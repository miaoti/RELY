"""
radMLBench RF robustness check (reviewer M1) -- the load-bearing test of the headline.

Re-runs the optimism sweep with a HIGH-VARIANCE learner (random forest) and, crucially,
with NO univariate pre-selection, so feature dimensionality CAN exert its effect. If the
optimism still tracks the class-count standard error and not dimensionality here, the
"sampling variance, not dimensionality" claim survives beyond the l2-logistic pipeline;
if dimensionality starts explaining the optimism, the claim must be scoped to regularized
linear models. Same cohorts, same 70/30 selection protocol, same gap = max_selected - honest.

Outputs results/radmlbench_sweep_rf.csv (mirrors radmlbench_sweep.csv columns).
Usage: python src/run_radmlbench_rf.py --repeats 3            # all 50 cohorts
"""
from __future__ import annotations
import argparse, csv, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     GridSearchCV, train_test_split)
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_radmlbench as rr   # reuse load_dataset + radMLBench listing

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEED = 20260625
N_TREES = 150
# RF's dimensionality knob is max_features; NO SelectKBest upstream, so all d features are visible
GRID = {"clf__max_features": ["sqrt", 0.3]}


def rf_pipe():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(0.0)),
        ("clf", RandomForestClassifier(n_estimators=N_TREES, class_weight="balanced_subsample",
                                       random_state=SEED, n_jobs=-1)),
    ])


def honest_rf(X, y, repeats=3):
    """Leakage-free repeated stratified nested CV, random forest, no feature preselection."""
    outer = RepeatedStratifiedKFold(n_splits=10, n_repeats=repeats, random_state=SEED)
    aucs = []
    for i, (tr, te) in enumerate(outer.split(X, y)):
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED + i)
        gs = GridSearchCV(rf_pipe(), GRID, scoring="roc_auc", cv=inner, n_jobs=1, refit=True)
        gs.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], gs.predict_proba(X[te])[:, 1]))
    a = np.array(aucs)
    return float(a.mean()), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def test_tuned_rf(X, y, n_splits=30):
    """Selection optimism: max over random 70/30 splits x the RF grid, tuned on the test set."""
    best = []
    for s in range(n_splits):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y, random_state=s)
        b = 0.0
        for mf in GRID["clf__max_features"]:
            p = rf_pipe(); p.set_params(clf__max_features=mf).fit(Xtr, ytr)
            b = max(b, roc_auc_score(yte, p.predict_proba(Xte)[:, 1]))
        best.append(b)
    a = np.array(best)
    return float(a.max()), float(a.mean())


FIELDS = ["dataset", "n", "d", "minority", "epv", "honest_auc", "honest_lo", "honest_hi",
          "test_selected_max", "test_selected_mean", "optimism_gap"]


def run_one(name, repeats):
    X, y = rr.load_dataset(name)
    n, d = X.shape
    minority = int(min(int(y.sum()), n - int(y.sum())))
    hm, hlo, hhi = honest_rf(X, y, repeats)
    tmax, tmean = test_tuned_rf(X, y)
    return {"dataset": name, "n": n, "d": d, "minority": minority, "epv": round(minority / d, 4),
            "honest_auc": round(hm, 4), "honest_lo": round(hlo, 3), "honest_hi": round(hhi, 3),
            "test_selected_max": round(tmax, 4), "test_selected_mean": round(tmean, 4),
            "optimism_gap": round(tmax - hm, 4)}


def main():
    import radMLBench as rb
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--only", type=str, default="")
    args = ap.parse_args()
    todo = [s.strip() for s in args.only.split(",") if s.strip()] or rb.listDatasets()
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "radmlbench_sweep_rf.csv"
    done = set()
    if out.exists():
        try:
            done = set(pd.read_csv(out)["dataset"].astype(str))
        except Exception:
            done = set()
    pend = [n for n in todo if n not in done]
    print(f"[rf-sweep] total={len(todo)} done={len(done)} pending={len(pend)}  RF n_estimators={N_TREES}, no feature preselection", flush=True)
    t0 = time.time()
    for i, name in enumerate(pend, 1):
        try:
            r = run_one(name, args.repeats)
            with open(out, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                if f.tell() == 0:
                    w.writeheader()
                w.writerow(r)
            print(f"  [{i}/{len(pend)}] {name:26} n={r['n']:>4} d={r['d']:>5} honest={r['honest_auc']:.3f} "
                  f"max={r['test_selected_max']:.3f} gap={r['optimism_gap']:+.3f}  ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"  [{i}/{len(pend)}] {name:26} FAILED {type(e).__name__}: {e}", flush=True)
    print(f"[done] {time.time()-t0:.0f}s -> {out}  ({len(pd.read_csv(out))} cohorts)", flush=True)


if __name__ == "__main__":
    main()
