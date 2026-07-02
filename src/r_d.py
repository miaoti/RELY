"""
阶段 R-D · 剂量-反应：在大队列【内部】分离"少数类事件数" vs "维度"对乐观差距的作用，
把 C1 的跨队列相关升级为同队列内的因果剂量-反应。
  Arm EVENTS：固定全部特征(d 不变)，逐步下采样少数类事件数 e → 预期 gap 单调上升。
  Arm DIM   ：固定全部样本(n、e 不变)，随机丢特征到 d' → 预期 gap 基本不动。
gap = cherry-pick(全数据选特征 + K 次随机划分取最大 test AUC) − 诚实(折内选择的重复分层 5 折 CV)。
自包含、对小子样本鲁棒(k 随 d/e 自适应)；增量写 results/rd_dose_response.csv，可续跑。
用法：python src/r_d.py [--cohorts 3] [--K 200]
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGS = ROOT / "results", ROOT / "figures"
SEED = 20260625
FIELDS = ["cohort", "arm", "level", "n", "minority", "d", "honest", "cherry", "gap"]


def make_pipe(d):
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("var", VarianceThreshold(0.0)), ("scaler", StandardScaler()),
                     ("select", SelectKBest(f_classif, k=min(20, d))),
                     ("clf", LogisticRegression(class_weight="balanced", solver="liblinear",
                                                max_iter=5000, C=0.1))])


def honest_auc(X, y, repeats=10):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=SEED)
    a = []
    for tr, te in cv.split(X, y):
        m = make_pipe(X.shape[1]).fit(X[tr], y[tr])
        a.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return float(np.mean(a))


def cherry_auc(X, y, K=200):
    Xi = SimpleImputer(strategy="median").fit_transform(X)
    Xs = StandardScaler().fit_transform(Xi)
    Xsel = SelectKBest(f_classif, k=min(30, Xs.shape[1])).fit_transform(Xs, y)  # 泄漏：全数据选
    best = 0.0
    for s in range(K):
        Xtr, Xte, ytr, yte = train_test_split(Xsel, y, test_size=0.3, stratify=y, random_state=s)
        m = LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=5000, C=0.1).fit(Xtr, ytr)
        best = max(best, roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
    return float(best)


def gap_of(X, y, K):
    h = honest_auc(X, y); c = cherry_auc(X, y, K)
    return h, c, c - h


def load_radml(name):
    import radMLBench as rb
    df = rb.loadData(name)
    y = df["Target"].astype(int).values
    X = df.drop(columns=[c for c in ("Target", "ID") if c in df.columns]).select_dtypes("number").values.astype(float)
    return X, y


def pick_cohorts(n):
    d = pd.read_csv(RESULTS / "radmlbench_sweep.csv")
    d = d[(d["d"] <= 1200) & (d["minority"] >= 120)].sort_values("minority", ascending=False)
    return list(d["dataset"].head(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", type=int, default=3)
    ap.add_argument("--K", type=int, default=200)
    args = ap.parse_args()
    cohorts = pick_cohorts(args.cohorts)
    print("[R-D] cohorts:", cohorts, flush=True)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "rd_dose_response.csv"
    done = set()
    if out.exists():
        dd = pd.read_csv(out); done = set(zip(dd["cohort"], dd["arm"], dd["level"]))
    rng = np.random.default_rng(SEED)

    def record(row):
        with open(out, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if f.tell() == 0:
                w.writeheader()
            w.writerow(row)

    for name in cohorts:
        X, y = load_radml(name)
        minority_lab = int(np.argmin(np.bincount(y)))
        idx_min = np.where(y == minority_lab)[0]; idx_maj = np.where(y != minority_lab)[0]
        M = len(idx_min); D = X.shape[1]
        # Arm EVENTS：固定 d，少数类事件数 e 下采样
        for frac in [1.0, 0.7, 0.5, 0.35, 0.25]:
            e = max(20, int(frac * M))
            key = (name, "events", e)
            if key in done or e > M:
                continue
            sel_min = rng.choice(idx_min, e, replace=False)
            ii = np.concatenate([sel_min, idx_maj])
            h, c, g = gap_of(X[ii], y[ii], args.K)
            record({"cohort": name, "arm": "events", "level": e, "n": len(ii),
                    "minority": e, "d": D, "honest": round(h, 4), "cherry": round(c, 4), "gap": round(g, 4)})
            print(f"  {name:18} EVENTS e={e:<4} gap={g:+.3f}", flush=True)
        # Arm DIM：固定 n、e，维度 d' 下采样
        for dfrac in [1.0, 0.5, 0.25, 0.1]:
            dd = max(50, int(dfrac * D))
            key = (name, "dim", dd)
            if key in done or dd > D:
                continue
            cols = rng.choice(D, dd, replace=False)
            h, c, g = gap_of(X[:, cols], y, args.K)
            record({"cohort": name, "arm": "dim", "level": dd, "n": len(y),
                    "minority": M, "d": dd, "honest": round(h, 4), "cherry": round(c, 4), "gap": round(g, 4)})
            print(f"  {name:18} DIM    d={dd:<5} gap={g:+.3f}", flush=True)

    # 图
    df = pd.read_csv(out)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    for name in df["cohort"].unique():
        a = df[(df.cohort == name) & (df.arm == "events")].sort_values("level")
        ax1.plot(a["level"], a["gap"], "o-", label=name)
        b = df[(df.cohort == name) & (df.arm == "dim")].sort_values("level")
        ax2.plot(b["level"], b["gap"], "s-", label=name)
    ax1.set_xlabel("minority events e (d fixed)"); ax1.set_ylabel("optimism gap")
    ax1.set_title("R-D - optimism RISES as minority events drop (d fixed; dominant lever)")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3); ax1.invert_xaxis()
    ax2.set_xlabel("dimensionality d (n & events fixed)"); ax2.set_ylabel("optimism gap")
    ax2.set_title("R-D - dimensionality: weaker secondary effect (not flat, not dominant)")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "C1_dose_response.png", dpi=200)
    fig.savefig(FIGS / "C1_dose_response.pdf"); plt.close(fig)
    print(f"\nsaved -> {out}, figures/C1_dose_response.png/.pdf", flush=True)


if __name__ == "__main__":
    main()
