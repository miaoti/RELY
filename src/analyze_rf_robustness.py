"""
Analyze the RF robustness sweep (reviewer M1). Question: does the class-count sampling-variance
law -- and the near-irrelevance of feature dimensionality -- survive a HIGH-VARIANCE learner
(random forest, NO univariate preselection, so dimensionality can act)? Mirrors
reliability_predictor.py exactly (same Hanley-McNeil SE at the 30%-test working point, same OLS)
but on results/radmlbench_sweep_rf.csv, and cross-checks against the l2-LR sweep.

Outputs results/rf_robustness.json and a printed summary to quote in the paper.
Usage: python src/analyze_rf_robustness.py
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def hm_se(auc, n_pos, n_neg):
    auc = min(max(auc, 1e-6), 1 - 1e-6)
    q1 = auc / (2 - auc); q2 = 2 * auc * auc / (1 + auc)
    return math.sqrt(max((auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2) + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg), 1e-12))


def r2(y, X):
    A = np.column_stack([np.ones(len(y))] + [X[:, i] for i in range(X.shape[1])])
    b, *_ = np.linalg.lstsq(A, y, rcond=None); yh = A @ b
    return 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum(), b


def analyze(csv, tag):
    d = pd.read_csv(csv)
    gap = (d["test_selected_max"] - d["honest_auc"]).values
    tpos = np.maximum(1, np.round(0.30 * d["minority"]).astype(int)).values
    tneg = np.maximum(1, np.round(0.30 * (d["n"] - d["minority"])).astype(int)).values
    se = np.array([hm_se(a, p, n) for a, p, n in zip(d["honest_auc"], tpos, tneg)])
    se05 = np.array([hm_se(0.5, p, n) for p, n in zip(tpos, tneg)])
    R2_se, b = r2(gap, se.reshape(-1, 1)); kappa = float(b[1])
    R2_se05, _ = r2(gap, se05.reshape(-1, 1))
    R2_d, _ = r2(gap, np.log10(d["d"].values.astype(float)).reshape(-1, 1))
    counts = np.column_stack([np.log10(tpos.astype(float)), np.log10(tneg.astype(float))])
    R2_counts, _ = r2(gap, counts)
    R2_counts_d, _ = r2(gap, np.column_stack([counts, np.log10(d["d"].values.astype(float))]))
    resid_sd = float(np.std(gap - b[0] - kappa * se, ddof=2))
    summary = {"tag": tag, "n_cohorts": int(len(d)), "mean_gap": round(float(gap.mean()), 3),
               "R2_SE_working_point": round(R2_se, 3), "R2_SE_a0.5_no_shared_term": round(R2_se05, 3),
               "R2_log_dimensionality": round(R2_d, 3),
               "R2_two_counts": round(R2_counts, 3), "R2_two_counts_plus_logd": round(R2_counts_d, 3),
               "incremental_R2_from_dimensionality": round(R2_counts_d - R2_counts, 3),
               "kappa_fitted": round(kappa, 3), "residual_sd": round(resid_sd, 3)}
    return summary, d


def main():
    rf, drf = analyze(RESULTS / "radmlbench_sweep_rf.csv", "RF (no preselection)")
    lr, dlr = analyze(RESULTS / "radmlbench_sweep.csv", "l2-LR (paper)")
    # cross-cohort agreement: do the two learners flag the same cohorts as optimistic?
    grf = drf[["dataset"]].copy(); grf["gap_rf"] = (drf["test_selected_max"] - drf["honest_auc"]).values
    glr = dlr[["dataset"]].copy(); glr["gap_lr"] = (dlr["test_selected_max"] - dlr["honest_auc"]).values
    m = pd.merge(grf, glr, on="dataset")
    r_cross = float(np.corrcoef(m["gap_rf"], m["gap_lr"])[0, 1]) if len(m) > 2 else float("nan")
    out = {"RF": rf, "LR": lr,
           "gap_rf_vs_gap_lr_pearson_r": round(r_cross, 3), "n_matched_cohorts": int(len(m))}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "rf_robustness.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
