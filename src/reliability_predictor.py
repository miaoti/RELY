"""
头条 · 诚实评测可靠性预测器（操作化已知抽样理论 + 验证 + 计算器）。诚实定位：不主张新定理。
把 Hanley-McNeil(1982) AUC 标准误（仅 n_pos/n_neg/AUC 闭式）操作化、跨 50 真实队列验证它能
预测"报告型乐观"(cherry-pick = test_selected_max − honest)，并做成预注册计算器。
⚠️ 自承：gap≈κ·SE 在形式上部分是"构造性恒等"（极值≈κ·SD、SD=AUC SE）；真正承重的是
(i) 该理想关系在 50 个真实管线上确实成立(操作化)，(ii) 与维度(公式里没有的量)的对比(R²≈0.09)。
近邻先验：Obuchowski(1998)/Hajian-Tilaki(2014) AUC 精度样本量、Riley(2019)/pmsampsize（开发期 EPV）——
本文把"AUC 精度样本量"从诚实估计区推广到"选择乐观区"并在真实影像组学上验证。
输出 results/reliability_predictor.json，图 figures/reliability_validation.png(.pdf)。
"""
from __future__ import annotations
import json, math, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGS = ROOT / "results", ROOT / "figures"
B_NOMINAL = 750


def hm_se(auc, n_pos, n_neg):
    auc = min(max(auc, 1e-6), 1 - 1e-6)
    q1 = auc / (2 - auc); q2 = 2 * auc * auc / (1 + auc)
    return math.sqrt(max((auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2) + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg), 1e-12))


def r2(y, X):
    A = np.column_stack([np.ones(len(y))] + [X[:, i] for i in range(X.shape[1])])
    b, *_ = np.linalg.lstsq(A, y, rcond=None); yh = A @ b
    return 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum(), b, yh


def main():
    d = pd.read_csv(RESULTS / "radmlbench_sweep.csv")
    gap = (d["test_selected_max"] - d["honest_auc"]).values
    tpos = np.maximum(1, np.round(0.30 * d["minority"]).astype(int)).values
    tneg = np.maximum(1, np.round(0.30 * (d["n"] - d["minority"])).astype(int)).values
    se = np.array([hm_se(a, p, n) for a, p, n in zip(d["honest_auc"], tpos, tneg)])
    src = np.array([re.sub(r'\d+.*$', '', re.split(r'[-_]', x)[0]) or x for x in d["dataset"]])

    # 主预测器：gap ~ SE（闭式，1 维）
    R2_se, b_se, _ = r2(gap, se.reshape(-1, 1))
    kappa_fit = float(b_se[1])                          # 实测斜率
    B_eff = float(math.exp(kappa_fit ** 2 / 2))         # 有效独立选择数（划分高度相关 → 远小于名义 B）
    # 对手对照
    R2_ev, *_ = r2(gap, np.log10(d["minority"].values.astype(float)).reshape(-1, 1))
    R2_n, *_ = r2(gap, np.log10(d["n"].values.astype(float)).reshape(-1, 1))
    R2_d, *_ = r2(gap, np.log10(d["d"].values.astype(float)).reshape(-1, 1))
    R2_se_a05, *_ = r2(gap, np.array([hm_se(0.5, p, n) for p, n in zip(tpos, tneg)]).reshape(-1, 1))  # de-circularized
    # R1(2a)：自由多元基线 {log n_pos, log n_neg, AUC}（3 维）vs 闭式 SE（1 维）
    multi = np.column_stack([np.log10(tpos), np.log10(tneg), d["honest_auc"].values])
    R2_multi, *_ = r2(gap, multi)
    resid_sd = float(np.std(gap - b_se[0] - kappa_fit * se, ddof=2))

    # 稳健性（写进代码、落 JSON —— 修复硬编码问题）
    loso = []
    for s in np.unique(src):
        m = src != s
        loso.append(r2(gap[m], se[m].reshape(-1, 1))[0])
    rng = np.random.default_rng(20260625)
    groups = [np.where(src == s)[0] for s in np.unique(src)]
    boot = []
    for _ in range(3000):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        boot.append(r2(gap[idx], se[idx].reshape(-1, 1))[0])

    out = {
        "n_cohorts": int(len(d)), "n_sources": int(len(np.unique(src))),
        "R2_observed_gap_vs": {
            "hanley_mcneil_SE (closed-form 1-param)": round(R2_se, 3),
            "free multivariate {log n_pos, log n_neg, AUC} (3-param)": round(R2_multi, 3),
            "log10(minority_events)": round(R2_ev, 3), "log10(n)": round(R2_n, 3),
            "log10(dimensionality)": round(R2_d, 3),
        },
        "robustness_R2": {
            "leave_one_source_out": [round(min(loso), 3), round(max(loso), 3)],
            "cluster_bootstrap_95CI": [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)],
            "fixed_working_point_a0.5 (no shared honest-AUC)": round(r2(gap, np.array([hm_se(0.5, p, n) for p, n in zip(tpos, tneg)]).reshape(-1, 1))[0], 3),
            "fixed_working_point_a0.7": round(r2(gap, np.array([hm_se(0.7, p, n) for p, n in zip(tpos, tneg)]).reshape(-1, 1))[0], 3),
        },
        "kappa_fitted": round(kappa_fit, 3),
        "kappa_theoretical_B750": round(math.sqrt(2 * math.log(B_NOMINAL)), 3),
        "effective_independent_selections_B_eff": round(B_eff, 1),
        "residual_sd": round(resid_sd, 3),
        "caveat": "gap≈kappa*SE is partly confirmatory-by-construction (extreme-value); the load-bearing "
                  "results are that it holds across 50 real pipelines and the contrast with dimensionality (R²≈0.09).",
        "scope": "predicts the split-max (selection) component of optimism; not the feature-leakage component.",
    }

    # 计算器：用【实测有效 κ】（非名义 B 的理论 κ）；给规划区间
    def events_needed(delta, prev=0.30, auc=0.5):
        for e in range(5, 8000):
            npos = max(1, int(0.30 * e)); nneg = max(1, int(0.30 * e * (1 - prev) / prev))
            if kappa_fit * hm_se(auc, npos, nneg) <= delta:
                return e
        return None
    out["calculator (uses FITTED kappa=%.2f, planning-level not per-study)" % kappa_fit] = {
        "min_minority_events_for_expected_gap<=0.05 (prev=0.3,AUC=0.5)": events_needed(0.05),
        "min_minority_events_for_expected_gap<=0.10": events_needed(0.10),
        "prediction_interval_note": f"±{1.96*resid_sd:.2f} AUC residual (95%); population-level planning heuristic",
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "reliability_predictor.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))

    # 图（用拟合 κ 画预测线）
    FIGS.mkdir(exist_ok=True)
    figstyle.apply()
    pred = b_se[0] + kappa_fit * se
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figstyle.SPAN, 2.9))
    ax1.scatter(pred, gap, c=figstyle.BLUE, s=15, alpha=0.85, edgecolor="white", linewidth=0.3)
    lim = [min(pred.min(), gap.min()), max(pred.max(), gap.max())]
    ax1.plot(lim, lim, "--", color="0.2", lw=1.1, label="identity")
    ax1.set_xlabel(r"predicted optimism  $\hat{\kappa}\cdot\mathrm{SE}_{\mathrm{AUC}}$  ($\hat{\kappa}$=%.2f)" % kappa_fit)
    ax1.set_ylabel("observed cherry-pick optimism")
    ax1.set_title("(a) Predicted vs. observed")
    ax1.annotate(f"$n$={len(d)} cohorts\n$R^2$ = {R2_se:.2f}\nLOSO [{min(loso):.2f}, {max(loso):.2f}]\ncluster CI [{np.percentile(boot,2.5):.2f}, {np.percentile(boot,97.5):.2f}]",
                 xy=(0.04, 0.97), xycoords="axes fraction", va="top", ha="left", fontsize=7.0)
    ax1.legend(loc="lower right")
    bars = ["HM SE\n(1-par)", "free\nmulti", "log\nevents", "log\n$n$", "log\ndim"]
    vals = [R2_se, R2_multi, R2_ev, R2_n, R2_d]
    cols = [figstyle.GREEN, figstyle.BLUE, figstyle.BLUE, figstyle.BLUE, figstyle.GREY]
    ax2.bar(range(len(bars)), vals, color=cols, width=0.72)
    ax2.set_xticks(range(len(bars))); ax2.set_xticklabels(bars)
    for i, v in enumerate(vals):
        ax2.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=6.8)
    ax2.annotate("our model\n(1 param)", xy=(0.0, 0.785), xytext=(0.0, 0.905),
                 fontsize=5.9, ha="center", va="center", color=figstyle.GREEN, fontweight="bold",
                 linespacing=1.0,
                 arrowprops=dict(arrowstyle="->", color=figstyle.GREEN, lw=1.1))
    ax2.text(3.45, 0.80, "de-circularized\n($a$=0.5): $R^2$=%.2f" % R2_se_a05,
             fontsize=5.8, ha="center", va="center", color="0.3", linespacing=1.1)
    ax2.set_ylabel(r"$R^2$ predicting optimism"); ax2.set_ylim(0, 1)
    ax2.set_title("(b) Variance explained")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(FIGS / "reliability_validation.png"); fig.savefig(FIGS / "reliability_validation.pdf"); plt.close(fig)
    print("\nsaved -> results/reliability_predictor.json , figures/reliability_validation.png/.pdf")


if __name__ == "__main__":
    main()
