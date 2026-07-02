"""
阶段 R-A · C1 事件数定律。
跨 50 个真实影像组学队列：乐观差距（cherry-pick = test_selected_max − honest）的幅度
由 log10(少数类事件数) 预测（R²≈0.59），加维度后斜率≈0 → 乐观由事件数而非维度决定。
+ 机制（挑划分天花板≈诚实抽样分布上尾）+ 可操作阈值。
输出 results/c1_law.json，图 figures/C1_law_gap_vs_events.png(.pdf)。
用法：python src/c1_law.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGS = ROOT / "results", ROOT / "figures"
SEED = 20260625


def ols(y, X):
    """X 不含截距；返回 (beta_with_intercept, R2, yhat)。"""
    A = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ b
    r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return b, float(r2), yhat


def main():
    d = pd.read_csv(RESULTS / "radmlbench_sweep.csv")
    gap = (d["test_selected_max"] - d["honest_auc"]).values
    lmin = np.log10(d["minority"].values.astype(float))
    ld = np.log10(d["d"].values.astype(float))

    # 主回归 gap ~ log10(minority)
    b1, r2_1, _ = ols(gap, lmin.reshape(-1, 1))
    # + 维度
    b2, r2_2, _ = ols(gap, np.column_stack([lmin, ld]))
    # 仅维度
    _, r2_d, _ = ols(gap, ld.reshape(-1, 1))

    # bootstrap R²(主回归) 的数据集级 CI
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(3000):
        idx = rng.integers(0, len(gap), len(gap))
        _, r2b, _ = ols(gap[idx], lmin[idx].reshape(-1, 1))
        boots.append(r2b)
    r2_lo, r2_hi = np.percentile(boots, [2.5, 97.5])

    # 维度的偏回归（added-variable）：残差化
    def resid(y, x):
        bb, _, yh = ols(y, x.reshape(-1, 1)); return y - yh
    e_gap = resid(gap, lmin); e_d = resid(ld, lmin)
    bav, r2av, _ = ols(e_gap, e_d.reshape(-1, 1))
    partial_d_slope = float(bav[1])

    # 机制
    mech_tail = float(np.corrcoef(d["test_selected_max"], d["honest_hi"])[0, 1])
    mech_half = float(np.corrcoef(gap, d["honest_hi"] - d["honest_auc"])[0, 1])
    cover_chance = int(((d["honest_lo"] <= 0.5) & (d["honest_hi"] >= 0.5)).sum())
    manufactured = int(((d["honest_auc"] < 0.60) & (d["test_selected_max"] >= 0.80)).sum())

    # 可操作阈值：期望 gap < 0.05 需多少少数类事件
    a, slope = b1[0], b1[1]
    m_for_05 = float(10 ** ((0.05 - a) / slope)) if slope != 0 else float("nan")

    # ICC（worked case study；非 radMLBench）
    he = json.loads((RESULTS / "honest_eval_summary.json").read_text(encoding="utf-8"))
    icc_gap = he["naive_single_split"]["max"] - he["honest_nested_cv"]["mean"]
    icc_min = 57

    out = {
        "n_cohorts": len(d),
        "cherry_gap_mean": float(gap.mean()), "cherry_gap_pos": int((gap > 0).sum()),
        "reg_gap~log10(minority)": {"R2": r2_1, "R2_boot95": [float(r2_lo), float(r2_hi)],
                                    "intercept": float(a), "slope": float(slope)},
        "reg_+log10(d)": {"R2": r2_2, "b_minority": float(b2[1]), "b_dim": float(b2[2])},
        "reg_dim_only_R2": r2_d,
        "partial_dim_slope_after_events": partial_d_slope,
        "mechanism": {"corr_cherrymax_vs_honesthi": mech_tail,
                      "corr_gap_vs_honest_CI_halfwidth": mech_half},
        "honest_CI_covers_0.5": f"{cover_chance}/{len(d)}",
        "manufactured_AUC>=0.80_from_noise": f"{manufactured}/{len(d)}",
        "min_events_for_expected_gap_below_0.05": round(m_for_05, 1),
        "ICC_worked_case": {"minority": icc_min, "cherry_gap": round(float(icc_gap), 3)},
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "c1_law.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))

    # 图：左=gap vs log10(events)+fit；右=维度 added-variable（斜率≈0）
    FIGS.mkdir(exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    ax1.scatter(lmin, gap, c="#1f77b4", s=30, label=f"radMLBench (N={len(d)})")
    ax1.scatter([np.log10(icc_min)], [icc_gap], c="#d62728", s=130, marker="*", zorder=5, label="ICC (case study)")
    xs = np.linspace(lmin.min(), lmin.max(), 50)
    ax1.plot(xs, a + slope * xs, "k-", lw=1.5, label=f"fit: R²={r2_1:.2f} [{r2_lo:.2f},{r2_hi:.2f}]")
    ax1.set_xlabel("log10(minority-class events)"); ax1.set_ylabel("optimism gap (cherry-pick − honest)")
    ax1.set_title("C1 - optimism is governed by event count"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax2.scatter(e_d, e_gap, c="#7f7f7f", s=28)
    xr = np.linspace(e_d.min(), e_d.max(), 50)
    ax2.plot(xr, bav[0] + partial_d_slope * xr, "k-", lw=1.5, label=f"partial slope ≈ {partial_d_slope:+.3f}")
    ax2.set_xlabel("log10(dimensionality) | events  (residual)")
    ax2.set_ylabel("optimism gap | events  (residual)")
    ax2.set_title("Dimensionality adds nothing (after event count)")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "C1_law_gap_vs_events.png", dpi=200)
    fig.savefig(FIGS / "C1_law_gap_vs_events.pdf"); plt.close(fig)
    print("\nsaved -> results/c1_law.json , figures/C1_law_gap_vs_events.png/.pdf")


if __name__ == "__main__":
    main()
