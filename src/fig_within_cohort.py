"""
Within-cohort 控制实验合图（round-18 新增）——把两个原本各占一张跨栏图的实验并成一张。

三个面板共用同一个 y 轴（论文主结局 Δ_selection = max_b − mean_b，折内选特征），
因此可以直接比较"什么会抬高选择乐观、什么不会"：
  (a) FIX-N  : n 固定 200，少数类事件数下降        → Δ 上升（实测 r=-0.38）
  (b) FIX-E  : 少数类事件固定 50，n 下降            → Δ 略升（实测 r=-0.12，弱）
  (c) DIM    : n 与两个类别计数都固定，维度 d 变化   → Δ 不动
⚠️ 关键：不要只看相关系数的符号，要看【闭式预测的变化幅度】——每个面板都叠加了
   κ·SE(0.5, 0.3n₊, 0.3n₋) 的虚线。(b) 之所以相关弱，是因为在少数类固定时增加多数类
   本来就只带来很小的 SE 变化（预测跨度 0.030，对比 (a) 的 0.074）：弱相关是【符合】
   闭式预测的，不是反例。(c) 的预测线按构造是平的，而实测也平。
   所以正确表述是"两个类别计数都进入公式，但权重高度偏向少数类"，
   而不是旧稿说的"任一类别计数都同等地抬高天花板"。

数据：results/re_identify_selection.csv（r_e_selection.py）
      results/rg_dim_channel.csv（r_g_dim_channel.py，只取 infold 臂）
输出：figures/within_cohort.png/.pdf
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGS = ROOT / "results", ROOT / "figures"
MKS = ["o", "s", "^", "D", "v"]
CALIB = {}          # 面板 (a)(b) 的跨度与校准斜率，落盘供核对


def wcorr(df, xcol, ycol):
    """within-cohort 相关：先按队列去均值，再合并（避免队列间差异冒充趋势）。"""
    g = df.groupby("cohort")
    xc = df[xcol] - g[xcol].transform("mean")
    yc = df[ycol] - g[ycol].transform("mean")
    return float(np.corrcoef(xc, yc)[0, 1])


def hm_se(a, n_pos, n_neg):
    a = min(max(a, 1e-6), 1 - 1e-6)
    q1 = a / (2 - a); q2 = 2 * a * a / (1 + a)
    return math.sqrt(max((a * (1 - a) + (n_pos - 1) * (q1 - a ** 2)
                          + (n_neg - 1) * (q2 - a ** 2)) / (n_pos * n_neg), 1e-12))


def counts_only_calibration():
    """(alpha0, kappa0) 拟合在 SE(0.5) 上。这三个面板都是"仅凭类别计数预测"，
    工作点只能是 a=0.5，所以必须用这一套，而不是 Fig. 3 那套以诚实 AUC 为工作点的系数。"""
    j = json.loads((RESULTS / "reliability_predictor.json").read_text(encoding="utf-8"))
    c = j["counts_only_fit (a=0.5; THIS is the calculator's calibration)"]
    return 0.0, float(c["kappa0"])      # 截距被约束为 0（round-27）


def panel(ax, df, xcol, title, xlabel, invert=False, logx=False, kap=None):
    cs = sorted(df.cohort.unique())
    for i, name in enumerate(cs):
        g = df[df.cohort == name].groupby(xcol)["delta_selection"].mean().sort_index()
        ax.plot(g.index, g.values, MKS[i % len(MKS)] + "-", color=figstyle.RED,
                ms=3.0, lw=0.85, alpha=0.30, zorder=2)
    piv = df.pivot_table(index=xcol, columns="cohort", values="delta_selection", aggfunc="mean")
    full = piv.notna().all(axis=1).values           # 只在所有队列都在场时画均值，避免组成偏差
    x, m = piv.index.values, piv.mean(axis=1).values
    ax.plot(x[full], m[full], "-", color=figstyle.RED, lw=2.6, zorder=5,
            solid_capstyle="round", label="observed")
    # 闭式预测：kappa * HM-SE(0.5, 0.3 n_pos, 0.3 n_neg)。三个面板同一个公式、同一个拟合 kappa，
    # 因此"预测的变化幅度"本身就是被检验的对象（不是只看相关的符号）。
    gp = df.groupby(xcol).agg(nn=("n", "mean"), mm=("minority", "mean"))
    tp = np.maximum(1, np.round(0.30 * gp["mm"])); tn = np.maximum(1, np.round(0.30 * (gp["nn"] - gp["mm"])))
    a0, k0 = kap
    pred = np.array([a0 + k0 * hm_se(0.5, p, q) for p, q in zip(tp, tn)])
    # 预测线只画在 complete-case 的水平上，与报告的跨度/校准斜率完全同一个估计量；
    # 否则 (b) 的虚线画到 n=500、实线只到 320，图上就仍留着"混合队列构成"的口实。
    _full = piv.notna().all(axis=1).reindex(gp.index).values
    ax.plot(gp.index.values[_full], pred[_full], "--", color="0.30", lw=1.5, zorder=6,
            label=r"predicted $\hat\kappa_0\,\mathrm{SE}$")
    # 面板 (c) 的 x 轴是对数的，正文引用的也是"每 decade"的斜率，
    # 所以相关系数必须在 log10(d) 上算，否则图上的 r 与正文口径不一致。
    # **complete-case estimand**：只在四个队列都在场的水平上算跨度、校准斜率和相关。
    # 否则 fixed-events 臂的 n=500 只有 2/4 队列，斜率会被组成变化污染（实测 0.67 -> 0.40）。
    keep = df[xcol].isin(piv.index.values[full])
    dfc = df[keep]
    if logx:
        r = wcorr(dfc.assign(_lx=np.log10(dfc[xcol])), "_lx", "delta_selection")
    else:
        r = wcorr(dfc, xcol, "delta_selection")
    ax.set_title(title, fontsize=7.2)
    ax.set_xlabel(xlabel)
    obs = piv.mean(axis=1).reindex(gp.index).values
    predf, obsf = pred[full], obs[full]
    span_p, span_o = float(predf.max() - predf.min()), float(obsf.max() - obsf.min())
    cal = float(np.polyfit(predf, obsf, 1)[0])
    n_drop = int((~full).sum())
    CALIB[xlabel] = {
        "levels_complete_case": int(full.sum()), "levels_total": len(full),
        "levels_dropped": n_drop,
        "predicted_span": round(span_p, 4), "observed_span": round(span_o, 4),
        "calibration_slope": round(cal, 3),
        "calibration_slope_if_incomplete_levels_pooled": round(float(np.polyfit(pred, obs, 1)[0]), 3),
        "within_cohort_r": round(r, 3),
    }
    print("   [%s] complete-case levels %d/%d (dropped %d), slope %.3f (all-levels %.3f)"
          % (xlabel, int(full.sum()), len(full), n_drop, cal,
             float(np.polyfit(pred, obs, 1)[0])))
    ax.annotate(r"span: predicted %.03f, observed %.03f" % (span_p, span_o),
                xy=(0.5, 0.98), xycoords="axes fraction", ha="center", va="top",
                fontsize=5.6, color="0.25")
    ax.annotate(r"calibration slope %.2f   ($r=%+.2f$)" % (cal, r),
                xy=(0.5, 0.90), xycoords="axes fraction", ha="center", va="top",
                fontsize=5.6, color="0.25")
    if logx:
        ax.set_xscale("log")
    if invert:
        ax.invert_xaxis()
    ax.grid(alpha=0.3)



ARM_STYLE = {                       # 三个选择臂：折内 / 全数据选特征(k 可调) / 全数据选特征(k 固定)
    "infold":       ("selection in-fold",            figstyle.C_INFOLD,      "-"),
    "leaky":        ("full-data selection, $k$ tuned", figstyle.C_LEAKY,     "--"),
    "leaky_fixedk": ("full-data selection, $k{=}30$",  figstyle.C_LEAKY_FIXK, ":"),
}


def panel_arms(ax, rg_all, kap):
    """(c) 三个选择臂的 Delta_split 对 d/D 的剂量-反应，共用 (a)(b) 的 y 轴。
    正文声称"三臂都不随 d 增长"，图就必须把三臂都画出来。"""
    for arm, (lab, col, ls) in ARM_STYLE.items():
        sub = rg_all[rg_all.arm == arm]
        if not len(sub):
            continue
        g = (sub.groupby(["cohort", "dfrac"], as_index=False)["delta_selection"].mean()
                .pivot(index="dfrac", columns="cohort", values="delta_selection"))
        full = g.notna().all(axis=1).values
        ax.plot(g.index.values[full], g.mean(axis=1).values[full], ls, color=col, lw=2.0,
                zorder=5, solid_capstyle="round", label=lab)
    sub = rg_all[rg_all.arm == "infold"]
    gp = sub.groupby("dfrac").agg(nn=("n", "mean"), mm=("minority", "mean"))
    tp = np.maximum(1, np.round(0.30 * gp["mm"])); tn = np.maximum(1, np.round(0.30 * (gp["nn"] - gp["mm"])))
    a0, k0 = kap
    pred = np.array([a0 + k0 * hm_se(0.5, p, q) for p, q in zip(tp, tn)])
    ax.plot(gp.index.values, pred, "--", color="0.30", lw=1.5, zorder=6)
    ax.set_title(r"(c) counts fixed, vary $d$", fontsize=7.2)
    ax.set_xlabel("retained feature fraction $d/D$")
    ax.set_xscale("log"); ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=5.0, framealpha=0.85, borderpad=0.3,
              handlelength=1.6, labelspacing=0.25)


def main():
    figstyle.apply()
    re = pd.read_csv(RESULTS / "re_identify_selection.csv")
    rg_all = pd.read_csv(RESULTS / "rg_dim_channel.csv")
    rg = rg_all[rg_all.arm == "infold"]
    # 每个 (cohort, dfrac) 对多个随机特征子集取平均，再进入面板
    rg = (rg.groupby(["cohort", "dfrac"], as_index=False)
            .agg(n=("n", "mean"), minority=("minority", "mean"),
                 delta_selection=("delta_selection", "mean"),
                 honest=("honest", "mean")))

    kap = counts_only_calibration()
    fig, axes = plt.subplots(1, 3, figsize=(figstyle.SPAN, 2.5), sharey=True)
    panel(axes[0], re[re.arm == "fix_n"], "minority",
          r"(a) $n=200$, vary balance", "minority events",
          invert=True, kap=kap)
    panel(axes[1], re[re.arm == "fix_e"], "n",
          r"(b) 50 minority events, vary $n$", "total sample size $n$", invert=True, kap=kap)
    panel_arms(axes[2], rg_all, kap)
    axes[0].set_ylabel(r"split-shopping optimism $\Delta_{\rm split}$")
    axes[0].legend(loc="lower left", fontsize=5.6, framealpha=0.85)
    fig.suptitle("Class counts move the optimism in the predicted direction, at attenuated magnitude; no detectable trend with dimensionality",
                 fontsize=7.8, y=1.03)
    fig.tight_layout(w_pad=1.1)
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "within_cohort.png"); fig.savefig(FIGS / "within_cohort.pdf")
    plt.close(fig)

    print("within-cohort correlations of Delta_selection (the paper's endpoint):")
    print(f"  (a) FIX-N  r(Delta, minority events) = {wcorr(re[re.arm=='fix_n'],'minority','delta_selection'):+.3f}"
          f"   [honest: {wcorr(re[re.arm=='fix_n'],'minority','honest'):+.3f}]")
    print(f"  (b) FIX-E  r(Delta, n)               = {wcorr(re[re.arm=='fix_e'],'n','delta_selection'):+.3f}"
          f"   [honest: {wcorr(re[re.arm=='fix_e'],'n','honest'):+.3f}]")
    print(f"  (c) DIM    r(Delta, log10 d/D)       = "
          f"{wcorr(rg.assign(ld=np.log10(rg['dfrac'])),'ld','delta_selection'):+.3f}"
          f"   [honest: {wcorr(rg.assign(ld=np.log10(rg['dfrac'])),'ld','honest'):+.3f}]")
    (RESULTS / "within_cohort_calibration.json").write_text(
        json.dumps({"note": "spans and calibration slopes for Fig. 4(a),(b). Complete-case: only "
                            "levels where all four cohorts are present. Prediction uses the "
                            "counts-only affine fit alpha0 + kappa0 * HM_SE(0.5, .).",
                    "calibration": {"alpha0": kap[0], "kappa0": kap[1]},
                    "arms": CALIB}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved -> figures/within_cohort.png/.pdf , results/within_cohort_calibration.json")


if __name__ == "__main__":
    main()
