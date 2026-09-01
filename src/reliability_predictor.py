"""
头条 · 选择乐观的闭式预测器（操作化已知抽样理论 + 验证 + 计算器）。不主张新定理。

把 Hanley-McNeil(1982) AUC 标准误（仅 n_pos/n_neg/AUC 的闭式）操作化，跨 50 个真实影像组学
队列验证它能预测【选择乐观】，并做成预注册计算器。

── round-23 口径修订（回应第二轮外部审稿，三处均已在代码里复现确认）──────────────
① **正类方向**：此前用 `minority` 当 n_+，但 AUC 的正类是 `Target==1`；实测 50 个队列中
   **27 个** 的 Target=1 是多数类，对这些队列 n_+/n_- 被交换。HM 公式对二者不对称
   （q1=a/(2-a) ≠ q2=2a²/(1+a)），交换会改变 SE。改由 `cohort_counts` 提供真实方向。
② **测试集计数**：此前用 round(0.3*count)，实测 **28/50** 与 sklearn 分层划分的真实计数
   差 1。改用 `cohort_counts.exact_test_counts` 精确值。
③ **极值参照写错**：此前把 √(2lnB) − (lnlnB+ln4π)/(2√(2lnB)) = 2.101 说成
   "50 次独立抽样的理论期望"。那是 Gumbel 的**位置参数** b_n，不是 E[max]。
   模拟得 E[max Z]=2.249，而与本文统计量对应的 E[(max−mean)/s]=**2.261**。
   现改为模拟给参照，并如实指出实测乘子低于独立值（因划分高度重叠）。
④ **estimand 统一**：诚实点估计、bootstrap 区间、置换检验此前分别用"折均值"和
   "池化 OOF"，最大差 0.033。现全部统一到 **pooled out-of-fold AUC**。
⑤ **endpoint 命名**：`test_tuned_single_splits` 是逐划分先在 test 上取网格最优，再跨划分
   取 max，所以 max−mean 的 B = **划分数 50**，不是 50×15=750 个候选。现按两级拆分报告：
   Δ_tuning（划分内调参）、Δ_split（划分间挑选）、Δ_total（两级合计）。

输出 results/reliability_predictor.json，图 figures/reliability_validation.png(.pdf)。
"""
from __future__ import annotations
import json, math, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle
from cohort_counts import counts_table

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGS = ROOT / "results", ROOT / "figures"
N_SPLITS = 50          # 选择臂的划分数 = Δ_split 的选择预算 B


def hm_se(auc, n_pos, n_neg):
    auc = min(max(auc, 1e-6), 1 - 1e-6)
    q1 = auc / (2 - auc); q2 = 2 * auc * auc / (1 + auc)
    return math.sqrt(max((auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2)
                          + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg), 1e-12))


def r2(y, X):
    A = np.column_stack([np.ones(len(y))] + [X[:, i] for i in range(X.shape[1])])
    b, *_ = np.linalg.lstsq(A, y, rcond=None); yh = A @ b
    return 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum(), b, yh


def source_oof_r2(y, x, src):
    """真·留一来源外推：对每个 source，用【其余 source】拟合，预测该 source 的队列。"""
    pred = np.full(len(y), np.nan)
    for s in np.unique(src):
        tr = src != s
        A = np.column_stack([np.ones(tr.sum()), x[tr]])
        b, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        pred[~tr] = b[0] + b[1] * x[~tr]
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()), pred


def delete_one_range(y, x, src):
    """delete-one 敏感性区间（影响力诊断，【不是】样本外验证）。"""
    out = [r2(y[m], x[m].reshape(-1, 1))[0] for m in (src != s for s in np.unique(src))]
    return [round(float(min(out)), 3), round(float(max(out)), 3)]


def iid_multiplier_reference(B=N_SPLITS, reps=400000, seed=0):
    """B 个独立标准正态下 (max − mean)/s 的期望，即与我们统计量【同形】的参照。
    不要用 Gumbel 位置参数 b_n 冒充它（那正是 round-23 修掉的错误）。"""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((reps, B))
    stat = (Z.max(axis=1) - Z.mean(axis=1)) / Z.std(axis=1, ddof=1)
    return float(stat.mean()), float(stat.std(ddof=1))


def main():
    d = pd.read_csv(RESULTS / "radmlbench_sweep.csv")
    cts = counts_table(list(d["dataset"])).set_index("dataset")
    d = d.join(cts[["n_pos", "n_neg", "te_pos", "te_neg", "pos_is_majority"]], on="dataset")

    # estimand 统一：诚实 AUC 一律用 pooled out-of-fold（与 bootstrap 区间、置换检验同一个量）
    honest = d["honest_pooled_auc"].values
    delta_split = d["delta_selection"].values          # 划分间挑选（B = 50）
    delta_tuning = d["delta_tuning"].values            # 划分内 test-set 调参
    delta_total = d["delta_total"].values              # 两级合计
    sd_obs = d["test_selected_sd"].values              # 逐划分最优值的实测标准差
    tp, tn = d["te_pos"].values, d["te_neg"].values    # 真实方向 + 精确的 held-out 计数

    se = np.array([hm_se(a, p, n) for a, p, n in zip(honest, tp, tn)])
    se0 = np.array([hm_se(0.5, p, n) for p, n in zip(tp, tn)])
    logd = np.log10(d["d"].values.astype(float))
    src = np.array([re.sub(r'\d+.*$', '', re.split(r'[-_]', x)[0]) or x for x in d["dataset"]])

    # ── step 1：闭式能否预测【实测】的逐划分标准差？（无极值假设）──
    R2_sd_se, b_sd_se, _ = r2(sd_obs, se.reshape(-1, 1))
    R2_sd_se0, b_sd_se0, _ = r2(sd_obs, se0.reshape(-1, 1))
    R2_sd_d, *_ = r2(sd_obs, logd.reshape(-1, 1))
    step1 = {
        "note": "no extreme-value assumption enters: does the closed form predict the OBSERVED "
                "split-to-split SD of the per-split best AUC?",
        "R2_SD_vs_HM_SE(at pooled honest AUC)": round(R2_sd_se, 3),
        "R2_SD_vs_HM_SE(a=0.5, class counts only)": round(R2_sd_se0, 3),
        "R2_SD_vs_log10_dimensionality": round(R2_sd_d, 3),
        "slope_SD_on_SE(a=0.5)": round(float(b_sd_se0[1]), 3),
    }

    # ── step 2：极值乘子 Δ_split/SD 是否稳定？参照用【模拟】而不是 Gumbel 位置参数 ──
    ratio = delta_split / np.maximum(sd_obs, 1e-9)
    iid_mean, iid_sd = iid_multiplier_reference()
    step2 = {
        "note": "known extreme-value part; verified, not claimed. B = number of SPLITS (50), "
                "because each split has already been reduced to its grid-best AUC.",
        "B_splits": N_SPLITS,
        "mean_ratio_delta_split_over_observedSD": round(float(ratio.mean()), 3),
        "sd_of_ratio": round(float(ratio.std(ddof=1)), 3),
        "cv_of_ratio": round(float(ratio.std(ddof=1) / ratio.mean()), 3),
        "range": [round(float(ratio.min()), 2), round(float(ratio.max()), 2)],
        "iid_reference_E[(max-mean)/s]_for_B50": round(iid_mean, 3),
        "interpretation": "the observed multiplier sits BELOW the independent-draw reference. "
                          "We report this as an empirical observation, not a theorem: "
                          "dependence among order statistics need not lower this ratio in "
                          "general, and we did not derive a bound.",
    }

    # ── 两级选择的拆分（endpoint 现在写清楚了）──
    decomposition = {
        "Delta_tuning (within-split, choosing the best of the grid on the test block)":
            round(float(delta_tuning.mean()), 3),
        "Delta_split (across splits, given each split is already test-tuned; B=50)":
            round(float(delta_split.mean()), 3),
        "Delta_total (both levels, relative to the unselected grand mean)":
            round(float(delta_total.mean()), 3),
        "note": "Delta_split is what the closed form models; it is NOT the whole selection "
                "optimism, because Delta_tuning is already absorbed into the per-split means.",
    }

    # ── 合成：Δ_split ≈ κ·SE ──
    R2_se, b_se, _ = r2(delta_split, se.reshape(-1, 1))
    kappa_fit, intercept = float(b_se[1]), float(b_se[0])
    # round-26：**两套标定，各用各的**。上面这套拟合在"以队列自身诚实 AUC 为工作点"的 SE 上
    # （R2=0.816，用于 Fig. 3 的验证）；但计算器是建模【前】用的，工作点只能取零假设 a=0.5，
    # 所以必须用【在 SE(0.5) 上拟合出来的】那套系数。此前把前一套系数直接套到 SE(0.5)，
    # 那既不是 0.816 的拟合、也不是论文引用的 0.674 的拟合：实测只有 R2=0.591，
    # 且平均高估 Delta 约 +0.018，把所需事件数从 234/74 抬到了 357/97。
    R2_se0, b_se0, _ = r2(delta_split, se0.reshape(-1, 1))
    kappa0_affine, intercept0 = float(b_se0[1]), float(b_se0[0])
    resid_sd0_affine = float(np.std(delta_split - intercept0 - kappa0_affine * se0, ddof=2))

    # 截距的显著性（正文要引用；两个工作点上都不显著）
    def _intercept_ci(y, x):
        A = np.column_stack([np.ones(len(y)), x]); b, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = y - A @ b; s2 = float((r ** 2).sum()) / (len(y) - 2)
        se_a = math.sqrt(float(s2 * np.linalg.inv(A.T @ A)[0, 0]))
        return float(b[0]), se_a, [float(b[0] - 2.011 * se_a), float(b[0] + 2.011 * se_a)]
    a0_hat, a0_se, a0_ci = _intercept_ci(delta_split, se0)

    # **零截距重拟合**（不是"把截距设 0 但保留原斜率"）：理论上 SE->0 必须给 Delta->0。
    kappa0 = float((se0 @ delta_split) / (se0 @ se0))
    R2_se0_noint = float(1 - ((delta_split - kappa0 * se0) ** 2).sum()
                         / ((delta_split - delta_split.mean()) ** 2).sum())
    resid_sd0 = float(np.std(delta_split - kappa0 * se0, ddof=1))
    kappa_wp_noint = float((se @ delta_split) / (se @ se))
    R2_se_noint = float(1 - ((delta_split - kappa_wp_noint * se) ** 2).sum()
                        / ((delta_split - delta_split.mean()) ** 2).sum())
    R2_ev, *_ = r2(delta_split, np.log10(d["minority"].values.astype(float)).reshape(-1, 1))
    R2_n, *_ = r2(delta_split, np.log10(d["n"].values.astype(float)).reshape(-1, 1))
    R2_d, *_ = r2(delta_split, logd.reshape(-1, 1))
    multi = np.column_stack([np.log10(tp.astype(float)), np.log10(tn.astype(float)), honest])
    R2_multi, *_ = r2(delta_split, multi)
    R2_multi_d, *_ = r2(delta_split, np.column_stack([multi, logd]))
    resid_sd = float(np.std(delta_split - intercept - kappa_fit * se, ddof=2))

    oof_se, _ = source_oof_r2(delta_split, se, src)
    oof_se0, _ = source_oof_r2(delta_split, se0, src)
    rng = np.random.default_rng(20260625)
    groups = [np.where(src == s)[0] for s in np.unique(src)]
    boot = []
    for _ in range(3000):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        boot.append(r2(delta_split[idx], se[idx].reshape(-1, 1))[0])

    out = {
        "endpoint": "Delta_split = max_s(best-of-grid AUC) - mean_s(best-of-grid AUC) over the "
                    "SAME 50 random 70/30 splits. The selection budget is B=50 SPLITS, not 750 "
                    "candidates: each split is reduced to its grid-best before the max is taken.",
        "class_direction": "n_pos = count of Target==1 (the class roc_auc_score treats as "
                           "positive); test counts are the exact stratified held-out counts.",
        "cohorts_where_positive_class_is_majority": int(d["pos_is_majority"].sum()),
        "n_cohorts": int(len(d)), "n_sources": int(len(np.unique(src))),
        "two_level_decomposition": decomposition,
        "step1_does_closed_form_predict_observed_SD": step1,
        "step2_is_the_extreme_value_multiplier_stable": step2,
        "R2_Delta_split_vs": {
            "hanley_mcneil_SE (one predictor + intercept)": round(R2_se, 3),
            "hanley_mcneil_SE at a=0.5 (class counts only)": round(R2_se0, 3),
            "free multivariate {log n_pos, log n_neg, AUC}": round(R2_multi, 3),
            "log10(minority_events)": round(R2_ev, 3), "log10(n)": round(R2_n, 3),
            "log10(dimensionality)": round(R2_d, 3),
            "incremental R2 of log10(d) over the count baseline": round(R2_multi_d - R2_multi, 4),
        },
        "robustness_R2": {
            "TRUE_leave_one_source_out_prediction (SE at pooled honest AUC)": round(oof_se, 3),
            "TRUE_leave_one_source_out_prediction (SE at a=0.5)": round(oof_se0, 3),
            "delete_one_source_SENSITIVITY_range (influence, NOT out-of-sample)":
                delete_one_range(delta_split, se, src),
            "cluster_bootstrap_95CI": [round(float(np.percentile(boot, 2.5)), 3),
                                       round(float(np.percentile(boot, 97.5)), 3)],
        },
        "kappa_fitted (protocol-specific selection multiplier)": round(kappa_fit, 3),
        "fit_intercept": round(intercept, 4),
        "counts_only_fit (a=0.5; THIS is the calculator's calibration)": {
            "kappa0": round(kappa0, 3), "R2": round(R2_se0_noint, 3),
            "residual_sd": round(resid_sd0, 3), "intercept": "constrained to zero",
            "affine_alternative": {"alpha0": round(intercept0, 4),
                                   "kappa0": round(kappa0_affine, 3), "R2": round(R2_se0, 3),
                                   "residual_sd": round(resid_sd0_affine, 3)},
        },
        "zero_intercept_refit_at_working_point": {
            "kappa": round(kappa_wp_noint, 3), "R2": round(R2_se_noint, 4),
            "affine_R2_for_comparison": round(R2_se, 4),
            "note": "constraining the intercept costs 0.0005 R2 here, so the comparative R2 "
                    "table keeps an intercept for every predictor (equal treatment) while the "
                    "planning tool uses the constrained one-parameter form.",
        },
        "residual_sd": round(resid_sd, 3),
        "scope": "models the split-shopping level of selection optimism, conditional on test-set "
                 "tuning already having happened; silent on feature-selection leakage and shift.",
    }

    def events_needed(delta, prev=0.30, a=0.0, k=kappa0, auc=0.5):
        for e in range(5, 40000):
            np_ = max(1, int(0.30 * e)); nn = max(1, int(0.30 * e * (1 - prev) / prev))
            if a + k * hm_se(auc, np_, nn) <= delta:
                return e
        return None
    # 计算器只用 counts-only 那套系数（拟合与使用同在 a=0.5 上，自洽）。
    mismatch_pred = intercept + kappa_fit * se0
    R2_mismatch = float(1 - ((delta_split - mismatch_pred) ** 2).sum()
                        / ((delta_split - delta_split.mean()) ** 2).sum())
    out["calculator"] = {
        "form": "expected Delta_split = kappa0 * HM_SE(0.5, exact held-out counts), fitted ON "
                "SE(0.5) with the intercept constrained to zero, because optimism must vanish as "
                "SE -> 0. The working-point fit belongs to Fig. 3 and must NOT be used here.",
        "kappa0_used": round(kappa0, 3),
        "R2_of_this_fit": round(R2_se0_noint, 3),
        "why_no_intercept": {
            "theory": "SE -> 0 implies Delta_split -> 0; the affine fit instead tends to "
                      f"{intercept0:+.4f}, which is not usable for extrapolation to large studies.",
            "intercept_not_significant": {"estimate": round(a0_hat, 4), "se": round(a0_se, 4),
                                          "ci95": [round(a0_ci[0], 4), round(a0_ci[1], 4)]},
            "cost_in_R2": {"affine": round(R2_se0, 4), "zero_intercept_refit": round(R2_se0_noint, 4)},
        },
        "min_minority_events": {f"delta<={x}": events_needed(x) for x in (0.05, 0.10, 0.15)},
        "SENSITIVITY_affine_fit": {
            "note": "the unconstrained affine fit, reported for sensitivity only.",
            "alpha0": round(intercept0, 4), "kappa0": round(kappa0_affine, 3),
            "R2": round(R2_se0, 3),
            "min_minority_events": {f"delta<={x}": events_needed(x, a=intercept0, k=kappa0_affine)
                                    for x in (0.05, 0.10, 0.15)},
        },
        "WRONG_drop_intercept_keep_affine_slope": {
            "note": "setting alpha to 0 while keeping the affine slope is neither fit; recorded "
                    "so nobody re-derives it by mistake.",
            "min_minority_events": {f"delta<={x}": events_needed(x, a=0.0, k=kappa0_affine)
                                    for x in (0.05, 0.10, 0.15)},
        },
        "prediction_interval_note": f"+/-{1.96*resid_sd0:.2f} AUC residual (95%)",
        "WRONG_transfer_kept_for_the_record": {
            "note": "applying the working-point (alpha, kappa) to SE(0.5), as the pre-round-26 "
                    "code did: neither of the two fits, and measurably worse.",
            "R2": round(R2_mismatch, 3),
            "mean_over_prediction_of_Delta": round(float((mismatch_pred - delta_split).mean()), 4),
            "min_minority_events": {f"delta<={x}": events_needed(x, a=intercept, k=kappa_fit)
                                    for x in (0.05, 0.10, 0.15)},
        },
    }
    # 计算器口径下，50 个队列里有多少个达不到 delta<=0.10 的门槛？（正文引用这个计数）
    need10 = events_needed(0.10)
    out["calculator"]["cohorts_below_delta0.10_target"] = int((d["minority"].values < need10).sum())
    out["calculator"]["patients_implied_at_prevalence_0.30"] = int(math.ceil(need10 / 0.30))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "reliability_predictor.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))

    # ── 图 ──
    FIGS.mkdir(exist_ok=True)
    figstyle.apply()
    pred = intercept + kappa_fit * se
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figstyle.SPAN, 2.9))
    ax1.scatter(pred, delta_split, c=figstyle.BLUE, s=15, alpha=0.85,
                edgecolor="white", linewidth=0.3)
    lim = [min(pred.min(), delta_split.min()), max(pred.max(), delta_split.max())]
    ax1.plot(lim, lim, "--", color="0.2", lw=1.1, label="identity")
    # 验证图用的是"以诚实 AUC 为工作点"的仿射拟合，轴标签必须写明这一点：
    # 计算器用的是另一套（零截距、null 工作点），两者不能互换。
    ax1.set_xlabel(r"predicted  $\hat\alpha+\hat{\kappa}\,\mathrm{SE}_{\mathrm{AUC}}$  at honest working point"
                   "\n" r"($\hat\alpha$=%+.3f, $\hat{\kappa}$=%.2f)" % (intercept, kappa_fit))
    ax1.set_ylabel(r"observed $\Delta_{\mathrm{split}}$")
    ax1.set_title("(a) Predicted vs. observed")
    # cluster CI 属于【样本内】R^2，不是 held-out-source R^2 的区间；两者必须分行写清，
    # 否则并排放置会让读者以为后者是前者的置信区间（第 5 轮外审就这么读的）。
    ax1.annotate("$n$=%d cohorts\n"
                 "in-sample $R^2$ = %.2f  [%.2f, %.2f]\n"
                 "  (cluster bootstrap over sources)\n"
                 "held-out-source $R^2$ = %.2f"
                 % (len(d), R2_se, np.percentile(boot, 2.5), np.percentile(boot, 97.5), oof_se),
                 xy=(0.04, 0.97), xycoords="axes fraction", va="top", ha="left", fontsize=6.5)
    ax1.legend(loc="lower right")
    bars = ["HM SE", "free\nmulti", "log\nevents", "log\n$n$", "log\ndim"]
    vals = [R2_se, R2_multi, R2_ev, R2_n, R2_d]
    cols = [figstyle.GREEN, figstyle.BLUE, figstyle.BLUE, figstyle.BLUE, figstyle.GREY]
    ax2.bar(range(len(bars)), vals, color=cols, width=0.72)
    ax2.set_xticks(range(len(bars))); ax2.set_xticklabels(bars)
    for i, v in enumerate(vals):
        ax2.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=6.8)
    ax2.text(3.45, 0.86, "class counts only\n($a$=0.5): $R^2$=%.2f" % R2_se0,
             fontsize=5.8, ha="center", va="center", color="0.3", linespacing=1.1)
    ax2.set_ylabel(r"$R^2$ predicting $\Delta_{\mathrm{split}}$"); ax2.set_ylim(0, 1)
    ax2.set_title("(b) Variance explained")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(FIGS / "reliability_validation.png")
    fig.savefig(FIGS / "reliability_validation.pdf"); plt.close(fig)
    print("\nsaved -> results/reliability_predictor.json , figures/reliability_validation.png/.pdf")


if __name__ == "__main__":
    main()
