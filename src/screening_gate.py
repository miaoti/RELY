"""
事前筛查闸门（decision-utility · 把"描述性预测"升级成"有实测性能的工具"）。
仅用【投稿前就知道的量】(n_pos, n_neg, 选择预算 B)——不跑任何模型——给每个队列一个风险分：
    risk = κ·SE_HM(AUC=0.5, n_pos, n_neg),  κ=√(2 ln B)
即"在 chance 工作点上，split-shopping 期望能把 AUC 抬高多少"。ROC-AUC 对 κ 这个单调因子不变，
故 B 的取值不影响闸门性能。

── round-18 标签修订（回应外部审稿）────────────────────────────────────────────
主标签改为【可直接测量的决策量】：Δ_selection ≥ δ（δ=0.15），即"光靠在多个评测里挑最好的，
报告 AUC 就会被抬高至少 δ"。

为什么换掉旧的 "honest 95%CI 覆盖 0.5"：
  ① 旧的 honest_lo/hi 是【逐折 AUC 的 2.5–97.5 百分位】，折之间相关、每折测试集很小，
     那是折间波动的离散度，不是置信区间——在多个队列上它宽到 [0.000, 1.000]。
  ② 更根本地：大样本真无信号的数据会精确估出 AUC≈0.5 并正确覆盖 0.5，那是【评测成功】，
     不是"不可评测"。用覆盖 0.5 当"不可靠"的标签，逻辑上就把成功当成了失败。
现在 honest CI 用的是患者级 bootstrap（见 honest_eval._pooled_oof_ci），仍作次要标签报告，
但主标签是 Δ_selection ≥ δ。
输出 results/screening_gate.json，图 figures/screening_gate_roc.png(.pdf)。
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGS = ROOT / "results", ROOT / "figures"
B = 50            # 与主协议同一个选择预算（round-25：原来是无关的 750）


def hm_se(auc, p, n):
    auc = min(max(auc, 1e-6), 1 - 1e-6)
    q1 = auc / (2 - auc); q2 = 2 * auc * auc / (1 + auc)
    return math.sqrt(max((auc * (1 - auc) + (p - 1) * (q1 - auc ** 2) + (n - 1) * (q2 - auc ** 2)) / (p * n), 1e-12))


def delete_one_auc(score, label, src):
    """delete-one **sensitivity** range: drop one source and re-score the REMAINING cohorts.

    round-23: this used to be reported as "leave-one-source-out AUC", which it is not.
    It measures how much one source influences the in-sample AUC, not held-out performance.
    The genuinely out-of-sample quantity is `heldout_source_auc` below (and the LOSO
    operating point, which was already out-of-sample)."""
    aucs = []
    for s_ in np.unique(src):
        m = src != s_
        if len(np.unique(label[m])) == 2:
            aucs.append(roc_auc_score(label[m], score[m]))
    return float(np.mean(aucs)), float(np.min(aucs)), float(np.max(aucs))


def heldout_source_auc(score, label, src):
    """真·留出来源 AUC：把每个 source 的队列留出，只在【留出的那些队列】上算 AUC，
    再对能算的 source 取平均。因为风险分不需要拟合，这衡量的是"分数在没见过的来源上
    还能不能排序"，而不是删掉一个来源后剩下的数据有多好排。
    只有同时含正负标签的 source 才可评（其余记为不可评，如实报告）。"""
    aucs, n_eval, n_skip = [], 0, 0
    for s_ in np.unique(src):
        te = src == s_
        if te.sum() >= 2 and len(np.unique(label[te])) == 2:
            aucs.append(roc_auc_score(label[te], score[te])); n_eval += 1
        else:
            n_skip += 1
    if not aucs:
        return float("nan"), 0, n_skip
    return float(np.mean(aucs)), n_eval, n_skip


def loso_operating_point(score, label, src):
    """样本外操作点：对每个源，用其余源选 Youden 阈值，应用到留出源，池化预测 → out-of-sample sens/spec。"""
    pred = np.full(len(label), np.nan)
    for s in np.unique(src):
        tr, te = src != s, src == s
        if len(np.unique(label[tr])) < 2:
            continue
        fpr, tpr, thr = roc_curve(label[tr], score[tr]); tau = float(thr[int(np.argmax(tpr - fpr))])
        pred[te] = (score[te] >= tau).astype(float)
    ok = ~np.isnan(pred); pr = pred[ok].astype(int); lb = label[ok]
    tp = int(((pr == 1) & (lb == 1)).sum()); fn = int(((pr == 0) & (lb == 1)).sum())
    tn = int(((pr == 0) & (lb == 0)).sum()); fp = int(((pr == 1) & (lb == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan"); spec = tn / (tn + fp) if tn + fp else float("nan")
    return {"sensitivity": round(sens, 3), "specificity": round(spec, 3), "n_evaluated": int(ok.sum())}


def main():
    import re
    d = pd.read_csv(RESULTS / "radmlbench_sweep.csv")
    # round-25：精确的分层留出计数 + 正类方向取 Target==1（JSON 一直这么声明，代码却还在
    # 用 minority 和 round(0.3n)）。Hanley--McNeil 对两个计数不对称，口径必须和预测器一致。
    from cohort_counts import counts_table
    ct = counts_table([str(x) for x in d["dataset"]]).set_index("dataset")
    tpos = np.array([int(ct.loc[str(x), "te_pos"]) for x in d["dataset"]])
    tneg = np.array([int(ct.loc[str(x), "te_neg"]) for x in d["dataset"]])
    kappa = math.sqrt(2 * math.log(B))
    risk = np.array([kappa * hm_se(0.5, p, n) for p, n in zip(tpos, tneg)])  # 事前风险分（仅类别计数）
    src = np.array([re.split(r'[-_]', x)[0] for x in d["dataset"]])
    src = np.array([re.sub(r'\d+.*$', '', s) or s for s in src])

    DELTA = 0.15
    labels = {  # 主标签 = 可直接测量的选择膨胀量；次要标签保留作更难的对照
        "selection inflation (Delta_selection >= %.2f)" % DELTA:
            (d["delta_selection"] >= DELTA).astype(int).values,
        "fabricate (max>=0.80 & honest<0.65)":
            ((d["test_selected_max"] >= 0.80) & (d["honest_pooled_auc"] < 0.65)).astype(int).values,
    }
    legname = {"selection inflation (Delta_selection >= %.2f)" % DELTA:
                   r"selection optimism $\Delta\geq%.2f$" % DELTA,
               "fabricate (max>=0.80 & honest<0.65)": "high-optimism (harder label)"}
    legstyle = {"selection inflation (Delta_selection >= %.2f)" % DELTA:
                    dict(color=figstyle.GREEN, lw=2.6, zorder=5),
                "fabricate (max>=0.80 & honest<0.65)":
                    dict(color="#9a9a9a", lw=1.0, alpha=0.85, zorder=2)}
    out = {"n_cohorts": int(len(d)), "B": B, "kappa": round(kappa, 3),
           "score": "risk = c * Hanley-McNeil SE(AUC=0.5, exact stratified held-out class counts from cohort_counts.exact_test_counts, n_pos = the class roc_auc_score treats as positive). ROC-AUC is invariant to the positive constant c, so this is a RANK score, not a calibrated expected optimism; the calculator (which does need a calibrated value) uses the fitted kappa instead.",
           "label_primary": "Delta_selection >= %.2f (measured: max_b - mean_b over the same splits x grid)" % DELTA,
           "caveat": "ROC-AUC is monotone-invariant to kappa. Score and label are both driven by the same "
                     "sampling variance, so the gate CALIBRATES the closed form rather than validating it "
                     "independently; what makes it usable is that the score is available before any modeling "
                     "while the label is not. Load-bearing: it holds across 50 real pipelines and contrasts "
                     "with dimensionality.",
           "decision_use": "a flag means split-shopping can inflate this cohort's reported AUC by >= delta; "
                           "treat any single favourable split as unverifiable and quote the calculator target.",
           "results": {}}
    figstyle.apply()
    fig, ax = plt.subplots(figsize=(figstyle.COL, figstyle.COL * 0.98))
    for name, lab in labels.items():
        pos = int(lab.sum())
        if pos == 0 or pos == len(lab):
            out["results"][name] = {"positives": pos, "note": "degenerate"}
            continue
        auc = roc_auc_score(lab, risk)
        m, lo, hi = delete_one_auc(risk, lab, src)
        fpr, tpr, thr = roc_curve(lab, risk)
        j = int(np.argmax(tpr - fpr))                      # Youden 最优操作点
        tau = float(thr[j]); pred = (risk >= tau).astype(int)
        tp = int(((pred == 1) & (lab == 1)).sum()); fp = int(((pred == 1) & (lab == 0)).sum())
        tn = int(((pred == 0) & (lab == 0)).sum()); fn = int(((pred == 0) & (lab == 1)).sum())
        sens = tp / (tp + fn) if tp + fn else float("nan"); spec = tn / (tn + fp) if tn + fp else float("nan")
        ppv = tp / (tp + fp) if tp + fp else float("nan"); npv = tn / (tn + fn) if tn + fn else float("nan")
        ax.plot(fpr, tpr, label=f"{legname[name]} (AUC={auc:.2f})", **legstyle[name])
        if name.startswith("chance"):
            ax.plot(fpr[j], tpr[j], "o", ms=5.5, color="black", zorder=6)
            ax.annotate(f"operating point (in-sample)\nspec=1.0, sens={sens:.2f}", xy=(fpr[j], tpr[j]),
                        xytext=(0.15, 0.94), fontsize=6, ha="left", va="center",
                        arrowprops=dict(arrowstyle="->", lw=0.6, connectionstyle="arc3,rad=0.15"))
        ho_auc, ho_n, ho_skip = heldout_source_auc(risk, lab, src)
        out["results"][name] = {
            "positives": pos, "base_rate": round(pos / len(lab), 3),
            "roc_auc_in_sample": round(auc, 3),
            "delete_one_source_SENSITIVITY_range (NOT out-of-sample)":
                [round(lo, 3), round(hi, 3)],
            "heldout_source_auc_mean (genuinely out-of-sample)":
                (round(ho_auc, 3) if ho_auc == ho_auc else None),
            "heldout_sources_evaluable": ho_n, "heldout_sources_single_label": ho_skip,
            "operating_point_Youden_insample": {"threshold_risk": round(tau, 4), "sensitivity": round(sens, 3),
                                       "specificity": round(spec, 3), "PPV": round(ppv, 3), "NPV": round(npv, 3)},
            "operating_point_LOSO_outofsample": loso_operating_point(risk, lab, src)}
    ax.plot([0, 1], [0, 1], ls=":", color="0.5", lw=1, label="chance")
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title("Pre-modeling evaluability screening gate\n(class counts only; $n$=50 cohorts)")
    ax.legend(loc="lower right", framealpha=1.0, fontsize=5.9, borderaxespad=0.25,
              handlelength=1.2, labelspacing=0.3); ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    fig.tight_layout(); FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "screening_gate_roc.png"); fig.savefig(FIGS / "screening_gate_roc.pdf"); plt.close(fig)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "screening_gate.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print("saved -> results/screening_gate.json , figures/screening_gate_roc.png/.pdf")


if __name__ == "__main__":
    main()
