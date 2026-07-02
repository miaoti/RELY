"""
阶段 8 · 生成 T1 结果阶梯表（markdown）。读 results/ 下各产物，汇成 docs/T1_results.md。
（T2 合规自评表见 docs/T2_compliance.md，阶段 9 产出。）
用法：python src/tables.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS, DOCS = ROOT / "results", ROOT / "docs"


def jload(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def main():
    base = pd.read_csv(RESULTS / "baselines_ICC.csv")
    he_sum = jload("honest_eval_summary.json")
    cal = jload("calibration_ICC.json")
    perm = jload("permutation_ICC.json")
    stab = jload("stability_ICC.json")
    stats = jload("stats_ICC.json")
    c1l = jload("c1_law.json")
    sweep = pd.read_csv(RESULTS / "radmlbench_sweep.csv")

    L = []
    L.append("# T1 · ICC 结果阶梯（Honest Results Ladder）\n")
    L.append("> 全部为无泄漏重复分层嵌套 CV 的折外结果（master seed 20260625）。阳性=poor=`Categories==0`（57 事件）。\n")
    L.append("## (a) 诚实基线阶梯\n")
    L.append("| 特征集 | 特征数 | 诚实 AUC [95%] | NB-SE | Brier | ECE | 敏感度 | 特异度 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for _, r in base.iterrows():
        L.append(f"| {r['subset']} | {int(r['n_features'])} | {r['auc_mean']:.3f} "
                 f"[{r['auc_lo']:.2f},{r['auc_hi']:.2f}] | {r['nb_se']:.3f} | {r['brier']:.3f} "
                 f"| {r['ece']:.3f} | {r['sens']:.2f} | {r['spec']:.2f} |")
    L.append("\n**主模型 = 全 1004 影像组学**（预指定）。**简约基准 = 年龄+体积**。\n")

    L.append("## (b) 关键对比 / 统计\n")
    pa = stats["paired_auc"]
    L.append(f"- **主模型 vs 年龄+体积（配对 AUC 检验，NB 校正）**：{pa['auc_main']:.3f} vs "
             f"{pa['auc_age_volume']:.3f}，差 {pa['mean_diff']:+.3f}，t={pa['t']}, **p={pa['pvalue']}** "
             f"→ 影像组学显著更差。")
    L.append(f"- **乐观差距（同一数据，只改协议）**：诚实 {he_sum['honest_nested_cv']['mean']:.3f} → "
             f"测试集选模型 {he_sum['test_set_selected_single_split']['mean']:.3f} → "
             f"挑划分(best/300) {he_sum['naive_single_split']['max']:.3f}（gap "
             f"+{he_sum['naive_single_split']['max']-he_sum['honest_nested_cv']['mean']:.3f}）。")
    L.append(f"- **置换检验**：观测 AUC {perm['observed_auc']:.3f}，1000 次零分布，**p={perm['pvalue']:.3f}** → 与随机不可区分。")
    L.append(f"- **校准**：未校准 ECE {cal['uncalibrated']['ece']:.3f} → Platt ECE {cal['Platt(sigmoid)']['ece']:.3f}"
             f"（Brier {cal['uncalibrated']['brier']:.3f}→{cal['Platt(sigmoid)']['brier']:.3f}）；可良好校准但无区分力。")
    L.append(f"- **决策曲线（DCA）**：主模型/年龄+体积均无净获益（max NB {stats['dca']['max_nb_main']:.3f} / "
             f"{stats['dca']['max_nb_age_volume']:.3f}，阈值>患病率后塌到 treat-none）。")
    L.append(f"- **Hosmer-Lemeshow**：p={stats['hosmer_lemeshow_main']['pvalue']}（校准拟合无异常）。")
    L.append(f"- **C3 稳定性（附注，降权）**：sgl(分组) {stab['nogueira_stability']['sgl(group)']:.3f} > "
             f"flatL1 {stab['nogueira_stability']['flatL1']:.3f}，但**两者绝对值均≈0（噪声级，该队列无信号）**，仅作方法学附注、不当卖点。")
    rp = jload("reliability_predictor.json"); r2s = rp["R2_observed_gap_vs"]; rob = rp["robustness_R2"]
    sg = jload("screening_gate.json")
    r2_se = r2s["hanley_mcneil_SE (closed-form 1-param)"]
    r2_mv = r2s["free multivariate {log n_pos, log n_neg, AUC} (3-param)"]
    chance = sg["results"]["chance (honest CI covers 0.5)"]
    L.append(f"- **★ C1 可靠性预测器（头条，N={rp['n_cohorts']} 队列 / {rp['n_sources']} 源）**：报告型乐观(cherry-pick) 由"
             f"**闭式 Hanley-McNeil AUC SE**（仅 n_pos/n_neg/AUC）预测 **R²={r2_se:.2f}**"
             f"（LOSO {rob['leave_one_source_out']}、cluster-bootstrap CI {rob['cluster_bootstrap_95CI']}；自由多元(3参) {r2_mv:.2f}）"
             f" ＞ 事件 {r2s['log10(minority_events)']:.2f} ≈ n {r2s['log10(n)']:.2f}"
             f" ≫ **维度 {r2s['log10(dimensionality)']:.2f}** → **评测乐观由样本量主导；维度只经'选择空间'弱进入(非主因)，降维不是对的杠杆**。"
             f" ⚠️ gap≈κ·SE 部分构造性恒等；承重的是跨 50 真实管线成立 + 维度对比。")
    L.append(f"- **事前筛查闸门（decision-utility）**：仅凭类别计数(建模前) 预测'诚实评测将与随机不可分' "
             f"**ROC-AUC={chance['roc_auc']:.2f}**（LOSO {chance['loso_range']}）。计算器用实测有效 κ≈{rp['kappa_fitted']}"
             f"（有效预算 B≈{rp['effective_independent_selections_B_eff']:.0f}，非名义 750）：δ≤0.10 需 ≈214 少数类事件。")
    L.append(f"- **R-E/R-D 队列内识别（因果）**：固定 n 变平衡 gap 对事件近平 + 固定事件变 n gap 随 n 降 + 固定 n 丢特征维度弱效应"
             f" → **样本量是主导杠杆、维度次要**（已证伪 v1'事件数 specifically'）。")
    L.append(f"- **C2 可靠性警示（经验审计，非定律）**：边际共形可欠覆盖少数类，但跨 50 队列 欠覆盖~m 噪声大（r≈−0.18）；"
             f"**真正稳的是 set-size 退化**——少数类校准点少时类条件'修复'近全集（size≈1.8–1.9）→ **覆盖须与 set size 并报**。")
    rd = pd.read_csv(RESULTS / "rd_dose_response.csv")
    ce = np.corrcoef(np.log10(rd[rd.arm == "events"].level), rd[rd.arm == "events"].gap)[0, 1]
    cd = np.corrcoef(np.log10(rd[rd.arm == "dim"].level), rd[rd.arm == "dim"].gap)[0, 1]
    L.append(f"- **R-D 剂量-反应（队列内因果，3 队列）**：固定维度下采样事件→乐观升（corr {ce:+.2f}）；"
             f"固定 n 丢特征→维度弱效应（corr {cd:+.2f}）→ **事件数主导、维度次要**。")

    DOCS.mkdir(exist_ok=True)
    (DOCS / "T1_results.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("saved -> docs/T1_results.md")
    print("\n".join(L[:14]))


if __name__ == "__main__":
    main()
