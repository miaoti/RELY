"""把工具箱（闸门 + 计算器）真正跑在 AIS 队列上，把 §V 的每个数字落成产物。

round-26 新增。此前 §V 里的"预测 0.13 / 闸门分 0.24 / 阈值 0.28 / 需要 97 事件"是手算的，
既没有产物、也没进核对脚本，正是最容易和标定口径脱节的地方（第 4 轮外审抓到的就是这个：
计算器混用了两套回归标定）。现在这些数一律从 results/*.json 现读现算。

依赖顺序：必须在 reliability_predictor.py 与 screening_gate.py 之后运行。
输出 results/ais_toolkit.json。
"""
from __future__ import annotations
import json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def hm_se(a, n_pos, n_neg):
    a = min(max(a, 1e-6), 1 - 1e-6)
    q1 = a / (2 - a); q2 = 2 * a * a / (1 + a)
    return math.sqrt(max((a * (1 - a) + (n_pos - 1) * (q1 - a ** 2)
                          + (n_neg - 1) * (q2 - a ** 2)) / (n_pos * n_neg), 1e-12))


def main():
    rp = json.loads((RESULTS / "reliability_predictor.json").read_text(encoding="utf-8"))
    gt = json.loads((RESULTS / "screening_gate.json").read_text(encoding="utf-8"))
    ais = json.loads((RESULTS / "ais_forensic.json").read_text(encoding="utf-8"))

    n, events = int(ais["n"]), int(ais["events"])
    # 与 benchmark 完全同一套 70/30 分层留出计数
    te_pos = int(round(0.30 * events))
    te_neg = int(round(0.30 * (n - events)))

    c0 = rp["counts_only_fit (a=0.5; THIS is the calculator's calibration)"]
    k0 = float(c0["kappa0"])            # 零截距重拟合；截距被约束为 0
    a0 = 0.0
    se0 = hm_se(0.5, te_pos, te_neg)

    gkey = [x for x in gt["results"] if x.startswith("selection")][0]
    gate_c = float(gt["kappa"])                      # 只定单位的常数 sqrt(2 ln B)
    thr = float(gt["results"][gkey]["operating_point_Youden_insample"]["threshold_risk"])
    score = gate_c * se0

    cal = rp["calculator"]
    need10 = cal["min_minority_events"]["delta<=0.1"]
    predicted = a0 + k0 * se0
    observed = float(ais["delta_selection"])

    out = {
        "note": "the toolkit run end to end on the AIS cohort, before any AUC exists. "
                "Every input is read from the other artifacts; nothing is hard-coded.",
        "n": n, "events": events, "held_out_counts": {"pos": te_pos, "neg": te_neg},
        "SE_null_point": round(se0, 4),
        "calculator": {
            "calibration": {"kappa0": k0, "R2": c0["R2"], "intercept": "constrained to zero",
                            "fitted_on": "SE at a=0.5, the only working point available "
                                         "before modeling"},
            "predicted_delta_split": round(predicted, 4),
            "observed_delta_split": round(observed, 4),
            "residual": round(observed - predicted, 4),
            "residual_sd_of_fit": c0["residual_sd"],
            "within_1_residual_sd": bool(abs(observed - predicted) <= float(c0["residual_sd"])),
            "min_events_for_delta_0.10": need10,
            "events_available": events,
            "verdict": "warns" if events < need10 else "clears",
        },
        "gate": {
            "score": round(score, 4), "threshold": round(thr, 4), "c_units_only": gate_c,
            "flags": bool(score >= thr),
            "label_is_true": bool(observed >= 0.15),
            "correct": bool((score >= thr) == (observed >= 0.15)),
        },
    }
    (RESULTS / "ais_toolkit.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
