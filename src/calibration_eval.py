"""
阶段 6a · 校准 + 可靠性图 F4（Brier/ECE 校准前后）。

主模型（SelectKBest k=20 + 逻辑回归 balanced）在分层 10 折下取折外预测概率，比较：
  uncalibrated / Platt(sigmoid，小样本默认) / isotonic(仅对照)。
报告各自 Brier、ECE，并画可靠性图 F4（无泄漏：校准在训练折内 fit）。
输出 results/calibration_ICC.json，图 figures/F4_reliability_ICC.png。
用法：python src/calibration_eval.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
import honest_eval as he

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGS = ROOT / "results", ROOT / "figures"


def main():
    X, y, _ = he.load_xy(ROOT / "data" / "ICC.csv", "Categories", 0)
    base = he.make_pipeline()
    base.set_params(select__k=20, clf__C=0.1)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=he.MASTER_SEED)
    methods = {
        "uncalibrated": base,
        "Platt(sigmoid)": CalibratedClassifierCV(base, method="sigmoid", cv=5),
        "isotonic": CalibratedClassifierCV(base, method="isotonic", cv=5),
    }
    out, curves = {}, {}
    for name, est in methods.items():
        p = cross_val_predict(est, X, y, cv=cv, method="predict_proba", n_jobs=3)[:, 1]
        brier, ece = he._calib_metrics(y, p)
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="uniform")
        out[name] = {"brier": round(brier, 4), "ece": round(ece, 4)}
        curves[name] = (mean_pred, frac_pos)
        print(f"{name:16}: Brier={brier:.4f}  ECE={ece:.4f}")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "calibration_ICC.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    FIGS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k:", lw=1, label="perfectly calibrated")
    colors = {"uncalibrated": "#d62728", "Platt(sigmoid)": "#2ca02c", "isotonic": "#1f77b4"}
    for name, (mp, fp) in curves.items():
        ax.plot(mp, fp, "o-", color=colors[name], ms=4,
                label=f"{name} (Brier {out[name]['brier']:.3f}, ECE {out[name]['ece']:.3f})")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency (poor outcome)")
    ax.set_title("F4 - ICC reliability diagram (before vs after calibration)")
    ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGS / "F4_reliability_ICC.png", dpi=200)
    fig.savefig(FIGS / "F4_reliability_ICC.pdf"); plt.close(fig)
    print("saved -> results/calibration_ICC.json , figures/F4_reliability_ICC.png")


if __name__ == "__main__":
    main()
