"""
Motivation figure (single column, Introduction) — the paper in one glance.
Story: ONE small radiomics cohort, evaluated two ways. The honest estimate is at chance,
while reporting the best of many evaluations of the SAME data reaches a publishable-looking
number; that selection optimism is the AUC's sampling variance in the class counts
(predictable BEFORE modeling), not the feature dimensionality.

round-18：图中每个数字都从 results/ 现读（ais_forensic.json / reliability_predictor.json），
不再硬编码；且 Δ 用的是与正文一致的【同协议内】选择乐观 max_b − mean_b，而不是跨协议差值。
Output figures/motivation.png(.pdf).
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import figstyle as fs

ROOT = Path(__file__).resolve().parents[1]
FIGS, RESULTS = ROOT / "figures", ROOT / "results"


def numbers():
    """从产物里读回图上要显示的每个数字（避免硬编码与正文脱节）。"""
    a = json.loads((RESULTS / "ais_forensic.json").read_text(encoding="utf-8"))
    r = json.loads((RESULTS / "reliability_predictor.json").read_text(encoding="utf-8"))
    rr = r["R2_Delta_split_vs"]
    return {
        "honest": a["honest"], "typical": a["tuned_mean"], "best": a["tuned_max"],
        "delta": a["delta_selection"], "events": a["events"], "n": a["n"],
        "nfeat": a["n_features"],
        "r2_se": rr["hanley_mcneil_SE (one predictor + intercept)"],
        "r2_d": rr["log10(dimensionality)"],
        # round-27：图上"predictable pre-modeling"这句话下面原来放的是 0.82，
        # 可那是以【每个队列自己的诚实 AUC】为工作点拟合的，建模前根本拿不到。
        # 真正 pre-modeling 的是仅凭类别计数的 0.67，以及留出来源外推的 0.64。
        "r2_counts_only": rr["hanley_mcneil_SE at a=0.5 (class counts only)"],
        "r2_heldout_source": r["robustness_R2"]["TRUE_leave_one_source_out_prediction (SE at a=0.5)"],
    }

BAND = "#eef2f7"      # bottom insight band fill
COHORT_FC = "#eaf1f8"
H_FC, C_FC = "#e8f5ef", "#fdece6"   # honest / cherry card tints


def fit_text(ax, x, y, txt, max_w, fontsize, **kw):
    """在卡片宽度 max_w（axes 坐标）内放置居中文字；超宽就自动缩字号。
    round-19：'REPORTED (BEST OF MANY)' 曾以 6.7pt 溢出 0.40 宽的卡片，
    这里改成量过再放，避免以后换文案又溢出。"""
    t = ax.text(x, y, txt, ha="center", va="center", fontsize=fontsize, **kw)
    fig = ax.figure
    fig.canvas.draw()
    for _ in range(12):
        bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
        w_ax = bb.transformed(ax.transAxes.inverted()).width
        if w_ax <= max_w * 0.92:
            break
        fontsize -= 0.2
        t.set_fontsize(fontsize)
    return t


def card(ax, x, y, w, h, ec, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.02",
                                fc=fc, ec=ec, lw=1.3, zorder=2))


def main():
    fs.apply()
    N = numbers()
    fig, ax = plt.subplots(figsize=(fs.COL, 3.7))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_axis_off()

    # ---------- cohort (top) ----------
    card(ax, 0.045, 0.875, 0.91, 0.108, fs.BLUE, COHORT_FC)
    ax.text(0.50, 0.948, "one small radiomics cohort", ha="center", va="center",
            fontsize=8, fontweight="bold")
    ax.text(0.50, 0.910, r"$n_+ = %d$ events,   $n_- = %d$,   %d features"
            % (N["events"], N["n"] - N["events"], N["nfeat"]),
            ha="center", va="center", fontsize=6.8, color="0.18")

    # ---------- fork arrows ----------
    ax.annotate("", xy=(0.24, 0.75), xytext=(0.44, 0.862),
                arrowprops=dict(arrowstyle="-|>", color=fs.C_HONEST, lw=1.8,
                                connectionstyle="arc3,rad=0.20"))
    ax.annotate("", xy=(0.76, 0.75), xytext=(0.56, 0.862),
                arrowprops=dict(arrowstyle="-|>", color=fs.C_CHERRY, lw=1.8,
                                connectionstyle="arc3,rad=-0.20"))
    ax.text(0.205, 0.815, "honest\nnested CV", ha="right", va="center",
            fontsize=7, color=fs.C_HONEST, fontweight="bold")
    ax.text(0.785, 0.815, "shop the best\nof many splits", ha="left", va="center",
            fontsize=7, color=fs.C_CHERRY, fontweight="bold")

    # ---------- two outcome cards ----------
    card(ax, 0.045, 0.50, 0.40, 0.25, fs.C_HONEST, H_FC)
    card(ax, 0.555, 0.50, 0.40, 0.25, fs.C_CHERRY, C_FC)
    # honest
    fit_text(ax, 0.245, 0.715, "HONEST EVALUATION", 0.40, 6.7,
             color=fs.C_HONEST, fontweight="bold")
    ax.text(0.245, 0.625, "%.2f" % N["honest"], ha="center", va="center",
            fontsize=20, color=fs.C_HONEST, fontweight="bold")
    ax.text(0.245, 0.558, r"$\approx$ chance", ha="center", va="center",
            fontsize=6.8, color="0.18")
    ax.text(0.245, 0.520, "(leakage-free)", ha="center", va="center",
            fontsize=6.5, color="0.30", style="italic")
    # cherry
    fit_text(ax, 0.755, 0.715, "REPORTED EVALUATION", 0.40, 6.7,
             color=fs.C_CHERRY, fontweight="bold")
    ax.text(0.755, 0.625, "%.2f" % N["best"], ha="center", va="center",
            fontsize=20, color=fs.C_CHERRY, fontweight="bold")
    ax.text(0.755, 0.558, "looks like signal", ha="center", va="center",
            fontsize=6.8, color="0.18")
    ax.text(0.755, 0.520, "(typical split: %.2f)" % N["typical"], ha="center", va="center",
            fontsize=6.5, color="0.30", style="italic")

    # ---------- optimism gap (explicit arithmetic in a pill — unambiguous) ----------
    ax.add_patch(FancyBboxPatch((0.255, 0.378), 0.49, 0.094,
                 boxstyle="round,pad=0.004,rounding_size=0.02", fc="#fdece6", ec="#b0410d", lw=1.2, zorder=2))
    ax.text(0.50, 0.448, "selection optimism", ha="center", va="center",
            fontsize=7.6, color="#b0410d", fontweight="bold")
    ax.text(0.50, 0.407, r"$%.2f - %.2f = %.2f$" % (N["best"], N["typical"], N["delta"]),
            ha="center", va="center",
            fontsize=8.6, color="#b0410d", fontweight="bold")

    # arrow into the insight band
    ax.annotate("", xy=(0.50, 0.312), xytext=(0.50, 0.372),
                arrowprops=dict(arrowstyle="-|>", color="0.3", lw=1.4))

    # ---------- insight band (bottom) ----------
    ax.add_patch(FancyBboxPatch((0.02, 0.004), 0.96, 0.296,
                 boxstyle="round,pad=0.004,rounding_size=0.02", fc=BAND, ec="0.75", lw=0.9, zorder=1))
    ax.text(0.50, 0.262, "this gap is sampling variance, set by class counts,", ha="center", va="center",
            fontsize=7.7, fontweight="bold")
    ax.text(0.50, 0.216, "not dimensionality; predictable pre-modeling:", ha="center", va="center",
            fontsize=7.7, fontweight="bold")
    ax.text(0.50, 0.156, r"$\Delta \;\approx\; \kappa\,\cdot\,\mathrm{SE}_{\mathrm{AUC}}(n_+, n_-)$",
            ha="center", va="center", fontsize=10)
    # round-27：这一带打的是"建模前可预测"，所以主位数字必须是【仅凭类别计数】的 0.67；
    # 0.82 是以每个队列自己的诚实 AUC 为工作点拟合的，建模前拿不到，降为下方括注。
    ax.text(0.295, 0.100, "closed-form SE\n$R^2\\!=\\!%.2f$" % N["r2_counts_only"],
            ha="center", va="center", fontsize=7, color=fs.GREEN, fontweight="bold")
    ax.text(0.705, 0.100, "dimensionality\n$R^2\\!=\\!%.2f$" % N["r2_d"],
            ha="center", va="center", fontsize=7, color="#555555", fontweight="bold")
    ax.text(0.50, 0.108, "vs", ha="center", va="center", fontsize=7.5, color="0.28")
    ax.text(0.50, 0.048, "across 50 cohorts, from class counts alone; %.2f predicting a held-out source."
            % N["r2_heldout_source"],
            ha="center", va="center", fontsize=5.0, color="0.42", style="italic")
    ax.text(0.50, 0.020, "The higher %.2f evaluates the SE at each cohort's own honest AUC, so it needs a model first."
            % N["r2_se"],
            ha="center", va="center", fontsize=5.0, color="0.42", style="italic")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(FIGS / "motivation.png")
    fig.savefig(FIGS / "motivation.pdf"); plt.close(fig)
    print("saved -> figures/motivation.png/.pdf")


if __name__ == "__main__":
    main()
