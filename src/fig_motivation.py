"""
Motivation figure (single column, Introduction) — the paper in one glance.
Story: ONE small radiomics cohort, evaluated two ways, yields a large AUC gap (the
"reported optimism"); that gap is the AUC's sampling variance in the class counts
(predictable BEFORE modeling), not the feature dimensionality. Output figures/motivation.png(.pdf).
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import figstyle as fs

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "figures"

BAND = "#eef2f7"      # bottom insight band fill
COHORT_FC = "#eaf1f8"
H_FC, C_FC = "#e8f5ef", "#fdece6"   # honest / cherry card tints


def card(ax, x, y, w, h, ec, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.02",
                                fc=fc, ec=ec, lw=1.3, zorder=2))


def main():
    fs.apply()
    fig, ax = plt.subplots(figsize=(fs.COL, 3.7))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_axis_off()

    # ---------- cohort (top) ----------
    card(ax, 0.045, 0.875, 0.91, 0.108, fs.BLUE, COHORT_FC)
    ax.text(0.50, 0.948, "one small radiomics cohort", ha="center", va="center",
            fontsize=8, fontweight="bold")
    ax.text(0.50, 0.910, r"$n_+ = 57$ events,   $n_- = 126$,   1004 features",
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
    ax.text(0.785, 0.815, "cherry-pick\nbest of splits", ha="left", va="center",
            fontsize=7, color=fs.C_CHERRY, fontweight="bold")

    # ---------- two outcome cards ----------
    card(ax, 0.045, 0.50, 0.40, 0.25, fs.C_HONEST, H_FC)
    card(ax, 0.555, 0.50, 0.40, 0.25, fs.C_CHERRY, C_FC)
    # honest
    ax.text(0.245, 0.715, "HONEST EVALUATION", ha="center", va="center",
            fontsize=6.7, color=fs.C_HONEST, fontweight="bold")
    ax.text(0.245, 0.625, "0.47", ha="center", va="center",
            fontsize=20, color=fs.C_HONEST, fontweight="bold")
    ax.text(0.245, 0.558, r"$\approx$ chance", ha="center", va="center",
            fontsize=6.8, color="0.18")
    ax.text(0.245, 0.520, "(leakage-free)", ha="center", va="center",
            fontsize=6.5, color="0.30", style="italic")
    # cherry
    ax.text(0.755, 0.715, "OPTIMISTIC EVALUATION", ha="center", va="center",
            fontsize=6.7, color=fs.C_CHERRY, fontweight="bold")
    ax.text(0.755, 0.625, "0.67", ha="center", va="center",
            fontsize=20, color=fs.C_CHERRY, fontweight="bold")
    ax.text(0.755, 0.558, "looks like signal", ha="center", va="center",
            fontsize=6.5, color="0.18")
    ax.text(0.755, 0.520, "(split-shopping + leakage)", ha="center", va="center",
            fontsize=6.5, color="0.30", style="italic")

    # ---------- optimism gap (explicit arithmetic in a pill — unambiguous) ----------
    ax.add_patch(FancyBboxPatch((0.255, 0.378), 0.49, 0.094,
                 boxstyle="round,pad=0.004,rounding_size=0.02", fc="#fdece6", ec="#b0410d", lw=1.2, zorder=2))
    ax.text(0.50, 0.448, "reported optimism", ha="center", va="center",
            fontsize=7.6, color="#b0410d", fontweight="bold")
    ax.text(0.50, 0.407, r"$0.67 - 0.47 = 0.20$", ha="center", va="center",
            fontsize=8.6, color="#b0410d", fontweight="bold")

    # arrow into the insight band
    ax.annotate("", xy=(0.50, 0.312), xytext=(0.50, 0.372),
                arrowprops=dict(arrowstyle="-|>", color="0.3", lw=1.4))

    # ---------- insight band (bottom) ----------
    ax.add_patch(FancyBboxPatch((0.02, 0.015), 0.96, 0.285,
                 boxstyle="round,pad=0.004,rounding_size=0.02", fc=BAND, ec="0.75", lw=0.9, zorder=1))
    ax.text(0.50, 0.262, "the gap is sampling variance, predictable pre-data,", ha="center", va="center",
            fontsize=7.7, fontweight="bold")
    ax.text(0.50, 0.216, "set by class counts, not dimensionality:", ha="center", va="center",
            fontsize=7.7, fontweight="bold")
    ax.text(0.50, 0.156, r"$\Delta \;\approx\; \kappa\,\cdot\,\mathrm{SE}_{\mathrm{AUC}}(n_+, n_-)$",
            ha="center", va="center", fontsize=10)
    ax.text(0.50, 0.103, "validated across 50 radiomics cohorts:", ha="center", va="center",
            fontsize=6.6, color="0.22", style="italic")
    ax.text(0.295, 0.050, "closed-form SE\n$R^2\\!=\\!0.73$", ha="center", va="center",
            fontsize=7, color=fs.GREEN, fontweight="bold")
    ax.text(0.705, 0.050, "dimensionality\n$R^2\\!=\\!0.09$", ha="center", va="center",
            fontsize=7, color="#555555", fontweight="bold")
    ax.text(0.50, 0.058, "vs", ha="center", va="center", fontsize=7.5, color="0.28")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(FIGS / "motivation.png")
    fig.savefig(FIGS / "motivation.pdf"); plt.close(fig)
    print("saved -> figures/motivation.png/.pdf")


if __name__ == "__main__":
    main()
