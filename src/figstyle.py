"""
统一出版级图风格（IEEE 双栏会议）。所有论文图 import 它，保证字号/尺寸/配色一致、按最终列宽作图
（避免 includegraphics 缩放后字糊）。COL=单栏宽、SPAN=跨栏宽（英寸），字号 8pt≈IEEE caption。
"""
from __future__ import annotations
import matplotlib as mpl

COL = 3.5      # 单栏宽 (in)
SPAN = 7.16    # 跨双栏宽 (in)

# 色盲友好（Okabe-Ito 子集）
BLUE, ORANGE, GREEN, RED, GREY, PURPLE = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#7f7f7f", "#CC79A7"
# 概念固定专色（跨图一致，降低读者重学成本）
C_HONEST, C_CHERRY, C_CHANCE, C_NULL, C_LUCK = GREEN, RED, "#555555", BLUE, ORANGE
COHORT = [BLUE, ORANGE, GREEN, RED]   # 4 队列固定顺序色


def apply():
    mpl.rcParams.update({
        "figure.dpi": 200, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.5,
        "lines.linewidth": 1.5, "lines.markersize": 4, "legend.frameon": True,
        "legend.framealpha": 0.92, "legend.edgecolor": "0.8", "axes.linewidth": 0.8,
        # softer, more polished spines/ticks (layout-preserving)
        "axes.edgecolor": "#3a3a3a", "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a",
        "axes.labelcolor": "#1a1a1a", "axes.titleweight": "bold", "axes.axisbelow": True,
    })


def panel(ax, label, dx=-0.16, dy=1.02):
    """子图角标 (a)/(b)。"""
    ax.text(dx, dy, label, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="right")
