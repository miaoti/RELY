"""
统一出版级图风格（IEEE 双栏会议）。所有论文图 import 它，保证字号/尺寸/配色一致、按最终列宽作图
（避免 includegraphics 缩放后字糊）。COL=单栏宽、SPAN=跨栏宽（英寸），字号 8pt≈IEEE caption。
"""
from __future__ import annotations
import matplotlib as mpl

COL = 3.5      # 单栏宽 (in)
SPAN = 7.16    # 跨双栏宽 (in)

# ── 配色（round-29 收敛）───────────────────────────────────────────────────
# 全篇只有两个叙事专色：绿 = 诚实评测，深红 = 选择乐观 / 被挑出来的那一个。
# 其余一律中性：蓝 = 不带叙事含义的数据色，灰 = 参照线与被证伪的预测量。
# 橙色系（旧 #E69F00、#D55E00、#b0410d 及浅底 #fbdcc9、#fdece6）整体去掉：双栏印刷里过跳，
# 且橙与红在色觉缺陷下难分。四个有彩色相经调色板校验：最差相邻对 ΔE=12.8（deutan），
# 对比度全部 >3:1，亮度全部落在 L 0.43-0.77 带内。
BLUE, GREEN, RED, GREY, PURPLE = "#0072B2", "#009E73", "#B2182B", "#7f7f7f", "#6A51A3"
SLATE = "#5a6672"                                          # 中性描边（不带含义的框）
TINT_H, TINT_C, TINT_N = "#e8f3ee", "#f7e7ea", "#eef2f7"   # 绿 / 红 / 中性 浅底
BAND_H, BAND_C = "#cfe9dd", "#f0d2d7"                      # 色条底纹（比浅底深一档）
# 概念固定专色（跨图一致，降低读者重学成本）
C_HONEST, C_CHERRY, C_CHANCE, C_NULL = GREEN, RED, "#555555", BLUE
# 三个选择臂在所有图里同一套颜色与线型（Fig. 4(c) 与 dim_channel 共用）
C_INFOLD, C_LEAKY, C_LEAKY_FIXK = RED, PURPLE, BLUE
COHORT = [BLUE, GREEN, RED, PURPLE]   # 4 队列固定顺序色


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
