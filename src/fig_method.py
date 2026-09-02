"""
Methods 图（Fig.2，跨栏）——两条评测臂：怎么切数据，各自报告什么。

设计原则（round-20 重做，旧版被批"就是几个文字框，和纯文字没区别"）：
  ① 用【视觉编码】而不是文字框。左半用分段色条画 train/test 的实际切分结构
     （sklearn 的 CV 可视化范式）：诚实臂 10 折里 1 折是 test（窄深条），
     选择臂 70/30 里 30% 是 test（宽深条）——两者的测试块宽度一眼可见。
  ② 右半用【真实数据】而不是示意：AIS 队列在建模预算 B=50 下的实测 tuned AUC 全量散点 +
     核密度，标出 mean 与 max，Δ 画成分布上的括号。数字从 results/ais_forensic.json 现读。
  ③ 图要能独立回答全文最容易被误读的一点：Δ = max_b − mean_b 只在【选择臂内部】取，
     两臂之间从不相减。所以 Δ 括号只画在右侧那一条分布上。

输出 figures/method.png(.pdf)。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import figstyle as fs

ROOT = Path(__file__).resolve().parents[1]
FIGS, RESULTS = ROOT / "figures", ROOT / "results"

TRAIN_H, TEST_H = fs.BAND_H, fs.C_HONEST      # honest arm: train / test
TRAIN_S, TEST_S = fs.BAND_C, fs.C_CHERRY      # selection arm: train / test


def split_rows(ax, y0, n_rows, row_h, gap, test_frac, kind, train_c, test_c, rng):
    """画 n_rows 条分段色条。kind='fold' 时 test 块依次右移（k 折），
    kind='random' 时 test 块随机落位（随机划分）。返回最后一行的 y。"""
    for i in range(n_rows):
        y = y0 - i * (row_h + gap)
        ax.add_patch(Rectangle((0, y), 1, row_h, fc=train_c, ec="none", zorder=2))
        if kind == "fold":
            x0 = i * test_frac
        else:
            x0 = float(rng.uniform(0, 1 - test_frac))
        ax.add_patch(Rectangle((x0, y), test_frac, row_h, fc=test_c, ec="none", zorder=3))
    return y0 - (n_rows - 1) * (row_h + gap)


def main():
    fs.apply()
    j = json.loads((RESULTS / "ais_forensic.json").read_text(encoding="utf-8"))
    # round-25：只画被建模的那个选择预算 B=50（数组里存了 500 个，但 kappa/闸门/计算器
    # 都是在 B=50 上拟合的，混用预算正是外审指出的问题）。
    B = int(j["protocol"]["B_modelled"])
    tuned = np.asarray(j["tuned_aucs"], float)[:B]
    mean_a, max_a = float(tuned.mean()), float(tuned.max())
    delta = max_a - mean_a
    honest = float(j["honest"])

    fig = plt.figure(figsize=(fs.SPAN, 2.45))
    gsL = fig.add_gridspec(2, 1, left=0.045, right=0.40, top=0.84, bottom=0.20, hspace=1.55)
    axH = fig.add_subplot(gsL[0]); axS = fig.add_subplot(gsL[1])
    axD = fig.add_axes([0.505, 0.16, 0.465, 0.64])

    rng = np.random.default_rng(7)

    # ---------------- (a) how each arm splits the cohort ----------------
    for ax, kind, tf, tr, te, title, sub in (
            (axH, "fold", 0.10, TRAIN_H, TEST_H, "HONEST ARM",
             r"10-fold $\times$ 10 repeats, nested; $(k,C)$ on an inner 5-fold"),
            (axS, "random", 0.30, TRAIN_S, TEST_S, "SELECTION ARM",
             r"50 random 70/30 splits; $(k,C)$ on the test block")):
        split_rows(ax, 0.78, 4, 0.17, 0.06, tf, kind, tr, te, rng)
        ax.text(0.0, 1.22, title, fontsize=6.8, fontweight="bold",
                color=te, ha="left", va="center", transform=ax.transAxes)
        ax.text(0.0, -0.22, sub, fontsize=5.6, color="0.30",
                ha="left", va="center", transform=ax.transAxes)
        ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.02); ax.set_axis_off()

    axH.text(1.012, 0.40, r"$\vdots$", fontsize=7, color="0.45",
             ha="left", va="center", transform=axH.transAxes)
    axS.text(1.012, 0.40, r"$\vdots$", fontsize=7, color="0.45",
             ha="left", va="center", transform=axS.transAxes)

    # shared legend for the two bars
    axS.add_patch(Rectangle((0.00, -0.60), 0.05, 0.12, fc=TRAIN_S,
                            ec="none", transform=axS.transAxes, clip_on=False, zorder=5))
    axS.text(0.060, -0.540, "train", fontsize=5.6, color="0.25",
             transform=axS.transAxes, va="center")
    axS.add_patch(Rectangle((0.150, -0.60), 0.05, 0.12, fc=TEST_S,
                            ec="none", transform=axS.transAxes, clip_on=False, zorder=5))
    axS.text(0.210, -0.540, "test block (where the AUC is measured)", fontsize=5.6,
             color="0.25", transform=axS.transAxes, va="center")

    # ---------------- (b) what the selection arm actually produces ----------------
    y = rng.normal(0, 1, tuned.size)
    axD.scatter(tuned, y, s=3.6, c=fs.C_CHERRY, alpha=0.38, linewidths=0, zorder=2)
    axD.axvline(mean_a, color="0.25", lw=1.3, zorder=4)
    axD.axvline(max_a, color=fs.RED, lw=1.6, zorder=4)
    axD.axvline(honest, color=fs.C_HONEST, lw=1.3, ls=(0, (3, 2)), zorder=4)

    ymax = 3.6
    axD.set_xlim(0.30, 0.80); axD.set_ylim(-ymax, ymax * 1.42)
    axD.set_yticks([])
    axD.set_xlabel("AUC of one evaluation (AIS cohort, %d splits)" % B, fontsize=6.2, labelpad=1.5)
    axD.tick_params(axis="x", labelsize=6.0, pad=1.5)
    for sp in ("left", "right", "top"):
        axD.spines[sp].set_visible(False)

    # Delta bracket, drawn only on this one distribution
    yb = ymax * 0.98
    axD.add_patch(FancyArrowPatch((mean_a, yb), (max_a, yb), arrowstyle="<|-|>",
                                  mutation_scale=7, color=fs.RED, lw=1.3, zorder=6))
    axD.text((mean_a + max_a) / 2, yb + 0.42,
             r"$\Delta=%.2f$" % delta, fontsize=7.4, color=fs.RED,
             fontweight="bold", ha="center", va="bottom", zorder=6)

    axD.text(mean_a - 0.006, -ymax * 0.92, "mean\n%.2f" % mean_a, fontsize=5.9, color="0.25",
             ha="right", va="bottom", linespacing=1.15)
    axD.text(max_a + 0.006, -ymax * 0.92, "max\n%.2f" % max_a, fontsize=5.9, color=fs.RED,
             ha="left", va="bottom", linespacing=1.15, fontweight="bold")
    axD.text(honest - 0.007, ymax * 1.05, "honest arm\n%.2f" % honest, fontsize=5.9,
             color=fs.C_HONEST, ha="right", va="center", linespacing=1.15)

    axD.text(0.0, 1.115, "WHAT SHOPPING BUYS", fontsize=6.8, fontweight="bold",
             color=fs.RED, ha="left", va="center", transform=axD.transAxes)
    axD.text(0.0, -0.30, r"$\Delta$ is taken inside the selection arm: the two arms are never subtracted",
             fontsize=5.6, color="0.30", ha="left", va="center", transform=axD.transAxes)

    fig.savefig(FIGS / "method.png", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIGS / "method.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("saved -> figures/method.png/.pdf  (mean %.3f, max %.3f, delta %.3f)"
          % (mean_a, max_a, delta))


if __name__ == "__main__":
    main()
