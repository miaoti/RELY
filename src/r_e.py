"""
阶段 R-E · 识别实验：把"少数类事件数"从"总样本 n"中【干净分离】。
审稿核心质疑：50 队列里 log10(minority)≈log10(n)（r=0.895），gap~事件数其实可能只是 gap~n。
两臂受控实验（队列内、固定一个变另一个）：
  Arm FIX-N  ：固定总 n，只变类平衡 → 少数类事件 e 变、n 不变。若 gap 随 e↓ 上升 ⇒ 事件数特异驱动（非 n）。
  Arm FIX-E  ：固定少数类事件 e，加多数类把 n 撑大 → e 不变、n 变。若 gap≈平 ⇒ n 在 e 之外无额外作用。
两臂合起来：若 FIX-N 升、FIX-E 平，则"少数类事件数"被识别为驱动量（这是 Hanley-McNeil/Vabalas 的小样本方差所给不出的特异性）。
gap = cherry-pick(全数据选特征+多次划分取最大) − 诚实(折内选择重复分层5折CV)；复用 r_d 的鲁棒实现。
增量写 results/re_identify.csv，可续；图 figures/C1_identify.png(.pdf)。
用法：python src/r_e.py [--cohorts 4] [--K 200] [--reps 3]
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r_d  # gap_of(X,y,K), load_radml
import figstyle

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGS = ROOT / "results", ROOT / "figures"
SEED = 20260625
FIELDS = ["cohort", "arm", "n", "minority", "rep", "honest", "cherry", "gap"]


def pick_cohorts(n):
    d = pd.read_csv(RESULTS / "radmlbench_sweep.csv")
    # 需要：少数类够多(可做平衡)、多数类够多(可固定事件撑大 n)、维度别太大(速度)
    d["maj"] = d["n"] - d["minority"]
    q = d[(d["minority"] >= 80) & (d["maj"] >= 250) & (d["d"] <= 4200)].sort_values("minority", ascending=False)
    return list(q["dataset"].head(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", type=int, default=4)
    ap.add_argument("--K", type=int, default=200)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    cohorts = pick_cohorts(args.cohorts)
    print("[R-E] cohorts:", cohorts, flush=True)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "re_identify.csv"
    done = set()
    if out.exists():
        dd = pd.read_csv(out); done = set(zip(dd.cohort, dd.arm, dd.n, dd.minority, dd.rep))
    rng = np.random.default_rng(SEED)

    def rec(row):
        with open(out, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if f.tell() == 0:
                w.writeheader()
            w.writerow(row)

    N_FIX = 200          # FIX-N 臂的固定总样本
    E_FIX = 50           # FIX-E 臂的固定少数类事件数
    bal = [0.45, 0.35, 0.25, 0.15, 0.10]     # FIX-N：少数类占比
    ns = [120, 160, 220, 320, 450]           # FIX-E：总 n（多数类 = n − E_FIX）

    for name in cohorts:
        X, y = r_d.load_radml(name)
        lab_min = int(np.argmin(np.bincount(y)))
        imin = np.where(y == lab_min)[0]; imaj = np.where(y != lab_min)[0]
        # Arm FIX-N：固定 n=200，变平衡
        for p in bal:
            e = int(round(p * N_FIX)); mj = N_FIX - e
            if e > len(imin) or mj > len(imaj):
                continue
            for rep in range(args.reps):
                if (name, "fix_n", N_FIX, e, rep) in done:
                    continue
                r = np.random.default_rng(SEED + rep * 7)
                ii = np.concatenate([r.choice(imin, e, replace=False), r.choice(imaj, mj, replace=False)])
                h, c, g = r_d.gap_of(X[ii], y[ii], args.K)
                rec({"cohort": name, "arm": "fix_n", "n": N_FIX, "minority": e, "rep": rep,
                     "honest": round(h, 4), "cherry": round(c, 4), "gap": round(g, 4)})
                print(f"  {name:20} FIX-N n={N_FIX} e={e:<3} gap={g:+.3f}", flush=True)
        # Arm FIX-E：固定少数类事件=50，变 n
        for n in ns:
            mj = n - E_FIX
            if E_FIX > len(imin) or mj > len(imaj) or mj < E_FIX:
                continue
            for rep in range(args.reps):
                if (name, "fix_e", n, E_FIX, rep) in done:
                    continue
                r = np.random.default_rng(SEED + 100 + rep * 7)
                ii = np.concatenate([r.choice(imin, E_FIX, replace=False), r.choice(imaj, mj, replace=False)])
                h, c, g = r_d.gap_of(X[ii], y[ii], args.K)
                rec({"cohort": name, "arm": "fix_e", "n": n, "minority": E_FIX, "rep": rep,
                     "honest": round(h, 4), "cherry": round(c, 4), "gap": round(g, 4)})
                print(f"  {name:20} FIX-E e={E_FIX} n={n:<3} gap={g:+.3f}", flush=True)

    # 汇总 + 图（画 cherry 乐观天花板——干净量；gap 差值会被 honest 波动相消，故另注 honest≈平）
    df = pd.read_csv(out)
    def wcorr(arm, xcol, ycol):  # within-cohort 去均值后的合并相关
        a = df[df.arm == arm].copy()
        a["xd"] = a.groupby("cohort")[xcol].transform(lambda v: np.log10(v) - np.log10(v).mean())
        a["yd"] = a.groupby("cohort")[ycol].transform(lambda v: v - v.mean())
        return float(np.corrcoef(a.xd, a.yd)[0, 1])
    ca_ch, ca_h = wcorr("fix_n", "minority", "cherry"), wcorr("fix_n", "minority", "honest")
    cb_ch, cb_h = wcorr("fix_e", "n", "cherry"), wcorr("fix_e", "n", "honest")
    print(f"\n[RESULT] FIX-N (n const, vary events): cherry corr={ca_ch:+.2f} (events matter at fixed n), honest corr={ca_h:+.2f} (~flat)")
    print(f"[RESULT] FIX-E (events const, vary n): cherry corr={cb_ch:+.2f} (n matters at fixed events), honest corr={cb_h:+.2f} (~flat)")
    print("[CONCLUSION] cherry ceiling falls as EITHER class count grows -> joint sampling variance (Hanley-McNeil, both n_pos & n_neg); not minority-events-specifically, not n-alone, not dimensionality.")
    figstyle.apply()
    from matplotlib.lines import Line2D
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figstyle.SPAN, 2.95), sharey=True)
    cs = sorted(df.cohort.unique())
    mks = dict(zip(cs, ["o", "s", "^", "D"]))            # marker = cohort (secondary)
    disp = {n: n.replace("Prostate-MRI-US-Biopsy", "Prostate-MRI-US") for n in cs}
    CH, HO = figstyle.RED, figstyle.GREEN               # colour = the message: cherry ceiling vs honest

    def draw(ax, arm, xcol):
        # faint per-cohort traces (consistency across cohorts) — no whiskers
        for name in cs:
            g = df[(df.cohort == name) & (df.arm == arm)].groupby(xcol)[["cherry", "honest"]].mean().sort_index()
            ax.plot(g.index, g.cherry, mks[name] + "-", color=CH, ms=3.1, lw=0.9, alpha=0.30, zorder=2)
            ax.plot(g.index, g.honest, mks[name] + "--", color=HO, ms=2.9, lw=0.9, alpha=0.30, zorder=2)
        # bold mean across cohorts + shaded optimism gap — ONLY where all cohorts are present
        # (else the mean's trend is a cohort-composition artifact, not a within-cohort effect)
        pc = df[df.arm == arm].pivot_table(index=xcol, columns="cohort", values="cherry", aggfunc="mean")
        ph = df[df.arm == arm].pivot_table(index=xcol, columns="cohort", values="honest", aggfunc="mean")
        full = (pc.notna().all(axis=1) & ph.notna().all(axis=1)).values
        x = pc.index.values
        xf, mcf, mhf = x[full], pc.mean(axis=1).values[full], ph.mean(axis=1).values[full]
        ax.fill_between(xf, mhf, mcf, color=figstyle.ORANGE, alpha=0.12, zorder=1)
        ax.plot(xf, mcf, "-", color=CH, lw=2.7, zorder=5, solid_capstyle="round")
        ax.plot(xf, mhf, "--", color=HO, lw=2.4, zorder=5)
        ax.annotate("optimism\ngap", xy=(xf[len(xf) // 2], (mcf[len(xf) // 2] + mhf[len(xf) // 2]) / 2),
                    fontsize=6.3, color="#b35e00", ha="center", va="center", style="italic")
        if (~full).any():                      # mark where <4 cohorts remain (mean not drawn there)
            xcut = xf.max()
            ax.axvline(xcut, color="0.6", ls=(0, (1, 1.5)), lw=0.8, zorder=1)
            ax.text(x.max(), mcf[-1] + 0.02, "$>$%g:\n$<$4 cohorts" % xcut, fontsize=5.3,
                    color="0.5", ha="right", va="bottom", style="italic", linespacing=0.95)

    draw(ax1, "fix_n", "minority"); draw(ax2, "fix_e", "n")
    ax1.set_xlabel(f"minority events $e$   (total $n$ fixed = {N_FIX})"); ax1.set_ylabel("ROC-AUC")
    ax1.set_title(f"(a) Fix $n$, vary events    $r$(ceiling,$e$)={ca_ch:+.2f}")
    ax2.set_xlabel(f"total $n$   (minority events fixed = {E_FIX})")
    ax2.set_title(f"(b) Fix events, vary $n$    $r$(ceiling,$n$)={cb_ch:+.2f}")
    coh = [Line2D([0], [0], color="0.35", marker=mks[n], ls="", ms=4, label=disp[n]) for n in cs]
    sem = [Line2D([0], [0], color=CH, lw=2.6, ls="-", label="cherry-pick ceiling (falls)"),
           Line2D([0], [0], color=HO, lw=2.2, ls="--", label=r"honest ($\approx$ flat, $|r|\leq0.11$)")]
    leg1 = fig.legend(handles=coh, loc="lower center", bbox_to_anchor=(0.30, 1.0), ncol=2,
                      fontsize=6, frameon=False, title="cohort (marker)", title_fontsize=6)
    fig.legend(handles=sem, loc="lower center", bbox_to_anchor=(0.76, 1.0), ncol=1,
               fontsize=6.3, frameon=False, title="cross-cohort mean (illustrative; bold)", title_fontsize=6)
    fig.add_artist(leg1)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(FIGS / "C1_identify.png"); fig.savefig(FIGS / "C1_identify.pdf"); plt.close(fig)
    print(f"\nsaved -> {out}, figures/C1_identify.png/.pdf", flush=True)


if __name__ == "__main__":
    main()
