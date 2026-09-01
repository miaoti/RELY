"""
阶段 R-E2 · 类别计数的 within-cohort 识别实验，改用【论文主结局】重跑（round-18）。

为什么要重跑：原 `r_e.py` 的 "ceiling" 用的是 `r_d.cherry_auc` —— 全数据选特征(泄漏) +
跨协议差值。外部审稿指出的核心问题正是"不同 endpoint 混在一起"，如果正文主结局已改成
同协议内的 Δ_selection = max_b − mean_b，那么识别实验也必须用同一个量，否则自相矛盾。

设计（与原 r_e.py 完全相同，只换 endpoint）：
  Arm FIX-N ：固定 n=200，只变类平衡（少数类事件 90→20）→ 事件数变、n 不变。
  Arm FIX-E ：固定少数类事件=50，加多数类把 n 撑大 → 事件数不变、n 变。
两臂特征空间自始至终固定，故任何 within-cohort 变化都不可能是维度造成的。
若【任一】类别计数缩小都会抬高 Δ_selection ⇒ 驱动量是 AUC 的联合抽样变异
（Hanley-McNeil 同时依赖 n_pos 与 n_neg），而不是"少数类事件数"单独作用。

输出 results/re_identify_selection.csv
用法：python src/r_e_selection.py [--cohorts 4] [--reps 3] [--splits 50]
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r_g_dim_channel as rg          # 复用 selection_optimism / honest_auc / load_radml

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEED = 20260625
N_FIX, E_FIX = 200, 50
FIELDS = ["cohort", "arm", "n", "minority", "rep", "honest",
          "sel_mean", "sel_sd", "sel_max", "delta_selection"]


def pick_cohorts(k):
    """够大的队列才能同时支撑 FIX-N 与 FIX-E 两个臂的下采样。与 r_e.py 同口径。"""
    cands = [RESULTS / "radmlbench_sweep.csv", RESULTS / "_archive" / "radmlbench_sweep_v1.csv"]
    tabs = [pd.read_csv(p) for p in cands if p.exists()]
    d = max(tabs, key=len)
    d = d[(d["minority"] >= 90) & (d["n"] >= 260)].sort_values("n", ascending=False)
    return list(d["dataset"].head(k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", type=int, default=4)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--splits", type=int, default=50)
    ap.add_argument("--fresh", action="store_true",
                    help="忽略断点续跑缓存：把已有产物归档到 results/_archive/ 后从零重跑")
    args = ap.parse_args()
    cohorts = pick_cohorts(args.cohorts)
    print(f"[R-E2] cohorts: {cohorts}  reps={args.reps} splits={args.splits}", flush=True)

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "re_identify_selection.csv"
    _fresh(out, getattr(args, "fresh", False))
    done = set()
    if out.exists():
        dd = pd.read_csv(out)
        done = set(zip(dd["cohort"], dd["arm"], dd["n"], dd["minority"], dd["rep"]))

    def record(row):
        with open(out, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if f.tell() == 0:
                w.writeheader()
            w.writerow(row)

    for name in cohorts:
        X, y = rg.load_radml(name)
        lab_min = int(np.argmin(np.bincount(y)))
        idx_min = np.where(y == lab_min)[0]; idx_maj = np.where(y != lab_min)[0]
        M, MJ = len(idx_min), len(idx_maj)

        for rep in range(args.reps):
            rng = np.random.default_rng(SEED + 1000 * rep)

            # ---- FIX-N：n 固定 200，少数类事件数下降 ----
            for e in (90, 70, 50, 35, 20):
                if e > M or (N_FIX - e) > MJ:
                    continue
                key = (name, "fix_n", N_FIX, e, rep)
                if key in done:
                    continue
                ii = np.concatenate([rng.choice(idx_min, e, replace=False),
                                     rng.choice(idx_maj, N_FIX - e, replace=False)])
                Xs, ys = X[ii], y[ii]
                h = rg.honest_auc(Xs, ys, repeats=5)
                m, sd, mx, delta = rg.selection_optimism(Xs, ys, leak=False, n_splits=args.splits)
                record({"cohort": name, "arm": "fix_n", "n": N_FIX, "minority": e, "rep": rep,
                        "honest": round(h, 4), "sel_mean": round(m, 4), "sel_sd": round(sd, 4),
                        "sel_max": round(mx, 4), "delta_selection": round(delta, 4)})
                print(f"  {name:22} FIX-N n=200 e={e:<3} Dsel={delta:+.4f} honest={h:.3f}", flush=True)

            # ---- FIX-E：少数类事件固定 50，n 增大 ----
            for n_tot in (120, 200, 320, 500, 700):
                nmaj = n_tot - E_FIX
                if E_FIX > M or nmaj > MJ or nmaj < 1:
                    continue
                key = (name, "fix_e", n_tot, E_FIX, rep)
                if key in done:
                    continue
                ii = np.concatenate([rng.choice(idx_min, E_FIX, replace=False),
                                     rng.choice(idx_maj, nmaj, replace=False)])
                Xs, ys = X[ii], y[ii]
                h = rg.honest_auc(Xs, ys, repeats=5)
                m, sd, mx, delta = rg.selection_optimism(Xs, ys, leak=False, n_splits=args.splits)
                record({"cohort": name, "arm": "fix_e", "n": n_tot, "minority": E_FIX, "rep": rep,
                        "honest": round(h, 4), "sel_mean": round(m, 4), "sel_sd": round(sd, 4),
                        "sel_max": round(mx, 4), "delta_selection": round(delta, 4)})
                print(f"  {name:22} FIX-E e=50  n={n_tot:<3} Dsel={delta:+.4f} honest={h:.3f}", flush=True)

    # ---- within-cohort 相关（与 r_e.py 同一 wcorr 口径：先按队列去均值再合并）----
    df = pd.read_csv(out)

    def wcorr(arm, xcol, ycol):
        s = df[df.arm == arm]
        g = s.groupby("cohort")
        xc = s[xcol] - g[xcol].transform("mean")
        yc = s[ycol] - g[ycol].transform("mean")
        return float(np.corrcoef(xc, yc)[0, 1])

    print("\n[R-E2 RESULT] endpoint = Delta_selection (in-fold, max-mean; the paper's endpoint)")
    for arm, xcol, lab in (("fix_n", "minority", "FIX-N (n=200 fixed, vary minority events)"),
                           ("fix_e", "n", "FIX-E (events=50 fixed, vary n)")):
        if not len(df[df.arm == arm]):
            continue
        print(f"  {lab}")
        print(f"     r(Delta_selection, {xcol}) = {wcorr(arm, xcol, 'delta_selection'):+.3f}"
              f"   r(honest, {xcol}) = {wcorr(arm, xcol, 'honest'):+.3f}")
    print("\n[CONCLUSION] if Delta_selection falls as EITHER class count grows while honest stays flat, "
          "the driver is the joint sampling variance of the AUC, not minority events alone and not d.")
    print(f"saved -> {out}")


def _fresh(out, enabled):
    """--fresh: 把已有产物挪进 results/_archive/，让断点续跑从零开始（可复现性要求）。"""
    if not enabled or not out.exists():
        return
    arch = out.parent / "_archive"
    arch.mkdir(exist_ok=True)
    i, dest = 0, arch / out.name
    while dest.exists():
        i += 1
        dest = arch / ("%s.%d%s" % (out.stem, i, out.suffix))
    out.replace(dest)
    print("[fresh] archived %s -> %s" % (out.name, dest))


if __name__ == "__main__":
    main()
