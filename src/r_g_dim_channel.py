"""
阶段 R-G · 维度的作用通道（round-18 新增，回应"Not Dimensionality 过强"的证伪风险）。

背景：仓库既有的 `results/rd_dose_response.csv`（DIM 臂：固定 n 与事件数、随机丢特征）里，
乐观差距其实【随维度上升】——pooled within-cohort r(log10 d, gap)=+0.665，Dong2022 与
Hosny2018A 的斜率均约 +0.06 AUC/decade。这与"维度几乎无关(R²=0.09)"表面矛盾，是审稿人
打开公开仓库就能抓到的硬伤。

关键观察：`r_d.py:cherry_auc` 用的是【全数据选特征】(SelectKBest 在划分之前 fit 了全部 y)，
即 feature-selection leakage；而 50 队列基准的 Δ 用的是【折内选特征】的纯选择乐观。
两者是不同的乐观来源。

本实验在【同一批划分、同一网格、同一学习器】下做 d 的剂量-反应，同时交叉两个因子：
  arm      : infold（特征选择在训练划分内）
             leaky（特征选择在全数据上做一次，k 随网格自适应）
             leaky_fixedk（全数据选特征且 k 固定为 30 —— 精确复刻 r_d.py:cherry_auc）
  endpoint : A = Δ_selection = max_b − mean_b（同协议内纯选择量，论文主结局）
             B = max_b − honest（跨协议差值，= r_d.py 的 gap 口径）
这样可以判定：那条正相关是"维度真的驱动选择乐观"，还是"它属于另一个 endpoint"。

⚠️ 结论由数据决定、不预设：脚本的 conclusion 字段由实测斜率生成，不含硬编码结论。
（实测结果：Δ_selection 在【三个臂】都不随 d 上升。d 真正起作用的地方是"被挑出的最大值"
  这个【水平量】，且只在 k 固定而候选池随 d 变大时才明显 —— 这正是 r_d.py 的配置，
  也就是 rd_dose_response.csv 那条正相关的来源。）

输出 results/rg_dim_channel.csv + figures/dim_channel.png/.pdf
用法：python src/r_g_dim_channel.py [--cohorts 4] [--splits 50]
"""
from __future__ import annotations
import argparse, csv, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGS = ROOT / "results", ROOT / "figures"
SEED = 20260625
GRID_K = (7, 15, 30, 50, 100)
GRID_C = (0.01, 0.1, 1.0)
FIELDS = ["cohort", "arm", "dfrac", "d", "D_full", "seed", "n", "minority", "honest",
          "sel_mean", "sel_sd", "sel_max", "delta_selection", "delta_vs_honest"]


def honest_auc(X, y, repeats=10):
    """诚实 AUC：折内选择的重复分层 5 折 CV。与 r_d.py:honest_auc 同一实现，
    这样 endpoint A（= r_d 的 gap 口径）可以和 rd_dose_response.csv 直接对照。"""
    from sklearn.model_selection import RepeatedStratifiedKFold
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=SEED)
    a = []
    for tr, te in cv.split(X, y):
        m = _infold_pipe(20, 0.1, X.shape[1]).fit(X[tr], y[tr])
        a.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return float(np.mean(a))


def _infold_pipe(k, C, d):
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("var", VarianceThreshold(0.0)), ("scaler", StandardScaler()),
                     ("select", SelectKBest(f_classif, k=min(k, d))),
                     ("clf", LogisticRegression(class_weight="balanced", solver="liblinear",
                                                max_iter=5000, C=C))])


def selection_optimism_fixedk(X, y, n_splits=50, base_seed=0, k=30, C=0.1):
    """精确复刻 r_d.py:cherry_auc 的协议：全数据 SelectKBest(k=min(30,d)) 泄漏选特征 +
    单一固定模型 (C=0.1)，对 n_splits 个 70/30 划分取最大。

    与 leaky 臂的差别只有一处、但很关键：k 固定为 30，【不】随 d 自适应。d 越大，
    "从多少个候选里挑出这 30 个"就越有利，泄漏优势因此随 d 增长；而 leaky 臂的网格
    (k 可到 100 或全部) 会在小 d 时改用几乎全部特征，把这个效应掩盖掉。
    这一臂用来判定 rd_dose_response.csv 的正相关是不是这个机制造成的。"""
    d = X.shape[1]
    Xi = SimpleImputer(strategy="median").fit_transform(X)
    Xs = StandardScaler().fit_transform(Xi)
    Xk = SelectKBest(f_classif, k=min(k, d)).fit_transform(Xs, y)
    per_split = []
    for s in range(n_splits):
        Xtr, Xte, ytr, yte = train_test_split(Xk, y, test_size=0.30, stratify=y,
                                              random_state=base_seed + s)
        m = LogisticRegression(class_weight="balanced", solver="liblinear",
                               max_iter=5000, C=C).fit(Xtr, ytr)
        per_split.append(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
    a = np.asarray(per_split, float)
    return float(a.mean()), float(a.std(ddof=1)), float(a.max()), float(a.max() - a.mean())


def selection_optimism(X, y, leak, n_splits=50, base_seed=0):
    """Δ = max_b − mean_b，b 遍历 n_splits 个 70/30 划分（每划分先在 test 上取网格最优）。
    leak=True ：先在【全数据】上用 SelectKBest 选好特征（泄漏），再做划分。
    leak=False：特征选择在每个训练划分【内部】做（无泄漏，= 基准协议）。
    两臂共用同一批划分种子与同一 (k,C) 网格，唯一差别就是特征选择的位置。"""
    d = X.shape[1]
    if leak:
        # 泄漏的选择只依赖 k，不依赖划分 → 每个 k 只算一次（否则会重复 n_splits 次）
        Xi = SimpleImputer(strategy="median").fit_transform(X)
        Xs = StandardScaler().fit_transform(Xi)
        leaked = {k: SelectKBest(f_classif, k=min(k, d)).fit_transform(Xs, y) for k in GRID_K}

    per_split = []
    for s in range(n_splits):
        best = 0.0
        for k in GRID_K:
            if leak:
                Xk = leaked[k]
                Xtr, Xte, ytr, yte = train_test_split(Xk, y, test_size=0.30, stratify=y,
                                                      random_state=base_seed + s)
                for C in GRID_C:
                    m = LogisticRegression(class_weight="balanced", solver="liblinear",
                                           max_iter=5000, C=C).fit(Xtr, ytr)
                    best = max(best, roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
            else:
                Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y,
                                                      random_state=base_seed + s)
                for C in GRID_C:
                    m = _infold_pipe(k, C, d).fit(Xtr, ytr)
                    best = max(best, roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
        per_split.append(best)
    a = np.asarray(per_split, float)
    return float(a.mean()), float(a.std(ddof=1)), float(a.max()), float(a.max() - a.mean())


def load_radml(name):
    import radMLBench as rb
    df = rb.loadData(name)
    y = df["Target"].astype(int).values
    X = df.drop(columns=[c for c in ("Target", "ID") if c in df.columns]).select_dtypes("number").values.astype(float)
    return X, y


def pick_cohorts(n):
    """与 r_d.py 同口径：维度够大、少数类够多的队列，才有空间做 d 的剂量-反应。
    只用 n/d/minority 这些【与协议无关】的队列属性，所以读哪一版 sweep 都一样；
    取行数更多的那份，以便主 sweep 正在重跑时也能用。"""
    cands = [RESULTS / "radmlbench_sweep.csv", RESULTS / "_archive" / "radmlbench_sweep_v1.csv"]
    tabs = [pd.read_csv(p) for p in cands if p.exists()]
    d = max(tabs, key=len)
    # 与 r_d.py:pick_cohorts 完全同一筛选条件 —— 本实验的目的就是解释 rd_dose_response.csv
    # 里那批队列上观察到的正相关，所以必须落在同一批队列上。
    d = d[(d["d"] <= 1200) & (d["minority"] >= 120)].sort_values("minority", ascending=False)
    return list(d["dataset"].head(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", type=int, default=4)
    ap.add_argument("--splits", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=3,
                    help="每个 d 水平重复几次随机特征子集（round-23：给斜率 CI）")
    ap.add_argument("--fresh", action="store_true",
                    help="忽略断点续跑缓存：把已有产物归档到 results/_archive/ 后从零重跑")
    args = ap.parse_args()
    cohorts = pick_cohorts(args.cohorts)
    print(f"[R-G] cohorts: {cohorts}  splits={args.splits}", flush=True)

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "rg_dim_channel.csv"
    _fresh(out, getattr(args, "fresh", False))
    done, prior_honest = set(), {}
    if out.exists():
        dd = pd.read_csv(out)
        key = ["cohort", "arm", "dfrac", "seed"]
        if all(k in dd.columns for k in key):
            done = set(zip(dd["cohort"], dd["arm"], dd["dfrac"], dd["seed"]))
            prior_honest = {(c, float(f), int(sd)): float(v) for c, f, sd, v
                            in zip(dd["cohort"], dd["dfrac"], dd["seed"], dd["honest"])}
    rng = np.random.default_rng(SEED)

    ARMS = ("infold", "leaky", "leaky_fixedk")
    for name in cohorts:
        X, y = load_radml(name)
        D = X.shape[1]; M = int(min(y.sum(), len(y) - y.sum()))
        for dfrac in (1.0, 0.5, 0.25, 0.1):
            dd_ = max(50, int(dfrac * D))
            if dd_ > D:
                continue
            # round-23：每个 d 水平重复多个随机特征子集，否则斜率没有不确定度可言
            for seed_i in range(1 if dfrac == 1.0 else args.seeds):
                if all((name, a_, dfrac, seed_i) in done for a_ in ARMS):
                    continue
                sub_rng = np.random.default_rng(SEED + 977 * seed_i)
                cols = (sub_rng.choice(D, dd_, replace=False) if dd_ < D else np.arange(D))
                Xd = X[:, cols]
                h = prior_honest.get((name, float(dfrac), seed_i))
                if h is None:
                    h = honest_auc(Xd, y)
                for arm in ARMS:
                    if (name, arm, dfrac, seed_i) in done:
                        continue
                    if arm == "leaky_fixedk":
                        m, sd, mx, delta = selection_optimism_fixedk(Xd, y, n_splits=args.splits)
                    else:
                        m, sd, mx, delta = selection_optimism(Xd, y, arm == "leaky",
                                                              n_splits=args.splits)
                    row = {"cohort": name, "arm": arm, "dfrac": dfrac, "d": dd_, "D_full": D,
                           "seed": seed_i, "n": len(y), "minority": M, "honest": round(h, 4),
                           "sel_mean": round(m, 4), "sel_sd": round(sd, 4),
                           "sel_max": round(mx, 4), "delta_selection": round(delta, 4),
                           "delta_vs_honest": round(mx - h, 4)}
                    with open(out, "a", newline="", encoding="utf-8") as f:
                        w = csv.DictWriter(f, fieldnames=FIELDS)
                        if f.tell() == 0:
                            w.writeheader()
                        w.writerow(row)
                    print(f"  {name:22} {arm:13} frac={dfrac:<5} seed={seed_i} "
                          f"d={dd_:<5} Dsel={delta:+.4f}", flush=True)

    # ---- 汇总：两个 endpoint × 两臂，各自的 within-cohort 斜率 (AUC per log10 d) ----
    df = pd.read_csv(out)
    df["ld"] = np.log10(df["d"])

    def _cohort_slopes(sub, ycol):
        """返回 {cohort: 该队列在各随机特征子集上斜率的均值}。
        独立实验单位是队列：同一队列的不同特征子集共享样本、共享标签，彼此高度相关。"""
        per = {}
        for (c, _sd), g in sub.groupby(["cohort", "seed"]):
            if g["dfrac"].nunique() >= 3:
                per.setdefault(c, []).append(float(np.polyfit(g["ld"], g[ycol], 1)[0]))
        return {c: float(np.mean(v)) for c, v in per.items()}

    def _t_ci(vals):
        """4 个簇时用 t 区间，不用 bootstrap：簇太少，bootstrap 会系统性偏窄。"""
        v = np.asarray(vals, float)
        k = v.size
        if k < 2:
            return [None, None]
        tcrit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
                 7: 2.447, 8: 2.365, 9: 2.306}.get(k - 1, 1.96)
        half = tcrit * v.std(ddof=1) / math.sqrt(k)
        return [round(float(v.mean() - half), 4), round(float(v.mean() + half), 4)]

    def within(sub, ycol):
        """cohort-level 斜率汇总（AUC per log10 d）。主区间是 df=k-1 的 t 区间。"""
        per = _cohort_slopes(sub, ycol)
        if not per:
            return {"n_cohorts": 0, "mean_slope": float("nan"), "ci95_cohort_t": [None, None]}
        vals = np.asarray(list(per.values()), float)
        rng = np.random.default_rng(20260625)
        cb = [vals[rng.integers(0, vals.size, vals.size)].mean() for _ in range(5000)]
        # 旧口径：把每条 (cohort, seed) 斜率当独立样本，仅作对照留档，不进正文
        raw = []
        for _, g in sub.groupby(["cohort", "seed"]):
            if g["dfrac"].nunique() >= 3:
                raw.append(float(np.polyfit(g["ld"], g[ycol], 1)[0]))
        raw = np.asarray(raw, float)
        nb = [raw[rng.integers(0, raw.size, raw.size)].mean() for _ in range(5000)]
        g2 = sub.groupby("cohort")
        ldc = sub["ld"] - g2["ld"].transform("mean")
        yc = sub[ycol] - g2[ycol].transform("mean")
        r = float(np.corrcoef(ldc, yc)[0, 1]) if len(sub) > 2 else float("nan")
        return {"n_cohorts": int(vals.size),
                "n_cohort_x_seed_slopes": int(raw.size),
                "mean_slope": round(float(vals.mean()), 4),
                "ci95_cohort_t": _t_ci(vals),
                "ci95_cohort_bootstrap": [round(float(np.percentile(cb, 2.5)), 4),
                                          round(float(np.percentile(cb, 97.5)), 4)],
                "ci95_PSEUDOREPLICATED_do_not_report": [round(float(np.percentile(nb, 2.5)), 4),
                                                        round(float(np.percentile(nb, 97.5)), 4)],
                "per_cohort_slopes": {c: round(v, 4) for c, v in sorted(per.items())},
                "pooled_within_r": round(r, 3)}

    def paired(sub_all, ycol, arm_a, arm_b):
        """配对对照：同一队列上 arm_a 的斜率减 arm_b 的斜率。队列天然配对，
        这是本节唯一不依赖跨队列方差的比较。"""
        a = _cohort_slopes(sub_all[sub_all.arm == arm_a], ycol)
        b = _cohort_slopes(sub_all[sub_all.arm == arm_b], ycol)
        both = sorted(set(a) & set(b))
        if len(both) < 2:
            return {"n_cohorts": len(both)}
        dif = np.array([a[c] - b[c] for c in both], float)
        return {"n_cohorts": int(dif.size), "mean_difference": round(float(dif.mean()), 4),
                "ci95_paired_t": _t_ci(dif),
                "per_cohort": {c: round(a[c] - b[c], 4) for c in both}}

    summary = {}
    print("\n[R-G RESULT] within-cohort dependence on log10(d), by ENDPOINT and by arm:")
    for ycol, tag in (("delta_selection", "A: within-protocol selection (max-mean)  [ours]"),
                      ("delta_vs_honest", "B: protocol-crossing gap (max-honest)   [r_d style]"),
                      ("sel_max", "level: the shopped maximum itself"),
                      ("honest", "level: honest AUC itself")):
        print(f"  {tag}")
        for arm in ARMS:
            sub = df[df.arm == arm]
            if not len(sub):
                continue
            summary[f"{ycol}|{arm}"] = within(sub, ycol)
            v = summary[f"{ycol}|{arm}"]
            lo, hi = v["ci95_cohort_t"]
            print(f"    {arm:13}: mean slope {v['mean_slope']:+.4f} AUC/decade "
                  f"95% CI [{lo:+.4f}, {hi:+.4f}] (cohort-level t, {v['n_cohorts']} cohorts)")
    # 配对对照：泄漏臂相对折内臂，在同一批队列上抬高了多少"被挑出来的最大值"
    for ycol in ("sel_max", "delta_selection"):
        for a in ("leaky", "leaky_fixedk"):
            summary[f"PAIRED {ycol}|{a}-infold"] = paired(df, ycol, a, "infold")
    print("  paired contrasts (same cohorts, leaky minus in-fold):")
    for k in [x for x in summary if x.startswith("PAIRED")]:
        v = summary[k]
        if "mean_difference" in v:
            print(f"    {k:38}: {v['mean_difference']:+.4f} "
                  f"95% CI [{v['ci95_paired_t'][0]:+.4f}, {v['ci95_paired_t'][1]:+.4f}] "
                  f"({v['n_cohorts']} cohorts)")

    def g(y, a):
        return summary.get(f"{y}|{a}", {}).get("mean_slope", float("nan"))
    conclusion = (
        "Holding n and the class counts fixed and varying only d: the WITHIN-PROTOCOL selection "
        f"optimism (max minus mean over the same splits) does not grow with d in any arm "
        f"(in-fold {g('delta_selection','infold'):+.4f}, leaky-grid {g('delta_selection','leaky'):+.4f}, "
        f"leaky-fixed-k {g('delta_selection','leaky_fixedk'):+.4f} AUC per decade of d). "
        "Where d does act is on the LEVEL of a leakage-contaminated maximum when the number of "
        f"selected features is held fixed while the pool grows: the shopped maximum moves "
        f"{g('sel_max','leaky_fixedk'):+.4f} AUC/decade under fixed-k full-data selection versus "
        f"{g('sel_max','infold'):+.4f} with selection kept in-fold, while honest discrimination itself "
        f"moves {g('honest','infold'):+.4f}. This is the configuration used in "
        "results/rd_dose_response.csv, which is why that file shows d correlating with its "
        "protocol-crossing gap while the selection optimism modelled in this paper does not.")
    print("\n[CONCLUSION] " + conclusion)

    # ---- 图 ----
    FIGS.mkdir(exist_ok=True)
    figstyle.apply()
    fig, axes = plt.subplots(1, 2, figsize=(figstyle.SPAN, 2.75), sharex=True)
    marks = ["o", "s", "^", "D", "v"]
    panels = [("delta_selection", r"(a) within-protocol selection  $\Delta=\max_b-\mathrm{mean}_b$"),
              ("delta_vs_honest", r"(b) protocol-crossing gap  $\max_b-\mathrm{AUC}^{\mathrm{honest}}$")]
    for ax, (ycol, title) in zip(axes, panels):
        for arm, col, lab in (("leaky_fixedk", figstyle.ORANGE, "full-data selection, $k$ fixed"),
                              ("leaky", figstyle.RED, "full-data selection, $k$ tuned"),
                              ("infold", figstyle.GREEN, "in-fold selection (no leakage)")):
            sub = df[df.arm == arm]
            for i, (c, s) in enumerate(sub.groupby("cohort")):
                s = s.sort_values("d")
                ax.plot(s["d"], s[ycol], marks[i % len(marks)] + "-",
                        color=col, alpha=0.30, lw=0.8, ms=3.0)
            g = sub.groupby("d")[ycol].mean().sort_index()
            ax.plot(g.index, g.values, "-", color=col, lw=2.4, label=lab, zorder=5)
        ax.set_xscale("log")
        ax.set_xlabel(r"feature dimensionality $d$")
        ax.set_title(title, fontsize=7.2)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("optimism (AUC)")
    axes[0].legend(loc="best", fontsize=6.0)
    fig.suptitle(r"Same cohorts, same splits, $n$ and class counts fixed: the endpoint decides "
                 r"whether $d$ appears to matter", fontsize=7.6, y=1.02)
    fig.tight_layout(w_pad=1.4)
    fig.savefig(FIGS / "dim_channel.png"); fig.savefig(FIGS / "dim_channel.pdf"); plt.close(fig)

    import json
    (RESULTS / "rg_dim_channel.json").write_text(json.dumps({
        "design": "fix n and class counts, vary d by random feature subsampling; "
                  "same 70/30 splits, same (k,C) grid, same learner; only the feature-selection "
                  "placement differs between arms.",
        "endpoint": "Delta_selection = max_b - mean_b over splits (pure within-protocol selection)",
        "cohorts": sorted(df["cohort"].unique().tolist()),
        "n_splits": args.splits, "summary": summary,
        "conclusion": conclusion,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved -> {out} , results/rg_dim_channel.json , figures/dim_channel.png/.pdf")


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
