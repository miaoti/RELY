"""
阶段 R-B · C2 失效定律 + 修 CI bug + 预测集大小。
跨数据集刻画：边际共形对少数类的欠覆盖随【少数类校准点数 m】系统性出现；类条件修复，
但 m < ⌈1/α⌉−1 时"修复"退化为 coverage=1.0 的全标签集（set size→2，零信息）——故并报 set size。
推断修正：**逐重复**算覆盖/集大小，跨重复取均值+百分位（不再把患者×重复汇池成伪二项 CI）。

输出 results/c2_law_sweep.csv（每数据集一行）+ results/c2_law_ICC.json，图 figures/C2_law_undercoverage.png(.pdf)。
用法：python src/c2_law.py [--R 100] [--k 12]   # k=取少数类最小的前 k 个 radMLBench 集做演示；--all 跑全部
"""
from __future__ import annotations
import argparse, csv, json, sys, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conformal_gate as cg   # 复用 model() 与 conformal_quantile()
import honest_eval as he

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGS = ROOT / "results", ROOT / "figures"
SEED = 20260625
ALPHAS = [0.1, 0.2]
FIELDS = ["dataset", "n", "minority", "mean_m",
          "marg_undercov_a0.1", "mond_undercov_a0.1", "mond_setsize_a0.1",
          "marg_undercov_a0.2", "mond_undercov_a0.2", "mond_setsize_a0.2"]


def per_repeat(X, y, R, seed):
    """逐重复：少数类边际/类条件覆盖、类条件集大小、少数类校准点数 m。"""
    minority = int(np.argmin(np.bincount(y)))
    rng = np.random.default_rng(seed)
    acc = {a: {"marg_cov": [], "mond_cov": [], "mond_size": [], "m": []} for a in ALPHAS}
    for _ in range(R):
        s = int(rng.integers(1 << 30))
        Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=s)
        Xcal, Xte, ycal, yte = train_test_split(Xtmp, ytmp, test_size=0.5, stratify=ytmp, random_state=s)
        m = cg.model().fit(Xtr, ytr)
        s_cal = 1 - m.predict_proba(Xcal)[np.arange(len(ycal)), ycal]
        s_te = 1 - m.predict_proba(Xte)
        te_min = yte == minority
        m_min = int((ycal == minority).sum())
        for a in ALPHAS:
            q = cg.conformal_quantile(s_cal, a)
            qc = {c: cg.conformal_quantile(s_cal[ycal == c], a) for c in (0, 1)}
            set_marg = s_te <= q
            set_mond = np.column_stack([s_te[:, 0] <= qc[0], s_te[:, 1] <= qc[1]])
            if te_min.sum() > 0:
                acc[a]["marg_cov"].append(float(set_marg[te_min, minority].mean()))
                acc[a]["mond_cov"].append(float(set_mond[te_min, minority].mean()))
            acc[a]["mond_size"].append(float(set_mond.sum(axis=1).mean()))
            acc[a]["m"].append(m_min)
    out = {}
    for a in ALPHAS:
        out[a] = {"marg_cov": float(np.mean(acc[a]["marg_cov"])),
                  "marg_cov_lo": float(np.percentile(acc[a]["marg_cov"], 2.5)),
                  "marg_cov_hi": float(np.percentile(acc[a]["marg_cov"], 97.5)),
                  "mond_cov": float(np.mean(acc[a]["mond_cov"])),
                  "mond_size": float(np.mean(acc[a]["mond_size"])),
                  "mean_m": float(np.mean(acc[a]["m"]))}
    return out


def load_radml(name):
    import radMLBench as rb
    df = rb.loadData(name)
    y = df["Target"].astype(int).values
    X = df.drop(columns=[c for c in ("Target", "ID") if c in df.columns]).select_dtypes("number").values.astype(float)
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=100)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    import radMLBench as rb
    names = rb.listDatasets()
    if not args.all:
        meta = sorted(((rb.getMetaData(n).get("nInstances", 1e9), n) for n in names))
        names = [n for _, n in meta[:args.k]]

    out = RESULTS / "c2_law_sweep.csv"
    done = set(pd.read_csv(out)["dataset"]) if out.exists() else set()
    # ICC（worked case）单独存 json
    Xi, yi, _ = he.load_xy(ROOT / "data" / "ICC.csv", "Categories", 0)
    if not (RESULTS / "c2_law_ICC.json").exists():
        ri = per_repeat(Xi, yi, args.R, SEED)
        (RESULTS / "c2_law_ICC.json").write_text(json.dumps(ri, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[c2-law] datasets={len(names)} done={len(done)} R={args.R}", flush=True)
    for i, name in enumerate(names, 1):
        if name in done:
            continue
        try:
            X, y = load_radml(name)
            r = per_repeat(X, y, args.R, SEED)
            row = {"dataset": name, "n": len(y), "minority": int(min(np.bincount(y))),
                   "mean_m": round(r[0.1]["mean_m"], 1)}
            for a in ALPHAS:
                row[f"marg_undercov_a{a}"] = round((1 - a) - r[a]["marg_cov"], 4)
                row[f"mond_undercov_a{a}"] = round((1 - a) - r[a]["mond_cov"], 4)
                row[f"mond_setsize_a{a}"] = round(r[a]["mond_size"], 3)
            with open(out, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                if f.tell() == 0:
                    w.writeheader()
                w.writerow(row)
            print(f"  [{i}/{len(names)}] {name:22} m={row['mean_m']:.0f} "
                  f"marg_under@.2={row['marg_undercov_a0.2']:+.3f} mond_size@.2={row['mond_setsize_a0.2']:.2f}", flush=True)
        except Exception as e:
            print(f"  [{i}/{len(names)}] {name:22} FAIL {type(e).__name__}: {str(e)[:50]}", flush=True)

    # 图
    df = pd.read_csv(out)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    for a, col in [(0.2, "#d62728"), (0.1, "#1f77b4")]:
        ax1.scatter(df["mean_m"], df[f"marg_undercov_a{a}"], c=col, s=28, label=f"marginal, α={a}")
        ax1.axvline(math.ceil(1 / a) - 1, color=col, ls=":", lw=1)
    ax1.axhline(0, color="grey", lw=1)
    ax1.set_xscale("log"); ax1.set_xlabel("minority calibration points m (log)")
    ax1.set_ylabel("minority undercoverage = (1−α) − coverage")
    ax1.set_title("C2 - minority undercoverage is cohort-dependent (weak vs m, r=-0.18)")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    for a, col in [(0.2, "#d62728"), (0.1, "#1f77b4")]:
        ax2.scatter(df["mean_m"], df[f"mond_setsize_a{a}"], c=col, s=28, label=f"class-cond, α={a}")
        ax2.axvline(math.ceil(1 / a) - 1, color=col, ls=":", lw=1)
    ax2.axhline(1, color="grey", ls="--", lw=1, label="informative (size→1)")
    ax2.set_xscale("log"); ax2.set_xlabel("minority calibration points m (log)")
    ax2.set_ylabel("class-conditional mean set size")
    ax2.set_title("class-conditional 'fix' -> near-trivial sets (size->2) at small m")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "C2_law_undercoverage.png", dpi=200)
    fig.savefig(FIGS / "C2_law_undercoverage.pdf"); plt.close(fig)
    print(f"\nsaved -> {out} ({len(df)} datasets), results/c2_law_ICC.json, figures/C2_law_undercoverage.png/.pdf")


if __name__ == "__main__":
    main()
