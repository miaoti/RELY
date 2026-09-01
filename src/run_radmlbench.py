"""
radMLBench 跨数据集 sweep —— C1 的"轻量"证据引擎。

对 radMLBench（50 个【真实提取特征 + 二分类标签】的公开影像组学表，n=51–969, d=101–11165）
逐个跑 honest_eval 的同一套协议，产出每个数据集的：
  honest 诚实嵌套 CV AUC（含【患者级 bootstrap 置信区间】与 Nadeau-Bengio 对照区间）、
  选择臂的逐划分 mean/sd/max、以及主结局 Δ_selection = max − mean、EPV/维度/不平衡。
汇总成 results/radmlbench_sweep.csv —— C1 主证据。

⚠️ v2 口径（round-18）：
  ① 主结局改为 `delta_selection = max_b − mean_b`（同一协议、同一批 70/30 划分、同一网格、
     同一估计量内部的【纯选择】乐观）。旧的 `max − honest` 跨了训练比例(90/10 vs 70/30)、
     网格(k∈{10,20,30} vs {7..100})与估计量(嵌套 CV vs 单划分)，不是纯选择量，降为对照
     列 `delta_vs_honest`。
  ② `honest_ci_lo/hi` 是真置信区间（池化 OOF + 患者级 bootstrap）；折分数的 p2.5/p97.5
     改名为 `fold_p2.5/p97.5`，仅作诊断，不得用于"覆盖 0.5"判定。

与 Gidwani 的差异化：这些是【真实】提取的影像组学特征（非合成）。
依赖：pip install radMLBench（纯 python wrapper，主环境即可，首次会缓存下载各表）。

用法：
  python src/run_radmlbench.py --k 6              # 取最小的 6 个数据集做演示
  python src/run_radmlbench.py --all --repeats 10 # 全部 50 个（较慢）
  python src/run_radmlbench.py --only Granata2024,Li2020
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import honest_eval as he   # 复用同一套评测函数，保证与 ICC 完全同协议

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load_dataset(name):
    import radMLBench as rb
    df = rb.loadData(name)
    y = df["Target"].astype(int).values
    drop = [c for c in ("Target", "ID") if c in df.columns]
    X = df.drop(columns=drop).select_dtypes("number").values.astype(float)
    return X, y


def run_one(name, repeats, n_jobs=3):
    X, y = load_dataset(name)
    n, d = X.shape
    minority = int(min(int(y.sum()), n - int(y.sum())))
    honest = he.honest_nested_cv(X, y, repeats=repeats, seed=he.MASTER_SEED, n_jobs=n_jobs)
    tuned = he.test_tuned_single_splits(X, y, n_splits=50)
    return {
        "dataset": name, "n": n, "d": d, "minority": minority,
        "epv": round(minority / d, 4),
        "honest_auc": round(honest["mean"], 4),
        # 诚实 AUC 的【真·置信区间】：逐 repeat 池化 OOF 预测 + 患者级 bootstrap
        "honest_pooled_auc": round(honest["pooled_auc"], 4),
        "honest_ci_lo": round(honest["boot_ci_lo"], 4),
        "honest_ci_hi": round(honest["boot_ci_hi"], 4),
        # Nadeau-Bengio 校正区间（对照口径）
        "honest_nb_se": round(honest["auc_nb_se"], 4),
        "honest_nb_lo": round(honest["nb_lo"], 4), "honest_nb_hi": round(honest["nb_hi"], 4),
        # ⚠️ 折分数离散度（不是置信区间；仅留作诊断，勿用于任何"覆盖 0.5"判定）
        "fold_p2.5": round(honest["p2.5"], 3), "fold_p97.5": round(honest["p97.5"], 3),
        # 选择臂：同一协议内的 50 次 70/30 划分 × 网格，逐划分取网格最优
        "test_selected_mean": round(tuned["mean"], 4),
        "test_selected_sd": round(tuned["std"], 4),
        "test_selected_max": round(tuned["max"], 4),
        # 全部 splits x grid 次评测（未经任何挑选）的均值/标准差，用于拆分两级选择
        "grid_grand_mean": round(tuned["grand_mean"], 4),
        "grid_grand_sd": round(tuned["grand_sd"], 4),
        "n_candidates_total": int(tuned["n_candidates_total"]),
        # 主结局 Δ：同协议内【纯选择】乐观（max − mean），不含训练比例/网格/估计量差异
        # Delta_split: 在【已经过 test-set 调参】的各划分之间继续挑最好（B = n_splits）
        "delta_selection": round(tuned["max"] - tuned["mean"], 4),
        # Delta_tuning: 每个划分内部在 test 上挑网格最优带来的那一级乐观
        "delta_tuning": round(tuned["mean"] - tuned["grand_mean"], 4),
        # 两级合计：从"谁都没挑"到"两级都挑"
        "delta_total": round(tuned["max"] - tuned["grand_mean"], 4),
        # 旧口径（max − honest）：跨协议差值，含训练比例与网格差异，仅留作对照
        "delta_vs_honest": round(tuned["max"] - honest["mean"], 4),
    }


FIELDS = ["dataset", "n", "d", "minority", "epv", "honest_auc",
          "honest_pooled_auc", "honest_ci_lo", "honest_ci_hi",
          "honest_nb_se", "honest_nb_lo", "honest_nb_hi",
          "fold_p2.5", "fold_p97.5",
          "test_selected_mean", "test_selected_sd", "test_selected_max",
          "grid_grand_mean", "grid_grand_sd", "n_candidates_total",
          "delta_selection", "delta_tuning", "delta_total", "delta_vs_honest"]


def main():
    import csv
    import radMLBench as rb
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6, help="取最小的 K 个数据集（演示）")
    ap.add_argument("--all", action="store_true", help="跑全部 50 个")
    ap.add_argument("--only", type=str, default="", help="逗号分隔的指定数据集")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--n-jobs", type=int, default=3, help="内层 GridSearch 并行度（控内存）")
    ap.add_argument("--fresh", action="store_true",
                    help="忽略断点续跑缓存：把已有产物归档到 results/_archive/ 后从零重跑")
    args = ap.parse_args()

    names = rb.listDatasets()
    if args.only:
        todo = [s.strip() for s in args.only.split(",") if s.strip()]
    elif args.all:
        todo = names
    else:
        meta = sorted(((rb.getMetaData(n).get("nInstances", 1e9), n) for n in names))
        todo = [n for _, n in meta[:args.k]]

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "radmlbench_sweep.csv"
    _fresh(out, getattr(args, "fresh", False))
    # 断点续跑：跳过已写入的数据集；增量追加，每个数据集写完即落盘（防中途被杀全丢）
    done = set()
    if out.exists():
        try:
            done = set(pd.read_csv(out)["dataset"].astype(str))
        except Exception:
            done = set()
    pending = [n for n in todo if n not in done]
    print(f"[sweep] total={len(todo)} done={len(done)} pending={len(pending)} "
          f"repeats={args.repeats} n_jobs={args.n_jobs}", flush=True)

    t0 = time.time()
    for i, name in enumerate(pending, 1):
        try:
            r = run_one(name, args.repeats, n_jobs=args.n_jobs)
            with open(out, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                if f.tell() == 0:
                    w.writeheader()
                w.writerow(r)
            print(f"  [{i}/{len(pending)}] {name:24} n={r['n']:>4} d={r['d']:>5} "
                  f"EPV={r['epv']:.3f}  honest={r['honest_auc']:.3f}  "
                  f"CI=[{r['honest_ci_lo']:.2f},{r['honest_ci_hi']:.2f}]  "
                  f"sel-mean={r['test_selected_mean']:.3f}  Dsel={r['delta_selection']:+.3f}", flush=True)
        except Exception as e:
            print(f"  [{i}/{len(pending)}] {name:24} FAILED: {type(e).__name__}: {e}", flush=True)

    if not out.exists():
        sys.exit("no datasets ran")
    df = pd.read_csv(out)
    print(f"\n[summary] datasets={len(df)}  "
          f"honest AUC median={df['honest_auc'].median():.3f}  "
          f"mean selection optimism={df['delta_selection'].mean():+.3f}", flush=True)
    print(f"  honest AUC within [0.45,0.55] (chance): "
          f"{int(((df['honest_auc']>=0.45)&(df['honest_auc']<=0.55)).sum())}/{len(df)}", flush=True)
    print(f"saved -> {out}   ({round(time.time()-t0,1)}s)", flush=True)


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
