"""
radMLBench 跨数据集 sweep —— C1 的"轻量"证据引擎。

对 radMLBench（50 个【真实提取特征 + 二分类标签】的公开影像组学表，n=51–969, d=101–11165）
逐个跑 honest_eval 的同一套协议，产出每个数据集的：
  honest 诚实嵌套 CV AUC、测试集选模型乐观 AUC、乐观差距、EPV/维度/不平衡。
汇总成 results/radmlbench_sweep.csv —— 即"乐观差距在多个真实队列上一致、并随 EPV 变化"的 C1 主证据。

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
        "honest_lo": round(honest["p2.5"], 3), "honest_hi": round(honest["p97.5"], 3),
        "test_selected_auc": round(tuned["mean"], 4),
        "test_selected_max": round(tuned["max"], 4),
        "optimism_gap": round(tuned["mean"] - honest["mean"], 4),
    }


FIELDS = ["dataset", "n", "d", "minority", "epv", "honest_auc", "honest_lo",
          "honest_hi", "test_selected_auc", "test_selected_max", "optimism_gap"]


def main():
    import csv
    import radMLBench as rb
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6, help="取最小的 K 个数据集（演示）")
    ap.add_argument("--all", action="store_true", help="跑全部 50 个")
    ap.add_argument("--only", type=str, default="", help="逗号分隔的指定数据集")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--n-jobs", type=int, default=3, help="内层 GridSearch 并行度（控内存）")
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
                  f"test-sel={r['test_selected_auc']:.3f}  gap={r['optimism_gap']:+.3f}", flush=True)
        except Exception as e:
            print(f"  [{i}/{len(pending)}] {name:24} FAILED: {type(e).__name__}: {e}", flush=True)

    if not out.exists():
        sys.exit("no datasets ran")
    df = pd.read_csv(out)
    print(f"\n[summary] datasets={len(df)}  "
          f"honest AUC median={df['honest_auc'].median():.3f}  "
          f"mean optimism gap={df['optimism_gap'].mean():+.3f}", flush=True)
    print(f"  honest AUC within [0.45,0.55] (chance): "
          f"{int(((df['honest_auc']>=0.45)&(df['honest_auc']<=0.55)).sum())}/{len(df)}", flush=True)
    print(f"saved -> {out}   ({round(time.time()-t0,1)}s)", flush=True)


if __name__ == "__main__":
    main()
