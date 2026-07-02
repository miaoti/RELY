"""
阶段 4 · 诚实基线阶梯（Honest Baselines）→ T1 表数据。

在同一套诚实嵌套 CV 协议下，报告各特征子集的诚实 AUC[95% pct] + NB-SE + Brier/ECE/sens/spec：
  clinical（Age,Bp,Sex,Diabetes）/ volume（梗死体积代理）/ age+volume（需被超越的简约基准）/
  all_radiomics（全 1004，=融合）/ DWI_only / T2_only。
增量写 results/baselines_ICC.csv（每个子集跑完即落盘，防中途被杀）。
主结果只认预指定主模型（all_radiomics 即主模型在全特征上的诚实表现）；其余为对照/探索。

用法：python src/baselines.py [--repeats 20]
"""
from __future__ import annotations
import argparse, csv, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import honest_eval as he

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIELDS = ["subset", "n_features", "auc_mean", "auc_lo", "auc_hi", "nb_se",
          "brier", "ece", "sens", "spec"]


def feature_subsets(df):
    cols = list(df.columns)
    clinical = [c for c in ["Age", "Bp", "Sex", "Diabetes"] if c in cols]
    volume = [c for c in cols if c.startswith("shape_MeshVolume") or c.startswith("shape_VoxelVolume")]
    meta = {"Unnamed: 0", "Image.number", "Smoke", "Categories", "Age", "Bp", "Sex", "Diabetes"}
    radiomics = [c for c in cols if c not in meta]
    dwi = [c for c in radiomics if c.split("_")[-1].startswith("DWI")]
    t2 = [c for c in radiomics if c.split("_")[-1].startswith("T2")]
    return [
        ("clinical", clinical),
        ("volume", volume),
        ("age+volume", (["Age"] if "Age" in cols else []) + volume),
        ("all_radiomics(主模型)", radiomics),
        ("DWI_only", dwi),
        ("T2_only", t2),
    ]


def grid_for(n_features):
    ks = [k for k in (10, 20, 30) if k < n_features]
    return {"select__k": ks or ["all"], "clf__C": [0.01, 0.1, 1.0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args()

    df = pd.read_csv(ROOT / "data" / "ICC.csv")
    y = (df["Categories"].values == 0).astype(int)      # 阳性=poor
    subsets = feature_subsets(df)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "baselines_ICC.csv"
    done = set()
    if out.exists():
        try:
            done = set(pd.read_csv(out)["subset"].astype(str))
        except Exception:
            done = set()
    print(f"[baselines] y events(poor)={int(y.sum())}/{len(y)}  subsets={len(subsets)} "
          f"done={len(done)} repeats={args.repeats}", flush=True)

    t0 = time.time()
    for name, cols in subsets:
        if name in done:
            continue
        if not cols:
            print(f"  {name:22} SKIP (no columns)"); continue
        X = df[cols].values.astype(float)
        r = he.honest_nested_cv(X, y, repeats=args.repeats, seed=he.MASTER_SEED,
                                n_jobs=3, grid=grid_for(X.shape[1]))
        row = {"subset": name, "n_features": X.shape[1],
               "auc_mean": round(r["mean"], 4), "auc_lo": round(r["p2.5"], 3),
               "auc_hi": round(r["p97.5"], 3), "nb_se": round(r["auc_nb_se"], 4),
               "brier": round(r["brier"], 4), "ece": round(r["ece"], 4),
               "sens": round(r["sensitivity"], 4), "spec": round(r["specificity"], 4)}
        with open(out, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if f.tell() == 0:
                w.writeheader()
            w.writerow(row)
        print(f"  {name:22} d={X.shape[1]:<5} AUC={row['auc_mean']:.3f} "
              f"[{row['auc_lo']:.2f},{row['auc_hi']:.2f}] NB-SE={row['nb_se']:.3f} "
              f"Brier={row['brier']:.3f}", flush=True)

    print(f"\nsaved -> {out}   ({round(time.time()-t0,1)}s)", flush=True)


if __name__ == "__main__":
    main()
