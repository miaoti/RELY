"""
B · 标签置换检验（permutation test）—— 给"诚实 AUC ≈ 随机"一个正式 p 值。

做法（遵循清单阶段 7）：
  - 预先指定的主模型 = 折内特征选择(SelectKBest k=20) + 逻辑回归(class_weight=balanced, C=0.1)，
    全部封进无泄漏 Pipeline；评测用分层 CV。
  - 观测：在真实标签上算 CV AUC。
  - 零分布：把 y 打乱 N(≥1000) 次，【每次都重跑整条流程（含特征选择）】算 CV AUC。
  - p 值 = (1 + #{perm_auc >= obs_auc}) / (N + 1)；并存零分布数组供画 F2。
  说明：用固定主模型（非每次置换再调参）是清单允许的"每次置换跑单趟"成本控制。

用法：
  python src/permutation_test.py                       # ICC，1000 次置换
  python src/permutation_test.py --n-perm 50 --n-jobs 2   # 冒烟
  python src/permutation_test.py --radmlbench Granata2024
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, permutation_test_score
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import honest_eval as he

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(args):
    if args.radmlbench:
        import radMLBench as rb
        df = rb.loadData(args.radmlbench)
        y = df["Target"].astype(int).values
        X = df.drop(columns=[c for c in ("Target", "ID") if c in df.columns]).select_dtypes("number").values.astype(float)
        return X, y, args.radmlbench
    X, y, _ = he.load_xy(Path(args.data), args.outcome_col, args.pos_label)
    return X, y, Path(args.data).stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=str(ROOT / "data" / "ICC.csv"))
    ap.add_argument("--outcome-col", type=str, default="Categories")
    ap.add_argument("--pos-label", type=int, default=0)
    ap.add_argument("--radmlbench", type=str, default="")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--n-jobs", type=int, default=3)
    args = ap.parse_args()

    X, y, name = load(args)
    print(f"[data] {name}  n={len(y)}  events={int(y.sum())}  d={X.shape[1]}", flush=True)

    model = he.make_pipeline()
    model.set_params(select__k=20, clf__C=0.1)          # 预先指定的主模型
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=he.MASTER_SEED)

    obs, perm_scores, p = permutation_test_score(
        model, X, y, scoring="roc_auc", cv=cv,
        n_permutations=args.n_perm, n_jobs=args.n_jobs, random_state=he.MASTER_SEED)

    perm = np.asarray(perm_scores, float)
    out = {
        "dataset": name, "n": int(len(y)), "events": int(y.sum()), "d": int(X.shape[1]),
        "model": "SelectKBest(k=20)+LogReg(balanced,C=0.1), 10-fold CV",
        "observed_auc": float(obs), "n_permutations": int(args.n_perm),
        "pvalue": float(p),
        "null_mean": float(perm.mean()), "null_std": float(perm.std(ddof=1)),
        "null_p2.5": float(np.percentile(perm, 2.5)),
        "null_p97.5": float(np.percentile(perm, 97.5)),
        "null_scores": perm.round(5).tolist(),     # 供画 F2 零分布
    }
    RESULTS.mkdir(exist_ok=True)
    f = RESULTS / f"permutation_{name}.json"
    f.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nobserved honest AUC : {obs:.4f}")
    print(f"null distribution   : mean={perm.mean():.4f}  95% [{out['null_p2.5']:.3f}, {out['null_p97.5']:.3f}]")
    print(f"p-value             : {p:.4f}  (N={args.n_perm})")
    verdict = ("观测 AUC 落在置换零分布内 → 与随机不可区分（支持论点）"
               if p > 0.05 else "p<=0.05 → 观测 AUC 显著高于置换零分布")
    print(f"verdict             : {verdict}")
    print(f"saved -> {f}")


if __name__ == "__main__":
    main()
