"""
B · 标签置换检验（permutation test）—— 给"诚实 AUC ≈ 随机"一个正式 p 值。

做法（遵循清单阶段 7）：
  - 预先指定的主模型 = 折内特征选择(SelectKBest k=20) + 逻辑回归(class_weight=balanced, C=0.1)，
    全部封进无泄漏 Pipeline；评测用分层 CV。
  - 观测：在真实标签上算 CV AUC。
  - 零分布：把 y 打乱 N(≥1000) 次，【每次都重跑整条流程（含特征选择）】算 CV AUC。
  - p 值 = (1 + #{perm_auc >= obs_auc}) / (N + 1)；并存零分布数组供画 F2。
  说明：用固定主模型（非每次置换再调参）是清单允许的"每次置换跑单趟"成本控制。

round-18 新增 --nested：每次置换都跑【完整嵌套 CV】（外层 10 折 × 内层 5 折网格搜索），
  使置换检验的被检验量与论文正文/图中报告的 nested-CV 估计【完全同一个估计量】。
  此前固定主模型给的观测值是 0.50，而正文 nested-CV 点估计是 0.47，口径不一致（已被外部审稿点名）。
round-18 新增 --radiomics-only：AIS 队列只用 1004 个影像组学列（排除 4 个临床列），
  与 baselines 的 all_radiomics 行及正文所称的 1004-feature pool 一致。

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
    if args.radiomics_only:
        import ais_forensic as af
        X, y, cols, _ = af.load_radiomics_only(Path(args.data))
        return X, y, Path(args.data).stem
    X, y, _ = he.load_xy(Path(args.data), args.outcome_col, args.pos_label)
    return X, y, Path(args.data).stem


def exact_nested_permutation(X, y, n_perm, repeats, seed, n_jobs):
    """置换检验，被检验量 = he.honest_nested_cv 本身（逐外层折的内层种子 seed+i 也一致）。

    这样 observed 与正文报告的诚实 AUC 【完全同一个数】，不再有"置换测的是另一个估计量"
    的口径缺口。并行放在【置换之间】，每次置换内部 n_jobs=1。"""
    from joblib import Parallel, delayed
    # round-23: 统一 estimand —— 报告的诚实 AUC、bootstrap 区间、置换检验都用 pooled OOF。
    # 此前这里用折均值 ["mean"]，与围绕 pooled OOF 的区间不是同一个量（跨 50 队列最大差 0.033）。
    obs = he.honest_nested_cv(X, y, repeats=repeats, seed=seed, n_jobs=1,
                              boot=0)["pooled_auc"]
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(y) for _ in range(n_perm)]

    def one(yp):
        return he.honest_nested_cv(X, yp, repeats=repeats, seed=seed, n_jobs=1,
                                   boot=0)["pooled_auc"]

    scores = Parallel(n_jobs=n_jobs, verbose=5)(delayed(one)(yp) for yp in perms)
    scores = np.asarray(scores, float)
    p = (1.0 + int((scores >= obs).sum())) / (n_perm + 1.0)
    return float(obs), scores, float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=str(ROOT / "data" / "ICC.csv"))
    ap.add_argument("--outcome-col", type=str, default="Categories")
    ap.add_argument("--pos-label", type=int, default=0)
    ap.add_argument("--radmlbench", type=str, default="")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--n-jobs", type=int, default=3)
    ap.add_argument("--nested", action="store_true",
                    help="每次置换跑完整嵌套 CV（与正文 nested-CV 估计同一估计量）")
    ap.add_argument("--radiomics-only", action="store_true",
                    help="AIS：只用 1004 个影像组学列，排除临床列")
    ap.add_argument("--exact", action="store_true",
                    help="被检验量直接用 he.honest_nested_cv（observed 与正文数字完全一致）")
    ap.add_argument("--repeats", type=int, default=10, help="--nested 时外层重复次数（与 honest 一致）")
    ap.add_argument("--out-suffix", type=str, default="")
    args = ap.parse_args()

    X, y, name = load(args)
    print(f"[data] {name}  n={len(y)}  events={int(y.sum())}  d={X.shape[1]}", flush=True)

    if args.nested:
        from sklearn.model_selection import GridSearchCV
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=he.MASTER_SEED)
        model = GridSearchCV(he.make_pipeline(), he.PARAM_GRID, scoring="roc_auc",
                             cv=inner, n_jobs=1, refit=True)
        model_desc = ("nested CV: outer 10-fold repeated %d times, inner 5-fold GridSearch over "
                      "k in {10,20,30} x C in {0.01,0.1,1.0} (same estimator AND same outer resampling "
                      "as the reported honest AUC; inner seed fixed rather than per-fold, the only "
                      "difference from honest_nested_cv)" % args.repeats)
    else:
        model = he.make_pipeline()
        model.set_params(select__k=20, clf__C=0.1)          # 预先指定的主模型
        model_desc = "SelectKBest(k=20)+LogReg(balanced,C=0.1), 10-fold CV"
    if args.nested:
        from sklearn.model_selection import RepeatedStratifiedKFold
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=args.repeats,
                                     random_state=he.MASTER_SEED)
    else:
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=he.MASTER_SEED)

    if args.exact:
        model_desc = ("EXACT: the permuted statistic is honest_nested_cv() itself "
                      "(outer 10-fold x %d repeats, inner 5-fold GridSearch with the same per-fold "
                      "seeding), so the observed value equals the honest AUC reported in the paper"
                      % args.repeats)
        obs, perm_scores, p = exact_nested_permutation(
            X, y, args.n_perm, args.repeats, he.MASTER_SEED, args.n_jobs)
    else:
        obs, perm_scores, p = permutation_test_score(
            model, X, y, scoring="roc_auc", cv=cv,
            n_permutations=args.n_perm, n_jobs=args.n_jobs, random_state=he.MASTER_SEED)

    perm = np.asarray(perm_scores, float)
    out = {
        "dataset": name, "n": int(len(y)), "events": int(y.sum()), "d": int(X.shape[1]),
        "model": model_desc,
        "observed_auc": float(obs), "n_permutations": int(args.n_perm),
        "pvalue": float(p),
        "null_mean": float(perm.mean()), "null_std": float(perm.std(ddof=1)),
        "null_p2.5": float(np.percentile(perm, 2.5)),
        "null_p97.5": float(np.percentile(perm, 97.5)),
        "null_scores": perm.round(5).tolist(),     # 供画 F2 零分布
    }
    RESULTS.mkdir(exist_ok=True)
    f = RESULTS / f"permutation_{name}{args.out_suffix}.json"
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
