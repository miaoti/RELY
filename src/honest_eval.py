"""
诚实评测脚手架（Honest Evaluation Harness） —— 项目地基。

全部在 `data/ICC.csv` 上由我们自己重新产生（不引用任何既有论文数字）。比较"只改评测协议"
下的三条流程，并做泄漏自检：
  1. HONEST       : 无泄漏的重复分层嵌套 CV（外层评估、内层 5 折选特征 k + 调 C），逐折算 AUC 聚合。
  2. LEAKY-CV     : 在【全数据】上缩放 + 选特征(泄漏)，再跑【同样的】重复分层 CV。与 HONEST 仅差
                    "特征选择在 CV 内还是 CV 外"——干净隔离"选择泄漏"的乐观量。
  3. SINGLE-SPLIT : 全数据缩放+选特征(泄漏) 后做单次 70/30；对很多随机划分取分布（mean / 区间 / max）。
                    max = "挑到的最好划分"，正是乐观作者会报的头条数。
  自检 CHECK      : 随机标签——打乱 y 后 HONEST 应 ≈ 0.5（区间覆盖 0.5 即判无泄漏）。

数据约定（据 docs/数据字典.md；标签语义来自未发表论文、仅作工作假设）：
  - 剔除 `Unnamed: 0`、`Image.number`（元数据）、`Smoke`（恒 0 废列）。
  - 标签 y = (Categories == 0) → 阳性 = poor outcome（少数类，57 事件）。
  - 其余列（Age/Bp/Sex/Diabetes + 1004 影像组学特征）作为特征；全表无缺失。

用法：
  python src/honest_eval.py                 # 默认 repeats=10
  python src/honest_eval.py --repeats 20
  python src/honest_eval.py --quick         # 冒烟测试 repeats=3
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV, train_test_split,
)
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ICC.csv"
RESULTS = ROOT / "results"
MASTER_SEED = 20260625

DROP_COLS = ["Unnamed: 0", "Image.number", "Smoke"]
OUTCOME = "Categories"
K_NAIVE = 30  # 乐观流程固定选 30 个特征（贴近既有论文做法）


def load_xy(path=DATA, outcome_col=OUTCOME, pos_label=0):
    df = pd.read_csv(path)
    assert outcome_col in df.columns, f"missing outcome column {outcome_col}"
    y = (df[outcome_col].values == pos_label).astype(int)   # ICC: 阳性=poor=(Categories==0)
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns] + [outcome_col])
    # 自动丢弃非数值列（如 SubjectID 等标识列），其余作为特征
    non_num = X.select_dtypes(exclude="number").columns.tolist()
    if non_num:
        print(f"[load] dropping non-numeric (id-like) columns: {non_num}")
        X = X.drop(columns=non_num)
    return X.values.astype(float), y, list(X.columns)


def _stats(aucs):
    a = np.asarray(aucs, float)
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std(ddof=1)),
            "p2.5": float(np.percentile(a, 2.5)), "p97.5": float(np.percentile(a, 97.5)),
            "min": float(a.min()), "max": float(a.max())}


def _calib_metrics(y, p, bins=10):
    """Brier 分数 + 期望校准误差 ECE（10 等宽分箱）。"""
    y = np.asarray(y, float); p = np.asarray(p, float)
    brier = float(np.mean((p - y) ** 2))
    edges = np.linspace(0, 1, bins + 1); ece = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1]) if i < bins - 1 else (p >= edges[i]) & (p <= edges[i + 1])
        if m.any():
            ece += (m.sum() / len(y)) * abs(y[m].mean() - p[m].mean())
    return brier, float(ece)


def _sens_spec(y, p, thr=0.5):
    y = np.asarray(y); pred = (np.asarray(p) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return sens, spec


def make_pipeline():
    """无泄漏 Pipeline：所有 fit 只在传入的训练数据上发生。"""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),   # ICC 无缺失=无操作；radMLBench 个别表稳妥
        ("var", VarianceThreshold(0.0)),
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif)),
        ("clf", LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=5000)),
    ])

PARAM_GRID = {"select__k": [10, 20, 30], "clf__C": [0.01, 0.1, 1.0]}


def honest_nested_cv(X, y, repeats, seed, n_jobs=-1, grid=None):
    """重复分层嵌套 CV：外层评估，内层(5折)选 k + 调 C。逐外层折算 AUC。
    grid 可覆盖默认 PARAM_GRID（基线子集特征少时用，避免 k 超过特征数）。"""
    grid = grid if grid is not None else PARAM_GRID
    outer = RepeatedStratifiedKFold(n_splits=10, n_repeats=repeats, random_state=seed)
    aucs, briers, eces, senss, specs = [], [], [], [], []
    for i, (tr, te) in enumerate(outer.split(X, y)):
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + i)
        gs = GridSearchCV(make_pipeline(), grid, scoring="roc_auc",
                          cv=inner, n_jobs=n_jobs, refit=True)
        gs.fit(X[tr], y[tr])
        proba = gs.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], proba))
        b, e = _calib_metrics(y[te], proba); briers.append(b); eces.append(e)
        s, sp = _sens_spec(y[te], proba); senss.append(s); specs.append(sp)
    out = _stats(aucs)
    # Nadeau & Bengio (2003) 校正方差：10 折 → n_test/n_train = (1/10)/(9/10) = 1/9
    out["auc_nb_se"] = float(np.sqrt((1.0 / out["n"] + 1.0 / 9.0) * np.var(aucs, ddof=1)))
    out["brier"] = float(np.nanmean(briers)); out["ece"] = float(np.nanmean(eces))
    out["sensitivity"] = float(np.nanmean(senss)); out["specificity"] = float(np.nanmean(specs))
    return out


def _leak_preprocess(X, y, k=K_NAIVE):
    """泄漏的预处理：在全数据(含全部 y)上缩放 + 选特征。返回选好的特征矩阵。"""
    Xi = SimpleImputer(strategy="median").fit_transform(X)
    Xs = StandardScaler().fit_transform(Xi)
    return SelectKBest(f_classif, k=k).fit_transform(Xs, y)


def leaky_cv(X, y, repeats, seed):
    """与 HONEST 仅差：特征选择在 CV 外、用了全部 y（选择泄漏）。同样的重复分层 CV。"""
    Xsel = _leak_preprocess(X, y)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=repeats, random_state=seed)
    aucs = []
    for tr, te in cv.split(Xsel, y):
        clf = LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=5000)
        clf.fit(Xsel[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(Xsel[te])[:, 1]))
    return _stats(aucs)


def single_splits(X, y, n_splits=300, base_seed=0):
    """泄漏预处理后做大量随机 70/30 单次划分；分布 + max（挑最好划分=乐观头条）。"""
    Xsel = _leak_preprocess(X, y)
    aucs = []
    for s in range(n_splits):
        Xtr, Xte, ytr, yte = train_test_split(Xsel, y, test_size=0.30, stratify=y,
                                              random_state=base_seed + s)
        clf = LogisticRegression(class_weight="balanced", solver="liblinear",
                                 max_iter=5000).fit(Xtr, ytr)
        aucs.append(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))
    return _stats(aucs)


def test_tuned_single_splits(X, y, n_splits=300, base_seed=0):
    """乐观机制（复刻既有论文做法）：每个划分在 TRAIN 上拟合多个候选(k×C)，
    但用 TEST 上的 AUC 去【挑最好的那个】（= 在测试集上选模型）。报告"测试集最优"的分布。
    特征选择在训练折内做（无选择泄漏），孤立出"在测试集上选模型"这一项乐观来源。"""
    cands = [(k, C) for k in (7, 15, 30, 50, 100) for C in (0.01, 0.1, 1.0)]
    best = []
    for s in range(n_splits):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y,
                                              random_state=base_seed + s)
        best_auc = 0.0
        for k, C in cands:
            pipe = Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("var", VarianceThreshold(0.0)),
                ("scaler", StandardScaler()),
                ("select", SelectKBest(f_classif, k=min(k, Xtr.shape[1]))),
                ("clf", LogisticRegression(class_weight="balanced", solver="liblinear",
                                           max_iter=5000, C=C)),
            ]).fit(Xtr, ytr)
            auc = roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1])
            best_auc = max(best_auc, auc)          # 在 TEST 上选最好 -> 乐观
        best.append(best_auc)
    return _stats(best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=str(DATA),
                    help="特征表路径（默认 data/ICC.csv；可指向 ISLES/导出的同构表）")
    ap.add_argument("--outcome-col", type=str, default=OUTCOME, help="结局列名（默认 Categories）")
    ap.add_argument("--pos-label", type=int, default=0, help="正类取值（ICC 默认 0=poor）")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    repeats = 3 if args.quick else args.repeats

    rng = np.random.default_rng(MASTER_SEED)
    X, y, cols = load_xy(Path(args.data), args.outcome_col, args.pos_label)
    print(f"[data] {args.data}  X={X.shape}  events(poor=1)={int(y.sum())}/{y.size}  features={len(cols)}")

    t0 = time.time()
    honest = honest_nested_cv(X, y, repeats=repeats, seed=MASTER_SEED)
    leaky = leaky_cv(X, y, repeats=repeats, seed=MASTER_SEED)
    splits = single_splits(X, y, n_splits=(60 if args.quick else 300))
    tuned = test_tuned_single_splits(X, y, n_splits=(60 if args.quick else 300))
    perm = honest_nested_cv(X, rng.permutation(y), repeats=max(3, repeats // 2),
                            seed=MASTER_SEED + 1)

    summary = {
        "data_shape": list(X.shape), "n_events_poor": int(y.sum()),
        "config": {"repeats": repeats, "outer": 10, "inner": 5, "grid": PARAM_GRID,
                   "k_naive": K_NAIVE, "master_seed": MASTER_SEED},
        "honest_nested_cv": honest,
        "leaky_selection_cv": leaky,
        "naive_single_split": splits,
        "test_set_selected_single_split": tuned,
        "gap_leaky_minus_honest": leaky["mean"] - honest["mean"],
        "gap_singlemean_minus_honest": splits["mean"] - honest["mean"],
        "gap_cherrypicked_minus_honest": splits["max"] - honest["mean"],
        "gap_testselected_minus_honest": tuned["mean"] - honest["mean"],
        "random_label_check": perm,
        "runtime_sec": round(time.time() - t0, 1),
    }
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "honest_eval_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    def line(name, s):
        print(f"{name:<28}: {s['mean']:.4f}  (95% pct {s['p2.5']:.3f}-{s['p97.5']:.3f}, "
              f"max {s['max']:.3f}, n={s['n']})")
    print("\n================ RESULTS ================")
    line("HONEST nested-CV", honest)
    print(f"  honest extras: NB-SE={honest['auc_nb_se']:.3f}  Brier={honest['brier']:.3f}  "
          f"ECE={honest['ece']:.3f}  sens={honest['sensitivity']:.3f}  spec={honest['specificity']:.3f}")
    line("LEAKY-selection CV", leaky)
    line("NAIVE single-split (300x)", splits)
    line("TEST-SELECTED split (300x)", tuned)
    print(f"{'gap  LEAKY-CV - HONEST':<28}: {summary['gap_leaky_minus_honest']:+.4f}")
    print(f"{'gap  single-mean - HONEST':<28}: {summary['gap_singlemean_minus_honest']:+.4f}")
    print(f"{'gap  cherry-pick - HONEST':<28}: {summary['gap_cherrypicked_minus_honest']:+.4f}")
    print(f"{'gap  test-selected - HONEST':<28}: {summary['gap_testselected_minus_honest']:+.4f}")
    ok = perm["p2.5"] <= 0.5 <= perm["p97.5"]
    print(f"{'RANDOM-LABEL HONEST':<28}: {perm['mean']:.4f}  (95% pct {perm['p2.5']:.3f}-{perm['p97.5']:.3f})")
    print(f"{'LEAKAGE SANITY':<28}: {'PASS (0.5 within interval)' if ok else 'FAIL — investigate leakage'}")
    print(f"\nsaved -> {out}   ({summary['runtime_sec']}s)")


if __name__ == "__main__":
    main()
