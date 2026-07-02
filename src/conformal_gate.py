"""
C2 · 类条件共形 + 选择性预测"可靠性闸门" —— 论文方法学新意核心。

演示（小样本 / 类不平衡下）：
  - 边际(marginal)共形：整体覆盖看似达标(≈1-α)，但【少数类严重欠覆盖】；
  - 类条件(Mondrian)共形：把每类覆盖修复到≈1-α；
  - 二项(Wilson)置信区间：少数类测试样本少 → 区间宽，如实报告；
  - 选择性分类：仅当预测集为单点时发声，否则弃权 → 风险-覆盖。

无泄漏：模型只在 proper-train 上 fit；共形阈值只在 calibration 上定；test 不参与任何拟合。
自实现 split-conformal（分类，nonconformity = 1 - p_hat(true class)），不依赖 MAPIE/crepes。

用法：
  python src/conformal_gate.py                              # 默认 ICC（data/ICC.csv）
  python src/conformal_gate.py --radmlbench Granata2024     # 更不平衡的 radMLBench 集
  python src/conformal_gate.py --data <csv> --outcome-col Target --pos-label 1
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MASTER_SEED = 20260625
ALPHAS = [0.05, 0.10, 0.20]


def model():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(0.0)),
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_classif, k=20)),
        ("clf", LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=5000, C=0.1)),
    ])


def load(args):
    if args.radmlbench:
        import radMLBench as rb
        df = rb.loadData(args.radmlbench)
        y = df["Target"].astype(int).values
        X = df.drop(columns=[c for c in ("Target", "ID") if c in df.columns]).select_dtypes("number").values.astype(float)
        return X, y, args.radmlbench
    df = pd.read_csv(args.data)
    y = (df[args.outcome_col].values == args.pos_label).astype(int)
    drop = [c for c in ["Unnamed: 0", "Image.number", "Smoke", args.outcome_col] if c in df.columns]
    X = df.drop(columns=drop).select_dtypes("number").values.astype(float)
    return X, y, Path(args.data).stem


def conformal_quantile(scores, alpha):
    """split-conformal 阈值：第 ceil((n+1)(1-α)) 小的分数（保守有限样本有效）。"""
    n = len(scores)
    if n == 0:
        return 1.0
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return 1.0  # 样本太少，无法保证该水平 → 阈值放到最大（集合更大）
    return float(np.sort(scores)[k - 1])


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def run(X, y, repeats, seed, alphas=ALPHAS):
    minority = int(np.argmin(np.bincount(y)))      # 少数类标签
    # 累计每类覆盖计数（按方法、按 alpha），用于覆盖率与二项 CI
    acc = {a: {"marg": {0: [0, 0], 1: [0, 0]}, "mond": {0: [0, 0], 1: [0, 0]}} for a in alphas}
    sel = {a: {"singleton": 0, "correct": 0, "total": 0} for a in alphas}  # mondrian 选择性
    rng = np.random.default_rng(seed)
    for r in range(repeats):
        s = int(rng.integers(1 << 30))
        # 60/20/20 stratified: train / calibration / test
        Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=s)
        Xcal, Xte, ycal, yte = train_test_split(Xtmp, ytmp, test_size=0.5, stratify=ytmp, random_state=s)
        m = model().fit(Xtr, ytr)
        pcal = m.predict_proba(Xcal)            # [:,0]=P(0), [:,1]=P(1)
        pte = m.predict_proba(Xte)
        s_cal = 1 - pcal[np.arange(len(ycal)), ycal]      # nonconformity of true label
        s_te = 1 - pte                                    # s_te[:,c] = nonconformity for candidate c
        for a in alphas:
            q_marg = conformal_quantile(s_cal, a)
            q0 = conformal_quantile(s_cal[ycal == 0], a)
            q1 = conformal_quantile(s_cal[ycal == 1], a)
            set_marg = s_te <= q_marg                      # (n_te, 2) bool
            set_mond = np.column_stack([s_te[:, 0] <= q0, s_te[:, 1] <= q1])
            for c in (0, 1):
                idx = yte == c
                acc[a]["marg"][c][0] += int(set_marg[idx, c].sum()); acc[a]["marg"][c][1] += int(idx.sum())
                acc[a]["mond"][c][0] += int(set_mond[idx, c].sum()); acc[a]["mond"][c][1] += int(idx.sum())
            # 选择性（用 mondrian 集合）：单点集才发声
            size = set_mond.sum(axis=1)
            single = size == 1
            pred = np.where(set_mond[:, 1] & (size == 1), 1, 0)
            sel[a]["singleton"] += int(single.sum())
            sel[a]["correct"] += int((single & (pred == yte)).sum())
            sel[a]["total"] += len(yte)
    return minority, acc, sel


def run_cvplus(X, y, alphas, repeats, seed, K=5):
    """CV+/cross-conformal 类条件覆盖：80% 池做 K 折交叉拟合当校准（每点用未见它的模型打分），
    20% 测试用 K 个折模型的平均概率，免切第三份独立校准集（Barber et al. 2021）。"""
    minority = int(np.argmin(np.bincount(y)))
    acc = {a: {"marg": {0: [0, 0], 1: [0, 0]}, "mond": {0: [0, 0], 1: [0, 0]}} for a in alphas}
    rng = np.random.default_rng(seed)
    for r in range(repeats):
        s = int(rng.integers(1 << 30))
        Xpool, Xte, ypool, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=s)
        skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=s)
        s_cal = np.empty(len(ypool)); tprobs = []
        for tr, ca in skf.split(Xpool, ypool):
            m = model().fit(Xpool[tr], ypool[tr])
            pca = m.predict_proba(Xpool[ca])
            s_cal[ca] = 1 - pca[np.arange(len(ca)), ypool[ca]]   # 交叉共形分数（模型未见该点）
            tprobs.append(m.predict_proba(Xte))
        s_te = 1 - np.mean(tprobs, axis=0)                       # CV+ 平均概率
        for a in alphas:
            q = conformal_quantile(s_cal, a)
            q0 = conformal_quantile(s_cal[ypool == 0], a); q1 = conformal_quantile(s_cal[ypool == 1], a)
            sm = s_te <= q
            so = np.column_stack([s_te[:, 0] <= q0, s_te[:, 1] <= q1])
            for c in (0, 1):
                idx = yte == c
                acc[a]["marg"][c][0] += int(sm[idx, c].sum()); acc[a]["marg"][c][1] += int(idx.sum())
                acc[a]["mond"][c][0] += int(so[idx, c].sum()); acc[a]["mond"][c][1] += int(idx.sum())
    return minority, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=str(ROOT / "data" / "ICC.csv"))
    ap.add_argument("--outcome-col", type=str, default="Categories")
    ap.add_argument("--pos-label", type=int, default=0)
    ap.add_argument("--radmlbench", type=str, default="")
    ap.add_argument("--repeats", type=int, default=300)
    ap.add_argument("--cvplus", action="store_true", help="用 CV+/cross-conformal（免切第三份校准集）")
    args = ap.parse_args()

    X, y, name = load(args)
    if args.cvplus:
        minority, acc = run_cvplus(X, y, ALPHAS, args.repeats, MASTER_SEED)
        print(f"[CV+] {name}  n={len(y)}  minority=class{minority}  repeats={args.repeats}\n")
        out = {"dataset": name, "method": "cvplus", "minority": minority, "levels": {}}
        for a in ALPHAS:
            ma = sum(acc[a]["marg"][c][0] for c in (0, 1)) / sum(acc[a]["marg"][c][1] for c in (0, 1))
            da = sum(acc[a]["mond"][c][0] for c in (0, 1)) / sum(acc[a]["mond"][c][1] for c in (0, 1))
            mk, mn = acc[a]["marg"][minority]; dk, dn = acc[a]["mond"][minority]
            mmin = mk / mn if mn else float("nan"); dmin = dk / dn if dn else float("nan")
            flag = " <-- UNDERCOVERAGE" if mmin < (1 - a) - 0.02 else ""
            print(f"alpha={a} target={1-a:.2f}: MARGINAL overall={ma:.3f} minority={mmin:.3f}{flag}"
                  f"   |   MONDRIAN overall={da:.3f} minority={dmin:.3f}")
            out["levels"][a] = {"target": 1 - a, "marg_overall": ma, "mond_overall": da,
                                "marg_minority": mmin, "mond_minority": dmin}
        sane = all(abs(out["levels"][a]["marg_overall"] - (1 - a)) < 0.05 for a in ALPHAS)
        print(f"\nSANITY (marginal overall ~ nominal): {'PASS' if sane else 'CHECK'}")
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / f"conformal_cvplus_{name}.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved -> results/conformal_cvplus_{name}.json")
        return
    minority, acc, sel = run(X, y, args.repeats, MASTER_SEED)
    n1 = int(y.sum()); n0 = len(y) - n1
    print(f"[data] {name}  n={len(y)}  class0={n0} class1={n1}  minority=class{minority}")
    print(f"[setup] {args.repeats} repeats x (60% train / 20% calib / 20% test), model=LR(k=20)\n")

    out = {"dataset": name, "n": len(y), "class0": n0, "class1": n1, "minority": minority,
           "repeats": args.repeats, "levels": {}}
    for a in ALPHAS:
        tgt = 1 - a
        def cov(meth, c):
            k, n = acc[a][meth][c]
            return (k / n if n else float("nan")), wilson(k, n), n
        m_all = (sum(acc[a]["marg"][c][0] for c in (0, 1)) / sum(acc[a]["marg"][c][1] for c in (0, 1)))
        d_all = (sum(acc[a]["mond"][c][0] for c in (0, 1)) / sum(acc[a]["mond"][c][1] for c in (0, 1)))
        (mmin, mci, mn) = cov("marg", minority)
        (dmin, dci, dn) = cov("mond", minority)
        s = sel[a]
        print(f"alpha={a}  target coverage={tgt:.2f}")
        print(f"  MARGINAL  overall={m_all:.3f}  minority(class{minority})={mmin:.3f} "
              f"[Wilson {mci[0]:.2f}-{mci[1]:.2f}, pooled n={mn}]   {'<-- UNDERCOVERAGE' if mmin < tgt-0.02 else ''}")
        print(f"  MONDRIAN  overall={d_all:.3f}  minority(class{minority})={dmin:.3f} "
              f"[Wilson {dci[0]:.2f}-{dci[1]:.2f}]   {'(fixed)' if dmin >= tgt-0.02 else ''}")
        print(f"  SELECTIVE(mondrian): coverage(发声率)={s['singleton']/s['total']:.3f}  "
              f"accuracy@singletons={(s['correct']/s['singleton'] if s['singleton'] else float('nan')):.3f}\n")
        out["levels"][a] = {
            "target": tgt, "marg_overall": m_all, "mond_overall": d_all,
            "marg_minority": mmin, "marg_minority_wilson": mci, "mond_minority": dmin,
            "mond_minority_wilson": dci, "minority_pooled_n": mn,
            "selective_coverage": s["singleton"] / s["total"],
            "selective_accuracy": (s["correct"] / s["singleton"] if s["singleton"] else None),
        }
    RESULTS.mkdir(exist_ok=True)
    f = RESULTS / f"conformal_{name}.json"
    f.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved -> {f}")


if __name__ == "__main__":
    main()
