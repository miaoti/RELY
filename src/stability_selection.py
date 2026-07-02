"""
阶段 5 · 分组稀疏稳定性选择 + Nogueira 稳定性指标（C3 方法学证据）。

- 三种选择器在 B 次分层半采样上做稳定性选择（Meinshausen-Bühlmann）：
    sgl   = sparse-group LASSO（asgl, model=logit），组=特征族（firstorder/glcm/.../shape）
    flatL1= L1 逻辑回归（sklearn）
    univ  = 单变量 SelectKBest(f_classif, k=15)
- 每个选择器记录 B×p 选择矩阵 → 每特征选择频率 + **Nogueira et al.(2018) 稳定性 Φ**。
  C3 主张：分组结构化选择比 flat 更稳（Φ 更高）= 可量化方法学增益。
- F6：按特征族展示选择频率（sgl）。
- 把 sgl 稳定集（top-by-freq）喂主模型（LR），嵌套 CV 报诚实 AUC（小样本无信号下预期≈随机）。

无泄漏：每次子采样内部各自标准化、各自选择。结果存 results/stability_ICC.json，图 figures/F6_stability_ICC.png。
用法：python src/stability_selection.py [--B 50]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
import asgl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import honest_eval as he

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGS = ROOT / "results", ROOT / "figures"
SEED = he.MASTER_SEED
TOL = 1e-8


def load():
    df = pd.read_csv(ROOT / "data" / "ICC.csv")
    y = (df["Categories"].values == 0).astype(int)
    meta = {"Unnamed: 0", "Image.number", "Smoke", "Categories", "Age", "Bp", "Sex", "Diabetes"}
    rad = [c for c in df.columns if c not in meta]
    fam = [c.split("_")[0] for c in rad]
    families = sorted(set(fam))
    gidx = np.array([families.index(f) for f in fam])
    X = df[rad].values.astype(float)
    return X, y, rad, np.array(fam), gidx, families


def sel_sgl(Xs, y, gidx):
    m = asgl.Regressor(model="logit", penalization="sgl", lambda1=0.05, alpha=0.5)
    m.fit(Xs, y, group_index=gidx)
    return np.abs(np.asarray(m.coef_).ravel()) > TOL


def sel_flatl1(Xs, y, gidx):
    m = LogisticRegression(penalty="l1", solver="liblinear", C=0.1,
                           class_weight="balanced", max_iter=5000).fit(Xs, y)
    return np.abs(m.coef_.ravel()) > TOL


def sel_univ(Xs, y, gidx):
    return SelectKBest(f_classif, k=15).fit(Xs, y).get_support()


def nogueira_stability(Z):
    """Nogueira et al. (2018, JMLR) 稳定性 Φ ∈ (-∞,1]，1=完全稳定。Z: M×d 二值。"""
    Z = np.asarray(Z, float); M, d = Z.shape
    pf = Z.mean(0)
    kbar = Z.sum(1).mean()
    denom = (kbar / d) * (1 - kbar / d)
    if denom <= 0:
        return float("nan")
    return float(1 - (M / (M - 1) * pf * (1 - pf)).mean() / denom)


def nested_sgl_auc(X, y, gidx, repeats=5, seed=SEED):
    """无泄漏：每外层折在【训练折内】做 sgl 选择 + LR 拟合，测试折评估。"""
    outer = RepeatedStratifiedKFold(n_splits=10, n_repeats=repeats, random_state=seed)
    aucs, nsel = [], []
    for tr, te in outer.split(X, y):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        m = asgl.Regressor(model="logit", penalization="sgl", lambda1=0.05, alpha=0.5)
        m.fit(Xtr, y[tr], group_index=gidx)
        coef = np.abs(np.asarray(m.coef_).ravel())
        sel = coef > TOL
        if sel.sum() == 0:
            sel = np.zeros(len(coef), bool); sel[int(np.argmax(coef))] = True
        clf = LogisticRegression(class_weight="balanced", solver="liblinear",
                                 max_iter=5000).fit(Xtr[:, sel], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(Xte[:, sel])[:, 1]))
        nsel.append(int(sel.sum()))
    a = np.array(aucs)
    return {"mean": float(a.mean()), "p2.5": float(np.percentile(a, 2.5)),
            "p97.5": float(np.percentile(a, 97.5)), "mean_n_selected": float(np.mean(nsel))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=50, help="子采样次数")
    args = ap.parse_args()
    X, y, rad, fam, gidx, families = load()
    p = X.shape[1]
    selectors = {"sgl(group)": sel_sgl, "flatL1": sel_flatl1, "univ_top15": sel_univ}
    Z = {k: np.zeros((args.B, p), bool) for k in selectors}
    rng = np.random.default_rng(SEED)
    print(f"[stability] X={X.shape} groups={len(families)} B={args.B}", flush=True)

    t0 = time.time()
    for b in range(args.B):
        s = int(rng.integers(1 << 30))
        Xh, _, yh, _ = train_test_split(X, y, test_size=0.5, stratify=y, random_state=s)
        Xs = StandardScaler().fit_transform(Xh)             # 折内标准化（无泄漏）
        for name, fn in selectors.items():
            try:
                Z[name][b] = fn(Xs, yh, gidx)
            except Exception as e:
                print(f"  b={b} {name} ERR {type(e).__name__}: {str(e)[:60]}", flush=True)
        if (b + 1) % 10 == 0:
            print(f"  {b+1}/{args.B} done ({time.time()-t0:.0f}s)", flush=True)

    stab = {k: nogueira_stability(Z[k]) for k in selectors}
    freq = {k: Z[k].mean(0) for k in selectors}
    print("\nNogueira stability:", {k: round(v, 3) for k, v in stab.items()})

    # sgl 选择 → 诚实 AUC（无泄漏：折内选择，替代旧的全数据选择版）
    order = np.argsort(-freq["sgl(group)"])
    nested = nested_sgl_auc(X, y, gidx, repeats=5, seed=SEED)
    print(f"nested sgl honest AUC = {nested['mean']:.3f} [{nested['p2.5']:.2f},{nested['p97.5']:.2f}] "
          f"(avg {nested['mean_n_selected']:.0f} feats/fold)")

    # 每特征族平均选择频率（sgl）
    fam_freq = {f: float(freq["sgl(group)"][fam == f].mean()) for f in families}

    out = {
        "B": args.B, "n_features": p, "families": families,
        "nogueira_stability": stab,
        "family_mean_selfreq_sgl": fam_freq,
        "n_selected_freq_ge_0.5": {k: int((freq[k] >= 0.5).sum()) for k in selectors},
        "nested_sgl_honest_auc": nested["mean"],
        "nested_sgl_auc_ci": [nested["p2.5"], nested["p97.5"]],
        "nested_sgl_mean_n_selected": nested["mean_n_selected"],
        "top10_features_sgl": [rad[i] for i in order[:10]],
        "runtime_sec": round(time.time() - t0, 1),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "stability_ICC.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # F6
    FIGS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    fams = families
    vals = [fam_freq[f] for f in fams]
    ax.bar(fams, vals, color="#4292c6", edgecolor="white")
    ax.axhline(0.5, color="grey", ls="--", lw=1, label="stability threshold 0.5")
    ax.set_ylabel("Mean selection frequency (sparse-group)")
    ax.set_xlabel("Radiomic feature family")
    ax.set_title(f"F6 - ICC stability selection by family  "
                 f"(Nogueira: sgl={stab['sgl(group)']:.2f} vs flatL1={stab['flatL1']:.2f})")
    ax.set_ylim(0, 1); ax.legend(fontsize=8); plt.xticks(rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(FIGS / "F6_stability_ICC.png", dpi=200)
    fig.savefig(FIGS / "F6_stability_ICC.pdf"); plt.close(fig)
    print(f"saved -> results/stability_ICC.json , figures/F6_stability_ICC.png  ({out['runtime_sec']}s)")


if __name__ == "__main__":
    main()
