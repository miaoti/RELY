"""
阶段 7 · 临床效用与统计检验。
  1) 配对 AUC 检验：主模型(全1004) vs 年龄+体积，同一外层折逐折 AUC 差，Nadeau-Bengio 校正成对 t（不用池化 DeLong）。
  2) DCA 决策曲线（F5）：主模型 / 年龄+体积 / 全治 / 全不治 的净获益；预设结论=无净获益。
  3) 亚组扫视：按 Sex、Age 三分位报告主模型诚实 AUC（探索性，功效有限）。
  4) Hosmer-Lemeshow：主模型折外校准概率的拟合优度（低功效补充）。
全部基于无泄漏的折外预测。输出 results/stats_ICC.json，图 figures/F5_dca_ICC.png。
用法：python src/stats_clinical.py [--repeats 10]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import honest_eval as he

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGS = ROOT / "results", ROOT / "figures"
SEED = he.MASTER_SEED


def load():
    df = pd.read_csv(ROOT / "data" / "ICC.csv")
    y = (df["Categories"].values == 0).astype(int)
    meta = {"Unnamed: 0", "Image.number", "Smoke", "Categories", "Age", "Bp", "Sex", "Diabetes"}
    rad = [c for c in df.columns if c not in meta]
    vol = [c for c in df.columns if c.startswith("shape_MeshVolume") or c.startswith("shape_VoxelVolume")]
    Xmain = df[rad].values.astype(float)
    Xbase = df[["Age"] + vol].values.astype(float)
    return df, y, Xmain, Xbase


def paired_auc_test(Xmain, Xbase, y, repeats):
    """同折逐折 AUC 差 + Nadeau-Bengio 校正成对 t 检验。"""
    outer = RepeatedStratifiedKFold(n_splits=10, n_repeats=repeats, random_state=SEED)
    base_grid = {"select__k": ["all"], "clf__C": [0.01, 0.1, 1.0]}
    am, ab = [], []
    for i, (tr, te) in enumerate(outer.split(Xmain, y)):
        inner = StratifiedKFold(5, shuffle=True, random_state=SEED + i)
        gm = GridSearchCV(he.make_pipeline(), he.PARAM_GRID, scoring="roc_auc", cv=inner, n_jobs=3).fit(Xmain[tr], y[tr])
        gb = GridSearchCV(he.make_pipeline(), base_grid, scoring="roc_auc", cv=inner, n_jobs=3).fit(Xbase[tr], y[tr])
        am.append(roc_auc_score(y[te], gm.predict_proba(Xmain[te])[:, 1]))
        ab.append(roc_auc_score(y[te], gb.predict_proba(Xbase[te])[:, 1]))
    am, ab = np.array(am), np.array(ab)
    d = am - ab
    J = len(d); se = np.sqrt((1.0 / J + 1.0 / 9.0) * np.var(d, ddof=1))
    t = float(np.mean(d) / se) if se > 0 else float("nan")
    p = float(2 * stats.t.sf(abs(t), df=J - 1)) if np.isfinite(t) else float("nan")
    return {"auc_main": round(float(am.mean()), 4), "auc_age_volume": round(float(ab.mean()), 4),
            "mean_diff": round(float(d.mean()), 4), "nb_se": round(se, 4),
            "t": round(t, 3), "pvalue": round(p, 4), "n_folds": J}


def oof_risk(X, y):
    """折外、Platt 校准的风险概率（无泄漏）。"""
    est = CalibratedClassifierCV(he.make_pipeline().set_params(select__k=20, clf__C=0.1), method="sigmoid", cv=5)
    cv = StratifiedKFold(10, shuffle=True, random_state=SEED)
    return cross_val_predict(est, X, y, cv=cv, method="predict_proba", n_jobs=3)[:, 1]


def net_benefit(y, risk, thr):
    n = len(y); out = []
    for pt in thr:
        pred = risk >= pt
        tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0))
        out.append(tp / n - fp / n * (pt / (1 - pt)))
    return np.array(out)


def hosmer_lemeshow(y, risk, g=10):
    df = pd.DataFrame({"y": y, "p": risk}).sort_values("p")
    df["bin"] = pd.qcut(df["p"], q=g, duplicates="drop")
    obs = df.groupby("bin", observed=True)["y"].agg(["sum", "count"])
    exp = df.groupby("bin", observed=True)["p"].sum()
    o1 = obs["sum"].values; n = obs["count"].values; e1 = exp.values
    e0 = n - e1; o0 = n - o1
    mask = (e1 > 0) & (e0 > 0)
    hl = float(np.sum((o1[mask] - e1[mask]) ** 2 / e1[mask] + (o0[mask] - e0[mask]) ** 2 / e0[mask]))
    dfree = mask.sum() - 2
    return {"HL_stat": round(hl, 3), "df": int(dfree), "pvalue": round(float(stats.chi2.sf(hl, dfree)), 4)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args()
    df, y, Xmain, Xbase = load()
    res = {}

    print("[1] paired AUC test (main vs age+volume)...", flush=True)
    res["paired_auc"] = paired_auc_test(Xmain, Xbase, y, args.repeats)
    print("   ", res["paired_auc"], flush=True)

    print("[2] DCA + OOF risks...", flush=True)
    rm, rb = oof_risk(Xmain, y), oof_risk(Xbase, y)
    thr = np.linspace(0.05, 0.6, 45)
    nb_main, nb_base = net_benefit(y, rm, thr), net_benefit(y, rb, thr)
    prev = y.mean()
    nb_all = prev - (1 - prev) * thr / (1 - thr)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(thr, nb_main, color="#d62728", label="all-radiomics model")
    ax.plot(thr, nb_base, color="#1f77b4", label="age+volume")
    ax.plot(thr, nb_all, color="grey", ls="--", lw=1, label="treat all")
    ax.axhline(0, color="black", lw=1, label="treat none")
    ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit")
    ax.set_title("F5 - ICC decision curve analysis (poor-outcome)")
    ax.set_ylim(-0.1, prev + 0.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "F5_dca_ICC.png", dpi=200)
    fig.savefig(FIGS / "F5_dca_ICC.pdf"); plt.close(fig)
    res["dca"] = {"prevalence": round(float(prev), 4),
                  "max_nb_main": round(float(nb_main.max()), 4),
                  "max_nb_age_volume": round(float(nb_base.max()), 4),
                  "note": "model net benefit ~ treat-none / below age+volume -> no clinical utility"}

    print("[3] subgroup scan...", flush=True)
    sub = {}
    sex = df["Sex"].values
    for v in (0, 1):
        m = sex == v
        if m.sum() > 20 and len(np.unique(y[m])) == 2:
            sub[f"Sex={v}"] = {"n": int(m.sum()), "auc": round(float(roc_auc_score(y[m], rm[m])), 3)}
    age = df["Age"].values; q1, q2 = np.quantile(age, [1 / 3, 2 / 3])
    for lab, m in [("Age_low", age <= q1), ("Age_mid", (age > q1) & (age <= q2)), ("Age_high", age > q2)]:
        if m.sum() > 20 and len(np.unique(y[m])) == 2:
            sub[lab] = {"n": int(m.sum()), "auc": round(float(roc_auc_score(y[m], rm[m])), 3)}
    res["subgroup_main_oof_auc"] = sub
    print("   ", sub, flush=True)

    print("[4] Hosmer-Lemeshow (main, calibrated OOF)...", flush=True)
    res["hosmer_lemeshow_main"] = hosmer_lemeshow(y, rm)
    print("   ", res["hosmer_lemeshow_main"], flush=True)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "stats_ICC.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved -> results/stats_ICC.json , figures/F5_dca_ICC.png")


if __name__ == "__main__":
    main()
