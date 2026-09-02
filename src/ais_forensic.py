"""
AIS forensic (highlight: hard-won real stroke cohort, cautionary worked example).
On the AIS cohort (n=183, 57 events) we show, on a shared ROC-AUC axis, an "optimism ladder"
whose every rung is reproduced by released code on data/ICC.csv with the same l2-LR family,
all at the paper's modelled budget of **B=50 splits**:
  - honest repeated nested CV = 0.46, sitting inside a label-permutation null (p=0.74): chance.
  - fixed protocol, selection IN-FOLD (k=30, C=1): per-split test-AUC "luck" distribution,
    typical 0.47, max over 50 splits = 0.56.
  - + test-set (k,C) tuning: the SAME in-fold protocol with (k,C) chosen on the test block,
    typical 0.54, max over 50 splits x grid = 0.68. The gap 0.68-0.54 is the paper's Delta_split.
Both rungs keep feature selection in-fold, so the ladder adds exactly one thing at a time.
Full-data feature selection is a DIFFERENT channel (leakage), reported separately as
LEAKY_full_data_selection_* and never drawn as a rung of this ladder.
Message: honest evaluation is at chance, yet split luck and test-set tuning climb the SAME
features to a publishable-looking number -- why evaluation reliability, not features, is the issue.
All rungs are persisted to results/ais_forensic.json and read
back for display (no hard-coded number).

round-18 修订：
  ① 特征池改为【纯 1004 影像组学列】（此前经 he.load_xy 混入 Age/Bp/Sex/Diabetes 共 1008 列，
     与正文所称 "1004-feature pool" 及 baselines 的 all_radiomics 行不一致）。
  ② 主报告量改为 Delta_selection = max(splits x grid) - mean(splits)，即【同协议内】纯选择
     乐观，与 50 队列基准的主结局同口径；旧的 max - honest 降为对照字段 delta_vs_honest。
  ③ 新增敏感性分析：剔除 3 对"共享同一 source identifier 且标签相反"的记录（6 行；其中 1 对
     1004 个特征完全相同）后重跑同一阶梯，检验结论不依赖这些冲突记录。
Outputs figures/AIS_forensic.png(.pdf) + results/ais_forensic.json.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import honest_eval as he
import figstyle

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGS = ROOT / "results", ROOT / "figures"
META_COLS = {"Unnamed: 0", "Image.number", "Smoke", "Categories", "Age", "Bp", "Sex", "Diabetes"}


def load_radiomics_only(path=None, drop_conflicting=False):
    """AIS 队列的【纯影像组学】特征池（1004 列），与 baselines.py 的 all_radiomics 子集完全一致。

    以前这里走 he.load_xy，会把 Age/Bp/Sex/Diabetes 4 个临床列一并带入 → 实际喂了 1008 列，
    与正文/表中所称的 "1004-feature pool" 不符。改为显式排除，口径统一。

    drop_conflicting=True 时，剔除共享同一 source identifier 且标签相反的记录对
    （3 对 / 6 行；其中 1 对 1004 个特征完全相同而标签相反）——用于敏感性分析。"""
    import pandas as pd
    df = pd.read_csv(path or (ROOT / "data" / "ICC.csv"))
    n_before = len(df)
    dropped = 0
    if drop_conflicting and "Image.number" in df.columns:
        dup = df[df.duplicated("Image.number", keep=False)]
        bad = [g for g, s in dup.groupby("Image.number") if s["Categories"].nunique() > 1]
        df = df[~df["Image.number"].isin(bad)].reset_index(drop=True)
        dropped = n_before - len(df)
    cols = [c for c in df.columns if c not in META_COLS]
    y = (df["Categories"].values == 0).astype(int)      # 阳性 = poor outcome
    X = df[cols].select_dtypes("number").values.astype(float)
    return X, y, cols, dropped

# selection-optimism protocol (= paper's Delta: max over splits x grid, feature selection in-fold)
KS, CS, NSPLIT = [7, 15, 30, 50, 100], [0.01, 0.1, 1.0], 500
def _lr(C):
    return LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=5000, C=C)


def compute_rungs(X, y):
    """Returns (split_shopping_aucs[500], tuned_aucs[500]) -- both real per-split distributions.
    split_shopping: conventional pipeline (full-data FS, k=30) + single l2-LR, per-split test AUC
                    (max over splits = the split-shopping rung, matching the paper's 0.67 headline).
    tuned:          feature selection IN-FOLD + (k,C) chosen on the TEST set, per-split best
                    (max over splits = max over splits x grid = the paper's Delta numerator; this
                    is the benchmark's own protocol -- pure selection optimism, no feature leakage)."""
    # split-shopping rung: conventional full-data feature selection + single fixed model
    Xsel = he._leak_preprocess(X, y, k=30)
    split_shopping = []
    for s in range(NSPLIT):
        Xtr, Xte, ytr, yte = train_test_split(Xsel, y, test_size=0.30, stratify=y, random_state=s)
        split_shopping.append(roc_auc_score(yte, _lr(1.0).fit(Xtr, ytr).predict_proba(Xte)[:, 1]))
    # tuned rung: in-fold feature selection, (k,C) tuned on the test set (benchmark protocol)
    tuned = []
    for s in range(NSPLIT):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y, random_state=s)
        best = 0.0
        for k in KS:
            for C in CS:
                pipe = Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("var", VarianceThreshold(0.0)),
                    ("scaler", StandardScaler()),
                    ("select", SelectKBest(f_classif, k=min(k, Xtr.shape[1]))),
                    ("clf", _lr(C)),
                ]).fit(Xtr, ytr)
                best = max(best, roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1]))
        tuned.append(best)
    return np.array(split_shopping), np.array(tuned)


def compute_infold_fixed(X, y, k=30, C=1.0):
    """干净的固定协议档：**折内**选特征 (k=30) + 固定 C=1 的 l2-LR，逐划分 test AUC。
    这一档与 tuned 档只差"是否在 test 上调 (k,C)"，所以 max 之间的差才真的只是"多了调参"。
    此前用的是 _leak_preprocess（全数据选特征），那是另一条通道，不能当同一梯子的下一级。"""
    out = []
    for sd in range(NSPLIT):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y, random_state=sd)
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("var", VarianceThreshold(0.0)),
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_classif, k=min(k, Xtr.shape[1]))),
            ("clf", _lr(C)),
        ]).fit(Xtr, ytr)
        out.append(roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1]))
    return np.array(out)


def main():
    X, y, cols, _ = load_radiomics_only()          # 1004 radiomics features (no clinical columns)
    # 诚实档在【同一个 1004 特征池】上现算（repeats=20，与 baselines 的 all_radiomics 行同协议
    # 同种子），而不是读 1008 列那次的 honest_eval_summary.json —— 保证阶梯与基线表同一个数。
    honest = he.honest_nested_cv(X, y, repeats=20, seed=he.MASTER_SEED, n_jobs=3,
                                 boot=0)["pooled_auc"]
    # 优先用【嵌套、exact】那次置换：它的被检验量就是 honest_nested_cv 本身，
    # 与图中"honest"这一档同一个估计量（旧的固定主模型版本给的是 0.50，与 0.47 口径不符）。
    # 按内容挑，不按文件名猜：置换检验的 observed 必须就是本次现算的 honest AUC，
    # 否则说明那份产物是改口径之前留下的（round-25 正是这么被坑到的）。
    cands = [RESULTS / "permutation_ICC.json", RESULTS / "permutation_ICC_nested.json"]
    cands = [c for c in cands if c.exists()]
    if not cands:
        raise SystemExit("no permutation artifact; run permutation_test.py --exact --radiomics-only")
    loaded = [(c, json.loads(c.read_text(encoding="utf-8"))) for c in cands]
    pf, perm = min(loaded, key=lambda t: abs(t[1]["observed_auc"] - honest))
    if abs(perm["observed_auc"] - honest) > 5e-4:
        raise SystemExit(
            "stale permutation artifact: observed %.4f != honest %.4f. Re-run\n"
            "  python src/permutation_test.py --exact --radiomics-only --n-perm 200 --repeats 20"
            % (perm["observed_auc"], honest))
    null = np.array(perm["null_scores"]); pval = perm["pvalue"]
    perm_observed = perm["observed_auc"]; perm_source = pf.name

    shop_aucs, tuned_aucs = compute_rungs(X, y)
    fixed_aucs = compute_infold_fixed(X, y)      # 干净的固定协议档（折内选特征）
    # 主口径 = benchmark 的选择预算 B=50（种子 0..49，与 test_tuned_single_splits 逐一对应）。
    # 500 个划分的结果保留为"加大搜索预算"的次级对照，绝不用来和 50 划分的 kappa/闸门比。
    B_MAIN = 50
    shop_b, tuned_b = shop_aucs[:B_MAIN], tuned_aucs[:B_MAIN]
    fixed_b = fixed_aucs[:B_MAIN]
    cherry = float(shop_b.max())           # split-shopping rung (single model), B=50
    top = float(tuned_b.max())             # + test-set tuning rung, B=50
    delta_sel = top - float(tuned_b.mean())        # paper's Delta_split at B=50

    # 敏感性分析：剔除 3 对"同 ID / 标签相反"的记录后重跑同一阶梯
    Xs, ys, _, n_dropped = load_radiomics_only(drop_conflicting=True)
    shop_s, tuned_s = compute_rungs(Xs, ys)
    shop_sb, tuned_sb = shop_s[:B_MAIN], tuned_s[:B_MAIN]
    sens = {"n_rows_dropped": n_dropped, "n_remaining": int(len(ys)),
            "events_remaining": int(ys.sum()),
            "split_shopping_max": float(shop_sb.max()),
            "tuned_mean": float(tuned_sb.mean()), "tuned_max": float(tuned_sb.max()),
            "delta_selection": float(tuned_sb.max() - tuned_sb.mean())}

    # persist every rung (reproducible + persisted; display reads from here, nothing hard-coded)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "ais_forensic.json").write_text(json.dumps({
        "n_features": int(X.shape[1]), "n": int(len(y)), "events": int(y.sum()),
        "honest": honest, "permutation_p": pval,
        "permutation_observed_auc": perm_observed, "permutation_source": perm_source,
        # 干净梯子：同一个折内协议，先"挑最幸运的划分"，再"额外在 test 上调 (k,C)"。
        "infold_fixed_mean": float(fixed_b.mean()),
        "infold_fixed_max": float(fixed_b.max()),
        "infold_fixed_delta_split": float(fixed_b.max() - fixed_b.mean()),
        "infold_fixed_aucs": [round(float(x), 5) for x in fixed_aucs],
        "LEAKY_full_data_selection_mean": float(shop_b.mean()),
        "LEAKY_full_data_selection_max": cherry,
        "leaky_note": "full-data feature selection (k=30) then one fixed model. NOT a rung of "
                      "the in-fold ladder: it is a different (leakage) channel, reported "
                      "separately so the ladder adds exactly one thing at a time.",
        "split_shopping_mean": float(shop_b.mean()), "split_shopping_max": cherry,
        "tuned_mean": float(tuned_b.mean()), "tuned_max": top,
        "delta_selection": delta_sel,
        # 逐划分 AUC 全量数组：供 fig_method.py 画真实分布（而不是示意图）
        "split_shopping_aucs": [round(float(x), 5) for x in shop_aucs],
        "tuned_aucs": [round(float(x), 5) for x in tuned_aucs],
        "delta_vs_honest": top - honest,
        "escalation_larger_budget": {
            "note": "同一协议，只把选择预算从 B=50 加到 B=500。kappa 随 log B 增长，所以这一档"
                    "不能用 B=50 拟合的 kappa/calculator/gate 去解释；仅作预算依赖性的证据。",
            "B": int(NSPLIT),
            "split_shopping_max": float(shop_aucs.max()),
            "tuned_mean": float(tuned_aucs.mean()), "tuned_max": float(tuned_aucs.max()),
            "delta_selection": float(tuned_aucs.max() - tuned_aucs.mean())},
        "protocol": {"B_modelled": B_MAIN, "grid_k": KS, "grid_C": CS,
                     "n_splits_computed": NSPLIT, "seeds": "0..%d (main rungs use 0..%d)" % (NSPLIT - 1, B_MAIN - 1),
                     "feature_pool": "1004 radiomics columns only (clinical Age/Bp/Sex/Diabetes excluded, "
                                     "matching the all_radiomics baseline row)",
                     "feature_selection": "in-fold SelectKBest(f_classif) (no leakage)",
                     "definition": "Delta_split = max over the B=50 splits of each split's grid-best "
                                   "AUC, minus the mean of those 50; same budget as the 50-cohort benchmark"},
        "sensitivity_drop_conflicting_ids": sens,
    }, indent=2), encoding="utf-8")

    figstyle.apply()
    from scipy.stats import gaussian_kde
    fixed_max = float(fixed_b.max())      # 与 rung/山脊同一个 B=50 的折内固定协议

    # ---- horizontal "optimism ladder": three lanes on a shared ROC-AUC axis ----
    fig, ax = plt.subplots(figsize=(figstyle.SPAN, 3.15))
    XMIN, XMAX = 0.32, 0.82
    xs = np.linspace(XMIN, XMAX, 400)
    RH, SP = 0.70, 1.16                       # ridge height, lane spacing
    yb = {"honest": 0.0, "cherry": SP, "top": 2 * SP}

    def ridge(samples, base, color):
        k = gaussian_kde(samples); dz = k(xs); dz = dz / dz.max() * RH
        ax.fill_between(xs, base, base + dz, color=color, alpha=0.32, lw=0, zorder=3)
        ax.plot(xs, base + dz, color=color, lw=1.2, alpha=0.9, zorder=3.1)

    for b in yb.values():                          # faint lane baselines
        ax.plot([XMIN, XMAX], [b, b], color="0.86", lw=0.8, zorder=0.5)
    ax.axvline(0.5, color=figstyle.C_CHANCE, ls=(0, (1, 1.7)), lw=1.2, zorder=1)
    ax.text(0.508, yb["top"] + RH + 0.10, "chance", color=figstyle.C_CHANCE, fontsize=7,
            ha="left", va="bottom", style="italic")

    ridge(null, yb["honest"], figstyle.GREEN)      # honest nested-CV null (green, paper-wide)
    # 一律用 B=50（与论文主口径、与下面的 rung 完全一致；此前山脊用 500、rung 用 50）。
    ridge(fixed_b, yb["cherry"], figstyle.BLUE)  # 折内固定协议 (k=30, C=1) 的逐划分分布
    ridge(tuned_b, yb["top"], figstyle.RED)        # 再加 test-set (k,C) 调参
    ax.text(0.383, 0.33, "permutation null\n(labels shuffled), $p$=%.2f" % pval, fontsize=6.1,
            color="#0a6b4e", style="italic", ha="center", va="center", zorder=12, linespacing=0.95)

    def rung(x, base, color, big):
        ax.plot([x, x], [base, base + RH], color=color, lw=2.6, zorder=10, solid_capstyle="round")
        ax.plot(x, base + RH, "o", color=color, ms=7.5, zorder=11,
                markeredgecolor="white", markeredgewidth=1.2)
        ax.annotate(big, xy=(x, base + RH), xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=11, fontweight="bold", color=color, zorder=12)

    rung(honest, yb["honest"], figstyle.GREEN, f"{honest:.2f}")
    rung(fixed_max, yb["cherry"], figstyle.BLUE, f"{fixed_max:.2f}")
    rung(top, yb["top"], figstyle.RED, f"{top:.2f}")
    ax.text(fixed_max + 0.012, yb["cherry"] + RH * 0.60, "max of\n50 splits", fontsize=5.7,
            color=figstyle.BLUE, style="italic", ha="left", va="center", linespacing=0.92, zorder=12)
    ax.text(top + 0.012, yb["top"] + RH * 0.60, "max of\n50$\\times$grid", fontsize=5.7,
            color=figstyle.RED, style="italic", ha="left", va="center", linespacing=0.92, zorder=12)

    def climb(x0, y0, x1, y1, txt, lx, ly, rad):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="0.45", lw=1.4,
                                    connectionstyle="arc3,rad=%g" % rad), zorder=8)
        ax.text(lx, ly, txt, fontsize=6.8, color="black", ha="center", va="center",
                style="italic", bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
    climb(honest, yb["honest"] + RH, fixed_max, yb["cherry"] + 0.02, "+ split luck", 0.560, 1.02, 0.06)
    # 这个标签往右上挪：在 0.585/2.06 处它压在 "0.56" 那个 rung 数字上（改配色之前就压着）。
    climb(fixed_max, yb["cherry"] + RH, top, yb["top"] + 0.02, "+ test-set\n($k,C$) tuning", 0.638, 2.14, 0.08)

    ax.set_yticks([yb["honest"] + RH / 2, yb["cherry"] + RH / 2, yb["top"] + RH / 2])
    ax.set_yticklabels(["honest\nnested CV", "fixed protocol\n(best of 50 splits)", "+ test-set\n($k,C$) tuning"])
    for t, col in zip(ax.get_yticklabels(), [figstyle.GREEN, figstyle.BLUE, figstyle.RED]):
        t.set_color(col); t.set_fontweight("bold"); t.set_fontsize(6.9); t.set_linespacing(1.4)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xlim(XMIN, XMAX); ax.set_ylim(-0.16, yb["top"] + RH + 0.30)
    ax.set_xlabel("ROC-AUC   (AIS cohort, $n$=183, 57 events)")
    ax.spines["left"].set_visible(False)
    ax.grid(False); ax.grid(axis="x", alpha=0.18, zorder=0)
    ax.set_title("One real cohort: honest evaluation is at chance;\nsplit-shopping and test-set "
                 "tuning climb it to a publishable-looking %.2f" % top, fontsize=8.5, pad=6, linespacing=1.15)
    fig.tight_layout()
    fig.savefig(FIGS / "AIS_forensic.png")
    fig.savefig(FIGS / "AIS_forensic.pdf"); plt.close(fig)
    print(f"[main]  d={X.shape[1]}  honest={honest:.3f}  split_shopping(max 500)={cherry:.3f}  "
          f"tuned mean/max={tuned_aucs.mean():.3f}/{top:.3f}  "
          f"Delta_selection={delta_sel:.3f}  (max-honest={top-honest:.3f})")
    print(f"[sens]  dropped {n_dropped} conflicting rows -> n={sens['n_remaining']} "
          f"events={sens['events_remaining']}  tuned mean/max="
          f"{sens['tuned_mean']:.3f}/{sens['tuned_max']:.3f}  "
          f"Delta_selection={sens['delta_selection']:.3f}")
    print("saved -> figures/AIS_forensic.png/.pdf + results/ais_forensic.json")


if __name__ == "__main__":
    main()
