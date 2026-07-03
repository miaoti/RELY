"""
AIS forensic (highlight: hard-won real stroke cohort, cautionary worked example).
On the AIS cohort (n=183, 57 events) we show, on a shared ROC-AUC axis, an "optimism ladder"
whose every rung is reproduced by released code on data/ICC.csv with the same l2-LR family:
  - honest repeated nested CV = 0.47, sitting inside a label-permutation null (p=0.54): chance.
  - split-shopping: the conventional pipeline (full-data feature selection + a single model),
    per-split test-AUC "luck" distribution; its max over 500 splits = 0.67.
  - + test-set (k,C) tuning: the SAME split-shopping with (k,C) chosen on the test set and
    feature selection kept IN-FOLD (the benchmark's own protocol; pure selection optimism, no
    feature leakage); per-split best distribution, max over 500 splits x grid = 0.72 -- this is
    the paper's Delta numerator realized on this cohort (Delta = 0.72 - 0.47 = 0.25), the same
    quantity validated across the fifty cohorts.
Message: honest evaluation is at chance, yet split luck and test-set tuning climb the SAME
features to a publishable-looking 0.72 -- why evaluation reliability, not features, is the issue.
The conventional full-data feature selection is inert here (single-split mean 0.46 ~ honest 0.47),
so the climb is selection-driven. All rungs are persisted to results/ais_forensic.json and read
back for display (no hard-coded number).
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


def main():
    X, y, _ = he.load_xy(ROOT / "data" / "ICC.csv", "Categories", 0)
    honest = json.loads((RESULTS / "honest_eval_summary.json").read_text(encoding="utf-8"))["honest_nested_cv"]["mean"]
    perm = json.loads((RESULTS / "permutation_ICC.json").read_text(encoding="utf-8"))
    null = np.array(perm["null_scores"]); pval = perm["pvalue"]

    shop_aucs, tuned_aucs = compute_rungs(X, y)
    cherry = float(shop_aucs.max())        # split-shopping rung (single model)
    top = float(tuned_aucs.max())          # + test-set tuning rung (= Delta numerator)

    # persist every rung (reproducible + persisted; display reads from here, nothing hard-coded)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "ais_forensic.json").write_text(json.dumps({
        "honest": honest, "permutation_p": pval,
        "split_shopping_mean": float(shop_aucs.mean()), "split_shopping_max": cherry,
        "tuned_mean": float(tuned_aucs.mean()), "tuned_max": top,
        "delta": top - honest,
        "protocol": {"grid_k": KS, "grid_C": CS, "n_splits": NSPLIT, "seeds": "0..499",
                     "feature_selection": "in-fold SelectKBest(f_classif) (no leakage)",
                     "definition": "max over splits x grid (= paper's Delta numerator, pure selection optimism)"},
    }, indent=2), encoding="utf-8")

    figstyle.apply()
    from scipy.stats import gaussian_kde

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
    ridge(shop_aucs, yb["cherry"], figstyle.ORANGE)   # split-shopping distribution (single model)
    ridge(tuned_aucs, yb["top"], figstyle.RED)     # test-set-tuned per split (in-fold; real distribution)
    ax.text(0.383, 0.33, "permutation null\n(labels shuffled), $p$=%.2f" % pval, fontsize=6.1,
            color="#0a6b4e", style="italic", ha="center", va="center", zorder=12, linespacing=0.95)

    def rung(x, base, color, big):
        ax.plot([x, x], [base, base + RH], color=color, lw=2.6, zorder=10, solid_capstyle="round")
        ax.plot(x, base + RH, "o", color=color, ms=7.5, zorder=11,
                markeredgecolor="white", markeredgewidth=1.2)
        ax.annotate(big, xy=(x, base + RH), xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=11, fontweight="bold", color=color, zorder=12)

    rung(honest, yb["honest"], figstyle.GREEN, f"{honest:.2f}")
    rung(cherry, yb["cherry"], figstyle.ORANGE, f"{cherry:.2f}")
    rung(top, yb["top"], figstyle.RED, f"{top:.2f}")
    ax.text(cherry + 0.012, yb["cherry"] + RH * 0.60, "max of\n500 splits", fontsize=5.7,
            color=figstyle.ORANGE, style="italic", ha="left", va="center", linespacing=0.92, zorder=12)
    ax.text(top + 0.012, yb["top"] + RH * 0.60, "max of\n500$\\times$grid", fontsize=5.7,
            color=figstyle.RED, style="italic", ha="left", va="center", linespacing=0.92, zorder=12)

    def climb(x0, y0, x1, y1, txt, lx, ly, rad):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="0.45", lw=1.4,
                                    connectionstyle="arc3,rad=%g" % rad), zorder=8)
        ax.text(lx, ly, txt, fontsize=6.8, color="black", ha="center", va="center",
                style="italic", bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
    climb(honest, yb["honest"] + RH, cherry, yb["cherry"] + 0.02, "+ split luck", 0.560, 1.02, 0.06)
    climb(cherry, yb["cherry"] + RH, top, yb["top"] + 0.02, "+ test-set\n($k,C$) tuning", 0.585, 2.06, 0.08)

    ax.set_yticks([yb["honest"] + RH / 2, yb["cherry"] + RH / 2, yb["top"] + RH / 2])
    ax.set_yticklabels(["honest\nnested CV", "split-shopping\n(best of splits)", "+ test-set\n($k,C$) tuning"])
    for t, col in zip(ax.get_yticklabels(), [figstyle.GREEN, figstyle.ORANGE, figstyle.RED]):
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
    print(f"honest={honest:.3f} split_shopping(max 500)={cherry:.3f} "
          f"tuned(max 500xgrid, in-fold)={top:.3f}  Delta={top-honest:.3f}  "
          f"-> display {honest:.2f}/{cherry:.2f}/{top:.2f}")
    print("saved -> figures/AIS_forensic.png/.pdf + results/ais_forensic.json")


if __name__ == "__main__":
    main()
