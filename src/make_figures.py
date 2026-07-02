"""
Figure generation: turn existing results into paper-ready figures (English labels for the
double-blind IEEE submission; no author/institution info).
  F2  ICC label-permutation null distribution + observed AUC (from results/permutation_ICC.json)
  C2  Marginal vs class-conditional minority coverage vs nominal level 1-alpha (ICC, Granata2024)
  F3  ICC risk-coverage (selective classification: accuracy of top-confidence retained fraction)
Outputs to figures/. Runs in the py3.12 main env; no PyRadiomics needed.
Usage: python src/make_figures.py
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGS = ROOT / "results", ROOT / "figures"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import honest_eval as he
import conformal_gate as cg


def load_named(name):
    if name == "ICC":
        X, y, _ = he.load_xy(ROOT / "data" / "ICC.csv", "Categories", 0)
        return X, y
    import radMLBench as rb
    df = rb.loadData(name)
    y = df["Target"].astype(int).values
    X = df.drop(columns=[c for c in ("Target", "ID") if c in df.columns]).select_dtypes("number").values.astype(float)
    return X, y


def fig_f2():
    d = json.loads((RESULTS / "permutation_ICC.json").read_text(encoding="utf-8"))
    null = np.array(d["null_scores"]); obs = d["observed_auc"]; p = d["pvalue"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(null, bins=30, color="#9ecae1", edgecolor="white", label=f"Permutation null (N={d['n_permutations']})")
    ax.axvline(0.5, color="grey", ls=":", lw=1, label="chance = 0.5")
    ax.axvline(obs, color="#d62728", lw=2, label=f"Observed AUC = {obs:.3f}")
    ax.set_xlabel("ROC-AUC"); ax.set_ylabel("Count")
    ax.set_title(f"F2 - ICC label-permutation test  (p = {p:.3f})")
    ax.legend(fontsize=8)
    fig.tight_layout(); out = FIGS / "F2_permutation_ICC.png"; fig.savefig(out, dpi=200); fig.savefig(out.with_suffix(".pdf")); plt.close(fig)
    return out


def fig_c2(name):
    X, y = load_named(name)
    alphas = [round(a, 2) for a in np.arange(0.05, 0.41, 0.05)]
    minority, acc, _ = cg.run(X, y, repeats=300, seed=he.MASTER_SEED, alphas=alphas)
    targets = [1 - a for a in alphas]
    def cov(meth):
        return [acc[a][meth][minority][0] / acc[a][meth][minority][1] for a in alphas]
    marg, mond = cov("marg"), cov("mond")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(targets, targets, color="grey", ls="--", lw=1, label="nominal (ideal)")
    ax.plot(targets, marg, "o-", color="#d62728", label="Marginal conformal - minority")
    ax.plot(targets, mond, "s-", color="#2ca02c", label="Class-conditional (Mondrian) - minority")
    ax.set_xlabel("Nominal coverage 1-alpha"); ax.set_ylabel("Empirical minority-class coverage")
    ax.set_title(f"C2 - Minority coverage: marginal vs class-conditional ({name})", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); out = FIGS / f"C2_coverage_{name}.png"; fig.savefig(out, dpi=200); fig.savefig(out.with_suffix(".pdf")); plt.close(fig)
    return out


def fig_f3():
    X, y = load_named("ICC")
    pipe = he.make_pipeline(); pipe.set_params(select__k=20, clf__C=0.1)
    n = len(y); psum = np.zeros(n); pcnt = np.zeros(n)
    for tr, te in RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=he.MASTER_SEED).split(X, y):
        pipe.fit(X[tr], y[tr]); psum[te] += pipe.predict_proba(X[te])[:, 1]; pcnt[te] += 1
    prob = psum / pcnt
    conf = np.abs(prob - 0.5)
    order = np.argsort(-conf)
    correct = ((prob > 0.5).astype(int) == y).astype(int)[order]
    cov = np.arange(1, n + 1) / n
    acc_at = np.cumsum(correct) / np.arange(1, n + 1)
    base = max(y.mean(), 1 - y.mean())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(cov, acc_at, color="#1f77b4", label="Accuracy of top-confidence retained fraction")
    ax.axhline(base, color="grey", ls="--", lw=1, label=f"Majority-class baseline = {base:.3f}")
    ax.set_xlabel("Coverage (retained fraction)"); ax.set_ylabel("Accuracy")
    ax.set_title("F3 - ICC risk-coverage (selective classification)")
    ax.set_ylim(0.4, 1.0); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); out = FIGS / "F3_risk_coverage_ICC.png"; fig.savefig(out, dpi=200); fig.savefig(out.with_suffix(".pdf")); plt.close(fig)
    return out


def fig_f1():
    """F1 头条：同一无信号数据，AUC 随评测协议从 honest(≈随机) 膨胀到 cherry-pick。"""
    d = json.loads((RESULTS / "honest_eval_summary.json").read_text(encoding="utf-8"))
    honest = d["honest_nested_cv"]["mean"]; naive = d["naive_single_split"]["mean"]
    cherry = d["naive_single_split"]["max"]; tsel = d["test_set_selected_single_split"]["mean"]
    labels = ["Honest\nnested-CV", "Naive single\nsplit (mean)", "Test-set\nmodel selection", "Cherry-picked\nsplit (best/300)"]
    vals = [honest, naive, tsel, cherry]
    colors = ["#2ca02c", "#9ecae1", "#fdae6b", "#d62728"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, vals, color=colors, edgecolor="white")
    ax.axhline(0.5, color="grey", ls=":", lw=1, label="chance = 0.5")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.2f}", ha="center", fontsize=9)
    ax.annotate("", xy=(3, cherry), xytext=(0, honest),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
    ax.text(1.3, (honest + cherry) / 2 + 0.04, f"optimism gap +{cherry-honest:.2f}", fontsize=9)
    ax.set_ylabel("ROC-AUC"); ax.set_ylim(0.3, 0.85)
    ax.set_title("F1 - ICC: same data, AUC inflates with evaluation protocol (honest ~ chance)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); out = FIGS / "F1_optimism_ICC.png"; fig.savefig(out, dpi=200); fig.savefig(out.with_suffix(".pdf")); plt.close(fig)
    print(f"[F1] honest {honest:.3f} vs cherry-pick {cherry:.3f}  gap +{cherry-honest:.3f}")
    return out


def fig_c1():
    """C1 主图：跨真实队列的乐观差距 vs EPV（含 ICC 标注）+ 诚实 AUC vs EPV。"""
    import pandas as pd
    df = pd.read_csv(RESULTS / "radmlbench_sweep.csv")
    icc = json.loads((RESULTS / "honest_eval_summary.json").read_text(encoding="utf-8"))
    icc_epv, icc_honest = 57 / 1004, icc["honest_nested_cv"]["mean"]
    icc_gap = icc["gap_testselected_minus_honest"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.scatter(df["epv"], df["optimism_gap"], c="#1f77b4", s=28, label=f"radMLBench (N={len(df)})")
    ax1.scatter([icc_epv], [icc_gap], c="#d62728", s=120, marker="*", zorder=5, label="ICC (our cohort)")
    ax1.axhline(0, color="grey", ls=":", lw=1)
    ax1.set_xscale("log"); ax1.set_xlabel("EPV (events per feature, log)")
    ax1.set_ylabel("Optimism gap (test-selected - honest)")
    ax1.set_title("C1 - optimism gap vs EPV across real radiomics cohorts")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax2.scatter(df["epv"], df["honest_auc"], c="#1f77b4", s=28)
    ax2.scatter([icc_epv], [icc_honest], c="#d62728", s=120, marker="*", zorder=5, label="ICC")
    ax2.axhline(0.5, color="grey", ls=":", lw=1, label="chance")
    ax2.set_xscale("log"); ax2.set_xlabel("EPV (log)"); ax2.set_ylabel("Honest nested-CV AUC")
    ax2.set_title("Honest AUC vs EPV"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig.tight_layout(); out = FIGS / "C1_gap_vs_epv.png"; fig.savefig(out, dpi=200); fig.savefig(out.with_suffix(".pdf")); plt.close(fig)
    pos = int((df["optimism_gap"] > 0).sum())
    print(f"[C1] N={len(df)}  mean gap={df['optimism_gap'].mean():+.3f}  gap>0: {pos}/{len(df)}  "
          f"median honest AUC={df['honest_auc'].median():.3f}  "
          f"corr(gap, log-EPV)={np.corrcoef(np.log(df['epv']), df['optimism_gap'])[0,1]:+.3f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["all", "core", "c1", "f1"], default="all",
                    help="all=全部; core=F2/C2/F3; c1=仅 C1; f1=仅 F1")
    args = ap.parse_args()
    FIGS.mkdir(exist_ok=True)
    made = []
    if args.only in ("all", "core"):
        made += [fig_f2(), fig_c2("ICC"), fig_c2("Granata2024"), fig_f3()]
    if args.only in ("all", "c1"):
        made.append(fig_c1())
    if args.only in ("all", "f1"):
        made.append(fig_f1())
    print("saved figures:")
    for p in made:
        print("  ", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
