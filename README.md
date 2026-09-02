# RELY: Reliability of Evaluation for small-sample radiomics

**Sampling Variance, Not Dimensionality, Predicts Split-Shopping Optimism in Small-Sample Radiomics.**

RELY is an open toolkit and benchmark for the question every small-sample radiomics study should ask *before* training a model: **how much of the resulting AUC will be bought by choosing among evaluations?** It shows that **split-shopping optimism** (the winner's curse of reporting the luckiest of many near-equivalent evaluations of the same data) is a **sampling-variance** phenomenon, a closed-form function of the two class counts, and that we detect no additional contribution from feature dimensionality. Cutting features can reduce *leakage*-driven optimism, but in our experiments it did not buy evaluation precision; more cases in both classes did.

> Double-blind: this repository contains **no author or institution identifiers**. Prior related write-ups are cited in the third person. (De-anonymize on publication.)

> **Public release:** this is a curated **code + benchmark** release. The single-center feature table behind the AIS worked example is **patient-level data and is not distributed, in any form**. Its outputs ship as the cached `figures/AIS_forensic.*` and the aggregate `results/ais_forensic.json`, `results/permutation_ICC.json`, `results/baselines_ICC.csv` and `results/ais_toolkit.json`, and the AIS scripts are included for inspection. The load-bearing evidence, the 50-cohort benchmark, reproduces fully from public data via `radMLBench`.

## What RELY gives you
- **A reliability calculator** (`src/reliability_predictor.py`): from class counts (and a selection budget), the expected reported optimism, or the minority events needed to bound it; validated across 50 real cohorts.
- **A pre-modeling evaluability gate** (`src/screening_gate.py`): from the exact stratified held-out class counts, before any modeling, predicts a split-shopping inflation of at least 0.15 AUC. In-sample ROC-AUC **0.91**. The score needs no fitting, so what can be made out-of-sample is the *threshold*: chosen leave-one-source-out it gives sensitivity **0.81** at specificity **0.90** (Wilson 95% intervals [0.60, 0.92] and [0.74, 0.96], from 17 of 21 positives and 26 of 29 negatives). A genuine held-out-source AUC cannot be estimated reliably on this benchmark (only 2 of 39 sources contain both label classes), and the delete-one range [0.899, 0.930] is reported as an influence diagnostic, not as held-out performance.
- **A high-variance-learner robustness check** (`src/run_radmlbench_rf.py`, `src/analyze_rf_robustness.py`): re-runs the whole sweep with random forests and no univariate preselection, confirming the dimensionality-irrelevance is not specific to the regularized linear pipeline.
- **An honest-evaluation audit** (`src/honest_eval.py`): leakage-free repeated stratified nested CV, a **patient-level bootstrap confidence interval** on pooled out-of-fold predictions (fold-score percentiles are *not* a CI), Nadeau-Bengio corrected variance, and a permutation null whose test statistic is the nested estimator itself.
- **A 50-cohort benchmark harness** (`src/run_radmlbench.py`) over radMLBench.
- **A pre-registration template** (`PREREGISTRATION.md`): freeze the primary model, metric, feature selection, optimism definition and interval method *before* results; aligned with PROBAST and TRIPOD+AI.

## Key findings (benchmark numbers regenerate from public data, AIS numbers from the cached outputs; see `src/paper_numbers.py`)
- **Two selection levels, kept apart.** Within-split tuning on the test block is worth **+0.064** AUC on average; shopping the luckiest of the 50 already-tuned splits adds a further **+0.137** (`Delta_split`); together **+0.201**. The closed form models `Delta_split`, whose selection budget is **B=50 splits**, not the 750 raw evaluations.
- **The closed form predicts *measured* sampling variability.** Across **50** real radiomics cohorts, the Hanley-McNeil AUC standard error predicts each cohort's observed split-to-split SD of the per-split best test AUC with **R2=0.885**, and **R2=0.784** from the two class counts alone. Feature dimensionality gives **R2=0.092**. No across-split maximum or extreme-value assumption enters this comparison; each per-split value is already test-tuned, so the variability of the within-split tuning gain is part of the SD being predicted.
- **The selection multiplier is stable.** The ratio (max - mean) / SD across the same splits is **2.02 +/- 0.31**, *below* the **2.26** that 50 independent draws give by simulation. That is reported as an observation: no bound is derived, and dependence need not lower this standardized ratio in general.
- **Composed, they predict split-shopping optimism.** `Delta_split` is predicted by the fitted affine form `alpha + kappa * SE` (**alpha=-0.004, kappa=1.61**) with **R2=0.816**, and **R2=0.798** when fitting on all data sources but one and *predicting* the held-out source. That held-source number uses the held-out cohort's own honest AUC as the working point, so the number a planner can actually use is the counts-only **R2=0.642**. The single predictor beats a *free* three-parameter fit in {log n_pos, log n_neg, AUC} (0.733). Dimensionality: **R2=0.091**, incremental **0.003** over the counts.
- **Robust to a high-variance learner.** Random forests with no univariate preselection, at the same 50-split budget: R2=0.84 (0.70 from counts alone), dimensionality R2=0.14 (incremental 0.000); the two learners' per-cohort optimism correlates **r=0.81**.
- **Within-cohort identification.** Both class counts move the optimism in the direction and order the closed form specifies, but at **attenuated magnitude**: calibration slope **0.43** when balance is varied (observed span 0.031 against predicted 0.070) and **0.76** when total size is varied (0.018 against 0.024). Both are complete-case, using only the levels where all four cohorts are present; the size arm's n=500 level has just two of four, and pooling it in would drag that slope to 0.46. Dimensionality shows **no detectable trend** rather than a demonstrated absence: with four cohorts every interval is wide. Varying dimensionality over a decade at fixed counts shows no detectable trend under **all three** selection protocols: mean cohort-level slopes -0.000 / +0.004 / +0.007 per decade, every interval covering zero.
- **Where dimensionality *may* act.** Selecting features on all the data raises the shopped *maximum* by **+0.016 AUC per decade of d** with the number of selected features fixed and **+0.013** with it tuned, against **-0.003** with selection kept in-fold. The ordering matches a leakage account and the fixed-k slope is positive in all four cohorts, but the independent unit is the **cohort** (n=4), and on cohort-level intervals every one of these covers zero. We report it as a descriptive contrast, not an identified effect. It is what our own earlier `rd_dose_response.csv` had been measuring.
- **Reliability calculator.** Two fits, each used only where its working point exists: validation evaluates the SE at each cohort's honest AUC (`alpha=-0.004, kappa=1.61`, R2=0.816), while the calculator and the gate run before any model and can use only the null point. The calculator's fit additionally **constrains the intercept to zero** (`kappa0=1.405`, R2=0.669), because optimism must vanish as SE does; the unconstrained intercept (-0.0131) is indistinguishable from zero (95% CI [-0.045, 0.019]) and dropping it costs 0.005 R2. Bounding the expected optimism at 0.10 needs about **80** minority events at prevalence 0.3 (**317** for 0.05); 36 of the 50 cohorts fall short. Two sensitivities: the unconstrained affine calculator asks 74 and 234, and merely zeroing its intercept *without* refitting the slope is neither fit and asks 94 and 370. Transferring the working-point coefficients onto the null point, as this repo did before round-26, is also neither fit: R2=0.591, over-predicting Delta by 0.018.
- **AIS worked example, at the same 50-split budget as the benchmark.** A real single-center acute ischemic stroke cohort (n=183, 57 events): honest nested-CV AUC **0.46** [0.38, 0.53] with permutation **p=0.74** (the permuted statistic is the nested estimator itself), yet a fixed in-fold protocol whose typical split gives 0.47 reaches **0.56** by split-shopping, and **0.68** once test-set tuning is added on top of a typical 0.54, so **Delta_split=0.139** against a closed-form prediction of **0.119** (residual 0.019, inside the fit's own 0.036). Every rung keeps feature selection in-fold; moving selection to the full data is a *different* channel and is listed separately (0.45 typical, 0.59 luckiest). The gate does *not* flag it (score 0.24 vs threshold 0.28) and is right not to: 0.14 is below the 0.15 bar. The calculator is what warns, asking 80 minority events against the 57 available. Raising only the budget to 500 splits lifts the maximum to 0.72 and Delta_split to 0.176, which is kappa's protocol dependence, not a second estimate. Dropping the six conflicting duplicate records moves Delta_split to 0.101, within the predictor's residual spread. Not load-bearing evidence; the 50-cohort benchmark carries the claims.

> **Interval caveat, and why it matters.** Earlier versions of this repo reported "honest 95% CI" as the 2.5-97.5 percentiles of the per-fold AUC scores. That is a spread of correlated small-fold scores, not a confidence interval, and it is far too wide: median width **0.512** versus **0.159** for the patient-level bootstrap, spanning at least 0.9 AUC on five cohorts. It also inflated the count of cohorts "covering chance" from **14/50** to 36/50. Use `honest_ci_lo/hi`; `fold_p2.5/p97.5` are retained for diagnosis only.

## Reproduce

One command runs everything in dependency order:

```bash
pip install -r requirements.txt          # Python 3.12; radMLBench; PyRadiomics not needed
python src/reproduce.py --fresh          # re-run the benchmark experiments from scratch (hours); AIS steps use cached outputs
python src/reproduce.py --quick          # derived JSON, figures and paper numbers only (seconds)
python src/reproduce.py --list           # print the plan without running it
```

Or step by step:

```bash
python src/honest_eval.py --repeats 20    # honest vs optimistic protocols + leakage self-check
python src/run_radmlbench.py --all        # 50-cohort sweep -> results/radmlbench_sweep.csv
python src/reliability_predictor.py       # the two-step predictor + calculator (Fig 3)
python src/screening_gate.py              # the pre-modeling gate (Fig 5)
python src/r_e_selection.py               # within-cohort: class counts, on the paper's endpoint
python src/r_g_dim_channel.py             # within-cohort: dimensionality, 3 selection protocols
python src/fig_within_cohort.py           # merges the two into Fig 4
python src/fig_motivation.py              # Fig 1 (teaser)
python src/fig_method.py                  # Fig 2 (the two evaluation arms)
python src/run_radmlbench_rf.py && python src/analyze_rf_robustness.py   # RF robustness
python src/permutation_test.py --exact --radiomics-only --n-perm 200 --repeats 20   # AIS null
python src/ais_forensic.py                # AIS worked example + duplicate-record sensitivity
python src/baselines.py && python src/paper_numbers.py   # AIS baselines + every paper number
```
All randomness is seeded (master seed 20260625). The AIS patient table is not distributed, so `reproduce.py` skips the three steps that need it (`permutation_test.py`, `ais_forensic.py`, `baselines.py`) and the downstream steps read their cached outputs in `results/`.

## Paper
The manuscript is not part of this repository. Figures in `figures/` are generated by the scripts above with the shared style in `src/figstyle.py`, and `src/paper_numbers.py` prints every number the paper quotes from `results/`.

## Data & scope
- **The AIS cohort** feature table is **patient-level data and is not distributed, in any form**. Only its aggregate outputs are here (`results/ais_forensic.json`, `results/permutation_ICC.json`, `results/baselines_ICC.csv`, `results/ais_toolkit.json`, `figures/AIS_forensic.*`), and the AIS scripts are included for inspection. It is a coded derived feature table only (original imaging accession numbers replaced by surrogate case identifiers; original images/masks/extraction code and clinical records are lost); its retrospective use was approved by the contributing center's institutional review board with the consent requirement waived, and the center and review-board name are withheld during double-blind review. Known defects reported in the paper: 183 rows but 180 unique source identifiers, with three pairs carrying opposite labels (one pair identical in all 1004 features); `src/ais_forensic.py` reports a sensitivity analysis excluding them. Outcome `Categories`: 1=good, 0=poor (positive=poor). Its 1004 features were **ICC (intra-class correlation) reliability-filtered**, which is why several artifacts carry `ICC` in their names; the paper names the cohort **"the AIS cohort"** so that it is not confused with that filtering step.
- **Class counts.** `src/cohort_counts.py` is the single source of truth for `n_pos`/`n_neg`: `n_pos` is the class `roc_auc_score` treats as positive (`Target==1`), which is the *majority* class on 27 of the 50 cohorts, and the held-out counts are the exact stratified split counts rather than `round(0.3n)`. The Hanley-McNeil form is not symmetric in the two, so both details are load-bearing.
- External cohorts via **radMLBench** (`pip install radMLBench`, public, real radiomics).

## License / use
Research and educational use. Public datasets retain their own licenses. Intended use: methodology audit / study planning, **not** a clinical tool.
