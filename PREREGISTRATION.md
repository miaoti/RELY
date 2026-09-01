# Pre-registration template for small-sample radiomics evaluation

A fill-in protocol to **freeze the evaluation before you see results**: the primary model, primary metric, feature-selection rule, optimism-gap definition, and statistics. Pre-registration is the strongest guard against the selection optimism this toolkit quantifies. It aligns with the open-science / pre-registration items of **TRIPOD+AI** (Collins et al., *BMJ* 2024;385:e078378) and the risk-of-bias domains of **PROBAST** (Moons et al., *Ann Intern Med* 2019;170:W1–W33), whose dominant controllable domain in small-sample radiomics is **Analysis**.

**How to freeze.** Fill this in and commit it *before* running confirmatory analyses; the git commit hash + timestamp is your freeze record. Use one fixed master seed throughout.

---

## 1. Primary outcome and positive class
- Outcome: `<define>`; positive class = `<define>`. State the class counts `n₊` (events) and `n₋`.
- If the label's provenance is uncertain, state it here and treat it as a limitation: it biases the honest estimate toward chance and also changes the split-to-split variability, so neither the honest estimate nor the optimism gap is immune to it.

## 2. Single primary model (confirmatory): one, fixed in advance
Leakage-free pipeline; every `fit` happens inside the inner **training** folds only:
`Impute(median) → VarianceThreshold → StandardScaler → SelectKBest(f_classif, k) → L2-LogisticRegression(class_weight="balanced")`
- Inner k-fold tuning grid: `k ∈ {…}`, `C ∈ {…}`; pre-specify the exact values.
- Imbalance: `class_weight="balanced"`; avoid resampling (e.g. SMOTE) that distorts calibration.
- Any other learner / feature count / modality subset is **secondary/exploratory**.

## 3. Primary metric
- **ROC-AUC under repeated stratified _nested_ cross-validation** (outer stratified k-fold × R repeats; inner folds tune and select).
- Within each repeat every case receives exactly one out-of-fold prediction; compute one **pooled out-of-fold (cross-fitted) AUC per repeat** and average over repeats. Take the interval by a **patient-level bootstrap** of the pooled out-of-fold predictions (resample cases, 2000 replicates; conditional on the cross-fitted predictions, so it captures sampling variation in the patients, not in the refitting). Do **not** report the 2.5–97.5 percentiles of per-fold AUC scores as a confidence interval: folds are dependent and small, and their spread is far too wide.
- Report the **Nadeau–Bengio (2003)** corrected-variance interval as a concordant second reading and use it for paired comparisons.

## 4. Secondary metrics (same nested CV)
Brier, ECE / calibration, sensitivity–specificity, and class-conditional conformal coverage with binomial (Wilson) intervals. Pre-specify these as secondary.

## 5. Feature selection
- In-fold only (`SelectKBest` inside each training fold); compute any selection-frequency/stability statistics on training folds. **Never inspect the outer test fold.**

## 6. Optimism definition (pre-specify)
Same data, same feature pool, same learner; vary **only the evaluation protocol**, and keep the two selection levels apart:
- **Within-split tuning** `Δ_tune`: the mean over splits of the best test-block setting, minus the mean over all splits and settings.
- **Split shopping** `Δ_split = max_s M_s − mean_s M_s`, where `M_s` is the best test AUC of split `s` over the tuning grid, across `S` random train/test splits of one fixed protocol (the paper uses S=50, 70/30 splits, feature selection inside each training split). Its selection budget is the number of splits, not the number of raw evaluations, and it is the quantity the RELY calculator predicts.
- The contrast `max_s M_s − honest AUC` is **secondary**: it also changes the estimator and the training fraction, so report it as a contrast, never as the modelled quantity.
- Any arm that selects features on all the data before splitting is a **leakage** arm; list it separately and label it as such.

## 7. Statistical tests
- Primary model vs a parsimonious clinical baseline: paired test on **per-fold AUC differences** (Nadeau–Bengio corrected), **not** DeLong on pooled-CV predictions.
- Label-permutation null: re-run the **entire nested estimator** (including feature selection) on each draw and report the p-value; use as many draws as the budget allows (the worked example uses 200). A cheaper stand-in that skips the refitting is not a substitute.
- Coverage / proportion estimates: binomial (Wilson) confidence intervals.

## 8. Pre-modeling evaluability (decide before modeling)
Before any modeling, run the RELY **gate** and **calculator** on the exact held-out class counts alone (`src/cohort_counts.py` defines them; `n₊` is the class the AUC treats as positive, and the Hanley–McNeil form is not symmetric in the two):
- The gate scores the cohort by the Hanley–McNeil standard error at AUC 0.5 and flags an expected split-shopping optimism of at least `δ = 0.15` AUC (in-sample ROC-AUC 0.91 on the 50-cohort benchmark; threshold chosen leave-one-source-out, sensitivity 0.81 at specificity 0.90). It does **not** predict honest performance, so a non-flag is not a clearance.
- The calculator returns the minority-event count needed to bound the expected split-shopping optimism at your tolerance `δ` (zero-intercept fit `κ₀ = 1.405` on the null working point; about 80 events for `δ ≤ 0.10` and 317 for `δ ≤ 0.05` at prevalence 0.3).
Read any single-split discrimination you later report against the optimism the counts already imply, and state the calculator's target next to your actual event count.

## 9. Multiplicity
One primary model and one primary metric. Everything else is secondary/exploratory and reported as-is: do **not** take the maximum over configurations, and do **not** select models/thresholds on the test set (except the explicitly-labeled "optimistic" demonstration arm).

## 10. Reproducibility
Fixed master seed (derive child seeds); environment pinned in `requirements.txt`; every number regenerated by `src/` into `results/` and `figures/`. Record the freeze commit here: `<git-hash>` · date `<YYYY-MM-DD>`.

---

*Scope note.* This is a methodology / study-planning template, not a clinical tool. Participant, predictor, and outcome (PROBAST) domains should be reported honestly; where data provenance is limited, state it as a limitation. The template targets the **Analysis** domain (data leakage and optimistic evaluation selection), which is the bias this toolkit measures and corrects.
