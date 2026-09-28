# Method A: budget-adaptive source-target fusion (pre-specified design)

Fixed on 2026-09-28, before any method-A result was computed and before any sealed-test data were
downloaded. Changes after this date are appended in a dated "Amendments" section with their reason and
are never applied silently.

## Question
A new user supplies k labeled trials per class after label-free, relative-floor SubjectEA. How should a
decoder combine the pooled source data with these k own trials?

## Why (development evidence already in hand)
`analysis_outputs/cmpb_decomposition_20260924/label_curve_summary.csv` (CSP-LDA, within-dataset LOSO):
- natural weighting of own trials changes nothing (about +0 to +1 pp; Dreyer2023 H6 +0.01 pp);
- equal weighting hurts at small k (Cho/Lee about -5 pp at k=5) and helps at k=40 (about +1 to +3 pp);
- an own-only model needs about 40 trials per class to match EA transfer.
The best weight therefore depends on k, and no current rule sets it. This is the gap method A targets.

## Pipeline (unchanged from the decomposition)
- CSP(8) + shrinkage LDA; relative-floor SubjectEA estimated from all of the target subject's trials
  (label-free); channels shared by all datasets in the analysis; 100 Hz, 8-30 Hz, first 200 samples.
- Per target subject: a fixed class-stratified split into a calibration pool P and an evaluation set E.
  Only E is scored. k labeled trials per class are drawn from P; k in {5, 10, 20, 30}; 3 draws per k.
- Source: cross-dataset. Development uses leave-one-dataset-out over Cho2017, Lee2019, BNCI2014-001 and
  PhysionetMI (source = the other three). The sealed test uses all four as source.
- Primary endpoint: subject-level balanced accuracy on E, averaged over draws.

## Candidates (exactly three; no others are added after results are seen)
- **M1 fixed schedule (data level).** Own trials are replicated so that they carry a share w of the total
  training weight, w in {natural, 0.05, 0.1, 0.2, 0.35, 0.5}. One w per k is chosen on the development
  folds (nested: the w used for a held-out dataset is chosen on the other three folds).
- **M2 decision-level shrinkage.** Standardized LDA decision values of a source model f_src and an own
  model f_own are combined as (1 - a_k) f_src + a_k f_own with a_k = k / (k + kappa). kappa in
  {5, 10, 20, 40, 80, 160} is chosen on the development folds (nested as for M1). k=0 gives the source model.
- **M3 per-subject inner selection.** For each subject, w from the M1 grid is chosen by stratified
  leave-one-pair-out cross-validation on the k labeled trials only (ties go to the smaller w).

## Baselines
src_only (EA transfer, k=0), src_own_nat, src_own_eq (w=0.5), own_only.

## Selection rule on development data (applied once)
1. For each candidate, compute the mean BAC over the four held-out datasets (subject-weighted) and over
   k in {5, 10, 20, 30}.
2. No-harm constraint: at k=5, the mean difference from src_only must be >= -0.5 pp.
3. Among candidates meeting the constraint, pick the highest mean BAC. It must exceed the best single
   baseline (max over the four baselines of their k-averaged BAC) by >= 0.5 pp; otherwise method A is
   reported as not supported and the sealed test is still run and reported as a negative confirmation.
The selected candidate and its hyperparameters are frozen in `FROZEN.json` (with code hash) before any
sealed data are loaded.

## Sealed test
- Dataset: MOABB `Stieger2021`, session 1 only, left- vs right-hand trials, subjects S21-S62.
- S1-S20 are excluded because they were already analyzed (including classification) in another project
  (`D:\2.연구\8.MI_selfpaced`). S21-S62 have not been downloaded or inspected before this design.
- Before freezing, only label-free checks are allowed: file integrity, channel names, units, trial counts.
- Exclusion: subjects with fewer than 60 left/right trials in session 1 (count recorded before scoring).

## Sealed-test hypotheses (alpha 0.05, Holm over A1-A2)
- **A1 (labels help when used well):** selected method minus src_only, averaged over k in {10, 20, 30},
  is > 0. One-sided paired Wilcoxon over subjects.
- **A2 (better than fixed rules):** selected method minus the best single baseline chosen on development
  data, averaged over k in {5, 10, 20, 30}, is > 0. One-sided paired Wilcoxon.
- **A3 (no harm with very few labels):** at k=5, the 95% paired-bootstrap CI lower bound of selected
  method minus src_only is > -1 pp (fixed CI rule, not in Holm).
- Secondary, exploratory: the smallest k at which the marginal gain from the next k step is < 0.5 pp
  ("enough labels" point), reported per dataset.

## Reporting
All results are reported whatever their direction. A failed hypothesis is reported as a failure to
confirm on Stieger2021, not re-tested with modified settings. Development results are labeled as
development, never as confirmation.

## Amendments
- 2026-09-28 (before any method-A result): (1) M3 inner selection uses stratified 5-fold CV on the k-shot
  trials instead of leave-one-pair-out, to bound compute (about 30 k-shot pairs x 6 weights per fit otherwise).
  (2) All conditions, including the baselines, are refitted with one covariance-based CSP-LDA
  implementation (`run_dev.py`, class `Weighted`); it reproduces ManualCSP features exactly and makes
  weighting identical to trial replication (both checked on synthetic data). Channel z-scoring is omitted
  for all conditions because CSP log-variance features are invariant to per-channel scaling.
- 2026-09-28, second amendment (closes open points found in a design review; written while `run_dev.py`
  was running and before any development result was opened; no candidate, grid or hyperparameter changed):
  1. **Transductive alignment.** SubjectEA is transductive and label-free: its covariance estimate uses all
     of the target subject's trials, including the evaluation set E, but never their labels. This keeps the
     pipeline identical to the decomposition. Method A is therefore described as "label-free target
     alignment followed by few-shot supervised adaptation", not as calibration-free.
  2. **Final hyperparameters for the sealed test.** After the candidate is selected, its hyperparameters are
     re-selected once on all four development datasets with the same subject-weighted objective:
     M1 w*_k = argmax_w mean_i BAC(i,k,w) for each k; M2 kappa* = argmax_kappa mean_(i,k) BAC(i,k,kappa).
     Ties go to the more source-heavy value (smaller w, larger kappa). The same tie rule applies inside the
     nested selection. M3 ties (inner CV) go to the smaller w. Values are written to FROZEN.json before
     Stieger2021 is loaded for analysis.
  3. **M2 score scaling.** Each model's LDA decision value is divided by the standard deviation of that
     model's decision values on its own training data (source model: source trials; own model: the k-shot
     trials). No centering is applied, so each model's decision threshold stays at 0. All scaling
     parameters come from training data only and are fixed before E is scored.
  4. **Sealed-data checks.** Before freezing, only non-signal metadata checks are allowed: file integrity,
     channel names, units, event codes and per-class trial counts (needed for the exclusion rule). No
     EEG-label relationship, feature, classifier output or performance statistic is inspected. Already
     checked on S21: all 21 development channels are present; data are in volts at 100 Hz.
  5. **Randomness and pairing.** Per subject, one generator seeded with 1000 x dataset_index + subject
     (development: Cho 0, Lee 1, BNCI 2, PhysionetMI 3; Stieger2021: 4) draws the P/E split and then the
     k-shot draws in the order k = 5, 10, 20, 30, draw = 0, 1, 2. Every method and baseline uses the same
     P/E split and the same k-shot trials for a given (subject, k, draw), so all comparisons are paired.
     M3 inner folds use StratifiedKFold(5, shuffle=True, random_state=0). CSP log-variance features do not
     depend on eigenvector sign; `numpy.linalg.eigh` is used throughout.
  6. **A3 bootstrap.** Subject-level paired percentile bootstrap, 10,000 resamples, seed 2026. The three
     draws are averaged within subject first; only subjects are resampled.
  7. **Secondary metric renamed.** "Enough labels" becomes the "empirical plateau on the tested grid": the
     first tested k for which the gain to the next tested k is < 0.5 pp (exploratory, not normalized).
  8. **Thresholds.** The 0.5-pp margin (selection rule) is a minimum practically relevant improvement below
     which the extra adaptation step is not considered worthwhile; the -1-pp bound in A3 is a tolerance
     margin for harm. Neither is a statistical significance threshold.
- 2026-09-28, third amendment (after the development run finished, from trial counts only, before any
  BAC value was inspected): a k level is skipped for a subject whose pool P has fewer than k trials of a
  class (already the behaviour of `run_dev.py`). PhysionetMI (about 45 trials per subject) therefore
  contributes only k = 5 and 10. Scores averaged over k use each subject's available k levels; per-k
  selections (M1) and per-k statistics use the subjects available at that k. The same rule applies to
  Stieger2021 (A1 averages over the available k in {10, 20, 30}; A2 over the available k in {5, 10, 20, 30}).
