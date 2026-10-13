# Dreyer external policy contrast: post-hoc extension

Date: 2026-10-08. User approved execution after reviewing development results and
the earlier Dreyer H1-H6 report. This is NOT a new preregistered or sealed
confirmation. Specification is fixed before the NEW four-condition predictions.

## Why four conditions

The original Dreyer manifest records Cho SHA256 f820119a...; the matched
development policy experiment records corrected Cho SHA256 9982fb28... . Existing
relative Dreyer scores cannot simply be joined to two new absolute fits. Rerun
all four conditions using the current development inputs and the original
Dreyer input; preserve all historical files. Use the original Dreyer common
18-channel montage, not the development-only 21-channel montage.

## Fixed design

- Source: Cho2017, Lee2019, BNCI2014-001, PhysionetMI, all 224 subjects.
- Target: Dreyer2023, all 87 subjects; same retained trials in all conditions.
- Inputs: hashes must match the development precision lock for four source
  files and the original Dreyer manifest for the target. Cho adjustment is
  already in the hashed NPZ; do not apply another multiplier. Its physical-unit
  provenance remains unresolved.
- Existing preprocessing: 8-30 Hz, 100 Hz; first 200 stored samples, following
  the original confirmation's cue-relative epoch convention. No new filtering.
- Montage/order: original Dreyer run_manifest.json shared_channels (18).
- Conditions: coherent_absolute/relative x subject/dataset_subject, scale=1.
  No external rescaling arm is added. At scale=1, raw and post-D absolute floors
  are both 1e-8. Relative floor is 1e-8 times trace(R)/C at each stage.
- EA operator: reuse precision_compare.align unchanged. Covariance is the mean
  of XX^T without division by T, accumulated in float64; stage output float32.
  Non-SPD references fail explicitly. Log all source/target clipping diagnostics.
- Each source dataset has its own DatasetEA reference. Target DatasetEA uses
  the complete unlabeled Dreyer cohort; SubjectEA uses that subject's complete
  unlabeled EEG. This is batch-transductive, not causal; S and D->S do NOT have
  equal target access. Access is held fixed across policies within each pipeline.
- One CSP-LDA fit per condition: concatenate all four source datasets, compute
  channel z-score on source only, ManualCSP(8), feature StandardScaler,
  LDA(lsqr, shrinkage=auto). Reuse original helpers with code hashes. CSP feature
  arithmetic retains the archived implementation, not an all-float64 rewrite.
- Target labels are not read during audit/alignment/fitting. Save all four
  complete target predictions first, then load labels solely for scoring.
- Fixed subject membership, trial order, montage and hyperparameters. No tuning,
  selective exclusions, early stopping or condition additions based on outcomes.

## Estimands and inference

Subject BAC in percentage points; each Dreyer subject has equal weight. Primary:

    delta = (BAC_abs_DS - BAC_abs_S) - (BAC_rel_DS - BAC_rel_S)

Expected direction, informed by development: delta > 0. A positive two-sided
95% percentile interval supports the directional pattern in this external
dataset under this protocol; a crossing interval is inconclusive, not equivalence.

Secondary descriptive contrasts: abs_DS-abs_S, rel_DS-rel_S,
abs_DS-rel_S, and rel_S-abs_S. Report condition means and subject SDs.
Do not claim equivalence from a narrow zero-containing interval. No new
equivalence margin, p-value or multiple-comparison significance family.

Bootstrap: 10,000 draws, NumPy default_rng(20261008), resample the 87 subject
rows with replacement, retaining all four condition scores together; same draw
matrix for every contrast. Report 2.5/97.5 percentiles. Secondary intervals are
pointwise exploratory summaries. Inference is conditional on the fixed source
models and batch target references; it does not resample source datasets,
retrain models or re-estimate cohort alignment in each draw. Subject outcomes
share fitted models/cohort references. Do not extrapolate these intervals to
variation across new acquisition sites or training pools.

## Execution and audit

Hash protocol, runner, original operator/helpers, input manifests and all NPZs
before fitting. Check all arrays finite, expected channels/sampling/dimensions,
87 target/224 source subjects, unique source-qualified IDs, and reference SPD.
Preserve positional target trial index tied to the hashed NPZ (not a new claim
of native acquisition trial IDs). Run synthetic operator and bootstrap tests
before opening EEG. Save predictions/decisions, normalizer/model parameters,
per-subject BAC, diagnostics, contrasts, bootstrap draws and status. Fail closed
on mismatches; failed runs remain documented. No overwrite of prior analyses.

## Reporting boundary

This is a new held-out-dataset policy comparison, not replication of all
development conditions: montage=18 and source pool=4 development datasets.
It is not independent research-team replication, new regularization
dose-response evidence, proof of optimal whitening, or evidence of prevalence
in the literature. Original Dreyer H1 (+5.478 pp LOSO relative-minus-legacy)
is separate historical evidence, not the new policy-by-stage interaction.
