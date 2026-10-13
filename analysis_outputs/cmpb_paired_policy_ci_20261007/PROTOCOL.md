# Paired policy-contrast uncertainty follow-up

This is an exploratory follow-up after observing the point estimates, not a prospective registration. The protocol fixes resampling before the new intervals are calculated. No EEG loading, model fitting, manuscript editing or significance-search loop.

## Inputs and estimands

Use the eight stored NPZ prediction archives from the precision-matched 32-fit comparison. Verify labels, subject arrays, shapes, class support and trial counts across policies/pipelines/scales. Independently recompute subject BAC and match subject_results.csv and comparison.csv. Files do not contain unique trial IDs; correspondence relies on the locked generator/order plus matching label and subject arrays, not independent raw-trial reconstruction.

Order four scores as absolute S, absolute D->S, relative S, relative D->S. All scores are percentages. Primary contrast: (absolute D->S - absolute S) - (relative D->S - relative S). Secondary contrast: absolute D->S - relative S. Positive primary contrast means a larger additional-stage gain under absolute regularization.

Reference multiplier1 is primary. Multiplier1e-6 is a numerical sensitivity check using the same subject resamples; it is not replication on new data. Dataset counts are fixed at52,54,9,109. The primary aggregate gives each of four dataset subject means weight1/4. The secondary pooled aggregate weights them by observed subject counts/224. Do not silently switch aggregation targets.

## Inference

Seed20261007, NumPy PCG64,10000 bootstrap repetitions. Independently resample subjects with replacement WITHIN each dataset, retaining the original dataset count. Use the same drawn subject indices for every condition and both amplitude representations. Compute the paired contrasts per subject before aggregation. Report two-sided percentile95% intervals (2.5th and97.5th percentiles; linear interpolation). No p-values, TOST, non-inferiority decisions, practical margin or multiplicity-adjusted claims. Intervals are marginal, not simultaneous. Report primary and secondary roles explicitly.

Training models, seeds, observed trial predictions, dataset identities, cohort references and harmonization remain fixed. These intervals describe conditional subject-level uncertainty; they omit retraining, reference-estimation uncertainty, trial sampling and between-new-dataset variation. Shared source models/cohort transforms induce dependence not captured by a simple subject bootstrap. Four fixed datasets and BNCI n9 limit generalization.

## Checks and optional descriptive support

Verify all1792 archived subject scores, reproduce published point estimates and reject missing pairs. Inspect per-dataset contrast distributions without excluding outliers. Test vectorized paired resampling against a literal loop on a synthetic fixture. No adaptive choice of resampling scheme from results.

From existing diagnostics only, derive tr(R)/C = relative_floor/1e-8 for raw SubjectEA references, and tabulate its across-subject distribution beside absolute epsilon and clipping counts. This is mean eigenvalue, NOT the full eigenvalue range or new evidence of physical units. Cho calibration provenance stays unresolved. Locate supplementary support for observed deterioration, prediction flips and the repository audit; do not edit these documents in this task.
