# Absolute-floor provenance and sensitivity: locked exploratory follow-up

Client date: 2026-10-15 (America/New_York). Covers coauthor item1 only. This is a planned sensitivity analysis on already examined development data, not preregistered independent confirmation. Cho amplitude-version robustness, independent cohort controls and neural matching are separate unexecuted items.

## Question and intervention

Does the observed DatasetEA->SubjectEA increment over SubjectEA vary across a fixed absolute-eigenvalue-floor grid, and does a large increment coincide with a weak single-stage comparator? Do not interpret a sweep as identifying an isolated transformation effect or an optimal clinical parameter.

Use the archived matched CSP-LDA implementation and identical four development datasets/224 subjects/21 channels/100Hz/200samples. Process every source dataset separately and apply the same policy to source and target; retrain the source classifier for each distinct aligned input. S uses own unlabeled EEG, D->S also uses its target cohort, unchanged across epsilon within a pipeline. Covariances are mean XX-transpose without T normalization. All other preprocessing, CSP/LDA constants, feature normalization and data order remain fixed.

Primary path: absolute raw-input epsilon in {0, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6} V^2 under the declared input contract. Apply to S's subject reference and D->S's dataset reference. Hold the dimensionless post-D subject floor at1e-8. The zero raw floor is a reference, not automatically proof that every stage is unclipped; stage diagnostics determine that description. Relative coefficient1e-8 is a separate archived anchor, not an optimized sweep setting.

Secondary path: retain the same raw grid and let the dimensionless post-D floor take the same numerical value as the raw constant. This is a clearly labeled two-stage policy path, NOT an equality of physical quantities. S is shared with the primary path. At1e-8 the two paths coincide. Their difference reveals whether the post-D policy changes the curve. Dimensionless stages do not receive squared unit-conversion factors.

The finite grid is fixed before new EEG/model outputs are read. It covers zero/weak/intermediate/the historical1e-8/stronger settings, without choosing levels by target accuracy. Do not refine it after observing scores. Up to31 unique pipeline specifications/124 LODO fits. Exact array-byte-identical conditions may reuse fitted predictions with recorded lineage; no tolerance-based reuse.

## Safeguards and controls

1. Freeze this protocol, config, runner, tests and imported helper hashes before EEG loading. Check the four archived input hashes and source helper hashes.
2. Synthetic checks first, without EEG. Require output equality with the archived operator at fixed settings, coherent unit handling, non-SPD failure, exact-whitening rotation and stage-policy anchor equality.
3. Run historical absolute1e-8 and relative controls first. Require exact prediction reproduction against archived scale1 four-condition predictions; abort on mismatch. This is a known-result implementation control, not new validation evidence.
4. Use only source labels to fit each fold; no target labels choose epsilon or change the protocol. Save all condition predictions before aggregate scoring. Source labels are necessarily available because each dataset is a source in other folds; do not claim complete label blindness.
5. Preserve archived scripts/outputs, all frozen plans and manuscripts. Do not execute old generators or modify FROZEN.json. Store results in a new research analysis directory, not a manuscript version.
6. Abort on nonfinite/nonpositive reference eigenvalues; do not silently add another floor or change rank. Record failures if encountered rather than discarding them from a favorable curve.

## Outcomes and uncertainty fixed in advance

For each level, report both pipeline BACs and Delta(epsilon)=BAC(DS)-BAC(S), dataset-equally weighted primary and subject-pooled secondary, plus dataset-specific values. Also report Delta(epsilon)-Delta(0) along the primary path. This compares increments against the zero-raw-floor reference; it is not a uniquely identified recovery mechanism.

Record dataset-stage and final subject-stage eigenvalue min/max/mean, epsilon/mean eigenvalue, clipping counts/fractions and whitening residual norm. Record these separately for all source/target dataset references. No monotonicity assumption, clipping-based failure threshold or accuracy-prediction claim.

Use10000 paired subject bootstrap draws within each fixed dataset, same draw indices for every condition, seed20261015. Provide95% pointwise percentile intervals. For the primary equal-dataset Delta curve AND its Delta-minus-zero-reference curve jointly, add an approximate95% simultaneous max-standardized-bootstrap-deviation band. Constant contrasts have zero-width intervals and are excluded from standardization. Supporting/tied-policy/dataset-specific intervals remain pointwise. Do not select a best epsilon, test a practical equivalence margin or claim all zero-containing intervals are null effects.

All intervals are exploratory and conditional on fitted models, observed trials and fixed/shared references; no retraining/reference-estimation/new-dataset uncertainty. Epsilon values are repeated conditions, not independent subjects. Target information differs between S and DS, so neither bands nor sweeps isolate a pure transformation effect. Sensitivity to the chosen Cho version is NOT answered here.

## Provenance question

Check the earliest available tracked implementation and exact code operation. An implementation's citation to the EA paper does not establish that its1e-8 floor was recommended by that paper. Separate first available upload, numerical-stability comment, true mathematical operation (clamping vs adding ridge) and absent tuning justification. Do not invent a first-author/original-choice date, widespread prevalence, or reasonable-default justification.

## Reporting

Create provenance evidence, test records, frozen hashes, all scores/predictions, diagnostic curves, a conditional uncertainty summary and a Korean interpretation report. Show unfavorable/nonmonotonic outcomes too. A narrow two-policy finding may remain valid even if no general boundary is established. Raw EEG is read locally only; no remote upload or manuscript/citation edit is authorized by this task.
