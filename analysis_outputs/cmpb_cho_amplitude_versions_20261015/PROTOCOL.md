# Cho amplitude-version robustness: matched four-condition comparison

Date: 2026-10-15 (America/New_York). Exploratory follow-up after the existing primary result and absolute-floor sweep. This is not prospective independent confirmation, physical calibration validation or parameter optimization.

## Question and estimands

Does the policy-dependent pipeline increment persist when the retained original Cho amplitude file replaces the adopted x0.03125 file? Compare both amplitude versions under absolute/relative policies and SubjectEA (S)/DatasetEA->SubjectEA (DS), in every one of the four LODO folds. Cho is also a source in the other three folds. All other datasets, channel/sample selection, labels, order, numerical precision and CSP-LDA definitions remain fixed.

For version v, Delta_abs(v)=BAC_abs_DS(v)-BAC_abs_S(v), Delta_rel(v)=BAC_rel_DS(v)-BAC_rel_S(v), and K(v)=Delta_abs(v)-Delta_rel(v). Report K(original)-K(adopted) directly with paired uncertainty, not by subtracting CI endpoints. Report every level/increment and dataset-specific/pooled results regardless of direction. No equivalence margin, hypothesis that K must stay positive, best-version selection or new threshold tuning.

## Input contract

Use actual retained MI_test/preprocessed_sfreq100/cho2017_moabb_uncorrected.npz for the original version, not a newly synthesized x32 array. Its expected SHA256 comes from the pre-amplitude-change archived unit-intervention manifest. Current adopted Cho and the other three datasets use the same hashes as the precision-matched primary experiment. Load the same21channels/first200samples at100Hz;224subjects. Use existing float32 input/aligned arrays, float64 reference/eigendecomposition, and unchanged archived source-only normalization, ManualCSP, feature scaler and shrinkage LDA (8CSP components).

Before fitting, require identical Cho used-array shape, channel ordering, sampling frequency, class labels and subject IDs; check the used data satisfy adopted=original*0.03125 within relative norm1e-6. Audit hashes before/after execution. This checks the analyzed data contract, not all discarded channels/time samples or acquisition provenance. Abort if trial/metadata/amplitude matching fails; do not silently substitute another file.

## Stage policy and access

Absolute policy: raw reference eigenvalue floor1e-8 and dimensionless post-D subject floor1e-8. Relative policy: floor1e-8*tr(R)/C independently at each stage. Apply the assigned policy to all source and target alignment stages. The amplitude-version intervention changes only Cho EEG; it deliberately retains numerical constants. It is NOT the coherent global unit-conversion experiment (which scales dimensional constants too). Process each dataset's references separately as in the archived primary experiment. DS additionally uses target-cohort EEG; the sweep does not isolate a transformation-only causal effect.

Eight version/policy/pipeline cells,4folds each=32actual fits, including16adopted refits as implementation anchors and16original-version fits. No cross-version prediction reuse, even if aligned arrays coincide. No target-label selection; save all eight prediction files before aggregate scoring. All datasets serve as labeled source in other folds, so complete label blindness is not claimed.

## Controls and stopping

Freeze protocol/config/runner/analyzer/tests/helpers/input definitions and anchor prediction hashes before opening new EEG. Synthetic tests must pass first. Refit the four adopted cells first, require exact predicted labels, labels and subject order against archived precision-matched scale1 predictions; record decision differences. Abort on anchor mismatch. Abort on non-SPD/nonfinite references without hidden repair.

Relative-policy alignment invariance across Cho versions is a prespecified implementation control: relative norm<=1e-5 for each dataset. Record byte equality, maximum output difference, downstream decision difference and prediction flips. Do not require bitwise equality in theory or discard nonidentical predictions to manufacture invariance. If the norm tolerance fails, retain outputs, flag inference as invalid pending investigation and do not reinterpret failure as physiology. If only labels change within output tolerance, report them and investigate sensitivity. No accuracy-based reruns or tolerance edits.

## Conditional uncertainty

Use10000 paired subject bootstrap draws within each of the four fixed dataset strata, seed20261015, same indices for all eight cells. Primary equal-dataset means, secondary pooled-subject means and per-dataset estimates. Provide95% percentile pointwise CIs for BAC levels, increments, K values and K-version difference. Additionally provide approximate95% max-standardized-bootstrap simultaneous bands for the three primary equal-dataset estimands K(adopted), K(original), K(original)-K(adopted). No new p-values or formal equivalence assessments.

Intervals condition on fixed fitted models, observed trials and fixed/shared alignment references; they do not include retraining, reference-estimation or new-site uncertainty. Repeated conditions are not additional independent subjects. Label both the follow-up and its uncertainty exploratory. Do not overwrite the manuscript's original CI with the new Monte Carlo realization.

## Deliverables and limits

Save lock, synthetic test log, input audit, all predictions/subject BACs, stage-specific clipping diagnostics, paired draws/indices, summary/contrasts, independent saved-output checks and a Korean interpretation report. No manuscript, Zotero, FROZEN.json, old outputs, GitHub or physical-unit claims changed. Neural matching, independent reference-cohort controls and fixed-c extensions are out of scope.
