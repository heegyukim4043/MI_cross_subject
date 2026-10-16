# Analysis closure: Dreyer robustness to the retained Cho source amplitude version

2026-10-15 America/New_York. User requested completion of analysis. This last targeted follow-up closes the agreed analysis scope; it does not authorize manuscript changes, a neural rerun, source-target mixed-policy study, same-information controls, fixed-c sweeps or target-tuned searches.

## Question and intervention

Does the existing post-hoc Dreyer policy contrast depend on the Cho source amplitude version? Refit adopted and original Cho source versions under absolute/relative x SubjectEA(S)/DatasetEA->SubjectEA(DS):8fits. Adopted4fits reproduce existing predicted labels before the original4fits. Original Cho uses the actual retained array, not a reconstructed x32 array. The only changed dataset is source Cho; target Dreyer data remain fixed.

Use the exact archived post-hoc18channel order, first200samples/100Hz,224source subjects from4development datasets and87Dreyer target subjects/20792trials. Preserve the archived source-only channel normalization, ManualCSP(8components), feature standardization, shrinkage LDA, float64 EA covariance/eigendecomposition and float32 stage outputs. All source datasets align separately and target alignment uses the same access as before: S own unlabeled batch,DS additionally target cohort. Source and target policy are matched, not crossed independently.

Absolute raw and dimensionless post-D floors remain1e-8; relative is1e-8tr(R)/C separately at every stage. Both versions keep these numerical policies. This is a source amplitude-version sensitivity, not coherent global unit conversion or physical calibration validation.

## Controls, freeze and labels

Before EEG array loading, pass synthetic tests, match expected source/target hashes to archived manifests, and freeze config/protocol/runner/analyzer/verifier/test/borrowed-code/anchor hashes. Verify the analyzed18channel Cho arrays have identical shape, channels, labels and subject order, and adopted=original*0.03125 with relative norm<=1e-6. Fail without changing the design if not. Read target X/subjects/channel/sampling metadata only; do not read target y until all8prediction files are saved and their hashes frozen.

Adopted predicted classes must match all4archived prediction arrays exactly, along with subject/trial row order. No target ground truth is needed for this check. Record decision differences too. Abort on anchor mismatch or non-SPD references; no silent repair. Before original-version fitting, the adopted4predictions exist. Relative alignment-output invariance across Cho versions is an implementation control with norm tolerance1e-5; record byte equality, flips and decision difference separately. Any failed relative-output control flags inference rather than changing tolerances or choosing favorable cases. No prediction reuse: all8fits executed.

## Estimands and uncertainty

Per version v, Delta_abs(v)=BAC_abs_DS-BAC_abs_S, Delta_rel(v)=BAC_rel_DS-BAC_rel_S;K(v)=Delta_abs-Delta_rel. Report bothKvalues and K(original)-K(adopted), irrespective of direction. Target subject-average BAC, all87paired subjects. Also report all8pipeline levels,4increments,4within-pipeline version changes, individual deltas descriptively and stage clipping diagnostics. No clinical margin, equivalence/null-absence claim or parameter/version selection.

10000paired subject-bootstrap resamples,seed20261015, same87subject indices applied to every8condition score.95%percentile pointwise CIs; approximate95%max-standardized-bootstrap simultaneous bands over the3K/version primary estimands. Derive K changes within each draw, never by subtracting CI endpoints. This new joint MonteCarlo realization does not overwrite the original archived seed20261008 CI. Save indices and joint draws for independent saved-output checking.

The protocol is fixed before this run, after prior development/Dreyer results and Cho sensitivity were known. This is exploratory robustness, not preregistration or new independent implementation replication. Intervals condition on source fits, observed trials and shared/fixed alignment references; no source resampling, retraining/reference-estimation or new-site uncertainty. Source-only version intervention can establish source-input sensitivity but does not isolate clipping, scalar scale, feature normalization or pure added-transformation effects.

## Closure outputs

Preserve prior inputs/results/manuscripts/FROZEN/GitHub. Save input/anchor/relative checks,8predictions, subject scores, joint bootstrap+indices, summaries, independent score/draw/CI checks and immutable input/code hashes. Finish with a Korean analysis synthesis linking the completed floor sweep, development Cho-version comparison, historical no-alignment anchor and this external sensitivity. Explicitly retain unresolved physical provenance and unmatched-neural/cohort-information scope limits. Close current analysis work after validation; do not pursue a favorable outcome by adding experiments.
