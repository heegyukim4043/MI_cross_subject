# Cross-dataset MI EEG alignment: controlled policy comparisons

Companion code and archived results for controlled cross-dataset motor-imagery
EEG alignment comparisons. This collection documents the reported analyses;
a portable clean-clone, raw-EEG-to-results workflow has not been verified.

The primary comparison crosses coherent absolute and relative regularization
policies with SubjectEA (S) and DatasetEA followed by SubjectEA (D->S), with
matched covariance precision, data and splits. The equal-dataset difference in
increments is +4.43 pp (95% conditional paired bootstrap CI 3.34 to 5.52).
The Dreyer post-hoc external extension yields +2.98 pp [1.98, 4.00].
This is not a new alignment algorithm, an isolated causal effect of DatasetEA,
a validated reporting standard, or evidence of universal regularizer optimality.

## Research and result navigation

- `docs/RESULT_FILE_MAP.md`: paper findings, corresponding records and file roles.
- `docs/RELEASE_STATUS.md`: validation boundary and outstanding release tasks.
- `docs/CHO_AMPLITUDE_PROVENANCE.md`: adopted amplitude version, not verified calibration.
- `docs/RESEARCH_SCOPE.md`: interpretation and limitations.
- `analysis_outputs/mi_four_checks_20261007/`: matched development operators/results.
- `analysis_outputs/cmpb_paired_policy_ci_20261007/`: paired uncertainty protocol/results.
- `analysis_outputs/cmpb_dreyer_policy_extension_20261008/`: post-hoc external extension.
- `figures/current/`: separate current result figures; not historical draft Figs. 1-6.

## Important conditions

Exact whitening cancels a positive global scalar. A fixed dimensional covariance
floor is coherent if the raw-input floor is converted with amplitude squared;
after DatasetEA the recomputed reference is dimensionless, so its constant is not
blindly rescaled. Unit consistency does not establish adequate regularization.
The code exposes multiple policies; inspect the explicit setting rather than
assuming an environment variable overrides a function argument.

Development uses the adopted Cho amplitude factor 0.03125. This is not a proven
ADC or physical-unit correction. Historical filenames and option labels are
retained for traceability and must not be treated as scientific claims.
The original pre-specified Dreyer analysis and the later post-hoc four-condition
extension use distinct input versions and must not be merged into one confirmation.

D->S additionally uses target-cohort EEG; S uses the target subject alone.
Subject/bootstrap intervals condition on fitted models and fixed references.
Offline causal replay is not closed-loop deployment. Observed individual decreases
are not established individual harm rates. Development equivalence criteria were
post-hoc; CSPNet CI containment is not a separately computed TOST.

## Reproduction boundary

No raw/preprocessed EEG is included. Recorded inputs must be independently
obtained under their dataset terms. Stored predictions contain class labels and coded identifiers from the public
research datasets; they do not contain raw EEG. Dataset terms still apply.
The standalone mi-ea-checks package is excluded pending its separate code review.
Do not rerun frozen selection to rebuild documentation.

Archived runners are preserved as evidence, not advertised as turnkey commands.
Some contain local paths and write outputs or progress records. Make a fresh run
directory and resolve portability before execution. The original inference script
deliberately refuses to create an already existing output directory. Do not run it
against the bundled historical results. No license has been chosen here.
