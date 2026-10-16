# Cross-dataset MI EEG alignment: controlled policy comparisons

Anonymous review copy: identifying filesystem locations and project commit
references have been removed; figure metadata has been cleared. Numerical result
tables and NPZ arrays are unchanged. Source/provenance copies are anonymized, not
the immutable private originals. See [anonymization scope](docs/ANONYMIZATION.md).

Companion code and archived results for controlled cross-dataset motor-imagery
EEG alignment comparisons. This collection includes the follow-up analyses
completed on 15 October 2026. It is an evidence archive, not a verified portable
clean-clone, raw-EEG-to-results workflow.

## Research question and findings

We quantify how the DatasetEA->SubjectEA increment over SubjectEA differs between
two stage-wise regularization policies under matched conditions. Each policy is
applied to all applicable source and target alignment stages in both pipelines.
D->S additionally uses target-cohort EEG, so this is a pipeline-level comparison,
not an isolated causal effect of the added transformation.

The latest joint amplitude-version analyses give the following K values, where
K = (D->S minus S under absolute) minus (D->S minus S under relative):

| Evaluation | Adopted Cho version | Original Cho version |
|---|---:|---:|
| Development, four datasets equally weighted, 224 subjects | +4.43 pp [3.37, 5.51] | +2.75 pp [1.86, 3.69] |
| Dreyer post-hoc extension, 87 subjects | +2.98 pp [1.99, 4.00] | +3.45 pp [2.38, 4.59] |

Brackets are pointwise 95% conditional paired bootstrap intervals from the joint
15 October analyses. Their separate three-estimand simultaneous bands are also
provided. The initial matched-development interval [3.34, 5.52] and earlier Dreyer
interval [1.98, 4.00] remain in their original archives and Figure 1 evidence;
different Monte Carlo draws must not be silently substituted for one another.

The adopted-Cho absolute-floor sweep shows an increment from +0.18 pp at zero raw
floor to +6.19 pp at 1e-6, as S performance falls while D->S is comparatively
stable. The primary sweep varies raw-reference floors with the dimensionless
post-D subject floor fixed at 1e-8. It does not identify a universal failure
threshold or optimize an epsilon using target accuracy.

Alignment improved LODO decoding, but selective reduction of the LOSO-LODO gap
was not established. Target access, individual variation and unsuccessful frozen
external hypotheses remain part of the reported evidence.

## Navigation

- [Result and Supplementary map](docs/RESULT_FILE_MAP.md)
- [Scope and interpretation](docs/RESEARCH_SCOPE.md)
- [Contents and validation limits](docs/RELEASE_STATUS.md)
- [Cho amplitude provenance](docs/CHO_AMPLITUDE_PROVENANCE.md)
- [Current Figures 1-3](figures/current/README.md)
- [Initial precision-matched comparison](analysis_outputs/mi_four_checks_20261007/)
- [Initial development paired uncertainty](analysis_outputs/cmpb_paired_policy_ci_20261007/)
- [Initial post-hoc Dreyer extension](analysis_outputs/cmpb_dreyer_policy_extension_20261008/)
- [Absolute-floor sweep](analysis_outputs/cmpb_absolute_floor_sweep_20261015/)
- [Development Cho-version sensitivity](analysis_outputs/cmpb_cho_amplitude_versions_20261015/)
- [Dreyer Cho-source-version sensitivity](analysis_outputs/cmpb_analysis_closure_20261015/)

## Interpretation boundaries

Unit re-expression is a numerical control, not a novelty claim or discovery of
a physical calibration error. Coherent conversion scales a raw-input dimensional
floor with amplitude squared; the post-DatasetEA reference is dimensionless.
Unit consistency does not establish appropriate regularization. The Cho-only
amplitude-version experiment keeps constants fixed and is a different comparison.
The factor 0.03125 is an operational amplitude choice, not verified ADC conversion.

The original pre-specified Dreyer confirmation, later four-condition policy
extension and subsequent Cho-version sensitivity are distinct analyses. Later
extensions are post-hoc, not independent confirmatory replications. Intervals
condition on fitted models, observed trials and fixed/shared alignment references.
They omit retraining, reference-estimation and new-site uncertainty.

Offline causal replay is not closed-loop deployment. Observed individual
decreases are not established harm rates. Development equivalence criteria were
post-hoc; CSPNet CI containment is not a separate TOST. Neural legacy/relative
contrasts differ in seed coverage and covariance precision and remain exploratory.

## Reproduction and release boundaries

No raw or preprocessed EEG, manuscript files, private reviews, canonical progress
log, implementation-audit inventory or bytecode caches are bundled. Obtain input
datasets separately under their access conditions. Stored predictions include
class labels and coded identifiers; dataset terms continue to apply.

Original records are preserved privately. Distributed scripts and provenance
copies have identifying paths normalized; historical source hashes still refer
to pre-anonymization originals, not a newly locked experiment. Some runners
retain historical hash checks, write output/progress records or refuse existing result directories.
Do not run prepare/run/analyze/verify commands against the bundled archive or
rerun frozen selection just to rebuild documentation. A new execution requires
a separate directory, resolved paths and explicit handling of historical locks.

Historical labels such as corrected, chofix or unitfix are version identifiers,
not physical-calibration claims. Consult the Cho provenance note before using
the old conversion helper. The standalone mi-ea-checks package remains excluded
pending its separate review. No software license has been assigned here.

`SHA256SUMS` lists the anonymized release files, excluding itself. Historical
per-analysis manifests can also refer to pre-anonymization files or files omitted
from this selection; they are provenance records, not current-file checksum lists.

