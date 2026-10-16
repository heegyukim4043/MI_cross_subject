# Paper results and archived source mapping

Supplementary reports research methods and results, not implementation filenames.
This map supplies computational provenance separately. BAC levels in the newest
summary files are percentages and effects are percentage points. Historical
subject_results files can use BAC fractions: inspect columns before aggregating.

| Research content | Directory below analysis_outputs/ | Principal records |
|---|---|---|
| Initial matched development comparison, 224 subjects | mi_four_checks_20261007/precision_results | comparison.csv, subject_results.csv, diagnostics.csv, predictions_*.npz, lock.json |
| Figure 1, seven conditional paired intervals; S14.1 | cmpb_paired_policy_ci_20261007/results | bootstrap_summary.csv, paired_subject_contrasts.csv, aggregate_bootstrap_draws.npz, analysis_lock.json |
| Initial post-hoc Dreyer policy extension, 87 subjects | cmpb_dreyer_policy_extension_20261008/results | contrasts.csv, performance_summary.csv, subject_results.csv, bootstrap_draws.npz |
| Absolute-floor sweep; Figure 2 and S14.5-S14.6 | cmpb_absolute_floor_sweep_20261015/results | summary.csv, diagnostics.csv, diagnostic_summary.csv, subject_scores.csv, bootstrap_draws.npz, predictions_*.npz |
| Development two-Cho-version sensitivity; S14.7 | cmpb_cho_amplitude_versions_20261015/results | contrasts.csv, summary.csv, version_changes.csv, subject_scores.csv, bootstrap_draws.npz, relative_invariance.json |
| Dreyer Cho-source-version sensitivity; S14.7 | cmpb_analysis_closure_20261015/results | contrasts.csv, increments.csv, performance_summary.csv, version_changes.csv, subject_scores.csv, bootstrap_draws.npz, relative_checks.json |
| Historical no-alignment benchmark in Figure 2 | cmpb_bdea_20260924 | lodo_cho2017_lee2019_bnci2014001_physionetmi_subject_results.csv: floor=none, alignment=none |
| Alignment gain versus LOSO-LODO gap; Figure 3 | cmpb_rq_strengthening_20260928 | rq1_components.csv, rq1_shapley.csv, rq1_robustness.csv |
| Unlabeled reference budget and offline causal replay | cmpb_rq_strengthening_20260928 | tiers_summary_budget.csv, tiers_summary_causal.csv, tiers_budget_results.csv, tiers_causal_results.csv |
| CSPNet policy contrasts and AdaBN | cmpb_neural_decomposition | neural_multiseed_summary_chofix.csv, neural_multiseed_per_seed_chofix.csv |
| Pre-specified Dreyer confirmation | cmpb_confirmation_dreyer2023 | CONFIRMATORY_PLAN.md, run_manifest.json, confirmatory_hypotheses.csv, cross_subject_results.csv, loso_subject_results.csv |
| Sealed few-shot rule | cmpb_method_A_budget_fusion | DESIGN.md, FROZEN.json, dev_results.csv, sealed_results.csv, sealed_hypotheses.csv |

The wildcard * denotes multiple files, not a literal filename. The no-alignment
benchmark uses the same subjects and aggregation but is a historical descriptive
reference, not a jointly refitted control proving exact performance equality.

## Exact uncertainty records

### Figure 1 and initial intervals

All seven Figure 1 intervals reuse 10,000 dataset-stratified paired subject draws,
four dataset means equally weighted, PCG64 seed 20261007, 95% percentile intervals.
The same sampled subject indices apply to all four conditions. K is computed in
each draw, not by subtracting interval endpoints. The initial Dreyer extension
uses its own seed 20261008. Preserve these initial intervals when citing them.

### S14.5 floor-sweep simultaneous bands

In `cmpb_absolute_floor_sweep_20261015/results/summary.csv`, select
`scope=equal_dataset` and `path=fixed_post`: ten rows, zero plus 1e-14 to 1e-6.

- `increment_ci_low/high`: pointwise 95% percentile CI for Delta(f).
- `increment_simultaneous_low/high`: approximate simultaneous 95% band for Delta(f).
- `excess_ci_low/high`: pointwise CI for Delta(f)-Delta(0).
- `excess_simultaneous_low/high`: simultaneous band for Delta(f)-Delta(0).

The max-standardized-deviation family comprises the ten increments and ten
zero-reference contrasts. The identically zero zero-minus-zero column is excluded
from standardization and receives [0, 0]. Joint draws use seed 20261015 and 10,000
paired within-dataset subject resamples. Source and target receive their assigned
policies; the primary post-D floor is fixed at 1e-8. The tied-post secondary path
is reported separately and is not an independent set of subjects.

### S14.7 Cho-version simultaneous bands

Use the development contrasts.csv rows with `scope=equal_dataset` and all three
rows of the Dreyer sensitivity contrasts.csv. Each analysis has a separate family:
`K_adopted`, `K_original`, `K_original_minus_adopted`. Columns `ci_low/high` are
pointwise 95% CIs; `simultaneous_low/high` are the approximate simultaneous 95%
bands. Each uses 10,000 paired draws, seed 20261015, but development and Dreyer are
not one joint family. Do not replace pointwise CIs with bands or pool the families.

The initial and subsequent joint analyses use different draws. Their small CI
endpoint differences are not changed point estimates or contradictory results.
All intervals are conditional; no retraining/reference/new-site uncertainty is
included. Public copies of saved checks document historical validation, not new
model execution during release preparation.

## Code and provenance

Each new analysis folder includes anonymized copies of its protocol, configuration, lock,
scientific runner/analyzer/tests/verifier and selected saved results. The Dreyer
folder also includes verify_roundtrip.py, documenting its CSV round-trip reader.
Identifying path prefixes have been removed. Historical source hashes refer to
pre-anonymization records, including some omitted internal reports/logs; they must
not be presented as a new cryptographic lock of the anonymous source. Root
SHA256SUMS is the current public inventory and integrity record.
Original input manifests needed for provenance are supplied without EEG arrays.

Current figure exports and archived plot functions are in figures/current/.
Archived functions retain historical calling assumptions and are not advertised
as portable generation commands. No Word-editing scripts are bundled.

