# Paper results and archived source mapping

Supplementary contains research methods and results, not implementation filenames.
This map preserves the computational provenance separately from the manuscript.

| Research content | Source directory | Principal records |
|---|---|---|
| Development two-policy comparison, 224 subjects, 32 fits | analysis_outputs/mi_four_checks_20261007/precision_results | comparison.csv, subject_results.csv, diagnostics.csv, predictions_*.npz, lock.json |
| Figure 1 and conditional paired CIs | analysis_outputs/cmpb_paired_policy_ci_20261007/results | bootstrap_summary.csv, paired_subject_contrasts.csv, aggregate_bootstrap_draws.npz, analysis_lock.json |
| Dreyer post-hoc policy extension, 87 subjects | analysis_outputs/cmpb_dreyer_policy_extension_20261008/results | contrasts.csv, performance_summary.csv, subject_results.csv, subject_contrasts.csv, bootstrap_draws.npz, predictions_*.npz |
| Alignment gain versus LOSO-LODO gap | analysis_outputs/cmpb_rq_strengthening_20260928 | rq1_components.csv, rq1_shapley.csv, rq1_robustness.csv |
| Unlabeled reference budget and offline causal replay | analysis_outputs/cmpb_rq_strengthening_20260928 | tiers_summary_budget.csv, tiers_summary_causal.csv, tiers_budget_results.csv, tiers_causal_results.csv |
| CSPNet policy contrasts and AdaBN | analysis_outputs/cmpb_neural_decomposition | neural_multiseed_summary_chofix.csv, neural_multiseed_per_seed_chofix.csv |
| Original pre-specified external confirmation | analysis_outputs/cmpb_confirmation_dreyer2023 | CONFIRMATORY_PLAN.md, run_manifest.json, confirmatory_hypotheses.csv, cross_subject_results.csv, loso_subject_results.csv |
| Sealed few-shot rule | analysis_outputs/cmpb_method_A_budget_fusion | DESIGN.md, FROZEN.json, dev_results.csv, sealed_results.csv, sealed_hypotheses.csv |

The wildcard * denotes multiple saved files, not a literal filename.
CSV scores in the historical subject_results.csv are BAC fractions, while
bootstrap summary estimates and contrasts use percentages/percentage points.
Always inspect column units before recomputing a contrast.

All seven Figure 1 intervals reuse 10,000 dataset-stratified paired subject draws,
four dataset means equally weighted, PCG64 seed 20261007, 95% percentile intervals.
The same sampled subject indices apply to all four conditions; K is computed in
each draw, not by subtracting interval endpoints. These are marginal conditional
intervals without retraining/reference/trial/new-site uncertainty.

Saved verification records demonstrate historical scoped checks, not a new
execution during this export. Source paths/hashes in historical locks are retained
unchanged. Independent portability and complete table coverage remain pending.

The archived figure sources are included as evidence but retain manuscript-folder
assumptions; they are not portable generators in their exported locations.
