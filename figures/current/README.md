# Current result figures

These PDF/PNG files retain the current manuscript figures without redrawing;
metadata was cleared while pixels/rendered page contents were checked unchanged.
The source is the manuscript
figure folder. Numbering follows the latest paper, not earlier repository drafts.

| Files | Content | Source |
|---|---|---|
| Figure_1_bar.pdf / .png | Four policy/pipeline BACs and incremental gains/paired K | cmpb_paired_policy_ci_20261007/results/bootstrap_summary.csv and aggregate_bootstrap_draws.npz |
| Figure_2_bar.pdf / .png | Absolute-floor sweep with bar/error/grid format | cmpb_absolute_floor_sweep_20261015/results/summary.csv, equal_dataset/fixed_post |
| Figure_3_bar.pdf / .png | Decoding gain versus LOSO-LODO gap interaction | cmpb_rq_strengthening_20260928 and source_archives/fig3_rq1_v3_source.csv |

Paths in the Source column are relative to analysis_outputs/ unless otherwise
specified. In the old uploaded tree the current Figure 3 was named Figure 2.
Figure 2 now denotes the new floor sweep; the old filename must be replaced.

Figure 1 shows initial adopted-Cho results, not every later amplitude-version
result. Its seven error bars are 95% paired subject-bootstrap intervals with
equal dataset weighting (10,000 draws, seed 20261007). K is calculated within
each draw, not by subtracting interval endpoints.

Figure 2 error bars are pointwise 95% intervals. Simultaneous bounds are separate
columns in the saved summary.csv and described in docs/RESULT_FILE_MAP.md.
The historical no-alignment line is a descriptive cross-analysis reference, not
exact identity or a new jointly fitted control. All result estimates/intervals
are conditional on fitted models and fixed/shared alignment references.

The archived plotting functions preserve the bar/error/grid styling. They retain
historical calling assumptions and are evidence, not portable turnkey commands.
The floor-sweep plotting module contains only the relevant drawing functions,
not the Word/table-editing program that originally invoked them.

