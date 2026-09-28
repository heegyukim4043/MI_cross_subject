# Pre-specified amplitude-unit intervention

Date frozen: 2026-09-25, before running the full analysis.

## Question

Does global amplitude recoding alter 4-dataset LODO conclusions under the
legacy absolute EA floor, while a scale-relative floor preserves outputs and
method rankings?

Amplitude is an intervention on numerical representation, not a component of
the biological/acquisition dataset shift. Every dataset is multiplied by the
same factor within each analysis cell; samples, labels and splits remain fixed.

## Fixed design

- Datasets: Cho2017, Lee2019, BNCI2014-001, PhysionetMI.
- Protocol: leave one dataset out; train on the other three and report
  subject-level macro balanced accuracy for the held-out dataset.
- Input contract: 21 shared channels, first 200 samples, existing 8-30 Hz and
  100 Hz files.
- Classifiers: CSP(8)+shrinkage LDA and OAS covariance+TS-LR.
- EA floor: legacy absolute `1e-8` versus relative
  `1e-8 * trace(R) / C`.
- Alignment: SubjectEA and DatasetEA-to-SubjectEA.
- Global amplitude multipliers: `1`, `1e-3`, `1e-6`.
- Target access: batch/transductive unlabeled target EEG for each subject,
  identical across conditions.
- CPU threads: 4, so the concurrent local neural GPU queue remains primary.

Full factorial: 2 classifiers x 2 floors x 2 alignments x 3 amplitude scales x
4 held-out datasets. No target labels are used for alignment or model fitting.

## Frozen endpoints

1. Subject-level BAC and accuracy.
2. Trial-level prediction disagreement after amplitude recoding relative to
   multiplier `1`.
3. DatasetEA-to-SubjectEA minus SubjectEA paired BAC difference.
4. Whether the ordering of the two alignment methods changes across scales.
5. Number of eigenvalues affected by each floor at dataset and subject stages.

## Decision rule

- Corrected EA passes if relative-floor predictions are unchanged across
  amplitude multipliers within numerical tolerance; exact prediction agreement
  is expected.
- The legacy representation-dependence hypothesis is supported if at least one
  classifier/held-out/alignment cell changes predictions or BAC materially, or
  if the SubjectEA versus hierarchy ranking changes across multipliers.
- Results are mechanistic deterministic interventions. Confidence intervals on
  subject-paired BAC contrasts quantify subject heterogeneity, not stochastic
  uncertainty in the amplitude intervention.

## Separation from shared code

All scripts and results live in
`analysis_outputs/cmpb_unit_intervention_20260925/`. No file under
`MI_test/MI_loso_project/` is modified.
