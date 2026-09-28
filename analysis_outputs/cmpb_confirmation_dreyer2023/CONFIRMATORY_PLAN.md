# Confirmatory test of the core claims on a sealed dataset (Dreyer2023)

Fixed on 2026-09-25, **before any Dreyer2023 data were downloaded or loaded** in this project.
Any deviation is logged in `MI_test/progress.md` with its reason.

## Data and preprocessing (identical to the development datasets)
- MOABB `Dreyer2023`, left- vs right-hand imagery, all subjects and all available runs/sessions.
- `LeftRightImagery(fmin=8, fmax=30, tmin=0.5, tmax=2.5, resample=100)`, first 200 samples, stored in volts.
- Preprocessing uses the isolated environment `.venv_moabb_new` (Python 3.11, current MOABB) only for
  loading; all analysis code is the existing development code.
- Sources for cross-dataset tests: the four development datasets pooled (Cho2017, Lee2019, BNCI2014-001,
  PhysionetMI). Channels: those shared by all five datasets (count recorded before any accuracy is computed).
- Classifier: CSP(8) + shrinkage LDA; balanced accuracy per subject; subject is the unit.

## Label-free check done first (does not use labels)
Count, per subject, the covariance eigenvalues below the absolute 1e-8 floor (legacy EA definition).
H1 is tested only if the median count is > 0; otherwise H1 is reported as "not applicable" (the dataset's
stored scale does not trigger the artifact).

## Hypotheses and decision rules (alpha 0.05, Holm over the hypotheses that are tested)
- **H1 (numerical artifact):** LOSO BAC with relative-floor EA exceeds legacy absolute-floor EA.
  Test: one-sided paired Wilcoxon; supported if Holm p < 0.05 and mean difference > 0.
- **H2 (marginal shift removed by EA):** cross-dataset (4 dev datasets -> Dreyer2023) BAC with relative
  SubjectEA exceeds no alignment. One-sided paired Wilcoxon, Holm p < 0.05.
- **H3 (dataset-level alignment adds at most little):** cross-dataset corrected DatasetEA->SubjectEA minus
  corrected SubjectEA lies within +/-1 pp. Paired TOST (90% CI inside +/-1 pp).
- **H4 (cross-dataset-specific loss is small):** LOSO(relative EA, within Dreyer2023) minus
  cross-dataset(relative EA) < 2 pp. Supported if the upper bound of the 95% paired-bootstrap CI < 2 pp.
- **H5 (intrinsic separability predicts accuracy):** split-half own class separability (calibration half)
  is positively rank-correlated with LOSO BAC on the evaluation half. One-sided Spearman, Holm p < 0.05.
- **H6 (few-shot labels do not beat unsupervised EA):** source + 5 own labeled trials per class
  (natural weight) minus LOSO(relative EA), on the evaluation half, has a 95% CI upper bound < +1 pp.

## Reporting
All six results are reported whatever their direction. Failure of a hypothesis is reported as a failure to
confirm on this dataset, not re-tested with modified settings.
