# Authorized four-check follow-up, 2026-10-07

User requested: verify BCI2000/BioSemi provenance, directly compare precision-matched
policies, reorganize literature evidence for reporting practices, and assess check validity.

## Numerical comparison fixed before this run

- Reuse the archived unit-intervention file hashes, ordered21 channels, first200
  samples at100Hz, 224 subjects, all four held-out datasets, full-reference
  transductive S and D->S. No new unit correction, reference split or exclusion.
- Covariance and eigendecomposition float64 for BOTH policies; alignment output
  and downstream ManualCSP input float32 for BOTH, as in existing pipeline.
- Policies: coherent absolute (first epsilon=1e-8*a^2; H second epsilon=1e-8),
  relative (each stage epsilon=1e-8*trace(R)/C). Units a=1 and1e-6.
- S and D->S for each policy/unit:8 cells x4 held-out datasets=32 source-trained
  CSP-LDA fits. Same source-only normalization, CSP8, StandardScaler and shrinkage
  LDA helper definitions as the archived run. All cells regenerated in one environment.
- Four CPU threads, sequential. Check source hashes before input use. No GPU,
  no alteration of running queues/shared code, no target-label model selection.
- Primary summaries: subject macro BAC per dataset, equal-dataset mean and pooled
  mean reported separately; I(policy)=BAC_DS-BAC_S; I(abs)-I(rel); all absolute BACs.
- Unit check: aligned-array relative L2<=1e-5; decision difference<=1e-3 and
  prediction agreement reported separately. Do not silently drop failed cells.
- Expected direction is already informed by previous runs: invariant rescaling
  within policies, larger hierarchy increment under absolute policy. This is a
  targeted development replication, NOT new preregistration/independent confirmation.
- No clinical success margin, significance-driven continuation or universal
  clipping threshold. Optional95% paired subject-bootstrap intervals are descriptive,
  fixed-model/fixed-dataset; no new confirmatory p-value family.

## Check validity

Freeze tests for specific events, not performance prediction: coherent transformations
should pass invariance; fixed epsilon under rescaling should be flagged; incorrectly
rescaling BOTH H epsilons should be flagged. Coherent strong regularization should
pass the unit test but retain a regularization-inspection flag. Flag does NOT mean
invalid or harmful regularization. Test actual operator on fixed synthetic SPD fixtures
and assess real pipeline outputs. No prospective accuracy/ROC claim from these tests.

## Evidence research

Only public queries. Distinguish manufacturer's digitization, BCI2000 stored raw gains,
and Cho exported MAT units. Official documentation/code cannot alone prove the actual
historical export path. Literature evidence is a bounded, source-located reconstruction,
not a prevalence estimate or completed systematic review. Preserve prior failed audits.

## Outputs

Staging `four_checks_20261007/`; publish to a new active output directory after QA.
Keep canonical progress updated on start/completion. No manuscript edits.
