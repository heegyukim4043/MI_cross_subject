# Research scope

The study first documents positive cross-dataset MI decoding gains, then asks
what those gains establish. Its primary finding is a difference between two
tested baseline regularization policies in the incremental DatasetEA gain.
Unit re-expression is a numerical control, not a claim of discovering a dataset
calibration error. Transfer-gap and target-access analyses contextualize the
primary comparison rather than claim those questions are new.

| Finding | Evidence | Boundary |
|---|---|---|
| Added-stage increment differed between policies | Development +4.43 pp [3.34,5.52], equal-dataset paired conditional CI | Two policies, no within-operator dose-response; D->S also uses target-cohort data |
| Two-stage absolute performance was near single-stage relative performance | Development difference +0.002 pp [-0.173,0.176] | Descriptive interval; not pre-specified equivalence |
| Same directional pattern in another dataset | Dreyer post-hoc extension +2.98 pp [1.98,4.00] | Post-hoc, not independently implemented confirmatory replication |
| Alignment improved performance but selective gap reduction was not established | LODO +6.29 pp, LOSO +6.32 pp; LOSO-minus-LODO gain interaction +0.04 pp [-1.16,1.22] | Absence of detected interaction is not equivalence or proof of no smaller effect |
| Access and individual variation matter | Reference-budget, offline causal replay, observed individual decreases and sealed few-shot outcomes | Not closed-loop deployment; decreases not established harm; failed external hypothesis retained |

Global unit consistency does not establish baseline regularization adequacy.
The study does not show that absolute constants are intrinsically invalid,
relative floors are universally optimal, clipping count predicts accuracy loss,
all transfer methods exhibit the same behavior, or a reporting standard is validated.

AdaBN Wilcoxon p-values are exploratory and unadjusted; the rank-based paired
test and bootstrap mean-difference CI are distinct inferential procedures.
CSP-LDA post-hoc TOST and CSPNet post-hoc 90% CI containment are distinguished.
All-trial label-informed diagnostics are not deployable label-free predictors.
The adopted Cho factor is an amplitude-version choice with unresolved physical
calibration; it is not part of the novelty claim.
