# Research scope

The study documents positive cross-dataset MI decoding gains, then tests what
those gains establish. Its primary comparison measures how the D->S increment
over S differs between two stage-wise regularization policies. Policies apply to
all applicable source and target alignment stages; K is not the effect of changing
S alone. D->S additionally consumes other target participants' EEG.

| Finding | Evidence | Boundary |
|---|---|---|
| Policy-dependent pipeline increments | Development K +4.43 pp adopted Cho, +2.75 original Cho; Dreyer +2.98 and +3.45 | Pipeline-level contrasts; not an isolated transformation or cohort-information effect |
| Strong-floor comparator deterioration | Tested raw-floor sweep Delta +0.18 to +6.19 pp while D->S levels remain comparatively stable | Adopted Cho, fixed post-D subject floor, finite exploratory grid; no universal threshold or optimal epsilon |
| Absolute two-stage near relative single-stage performance | Initial development contrast +0.002 pp [-0.173, 0.176]; later version-specific records supplied separately | Descriptive interval, not pre-specified equivalence |
| Direction persists with other data and versions | Dreyer post-hoc policy and source-version extensions | Not independently implemented confirmatory replication; original pre-specified tests remain separate |
| Performance gain is not selective gap reduction | LODO +6.29 pp, LOSO +6.32 pp; LOSO-minus-LODO interaction +0.04 pp [-1.16, 1.22] | No detected interaction is not equivalence or absence of a smaller effect |
| Target access and individuals matter | Budget, offline causal replay, observed decreases and sealed few-shot results | Not closed-loop deployment; decreases are not established harm; unsuccessful external hypothesis retained |

## Mathematical and numerical scope

Exact subject whitening after an invertible dataset transform can differ from
direct subject whitening by a subject-specific orthogonal rotation. The final
subject whitening must be exact; the dataset step may itself use a floored SPD
transform. This identity does not predict identical decoding or a small gain.
The absence of final-stage clipping in the tested sweep is recorded, not a theorem
that flooring can never matter.

Coherent global unit conversion and Cho-only amplitude-version sensitivity are
distinct experiments. The latter deliberately changes only Cho and retains
numerical constants; it does not verify physical calibration. Absolute constants
are not intrinsically invalid, and relative floors are not universally optimal.
The study does not validate clipping count as a predictor or establish prevalence
of numerical failures across the field.

## Inferential scope

The follow-up sweep and version analyses were planned after earlier findings.
They are exploratory, with saved protocols fixed before their particular runs,
not prospective preregistration. Pointwise CIs and simultaneous bands have distinct
roles and separate analysis families. Intervals condition on fitted models,
observed trials and fixed/shared references; no new-site uncertainty is modeled.

Neural legacy versus relative comparisons retain unequal seed coverage and
float32/float64 covariance precision and are exploratory support, not a fully
matched four-condition neural replication. AdaBN Wilcoxon p-values are unadjusted
exploratory rank tests, distinct from mean-difference bootstrap intervals.
CSP-LDA post-hoc TOST and CSPNet 90% CI containment must not be conflated.
All-trial label-informed diagnostics are not deployable label-free predictors.

This is not a new alignment algorithm, universally optimal regularizer, isolated
causal attribution, validated diagnostic cutoff or independently validated
reporting standard.

