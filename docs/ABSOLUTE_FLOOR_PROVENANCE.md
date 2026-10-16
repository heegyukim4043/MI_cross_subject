# Absolute-floor provenance and sweep boundary

The fixed numerical floor 1e-8 is traced to the earliest available archived
project implementation, where the inverse square
root clips eigenvalues with max(eigenvalue, eps). Its docstring describes
numerical stability. This records the motivating implementation, not a verified
source-only tuning decision, recommended EEG default or widespread field practice.

The source extraction is preserved in
analysis_outputs/cmpb_absolute_floor_sweep_20261015/ as
first_available_eeg_ea.py.txt. Identifying project commit IDs and the local history
listing are withheld from this anonymous copy; original records remain private.
The original project
snapshot is not the EA publication's reference algorithm or evidence that the
EA authors recommended that floor. Earliest available history is not proof that
the constant first originated in that snapshot.

The follow-up protocol uses zero and nine absolute raw-reference floors from
1e-14 to 1e-6. Policies are applied symmetrically to source and target. The primary
path varies the raw subject reference in S and raw dataset reference in D->S;
the dimensionless final subject floor remains 1e-8. The secondary tied-post path
is explicitly a numerical coupling of constants with different dimensions.

The observed increment grows as S deteriorates, while D->S levels stay relatively
stable in this grid. This supports a comparator-recovery interpretation as a
descriptive pattern, not proof that clipping alone mediates every difference.
Additional cohort access and subject-specific rotations remain unisolated.
The threshold was not optimized on target accuracy; no universal cutoff follows.

