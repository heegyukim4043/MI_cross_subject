# Cho2017 amplitude provenance

Status: 2026-10-12. Physical calibration remains unresolved.

## Evidence contract

The project uses an archived loader-produced Cho2017 array and a second version
obtained by multiplying it by 0.03125. The multiplication and array identities
are reproducible operations. They do not establish which array has the correct
physical amplitude calibration.

The author reports having heard an informal explanation of the laboratory's
acquisition circumstances. No documentary acquisition/export/calibration record
has been supplied that establishes the 0.03125 factor for these distributed arrays.
This account motivates an uncertainty note, not a verified technical attribution.
No speaker identity, date or quotation is asserted here.

| Item | What is established | What is not established |
|---|---|---|
| Factor 0.03125 | Adopted operational scaling; applied to the archived array | A uniquely correct physical-unit conversion |
| File/array hashes | Identity of the recorded analysis versions | Accuracy of physical calibration |
| RMS comparisons | Differences in observed amplitudes | Which acquisition/export step caused them, or whether units are correct |
| Informal acquisition account | Author-reported context | Documentary evidence of ADC counts, gain, export settings or a loader fault |
| BioSemi/BCI2000 context | A possible subject for provenance clarification | A demonstrated device, software or public-dataset defect |

Do not describe this as a confirmed 32-fold error in Cho2017, MOABB or the
acquisition system. Do not infer the factor from cross-dataset RMS similarity.
Manufacturer specifications alone would not connect the archived array to the
specific gain/export configuration; that connection would require additional records.

## Historical identifiers

The following names are preserved to avoid breaking scripts and archived records.
Their wording is not a physical-correctness assertion.

| Historical identifier | Meaning in this project |
|---|---|
| `cho2017_moabb_uncorrected.npz`, `--select uncorrected` | Original archived project-loader array |
| `cho2017_unitcorrected.npz`, `--select corrected` | Adopted array multiplied by 0.03125 |
| `chofix`, `unitfix`, `corrected` in historical result names | Analysis-version labels; consult the originating runner and hash |

The tool's recorded SHA256 values for the contiguous X array are:

```text
original: 188f759a4ddfb49413afa2dd1b8815942926a8041000ceff802c1f7f4880ca19
scaled:   6fbc8deaa2aaedae4b6f57a7938251c0bf825304ef7655e71fd4607987b77d50
```

Whole-file hashes are separately listed in `data_manifest/SHA256SUMS`; an array
hash and an NPZ-file hash are different checks. Neither was newly recomputed from
EEG during this documentation update.

## Loader and analysis scope

The Cho/Lee preparation script requests MOABB output as MNE Epochs and then calls
`epochs.get_data()`, following the default MNE V convention. This must not be
confused with MOABB's ndarray output path, which can apply the dataset unit factor.
The actual API path and installed version belong in each data contract. A V
convention does not itself verify the original Cho acquisition/export calibration.

The later development analyses use the adopted scaled version. The original
Dreyer pre-specified confirmation used the original Cho source version; its later
amplitude sensitivity and post-hoc policy extension are distinct analyses.
Do not retrospectively relabel or overwrite the original confirmation.

Subject-wise relative EA is invariant to a positive global scalar in exact
arithmetic. Numerical tolerance and rank handling still matter. This property
does not imply invariance for pooled preprocessing, all classifiers, or every
unscaled/no-alignment baseline. State the input version for those conditions.

## Tool cautions

`tools/correct_cho2017_units.py` retains historical behavior in this documentation
update. Its `--npz_dir` export always selects scaled Cho, independently of
`--select`. The current helper can copy a selected file before returning a
nonzero status for a failed hash check; a fail-closed implementation remains a
release-readiness task. Do not use it as a certified calibration or validation tool.

The archived `pilot_cho_unitfix.py` amplitude heuristic is not reliable physical-unit
or version identification for new inputs. Use explicit provenance and recorded
array hashes instead of extending that heuristic to other data.

No additional physical-unit evidence is claimed. Until such evidence is available,
retain both versions, report sensitivity where relevant, and keep this limitation
in a brief manuscript Methods disclosure and these technical records. Cho unit
provenance is not presented as a novelty claim or a confirmed dataset defect.
