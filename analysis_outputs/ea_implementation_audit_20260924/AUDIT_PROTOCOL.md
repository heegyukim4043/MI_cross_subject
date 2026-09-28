# Public EA implementation audit: protocol (fixed before searching, 2026-09-24)

## Question
Do publicly available EEG Euclidean/Riemannian alignment implementations regularise the reference
covariance in a way that depends on the stored signal scale (e.g. volts from MOABB), so that
alignment silently stops whitening low-amplitude data?

## Candidate identification (fixed)
1. GitHub code search (via `gh search code`, public repositories, Python), queries:
   - `"euclidean alignment" eeg`
   - `fractional_matrix_power eeg align`
   - `"euclidean_alignment"`
   - `"EA(" eeg "-0.5"` style reference whitening (`R ** -0.5`, `inv(sqrtm(`)
2. Plus the libraries most likely to be used by MI-EEG researchers: pyRiemann (TLCenter / invsqrtm),
   braindecode, MOABB, TorchEEG, and the Wu-group reference code (DeepTransferEEG).
3. Inclusion: public repository; Python implementation that computes a reference covariance from
   EEG trials and applies its inverse square root (EA / RA re-centering); code is readable.
   Exclusion: forks/duplicates of an included file; notebooks without a reusable function;
   implementations that only call an already-included library function.
4. Cap: the 20 most-starred distinct repositories among the search hits, plus the libraries in (2).
   Stars and search date are recorded; no candidate is dropped after its code has been inspected.

## Classification of the reference inverse square root (fixed)
- **S-inv (scale-invariant)**: exact `R^{-1/2}` (eigh / fractional_matrix_power / sqrtm+inv) with no
  regularisation, OR regularisation proportional to the matrix scale (trace/C, lambda_max), OR a
  relative pseudo-inverse cutoff.
- **S-dep (scale-dependent)**: absolute additive ridge (`R + eps*I`) or absolute eigenvalue clamp
  (`max(lambda, eps)`) with eps fixed in data units, applied without prior rescaling.
- **S-mitigated**: absolute eps, but the data are rescaled before EA in the same pipeline
  (e.g. per-trial/channel z-scoring, microvolt conversion), so the stored scale does not reach EA.
- **Unclear**: cannot be determined from the code.

## Functional test (for every implementation that can be imported or copied verbatim)
Input: Lee2019 subject 1, common48, 100 Hz, 8-30 Hz, as stored (volts) and multiplied by 1e6 (uV).
Metrics: (i) whitening residual ||mean(Z Z^T) - I||_F / sqrt(C); (ii) relative difference between the
aligned outputs of the volt and microvolt inputs after removing the global scale; (iii) number of
eigenvalues affected by the regulariser. An implementation "fails" if (i) > 0.1 on volt input while
< 0.1 on microvolt input.

## Reporting
Every candidate is listed with repository, file/line, stars, class, and test outcome (or the reason
it could not be tested). Absence of affected implementations is reported as such.
