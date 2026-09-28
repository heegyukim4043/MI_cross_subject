"""Functional test of public EA implementations (core lines copied verbatim from each repository).

Input: Lee2019 subject 1, 48 common channels, 100 Hz, 8-30 Hz band, as stored (volts) and x1e6 (uV).
Metrics per protocol: whitening residual ||mean(Z Z^T)/scale - I||_F / sqrt(C) where the aligned mean
covariance is normalised by its trace/C (so a pure global scale does not count as failure); an
implementation FAILS if the residual is > 0.1 on volt input while < 0.1 on microvolt input.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.linalg import fractional_matrix_power, inv, sqrtm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


# ---- verbatim cores ------------------------------------------------------------------------------
def ea_wu_family(x):  # MIRepNet, EEG-FM-Benchmark, DeepTransferEEG, SDDA, DBConformer, MVCNet, EEGAug, FedBS, PGAP, MotorImageryTutorial
    cov = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        cov[i] = np.cov(x[i])
    refEA = np.mean(cov, 0)
    sqrtRefEA = fractional_matrix_power(refEA, -0.5)
    return np.stack([np.dot(sqrtRefEA, t) for t in x])


def ea_spdsafe(x, epsilon=1e-6):  # DeepTransferEEG EA_SPDsafe (trace-relative ridge)
    n = x.shape[0]
    C = np.zeros((x[0].shape[0], x[0].shape[0]))
    for X in x:
        C += X @ X.T
    R_bar = C / n
    trace = np.trace(R_bar)
    R_bar += epsilon * (trace / R_bar.shape[0]) * np.eye(R_bar.shape[0])
    eigvals, eigvecs = np.linalg.eigh(R_bar)
    ref = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    return ref @ x


def ea_singlem(data):  # ttlabtuat/SingLEM preprocessing/extract_features/adapters/policies.py:214-221
    cov = np.zeros((data.shape[1], data.shape[1]), dtype="float64")
    for trial in data:
        cov += trial @ trial.T / max(trial.shape[1] - 1, 1)
    cov /= max(data.shape[0], 1)
    eigval, eigvec = np.linalg.eigh(cov + np.eye(cov.shape[0]) * 1e-6)
    inv_sqrt = eigvec @ np.diag(1.0 / np.sqrt(np.maximum(eigval, 1e-6))) @ eigvec.T
    return np.asarray([inv_sqrt @ trial for trial in data], dtype="float32")


def ea_miaosonic(x):  # miaosonic/EA-EEG main2.py:160-180 (absolute ridge 1e-5)
    num_samples, num_channels, _ = x.shape
    cov = np.zeros((num_samples, num_channels, num_channels))
    for i in range(num_samples):
        cov[i] = np.cov(x[i])
    refEA = np.mean(cov, axis=0)
    refEA += 1e-5 * np.eye(num_channels)
    sqrtRefEA = fractional_matrix_power(refEA, -0.5)
    return np.stack([np.dot(sqrtRefEA, t) for t in x])


def ea_frdw_sacm(x):  # XinRu2001/FRDW utils/EA.py:46, WangHongbinary/SACM dataset_SEEG.py:108 (+1e-8*I after R^-1/2)
    cov = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        cov[i] = np.cov(x[i])
    refEA = np.mean(cov, 0)
    sqrtRefEA = fractional_matrix_power(refEA, -0.5) + (0.00000001) * np.eye(x.shape[1])
    return np.stack([np.dot(sqrtRefEA, t) for t in x])


def ea_sqrtm_inv(x):  # orvindemsy/EA-wLTL, ea-tca-mi-bci func/EA.py: sqrtm(inv(RefEA))
    RefEA = np.mean([t @ t.T for t in x], axis=0)
    R_inv = np.real(sqrtm(inv(RefEA)))
    return np.stack([R_inv @ t for t in x])


def ea_inv_sqrtm(x):  # mcd4874/NeurIPS_competition, chchenhui/fabscore: inv(sqrtm(R))
    r = np.mean([t @ t.T for t in x], axis=0)
    r_op = inv(np.real(sqrtm(r)))
    return np.stack([r_op @ t for t in x])


def ea_pyriemann(x):  # BrynhildrW/SSVEP_algorithms (pyriemann invsqrtm)
    from pyriemann.utils.base import invsqrtm
    P = invsqrtm(np.mean([t @ t.T for t in x], axis=0))
    return np.stack([P @ t for t in x])


def ea_this_project_legacy(x):  # this project's original eeg_ea.py (absolute eigenvalue floor 1e-8, no /T)
    R = np.mean([t @ t.T for t in x], axis=0)
    w, V = np.linalg.eigh(R)
    w = np.maximum(w, 1e-8)
    return np.einsum("cd,ndt->nct", V @ np.diag(w ** -0.5) @ V.T, x)


IMPLEMENTATIONS = {
    "wu_family_exact(fractional_matrix_power)": ea_wu_family,
    "DeepTransferEEG_EA_SPDsafe(relative ridge)": ea_spdsafe,
    "SingLEM_euclidean_alignment(abs ridge+clamp 1e-6)": ea_singlem,
    "miaosonic_EA-EEG(abs ridge 1e-5)": ea_miaosonic,
    "FRDW/SACM(+1e-8 I after R^-1/2)": ea_frdw_sacm,
    "orvindemsy(sqrtm(inv))": ea_sqrtm_inv,
    "NeurIPS_competition/fabscore(inv(sqrtm))": ea_inv_sqrtm,
    "pyRiemann invsqrtm": ea_pyriemann,
    "this_project_legacy(abs eig floor 1e-8)": ea_this_project_legacy,
}


def residual(z):
    c = np.mean([t @ t.T for t in z.astype(np.float64)], axis=0)
    c = c / (np.trace(c) / len(c))
    return float(np.linalg.norm(c - np.eye(len(c))) / np.sqrt(len(c)))


def main():
    with np.load(ROOT / "MI_test" / "preprocessed_sfreq100" / "lee2019.npz", allow_pickle=True) as d:
        names = [str(c) for c in d["ch_names"]]
        with np.load(ROOT / "MI_test" / "preprocessed_sfreq100" / "cho2017.npz", allow_pickle=True) as c:
            common = [n for n in [str(x) for x in c["ch_names"]] if n in names]
        idx = [names.index(n) for n in common]
        x_v = d["X"][d["subjects"] == d["subjects"].min()][:, idx, :201].astype(np.float64)
    rows = []
    for name, fn in IMPLEMENTATIONS.items():
        r_v = residual(fn(x_v.copy()))
        r_uv = residual(fn(x_v.copy() * 1e6))
        rows.append({"implementation": name, "residual_volts": round(r_v, 6), "residual_microvolts": round(r_uv, 6),
                     "fails_on_volts": bool(r_v > 0.1 and r_uv < 0.1)})
        print(f"{name:52s} volts={r_v:9.4f}  uV={r_uv:9.4f}  {'FAIL' if rows[-1]['fails_on_volts'] else 'ok'}")
    with open(HERE / "functional_test_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
