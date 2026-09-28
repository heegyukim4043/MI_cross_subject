"""Numerical checks for covariance-whitening alignment (Euclidean Alignment).

For an EEG array X (trials x channels x samples) and subject labels, and for each EA regularization mode,
the script reports:
  rescaling   max relative difference of the aligned data when the same EEG is expressed at other amplitude
              scales (x1e-6 ... x1e6). Exact whitening is scale-equivariant, so this should be ~0.
  whitening   max over subjects of ||mean aligned covariance / its mean eigenvalue - I||_F / sqrt(C);
              ~0 means each subject was actually whitened.
  rank        min numerical rank of the subject reference covariances (relative tolerance 1e-10) vs channels.
  floored     max number of reference eigenvalues below the absolute 1e-8 floor (legacy mode clips these).
  order       S->D vs S (dataset EA after subject EA should change nothing) and orthogonality of Q_s in
              D->S = Q_s S (subject EA after dataset EA should differ from S only by a rotation).
A mode passes when rescaling, whitening and order errors are below --tol.

Usage:
  python tools/check_scale_equivariance.py --npz data.npz            # keys X, subjects (y is not used)
  python tools/check_scale_equivariance.py --synthetic               # volt-scaled synthetic data
Exit code 1 if the default mode (relative) fails.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "MI_test" / "MI_loso_project"))
from eeg_ea import EA_EPS_MODES, apply_ea_loso, euclidean_align  # noqa: E402

SCALES = (1e-6, 1e-3, 1.0, 1e3, 1e6)


def mean_cov(x):
    x = x.astype(np.float64)
    return np.einsum("nct,ndt->cd", x, x) / len(x)


def rel_diff(a, b):
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-300))


def check_mode(x, subjects, mode):
    base = apply_ea_loso(x, subjects, eps_mode=mode).astype(np.float64)
    rescale = max(rel_diff(apply_ea_loso((x * s).astype(np.float32), subjects, eps_mode=mode), base) for s in SCALES)
    white, ranks, floored, order_sd, orth = [], [], [], [], []
    ds = euclidean_align(x, eps_mode=mode)
    d_then_s = apply_ea_loso(ds, subjects, eps_mode=mode).astype(np.float64)
    s_then_d = euclidean_align(base.astype(np.float32), eps_mode=mode).astype(np.float64)
    order_sd.append(rel_diff(s_then_d, base))
    for s in np.unique(subjects):
        m = subjects == s
        c = mean_cov(base[m]); c = c / np.mean(np.linalg.eigvalsh(c))
        white.append(np.linalg.norm(c - np.eye(len(c))) / np.sqrt(len(c)))
        r = mean_cov(x[m]); ev = np.linalg.eigvalsh(r)
        ranks.append(int((ev > 1e-10 * ev[-1]).sum())); floored.append(int((ev < 1e-8).sum()))
        # Q_s from least squares on concatenated trials: d_then_s ~ Q_s @ base
        a = np.concatenate(list(base[m]), axis=1); b = np.concatenate(list(d_then_s[m]), axis=1)
        q = b @ np.linalg.pinv(a)
        orth.append(float(np.abs(q @ q.T - np.eye(len(q))).max()))
    return {"mode": mode, "rescaling": rescale, "whitening": max(white), "min_rank": min(ranks),
            "channels": x.shape[1], "max_floored": max(floored), "order_S->D_vs_S": max(order_sd),
            "order_D->S_rotation": max(orth)}


def synthetic(rng):
    n_sub, n_tr, c, t = 6, 60, 16, 200
    xs, ss = [], []
    for s in range(n_sub):
        mix = rng.standard_normal((c, c)) * rng.uniform(0.5, 2.0)
        xs.append(np.einsum("cd,ndt->nct", mix, rng.standard_normal((n_tr, c, t))) * 1e-6)  # volts
        ss.append(np.full(n_tr, s))
    return np.concatenate(xs).astype(np.float32), np.concatenate(ss)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", type=Path)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--modes", nargs="+", default=list(EA_EPS_MODES))
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()
    if args.npz:
        d = np.load(args.npz, allow_pickle=True); x = d["X"].astype(np.float32); subjects = d["subjects"]
    elif args.synthetic:
        x, subjects = synthetic(np.random.default_rng(0))
    else:
        ap.error("give --npz or --synthetic")
    print(f"data: {x.shape}, subjects: {len(np.unique(subjects))}, median |x| = {np.median(np.abs(x)):.3g}")
    header = ["mode", "rescaling", "whitening", "order_S->D_vs_S", "order_D->S_rotation", "min_rank", "channels", "max_floored"]
    print(" | ".join(header) + " | result")
    default_ok = True
    for mode in args.modes:
        r = check_mode(x, subjects, mode)
        ok = all(r[k] < args.tol for k in ("rescaling", "whitening", "order_S->D_vs_S", "order_D->S_rotation"))
        if mode == "relative":
            default_ok = ok
        print(" | ".join(f"{r[k]:.2e}" if isinstance(r[k], float) else str(r[k]) for k in header) + f" | {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if default_ok else 1)


if __name__ == "__main__":
    os.environ.pop("MI_EA_FLOOR", None)
    main()
