"""
MRFBCSP (Multi-Resolution Filter Bank CSP) + LDA  LOSO evaluation.

Reference: Soekadar et al. / Kwon et al. style multi-band CSP;
  specifically the version evaluated in
  "Selective Subject Pooling for BCI", Sensors 2021 (MDPI).

Protocol
--------
  Filter bank : 10 sub-bands, 4 Hz wide, 2 Hz step (8-30 Hz)
                [8-12, 10-14, 12-16, 14-18, 16-20,
                 18-22, 20-24, 22-26, 24-28, 26-30]
  CSP         : n_components=4 per band  (2 from each end of spectrum)
  Features    : log-variance of CSP-filtered epochs
  Classifier  : Linear Discriminant Analysis (sklearn)
  Evaluation  : LOSO (Leave-One-Subject-Out)

Usage
-----
    python mrfbcsp_loso.py                      # both datasets
    python mrfbcsp_loso.py --dataset cho2017
    python mrfbcsp_loso.py --dataset lee2019
    python mrfbcsp_loso.py --n_csp 6           # CSP components per band
"""

import os
import glob
import argparse
import csv
import time
from datetime import datetime

import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score

# Attempt MNE CSP first; fall back to manual if unavailable
try:
    from mne.decoding import CSP
    _USE_MNE_CSP = True
except ImportError:
    _USE_MNE_CSP = False

# ---------------------------------------------------------------------------
_BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAVE_DIR    = os.path.join(_BASE_DIR, "preprocessed")
RESULTS_DIR = os.path.join(_BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 2026

# Filter bank definition  (lo, hi)  in Hz
FILTER_BANK = [
    (8,  12), (10, 14), (12, 16), (14, 18), (16, 20),
    (18, 22), (20, 24), (22, 26), (24, 28), (26, 30),
]


# ---------------------------------------------------------------------------
# Signal utilities
# ---------------------------------------------------------------------------

def bandpass(X: np.ndarray, lo: float, hi: float, sfreq: float,
             order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass.  X: (N, C, T) -> (N, C, T)."""
    nyq = sfreq / 2.0
    sos = butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, X, axis=-1)


# ---------------------------------------------------------------------------
# Manual CSP (fallback / reference)
# ---------------------------------------------------------------------------

class ManualCSP:
    """Analytically computed CSP (whitened joint diagonalisation)."""

    def __init__(self, n_components: int = 4):
        self.n_components = n_components
        self.filters_ = None    # (n_components, C)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """X: (N, C, T), y: (N,) binary {0,1}."""
        classes = np.unique(y)
        assert len(classes) == 2
        Sigma = {}
        for c in classes:
            Xc    = X[y == c]                       # (N_c, C, T)
            S     = np.mean(
                np.einsum("nct,ndt->ncd", Xc, Xc), axis=0
            )                                       # (C, C)
            Sigma[c] = S / np.trace(S)
        S0, S1 = Sigma[classes[0]], Sigma[classes[1]]
        eigvals, eigvecs = np.linalg.eigh(S0 + S1)
        # Whitening
        D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals + 1e-9))
        W = D_inv_sqrt @ eigvecs.T                  # (C, C)
        # Joint diagonalisation
        S0w = W @ S0 @ W.T
        eigvals2, eigvecs2 = np.linalg.eigh(S0w)
        # Sort: first and last n_components//2 eigenvectors
        idx  = np.argsort(eigvals2)
        half = self.n_components // 2
        sel  = np.concatenate([idx[:half], idx[-half:]])
        self.filters_ = (eigvecs2[:, sel].T @ W)    # (n_comp, C)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """X: (N, C, T) -> (N, n_components)  log-variance features."""
        Z    = np.einsum("kc,nct->nkt", self.filters_, X)  # (N, k, T)
        var  = Z.var(axis=-1)                               # (N, k)
        return np.log(var + 1e-9)


# ---------------------------------------------------------------------------
# MNE CSP wrapper (preferred when available)
# ---------------------------------------------------------------------------

def _mne_csp_features(X_train, y_train, X_test, n_csp):
    """Return log-variance features using mne.decoding.CSP."""
    csp = CSP(n_components=n_csp, reg=None, log=True, norm_trace=False)
    csp.fit(X_train.astype(np.float64), y_train)
    return csp.transform(X_train.astype(np.float64)), \
           csp.transform(X_test.astype(np.float64))


def _manual_csp_features(X_train, y_train, X_test, n_csp):
    csp = ManualCSP(n_components=n_csp)
    csp.fit(X_train, y_train)
    return csp.transform(X_train), csp.transform(X_test)


def extract_fb_features(X_train, y_train, X_test, sfreq, n_csp):
    """
    Apply filterbank + CSP across all bands and concatenate log-variance features.

    Returns
    -------
    F_train : (N_train, n_bands * n_csp)
    F_test  : (N_test,  n_bands * n_csp)
    """
    feat_train_list, feat_test_list = [], []
    for lo, hi in FILTER_BANK:
        Xtr_f = bandpass(X_train, lo, hi, sfreq)
        Xte_f = bandpass(X_test,  lo, hi, sfreq)
        try:
            if _USE_MNE_CSP:
                ftr, fte = _mne_csp_features(Xtr_f, y_train, Xte_f, n_csp)
            else:
                ftr, fte = _manual_csp_features(Xtr_f, y_train, Xte_f, n_csp)
        except Exception:
            # Fallback if band fails (e.g., rank deficiency)
            ftr, fte = _manual_csp_features(Xtr_f, y_train, Xte_f, n_csp)
        feat_train_list.append(ftr)
        feat_test_list.append(fte)
    return np.hstack(feat_train_list), np.hstack(feat_test_list)


# ---------------------------------------------------------------------------
# LOSO loop
# ---------------------------------------------------------------------------

def load_data(dataset_name):
    path = os.path.join(SAVE_DIR, f"{dataset_name}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run preprocess_data.py first.")
    d        = np.load(path, allow_pickle=True)
    X        = d["X"].astype(np.float32)
    y        = d["y"].astype(np.int64)
    subjects = d["subjects"].astype(np.int64)
    sfreq    = float(d["sfreq"])
    print(f"  Loaded {dataset_name}: X={X.shape}, "
          f"{len(np.unique(subjects))} subjects, sfreq={sfreq} Hz")
    return X, y, subjects, sfreq


def run_loso(dataset_name: str, n_csp: int = 4, resume: bool = False,
             out_dir: str = RESULTS_DIR, run_id: str = None):
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    print(f"\n{'='*60}")
    print(f" MRFBCSP+LDA LOSO - {dataset_name.upper()}")
    print(f" Filter bank : {len(FILTER_BANK)} bands  |  CSP components/band : {n_csp}")
    print(f"{'='*60}")

    X, y, subjects, sfreq = load_data(dataset_name)
    subj_ids = np.unique(subjects)

    # -- Incremental CSV setup ------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    fields = ["dataset", "subject", "n_train", "n_test",
              "acc", "bac", "kappa", "time_min"]
    tag = f"_ncsp{n_csp}"

    done_subjects = set()
    out_csv = None

    if resume:
        if run_id:
            out_csv = os.path.join(out_dir, f"mrfbcsp_results_{run_id}{tag}.csv")
            if not os.path.exists(out_csv):
                raise FileNotFoundError(
                    f"--resume requested but results not found: {out_csv}"
                )
        else:
            pattern = os.path.join(out_dir, f"mrfbcsp_results_*{tag}.csv")
            existing = sorted(glob.glob(pattern))
            if existing:
                out_csv = existing[-1]
            else:
                print("  --resume: no existing CSV found, starting fresh.")
                resume = False

    if not resume:
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = os.path.join(out_dir, f"mrfbcsp_results_{run_id}{tag}.csv")
        if os.path.exists(out_csv):
            resume = True
        else:
            with open(out_csv, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()

    if resume and out_csv:
        with open(out_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("dataset") == dataset_name:
                    try:
                        done_subjects.add(int(row["subject"]))
                    except Exception:
                        pass
        print(f"  Resuming: {len(done_subjects)} subjects done -> {out_csv}")

    print(f"  Results -> {out_csv}")

    results = []
    t_total = time.time()

    for i, test_subj in enumerate(subj_ids):
        if test_subj in done_subjects:
            print(f"  [{i+1:02d}/{len(subj_ids)}] S{test_subj:02d} -- skipped (already done)")
            continue
        t0 = time.time()

        test_mask  = subjects == test_subj
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        # Channel-wise z-score normalisation (train stats only)
        mu  = X_train.mean(axis=(0, 2), keepdims=True)
        std = X_train.std(axis=(0, 2),  keepdims=True) + 1e-8
        X_train = (X_train - mu) / std
        X_test  = (X_test  - mu) / std

        # Filter bank CSP features
        F_train, F_test = extract_fb_features(
            X_train, y_train, X_test, sfreq, n_csp
        )

        # LDA classifier
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        lda.fit(F_train, y_train)
        y_pred = lda.predict(F_test)

        acc   = accuracy_score(y_test, y_pred)
        bac   = balanced_accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        elapsed = time.time() - t0

        row = dict(
            dataset=dataset_name,
            subject=int(test_subj),
            n_train=int(train_mask.sum()),
            n_test=int(test_mask.sum()),
            acc=acc, bac=bac, kappa=kappa,
            time_min=round(elapsed / 60, 3),
        )
        results.append(row)
        remaining = elapsed * (len(subj_ids) - i - 1 - len(done_subjects))
        print(f"  [{i+1:02d}/{len(subj_ids)}] S{test_subj:02d} | "
              f"Acc={acc*100:.1f}%  BAcc={bac*100:.1f}%  k={kappa:.3f}  "
              f"[{elapsed:.1f}s | ETA {remaining/60:.1f}min]")

        # Incremental save
        with open(out_csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)

    total_time = time.time() - t_total
    accs   = [r["acc"]   for r in results]
    bacs   = [r["bac"]   for r in results]
    kappas = [r["kappa"] for r in results]

    print(f"\n{'-'*60}")
    print(f"  {dataset_name.upper()} MRFBCSP+LDA Summary ({len(results)} subjects)")
    print(f"  Accuracy  : {np.mean(accs)*100:.2f} +/- {np.std(accs)*100:.2f} %")
    print(f"  Bal. Acc  : {np.mean(bacs)*100:.2f} +/- {np.std(bacs)*100:.2f} %")
    print(f"  Cohen k   : {np.mean(kappas):.3f} +/- {np.std(kappas):.3f}")
    print(f"  Total time: {total_time/60:.1f} min  "
          f"({total_time/len(results):.1f} s/subject)")
    print(f"{'-'*60}\n")

    return results




# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cho2017", "lee2019", "both"],
                        default="both")
    parser.add_argument("--n_csp", type=int, default=4,
                        help="CSP components per sub-band (default 4)")
    parser.add_argument("--out_dir", type=str, default=RESULTS_DIR,
                        help="Directory to save results (default: results/)")
    parser.add_argument("--run_id", type=str, default=None,
                        help="Run id for output filenames (default: timestamp)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing CSV for the same run_id (or latest if run_id is not set)")
    args = parser.parse_args()

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.dataset in ("cho2017", "both"):
        run_loso("cho2017", n_csp=args.n_csp, resume=args.resume,
                 out_dir=args.out_dir, run_id=run_id)
    if args.dataset in ("lee2019", "both"):
        run_loso("lee2019", n_csp=args.n_csp, resume=args.resume,
                 out_dir=args.out_dir, run_id=run_id)
