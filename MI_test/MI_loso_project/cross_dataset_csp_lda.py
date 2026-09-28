"""Cross-dataset CSP-LDA baselines.

Runs source-dataset training and target-dataset per-subject evaluation using
fixed channel order from cross_dataset.py. Intended baselines:
- CSP-LDA
- EA-CSP-LDA
- DatasetEA + SubjectEA + CSP-LDA
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from cross_dataset import (  # noqa: E402
    RESULTS_DIR,
    SEED,
    N_TIMES,
    find_common_channels,
    load_dataset,
    normalize,
    apply_dataset_ea,
)
from eeg_ea import apply_ea_loso  # noqa: E402
from mrfbcsp_loso import ManualCSP  # noqa: E402


def features_csp_lda(X_train, y_train, X_test, n_csp=8):
    csp = ManualCSP(n_components=n_csp)
    csp.fit(X_train, y_train)
    F_train = csp.transform(X_train)
    F_test = csp.transform(X_test)
    scaler = StandardScaler()
    F_train = scaler.fit_transform(F_train)
    F_test = scaler.transform(F_test)
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit(F_train, y_train)
    return lda, F_test


def run_cross(
    train_name: str,
    test_name: str,
    run_id: str,
    channel_set: str = "standard_mi",
    use_ea: bool = False,
    use_dataset_ea: bool = False,
    ea_order: str = "dataset_subject",
    n_csp: int = 8,
):
    common_ch, cho_idx, lee_idx = find_common_channels(channel_set=channel_set)
    train_idx = cho_idx if train_name == "cho2017" else lee_idx
    test_idx = cho_idx if test_name == "cho2017" else lee_idx

    print("\n" + "=" * 60)
    print(f"  CSP-LDA Cross-dataset: TRAIN={train_name.upper()} -> TEST={test_name.upper()}")
    print(f"  channel_set={channel_set}  n_channels={len(common_ch)}  n_csp={n_csp}")
    print(f"  EA={use_ea}  DatasetEA={use_dataset_ea}  EA order={ea_order}")
    print("=" * 60)

    X_train, y_train, subj_train = load_dataset(train_name, train_idx, n_times=N_TIMES)
    X_test, y_test, subj_test = load_dataset(test_name, test_idx, n_times=N_TIMES)

    def subject_pair(X_src, X_tgt):
        print("  Applying subject-level EA")
        return apply_ea_loso(X_src, subj_train), apply_ea_loso(X_tgt, subj_test)

    def dataset_pair(X_src, X_tgt):
        print("  Applying dataset-level EA")
        return apply_dataset_ea(X_src), apply_dataset_ea(X_tgt)

    if ea_order == "subject_dataset":
        if use_ea:
            X_train, X_test = subject_pair(X_train, X_test)
        if use_dataset_ea:
            X_train, X_test = dataset_pair(X_train, X_test)
    elif ea_order == "dataset_subject":
        if use_dataset_ea:
            X_train, X_test = dataset_pair(X_train, X_test)
        if use_ea:
            X_train, X_test = subject_pair(X_train, X_test)
    else:
        raise ValueError(f"unknown ea_order: {ea_order}")

    X_train, X_test = normalize(X_train, X_test)

    tag = f"cross_{train_name}_to_{test_name}_csp_lda"
    out_csv = Path(RESULTS_DIR) / f"loso_results_{run_id}_{tag}.csv"
    fields = [
        "train_ds",
        "test_ds",
        "model",
        "subject",
        "n_test",
        "acc",
        "precision_macro",
        "f1_macro",
        "bac",
        "kappa",
    ]
    with out_csv.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    rows = []
    for subj in np.unique(subj_test):
        mask = subj_test == subj
        lda, F_test = features_csp_lda(X_train, y_train, X_test[mask], n_csp=n_csp)
        pred = lda.predict(F_test)
        y_true = y_test[mask]
        acc = accuracy_score(y_true, pred)
        prec = precision_score(y_true, pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, pred, average="macro", zero_division=0)
        bac = balanced_accuracy_score(y_true, pred)
        kap = cohen_kappa_score(y_true, pred)
        row = {
            "train_ds": train_name,
            "test_ds": test_name,
            "model": "csp_lda",
            "subject": int(subj),
            "n_test": int(mask.sum()),
            "acc": round(acc * 100, 1),
            "precision_macro": round(prec * 100, 1),
            "f1_macro": round(f1 * 100, 1),
            "bac": round(bac * 100, 1),
            "kappa": round(kap, 3),
        }
        rows.append(row)
        with out_csv.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        print(f"    S{subj:02d}: acc={acc*100:.1f}%  bac={bac*100:.1f}%  k={kap:.3f}")

    accs = [r["acc"] for r in rows]
    kaps = [r["kappa"] for r in rows]
    print("\n  " + "-" * 50)
    print(f"  {train_name.upper()} -> {test_name.upper()} | CSP-LDA")
    print(f"  Accuracy : {np.mean(accs):.2f} +/- {np.std(accs):.2f} %")
    print(f"  Cohen k  : {np.mean(kaps):.3f} +/- {np.std(kaps):.3f}")
    print(f"  Saved    : {out_csv}")
    return float(np.mean(accs)), float(np.mean(kaps)), out_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--both", action="store_true")
    parser.add_argument("--train", choices=["cho2017", "lee2019"], default="cho2017")
    parser.add_argument("--test", choices=["cho2017", "lee2019"], default="lee2019")
    parser.add_argument("--channel_set", choices=["common", "standard_mi"], default="standard_mi")
    parser.add_argument("--ea", action="store_true")
    parser.add_argument("--dataset_ea", action="store_true")
    parser.add_argument("--ea_order", choices=["dataset_subject", "subject_dataset"], default="dataset_subject")
    parser.add_argument("--n_csp", type=int, default=8)
    parser.add_argument("--run_id", default=None)
    args = parser.parse_args()

    np.random.seed(SEED)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.both:
        r1 = run_cross("cho2017", "lee2019", run_id, args.channel_set, args.ea,
                       args.dataset_ea, args.ea_order, args.n_csp)
        r2 = run_cross("lee2019", "cho2017", run_id, args.channel_set, args.ea,
                       args.dataset_ea, args.ea_order, args.n_csp)
        print("\n" + "=" * 60)
        print(f"  Cho->Lee : {r1[0]:.2f}%  k={r1[1]:.3f}")
        print(f"  Lee->Cho : {r2[0]:.2f}%  k={r2[1]:.3f}")
    else:
        run_cross(args.train, args.test, run_id, args.channel_set, args.ea,
                  args.dataset_ea, args.ea_order, args.n_csp)


if __name__ == "__main__":
    main()
