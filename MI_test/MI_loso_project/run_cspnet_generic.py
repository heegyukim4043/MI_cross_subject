"""Generic neural LOSO / LODO runner for any npz dataset (keys: X, y, subjects, ch_names).

Reuses the training code of cross_dataset.py (build_model, init_csp, make_loader, normalize,
train_model, evaluate), so optimisation settings are identical to the Cho2017/Lee2019 runs.
EA floor is set by MI_EA_FLOOR (relative by default; legacy reproduces superseded results); seeds by MI_SEED / MI_SPLIT_SEED.

  --mode loso  --datasets physionetmi           LOSO within one dataset
  --mode lodo  --datasets cho2017 lee2019 bnci2014001 physionetmi
                                                train on all but one dataset, test on the held-out one
  --align none | subject | dataset_subject      SubjectEA per subject; dataset_subject applies one DatasetEA per
                                                dataset first (label-free), then SubjectEA
  --channels shared (channels common to all listed datasets; default) | all (single-dataset LOSO only)
  --shared_with D1 D2 ...                       also restrict the shared set to channels present in these datasets
                                                (e.g. LOSO on the four-dataset LODO channel set)
  --adabn                                       per target subject, re-estimate BatchNorm statistics on that subject's
                                                (unlabeled) test trials in a copy of the model (batch-transductive);
                                                reports adapted metrics and the unadapted BAC in bac_noadapt
Data files are read from MI_NPZ_DIR (default: ../preprocessed_generic).
Output: ../results/generic_<run_id>_<mode>.csv (one row per target subject).
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import time

import numpy as np

import cross_dataset as cd
from eeg_ea import apply_ea_loso, euclidean_align

NPZ_DIR = os.environ.get("MI_NPZ_DIR", os.path.join(os.path.dirname(__file__), "..", "preprocessed_generic"))
N_T = 200


def load(name):
    d = np.load(os.path.join(NPZ_DIR, f"{name}.npz"), allow_pickle=True)
    return {"X": d["X"][:, :, :N_T].astype(np.float32), "y": d["y"].astype(np.int64),
            "subjects": d["subjects"].astype(np.int64), "ch": [str(c) for c in d["ch_names"]]}


def pick(d, chans):
    return {**d, "X": d["X"][:, [d["ch"].index(c) for c in chans], :]}


def train_and_score(model_name, x_tr, y_tr, s_tr, x_te, y_te, s_te, adabn=False):
    ids = np.unique(s_tr)
    rng = np.random.RandomState(cd.SPLIT_SEED)
    val_ids = rng.choice(ids, max(1, int(len(ids) * 0.1)), replace=False)
    val = np.isin(s_tr, val_ids)
    x_tr_n, x_te_n = cd.normalize(x_tr, x_te)
    cd.torch.manual_seed(cd.SEED)
    model = cd.build_model(model_name, x_tr.shape[1], x_tr.shape[2]).to(cd.DEVICE)
    cd.init_csp(model, model_name, x_tr_n[~val], y_tr[~val])
    cd.train_model(model, model_name, cd.make_loader(x_tr_n[~val], y_tr[~val], shuffle=True),
                   cd.make_loader(x_tr_n[val], y_tr[val]))
    out = []
    for s in np.unique(s_te):
        m = s_te == s
        acc, bac, kappa, _, _ = cd.evaluate(model, cd.make_loader(x_te_n[m], y_te[m]))
        row = {"subject": int(s), "n_test": int(m.sum()), "acc": acc, "bac": bac, "kappa": kappa}
        if adabn:
            adapted = copy.deepcopy(model)
            cd.apply_adabn(adapted, x_te_n[m], cd.DEVICE, batch_size=cd.BATCH_SIZE, n_passes=3)
            a_acc, a_bac, a_kappa, _, _ = cd.evaluate(adapted, cd.make_loader(x_te_n[m], y_te[m]))
            row = {**row, "acc": a_acc, "bac": a_bac, "kappa": a_kappa, "bac_noadapt": bac}
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["loso", "lodo"], required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--align", choices=["none", "subject", "dataset_subject"], default="subject")
    ap.add_argument("--channels", choices=["shared", "all"], default="shared")
    ap.add_argument("--shared_with", nargs="*", default=[])
    ap.add_argument("--adabn", action="store_true")
    ap.add_argument("--model", default="cspnet")
    ap.add_argument("--run_id", required=True)
    args = ap.parse_args()
    data = {n: load(n) for n in args.datasets}
    ref = [load(n)["ch"] for n in args.shared_with if n not in data]
    chans = data[args.datasets[0]]["ch"] if args.channels == "all" else \
        [c for c in data[args.datasets[0]]["ch"]
         if all(c in data[n]["ch"] for n in args.datasets) and all(c in r for r in ref)]
    data = {n: pick(d, chans) for n, d in data.items()}
    if args.align in ("subject", "dataset_subject"):
        for d in data.values():
            if args.align == "dataset_subject":
                d["X"] = euclidean_align(d["X"])
            d["X"] = apply_ea_loso(d["X"], d["subjects"])
    out_path = os.path.join(cd.RESULTS_DIR, f"generic_{args.run_id}_{args.mode}.csv")
    done = set()
    if os.path.exists(out_path):  # resume: skip finished (held_out, subject) pairs
        with open(out_path, newline="") as f:
            done = {(r["held_out"], int(r["subject"])) for r in csv.DictReader(f)}
    fields = ["mode", "held_out", "sources", "n_channels", "ea_floor", "align", "model", "seed", "subject", "n_test", "acc", "bac", "kappa", "time_min"]
    if args.adabn:
        fields += ["adapt", "bac_noadapt"]
    new = not os.path.exists(out_path) or os.path.getsize(out_path) == 0  # a killed run can leave an empty file
    with open(out_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        base = {"mode": args.mode, "n_channels": len(chans),
                "ea_floor": os.environ.get("MI_EA_FLOOR", "relative") if args.align != "none" else "n/a",
                "align": args.align, "model": args.model, "seed": cd.SEED}
        if args.adabn:
            base["adapt"] = "adabn"
        if args.mode == "loso":
            n = args.datasets[0]; d = data[n]
            for s in np.unique(d["subjects"]):
                if (n, int(s)) in done:
                    continue
                t0 = time.time(); m = d["subjects"] == s
                for r in train_and_score(args.model, d["X"][~m], d["y"][~m], d["subjects"][~m], d["X"][m], d["y"][m], d["subjects"][m],
                                         adabn=args.adabn):
                    w.writerow({**base, "held_out": n, "sources": n, **r, "time_min": round((time.time() - t0) / 60, 2)}); f.flush()
                print(f"[generic] loso {n} S{s} done", flush=True)
        else:
            for held in args.datasets:
                if any(h == held for h, _ in done):
                    continue
                t0 = time.time(); src = [n for n in args.datasets if n != held]
                off, s_tr = 0, []
                for n in src:  # unique subject ids across source datasets
                    s_tr.append(data[n]["subjects"] + off); off = int(s_tr[-1].max()) + 1000
                rows = train_and_score(args.model, np.concatenate([data[n]["X"] for n in src]), np.concatenate([data[n]["y"] for n in src]),
                                       np.concatenate(s_tr), data[held]["X"], data[held]["y"], data[held]["subjects"],
                                       adabn=args.adabn)
                for r in rows:
                    w.writerow({**base, "held_out": held, "sources": "+".join(src), **r, "time_min": round((time.time() - t0) / 60, 2)})
                f.flush()
                print(f"[generic] lodo held-out {held} done ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
