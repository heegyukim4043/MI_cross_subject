"""Replicate the drift-triggered EA reset on BNCI2014-001 (2 sessions x 6 runs, chronological).

Drift parameters are the ones fixed for the Lee2019 prototype (w=10, z=3, m=20); nothing is tuned here.
Settings (CSP-LDA, relative-floor EA, label-free causal target alignment):
  loso        : BNCI2014-001 leave-one-subject-out (source = other 8 subjects)
  cross       : Lee2019 -> BNCI2014-001 and Cho2017 -> BNCI2014-001 on the shared channels
Target stream: session 0 (144 trials) then session 1 (144 trials); the boundary is hidden from all
causal variants except online_oracle_reset.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "cmpb_floor_gate_20260924"))
sys.path.insert(0, str(HERE.parent / "cmpb_budget_20260924"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "MI_test" / "MI_loso_project"))
from run_floor_gate import metrics, write_rows  # noqa: E402
from run_budget_causal import Model  # noqa: E402
from run_bdea import align_stream, batch_align, references, trial_covs  # noqa: E402

VARIANTS = ("online_plain", "online_oracle_reset", "drift_reset", "ewma_0.95", "ewma_0.98")


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as d:
        out = {k: d[k] for k in d.files}
    out["ch_names"] = [str(c) for c in out["ch_names"]]
    return out


def pick(data: dict, channels: list[str], n_times: int) -> dict:
    idx = [data["ch_names"].index(c) for c in channels]
    keys = [k for k in ("y", "subjects", "sessions", "runs") if k in data]
    return {"X": data["X"][:, idx, :n_times].astype(np.float32), **{k: data[k] for k in keys}}


def evaluate(ctx: dict, model: Model, x: np.ndarray, y: np.ndarray, sessions: np.ndarray, rows, resets_log):
    boundary = int(np.flatnonzero(sessions == sessions.max())[0])
    groups_session = [np.arange(boundary), np.arange(boundary, len(y))]
    c = trial_covs(x)
    pos2 = np.where(np.arange(len(y)) >= boundary, np.arange(len(y)) - boundary + 1, 0)
    preds = {"batch_subject": model.predict(batch_align(x, [np.arange(len(y))])),
             "batch_session": model.predict(batch_align(x, groups_session))}
    for v in VARIANTS:
        refs, resets = references(c, v, {"src": np.eye(x.shape[1])}, boundary)
        preds[v] = model.predict(align_stream(x, refs))
        if v == "drift_reset":
            after = [r for r in resets if r >= boundary]
            resets_log.append({**ctx, "n_resets": len(resets), "resets": " ".join(map(str, resets)),
                               "first_reset_after_boundary": after[0] if after else -1, "boundary": boundary})
    for name, p in preds.items():
        for wname, keep in (("all", np.ones(len(y), bool)), ("s2_first20", (pos2 >= 1) & (pos2 <= 20)),
                            ("session2", pos2 >= 1)):
            rows.append({**ctx, "variant": name, "window": wname, "n": int(keep.sum()), **metrics(y[keep], p[keep])})


def main() -> int:
    from mrfbcsp_loso import ManualCSP
    bnci = load_npz(HERE / "bnci2014001.npz")
    n_times = bnci["X"].shape[2]
    rows, resets = [], []
    with threadpool_limits(limits=8):
        # LOSO within BNCI2014-001
        t0 = time.perf_counter()
        b = pick(bnci, bnci["ch_names"], n_times)
        for s in np.unique(b["subjects"]):
            tr = b["subjects"] != s
            model = Model(ManualCSP, b["X"][tr], b["y"][tr], b["subjects"][tr], 8, aligned=True)
            m = ~tr
            evaluate({"scope": "loso", "source": "bnci2014001", "target": "bnci2014001", "subject": int(s),
                      "n_channels": len(bnci["ch_names"])}, model, b["X"][m], b["y"][m], b["sessions"][m], rows, resets)
        print(f"[bnci] loso done in {(time.perf_counter()-t0)/60:.1f} min", flush=True)
        # cross-dataset: Lee2019 / Cho2017 -> BNCI2014-001 on shared channels
        for src_name in ("lee2019", "cho2017"):
            t0 = time.perf_counter()
            src = load_npz(ROOT / "MI_test" / "preprocessed_sfreq100" / f"{src_name}.npz")
            shared = [c for c in bnci["ch_names"] if c in src["ch_names"]]
            s_data, t_data = pick(src, shared, n_times), pick(bnci, shared, n_times)
            model = Model(ManualCSP, s_data["X"], s_data["y"], s_data["subjects"], 8, aligned=True)
            for s in np.unique(t_data["subjects"]):
                m = t_data["subjects"] == s
                evaluate({"scope": "cross", "source": src_name, "target": "bnci2014001", "subject": int(s),
                          "n_channels": len(shared)}, model, t_data["X"][m], t_data["y"][m], t_data["sessions"][m],
                         rows, resets)
            print(f"[bnci] cross {src_name}->bnci ({len(shared)} ch: {shared}) done in "
                  f"{(time.perf_counter()-t0)/60:.1f} min", flush=True)
    write_rows(HERE / "bnci_subject_results.csv", rows)
    write_rows(HERE / "bnci_resets.csv", resets)
    print("[done]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
