"""Interruption-safe queue for the neural (CSPNet, EEGNet) runs used in the paper.

Run with the neural environment from the repository root:
  python tools/correct_cho2017_units.py --npz_dir MI_test/preprocessed_generic_unitcorrected
  QUEUE_PARALLEL=2 python analysis_outputs/cmpb_neural_decomposition/run_neural_queue.py
then analyze with (CPU environment)
  python analysis_outputs/cmpb_neural_decomposition/analyze_neural_multiseed.py --chofix
  python analysis_outputs/cmpb_neural_decomposition/analyze_neural_loso21.py

- All runs read the four development files from MI_NPZ_DIR (default MI_test/preprocessed_generic_unitcorrected,
  Cho2017 unit-corrected). Run ids ending in "_chofix" mark conditions whose results depend on Cho2017's absolute scale
  (no alignment and absolute floor); relative-floor runs do not depend on it (scale equivariance).
- A finished job leaves done/<job_id>; rerunning skips it. An interrupted job is restarted and run_cspnet_generic.py
  skips the (held-out, subject) pairs already in its CSV.
- queue.lock prevents two drivers from writing the same CSVs. Delete it only if no driver is running.
Results: MI_test/results/generic_<job_id>_<mode>.csv; logs: logs/<job_id>.log.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CODE = ROOT / "MI_test" / "MI_loso_project"
NPZ = Path(os.environ.get("MI_NPZ_DIR", ROOT / "MI_test" / "preprocessed_generic_unitcorrected"))
PARALLEL = int(os.environ.get("QUEUE_PARALLEL", "2"))
DONE, LOGS, LOCK = HERE / "done", HERE / "logs", HERE / "queue.lock"
FOUR = ["cho2017", "lee2019", "bnci2014001", "physionetmi"]
SEEDS = ["2026", "1", "2", "3", "4"]


def jid(name, seed):
    return name + ("" if seed == "2026" else f"_s{seed}")


# (job id, EA floor, runner arguments, seed)
JOBS = []
# CSPNet LODO over the four development datasets, five seeds (Table 8, Table 10, Fig. 3b, Fig. 5)
for seed in SEEDS:
    for name, floor, extra in (("gen_lodo4_relative", "relative", ["--align", "subject"]),
                               ("gen_lodo4_none_chofix", "relative", ["--align", "none"]),
                               ("gen_lodo4_legacy_chofix", "legacy", ["--align", "subject"]),
                               ("gen_lodo4_dsea_relative", "relative", ["--align", "dataset_subject"]),
                               ("gen_lodo4_adabn_relative", "relative", ["--align", "subject", "--adabn"])):
        JOBS.append((jid(name, seed), floor, ["--mode", "lodo", "--datasets", *FOUR, *extra], seed))
JOBS.append(("gen_lodo4_dsea_legacy_chofix", "legacy", ["--mode", "lodo", "--datasets", *FOUR, "--align", "dataset_subject"],
             "2026"))
# CSPNet LOSO within each dataset on the 21 shared channels, seed 2026 (Table 8)
for ds in FOUR:
    others = [d for d in FOUR if d != ds]
    for cond, floor, align in (("relative", "relative", "subject"), ("none", "relative", "none"),
                               ("legacy_chofix" if ds == "cho2017" else "legacy", "legacy", "subject")):
        JOBS.append((f"gen_loso21_{ds}_{cond}", floor, ["--mode", "loso", "--datasets", ds, "--channels", "shared",
                                                        "--shared_with", *others, "--align", align], "2026"))
# EEGNet LODO, seed 2026 (numerical artifact in a second architecture)
for name, floor in (("gen_lodo4_eegnet_relative", "relative"), ("gen_lodo4_eegnet_legacy_chofix", "legacy")):
    JOBS.append((name, floor, ["--mode", "lodo", "--datasets", *FOUR, "--align", "subject", "--model", "eegnet"], "2026"))


def main() -> int:
    if not (NPZ / "cho2017.npz").exists():
        print(f"{NPZ} has no cho2017.npz; run tools/correct_cho2017_units.py --npz_dir {NPZ}", flush=True)
        return 1
    if LOCK.exists():
        print(f"{LOCK} exists: another driver may be running (delete the file if not)", flush=True)
        return 1
    DONE.mkdir(exist_ok=True); LOGS.mkdir(exist_ok=True)
    LOCK.write_text(str(os.getpid()))
    try:
        pending = [j for j in JOBS if not (DONE / j[0]).exists()]
        running = {}
        while pending or running:
            while pending and len(running) < PARALLEL:
                job_id, floor, args, seed = pending.pop(0)
                env = {**os.environ, "MI_EA_FLOOR": floor, "MI_SEED": seed, "MI_NPZ_DIR": str(NPZ),
                       "PYTHONIOENCODING": "utf-8"}
                log = open(LOGS / f"{job_id}.log", "a", encoding="utf-8")
                log.write(f"\n==== start {time.strftime('%F %T')} ====\n"); log.flush()
                p = subprocess.Popen([sys.executable, "run_cspnet_generic.py", *args, "--run_id", job_id], cwd=CODE,
                                     env=env, stdout=log, stderr=subprocess.STDOUT)
                running[job_id] = (p, log, time.time())
                print(f"[{time.strftime('%F %T')}] START {job_id}", flush=True)
            time.sleep(15)
            for job_id in list(running):
                p, log, t0 = running[job_id]
                if p.poll() is not None:
                    log.close(); del running[job_id]
                    if p.returncode == 0:
                        (DONE / job_id).touch()
                    print(f"[{time.strftime('%F %T')}] END {job_id} rc={p.returncode} {(time.time() - t0) / 60:.1f} min",
                          flush=True)
        print(f"[{time.strftime('%F %T')}] QUEUE DONE", flush=True)
    finally:
        LOCK.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
