"""Apply the adopted Cho2017 amplitude factor and select the analysis array version.

The factor 0.03125 is an operational representation choice, NOT a verified physical-unit correction.
An informal acquisition account is not a calibration/export record. This tool does not establish a
BioSemi, BCI2000 or MOABB conversion error. See docs/CHO_AMPLITUDE_PROVENANCE.md. Historical filenames
and the corrected/uncorrected CLI choices are preserved for compatibility, not as calibration claims.
Recorded array hashes identify the two study inputs; they do not certify their physical units.

All analyses read MI_test/preprocessed_sfreq100/cho2017.npz. This tool keeps both versions next to it:
  cho2017_moabb_uncorrected.npz   original project-loader array (MNE Epochs path)
  cho2017_unitcorrected.npz       x 0.03125
and copies the selected one to cho2017.npz. Development uses the scaled version; the original
pre-specified Dreyer2023 analysis used the original version. --npz_dir always exports scaled Cho.

Usage (from the repository root):
  python tools/correct_cho2017_units.py                     # create both versions, select scaled input
  python tools/correct_cho2017_units.py --select uncorrected
  python tools/correct_cho2017_units.py --npz_dir DIR       # also build a folder for run_cspnet_generic.py (MI_NPZ_DIR)
The EEG arrays are checked against the SHA-256 values in data_manifest/SHA256SUMS (file hashes of .npz archives can
differ between NumPy versions and compression settings; array hashes do not).
"""
import argparse
import hashlib
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D100 = ROOT / "MI_test" / "preprocessed_sfreq100"
FACTOR = np.float32(0.03125)
X_SHA256 = {"uncorrected": "188f759a4ddfb49413afa2dd1b8815942926a8041000ceff802c1f7f4880ca19",
            "corrected": "6fbc8deaa2aaedae4b6f57a7938251c0bf825304ef7655e71fd4607987b77d50"}
OTHERS = {"lee2019.npz": D100 / "lee2019.npz",
          "bnci2014001.npz": ROOT / "analysis_outputs" / "cmpb_bdea_20260924" / "bnci2014001.npz",
          "physionetmi.npz": ROOT / "analysis_outputs" / "cmpb_bdea_20260924" / "physionetmi.npz"}


def x_hash(path):
    x = np.load(path, allow_pickle=True)["X"]
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def check(path, version):
    h = x_hash(path)
    ok = h == X_SHA256[version]
    print(f"{path.name}: X sha256 {h[:16]}... {'matches' if ok else 'DOES NOT MATCH'} the {version} reference")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--select", choices=["corrected", "uncorrected"], default="corrected")
    ap.add_argument("--npz_dir", type=Path, help="also write the four development files (Cho corrected) to this folder")
    args = ap.parse_args()

    cur, unc, cor = D100 / "cho2017.npz", D100 / "cho2017_moabb_uncorrected.npz", D100 / "cho2017_unitcorrected.npz"
    if not unc.exists():
        if not cur.exists():
            raise SystemExit(f"{cur} not found; run MI_test/MI_loso_project/preprocess_moabb_sfreq100.py first")
        if x_hash(cur) != X_SHA256["uncorrected"]:
            raise SystemExit(f"{cur} is not the MOABB output listed in the manifest; refusing to treat it as uncorrected")
        shutil.copyfile(cur, unc)
    if not cor.exists():
        d = dict(np.load(unc, allow_pickle=True))
        d["X"] = (d["X"] * FACTOR).astype(np.float32)
        np.savez(cor, **d)
    ok = check(unc, "uncorrected") & check(cor, "corrected")
    shutil.copyfile(cor if args.select == "corrected" else unc, cur)
    print(f"cho2017.npz <- {args.select}")
    if args.npz_dir:
        args.npz_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cor, args.npz_dir / "cho2017.npz")
        for name, src in OTHERS.items():
            shutil.copyfile(src, args.npz_dir / name)
        print(f"wrote the four development files (Cho2017 corrected) to {args.npz_dir}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
