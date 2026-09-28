"""Download only S21-S62 Session_1 .mat files of Stieger2021 from Figshare (article 13123148), md5-verified.
Resumable: files with a matching md5 are skipped. MOABB's own loader fetches every session of a subject."""

import hashlib
import time
from pathlib import Path

import requests

DEST = Path(r"D:\mne_data\stieger2021_session1")
WANT = {f"S{s}_Session_1.mat" for s in range(21, 63)}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    files, page = [], 1
    while True:
        r = requests.get("https://api.figshare.com/v2/articles/13123148/files", params={"page": page, "page_size": 100}, timeout=60)
        r.raise_for_status(); batch = r.json()
        if not batch:
            break
        files += batch; page += 1
    todo = sorted((f for f in files if f["name"] in WANT), key=lambda f: int(f["name"].split("_")[0][1:]))
    print(f"figshare files: {len(files)}; session-1 targets found: {len(todo)} of {len(WANT)}", flush=True)
    for f in todo:
        out = DEST / f["name"]
        if out.exists() and out.stat().st_size == f["size"] and md5(out) == f["computed_md5"]:
            continue
        for attempt in range(6):
            try:
                tmp = out.with_suffix(".part")
                with requests.get(f["download_url"], stream=True, timeout=120) as r:
                    r.raise_for_status()
                    with open(tmp, "wb") as fh:
                        for chunk in r.iter_content(1 << 22):
                            fh.write(chunk)
                if md5(tmp) != f["computed_md5"]:
                    raise IOError("md5 mismatch")
                tmp.replace(out); print(f"  {f['name']} ok ({f['size'] / 1e6:.0f} MB)", flush=True); break
            except Exception as err:
                print(f"  {f['name']} attempt {attempt + 1} failed: {err}", flush=True); time.sleep(30 * (attempt + 1))
    print("download done", flush=True)


if __name__ == "__main__":
    main()
