#!/usr/bin/env python3
"""Fetch a reproducible NIH ChestX-ray14 subset for the confirmatory image-conditioned audit (revision R2-2).

Reads only the needed members of images_001.zip through HTTP range requests (the archive's central
directory is read remotely). Selection: one image per patient (the first follow-up), frontal view,
patients present in images_001, stratified 50/50 between 'No Finding' and at least one finding label,
shuffled with a fixed seed.

  python tools/nih_subset.py --meta ../external/nih/Data_Entry_2017_v2020.csv --n 400 --out ../external/nih
"""
import argparse
import csv
import io
import json
import random
import urllib.request
import zipfile
from pathlib import Path

URL = "https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/data/images/images_001.zip"


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url):
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.url, self.size = r.geturl(), int(r.headers["Content-Length"])
        self.pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.size, self.pos + n) - 1
        for attempt in range(10):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                break
            except Exception:
                if attempt == 9:
                    raise
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    f = HTTPRangeFile(URL)
    z = zipfile.ZipFile(io.BufferedReader(f, buffer_size=1 << 16))
    members = {Path(n).name: n for n in z.namelist() if n.endswith(".png")}
    rows = [r for r in csv.DictReader(open(a.meta)) if r["Image Index"] in members]
    first = {}
    for r in rows:
        pid = r["Patient ID"]
        if r["View Position"] in ("PA", "AP") and (pid not in first or int(r["Follow-up #"]) < int(first[pid]["Follow-up #"])):
            first[pid] = r
    pool = sorted(first.values(), key=lambda r: r["Image Index"])
    rng = random.Random(a.seed)
    normal = [r for r in pool if r["Finding Labels"] == "No Finding"]
    abnormal = [r for r in pool if r["Finding Labels"] != "No Finding"]
    rng.shuffle(normal)
    rng.shuffle(abnormal)
    chosen = abnormal[: a.n // 2] + normal[: a.n - a.n // 2]
    rng.shuffle(chosen)
    img_dir = a.out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for r in chosen:
        dst = img_dir / r["Image Index"]
        if not dst.exists():
            dst.write_bytes(z.read(members[r["Image Index"]]))
    meta = [{"image": r["Image Index"], "patient": r["Patient ID"], "view": r["View Position"],
             "labels": r["Finding Labels"].split("|"), "age": r["Patient Age"], "sex": r["Patient Gender"]} for r in chosen]
    (a.out / "subset.json").write_text(json.dumps({"source": URL, "seed": a.seed, "n": len(meta),
                                                    "patients_in_archive": len(first), "images": meta}, indent=1))
    print("images_001 members", len(members), "eligible patients", len(first), "chosen", len(meta),
          "abnormal", sum(m["labels"] != ["No Finding"] for m in meta))


if __name__ == "__main__":
    main()
