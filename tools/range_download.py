#!/usr/bin/env python3
"""Parallel HTTP range download with retries (the network proxy throttles single long transfers)."""
import concurrent.futures as cf, hashlib, os, sys, time, urllib.request

def size_of(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"]), r.geturl()

def fetch(url, start, end, path, tries=20):
    for t in range(tries):
        try:
            if os.path.exists(path) and os.path.getsize(path) == end - start + 1:
                return
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b: break
                    f.write(b)
            if os.path.getsize(path) == end - start + 1:
                return
        except Exception as e:
            time.sleep(3 + t)
    raise RuntimeError(f"chunk {start}-{end} failed")

def main(url, out, sha=None, chunk=16 << 20, workers=16):
    n, final = size_of(url)
    parts = [(i, min(i + chunk, n) - 1) for i in range(0, n, chunk)]
    tmp = out + ".parts"; os.makedirs(tmp, exist_ok=True)
    with cf.ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda p: fetch(final, p[0], p[1], f"{tmp}/{p[0]:012d}"), parts))
    h = hashlib.sha256()
    with open(out, "wb") as f:
        for s, _ in parts:
            b = open(f"{tmp}/{s:012d}", "rb").read(); f.write(b); h.update(b)
    if sha and h.hexdigest() != sha:
        raise SystemExit(f"sha mismatch {out}")
    for s, _ in parts: os.remove(f"{tmp}/{s:012d}")
    os.rmdir(tmp)
    print("ok", out, n, h.hexdigest())

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
