#!/usr/bin/env python3
"""Regenerate all precisions from retained checkpoints with a modified inference configuration.

Used for the revision's decoding-corrected rerun (EOS exempt from the repetition penalty). The
archived run supplies data, manifest and base configuration (read only); checkpoints are read
only; every output goes to --out. One worker process per MIG device, models split round-robin.

  python scripts/regenerate_checkpoints.py launch --run-dir runs/full_seed42_corrected_20260911 \
      --checkpoints <CHECKPOINT_ROOT>/full_seed42_corrected_20260911/checkpoints \
      --set repetition_penalty_exempt_eos=true --out ../results/eosfix/seed42 \
      --devices MIG-a,MIG-b
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MODELS = ["distilgpt2", "gpt2-medium", "pythia-1b", "gpt2", "biogpt", "biogpt-large"]  # worker0 / worker1 blocks


def parse_sets(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k] = json.loads(v) if v in ("true", "false") or v.replace(".", "", 1).isdigit() else v
    return out


def worker(a):
    from cxrquant.pipeline import generate, gpu_preflight, load_data
    from cxrquant.runio import atomic_json
    from cxrquant.validation import score

    run = Path(a.run_dir)
    cfg = json.loads((run / "config.json").read_text())
    cfg.update(parse_sets(a.set))
    manifest = json.loads((run / "manifest.json").read_text())
    splits = load_data(run / "data.json", cfg)
    assert [r["uid"] for r in splits["test"]] == manifest["sample_ids"]
    out = Path(a.out)
    for name in a.models.split(","):
        hardware = gpu_preflight()
        preds = {}
        for precision in cfg["configs"]:
            p = out / "models" / name / f"{precision}.json"
            if p.exists():
                preds[precision] = json.loads(p.read_text())["predictions"]
                continue
            t0 = time.time()
            texts, samples, efficiency, backend = generate(Path(a.checkpoints) / name / "final", precision, cfg,
                                                           splits["test"])
            metrics = score(texts, manifest["references"], preds.get("FP32") if precision != "FP32" else None)
            ap_ = run / "models" / name / f"{precision}.json"
            archived = json.loads(ap_.read_text())["predictions"] if ap_.exists() else []
            atomic_json(p, {"status": "complete", "config": precision, "overrides": parse_sets(a.set),
                            "device": os.environ.get("CUDA_VISIBLE_DEVICES"), "hardware": hardware,
                            "predictions": texts, "samples": samples, "efficiency": efficiency, "backend": backend,
                            "n_identical_to_archive": sum(x == y for x, y in zip(texts, archived)),
                            "wall_s": time.time() - t0, **metrics})
            preds[precision] = texts
            print(name, precision, "FNF", metrics.get("FNF_rate"), "BLEU4", metrics.get("bleu_4"), flush=True)


def launch(a):
    devices = a.devices.split(",")
    groups = {d: [] for d in devices}
    models = a.models.split(",") if a.models else MODELS
    for i, m in enumerate(models):
        groups[devices[(i // 3) % len(devices)] if len(devices) > 1 else devices[0]].append(m)
    groups = {d: ms for d, ms in groups.items() if ms}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "launch.json").write_text(json.dumps({"groups": groups, "set": a.set, "run_dir": a.run_dir,
                                                         "checkpoints": a.checkpoints}, indent=1))
    procs = []
    for d, ms in groups.items():
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=d)
        cmd = [sys.executable, __file__, "worker", "--run-dir", a.run_dir, "--checkpoints", a.checkpoints,
               "--out", a.out, "--models", ",".join(ms)] + sum([["--set", s] for s in a.set or []], [])
        log = open(Path(a.out) / f"worker_{d[:12]}.log", "a")
        procs.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT))
    codes = [p.wait() for p in procs]
    print("exit", codes)
    sys.exit(max(codes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["launch", "worker"])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoints", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append")
    ap.add_argument("--devices", default="")
    ap.add_argument("--models", default="")
    a = ap.parse_args()
    (launch if a.cmd == "launch" else worker)(a)


if __name__ == "__main__":
    main()
