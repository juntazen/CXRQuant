#!/usr/bin/env python3
"""Repeated, randomised efficiency measurement for the retained checkpoints (revision R1.8).

Protocol (one job = one model x precision x repetition):
  * every job runs in a fresh Python process, so CUDA context creation, kernel loading and
    BitsAndBytes first-use costs never leak between jobs;
  * job order is randomised per device with a recorded seed, and every model x precision is
    measured on both MIG devices (repetitions are split evenly across devices);
  * before timing: one full warm-up pass over the first batch (8 reports, 64 new tokens) that is
    discarded, then torch.cuda.synchronize(), torch.cuda.empty_cache() and
    torch.cuda.reset_peak_memory_stats();
  * each batch is timed with CUDA events and wall clock between synchronize() calls;
  * decoding, batching and prompts are identical to the benchmark, and the generated text is
    compared with the archived predictions (determinism check).

Usage (two MIG devices, 6 repetitions -> 3 per device):
  python scripts/efficiency_repeat.py plan --checkpoints <CHECKPOINT_ROOT>/full_seed42_corrected_20260911/checkpoints \
      --archive runs/full_seed42_corrected_20260911 --devices MIG-a,MIG-b --reps 6 --out ../results/efficiency_repeat
  python scripts/efficiency_repeat.py run --out ../results/efficiency_repeat      # launches one worker per device
  python scripts/efficiency_repeat.py summarise --out ../results/efficiency_repeat
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

MODELS = ["distilgpt2", "gpt2", "gpt2-medium", "biogpt", "pythia-1b", "biogpt-large"]
CONFIGS = ["FP32", "FP16", "INT8-BnB", "NF4-BnB"]
ROOT = Path(__file__).resolve().parents[1]


def plan(a):
    devices = a.devices.split(",")
    jobs = {d: [] for d in devices}
    for rep in range(a.reps):
        for m in MODELS:
            for c in CONFIGS:
                d = devices[(rep + MODELS.index(m)) % len(devices)]
                jobs[d].append({"model": m, "config": c, "rep": rep})
    rng = random.Random(a.seed)
    for d in devices:
        rng.shuffle(jobs[d])
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "plan.json").write_text(json.dumps({"seed": a.seed, "checkpoints": str(a.checkpoints),
                                                 "archive": str(a.archive), "jobs": jobs}, indent=1))
    print({d: len(v) for d, v in jobs.items()})


def job(a):
    import torch
    from transformers import AutoTokenizer

    sys.path.insert(0, str(ROOT / "src"))
    from cxrquant.pipeline import load_precision
    from cxrquant.training_data import encode_prompt

    archive = Path(a.archive)
    cfg = json.loads((archive / "config.json").read_text())
    rec = json.loads((archive / "models" / a.model / f"{a.config}.json").read_text())
    inputs = [s["input"] for s in rec["samples"]]
    findings = [x[len("FINDINGS: "):-len(" IMPRESSION:")] for x in inputs]
    path = Path(a.checkpoints) / a.model / "final"
    tok = AutoTokenizer.from_pretrained(path)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"
    enc = [encode_prompt(tok, f, cfg["generation_input_limit"])[0] for f in findings]
    torch.cuda.synchronize()
    t_load = time.perf_counter()
    model, _ = load_precision(path, a.config)
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t_load
    dec = dict(do_sample=cfg["do_sample"], num_beams=cfg["num_beams"], repetition_penalty=cfg["repetition_penalty"],
               pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    bs = cfg["generation_batch_size"]

    def batch(items):
        return tok.pad({"input_ids": items}, padding=True, return_tensors="pt").to("cuda")

    with torch.no_grad():  # full-size warm-up, discarded
        model.generate(**batch(enc[:bs]), max_new_tokens=cfg["max_new_tokens"], **dec)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    preds, tokens, ev_ms, wall_s = [], 0, 0.0, 0.0
    for i in range(0, len(enc), bs):
        b = batch(enc[i:i + bs])
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        w0 = time.perf_counter()
        start.record()
        with torch.no_grad():
            out = model.generate(**b, max_new_tokens=cfg["max_new_tokens"], **dec)
        end.record()
        torch.cuda.synchronize()
        wall_s += time.perf_counter() - w0
        ev_ms += start.elapsed_time(end)
        new = out[:, b["input_ids"].shape[1]:].tolist()
        for ids, text in zip(new, tok.batch_decode(new, skip_special_tokens=True)):
            tokens += ids.index(tok.eos_token_id) + 1 if tok.eos_token_id in ids else len(ids)
            clean = text.strip()
            for p in ("IMPRESSION:", "Impression:", "impression:"):
                if clean.startswith(p):
                    clean = clean[len(p):].strip()
                    break
            preds.append(clean)
    n = len(enc)
    res = {"model": a.model, "config": a.config, "rep": a.rep, "device": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "load_s": load_s, "event_ms_per_report": ev_ms / n, "wall_ms_per_report": 1000 * wall_s / n,
           "ms_per_token": ev_ms / tokens, "generated_tokens": tokens,
           "peak_alloc_mib": torch.cuda.max_memory_allocated() / 2**20,
           "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
           "predictions_identical_to_archive": preds == rec["predictions"],
           "n_prediction_mismatch": sum(x != y for x, y in zip(preds, rec["predictions"])),
           "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    print(json.dumps(res))


def run(a):
    p = json.loads((a.out / "plan.json").read_text())
    procs = []
    for dev, jobs in p["jobs"].items():
        log = (a.out / f"worker_{dev[:12]}.jsonl").open("a")
        cmd = [sys.executable, __file__, "worker", "--out", str(a.out), "--device", dev]
        procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.DEVNULL))
    for pr in procs:
        pr.wait()
    print([pr.returncode for pr in procs])


def worker(a):
    p = json.loads((a.out / "plan.json").read_text())
    done_file = a.out / f"worker_{a.device[:12]}.jsonl"
    done = set()
    if done_file.exists():
        for line in done_file.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["model"], r["config"], r["rep"]))
            except (ValueError, KeyError):
                pass
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.device)
    for j in p["jobs"][a.device]:
        if (j["model"], j["config"], j["rep"]) in done:
            continue
        cmd = [sys.executable, __file__, "job", "--checkpoints", p["checkpoints"], "--archive", p["archive"],
               "--model", j["model"], "--config", j["config"], "--rep", str(j["rep"])]
        out = subprocess.run(cmd, env=env, capture_output=True, text=True)
        line = [x for x in out.stdout.splitlines() if x.startswith("{")]
        print(line[-1] if line else json.dumps({**j, "error": out.stderr[-2000:]}), flush=True)


def summarise(a):
    import numpy as np

    rows = []
    for f in a.out.glob("worker_*.jsonl"):
        rows += [json.loads(x) for x in f.read_text().splitlines() if x.startswith("{") and '"error"' not in x]
    out = []
    for m in MODELS:
        base = [r["ms_per_token"] for r in rows if r["model"] == m and r["config"] == "FP32"]
        for c in CONFIGS:
            g = [r for r in rows if r["model"] == m and r["config"] == c]
            if not g:
                continue
            t = np.array([r["event_ms_per_report"] for r in g])
            k = np.array([r["ms_per_token"] for r in g])
            half = 1.96 * t.std(ddof=1) / np.sqrt(len(t)) if len(t) > 1 else float("nan")
            out.append({"model": m, "config": c, "n": len(g), "devices": sorted({r["device"] for r in g}),
                        "ms_report_mean": round(float(t.mean()), 1), "ms_report_ci95_half": round(float(half), 1),
                        "ms_report_cv_pct": round(float(100 * t.std(ddof=1) / t.mean()), 1) if len(t) > 1 else None,
                        "ms_token_mean": round(float(k.mean()), 2),
                        "ratio_vs_fp32_ms_token": round(float(k.mean() / np.mean(base)), 2) if base else None,
                        "peak_mib_mean": round(float(np.mean([r["peak_alloc_mib"] for r in g])), 0),
                        "all_predictions_identical": all(r["predictions_identical_to_archive"] for r in g)})
    (a.out / "summary.json").write_text(json.dumps(out, indent=1))
    for r in out:
        print(r)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--checkpoints", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--devices", required=True)
    p.add_argument("--reps", type=int, default=6)
    p.add_argument("--seed", type=int, default=20260926)
    p.add_argument("--out", type=Path, required=True)
    for name in ("run", "summarise"):
        s = sub.add_parser(name)
        s.add_argument("--out", type=Path, required=True)
    w = sub.add_parser("worker")
    w.add_argument("--out", type=Path, required=True)
    w.add_argument("--device", required=True)
    j = sub.add_parser("job")
    for k in ("checkpoints", "archive", "model", "config"):
        j.add_argument(f"--{k}", required=True)
    j.add_argument("--rep", type=int, required=True)
    a = ap.parse_args()
    {"plan": plan, "run": run, "summarise": summarise, "worker": worker, "job": job}[a.cmd](a)


if __name__ == "__main__":
    main()
