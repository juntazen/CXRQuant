#!/usr/bin/env python3
"""Tabulate measured generation efficiency for every model x precision of one run.

Reads the ``efficiency`` and ``backend`` blocks written by the GPU worker
(synchronised timing after warm-up, peak allocated/reserved CUDA memory) and the
FP32 checkpoint size from ``training.json``. Nothing is re-measured: values are
exactly those recorded during generation. Ratios are relative to the same
checkpoint under FP32. Per-token latency is reported alongside per-report
latency because reduced-precision outputs differ in length from FP32 outputs.

Outputs new files only: ``efficiency_summary.json`` and
``tables/efficiency_summary.csv``.
"""
import argparse
import csv
import json
from pathlib import Path
from statistics import pstdev

from cxrquant.runio import atomic_json, sha256_file

MIB = 1024 * 1024


def summarise(run, manifest):
    rows = []
    for model in manifest["models"]:
        training = json.loads((run / "models" / model / "training.json").read_text())
        base = None
        for cfg in manifest["configs"]:
            rec = json.loads((run / "models" / model / (cfg + ".json")).read_text())
            eff, backend = rec["efficiency"], rec["backend"]
            n_reports = sum(b["n_reports"] for b in eff["batch_timings"])
            if n_reports != len(manifest["sample_ids"]):
                raise ValueError(f"timed reports do not cover the test set: {model}/{cfg}")
            per_report_batch_ms = [1000 * b["latency_s"] / b["n_reports"] for b in eff["batch_timings"]]
            row = {
                "model": model,
                "config": cfg,
                "latency_ms_per_report": round(eff["amortized_latency_ms"], 2),
                "batch_latency_ms_per_report_sd": round(pstdev(per_report_batch_ms), 2),
                "reports_per_s": round(eff["reports_per_second"], 2),
                "generated_tokens": eff["generated_tokens"],
                "tokens_per_s": round(eff["tokens_per_second"], 1),
                "ms_per_generated_token": round(1000 * eff["total_generation_s"] / eff["generated_tokens"], 3),
                "peak_allocated_MiB": round(eff["peak_allocated_bytes"] / MIB, 1),
                "peak_reserved_MiB": round(eff["peak_reserved_bytes"] / MIB, 1),
                "quantized_linear_modules": len(backend.get("quantized_modules") or {}),
                "fp32_checkpoint_weight_MiB": round(training["checkpoint_weight_bytes"] / MIB, 1),
            }
            if cfg == "FP32":
                base = row
            row["peak_allocated_reduction_pct"] = round(
                100 * (1 - row["peak_allocated_MiB"] / base["peak_allocated_MiB"]), 1
            )
            row["latency_ratio_vs_fp32"] = round(
                row["latency_ms_per_report"] / base["latency_ms_per_report"], 2
            )
            row["per_token_latency_ratio_vs_fp32"] = round(
                row["ms_per_generated_token"] / base["ms_per_generated_token"], 2
            )
            rows.append(row)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    run = args.run_dir.resolve()
    out_json = run / "efficiency_summary.json"
    out_csv = run / "tables" / "efficiency_summary.csv"
    if not args.overwrite and (out_json.exists() or out_csv.exists()):
        raise SystemExit("efficiency output exists; pass --overwrite to regenerate")
    manifest = json.loads((run / "manifest.json").read_text())
    cfg = json.loads((run / "config.json").read_text())
    status = json.loads((run / "status.json").read_text())
    rows = summarise(run, manifest)
    by_cfg = {}
    for c in manifest["configs"]:
        if c == "FP32":
            continue
        sel = [r for r in rows if r["config"] == c]
        by_cfg[c] = {
            "peak_allocated_reduction_pct_range": [
                min(r["peak_allocated_reduction_pct"] for r in sel),
                max(r["peak_allocated_reduction_pct"] for r in sel),
            ],
            "latency_ratio_range": [
                min(r["latency_ratio_vs_fp32"] for r in sel),
                max(r["latency_ratio_vs_fp32"] for r in sel),
            ],
            "per_token_latency_ratio_range": [
                min(r["per_token_latency_ratio_vs_fp32"] for r in sel),
                max(r["per_token_latency_ratio_vs_fp32"] for r in sel),
            ],
        }
    slowest = {}
    for model in manifest["models"]:
        sel = [r for r in rows if r["model"] == model]
        slowest[model] = max(sel, key=lambda r: r["latency_ms_per_report"])["config"]
    out = {
        "analysis": "recorded generation efficiency (no re-measurement)",
        "protocol": {
            "generation_batch_size": cfg["generation_batch_size"],
            "max_new_tokens": cfg["max_new_tokens"],
            "decoding": {"do_sample": cfg["do_sample"], "num_beams": cfg["num_beams"],
                         "repetition_penalty": cfg["repetition_penalty"]},
            "warmup_reports": cfg["warmup_reports"],
            "warmup_new_tokens": cfg["warmup_new_tokens"],
            "timing": "torch.cuda.synchronize around each batch generate call; tokenisation excluded",
            "memory": "torch.cuda.max_memory_allocated/reserved after warm-up reset",
            "devices": [j["device"] for j in status.get("jobs", [])],
            "device_type": "NVIDIA A100 MIG 7g.40gb, one process per MIG instance",
        },
        "rows": rows,
        "summary_by_config": by_cfg,
        "slowest_config_per_model": slowest,
        "provenance": {
            "run_id": manifest["run_id"],
            "manifest_hash": sha256_file(run / "manifest.json"),
            "script_hash": sha256_file(__file__),
        },
    }
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(out_json, out)
    print(json.dumps({k: out[k] for k in ("summary_by_config", "slowest_config_per_model")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
