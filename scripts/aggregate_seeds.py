#!/usr/bin/env python3
"""Aggregate completed full runs that differ only in training seed.

Before any statistic is computed, every run must be complete and share the
same data hash, the same held-out sample order, the same split identifiers and
the same configuration apart from ``training_seed``/``data_seed``. Any
difference is a hard error: silently mixing held-out sets would invalidate the
between-seed estimate.

Writes ``multiseed_summary.json`` and ``tables/multiseed_summary.csv`` to a new
output directory; mean and sample standard deviation (n - 1) are reported per
model x precision. n is the number of independently trained seeds, which is a
limited estimate of training variability rather than a distributional claim.
"""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

from cxrquant.runio import atomic_json, sha256_file

METRICS = ("FNF_rate", "FPF_rate", "CFPR", "bleu_4", "ce_micro_f1")
SEED_KEYS = ("training_seed", "data_seed")
REDUCED = ("FP16", "INT8-BnB", "NF4-BnB")


def load(run):
    run = Path(run).resolve()
    status = json.loads((run / "status.json").read_text())
    if status.get("status") != "complete" or status.get("worker_exit_codes") != [0, 0]:
        raise ValueError(f"run is not complete with two successful workers: {run}")
    with (run / "tables" / "metrics.csv").open(newline="") as f:
        metrics = list(csv.DictReader(f))
    return {
        "path": run,
        "identity": json.loads((run / "identity.json").read_text()),
        "manifest": json.loads((run / "manifest.json").read_text()),
        "config": json.loads((run / "config.json").read_text()),
        "metrics": metrics,
    }


def check_comparable(runs):
    if len(runs) < 2:
        raise ValueError("at least two runs are required")
    ref = runs[0]
    stripped = {k: v for k, v in ref["config"].items() if k not in SEED_KEYS}
    seeds = []
    for r in runs:
        name = r["path"].name
        if r["identity"]["data_hash"] != ref["identity"]["data_hash"]:
            raise ValueError(f"data_hash differs: {name}")
        for key in ("sample_ids", "references", "split_ids", "models", "configs"):
            if r["manifest"].get(key) != ref["manifest"].get(key):
                raise ValueError(f"manifest {key} differs: {name}")
        if {k: v for k, v in r["config"].items() if k not in SEED_KEYS} != stripped:
            raise ValueError(f"configuration differs beyond training/data seed: {name}")
        if r["config"]["training_seed"] != r["config"]["data_seed"]:
            raise ValueError(f"training_seed and data_seed differ within run: {name}")
        seeds.append(r["config"]["training_seed"])
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate training seed")
    return seeds


def value(run, model, cfg, key):
    row = next(
        (x for x in run["metrics"] if x["model"] == model and x["config"] == cfg), None
    )
    if row is None:
        raise ValueError(f"missing metrics row {model}/{cfg} in {run['path'].name}")
    return None if row[key] == "" else float(row[key])


def aggregate(runs):
    seeds = check_comparable(runs)
    manifest = runs[0]["manifest"]
    rows = []
    for model in manifest["models"]:
        for cfg in manifest["configs"]:
            row = {"model": model, "config": cfg, "n_seeds": len(runs)}
            for key in METRICS:
                vals = [value(r, model, cfg, key) for r in runs]
                if cfg == "FP32" and key in ("FNF_rate", "FPF_rate", "CFPR"):
                    continue
                defined = [v for v in vals if v is not None]
                row[key + "_n_defined"] = len(defined)
                row[key + "_mean"] = round(mean(defined), 4) if defined else None
                row[key + "_sd"] = round(stdev(defined), 4) if len(defined) > 1 else None
                row[key + "_min"] = min(defined) if defined else None
                row[key + "_max"] = max(defined) if defined else None
            rows.append(row)
    ordering = {}
    for model in manifest["models"]:
        per_seed, non_decreasing, ties = [], [], []
        for r in runs:
            vals = [value(r, model, c, "FNF_rate") for c in REDUCED]
            defined = all(v is not None for v in vals)
            per_seed.append(defined and vals[0] < vals[1] < vals[2])
            non_decreasing.append(defined and vals[0] <= vals[1] <= vals[2])
            if defined and vals[0] <= vals[1] <= vals[2] and not vals[0] < vals[1] < vals[2]:
                ties.append({"training_seed": r["config"]["training_seed"], "FNF": dict(zip(REDUCED, vals, strict=True))})
        ordering[model] = {"seeds_with_strict_order": sum(per_seed), "seeds_non_decreasing": sum(non_decreasing),
                           "ties": ties, "n_seeds": len(runs)}
    return seeds, rows, ordering


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", required=True, help="comma-separated completed run directories")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    runs = [load(p) for p in args.run_dirs.split(",") if p.strip()]
    seeds, rows, ordering = aggregate(runs)
    out_dir = args.out_dir.resolve()
    if (out_dir / "multiseed_summary.json").exists():
        raise SystemExit("output directory already holds a summary; choose a new directory")
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("model", "config", "n_seeds"), k))
    with (out_dir / "tables" / "multiseed_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(
        out_dir / "multiseed_summary.json",
        {
            "analysis": "between-training-seed variability with fixed split and evaluation set",
            "training_seeds": seeds,
            "std": "sample standard deviation (n - 1)",
            "rows": rows,
            "fnf_ordering_FP16_lt_INT8_lt_NF4": ordering,
            "provenance": {
                "runs": [
                    {
                        "run_id": r["identity"]["run_id"],
                        "identity": r["identity"],
                        "metrics_hash": sha256_file(r["path"] / "tables" / "metrics.csv"),
                    }
                    for r in runs
                ],
                "shared_data_hash": runs[0]["identity"]["data_hash"],
                "script_hash": sha256_file(__file__),
            },
        },
    )
    print(json.dumps({"training_seeds": seeds, "ordering": ordering}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
