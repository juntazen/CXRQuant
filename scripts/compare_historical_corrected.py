#!/usr/bin/env python3
"""Paired report-bootstrap comparison of historical and corrected predictions."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from cxrquant.clinical_safety.fact_extractor import (
    _PATHOLOGY_CATEGORIES,
    extract_labels,
    labels_to_binary,
)
from cxrquant.runio import atomic_json


def matrix(texts):
    return np.asarray([
        [labels_to_binary(extract_labels(text))[cat] for cat in _PATHOLOGY_CATEGORIES]
        for text in texts
    ], dtype=np.int8)


def ce_f1(pred, ref, indices):
    p, r = pred[indices], ref[indices]
    tp = np.sum((p == 1) & (r == 1), axis=(1, 2))
    fp = np.sum((p == 1) & (r == 0), axis=(1, 2))
    fn = np.sum((p == 0) & (r == 1), axis=(1, 2))
    denominator = 2 * tp + fp + fn
    return np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator > 0)


def fnf(quant, base, indices):
    q, b = quant[indices], base[indices]
    lost = np.sum((b == 1) & (q == 0), axis=(1, 2))
    present = np.sum(b == 1, axis=(1, 2))
    return np.divide(lost, present, out=np.full_like(lost, np.nan, dtype=float), where=present > 0)


def interval(values):
    finite = values[np.isfinite(values)]
    if not len(finite):
        return None, None
    return tuple(float(x) for x in np.percentile(finite, [2.5, 97.5]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-run", type=Path, required=True)
    parser.add_argument("--corrected-run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    old = json.loads((args.historical_run / "multimodel_benchmark.json").read_text())
    new = json.loads((args.corrected_run / "multimodel_benchmark.json").read_text())
    old_manifest = json.loads((args.historical_run / "manifest.json").read_text())
    new_manifest = json.loads((args.corrected_run / "manifest.json").read_text())
    if old_manifest["sample_ids"] != new_manifest["sample_ids"]:
        raise ValueError("Historical and corrected report IDs are not paired")
    if old_manifest["references"] != new_manifest["references"]:
        raise ValueError("Historical and corrected reference order differs")
    models = new_manifest["models"]
    configs = new_manifest["configs"]
    old_models = {m["model"]: m for m in old["models"]}
    new_models = {m["model"]: m for m in new["models"]}
    n = len(new_manifest["sample_ids"])
    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, n, size=(args.resamples, n))
    references = matrix(new_manifest["references"])
    cache = {}
    for version, mapping in (("historical", old_models), ("corrected", new_models)):
        for model in models:
            for config in configs:
                cache[(version, model, config)] = matrix(mapping[model]["predictions"][config])

    rows = []
    ce_boot = {}
    fnf_boot = {}
    for model in models:
        for config in configs:
            old_values = ce_f1(cache[("historical", model, config)], references, indices)
            new_values = ce_f1(cache[("corrected", model, config)], references, indices)
            delta = new_values - old_values
            low, high = interval(delta)
            old_point = ce_f1(cache[("historical", model, config)], references, np.arange(n)[None, :])[0]
            new_point = ce_f1(cache[("corrected", model, config)], references, np.arange(n)[None, :])[0]
            rows.append({
                "model": model, "config": config, "metric": "ce_micro_f1",
                "historical": old_point, "corrected": new_point,
                "delta_new_minus_old": new_point - old_point,
                "paired_ci95_low": low, "paired_ci95_high": high,
                "direction": "improved" if low > 0 else "worse" if high < 0 else "inconclusive",
                "n_reports": n, "resamples": args.resamples, "seed": args.seed,
            })
            ce_boot[(model, config)] = delta
            if config != "FP32":
                old_f = fnf(cache[("historical", model, config)], cache[("historical", model, "FP32")], indices)
                new_f = fnf(cache[("corrected", model, config)], cache[("corrected", model, "FP32")], indices)
                fdelta = new_f - old_f
                low, high = interval(fdelta)
                old_point = fnf(cache[("historical", model, config)], cache[("historical", model, "FP32")], np.arange(n)[None, :])[0]
                new_point = fnf(cache[("corrected", model, config)], cache[("corrected", model, "FP32")], np.arange(n)[None, :])[0]
                rows.append({
                    "model": model, "config": config, "metric": "FNF_rate",
                    "historical": old_point, "corrected": new_point,
                    "delta_new_minus_old": new_point - old_point,
                    "paired_ci95_low": low, "paired_ci95_high": high,
                    "direction": "improved" if high < 0 else "worse" if low > 0 else "inconclusive",
                    "n_reports": n, "resamples": args.resamples, "seed": args.seed,
                })
                fnf_boot[(model, config)] = fdelta

    aggregates = []
    for config in configs:
        values = np.mean([ce_boot[(model, config)] for model in models], axis=0)
        low, high = interval(values)
        points = [r for r in rows if r["config"] == config and r["metric"] == "ce_micro_f1"]
        aggregates.append({
            "scope": f"mean_across_6_models/{config}", "metric": "ce_micro_f1",
            "delta_new_minus_old": float(np.mean([r["delta_new_minus_old"] for r in points])),
            "paired_ci95_low": low, "paired_ci95_high": high,
            "direction": "improved" if low > 0 else "worse" if high < 0 else "inconclusive",
        })
    all_ce = np.mean(list(ce_boot.values()), axis=0)
    low, high = interval(all_ce)
    aggregates.append({
        "scope": "mean_across_24_model_configs", "metric": "ce_micro_f1",
        "delta_new_minus_old": float(np.mean([r["delta_new_minus_old"] for r in rows if r["metric"] == "ce_micro_f1"])),
        "paired_ci95_low": low, "paired_ci95_high": high,
        "direction": "improved" if low > 0 else "worse" if high < 0 else "inconclusive",
    })
    for config in configs[1:]:
        values = np.nanmean([fnf_boot[(model, config)] for model in models], axis=0)
        low, high = interval(values)
        points = [r for r in rows if r["config"] == config and r["metric"] == "FNF_rate"]
        aggregates.append({
            "scope": f"mean_across_6_models/{config}", "metric": "FNF_rate",
            "delta_new_minus_old": float(np.mean([r["delta_new_minus_old"] for r in points])),
            "paired_ci95_low": low, "paired_ci95_high": high,
            "direction": "improved" if high < 0 else "worse" if low > 0 else "inconclusive",
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "OLD_VS_NEW_PAIRED_BOOTSTRAP.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "status": "complete",
        "pairing": {"same_ordered_report_ids": True, "same_references": True, "n_reports": n},
        "resamples": args.resamples,
        "seed": args.seed,
        "interpretation": "CE-F1 higher is better against references; FNF lower is better agreement with each run's FP32 output.",
        "per_model_config": rows,
        "aggregates": aggregates,
        "limitations": [
            "Historical and corrected runs differ in padding loss and explicit software recipe.",
            "Only one corrected training seed; bootstrap quantifies report sampling, not training variance.",
            "CXRQuant reference labels are rule-based and not radiologist adjudication.",
        ],
    }
    atomic_json(args.out_dir / "OLD_VS_NEW_PAIRED_BOOTSTRAP.json", result)
    print(json.dumps({"pairing": result["pairing"], "aggregates": aggregates}, indent=2))


if __name__ == "__main__":
    main()
