#!/usr/bin/env python3
"""Create manuscript-ready audit tables from a validated corrected GPU run."""

import argparse
import csv
import json
from pathlib import Path

from cxrquant.runio import atomic_json
from cxrquant.validation import validate_benchmark


METRICS = (
    "bleu_1",
    "bleu_4",
    "rouge_1",
    "rouge_2",
    "rouge_l",
    "ce_micro_f1",
    "ce_macro_f1",
    "FNF_rate",
    "FPF_rate",
    "CFPR",
    "CFFR",
)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def config_map(benchmark):
    return {
        (model["model"], config["config"]): config
        for model in benchmark["models"]
        for config in model["configs"]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--archive-replay", type=Path, required=True)
    parser.add_argument("--corrected-run", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    args = parser.parse_args()

    old = json.loads(args.historical.read_text())
    replay = json.loads((args.archive_replay / "multimodel_benchmark.json").read_text())
    new = json.loads((args.corrected_run / "multimodel_benchmark.json").read_text())
    manifest = json.loads((args.corrected_run / "manifest.json").read_text())
    validation = validate_benchmark(new, manifest)
    old_map, replay_map, new_map = map(config_map, (old, replay, new))

    comparisons = []
    for key in new_map:
        model, config = key
        for metric in METRICS:
            old_value = old_map[key].get(metric)
            replay_value = replay_map[key].get(metric)
            new_value = new_map[key].get(metric)
            if old_value is None and replay_value is None and new_value is None:
                continue
            comparisons.append(
                {
                    "model": model,
                    "config": config,
                    "metric": metric,
                    "old_value": old_value,
                    "cpu_replay_archived_predictions": replay_value,
                    "new_gpu_value": new_value,
                    "delta_new_minus_old": (
                        None
                        if old_value is None or new_value is None
                        else new_value - old_value
                    ),
                    "source": str(args.corrected_run / "multimodel_benchmark.json"),
                    "status": "CORRECTED_GPU_RUN_COMPLETE",
                }
            )
    write_csv(args.audit_dir / "OLD_VS_NEW_RESULTS.csv", comparisons)

    training = []
    efficiency = []
    fp32_nf4 = []
    for model in new["models"]:
        training.append(
            {
                "model": model["model"],
                "family": model["family"],
                "params_m": model["params_m"],
                "epochs": next(
                    spec["epochs"]
                    for spec in json.loads((args.corrected_run / "config.json").read_text())[
                        "models"
                    ]
                    if spec["name"] == model["model"]
                ),
                "train_time_min": model["train_time_s_this_invocation"] / 60,
                "token_weighted_val_loss": model["token_weighted_val_loss"],
                "val_perplexity": model["val_perplexity"],
                "valid_target_tokens": model["valid_target_tokens"],
                "checkpoint_weight_mib": model["checkpoint_weight_bytes"] / 2**20,
                "checkpoint_hash": model["checkpoint_hash"],
            }
        )
        configs = {row["config"]: row for row in model["configs"]}
        fp32, nf4 = configs["FP32"], configs["NF4-BnB"]
        bootstrap = list(
            csv.DictReader((args.corrected_run / "tables/bootstrap.csv").open())
        )
        fnf_ci = next(
            row
            for row in bootstrap
            if row["model"] == model["model"]
            and row["config"] == "NF4-BnB"
            and row["metric"] == "FNF_rate"
        )
        fp32_nf4.append(
            {
                "model": model["model"],
                "FP32_BLEU4": fp32["bleu_4"],
                "NF4_BLEU4": nf4["bleu_4"],
                "delta_BLEU4": nf4["bleu_4"] - fp32["bleu_4"],
                "FP32_CE_micro_F1": fp32["ce_micro_f1"],
                "NF4_CE_micro_F1": nf4["ce_micro_f1"],
                "delta_CE_micro_F1": nf4["ce_micro_f1"] - fp32["ce_micro_f1"],
                "NF4_FNF": nf4["FNF_rate"],
                "NF4_FNF_CI95_low": float(fnf_ci["ci95_low"]),
                "NF4_FNF_CI95_high": float(fnf_ci["ci95_high"]),
                "NF4_FPF": nf4["FPF_rate"],
                "NF4_CFPR": nf4["CFPR"],
            }
        )
        for config in model["configs"]:
            stats = config["efficiency"]
            efficiency.append(
                {
                    "model": model["model"],
                    "config": config["config"],
                    "amortized_latency_ms": stats["amortized_latency_ms"],
                    "reports_per_second": stats["reports_per_second"],
                    "tokens_per_second": stats["tokens_per_second"],
                    "peak_allocated_gib": stats["peak_allocated_bytes"] / 2**30,
                    "peak_reserved_gib": stats["peak_reserved_bytes"] / 2**30,
                    "generated_tokens": stats["generated_tokens"],
                }
            )
    write_csv(args.audit_dir / "CORRECTED_TRAINING_TABLE.csv", training)
    write_csv(args.audit_dir / "CORRECTED_FP32_NF4_TABLE.csv", fp32_nf4)
    write_csv(args.audit_dir / "CORRECTED_EFFICIENCY_TABLE.csv", efficiency)

    reduced = [
        row
        for row in new_map.values()
        if row["config"] != "FP32"
    ]
    pooled = list(csv.DictReader((args.corrected_run / "tables/mcnemar_pooled.csv").open()))
    significant = [row for row in pooled if float(row["p"]) < 0.05]
    bonferroni = 0.05 / len(pooled)
    summary = {
        "status": "complete",
        "validation": validation,
        "run_id": new["run_id"],
        "training_seed_count": 1,
        "repeat_training": False,
        "nf4_fnf_range": [
            min(row["FNF_rate"] for row in reduced if row["config"] == "NF4-BnB"),
            max(row["FNF_rate"] for row in reduced if row["config"] == "NF4-BnB"),
        ],
        "int8_fnf_range": [
            min(row["FNF_rate"] for row in reduced if row["config"] == "INT8-BnB"),
            max(row["FNF_rate"] for row in reduced if row["config"] == "INT8-BnB"),
        ],
        "fp16_fnf_range": [
            min(row["FNF_rate"] for row in reduced if row["config"] == "FP16"),
            max(row["FNF_rate"] for row in reduced if row["config"] == "FP16"),
        ],
        "best_ce_micro_f1": max(
            (
                {
                    "model": model,
                    "config": config,
                    "value": row["ce_micro_f1"],
                }
                for (model, config), row in new_map.items()
            ),
            key=lambda row: row["value"],
        ),
        "best_fp32_ce_micro_f1": max(
            (
                {"model": model, "value": row["ce_micro_f1"]}
                for (model, config), row in new_map.items()
                if config == "FP32"
            ),
            key=lambda row: row["value"],
        ),
        "nf4_ce_improved_models": [
            row["model"] for row in fp32_nf4 if row["delta_CE_micro_F1"] > 0
        ],
        "nf4_bleu4_absolute_delta_range": [
            min(abs(row["delta_BLEU4"]) for row in fp32_nf4),
            max(abs(row["delta_BLEU4"]) for row in fp32_nf4),
        ],
        "pooled_mcnemar_p_lt_0_05": significant,
        "pooled_mcnemar_bonferroni_threshold": bonferroni,
        "pooled_mcnemar_bonferroni_significant": [
            row for row in pooled if float(row["p"]) < bonferroni
        ],
        "artifacts": {
            "benchmark": str(args.corrected_run / "multimodel_benchmark.json"),
            "bootstrap": str(args.corrected_run / "bootstrap_fnf_ci.json"),
            "mcnemar": str(args.corrected_run / "mcnemar_results.json"),
        },
    }
    atomic_json(args.audit_dir / "corrected_run_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
