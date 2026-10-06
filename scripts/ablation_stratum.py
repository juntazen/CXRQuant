#!/usr/bin/env python3
"""Degenerate-versus-functional stratum ablation and floor-effect diagnostics.

Recomputes pooled reduced-precision summaries from one validated run under
three model sets: all checkpoints, the functional stratum and the degenerate
stratum. On the primary run, strata are derived from the stored FP32 BLEU-4 with
the thresholds stated in the manuscript (degenerate < 0.02, functional > 0.15);
a checkpoint between the thresholds is a hard error rather than a silent
assignment. Replication runs reuse the pre-specified primary-run strata and
report their own FP32 BLEU-4 classes as a diagnostic.

Inputs are the replayed tables (``tables/metrics.csv``, ``mcnemar_pooled.csv``)
and, for the floor-effect diagnostics, the stored per-configuration records.
Outputs are new files only: ``ablation_stratum.json`` and
``tables/ablation_stratum.csv``. Discordant-pair sums are descriptive; no pooled
test is computed across models because models are not exchangeable units.
"""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from cxrquant.paths import REPO_ROOT
from cxrquant.runio import atomic_json, sha256_file

PRIMARY_RUN = REPO_ROOT / "runs" / "full_seed42_corrected_20260911"

REDUCED = ("FP16", "INT8-BnB", "NF4-BnB")
DEGENERATE_BELOW = 0.02
FUNCTIONAL_ABOVE = 0.15


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def num(value):
    return None if value in ("", None) else float(value)


def classify(metrics, degenerate_below=DEGENERATE_BELOW, functional_above=FUNCTIONAL_ABOVE):
    strata = {"functional": [], "degenerate": []}
    for row in metrics:
        if row["config"] != "FP32":
            continue
        bleu = float(row["bleu_4"])
        if bleu < degenerate_below:
            strata["degenerate"].append(row["model"])
        elif bleu > functional_above:
            strata["functional"].append(row["model"])
        else:
            raise ValueError(f"{row['model']} FP32 BLEU-4 {bleu} lies between stratum thresholds")
    return strata


def diagnostic_classes(metrics):
    """Per-run FP32 BLEU-4 classes without failing on values between thresholds."""
    out = {}
    for row in metrics:
        if row["config"] == "FP32":
            bleu = float(row["bleu_4"])
            out[row["model"]] = {"fp32_bleu_4": bleu, "class": "degenerate" if bleu < DEGENERATE_BELOW
                                 else "functional" if bleu > FUNCTIONAL_ABOVE else "between_thresholds"}
    return out


def summarise(metrics, pooled, models):
    fp32 = {r["model"]: r for r in metrics if r["config"] == "FP32"}
    out = []
    for cfg in REDUCED:
        rows = [r for r in metrics if r["config"] == cfg and r["model"] in models]
        counts = [r for r in pooled if r["config"] == cfg and r["model"] in models]
        if len(rows) != len(models) or len(counts) != len(models):
            raise ValueError(f"missing rows for {cfg}")
        fnf = [num(r["FNF_rate"]) for r in rows]
        fpf = [num(r["FPF_rate"]) for r in rows]
        if any(v is None for v in fnf + fpf):
            raise ValueError(f"undefined rate in {cfg}; pooled mean is not defined")
        d_bleu = [100 * (float(r["bleu_4"]) - float(fp32[r["model"]]["bleu_4"])) for r in rows]
        d_ce = [
            100 * (float(r["ce_micro_f1"]) - float(fp32[r["model"]]["ce_micro_f1"])) for r in rows
        ]
        out.append(
            {
                "config": cfg,
                "n_models": len(rows),
                "FNF_mean": round(mean(fnf), 4),
                "FNF_min": min(fnf),
                "FNF_max": max(fnf),
                "FPF_mean": round(mean(fpf), 4),
                "BLEU4_delta_pp_mean": round(mean(d_bleu), 2),
                "BLEU4_abs_delta_pp_max": round(max(abs(x) for x in d_bleu), 2),
                "CE_micro_F1_delta_pp_mean": round(mean(d_ce), 2),
                "b_sum": sum(int(r["b"]) for r in counts),
                "c_sum": sum(int(r["c"]) for r in counts),
            }
        )
    return out


def ordering_holds(metrics, models):
    result = {}
    for model in models:
        vals = [
            num(next(r["FNF_rate"] for r in metrics if r["model"] == model and r["config"] == c))
            for c in REDUCED
        ]
        result[model] = all(v is not None for v in vals) and vals[0] < vals[1] < vals[2]
    return result


def floor_diagnostics(run, models, max_new_tokens):
    """Generation-length and extracted-positive profile of the stored FP32 outputs."""
    out = {}
    for model in models:
        fp32 = json.loads((run / "models" / model / "FP32.json").read_text())
        nf4 = json.loads((run / "models" / model / "NF4-BnB.json").read_text())
        samples = fp32["samples"]
        n = len(samples)
        out[model] = {
            "n_reports": n,
            "empty_outputs": sum(s["output_status"] == "empty" for s in samples),
            "reached_max_new_tokens": sum(s["generated_tokens"] >= max_new_tokens for s in samples),
            "mean_words": round(mean(len(s["prediction"].split()) for s in samples), 1),
            "fp32_positive_category_pairs": nf4["safety_counts"]["fp32_present"],
            "fp32_positive_pairs_per_report": round(nf4["safety_counts"]["fp32_present"] / n, 3),
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--reference-strata", type=Path, help="ablation_stratum.json whose strata are reused")
    args = ap.parse_args(argv)
    run = args.run_dir.resolve()
    out_json = run / "ablation_stratum.json"
    out_csv = run / "tables" / "ablation_stratum.csv"
    if not args.overwrite and (out_json.exists() or out_csv.exists()):
        raise SystemExit("ablation output exists; pass --overwrite to regenerate")
    metrics_path = run / "tables" / "metrics.csv"
    pooled_path = run / "tables" / "mcnemar_pooled.csv"
    metrics, pooled = read_csv(metrics_path), read_csv(pooled_path)
    reference = args.reference_strata.resolve() if args.reference_strata else None
    if reference is None and run != PRIMARY_RUN.resolve() and (PRIMARY_RUN / "ablation_stratum.json").exists():
        reference = (PRIMARY_RUN / "ablation_stratum.json").resolve()
    if reference is not None:
        # Strata are pre-specified on the primary run and held fixed for replication runs.
        strata = json.loads(reference.read_text())["strata"]
        strata_source = {"source": "reference", "path": str(reference), "sha256": sha256_file(reference)}
    else:
        strata = classify(metrics)
        strata_source = {"source": "this run FP32 BLEU-4 thresholds"}
    everyone = strata["degenerate"] + strata["functional"]
    sets = {"all": everyone, **strata}
    rows = []
    for label, models in sets.items():
        for row in summarise(metrics, pooled, models):
            rows.append({"model_set": label, "models": "|".join(sorted(models)), **row})
    cfg = json.loads((run / "config.json").read_text())
    nf4 = {r["model_set"]: r for r in rows if r["config"] == "NF4-BnB"}
    out = {
        "analysis": "stratum ablation (re-aggregation of replayed tables; no training or generation)",
        "thresholds": {
            "degenerate_fp32_bleu4_below": DEGENERATE_BELOW,
            "functional_fp32_bleu4_above": FUNCTIONAL_ABOVE,
        },
        "strata": strata,
        "strata_source": strata_source,
        "this_run_fp32_bleu4_classes": diagnostic_classes(metrics),
        "rows": rows,
        "ordering_FP16_lt_INT8_lt_NF4": {k: ordering_holds(metrics, v) for k, v in sets.items()},
        "nf4_fnf_range": {k: [v["FNF_min"], v["FNF_max"]] for k, v in nf4.items()},
        "floor_effect_diagnostics": floor_diagnostics(run, everyone, cfg["max_new_tokens"]),
        "note": "b_sum/c_sum are descriptive totals; no across-model pooled test is reported.",
        "provenance": {
            "run_id": json.loads((run / "identity.json").read_text())["run_id"],
            "metrics_hash": sha256_file(metrics_path),
            "mcnemar_pooled_hash": sha256_file(pooled_path),
            "data_hash": sha256_file(run / "data.json"),
            "script_hash": sha256_file(__file__),
        },
    }
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(out_json, out)
    print(json.dumps({k: out[k] for k in ("strata", "nf4_fnf_range", "floor_effect_diagnostics")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
