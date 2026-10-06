#!/usr/bin/env python3
"""Sensitivity of the differential audit to the uncertain-finding policy.

Re-scores the stored predictions of one validated run under both binarisation
policies already supported by the metric layer: ``positive`` (U-Ones, the
primary analysis used by ``validation.score``) and ``negative`` (U-Zeros).
No training and no generation are performed; only ``models/<model>/<CONFIG>.json``
predictions and ``manifest.json`` references are read.

Non-destructive: writes ``ablation_uncertain_policy.json`` in the run directory
and ``tables/ablation_uncertain_policy.csv``; refuses to overwrite either unless
``--overwrite`` is given. The primary (positive) values are cross-checked
against the archived per-configuration metrics so that a silent drift of the
extractor or metric code fails loudly instead of producing a mismatched table.
"""
import argparse
import csv
import json
from pathlib import Path

from cxrquant.clinical_safety.fact_extractor import extract_labels
from cxrquant.clinical_safety.safety_metrics import clinical_efficacy_f1, fact_preservation
from cxrquant.runio import atomic_json, sha256_file

POLICIES = ("positive", "negative")
REDUCED = ("FP16", "INT8-BnB", "NF4-BnB")


def pp(new, old):
    if new is None or old is None:
        return None
    return round(100 * (new - old), 2)


def load_run(run):
    manifest = json.loads((run / "manifest.json").read_text())
    identity = json.loads((run / "identity.json").read_text())
    records = {}
    for model in manifest["models"]:
        for cfg in manifest["configs"]:
            rec = json.loads((run / "models" / model / (cfg + ".json")).read_text())
            if rec.get("status") != "complete" or rec.get("identity") != identity:
                raise ValueError(f"incomplete or foreign record: {model}/{cfg}")
            if [s["sample_id"] for s in rec["samples"]] != manifest["sample_ids"]:
                raise ValueError(f"sample order mismatch: {model}/{cfg}")
            records[model, cfg] = rec
    return manifest, identity, records


def ablate(manifest, records, configs=REDUCED):
    refs = [extract_labels(x) for x in manifest["references"]]
    rows, detail = [], []
    for model in manifest["models"]:
        base = [extract_labels(x) for x in records[model, "FP32"]["predictions"]]
        ce_base = {p: clinical_efficacy_f1(base, refs, p) for p in POLICIES}
        for cfg in configs:
            rec = records[model, cfg]
            quant = [extract_labels(x) for x in rec["predictions"]]
            flips = {p: fact_preservation(quant, base, p) for p in POLICIES}
            ce = {p: clinical_efficacy_f1(quant, refs, p) for p in POLICIES}
            archived = {k: rec[k] for k in ("FNF_rate", "FPF_rate", "CFPR", "ce_micro_f1")}
            primary = {
                "FNF_rate": flips["positive"]["FNF_rate"],
                "FPF_rate": flips["positive"]["FPF_rate"],
                "CFPR": flips["positive"]["CFPR"],
                "ce_micro_f1": ce["positive"]["micro_f1"],
            }
            if primary != archived:
                raise ValueError(f"primary policy does not reproduce archive: {model}/{cfg}")
            fp, fn = flips["positive"], flips["negative"]
            rows.append(
                {
                    "model": model,
                    "config": cfg,
                    "FNF_positive": fp["FNF_rate"],
                    "FNF_negative": fn["FNF_rate"],
                    "FNF_delta_pp": pp(fn["FNF_rate"], fp["FNF_rate"]),
                    "FPF_positive": fp["FPF_rate"],
                    "FPF_negative": fn["FPF_rate"],
                    "FPF_delta_pp": pp(fn["FPF_rate"], fp["FPF_rate"]),
                    "fp32_present_positive": fp["counts"]["fp32_present"],
                    "fp32_present_negative": fn["counts"]["fp32_present"],
                    "b_positive": fp["counts"]["false_negative_flips"],
                    "c_positive": fp["counts"]["false_positive_flips"],
                    "b_negative": fn["counts"]["false_negative_flips"],
                    "c_negative": fn["counts"]["false_positive_flips"],
                    "CE_micro_F1_positive": ce["positive"]["micro_f1"],
                    "CE_micro_F1_negative": ce["negative"]["micro_f1"],
                    "CE_micro_F1_FP32_positive": ce_base["positive"]["micro_f1"],
                    "CE_micro_F1_FP32_negative": ce_base["negative"]["micro_f1"],
                }
            )
            detail.append(
                {
                    "model": model,
                    "config": cfg,
                    "fact_preservation": {p: flips[p] for p in POLICIES},
                    "clinical_efficacy_f1": {p: ce[p] for p in POLICIES},
                }
            )
    return rows, detail


def ordering(rows, models, policy):
    """Per-model check of FNF(FP16) < FNF(INT8) < FNF(NF4) under one policy."""
    out = {}
    for model in models:
        vals = [
            next(r["FNF_" + policy] for r in rows if r["model"] == model and r["config"] == c)
            for c in REDUCED
        ]
        defined = all(v is not None for v in vals)
        out[model] = {
            "FNF": dict(zip(REDUCED, vals, strict=True)),
            "strict_FP16_lt_INT8_lt_NF4": defined and vals[0] < vals[1] < vals[2],
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    run = args.run_dir.resolve()
    out_json = run / "ablation_uncertain_policy.json"
    out_csv = run / "tables" / "ablation_uncertain_policy.csv"
    if not args.overwrite and (out_json.exists() or out_csv.exists()):
        raise SystemExit("ablation output exists; pass --overwrite to regenerate")
    manifest, identity, records = load_run(run)
    rows, detail = ablate(manifest, records)
    nf4 = [r for r in rows if r["config"] == "NF4-BnB"]
    largest = max(nf4, key=lambda r: abs(r["FNF_delta_pp"] or 0))
    summary = {
        "ordering": {p: ordering(rows, manifest["models"], p) for p in POLICIES},
        "nf4_fnf_range": {
            p: [min(r["FNF_" + p] for r in nf4), max(r["FNF_" + p] for r in nf4)]
            for p in POLICIES
        },
        "nf4_largest_absolute_fnf_shift": {
            "model": largest["model"],
            "delta_pp": largest["FNF_delta_pp"],
        },
    }
    summary["ordering_holds_all_models"] = {
        p: all(v["strict_FP16_lt_INT8_lt_NF4"] for v in summary["ordering"][p].values())
        for p in POLICIES
    }
    out = {
        "analysis": "uncertain-policy sensitivity (re-scoring of stored predictions only)",
        "primary_policy": "positive",
        "sensitivity_policy": "negative",
        "categories": 13,
        "rows": rows,
        "summary": summary,
        "detail": detail,
        "provenance": {
            "run_id": identity["run_id"],
            "run_identity": identity,
            "data_hash": sha256_file(run / "data.json"),
            "manifest_hash": sha256_file(run / "manifest.json"),
            "script_hash": sha256_file(__file__),
            "training_or_generation": "none",
            "archived_primary_metrics_reproduced": True,
        },
    }
    out_csv.parent.mkdir(exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(out_json, out)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
