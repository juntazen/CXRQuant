#!/usr/bin/env python3
"""Where reduced-precision outputs start to differ from FP32, and where the flips arise.

CPU-only analysis of stored predictions. For every report and reduced-precision
configuration the word sequence is compared with the FP32 output of the same
checkpoint to find the first differing word. Each flipped category is then
located relative to that point by extracting labels from the shared prefix:

* "decided after divergence": the category is not mentioned in the shared prefix,
  so its state is set entirely by the diverging continuations;
* "revised after divergence": the category is mentioned in the prefix and a later,
  diverging sentence changes its aggregated state.

A flip can only occur in a report whose text differs, which is checked explicitly.
Loss (b) and gain (c) counts are also reported by the relative position of the
divergence (early, middle, late third of the FP32 output). Writes
``divergence_analysis.json`` and ``tables/divergence_analysis.csv``.
"""
import argparse
import csv
import json
from pathlib import Path
from statistics import median

from cxrquant.clinical_safety.fact_extractor import _PATHOLOGY_CATEGORIES, extract_labels, labels_to_binary
from cxrquant.runio import atomic_json, sha256_file

REDUCED = ("FP16", "INT8-BnB", "NF4-BnB")


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def analyse_pair(base_texts, quant_texts):
    stats = {"reports": len(base_texts), "identical_text": 0, "diverged": 0, "flips": 0,
             "flips_in_identical_text": 0, "decided_after_divergence": 0, "revised_after_divergence": 0,
             "b": 0, "c": 0}
    onset_words, onset_rel = [], []
    thirds = {"early": [0, 0], "middle": [0, 0], "late": [0, 0]}
    for x32, xq in zip(base_texts, quant_texts, strict=True):
        w32, wq = x32.split(), xq.split()
        d = first_divergence(w32, wq)
        l32, lq = extract_labels(x32), extract_labels(xq)
        b32, bq = labels_to_binary(l32), labels_to_binary(lq)
        flipped = [c for c in _PATHOLOGY_CATEGORIES if b32[c] != bq[c]]
        if d is None:
            stats["identical_text"] += 1
            stats["flips_in_identical_text"] += len(flipped)
            continue
        stats["diverged"] += 1
        onset_words.append(d)
        rel = d / max(len(w32), 1)
        onset_rel.append(rel)
        third = "early" if rel < 1 / 3 else "middle" if rel < 2 / 3 else "late"
        prefix = extract_labels(" ".join(w32[:d]))
        for c in flipped:
            stats["flips"] += 1
            if prefix[c] is None:
                stats["decided_after_divergence"] += 1
            else:
                stats["revised_after_divergence"] += 1
            if b32[c] == 1:
                stats["b"] += 1
                thirds[third][0] += 1
            else:
                stats["c"] += 1
                thirds[third][1] += 1
    stats["median_onset_word"] = median(onset_words) if onset_words else None
    stats["median_onset_relative"] = round(median(onset_rel), 3) if onset_rel else None
    stats["b_c_by_onset_third"] = {k: {"b": v[0], "c": v[1]} for k, v in thirds.items()}
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    run = args.run_dir.resolve()
    out_json, out_csv = run / "divergence_analysis.json", run / "tables" / "divergence_analysis.csv"
    if not args.overwrite and (out_json.exists() or out_csv.exists()):
        raise SystemExit("output exists; pass --overwrite to regenerate")
    manifest = json.loads((run / "manifest.json").read_text())
    rows, per_model = [], {}
    for model in manifest["models"]:
        preds = {c: json.loads((run / "models" / model / (c + ".json")).read_text())["predictions"] for c in manifest["configs"]}
        per_model[model] = {}
        for cfg in REDUCED:
            s = analyse_pair(preds["FP32"], preds[cfg])
            if s["flips_in_identical_text"]:
                raise ValueError(f"flip without text change: {model}/{cfg}")
            per_model[model][cfg] = s
            rows.append({"model": model, "config": cfg, **{k: v for k, v in s.items() if k != "b_c_by_onset_third"},
                         **{f"{t}_{x}": s["b_c_by_onset_third"][t][x] for t in ("early", "middle", "late") for x in ("b", "c")}})
    pooled = {}
    for cfg in REDUCED:
        sel = [per_model[m][cfg] for m in manifest["models"]]
        flips = sum(x["flips"] for x in sel)
        pooled[cfg] = {
            "reports": sum(x["reports"] for x in sel), "diverged": sum(x["diverged"] for x in sel),
            "flips": flips, "decided_after_divergence": sum(x["decided_after_divergence"] for x in sel),
            "share_decided_after_divergence": round(sum(x["decided_after_divergence"] for x in sel) / flips, 4) if flips else None,
            "b": sum(x["b"] for x in sel), "c": sum(x["c"] for x in sel),
            "b_c_by_onset_third": {t: {k: sum(x["b_c_by_onset_third"][t][k] for x in sel) for k in ("b", "c")} for t in ("early", "middle", "late")},
        }
    (run / "tables").mkdir(exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(out_json, {"analysis": "word-level divergence onset and localisation of category flips", "per_model": per_model,
                           "pooled": pooled, "provenance": {"run_id": manifest["run_id"], "script_hash": sha256_file(__file__),
                                                            "manifest_hash": sha256_file(run / "manifest.json")}})
    print(json.dumps(pooled, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
