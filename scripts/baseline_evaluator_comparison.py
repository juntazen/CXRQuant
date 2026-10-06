#!/usr/bin/env python3
"""Cross-check the audit with independent labellers (CheXbert, CheXpert labeller).

For one validated run, every stored prediction, every reference impression and
every extractor-validation text is labelled by the CXRQuant rule-based extractor
and by each available independent labeller:

* CheXbert (Smit et al., EMNLP 2020), executed here through
  ``cxrquant.clinical_safety.chexbert_labeler``;
* the original CheXpert labeller (Irvin et al., AAAI 2019), executed externally
  by ``scripts/chexpert_labeler_bridge.py`` and read from ``chexpert_labels.json``.

Each extractor then feeds the unchanged metric layer separately: reference CE
micro/macro F1, the FP32-relative differential audit (CFPR, FNF, FPF, pooled b/c)
and the extractor-validation protocol of ``extractor_comparison.py`` (MeSH silver
labels, definite positives only). Pairwise agreement (raw agreement and Cohen's
kappa on binary labels, uncertain -> positive) is reported per category.

Honest by design: an unavailable labeller is recorded with its error and no score
is imputed. No training or generation is performed; outputs are new files only.
"""
import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import os
import platform
import time
from pathlib import Path

from cxrquant.clinical_safety.fact_extractor import _PATHOLOGY_CATEGORIES, extract_labels, labels_to_binary
from cxrquant.clinical_safety.safety_metrics import clinical_efficacy_f1, fact_preservation
from cxrquant.paths import REPO_ROOT
from cxrquant.runio import atomic_json, sha256_file

REDUCED = ("FP16", "INT8-BnB", "NF4-BnB")
METRIC_KEYS = ("ce_micro_f1", "ce_macro_f1", "CFPR", "FNF_rate", "FPF_rate", "b", "c", "fp32_present")


def text_key(text):
    return hashlib.sha256(text.encode()).hexdigest()


def kappa(a, b):
    n = len(a)
    if n == 0 or n != len(b):
        raise ValueError("paired nonempty labels required")
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return {"agreement": round(po, 4), "kappa": None if pe == 1 else round((po - pe) / (1 - pe), 4),
            "positive_rate_a": round(pa, 4), "positive_rate_b": round(pb, 4), "n": n}


def agreement(labels_a, labels_b):
    out, pooled_a, pooled_b = {}, [], []
    bin_a = [labels_to_binary(x) for x in labels_a]
    bin_b = [labels_to_binary(x) for x in labels_b]
    for cat in _PATHOLOGY_CATEGORIES:
        a = [x[cat] for x in bin_a]
        b = [x[cat] for x in bin_b]
        pooled_a += a
        pooled_b += b
        out[cat] = kappa(a, b)
    out["ALL_13_POOLED"] = kappa(pooled_a, pooled_b)
    return out


def audit(manifest, records, label_of):
    refs = [label_of(x) for x in manifest["references"]]
    rows = []
    for model in manifest["models"]:
        base = [label_of(x) for x in records[model, "FP32"]["predictions"]]
        for cfg in manifest["configs"]:
            preds = base if cfg == "FP32" else [label_of(x) for x in records[model, cfg]["predictions"]]
            ce = clinical_efficacy_f1(preds, refs)
            row = {"model": model, "config": cfg, "ce_micro_f1": ce["micro_f1"], "ce_macro_f1": ce["macro_f1"]}
            if cfg != "FP32":
                fp = fact_preservation(preds, base)
                row.update(CFPR=fp["CFPR"], FNF_rate=fp["FNF_rate"], FPF_rate=fp["FPF_rate"],
                           b=fp["counts"]["false_negative_flips"], c=fp["counts"]["false_positive_flips"],
                           fp32_present=fp["counts"]["fp32_present"])
            rows.append(row)
    return rows


def ordering(rows, models):
    out = {}
    for model in models:
        vals = [next(r["FNF_rate"] for r in rows if r["model"] == model and r["config"] == c) for c in REDUCED]
        defined = all(v is not None for v in vals)
        out[model] = {"FNF": dict(zip(REDUCED, vals, strict=True)),
                      "strict_FP16_lt_INT8_lt_NF4": defined and vals[0] < vals[1] < vals[2],
                      "FP16_le_INT8_le_NF4": defined and vals[0] <= vals[1] <= vals[2]}
    return out


def silver_validation(comparator, label_of):
    test = comparator.load_test()
    preds = [{c for c in comparator.SCORABLE if label_of(t["text"]).get(c) == 1} for t in test]
    result = comparator.score(preds, [t["gt"] for t in test])
    result.update(n_reports=len(test), gt_positive_pairs=sum(len(t["gt"]) for t in test),
                  positive_policy="only definite positive; uncertain excluded")
    return result


def load_chexbert(texts, device, batch_size):
    from cxrquant.clinical_safety import chexbert_labeler as cb
    import torch

    started = time.perf_counter()
    labeler = cb.CheXbertLabeler(device=device)
    load_s = time.perf_counter() - started
    started = time.perf_counter()
    classes = labeler.label_classes(texts, batch_size)
    label_s = time.perf_counter() - started
    probe = texts[: min(64, len(texts))]
    if labeler.label_classes(probe, 1) != classes[: len(probe)]:
        raise RuntimeError("CheXbert classes depend on batch composition")
    info = {
        "available": True,
        "checkpoint_repo": cb.CHECKPOINT_REPO,
        "checkpoint_file": cb.CHECKPOINT_FILE,
        "checkpoint_revision": cb.CHECKPOINT_REVISION,
        "checkpoint_sha256": sha256_file(labeler.checkpoint),
        "tokenizer": cb.TOKENIZER_ID,
        "load_strict": True,
        "batch_invariance_probe_texts": len(probe),
        "hardware": {"device": str(labeler.device), "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
                     "cpu_count": os.cpu_count(), "platform": platform.platform()},
        "load_s": round(load_s, 2),
        "labelling_s": round(label_s, 2),
        "ms_per_text": round(1000 * label_s / len(texts), 3),
    }
    return {t: cb.decode(c) for t, c in zip(texts, classes, strict=True)}, info, {text_key(t): c for t, c in zip(texts, classes, strict=True)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "iuxray_paired.json")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    run = args.run_dir.resolve()
    out_json = run / "baseline_evaluator_comparison.json"
    if out_json.exists() and not args.overwrite:
        raise SystemExit("output exists; pass --overwrite to regenerate")
    manifest = json.loads((run / "manifest.json").read_text())
    identity = json.loads((run / "identity.json").read_text())
    records = {}
    for model in manifest["models"]:
        for cfg in manifest["configs"]:
            rec = json.loads((run / "models" / model / (cfg + ".json")).read_text())
            if rec.get("status") != "complete" or rec.get("identity") != identity:
                raise ValueError(f"incomplete or foreign record: {model}/{cfg}")
            records[model, cfg] = rec
    spec = importlib.util.spec_from_file_location("extractor_comparison", REPO_ROOT / "scripts" / "extractor_comparison.py")
    comparator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparator)
    comparator.DATA = args.data
    texts = set(manifest["references"]) | {t["text"] for t in comparator.load_test()}
    for rec in records.values():
        texts.update(rec["predictions"])
    texts = sorted(texts)

    labellers = {"cxrquant": {t: extract_labels(t) for t in texts}}
    availability = {"cxrquant": {"available": True, "note": "primary rule-based extractor"}}
    try:
        labels, info, raw = load_chexbert(texts, args.device, args.batch_size)
        labellers["chexbert"] = labels
        availability["chexbert"] = info
        atomic_json(run / "chexbert_labels.json", {"provenance": info, "labels": raw})
    except Exception as e:
        availability["chexbert"] = {"available": False, "error": f"{type(e).__name__}: {str(e)[:300]}",
                                    "note": "not executed; no score imputed"}
    chexpert_path = run / "chexpert_labels.json"
    if chexpert_path.exists():
        stored = json.loads(chexpert_path.read_text())
        missing = [t for t in texts if text_key(t) not in stored["labels"]]
        if missing:
            availability["chexpert"] = {"available": False, "error": f"{len(missing)} texts without CheXpert labels"}
        else:
            labellers["chexpert"] = {t: stored["labels"][text_key(t)] for t in texts}
            availability["chexpert"] = {"available": True, **stored["provenance"],
                                        "labels_hash": sha256_file(chexpert_path)}
    else:
        availability["chexpert"] = {"available": False, "error": "chexpert_labels.json absent",
                                    "note": "run scripts/chexpert_labeler_bridge.py; no score imputed"}

    rows = {name: audit(manifest, records, labels.__getitem__) for name, labels in labellers.items()}
    combined = []
    for i, base in enumerate(rows["cxrquant"]):
        row = {"model": base["model"], "config": base["config"]}
        for key in METRIC_KEYS:
            if key in base:
                for name in labellers:
                    row[f"{key}_{name}"] = rows[name][i][key]
        combined.append(row)
    prediction_texts = [t for rec in records.values() for t in rec["predictions"]]
    agree = {}
    for a, b in itertools.combinations(labellers, 2):
        agree[f"{a}_vs_{b}"] = {
            "references": agreement([labellers[a][t] for t in manifest["references"]],
                                    [labellers[b][t] for t in manifest["references"]]),
            "predictions_all_configs": agreement([labellers[a][t] for t in prediction_texts],
                                                 [labellers[b][t] for t in prediction_texts]),
        }
    out = {
        "analysis": "independent-labeller cross-check (re-labelling of stored text; no training or generation)",
        "primary_extractor": "cxrquant (primary analysis unchanged)",
        "uncertain_policy": "positive for CE-F1, flips and agreement; silver validation counts definite positives only",
        "labellers": availability,
        "rows": combined,
        "fnf_ordering": {name: ordering(r, manifest["models"]) for name, r in rows.items()},
        "inter_extractor_agreement": agree,
        "silver_label_validation": {name: silver_validation(comparator, labels.__getitem__) for name, labels in labellers.items()},
        "provenance": {
            "run_id": identity["run_id"],
            "run_identity": identity,
            "data_hash": sha256_file(args.data),
            "manifest_hash": sha256_file(run / "manifest.json"),
            "script_hash": sha256_file(__file__),
            "unique_texts_labelled": len(texts),
        },
    }
    tables = run / "tables"
    tables.mkdir(exist_ok=True)
    with (tables / "baseline_evaluator_comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(combined[-1]))
        writer.writeheader()
        writer.writerows(combined)
    with (tables / "extractor_agreement.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pair", "text_set", "category", "agreement", "kappa", "positive_rate_a", "positive_rate_b", "n"])
        for pair, sets in agree.items():
            for text_set, cats in sets.items():
                for cat, v in cats.items():
                    writer.writerow([pair, text_set, cat, v["agreement"], v["kappa"], v["positive_rate_a"], v["positive_rate_b"], v["n"]])
    with (tables / "silver_validation_labellers.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["extractor", "precision", "recall", "f1", "tp", "fp", "fn"])
        for name, v in out["silver_label_validation"].items():
            writer.writerow([name, v["precision"], v["recall"], v["f1"], v["tp"], v["fp"], v["fn"]])
    atomic_json(out_json, out)
    print(json.dumps({
        "labellers": {k: v["available"] for k, v in availability.items()},
        "strict_ordering_all_models": {k: all(m["strict_FP16_lt_INT8_lt_NF4"] for m in o.values()) for k, o in out["fnf_ordering"].items()},
        "agreement_predictions_pooled": {k: v["predictions_all_configs"]["ALL_13_POOLED"] for k, v in agree.items()},
        "silver": {k: {x: v[x] for x in ("precision", "recall", "f1")} for k, v in out["silver_label_validation"].items()},
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
