"""Fail-closed, manifest-driven completeness and numerical verification."""

import math

from cxrquant.clinical_safety.fact_extractor import extract_labels
from cxrquant.clinical_safety.safety_metrics import clinical_efficacy_f1, fact_preservation
from cxrquant.evaluation.nlg_metrics import evaluate_reports

NLG_KEYS = ("bleu_1", "bleu_4", "rouge_1", "rouge_2", "rouge_l")
FLIP_KEYS = ("FNF_rate", "FPF_rate", "CFPR", "CFFR")


def unique_strings(values, label):
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(x, str) or not x for x in values)
    ):
        raise ValueError(label + ": expected nonempty list of names")
    if len(values) != len(set(values)):
        raise ValueError(label + ": duplicate names")


def score(preds, refs, base=None):
    labels = [extract_labels(x) for x in preds]
    efficacy = clinical_efficacy_f1(labels, [extract_labels(x) for x in refs])
    out = {k: round(v, 4) for k, v in evaluate_reports(preds, refs).items() if k in NLG_KEYS}
    out.update(ce_micro_f1=efficacy["micro_f1"], ce_macro_f1=efficacy["macro_f1"])
    if base is not None:
        flips = fact_preservation(labels, [extract_labels(x) for x in base])
        out.update({k: flips[k] for k in FLIP_KEYS})
        out["safety_counts"] = flips["counts"]
        out["per_category_flips"] = flips["per_category_flips"]
    return out


def validate_manifest(manifest):
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be an object")
    for key in ("models", "configs", "sample_ids"):
        unique_strings(manifest.get(key), key)
    if "FP32" not in manifest["configs"]:
        raise ValueError("manifest requires FP32")
    refs = manifest.get("references")
    if (
        not isinstance(refs, list)
        or len(refs) != len(manifest["sample_ids"])
        or any(not isinstance(x, str) for x in refs)
    ):
        raise ValueError("invalid manifest references")
    if manifest.get("rounding_digits") != 4:
        raise ValueError("supported rounding: Python round(x, 4)")


def validate_benchmark(bench, manifest, verify_metrics=True):
    validate_manifest(manifest)
    if not isinstance(bench, dict) or not isinstance(bench.get("models"), list):
        raise ValueError("benchmark must contain models")
    models = bench["models"]
    if any(not isinstance(m, dict) for m in models):
        raise ValueError("invalid model record")
    names = [m.get("model") for m in models]
    unique_strings(names, "benchmark models")
    if set(names) != set(manifest["models"]):
        raise ValueError("model matrix incomplete or unexpected models")
    if bench.get("run_id") != manifest.get("run_id"):
        raise ValueError("run_id mismatch")
    n = len(manifest["sample_ids"])
    comparisons = 0
    for m in models:
        if "error" in m or m.get("status") != "complete":
            raise ValueError("failed/partial model: " + str(m.get("model")))
        if (
            m.get("sample_ids") != manifest["sample_ids"]
            or m.get("references") != manifest["references"]
            or m.get("n_test") != n
        ):
            raise ValueError("sample/reference order or length mismatch: " + m["model"])
        preds = m.get("predictions")
        if not isinstance(preds, dict) or set(preds) != set(manifest["configs"]):
            raise ValueError("prediction configurations missing/unexpected: " + m["model"])
        for cfg, texts in preds.items():
            if m.get("prediction_sample_ids", {}).get(cfg) != manifest["sample_ids"]:
                raise ValueError("per-configuration sample order mismatch: " + cfg)
            if (
                not isinstance(texts, list)
                or len(texts) != n
                or any(not isinstance(x, str) for x in texts)
            ):
                raise ValueError("invalid prediction array: " + cfg)
        configs = m.get("configs")
        if not isinstance(configs, list) or any(not isinstance(c, dict) for c in configs):
            raise ValueError("invalid configs")
        config_names = [c.get("config") for c in configs]
        unique_strings(config_names, "config names")
        if set(config_names) != set(manifest["configs"]):
            raise ValueError("config matrix incomplete")
        for c in configs:
            cfg = c["config"]
            if "error" in c or c.get("status") != "complete":
                raise ValueError("failed/partial config: " + cfg)
            required = list(NLG_KEYS) + ["ce_micro_f1", "ce_macro_f1"]
            if cfg != "FP32":
                required += list(FLIP_KEYS)
                comparisons += 1
            calculated = (
                score(preds[cfg], m["references"], preds["FP32"] if cfg != "FP32" else None)
                if verify_metrics
                else None
            )
            for key in required:
                if key not in c:
                    raise ValueError("missing metric: " + key)
                val = c[key]
                if val is None:
                    if key not in ("FNF_rate", "FPF_rate") or not isinstance(
                        c.get("safety_counts"), dict
                    ):
                        raise ValueError("unexpected undefined metric: " + key)
                    denominator = "fp32_present" if key == "FNF_rate" else "fp32_absent"
                    if c["safety_counts"].get(denominator) != 0:
                        raise ValueError("null rate with nonzero denominator")
                elif (
                    isinstance(val, bool)
                    or not isinstance(val, (int, float))
                    or not math.isfinite(val)
                    or not 0 <= val <= 1
                ):
                    raise ValueError("invalid/nonfinite metric: " + key)
                if calculated is not None and val != calculated[key]:
                    raise ValueError(
                        f"metric mismatch {m['model']}/{cfg}/{key}: {val} != {calculated[key]}"
                    )
            if cfg != "FP32":
                counts = c.get("safety_counts")
                if not isinstance(counts, dict) or any(
                    type(x) is not int or x < 0 for x in counts.values()
                ):
                    raise ValueError("invalid/missing raw counts")
                if calculated is not None and counts != calculated["safety_counts"]:
                    raise ValueError("raw count mismatch")
    return {
        "models": len(models),
        "configurations": len(models) * len(manifest["configs"]),
        "predictions": len(models) * len(manifest["configs"]) * n,
        "comparisons": comparisons,
    }
