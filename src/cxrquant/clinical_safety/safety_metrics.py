"""
Clinical safety metrics for quantized CXR report generation.

Provides two complementary analyses:

  (A) Clinical Efficacy F1 (CE-F1) vs. the REFERENCE report — the established
      CXRQuant-extractor clinical-correctness metric (related to the
      literature; Yu et al., Patterns 2023).

  (B) Quantization fact-preservation vs. the FP32 BASELINE model — the novel
      contribution. Isolates the effect of compression on clinical facts by
      comparing each quantized model's outputs against the full-precision
      model's outputs on the same inputs:

        CFPR  Clinical Fact Preservation Rate  — (sample,finding) pairs unchanged
        CFFR  Clinical Fact Flip Rate          — (sample,finding) pairs flipped
        FNF   False-Negative Flip rate         — FP32 present -> quant absent
                                                 (a MISSED finding; most dangerous)
        FPF   False-Positive Flip rate         — FP32 absent -> quant present
                                                 (a false alarm)

All inputs are lists of 14-category label dicts as produced by
fact_extractor.extract_labels().
"""

from __future__ import annotations

from .fact_extractor import (
    _PATHOLOGY_CATEGORIES,
    labels_to_binary,
)


def _binarize_all(label_dicts: list[dict[str, int | None]],
                  uncertain_policy: str) -> list[dict[str, int]]:
    return [labels_to_binary(d, uncertain_policy) for d in label_dicts]


def clinical_efficacy_f1(pred_labels: list[dict[str, int | None]],
                         ref_labels: list[dict[str, int | None]],
                         uncertain_policy: str = "positive") -> dict:
    """Compute CXRQuant-extractor clinical-efficacy F1 vs references.

    Returns micro/macro precision, recall, F1 over the 13 pathology categories,
    plus a per-category breakdown.
    """
    if len(pred_labels) != len(ref_labels) or not pred_labels:
        raise ValueError("expected nonempty paired pred/ref labels")
    if uncertain_policy not in ("positive", "negative"):
        raise ValueError("uncertain_policy must be positive or negative; ignore is unsupported")
    preds = _binarize_all(pred_labels, uncertain_policy)
    refs = _binarize_all(ref_labels, uncertain_policy)

    per_cat = {}
    tp_tot = fp_tot = fn_tot = 0
    macro_f1_sum = 0.0

    for cat in _PATHOLOGY_CATEGORIES:
        tp = fp = fn = 0
        for p, r in zip(preds, refs, strict=True):
            if p[cat] == 1 and r[cat] == 1:
                tp += 1
            elif p[cat] == 1 and r[cat] == 0:
                fp += 1
            elif p[cat] == 0 and r[cat] == 1:
                fn += 1
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_cat[cat] = {"precision": round(prec, 4), "recall": round(rec, 4),
                        "f1": round(f1, 4), "support": tp + fn}
        tp_tot += tp; fp_tot += fp; fn_tot += fn
        macro_f1_sum += f1

    micro_p = tp_tot / (tp_tot + fp_tot) if (tp_tot + fp_tot) else 0.0
    micro_r = tp_tot / (tp_tot + fn_tot) if (tp_tot + fn_tot) else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    macro_f1 = macro_f1_sum / len(_PATHOLOGY_CATEGORIES)

    return {
        "micro_precision": round(micro_p, 4),
        "micro_recall": round(micro_r, 4),
        "micro_f1": round(micro_f1, 4),
        "macro_f1": round(macro_f1, 4),
        "per_category": per_cat,
        "uncertain_policy": uncertain_policy,
    }


def fact_preservation(quant_labels: list[dict[str, int | None]],
                      fp32_labels: list[dict[str, int | None]],
                      uncertain_policy: str = "positive") -> dict:
    """Compute quantization fact-preservation metrics vs. the FP32 baseline.

    Returns CFPR, CFFR, FNF rate, FPF rate, raw counts, and per-category flips.
    """
    if len(quant_labels) != len(fp32_labels) or not quant_labels:
        raise ValueError("expected nonempty paired quant/fp32 labels")
    if uncertain_policy not in ("positive", "negative"):
        raise ValueError("uncertain_policy must be positive or negative; ignore is unsupported")
    quant = _binarize_all(quant_labels, uncertain_policy)
    base = _binarize_all(fp32_labels, uncertain_policy)

    total = preserved = flips = 0
    fnf = fpf = 0
    present_total = absent_total = 0
    per_cat = {c: {"fnf": 0, "fpf": 0, "present": 0} for c in _PATHOLOGY_CATEGORIES}

    for q, b in zip(quant, base, strict=True):
        for cat in _PATHOLOGY_CATEGORIES:
            total += 1
            bv, qv = b[cat], q[cat]
            if bv == 1:
                present_total += 1
                per_cat[cat]["present"] += 1
            else:
                absent_total += 1
            if qv == bv:
                preserved += 1
            else:
                flips += 1
                if bv == 1 and qv == 0:
                    fnf += 1
                    per_cat[cat]["fnf"] += 1
                elif bv == 0 and qv == 1:
                    fpf += 1
                    per_cat[cat]["fpf"] += 1

    return {
        "CFPR": round(preserved / total, 4) if total else None,
        "CFFR": round(flips / total, 4) if total else None,
        "FNF_rate": round(fnf / present_total, 4) if present_total else None,
        "FPF_rate": round(fpf / absent_total, 4) if absent_total else None,
        "metric_version": "2.1-null-denominator",
        "counts": {
            "total_pairs": total, "preserved": preserved, "flips": flips,
            "false_negative_flips": fnf, "false_positive_flips": fpf,
            "fp32_present": present_total, "fp32_absent": absent_total,
        },
        "per_category_flips": per_cat,
        "uncertain_policy": uncertain_policy,
    }


if __name__ == "__main__":
    from .fact_extractor import extract_labels
    ref = [extract_labels("Mild cardiomegaly. Small pleural effusion."),
           extract_labels("No acute cardiopulmonary abnormality.")]
    fp32 = [extract_labels("Mild cardiomegaly. Small pleural effusion."),
            extract_labels("No acute cardiopulmonary abnormality.")]
    quant = [extract_labels("Mild cardiomegaly. Lungs are clear."),   # dropped effusion (FNF)
             extract_labels("Possible pneumonia.")]                   # added pneumonia (FPF)
    print("CE-F1 (quant vs ref):", clinical_efficacy_f1(quant, ref)["micro_f1"])
    fp = fact_preservation(quant, fp32)
    print("CFPR:", fp["CFPR"], "CFFR:", fp["CFFR"],
          "FNF_rate:", fp["FNF_rate"], "FPF_rate:", fp["FPF_rate"])
    print("counts:", fp["counts"])
