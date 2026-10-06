#!/usr/bin/env python3
"""Differential audit re-run with pyConTextNLP on the categories it can express.

pyConTextNLP is applied with the target and modifier rules of
``scripts/extractor_comparison.py`` to every stored prediction and reference of
one run. Its output covers the nine categories of that comparator and counts only
definite positive, non-negated mentions. For a like-for-like comparison the
CXRQuant extractor is scored on the same nine categories with uncertain mapped to
negative. For both extractors the script reports FP32-relative FNF and FPF, the
FP16 < INT8 < NF4 ordering, NF4 discordant counts and Cohen's kappa between the two
extractors. Requires pyConTextNLP and networkx; writes ``pycontext_audit.json``.
"""
import argparse
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path

from cxrquant.clinical_safety.fact_extractor import extract_labels, labels_to_binary
from cxrquant.paths import REPO_ROOT
from cxrquant.runio import atomic_json, sha256_file

REDUCED = ("FP16", "INT8-BnB", "NF4-BnB")
_FN = None
_CATS = None


def _load_comparator():
    spec = importlib.util.spec_from_file_location("extractor_comparison", REPO_ROOT / "scripts" / "extractor_comparison.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _init():
    global _FN, _CATS
    module = _load_comparator()
    _CATS = module.SCORABLE
    for name, fn, note in module.build_external():
        if name.startswith("pyConTextNLP") and callable(fn):
            _FN = fn
    if _FN is None:
        raise RuntimeError("pyConTextNLP is not available")


def _label(text):
    return sorted(_FN(text))


def rates(base, quant, cats):
    b = c = present = absent = 0
    for x, y in zip(base, quant, strict=True):
        for cat in cats:
            if x[cat]:
                present += 1
                b += int(not y[cat])
            else:
                absent += 1
                c += int(bool(y[cat]))
    return {"FNF_rate": round(b / present, 4) if present else None, "FPF_rate": round(c / absent, 4) if absent else None,
            "b": b, "c": c, "fp32_present": present}


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return {"agreement": round(po, 4), "kappa": None if pe == 1 else round((po - pe) / (1 - pe), 4), "n": n}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    run = args.run_dir.resolve()
    out = run / "pycontext_audit.json"
    if out.exists() and not args.overwrite:
        raise SystemExit("output exists; pass --overwrite to regenerate")
    manifest = json.loads((run / "manifest.json").read_text())
    preds = {(m, c): json.loads((run / "models" / m / (c + ".json")).read_text())["predictions"]
             for m in manifest["models"] for c in manifest["configs"]}
    texts = sorted(set(manifest["references"]).union(*[set(v) for v in preds.values()]))
    _init()
    cats = list(_CATS)
    with mp.get_context("fork").Pool(args.workers) as pool:
        labelled = dict(zip(texts, pool.map(_label, texts, chunksize=8), strict=True))
    pyc = {t: {cat: cat in set(v) for cat in cats} for t, v in labelled.items()}
    cxr = {t: {cat: bool(labels_to_binary(extract_labels(t), "negative")[cat]) for cat in cats} for t in texts}
    extractors = {"pycontextnlp": pyc, "cxrquant_9cat_uncertain_negative": cxr}
    result = {"analysis": "differential audit with pyConTextNLP on nine categories", "categories": cats,
              "positive_policy": "definite, non-negated mentions (uncertain counted as absent) for both extractors",
              "per_extractor": {}, "provenance": {"run_id": manifest["run_id"], "script_hash": sha256_file(__file__),
                                                   "unique_texts": len(texts)}}
    for name, lab in extractors.items():
        per_model, ordering = {}, {}
        for m in manifest["models"]:
            base = [lab[t] for t in preds[m, "FP32"]]
            per_model[m] = {c: rates(base, [lab[t] for t in preds[m, c]], cats) for c in REDUCED}
            vals = [per_model[m][c]["FNF_rate"] for c in REDUCED]
            ordering[m] = {"strict": all(v is not None for v in vals) and vals[0] < vals[1] < vals[2],
                           "non_decreasing": all(v is not None for v in vals) and vals[0] <= vals[1] <= vals[2]}
        result["per_extractor"][name] = {"per_model": per_model, "ordering": ordering}
    pred_texts = [t for v in preds.values() for t in v]
    a = [int(pyc[t][cat]) for t in pred_texts for cat in cats]
    b = [int(cxr[t][cat]) for t in pred_texts for cat in cats]
    result["agreement_predictions_pooled"] = kappa(a, b)
    atomic_json(out, result)
    summary = {name: {"ordering_strict": sum(v["strict"] for v in r["ordering"].values()),
                      "ordering_non_decreasing": sum(v["non_decreasing"] for v in r["ordering"].values()),
                      "nf4_fnf": {m: r["per_model"][m]["NF4-BnB"]["FNF_rate"] for m in manifest["models"]}}
               for name, r in result["per_extractor"].items()}
    print(json.dumps({"summary": summary, "agreement": result["agreement_predictions_pooled"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
