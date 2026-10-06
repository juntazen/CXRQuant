#!/usr/bin/env python3
"""Summaries of the revision GPU experiments, read from their stored outputs.

  eosfix     decoding-corrected reruns (EOS exempt from the repetition penalty), five seeds
  ablation   impression-only loss ablation, seed 42
  llm7b      LoRA-fine-tuned 7B instruction model, four precisions
  chexagent  image-conditioned CheXagent-8b, FINDINGS generation, four precisions

  python scripts/summarise_reruns.py --results ../results --out ../results/rerun_summary.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from cxrquant.clinical_safety.fact_extractor import _PATHOLOGY_CATEGORIES, extract_labels, labels_to_binary
from cxrquant.evaluation.nlg_metrics import corpus_bleu

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ["FP32", "FP16", "INT8-BnB", "NF4-BnB"]
SHORT = {"FP16": "FP16", "INT8-BnB": "INT8", "NF4-BnB": "NF4"}
_cache = {}


def lab(t):
    if t not in _cache:
        _cache[t] = labels_to_binary(extract_labels(t), "positive")
    return _cache[t]


def paired(base_txt, q_txt, draws=2000, seed=42):
    base, q = [lab(x) for x in base_txt], [lab(x) for x in q_txt]
    per = np.array([[sum(b[k] for k in _PATHOLOGY_CATEGORIES), sum(b[k] and not y[k] for k in _PATHOLOGY_CATEGORIES),
                     sum(1 - b[k] for k in _PATHOLOGY_CATEGORIES), sum((not b[k]) and y[k] for k in _PATHOLOGY_CATEGORIES)]
                    for b, y in zip(base, q)])
    tot = per.sum(0)
    idx = np.random.default_rng(seed).integers(0, len(per), size=(draws, len(per)))
    s = per[idx].sum(1)
    fnf = np.where(s[:, 0] > 0, s[:, 1] / np.maximum(s[:, 0], 1), np.nan)
    fpf = s[:, 3] / s[:, 2]
    return {"fp32_pos": int(tot[0]), "b": int(tot[1]), "c": int(tot[3]),
            "FNF": round(tot[1] / tot[0], 4) if tot[0] else None,
            "FNF_ci": [round(float(np.nanpercentile(fnf, 2.5)), 4), round(float(np.nanpercentile(fnf, 97.5)), 4)],
            "FPF": round(tot[3] / tot[2], 4),
            "FPF_ci": [round(float(np.percentile(fpf, 2.5)), 4), round(float(np.percentile(fpf, 97.5)), 4)],
            "reports_changed": int(sum(any(b[k] != y[k] for k in _PATHOLOGY_CATEGORIES) for b, y in zip(base, q)))}


def model_dir_summary(mdir):
    recs = {c: json.loads((mdir / f"{c}.json").read_text()) for c in CONFIGS if (mdir / f"{c}.json").exists()}
    if len(recs) < 4:
        return None
    refs = [s["reference"] for s in recs["FP32"]["samples"]]
    out = {"fp32_bleu4": recs["FP32"]["bleu_4"],
           "fp32_hit_limit": sum(s["generated_tokens"] >= 64 for s in recs["FP32"]["samples"]),
           "fp32_mean_tokens": round(float(np.mean([s["generated_tokens"] for s in recs["FP32"]["samples"]])), 1)}
    for c in CONFIGS[1:]:
        r = paired(recs["FP32"]["predictions"], recs[c]["predictions"])
        assert abs((r["FNF"] or 0) - (recs[c]["FNF_rate"] or 0)) < 1e-3, (mdir, c)
        r.update(bleu4=recs[c]["bleu_4"], d_bleu4_pp=round(100 * (recs[c]["bleu_4"] - recs["FP32"]["bleu_4"]), 3),
                 d_ce_micro_pp=round(100 * (recs[c]["ce_micro_f1"] - recs["FP32"]["ce_micro_f1"]), 2),
                 ms_per_report=round(recs[c]["efficiency"]["amortized_latency_ms"], 1),
                 peak_mib=round(recs[c]["efficiency"]["peak_allocated_bytes"] / 2**20))
        out[SHORT[c]] = r
    out["FP32_ms_per_report"] = round(recs["FP32"]["efficiency"]["amortized_latency_ms"], 1)
    out["FP32_peak_mib"] = round(recs["FP32"]["efficiency"]["peak_allocated_bytes"] / 2**20)
    v = [out[p]["FNF"] for p in ("FP16", "INT8", "NF4")]
    out["strict"] = bool(None not in v and v[0] < v[1] < v[2])
    out["non_decreasing"] = bool(None not in v and v[0] <= v[1] <= v[2])
    return out


def collection(models_root):
    res = {}
    for mdir in sorted(Path(models_root).glob("*")):
        if mdir.is_dir():
            s = model_dir_summary(mdir)
            if s:
                res[mdir.name] = s
    return res


def acceptance(summ):
    acc = [(m, p) for m, s in summ.items() for p in ("FP16", "INT8", "NF4") if abs(s[p]["d_bleu4_pp"]) <= 1.0]
    return {"bleu_1pp_accepted": len(acc), "of": 3 * len(summ),
            "accepted_with_fnf_ge_30": sum(summ[m][p]["FNF"] >= 0.30 for m, p in acc)}


NIH_MAP = {"Atelectasis": "Atelectasis", "Cardiomegaly": "Cardiomegaly", "Effusion": "Pleural Effusion",
           "Edema": "Edema", "Consolidation": "Consolidation", "Pneumonia": "Pneumonia", "Pneumothorax": "Pneumothorax",
           "Mass": "Lung Lesion", "Nodule": "Lung Lesion", "Infiltration": "Lung Opacity"}


def nih_reference(pred_texts, label_strings):
    """Micro-F1 of extracted categories against NIH image labels on the ten mappable categories."""
    cats = sorted(set(NIH_MAP.values()))
    tp = fp = fn = 0
    for t, labs in zip(pred_texts, label_strings):
        gold = {NIH_MAP[x] for x in labs.split("|") if x in NIH_MAP}
        pred = {c for c in cats if lab(t)[c]}
        tp += len(gold & pred); fp += len(pred - gold); fn += len(gold - pred)
    return round(2 * tp / (2 * tp + fp + fn), 4) if tp else 0.0


def chexagent(d, nih_labels=False):
    d = Path(d)
    runs = {}
    for c in CONFIGS:
        f = d / f"{c}.jsonl"
        if f.exists():
            runs[c] = {json.loads(x)["uid"]: json.loads(x) for x in f.read_text().splitlines() if x.strip()}
    if "FP32" not in runs:
        return None
    out = {"n_complete": {c: len(v) for c, v in runs.items()}}
    common = sorted(set.intersection(*[set(v) for v in runs.values()]))
    out["n_common"] = len(common)
    base = [runs["FP32"][u]["prediction"] for u in common]
    refs = [runs["FP32"][u]["reference_findings"] or "" for u in common]
    keep = [i for i, r in enumerate(refs) if r.strip()]
    if keep:
        out["fp32_bleu4_vs_findings"] = round(corpus_bleu([base[i] for i in keep], [[refs[i]] for i in keep])["bleu_4"], 4)
    labs = [runs["FP32"][u]["reference_impression"] or "" for u in common]
    if nih_labels:
        out["fp32_nih_label_microf1"] = nih_reference(base, labs)
    for c in CONFIGS:
        if c not in runs:
            continue
        q = [runs[c][u]["prediction"] for u in common]
        meta = json.loads((d / f"{c}.meta.json").read_text()) if (d / f"{c}.meta.json").exists() else {}
        e = {"s_per_report": round(float(np.mean([runs[c][u]["latency_s"] for u in common])), 3),
             "peak_mib": round(meta.get("peak_alloc_mib", float("nan")))}
        if c != "FP32":
            e.update(paired(base, q))
            if keep:
                e["bleu4_vs_findings"] = round(corpus_bleu([q[i] for i in keep], [[refs[i]] for i in keep])["bleu_4"], 4)
            if nih_labels:
                e["nih_label_microf1"] = nih_reference(q, labs)
            e["identical_text"] = sum(x == y for x, y in zip(base, q))
        out[c] = e
    if all(c in out for c in CONFIGS[1:]):
        v = [out[c]["FNF"] for c in CONFIGS[1:]]
        out["strict"] = bool(v[0] < v[1] < v[2])
        out["non_decreasing"] = bool(v[0] <= v[1] <= v[2])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    R = {}
    ef = {}
    for sd in sorted((a.results / "eosfix").glob("seed*")):
        s = collection(sd / "models")
        if len(s) == 6:
            ef[sd.name] = {"models": s, "strict": sum(v["strict"] for v in s.values()),
                           "non_decreasing": sum(v["non_decreasing"] for v in s.values()), **acceptance(s)}
    R["eosfix"] = ef
    abl = ROOT / "runs/ablation_impression_only_seed42/models"
    if abl.exists():
        s = collection(abl)
        tr = {m: json.loads((abl / m / "training.json").read_text()).get("val_perplexity")
              for m in s if (abl / m / "training.json").exists()}
        R["ablation_impression_only_seed42"] = {"models": s, "val_perplexity": tr,
                                               "strict": sum(v["strict"] for v in s.values()),
                                               "non_decreasing": sum(v["non_decreasing"] for v in s.values()),
                                               **(acceptance(s) if s else {})}
    llm = a.results / "llm7b/audit/models"
    if llm.exists():
        R["llm7b"] = {"models": collection(llm)}
        tj = a.results / "llm7b/mistral-7b/training.json"
        if tj.exists():
            R["llm7b"]["training"] = json.loads(tj.read_text())
    cx = a.results / "chexagent"
    if cx.exists():
        R["chexagent"] = chexagent(cx)
    nih = a.results / "chexagent_nih"
    if nih.exists():
        R["chexagent_nih"] = chexagent(nih, nih_labels=True)
    a.out.write_text(json.dumps(R, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps({k: (v if k != "eosfix" else {s: {kk: vv for kk, vv in x.items() if kk != "models"} for s, x in v.items()})
                      for k, v in R.items() if k in ("eosfix",)}, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    main()
