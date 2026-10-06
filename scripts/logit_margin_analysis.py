#!/usr/bin/env python3
"""Decision margins at the point where reduced-precision generation diverges from FP32.

Regenerates the held-out impressions of one run from its retained checkpoints with
the archived prompts, batch composition and greedy decoding, recording for every
generated token the processed log-probability margin between the top-1 and top-2
candidates. For each report the first token at which a reduced-precision output
departs from the FP32 output is located, and the FP32 margin at that step is
compared with FP32 margins at steps where both precisions agreed.

Tests the hypothesis that divergence starts at near-tie decisions (small margins)
rather than at confident ones. Checkpoint hashes are verified before use and the
agreement of the regenerated text with the archived predictions is reported. No
timing is measured. Requires CUDA; writes ``logit_margin_analysis.json``.
"""
import argparse
import json
import math
from pathlib import Path
from statistics import median

from cxrquant.clinical_safety.fact_extractor import extract_labels, labels_to_binary, _PATHOLOGY_CATEGORIES
from cxrquant.pipeline import load_data, load_precision
from cxrquant.runio import atomic_json, sha256_file, tree_hash
from cxrquant.training_data import encode_prompt

CONFIGS = ["FP32", "FP16", "INT8-BnB", "NF4-BnB"]


def generate_with_margins(path, config, cfg, rows):
    import torch
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"
    encoded = [encode_prompt(tok, r["findings"], cfg["generation_input_limit"]) for r in rows]
    model, _ = load_precision(path, config)
    out_tokens, out_margins, texts = [], [], []
    for i in range(0, len(rows), cfg["generation_batch_size"]):
        chunk = encoded[i : i + cfg["generation_batch_size"]]
        enc = tok.pad({"input_ids": [x[0] for x in chunk]}, padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=cfg["max_new_tokens"], do_sample=cfg["do_sample"],
                                 num_beams=cfg["num_beams"], repetition_penalty=cfg["repetition_penalty"],
                                 pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                                 output_scores=True, return_dict_in_generate=True)
        new = gen.sequences[:, enc["input_ids"].shape[1]:]
        logp = torch.stack([torch.log_softmax(s.float(), dim=-1) for s in gen.scores], dim=1)  # batch, steps, vocab
        top2 = torch.topk(logp, 2, dim=-1).values
        margins = (top2[..., 0] - top2[..., 1]).cpu().tolist()
        for b, ids in enumerate(new.tolist()):
            n = ids.index(tok.eos_token_id) + 1 if tok.eos_token_id in ids else len(ids)
            out_tokens.append(ids[:n])
            out_margins.append([round(x, 5) for x in margins[b][:n]])
        for text in tok.batch_decode(new, skip_special_tokens=True):
            clean = text.strip()
            for prefix in ("IMPRESSION:", "Impression:", "impression:"):
                if clean.startswith(prefix):
                    clean = clean[len(prefix):].strip()
                    break
            texts.append(clean)
        del gen, logp, top2
    del model
    torch.cuda.empty_cache()
    return out_tokens, out_margins, texts


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def quantile_bins(values, k=5):
    s = sorted(values)
    return [s[min(len(s) - 1, int(len(s) * q / k))] for q in range(1, k)]


def analyse(base, quant, base_texts, quant_texts):
    div_margins_fp32, div_margins_q, agree_margins, hazard_rows = [], [], [], []
    diverged = 0
    flips_in_diverged = flips_in_identical = 0
    for (t32, m32), (tq, mq), x32, xq in zip(base, quant, base_texts, quant_texts, strict=True):
        d = first_divergence(t32, tq)
        b32 = labels_to_binary(extract_labels(x32))
        bq = labels_to_binary(extract_labels(xq))
        flips = sum(b32[c] != bq[c] for c in _PATHOLOGY_CATEGORIES)
        if d is None:
            agree_margins.extend(m32)
            hazard_rows.extend((m, 0) for m in m32)
            flips_in_identical += flips
            continue
        diverged += 1
        flips_in_diverged += flips
        agree_margins.extend(m32[:d])
        hazard_rows.extend((m, 0) for m in m32[:d])
        if d < len(m32):
            div_margins_fp32.append(m32[d])
            hazard_rows.append((m32[d], 1))
        if d < len(mq):
            div_margins_q.append(mq[d])
    cuts = quantile_bins([m for m, _ in hazard_rows]) if hazard_rows else []
    bins = [[0, 0] for _ in range(len(cuts) + 1)]
    for m, event in hazard_rows:
        k = sum(m > c for c in cuts)
        bins[k][0] += 1
        bins[k][1] += event
    low_share = (sum(m <= cuts[0] for m in div_margins_fp32) / len(div_margins_fp32)) if div_margins_fp32 and cuts else None
    return {
        "reports": len(base), "diverged_reports": diverged,
        "flips_in_diverged_reports": flips_in_diverged, "flips_in_identical_token_reports": flips_in_identical,
        "median_fp32_margin_at_divergence": round(median(div_margins_fp32), 4) if div_margins_fp32 else None,
        "median_quant_margin_at_divergence": round(median(div_margins_q), 4) if div_margins_q else None,
        "median_fp32_margin_at_agreeing_steps": round(median(agree_margins), 4) if agree_margins else None,
        "share_divergences_in_lowest_margin_quintile": round(low_share, 4) if low_share is not None else None,
        "hazard_by_fp32_margin_quintile": [
            {"quintile": i + 1, "steps": n, "divergences": e, "rate": round(e / n, 5) if n else None} for i, (n, e) in enumerate(bins)
        ],
        "margin_quintile_cuts": [round(c, 4) for c in cuts],
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--checkpoint-root", type=Path, required=True)
    ap.add_argument("--models", default="distilgpt2,gpt2,gpt2-medium,biogpt,pythia-1b,biogpt-large")
    ap.add_argument("--memory-fraction", type=float, default=0.25, help="cap on this process's share of device memory")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    run = args.run_dir.resolve()
    out_path = args.out or run / "logit_margin_analysis.json"
    cfg = json.loads((run / "config.json").read_text())
    rows = load_data(run / "data.json", cfg)["test"]
    manifest = json.loads((run / "manifest.json").read_text())
    if [r["uid"] for r in rows] != manifest["sample_ids"]:
        raise ValueError("held-out order differs from the archived manifest")
    result = {"analysis": "top-1/top-2 log-probability margin at first token divergence", "models": {},
              "hardware": {"device": torch.cuda.get_device_name(0), "memory_fraction_cap": args.memory_fraction},
              "provenance": {"run_id": manifest["run_id"], "script_hash": sha256_file(__file__)}}
    for name in args.models.split(","):
        training = json.loads((run / "models" / name / "training.json").read_text())
        ckpt = args.checkpoint_root / name / "final"
        if tree_hash(ckpt)["sha256"] != training["checkpoint_hash"]:
            raise ValueError(f"checkpoint hash mismatch for {name}")
        gens = {}
        for config in CONFIGS:
            tokens, margins, texts = generate_with_margins(ckpt, config, cfg, rows)
            archived = json.loads((run / "models" / name / (config + ".json")).read_text())["predictions"]
            gens[config] = {"tokens": tokens, "margins": margins, "texts": texts,
                            "reproduced_archived_text": sum(a == b for a, b in zip(texts, archived, strict=True))}
            print(name, config, "reproduced", gens[config]["reproduced_archived_text"], "/", len(texts), flush=True)
        base = list(zip(gens["FP32"]["tokens"], gens["FP32"]["margins"]))
        result["models"][name] = {
            "checkpoint_hash_verified": True,
            "reproduced_archived_text": {c: gens[c]["reproduced_archived_text"] for c in CONFIGS},
            "comparisons": {c: analyse(base, list(zip(gens[c]["tokens"], gens[c]["margins"])), gens["FP32"]["texts"], gens[c]["texts"])
                            for c in CONFIGS[1:]},
        }
        atomic_json(out_path, result)
    pooled = {}
    for c in CONFIGS[1:]:
        comps = [m["comparisons"][c] for m in result["models"].values()]
        pooled[c] = {
            "diverged_reports": sum(x["diverged_reports"] for x in comps), "reports": sum(x["reports"] for x in comps),
            "flips_in_identical_token_reports": sum(x["flips_in_identical_token_reports"] for x in comps),
            "hazard_by_fp32_margin_quintile": [
                {"quintile": q + 1, "steps": sum(x["hazard_by_fp32_margin_quintile"][q]["steps"] for x in comps if len(x["hazard_by_fp32_margin_quintile"]) > q),
                 "divergences": sum(x["hazard_by_fp32_margin_quintile"][q]["divergences"] for x in comps if len(x["hazard_by_fp32_margin_quintile"]) > q)}
                for q in range(5)],
        }
    result["pooled_within_model_quintiles"] = pooled
    atomic_json(out_path, result)
    print(json.dumps(pooled, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
