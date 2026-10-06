#!/usr/bin/env python3
"""Confirmatory audit with an image-conditioned chest X-ray model (revision R2-2).

StanfordAIMI/CheXagent-8b (pinned snapshot) generates the FINDINGS section from the frontal IU-XRAY
radiograph of every report in the 667-report benchmark corpus, under FP32, FP16, BitsAndBytes INT8
and NF4, from the same source checkpoint. Greedy decoding, batch size 1 (no padding), at most 160 new
tokens, identical prompt. Runs in a Python 3.11 / transformers 4.40.2 environment because the model's
remote code predates transformers 5. Scoring is done separately by scripts/score_generations.py.

  CUDA_VISIBLE_DEVICES=<MIG> python scripts/chexagent_audit.py --precision FP32 --out ../results/chexagent
"""
import argparse
import glob
import io
import json
import os
import time
from pathlib import Path

PROMPT = "Write the findings section of the radiology report for this chest X-ray."
MODEL = "StanfordAIMI/CheXagent-8b"


def load_images(parquet_dir, uids):
    import pandas as pd

    want = {int(u[3:]) for u in uids}
    frames = []
    for f in sorted(glob.glob(os.path.join(parquet_dir, "data", "*.parquet"))):
        df = pd.read_parquet(f, columns=["uid", "img_frontal", "findings", "impression"])
        frames.append(df[df.uid.isin(want)])
    df = pd.concat(frames)
    return {f"CXR{int(r.uid)}": r for r in df.itertuples()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", required=True, choices=["FP32", "FP16", "INT8-BnB", "NF4-BnB"])
    ap.add_argument("--corpus", default=str(Path(__file__).resolve().parents[1] / "../results/split_ids.csv"))
    ap.add_argument("--parquet", default=str(Path(__file__).resolve().parents[1] / "../external/openi_hf"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--source", choices=["iuxray", "nih"], default="iuxray")
    ap.add_argument("--nih-dir", default=str(Path(__file__).resolve().parents[1] / "../external/nih"))
    a = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig, GenerationConfig

    import csv
    from types import SimpleNamespace

    if a.source == "iuxray":
        rows = list(csv.DictReader(open(a.corpus)))
        order = {"test": 0, "val": 1, "train": 2}
        rows.sort(key=lambda r: (order[r["partition"]], r["uid"]))
        uids = [r["uid"] for r in rows]
        if a.limit:
            uids = uids[: a.limit]
        imgs = load_images(a.parquet, uids)
    else:  # NIH ChestX-ray14 subset: image-level labels only, no report text
        sub = json.loads((Path(a.nih_dir) / "subset.json").read_text())["images"]
        if a.limit:
            sub = sub[: a.limit]
        uids = [m["image"] for m in sub]
        imgs = {m["image"]: SimpleNamespace(img_frontal=(Path(a.nih_dir) / "images" / m["image"]).read_bytes(),
                                            findings="", impression="|".join(m["labels"])) for m in sub}
    missing = [u for u in uids if u not in imgs or imgs[u].img_frontal is None]
    uids = [u for u in uids if u not in missing]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{a.precision}.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(x)["uid"] for x in path.read_text().splitlines() if x.strip()}

    kwargs = {"trust_remote_code": True, "device_map": {"": 0}}
    if a.precision == "FP32":
        kwargs["torch_dtype"] = torch.float32
    else:
        kwargs["torch_dtype"] = torch.float16
    if a.precision == "INT8-BnB":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
    if a.precision == "NF4-BnB":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_use_double_quant=True,
                                                           bnb_4bit_compute_dtype=torch.float16)
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, **kwargs).eval()
    dtype = torch.float32 if a.precision == "FP32" else torch.float16
    gen = GenerationConfig.from_pretrained(MODEL)
    gen.num_beams, gen.do_sample, gen.max_new_tokens, gen.max_length = 1, False, a.max_new_tokens, None
    quantized = sorted({type(m).__name__ for m in model.modules() if type(m).__name__ in ("Linear8bitLt", "Linear4bit")})
    meta = {"model": MODEL, "precision": a.precision, "prompt": PROMPT, "n_requested": len(uids) + len(missing), "source": a.source,
            "missing_images": missing, "quantized_module_types": quantized,
            "n_quantized_modules": sum(type(m).__name__ in ("Linear8bitLt", "Linear4bit") for m in model.modules()),
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "device": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "generation": {"num_beams": 1, "do_sample": False, "max_new_tokens": a.max_new_tokens}}
    (out / f"{a.precision}.meta.json").write_text(json.dumps(meta, indent=1))
    torch.cuda.reset_peak_memory_stats()
    with path.open("a") as fh:
        for u in uids:
            if u in done:
                continue
            image = Image.open(io.BytesIO(imgs[u].img_frontal)).convert("RGB")
            inputs = processor(images=[image], text=f" USER: <s>{PROMPT} ASSISTANT: <s>", return_tensors="pt")
            inputs = {k: (v.to("cuda", dtype=dtype) if v.dtype.is_floating_point else v.to("cuda")) for k, v in inputs.items()}
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                output = model.generate(**inputs, generation_config=gen)[0]
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            text = processor.tokenizer.decode(output, skip_special_tokens=True)
            fh.write(json.dumps({"uid": u, "prediction": text.strip(), "reference_findings": imgs[u].findings,
                                 "reference_impression": imgs[u].impression, "latency_s": dt,
                                 "n_tokens": int(output.shape[-1])}) + "\n")
            fh.flush()
    meta["peak_alloc_mib"] = torch.cuda.max_memory_allocated() / 2**20
    (out / f"{a.precision}.meta.json").write_text(json.dumps(meta, indent=1))
    print("done", a.precision)


if __name__ == "__main__":
    main()
