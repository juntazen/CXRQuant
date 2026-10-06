"""Resolve public model revisions and audit real tokenizers without loading weights."""

import argparse
import json
from pathlib import Path
import time

from cxrquant.runio import atomic_json, sha256_file
from cxrquant.training_data import encode_prompt, encode_training


def main():
    from huggingface_hub import HfApi, constants
    from transformers import AutoTokenizer, TrainingArguments
    from cxrquant.data.iuxray_dataset import create_splits

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    splits = create_splits(json.loads(Path(args.data).read_text()), seed=cfg["split_seed"])
    splits = {
        k: [r for r in v if r.get("findings", "").strip() and r.get("impression", "").strip()]
        for k, v in splits.items()
    }
    report = {"data_hash": sha256_file(args.data), "models": [], "status": "partial"}
    for spec in cfg["models"]:
        start = time.time()
        rec = {"model": spec["name"], "hf_id": spec["hf_id"]}
        try:
            revision = spec.get("revision")
            if not revision:
                ref = (
                    Path(constants.HF_HUB_CACHE)
                    / ("models--" + spec["hf_id"].replace("/", "--"))
                    / "refs/main"
                )
                if ref.exists():
                    revision = ref.read_text().strip()
                    rec["revision_source"] = "existing local Hugging Face cache ref"
                else:
                    revision = HfApi(token=False).model_info(spec["hf_id"], timeout=20).sha
                    rec["revision_source"] = "public Hub model_info"
            tok = AutoTokenizer.from_pretrained(spec["hf_id"], revision=revision, token=False)
            tok.pad_token = tok.pad_token or tok.eos_token
            spec["revision"] = revision
            rec.update(
                revision=revision,
                tokenizer_class=type(tok).__name__,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
                splits={},
            )
            for split, rows in splits.items():
                rec["splits"][split] = [
                    encode_training(tok, r, cfg["train_token_limit"])[1] for r in rows
                ]
            rec["generation"] = [
                dict(
                    sample_id=r["uid"],
                    **encode_prompt(tok, r["findings"], cfg["generation_input_limit"])[1],
                )
                for r in splits["test"]
            ]
            rec["status"] = "complete"
        except Exception as e:
            rec.update(status="failed", error=str(e))
        rec["duration_s"] = time.time() - start
        report["models"].append(rec)
        atomic_json(args.out, report)
        print(spec["name"], rec["status"], rec.get("revision", rec.get("error")), flush=True)
    report["status"] = (
        "complete" if all(r["status"] == "complete" for r in report["models"]) else "partial"
    )
    fields = TrainingArguments.__dataclass_fields__
    report["installed_trainer_defaults_not_historical_proof"] = {
        k: str(fields[k].default)
        for k in (
            "learning_rate",
            "adam_beta1",
            "adam_beta2",
            "adam_epsilon",
            "seed",
            "data_seed",
            "lr_scheduler_type",
        )
    }
    atomic_json(args.out, report)
    atomic_json(args.config, cfg)
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
