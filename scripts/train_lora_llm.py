#!/usr/bin/env python3
"""Fine-tune a 7B instruction model for FINDINGS -> IMPRESSION with LoRA, then merge to one FP32 source checkpoint.

Revision R2-2 (stronger generator). Same partition (465/68/134, split seed 42) and prompt as the benchmark,
loss restricted to IMPRESSION tokens and EOS. LoRA is trained in bf16; the adapters are merged into the
FP32 base weights and saved as one checkpoint, which the standard four-precision audit then re-executes.

  python scripts/train_lora_llm.py --base mistralai/Mistral-7B-Instruct-v0.2 --run-dir runs/full_seed42_corrected_20260911 \
      --out ../results/llm7b/checkpoint --seed 42
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    a = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, set_seed

    from cxrquant.pipeline import load_data
    from cxrquant.training_data import encode_training

    set_seed(a.seed)
    run = Path(a.run_dir)
    cfg = json.loads((run / "config.json").read_text())
    splits = load_data(run / "data.json", cfg)
    tok = AutoTokenizer.from_pretrained(a.base)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "right"

    def encode(rows):
        data = []
        for r in rows:
            enc, _ = encode_training(tok, r, cfg["train_token_limit"], mask_prompt=True)
            data.append({k: torch.tensor(v) for k, v in enc.items()})
        return data

    train, val = encode(splits["train"]), encode(splits["val"])
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16).to("cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lcfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lcfg)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    bs, accum = 4, 4
    steps = math.ceil(len(train) / (bs * accum)) * a.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    g = torch.Generator().manual_seed(a.seed)
    t0, log = time.time(), []
    model.train()
    for ep in range(a.epochs):
        order = torch.randperm(len(train), generator=g).tolist()
        for i in range(0, len(order), bs):
            batch = {k: torch.stack([train[j][k] for j in order[i:i + bs]]).to("cuda") for k in train[0]}
            loss = model(**batch).loss / accum
            loss.backward()
            if (i // bs + 1) % accum == 0 or i + bs >= len(order):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(), sched.step(), opt.zero_grad()
            log.append(float(loss) * accum)
        model.eval()
        nll = n = 0
        with torch.no_grad():
            for i in range(0, len(val), bs):
                batch = {k: torch.stack([x[k] for x in val[i:i + bs]]).to("cuda") for k in val[0]}
                out = model(**batch)
                t = (batch["labels"][:, 1:] != -100).sum().item()
                nll += out.loss.item() * t
                n += t
        model.train()
        print(f"epoch {ep + 1} train_loss {sum(log[-50:]) / len(log[-50:]):.4f} val_ppl {math.exp(nll / n):.3f}", flush=True)
    model = model.merge_and_unload().to(torch.float32)
    out = Path(a.out)
    (out / "final").mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out / "final", safe_serialization=True)
    tok.save_pretrained(out / "final")
    (out / "training.json").write_text(json.dumps({
        "base": a.base, "seed": a.seed, "epochs": a.epochs, "lr": a.lr, "lora_rank": a.rank, "lora_alpha": 2 * a.rank,
        "target_modules": lcfg.target_modules and sorted(lcfg.target_modules), "loss": "IMPRESSION tokens and EOS only",
        "batch": bs * accum, "train_time_s": time.time() - t0, "val_perplexity": math.exp(nll / n),
        "saved_dtype": "float32 (LoRA merged)"}, indent=1))
    print("saved", out / "final")


if __name__ == "__main__":
    main()
