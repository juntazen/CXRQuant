"""Manifest-driven smoke/full GPU execution and CPU replay entry point."""

import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from cxrquant.runio import (
    atomic_json,
    checked_result,
    code_hash,
    object_hash,
    require_identity,
    sha256_file,
    tree_hash,
)
from cxrquant.training_data import encode_prompt, encode_training
from cxrquant.validation import score, validate_benchmark, validate_manifest


def select_devices(environ=None):
    env = os.environ if environ is None else environ
    visible = env.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        values = [x.strip() for x in visible.split(",") if x.strip()]
    else:
        values = [x.strip() for x in env.get("NVIDIA_VISIBLE_DEVICES", "").split(",") if x.strip()]
    if visible is not None and not values:
        return []
    if not values or values == ["all"]:
        import torch

        values = [str(i) for i in range(torch.cuda.device_count())]
    if any(x in ("none", "void", "-1") for x in values):
        return []
    if len(set(values)) != len(values):
        raise ValueError("duplicate GPU allocation")
    return values


def load_data(path, cfg):
    from cxrquant.data.iuxray_dataset import create_splits

    splits = create_splits(json.loads(Path(path).read_text()), seed=cfg["split_seed"])
    out = {}
    for key, rows in splits.items():
        rows = [
            dict(r, findings=r["findings"].strip(), impression=r["impression"].strip())
            for r in rows
            if r.get("findings", "").strip() and r.get("impression", "").strip()
        ]
        out[key] = rows[: cfg.get(key + "_n")] if cfg.get(key + "_n") else rows
    return out


def gpu_preflight():
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.cuda.init()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU_RUN_PENDING: CUDA is unavailable")
    x = torch.ones(8, device="cuda")
    torch.cuda.synchronize()
    if x.sum().item() != 8:
        raise RuntimeError("CUDA numerical preflight failed")
    p = torch.cuda.get_device_properties(0)
    return {
        "name": p.name,
        "memory_bytes": p.total_memory,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "free_total_bytes": list(torch.cuda.mem_get_info()),
    }


def release_memory():
    import torch

    gc.collect()
    torch.cuda.empty_cache()


def build_dataset(tok, rows, limit, mask_prompt=False):
    import torch

    data, audits = [], []
    for row in rows:
        enc, audit = encode_training(tok, row, limit, mask_prompt=mask_prompt)
        data.append({k: torch.tensor(v, dtype=torch.long) for k, v in enc.items()})
        audits.append(audit)
    return data, audits


def train_model(run, spec, cfg, splits, identity):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        default_data_collator,
        set_seed,
    )
    from transformers.trainer_utils import get_last_checkpoint

    name = spec["name"]
    target = run / "checkpoints" / name / "final"
    info_path = run / "models" / name / "training.json"
    if info_path.exists():
        info = checked_result(info_path, identity)
        if tree_hash(target)["sha256"] != info["checkpoint_hash"]:
            raise ValueError("checkpoint was modified")
        return target, info
    revision = spec.get("revision")
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("Resolve and pin the model revision using scripts/prepare_models.py")
    set_seed(cfg["training_seed"])
    tok = AutoTokenizer.from_pretrained(spec["hf_id"], revision=revision, token=False)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "right"
    mask_prompt = cfg.get("loss_mask", "full_sequence") == "impression_only"
    train, taudit = build_dataset(tok, splits["train"], cfg["train_token_limit"], mask_prompt)
    val, vaudit = build_dataset(tok, splits["val"], cfg["train_token_limit"], mask_prompt)
    atomic_json(run / "models" / name / "token_audit.json", {"train": taudit, "val": vaudit})
    if any(a["impression_fully_truncated"] for a in taudit + vaudit):
        raise ValueError(
            "Impression entirely truncated; inspect token_audit and revise token policy in a new run"
        )
    model = AutoModelForCausalLM.from_pretrained(
        spec["hf_id"], revision=revision, token=False, dtype=torch.float32
    ).to("cuda")
    model.config.use_cache = not spec["gradient_checkpointing"]
    trainer_dir = run / "checkpoints" / name / "trainer"
    args = TrainingArguments(
        output_dir=str(trainer_dir),
        num_train_epochs=spec["epochs"],
        per_device_train_batch_size=spec["microbatch"],
        per_device_eval_batch_size=spec["microbatch"],
        gradient_accumulation_steps=spec["accumulation"],
        learning_rate=cfg["learning_rate"],
        optim=cfg["optimizer"],
        adam_beta1=cfg["adam_beta1"],
        adam_beta2=cfg["adam_beta2"],
        adam_epsilon=cfg["adam_epsilon"],
        weight_decay=cfg["weight_decay"],
        max_grad_norm=cfg["max_grad_norm"],
        lr_scheduler_type=cfg["scheduler"],
        warmup_steps=cfg["warmup_steps"],
        seed=cfg["training_seed"],
        data_seed=cfg["data_seed"],
        fp16=True,
        bf16=False,
        tf32=False,
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="no",
        report_to="none",
        dataloader_num_workers=0,
        gradient_checkpointing=spec["gradient_checkpointing"],
        logging_steps=25,
    )
    atomic_json(run / "models" / name / "effective_training_args.json", args.to_dict())
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=val,
        data_collator=default_data_collator,
    )
    last = get_last_checkpoint(str(trainer_dir)) if trainer_dir.exists() else None
    torch.cuda.synchronize()
    start = time.perf_counter()
    trainer.train(resume_from_checkpoint=last)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    eval_loss = float(trainer.evaluate()["eval_loss"])
    # Count shifted, unmasked targets explicitly for token-weighted FP32 validation.
    model.eval()
    nll, targets = 0.0, 0
    with torch.no_grad():
        for batch in DataLoader(
            val, batch_size=spec["microbatch"], collate_fn=default_data_collator
        ):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            labels = batch.pop("labels")[:, 1:].contiguous()
            logits = model(**batch).logits[:, :-1].float().contiguous()
            nll += F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            ).item()
            targets += (labels != -100).sum().item()
    if targets <= 0:
        raise ValueError("validation has no shifted targets")
    import math

    target.mkdir(parents=True, exist_ok=True)
    model.config.use_cache = True
    trainer.save_model(target)
    tok.save_pretrained(target)
    info = {
        "status": "complete",
        "identity": identity,
        "checkpoint_hash": tree_hash(target)["sha256"],
        "checkpoint_files": tree_hash(target)["files"],
        "model_revision": revision,
        "train_time_s_this_invocation": elapsed,
        "resumed_from": last,
        "trainer_eval_loss": eval_loss,
        "token_nll_sum": nll,
        "valid_target_tokens": targets,
        "token_weighted_val_loss": nll / targets,
        "val_perplexity": math.exp(nll / targets),
        "params_m": sum(p.numel() for p in model.parameters()) / 1e6,
        "checkpoint_weight_bytes": sum(p.stat().st_size for p in target.rglob("*.safetensors")),
    }
    atomic_json(info_path, info)
    del trainer, model
    release_memory()
    return target, info


def load_precision(path, config):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    kwargs = {"device_map": {"": 0}, "dtype": torch.float32 if config == "FP32" else torch.float16}
    bnb = None
    if config == "INT8-BnB":
        bnb = BitsAndBytesConfig(
            load_in_8bit=True, llm_int8_threshold=6.0, llm_int8_enable_fp32_cpu_offload=False
        )
    elif config == "NF4-BnB":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif config not in ("FP32", "FP16"):
        raise ValueError("unknown precision")
    if bnb:
        kwargs["quantization_config"] = bnb
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs).eval()
    modules = {
        name: type(mod).__name__
        for name, mod in model.named_modules()
        if type(mod).__name__ in ("Linear8bitLt", "Linear4bit")
    }
    expected = "Linear8bitLt" if config == "INT8-BnB" else "Linear4bit"
    if bnb and (
        expected not in modules.values()
        or not getattr(
            model, "is_loaded_in_8bit" if config == "INT8-BnB" else "is_loaded_in_4bit", False
        )
    ):
        raise RuntimeError("Requested BitsAndBytes backend is not active")
    if any(p.device.type != "cuda" for p in model.parameters()):
        raise RuntimeError("CPU/disk offload is outside this benchmark protocol")
    audit = {
        "quantized_modules": modules,
        "bnb_config": bnb.to_dict() if bnb else None,
        "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
        "device_map": {k: str(v) for k, v in getattr(model, "hf_device_map", {}).items()},
        "size_note": "Only the saved FP32 checkpoint is measured on disk; quantized disk compression is not reported.",
    }
    return model, audit


class EosExemptRepetitionPenalty:
    """HF-equivalent repetition penalty over input_ids that never penalises the EOS token."""

    def __init__(self, penalty, eos_token_id):
        self.penalty, self.eos = float(penalty), eos_token_id

    def __call__(self, input_ids, scores):
        import torch

        score = torch.gather(scores, 1, input_ids)
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)
        keep = input_ids == self.eos
        score = torch.where(keep, torch.gather(scores, 1, input_ids), score)
        return scores.scatter(1, input_ids, score)


def generate(path, config, cfg, rows):
    import torch
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"
    encoded = [encode_prompt(tok, r["findings"], cfg["generation_input_limit"]) for r in rows]
    model, backend = load_precision(path, config)
    decoding = {
        "do_sample": cfg["do_sample"],
        "num_beams": cfg["num_beams"],
        "repetition_penalty": cfg["repetition_penalty"],
        "pad_token_id": tok.pad_token_id,
        "eos_token_id": tok.eos_token_id,
    }
    if cfg.get("repetition_penalty_exempt_eos", False):
        # Tokenizers without a pad token reuse EOS for left padding; the standard repetition
        # penalty then also penalises EOS and suppresses termination. Apply the same penalty to
        # every other token that appears in the context, but never to EOS.
        decoding["repetition_penalty"] = 1.0
        decoding["logits_processor"] = [EosExemptRepetitionPenalty(cfg["repetition_penalty"], tok.eos_token_id)]

    def batch(items):
        return tok.pad({"input_ids": [x[0] for x in items]}, padding=True, return_tensors="pt").to(
            "cuda"
        )

    with torch.no_grad():
        model.generate(
            **batch(encoded[: cfg["warmup_reports"]]),
            max_new_tokens=cfg["warmup_new_tokens"],
            **decoding,
        )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    predictions, raw, token_counts, batch_stats = [], [], [], []
    for i in range(0, len(rows), cfg["generation_batch_size"]):
        enc = batch(encoded[i : i + cfg["generation_batch_size"]])
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            output = model.generate(**enc, max_new_tokens=cfg["max_new_tokens"], **decoding)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        new = output[:, enc["input_ids"].shape[1] :].tolist()
        texts = tok.batch_decode(new, skip_special_tokens=True)
        for ids, text in zip(new, texts, strict=True):
            token_counts.append(
                ids.index(tok.eos_token_id) + 1 if tok.eos_token_id in ids else len(ids)
            )
            raw.append(text)
            clean = text.strip()
            for prefix in ("IMPRESSION:", "Impression:", "impression:"):
                if clean.startswith(prefix):
                    clean = clean[len(prefix) :].strip()
                    break
            predictions.append(clean)
        batch_stats.append({"n_reports": len(new), "latency_s": seconds})
    total_s = sum(x["latency_s"] for x in batch_stats)
    efficiency = {
        "batch_timings": batch_stats,
        "total_generation_s": total_s,
        "amortized_latency_ms": 1000 * total_s / len(rows),
        "reports_per_second": len(rows) / total_s,
        "generated_tokens": sum(token_counts),
        "tokens_per_second": sum(token_counts) / total_s,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "warmup_reports": cfg["warmup_reports"],
        "warmup_new_tokens": cfg["warmup_new_tokens"],
    }
    samples = [
        {
            "sample_id": r["uid"],
            "input": f"FINDINGS: {r['findings']} IMPRESSION:",
            "reference": r["impression"],
            "raw_prediction": a,
            "prediction": b,
            "generated_tokens": n,
            "output_status": "empty" if b == "" else "generated",
            "prompt_audit": enc[1],
        }
        for r, a, b, n, enc in zip(rows, raw, predictions, token_counts, encoded, strict=True)
    ]
    del model
    release_memory()
    return predictions, samples, efficiency, backend


def run_worker(run, names):
    manifest = json.loads((run / "manifest.json").read_text())
    cfg = json.loads((run / "config.json").read_text())
    identity = json.loads((run / "identity.json").read_text())
    splits = load_data(run / "data.json", cfg)
    failures = []
    for spec in cfg["models"]:
        if spec["name"] not in names:
            continue
        name = spec["name"]
        try:
            hardware = gpu_preflight()
            target, info = train_model(run, spec, cfg, splits, identity)
            entries, preds = [], {}
            for precision in cfg["configs"]:
                p = run / "models" / name / (precision + ".json")
                if p.exists():
                    result = checked_result(p, identity, info["checkpoint_hash"])
                else:
                    texts, samples, efficiency, backend = generate(
                        target, precision, cfg, splits["test"]
                    )
                    metrics = score(
                        texts,
                        manifest["references"],
                        preds.get("FP32") if precision != "FP32" else None,
                    )
                    result = {
                        "identity": identity,
                        "status": "complete",
                        "config": precision,
                        "checkpoint_hash": info["checkpoint_hash"],
                        "predictions": texts,
                        "samples": samples,
                        "efficiency": efficiency,
                        "backend": backend,
                        **metrics,
                    }
                    atomic_json(p, result)
                preds[precision] = result["predictions"]
                entries.append(
                    {
                        k: v
                        for k, v in result.items()
                        if k not in ("predictions", "samples", "identity")
                    }
                )
            model = {
                "model": name,
                "status": "complete",
                "identity": identity,
                "hardware": hardware,
                "sample_ids": manifest["sample_ids"],
                "references": manifest["references"],
                "n_test": len(splits["test"]),
                "predictions": preds,
                "prediction_sample_ids": {
                    c: [
                        s["sample_id"]
                        for s in json.loads((run / "models" / name / (c + ".json")).read_text())[
                            "samples"
                        ]
                    ]
                    for c in cfg["configs"]
                },
                "configs": entries,
                "family": spec["family"],
                **{k: v for k, v in info.items() if k not in ("status", "identity")},
            }
            atomic_json(run / "models" / name / "result.json", model)
        except Exception as e:
            traceback.print_exc()
            failures.append(name)
            atomic_json(
                run / "models" / name / "failure.json",
                {"status": "failed", "error": str(e), "identity": identity},
            )
    return 1 if failures else 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full", "replay", "preflight"), required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--data", type=Path)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--worker-models", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    if args.worker_models:
        return run_worker(run, args.worker_models.split(","))
    with (run / ".pipeline.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another launcher holds this run lock")
        if args.mode == "replay":
            from cxrquant.replay import replay

            replay(run)
            return 0
        started = time.time()
        if args.mode == "preflight":
            try:
                hardware = gpu_preflight()
                if not args.config:
                    raise ValueError("--config required for largest-model preflight")
                cfg = json.loads(args.config.read_text())
                spec = next(s for s in cfg["models"] if s["name"] == "biogpt-large")
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                tok = AutoTokenizer.from_pretrained(
                    spec["hf_id"], revision=spec["revision"], token=False
                )
                model = AutoModelForCausalLM.from_pretrained(
                    spec["hf_id"], revision=spec["revision"], token=False, dtype=torch.float32
                ).to("cuda")
                inp = tok("FINDINGS: Lungs are clear. IMPRESSION:", return_tensors="pt").to("cuda")
                model(**inp, labels=inp["input_ids"]).loss.backward()
                for p in model.parameters():
                    p.grad = None
                del model
                release_memory()
                atomic_json(
                    run / "preflight.json",
                    {
                        "status": "complete",
                        "hardware": hardware,
                        "largest_model_forward_backward": True,
                    },
                )
                return 0
            except Exception as e:
                atomic_json(
                    run / "preflight.json",
                    {
                        "status": "GPU_RUN_PENDING",
                        "error": str(e),
                        "duration_s": time.time() - started,
                    },
                )
                return 1
        if not args.config or not args.data:
            raise SystemExit("--config and --data required")
        cfg = json.loads(args.config.read_text())
        from cxrquant.paths import REPO_ROOT

        root = REPO_ROOT
        identity = {
            "run_id": run.name,
            "code_hash": code_hash(root),
            "config_hash": object_hash(cfg),
            "data_hash": sha256_file(args.data),
        }
        if (run / "identity.json").exists():
            if not args.resume:
                raise ValueError("existing run requires --resume")
            require_identity(json.loads((run / "identity.json").read_text()), identity)
        else:
            atomic_json(run / "identity.json", identity)
            atomic_json(run / "config.json", cfg)
            import shutil

            shutil.copyfile(args.data, run / "data.json")
        if (
            sha256_file(run / "data.json") != identity["data_hash"]
            or object_hash(json.loads((run / "config.json").read_text())) != identity["config_hash"]
        ):
            raise ValueError("Persisted run data/config changed")
        splits = load_data(run / "data.json", cfg)
        manifest = {
            "run_id": run.name,
            "models": [m["name"] for m in cfg["models"]],
            "configs": cfg["configs"],
            "sample_ids": [r["uid"] for r in splits["test"]],
            "references": [r["impression"] for r in splits["test"]],
            "split_ids": {k: [r["uid"] for r in v] for k, v in splits.items()},
            "rounding_digits": 4,
            "mode": args.mode,
        }
        validate_manifest(manifest)
        atomic_json(run / "manifest.json", manifest)
        state = {
            "status": "running",
            "pid": os.getpid(),
            "started_at": started,
            "identity": identity,
            "jobs": [],
        }
        atomic_json(run / "status.json", state)
        try:
            devices = select_devices()[:2]
            if not devices:
                raise RuntimeError("No allocated CUDA devices")
            # Do not initialize CUDA in the launcher. On MIG systems NVML may
            # enumerate multiple allocated UUIDs while one process can create
            # a CUDA context for only one. Each worker performs its own real
            # tensor preflight after CUDA_VISIBLE_DEVICES isolates one UUID.
            hardware = {
                "launcher_visible_devices": devices,
                "worker_preflight": "recorded per model after device isolation",
            }
        except Exception as e:
            state.update(
                status="GPU_RUN_PENDING",
                error=str(e),
                duration_s=time.time() - started,
                exit_code=1,
            )
            atomic_json(run / "status.json", state)
            return 1
        groups = [manifest["models"][i :: len(devices)] for i in range(len(devices))]
        processes = []
        for device, names in zip(devices, groups, strict=True):
            if not names:
                continue
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=device,
                HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
                TOKENIZERS_PARALLELISM="false",
            )
            env["OMP_NUM_THREADS"] = "4"
            cmd = [
                sys.executable,
                "-m",
                "cxrquant.pipeline",
                "--mode",
                args.mode,
                "--run-dir",
                str(run),
                "--worker-models",
                ",".join(names),
            ]
            log = (run / ("worker_" + str(len(processes)) + ".log")).open("a")
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            processes.append((proc, log))
            state["jobs"].append(
                {"pid": proc.pid, "device": device, "models": names, "command": cmd}
            )
            atomic_json(run / "status.json", state)
        codes = [p.wait() for p, log in processes]
        for p, log in processes:
            log.close()
        state["worker_exit_codes"] = codes
        try:
            if any(c != 0 for c in codes):
                raise RuntimeError("required worker failed")
            models = [
                checked_result(run / "models" / name / "result.json", identity)
                for name in manifest["models"]
            ]
            bench = {
                "run_id": run.name,
                "identity": identity,
                "hardware_preflight": hardware,
                "models": models,
            }
            state["matrix"] = validate_benchmark(bench, manifest)
            atomic_json(run / "multimodel_benchmark.json", bench)
            from cxrquant.replay import replay

            replay(run)
            state.update(status="complete", exit_code=0)
        except Exception as e:
            traceback.print_exc()
            state.update(status="partial", error=str(e), exit_code=1)
        state["duration_s"] = time.time() - started
        atomic_json(run / "status.json", state)
        return state["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
