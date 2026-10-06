"""Behavioral regression tests independent of historical model conclusions."""

import copy
import json
import subprocess
import sys

import pytest

from cxrquant.runio import atomic_json, checked_result, require_identity, tree_hash
from cxrquant.training_data import encode_prompt, encode_training
from cxrquant.validation import score, validate_benchmark
from cxrquant.pipeline import select_devices
from cxrquant.replay import exact_mcnemar, report_counts, replay
from cxrquant.clinical_safety.fact_extractor import extract_labels
from cxrquant.clinical_safety.safety_metrics import fact_preservation


@pytest.fixture
def capsule():
    refs = ["Pneumonia.", "Lungs are clear."]
    manifest = {
        "run_id": "test",
        "models": ["tiny"],
        "configs": ["FP32", "FP16", "INT8-BnB", "NF4-BnB"],
        "sample_ids": ["a", "b"],
        "references": refs,
        "rounding_digits": 4,
    }
    preds = {c: list(refs) for c in manifest["configs"]}
    preds["NF4-BnB"] = ["", ""]
    configs = [
        dict(config=c, status="complete", **score(p, refs, refs if c != "FP32" else None))
        for c, p in preds.items()
    ]
    bench = {
        "run_id": "test",
        "models": [
            {
                "model": "tiny",
                "status": "complete",
                "n_test": 2,
                "sample_ids": ["a", "b"],
                "prediction_sample_ids": {c: ["a", "b"] for c in manifest["configs"]},
                "references": refs,
                "configs": configs,
                "predictions": preds,
            }
        ],
    }
    return bench, manifest


def test_empty_strings_are_counted(capsule):
    bench, manifest = capsule
    assert validate_benchmark(bench, manifest) == {
        "models": 1,
        "configurations": 4,
        "predictions": 8,
        "comparisons": 3,
    }
    assert bench["models"][0]["configs"][-1]["bleu_4"] == 0


@pytest.mark.parametrize(
    "damage",
    [
        "nf4_missing",
        "fp32_missing",
        "model_missing",
        "duplicate_model",
        "length",
        "sample_order",
        "reference_order",
        "metric_missing",
        "metric_nan",
        "config_error",
        "duplicate_config",
        "missing_counts",
        "wrong_counts",
        "nonstring",
        "empty_array",
        "wrong_run",
        "partial",
        "null_rate",
        "config_sample_order",
    ],
)
def test_verifier_rejects_incomplete_or_corrupt(capsule, damage):
    bench, manifest = capsule
    m = bench["models"][0]
    if damage == "nf4_missing":
        del m["predictions"]["NF4-BnB"]
    elif damage == "fp32_missing":
        del m["predictions"]["FP32"]
    elif damage == "model_missing":
        bench["models"] = []
    elif damage == "duplicate_model":
        bench["models"].append(copy.deepcopy(m))
    elif damage == "length":
        m["predictions"]["FP16"].append("x")
    elif damage == "sample_order":
        m["sample_ids"].reverse()
    elif damage == "reference_order":
        m["references"] = list(reversed(m["references"]))
    elif damage == "metric_missing":
        del m["configs"][-1]["FNF_rate"]
    elif damage == "metric_nan":
        m["configs"][-1]["FNF_rate"] = float("nan")
    elif damage == "config_error":
        m["configs"][-1]["error"] = "OOM"
    elif damage == "duplicate_config":
        m["configs"].append(copy.deepcopy(m["configs"][-1]))
    elif damage == "missing_counts":
        del m["configs"][-1]["safety_counts"]
    elif damage == "wrong_counts":
        m["configs"][-1]["safety_counts"]["false_negative_flips"] = 3
    elif damage == "nonstring":
        m["predictions"]["FP16"][0] = None
    elif damage == "empty_array":
        m["predictions"]["FP16"] = []
    elif damage == "wrong_run":
        bench["run_id"] = "other"
    elif damage == "partial":
        m["status"] = "partial"
    elif damage == "null_rate":
        m["configs"][-1]["FNF_rate"] = None
    elif damage == "config_sample_order":
        m["prediction_sample_ids"]["FP16"].reverse()
    with pytest.raises(ValueError):
        validate_benchmark(bench, manifest)


class CharTokenizer:
    eos_token = "~"
    eos_token_id = ord("~")
    pad_token_id = ord("~")

    def encode(self, text, **kwargs):
        return [ord(x) for x in text]

    def decode(self, ids):
        return "".join(chr(x) for x in ids)


def test_padding_mask_retains_real_eos_and_collator():
    pytest.importorskip("transformers")
    from transformers import default_data_collator

    tok = CharTokenizer()
    row, audit = encode_training(tok, {"uid": "a", "findings": "clear", "impression": "normal"}, 80)
    import torch

    batch = default_data_collator([{k: torch.tensor(v) for k, v in row.items()}])
    assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()
    assert (batch["labels"][batch["attention_mask"] == 1] == tok.eos_token_id).sum() == 1
    assert audit["valid_shifted_tokens"] > 0
    assert audit["impression_tokens_kept"] > 0


def test_truncated_impression_is_detected():
    _, audit = encode_training(
        CharTokenizer(), {"uid": "a", "findings": "x" * 100, "impression": "normal"}, 30
    )
    assert audit["impression_fully_truncated"]
    with pytest.raises(ValueError):
        encode_training(CharTokenizer(), {"uid": "a", "findings": "x", "impression": "y"}, 1)


def test_generation_preserves_impression_marker():
    tok = CharTokenizer()
    ids, audit = encode_prompt(tok, "x" * 200, 40)
    assert tok.decode(ids).endswith(" IMPRESSION:")
    assert len(ids) == 40 and audit["legacy_marker_lost"]


def test_null_denominator_and_unsupported_uncertainty():
    labels = [extract_labels("Lungs are clear.")]
    assert fact_preservation(labels, labels)["FNF_rate"] is None
    with pytest.raises(ValueError):
        fact_preservation(labels, labels, "ignore")


def test_safe_allocation():
    assert select_devices({"CUDA_VISIBLE_DEVICES": ""}) == []
    assert select_devices(
        {"CUDA_VISIBLE_DEVICES": "MIG-a", "NVIDIA_VISIBLE_DEVICES": "MIG-b,MIG-c"}
    ) == ["MIG-a"]
    assert select_devices({"NVIDIA_VISIBLE_DEVICES": "MIG-a, MIG-b"}) == ["MIG-a", "MIG-b"]
    assert select_devices({"CUDA_VISIBLE_DEVICES": "-1"}) == []
    with pytest.raises(ValueError):
        select_devices({"CUDA_VISIBLE_DEVICES": "0,0"})


def test_resume_identity_and_checkpoint_guard(tmp_path):
    path = tmp_path / "result.json"
    identity = {"code": "a", "config": "b", "data": "c"}
    atomic_json(path, {"status": "complete", "identity": identity, "checkpoint_hash": "d"})
    assert checked_result(path, identity, "d")["status"] == "complete"
    with pytest.raises(ValueError):
        checked_result(path, identity, "wrong")
    with pytest.raises(ValueError):
        require_identity(identity, dict(identity, code="changed"))
    atomic_json(path, {"status": "partial", "identity": identity})
    with pytest.raises(ValueError):
        checked_result(path, identity)
    assert not list(tmp_path.glob(".result.json*"))


def test_mcnemar_handles_large_counts_without_underflow():
    assert exact_mcnemar(10, 0) == pytest.approx(2 / 1024)
    assert exact_mcnemar(2000, 2000) == 1
    with pytest.raises(ValueError):
        report_counts([{}], [])


def test_pooled_regeneration_and_run_paths(tmp_path, capsule):
    bench, manifest = capsule
    run = tmp_path / "nested" / "run"
    atomic_json(run / "manifest.json", manifest)
    atomic_json(run / "multimodel_benchmark.json", bench)
    atomic_json(run / "config.json", {"bootstrap_seed": 42, "bootstrap_resamples": 50})
    atomic_json(run / "mcnemar_pooled.json", {"stale": True})
    replay(run)
    pooled = json.loads((run / "mcnemar_pooled.json").read_text())
    assert "stale" not in pooled and len(pooled["rows"]) == 3
    assert pooled["rows"][-1]["b"] == 1
    first = tree_hash(run)["sha256"]
    replay(run)
    assert tree_hash(run)["sha256"] == first
    assert not (tmp_path / "mcnemar_pooled.json").exists()


def test_worker_failure_exits_nonzero(tmp_path):
    # Empty run: no manifest. The real child entry point must propagate failure.
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "cxrquant.pipeline",
            "--mode",
            "smoke",
            "--run-dir",
            str(tmp_path),
            "--worker-models",
            "tiny",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert not (tmp_path / "multimodel_benchmark.json").exists()


def test_cli_rejects_corrupt_matrix_under_optimized_python(tmp_path, capsule):
    bench, manifest = capsule
    del bench["models"][0]["predictions"]["NF4-BnB"]
    atomic_json(tmp_path / "manifest.json", manifest)
    atomic_json(tmp_path / "multimodel_benchmark.json", bench)
    proc = subprocess.run(
        [sys.executable, "-O", "-m", "cxrquant.cli", "verify", "--results", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1
