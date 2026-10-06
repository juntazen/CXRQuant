"""CPU tests for the IJIES extension analyses (ablation, strata, seeds, CheXbert).

Synthetic inputs lock the logic; the archived-run tests re-derive the reported
ablation facts from the shipped corrected run and skip when it is absent.
"""

import csv
import importlib.util
import json

import pytest

from cxrquant.clinical_safety.fact_extractor import CHEXPERT_CATEGORIES
from cxrquant.clinical_safety.safety_metrics import fact_preservation
from cxrquant.paths import REPO_ROOT

RUN = REPO_ROOT / "runs" / "full_seed42_corrected_20260911"
FUNCTIONAL = {"biogpt", "pythia-1b", "biogpt-large"}


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def labels(**kwargs):
    out = {c: None for c in CHEXPERT_CATEGORIES}
    out.update(kwargs)
    return out


def valid_rate(value):
    return value is None or 0 <= value <= 1


def test_uncertain_policy_changes_flip_rates_and_keeps_rates_valid():
    base = [labels(Pneumonia=-1, Edema=1), labels(Cardiomegaly=1)]
    quant = [labels(Pneumonia=0, Edema=1), labels(Cardiomegaly=-1)]
    pos = fact_preservation(quant, base, "positive")
    neg = fact_preservation(quant, base, "negative")
    assert (pos["FNF_rate"], pos["FPF_rate"]) != (neg["FNF_rate"], neg["FPF_rate"])
    for result in (pos, neg):
        assert valid_rate(result["FNF_rate"]) and valid_rate(result["FPF_rate"])
    # Under U-Zeros the uncertain-only FP32 positive disappears from the denominator.
    assert pos["counts"]["fp32_present"] == 3 and neg["counts"]["fp32_present"] == 2


def test_uncertain_policy_null_denominator_is_preserved():
    base = [labels(Pneumonia=-1)]
    quant = [labels(Pneumonia=-1)]
    assert fact_preservation(quant, base, "negative")["FNF_rate"] is None
    assert fact_preservation(quant, base, "positive")["FNF_rate"] == 0


def write_metrics(path, rows):
    fields = ["model", "config", "FNF_rate", "FPF_rate", "CFPR", "CFFR", "ce_micro_f1",
              "ce_macro_f1", "bleu_1", "bleu_4", "rouge_1", "rouge_2", "rouge_l"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def synthetic_metrics(scale=0.0):
    rows = []
    for model, bleu in (("good", 0.18), ("bad", 0.01)):
        rows.append({"model": model, "config": "FP32", "bleu_4": bleu, "ce_micro_f1": 0.3})
        for i, cfg in enumerate(("FP16", "INT8-BnB", "NF4-BnB")):
            rows.append({"model": model, "config": cfg, "FNF_rate": 0.1 * (i + 1) + scale,
                         "FPF_rate": 0.01, "CFPR": 0.9, "bleu_4": bleu, "ce_micro_f1": 0.3})
    return rows


def test_stratum_classification_and_threshold_gap():
    strata = load_script("ablation_stratum")
    rows = [{"model": m, "config": "FP32", "bleu_4": str(b)} for m, b in (("a", 0.01), ("b", 0.2))]
    assert strata.classify(rows) == {"functional": ["b"], "degenerate": ["a"]}
    with pytest.raises(ValueError):
        strata.classify(rows + [{"model": "c", "config": "FP32", "bleu_4": "0.08"}])


@pytest.mark.skipif(not (RUN / "tables" / "metrics.csv").exists(), reason="corrected run absent")
def test_functional_stratum_is_exactly_the_three_non_degenerate_models():
    strata = load_script("ablation_stratum")
    result = strata.classify(strata.read_csv(RUN / "tables" / "metrics.csv"))
    assert set(result["functional"]) == FUNCTIONAL
    assert set(result["degenerate"]) == {"distilgpt2", "gpt2", "gpt2-medium"}


@pytest.mark.skipif(not (RUN / "models").exists(), reason="corrected run absent")
def test_archived_ordering_holds_under_both_uncertain_policies():
    ablation = load_script("ablation_uncertain_policy")
    manifest, _, records = ablation.load_run(RUN)
    rows, _ = ablation.ablate(manifest, records)
    for policy in ablation.POLICIES:
        order = ablation.ordering(rows, manifest["models"], policy)
        assert all(v["strict_FP16_lt_INT8_lt_NF4"] for v in order.values())


def fake_run(root, name, seed, scale, data_hash="d" * 64, sample_ids=("s1", "s2")):
    run = root / name
    (run / "tables").mkdir(parents=True)
    cfg = {"training_seed": seed, "data_seed": seed, "split_seed": 42, "bootstrap_seed": 42}
    manifest = {"sample_ids": list(sample_ids), "references": ["r1", "r2"],
                "split_ids": {"test": list(sample_ids)}, "models": ["good", "bad"],
                "configs": ["FP32", "FP16", "INT8-BnB", "NF4-BnB"]}
    for file, value in (
        ("config.json", cfg),
        ("manifest.json", manifest),
        ("identity.json", {"run_id": name, "data_hash": data_hash}),
        ("status.json", {"status": "complete", "worker_exit_codes": [0, 0]}),
    ):
        (run / file).write_text(json.dumps(value))
    write_metrics(run / "tables" / "metrics.csv", synthetic_metrics(scale))
    return run


def test_aggregate_seeds_mean_sd_and_ordering(tmp_path):
    agg = load_script("aggregate_seeds")
    runs = [agg.load(fake_run(tmp_path, f"r{s}", s, x)) for s, x in ((1, 0.0), (2, 0.1))]
    seeds, rows, ordering = agg.aggregate(runs)
    nf4 = next(r for r in rows if r["model"] == "good" and r["config"] == "NF4-BnB")
    assert seeds == [1, 2]
    assert nf4["FNF_rate_mean"] == pytest.approx(0.35)
    assert nf4["FNF_rate_sd"] == pytest.approx(0.0707, abs=1e-4)
    assert ordering["good"]["seeds_with_strict_order"] == 2


@pytest.mark.parametrize(
    "change", [{"data_hash": "e" * 64}, {"sample_ids": ("s2", "s1")}]
)
def test_aggregate_seeds_rejects_different_held_out_sets(tmp_path, change):
    agg = load_script("aggregate_seeds")
    a = agg.load(fake_run(tmp_path, "a", 1, 0.0))
    b = agg.load(fake_run(tmp_path, "b", 2, 0.0, **change))
    with pytest.raises(ValueError):
        agg.aggregate([a, b])


def test_aggregate_seeds_rejects_config_drift_beyond_seed(tmp_path):
    agg = load_script("aggregate_seeds")
    a = fake_run(tmp_path, "a", 1, 0.0)
    b = fake_run(tmp_path, "b", 2, 0.0)
    cfg = json.loads((b / "config.json").read_text())
    cfg["split_seed"] = 7
    (b / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        agg.aggregate([agg.load(a), agg.load(b)])


def test_chexbert_decode_maps_to_cxrquant_encoding():
    from cxrquant.clinical_safety.chexbert_labeler import CHEXBERT_ORDER, decode

    classes = [0] * 14
    classes[CHEXBERT_ORDER.index("Pneumonia")] = 3
    classes[CHEXBERT_ORDER.index("Edema")] = 1
    classes[CHEXBERT_ORDER.index("Pleural Effusion")] = 2
    classes[CHEXBERT_ORDER.index("No Finding")] = 1
    out = decode(classes)
    assert list(out) == CHEXPERT_CATEGORIES
    assert out["Pneumonia"] == -1 and out["Edema"] == 1 and out["Pleural Effusion"] == 0
    assert out["No Finding"] == 1 and out["Cardiomegaly"] is None
    classes[CHEXBERT_ORDER.index("No Finding")] = 3
    with pytest.raises(ValueError):
        decode(classes)


def test_chexbert_labeler_runs_when_checkpoint_is_available():
    import os

    checkpoint = os.environ.get("CXRQUANT_CHEXBERT_CHECKPOINT")
    if not checkpoint or not os.path.exists(checkpoint):
        pytest.skip("CheXbert checkpoint not provided (optional dependency)")
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from cxrquant.clinical_safety.chexbert_labeler import CheXbertLabeler

    out = CheXbertLabeler(checkpoint=checkpoint).extract(
        ["Moderate cardiomegaly.", "No pleural effusion or pneumothorax."]
    )
    assert out[0]["Cardiomegaly"] == 1
    assert out[1]["Pleural Effusion"] == 0 and out[1]["Pneumothorax"] == 0


def test_divergence_analysis_localises_flips_after_first_difference():
    div = load_script("divergence_analysis")
    assert div.first_divergence("a b c".split(), "a b c".split()) is None
    assert div.first_divergence("a b c".split(), "a x c".split()) == 1
    assert div.first_divergence("a b".split(), "a b c".split()) == 2
    base = ["No acute findings. Mild cardiomegaly.", "Lungs are clear."]
    quant = ["No acute findings. Small left pleural effusion.", "Lungs are clear."]
    s = div.analyse_pair(base, quant)
    assert s["identical_text"] == 1 and s["diverged"] == 1 and s["flips_in_identical_text"] == 0
    assert s["b"] == 1 and s["c"] == 1 and s["decided_after_divergence"] == 2


def test_logit_margin_divergence_helpers():
    pytest.importorskip("numpy")
    spec = importlib.util.spec_from_file_location("lm", REPO_ROOT / "scripts" / "logit_margin_analysis.py")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError:
        pytest.skip("pipeline dependencies unavailable")
    assert module.first_divergence([1, 2, 3], [1, 2, 3]) is None
    assert module.first_divergence([1, 2, 3], [1, 9, 3]) == 1
    base = [([1, 2, 3], [5.0, 0.01, 4.0])]
    quant = [([1, 7, 3], [5.0, 0.2, 4.0])]
    r = module.analyse(base, quant, ["Mild cardiomegaly."], ["Mild cardiomegaly."])
    assert r["diverged_reports"] == 1 and r["median_fp32_margin_at_divergence"] == 0.01
