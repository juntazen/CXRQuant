"""Reproducibility tests — the ones that matter for the paper.

Each test re-derives a published number from the artefacts committed in this
repository, on CPU, with no network access. If any of these fail, a claim in the
manuscript is no longer supported by the shipped data.
"""

from __future__ import annotations

import json

import pytest

from cxrquant.clinical_safety.fact_extractor import extract_labels
from cxrquant.clinical_safety.safety_metrics import fact_preservation
from cxrquant.paths import DATA_JSON, REPO_ROOT

# These hypotheses belong only to the frozen legacy capsule, never to a rerun.
RESULTS_DIR = REPO_ROOT / "historical" / "results"

BENCH = RESULTS_DIR / "multimodel_benchmark.json"

pytestmark = pytest.mark.skipif(
    not BENCH.exists(), reason="results/multimodel_benchmark.json not present"
)


@pytest.fixture(scope="module")
def bench():
    return json.loads(BENCH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def models(bench):
    return [m for m in bench["models"] if "predictions" in m and "configs" in m]


def test_six_models_with_stored_predictions(models):
    assert len(models) == 6, "the paper benchmarks six decoders"


def test_every_model_has_all_four_precisions(models):
    for m in models:
        assert set(m["predictions"]) >= {"FP32", "FP16", "INT8-BnB", "NF4-BnB"}, m["model"]


def test_test_set_is_the_published_n_134(models):
    for m in models:
        assert len(m["predictions"]["FP32"]) == 134, m["model"]
        assert m["n_test"] == 134, m["model"]


def test_data_split_reproduces_n_134_from_the_shipped_corpus():
    """The held-out N=134 must fall out of the committed corpus + fixed seed,
    not out of a file the reader does not have."""
    if not DATA_JSON.exists():
        pytest.skip("data/iuxray_paired.json not present")
    from cxrquant.data.iuxray_dataset import create_splits

    records = json.loads(DATA_JSON.read_text(encoding="utf-8"))
    splits = create_splits(records, seed=42)
    usable = [
        r for r in splits["test"]
        if r.get("findings", "").strip() and r.get("impression", "").strip()
    ]
    assert len(usable) == 134


@pytest.mark.parametrize("config", ["FP16", "INT8-BnB", "NF4-BnB"])
def test_stored_fnf_is_recomputable_from_stored_predictions(models, config):
    """Recompute each published FNF from the raw predictions and compare with
    the value recorded in the results table."""
    for m in models:
        base = [extract_labels(p) for p in m["predictions"]["FP32"]]
        quant = [extract_labels(p) for p in m["predictions"][config]]
        recomputed = fact_preservation(quant, base)["FNF_rate"]
        stored = next(c["FNF_rate"] for c in m["configs"] if c["config"] == config)
        assert recomputed == pytest.approx(stored, abs=1e-6), f"{m['model']} / {config}"


def test_headline_claim_nf4_drops_41_to_78_percent(models):
    """The abstract's headline range, asserted directly."""
    rates = []
    for m in models:
        base = [extract_labels(p) for p in m["predictions"]["FP32"]]
        quant = [extract_labels(p) for p in m["predictions"]["NF4-BnB"]]
        rates.append(fact_preservation(quant, base)["FNF_rate"])
    assert min(rates) == pytest.approx(0.41, abs=0.01), f"min FNF was {min(rates):.3f}"
    assert max(rates) == pytest.approx(0.78, abs=0.01), f"max FNF was {max(rates):.3f}"


def test_nf4_is_never_safer_than_fp16(models):
    """Direction of the effect, model by model."""
    for m in models:
        base = [extract_labels(p) for p in m["predictions"]["FP32"]]
        fp16 = fact_preservation([extract_labels(p) for p in m["predictions"]["FP16"]], base)
        nf4 = fact_preservation([extract_labels(p) for p in m["predictions"]["NF4-BnB"]], base)
        assert nf4["FNF_rate"] >= fp16["FNF_rate"], m["model"]


def test_bleu4_stays_within_two_points_while_facts_are_lost(models):
    """The 'aggregate NLG masks clinical loss' claim: BLEU-4 barely moves."""
    for m in models:
        fp32 = next(c for c in m["configs"] if c["config"] == "FP32")
        nf4 = next(c for c in m["configs"] if c["config"] == "NF4-BnB")
        delta = abs(nf4["bleu_4"] - fp32["bleu_4"]) * 100
        assert delta <= 2.0, f"{m['model']}: BLEU-4 moved {delta:.2f} points"


def test_bootstrap_cis_cover_their_point_estimates():
    path = RESULTS_DIR / "bootstrap_fnf_ci.json"
    if not path.exists():
        pytest.skip("bootstrap_fnf_ci.json not present")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["models"], "no bootstrap rows"
    for row in payload["models"]:
        lo, point, hi = row["ci95_low"], row["nf4_fnf_point"], row["ci95_high"]
        assert lo <= point <= hi, f"{row['model']}: {lo} <= {point} <= {hi} violated"
        assert 0.0 <= lo < hi <= 1.0, row["model"]
