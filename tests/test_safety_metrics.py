"""Tests for the quantization-safety metrics (CFPR / CFFR / FNF / FPF).

The metrics are built from synthetic label vectors with known answers, so a
failure here means the metric definition changed — not that a model changed.
"""

from __future__ import annotations

import pytest

from cxrquant.clinical_safety.fact_extractor import CHEXPERT_CATEGORIES
from cxrquant.clinical_safety.safety_metrics import clinical_efficacy_f1, fact_preservation


def labels(**kwargs):
    """Build a full 14-category label dict; unnamed categories are 'not mentioned'."""
    out = {c: None for c in CHEXPERT_CATEGORIES}
    out.update(kwargs)
    return out


def test_identical_outputs_preserve_every_fact():
    base = [labels(Pneumonia=1, **{"Pleural Effusion": 0})]
    fp = fact_preservation(base, base)
    assert fp["CFPR"] == pytest.approx(1.0)
    assert fp["CFFR"] == pytest.approx(0.0)

    assert fp["FNF_rate"] == pytest.approx(0.0)
    assert fp["FPF_rate"] == pytest.approx(0.0)


def test_dropped_finding_counts_as_false_negative_flip():
    base = [labels(Pneumonia=1)]
    quant = [labels(Pneumonia=0)]
    fp = fact_preservation(quant, base)
    assert fp["FNF_rate"] == pytest.approx(1.0), "a dropped present finding must score FNF = 1"
    assert fp["FPF_rate"] == pytest.approx(0.0)


def test_invented_finding_counts_as_false_positive_flip():
    """FPF is normalised over every category the baseline did NOT report (13 of
    them here), so one invented finding scores 1/13 — not 1.0. Pinning this
    denominator down is the point of the test: it is what keeps FPF and FNF
    asymmetric, exactly as the paper describes them."""
    base = [labels(Pneumonia=0)]
    quant = [labels(Pneumonia=1)]
    fp = fact_preservation(quant, base)
    assert fp["counts"]["false_positive_flips"] == 1
    assert fp["counts"]["fp32_absent"] == 13
    assert fp["FPF_rate"] == pytest.approx(1 / 13, abs=1e-4)
    assert fp["FNF_rate"] is None  # no positive baseline denominator


def test_fnf_is_a_fraction_of_present_findings_only():
    """Two present findings, one dropped -> FNF = 0.5, regardless of how many
    categories were never mentioned."""
    base = [labels(Pneumonia=1, Edema=1)]
    quant = [labels(Pneumonia=1, Edema=0)]
    assert fact_preservation(quant, base)["FNF_rate"] == pytest.approx(0.5)


def test_preservation_and_flip_rates_are_complementary():
    base = [labels(Pneumonia=1, Edema=1, Fracture=0)]
    quant = [labels(Pneumonia=0, Edema=1, Fracture=1)]
    fp = fact_preservation(quant, base)
    assert fp["CFPR"] + fp["CFFR"] == pytest.approx(1.0)


def test_metrics_survive_a_report_with_no_findings_at_all():
    base = [labels()]
    quant = [labels()]
    fp = fact_preservation(quant, base)
    assert fp["FNF_rate"] in (0.0, None)


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError):
        clinical_efficacy_f1([labels(Pneumonia=1)], [labels(Pneumonia=1), labels(Edema=1)])


def test_clinical_efficacy_perfect_prediction_scores_one():
    ref = [labels(Pneumonia=1, Edema=0)]
    out = clinical_efficacy_f1(ref, ref)
    assert out["micro_f1"] == pytest.approx(1.0)


def test_clinical_efficacy_completely_wrong_prediction_scores_zero():
    ref = [labels(Pneumonia=1)]
    pred = [labels(Pneumonia=0, Edema=1)]
    out = clinical_efficacy_f1(pred, ref)
    assert out["micro_f1"] == pytest.approx(0.0)
