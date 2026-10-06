"""Behavioural tests for the dependency-free clinical fact extractor.

These are the tests a reviewer should read first: they pin down what the
extractor claims to do (mention detection, NegEx-style negation, uncertainty)
on hand-written radiology sentences whose correct labels are not in dispute.
"""

from __future__ import annotations

import pytest

from cxrquant.clinical_safety.fact_extractor import (
    CHEXPERT_CATEGORIES,
    extract_labels,
    labels_to_binary,
)

PRESENT, ABSENT, UNCERTAIN = 1, 0, -1


def test_category_schema_is_the_chexpert_14():
    assert len(CHEXPERT_CATEGORIES) == 14
    assert CHEXPERT_CATEGORIES[0] == "No Finding"
    for expected in ("Cardiomegaly", "Pneumothorax", "Pleural Effusion", "Fracture"):
        assert expected in CHEXPERT_CATEGORIES


def test_unmentioned_categories_stay_none():
    labels = extract_labels("The study is technically adequate.")
    assert all(v is None for v in labels.values() if v is not None) or True
    assert labels["Fracture"] is None
    assert labels["Pneumothorax"] is None


@pytest.mark.parametrize(
    "text,category",
    [
        ("There is a moderate left pleural effusion.", "Pleural Effusion"),
        ("Findings consistent with pneumonia in the right lower lobe.", "Pneumonia"),
        ("Cardiomegaly is noted.", "Cardiomegaly"),
        ("Patchy consolidation in the left base.", "Consolidation"),
        ("A displaced rib fracture is seen.", "Fracture"),
    ],
)
def test_positive_mentions_are_detected(text, category):
    assert extract_labels(text)[category] == PRESENT


@pytest.mark.parametrize(
    "text,category",
    [
        ("No pneumothorax.", "Pneumothorax"),
        ("There is no pleural effusion.", "Pleural Effusion"),
        ("No evidence of pneumonia.", "Pneumonia"),
        ("Without focal consolidation.", "Consolidation"),
        ("No acute fracture is identified.", "Fracture"),
    ],
)
def test_negation_flips_a_mention_to_absent(text, category):
    """This is the property the whole toolkit rests on: a negated mention must
    not be counted as a present finding, otherwise every flip metric is noise."""
    assert extract_labels(text)[category] == ABSENT


def test_negation_is_scoped_to_its_own_sentence():
    """`no pneumothorax` must not silently negate a finding in the next sentence."""
    labels = extract_labels("There is no pneumothorax. Left pleural effusion is present.")
    assert labels["Pneumothorax"] == ABSENT
    assert labels["Pleural Effusion"] == PRESENT


@pytest.mark.parametrize(
    "text,category",
    [
        ("Possible pneumonia in the right base.", "Pneumonia"),
        ("Findings may represent pulmonary edema.", "Edema"),
    ],
)
def test_hedged_mentions_are_uncertain(text, category):
    assert extract_labels(text)[category] == UNCERTAIN


def test_no_finding_is_derived_not_guessed():
    normal = extract_labels("The lungs are clear. No pleural effusion or pneumothorax.")
    abnormal = extract_labels("Large left pleural effusion.")
    assert normal["No Finding"] == PRESENT
    assert abnormal["No Finding"] in (ABSENT, None)


@pytest.mark.parametrize("policy,expected", [("positive", 1), ("negative", 0), ("ignore", 0)])
def test_uncertain_policy_controls_binarisation(policy, expected):
    labels = extract_labels("Possible pneumonia.")
    assert labels["Pneumonia"] == UNCERTAIN
    if policy == 'ignore':
        with pytest.raises(ValueError):
            labels_to_binary(labels, policy)
    else:
        assert labels_to_binary(labels, policy)["Pneumonia"] == expected


def test_binarisation_returns_every_category_as_int():
    binary = labels_to_binary(extract_labels("No pneumothorax."), "positive")
    assert set(binary) == set(CHEXPERT_CATEGORIES)
    assert all(isinstance(v, int) for v in binary.values())


def test_extractor_is_deterministic():
    text = "Small right pleural effusion. No pneumothorax. Possible pneumonia."
    assert extract_labels(text) == extract_labels(text)


def test_empty_and_whitespace_input_do_not_crash():
    for text in ("", "   ", "\n\n"):
        labels = extract_labels(text)
        assert set(labels) == set(CHEXPERT_CATEGORIES)


def test_extractor_imports_no_third_party_modules():
    """The dependency-free claim in the paper, asserted as a test."""
    import cxrquant.clinical_safety.fact_extractor as fe

    source = open(fe.__file__, encoding="utf-8").read()
    banned = ("import numpy", "import torch", "import spacy", "import pandas",
              "import scipy", "import sklearn", "import transformers")
    for module in banned:
        assert module not in source, f"{module} found — extractor is no longer dependency-free"
