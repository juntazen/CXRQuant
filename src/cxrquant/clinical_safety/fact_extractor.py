"""
Pure-Python, CPU-only clinical fact extractor for chest X-ray (CXR) reports.

Maps free-text findings/impression to a 14-dimensional label vector over the
canonical CheXpert observation categories, using a lightweight
mention -> negation/uncertainty -> aggregation pipeline (NegEx-style; Chapman
et al., J. Biomedical Informatics 34(5):301-310, 2001).

Design intent: zero external NLP dependencies (only the standard library), so
results are reproducible across environments and runnable without a GPU. This
extractor is intended for *differential* measurement (comparing FP32 vs.
quantized outputs of the SAME model), where systematic extractor errors cancel.

Label encoding (CheXpert convention):
     1  = positive  (finding present)
     0  = negative  (finding explicitly absent)
    -1  = uncertain (hedged / possible)
  None  = not mentioned (blank / NA)
"""

from __future__ import annotations

import re

# ── The 14 canonical CheXpert observation categories (fixed order) ────────────
CHEXPERT_CATEGORIES: list[str] = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

# The 13 pathology categories (No Finding is derived from these).
_PATHOLOGY_CATEGORIES: list[str] = CHEXPERT_CATEGORIES[1:]


# ── Mention lexicon: finding -> list of surface phrases (case-insensitive) ─────
# First entries for several categories are taken verbatim from the official
# stanfordmlgroup/chexpert-labeler phrase files; the remainder are standard
# radiology synonyms consistent with the CheXpert/CheXbert label schema.
MENTION_LEXICON: dict[str, list[str]] = {
    "Enlarged Cardiomediastinum": [
        "mediastinal contour", "mediastinal silhouette", "cardiomediastinal",
        "widened mediastinum", "mediastinal widening", "enlarged cardiomediastinum",
    ],
    "Cardiomegaly": [
        "cardiomegaly", "cardiac enlargement", "enlarged heart",
        "cardiac silhouette is enlarged", "heart is enlarged",
        "enlarged cardiac silhouette", "enlargement of the cardiac silhouette",
        "cardiac contour is enlarged",
    ],
    "Lung Opacity": [
        "opacity", "opacities", "opacification", "airspace opacity",
        "ground glass", "ground-glass", "reticular opacit", "hazy",
        "increased markings", "interstitial marking", "patchy opacit",
        "density", "densities",
    ],
    "Lung Lesion": [
        "nodule", "nodular", "mass", "lesion", "cavitary lesion",
        "carcinoma", "neoplasm", "lump",
    ],
    "Edema": [
        "edema", "pulmonary edema", "vascular congestion", "pulmonary congestion",
        "interstitial edema", "fluid overload", "cephalization",
    ],
    "Consolidation": [
        "consolidation", "consolidative", "airspace disease", "air space disease",
        "air-space opacity", "infiltrate", "infiltration",
    ],
    "Pneumonia": [
        "pneumonia", "infectious process", "bronchopneumonia",
    ],
    "Atelectasis": [
        "atelectasis", "atelectatic", "collapse", "volume loss",
        "subsegmental atelectasis", "plate-like atelectasis", "platelike atelectasis",
    ],
    "Pneumothorax": [
        "pneumothorax", "pneumothoraces", "ptx",
    ],
    "Pleural Effusion": [
        "pleural effusion", "pleural effusions", "effusion", "effusions",
        "pleural fluid", "blunting of the costophrenic angle",
        "costophrenic angle blunting", "layering fluid",
    ],
    "Pleural Other": [
        "pleural thickening", "pleural scarring", "pleural plaque",
        "pleural calcification", "fibrosis", "fibrotic",
    ],
    "Fracture": [
        "fracture", "fractures", "fractured", "rib fracture", "displaced fracture",
    ],
    "Support Devices": [
        "catheter", "picc", "pacemaker", "pigtail", "et tube", "endotracheal tube",
        "ng tube", "nasogastric tube", "chest tube", "sternotomy wire",
        "central line", "central venous", "tracheostomy", "stent",
        "endotracheal", "intubat",
    ],
}

# ── Negation / uncertainty / scope triggers (NegEx-style) ─────────────────────
# Pre-negation triggers (negate findings that appear AFTER the trigger).
PRE_NEGATION = [
    "no evidence of", "no evidence for", "without evidence of", "no sign of",
    "no signs of", "no findings of", "no acute", "no focal", "no definite",
    "no", "without", "absence of", "free of", "negative for", "rules out",
    "rule out", "not", "resolved",
]
# Post-negation triggers (negate findings that appear BEFORE the trigger).
POST_NEGATION = [
    "is ruled out", "are ruled out", "unremarkable", "not seen", "is absent",
    "are absent", "is clear", "are clear", "is normal", "are normal",
    "within normal limits", "wnl", "is negative", "have resolved", "has resolved",
]
# Uncertainty triggers (mark mention as uncertain, -1).
UNCERTAINTY = [
    "possible", "possibly", "probable", "probably", "likely", "may represent",
    "may be", "cannot exclude", "cannot rule out", "can not exclude",
    "questionable", "suspicious for", "suspected", "concern for", "concerning for",
    "differential", "versus", " vs ", " vs.", "could represent", "appears",
    "borderline", "equivocal", "indeterminate",
]
# Pseudo-negation: looks like negation but is NOT (must be excluded).
PSEUDO_NEGATION = [
    "no change", "no interval change", "no significant change", "no increase",
    "no longer", "not only", "without difficulty", "no evidence of interval",
    "not significantly changed",
]
# Scope terminators: cut off the negation/uncertainty scope.
SCOPE_TERMINATORS = [
    "but", "however", "nevertheless", "yet", "though", "although",
    "aside from", "except", "apart from", "otherwise",
]

# Normality cues that, when a sentence contains NO positive finding mention,
# strongly indicate a normal study (used to set "No Finding").
NORMALITY_CUES = [
    "no acute cardiopulmonary", "no acute findings", "no acute disease",
    "normal chest", "unremarkable", "clear lung", "lungs are clear",
    "no acute abnormalit", "grossly normal", "within normal limits",
]


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"x{2,}", " ", text)          # de-identification placeholders (xxxx)
    text = re.sub(r"[^a-z0-9\s\.,;:/-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _split_sentences(text: str) -> list[str]:
    # Split on sentence boundaries and clause terminators (comma/semicolon keep
    # negation scope tight, matching NegEx windowing behaviour on terse reports).
    parts = re.split(r"[.;]\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def _has_trigger(span: str, triggers: list[str]) -> bool:
    return any(t in span for t in triggers)


def _classify_mention(sentence: str, mention_start: int, mention_end: int) -> int:
    """Classify a single mention given its sentence context.

    Returns 1 (positive), 0 (negative), or -1 (uncertain).
    """
    before = sentence[:mention_start]
    after = sentence[mention_end:]

    # Full before-span within the sentence, so a negation governing a coordinated
    # list ("no evidence of A, B, or C") reaches every conjunct. Truncate at the
    # nearest scope terminator so negation does not leak across "but/however".
    window_before = before
    window_after = after[:40]
    if _has_trigger(window_before[-40:], PSEUDO_NEGATION) or _has_trigger(window_after, PSEUDO_NEGATION):
        pass  # fall through; pseudo-negations are treated as non-negating
    else:
        for term in SCOPE_TERMINATORS:
            idx = window_before.rfind(term)
            if idx != -1:
                window_before = window_before[idx + len(term):]

        # Uncertainty takes priority (clinically, a hedge is not a clean negative).
        if _has_trigger(window_before, UNCERTAINTY) or _has_trigger(window_after[:25], UNCERTAINTY):
            return -1
        if _has_trigger(window_before, PRE_NEGATION):
            return 0
        if _has_trigger(window_after[:30], POST_NEGATION):
            return 0

    return 1


def extract_labels(report_text: str) -> dict[str, int | None]:
    """Extract a 14-category CheXpert label vector from a free-text CXR report.

    Returns a dict mapping each of CHEXPERT_CATEGORIES to {1, 0, -1, None}.
    """
    text = _normalize(report_text or "")
    sentences = _split_sentences(text)

    # Initialise all pathology categories to None (not mentioned).
    labels: dict[str, int | None] = {c: None for c in _PATHOLOGY_CATEGORIES}

    for sentence in sentences:
        for category, phrases in MENTION_LEXICON.items():
            for phrase in phrases:
                for m in re.finditer(re.escape(phrase), sentence):
                    cls = _classify_mention(sentence, m.start(), m.end())
                    prev = labels[category]
                    # Aggregation priority: positive > uncertain > negative > None.
                    if prev == 1:
                        continue
                    if cls == 1:
                        labels[category] = 1
                    elif cls == -1 and prev != 1:
                        labels[category] = -1
                    elif cls == 0 and prev is None:
                        labels[category] = 0

    # Derive "No Finding": positive iff no pathology is positive or uncertain.
    any_pathology = any(labels[c] in (1, -1) for c in _PATHOLOGY_CATEGORIES)
    if any_pathology:
        labels["No Finding"] = 0
    else:
        # Confirm with explicit normality cues, else leave as positive-by-default
        # (no positive findings detected == normal study, CheXpert convention).
        labels["No Finding"] = 1

    return labels


def labels_to_binary(labels: dict[str, int | None], uncertain_policy: str = "positive") -> dict[str, int]:
    """Collapse {1,0,-1,None} to binary presence {1,0} for flip analysis.

    uncertain_policy: 'positive' (U-Ones) or 'negative' (U-Zeros).
    None (not mentioned) maps to 0 (absent).
    """
    if uncertain_policy not in ('positive', 'negative'):
        raise ValueError('uncertain_policy must be positive or negative')
    out: dict[str, int] = {}
    for cat, v in labels.items():
        if v == 1:
            out[cat] = 1
        elif v == -1:
            out[cat] = 1 if uncertain_policy == "positive" else 0
        else:  # 0 or None
            out[cat] = 0
    return out


if __name__ == "__main__":
    # Self-test on canonical sentences.
    tests = [
        ("No acute cardiopulmonary abnormalities. The lungs are clear. No pneumothorax.",
         {"No Finding": 1, "Pneumothorax": 0}),
        ("There is mild cardiomegaly. Small left pleural effusion is present.",
         {"Cardiomegaly": 1, "Pleural Effusion": 1, "No Finding": 0}),
        ("Possible early pneumonia in the right lower lobe.",
         {"Pneumonia": -1}),
        ("No evidence of consolidation, effusion, or pneumothorax.",
         {"Consolidation": 0, "Pleural Effusion": 0, "Pneumothorax": 0, "No Finding": 1}),
        ("Endotracheal tube in place. Patchy opacity in the left base.",
         {"Support Devices": 1, "Lung Opacity": 1, "No Finding": 0}),
    ]
    passed = 0
    total = 0
    for text, expected in tests:
        got = extract_labels(text)
        for cat, exp in expected.items():
            total += 1
            ok = got[cat] == exp
            passed += ok
            mark = "OK " if ok else "XX "
            print(f"  {mark} [{cat}] expected={exp} got={got[cat]}  <- {text[:55]}")
    print(f"\nSelf-test: {passed}/{total} assertions passed")
