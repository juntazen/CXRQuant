#!/usr/bin/env python3
"""
Extractor comparison against the IU-XRAY MeSH silver standard.

Scores each clinical-fact extractor against the IU-XRAY MeSH silver-standard on the
held-out test set, over the categories MeSH can express. Reports precision / recall /
F1 (micro), runtime, and dependency footprint.

Comparators (auto-detected — whichever import successfully):
  - CXRQuant NegEx extractor  (this work; pure-Python stdlib)
  - No-negation keyword baseline (ablation: counts mentions, ignores negation)
  - Window-negation baseline  (simple ±N-token "no/without/negative" window)
  - medspaCy ConText          (if installed)
  - pyConTextNLP              (if installed)

Honest by design: tools that do not install are reported as "not installable in this
environment (Python 3.13)" rather than fabricated.
"""
import json
import argparse
import re
import time

from cxrquant.clinical_safety.fact_extractor import extract_labels
from cxrquant.data.iuxray_dataset import create_splits
from cxrquant.paths import DATA_JSON, REPO_ROOT
from cxrquant.runio import atomic_json, sha256_file

V2 = REPO_ROOT  # kept as an alias for readability of the original scripts
DATA = DATA_JSON

MESH_TO_CHEXPERT = {
    'cardiomegaly': 'Cardiomegaly', 'pulmonary atelectasis': 'Atelectasis',
    'atelectasis': 'Atelectasis', 'pleural effusion': 'Pleural Effusion',
    'effusion': 'Pleural Effusion', 'pneumonia': 'Pneumonia',
    'pneumothorax': 'Pneumothorax', 'consolidation': 'Consolidation',
    'edema': 'Edema', 'pulmonary edema': 'Edema', 'nodule': 'Lung Lesion',
    'mass': 'Lung Lesion', 'fracture': 'Fracture',
}
SCORABLE = sorted(set(MESH_TO_CHEXPERT.values()))

# keyword surface forms for the scorable categories (shared by baselines + spaCy tools)
CAT_KEYWORDS = {
    'Cardiomegaly': ['cardiomegaly', 'enlarged heart', 'cardiac enlargement', 'enlarged cardiac silhouette', 'cardiac silhouette is enlarged'],
    'Atelectasis': ['atelectasis', 'atelectatic', 'atelectases'],
    'Pleural Effusion': ['pleural effusion', 'effusion', 'pleural fluid'],
    'Pneumonia': ['pneumonia', 'pneumonic'],
    'Pneumothorax': ['pneumothorax', 'pneumothoraces'],
    'Consolidation': ['consolidation', 'consolidative', 'airspace disease', 'air space disease'],
    'Edema': ['edema', 'oedema', 'pulmonary edema'],
    'Lung Lesion': ['nodule', 'nodular', 'mass', 'lung lesion'],
    'Fracture': ['fracture', 'fractured', 'fractures'],
}
NEG_CUES = ['no ', 'without', 'negative for', 'free of', 'absence of', 'no evidence of',
            'not ', 'clear of', 'resolved', 'rule out', 'ruled out', 'unremarkable']


def load_test():
    recs = json.load(open(DATA))
    splits = create_splits(recs, seed=42)
    out = []
    for r in splits['test']:
        f = r.get('findings', '').strip(); i = r.get('impression', '').strip()
        if not (f and i):
            continue
        mesh = r.get('mesh_labels') or []
        if isinstance(mesh, str):
            mesh = [mesh]
        gt = set()
        for m in mesh:
            ml = m.lower()
            for key, cat in MESH_TO_CHEXPERT.items():
                if key in ml:
                    gt.add(cat)
        out.append({'text': f + '. ' + i, 'gt': gt})
    return out


def score(pred_sets, gt_sets):
    tp = fp = fn = 0
    per = {c: {'tp': 0, 'fp': 0, 'fn': 0} for c in SCORABLE}
    if len(pred_sets) != len(gt_sets) or not pred_sets:
        raise ValueError('expected nonempty paired comparator inputs')
    for pred, gt in zip(pred_sets, gt_sets, strict=True):
        for c in SCORABLE:
            ig, ip = c in gt, c in pred
            if ig and ip: tp += 1; per[c]['tp'] += 1
            elif ip and not ig: fp += 1; per[c]['fp'] += 1
            elif ig and not ip: fn += 1; per[c]['fn'] += 1
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {'precision': round(p, 4), 'recall': round(r, 4), 'f1': round(f1, 4),
            'tp': tp, 'fp': fp, 'fn': fn}


# ---------- extractors ----------
def ex_cxrquant(text):
    labels = extract_labels(text)
    return {c for c in SCORABLE if labels.get(c) == 1}


def ex_keyword_nonneg(text):
    t = ' ' + text.lower() + ' '
    return {c for c, kws in CAT_KEYWORDS.items() if any(k in t for k in kws)}


def ex_window_neg(text, window=6):
    """Positive if a keyword appears and is NOT preceded by a negation cue within `window` tokens."""
    toks = re.findall(r"[a-zA-Z]+", text.lower())
    joined = ' '.join(toks)
    pos = set()
    for c, kws in CAT_KEYWORDS.items():
        for kw in kws:
            for m in re.finditer(re.escape(kw), joined):
                start = joined[:m.start()].count(' ')
                pre = toks[max(0, start - window):start]
                pre_join = ' ' + ' '.join(pre) + ' '
                negated = any(cue.strip() in pre_join for cue in NEG_CUES)
                if not negated:
                    pos.add(c); break
    return pos


def build_external():
    """Return list of (name, fn, dependency_note) for any installable external tools."""
    ext = []
    # medspaCy
    try:
        import medspacy
        from medspacy.ner import TargetRule
        nlp = medspacy.load(medspacy_disable=["medspacy_pyrush"]) if False else medspacy.load()
        rules = []
        for c, kws in CAT_KEYWORDS.items():
            for kw in kws:
                rules.append(TargetRule(literal=kw, category=c))
        nlp.get_pipe("medspacy_target_matcher").add(rules)

        def ex_medspacy(text):
            doc = nlp(text)
            pos = set()
            for ent in doc.ents:
                if not getattr(ent._, "is_negated", False) and not getattr(ent._, "is_uncertain", False):
                    pos.add(ent.label_)
            return pos
        ext.append(("medspaCy ConText", ex_medspacy, "spaCy + medspaCy (heavy; build needed)"))
    except Exception as e:
        ext.append(("medspaCy ConText", None, f"unavailable ({type(e).__name__}): {str(e)[:120]}"))
    # pyConTextNLP (canonical NegEx/ConText, Chapman et al.)
    try:
        import pyConTextNLP.itemData as itemData
        import pyConTextNLP.pyConTextGraph as pyConText

        # negation/uncertainty modifiers
        mod_specs = [(cue, "DEFINITE_NEGATED_EXISTENCE", "forward") for cue in
                     ["no", "without", "no evidence of", "negative for", "free of",
                      "absence of", "clear of", "not", "rule out", "ruled out",
                      "resolved", "unremarkable for"]]
        mod_specs += [(cue, "PROBABLE_EXISTENCE", "forward") for cue in
                      ["possible", "probable", "likely", "may represent", "suggestive of",
                       "cannot exclude", "concerning for", "question of"]]
        modifiers = [itemData.contextItem((lit, cat, r"\b" + re.escape(lit) + r"\b", rule))
                     for lit, cat, rule in mod_specs]
        cat_back = {}
        targets = []
        for c, kws in CAT_KEYWORDS.items():
            tag = c.lower().replace(" ", "_")
            cat_back[tag] = c
            for kw in kws:
                targets.append(itemData.contextItem((kw, tag, r"\b" + re.escape(kw) + r"\b", "")))

        def _markup(sent):
            m = pyConText.ConTextMarkup()
            m.setRawText(sent)
            m.cleanText()
            m.markItems(modifiers, mode="modifier")
            m.markItems(targets, mode="target")
            m.pruneMarks()
            m.applyModifiers()
            m.dropInactiveModifiers()
            return m

        def ex_pycontext(text):
            pos = set()
            for sent in re.split(r"[.;\n]", text):
                sent = sent.strip()
                if not sent:
                    continue
                m = _markup(sent)
                for tnode in m.getMarkedTargets():
                    if m.isModifiedByCategory(tnode, "DEFINITE_NEGATED_EXISTENCE"):
                        continue
                    if m.isModifiedByCategory(tnode, "PROBABLE_EXISTENCE"):
                        continue  # uncertain -> treat as absent (conservative, matches our 'positive' policy off)
                    for cat in tnode.getCategory():
                        if cat in cat_back:
                            pos.add(cat_back[cat])
            return pos
        ext.append(("pyConTextNLP (NegEx)", ex_pycontext, "pyConTextNLP + networkx"))
    except Exception as e:
        ext.append(("pyConTextNLP (NegEx)", None, f"unavailable ({type(e).__name__}): {str(e)[:120]}"))
    return ext


def main():
    global DATA
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--run-dir', type=Path, required=True)
    args = ap.parse_args()
    DATA = args.data
    args.run_dir.mkdir(parents=True, exist_ok=True)
    test = load_test()
    gt = [t['gt'] for t in test]
    texts = [t['text'] for t in test]
    n_gt_pos = sum(len(g) for g in gt)
    print(f"Test reports: {len(test)} | scorable categories: {SCORABLE}")
    print(f"Ground-truth positive (category,report) pairs from MeSH: {n_gt_pos}\n")

    runs = [
        ("CXRQuant NegEx (ours)", ex_cxrquant, "pure-Python stdlib only", True),
        ("Keyword (no negation)", ex_keyword_nonneg, "stdlib (ablation)", True),
        ("Window-negation baseline", ex_window_neg, "stdlib (simple ±6-tok)", True),
    ]
    # external (best-effort)
    for name, fn, note in build_external():
        if callable(fn):
            runs.append((name, fn, note, True))
        else:
            runs.append((name, None, note, False))

    results = []
    for name, fn, note, ok in runs:
        if not ok or fn is None:
            print(f"  [SKIP] {name:<26} — {note}")
            results.append({"extractor": name, "available": False, "dependencies": note})
            continue
        t0 = time.perf_counter()
        try:
            preds = [fn(x) for x in texts]
        except Exception as e:
            results.append({'extractor':name,'available':False,'dependencies':note,'error':str(e)})
            print(f'  [FAILED] {name}: {e}')
            continue
        dt = (time.perf_counter() - t0) * 1000
        s = score(preds, gt)
        s.update({"extractor": name, "available": True, "dependencies": note,
                  "runtime_ms_total": round(dt, 1), "runtime_ms_per_report": round(dt / len(texts), 3)})
        results.append(s)
        print(f"  {name:<26} P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1']:.3f} "
              f"(tp{s['tp']} fp{s['fp']} fn{s['fn']})  {dt:.0f}ms  [{note}]")

    out = {"task": "extractor comparison vs IU-XRAY MeSH silver standard",
           "n_reports": len(test), "scorable_categories": SCORABLE,
           "gt_positive_pairs": n_gt_pos, "results": results}
    # Non-destructive by design: the committed results/extractor_comparison.json is the
    # reference run made with every optional comparator installed. When a comparator is
    # missing we write a clearly-named local file instead of silently overwriting it.
    complete = all(r.get("available") for r in results)
    name = 'extractor_comparison.json'
    out['provenance'] = {'run_id':args.run_dir.name,'data_hash':sha256_file(DATA),'script_hash':sha256_file(__file__),
                         'evaluation_text':'human findings + impression','positive_policy':'only definite positive; uncertain excluded',
                         'status':'complete' if complete else 'partial_optional_comparators'}
    atomic_json(args.run_dir / name, out)
    print(f"\nSaved -> {args.run_dir / name}")
    if not complete:
        print('Optional comparator unavailable; current output records this explicitly. No archived row substituted.')


if __name__ == "__main__":
    main()
