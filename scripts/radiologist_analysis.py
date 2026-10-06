#!/usr/bin/env python3
"""Analyse the completed radiologist workbooks (R2-3, R1-6).

  python scripts/radiologist_analysis.py --key ../radiologist_review/answer_key_DO_NOT_SHARE.json \
      --readers ../radiologist_review/reader_R1.xlsx ../radiologist_review/reader_R2.xlsx \
      [--adjudicator ../radiologist_review/reader_ADJ.xlsx] --out ../results/radiologist_results.json

Part A: inter-reader Cohen's kappa for Q1 and weighted kappa for Q3; consensus (adjudicator resolves
disagreements, otherwise items with disagreement are reported separately); agreement between the
extractor's transition flag and the consensus; share of clinically significant differences by
precision and stratum. Part B: per-category agreement between the extractor and the consensus labels.
"""
import argparse
import json
from collections import Counter

CATS = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
        "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"]
SIG = {"none": 0, "minor": 1, "significant": 2}


def read(path):
    from openpyxl import load_workbook

    wb = load_workbook(path, data_only=True)
    a = {r[0]: {"q1": r[4], "q3": r[6], "q4": r[7]} for r in wb["Part A"].iter_rows(min_row=2, values_only=True) if r[0]}
    b = {r[0]: dict(zip(CATS, r[2:15])) for r in wb["Part B"].iter_rows(min_row=2, values_only=True) if r[0]}
    return a, b


def kappa(x, y, weights=None, cats=None):
    pairs = [(p, q) for p, q in zip(x, y) if p is not None and q is not None]
    if not pairs:
        return None
    cats = cats or sorted({v for p in pairs for v in p})
    k = len(cats)
    idx = {c: i for i, c in enumerate(cats)}
    import numpy as np

    O = np.zeros((k, k))
    for p, q in pairs:
        O[idx[p], idx[q]] += 1
    n = O.sum()
    E = np.outer(O.sum(1), O.sum(0)) / n
    W = np.array([[((i - j) ** 2 if weights == "quadratic" else (0 if i == j else 1)) for j in range(k)] for i in range(k)], float)
    den = (W * E).sum()
    return None if den == 0 else round(float(1 - (W * O).sum() / den), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--readers", nargs=2, required=True)
    ap.add_argument("--adjudicator")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    key = json.load(open(a.key))
    (a1, b1), (a2, b2) = read(a.readers[0]), read(a.readers[1])
    adj_a, adj_b = read(a.adjudicator) if a.adjudicator else ({}, {})
    items = key["part_a"]
    ids = [it["item_id"] for it in items]
    out = {"n_items": len(ids)}
    out["kappa_q1"] = kappa([a1.get(i, {}).get("q1") for i in ids], [a2.get(i, {}).get("q1") for i in ids])
    out["weighted_kappa_q3"] = kappa([SIG.get(a1.get(i, {}).get("q3")) for i in ids], [SIG.get(a2.get(i, {}).get("q3")) for i in ids],
                                     weights="quadratic", cats=[0, 1, 2])

    def consensus(i, field):
        x, y = a1.get(i, {}).get(field), a2.get(i, {}).get(field)
        if x == y:
            return x
        return adj_a.get(i, {}).get(field)

    rows = []
    for it in items:
        i = it["item_id"]
        rows.append({"id": i, "precision": it["precision"], "stratum": it["stratum"], "model": it["model"],
                     "extractor_change": bool(it["extractor_lost"] or it["extractor_gained"]),
                     "q1": consensus(i, "q1"), "q3": consensus(i, "q3"), "q4": consensus(i, "q4")})
    done = [r for r in rows if r["q1"] in ("yes", "no")]
    out["n_consensus"] = len(done)
    tp = sum(r["extractor_change"] and r["q1"] == "yes" for r in done)
    fp = sum(r["extractor_change"] and r["q1"] == "no" for r in done)
    fn = sum((not r["extractor_change"]) and r["q1"] == "yes" for r in done)
    tn = sum((not r["extractor_change"]) and r["q1"] == "no" for r in done)
    out["extractor_vs_radiologist_change"] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                                             "ppv": round(tp / (tp + fp), 3) if tp + fp else None,
                                             "sensitivity": round(tp / (tp + fn), 3) if tp + fn else None}
    out["significance_by_precision"] = {p: dict(Counter(r["q3"] for r in done if r["precision"] == p and r["q1"] == "yes"))
                                        for p in ("FP16", "INT8", "NF4")}
    out["significance_extracted_change_stratum"] = dict(Counter(r["q3"] for r in done if r["extractor_change"]))
    out["inconsistent_with_findings"] = dict(Counter(r["q4"] for r in done))
    # Part B
    kb = {it["item_id"]: it for it in key["part_b"]}
    percat = {}
    for c in CATS:
        r1 = [b1.get(i, {}).get(c) for i in kb]
        r2 = [b2.get(i, {}).get(c) for i in kb]
        cons = [x if x == y else adj_b.get(i, {}).get(c) for i, x, y in zip(kb, r1, r2)]
        ext = [kb[i]["extractor"][c] for i in kb]
        pos = lambda v: None if v is None else int(v in ("positive", "uncertain"))
        percat[c] = {"kappa_readers": kappa([pos(x) for x in r1], [pos(y) for y in r2]),
                     "kappa_extractor_vs_consensus": kappa([pos(x) for x in ext], [pos(y) for y in cons]),
                     "consensus_positive": sum(pos(x) == 1 for x in cons)}
    out["part_b_per_category"] = percat
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
