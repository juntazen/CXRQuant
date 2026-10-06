"""CPU analyses added for the IJIES second-round revision (paper ID 20266214).

Reads only stored run archives and the public OpenI report release; writes new files under
``--out``. Nothing in the archived runs is modified. One section per reviewer point:

  data_flow         R1.3  OpenI release -> 781-record corpus -> 465/68/134 partitions
  overlap           R1.2  exact and near-duplicate FINDINGS/IMPRESSION across partitions,
                          audit recomputed on held-out reports without duplicated FINDINGS
  denominators      R1.7  FP32-positive/negative denominators, b/c, FNF/FPF with report-level
                          bootstrap CIs for every precision, per-category transitions (CSV)
  extractor         R1.6  per-category silver validation and per-category labeller agreement
  strata            R1.4  threshold gap, all-model conclusions, continuous BLEU-FNF relation
  hierarchical      R2.4  report-level paired summaries, cluster bootstrap over reports,
                          GEE (report clusters) and crossed random-effects logistic model
  efficiency        R1.8  repeated measurements over five independent runs and two devices
  tolerance         R2.6  acceptance counts across lexical and stability tolerances

Usage: python scripts/revision_analyses.py --openi ../external/openi_parsed.json --out ../results
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from cxrquant.clinical_safety.fact_extractor import (
    _PATHOLOGY_CATEGORIES,
    extract_labels,
    labels_to_binary,
)
from cxrquant.data.iuxray_dataset import create_splits
from cxrquant.evaluation.nlg_metrics import corpus_bleu

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    42: ROOT / "runs/full_seed42_corrected_20260911",
    123: ROOT / "runs/full_seed123_20260915",
    777: ROOT / "runs/full_seed777_20260915",
    2024: ROOT / "runs/full_seed2024_20260915",
    31415: ROOT / "runs/full_seed31415_20260915",
}
MODELS = ["distilgpt2", "gpt2", "gpt2-medium", "biogpt", "pythia-1b", "biogpt-large"]
CONFIGS = ["FP32", "FP16", "INT8-BnB", "NF4-BnB"]
REDUCED = CONFIGS[1:]
SHORT = {"FP16": "FP16", "INT8-BnB": "INT8", "NF4-BnB": "NF4"}
FUNCTIONAL = {"biogpt", "pythia-1b", "biogpt-large"}
# Silver-label categories (scripts/extractor_comparison.py MESH_TO_CHEXPERT)
SILVER = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Fracture", "Lung Lesion",
          "Pleural Effusion", "Pneumonia", "Pneumothorax"]
# Launcher order decides the MIG: worker 0 (first device) runs distilgpt2/gpt2-medium/pythia-1b.
WORKER0 = {"distilgpt2", "gpt2-medium", "pythia-1b"}
FIRST_DEVICE = {42: "MIG-b229", 123: "MIG-75de", 777: "MIG-75de", 2024: "MIG-75de", 31415: "MIG-75de"}

_label_cache: dict[str, dict] = {}


def labels(text: str, policy: str = "positive") -> dict[str, int]:
    if text not in _label_cache:
        _label_cache[text] = extract_labels(text)
    return labels_to_binary(_label_cache[text], policy)


def load_run(seed):
    run = RUNS[seed]
    manifest = json.loads((run / "manifest.json").read_text())
    recs = {}
    for m in MODELS:
        for c in CONFIGS:
            recs[m, c] = json.loads((run / "models" / m / f"{c}.json").read_text())
    return manifest, recs


def rate(num, den):
    return None if den == 0 else num / den


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- R1.3 data flow
def data_flow(openi_path: Path, corpus: list[dict]):
    openi = json.loads(openi_path.read_text())
    with_img = {u for u, r in openi.items() if r["n_img"] > 0}
    both = {u for u, r in openi.items() if r["findings"] and r["impression"]}
    uids = {r["uid"] for r in corpus}
    text_identical = sum(
        r["findings"].strip() == openi[r["uid"]]["findings"]
        and r["impression"].strip() == openi[r["uid"]]["impression"]
        for r in corpus
    )
    splits = create_splits(corpus, seed=42)
    raw = {k: len(v) for k, v in splits.items()}
    usable = {k: [r for r in v if r["findings"].strip() and r["impression"].strip()] for k, v in splits.items()}
    excl = {k: raw[k] - len(usable[k]) for k in raw}
    excl_reason = Counter()
    for v in splits.values():
        for r in v:
            f, i = bool(r["findings"].strip()), bool(r["impression"].strip())
            if not f and not i:
                excl_reason["both_empty"] += 1
            elif not f:
                excl_reason["findings_empty"] += 1
            elif not i:
                excl_reason["impression_empty"] += 1
    sub_imgs = sum(len(r["images"]) for r in corpus)
    openi_imgs_same = sum(openi[u]["n_img"] for u in uids)

    # Representativeness: usable corpus vs the remaining usable OpenI reports with images
    rest = [u for u in (with_img & both) if u not in uids]
    used = [r["uid"] for v in usable.values() for r in v]

    def profile(ids):
        n = len(ids)
        fw = [len(openi[u]["findings"].split()) for u in ids]
        iw = [len(openi[u]["impression"].split()) for u in ids]
        normal = sum(any(m.lower() == "normal" for m in openi[u]["mesh"]) for u in ids)
        prev = Counter()
        for u in ids:
            lab = labels(openi[u]["impression"])
            for c in _PATHOLOGY_CATEGORIES:
                prev[c] += lab[c]
        return {
            "n": n,
            "findings_words_mean": round(float(np.mean(fw)), 1),
            "impression_words_mean": round(float(np.mean(iw)), 1),
            "mesh_normal_pct": round(100 * normal / n, 1),
            "impression_any_positive_category_pct": round(
                100 * sum(any(labels(openi[u]["impression"])[c] for c in _PATHOLOGY_CATEGORIES) for u in ids) / n, 1),
            "category_prevalence_pct": {c: round(100 * prev[c] / n, 1) for c in _PATHOLOGY_CATEGORIES},
        }

    split_rows = [
        {"uid": r["uid"], "partition": k} for k, v in usable.items() for r in sorted(v, key=lambda x: x["uid"])
    ]
    return {
        "openi_reports": len(openi),
        "openi_reports_with_images": len(with_img),
        "openi_reports_without_images": len(openi) - len(with_img),
        "openi_reports_with_findings_and_impression": len(both),
        "openi_with_images_and_both_sections": len(with_img & both),
        "corpus_records": len(corpus),
        "corpus_records_in_openi": sum(u in openi for u in uids),
        "corpus_records_text_identical_to_openi": text_identical,
        "corpus_records_without_images": sum(openi[u]["n_img"] == 0 for u in uids),
        "corpus_image_files_listed": sub_imgs,
        "openi_image_files_for_same_records": openi_imgs_same,
        "split_procedure": "unique source UIDs sorted, shuffled with numpy default_rng(42), 70/10/20 cut "
                           "(int truncation), filtering of empty sections applied after the split",
        "raw_partition_sizes": raw,
        "excluded_empty_sections": excl,
        "excluded_reason_counts": dict(excl_reason),
        "usable_partition_sizes": {k: len(v) for k, v in usable.items()},
        "duplicate_uids": len(corpus) - len(uids),
        "representativeness": {"used_667": profile(used), "remaining_openi_with_images": profile(rest)},
    }, split_rows, usable


# ---------------------------------------------------------------- R1.2 overlap
def norm(t: str) -> str:
    t = t.lower().replace("xxxx", " ")
    return " ".join(re.findall(r"[a-z0-9]+", t))


def jacc(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def overlap(usable, threshold=0.9):
    out, flagged = {}, {}
    pools = {"train": usable["train"], "train+val": usable["train"] + usable["val"]}
    for field in ("findings", "impression"):
        ref_sets = {k: [(norm(r[field]), set(norm(r[field]).split())) for r in v] for k, v in pools.items()}
        for part, pool in (("val", "train"), ("test", "train+val")):
            exact = near = 0
            ids_exact, ids_near = set(), set()
            for r in usable[part]:
                n = norm(r[field]); s = set(n.split())
                if any(n == x for x, _ in ref_sets[pool]):
                    exact += 1; ids_exact.add(r["uid"])
                if any(jacc(s, y) >= threshold for _, y in ref_sets[pool]):
                    near += 1; ids_near.add(r["uid"])
            out[f"{part}_vs_{pool}_{field}"] = {"n": len(usable[part]), "exact": exact, f"jaccard_ge_{threshold}": near}
            flagged[(part, field)] = (ids_exact, ids_near)
    # Within-test repetition and templated impressions
    imp = Counter(norm(r["impression"]) for r in usable["test"])
    out["test_distinct_impressions"] = len(imp)
    out["test_top_impressions"] = imp.most_common(5)
    return out, flagged


def audit_subset(manifest, recs, keep_idx, policy="positive"):
    """FNF/FPF and BLEU-4 per model x precision on a subset of held-out positions."""
    refs = [manifest["references"][i] for i in keep_idx]
    rows = []
    for m in MODELS:
        base = [labels(recs[m, "FP32"]["predictions"][i], policy) for i in keep_idx]
        bleu32 = corpus_bleu([recs[m, "FP32"]["predictions"][i] for i in keep_idx], [[r] for r in refs])["bleu_4"]
        for c in REDUCED:
            q = [labels(recs[m, c]["predictions"][i], policy) for i in keep_idx]
            pos = neg = b = cc = 0
            for x, y in zip(base, q):
                for k in _PATHOLOGY_CATEGORIES:
                    if x[k]:
                        pos += 1; b += (y[k] == 0)
                    else:
                        neg += 1; cc += (y[k] == 1)
            bleu = corpus_bleu([recs[m, c]["predictions"][i] for i in keep_idx], [[r] for r in refs])["bleu_4"]
            rows.append({"model": m, "precision": SHORT[c], "n_reports": len(keep_idx), "fp32_bleu4": round(bleu32, 4),
                         "bleu4": round(bleu, 4), "fp32_pos": pos, "fp32_neg": neg, "b": b, "c": cc,
                         "FNF": None if not pos else round(b / pos, 4), "FPF": None if not neg else round(cc / neg, 4)})
    return rows


def ordering(rows):
    res = {}
    for m in MODELS:
        v = [next(r["FNF"] for r in rows if r["model"] == m and r["precision"] == p) for p in ("FP16", "INT8", "NF4")]
        res[m] = {"FNF": v, "strict": None not in v and v[0] < v[1] < v[2],
                  "non_decreasing": None not in v and v[0] <= v[1] <= v[2]}
    return res


# ---------------------------------------------------------------- R1.7 denominators + CIs
def bootstrap_ci(base, quant, idx_draws):
    """Report-level bootstrap of FNF and FPF; base/quant are per-report binary dict lists."""
    per = np.array([[sum(x[k] for k in _PATHOLOGY_CATEGORIES),
                     sum(x[k] and not y[k] for k in _PATHOLOGY_CATEGORIES),
                     sum(1 - x[k] for k in _PATHOLOGY_CATEGORIES),
                     sum((not x[k]) and y[k] for k in _PATHOLOGY_CATEGORIES)] for x, y in zip(base, quant)])
    s = per[idx_draws].sum(axis=1)  # draws x 4
    fnf = np.where(s[:, 0] > 0, s[:, 1] / np.maximum(s[:, 0], 1), np.nan)
    fpf = np.where(s[:, 2] > 0, s[:, 3] / np.maximum(s[:, 2], 1), np.nan)
    q = lambda a: (float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5)), int(np.isnan(a).sum()))
    return q(fnf), q(fpf)


def denominators(all_runs, draws=2000):
    rows, percat = [], []
    for seed, (manifest, recs) in all_runs.items():
        n = len(manifest["references"])
        rng = np.random.default_rng(42)
        idx = rng.integers(0, n, size=(draws, n))
        for m in MODELS:
            base = [labels(p) for p in recs[m, "FP32"]["predictions"]]
            for c in REDUCED:
                q = [labels(p) for p in recs[m, c]["predictions"]]
                pos = sum(x[k] for x in base for k in _PATHOLOGY_CATEGORIES)
                neg = n * 13 - pos
                b = sum(x[k] and not y[k] for x, y in zip(base, q) for k in _PATHOLOGY_CATEGORIES)
                cc = sum((not x[k]) and y[k] for x, y in zip(base, q) for k in _PATHOLOGY_CATEGORIES)
                (fl, fh, fu), (pl, ph, pu) = bootstrap_ci(base, q, idx)
                reports_changed = sum(any(x[k] != y[k] for k in _PATHOLOGY_CATEGORIES) for x, y in zip(base, q))
                rows.append({"seed": seed, "model": m, "precision": SHORT[c], "fp32_pos": pos, "fp32_neg": neg,
                             "b_lost": int(b), "c_gained": int(cc),
                             "FNF": round(b / pos, 4) if pos else None, "FNF_ci_low": round(fl, 4), "FNF_ci_high": round(fh, 4),
                             "FNF_undefined_resamples": fu,
                             "FPF": round(cc / neg, 4), "FPF_ci_low": round(pl, 4), "FPF_ci_high": round(ph, 4),
                             "reports_with_any_change": reports_changed, "n_reports": n})
                for k in _PATHOLOGY_CATEGORIES:
                    p_ = sum(x[k] for x in base)
                    percat.append({"seed": seed, "model": m, "precision": SHORT[c], "category": k,
                                   "fp32_pos": p_, "fp32_neg": n - p_,
                                   "pos_to_neg": sum(x[k] and not y[k] for x, y in zip(base, q)),
                                   "neg_to_pos": sum((not x[k]) and y[k] for x, y in zip(base, q)),
                                   "pos_to_pos": sum(x[k] and y[k] for x, y in zip(base, q)),
                                   "neg_to_neg": sum((not x[k]) and (not y[k]) for x, y in zip(base, q))})
    return rows, percat


# ---------------------------------------------------------------- R1.6 extractor per category
def extractor_per_category(seed42_dir: Path):
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import extractor_comparison as comp

    test = comp.load_test()
    per = {c: {"tp": 0, "fp": 0, "fn": 0, "silver_pos": 0} for c in SILVER}
    unc = Counter()
    for t in test:
        lab = extract_labels(t["text"])
        for c in SILVER:
            p, g = lab.get(c) == 1, c in t["gt"]
            per[c]["silver_pos"] += g
            per[c]["tp"] += p and g; per[c]["fp"] += p and not g; per[c]["fn"] += g and not p
            if lab.get(c) == -1 or lab.get(c) == 2:
                unc[c] += 1
    for c, v in per.items():
        P = v["tp"] / (v["tp"] + v["fp"]) if v["tp"] + v["fp"] else None
        R = v["tp"] / (v["tp"] + v["fn"]) if v["tp"] + v["fn"] else None
        v.update(precision=None if P is None else round(P, 3), recall=None if R is None else round(R, 3),
                 f1=None if not P or not R else round(2 * P * R / (P + R), 3))
    bec = json.loads((seed42_dir / "baseline_evaluator_comparison.json").read_text())
    agree = bec["inter_extractor_agreement"]
    return {"silver_per_category": per, "n_reports": len(test),
            "uncovered_categories": [c for c in _PATHOLOGY_CATEGORIES if c not in SILVER],
            "agreement_keys": list(agree.keys()), "agreement": agree}


# ---------------------------------------------------------------- R1.4 strata
def strata(all_runs):
    bleu = {s: {m: recs[m, "FP32"]["bleu_4"] for m in MODELS} for s, (_, recs) in all_runs.items()}
    fnf = {s: {m: {c: recs[m, c]["FNF_rate"] for c in REDUCED} for m in MODELS} for s, (_, recs) in all_runs.items()}
    s42 = sorted(bleu[42].values())
    gap = (max(v for v in s42 if v < 0.1), min(v for v in s42 if v > 0.1))
    allv = [bleu[s][m] for s in bleu for m in MODELS]
    deg = [bleu[s][m] for s in bleu for m in MODELS if m not in FUNCTIONAL]
    fun = [bleu[s][m] for s in bleu for m in MODELS if m in FUNCTIONAL]
    x = np.array([bleu[s][m] for s in bleu for m in MODELS])
    y = np.array([fnf[s][m]["NF4-BnB"] for s in bleu for m in MODELS])
    from scipy.stats import spearmanr

    rho = spearmanr(x, y)
    return {"seed42_gap": gap, "degenerate_range_all_seeds": (min(deg), max(deg)),
            "functional_range_all_seeds": (min(fun), max(fun)),
            "any_threshold_in_gap_gives_same_partition_all_seeds": max(deg) < min(fun),
            "spearman_fp32bleu_vs_nf4fnf_30runs": {"rho": round(float(rho.statistic), 3), "p": float(rho.pvalue)},
            "all_model_nf4_fnf_seed42": {m: fnf[42][m]["NF4-BnB"] for m in MODELS},
            "all_model_ordering_seed42_strict": sum(fnf[42][m]["FP16"] < fnf[42][m]["INT8-BnB"] < fnf[42][m]["NF4-BnB"] for m in MODELS),
            "fp32_bleu4": bleu}


# ---------------------------------------------------------------- R2.4 hierarchical
def long_table(all_runs):
    """One row per FP32-positive (FNF) or FP32-negative (FPF) category-report pair."""
    rows = []
    for seed, (manifest, recs) in all_runs.items():
        ids = manifest["sample_ids"]
        for m in MODELS:
            base = [labels(p) for p in recs[m, "FP32"]["predictions"]]
            for c in REDUCED:
                q = [labels(p) for p in recs[m, c]["predictions"]]
                for i, (x, y) in enumerate(zip(base, q)):
                    for k in _PATHOLOGY_CATEGORIES:
                        rows.append((seed, m, SHORT[c], ids[i], k, x[k], int(x[k] != y[k])))
    import pandas as pd

    return pd.DataFrame(rows, columns=["seed", "model", "precision", "report", "category", "fp32_pos", "flip"])


def hierarchical(df, draws=2000):
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    out = {}
    pos = df[df.fp32_pos == 1].copy()
    neg = df[df.fp32_pos == 0].copy()

    # 1) Report-level paired summaries: per report and model x seed, FNF computed within report.
    rep = pos.groupby(["seed", "model", "report", "precision"]).flip.mean().unstack("precision")
    rep = rep.dropna()
    out["report_level"] = {
        "n_report_model_seed_units_with_fp32_positive": int(len(rep)),
        "mean_report_fnf": {p: round(float(rep[p].mean()), 4) for p in ("FP16", "INT8", "NF4")},
        "pct_units_INT8_ge_FP16": round(100 * float((rep.INT8 >= rep.FP16).mean()), 1),
        "pct_units_NF4_ge_INT8": round(100 * float((rep.NF4 >= rep.INT8).mean()), 1),
        "pct_units_INT8_gt_FP16": round(100 * float((rep.INT8 > rep.FP16).mean()), 1),
        "pct_units_NF4_gt_INT8": round(100 * float((rep.NF4 > rep.INT8).mean()), 1),
        "pct_units_INT8_lt_FP16": round(100 * float((rep.INT8 < rep.FP16).mean()), 1),
        "pct_units_NF4_lt_INT8": round(100 * float((rep.NF4 < rep.INT8).mean()), 1),
    }
    from scipy.stats import wilcoxon

    out["report_level"]["wilcoxon_INT8_vs_FP16_p"] = float(wilcoxon(rep.INT8, rep.FP16, zero_method="wilcox").pvalue)
    out["report_level"]["wilcoxon_NF4_vs_INT8_p"] = float(wilcoxon(rep.NF4, rep.INT8, zero_method="wilcox").pvalue)

    # 2) Cluster bootstrap over held-out reports (reports resampled jointly across models, seeds, categories).
    reports = np.array(sorted(pos.report.unique()))
    agg = pos.groupby(["report", "precision"]).flip.agg(["sum", "count"]).unstack("precision").fillna(0)
    S = {p: agg[("sum", p)].reindex(reports).to_numpy() for p in ("FP16", "INT8", "NF4")}
    N = {p: agg[("count", p)].reindex(reports).to_numpy() for p in ("FP16", "INT8", "NF4")}
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(reports), size=(draws, len(reports)))
    boot = {p: S[p][idx].sum(1) / N[p][idx].sum(1) for p in S}
    ci = lambda a: [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]
    out["cluster_bootstrap_pooled_fnf"] = {
        "point": {p: round(float(S[p].sum() / N[p].sum()), 4) for p in S},
        "ci95": {p: ci(boot[p]) for p in boot},
        "diff_INT8_minus_FP16": ci(boot["INT8"] - boot["FP16"]),
        "diff_NF4_minus_INT8": ci(boot["NF4"] - boot["INT8"]),
        "n_reports": int(len(reports)), "draws": draws,
    }
    for stratum, models in (("functional", FUNCTIONAL), ("degenerate", set(MODELS) - FUNCTIONAL)):
        sub = pos[pos.model.isin(models)]
        a = sub.groupby(["report", "precision"]).flip.agg(["sum", "count"]).unstack("precision").fillna(0)
        rr = np.array(sorted(sub.report.unique()))
        Ss = {p: a[("sum", p)].reindex(rr).to_numpy() for p in ("FP16", "INT8", "NF4")}
        Nn = {p: a[("count", p)].reindex(rr).to_numpy() for p in ("FP16", "INT8", "NF4")}
        ii = np.random.default_rng(42).integers(0, len(rr), size=(draws, len(rr)))
        bb = {p: Ss[p][ii].sum(1) / Nn[p][ii].sum(1) for p in Ss}
        out[f"cluster_bootstrap_{stratum}"] = {
            "point": {p: round(float(Ss[p].sum() / Nn[p].sum()), 4) for p in Ss},
            "ci95": {p: ci(bb[p]) for p in bb},
            "diff_INT8_minus_FP16": ci(bb["INT8"] - bb["FP16"]),
            "diff_NF4_minus_INT8": ci(bb["NF4"] - bb["INT8"])}

    # 3) GEE logistic regression, exchangeable working correlation within held-out report.
    pos["precision"] = pd.Categorical(pos.precision, ["FP16", "INT8", "NF4"])
    pos["group"] = pos.report.astype("category").cat.codes
    gee = smf.gee("flip ~ C(precision) + C(model) + C(category) + C(seed)", groups="group", data=pos,
                  family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
    terms = [t for t in gee.params.index if t.startswith("C(precision)")]
    out["gee_fnf"] = {t: {"OR": round(float(np.exp(gee.params[t])), 2),
                          "ci95": [round(float(np.exp(gee.conf_int().loc[t, 0])), 2), round(float(np.exp(gee.conf_int().loc[t, 1])), 2)],
                          "p": float(gee.pvalues[t])} for t in terms}
    out["gee_fnf"]["n_pairs"] = int(len(pos)); out["gee_fnf"]["n_clusters"] = int(pos.group.nunique())
    # INT8 -> NF4 contrast with NF4 as reference level
    pos2 = pos.copy(); pos2["precision"] = pd.Categorical(pos2.precision.astype(str), ["INT8", "FP16", "NF4"])
    g2 = smf.gee("flip ~ C(precision) + C(model) + C(category) + C(seed)", groups="group", data=pos2,
                 family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
    t = "C(precision)[T.NF4]"
    out["gee_fnf"]["NF4_vs_INT8"] = {"OR": round(float(np.exp(g2.params[t])), 2),
                                     "ci95": [round(float(np.exp(g2.conf_int().loc[t, 0])), 2), round(float(np.exp(g2.conf_int().loc[t, 1])), 2)],
                                     "p": float(g2.pvalues[t])}

    # 4) Crossed random-effects logistic model (variational Bayes): report, category, model, seed.
    vc = {"report": "0 + C(report)", "category": "0 + C(category)", "model": "0 + C(model)", "seed": "0 + C(seed)"}
    mixed = BinomialBayesMixedGLM.from_formula("flip ~ C(precision, Treatment('FP16'))", vc, pos).fit_vb()
    names = mixed.model.exog_names
    fe = {}
    for i, n in enumerate(names):
        if "precision" in n:
            m_, s_ = mixed.fe_mean[i], mixed.fe_sd[i]
            fe[n] = {"OR": round(float(np.exp(m_)), 2), "ci95": [round(float(np.exp(m_ - 1.96 * s_)), 2), round(float(np.exp(m_ + 1.96 * s_)), 2)]}
    vcn = mixed.model.vcp_names
    out["mixed_glmm_fnf"] = {"fixed_effects_vs_FP16": fe,
                             "random_effect_sd": {n: round(float(np.exp(v)), 3) for n, v in zip(vcn, mixed.vcp_mean)},
                             "method": "BinomialBayesMixedGLM (statsmodels), variational Bayes, crossed random intercepts"}

    # 5) Concentration: is instability driven by a few categories or reports?
    nf4 = pos[pos.precision == "NF4"]
    by_cat = nf4.groupby("category").flip.agg(["sum", "count"])
    loco = {}
    for cat in _PATHOLOGY_CATEGORIES:
        sub = pos[pos.category != cat]
        v = sub.groupby(["seed", "model", "precision"]).flip.mean().unstack("precision")
        loco[cat] = int(((v.FP16 <= v.INT8) & (v.INT8 <= v.NF4)).sum())
    by_rep = nf4.groupby("report").flip.sum().sort_values(ascending=False)
    top10 = by_rep.head(int(np.ceil(0.1 * len(by_rep)))).sum() / by_rep.sum()
    out["concentration"] = {
        "nf4_flips_by_category": {k: int(v) for k, v in by_cat["sum"].items()},
        "nf4_fnf_by_category": {k: round(float(r["sum"] / r["count"]), 3) for k, r in by_cat.iterrows()},
        "leave_one_category_out_non_decreasing_runs_of_30": loco,
        "share_of_nf4_losses_in_top10pct_reports": round(float(top10), 3),
        "reports_with_any_nf4_loss": int((by_rep > 0).sum()), "reports_with_fp32_positive": int(len(by_rep)),
    }
    # FPF counterpart for completeness
    out["cluster_bootstrap_pooled_fpf_point"] = {p: round(float(neg[neg.precision == p].flip.mean()), 4) for p in ("FP16", "INT8", "NF4")}
    return out


# ---------------------------------------------------------------- R1.8 efficiency
def efficiency(all_runs):
    rows = []
    for seed, (_, recs) in all_runs.items():
        for m in MODELS:
            dev = FIRST_DEVICE[seed] if m in WORKER0 else ({"MIG-b229": "MIG-75de", "MIG-75de": "MIG-b229"}[FIRST_DEVICE[seed]])
            for c in CONFIGS:
                e = recs[m, c]["efficiency"]
                bt = [b["latency_s"] / b["n_reports"] * 1000 for b in e["batch_timings"]]
                rows.append({"seed": seed, "model": m, "precision": c.replace("-BnB", ""), "device": dev,
                             "ms_per_report": e["amortized_latency_ms"],
                             "ms_per_token": 1000 * e["total_generation_s"] / e["generated_tokens"],
                             "peak_alloc_mib": e["peak_allocated_bytes"] / 2**20,
                             "batch_ms_per_report_cv": float(np.std(bt, ddof=1) / np.mean(bt))})
    import pandas as pd

    df = pd.DataFrame(rows)
    summ = []
    for (m, c), g in df.groupby(["model", "precision"], sort=False):
        n = len(g)
        t = 2.776  # t_{0.975, 4}
        for col in ("ms_per_report", "ms_per_token", "peak_alloc_mib"):
            pass
        summ.append({"model": m, "precision": c, "n_runs": n,
                     "ms_report_mean": round(g.ms_per_report.mean(), 1), "ms_report_sd": round(g.ms_per_report.std(ddof=1), 1),
                     "ms_report_ci_half": round(t * g.ms_per_report.std(ddof=1) / np.sqrt(n), 1),
                     "ms_token_mean": round(g.ms_per_token.mean(), 2), "ms_token_sd": round(g.ms_per_token.std(ddof=1), 2),
                     "ms_token_cv_pct": round(100 * g.ms_per_token.std(ddof=1) / g.ms_per_token.mean(), 1),
                     "mem_mib_mean": round(g.peak_alloc_mib.mean(), 0), "mem_mib_sd": round(g.peak_alloc_mib.std(ddof=1), 1)})
    s = pd.DataFrame(summ)
    # Ratios vs FP32 within each run, then summarised across runs
    piv = df.pivot_table(index=["seed", "model"], columns="precision", values="ms_per_token")
    ratio = {p: piv[p] / piv["FP32"] for p in ("FP16", "INT8", "NF4")}
    ratio_summary = {p: {"min_over_runs": round(float(r.min()), 2), "max_over_runs": round(float(r.max()), 2)} for p, r in ratio.items()}
    slowest_int8 = int((piv["INT8"] > piv[["FP32", "FP16", "NF4"]].max(axis=1)).sum())
    nf4_slower = int((piv["NF4"] > piv["FP32"]).sum())
    fp16_not_slower = int((piv["FP16"] <= piv["FP32"] * 1.05).sum())
    # Device effect: seed 42 ran each model on the other MIG than seeds 123-31415
    dev = df.groupby(["model", "precision", "device"]).ms_per_token.mean().unstack("device")
    dev_ratio = (dev["MIG-b229"] / dev["MIG-75de"]).dropna()
    return df, s, {"ms_per_token_ratio_vs_fp32": ratio_summary, "runs": int(len(piv)),
                   "INT8_slowest_runs": slowest_int8, "NF4_slower_than_FP32_runs": nf4_slower,
                   "FP16_within_5pct_or_faster_runs": fp16_not_slower,
                   "device_ratio_b229_over_75de_ms_token": {"min": round(float(dev_ratio.min()), 2), "max": round(float(dev_ratio.max()), 2),
                                                            "median": round(float(dev_ratio.median()), 2)},
                   "within_run_batch_cv_pct_median": round(100 * float(df.batch_ms_per_report_cv.median()), 1),
                   "ms_token_between_run_cv_pct_range": [float(s.ms_token_cv_pct.min()), float(s.ms_token_cv_pct.max())]}


# ---------------------------------------------------------------- R2.6 tolerance
def tolerance(den_rows, all_runs):
    manifest, recs = all_runs[42]
    rows42 = [r for r in den_rows if r["seed"] == 42]
    out = {"bleu_tolerance": {}, "fnf_upper_ci_tolerance": {}, "critical_category_fnf": {}}
    for tol in (0.5, 1.0, 2.0):
        acc = [r for r in rows42 if abs(recs[r["model"], {"FP16": "FP16", "INT8": "INT8-BnB", "NF4": "NF4-BnB"}[r["precision"]]]["bleu_4"]
                                           - recs[r["model"], "FP32"]["bleu_4"]) * 100 <= tol]
        out["bleu_tolerance"][tol] = {"accepted": len(acc), "accepted_with_fnf_ge_30": sum(r["FNF"] >= 0.30 for r in acc)}
    for tau in (0.05, 0.10, 0.20, 0.30):
        out["fnf_upper_ci_tolerance"][tau] = {
            "accepted": [f"{r['model']}:{r['precision']}" for r in rows42 if r["FNF_ci_high"] <= tau and r["FPF_ci_high"] <= tau / 2]}
    critical = ["Pneumothorax", "Edema", "Consolidation", "Pneumonia", "Pleural Effusion"]
    for m in MODELS:
        base = [labels(p) for p in recs[m, "FP32"]["predictions"]]
        for c in REDUCED:
            q = [labels(p) for p in recs[m, c]["predictions"]]
            pos = sum(x[k] for x in base for k in critical)
            lost = sum(x[k] and not y[k] for x, y in zip(base, q) for k in critical)
            out["critical_category_fnf"][f"{m}:{SHORT[c]}"] = {"fp32_pos": pos, "lost": lost,
                                                                "FNF": None if not pos else round(lost / pos, 3)}
    out["critical_categories"] = critical
    return out


def restricted_categories(df, drop):
    """FNF recomputed after removing low-agreement categories (R1.6)."""
    sub = df[(df.fp32_pos == 1) & (~df.category.isin(drop))]
    v = sub.groupby(["seed", "model", "precision"]).flip.mean().unstack("precision")
    s42 = v.loc[42]
    return {"dropped": drop,
            "seed42_fnf": {m: {p: round(float(s42.loc[m, p]), 4) for p in ("FP16", "INT8", "NF4")} for m in MODELS},
            "seed42_denominators": {m: int(((sub.seed == 42) & (sub.model == m) & (sub.precision == "NF4")).sum()) for m in MODELS},
            "strict_runs": int(((v.FP16 < v.INT8) & (v.INT8 < v.NF4)).sum()),
            "non_decreasing_runs": int(((v.FP16 <= v.INT8) & (v.INT8 <= v.NF4)).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--openi", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    corpus = json.loads((ROOT / "data/iuxray_paired.json").read_text())
    all_runs = {s: load_run(s) for s in RUNS}
    summary = {}

    flow, split_rows, usable = data_flow(args.openi, corpus)
    summary["data_flow"] = flow
    write_csv(out / "split_ids.csv", split_rows)
    # held-out order check against the archived manifest
    assert [r["uid"] for r in usable["test"]] == all_runs[42][0]["sample_ids"], "held-out order mismatch"

    ov, flagged = overlap(usable)
    summary["overlap"] = ov
    dup_findings = flagged[("test", "findings")][0] | flagged[("test", "findings")][1]
    ids = all_runs[42][0]["sample_ids"]
    keep = [i for i, u in enumerate(ids) if u not in dup_findings]
    sens = {}
    for seed, (manifest, recs) in all_runs.items():
        full = audit_subset(manifest, recs, list(range(len(ids))))
        sub = audit_subset(manifest, recs, keep)
        sens[seed] = {"full": full, "dedup": sub, "ordering_full": ordering(full), "ordering_dedup": ordering(sub)}
    summary["overlap_sensitivity"] = {
        "excluded_test_reports": len(ids) - len(keep), "kept": len(keep),
        "seed42_dedup_rows": sens[42]["dedup"], "seed42_full_rows": sens[42]["full"],
        "strict_ordering_dedup_runs": sum(v["strict"] for s in sens.values() for v in s["ordering_dedup"].values()),
        "non_decreasing_dedup_runs": sum(v["non_decreasing"] for s in sens.values() for v in s["ordering_dedup"].values()),
        "strict_ordering_full_runs": sum(v["strict"] for s in sens.values() for v in s["ordering_full"].values()),
    }
    write_csv(out / "overlap_sensitivity_all_seeds.csv",
              [dict(seed=s, subset=k, **r) for s, v in sens.items() for k in ("full", "dedup") for r in v[k]])

    den, percat = denominators(all_runs)
    write_csv(out / "denominators_ci_all_precisions.csv", den)
    write_csv(out / "per_category_transitions.csv", percat)
    # sanity: reproduce archived seed-42 rates
    for r in den:
        if r["seed"] == 42:
            arch = all_runs[42][1][r["model"], {"FP16": "FP16", "INT8": "INT8-BnB", "NF4": "NF4-BnB"}[r["precision"]]]
            assert abs(arch["FNF_rate"] - r["FNF"]) < 1e-3 and abs(arch["FPF_rate"] - r["FPF"]) < 1e-3, r
    summary["denominators_seed42"] = [r for r in den if r["seed"] == 42]

    summary["extractor"] = extractor_per_category(RUNS[42])
    summary["strata"] = strata(all_runs)

    df = long_table(all_runs)
    summary["hierarchical"] = hierarchical(df)
    summary["restricted_categories"] = restricted_categories(
        df, ["Consolidation", "Enlarged Cardiomediastinum", "Lung Opacity", "Support Devices"])

    eff_rows, eff_sum, eff_info = efficiency(all_runs)
    eff_rows.to_csv(out / "efficiency_all_runs.csv", index=False)
    eff_sum.to_csv(out / "efficiency_repeated_summary.csv", index=False)
    summary["efficiency"] = eff_info

    summary["tolerance"] = tolerance(den, all_runs)
    (out / "revision_analyses.json").write_text(json.dumps(summary, indent=1, default=str))
    print("wrote", out)


if __name__ == "__main__":
    main()
