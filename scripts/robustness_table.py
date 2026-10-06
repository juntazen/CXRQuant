#!/usr/bin/env python3
"""Rows of the manuscript's robustness table: NF4 FNF range and FP16 < INT8 < NF4 ordering per specification.

Every row is recomputed from stored outputs (archived runs, revision analyses, GPU reruns).
  python scripts/robustness_table.py --results ../results --out ../results/robustness.json
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
MODELS = ["distilgpt2", "gpt2", "gpt2-medium", "biogpt", "pythia-1b", "biogpt-large"]
RED = ["FP16", "INT8-BnB", "NF4-BnB"]


def order(v):
    return bool(v[0] < v[1] < v[2]), bool(v[0] <= v[1] <= v[2])


def row(name, triples, note=""):
    s = [order(t) for t in triples]
    nf4 = [t[2] for t in triples]
    return {"analysis": name, "runs": len(triples), "nf4_min": round(100 * min(nf4), 1), "nf4_max": round(100 * max(nf4), 1),
            "strict": sum(x[0] for x in s), "non_decreasing": sum(x[1] for x in s), "note": note}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import revision_analyses as RA

    run42 = ROOT / "runs/full_seed42_corrected_20260911"
    runs = {s: RA.load_run(s) for s in RA.RUNS}
    rows = []
    prim = [[runs[42][1][m, c]["FNF_rate"] for c in RED] for m in MODELS]
    rows.append(row("Primary specification, seed 42", prim))
    rows.append(row("Five training seeds", [[runs[s][1][m, c]["FNF_rate"] for c in RED] for s in runs for m in MODELS]))
    unc = json.loads((run42 / "ablation_uncertain_policy.json").read_text())["summary"]["ordering"]["negative"]
    rows.append(row("Uncertain mapped to negative", [[unc[m]["FNF"][c] for c in RED] for m in MODELS]))
    lab = json.loads((run42 / "baseline_evaluator_comparison.json").read_text())["fnf_ordering"]
    rows.append(row("CheXbert labels", [[lab["chexbert"][m]["FNF"][c] for c in RED] for m in MODELS]))
    rows.append(row("CheXpert labeller", [[lab["chexpert"][m]["FNF"][c] for c in RED] for m in MODELS]))
    pc = json.loads((run42 / "pycontext_audit.json").read_text())["per_extractor"]["pycontextnlp"]["per_model"]
    rows.append(row("pyConTextNLP, nine categories", [[pc[m][c]["FNF_rate"] for c in RED] for m in MODELS]))
    dd = [r for r in csv.DictReader((a.results / "overlap_sensitivity_all_seeds.csv").open()) if r["subset"] == "dedup"]
    g = {}
    for r in dd:
        g.setdefault((r["seed"], r["model"]), {})[r["precision"]] = float(r["FNF"])
    rows.append(row("Near-duplicate held-out reports removed", [[v["FP16"], v["INT8"], v["NF4"]] for v in g.values()]))
    df = RA.long_table(runs)
    sub = df[(df.fp32_pos == 1) & (~df.category.isin(["Consolidation", "Enlarged Cardiomediastinum", "Lung Opacity", "Support Devices"]))]
    v = sub.groupby(["seed", "model", "precision"]).flip.mean().unstack("precision")
    rows.append(row("Four low-agreement categories removed", [[r.FP16, r.INT8, r.NF4] for r in v.itertuples()]))
    X = json.loads((a.results / "rerun_summary.json").read_text())
    ef = X["eosfix"]
    rows.append(row("EOS exempt from repetition penalty", [[x[p]["FNF"] for p in ("FP16", "INT8", "NF4")]
                                                          for s in ef.values() for x in s["models"].values()]))
    ab = X["ablation_impression_only_seed42"]["models"]
    rows.append(row("Loss on IMPRESSION tokens only", [[x[p]["FNF"] for p in ("FP16", "INT8", "NF4")] for x in ab.values()]))
    ll = X["llm7b"]["models"]["mistral-7b"]
    rows.append(row("Mistral-7B-Instruct, LoRA", [[ll[p]["FNF"] for p in ("FP16", "INT8", "NF4")]]))
    for key, name in (("chexagent", "CheXagent-8b, IU-XRAY radiographs"), ("chexagent_nih", "CheXagent-8b, NIH ChestX-ray14")):
        c = X[key]
        rows.append(row(name, [[c[p]["FNF"] for p in RED]]))
    a.out.write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
