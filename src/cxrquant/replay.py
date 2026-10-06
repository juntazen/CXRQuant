"""Deterministic report-level bootstrap, McNemar, tables and figures."""

import csv
import json
from math import comb
from pathlib import Path

from cxrquant.clinical_safety.fact_extractor import (
    extract_labels,
    labels_to_binary,
    _PATHOLOGY_CATEGORIES,
)
from cxrquant.runio import atomic_json, object_hash, sha256_file
from cxrquant.validation import validate_benchmark


def exact_mcnemar(b, c):
    if type(b) is not int or type(c) is not int or min(b, c) < 0:
        raise ValueError("McNemar counts must be nonnegative integers")
    n = b + c
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / (2**n)) if n else 1.0


def report_counts(base, quant):
    import numpy as np

    if len(base) != len(quant) or not base:
        raise ValueError("paired nonempty reports required")
    # Columns: lost, gained, baseline present, baseline absent, unchanged, pairs.
    rows = []
    percat = {cat: {"b": 0, "c": 0} for cat in _PATHOLOGY_CATEGORIES}
    for b, q in zip(base, quant, strict=True):
        row = [0] * 6
        for cat in _PATHOLOGY_CATEGORIES:
            bv, qv = b[cat], q[cat]
            lost, gained = int(bv == 1 and qv == 0), int(bv == 0 and qv == 1)
            row = [
                x + y
                for x, y in zip(row, [lost, gained, bv, 1 - bv, int(bv == qv), 1], strict=True)
            ]
            percat[cat]["b"] += lost
            percat[cat]["c"] += gained
        rows.append(row)
    return np.asarray(rows, dtype="int64"), percat


def statistics(bench, manifest, seed=42, resamples=2000):
    import numpy as np

    validate_benchmark(bench, manifest)
    if resamples < 1:
        raise ValueError("resamples must be positive")
    rng = np.random.default_rng(seed)
    boots, pooled, categories = [], [], []
    for model in bench["models"]:
        base = [
            labels_to_binary(extract_labels(p), "positive") for p in model["predictions"]["FP32"]
        ]
        for cfg in manifest["configs"]:
            if cfg == "FP32":
                continue
            quant = [
                labels_to_binary(extract_labels(p), "positive") for p in model["predictions"][cfg]
            ]
            counts, percat = report_counts(base, quant)
            sampled = counts[rng.integers(0, len(counts), size=(resamples, len(counts)))].sum(
                axis=1
            )
            total = counts.sum(axis=0)
            for metric, num, den in [("FNF_rate", 0, 2), ("FPF_rate", 1, 3), ("CFPR", 4, 5)]:
                valid = sampled[:, den] > 0
                values = sampled[valid, num] / sampled[valid, den]
                ci = (
                    [float(x) for x in np.percentile(values, [2.5, 97.5])]
                    if len(values)
                    else [None, None]
                )
                boots.append(
                    {
                        "model": model["model"],
                        "config": cfg,
                        "metric": metric,
                        "point": float(total[num] / total[den]) if total[den] else None,
                        "ci95_low": ci[0],
                        "ci95_high": ci[1],
                        "undefined_resamples": int((~valid).sum()),
                        "n_reports": len(counts),
                    }
                )
            previous = boots[-1]
            boots.append(
                dict(
                    previous,
                    metric="CFFR",
                    point=1 - previous["point"],
                    ci95_low=1 - previous["ci95_high"],
                    ci95_high=1 - previous["ci95_low"],
                )
            )
            b, c = int(total[0]), int(total[1])
            pooled.append(
                {"model": model["model"], "config": cfg, "b": b, "c": c, "p": exact_mcnemar(b, c)}
            )
            for cat, values in percat.items():
                b, c = values["b"], values["c"]
                p = exact_mcnemar(b, c)
                categories.append(
                    {
                        "model": model["model"],
                        "config": cfg,
                        "category": cat,
                        "b": b,
                        "c": c,
                        "p": p,
                        "bonferroni_significant": p < 0.05 / len(percat),
                    }
                )
    return {
        "seed": seed,
        "resamples": resamples,
        "unit": "report",
        "undefined_policy": "null point; exclude zero-denominator resamples and report count",
        "bootstrap": boots,
        "pooled": pooled,
        "categories": categories,
    }


def write_csv(path, rows):
    if not rows:
        raise ValueError("empty table")
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def replay(run):
    run = Path(run)
    bench_path = run / "multimodel_benchmark.json"
    manifest_path = run / "manifest.json"
    bench = json.loads(bench_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    matrix = validate_benchmark(bench, manifest)
    cfg = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else {}
    stats = statistics(
        bench, manifest, cfg.get("bootstrap_seed", 42), cfg.get("bootstrap_resamples", 2000)
    )
    provenance = {
        "run_id": manifest["run_id"],
        "benchmark_hash": sha256_file(bench_path),
        "manifest_hash": sha256_file(manifest_path),
        "config_hash": object_hash(cfg),
        "metric_version": "2.1-null-denominator",
        "prediction_provenance": bench.get(
            "prediction_provenance", "GPU run; see identity and checkpoints"
        ),
    }
    atomic_json(
        run / "bootstrap_fnf_ci.json",
        {
            "provenance": provenance,
            "seed": stats["seed"],
            "resamples": stats["resamples"],
            "unit": stats["unit"],
            "undefined_policy": stats["undefined_policy"],
            "rows": stats["bootstrap"],
        },
    )
    atomic_json(
        run / "mcnemar_pooled.json",
        {
            "provenance": provenance,
            "note": "Exploratory: categories nested within reports; no independent-pair clinical inference",
            "rows": stats["pooled"],
        },
    )
    atomic_json(
        run / "mcnemar_results.json", {"provenance": provenance, "rows": stats["categories"]}
    )
    tables = run / "tables"
    tables.mkdir(exist_ok=True)
    metric_rows = []
    for model in bench["models"]:
        for c in model["configs"]:
            metric_rows.append(
                {
                    "model": model["model"],
                    "config": c["config"],
                    **{
                        k: c.get(k)
                        for k in (
                            "FNF_rate",
                            "FPF_rate",
                            "CFPR",
                            "CFFR",
                            "ce_micro_f1",
                            "ce_macro_f1",
                            "bleu_1",
                            "bleu_4",
                            "rouge_1",
                            "rouge_2",
                            "rouge_l",
                        )
                    },
                }
            )
    write_csv(tables / "metrics.csv", metric_rows)
    write_csv(tables / "bootstrap.csv", stats["bootstrap"])
    write_csv(tables / "mcnemar_pooled.csv", stats["pooled"])
    comparator_path = run / "extractor_comparison.json"
    if comparator_path.exists():
        comparator = json.loads(comparator_path.read_text())
        if comparator.get("provenance", {}).get("run_id") != manifest["run_id"]:
            raise ValueError("Comparator belongs to another run")
        write_csv(
            tables / "extractor_comparison.csv",
            [
                {
                    k: r.get(k)
                    for k in (
                        "extractor",
                        "available",
                        "precision",
                        "recall",
                        "f1",
                        "dependencies",
                        "error",
                    )
                }
                for r in comparator["results"]
            ],
        )
    atomic_json(tables / "provenance.json", provenance)
    figures = run / "figures"
    figures.mkdir(exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = manifest["models"]
    configs = [c for c in manifest["configs"] if c != "FP32"]
    values = np.asarray(
        [
            [
                next(
                    c["FNF_rate"]
                    for m in bench["models"]
                    if m["model"] == name
                    for c in m["configs"]
                    if c["config"] == cfg
                )
                for cfg in configs
            ]
            for name in names
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(7, max(3, len(names) * 0.55)))
    im = ax.imshow(values, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(configs)), configs)
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(configs)):
            ax.text(
                j,
                i,
                "undefined" if np.isnan(values[i, j]) else f"{values[i, j]:.1%}",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if values[i, j] > 0.65 else "black",
            )
    ax.set_title("Extracted-category loss relative to FP32")
    fig.colorbar(im, ax=ax, label="FNF rate")
    fig.tight_layout()
    fig.savefig(figures / "fnf_matrix.png", dpi=180)
    plt.close(fig)
    atomic_json(figures / "provenance.json", provenance)
    summary = {
        "status": "complete",
        "matrix": matrix,
        "provenance": provenance,
        "validated_metrics": ["FNF", "FPF", "CFPR", "CFFR", "CE-F1", "BLEU-1/4", "ROUGE-1/2/L"],
    }
    atomic_json(run / "replay_status.json", summary)
    print(json.dumps(summary))
    return summary
