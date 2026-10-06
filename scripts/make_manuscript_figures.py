#!/usr/bin/env python3
"""Render corrected-run figures used by the revised manuscript."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics, training = read_rows(args.metrics), read_rows(args.training)
    models = [row["model"] for row in training]
    precisions = ["FP16", "INT8-BnB", "NF4-BnB"]
    fnf = np.array([[float(next(r["FNF_rate"] for r in metrics if r["model"] == m and r["config"] == p)) for p in precisions] for m in models])
    delta_bleu, delta_ce = [], []
    for model in models:
        cfg = {r["config"]: r for r in metrics if r["model"] == model}
        delta_bleu.append(float(cfg["NF4-BnB"]["bleu_4"]) - float(cfg["FP32"]["bleu_4"]))
        delta_ce.append(float(cfg["NF4-BnB"]["ce_micro_f1"]) - float(cfg["FP32"]["ce_micro_f1"]))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8.825), gridspec_kw={"width_ratios": [1, 1.15]})
    image = ax1.imshow(fnf * 100, cmap="YlGnBu", vmin=0, vmax=85, aspect="auto")
    ax1.set_xticks(range(3), ["FP16", "INT8", "NF4"])
    ax1.set_yticks(range(len(models)), models)
    ax1.set_title("A. False-negative flips vs FP32")
    for i in range(len(models)):
        for j in range(3):
            ax1.text(j, i, f"{fnf[i, j] * 100:.1f}%", ha="center", va="center",
                     color="white" if fnf[i, j] > 0.5 else "black")
    y = np.arange(len(models))
    ax2.axvline(0, color="#333333", linewidth=0.8)
    ax2.barh(y - 0.18, np.array(delta_bleu) * 100, 0.34, label="BLEU-4", color="#d97706")
    ax2.barh(y + 0.18, np.array(delta_ce) * 100, 0.34, label="CE micro-F1", color="#0f766e")
    ax2.set_yticks(y, models)
    ax2.invert_yaxis()
    ax2.set_xlabel("NF4 minus FP32 (percentage points)")
    ax2.set_title("B. Reference-based score changes")
    ax2.legend(loc="lower right", frameon=False)
    ax2.grid(axis="x", color="#d1d5db", linewidth=0.6)
    fig.colorbar(image, ax=ax1, fraction=0.046, pad=0.04, label="FNF (%)")
    fig.suptitle("Corrected seed-42 GPU benchmark", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out_dir / "figure3_corrected_precision.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 9.7))
    colors, markers = {"general": "#2563eb", "biomedical": "#dc2626"}, {"general": "o", "biomedical": "s"}
    for family in ("general", "biomedical"):
        selected = [row for row in training if row["family"] == family]
        x = [float(row["params_m"]) for row in selected]
        y = [100 * float(next(r["FNF_rate"] for r in metrics if r["model"] == item["model"] and r["config"] == "NF4-BnB")) for item in selected]
        ax.scatter(x, y, s=100, color=colors[family], marker=markers[family],
                   label=family.capitalize(), edgecolor="white", linewidth=0.8)
        for item, px, py in zip(selected, x, y, strict=True):
            ax.annotate(item["model"], (px, py), xytext=(7, 7), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Model parameters (millions, log scale)")
    ax.set_ylabel("NF4 false-negative flip rate vs FP32 (%)")
    ax.set_title("Corrected NF4 differential flips by model size and pretraining family")
    ax.grid(color="#d1d5db", linewidth=0.6)
    ax.legend(frameon=False)
    ax.text(0.01, 0.01, "Descriptive only: architecture, tokenizer, epochs, and pretraining are not controlled.",
            transform=ax.transAxes, fontsize=9, color="#374151")
    fig.tight_layout()
    fig.savefig(args.out_dir / "figure4_corrected_scale.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
