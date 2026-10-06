#!/usr/bin/env python3
"""Journal figures for the IJIES manuscript, grouped by generation-quality stratum.

Figure 2 (three axes): panel A shows FP32-relative FNF under FP16, INT8 and NF4;
panel B shows signed NF4-minus-FP32 changes in BLEU-4 and reference CE micro-F1.
Figure 3 (descriptive scale plot, full version only): NF4 FNF against parameter
count, general versus biomedical pretraining, with functional checkpoints filled
and degenerate checkpoints hollow. No in-figure titles; 10-pt text throughout.
"""
import argparse
import csv
import json
from pathlib import Path

ORDER = [("Functional", ["biogpt", "pythia-1b", "biogpt-large"]),
         ("Degenerate", ["distilgpt2", "gpt2", "gpt2-medium"])]
NAMES = {"distilgpt2": "DistilGPT-2", "gpt2": "GPT-2", "gpt2-medium": "GPT-2 Medium", "biogpt": "BioGPT",
         "pythia-1b": "Pythia-1B", "biogpt-large": "BioGPT-Large"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLEU, CE = "#eb6834", "#2a78d6"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    run = args.run_dir.resolve()
    m = {(r["model"], r["config"]): r for r in csv.DictReader((run / "tables" / "metrics.csv").open())}
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
    models = [x for _, group in ORDER for x in group]
    labels = [NAMES[x] for x in models]
    cfgs = ["FP16", "INT8-BnB", "NF4-BnB"]
    fnf = np.array([[100 * float(m[x, c]["FNF_rate"]) for c in cfgs] for x in models])
    d_bleu = np.array([100 * (float(m[x, "NF4-BnB"]["bleu_4"]) - float(m[x, "FP32"]["bleu_4"])) for x in models])
    d_ce = np.array([100 * (float(m[x, "NF4-BnB"]["ce_micro_f1"]) - float(m[x, "FP32"]["ce_micro_f1"])) for x in models])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 4.0), gridspec_kw={"width_ratios": [1.25, 1]})
    im = ax1.imshow(fnf, cmap="YlGnBu", vmin=0, vmax=85, aspect="auto")
    ax1.set_xticks(range(3), ["FP16", "INT8", "NF4"])
    ax1.set_yticks(range(len(models)), labels)
    for i in range(len(models)):
        for j in range(3):
            ax1.text(j, i, f"{fnf[i, j]:.1f}%", ha="center", va="center", fontsize=9,
                     color="white" if fnf[i, j] > 45 else INK)
    ax1.axhline(2.5, color=INK, linewidth=1.6)
    ax1.set_title("A", loc="left", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.03)
    cbar.set_label("FP32-relative FNF (%)")

    y = np.arange(len(models))
    ax2.barh(y - 0.19, d_bleu, 0.36, color=BLEU, label="BLEU-4")
    ax2.barh(y + 0.19, d_ce, 0.36, color=CE, label="CE micro-F1")
    ax2.axvline(0, color=MUTED, linewidth=0.8)
    ax2.axhline(2.5, color=INK, linewidth=1.6)
    ax2.set_yticks(y, [""] * len(models))
    ax2.invert_yaxis()
    ax2.set_ylim(len(models) - 0.5, -0.5)
    ax2.grid(axis="x", color=GRID, linewidth=0.6)
    ax2.set_axisbelow(True)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    ax2.set_xlabel("NF4 minus FP32 (pp)")
    ax2.set_title("B", loc="left", fontweight="bold")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
    fig.text(0.005, 0.70, "Functional", rotation=90, va="center", fontsize=10, color=MUTED)
    fig.text(0.005, 0.30, "Degenerate", rotation=90, va="center", fontsize=10, color=MUTED)
    fig.tight_layout(rect=(0.02, 0, 1, 1))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_dir / "fig2_three_axes_strata.png", dpi=300, facecolor="white")
    plt.close(fig)

    training = {x: json.loads((run / "models" / x / "training.json").read_text()) for x in models}
    cfg = json.loads((run / "config.json").read_text())
    family = {s["name"]: s["family"] for s in cfg["models"]}
    functional = set(ORDER[0][1])
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    for x in models:
        marker = "s" if family[x] == "biomedical" else "o"
        colour = CE if family[x] == "biomedical" else BLEU
        filled = x in functional
        ax.scatter(training[x]["params_m"], 100 * float(m[x, "NF4-BnB"]["FNF_rate"]), s=46, marker=marker,
                   facecolors=colour if filled else "white", edgecolors=colour, linewidths=1.4, zorder=3)
        ax.annotate(NAMES[x], (training[x]["params_m"], 100 * float(m[x, "NF4-BnB"]["FNF_rate"])), xytext=(4, 4),
                    textcoords="offset points", fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (millions, log scale)")
    ax.set_ylabel("NF4 FP32-relative FNF (%)")
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], linestyle="", marker="o", markerfacecolor=BLEU, markeredgecolor=BLEU, label="General"),
               Line2D([], [], linestyle="", marker="s", markerfacecolor=CE, markeredgecolor=CE, label="Biomedical"),
               Line2D([], [], linestyle="", marker="o", markerfacecolor="white", markeredgecolor=MUTED, label="Degenerate (hollow)")]
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(args.out_dir / "fig3_scale_strata.png", dpi=300, facecolor="white")
    plt.close(fig)
    print(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
