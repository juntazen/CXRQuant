#!/usr/bin/env python3
"""Figures for the IJIES second-round revision, drawn at their printed size.

IJIES format rules applied: every letter is at least 10 pt at print size, a Times-metric serif
font is used, there are no in-figure titles, no (a)/(b) panel labels, no outer border, and each
figure is a single plot (the former two-panel Fig. 2 is split into Fig. 2 and Fig. 3).

  fig1_workflow.png  full text width (17.2 cm) - placed in a one-column page section
  fig2_fnf.png       column width (8.2 cm)     - FP32-relative FNF per model and precision
  fig3_axes.png      column width (8.2 cm)     - NF4 minus FP32 change in BLEU-4 and CE micro-F1
"""
import argparse
import csv
from pathlib import Path

ORDER = ["biogpt", "pythia-1b", "biogpt-large", "distilgpt2", "gpt2", "gpt2-medium"]
NAMES = {"distilgpt2": "DistilGPT-2", "gpt2": "GPT-2", "gpt2-medium": "GPT-2 Medium", "biogpt": "BioGPT",
         "pythia-1b": "Pythia-1B", "biogpt-large": "BioGPT-Large"}
INK, MUTED, GRID = "#111111", "#4a4a4a", "#dddddd"
BLEU, CE = "#c4501b", "#1f64b4"
CM = 1 / 2.54
import os
FONT_DIRS = [d for d in os.environ.get("CXRQUANT_FONT_DIRS", "/usr/share/fonts/truetype/liberation").split(":") if d]


def setup():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt

    family = "DejaVu Serif"
    for d in FONT_DIRS:
        for f in Path(d).glob("LiberationSerif-*.ttf") if Path(d).is_dir() else []:
            font_manager.fontManager.addfont(str(f))
            family = "Liberation Serif"
    plt.rcParams.update({"font.family": family, "font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 10,
                         "ytick.labelsize": 10, "legend.fontsize": 10, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "savefig.dpi": 600})
    return plt


def workflow(plt, out):
    from matplotlib.patches import FancyBboxPatch

    fig = plt.figure(figsize=(17.2 * CM, 6.4 * CM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 172)
    ax.set_ylim(0, 64)
    ax.axis("off")

    def box(x, y, w, h, text, fc, bold_first=True):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=fc, ec="#6b6b6b", lw=0.8))
        lines = text.split("\n")
        n = len(lines)
        for i, line in enumerate(lines):
            ax.text(x + w / 2, y + h / 2 + (n - 1) * 2.35 - i * 4.7, line, ha="center", va="center",
                    fontsize=10, fontweight="bold" if (bold_first and i == 0) else "normal", color=INK)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="#333333", lw=0.9, shrinkA=0, shrinkB=0))

    # Row 1: data -> training -> one checkpoint -> four precisions -> stored predictions
    y1, h1 = 40, 21
    box(1, y1, 31, h1, "Data\nIU-XRAY/OpenI\nFINDINGS to\nIMPRESSION", "#e3eefb")
    box(36, y1, 31, h1, "Training\n6 language\nmodels, 5 seeds", "#e3eefb")
    box(71, y1, 31, h1, "Checkpoint\none final\ncheckpoint\nper model", "#fbf1d0")
    box(106, y1, 31, h1, "Inference\nFP32, FP16,\nINT8, NF4 from\none checkpoint", "#ece3fb")
    box(141, y1, 30, h1, "Records\n3,216 stored\npredictions\nper seed", "#f2f2f2")
    for x in (32, 67, 102, 137):
        arrow(x + 0.3, y1 + h1 / 2, x + 3.7, y1 + h1 / 2)
    # Row 2: separate evaluation axes -> replay layer
    y2, h2 = 3, 25
    box(1, y2, 38, h2, "Lexical axis\nBLEU and ROUGE\nversus the reference", "#e2f4e6")
    box(44, y2, 38, h2, "Stability axis\nFNF and FPF versus\nthe same model's\nFP32 output", "#e2f4e6")
    box(87, y2, 38, h2, "Reference axis\nCE micro-F1\nversus the reference", "#e2f4e6")
    box(130, y2, 41, h2, "Replay layer\nverification,\nbootstrap, McNemar,\ntables, figures", "#fbe3e3")
    ax.plot([156, 156], [y1 - 0.5, 34], color="#333333", lw=0.9)
    ax.plot([20, 156], [34, 34], color="#333333", lw=0.9)
    for x in (20, 63, 106):
        arrow(x, 34, x, y2 + h2 + 0.6)
    arrow(156, 34, 156, y2 + h2 + 0.6)
    fig.savefig(out / "fig1_workflow.png", facecolor="white")
    plt.close(fig)


def fnf_heatmap(plt, m, out):
    import numpy as np

    cfgs = ["FP16", "INT8-BnB", "NF4-BnB"]
    fnf = np.array([[100 * float(m[x, c]["FNF_rate"]) for c in cfgs] for x in ORDER])
    fig, ax = plt.subplots(figsize=(8.2 * CM, 6.6 * CM))
    im = ax.imshow(fnf, cmap="Blues", vmin=0, vmax=90, aspect="auto")
    ax.set_xticks(range(3), ["FP16", "INT8", "NF4"])
    ax.set_yticks(range(len(ORDER)), [NAMES[x] for x in ORDER])
    for i in range(len(ORDER)):
        for j in range(3):
            ax.text(j, i, f"{fnf[i, j]:.1f}", ha="center", va="center", fontsize=10,
                    color="white" if fnf[i, j] > 50 else INK)
    ax.axhline(2.5, color=INK, linewidth=1.4)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.03)
    cbar.set_label("FP32-relative FNF (%)")
    cbar.outline.set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "fig2_fnf.png", facecolor="white")
    plt.close(fig)


def axes_change(plt, m, out):
    import numpy as np

    d_bleu = np.array([100 * (float(m[x, "NF4-BnB"]["bleu_4"]) - float(m[x, "FP32"]["bleu_4"])) for x in ORDER])
    d_ce = np.array([100 * (float(m[x, "NF4-BnB"]["ce_micro_f1"]) - float(m[x, "FP32"]["ce_micro_f1"])) for x in ORDER])
    fig, ax = plt.subplots(figsize=(8.2 * CM, 7.0 * CM))
    y = np.arange(len(ORDER))
    ax.barh(y - 0.2, d_bleu, 0.38, color=BLEU, label="BLEU-4")
    ax.barh(y + 0.2, d_ce, 0.38, color=CE, label="CE micro-F1")
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.axhline(2.5, color=INK, linewidth=1.4)
    ax.set_yticks(y, [NAMES[x] for x in ORDER])
    ax.set_ylim(len(ORDER) - 0.5, -0.5)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("NF4 minus FP32 (percentage points)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=2, frameon=False, handlelength=1.2)
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "fig3_axes.png", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    m = {(r["model"], r["config"]): r for r in csv.DictReader((a.run_dir / "tables" / "metrics.csv").open())}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    plt = setup()
    workflow(plt, a.out_dir)
    fnf_heatmap(plt, m, a.out_dir)
    axes_change(plt, m, a.out_dir)
    print(a.out_dir)


if __name__ == "__main__":
    main()
