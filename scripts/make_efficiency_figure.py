#!/usr/bin/env python3
"""Figure: measured efficiency against FP32-relative finding stability.

Panel A plots peak allocated memory reduction and panel B per-report latency
relative to FP32 against FNF for every reduced-precision configuration. Each
panel has one y-axis. Precision is encoded by colour and marker shape; stratum
by filled (functional) or hollow (degenerate) markers.
"""
import argparse
import json
from pathlib import Path

PRECISIONS = {
    "FP16": ("#2a78d6", "o"),
    "INT8-BnB": ("#eb6834", "s"),
    "NF4-BnB": ("#1baf7a", "^"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    run = args.run_dir.resolve()
    eff = {(r["model"], r["config"]): r for r in json.loads((run / "efficiency_summary.json").read_text())["rows"]}
    ablation = json.loads((run / "ablation_stratum.json").read_text())
    functional = set(ablation["strata"]["functional"])
    fnf = {}
    for row in json.loads((run / "ablation_uncertain_policy.json").read_text())["rows"]:
        fnf[row["model"], row["config"]] = row["FNF_positive"]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), sharey=True)
    panels = (("peak_allocated_reduction_pct", "Peak allocated memory reduction vs FP32 (%)", "A"),
              ("latency_ratio_vs_fp32", "Per-report latency relative to FP32 (×)", "B"))
    for ax, (key, label, tag) in zip(axes, panels):
        for cfg, (colour, marker) in PRECISIONS.items():
            for (model, c), value in fnf.items():
                if c != cfg:
                    continue
                filled = model in functional
                ax.scatter(eff[model, cfg][key], 100 * value, s=42, marker=marker, linewidths=1.4,
                           facecolors=colour if filled else "white", edgecolors=colour, zorder=3)
        ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel(label)
        ax.set_title(tag, loc="left", fontweight="bold", color=INK)
    axes[1].axvline(1.0, color=MUTED, linewidth=0.8, linestyle="--", zorder=1)
    axes[0].set_ylabel("FP32-relative FNF (%)")
    axes[0].set_ylim(-3, 85)
    handles = [Line2D([], [], linestyle="", marker=m, markersize=6, markerfacecolor=c, markeredgecolor=c, label=p)
               for p, (c, m) in PRECISIONS.items()]
    handles += [Line2D([], [], linestyle="", marker="o", markersize=6, markerfacecolor=MUTED, markeredgecolor=MUTED,
                       label="functional stratum"),
                Line2D([], [], linestyle="", marker="o", markersize=6, markerfacecolor="white", markeredgecolor=MUTED,
                       label="degenerate stratum")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, facecolor="white")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
