#!/usr/bin/env python3
"""Draw HealthSec-AI architecture diagram for the manuscript."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FIG = Path(__file__).resolve().parents[2] / "manuscript" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def box(ax, xy, w, h, text, fc="#e8f1f5", ec="#2c5f7c"):
    p = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(p)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=8)


def main():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    # No in-image "Figure 1" title — caption lives in the manuscript only.

    box(ax, (0.3, 4.3), 2.2, 1.2, "Dataset adapters\nFramingham / CinC2012\nSynthetic (+ UCI smoke)")
    box(ax, (3.0, 4.3), 2.0, 1.2, "Stress engine\nnoise / missingness\nrandom label flips")
    box(ax, (5.5, 4.3), 2.0, 1.2, "Mitigation engine\nclipping / anomaly\nabstention")
    box(ax, (7.9, 4.3), 1.8, 1.2, "Models\nLR / RF / XGB")

    box(ax, (1.5, 2.3), 3.0, 1.2, "Evaluation engine\nAUROC / AUPRC / Brier\ncalibration / bootstrap CIs\nrisk-coverage")
    box(ax, (5.0, 2.3), 2.5, 1.2, "Experiment tracker\nseed / configs\nCSV + JSON provenance")
    box(ax, (7.8, 2.3), 1.9, 1.2, "CLI\nhealthsec run\nlist-adapters")

    box(ax, (2.5, 0.5), 5.0, 1.1, "Outputs: benchmark tables, multi-seed flips, risk-coverage, figures")

    for x0, x1 in [(2.5, 3.0), (5.0, 5.5), (7.5, 7.9)]:
        ax.annotate(
            "",
            xy=(x1, 4.9),
            xytext=(x0, 4.9),
            arrowprops=dict(arrowstyle="->", color="#333"),
        )
    ax.annotate("", xy=(3.0, 3.5), xytext=(4.0, 4.3), arrowprops=dict(arrowstyle="->", color="#333"))
    ax.annotate("", xy=(6.2, 3.5), xytext=(6.5, 4.3), arrowprops=dict(arrowstyle="->", color="#333"))
    ax.annotate("", xy=(5.0, 1.6), xytext=(5.0, 2.3), arrowprops=dict(arrowstyle="->", color="#333"))

    fig.tight_layout()
    for name in ("Figure1_architecture.png", "FigureA1_architecture.png"):
        out = FIG / name
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        print(out)
    plt.close()


if __name__ == "__main__":
    main()
