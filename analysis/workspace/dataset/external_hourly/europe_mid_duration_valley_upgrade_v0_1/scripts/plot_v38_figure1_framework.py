#!/usr/bin/env python3
"""Conceptual framework for the European observed-gas mainline."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures" / "europe_observed_gas_mainline_v0_1"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "savefig.dpi": 600,
    }
)

TEAL = "#009E73"
BLUE = "#0072B2"
ORANGE = "#D55E00"
PURPLE = "#6A51A3"
GREY = "#555555"
LIGHT = "#F4F7F6"


def box(ax, xy, width, height, title, body, edge=TEAL):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        facecolor="white",
        edgecolor=edge,
        linewidth=1.1,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + 0.03, xy[1] + height - 0.06, title, weight="bold", color=edge, va="top")
    ax.text(xy[0] + 0.03, xy[1] + height - 0.15, body, color="#333333", va="top", linespacing=1.35)


def main():
    fig, ax = plt.subplots(figsize=(7.2, 3.05))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1, 1, color=LIGHT, zorder=-5))

    # Empirical input
    box(
        ax,
        (0.035, 0.14),
        0.24,
        0.72,
        "Observed European systems",
        "15 countries · 132 country–years\n1.16 million observed hours",
        edge=BLUE,
    )
    for i, height in enumerate([0.18, 0.28, 0.14, 0.35, 0.22, 0.31]):
        x = 0.065 + 0.028 * i
        ax.add_patch(Rectangle((x, 0.22), 0.016, height, color=BLUE, alpha=0.75))
    ax.plot([0.06, 0.235], [0.56, 0.56], color=ORANGE, lw=1.0)
    ax.text(0.148, 0.58, "actual fossil-gas task", ha="center", color=ORANGE, fontsize=6)

    # Paired intervention
    box(
        ax,
        (0.38, 0.14),
        0.24,
        0.72,
        "Chronology intervention",
        "same blocks · annual energy · recharge",
        edge=TEAL,
    )
    x = np.linspace(0.405, 0.585, 12)
    dispersed = np.array([0.53, 0.29, 0.51, 0.28, 0.55, 0.29, 0.52, 0.28, 0.54, 0.30, 0.50, 0.28])
    clustered = np.array([0.27, 0.29, 0.30, 0.50, 0.54, 0.52, 0.49, 0.28, 0.30, 0.27, 0.31, 0.28])
    ax.plot(x, dispersed, color=GREY, marker="o", ms=2.2, lw=0.9)
    ax.plot(x, clustered, color=TEAL, marker="s", ms=2.2, lw=0.9)
    ax.text(0.50, 0.57, "dispersed", color=GREY, ha="center", fontsize=6)
    ax.text(0.50, 0.39, "clustered", color=TEAL, ha="center", fontsize=6)
    ax.text(0.50, 0.21, "reorder intact 24 h blocks", ha="center", fontsize=6, color="#333333")

    # Mechanism and value
    box(
        ax,
        (0.725, 0.14),
        0.24,
        0.72,
        "Duration–value outcome",
        "cycles · reliability · system cost",
        edge=PURPLE,
    )
    xx = np.linspace(0.755, 0.93, 120)
    turnover = 0.59 - 1.05 * (xx - 0.755)
    inventory = 0.25 + 2.0 * np.maximum(xx - 0.855, 0)
    ax.plot(xx, turnover, color=TEAL, lw=1.7)
    ax.plot(xx, inventory, color=PURPLE, lw=1.7)
    ax.plot([0.755, 0.93], [0.39, 0.39], color="#888888", lw=0.7, ls="--")
    ax.fill_between(xx, 0.23, 0.39, where=(xx > 0.84) & (xx < 0.885), color=ORANGE, alpha=0.17)
    ax.text(0.845, 0.205, "conditional\nmid-duration gap", color=ORANGE, ha="center", fontsize=5.7)
    ax.text(0.77, 0.60, "reuse value", color=TEAL, fontsize=5.7)
    ax.text(0.875, 0.46, "firm value", color=PURPLE, fontsize=5.7)
    ax.text(0.842, 0.16, "storage duration", ha="center", fontsize=6)

    for start, end in [((0.285, 0.50), (0.37, 0.50)), ((0.63, 0.50), (0.715, 0.50))]:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.1,
                color="#777777",
            )
        )

    ax.text(
        0.5,
        0.955,
        "Does clustering increase storage duration needs—or reduce the reuse that pays for them?",
        ha="center",
        va="top",
        fontsize=9,
        weight="bold",
        color="#222222",
    )
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "fig_v38_figure1_europe_framework"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=600)
    fig.savefig(stem.with_suffix(".tiff"), bbox_inches="tight", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    with Image.open(stem.with_suffix(".png")) as image:
        image.convert("L").save(OUT / "fig_v38_figure1_europe_framework_grayscale.png", dpi=(300, 300))
    plt.close(fig)


if __name__ == "__main__":
    main()
