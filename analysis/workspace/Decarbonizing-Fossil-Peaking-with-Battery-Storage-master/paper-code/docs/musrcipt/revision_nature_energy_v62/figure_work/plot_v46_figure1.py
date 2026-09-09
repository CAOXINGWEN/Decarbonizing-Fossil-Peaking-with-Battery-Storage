#!/usr/bin/env python3
"""Build the v46 single-canvas conceptual framework figure."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "main_figures"
SRC = ROOT / "source_data"
OUT.mkdir(parents=True, exist_ok=True)
SRC.mkdir(parents=True, exist_ok=True)

INK = "#223746"
MUTED = "#687C87"
GRID = "#D7E2E6"
PANEL = "#F5F8F8"
BLUE = "#2C77A8"
BLUE_LIGHT = "#D9EAF3"
TEAL = "#15958A"
TEAL_LIGHT = "#D8F0EC"
ORANGE = "#DD6B2C"
ORANGE_LIGHT = "#F8E2D7"
PURPLE = "#7862A9"
PURPLE_LIGHT = "#E7E0F2"
GOLD = "#D69C22"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.0,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.unicode_minus": False,
    }
)


def box(ax, xy, width, height, face="white", edge=GRID, radius=0.012, lw=0.7, z=2):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        transform=ax.transAxes,
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, color=MUTED, lw=0.9, size=7, z=8, style="-|>"):
    patch = FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle=style,
        mutation_scale=size,
        color=color,
        linewidth=lw,
        shrinkA=0,
        shrinkB=0,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def event_strip(ax, x0, y0, width, clustered=False):
    slots = np.linspace(x0, x0 + width, 8)
    heights = np.array([0.030, 0.050, 0.039, 0.056, 0.035, 0.049])
    if clustered:
        xpos = np.r_[slots[:3], slots[5:8]]
    else:
        xpos = np.linspace(x0, x0 + width, 6)
    for x, h in zip(xpos, heights):
        ax.add_patch(
            Rectangle(
                (x - 0.006, y0),
                0.012,
                h,
                transform=ax.transAxes,
                facecolor=ORANGE,
                edgecolor="white",
                linewidth=0.25,
                zorder=6,
            )
        )
    xx = np.linspace(x0 - 0.006, x0 + width + 0.006, 180)
    yy = y0 - 0.030 + 0.009 * np.sin(np.linspace(0, 8 * np.pi, len(xx)))
    ax.plot(xx, yy, transform=ax.transAxes, color=TEAL, lw=1.15, zorder=5)


def battery(ax, xy, width, height, color, cells=4, fill_fraction=0.75, label=None):
    x, y = xy
    box(ax, (x, y), width, height, face=mpl.colors.to_rgba(color, 0.10), edge=color, radius=0.006, lw=0.7, z=5)
    inner = width - 0.018
    gap = 0.004
    cw = (inner - gap * (cells - 1)) / cells
    for j in range(cells):
        ax.add_patch(
            Rectangle(
                (x + 0.009 + j * (cw + gap), y + 0.014),
                cw,
                (height - 0.028) * fill_fraction,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor="none",
                alpha=0.88 if j < max(1, round(cells * fill_fraction)) else 0.25,
                zorder=7,
            )
        )
    if label:
        ax.text(x + width / 2, y - 0.014, label, transform=ax.transAxes, ha="center", va="top", fontsize=5.4, color=color, fontweight="bold", zorder=9)


def loop_icon(ax, center, radius, color, label, sublabel):
    cx, cy = center
    ax.add_patch(Arc((cx, cy), 2 * radius, 2 * radius, theta1=35, theta2=335, transform=ax.transAxes, color=color, lw=1.15, zorder=6))
    arrow(ax, (cx + 0.70 * radius, cy + 0.55 * radius), (cx + 0.90 * radius, cy + 0.25 * radius), color=color, lw=0.8, size=5)
    ax.text(cx, cy + 0.003, label, transform=ax.transAxes, ha="center", va="center", fontsize=5.5, color=color, fontweight="bold", zorder=9)
    ax.text(cx, cy - radius - 0.016, sublabel, transform=ax.transAxes, ha="center", va="top", fontsize=5.0, color=MUTED, zorder=9)


def build():
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    fig.subplots_adjust(left=0.012, right=0.988, bottom=0.025, top=0.985)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Quiet asymmetric stage fields.
    ax.add_patch(Polygon([(0.015, 0.95), (0.335, 0.93), (0.335, 0.25), (0.02, 0.27)], transform=ax.transAxes, facecolor=PANEL, edgecolor="none", zorder=0))
    ax.add_patch(Polygon([(0.350, 0.93), (0.655, 0.96), (0.655, 0.25), (0.350, 0.25)], transform=ax.transAxes, facecolor="#F2F7F8", edgecolor="none", zorder=0))
    ax.add_patch(Polygon([(0.675, 0.94), (0.985, 0.91), (0.980, 0.25), (0.675, 0.27)], transform=ax.transAxes, facecolor="#F6F7F8", edgecolor="none", zorder=0))

    # Stage 1: controlled chronology.
    ax.text(0.035, 0.910, "Conserved peaking task and recharge\nunder alternative event timing", transform=ax.transAxes, fontsize=6.5, fontweight="bold", color=INK, linespacing=1.05)
    ax.text(0.035, 0.817, "dispersed", transform=ax.transAxes, fontsize=5.7, fontweight="bold", color=BLUE)
    event_strip(ax, 0.045, 0.737, 0.235, clustered=False)
    ax.text(0.035, 0.610, "clustered", transform=ax.transAxes, fontsize=5.7, fontweight="bold", color=ORANGE)
    event_strip(ax, 0.045, 0.530, 0.235, clustered=True)
    # Recovery-window callouts.
    arrow(ax, (0.093, 0.697), (0.151, 0.697), color=TEAL, lw=0.65, size=4, style="<->")
    ax.text(0.122, 0.708, "recovery window", transform=ax.transAxes, ha="center", va="bottom", fontsize=5.0, color=TEAL)
    arrow(ax, (0.091, 0.490), (0.111, 0.490), color=TEAL, lw=0.65, size=4, style="<->")
    ax.text(0.101, 0.479, "contracted", transform=ax.transAxes, ha="center", va="top", fontsize=5.0, color=TEAL)
    ax.text(0.163, 0.355, "identical 24 h blocks", transform=ax.transAxes, ha="center", fontsize=5.3, color=INK, fontweight="bold")
    ax.text(0.163, 0.326, "annual task + renewable surplus conserved", transform=ax.transAxes, ha="center", fontsize=5.0, color=MUTED)

    arrow(ax, (0.322, 0.605), (0.355, 0.605), color=MUTED, lw=0.9, size=7)

    # Stage 2: symmetric layered portfolio and charging interface.
    ax.text(0.370, 0.910, "Common 4 h LFP layer\nwith technology-specific event tails", transform=ax.transAxes, fontsize=6.5, fontweight="bold", color=INK, linespacing=1.05)
    battery(ax, (0.420, 0.370), 0.165, 0.105, BLUE, cells=6, fill_fraction=0.80, label="shared 4 h LFP layer")
    ax.text(0.503, 0.320, "serves frequent intraday task", transform=ax.transAxes, ha="center", fontsize=5.0, color=MUTED, zorder=9)

    # Three mutually exclusive tail choices.
    tail_x = [0.385, 0.475, 0.565]
    tail_labels = ["no tail", "+ 4 h LFP", "+ LDES"]
    tail_colors = [MUTED, BLUE, PURPLE]
    for x, label, color in zip(tail_x, tail_labels, tail_colors):
        if label == "no tail":
            ax.add_patch(Circle((x, 0.650), 0.025, transform=ax.transAxes, facecolor="white", edgecolor=color, linewidth=0.8, zorder=6))
            ax.plot([x - 0.012, x + 0.012], [0.638, 0.662], transform=ax.transAxes, color=color, lw=1.0, zorder=7)
        elif label == "+ 4 h LFP":
            battery(ax, (x - 0.031, 0.612), 0.062, 0.076, color, cells=3, fill_fraction=0.75)
        else:
            # Catalogue fan: multiple duration tanks without naming one technology.
            for j, (dx, h, c) in enumerate([(-0.028, 0.060, TEAL), (-0.008, 0.082, GOLD), (0.013, 0.105, PURPLE)]):
                box(ax, (x + dx, 0.606), 0.024, h, face=mpl.colors.to_rgba(c, 0.12), edge=c, radius=0.004, lw=0.6, z=5 + j)
        ax.text(x, 0.575, label, transform=ax.transAxes, ha="center", fontsize=5.2, color=color, fontweight="bold")
    ax.text(0.503, 0.721, "choose one tail option", transform=ax.transAxes, ha="center", fontsize=5.5, color=INK, fontweight="bold", zorder=9)

    # Surplus arrives through a finite interface during recovery windows.
    ax.plot([0.382, 0.623], [0.800, 0.800], transform=ax.transAxes, color=TEAL, lw=1.3, zorder=5)
    for x in np.linspace(0.397, 0.607, 6):
        ax.add_patch(Circle((x, 0.800), 0.006, transform=ax.transAxes, facecolor=TEAL, edgecolor="white", linewidth=0.25, zorder=6))
    funnel = Polygon([(0.485, 0.792), (0.520, 0.792), (0.510, 0.755), (0.495, 0.755)], transform=ax.transAxes, closed=True, facecolor=TEAL_LIGHT, edgecolor=TEAL, linewidth=0.7, zorder=7)
    ax.add_patch(funnel)
    arrow(ax, (0.503, 0.755), (0.503, 0.713), color=TEAL, lw=0.9, size=6)
    ax.text(0.503, 0.838, "renewable surplus during recovery windows", transform=ax.transAxes, ha="center", fontsize=5.1, color=TEAL)
    ax.text(0.530, 0.754, "charging\ninterface", transform=ax.transAxes, ha="left", va="center", fontsize=5.0, color=TEAL, linespacing=1.0)

    arrow(ax, (0.645, 0.605), (0.681, 0.605), color=MUTED, lw=0.9, size=7)

    # Stage 3: mechanism and choice, not a generic workflow box.
    ax.text(0.695, 0.910, "Event timing redistributes value\nacross storage duration", transform=ax.transAxes, fontsize=6.5, fontweight="bold", color=INK, linespacing=1.05)
    loop_icon(ax, (0.760, 0.770), 0.041, BLUE, "cycling", "repeated energy service")
    loop_icon(ax, (0.895, 0.770), 0.041, PURPLE, "coverage", "firm supply across clusters")

    # Signed cross-duration response.
    x = np.linspace(0.705, 0.962, 260)
    u = (x - x.min()) / (x.max() - x.min())
    y = 0.600 - 0.115 * np.exp(-((u - 0.38) / 0.22) ** 2) + 0.105 / (1 + np.exp(-(u - 0.70) * 20)) - 0.035
    zero = 0.568
    ax.plot([0.705, 0.962], [zero, zero], transform=ax.transAxes, color="#89969C", lw=0.65, zorder=3)
    ax.fill_between(x, y, zero, where=y < zero, transform=ax.transAxes, color=ORANGE_LIGHT, alpha=0.95, linewidth=0, zorder=2)
    ax.fill_between(x, y, zero, where=y >= zero, transform=ax.transAxes, color=PURPLE_LIGHT, alpha=0.75, linewidth=0, zorder=2)
    ax.plot(x, y, transform=ax.transAxes, color=INK, lw=1.35, zorder=6)
    ax.text(0.781, 0.507, "cycling loss dominates", transform=ax.transAxes, ha="center", fontsize=5.0, color=ORANGE, fontweight="bold")
    ax.text(0.916, 0.650, "coverage gains weight", transform=ax.transAxes, ha="center", fontsize=5.0, color=PURPLE, fontweight="bold")
    arrow(ax, (0.707, 0.458), (0.963, 0.458), color=INK, lw=0.75, size=6)
    ax.text(0.835, 0.432, "storage duration", transform=ax.transAxes, ha="center", fontsize=5.1, color=INK, fontweight="bold")

    # Decision readout.
    ax.text(0.695, 0.360, "Near an economic margin, the selected portfolio can change", transform=ax.transAxes, fontsize=5.5, color=INK, fontweight="bold")
    decision_x = [0.735, 0.835, 0.935]
    decision = [("technology", "LFP  ↔  LDES"), ("duration", "24 h  ↔  96 h"), ("tail layer", "enter  ↔  exit")]
    for x0, (head, body) in zip(decision_x, decision):
        box(ax, (x0 - 0.043, 0.270), 0.086, 0.060, face="white", edge=GRID, radius=0.008, lw=0.65, z=5)
        ax.text(x0, 0.315, head, transform=ax.transAxes, ha="center", va="center", fontsize=5.0, color=MUTED, zorder=9)
        ax.text(x0, 0.286, body, transform=ax.transAxes, ha="center", va="center", fontsize=5.0, color=INK, fontweight="bold", zorder=9)

    # Bottom evidence and method chain. Three empirical scales converge on
    # the same chronology intervention and portfolio model.
    box(ax, (0.030, 0.045), 0.940, 0.145, face="white", edge=GRID, radius=0.014, lw=0.75, z=2)
    inputs = [
        (0.085, "Europe", "15 countries\n132 country-years", BLUE),
        (0.205, "China", "26 provinces\n26 province-years", ORANGE),
        (0.325, "Japan", "5 prefectures\n27 community-years", TEAL),
    ]
    for x0, head, sub, color in inputs:
        ax.add_patch(Circle((x0, 0.128), 0.018, transform=ax.transAxes, facecolor=mpl.colors.to_rgba(color, 0.14), edgecolor=color, lw=0.7, zorder=7))
        ax.text(x0, 0.097, head, transform=ax.transAxes, ha="center", fontsize=5.1, color=INK, fontweight="bold")
        ax.text(x0, 0.065, sub, transform=ax.transAxes, ha="center", va="center", fontsize=5.0, color=MUTED, linespacing=1.15)
        arrow(ax, (x0 + 0.026, 0.128), (0.445, 0.128), color="#AAB8BE", lw=0.55, size=4)
    nodes = [
        (0.475, "Conserved chronology", "same blocks and annual energy", ORANGE),
        (0.690, "Joint portfolio dispatch", "51 choices · cyclic SOC", TEAL),
        (0.895, "Value and cost choice", "price · coverage · technology", PURPLE),
    ]
    for idx, (x0, head, sub, color) in enumerate(nodes):
        ax.add_patch(Circle((x0, 0.128), 0.018, transform=ax.transAxes, facecolor=mpl.colors.to_rgba(color, 0.14), edgecolor=color, lw=0.7, zorder=7))
        ax.text(x0, 0.093, head, transform=ax.transAxes, ha="center", fontsize=5.0, color=INK, fontweight="bold")
        ax.text(x0, 0.066, sub, transform=ax.transAxes, ha="center", fontsize=5.0, color=MUTED)
        if idx < len(nodes) - 1:
            arrow(ax, (x0 + 0.035, 0.128), (nodes[idx + 1][0] - 0.035, 0.128), color="#AAB8BE", lw=0.7, size=5)

    source = pd.DataFrame(
        [
            ("Europe", "15 countries", "132 country-years", "1,156,320 country-hours"),
            ("China", "26 provinces", "26 province-years", "227,760 province-hours"),
            ("Japan", "5 prefectures", "27 community-years", "236,520 community-hours"),
            ("chronology", "dispersed versus clustered", "24 h blocks conserved", "annual task and surplus conserved"),
            ("portfolio", "common 4 h LFP layer", "no tail or additional 4 h LFP", "49 candidate LDES configurations"),
            ("mechanism", "recovery window", "charging interface", "cycling and coverage returns"),
            ("decision", "technology", "duration and power", "tail-layer participation"),
        ],
        columns=["stage", "element_1", "element_2", "element_3"],
    )
    source.to_csv(SRC / "figure1_v48_framework_contract.csv", index=False)

    stem = OUT / "fig_v48_figure1_framework"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)

    # Grayscale proof for redundant-encoding QA.
    rgba = plt.imread(stem.with_suffix(".png"))
    rgb = rgba[..., :3]
    gray = np.dot(rgb, [0.2126, 0.7152, 0.0722])
    plt.imsave(OUT / "fig_v48_figure1_framework_grayscale.png", gray, cmap="gray", vmin=0, vmax=1)
    plt.close(fig)


if __name__ == "__main__":
    build()
