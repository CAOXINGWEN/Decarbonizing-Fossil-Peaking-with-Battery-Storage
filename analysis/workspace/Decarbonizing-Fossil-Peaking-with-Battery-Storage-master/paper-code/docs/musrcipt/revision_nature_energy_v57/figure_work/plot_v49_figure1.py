#!/usr/bin/env python3
"""Build the v49 reference-informed conceptual framework figure."""

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


def build_v46_legacy():
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


def build():
    """Asymmetric hypothesis schematic following a map--mechanism--outcome hierarchy."""
    fig, ax = plt.subplots(figsize=(7.2, 4.35))
    fig.subplots_adjust(left=0.012, right=0.988, bottom=0.025, top=0.985)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # Three quiet fields create hierarchy without turning the figure into a dashboard.
    ax.add_patch(Polygon([(0.018, 0.96), (0.215, 0.92), (0.205, 0.22), (0.025, 0.25)],
                         transform=ax.transAxes, facecolor="#F6F8F8", edgecolor="none", zorder=0))
    ax.add_patch(Polygon([(0.225, 0.93), (0.650, 0.96), (0.640, 0.22), (0.215, 0.22)],
                         transform=ax.transAxes, facecolor="#F2F7F8", edgecolor="none", zorder=0))
    ax.add_patch(Polygon([(0.655, 0.95), (0.985, 0.92), (0.980, 0.22), (0.645, 0.24)],
                         transform=ax.transAxes, facecolor="#F7F7F9", edgecolor="none", zorder=0))

    # Empirical scales: three equally weighted applications, not a primary/validation hierarchy.
    ax.text(0.040, 0.905, "Three empirical scales", transform=ax.transAxes,
            fontsize=6.6, color=INK, fontweight="bold")
    scales = [
        (0.105, 0.770, BLUE, "Europe", "15 countries\n132 country–years", "national\nbalance"),
        (0.105, 0.565, ORANGE, "China", "26 provinces", "gas peaking"),
        (0.105, 0.360, TEAL, "Japan", "27 community–years", "PV community"),
    ]
    for cx, cy, colour, name, sample, task in scales:
        ax.add_patch(Circle((cx, cy), 0.046, transform=ax.transAxes,
                            facecolor=mpl.colors.to_rgba(colour, 0.12), edgecolor=colour, lw=0.8, zorder=5))
        # Simple system glyph: demand bars and a recharge wave.
        for j, h in enumerate([0.020, 0.035, 0.025]):
            ax.add_patch(Rectangle((cx-0.027+j*0.018, cy-0.006), 0.010, h,
                                   transform=ax.transAxes, facecolor=colour, edgecolor="none", zorder=7))
        xx = np.linspace(cx-0.030, cx+0.030, 50)
        yy = cy-0.020 + 0.006*np.sin(np.linspace(0, 2*np.pi, len(xx)))
        ax.plot(xx, yy, transform=ax.transAxes, color=TEAL, lw=0.65, zorder=7)
        ax.text(cx, cy-0.068, name, transform=ax.transAxes, ha="center", fontsize=5.6,
                color=colour, fontweight="bold")
        ax.text(cx, cy-0.104, sample, transform=ax.transAxes, ha="center", va="top",
                fontsize=5.0, color=MUTED, linespacing=1.10)
        ax.text(cx+0.061, cy+0.006, task, transform=ax.transAxes, ha="left", va="center",
                fontsize=5.0, color=INK, linespacing=1.05)
        arrow(ax, (0.183, cy), (0.232, 0.625), color="#A6B4B9", lw=0.55, size=4)

    # The conserved chronology intervention is the hero mechanism.
    ax.text(0.250, 0.905, "Conserved task and recharge · alternative event timing",
            transform=ax.transAxes, fontsize=6.6, color=INK, fontweight="bold")
    ax.text(0.255, 0.806, "dispersed", transform=ax.transAxes, fontsize=5.7,
            color=BLUE, fontweight="bold")
    event_strip(ax, 0.295, 0.744, 0.285, clustered=False)
    arrow(ax, (0.347, 0.706), (0.438, 0.706), color=TEAL, lw=0.7, size=4, style="<->")
    ax.text(0.392, 0.718, "long recovery window", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=5.0, color=TEAL)

    ax.text(0.255, 0.623, "clustered", transform=ax.transAxes, fontsize=5.7,
            color=ORANGE, fontweight="bold")
    event_strip(ax, 0.295, 0.560, 0.285, clustered=True)
    arrow(ax, (0.346, 0.522), (0.376, 0.522), color=TEAL, lw=0.7, size=4, style="<->")
    ax.text(0.361, 0.510, "contracted", transform=ax.transAxes,
            ha="center", va="top", fontsize=5.0, color=TEAL)
    ax.text(0.450, 0.455, "same 24 h blocks · same annual energy",
            transform=ax.transAxes, ha="center", fontsize=5.0, color=MUTED)

    # Shared short-duration layer and technology-specific tail catalogue.
    battery(ax, (0.275, 0.280), 0.170, 0.105, BLUE, cells=6, fill_fraction=0.82,
            label="common 4 h LFP layer")
    ax.text(0.360, 0.231, "frequent intraday task", transform=ax.transAxes,
            ha="center", fontsize=5.0, color=MUTED)
    ax.plot([0.478, 0.595], [0.378, 0.378], transform=ax.transAxes, color=TEAL, lw=1.2)
    for x0 in np.linspace(0.485, 0.590, 5):
        ax.add_patch(Circle((x0, 0.378), 0.005, transform=ax.transAxes,
                            facecolor=TEAL, edgecolor="white", lw=0.2, zorder=6))
    funnel = Polygon([(0.518, 0.372), (0.554, 0.372), (0.545, 0.335), (0.527, 0.335)],
                     transform=ax.transAxes, closed=True, facecolor=TEAL_LIGHT,
                     edgecolor=TEAL, lw=0.7, zorder=7)
    ax.add_patch(funnel)
    ax.text(0.536, 0.405, "recovery window × charging interface",
            transform=ax.transAxes, ha="center", fontsize=5.0, color=TEAL)
    tail_x = [0.485, 0.535, 0.585]
    tail_h = [0.052, 0.085, 0.122]
    for j, (x0, hh, colour) in enumerate(zip(tail_x, tail_h, [BLUE, GOLD, PURPLE])):
        box(ax, (x0-0.016, 0.248), 0.032, hh,
            face=mpl.colors.to_rgba(colour, 0.13), edge=colour, radius=0.004, lw=0.65, z=6+j)
    ax.text(0.535, 0.209, "49 technology–duration tails",
            transform=ax.transAxes, ha="center", fontsize=5.1, color=PURPLE, fontweight="bold")

    arrow(ax, (0.632, 0.575), (0.674, 0.575), color=MUTED, lw=0.9, size=7)

    # Cross-duration value redistribution and the mid-duration valley hypothesis.
    ax.text(0.690, 0.905, "Value redistribution across storage duration",
            transform=ax.transAxes, fontsize=6.6, color=INK, fontweight="bold")
    loop_icon(ax, (0.760, 0.795), 0.038, BLUE, "cycling", "repeat service")
    loop_icon(ax, (0.900, 0.795), 0.038, PURPLE, "coverage", "cluster supply")

    x = np.linspace(0.690, 0.963, 280)
    u = (x-x.min())/(x.max()-x.min())
    y = 0.610 - 0.135*np.exp(-((u-0.42)/0.24)**2) + 0.105/(1+np.exp(-(u-0.73)*22)) - 0.046
    zero = 0.575
    ax.plot([0.690, 0.963], [zero, zero], transform=ax.transAxes, color="#8A979D", lw=0.65)
    ax.fill_between(x, y, zero, where=y<zero, transform=ax.transAxes,
                    color=ORANGE_LIGHT, alpha=0.95, linewidth=0)
    ax.fill_between(x, y, zero, where=y>=zero, transform=ax.transAxes,
                    color=PURPLE_LIGHT, alpha=0.80, linewidth=0)
    ax.plot(x, y, transform=ax.transAxes, color=INK, lw=1.35)
    # Boundary markers reveal the positive–negative–positive topology.
    for xx, colour in [(0.735, BLUE), (0.865, PURPLE)]:
        ax.plot([xx, xx], [0.485, 0.660], transform=ax.transAxes,
                color=colour, lw=0.75, ls=(0, (2, 2)))
    ax.text(0.800, 0.515, "mid-duration valley", transform=ax.transAxes,
            ha="center", fontsize=5.3, color=ORANGE, fontweight="bold")
    ax.text(0.735, 0.666, "cycling entry", transform=ax.transAxes,
            ha="center", fontsize=5.0, color=BLUE)
    ax.text(0.865, 0.666, "coverage re-entry", transform=ax.transAxes,
            ha="center", fontsize=5.0, color=PURPLE)
    arrow(ax, (0.692, 0.465), (0.965, 0.465), color=INK, lw=0.75, size=6)
    ax.text(0.828, 0.438, "storage duration", transform=ax.transAxes,
            ha="center", fontsize=5.1, color=INK, fontweight="bold")

    # Choice readout stays compact and subordinate to the mechanism.
    ax.text(0.690, 0.370, "Economic consequence", transform=ax.transAxes,
            fontsize=5.7, color=INK, fontweight="bold")
    decisions = [("technology", "LFP ↔ LDES"), ("duration", "24 h ↔ 96 h"), ("tail layer", "enter ↔ exit")]
    for x0, (head, body) in zip([0.735, 0.830, 0.925], decisions):
        box(ax, (x0-0.041, 0.272), 0.082, 0.065, face="white", edge=GRID,
            radius=0.008, lw=0.65, z=5)
        ax.text(x0, 0.317, head, transform=ax.transAxes, ha="center",
                fontsize=5.0, color=MUTED, zorder=9)
        ax.text(x0, 0.288, body, transform=ax.transAxes, ha="center",
                fontsize=5.0, color=INK, fontweight="bold", zorder=9)

    # Bottom method chain is deliberately icon-led and concise.
    box(ax, (0.030, 0.035), 0.940, 0.125, face="white", edge=GRID, radius=0.014, lw=0.75)
    nodes = [
        (0.105, BLUE, "empirical task\n+ recharge"),
        (0.305, ORANGE, "conserved\nchronology"),
        (0.505, TEAL, "joint cyclic-SOC\ndispatch"),
        (0.705, PURPLE, "cycling + coverage\naccount"),
        (0.895, INK, "technology · duration\nparticipation choice"),
    ]
    for idx, (x0, colour, label) in enumerate(nodes):
        ax.add_patch(Circle((x0, 0.108), 0.018, transform=ax.transAxes,
                            facecolor=mpl.colors.to_rgba(colour, 0.12), edgecolor=colour, lw=0.7, zorder=7))
        ax.text(x0, 0.070, label, transform=ax.transAxes, ha="center", va="center",
                fontsize=5.0, color=INK, linespacing=1.10)
        if idx < len(nodes)-1:
            arrow(ax, (x0+0.032, 0.108), (nodes[idx+1][0]-0.032, 0.108),
                  color="#A8B6BB", lw=0.65, size=5)

    source = pd.DataFrame(
        [
            ("applications", "Europe national", "China provincial", "Japan community"),
            ("intervention", "dispersed chronology", "clustered chronology", "annual energy conserved"),
            ("portfolio", "common 4 h LFP", "additional 4 h LFP", "49 LDES tails"),
            ("mechanism", "recovery window", "charging interface", "cycling and coverage returns"),
            ("outcome", "mid-duration valley", "technology and duration", "tail-layer participation"),
        ],
        columns=["stage", "element_1", "element_2", "element_3"],
    )
    source.to_csv(SRC / "figure1_v49_framework_contract.csv", index=False)

    stem = OUT / "fig_v49_figure1_framework"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    rgba = plt.imread(stem.with_suffix(".png"))
    gray = np.dot(rgba[..., :3], [0.2126, 0.7152, 0.0722])
    plt.imsave(OUT / "fig_v49_figure1_framework_grayscale.png", gray,
               cmap="gray", vmin=0, vmax=1)
    plt.close(fig)


if __name__ == "__main__":
    build()
