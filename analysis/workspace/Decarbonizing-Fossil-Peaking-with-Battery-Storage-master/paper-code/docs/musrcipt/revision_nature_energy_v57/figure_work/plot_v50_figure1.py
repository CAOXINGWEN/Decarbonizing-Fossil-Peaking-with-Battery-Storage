#!/usr/bin/env python3
"""Build the v54 sequencing-parity conceptual framework figure."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "main_figures"
SRC = ROOT / "source_data_v50"
OUT.mkdir(parents=True, exist_ok=True)
SRC.mkdir(parents=True, exist_ok=True)

SPEC = importlib.util.spec_from_file_location("v49_figure1", ROOT / "plot_v49_figure1.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Cannot import v49 figure helpers")
old = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(old)

INK = old.INK
MUTED = old.MUTED
GRID = old.GRID
PANEL = old.PANEL
BLUE = old.BLUE
TEAL = old.TEAL
ORANGE = old.ORANGE
PURPLE = old.PURPLE
GOLD = old.GOLD
BLUE_LIGHT = old.BLUE_LIGHT
TEAL_LIGHT = old.TEAL_LIGHT
ORANGE_LIGHT = old.ORANGE_LIGHT
PURPLE_LIGHT = old.PURPLE_LIGHT

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


def box(ax, xy, width, height, face="white", edge=GRID, radius=0.010, lw=0.7, z=2):
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


def arrow(ax, start, end, color=MUTED, lw=0.8, size=6):
    return old.arrow(ax, start, end, color=color, lw=lw, size=size)


def regime_icon(ax, x, y, color, title, subtitle, source):
    ax.add_patch(
        Circle(
            (x, y),
            0.036,
            transform=ax.transAxes,
            facecolor=mpl.colors.to_rgba(color, 0.13),
            edgecolor=color,
            linewidth=0.8,
            zorder=5,
        )
    )
    for j, h in enumerate((0.017, 0.031, 0.023)):
        ax.add_patch(
            Rectangle(
                (x - 0.022 + j * 0.015, y - 0.005),
                0.009,
                h,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor="none",
                zorder=7,
            )
        )
    wave_x = np.linspace(x - 0.025, x + 0.025, 50)
    wave_y = y - 0.016 + 0.004 * np.sin(np.linspace(0, 2 * np.pi, 50))
    ax.plot(wave_x, wave_y, transform=ax.transAxes, color=TEAL, lw=0.65, zorder=7)
    ax.text(x + 0.050, y + 0.023, title, transform=ax.transAxes, ha="left", va="center", fontsize=5.7, color=color, fontweight="bold")
    ax.text(x + 0.050, y - 0.006, subtitle, transform=ax.transAxes, ha="left", va="center", fontsize=5.0, color=INK)
    ax.text(x + 0.050, y - 0.035, source, transform=ax.transAxes, ha="left", va="center", fontsize=5.0, color=MUTED)


def event_sequence(ax, x0, y0, width, clustered):
    positions = np.linspace(x0, x0 + width, 7)
    if clustered:
        positions = np.array([positions[0], positions[1], positions[2], positions[4], positions[5], positions[6]])
    else:
        positions = np.linspace(x0, x0 + width, 6)
    heights = (0.032, 0.048, 0.038, 0.052, 0.035, 0.045)
    for x, h in zip(positions, heights):
        ax.add_patch(Rectangle((x - 0.006, y0), 0.012, h, transform=ax.transAxes, facecolor=ORANGE, edgecolor="white", linewidth=0.25, zorder=6))
    wave_x = np.linspace(x0 - 0.006, x0 + width + 0.006, 180)
    wave_y = y0 - 0.026 + 0.007 * np.sin(np.linspace(0, 8 * np.pi, 180))
    ax.plot(wave_x, wave_y, transform=ax.transAxes, color=TEAL, lw=1.05, zorder=5)


def build():
    sensitivity = pd.read_csv(ROOT.parent / "sequencing_parity" / "cost_duration_discount_sensitivity.csv")
    central_cost_movement = float(sensitivity[
        np.isclose(sensitivity["discount_rate"], 0.036)
        & np.isclose(sensitivity["caes_duration_h"], 24.0)
    ]["absolute_relative_cost_movement_eur_kw_year"].iloc[0])
    parity_summary = pd.read_csv(
        ROOT / "source_data" / "v54_parity_exposure" / "figure6b_24h_cross_regime_parity_summary.csv"
    ).set_index("regime")
    europe_above_parity = int(parity_summary.loc["Europe", "above_parity_count"])

    fig, ax = plt.subplots(figsize=(7.2, 4.15))
    fig.subplots_adjust(left=0.012, right=0.988, bottom=0.025, top=0.985)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.add_patch(Polygon([(0.015, 0.96), (0.225, 0.93), (0.215, 0.205), (0.02, 0.225)], transform=ax.transAxes, facecolor="#F6F8F8", edgecolor="none"))
    ax.add_patch(Polygon([(0.235, 0.94), (0.640, 0.96), (0.635, 0.205), (0.225, 0.205)], transform=ax.transAxes, facecolor="#F1F7F7", edgecolor="none"))
    ax.add_patch(Polygon([(0.650, 0.96), (0.985, 0.93), (0.980, 0.205), (0.645, 0.220)], transform=ax.transAxes, facecolor="#F7F6F9", edgecolor="none"))

    # Three empirical balancing regimes.
    ax.text(0.035, 0.910, "Three balancing regimes", transform=ax.transAxes, fontsize=6.7, color=INK, fontweight="bold")
    regime_icon(ax, 0.072, 0.762, BLUE, "Europe", "national energy balance", "15 countries · 132 country–years")
    regime_icon(ax, 0.072, 0.560, ORANGE, "China", "provincial gas peaking", "26 provinces")
    regime_icon(ax, 0.072, 0.358, TEAL, "Japan", "community PV balance", "27 community–years")
    ax.text(0.035, 0.270, "Different tasks and recharge sources", transform=ax.transAxes, fontsize=5.1, color=INK, fontweight="bold")
    ax.text(0.035, 0.240, "the same conserved sequencing operation", transform=ax.transAxes, fontsize=5.0, color=MUTED)
    arrow(ax, (0.210, 0.565), (0.242, 0.565), color="#9AAAB0", lw=0.8, size=6)

    # Conserved sequencing and recovery mechanism.
    ax.text(0.255, 0.910, "Conserved event sequencing", transform=ax.transAxes, fontsize=6.7, color=INK, fontweight="bold")
    ax.text(0.258, 0.817, "dispersed", transform=ax.transAxes, fontsize=5.7, color=BLUE, fontweight="bold")
    event_sequence(ax, 0.315, 0.758, 0.265, clustered=False)
    old.arrow(ax, (0.360, 0.717), (0.455, 0.717), color=TEAL, lw=0.7, size=4, style="<->")
    ax.text(0.408, 0.729, "recovery window", transform=ax.transAxes, ha="center", va="bottom", fontsize=5.0, color=TEAL)
    ax.text(0.258, 0.641, "clustered", transform=ax.transAxes, fontsize=5.7, color=ORANGE, fontweight="bold")
    event_sequence(ax, 0.315, 0.582, 0.265, clustered=True)
    old.arrow(ax, (0.359, 0.541), (0.385, 0.541), color=TEAL, lw=0.7, size=4, style="<->")
    ax.text(0.372, 0.529, "contracted", transform=ax.transAxes, ha="center", va="top", fontsize=5.0, color=TEAL)
    box(ax, (0.276, 0.420), 0.320, 0.074, face="white", edge=GRID, radius=0.009, lw=0.65)
    ax.text(0.436, 0.471, "same 24 h blocks · same annual task and recharge", transform=ax.transAxes, ha="center", va="center", fontsize=5.2, color=INK, fontweight="bold")
    ax.text(0.436, 0.443, "median recovery 72 h → 0 h · zero-window share 0 → 74.7%", transform=ax.transAxes, ha="center", va="center", fontsize=5.0, color=MUTED)

    # Layered portfolio and finite interface.
    old.battery(ax, (0.274, 0.265), 0.145, 0.090, BLUE, cells=5, fill_fraction=0.8, label="common 4 h LFP")
    ax.plot([0.448, 0.595], [0.362, 0.362], transform=ax.transAxes, color=TEAL, lw=1.2)
    for x in np.linspace(0.458, 0.585, 6):
        ax.add_patch(Circle((x, 0.362), 0.005, transform=ax.transAxes, facecolor=TEAL, edgecolor="white", linewidth=0.2, zorder=7))
    funnel = Polygon([(0.503, 0.354), (0.538, 0.354), (0.529, 0.322), (0.512, 0.322)], transform=ax.transAxes, closed=True, facecolor=TEAL_LIGHT, edgecolor=TEAL, linewidth=0.7, zorder=7)
    ax.add_patch(funnel)
    ax.text(0.520, 0.390, "charging interface", transform=ax.transAxes, ha="center", fontsize=5.0, color=TEAL, fontweight="bold")
    for x, h, color in [(0.470, 0.044, BLUE), (0.516, 0.066, GOLD), (0.562, 0.090, PURPLE)]:
        box(ax, (x - 0.014, 0.247), 0.028, h, face=mpl.colors.to_rgba(color, 0.13), edge=color, radius=0.004, lw=0.65, z=6)
    ax.text(0.520, 0.224, "49 LDES technology–duration tails", transform=ax.transAxes, ha="center", fontsize=5.0, color=PURPLE, fontweight="bold")
    arrow(ax, (0.630, 0.565), (0.660, 0.565), color="#9AAAB0", lw=0.8, size=6)

    # Sequencing parity as the hero outcome.
    ax.text(0.675, 0.910, "Sequencing parity", transform=ax.transAxes, fontsize=6.7, color=INK, fontweight="bold")
    ax.text(0.675, 0.875, "recovery-window value weighed against cost learning", transform=ax.transAxes, fontsize=5.0, color=MUTED)
    x = np.linspace(0.682, 0.960, 260)
    u = (x - x.min()) / (x.max() - x.min())
    y = 0.737 - 0.105 * np.exp(-((u - 0.42) / 0.23) ** 2) + 0.090 / (1 + np.exp(-(u - 0.73) * 22)) - 0.035
    zero = 0.704
    ax.plot([0.682, 0.960], [zero, zero], transform=ax.transAxes, color="#8B989E", lw=0.65)
    ax.fill_between(x, y, zero, where=y < zero, transform=ax.transAxes, color=ORANGE_LIGHT, alpha=0.95, linewidth=0)
    ax.fill_between(x, y, zero, where=y >= zero, transform=ax.transAxes, color=PURPLE_LIGHT, alpha=0.80, linewidth=0)
    ax.plot(x, y, transform=ax.transAxes, color=INK, lw=1.35)
    ax.text(0.785, 0.650, "intermediate-duration loss", transform=ax.transAxes, ha="center", fontsize=5.0, color=ORANGE, fontweight="bold")
    ax.text(0.915, 0.782, "long-duration recovery", transform=ax.transAxes, ha="center", fontsize=5.0, color=PURPLE, fontweight="bold")
    arrow(ax, (0.684, 0.600), (0.962, 0.600), color=INK, lw=0.7, size=5)
    ax.text(0.823, 0.578, "storage duration", transform=ax.transAxes, ha="center", fontsize=5.0, color=INK)

    box(ax, (0.677, 0.335), 0.294, 0.190, face="white", edge=GRID, radius=0.012, lw=0.75)
    maxv = 6.0
    left = 0.720
    width = 0.205
    for y0, value, color, title, sub in [
        (0.455, 5.48, ORANGE, "event sequencing", "median absolute value shift"),
        (0.390, central_cost_movement, PURPLE, "2025–2035 cost path", "24 h A-CAES vs 4 h LFP"),
    ]:
        ax.add_patch(Rectangle((left, y0), width, 0.020, transform=ax.transAxes, facecolor="#E8ECEE", edgecolor="none", zorder=4))
        ax.add_patch(Rectangle((left, y0), width * value / maxv, 0.020, transform=ax.transAxes, facecolor=color, edgecolor="none", zorder=5))
        ax.text(left, y0 + 0.031, title, transform=ax.transAxes, fontsize=5.0, color=INK, fontweight="bold")
        ax.text(left + width + 0.010, y0 + 0.010, f"{value:.2f}", transform=ax.transAxes, ha="left", va="center", fontsize=5.4, color=color, fontweight="bold")
        ax.text(left, y0 - 0.012, sub, transform=ax.transAxes, fontsize=5.0, color=MUTED)
    ax.text(0.824, 0.359, f"{europe_above_parity}/132 country–years", transform=ax.transAxes, ha="center", fontsize=5.0, color=INK, fontweight="bold")
    ax.text(0.824, 0.340, "sequencing shift ≥ matched decade cost movement", transform=ax.transAxes, ha="center", fontsize=5.0, color=INK)

    # Compact evidence chain.
    box(ax, (0.030, 0.035), 0.940, 0.125, face="white", edge=GRID, radius=0.014, lw=0.75)
    nodes = [
        (0.105, BLUE, "observed peaking\ntask + recharge"),
        (0.300, ORANGE, "conserved\nday-block sequencing"),
        (0.500, TEAL, "cyclic-SOC\nportfolio dispatch"),
        (0.700, PURPLE, "cycling + coverage\nvalue account"),
        (0.895, INK, "cost trajectory +\nconfiguration choice"),
    ]
    for idx, (x0, color, label) in enumerate(nodes):
        ax.add_patch(Circle((x0, 0.108), 0.018, transform=ax.transAxes, facecolor=mpl.colors.to_rgba(color, 0.12), edgecolor=color, linewidth=0.7, zorder=7))
        ax.text(x0, 0.070, label, transform=ax.transAxes, ha="center", va="center", fontsize=5.0, color=INK, linespacing=1.08)
        if idx < len(nodes) - 1:
            arrow(ax, (x0 + 0.032, 0.108), (nodes[idx + 1][0] - 0.032, 0.108), color="#A8B6BB", lw=0.65, size=5)

    pd.DataFrame(
        [
            ("Europe", "national energy balance", 132, "synchronous wind-solar balance"),
            ("China", "provincial gas peaking", 26, "wind-solar curtailment"),
            ("Japan", "community grid import", 27, "distributed-PV reverse flow"),
        ],
        columns=["regime", "peaking_task", "case_years", "recharge_source"],
    ).to_csv(SRC / "figure1_three_regime_contract.csv", index=False)
    pd.DataFrame(
        [
            ("sequencing_value_shift", 5.48, "EUR kW-1 yr-1", "median absolute shift"),
            ("decade_relative_cost_movement", central_cost_movement, "EUR kW-1 yr-1", "24 h A-CAES versus 4 h LFP"),
            ("sequencing_ge_cost_cases", europe_above_parity, "of 132", "matched selected configuration"),
        ],
        columns=["metric", "value", "unit", "definition"],
    ).to_csv(SRC / "figure1_sequencing_parity_metrics.csv", index=False)

    stem = OUT / "fig_v54_figure1_sequencing_parity_framework"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    rgba = plt.imread(stem.with_suffix(".png"))
    gray = np.dot(rgba[..., :3], [0.2126, 0.7152, 0.0722])
    plt.imsave(OUT / "fig_v54_figure1_sequencing_parity_framework_grayscale.png", gray, cmap="gray", vmin=0, vmax=1)
    plt.close(fig)


if __name__ == "__main__":
    build()
