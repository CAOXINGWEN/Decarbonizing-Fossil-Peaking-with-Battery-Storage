#!/usr/bin/env python3
"""Icon-led standalone Figure 1 preview for the sequencing-parity framework."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Polygon, Rectangle


HERE = Path(__file__).resolve().parent
FIGURE_WORK = HERE.parent
sys.path.insert(0, str(FIGURE_WORK))

from v60_figure_style import (  # noqa: E402
    BLUE,
    CORAL,
    GOLD,
    GRID,
    INK,
    MM,
    MUTED,
    PURPLE,
    REGIME_COLORS,
    TEAL,
    WHITE,
    configure,
)


OUT = HERE / "outputs"
SOURCE = HERE / "source_data" / "figure1_framework_values.csv"
STEM = "fig_v62_preview_figure1_research_framework_icons"
WIDTH_MM = 183.0
HEIGHT_MM = 150.0

TASK = CORAL
RECHARGE = BLUE
RECOVERY = TEAL
NATURAL = "#2F8B7C"
CONTROLLED = "#D8775E"
COST = PURPLE
CHARCOAL = "#3E4D53"
SOFT = "#C8D1CE"
NATURAL_PALE = "#E8F2EF"
CONTROLLED_PALE = "#F8EDE9"
PURPLE_PALE = "#EFECF7"


def load_values() -> dict[str, float]:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        return {row["key"]: float(row["value"]) for row in csv.DictReader(handle)}


VALUES = load_values()


def setup() -> None:
    configure()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.1,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
        }
    )


def text(
    ax: plt.Axes,
    x: float,
    y: float,
    value: str,
    *,
    size: float = 5.5,
    colour: str = INK,
    weight: str = "normal",
    ha: str = "center",
    va: str = "center",
    rotation: float = 0,
    zorder: int = 9,
) -> None:
    ax.text(
        x,
        y,
        value,
        transform=ax.transAxes,
        fontsize=size,
        color=colour,
        fontweight=weight,
        ha=ha,
        va=va,
        rotation=rotation,
        zorder=zorder,
    )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    colour: str = MUTED,
    lw: float = 0.85,
    mutation: float = 6.0,
    connection: str = "arc3,rad=0",
    zorder: int = 5,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle="-|>",
            connectionstyle=connection,
            mutation_scale=mutation,
            lw=lw,
            color=colour,
            shrinkA=0,
            shrinkB=0,
            zorder=zorder,
        )
    )


def header(ax: plt.Axes, label: str, title: str, x: float) -> None:
    text(ax, x, 0.966, label, size=8.0, weight="bold", ha="left", va="top")
    text(ax, x + 0.021, 0.965, title, size=7.1, weight="bold", ha="left", va="top")


def bracket(ax: plt.Axes, x0: float, x1: float, y: float, label: str, colour: str, label_y: float) -> None:
    ax.plot([x0, x1], [y, y], color=colour, lw=0.85, transform=ax.transAxes, zorder=4)
    ax.plot([x0, x0], [y - 0.009, y + 0.009], color=colour, lw=0.85, transform=ax.transAxes, zorder=4)
    ax.plot([x1, x1], [y - 0.009, y + 0.009], color=colour, lw=0.85, transform=ax.transAxes, zorder=4)
    text(ax, (x0 + x1) / 2, label_y, label, size=5.2, colour=colour, weight="bold")


def draw_calendar(ax: plt.Axes, x: float, y: float, w: float, h: float, colour: str) -> None:
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, transform=ax.transAxes, facecolor=WHITE, edgecolor=colour, lw=0.75, zorder=3))
    ax.add_patch(Rectangle((x - w / 2, y + h * 0.17), w, h * 0.33, transform=ax.transAxes, facecolor=mpl.colors.to_rgba(colour, 0.20), edgecolor="none", zorder=3))
    for dx in (-w * 0.24, w * 0.24):
        ax.plot([x + dx, x + dx], [y + h * 0.40, y + h * 0.57], color=colour, lw=1.2, transform=ax.transAxes, zorder=4)
    for row in range(2):
        for col in range(3):
            ax.add_patch(Circle((x - w * 0.24 + col * w * 0.24, y - h * 0.18 + row * h * 0.23), 0.0028, transform=ax.transAxes, facecolor=colour, edgecolor="none", zorder=4))


def draw_globe(ax: plt.Axes, x: float, y: float, r: float, colour: str) -> None:
    ax.add_patch(Circle((x, y), r, transform=ax.transAxes, facecolor=mpl.colors.to_rgba(colour, 0.08), edgecolor=colour, lw=0.75, zorder=3))
    ax.add_patch(Arc((x, y), 1.0 * r, 2 * r, theta1=-90, theta2=90, transform=ax.transAxes, color=colour, lw=0.55, zorder=4))
    ax.add_patch(Arc((x, y), 1.0 * r, 2 * r, theta1=90, theta2=270, transform=ax.transAxes, color=colour, lw=0.55, zorder=4))
    ax.add_patch(Arc((x, y), 2 * r, 0.82 * r, theta1=0, theta2=360, transform=ax.transAxes, color=colour, lw=0.55, zorder=4))


def draw_eye(ax: plt.Axes, x: float, y: float, w: float, h: float, colour: str) -> None:
    verts = [(x - w / 2, y), (x - w / 4, y + h / 2), (x, y + h * 0.62), (x + w / 4, y + h / 2), (x + w / 2, y), (x + w / 4, y - h / 2), (x, y - h * 0.62), (x - w / 4, y - h / 2)]
    ax.add_patch(Polygon(verts, closed=True, transform=ax.transAxes, facecolor=WHITE, edgecolor=colour, lw=0.75, zorder=3))
    ax.add_patch(Circle((x, y), h * 0.23, transform=ax.transAxes, facecolor=colour, edgecolor="none", zorder=4))


def draw_gear(ax: plt.Axes, x: float, y: float, r: float, colour: str) -> None:
    ax.add_patch(Circle((x, y), r * 0.68, transform=ax.transAxes, facecolor=mpl.colors.to_rgba(colour, 0.14), edgecolor=colour, lw=0.8, zorder=3))
    ax.add_patch(Circle((x, y), r * 0.22, transform=ax.transAxes, facecolor=WHITE, edgecolor=colour, lw=0.7, zorder=4))
    for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        x0 = x + r * 0.68 * np.cos(angle)
        y0 = y + r * 0.68 * np.sin(angle)
        x1 = x + r * 1.03 * np.cos(angle)
        y1 = y + r * 1.03 * np.sin(angle)
        ax.plot([x0, x1], [y0, y1], color=colour, lw=1.5, transform=ax.transAxes, zorder=3)


def draw_clock(ax: plt.Axes, x: float, y: float, r: float, colour: str) -> None:
    ax.add_patch(Circle((x, y), r, transform=ax.transAxes, facecolor=WHITE, edgecolor=colour, lw=0.85, zorder=3))
    for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        xx = x + 0.78 * r * np.cos(angle)
        yy = y + 0.78 * r * np.sin(angle)
        ax.add_patch(Circle((xx, yy), 0.0017, transform=ax.transAxes, facecolor=colour, edgecolor="none", zorder=4))
    ax.plot([x, x], [y, y + r * 0.53], color=colour, lw=0.8, transform=ax.transAxes, zorder=4)
    ax.plot([x, x + r * 0.46], [y, y - r * 0.24], color=colour, lw=0.8, transform=ax.transAxes, zorder=4)


def draw_inventory(ax: plt.Axes, x: float, y: float, w: float, h: float, colour: str, level: float = 0.58) -> None:
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, transform=ax.transAxes, facecolor=WHITE, edgecolor=CHARCOAL, lw=0.8, zorder=3))
    ax.add_patch(Rectangle((x - w / 2 + 0.003, y - h / 2 + 0.003), w - 0.006, (h - 0.006) * level, transform=ax.transAxes, facecolor=mpl.colors.to_rgba(colour, 0.50), edgecolor="none", zorder=3))
    ax.plot([x - w * 0.65, x + w * 0.65], [y + h / 2, y + h / 2], color=CHARCOAL, lw=0.8, transform=ax.transAxes, zorder=4)


def draw_battery(ax: plt.Axes, x: float, y: float, w: float, h: float, colour: str) -> None:
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, transform=ax.transAxes, facecolor=mpl.colors.to_rgba(colour, 0.28), edgecolor=colour, lw=0.75, zorder=3))
    ax.add_patch(Rectangle((x + w / 2, y - h * 0.16), w * 0.13, h * 0.32, transform=ax.transAxes, facecolor=colour, edgecolor="none", zorder=3))


def draw_lock(ax: plt.Axes, x: float, y: float, colour: str) -> None:
    ax.add_patch(Rectangle((x - 0.009, y - 0.010), 0.018, 0.017, transform=ax.transAxes, facecolor=mpl.colors.to_rgba(colour, 0.16), edgecolor=colour, lw=0.7, zorder=4))
    ax.add_patch(Arc((x, y + 0.007), 0.014, 0.020, theta1=0, theta2=180, transform=ax.transAxes, color=colour, lw=0.8, zorder=4))


def draw_day_blocks(ax: plt.Axes, x: float, y: float, w: float, colour: str, clustered: bool) -> None:
    positions = [0.03, 0.20, 0.39, 0.61, 0.80, 0.96] if not clustered else [0.03, 0.09, 0.15, 0.77, 0.83, 0.89]
    ax.plot([x - w / 2, x + w / 2], [y, y], color=SOFT, lw=0.6, transform=ax.transAxes, zorder=2)
    for p in positions:
        xx = x - w / 2 + p * w
        ax.add_patch(Rectangle((xx - 0.004, y - 0.013), 0.008, 0.026, transform=ax.transAxes, facecolor=mpl.colors.to_rgba(colour, 0.80), edgecolor=WHITE, lw=0.3, zorder=3))


def draw_wave_pair(ax: plt.Axes, x0: float, x1: float, y: float) -> None:
    t = np.linspace(0, 1, 100)
    x = x0 + (x1 - x0) * t
    up = 0.017 * np.maximum(0, np.sin(2 * np.pi * (4.1 * t - 0.08)))
    down = -0.015 * np.maximum(0, np.sin(2 * np.pi * (4.1 * t + 0.42)))
    ax.fill_between(x, y, y + up, transform=ax.transAxes, color=mpl.colors.to_rgba(TASK, 0.62), lw=0, zorder=3)
    ax.fill_between(x, y, y + down, transform=ax.transAxes, color=mpl.colors.to_rgba(RECHARGE, 0.62), lw=0, zorder=3)
    ax.plot([x0, x1], [y, y], color=MUTED, lw=0.55, transform=ax.transAxes, zorder=4)


def draw_cycle(ax: plt.Axes, x: float, y: float, r: float, colour: str) -> None:
    ax.add_patch(Arc((x, y), 2 * r, 2 * r, theta1=20, theta2=215, transform=ax.transAxes, color=colour, lw=1.1, zorder=3))
    ax.add_patch(Arc((x, y), 2 * r, 2 * r, theta1=200, theta2=385, transform=ax.transAxes, color=colour, lw=1.1, zorder=3))
    arrow(ax, (x + r * 0.88, y + r * 0.15), (x + r * 0.70, y + r * 0.47), colour=colour, lw=0.8, mutation=4.5)
    arrow(ax, (x - r * 0.88, y - r * 0.15), (x - r * 0.70, y - r * 0.47), colour=colour, lw=0.8, mutation=4.5)


def draw_duration_curve(ax: plt.Axes, x: float, y: float, w: float, h: float, colour: str) -> None:
    ax.plot([x - w / 2, x - w / 2, x + w / 2], [y + h / 2, y - h / 2, y - h / 2], color=MUTED, lw=0.5, transform=ax.transAxes, zorder=2)
    t = np.linspace(-1, 1, 60)
    xx = x + t * w * 0.43
    yy = y - h * 0.15 + h * 0.46 * (t**2)
    ax.plot(xx, yy, color=colour, lw=1.0, transform=ax.transAxes, zorder=3)


def draw_coin(ax: plt.Axes, x: float, y: float, r: float, colour: str) -> None:
    ax.add_patch(Circle((x, y), r, transform=ax.transAxes, facecolor=mpl.colors.to_rgba(colour, 0.18), edgecolor=colour, lw=0.85, zorder=3))
    text(ax, x, y, "V", size=5.7, colour=colour, weight="bold")


def panel_a(ax: plt.Axes) -> None:
    x0, x1 = 0.025, 0.302
    header(ax, "a", "Missing time coordinate", x0)

    line0, line1 = x0 + 0.025, x1 - 0.010
    y = 0.833
    ax.plot([line0, line1], [y, y], color=INK, lw=0.75, transform=ax.transAxes)
    positions = [0.05, 0.18, 0.31, 0.55, 0.68, 0.84]
    widths = [0.035, 0.050, 0.027, 0.063, 0.040, 0.052]
    for pos, width in zip(positions, widths):
        xx = line0 + pos * (line1 - line0)
        ww = width * (line1 - line0)
        ax.add_patch(Rectangle((xx, y - 0.026), ww, 0.052, transform=ax.transAxes, facecolor=mpl.colors.to_rgba(TASK, 0.72), edgecolor=WHITE, lw=0.35, zorder=3))

    bracket(ax, line0, line1, 0.890, "annual task", CHARCOAL, 0.907)
    event_x0 = line0 + positions[3] * (line1 - line0)
    event_x1 = event_x0 + widths[3] * (line1 - line0)
    bracket(ax, event_x0, event_x1, 0.786, "event duration", TASK, 0.769)
    gap_x0 = line0 + (positions[2] + widths[2]) * (line1 - line0)
    gap_x1 = event_x0
    ax.add_patch(Rectangle((gap_x0, y - 0.028), gap_x1 - gap_x0, 0.056, transform=ax.transAxes, facecolor=mpl.colors.to_rgba(RECOVERY, 0.12), edgecolor="none", zorder=1))
    bracket(ax, gap_x0, gap_x1, 0.714, "recovery window", RECOVERY, 0.695)
    arrow(ax, ((gap_x0 + gap_x1) / 2, 0.723), ((gap_x0 + gap_x1) / 2, y - 0.027), colour=RECOVERY, lw=0.75, mutation=5.0)

    ax.plot([x0 + 0.002, x1 - 0.002], [0.647, 0.647], color=GRID, lw=0.55, transform=ax.transAxes)
    text(ax, (x0 + x1) / 2, 0.620, "same 24 h blocks", size=5.5, weight="bold")
    draw_day_blocks(ax, (line0 + line1) / 2, 0.566, line1 - line0, BLUE, clustered=False)
    draw_day_blocks(ax, (line0 + line1) / 2, 0.486, line1 - line0, TASK, clustered=True)
    text(ax, line0 - 0.007, 0.566, "dispersed", size=5.1, colour=BLUE, weight="bold", ha="right")
    text(ax, line0 - 0.007, 0.486, "clustered", size=5.1, colour=TASK, weight="bold", ha="right")

    icon_x = [x0 + 0.065, x0 + 0.139, x0 + 0.213]
    draw_lock(ax, icon_x[0], 0.409, CHARCOAL)
    draw_lock(ax, icon_x[1], 0.409, CHARCOAL)
    draw_lock(ax, icon_x[2], 0.409, CHARCOAL)
    text(ax, icon_x[0], 0.376, "task", size=4.8, colour=MUTED)
    text(ax, icon_x[1], 0.376, "recharge", size=4.8, colour=MUTED)
    text(ax, icon_x[2], 0.376, "day profile", size=4.8, colour=MUTED)

    ax.plot([x0 + 0.002, x1 - 0.002], [0.336, 0.336], color=GRID, lw=0.55, transform=ax.transAxes)
    chain_y = 0.235
    # Spacing glyph.
    for dx in (-0.018, 0.0, 0.029):
        ax.add_patch(Rectangle((x0 + 0.020 + dx, chain_y - 0.015), 0.012, 0.030, transform=ax.transAxes, facecolor=mpl.colors.to_rgba(TASK, 0.75), edgecolor="none", zorder=3))
    sx = x0 + 0.031
    cx = x0 + 0.103
    ix = x0 + 0.177
    vx = x0 + 0.248
    draw_clock(ax, cx, chain_y, 0.023, RECOVERY)
    draw_inventory(ax, ix, chain_y, 0.038, 0.055, RECOVERY)
    draw_coin(ax, vx, chain_y, 0.020, COST)
    arrow(ax, (sx + 0.027, chain_y), (cx - 0.029, chain_y), colour=RECOVERY)
    arrow(ax, (cx + 0.029, chain_y), (ix - 0.026, chain_y), colour=RECOVERY)
    arrow(ax, (ix + 0.027, chain_y), (vx - 0.027, chain_y), colour=COST)
    for xx, label, colour in zip([sx, cx, ix, vx], ["spacing", "recovery", "inventory", "value"], [TASK, RECOVERY, INK, COST]):
        text(ax, xx, 0.184, label, size=4.9, colour=colour, weight="bold")
    text(ax, (x0 + x1) / 2, 0.116, "SEQUENCING HYPOTHESIS", size=5.8, colour=COST, weight="bold")


def draw_wind_icon(ax: plt.Axes, x: float, y: float, colour: str) -> None:
    ax.plot([x, x], [y - 0.025, y + 0.005], color=colour, lw=0.75, transform=ax.transAxes, zorder=3)
    ax.add_patch(Circle((x, y + 0.008), 0.003, transform=ax.transAxes, facecolor=colour, edgecolor="none", zorder=4))
    for angle in (90, 210, 330):
        rad = np.deg2rad(angle)
        ax.plot([x, x + 0.025 * np.cos(rad)], [y + 0.008, y + 0.008 + 0.025 * np.sin(rad)], color=colour, lw=0.9, transform=ax.transAxes, zorder=3)


def draw_solar_icon(ax: plt.Axes, x: float, y: float, colour: str) -> None:
    ax.add_patch(Polygon([(x - 0.024, y - 0.012), (x + 0.023, y - 0.012), (x + 0.017, y + 0.017), (x - 0.018, y + 0.017)], closed=True, transform=ax.transAxes, facecolor=mpl.colors.to_rgba(colour, 0.20), edgecolor=colour, lw=0.65, zorder=3))
    ax.plot([x, x], [y - 0.012, y + 0.017], color=colour, lw=0.4, transform=ax.transAxes)
    ax.plot([x - 0.020, x + 0.020], [y + 0.002, y + 0.002], color=colour, lw=0.4, transform=ax.transAxes)
    ax.plot([x, x], [y - 0.012, y - 0.026], color=colour, lw=0.65, transform=ax.transAxes)


def draw_thermometer(ax: plt.Axes, x: float, y: float, colour: str) -> None:
    ax.add_patch(Circle((x, y - 0.016), 0.010, transform=ax.transAxes, facecolor=mpl.colors.to_rgba(colour, 0.25), edgecolor=colour, lw=0.7, zorder=3))
    ax.add_patch(Rectangle((x - 0.004, y - 0.016), 0.008, 0.038, transform=ax.transAxes, facecolor=WHITE, edgecolor=colour, lw=0.7, zorder=3))
    ax.plot([x, x], [y - 0.014, y + 0.014], color=colour, lw=1.7, transform=ax.transAxes, zorder=4)


def panel_b(ax: plt.Axes) -> None:
    x0, x1 = 0.332, 0.690
    header(ax, "b", "Evidence architecture", x0)

    # Natural occurrence band.
    ax.add_patch(Rectangle((x0, 0.652), x1 - x0, 0.255, transform=ax.transAxes, facecolor=NATURAL_PALE, edgecolor="none", zorder=0))
    ax.plot([x0, x0], [0.652, 0.907], color=NATURAL, lw=2.3, transform=ax.transAxes)
    text(ax, x0 + 0.014, 0.879, "Natural occurrence", size=6.2, colour=NATURAL, weight="bold", ha="left")

    inputs_x = [x0 + 0.045, x0 + 0.105, x0 + 0.165]
    draw_wind_icon(ax, inputs_x[0], 0.789, BLUE)
    draw_solar_icon(ax, inputs_x[1], 0.789, GOLD)
    draw_thermometer(ax, inputs_x[2], 0.789, TASK)
    for xx, label, colour in zip(inputs_x, ["wind", "solar", "demand"], [BLUE, GOLD, TASK]):
        text(ax, xx, 0.743, label, size=4.7, colour=colour)

    arrow(ax, (x0 + 0.197, 0.790), (x0 + 0.220, 0.790), colour=NATURAL)
    draw_calendar(ax, x0 + 0.250, 0.790, 0.044, 0.060, NATURAL)
    text(ax, x0 + 0.250, 0.742, f"{int(VALUES['natural_start_year'])}–{int(VALUES['natural_end_year'])}", size=4.9, colour=INK, weight="bold")

    arrow(ax, (x0 + 0.277, 0.790), (x0 + 0.296, 0.790), colour=NATURAL)
    draw_globe(ax, x0 + 0.320, 0.790, 0.024, NATURAL)
    text(ax, x0 + 0.320, 0.742, f"Europe-{int(VALUES['natural_countries'])}", size=4.8, colour=INK, weight="bold")
    text(ax, x0 + 0.250, 0.686, f"{int(VALUES['natural_operating_years'])} unrearranged years", size=5.0, colour=MUTED)

    draw_eye(ax, x1 - 0.029, 0.858, 0.040, 0.025, NATURAL)
    text(ax, x1 - 0.029, 0.823, "occurrence", size=4.8, colour=NATURAL, weight="bold")

    # Controlled mechanism band.
    ax.add_patch(Rectangle((x0, 0.225), x1 - x0, 0.382, transform=ax.transAxes, facecolor=CONTROLLED_PALE, edgecolor="none", zorder=0))
    ax.plot([x0, x0], [0.225, 0.607], color=CONTROLLED, lw=2.3, transform=ax.transAxes)
    text(ax, x0 + 0.014, 0.580, "Controlled mechanism", size=6.2, colour=CONTROLLED, weight="bold", ha="left")

    regime_x = [x0 + 0.050, x0 + 0.132, x0 + 0.214]
    regimes = ["Europe", "China", "Japan"]
    sources = ["gas / VRE", "gas / curtail.", "grid / PV"]
    for xx, regime, source in zip(regime_x, regimes, sources):
        colour = REGIME_COLORS[regime]
        ax.add_patch(Circle((xx, 0.502), 0.016, transform=ax.transAxes, facecolor=colour, edgecolor=WHITE, lw=0.55, zorder=3))
        text(ax, xx, 0.467, regime, size=5.2, colour=colour, weight="bold")
        text(ax, xx, 0.438, f"n={int(VALUES[f'{regime}_system_years'])}", size=4.7, colour=MUTED)
        # Paired task-recharge marks.
        arrow(ax, (xx - 0.012, 0.398), (xx - 0.012, 0.420), colour=TASK, lw=0.75, mutation=4.5)
        arrow(ax, (xx + 0.012, 0.420), (xx + 0.012, 0.398), colour=RECHARGE, lw=0.75, mutation=4.5)
        text(ax, xx, 0.368, source, size=4.4, colour=INK)

    merge_x = x0 + 0.286
    collector_y = 0.338
    for xx, regime in zip(regime_x, regimes):
        ax.plot([xx, xx], [0.350, collector_y], color=REGIME_COLORS[regime], lw=0.65, transform=ax.transAxes, zorder=3)
    ax.plot([regime_x[0], regime_x[-1]], [collector_y, collector_y], color=CONTROLLED, lw=0.75, transform=ax.transAxes, zorder=3)
    arrow(ax, (regime_x[-1], collector_y), (merge_x - 0.046, collector_y), colour=CONTROLLED, lw=0.75, mutation=5.0)

    draw_day_blocks(ax, merge_x, 0.370, 0.080, CONTROLLED, clustered=False)
    draw_day_blocks(ax, merge_x, 0.306, 0.080, CONTROLLED, clustered=True)
    arrow(ax, (merge_x, 0.353), (merge_x, 0.324), colour=CONTROLLED, lw=0.8, mutation=5.0)
    draw_lock(ax, merge_x, 0.258, CONTROLLED)
    text(ax, merge_x, 0.229, "24 h blocks", size=4.8, colour=CONTROLLED, weight="bold")

    arrow(ax, (merge_x + 0.046, collector_y), (x1 - 0.040, collector_y), colour=CONTROLLED)
    draw_gear(ax, x1 - 0.025, collector_y, 0.020, CONTROLLED)
    text(ax, x1 - 0.025, 0.291, "mechanism", size=4.8, colour=CONTROLLED, weight="bold")

    text(ax, (x0 + x1) / 2, 0.177, "annual ledger fixed", size=4.9, colour=MUTED, weight="bold")


def draw_sequence_icon(ax: plt.Axes, x: float, y: float, colour: str, clustered: bool) -> None:
    positions = [0.05, 0.25, 0.47, 0.70, 0.93] if not clustered else [0.05, 0.12, 0.19, 0.82, 0.89]
    ax.plot([x - 0.050, x + 0.050], [y, y], color=SOFT, lw=0.55, transform=ax.transAxes)
    for p in positions:
        xx = x - 0.050 + p * 0.100
        ax.add_patch(Rectangle((xx - 0.004, y - 0.012), 0.008, 0.024, transform=ax.transAxes, facecolor=mpl.colors.to_rgba(colour, 0.82), edgecolor="none", zorder=3))


def panel_c(ax: plt.Axes) -> None:
    x0, x1 = 0.716, 0.983
    header(ax, "c", "Common method chain", x0)

    draw_sequence_icon(ax, x0 + 0.063, 0.856, NATURAL, clustered=False)
    draw_sequence_icon(ax, x0 + 0.204, 0.856, CONTROLLED, clustered=True)
    text(ax, x0 + 0.063, 0.817, "natural", size=4.9, colour=NATURAL, weight="bold")
    text(ax, x0 + 0.204, 0.817, "controlled", size=4.9, colour=CONTROLLED, weight="bold")
    arrow(ax, (x0 + 0.063, 0.801), (x0 + 0.128, 0.763), colour=NATURAL)
    arrow(ax, (x0 + 0.204, 0.801), (x0 + 0.140, 0.763), colour=CONTROLLED)

    draw_wave_pair(ax, x0 + 0.018, x1 - 0.018, 0.720)
    text(ax, x0 + 0.020, 0.680, "task", size=4.7, colour=TASK, weight="bold", ha="left")
    text(ax, x1 - 0.020, 0.680, "recharge", size=4.7, colour=RECHARGE, weight="bold", ha="right")
    arrow(ax, (x0 + 0.134, 0.665), (x0 + 0.134, 0.630), colour=RECOVERY)

    draw_clock(ax, x0 + 0.064, 0.578, 0.027, RECOVERY)
    draw_inventory(ax, x0 + 0.202, 0.578, 0.050, 0.074, RECOVERY)
    arrow(ax, (x0 + 0.098, 0.578), (x0 + 0.169, 0.578), colour=RECOVERY)
    text(ax, x0 + 0.064, 0.530, "recovery", size=4.9, colour=RECOVERY, weight="bold")
    text(ax, x0 + 0.202, 0.530, "inventory", size=4.9, colour=INK, weight="bold")
    arrow(ax, (x0 + 0.134, 0.507), (x0 + 0.134, 0.474), colour=CHARCOAL)

    draw_battery(ax, x0 + 0.052, 0.420, 0.036, 0.040, BLUE)
    draw_battery(ax, x0 + 0.112, 0.420, 0.040, 0.072, RECOVERY)
    text(ax, x0 + 0.052, 0.372, "4 h LFP", size=4.8, colour=BLUE, weight="bold")
    text(ax, x0 + 0.112, 0.372, "LDES", size=4.8, colour=RECOVERY, weight="bold")

    draw_calendar(ax, x0 + 0.185, 0.422, 0.036, 0.049, CONTROLLED)
    text(ax, x0 + 0.185, 0.372, "1–12 d", size=4.7, colour=CONTROLLED, weight="bold")
    ax.add_patch(Polygon([(x0 + 0.239, 0.449), (x0 + 0.224, 0.419), (x0 + 0.237, 0.419), (x0 + 0.226, 0.386), (x0 + 0.252, 0.424), (x0 + 0.239, 0.424)], closed=True, transform=ax.transAxes, facecolor=RECHARGE, edgecolor="none", zorder=3))
    text(ax, x0 + 0.239, 0.372, "1×–4×", size=4.7, colour=RECHARGE, weight="bold")
    text(ax, x0 + 0.214, 0.477, "2025→2035", size=4.8, colour=COST, weight="bold")
    arrow(ax, (x0 + 0.134, 0.342), (x0 + 0.134, 0.314), colour=CHARCOAL)

    output_x = [x0 + 0.050, x0 + 0.134, x0 + 0.218]
    draw_cycle(ax, output_x[0], 0.269, 0.020, RECOVERY)
    draw_duration_curve(ax, output_x[1], 0.269, 0.046, 0.045, GOLD)
    draw_coin(ax, output_x[2], 0.269, 0.020, COST)
    for xx, label, colour in zip(output_x, ["cycling", "duration", "value"], [RECOVERY, GOLD, COST]):
        text(ax, xx, 0.222, label, size=4.8, colour=colour, weight="bold")
    arrow(ax, (x0 + 0.134, 0.200), (x0 + 0.134, 0.177), colour=COST)

    centre = x0 + 0.134
    ax.plot([centre - 0.083, centre + 0.083], [0.142, 0.142], color=CHARCOAL, lw=1.05, transform=ax.transAxes, zorder=3)
    ax.plot([centre, centre], [0.107, 0.142], color=CHARCOAL, lw=0.95, transform=ax.transAxes, zorder=3)
    ax.add_patch(Polygon([[centre - 0.019, 0.103], [centre + 0.019, 0.103], [centre, 0.124]], closed=True, transform=ax.transAxes, facecolor=CHARCOAL, edgecolor="none", zorder=3))
    ax.add_patch(Circle((centre - 0.083, 0.142), 0.012, transform=ax.transAxes, facecolor=CONTROLLED, edgecolor=WHITE, lw=0.5, zorder=4))
    ax.add_patch(Rectangle((centre + 0.072, 0.131), 0.022, 0.022, transform=ax.transAxes, facecolor=COST, edgecolor=WHITE, lw=0.5, zorder=4))
    text(ax, centre - 0.083, 0.171, "|ΔVseq|", size=5.0, colour=CONTROLLED, weight="bold")
    text(ax, centre + 0.083, 0.171, f"{VALUES['technology_cost_scale']:.2f}", size=5.0, colour=COST, weight="bold")

    ax.add_patch(Rectangle((x0 + 0.026, 0.030), x1 - x0 - 0.052, 0.054, transform=ax.transAxes, facecolor=PURPLE_PALE, edgecolor=COST, lw=0.75, zorder=2))
    text(ax, centre, 0.057, "SEQUENCING PARITY", size=6.3, colour=COST, weight="bold")


def build_figure() -> plt.Figure:
    setup()
    fig = plt.figure(figsize=(WIDTH_MM * MM, HEIGHT_MM * MM))
    ax = fig.add_axes([0.020, 0.026, 0.964, 0.954])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.plot([0.316, 0.316], [0.040, 0.948], color=SOFT, lw=0.7, transform=ax.transAxes)
    ax.plot([0.701, 0.701], [0.040, 0.948], color=SOFT, lw=0.7, transform=ax.transAxes)

    panel_a(ax)
    panel_b(ax)
    panel_c(ax)
    return fig


def main() -> None:
    fig = build_figure()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{STEM}.pdf", metadata={"Creator": "Python/matplotlib"})
    fig.savefig(OUT / f"{STEM}.svg")
    fig.savefig(OUT / f"{STEM}.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(OUT / f"{STEM}.png", dpi=600)
    rgba = plt.imread(OUT / f"{STEM}.png")
    grey = np.dot(rgba[..., :3], [0.2126, 0.7152, 0.0722])
    plt.imsave(OUT / f"{STEM}_grayscale.png", grey, cmap="gray", vmin=0, vmax=1)
    plt.close(fig)


if __name__ == "__main__":
    main()
