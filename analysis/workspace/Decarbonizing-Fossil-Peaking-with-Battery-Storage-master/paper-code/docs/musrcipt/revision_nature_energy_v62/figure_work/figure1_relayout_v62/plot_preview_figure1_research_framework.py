#!/usr/bin/env python3
"""Standalone Figure 1 preview: sequencing-parity research framework."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle


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
    PALE,
    PURPLE,
    REGIME_COLORS,
    TEAL,
    WHITE,
    configure,
)


OUT = HERE / "outputs"
SOURCE = HERE / "source_data" / "figure1_framework_values.csv"
STEM = "fig_v62_preview_figure1_research_framework"
WIDTH_MM = 183.0
HEIGHT_MM = 154.0

TASK = CORAL
RECHARGE = BLUE
RECOVERY = TEAL
NATURAL = "#2F8B7C"
CONTROLLED = "#D8775E"
COST = PURPLE
CHARCOAL = "#3E4D53"
SOFT_LINE = "#C8D1CE"
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
            "axes.titlesize": 7.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
        }
    )


def add_text(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    size: float = 6.0,
    colour: str = INK,
    weight: str = "normal",
    ha: str = "left",
    va: str = "center",
    rotation: float = 0,
    zorder: int = 8,
) -> None:
    ax.text(
        x,
        y,
        text,
        fontsize=size,
        color=colour,
        fontweight=weight,
        ha=ha,
        va=va,
        rotation=rotation,
        transform=ax.transAxes,
        zorder=zorder,
    )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    colour: str = MUTED,
    width: float = 0.85,
    mutation: float = 7.0,
    style: str = "-|>",
    connection: str = "arc3,rad=0",
    zorder: int = 5,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle=style,
            connectionstyle=connection,
            mutation_scale=mutation,
            lw=width,
            color=colour,
            shrinkA=0,
            shrinkB=0,
            zorder=zorder,
        )
    )


def section_header(ax: plt.Axes, label: str, title: str, x: float, y: float) -> None:
    add_text(ax, x, y, label, size=8.0, weight="bold", va="top")
    add_text(ax, x + 0.021, y - 0.001, title, size=7.2, weight="bold", va="top")


def bracket(
    ax: plt.Axes,
    x0: float,
    x1: float,
    y: float,
    text: str,
    *,
    colour: str,
    text_y: float,
) -> None:
    ax.plot([x0, x1], [y, y], color=colour, lw=0.85, transform=ax.transAxes, zorder=5)
    tick = 0.009
    ax.plot([x0, x0], [y - tick, y + tick], color=colour, lw=0.85, transform=ax.transAxes, zorder=5)
    ax.plot([x1, x1], [y - tick, y + tick], color=colour, lw=0.85, transform=ax.transAxes, zorder=5)
    add_text(ax, (x0 + x1) / 2, text_y, text, size=5.5, colour=colour, weight="bold", ha="center")


def event_sequence(
    ax: plt.Axes,
    x0: float,
    x1: float,
    y: float,
    positions: list[float],
    *,
    label: str,
    colour: str,
    height: float = 0.027,
) -> None:
    ax.plot([x0, x1], [y, y], color=SOFT_LINE, lw=0.65, transform=ax.transAxes, zorder=2)
    width = (x1 - x0) * 0.033
    for p in positions:
        x = x0 + p * (x1 - x0)
        ax.add_patch(
            Rectangle(
                (x - width / 2, y - height / 2),
                width,
                height,
                transform=ax.transAxes,
                facecolor=mpl.colors.to_rgba(colour, 0.82),
                edgecolor=WHITE,
                lw=0.35,
                zorder=4,
            )
        )
    add_text(ax, x0 - 0.007, y, label, size=5.5, colour=colour, weight="bold", ha="right")


def draw_problem_panel(ax: plt.Axes) -> None:
    x0, x1 = 0.025, 0.302
    section_header(ax, "a", "The missing temporal coordinate", x0, 0.968)

    add_text(ax, x0 + 0.002, 0.914, "Annual adequacy is usually priced on two clocks", size=5.7, colour=MUTED)

    line0, line1 = x0 + 0.025, x1 - 0.008
    y = 0.842
    ax.plot([line0, line1], [y, y], color=INK, lw=0.75, transform=ax.transAxes, zorder=2)
    positions = [0.05, 0.18, 0.31, 0.55, 0.68, 0.84]
    widths = [0.035, 0.050, 0.027, 0.063, 0.040, 0.052]
    for pos, width in zip(positions, widths):
        xx = line0 + pos * (line1 - line0)
        ww = width * (line1 - line0)
        ax.add_patch(
            Rectangle(
                (xx, y - 0.025),
                ww,
                0.05,
                transform=ax.transAxes,
                facecolor=mpl.colors.to_rgba(TASK, 0.72),
                edgecolor=WHITE,
                lw=0.4,
                zorder=3,
            )
        )

    bracket(ax, line0, line1, 0.889, "annual task", colour=CHARCOAL, text_y=0.903)
    event_x0 = line0 + positions[3] * (line1 - line0)
    event_x1 = event_x0 + widths[3] * (line1 - line0)
    bracket(ax, event_x0, event_x1, 0.789, "event duration", colour=TASK, text_y=0.772)
    gap_x0 = line0 + (positions[2] + widths[2]) * (line1 - line0)
    gap_x1 = event_x0
    bracket(ax, gap_x0, gap_x1, 0.723, "recovery window", colour=RECOVERY, text_y=0.705)
    ax.add_patch(
        Rectangle(
            (gap_x0, 0.818),
            gap_x1 - gap_x0,
            0.048,
            transform=ax.transAxes,
            facecolor=mpl.colors.to_rgba(RECOVERY, 0.12),
            edgecolor="none",
            zorder=1,
        )
    )
    arrow(ax, ((gap_x0 + gap_x1) / 2, 0.713), ((gap_x0 + gap_x1) / 2, 0.817), colour=RECOVERY, width=0.75, mutation=5.5)

    ax.plot([x0 + 0.002, x1 - 0.002], [0.660, 0.660], color=GRID, lw=0.55, transform=ax.transAxes)
    add_text(ax, x0 + 0.002, 0.637, "Same intact 24 h blocks, different spacing", size=5.7, weight="bold")

    dispersed = [0.03, 0.18, 0.35, 0.51, 0.70, 0.88]
    clustered = [0.04, 0.08, 0.12, 0.63, 0.67, 0.71]
    event_sequence(ax, line0, line1, 0.581, dispersed, label="dispersed", colour=BLUE)
    event_sequence(ax, line0, line1, 0.499, clustered, label="clustered", colour=TASK)

    for xa, xb in zip(dispersed, clustered):
        x_start = line0 + xa * (line1 - line0)
        x_end = line0 + xb * (line1 - line0)
        ax.plot(
            [x_start, x_end],
            [0.563, 0.517],
            color=mpl.colors.to_rgba(MUTED, 0.26),
            lw=0.45,
            transform=ax.transAxes,
            zorder=1,
        )

    add_text(ax, (line0 + line1) / 2, 0.452, "annual task + recharge fixed", size=4.8, colour=MUTED, ha="center")
    add_text(ax, (line0 + line1) / 2, 0.429, "within-day profiles fixed", size=4.8, colour=MUTED, ha="center")

    arrow(ax, (x0 + 0.139, 0.408), (x0 + 0.139, 0.363), colour=RECOVERY, width=0.9, mutation=7)
    add_text(ax, x0 + 0.139, 0.385, "spacing changes", size=5.3, colour=RECOVERY, weight="bold", ha="center")

    y0 = 0.314
    ax.add_patch(
        Rectangle(
            (x0 + 0.015, y0 - 0.043),
            x1 - x0 - 0.03,
            0.086,
            transform=ax.transAxes,
            facecolor=NATURAL_PALE,
            edgecolor="none",
            zorder=1,
        )
    )
    add_text(ax, x0 + 0.026, y0 + 0.020, "recovery time", size=5.6, colour=RECOVERY, weight="bold")
    arrow(ax, (x0 + 0.103, y0 + 0.020), (x0 + 0.148, y0 + 0.020), colour=RECOVERY, width=0.8, mutation=6)
    add_text(ax, x0 + 0.158, y0 + 0.020, "inventory rebuild", size=5.6, colour=INK, weight="bold")
    add_text(ax, x0 + 0.026, y0 - 0.019, "The time between calls becomes a value coordinate", size=5.2, colour=MUTED)

    add_text(ax, x0 + 0.003, 0.216, "Hypothesis", size=5.5, colour=COST, weight="bold")
    ax.plot([x0 + 0.003, x0 + 0.058], [0.201, 0.201], color=COST, lw=1.2, transform=ax.transAxes)
    add_text(
        ax,
        x0 + 0.003,
        0.163,
        "Event sequencing can move annual storage\nvalue by as much as a decade of\ntechnology learning.",
        size=5.7,
        colour=INK,
        weight="bold",
        va="top",
    )


def draw_wave_icon(ax: plt.Axes, x: float, y: float, colour: str, phase: float, amp: float = 0.010) -> None:
    t = np.linspace(0, 1, 50)
    xx = x + 0.050 * t
    yy = y + amp * np.sin(2 * np.pi * (1.5 * t + phase)) + 0.003 * np.sin(2 * np.pi * (4 * t + phase))
    ax.plot(xx, yy, color=colour, lw=0.8, transform=ax.transAxes, zorder=4)


def draw_evidence_panel(ax: plt.Axes) -> None:
    x0, x1 = 0.332, 0.687
    section_header(ax, "b", "Two evidence layers", x0, 0.968)

    # Natural chronology branch.
    ax.add_patch(
        Rectangle(
            (x0, 0.665),
            x1 - x0,
            0.245,
            transform=ax.transAxes,
            facecolor=NATURAL_PALE,
            edgecolor="none",
            zorder=0,
        )
    )
    ax.plot([x0, x0], [0.665, 0.910], color=NATURAL, lw=2.3, transform=ax.transAxes, zorder=2)
    add_text(ax, x0 + 0.014, 0.884, "Natural occurrence", size=6.4, colour=NATURAL, weight="bold")
    add_text(ax, x0 + 0.014, 0.852, "unrearranged hourly weather replay", size=5.2, colour=MUTED)

    draw_wave_icon(ax, x0 + 0.020, 0.802, BLUE, 0.00)
    draw_wave_icon(ax, x0 + 0.020, 0.774, GOLD, 0.23)
    draw_wave_icon(ax, x0 + 0.020, 0.746, CORAL, 0.47)
    add_text(ax, x0 + 0.076, 0.802, "wind", size=5.1, colour=BLUE)
    add_text(ax, x0 + 0.076, 0.774, "solar", size=5.1, colour=GOLD)
    add_text(ax, x0 + 0.076, 0.746, "temperature-driven demand", size=5.1, colour=CORAL)

    arrow(ax, (x0 + 0.164, 0.775), (x0 + 0.198, 0.775), colour=NATURAL, width=0.9, mutation=6)
    natural_years = f"{int(VALUES['natural_start_year'])}–{int(VALUES['natural_end_year'])}"
    add_text(ax, x0 + 0.208, 0.808, natural_years, size=6.3, colour=INK, weight="bold")
    add_text(ax, x0 + 0.208, 0.778, f"{int(VALUES['natural_operating_years'])} complete operating years", size=5.3, colour=INK)
    add_text(ax, x0 + 0.208, 0.749, f"{int(VALUES['natural_countries'])}-country European aggregate", size=5.3, colour=MUTED)
    add_text(ax, x1 - 0.012, 0.694, "Does nature realize the cost scale?", size=5.3, colour=NATURAL, weight="bold", ha="right")

    # Controlled mechanism branch.
    ax.add_patch(
        Rectangle(
            (x0, 0.235),
            x1 - x0,
            0.385,
            transform=ax.transAxes,
            facecolor=CONTROLLED_PALE,
            edgecolor="none",
            zorder=0,
        )
    )
    ax.plot([x0, x0], [0.235, 0.620], color=CONTROLLED, lw=2.3, transform=ax.transAxes, zorder=2)
    add_text(ax, x0 + 0.014, 0.594, "Controlled mechanism", size=6.4, colour=CONTROLLED, weight="bold")
    add_text(ax, x0 + 0.014, 0.562, "parallel task–recharge regimes", size=5.2, colour=MUTED)

    lane_y = [0.497, 0.421, 0.345]
    labels = [
        ("Europe", f"national  |  n={int(VALUES['Europe_system_years'])}", "gas peaking  +  VRE surplus"),
        ("China", f"provincial  |  n={int(VALUES['China_system_years'])}", "gas peaking  +  curtailment"),
        ("Japan", f"community  |  n={int(VALUES['Japan_system_years'])}", "grid import  +  PV reverse flow"),
    ]
    for y, (regime, scale, streams) in zip(lane_y, labels):
        colour = REGIME_COLORS[regime]
        ax.add_patch(Circle((x0 + 0.023, y), 0.008, transform=ax.transAxes, facecolor=colour, edgecolor=WHITE, lw=0.5, zorder=4))
        add_text(ax, x0 + 0.039, y + 0.013, regime, size=5.7, colour=colour, weight="bold")
        add_text(ax, x0 + 0.098, y + 0.013, scale, size=5.0, colour=MUTED)
        add_text(ax, x0 + 0.039, y - 0.017, streams, size=5.1, colour=INK)
        ax.plot([x0 + 0.023, x1 - 0.025], [y - 0.038, y - 0.038], color=mpl.colors.to_rgba(colour, 0.28), lw=0.6, transform=ax.transAxes)

    join_y = 0.274
    join_x = x1 - 0.026
    for y, regime in zip(lane_y, ["Europe", "China", "Japan"]):
        ax.add_patch(Circle((join_x, y), 0.0045, transform=ax.transAxes, facecolor=REGIME_COLORS[regime], edgecolor=WHITE, lw=0.35, zorder=4))
    ax.plot([join_x, join_x], [lane_y[-1], lane_y[0]], color=CONTROLLED, lw=0.8, transform=ax.transAxes, zorder=3)
    arrow(ax, (join_x, lane_y[-1]), (join_x, join_y + 0.031), colour=CONTROLLED, width=0.8, mutation=5.5)
    ax.add_patch(
        Rectangle(
            (x0 + 0.040, join_y - 0.030),
            x1 - x0 - 0.055,
            0.058,
            transform=ax.transAxes,
            facecolor=WHITE,
            edgecolor=CONTROLLED,
            lw=0.70,
            zorder=3,
        )
    )
    add_text(ax, (x0 + x1) / 2 + 0.013, join_y + 0.010, "conserved 24 h day-block sequencing", size=5.3, colour=CONTROLLED, weight="bold", ha="center")
    add_text(ax, (x0 + x1) / 2 + 0.013, join_y - 0.014, "task · recharge · within-day shape fixed", size=4.7, colour=MUTED, ha="center")

    add_text(ax, x0 + 0.002, 0.194, "Evidence roles remain distinct", size=5.5, colour=INK, weight="bold")
    ax.plot([x0 + 0.002, x1 - 0.002], [0.178, 0.178], color=GRID, lw=0.55, transform=ax.transAxes)
    add_text(ax, x0 + 0.008, 0.145, "natural chronology", size=5.2, colour=NATURAL, weight="bold")
    add_text(ax, x0 + 0.142, 0.145, "occurrence", size=5.2, colour=INK)
    add_text(ax, x0 + 0.008, 0.109, "controlled sequencing", size=5.2, colour=CONTROLLED, weight="bold")
    add_text(ax, x0 + 0.142, 0.109, "mechanism", size=5.2, colour=INK)


def draw_hourly_stream(ax: plt.Axes, x0: float, x1: float, y: float) -> None:
    t = np.linspace(0, 1, 80)
    task = 0.018 * np.maximum(0, np.sin(2 * np.pi * (4.2 * t - 0.1))) * (0.55 + 0.45 * np.sin(2 * np.pi * 0.8 * t) ** 2)
    recharge = -0.016 * np.maximum(0, np.sin(2 * np.pi * (4.2 * t + 0.40)))
    x = x0 + (x1 - x0) * t
    ax.fill_between(x, y, y + task, color=mpl.colors.to_rgba(TASK, 0.60), lw=0, transform=ax.transAxes, zorder=3)
    ax.fill_between(x, y, y + recharge, color=mpl.colors.to_rgba(RECHARGE, 0.60), lw=0, transform=ax.transAxes, zorder=3)
    ax.plot([x0, x1], [y, y], color=MUTED, lw=0.5, transform=ax.transAxes, zorder=4)


def stage_rule(ax: plt.Axes, x0: float, x1: float, y: float) -> None:
    ax.plot([x0, x1], [y, y], color=GRID, lw=0.55, transform=ax.transAxes, zorder=1)


def draw_inventory(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    ax.add_patch(Rectangle((x, y), w, h, transform=ax.transAxes, facecolor=WHITE, edgecolor=CHARCOAL, lw=0.75, zorder=3))
    level = 0.57
    ax.add_patch(Rectangle((x + 0.003, y + 0.003), w - 0.006, (h - 0.006) * level, transform=ax.transAxes, facecolor=mpl.colors.to_rgba(RECOVERY, 0.52), edgecolor="none", zorder=3))
    ax.plot([x - 0.006, x + w + 0.006], [y + h, y + h], color=CHARCOAL, lw=0.8, transform=ax.transAxes, zorder=3)
    arrow(ax, (x - 0.029, y + h * 0.72), (x - 0.002, y + h * 0.72), colour=RECHARGE, width=0.75, mutation=5.5)


def draw_method_panel(ax: plt.Axes) -> None:
    x0, x1 = 0.714, 0.982
    section_header(ax, "c", "Common analysis chain", x0, 0.968)

    # Two incoming chronology definitions.
    add_text(ax, x0 + 0.002, 0.892, "chronology", size=5.3, colour=MUTED, weight="bold")
    ax.add_patch(Rectangle((x0 + 0.002, 0.820), 0.121, 0.048, transform=ax.transAxes, facecolor=NATURAL_PALE, edgecolor=NATURAL, lw=0.65, zorder=2))
    ax.add_patch(Rectangle((x0 + 0.137, 0.820), 0.125, 0.048, transform=ax.transAxes, facecolor=CONTROLLED_PALE, edgecolor=CONTROLLED, lw=0.65, zorder=2))
    add_text(ax, x0 + 0.0625, 0.844, "weather order retained", size=5.0, colour=NATURAL, weight="bold", ha="center")
    add_text(ax, x0 + 0.1995, 0.844, "day blocks reordered", size=5.0, colour=CONTROLLED, weight="bold", ha="center")
    arrow(ax, (x0 + 0.063, 0.820), (x0 + 0.126, 0.769), colour=NATURAL, width=0.75, mutation=5.5)
    arrow(ax, (x0 + 0.199, 0.820), (x0 + 0.137, 0.769), colour=CONTROLLED, width=0.75, mutation=5.5)

    stage_rule(ax, x0, x1, 0.793)
    add_text(ax, x0 + 0.131, 0.756, "paired hourly task and recharge", size=5.7, colour=INK, weight="bold", ha="center")
    draw_hourly_stream(ax, x0 + 0.016, x1 - 0.016, 0.709)
    add_text(ax, x0 + 0.016, 0.681, "task", size=4.9, colour=TASK, weight="bold")
    add_text(ax, x1 - 0.016, 0.681, "recharge", size=4.9, colour=RECHARGE, weight="bold", ha="right")
    arrow(ax, (x0 + 0.134, 0.670), (x0 + 0.134, 0.634), colour=RECOVERY, width=0.9, mutation=6)

    stage_rule(ax, x0, x1, 0.652)
    add_text(ax, x0 + 0.012, 0.610, "recovery window", size=5.8, colour=RECOVERY, weight="bold")
    ax.plot([x0 + 0.012, x0 + 0.101], [0.582, 0.582], color=RECOVERY, lw=0.9, transform=ax.transAxes)
    ax.plot([x0 + 0.012, x0 + 0.012], [0.574, 0.590], color=RECOVERY, lw=0.8, transform=ax.transAxes)
    ax.plot([x0 + 0.101, x0 + 0.101], [0.574, 0.590], color=RECOVERY, lw=0.8, transform=ax.transAxes)
    add_text(ax, x0 + 0.056, 0.556, "time available", size=4.9, colour=MUTED, ha="center")

    arrow(ax, (x0 + 0.112, 0.582), (x0 + 0.156, 0.582), colour=RECOVERY, width=0.85, mutation=6)
    draw_inventory(ax, x0 + 0.172, 0.545, 0.050, 0.071)
    add_text(ax, x0 + 0.197, 0.630, "inventory", size=5.3, colour=INK, weight="bold", ha="center")
    add_text(ax, x0 + 0.197, 0.527, "rebuild", size=4.9, colour=MUTED, ha="center")
    arrow(ax, (x0 + 0.134, 0.512), (x0 + 0.134, 0.477), colour=CHARCOAL, width=0.9, mutation=6)

    stage_rule(ax, x0, x1, 0.495)
    add_text(ax, x0 + 0.008, 0.452, "hourly portfolio dispatch", size=5.8, colour=INK, weight="bold")
    add_text(ax, x0 + 0.008, 0.422, "4 h LFP", size=5.3, colour=BLUE, weight="bold")
    add_text(ax, x0 + 0.058, 0.422, "+", size=5.3, colour=MUTED, weight="bold")
    add_text(ax, x0 + 0.077, 0.422, "candidate LDES", size=5.3, colour=RECOVERY, weight="bold")

    durations_x = [x0 + 0.011, x0 + 0.042, x0 + 0.079, x0 + 0.123]
    durations_h = [0.012, 0.021, 0.032, 0.045]
    for xx, hh in zip(durations_x, durations_h):
        ax.add_patch(Rectangle((xx, 0.353), 0.019, hh, transform=ax.transAxes, facecolor=mpl.colors.to_rgba(RECOVERY, 0.45), edgecolor=RECOVERY, lw=0.45, zorder=3))
    add_text(ax, x0 + 0.076, 0.334, "4–720 h", size=4.9, colour=MUTED, ha="center")

    add_text(ax, x0 + 0.169, 0.386, "stress tests", size=5.1, colour=INK, weight="bold")
    add_text(ax, x0 + 0.169, 0.358, "cluster 1–12 d", size=4.9, colour=CONTROLLED)
    add_text(ax, x0 + 0.169, 0.331, "recharge 1×–4×", size=4.9, colour=RECHARGE)
    add_text(ax, x1 - 0.005, 0.422, "cost years", size=4.9, colour=MUTED, ha="right")
    add_text(ax, x1 - 0.005, 0.395, "2025 · 2030 · 2035", size=5.2, colour=COST, weight="bold", ha="right")
    arrow(ax, (x0 + 0.134, 0.316), (x0 + 0.134, 0.282), colour=CHARCOAL, width=0.9, mutation=6)

    stage_rule(ax, x0, x1, 0.300)
    outputs = [("repeat cycling", RECOVERY), ("duration response", GOLD), ("relative value", PURPLE)]
    ox = [x0 + 0.043, x0 + 0.134, x0 + 0.225]
    for xx, (label, colour) in zip(ox, outputs):
        ax.add_patch(Circle((xx, 0.258), 0.006, transform=ax.transAxes, facecolor=colour, edgecolor=WHITE, lw=0.4, zorder=4))
        add_text(ax, xx, 0.232, label, size=4.7, colour=INK, ha="center")

    arrow(ax, (x0 + 0.134, 0.222), (x0 + 0.134, 0.188), colour=COST, width=0.95, mutation=6.5)
    stage_rule(ax, x0, x1, 0.205)

    # Cost-scale comparison, shown as a compact balance.
    centre = x0 + 0.134
    ax.plot([centre - 0.084, centre + 0.084], [0.148, 0.148], color=CHARCOAL, lw=1.1, transform=ax.transAxes, zorder=3)
    ax.plot([centre, centre], [0.112, 0.148], color=CHARCOAL, lw=1.0, transform=ax.transAxes, zorder=3)
    ax.add_patch(Polygon([[centre - 0.020, 0.108], [centre + 0.020, 0.108], [centre, 0.130]], closed=True, transform=ax.transAxes, facecolor=CHARCOAL, edgecolor="none", zorder=3))
    ax.add_patch(Circle((centre - 0.083, 0.148), 0.012, transform=ax.transAxes, facecolor=CONTROLLED, edgecolor=WHITE, lw=0.55, zorder=4))
    ax.add_patch(Rectangle((centre + 0.072, 0.137), 0.022, 0.022, transform=ax.transAxes, facecolor=COST, edgecolor=WHITE, lw=0.55, zorder=4))
    add_text(ax, centre - 0.083, 0.178, "|ΔV sequencing|", size=5.0, colour=CONTROLLED, weight="bold", ha="center")
    add_text(ax, centre + 0.083, 0.178, "cost learning", size=5.0, colour=COST, weight="bold", ha="center")
    cost_scale = VALUES["technology_cost_scale"]
    add_text(ax, centre + 0.083, 0.119, f"{cost_scale:.2f} EUR kW⁻¹ yr⁻¹", size=4.8, colour=COST, ha="center")

    ax.add_patch(Rectangle((x0 + 0.025, 0.030), x1 - x0 - 0.050, 0.057, transform=ax.transAxes, facecolor=PURPLE_PALE, edgecolor=COST, lw=0.75, zorder=2))
    add_text(ax, centre, 0.059, "SEQUENCING PARITY", size=6.6, colour=COST, weight="bold", ha="center")


def build_figure() -> plt.Figure:
    setup()
    fig = plt.figure(figsize=(WIDTH_MM * MM, HEIGHT_MM * MM))
    ax = fig.add_axes([0.020, 0.026, 0.964, 0.954])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.plot([0.316, 0.316], [0.040, 0.948], color=SOFT_LINE, lw=0.7, transform=ax.transAxes, zorder=0)
    ax.plot([0.701, 0.701], [0.040, 0.948], color=SOFT_LINE, lw=0.7, transform=ax.transAxes, zorder=0)

    draw_problem_panel(ax)
    draw_evidence_panel(ax)
    draw_method_panel(ax)
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
