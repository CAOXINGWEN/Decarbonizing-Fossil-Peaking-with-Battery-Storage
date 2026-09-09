#!/usr/bin/env python3
"""Build the v41 Europe-main figures with an evidence-led asymmetric layout.

The script redraws the five main figures from the frozen v40 Source Data.
It removes the Chinese external-validation plate, replaces the redundant
country-year coverage panel, and presents the central no-valley case as an
inset rather than as a second full-size value envelope.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import (
    Arc,
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
    Wedge,
)
from PIL import Image


HERE = Path(__file__).resolve().parent
MUSCRIPT = HERE.parent
V40 = MUSCRIPT / "revision_nature_energy_v40"
V40_SOURCE = V40 / "source_data/figure_source_data"
V40_RESULTS = V40 / "currency_work"
V40_PLOT = V40 / "plot_v40_figures.py"
EU = HERE.parents[4] / "dataset/external_hourly/europe_mid_duration_valley_upgrade_v0_1"
PANEL = EU / "processed/observed_gas_event_task_definitions_v0_2.csv.gz"
MAP = EU / "raw/gisco_countries_2024/CNTR_RG_20M_2024_4326.geojson"
OUT = HERE / "main_figures"
SOURCE_OUT = HERE / "source_data/figure_source_data"

# Nature double-column target.  The explicit millimetre constant is retained
# so that the delivered size is auditable without opening the rendered file.
FIG_WIDTH_MM = 182.88
WIDTH = FIG_WIDTH_MM / 25.4
INK = "#243746"
MUTED = "#667780"
GRID = "#E2E8EB"
PALE = "#F4F7F8"
DISPERSED = "#355F8D"
CLUSTERED = "#D95F02"
RECHARGE = "#159D89"
PURPLE = "#6B55A3"
RED = "#C44E52"
TECH_ORDER = [
    "li_lfp",
    "vrfb",
    "fecr_flow",
    "gravity",
    "molten_salt_tes",
    "caes",
    "laes",
    "h2_cavern",
]
TECH_COLOR = {
    "li_lfp": "#0072B2",
    "vrfb": "#56B4E9",
    "fecr_flow": "#CC79A7",
    "gravity": "#767676",
    "molten_salt_tes": "#E69F00",
    "caes": "#009E73",
    "laes": "#D55E00",
    "h2_cavern": "#6A51A3",
}
TECH_LABEL = {
    "li_lfp": "LFP",
    "vrfb": "VRFB",
    "fecr_flow": "Fe–Cr flow",
    "gravity": "Gravity",
    "molten_salt_tes": "Thermal",
    "caes": "A-CAES",
    "laes": "LAES",
    "h2_cavern": "Cavern H$_2$",
}
CONFIG_LABEL = {
    "li_lfp_4h": "LFP 4 h",
    "caes_24h": "A-CAES 24 h",
    "h2_cavern_48h": "Cavern H$_2$ 48 h",
    "h2_cavern_168h": "Cavern H$_2$ 168 h",
}


def load_v40():
    spec = importlib.util.spec_from_file_location("plot_v40", V40_PLOT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {V40_PLOT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V40_MODULE = load_v40()
RUNNER = V40_MODULE.load_runner()


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6.3,
            "axes.labelsize": 6.5,
            "axes.titlesize": 6.8,
            "xtick.labelsize": 5.7,
            "ytick.labelsize": 5.7,
            "legend.fontsize": 5.5,
            "axes.linewidth": 0.55,
            "lines.linewidth": 1.15,
            "lines.markersize": 3.2,
            "xtick.major.size": 2.3,
            "ytick.major.size": 2.3,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
        }
    )


def export(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches=None)
    fig.savefig(OUT / f"{stem}.svg", bbox_inches=None)
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches=None)
    fig.savefig(
        OUT / f"{stem}.tiff",
        dpi=600,
        bbox_inches=None,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    with Image.open(OUT / f"{stem}.png") as image:
        image.convert("L").save(OUT / f"{stem}_grayscale.png", dpi=(300, 300))
    plt.close(fig)


def save_source(frame: pd.DataFrame, filename: str) -> None:
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(SOURCE_OUT / filename, index=False)


def require_strictly_positive(values, label: str) -> None:
    """Reject non-positive values before any logarithmic transform or axis."""
    array = np.asarray(values, dtype=float)
    if np.any(array <= 0):
        raise ValueError(f"{label} must be strictly positive for a log scale")


def panel_labels(fig: plt.Figure, axes: list[plt.Axes]) -> None:
    fig.canvas.draw()
    for label, ax in zip("abcdefghijklmnopqrstuvwxyz", axes):
        box = ax.get_position()
        fig.text(
            box.x0 - 0.012,
            box.y1 + 0.008,
            label,
            ha="right",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="#111111",
        )


def duration_axis(ax: plt.Axes, maximum: int = 720) -> None:
    ticks = [4, 8, 12, 24, 48, 96, 168]
    if maximum > 168:
        ticks += [336, 720]
    require_strictly_positive([3.7, maximum * 1.08, *ticks], "duration axis")
    ax.set_xscale("log")
    ax.set_xlim(3.7, maximum * 1.08)
    ax.set_xticks([value for value in ticks if value <= maximum])
    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())


def rounded(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    face: str,
    edge: str = "none",
    linewidth: float = 0.6,
    radius: float = 0.012,
    zorder: int = 2,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        transform=ax.transAxes,
        boxstyle=f"round,pad=0.002,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = MUTED,
    linewidth: float = 0.8,
    mutation: float = 7,
    connection: str = "arc3",
    zorder: int = 8,
) -> FancyArrowPatch:
    patch = FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=mutation,
        linewidth=linewidth,
        color=color,
        connectionstyle=connection,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def event_strip(
    ax: plt.Axes,
    x0: float,
    x1: float,
    y: float,
    clustered: bool,
) -> None:
    heights = np.array([0.050, 0.065, 0.052, 0.072, 0.061, 0.068, 0.055])
    fractions = (
        np.array([0.04, 0.195, 0.35, 0.505, 0.66, 0.815, 0.96])
        if not clustered
        else np.array([0.40, 0.485, 0.565, 0.645, 0.725, 0.805, 0.885])
    )
    width = x1 - x0
    xs = x0 + fractions * width
    ax.plot([x0, x1], [y, y], transform=ax.transAxes, color=GRID, lw=0.8)
    bar_width = 0.018
    for x, height in zip(xs, heights):
        rounded(ax, (x - bar_width / 2, y), bar_width, height, "#D99023", radius=0.003, zorder=5)
    for left, right in zip(xs[:-1], xs[1:]):
        gap = right - left - bar_width
        if gap < 0.006:
            continue
        xx = np.linspace(left + bar_width / 2, right - bar_width / 2, 40)
        yy = y - 0.014 + min(0.018, gap * 0.28) * np.sin(np.linspace(0, np.pi, 40))
        ax.plot(xx, yy, transform=ax.transAxes, color=RECHARGE, lw=1.25, solid_capstyle="round")


def retention_ring(
    ax: plt.Axes,
    x: float,
    y: float,
    value: float,
    color: str,
    text: str,
) -> None:
    radius = 0.030
    ax.add_patch(
        Wedge(
            (x, y), radius, 90, 450, width=0.007,
            transform=ax.transAxes, facecolor=GRID, edgecolor="none", zorder=4,
        )
    )
    ax.add_patch(
        Wedge(
            (x, y), radius, 90, 90 + 360 * value, width=0.007,
            transform=ax.transAxes, facecolor=color, edgecolor="none", zorder=5,
        )
    )
    ax.text(
        x, y, f"{100 * value:.0f}%",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=7.2, fontweight="bold", color=color,
    )
    ax.text(
        x, y - 0.048, text,
        transform=ax.transAxes, ha="center", va="top",
        fontsize=5.4, color=MUTED,
    )


def paired_gauge(
    ax: plt.Axes,
    x: float,
    y: float,
    clustered_fraction: float,
    text: str,
) -> None:
    """Draw a qualitative dispersed-versus-clustered response gauge."""
    left = x - 0.034
    right = x + 0.034
    for offset, label, fraction, colour in [
        (0.016, "D", 1.0, DISPERSED),
        (-0.016, "C", clustered_fraction, CLUSTERED),
    ]:
        ax.plot(
            [left, right],
            [y + offset, y + offset],
            transform=ax.transAxes,
            color="#D8E0E4",
            lw=3.1,
            solid_capstyle="round",
            zorder=5,
        )
        ax.plot(
            [left, left + fraction * (right - left)],
            [y + offset, y + offset],
            transform=ax.transAxes,
            color=colour,
            lw=3.1,
            solid_capstyle="round",
            zorder=6,
        )
        ax.text(
            left - 0.008,
            y + offset,
            label,
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=5.0,
            fontweight="bold",
            color=colour,
            zorder=7,
        )
    ax.text(
        x,
        y - 0.048,
        text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=5.4,
        color=MUTED,
        zorder=9,
    )


def figure1() -> None:
    """A continuous scientific hypothesis figure modelled on the v35 grammar."""
    fig, ax = plt.subplots(figsize=(WIDTH, 4.05))
    fig.subplots_adjust(left=0.012, right=0.992, bottom=0.025, top=0.985)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # One continuous, shallow background field rather than a dashboard frame.
    cloud = Polygon(
        [
            (0.018, 0.92), (0.25, 0.95), (0.49, 0.90), (0.69, 0.80),
            (0.97, 0.86), (0.97, 0.31), (0.72, 0.30), (0.47, 0.39),
            (0.20, 0.40), (0.04, 0.37),
        ],
        closed=True,
        transform=ax.transAxes,
        facecolor="#F3F7F7",
        edgecolor="none",
        zorder=0,
    )
    ax.add_patch(cloud)

    # Paired chronology intervention.
    ax.text(0.035, 0.905, "same daily blocks and annual energy", transform=ax.transAxes,
            fontsize=6.2, fontweight="bold", color=MUTED)
    ax.text(0.035, 0.825, "dispersed", transform=ax.transAxes,
            fontsize=6.5, fontweight="bold", color=DISPERSED)
    event_strip(ax, 0.035, 0.330, 0.755, clustered=False)
    ax.text(0.035, 0.605, "clustered", transform=ax.transAxes,
            fontsize=6.5, fontweight="bold", color=CLUSTERED)
    event_strip(ax, 0.035, 0.330, 0.535, clustered=True)
    arrow(ax, (0.343, 0.752), (0.343, 0.590), CLUSTERED, 1.0, 8)
    ax.text(0.352, 0.670, "reorder only", transform=ax.transAxes,
            fontsize=5.5, fontweight="bold", color=CLUSTERED, rotation=90,
            ha="center", va="center")
    ax.text(0.165, 0.465, "recharge intervals contract", transform=ax.transAxes,
            ha="center", fontsize=5.7, fontweight="bold", color=RECHARGE)

    # Realized response across a technology family.
    arrow(ax, (0.370, 0.680), (0.405, 0.680), MUTED)
    ax.text(0.405, 0.885, "8 storage families", transform=ax.transAxes,
            fontsize=6.5, fontweight="bold", color=INK)
    ax.text(0.405, 0.852, "realized use under the same event energy", transform=ax.transAxes,
            fontsize=5.6, color=MUTED)

    # Compact LFP and storage-family glyphs.
    rounded(ax, (0.415, 0.685), 0.070, 0.078, "#DDEAF3", TECH_COLOR["li_lfp"], 0.8, 0.008, 6)
    for idx in range(4):
        ax.add_patch(
            Rectangle(
                (0.424 + idx * 0.014, 0.697), 0.011, 0.054,
                transform=ax.transAxes,
                facecolor=TECH_COLOR["li_lfp"] if idx < 3 else "#DDEAF3",
                edgecolor="none",
                zorder=7,
            )
        )
    ax.text(0.450, 0.655, "4 h LFP", transform=ax.transAxes,
            ha="center", fontsize=5.8, fontweight="bold", color=TECH_COLOR["li_lfp"])
    paired_gauge(ax, 0.545, 0.727, 0.70, "realized cycles")

    rounded(ax, (0.395, 0.462), 0.198, 0.150, "white", GRID, 0.7, 0.012, 4)
    glyphs = [
        (0.421, 0.550, "VRFB", TECH_COLOR["vrfb"]),
        (0.469, 0.550, "Fe–Cr", TECH_COLOR["fecr_flow"]),
        (0.517, 0.550, "Gravity", TECH_COLOR["gravity"]),
        (0.565, 0.550, "TES", TECH_COLOR["molten_salt_tes"]),
        (0.445, 0.500, "A-CAES", TECH_COLOR["caes"]),
        (0.493, 0.500, "LAES", TECH_COLOR["laes"]),
        (0.541, 0.500, "H$_2$", TECH_COLOR["h2_cavern"]),
    ]
    for x, y, label, color in glyphs:
        rounded(
            ax,
            (x - 0.021, y - 0.013),
            0.042,
            0.026,
            mpl.colors.to_rgba(color, 0.10),
            edge=color,
            linewidth=0.60,
            radius=0.005,
            zorder=6,
        )
        ax.text(
            x,
            y,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=5.0,
            fontweight="bold",
            color=color,
            zorder=7,
        )
    ax.text(0.493, 0.622, "7 long-duration families", transform=ax.transAxes,
            ha="center", fontsize=5.7, fontweight="bold", color=INK)
    paired_gauge(ax, 0.635, 0.535, 0.34, "task-firm supply")

    # Strict valley hypothesis as the dominant visual target.
    arrow(ax, (0.665, 0.655), (0.695, 0.655), MUTED)
    x = np.linspace(0.705, 0.965, 220)
    u = (x - x.min()) / (x.max() - x.min())
    dispersed = 0.69 - 0.20 * u + 0.05 * np.sin(1.4 * np.pi * u)
    clustered = (
        0.62 - 0.25 * u
        - 0.12 * np.exp(-((u - 0.49) / 0.16) ** 2)
        + 0.18 / (1 + np.exp(-(u - 0.76) * 22))
    )
    zero = 0.475
    ax.plot(x, dispersed, transform=ax.transAxes, color=DISPERSED, lw=1.65, zorder=6)
    ax.plot(x, clustered, transform=ax.transAxes, color=CLUSTERED, lw=1.65, zorder=6)
    ax.plot([0.700, 0.970], [zero, zero], transform=ax.transAxes, color="#7D858A", lw=0.75, zorder=3)
    ax.text(0.815, 0.670, "dispersed", transform=ax.transAxes,
            fontsize=5.4, fontweight="bold", color=DISPERSED)
    ax.text(0.718, 0.620, "clustered", transform=ax.transAxes,
            fontsize=5.4, fontweight="bold", color=CLUSTERED)
    ax.text(0.968, zero + 0.010, "zero net value", transform=ax.transAxes,
            ha="right", fontsize=5.1, color=MUTED)
    left, right = 0.790, 0.875
    rounded(ax, (left, 0.385), right - left, 0.050, "#E8ECEE", GRID, 0.5, 0.004, 3)
    rounded(ax, (0.705, 0.385), left - 0.705, 0.050, "#DCECE8", radius=0.008, zorder=3)
    rounded(ax, (right, 0.385), 0.965 - right, 0.050, "#E9E5F4", radius=0.008, zorder=3)
    ax.text((0.705 + left) / 2, 0.410, "short", transform=ax.transAxes,
            ha="center", va="center", fontsize=5.2, fontweight="bold", color=TECH_COLOR["caes"])
    ax.text((left + right) / 2, 0.410, "mid-duration\nvalley", transform=ax.transAxes,
            ha="center", va="center", fontsize=5.2, fontweight="bold", color=INK, linespacing=0.9)
    ax.text((right + 0.965) / 2, 0.410, "long", transform=ax.transAxes,
            ha="center", va="center", fontsize=5.2, fontweight="bold", color=PURPLE)
    arrow(ax, (0.705, 0.350), (0.968, 0.350), INK, 0.75, 6)
    ax.text(0.835, 0.320, "discharge duration", transform=ax.transAxes,
            ha="center", fontsize=5.5, fontweight="bold", color=INK)

    # Compact evidence ribbon.
    rounded(ax, (0.035, 0.075), 0.930, 0.160, "white", GRID, 0.6, 0.016, 1)
    nodes = [
        (0.130, "15 countries", "1,156,320 h", "#2685A7"),
        (0.355, "132 country-years", "paired chronologies", DISPERSED),
        (0.590, "53 configurations", "4–720 h", TECH_COLOR["caes"]),
        (0.825, "36 value tests", "joint valley criterion", PURPLE),
    ]
    for idx, (xnode, title, subtitle, color) in enumerate(nodes):
        ax.add_patch(
            Circle((xnode, 0.155), 0.028, transform=ax.transAxes,
                   facecolor=mpl.colors.to_rgba(color, 0.13),
                   edgecolor=color, lw=0.8, zorder=5)
        )
        if idx == 0:
            ax.add_patch(
                Circle(
                    (xnode, 0.155),
                    0.017,
                    transform=ax.transAxes,
                    facecolor="white",
                    edgecolor=color,
                    lw=0.65,
                    zorder=7,
                )
            )
            ax.add_patch(
                Arc(
                    (xnode, 0.155),
                    0.018,
                    0.034,
                    theta1=0,
                    theta2=360,
                    transform=ax.transAxes,
                    color=color,
                    lw=0.5,
                    zorder=8,
                )
            )
            ax.plot(
                [xnode - 0.015, xnode + 0.015],
                [0.155, 0.155],
                transform=ax.transAxes,
                color=color,
                lw=0.5,
                zorder=8,
            )
        elif idx == 1:
            arrow(
                ax,
                (xnode - 0.015, 0.163),
                (xnode + 0.015, 0.147),
                CLUSTERED,
                0.65,
                4,
            )
            arrow(
                ax,
                (xnode - 0.015, 0.147),
                (xnode + 0.015, 0.163),
                RECHARGE,
                0.65,
                4,
            )
        elif idx == 2:
            for dx, height in zip(
                [-0.014, -0.004, 0.006, 0.016],
                [0.012, 0.019, 0.026, 0.034],
            ):
                ax.add_patch(
                    Rectangle(
                        (xnode + dx - 0.003, 0.138),
                        0.006,
                        height,
                        transform=ax.transAxes,
                        facecolor=color,
                        edgecolor="none",
                        zorder=7,
                    )
                )
        else:
            for dx, outcome_color in zip(
                [-0.014, 0, 0.014],
                [TECH_COLOR["molten_salt_tes"], RECHARGE, PURPLE],
            ):
                ax.add_patch(
                    Circle(
                        (xnode + dx, 0.155),
                        0.008,
                        transform=ax.transAxes,
                        facecolor="white",
                        edgecolor=outcome_color,
                        lw=0.60,
                        zorder=7,
                    )
                )
        ax.text(xnode, 0.113, title, transform=ax.transAxes,
                ha="center", va="top", fontsize=5.6, fontweight="bold", color=INK)
        ax.text(xnode, 0.088, subtitle, transform=ax.transAxes,
                ha="center", va="top", fontsize=5.0, color=MUTED)
        if idx < len(nodes) - 1:
            arrow(ax, (xnode + 0.036, 0.155), (nodes[idx + 1][0] - 0.036, 0.155), "#AAB6BC", 0.7, 5)

    export(fig, "fig_v41_figure1_framework")


def map_event_geometry(ax: plt.Axes, stats: pd.DataFrame) -> None:
    with MAP.open(encoding="utf-8") as stream:
        geo = json.load(stream)
    medians = stats.groupby("country")["natural_longest_event_run_days"].median().to_dict()
    code_map = {"GB": "UK", "GR": "EL"}
    medians = {code_map.get(key, key): value for key, value in medians.items()}
    norm = mpl.colors.Normalize(vmin=4, vmax=15)
    cmap = mpl.colormaps["YlOrBr"]
    background: list[Polygon] = []
    selected: list[Polygon] = []
    values: list[float] = []
    for feature in geo["features"]:
        code = feature["properties"].get("CNTR_ID")
        geometry = feature.get("geometry")
        if not geometry:
            continue
        polygons = (
            [geometry["coordinates"]]
            if geometry["type"] == "Polygon"
            else geometry["coordinates"]
            if geometry["type"] == "MultiPolygon"
            else []
        )
        for polygon in polygons:
            if not polygon:
                continue
            outer = np.asarray(polygon[0], dtype=float)
            if outer.ndim != 2 or outer.shape[1] != 2:
                continue
            if outer[:, 0].max() < -13 or outer[:, 0].min() > 32:
                continue
            if outer[:, 1].max() < 34 or outer[:, 1].min() > 72:
                continue
            patch = Polygon(outer, closed=True)
            if code in medians:
                selected.append(patch)
                values.append(float(medians[code]))
            else:
                background.append(patch)
    ax.add_collection(
        PatchCollection(background, facecolor="#F1F3F4", edgecolor="white", linewidth=0.25)
    )
    collection = PatchCollection(
        selected, cmap=cmap, norm=norm, edgecolor="white", linewidth=0.45
    )
    collection.set_array(np.asarray(values))
    ax.add_collection(collection)
    ax.set_xlim(-12, 31)
    ax.set_ylim(35, 71.5)
    ax.set_aspect(1.13)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    cax = ax.inset_axes([0.06, -0.01, 0.50, 0.027])
    mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation="horizontal")
    cax.set_xlabel("Median longest natural event (d)", labelpad=1)
    cax.tick_params(labelsize=5, length=1.5)


def longest_run(flag: np.ndarray) -> tuple[int, int, int]:
    best = current = start = best_start = 0
    for index, value in enumerate(flag):
        if value:
            if current == 0:
                start = index
            current += 1
            if current > best:
                best = current
                best_start = start
        else:
            current = 0
    return best, best_start, best_start + best


def figure2() -> None:
    daily = pd.read_csv(V40_SOURCE / "europe_daily_task_recharge.csv")
    stats = pd.read_csv(V40_SOURCE / "europe_country_year_event_geometry.csv")
    chronology = pd.read_csv(V40_SOURCE / "figure2_panel_c_germany_2023_chronologies.csv")
    fig = plt.figure(figsize=(WIDTH, 5.55))
    fig.subplots_adjust(left=0.072, right=0.985, bottom=0.083, top=0.965)
    gs = fig.add_gridspec(
        3, 12,
        height_ratios=[1.0, 0.82, 0.82],
        hspace=0.62,
        wspace=0.90,
    )
    axa = fig.add_subplot(gs[0:2, 0:5])
    axb = fig.add_subplot(gs[0, 5:12])
    axc = fig.add_subplot(gs[1, 5:12])
    axd = fig.add_subplot(gs[2, 0:4])
    axe = fig.add_subplot(gs[2, 4:8])
    axf = fig.add_subplot(gs[2, 8:12])
    axes = [axa, axb, axc, axd, axe, axf]

    map_event_geometry(axa, stats)

    # A matched natural-year contrast with similar annual task and recharge.
    ro = daily[daily["country"].eq("RO") & daily["year"].isin([2023, 2024])]
    for row_index, year in enumerate([2023, 2024]):
        group = ro[ro["year"].eq(year)].sort_values("source_day_id")
        flag = group["event"].to_numpy(bool)
        task = group["task_mwh"].to_numpy(float)
        task = task / max(task.max(), 1)
        y0 = 1 - row_index
        axb.fill_between(np.arange(365), y0, y0 + 0.42 * task, color="#D99023", linewidth=0)
        run, start, stop = longest_run(flag)
        axb.add_patch(
            Rectangle(
                (start, y0 - 0.07), stop - start, 0.57,
                fill=False, edgecolor=RED, linewidth=0.9,
            )
        )
        info = stats[(stats["country"].eq("RO")) & (stats["year"].eq(year))].iloc[0]
        axb.text(
            368, y0 + 0.20,
            f"{year}  {info['annual_task_twh']:.2f} TWh · {run} d",
            ha="left", va="center", fontsize=5.5, color=INK,
        )
    axb.set_xlim(0, 470)
    axb.set_ylim(-0.10, 1.55)
    axb.set_yticks([])
    axb.set_xticks([0, 90, 181, 273, 364])
    axb.set_xticklabels(["Jan", "Apr", "Jul", "Oct", "Dec"])
    axb.set_xlabel("Natural chronology within the year")
    axb.text(
        0.01, 0.98,
        "Romania: nearly matched annual task and recharge, different event runs",
        transform=axb.transAxes, va="top", fontsize=5.7, fontweight="bold", color=MUTED,
    )

    days = chronology["day"].to_numpy()
    for chronology_name, base, color in [
        ("Dispersed", 1.0, DISPERSED),
        ("Clustered", 0.0, CLUSTERED),
    ]:
        task = chronology[f"{chronology_name.lower()}_task_normalized"].to_numpy()
        recharge = chronology[f"{chronology_name.lower()}_recharge_normalized"].to_numpy()
        axc.fill_between(days, base, base + 0.43 * task, color=color, alpha=0.86, linewidth=0)
        axc.plot(days, base - 0.26 * recharge, color=RECHARGE, lw=0.65)
        axc.text(-8, base + 0.15, chronology_name, ha="right", va="center",
                 color=color, fontsize=5.7, fontweight="bold")
    axc.set_xlim(-30, 365)
    axc.set_ylim(-0.33, 1.50)
    axc.set_yticks([])
    axc.set_xticks([0, 90, 181, 273, 364])
    axc.set_xticklabels(["Jan", "Apr", "Jul", "Oct", "Dec"])
    axc.set_xlabel("Conserved reordering of identical German daily blocks")
    axc.text(0.01, 0.95, "task", transform=axc.transAxes, color=CLUSTERED,
             fontsize=5.3, fontweight="bold")
    axc.text(0.09, 0.95, "potential recharge", transform=axc.transAxes, color=RECHARGE,
             fontsize=5.3, fontweight="bold")

    x = stats["annual_task_twh"].to_numpy()
    y = stats["natural_longest_event_run_days"].to_numpy()
    recharge_ratios = stats["recharge_to_task_ratio"].to_numpy()
    require_strictly_positive(x, "annual peaking task")
    require_strictly_positive(recharge_ratios, "recharge-to-task ratio")
    colours = np.log10(recharge_ratios)
    axd.scatter(x, y, c=colours, cmap="viridis", s=15, alpha=0.72,
                edgecolor="white", linewidth=0.25)
    axd.set_xscale("log")
    rho = stats["annual_task_twh"].corr(
        stats["natural_longest_event_run_days"], method="spearman"
    )
    axd.text(0.05, 0.94, f"Spearman $\\rho$ = {rho:.2f}",
             transform=axd.transAxes, va="top", fontsize=5.4)
    axd.set_xlabel("Annual peaking task (TWh)")
    axd.set_ylabel("Longest natural event (d)")

    values = np.sort(stats["natural_longest_event_run_days"].to_numpy())
    ecdf = np.arange(1, len(values) + 1) / len(values)
    axe.step(values, ecdf, where="post", color="#D99023")
    axe.axvline(np.median(values), color=INK, ls="--", lw=0.7)
    axe.text(np.median(values) + 0.5, 0.12, f"median {np.median(values):.1f} d",
             fontsize=5.2, color=INK)
    axe.set_xlabel("Longest natural event (d)")
    axe.set_ylabel("Cumulative country-years")
    axe.set_ylim(0, 1.02)

    ratio = np.sort(recharge_ratios)
    ratio_ecdf = np.arange(1, len(ratio) + 1) / len(ratio)
    axf.step(ratio, ratio_ecdf, where="post", color=RECHARGE)
    axf.set_xscale("log")
    axf.axvline(np.median(ratio), color=INK, ls="--", lw=0.7)
    axf.text(np.median(ratio) * 1.12, 0.12, f"median {np.median(ratio):.1f}",
             fontsize=5.2, color=INK)
    axf.set_xlabel("Annual recharge / peaking task")
    axf.set_ylabel("Cumulative country-years")
    axf.set_ylim(0, 1.02)

    panel_labels(fig, axes)
    export(fig, "fig_v41_figure2_europe_data_geometry")
    save_source(ro, "figure2_panel_b_romania_matched_years.csv")
    save_source(chronology, "figure2_panel_c_conserved_chronology.csv")
    save_source(stats, "figure2_panel_adef_event_geometry.csv")


def figure3() -> None:
    summary = pd.read_csv(V40_SOURCE / "figure3_panel_a_configuration_summary.csv")
    selected = pd.read_csv(V40_SOURCE / "figure3_panel_c_selected_configuration_changes.csv")
    mean_cycles = pd.read_csv(V40_SOURCE / "figure3_panel_d_mean_cycles.csv")
    physical = pd.read_csv(V40_SOURCE / "figure3_panel_e_physical_effects.csv")
    benefit = pd.read_csv(V40_SOURCE / "figure3_panel_f_net_benefit.csv")
    fig = plt.figure(figsize=(WIDTH, 5.15))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.092, top=0.965)
    gs = fig.add_gridspec(2, 12, height_ratios=[1.2, 1.0], hspace=0.56, wspace=0.95)
    axa = fig.add_subplot(gs[0, 0:8])
    axb = fig.add_subplot(gs[0, 8:12])
    axc = fig.add_subplot(gs[1, 0:3])
    axd = fig.add_subplot(gs[1, 3:7])
    axe = fig.add_subplot(gs[1, 7:12])
    axes = [axa, axb, axc, axd, axe]

    for tech in TECH_ORDER:
        group = summary[summary["tech_id"].eq(tech)].sort_values("duration_h")
        if group.empty:
            continue
        axa.fill_between(
            group["duration_h"],
            100 * group["retained_equivalent_full_cycles_ci_low"],
            100 * group["retained_equivalent_full_cycles_ci_high"],
            color=TECH_COLOR[tech], alpha=0.12, linewidth=0,
        )
        axa.plot(
            group["duration_h"],
            100 * group["retained_equivalent_full_cycles_mean"],
            color=TECH_COLOR[tech], marker="o", ms=2.7, label=TECH_LABEL[tech],
        )
    duration_axis(axa, 720)
    axa.axhline(100, color="#7B858A", lw=0.65, ls="--")
    axa.set_ylim(20, 104)
    axa.set_ylabel("Equivalent-full-cycle retention (%)")
    axa.set_xlabel("Discharge duration (h)")
    axa.legend(loc="upper left", ncol=4, fontsize=5.1, columnspacing=0.8, handletextpad=0.3)
    axa.text(
        0.99, 0.06, "6,814 / 6,996 paired trajectories lost cycles",
        transform=axa.transAxes, ha="right", fontsize=5.5, fontweight="bold", color=MUTED,
    )

    for tech in TECH_ORDER:
        group = summary[summary["tech_id"].eq(tech)].sort_values("duration_h")
        if group.empty:
            continue
        axb.plot(
            group["duration_h"],
            100 * group["delta_efc_ratio_mean"],
            color=TECH_COLOR[tech], marker="o", ms=2.4,
        )
    duration_axis(axb, 720)
    axb.axhline(0, color="#777777", lw=0.65)
    axb.set_ylabel("$\\Delta$ task-firm-supply ratio (pp)")
    axb.set_xlabel("Discharge duration (h)")

    configs = list(CONFIG_LABEL)
    positions = np.arange(len(configs))
    data = [
        selected[selected["config_id"].eq(config)]["retained_equivalent_full_cycles"].to_numpy() * 100
        for config in configs
    ]
    violin = axc.violinplot(data, positions=positions, widths=0.72, showextrema=False)
    for body, config in zip(violin["bodies"], configs):
        tech = selected[selected["config_id"].eq(config)]["tech_id"].iloc[0]
        body.set_facecolor(TECH_COLOR[tech])
        body.set_edgecolor("none")
        body.set_alpha(0.42)
    for position, values in zip(positions, data):
        jitter = np.linspace(-0.09, 0.09, len(values))
        axc.scatter(np.full(len(values), position) + jitter, values, s=3, color="#333333", alpha=0.11)
        axc.plot([position - 0.17, position + 0.17], [np.median(values)] * 2, color=INK, lw=1.4)
    axc.set_xticks(positions)
    axc.set_xticklabels([CONFIG_LABEL[c] for c in configs], rotation=35, ha="right")
    axc.set_ylabel("Cycle retention (%)")

    wide = mean_cycles.pivot(index="config_id", columns="chronology", values="equivalent_full_cycles")
    wide = wide.loc[configs]
    y = np.arange(len(wide))
    for index, (config, row) in enumerate(wide.iterrows()):
        axd.plot([row["dispersed"], row["clustered"]], [index, index], color="#B8C0C4", lw=1.1)
        axd.scatter(row["dispersed"], index, color=DISPERSED, s=20, zorder=3)
        axd.scatter(row["clustered"], index, color=CLUSTERED, s=20, zorder=3)
    axd.set_yticks(y)
    axd.set_yticklabels([CONFIG_LABEL[c] for c in wide.index])
    axd.invert_yaxis()
    axd.set_xlabel("Equivalent full cycles yr$^{-1}$")
    axd.scatter([], [], color=DISPERSED, label="Dispersed")
    axd.scatter([], [], color=CLUSTERED, label="Clustered")
    axd.legend(loc="lower right")

    for tech in TECH_ORDER:
        group = benefit[benefit["tech_id"].eq(tech)].sort_values("duration_h")
        if group.empty:
            continue
        axe.fill_between(group["duration_h"], group["low"], group["high"],
                         color=TECH_COLOR[tech], alpha=0.10, linewidth=0)
        axe.plot(group["duration_h"], group["mean"], color=TECH_COLOR[tech],
                 marker="o", ms=2.2)
    duration_axis(axe, 720)
    axe.axhline(0, color="#777777", lw=0.65)
    axe.set_xlabel("Discharge duration (h)")
    axe.set_ylabel("$\\Delta$ annual net benefit\n(EUR kW$^{-1}$ yr$^{-1}$)")
    axe.text(0.98, 0.95, "2035 cost snapshot", transform=axe.transAxes,
             ha="right", va="top", fontsize=5.4, color=MUTED)

    panel_labels(fig, axes)
    export(fig, "fig_v41_figure3_duration_spectrum")
    save_source(summary, "figure3_panel_ab_configuration_summary.csv")
    save_source(selected, "figure3_panel_c_selected_distributions.csv")
    save_source(mean_cycles, "figure3_panel_d_mean_cycles.csv")
    save_source(physical, "figure3_panel_d_task_effects.csv")
    save_source(benefit, "figure3_panel_e_net_benefit.csv")


def build_mechanism_window() -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = pd.read_csv(PANEL)
    results = pd.read_csv(V40_RESULTS / "configuration_results_source_aware_eur.csv.gz").rename(
        columns={
            "annual_fixed_cost_eur_per_kw_year_source_aware": "annual_fixed_cost_eur_per_kw_year",
            "storage_vom_eur_per_kw_year_source_aware": "storage_vom_eur_per_kw_year",
            "turnover_net_benefit_eur_per_kw_year_source_aware": "turnover_net_benefit_eur_per_kw_year",
            "joint_net_benefit_eur_per_kw_year_source_aware": "joint_net_benefit_eur_per_kw_year",
        }
    )
    example = panel[(panel["country"].eq("DE")) & (panel["year"].eq(2023))].sort_values("source_hour_id")
    orders, _ = RUNNER.day_orders(example)
    task0 = example["task_c3_q75_mw"].to_numpy(float)
    recharge0 = example["r2_potential_recharge_mw"].to_numpy(float)
    clustered_task = RUNNER.reorder_days(task0, orders["clustered"])
    clustered_recharge = RUNNER.reorder_days(recharge0, orders["clustered"])
    daily_task = clustered_task.reshape(365, 24).sum(axis=1)
    daily_recharge = clustered_recharge.reshape(365, 24).sum(axis=1)
    window = 45
    start = int(np.argmax(np.convolve(daily_task, np.ones(window), mode="valid")))
    stop = start + window
    source = pd.DataFrame(
        {
            "window_day": np.arange(window),
            "task_normalized": daily_task[start:stop] / max(daily_task[start:stop].max(), 1),
            "recharge_normalized": daily_recharge[start:stop] / max(daily_recharge[start:stop].max(), 1),
        }
    )
    pdis = float(example["forced_discharge_power_mw"].iloc[0])
    configs = results.drop_duplicates("config_id").set_index("config_id")
    traces: list[pd.DataFrame] = []
    for config in ["li_lfp_4h", "caes_24h"]:
        row = configs.loc[config]
        for chronology in ["dispersed", "clustered"]:
            task = RUNNER.reorder_days(task0, orders[chronology])
            recharge = RUNNER.reorder_days(recharge0, orders[chronology])
            trace = V40_MODULE.make_soc_trace(
                task,
                recharge,
                float(row["duration_h"]),
                float(row["rte"]),
                float(row["self_discharge_pct_day"]),
                float(row["charge_power_ratio"]),
                pdis,
            )
            daily_min = pd.Series(trace[:-1]).groupby(np.arange(8760) // 24).min().to_numpy()
            daily_max = pd.Series(trace[:-1]).groupby(np.arange(8760) // 24).max().to_numpy()
            traces.append(
                pd.DataFrame(
                    {
                        "window_day": np.arange(window),
                        "config_id": config,
                        "chronology": chronology,
                        "soc_min": daily_min[start:stop],
                        "soc_mid": 0.5 * (daily_min[start:stop] + daily_max[start:stop]),
                        "soc_max": daily_max[start:stop],
                    }
                )
            )
    return source, pd.concat(traces, ignore_index=True)


def figure4() -> None:
    window, traces = build_mechanism_window()
    retention = pd.read_csv(V40_SOURCE / "figure4_panel_c_retention_summary.csv")
    saturation = pd.read_csv(V40_SOURCE / "figure4_panel_d_charge_saturation_cycles.csv")
    countries = pd.read_csv(V40_SOURCE / "figure4_panel_e_country_relative_value.csv")
    link = pd.read_csv(V40_SOURCE / "figure4_panel_f_charge_retention_value_link.csv")

    fig = plt.figure(figsize=(WIDTH, 5.30))
    fig.subplots_adjust(left=0.082, right=0.985, bottom=0.092, top=0.965)
    gs = fig.add_gridspec(2, 12, height_ratios=[1.25, 1.0], hspace=0.57, wspace=1.24)
    axa = fig.add_subplot(gs[0, 0:8])
    axb = fig.add_subplot(gs[0, 8:12])
    axc = fig.add_subplot(gs[1, 0:4])
    axd = fig.add_subplot(gs[1, 4:8])
    axe = fig.add_subplot(gs[1, 8:12])
    axes = [axa, axb, axc, axd, axe]

    # Hero: task/recharge and SOC share one horizontal time axis.
    days = window["window_day"].to_numpy()
    axa.bar(days, window["task_normalized"], width=0.82, color=CLUSTERED, alpha=0.83)
    axa.fill_between(days, 0, -0.52 * window["recharge_normalized"],
                     color=RECHARGE, alpha=0.30, linewidth=0)
    axa.axhline(0, color="#AAB4B9", lw=0.6)
    for config, offset, color in [
        ("li_lfp_4h", 1.24, TECH_COLOR["li_lfp"]),
        ("caes_24h", 2.02, TECH_COLOR["caes"]),
    ]:
        for chronology, linestyle, alpha in [
            ("dispersed", "--", 0.88),
            ("clustered", "-", 1.0),
        ]:
            group = traces[
                traces["config_id"].eq(config) & traces["chronology"].eq(chronology)
            ]
            axa.plot(days, offset + 0.58 * group["soc_mid"].to_numpy(),
                     color=color, ls=linestyle, lw=1.1, alpha=alpha)
            if chronology == "clustered":
                axa.fill_between(
                    days,
                    offset + 0.58 * group["soc_min"].to_numpy(),
                    offset + 0.58 * group["soc_max"].to_numpy(),
                    color=color, alpha=0.09, linewidth=0,
                )
    axa.text(-1.2, 0.30, "task", ha="right", va="center", color=CLUSTERED,
             fontsize=5.5, fontweight="bold")
    axa.text(-1.2, -0.28, "recharge", ha="right", va="center", color=RECHARGE,
             fontsize=5.5, fontweight="bold")
    axa.text(-1.2, 1.53, "LFP 4 h", ha="right", va="center",
             color=TECH_COLOR["li_lfp"], fontsize=5.5, fontweight="bold")
    axa.text(-1.2, 2.31, "A-CAES 24 h", ha="right", va="center",
             color=TECH_COLOR["caes"], fontsize=5.5, fontweight="bold")
    axa.set_xlim(-4.5, 44.5)
    axa.set_ylim(-0.58, 2.78)
    axa.set_yticks([])
    axa.set_xlabel("Day within the highest-task 45-d clustered window")
    axa.plot([], [], color=INK, ls="-", label="Clustered SOC")
    axa.plot([], [], color=INK, ls="--", label="Dispersed SOC")
    axa.legend(loc="upper left", ncol=2)

    # Similar cycle retention, different absolute task-firm loss.
    summary = (
        pd.read_csv(V40_SOURCE / "figure3_panel_a_configuration_summary.csv")
        .set_index("config_id")
        .loc[["li_lfp_4h", "caes_24h"]]
    )
    labels = ["Cycle retention", "Task-firm loss"]
    values = {
        "li_lfp_4h": [
            100 * summary.loc["li_lfp_4h", "retained_equivalent_full_cycles_mean"],
            -100 * summary.loc["li_lfp_4h", "delta_efc_ratio_mean"],
        ],
        "caes_24h": [
            100 * summary.loc["caes_24h", "retained_equivalent_full_cycles_mean"],
            -100 * summary.loc["caes_24h", "delta_efc_ratio_mean"],
        ],
    }
    x = np.arange(2)
    width = 0.32
    axb.bar(x - width / 2, values["li_lfp_4h"], width, color=TECH_COLOR["li_lfp"], label="LFP 4 h")
    axb.bar(x + width / 2, values["caes_24h"], width, color=TECH_COLOR["caes"], label="A-CAES 24 h")
    axb.set_xticks(x)
    axb.set_xticklabels(labels, rotation=17, ha="right")
    axb.set_ylabel("Relative to dispersed (%)", labelpad=2)
    axb.legend(loc="upper left")
    for config, offset in [("li_lfp_4h", -width / 2), ("caes_24h", width / 2)]:
        for index, value in enumerate(values[config]):
            axb.text(index + offset, value + 1.2, f"{value:.1f}", ha="center", fontsize=5.2)

    for tech in TECH_ORDER:
        group = saturation[saturation["tech_id"].eq(tech)]
        if group.empty:
            continue
        axc.scatter(group["delta_charge_saturation_h"], group["delta_cycles"],
                    color=TECH_COLOR[tech], alpha=0.06, s=4, linewidth=0)
        axc.scatter(group["delta_charge_saturation_h"].median(),
                    group["delta_cycles"].median(), color=TECH_COLOR[tech],
                    s=24, marker="D", edgecolor="white", linewidth=0.4,
                    label=TECH_LABEL[tech])
    axc.axhline(0, color="#777777", lw=0.6)
    axc.set_xlabel("$\\Delta$ charge saturation (h yr$^{-1}$)")
    axc.set_ylabel("$\\Delta$ equivalent full cycles")
    axc.legend(loc="upper left", ncol=2, fontsize=5.0, columnspacing=0.6, handletextpad=0.2)

    countries = countries.sort_values("delta_caes_minus_lfp_eur_per_kw_year")
    y = np.arange(len(countries))
    axd.hlines(
        y,
        countries["dispersed_caes_minus_lfp_eur_per_kw_year"],
        countries["clustered_caes_minus_lfp_eur_per_kw_year"],
        color="#B5BEC2", lw=0.9,
    )
    axd.scatter(countries["dispersed_caes_minus_lfp_eur_per_kw_year"], y,
                color=DISPERSED, s=16, label="Dispersed")
    axd.scatter(countries["clustered_caes_minus_lfp_eur_per_kw_year"], y,
                color=CLUSTERED, s=16, label="Clustered")
    axd.axvline(0, color="#777777", lw=0.65)
    axd.set_yticks(y)
    axd.set_yticklabels(countries["country"])
    axd.set_xlabel("A-CAES24 − LFP4 annual net benefit\n(EUR kW$^{-1}$ yr$^{-1}$)")
    axd.legend(
        loc="lower center",
        bbox_to_anchor=(0.50, 1.005),
        ncol=2,
        borderaxespad=0,
    )

    xlink = 100 * link["caes_minus_lfp_charge_retention"].to_numpy()
    ylink = link["relative_value_shift_eur_per_kw_year"].to_numpy()
    axe.scatter(xlink, ylink, s=12, color=TECH_COLOR["caes"],
                alpha=0.40, edgecolor="white", linewidth=0.25)
    fit = np.polyfit(xlink, ylink, 1)
    xx = np.linspace(xlink.min(), xlink.max(), 100)
    axe.plot(xx, np.polyval(fit, xx), color=INK, lw=0.9)
    rho = pd.Series(xlink).corr(pd.Series(ylink), method="spearman")
    axe.text(0.04, 0.95, f"Spearman $\\rho$ = {rho:.2f}",
             transform=axe.transAxes, va="top", fontsize=5.4)
    axe.axhline(0, color="#777777", lw=0.6)
    axe.set_xlabel("A-CAES24 − LFP4\ncharge-retention gap (pp)")
    axe.set_ylabel("Relative annual-value shift\n(EUR kW$^{-1}$ yr$^{-1}$)")

    panel_labels(fig, axes)
    export(fig, "fig_v41_figure4_replenishment_mechanism")
    save_source(window, "figure4_panel_a_task_recharge_window.csv")
    save_source(traces, "figure4_panel_a_soc_window.csv")
    save_source(retention, "figure4_panel_b_retention_summary.csv")
    save_source(saturation, "figure4_panel_c_charge_saturation_cycles.csv")
    save_source(countries, "figure4_panel_d_country_relative_value.csv")
    save_source(link, "figure4_panel_e_charge_retention_value_link.csv")


def figure5() -> None:
    envelope = pd.read_csv(V40_SOURCE / "figure5_panel_abc_duration_value_envelopes.csv")
    dominant = pd.read_csv(V40_SOURCE / "figure5_panel_b_dominant_technology_by_duration.csv")
    widths = pd.read_csv(V40_SOURCE / "figure5_panel_d_gap_widths.csv")
    phase = pd.read_csv(V40_SOURCE / "figure5_panel_e_gap_phase_map.csv")
    components = pd.read_csv(V40_SOURCE / "figure5_panel_f_value_components.csv")
    fig = plt.figure(figsize=(WIDTH, 5.70))
    fig.subplots_adjust(left=0.080, right=0.985, bottom=0.100, top=0.965)
    gs = fig.add_gridspec(3, 12, height_ratios=[1.00, 0.72, 0.92], hspace=0.58, wspace=1.05)
    axa = fig.add_subplot(gs[0:2, 0:8])
    axb = fig.add_subplot(gs[0, 8:12])
    axc = fig.add_subplot(gs[1, 8:12])
    axd = fig.add_subplot(gs[2, 0:4])
    axe = fig.add_subplot(gs[2, 4:8])
    axf = fig.add_subplot(gs[2, 8:12])
    axes = [axa, axb, axc, axd, axe, axf]

    qualifying = envelope[envelope["scenario_id"].eq(31)]
    for chronology, color in [("dispersed", DISPERSED), ("clustered", CLUSTERED)]:
        group = qualifying[qualifying["chronology"].eq(chronology)].sort_values("duration_h")
        axa.fill_between(group["duration_h"], group["q25_eur_per_kw_year"],
                         group["q75_eur_per_kw_year"], color=color, alpha=0.14, linewidth=0)
        axa.plot(group["duration_h"], group["mean_eur_per_kw_year"],
                 color=color, marker="o", ms=3, label=chronology.capitalize())
    axa.axhline(0, color="#777777", lw=0.7)
    axa.axvspan(28, 40, color=PURPLE, alpha=0.10)
    axa.axvline(44, color=PURPLE, ls=":", lw=1.0)
    duration_axis(axa, 168)
    axa.set_ylabel("Best annual net benefit\n(EUR kW$^{-1}$ yr$^{-1}$)")
    axa.set_xlabel("Discharge duration (h)")
    axa.legend(loc="upper left", ncol=2)
    axa.text(33.5, axa.get_ylim()[1] * 0.88, "negative-value interval",
             ha="center", color=PURPLE, fontsize=5.6, fontweight="bold")
    axa.text(44.8, 5, "re-entry", color=PURPLE, fontsize=5.3, fontweight="bold")
    for _, row in dominant.iterrows():
        axa.scatter(row["duration_h"], -18.0, s=17, color=TECH_COLOR[row["tech_id"]],
                    clip_on=False, zorder=5)
    axa.text(6.2, -28, "LFP", color=TECH_COLOR["li_lfp"], fontweight="bold", fontsize=5.3)
    axa.text(15.5, -28, "A-CAES", color=TECH_COLOR["caes"], fontweight="bold", fontsize=5.3)
    axa.text(73, -28, "Cavern H$_2$", color=TECH_COLOR["h2_cavern"],
             fontweight="bold", fontsize=5.3)

    # Central case is a small negative-control inset, not a duplicate full panel.
    inset = axa.inset_axes([0.055, 0.56, 0.34, 0.30])
    central = envelope[envelope["scenario_id"].eq(20)]
    for chronology, color in [("dispersed", DISPERSED), ("clustered", CLUSTERED)]:
        group = central[central["chronology"].eq(chronology)].sort_values("duration_h")
        require_strictly_positive(group["duration_h"], "central-case duration")
        inset.plot(group["duration_h"], group["mean_eur_per_kw_year"],
                   color=color, lw=0.85)
    inset.axhline(0, color="#777777", lw=0.5)
    inset.set_xscale("log")
    inset.set_xlim(4, 168)
    inset.set_xticks([4, 24, 168])
    inset.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    inset.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    inset.tick_params(labelsize=5.0, length=1.5)
    inset.set_title("Central value: whole-frontier exit, no strict valley",
                    loc="left", fontsize=5.0, color=MUTED, pad=1)

    for chronology, color in [("dispersed", DISPERSED), ("clustered", CLUSTERED)]:
        group = qualifying[qualifying["chronology"].eq(chronology)].sort_values("duration_h")
        axb.plot(group["duration_h"], 100 * group["positive_share"],
                 color=color, marker="o", ms=2.5)
    duration_axis(axb, 168)
    axb.set_ylim(-3, 103)
    axb.set_ylabel("Positive country-years (%)")
    axb.set_xlabel("Duration (h)")
    axb.axvspan(28, 40, color=PURPLE, alpha=0.08)

    clustered_widths = widths[
        widths["scenario_id"].eq(31) & widths["chronology"].eq("clustered")
    ]["gap_width_h"].to_numpy()
    dispersed_widths = widths[
        widths["scenario_id"].eq(31) & widths["chronology"].eq("dispersed")
    ]["gap_width_h"].to_numpy()
    positions = [0, 1]
    for pos, values, color in zip(positions, [dispersed_widths, clustered_widths], [DISPERSED, CLUSTERED]):
        jitter = np.linspace(-0.12, 0.12, len(values))
        axc.scatter(np.full(len(values), pos) + jitter, values, s=5, alpha=0.18, color=color)
        axc.plot([pos - 0.18, pos + 0.18], [np.median(values)] * 2, color=INK, lw=1.5)
    axc.set_xticks(positions)
    axc.set_xticklabels(["Dispersed", "Clustered"])
    axc.set_ylabel("Strict valley width (h)")
    axc.text(1, clustered_widths.mean() + 2.4, f"mean {clustered_widths.mean():.1f} h",
             ha="center", fontsize=5.3, color=CLUSTERED, fontweight="bold")

    pivot = phase.pivot(
        index="charge_price_eur_per_mwh",
        columns="energy_value_eur_per_mwh",
        values="valley_prevalence_change_pp",
    ).sort_index(ascending=False)
    image = axd.imshow(pivot.to_numpy(), cmap="RdBu_r", vmin=-100, vmax=100, aspect="auto")
    for iy in range(pivot.shape[0]):
        for ix in range(pivot.shape[1]):
            value = pivot.iloc[iy, ix]
            axd.text(ix, iy, f"{value:.0f}", ha="center", va="center",
                     fontsize=5.4, color="white" if abs(value) > 55 else INK,
                     fontweight="bold")
    axd.set_xticks(np.arange(pivot.shape[1]))
    axd.set_xticklabels([f"{value:.1f}" for value in pivot.columns])
    axd.set_yticks(np.arange(pivot.shape[0]))
    axd.set_yticklabels([f"{value:.2f}" for value in pivot.index])
    axd.set_xlabel("Energy displacement value (EUR MWh$^{-1}$)")
    axd.set_ylabel("Charging opportunity price\n(EUR MWh$^{-1}$)")
    # Highlight central and qualifying tests.
    central_x = list(pivot.columns).index(min(pivot.columns, key=lambda value: abs(value - 65.233)))
    central_y = list(pivot.index).index(min(pivot.index, key=lambda value: abs(value - 6.16)))
    qual_x = list(pivot.columns).index(max(pivot.columns))
    qual_y = list(pivot.index).index(min(pivot.index))
    for xcell, ycell, colour in [(central_x, central_y, INK), (qual_x, qual_y, PURPLE)]:
        axd.add_patch(Rectangle((xcell - 0.47, ycell - 0.47), 0.94, 0.94,
                                fill=False, edgecolor=colour, linewidth=1.1))
    axd.text(
        0.99,
        1.02,
        "cell value = $\\Delta$ prevalence (percentage points)",
        transform=axd.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.0,
        color=MUTED,
    )

    # Three jointly necessary signatures.
    requirements = [
        ("Short-duration positive value", 132, TECH_COLOR["caes"]),
        ("Contiguous negative interval", 132, CLUSTERED),
        ("Long-duration re-entry", 132, PURPLE),
    ]
    y_positions = [0.76, 0.49, 0.22]
    for index, ((label, count, color), ypos) in enumerate(zip(requirements, y_positions)):
        rounded(
            axe, (0.06, ypos - 0.09), 0.88, 0.18,
            mpl.colors.to_rgba(color, 0.10),
            edge=color, linewidth=0.75, radius=0.018, zorder=2,
        )
        axe.text(0.19, ypos, str(count), transform=axe.transAxes,
                 ha="center", va="center", fontsize=7.0,
                 fontweight="bold", color=color)
        axe.text(0.36, ypos, label, transform=axe.transAxes,
                 ha="left", va="center", fontsize=5.2, color=INK)
        if index < 2:
            axe.annotate(
                "", xy=(0.50, ypos - 0.14), xytext=(0.50, ypos - 0.10),
                xycoords=axe.transAxes,
                arrowprops={"arrowstyle": "-|>", "color": "#AAB4B9", "lw": 0.7},
            )
    axe.set_xlim(0, 1)
    axe.set_ylim(0, 1)
    axe.axis("off")
    axe.text(0.50, 0.98, "Joint criterion under clustered chronology",
             transform=axe.transAxes, ha="center", va="top",
             fontsize=5.3, fontweight="bold", color=MUTED)

    xbars = np.arange(len(components))
    energy = components["energy_value_eur_per_kw_year"].to_numpy()
    firm = components["firm_value_eur_per_kw_year"].to_numpy()
    cost = components["storage_cost_eur_per_kw_year"].to_numpy()
    charge = components["charge_cost_eur_per_kw_year"].to_numpy()
    axf.bar(xbars, energy, color="#4C78A8", label="Energy")
    axf.bar(xbars, firm, bottom=energy, color="#59A14F", label="Task-firm supply")
    axf.bar(xbars, cost, color="#B5B5B5", label="Storage cost")
    axf.bar(xbars, charge, bottom=cost, color="#F28E2B", label="Charge cost")
    axf.scatter(xbars, components["net_value_eur_per_kw_year"], color=INK, marker="D",
                s=24, edgecolor="white", linewidth=0.6, label="Net", zorder=5)
    axf.axhline(0, color="#777777", lw=0.65)
    axf.set_xticks(xbars)
    component_labels = {
        "li_lfp_10h": "LFP 10 h",
        "caes_24h": "A-CAES 24 h",
        "h2_cavern_36h": "Cavern H$_2$ 36 h",
        "h2_cavern_60h": "Cavern H$_2$ 60 h",
    }
    axf.set_xticklabels(
        [component_labels.get(value, value) for value in components["config_id"]],
        rotation=31, ha="right",
    )
    axf.set_ylabel("Annual value component\n(EUR kW$^{-1}$ yr$^{-1}$)")
    axf.legend(loc="upper left", ncol=2, fontsize=5.0, columnspacing=0.6, handletextpad=0.25)

    panel_labels(fig, axes)
    export(fig, "fig_v41_figure5_conditional_gap")
    save_source(envelope, "figure5_panel_abc_duration_value_envelopes.csv")
    save_source(dominant, "figure5_panel_a_dominant_technology.csv")
    save_source(widths, "figure5_panel_c_gap_widths.csv")
    save_source(phase, "figure5_panel_d_gap_phase_map.csv")
    save_source(components, "figure5_panel_f_value_components.csv")


def main() -> None:
    configure()
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    manifest = {
        "status": "pass",
        "main_figures": 5,
        "primary_sample": {
            "countries": 15,
            "country_years": 132,
            "country_hours": 1156320,
            "configurations": 53,
        },
        "removed": ["Chinese provincial validation figure", "redundant coverage matrix"],
        "figure5_control": "central-value envelope retained only as a negative-control inset",
        "input_source_data": str(V40_SOURCE),
    }
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)
    (SOURCE_OUT / "figure_build_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
