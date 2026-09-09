#!/usr/bin/env python3
"""Build v57 Figure 1: natural occurrence, mechanism and investment endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parent
V57 = ROOT.parent
RESULTS = V57 / "weather_year_parity_v42" / "results"
RESPONSE = ROOT / "source_data_v49"
OUT = ROOT / "main_figures"
SOURCE = ROOT / "source_data_v57"
OUT.mkdir(parents=True, exist_ok=True)
SOURCE.mkdir(parents=True, exist_ok=True)

MM = 1.0 / 25.4
INK = "#263842"
MUTED = "#6D7D84"
GRID = "#D8E0E2"
BLUE = "#3B82A8"
TEAL = "#178F83"
CORAL = "#D76F55"
GOLD = "#D2A13B"
VIOLET = "#7062A3"
RED = "#BB4D48"
GREEN = "#3C8766"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 6.2,
    "axes.labelsize": 6.3,
    "axes.titlesize": 6.8,
    "xtick.labelsize": 5.5,
    "ytick.labelsize": 5.5,
    "legend.fontsize": 5.4,
    "axes.linewidth": 0.65,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.55,
    "ytick.major.width": 0.55,
    "xtick.major.size": 2.4,
    "ytick.major.size": 2.4,
    "axes.unicode_minus": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})


def panel_label(ax: plt.Axes, letter: str, x: float = -0.08, y: float = 1.04) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8.2, fontweight="bold", va="bottom", ha="left")


def quiet(ax: plt.Axes, grid: str | None = None) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.45)
        ax.set_axisbelow(True)


def save_bundle(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), metadata={"Creator": "Python/matplotlib"})
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    rgba = plt.imread(stem.with_suffix(".png"))
    grey = np.dot(rgba[..., :3], [0.2126, 0.7152, 0.0722])
    plt.imsave(stem.with_name(stem.name + "_grayscale").with_suffix(".png"), grey, cmap="gray", vmin=0, vmax=1)


def first_sustained_positive(duration: np.ndarray, effect: np.ndarray) -> float:
    order = np.argsort(duration)
    duration = np.asarray(duration, float)[order]
    effect = np.asarray(effect, float)[order]
    for index in range(len(duration)):
        if np.all(effect[index:] > 0):
            return float(duration[index])
    return float("nan")


def plot_natural_record(ax: plt.Axes, values: pd.DataFrame, decades: pd.DataFrame) -> None:
    frame = values.sort_values("window_start_year").reset_index(drop=True)
    x = np.arange(len(frame))
    y = frame.value_2025_eur_kw_year.to_numpy(float)
    ax.fill_between(x, y, np.nanmax(y) + 0.8, color=mpl.colors.to_rgba(TEAL, 0.08), lw=0)
    ax.plot(x, y, color=TEAL, lw=1.05)
    ax.scatter(x, y, s=7, color=TEAL, edgecolor="white", lw=0.25, zorder=3)

    worst = decades.loc[decades.weather_value_change_eur_kw_year.idxmin()]
    start = int(np.flatnonzero(frame.window_start_year.to_numpy(int) == int(worst.start_year))[0])
    end = int(np.flatnonzero(frame.window_start_year.to_numpy(int) == int(worst.end_year))[0])
    ax.scatter([start, end], [y[start], y[end]], s=34, color=[GREEN, RED], edgecolor="white", lw=0.6, zorder=5)
    arc_y = max(y[start], y[end]) + 0.95
    ax.add_patch(FancyArrowPatch(
        (start, arc_y), (end, arc_y), arrowstyle="-|>", mutation_scale=7,
        connectionstyle="arc3,rad=-0.18", color=RED, lw=0.8,
    ))
    ax.text(
        (start + end) / 2, arc_y + 0.22,
        f"{worst.start_window} to {worst.end_window}  {worst.weather_value_change_eur_kw_year:.2f}",
        ha="center", va="bottom", color=RED, fontsize=5.4, fontweight="bold",
    )
    ticks = np.arange(0, len(frame), 10)
    ax.set_xticks(ticks, frame.weather_window.iloc[ticks])
    ax.set_xlim(-1, len(frame))
    ax.set_ylabel("24 h A-CAES relative value\n(EUR kW$^{-1}$ yr$^{-1}$)")
    ax.set_xlabel("July-June operating year")
    ax.set_title("Natural occurrence | 75 synchronized European operating years", loc="left", fontweight="bold", pad=4)
    quiet(ax, "y")
    panel_label(ax, "a", -0.075, 1.05)


def plot_scale_ledger(ax: plt.Axes, summary: dict[str, object]) -> None:
    threshold = float(summary["cost_improvement_2025_2035_eur_kw_year"])
    metrics = [
        ("Full natural span", float(summary["natural_full_span_eur_kw_year"]), INK),
        ("Central 90% span", float(summary["natural_q95_q05_span_eur_kw_year"]), TEAL),
        ("2025-2035 learning", threshold, VIOLET),
    ]
    for row, (label, value, colour) in enumerate(metrics[::-1]):
        ax.plot([0, value], [row, row], color=colour, lw=5.0, solid_capstyle="butt")
        ax.scatter(value, row, s=25, color=colour, edgecolor="white", lw=0.45, zorder=4)
        ax.text(value + 0.20, row, f"{value:.2f}", color=colour, va="center", ha="left", fontsize=5.6, fontweight="bold")
    ax.set_yticks(range(3), [item[0] for item in metrics[::-1]])
    ax.set_xlim(0, 12.6)
    ax.set_ylim(-0.8, 3.85)
    ax.set_xlabel("Annual value scale (EUR kW$^{-1}$ yr$^{-1}$)")
    quiet(ax, "x")
    ax.set_title("Weather risk and economic endpoint", loc="left", fontweight="bold", pad=4)

    counts = [
        (f"{summary['all_pairs_reaching_cost_scale']}/{summary['all_unordered_weather_pairs']}", "natural pairs", CORAL),
        (f"{summary['matched_decades_erasing_learning']}/{summary['matched_ten_year_transitions']}", "adverse decades", RED),
        (f"{summary['economic_crossovers_by_cost_year']['2035']}/{summary['complete_july_june_operating_years']}", "2035 crossovers", INK),
    ]
    for index, (count, label, colour) in enumerate(counts):
        xpos = 0.02 + 0.33 * index
        ax.text(xpos, 0.98, count, transform=ax.transAxes, color=colour, fontsize=6.4, fontweight="bold", ha="left", va="top")
        ax.text(xpos, 0.88, label, transform=ax.transAxes, color=MUTED, fontsize=5.0, ha="left", va="top")
    panel_label(ax, "b", -0.16, 1.05)


def plot_mechanism(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Conserved sequencing | task and recharge held fixed", loc="left", fontweight="bold", pad=4)
    panel_label(ax, "c", -0.075, 1.05)

    for xpos, label, colour in [(0.07, "Europe", BLUE), (0.19, "China", CORAL), (0.29, "Japan", TEAL)]:
        ax.scatter(xpos, 0.89, s=24, color=colour, edgecolor="white", lw=0.4)
        ax.text(xpos + 0.025, 0.89, label, va="center", ha="left", color=colour, fontsize=5.4, fontweight="bold")

    event_heights = np.array([0.20, 0.28, 0.22, 0.31, 0.24])
    for y0, label, positions, colour in [
        (0.62, "dispersed", np.array([0.08, 0.26, 0.45, 0.66, 0.86]), BLUE),
        (0.35, "clustered", np.array([0.31, 0.39, 0.47, 0.55, 0.63]), CORAL),
    ]:
        ax.text(0.01, y0 + 0.11, label, color=colour, fontsize=5.6, fontweight="bold", va="center")
        ax.plot([0.08, 0.91], [y0, y0], color=GRID, lw=0.7)
        for xpos, height in zip(positions, event_heights):
            ax.add_patch(Rectangle((xpos - 0.012, y0), 0.024, 0.42 * height, facecolor=CORAL, edgecolor="white", lw=0.25))
        wave_x = np.linspace(0.08, 0.91, 180)
        wave_y = y0 - 0.07 + 0.022 * np.sin(np.linspace(0, 10 * np.pi, 180))
        ax.plot(wave_x, wave_y, color=TEAL, lw=0.9)

    ax.annotate(
        "median recovery 72 h", xy=(0.31, 0.55), xytext=(0.49, 0.55), ha="center", va="center",
        arrowprops={"arrowstyle": "<->", "color": TEAL, "lw": 0.7}, color=TEAL, fontsize=5.2,
    )
    ax.annotate(
        "0 h", xy=(0.46, 0.28), xytext=(0.51, 0.28), ha="center", va="center",
        arrowprops={"arrowstyle": "<->", "color": TEAL, "lw": 0.7}, color=TEAL, fontsize=5.2,
    )

    ledger = [
        ("Recovery", "72 h", "0 h", TEAL),
        ("A-CAES cycles", "19.01", "11.48", BLUE),
        ("Portfolio value", "0", "-5.21", CORAL),
    ]
    for index, (label, before, after, colour) in enumerate(ledger):
        xpos = 0.08 + index * 0.30
        ax.text(xpos, 0.13, label, color=MUTED, fontsize=5.0, ha="left")
        ax.text(xpos, 0.065, before, color=INK, fontsize=6.2, fontweight="bold", ha="left")
        ax.annotate(
            "", xy=(xpos + 0.17, 0.072), xytext=(xpos + 0.09, 0.072),
            arrowprops={"arrowstyle": "-|>", "color": colour, "lw": 0.8},
        )
        ax.text(xpos + 0.19, 0.065, after, color=colour, fontsize=6.2, fontweight="bold", ha="left")


def plot_duration_consequence(fig: plt.Figure, spec, eu: pd.DataFrame, cross: pd.DataFrame) -> pd.DataFrame:
    nested = spec.subgridspec(3, 1, hspace=0.12)
    panels: list[pd.DataFrame] = []
    definitions = [
        ("Europe", eu.duration_h.to_numpy(float), eu.point_estimate.to_numpy(float), BLUE),
        (
            "China",
            cross.loc[cross.application_case.eq("china"), "duration_h"].to_numpy(float),
            cross.loc[cross.application_case.eq("china"), "mean_effect_eur_kw_year"].to_numpy(float),
            CORAL,
        ),
        (
            "Japan",
            cross.loc[cross.application_case.eq("japan"), "duration_h"].to_numpy(float),
            cross.loc[cross.application_case.eq("japan"), "mean_effect_eur_kw_year"].to_numpy(float),
            TEAL,
        ),
    ]
    first_ax: plt.Axes | None = None
    for index, (name, duration, effect, colour) in enumerate(definitions):
        ax = fig.add_subplot(nested[index, 0])
        if first_ax is None:
            first_ax = ax
            ax.set_title("Duration response | long-duration reversal", loc="left", fontweight="bold", pad=4)
            panel_label(ax, "d", -0.15, 1.18)
        scale = np.nanmax(np.abs(effect))
        normalized = effect / scale if scale > 0 else effect
        order = np.argsort(duration)
        duration = duration[order]
        normalized = normalized[order]
        if np.any(duration <= 0):
            raise ValueError(f"{name} contains nonpositive storage duration")
        ax.axvspan(8, 24, color=mpl.colors.to_rgba(GOLD, 0.12), lw=0)
        ax.axhline(0, color=INK, lw=0.55)
        ax.fill_between(duration, normalized, 0, where=normalized < 0, color=mpl.colors.to_rgba(colour, 0.15), lw=0)
        ax.plot(duration, normalized, color=colour, lw=1.05)
        if index == 0:
            ax.text(0.98, 0.80, "normalized within regime", transform=ax.transAxes, color=MUTED,
                    fontsize=5.0, ha="right", va="top")
        ax.text(0.01, 0.80, name, transform=ax.transAxes, color=colour, fontsize=5.4, fontweight="bold", ha="left", va="top")
        ax.set_xscale("log")
        ax.set_ylim(-1.08, 1.08)
        ax.set_yticks([-1, 0, 1] if index == 1 else [])
        if index < 2:
            ax.tick_params(axis="x", bottom=False, labelbottom=False)
        else:
            ax.set_xlabel("Storage duration (h)")
        quiet(ax)
        panels.append(pd.DataFrame({"regime": name, "duration_h": duration, "normalized_effect": normalized}))
    return pd.concat(panels, ignore_index=True)


def build() -> None:
    values = pd.read_csv(RESULTS / "pecd_v42_75_window_relative_values_2025.csv")
    decades = pd.read_csv(RESULTS / "pecd_v42_65_matched_decade_transitions.csv")
    summary = json.loads((RESULTS / "pecd_v42_75y_parity_summary.json").read_text(encoding="utf-8"))
    eu = pd.read_csv(RESPONSE / "figure5a_europe_duration_response.csv")
    cross = pd.read_csv(RESPONSE / "figure5bc_china_japan_duration_response.csv")

    fig = plt.figure(figsize=(7.20, 4.80))
    grid = fig.add_gridspec(
        2, 12, left=0.075, right=0.985, bottom=0.075, top=0.955,
        height_ratios=[1.02, 1.0], hspace=0.48, wspace=0.90,
    )
    ax_a = fig.add_subplot(grid[0, :8])
    ax_b = fig.add_subplot(grid[0, 8:])
    ax_c = fig.add_subplot(grid[1, :7])

    plot_natural_record(ax_a, values, decades)
    plot_scale_ledger(ax_b, summary)
    plot_mechanism(ax_c)
    duration_source = plot_duration_consequence(fig, grid[1, 7:], eu, cross)

    values[["window_start_year", "weather_window", "value_2025_eur_kw_year"]].to_csv(
        SOURCE / "figure1a_natural_record.csv", index=False,
    )
    pd.DataFrame({
        "metric": [
            "natural_full_span", "natural_central_90_span", "relative_cost_improvement",
            "all_pair_hits", "adverse_decade_hits", "economic_crossovers_2035",
        ],
        "value": [
            summary["natural_full_span_eur_kw_year"], summary["natural_q95_q05_span_eur_kw_year"],
            summary["cost_improvement_2025_2035_eur_kw_year"], summary["all_pairs_reaching_cost_scale"],
            summary["matched_decades_erasing_learning"], summary["economic_crossovers_by_cost_year"]["2035"],
        ],
    }).to_csv(SOURCE / "figure1b_scale_ledger.csv", index=False)
    duration_source.to_csv(SOURCE / "figure1d_duration_consequence.csv", index=False)

    stem = OUT / "fig_v57_figure1_natural_sequencing_framework"
    save_bundle(fig, stem)
    plt.close(fig)
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    build()
