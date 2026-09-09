#!/usr/bin/env python3
"""Create the submission-scale PECD 35-weather-year parity summary figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"

COUNTRY_ORDER = ["AT", "BE", "BG", "CZ", "DE", "DK", "ES", "FR", "GB", "GR", "HR", "IT", "PT", "RO", "SI"]

BLUE = "#397AA5"
TEAL = "#278485"
RED = "#C95545"
PEACH = "#D88974"
DARK = "#252525"
MID = "#777777"
LIGHT = "#D8D8D8"
PALE = "#F2F2F2"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 6.5,
        "axes.labelsize": 6.5,
        "axes.titlesize": 7.0,
        "axes.linewidth": 0.6,
        "xtick.labelsize": 5.8,
        "ytick.labelsize": 5.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "legend.fontsize": 5.8,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def panel_label(ax: plt.Axes, label: str, x: float = -0.13, y: float = 1.05) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    values = pd.read_csv(RESULTS / "pecd_35y_relative_values.csv")
    country = pd.read_csv(RESULTS / "pecd_35y_country_parity_summary.csv")
    climate = pd.read_csv(RESULTS / "pecd_35y_climate_year_summary.csv")
    pairs = pd.read_csv(RESULTS / "pecd_35y_within_country_year_pairs.csv.gz")
    summary = json.loads((RESULTS / "pecd_35y_parity_summary.json").read_text())
    threshold = float(summary["cost_parity_threshold_eur_kw_year"])

    # 183 x 113 mm, expressed in inches for deterministic journal-width QA.
    figure = plt.figure(figsize=(7.204724, 4.448819), constrained_layout=False)
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=[1.55, 1.0],
        height_ratios=[1.08, 0.92],
        left=0.065,
        right=0.985,
        bottom=0.10,
        top=0.96,
        wspace=0.34,
        hspace=0.42,
    )
    ax_a = figure.add_subplot(grid[0, 0])
    ax_b = figure.add_subplot(grid[:, 1])
    lower_left = grid[1, 0].subgridspec(1, 2, width_ratios=[1.18, 1.0], wspace=0.46)
    ax_c = figure.add_subplot(lower_left[0, 0])
    ax_d = figure.add_subplot(lower_left[0, 1])

    # a, natural chronology value deviations across all country-weather years.
    heat = (
        values.pivot(index="country", columns="weather_year", values="within_country_value_deviation")
        .reindex(COUNTRY_ORDER)
        .sort_index(axis=1)
    )
    cmap = LinearSegmentedColormap.from_list("chronology", [BLUE, "#F7F7F7", PEACH], N=256)
    norm = TwoSlopeNorm(vmin=-threshold, vcenter=0.0, vmax=threshold)
    image = ax_a.imshow(heat.to_numpy(), aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    ax_a.set_yticks(np.arange(len(heat.index)), heat.index)
    tick_positions = np.arange(0, len(heat.columns), 5)
    ax_a.set_xticks(tick_positions, heat.columns.to_numpy()[tick_positions])
    ax_a.set_xlabel("Weather year")
    ax_a.set_ylabel("Country")
    ax_a.set_title("Natural chronology shifts 24 h A-CAES value", loc="left", pad=4, fontweight="bold")
    crossings = np.argwhere(np.abs(heat.to_numpy()) >= threshold)
    if len(crossings):
        ax_a.scatter(crossings[:, 1], crossings[:, 0], s=8, marker="o", facecolors="none", edgecolors=DARK, linewidths=0.55)
    ax_a.text(
        0.992,
        0.025,
        "open circles  |deviation| ≥ parity",
        transform=ax_a.transAxes,
        ha="right",
        va="bottom",
        color=MID,
        fontsize=5.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.0},
    )
    colorbar = figure.colorbar(image, ax=ax_a, orientation="horizontal", fraction=0.055, pad=0.18, aspect=28)
    colorbar.set_label("Deviation from country median (EUR kW$^{-1}$ yr$^{-1}$)", labelpad=2)
    colorbar.set_ticks(
        [-threshold, 0.0, threshold],
        labels=[f"-{threshold:.1f}", "0", f"+{threshold:.1f}"],
    )
    panel_label(ax_a, "a", x=-0.11, y=1.04)

    # b, country-level full and central spans against the cost movement.
    ordered = country.sort_values("natural_full_span_eur_kw_year", ascending=True).reset_index(drop=True)
    y = np.arange(len(ordered))
    ax_b.hlines(y, 0.0, ordered.natural_full_span_eur_kw_year, color=LIGHT, linewidth=3.2, zorder=1)
    ax_b.hlines(y, 0.0, ordered.natural_q95_q05_span_eur_kw_year, color=TEAL, linewidth=3.2, zorder=2)
    ax_b.scatter(ordered.natural_full_span_eur_kw_year, y, color=DARK, s=10, zorder=3)
    below = ~ordered.q95_q05_span_reaches_cost_parity
    ax_b.scatter(ordered.loc[below, "natural_q95_q05_span_eur_kw_year"], y[below], color="white", edgecolor=TEAL, linewidth=0.7, s=11, zorder=4)
    ax_b.axvline(threshold, color=RED, linestyle=(0, (3, 2)), linewidth=1.0, zorder=0)
    ax_b.text(threshold + 0.12, len(ordered) - 0.3, "2025–2035 cost movement", color=RED, fontsize=5.7, rotation=90, va="top")
    ax_b.set_yticks(y, ordered.country)
    ax_b.set_xlim(0.0, max(12.0, ordered.natural_full_span_eur_kw_year.max() + 0.7))
    ax_b.set_ylim(-1.1, len(ordered) + 0.9)
    ax_b.set_xlabel("Natural value span (EUR kW$^{-1}$ yr$^{-1}$)")
    ax_b.set_title("Country-level natural spans", loc="left", pad=4, fontweight="bold")
    ax_b.text(0.25, len(ordered) + 0.55, "teal  5th–95th percentile    grey  full span", color=MID, va="center")
    ax_b.text(0.25, len(ordered) + 0.18, "open endpoint  5th–95th span below parity", color=MID, va="center", fontsize=5.5)
    ax_b.text(11.7, -0.72, f"{int(country.q95_q05_span_reaches_cost_parity.sum())}/15 robust spans cross parity", ha="right", va="center", color=TEAL, fontweight="bold")
    panel_label(ax_b, "b", x=-0.18, y=1.02)

    # c, empirical distribution of all within-country weather-year pairs.
    pair_values = np.sort(pairs.absolute_value_difference_eur_kw_year.to_numpy(float))
    exceedance = 1.0 - np.arange(1, len(pair_values) + 1) / len(pair_values)
    ax_c.step(pair_values, 100.0 * exceedance, where="post", color=DARK, linewidth=1.25)
    ax_c.axvspan(threshold, max(pair_values.max(), threshold), color=RED, alpha=0.08, linewidth=0)
    ax_c.axvline(threshold, color=RED, linestyle=(0, (3, 2)), linewidth=1.0)
    ax_c.set_xlim(0.0, 12.0)
    ax_c.set_ylim(0.0, 100.0)
    ax_c.set_xlabel("Pairwise difference\n(EUR kW$^{-1}$ yr$^{-1}$)")
    ax_c.set_ylabel("Pairs exceeding difference (%)")
    ax_c.set_title("Natural-year pairwise tail", loc="left", pad=4, fontweight="bold")
    ax_c.text(
        threshold + 0.2,
        22,
        f"{int(summary['within_country_pairs_reaching_parity']):,} / {int(summary['within_country_weather_year_pairs']):,}\n({100 * summary['within_country_pair_fraction_reaching_parity']:.1f}%)",
        color=RED,
        fontweight="bold",
        va="center",
    )
    panel_label(ax_c, "c", x=-0.11, y=1.05)

    # d, common weather-year effect averaged across all 15 countries.
    ax_d.axhspan(-threshold, threshold, color=PALE, zorder=0)
    ax_d.axhline(0.0, color=MID, linewidth=0.6, zorder=1)
    ax_d.plot(climate.weather_year, climate.mean_within_country_value_deviation, color=BLUE, linewidth=1.0, marker="o", markersize=2.0, zorder=2)
    low = climate.loc[climate.mean_within_country_value_deviation.idxmin()]
    high = climate.loc[climate.mean_within_country_value_deviation.idxmax()]
    ax_d.scatter([low.weather_year], [low.mean_within_country_value_deviation], color=RED, s=14, zorder=3)
    ax_d.scatter([high.weather_year], [high.mean_within_country_value_deviation], color=TEAL, s=14, zorder=3)
    ax_d.set_xlim(1981, 2017)
    ax_d.set_ylim(-threshold * 1.04, threshold * 1.04)
    ax_d.set_xticks([1982, 1990, 2000, 2010, 2016])
    ax_d.set_yticks([-5, 0, 5])
    ax_d.set_xlabel("Weather year")
    ax_d.set_ylabel("15-country mean deviation\n(EUR kW$^{-1}$ yr$^{-1}$)")
    ax_d.set_title("Europe-wide common effect", loc="left", pad=4, fontweight="bold", fontsize=6.5)
    ax_d.text(0.97, 0.05, f"full span {summary['cross_country_mean_weather_year_effect_full_span_eur_kw_year']:.2f}", transform=ax_d.transAxes, ha="right", va="bottom", color=BLUE)
    panel_label(ax_d, "d", x=-0.20, y=1.05)

    for axis in [ax_a, ax_b, ax_c, ax_d]:
        axis.tick_params(direction="out")
    stem = FIGURES / "pecd_35y_natural_chronology_parity"
    figure.savefig(stem.with_suffix(".svg"))
    figure.savefig(stem.with_suffix(".pdf"))
    figure.savefig(stem.with_suffix(".png"), dpi=300)
    figure.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
