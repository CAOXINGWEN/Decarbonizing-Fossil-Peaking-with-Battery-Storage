#!/usr/bin/env python3
"""Build v57 Extended Data Figures 2 and 3 from frozen PECD results."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROCESSED = ROOT / "weather_year_parity_v42" / "processed"
RESULTS = ROOT / "weather_year_parity_v42" / "results"
OUT = HERE / "extended_data_figures"
SOURCE = HERE / "source_data_v57"
OUT.mkdir(parents=True, exist_ok=True)
SOURCE.mkdir(parents=True, exist_ok=True)

MM = 1.0 / 25.4
INK = "#243740"
MUTED = "#6B7D84"
GRID = "#D7DFE2"
PALE = "#F2F4F2"
BLUE = "#4C91B5"
TEAL = "#188F83"
SAGE = "#7CA995"
CORAL = "#D96F55"
GOLD = "#D6A43A"
PURPLE = "#7568AA"
GREY = "#96A2A6"


def configure() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 6.2,
        "axes.labelsize": 6.5,
        "axes.titlesize": 6.7,
        "xtick.labelsize": 5.5,
        "ytick.labelsize": 5.5,
        "legend.fontsize": 5.3,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })


def label_panel(ax: plt.Axes, letter: str, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8.4, fontweight="bold",
            ha="left", va="bottom", color=INK)


def finish(ax: plt.Axes, axis: str = "x") -> None:
    ax.grid(axis=axis, color=GRID, lw=0.45, alpha=0.9)
    ax.set_axisbelow(True)


def save_bundle(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), metadata={"Creator": "Python/matplotlib"})
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=350)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600,
                pil_kwargs={"compression": "tiff_lzw"})


def ed2_inputs_and_construction() -> None:
    inventory = pd.read_csv(PROCESSED / "pecd_v42_zone_inventory.csv")
    weights = pd.read_csv(PROCESSED / "fixed_2030_vre_capacity_weights.csv")
    matching = pd.read_csv(PROCESSED / "pecd_v42_zone_weight_matching_qa.csv")

    fig = plt.figure(figsize=(7.20, 5.91))
    gs = fig.add_gridspec(
        3, 12, left=0.075, right=0.985, bottom=0.07, top=0.95,
        hspace=0.68, wspace=1.05, height_ratios=[0.82, 1.32, 0.92],
    )

    ax = fig.add_subplot(gs[0, :6])
    rows = [
        ("Onshore", BLUE, [(1950, 2025)]),
        ("Offshore", PURPLE, [(1950, 2025)]),
        ("Solar PV", GOLD, [(1950, 2025)]),
        ("Temperature", TEAL, [(1950, 2021), (2022, 2025)]),
    ]
    for i, (name, colour, spans) in enumerate(rows):
        ax.plot([1950, 2025], [i, i], color=GRID, lw=7, solid_capstyle="butt")
        for j, (start, end) in enumerate(spans):
            segment_colour = CORAL if name.startswith("Population") and j == 1 else colour
            ax.plot([start, end], [i, i], color=segment_colour, lw=7, solid_capstyle="butt")
    ax.axvline(2021.5, color=INK, lw=0.6, ls="--")
    ax.text(2021.2, 3.42, "fv2", ha="right", va="bottom", color=TEAL, fontsize=5.2)
    ax.text(2022.0, 3.42, "fv1", ha="left", va="bottom", color=CORAL, fontsize=5.2)
    ax.set_xlim(1948, 2027)
    ax.set_ylim(-0.6, 3.65)
    ax.set_yticks(range(4), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xticks([1950, 1975, 2000, 2025])
    ax.set_xlabel("PECD v4.2 calendar-year coverage")
    finish(ax, "x")
    label_panel(ax, "a", -0.12, 1.05)

    ax = fig.add_subplot(gs[0, 6:])
    ax.axis("off")
    headers = ["field", "technology", "space", "resource / version"]
    table_rows = [
        ("onshore", "35", "PEON", "ReGrB / fv1"),
        ("offshore", "21", "PEOF", "ReGrB / fv1"),
        ("solar", "62", "PEON", "fixed utility / fv1"),
        ("temperature", "--", "SZON", "population weighted / fv2+fv1"),
    ]
    col_x = [0.02, 0.30, 0.48, 0.64]
    widths = [0.27, 0.17, 0.15, 0.35]
    for x, width, header in zip(col_x, widths, headers):
        ax.add_patch(Rectangle((x, 0.80), width, 0.16, transform=ax.transAxes,
                               facecolor=INK, edgecolor="white", lw=0.5))
        ax.text(x + 0.01, 0.88, header, transform=ax.transAxes, color="white",
                fontweight="bold", ha="left", va="center", fontsize=5.2)
    row_colours = [BLUE, PURPLE, GOLD, TEAL]
    for i, (row, colour) in enumerate(zip(table_rows, row_colours)):
        y = 0.62 - i * 0.18
        for x, width, value in zip(col_x, widths, row):
            ax.add_patch(Rectangle((x, y), width, 0.16, transform=ax.transAxes,
                                   facecolor=mpl.colors.to_rgba(colour, 0.10),
                                   edgecolor="white", lw=0.5))
            ax.text(x + 0.01, y + 0.08, value, transform=ax.transAxes, color=INK,
                    ha="left", va="center", fontsize=5.1)
    label_panel(ax, "b", -0.08, 1.02)

    ax = fig.add_subplot(gs[1, :7])
    weights = weights.copy()
    cap_cols = ["onshore_capacity_mw", "offshore_capacity_mw", "solar_capacity_mw"]
    weights["total"] = weights[cap_cols].sum(axis=1)
    top = weights.nlargest(11, "total").copy()
    other = weights.loc[~weights.country.isin(top.country), cap_cols].sum()
    other_row = {"country": "Other 22", **other.to_dict(), "total": float(other.sum())}
    plot = pd.concat([top, pd.DataFrame([other_row])], ignore_index=True)
    y = np.arange(len(plot))
    left = np.zeros(len(plot))
    for column, colour, name in zip(cap_cols, [BLUE, PURPLE, GOLD],
                                    ["onshore", "offshore", "solar"]):
        values = plot[column].to_numpy() / 1000
        ax.barh(y, values, left=left, color=colour, height=0.68, label=name,
                edgecolor="white", lw=0.25)
        left += values
    ax.set_yticks(y, plot.country)
    ax.invert_yaxis()
    ax.set_xlabel("Fixed ERAA 2030 VRE capacity weight (GW)")
    ax.legend(ncol=3, loc="lower right", bbox_to_anchor=(1.0, 1.01), frameon=False)
    finish(ax, "x")
    label_panel(ax, "c", -0.14, 1.02)

    ax = fig.add_subplot(gs[1, 8:])
    method_names = {
        "matched_zone_weighted": "direct zone match",
        "country_total_distributed_to_pecd_regions": "country-balance allocation",
        "zero_capacity": "zero capacity",
    }
    inventory["method_label"] = inventory.aggregation_method.map(method_names).fillna(inventory.aggregation_method)
    country_methods = inventory[["product", "country", "method_label"]].drop_duplicates()
    counts = country_methods.groupby(["product", "method_label"]).size().unstack(fill_value=0)
    products = [p for p in ["onshore", "offshore", "solar"] if p in counts.index]
    method_order = [m for m in method_names.values() if m in counts.columns]
    colours = [TEAL, CORAL, GREY]
    y = np.arange(len(products))
    left = np.zeros(len(products))
    for method, colour in zip(method_order, colours):
        values = counts.reindex(products)[method].to_numpy()
        ax.barh(y, values, left=left, color=colour, height=0.58, label=method)
        for yi, start, value in zip(y, left, values):
            if value >= 3:
                ax.text(start + value / 2, yi, str(int(value)), ha="center", va="center",
                        color="white", fontsize=5.0, fontweight="bold")
        left += values
    ax.set_yticks(y, products)
    ax.invert_yaxis()
    ax.set_xlabel("Countries")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.03), ncol=2, frameon=False)
    finish(ax, "x")
    label_panel(ax, "d", -0.24, 1.02)

    ax = fig.add_subplot(gs[2, :6])
    q = matching.copy()
    y = np.arange(len(q))
    overall = 100 * q.matched_weight_share.to_numpy()
    minimum = 100 * q.minimum_positive_country_weight_share.to_numpy()
    ax.hlines(y, 95, overall, color=mpl.colors.to_rgba(TEAL, 0.35), lw=5)
    ax.scatter(overall, y, s=28, color=TEAL, edgecolor="white", lw=0.5, label="all weights")
    ax.scatter(minimum, y, s=23, marker="D", color=CORAL, edgecolor="white", lw=0.45,
               label="minimum country")
    for value, yi in zip(overall, y):
        ax.text(value - 0.08, yi, f"{value:.1f}%", ha="right", va="center",
                color=TEAL, fontsize=5.1, fontweight="bold")
    short_products = q["product"].str.replace("temperature_", "temp ", regex=False)
    ax.set_yticks(y, short_products)
    ax.invert_yaxis()
    ax.set_xlim(95, 100.35)
    ax.set_xlabel("ERAA weight represented in PECD zones (%)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2, frameon=False)
    finish(ax, "x")
    label_panel(ax, "e", -0.12, 1.03)

    ax = fig.add_subplot(gs[2, 6:])
    ax.axis("off")
    stages = [
        (0.02, 0.54, 0.25, 0.28, "76 calendar years", "1950--2025"),
        (0.38, 0.54, 0.25, 0.28, "July--June cut", "leap day removed"),
        (0.74, 0.54, 0.24, 0.28, "75 operating years", "1950/51--2024/25"),
    ]
    for i, (x, y0, w, h, title, detail) in enumerate(stages):
        colour = [BLUE, GOLD, TEAL][i]
        ax.add_patch(Rectangle((x, y0), w, h, transform=ax.transAxes,
                               facecolor=mpl.colors.to_rgba(colour, 0.13),
                               edgecolor=colour, lw=0.9))
        ax.text(x + w / 2, y0 + 0.18, title, transform=ax.transAxes, ha="center",
                va="center", fontweight="bold", color=colour, fontsize=5.5)
        ax.text(x + w / 2, y0 + 0.08, detail, transform=ax.transAxes, ha="center",
                va="center", color=INK, fontsize=5.0)
        if i < 2:
            ax.annotate("", xy=(stages[i + 1][0] - 0.015, y0 + h / 2),
                        xytext=(x + w + 0.015, y0 + h / 2), xycoords=ax.transAxes,
                        arrowprops={"arrowstyle": "-|>", "lw": 0.9, "color": INK})
    checks = ["8,760 h per window", "92 event days", "task/recharge fixed"]
    for i, text in enumerate(checks):
        x = [0.08, 0.43, 0.74][i]
        ax.scatter(x, 0.25, s=25, color=[BLUE, GOLD, TEAL][i], transform=ax.transAxes,
                   edgecolor="white", lw=0.4)
        ax.text(x + 0.035, 0.25, text, transform=ax.transAxes, ha="left", va="center",
                color=INK, fontsize=5.1)
    ax.text(0.02, 0.05, "Preparation QA: pass", transform=ax.transAxes, ha="left",
            va="bottom", color=TEAL, fontsize=5.4, fontweight="bold")
    label_panel(ax, "f", -0.08, 1.03)

    inventory.to_csv(SOURCE / "extended_data_figure2_zone_inventory.csv", index=False)
    weights.to_csv(SOURCE / "extended_data_figure2_capacity_weights.csv", index=False)
    matching.to_csv(SOURCE / "extended_data_figure2_weight_matching_qa.csv", index=False)
    stem = OUT / "ed_fig_v57_pecd_inputs_operating_years"
    save_bundle(fig, stem)
    plt.close(fig)


def ed3_weather_cost_ledger() -> None:
    cost = pd.read_csv(RESULTS / "pecd_v42_focal_cost_path_2025_2035.csv")
    values = pd.read_csv(RESULTS / "pecd_v42_75_window_relative_values_by_cost_year.csv")
    decades = pd.read_csv(RESULTS / "pecd_v42_65_matched_decade_transitions.csv")
    cost_summary = pd.read_csv(RESULTS / "pecd_v42_75_window_cost_year_summary.csv")
    pairs = pd.read_csv(RESULTS / "pecd_v42_75_window_all_pairs.csv.gz")
    threshold = float(cost.relative_value_shift_from_2025_eur_kw_year.iloc[-1])

    fig = plt.figure(figsize=(7.20, 5.98))
    gs = fig.add_gridspec(
        3, 12, left=0.075, right=0.985, bottom=0.07, top=0.95,
        hspace=0.66, wspace=1.05, height_ratios=[0.94, 1.15, 1.05],
    )

    ax = fig.add_subplot(gs[0, :7])
    ax.plot(cost.learning_year, cost.caes_24h, color=CORAL, marker="o", lw=1.6,
            ms=4.2, label="24 h A-CAES")
    ax.plot(cost.learning_year, cost.li_lfp_4h, color=BLUE, marker="s", lw=1.6,
            ms=3.8, label="additional 4 h LFP")
    for row in cost.itertuples():
        ax.text(row.learning_year, row.caes_24h + 1.7, f"{row.caes_24h:.2f}",
                ha="center", va="bottom", color=CORAL, fontsize=5.1)
        ax.text(row.learning_year, row.li_lfp_4h - 1.7, f"{row.li_lfp_4h:.2f}",
                ha="center", va="top", color=BLUE, fontsize=5.1)
    ax.set_xticks(cost.learning_year)
    ax.set_ylabel("Annualized fixed cost\n(EUR kW$^{-1}$ yr$^{-1}$)")
    ax.set_xlabel("Technology cost year")
    ax.set_ylim(8, 63)
    ax.legend(loc="center right", frameon=False)
    finish(ax, "y")
    label_panel(ax, "a", -0.15, 1.04)

    ax = fig.add_subplot(gs[0, 8:])
    movements = pd.DataFrame({
        "interval": ["2025--2030", "2030--2035", "2025--2035"],
        "movement": [
            float(cost.relative_value_shift_from_2025_eur_kw_year.iloc[1]),
            float(cost.relative_value_shift_from_2025_eur_kw_year.iloc[2] - cost.relative_value_shift_from_2025_eur_kw_year.iloc[1]),
            threshold,
        ],
    })
    colours = [BLUE, GOLD, PURPLE]
    y = np.arange(3)
    ax.hlines(y, 0, movements.movement, color=colours, lw=5, alpha=0.85)
    ax.scatter(movements.movement, y, s=35, color=colours, edgecolor="white", lw=0.5)
    for yi, value, colour in zip(y, movements.movement, colours):
        ax.text(value + 0.12, yi, f"{value:.3f}", ha="left", va="center",
                color=colour, fontweight="bold", fontsize=5.2)
    ax.set_yticks(y, movements.interval)
    ax.invert_yaxis()
    ax.set_xlim(0, 6.1)
    ax.set_xlabel("Relative cost improvement\n(EUR kW$^{-1}$ yr$^{-1}$)")
    finish(ax, "x")
    label_panel(ax, "b", -0.25, 1.04)

    ax = fig.add_subplot(gs[1, :5])
    positions = np.arange(3)
    year_colours = [GREY, BLUE, TEAL]
    for pos, year, colour in zip(positions, [2025, 2030, 2035], year_colours):
        vals = values.loc[values.cost_year.eq(year), "caes24_minus_lfp4_eur_kw_year"].to_numpy()
        violin = ax.violinplot([vals], positions=[pos], widths=0.72, showextrema=False)
        body = violin["bodies"][0]
        body.set_facecolor(mpl.colors.to_rgba(colour, 0.25))
        body.set_edgecolor(colour)
        body.set_linewidth(0.8)
        jitter = np.linspace(-0.14, 0.14, len(vals))
        ax.scatter(pos + jitter, np.sort(vals), s=6, color=mpl.colors.to_rgba(colour, 0.35),
                   edgecolor="none", rasterized=True)
        q1, med, q3 = np.quantile(vals, [0.25, 0.5, 0.75])
        ax.plot([pos, pos], [q1, q3], color=colour, lw=3, solid_capstyle="round")
        ax.scatter(pos, med, s=31, color=colour, edgecolor="white", lw=0.5, zorder=4)
        ax.text(pos, vals.max() + 0.55, "0/75", ha="center", va="bottom",
                color=colour, fontsize=5.2, fontweight="bold")
    ax.axhline(0, color=INK, ls="--", lw=0.7)
    ax.set_xticks(positions, ["2025", "2030", "2035"])
    ax.set_ylabel("24 h A-CAES relative value\n(EUR kW$^{-1}$ yr$^{-1}$)")
    ax.set_xlabel("Technology cost year")
    finish(ax, "y")
    label_panel(ax, "c", -0.20, 1.03)

    ax = fig.add_subplot(gs[1, 5:])
    base = values.loc[values.cost_year.eq(2025)].sort_values("window_start_year")
    x = base.window_start_year.to_numpy()
    yv = base.caes24_minus_lfp4_eur_kw_year.to_numpy()
    q05, q95 = np.quantile(yv, [0.05, 0.95])
    ax.axhspan(q05, q95, color=mpl.colors.to_rgba(TEAL, 0.10), lw=0)
    ax.plot(x, yv, color=TEAL, lw=1.1, marker="o", ms=2.4,
            markeredgecolor="white", markeredgewidth=0.25)
    min_row = base.loc[base.caes24_minus_lfp4_eur_kw_year.idxmin()]
    max_row = base.loc[base.caes24_minus_lfp4_eur_kw_year.idxmax()]
    ax.scatter([min_row.window_start_year, max_row.window_start_year],
               [min_row.caes24_minus_lfp4_eur_kw_year, max_row.caes24_minus_lfp4_eur_kw_year],
               s=32, color=[CORAL, SAGE], edgecolor="white", lw=0.5, zorder=5)
    full_span = float(yv.max() - yv.min())
    central_span = float(q95 - q05)
    ax.text(0.02, 0.04, f"full span {full_span:.2f}  |  5--95% span {central_span:.2f}",
            transform=ax.transAxes, ha="left", va="bottom", color=INK,
            fontsize=5.4, fontweight="bold")
    ax.set_xlabel("July--June operating-year start")
    ax.set_ylabel("2025 relative value")
    finish(ax, "y")
    label_panel(ax, "d", -0.14, 1.03)

    ax = fig.add_subplot(gs[2, :7])
    frame = decades.sort_values("start_year")
    xx = np.arange(len(frame))
    changes = frame.weather_value_change_eur_kw_year.to_numpy()
    colours = np.where(changes <= -threshold, CORAL,
                       np.where(changes >= threshold, SAGE, BLUE))
    ax.vlines(xx, 0, changes, color=mpl.colors.to_rgba(BLUE, 0.45), lw=0.8)
    ax.scatter(xx, changes, s=14, color=colours, edgecolor="white", lw=0.35, zorder=4)
    ax.axhline(0, color=INK, lw=0.6)
    ax.axhline(-threshold, color=PURPLE, lw=0.75, ls="--")
    ax.axhline(threshold, color=PURPLE, lw=0.75, ls=":")
    worst = frame.loc[frame.weather_value_change_eur_kw_year.idxmin()]
    wi = int(frame.index.get_loc(worst.name))
    ax.annotate(
        f"{worst.start_window} to {worst.end_window}\n{worst.weather_value_change_eur_kw_year:.2f}; net {worst.net_2025_to_2035_value_change_eur_kw_year:.2f}",
        xy=(wi, worst.weather_value_change_eur_kw_year), xytext=(wi + 3, -7.9),
        fontsize=5.1, color=CORAL, ha="left", va="bottom",
        arrowprops={"arrowstyle": "->", "lw": 0.7, "color": CORAL},
    )
    ticks = np.arange(0, len(frame), 12)
    ax.set_xticks(ticks, frame.start_window.iloc[ticks], rotation=35, ha="right")
    ax.set_ylabel("Weather-value change\n(EUR kW$^{-1}$ yr$^{-1}$)")
    ax.set_xlabel("Time-matched ten-year transition")
    ax.text(0.99, 0.95, "3/65 adverse transitions", transform=ax.transAxes,
            ha="right", va="top", color=CORAL, fontsize=5.4, fontweight="bold")
    finish(ax, "y")
    label_panel(ax, "e", -0.15, 1.03)

    ax = fig.add_subplot(gs[2, 8:])
    pair_hit_column = next(c for c in pairs.columns if "reach" in c.lower() or "parity" in c.lower())
    pair_hits = int(pairs[pair_hit_column].astype(bool).sum())
    endpoints = pd.DataFrame({
        "endpoint": ["all natural pairs", "adverse decades", "2025 crossover", "2030 crossover", "2035 crossover"],
        "hits": [pair_hits, int(frame.erases_2025_2035_learning.sum()), 0, 0, 0],
        "total": [len(pairs), len(frame), 75, 75, 75],
    })
    endpoints["percent"] = 100 * endpoints.hits / endpoints.total
    y = np.arange(len(endpoints))
    endpoint_colours = [CORAL, PURPLE, GREY, BLUE, TEAL]
    ax.hlines(y, 0, endpoints.percent, color=endpoint_colours, lw=5, alpha=0.75)
    ax.scatter(endpoints.percent, y, s=31, color=endpoint_colours,
               edgecolor="white", lw=0.5, zorder=4)
    for yi, row, colour in zip(y, endpoints.itertuples(), endpoint_colours):
        ax.text(row.percent + 0.18, yi, f"{row.hits}/{row.total} ({row.percent:.1f}%)",
                ha="left", va="center", color=colour, fontsize=5.0, fontweight="bold")
    ax.set_yticks(y, endpoints.endpoint)
    ax.invert_yaxis()
    ax.set_xlim(0, 8.1)
    ax.set_xlabel("Archived comparisons reaching endpoint (%)")
    finish(ax, "x")
    label_panel(ax, "f", -0.30, 1.03)

    cost.to_csv(SOURCE / "extended_data_figure3_cost_path.csv", index=False)
    values.to_csv(SOURCE / "extended_data_figure3_weather_values_by_cost_year.csv", index=False)
    decades.to_csv(SOURCE / "extended_data_figure3_matched_decades.csv", index=False)
    cost_summary.to_csv(SOURCE / "extended_data_figure3_cost_year_summary.csv", index=False)
    endpoints.to_csv(SOURCE / "extended_data_figure3_endpoint_accounting.csv", index=False)
    stem = OUT / "ed_fig_v57_natural_weather_cost_ledger"
    save_bundle(fig, stem)
    plt.close(fig)


def main() -> None:
    configure()
    ed2_inputs_and_construction()
    ed3_weather_cost_ledger()
    print(OUT / "ed_fig_v57_pecd_inputs_operating_years.pdf")
    print(OUT / "ed_fig_v57_natural_weather_cost_ledger.pdf")


if __name__ == "__main__":
    main()
