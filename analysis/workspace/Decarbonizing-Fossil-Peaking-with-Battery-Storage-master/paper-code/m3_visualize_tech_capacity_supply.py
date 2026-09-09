"""
Standalone technology-faceted bubble figure for M3 dispatch results.

This script creates a new figure without overwriting fig3_m3_dispatch_nature.
Each subplot corresponds to one storage technology. Within each technology,
provinces are sorted by that technology's maximum optimal capacity across
available durations. Bubble size encodes optimal capacity, and bubble colour
encodes maximum supply ratio.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from m3_visualize_nature import (
    DURATION_ORDER,
    FIGURE_DIR,
    PALETTE,
    PROVINCE_SHORT_LABELS,
    SOURCE_DIR,
    TECH_LABELS,
    TECH_ORDER,
    build_annual_curtailment_table,
    build_best_combo_table,
    load_inputs,
)


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.size"] = 8.0
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.85
plt.rcParams["legend.frameon"] = False


FIG_BASE = FIGURE_DIR / "fig3b_tech_capacity_supply_matrix"
SUPPLY_CONTRAST_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "nature_supply_ratio",
    [
        "#A9343A",  # low supply: high-risk red
        "#E05D3C",  # constrained supply: pipeline orange
        "#EFEAE1",  # transition: warm neutral
        "#7EA9C9",  # high supply: pale operating blue
        "#2F78A8",  # near-saturated supply: operating blue
    ],
)
CAPACITY_BINS_GWH = [0.0, 10.0, 50.0, 150.0, 500.0, np.inf]
CAPACITY_BIN_LABELS = ["<10 GWh", "10-50 GWh", "50-150 GWh", "150-500 GWh", ">=500 GWh"]
CAPACITY_BIN_MARKERS = ["o", "^", "s", "D", "h"]
CAPACITY_BIN_SIZES = [28.0, 52.0, 86.0, 132.0, 205.0]


def build_technology_orders(best_combo: pd.DataFrame) -> tuple[dict[str, list[str]], pd.DataFrame]:
    """Return province order for each technology by optimal capacity."""
    orders: dict[str, list[str]] = {}
    rows = []

    for tech_id in TECH_ORDER:
        tech_rows = best_combo[best_combo["tech_id"] == tech_id].copy()
        if tech_rows.empty:
            orders[tech_id] = []
            continue

        summary = (
            tech_rows.groupby(["province_en", "province_cn"], as_index=False)
            .agg(
                max_optimal_capacity_mwh=("capacity_mwh", "max"),
                median_optimal_capacity_mwh=("capacity_mwh", "median"),
                max_supply_ratio_pct=("satisfy_rate_pct", "max"),
            )
            .sort_values(
                ["max_optimal_capacity_mwh", "median_optimal_capacity_mwh", "province_en"],
                ascending=[False, False, True],
            )
            .reset_index(drop=True)
        )
        summary.insert(0, "tech_id", tech_id)
        summary.insert(1, "technology", TECH_LABELS[tech_id])
        summary["rank_within_technology"] = np.arange(1, len(summary) + 1)
        rows.append(summary)
        orders[tech_id] = summary["province_en"].tolist()

    order_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return orders, order_df


def capacity_to_area(capacity_gwh: np.ndarray | float, max_capacity_gwh: float) -> np.ndarray | float:
    """Map capacity in GWh to bubble area."""
    if max_capacity_gwh <= 0:
        return np.asarray(capacity_gwh) * 0.0 + 10.0
    return 7.0 + 360.0 * np.power(np.asarray(capacity_gwh) / max_capacity_gwh, 0.68)


def assign_capacity_bin(capacity_gwh: pd.Series) -> pd.Series:
    """Assign capacity classes used for marker shape."""
    return pd.cut(
        capacity_gwh,
        bins=CAPACITY_BINS_GWH,
        labels=range(len(CAPACITY_BIN_LABELS)),
        include_lowest=True,
        right=False,
    ).astype(int)


def add_capacity_legend(fig: plt.Figure, max_capacity_gwh: float) -> None:
    legend_examples = [5, 25, 90, 250, 650]
    handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="none",
            markerfacecolor="#B8A86A",
            markeredgecolor="white",
            markeredgewidth=0.50,
            alpha=0.90,
            markersize=float(np.sqrt(size)),
            label=label,
        )
        for marker, size, label in zip(
            CAPACITY_BIN_MARKERS,
            CAPACITY_BIN_SIZES,
            CAPACITY_BIN_LABELS,
        )
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.41, 0.018),
        ncol=len(legend_examples),
        title="Optimal capacity class",
        fontsize=6.7,
        title_fontsize=7.0,
        columnspacing=0.95,
        handletextpad=0.45,
    )


def plot_technology_panel(
    ax: plt.Axes,
    tech_id: str,
    best_combo: pd.DataFrame,
    province_order: list[str],
    norm: mcolors.Normalize,
    max_capacity_gwh: float,
) -> None:
    tech_rows = best_combo[best_combo["tech_id"] == tech_id].copy()
    tech_rows["duration_h"] = tech_rows["duration_h"].astype(int)
    tech_rows["capacity_gwh"] = tech_rows["capacity_mwh"] / 1000.0
    tech_rows["capacity_bin"] = assign_capacity_bin(tech_rows["capacity_gwh"])

    durations = [
        duration_h
        for duration_h in DURATION_ORDER
        if duration_h in set(tech_rows["duration_h"])
    ]
    duration_to_y = {duration_h: idx for idx, duration_h in enumerate(durations)}
    province_to_x = {province: idx for idx, province in enumerate(province_order)}

    tech_rows["x_pos"] = tech_rows["province_en"].map(province_to_x)
    tech_rows["y_pos"] = tech_rows["duration_h"].map(duration_to_y)
    tech_rows = tech_rows.dropna(subset=["x_pos", "y_pos"]).copy()

    for x in range(len(province_order)):
        ax.axvline(x, color="#F0F0F0", lw=0.30, zorder=0)
    for y in range(len(durations)):
        ax.axhline(y, color="#E6E6E6", lw=0.45, zorder=0)

    for bin_id, marker in enumerate(CAPACITY_BIN_MARKERS):
        marker_rows = tech_rows[tech_rows["capacity_bin"] == bin_id]
        if marker_rows.empty:
            continue
        ax.scatter(
            marker_rows["x_pos"],
            marker_rows["y_pos"],
            s=CAPACITY_BIN_SIZES[bin_id],
            c=marker_rows["satisfy_rate_pct"],
            cmap=SUPPLY_CONTRAST_CMAP,
            norm=norm,
            marker=marker,
            alpha=0.90,
            edgecolor="white",
            linewidth=0.34,
            zorder=3,
        )

    ax.set_xlim(-0.65, len(province_order) - 0.35)
    ax.set_ylim(-0.55, len(durations) - 0.55)
    ax.set_xticks(np.arange(len(province_order)))
    ax.set_xticklabels(
        [PROVINCE_SHORT_LABELS.get(province, province) for province in province_order],
        fontsize=7.2,
        rotation=90,
        ha="center",
        va="top",
    )
    ax.set_yticks(range(len(durations)))
    ax.set_yticklabels([str(duration_h) for duration_h in durations], fontsize=6.0)
    ax.set_title(
        TECH_LABELS[tech_id],
        fontsize=8.0,
        color=PALETTE["neutral_black"],
        fontweight="semibold",
        pad=3.0,
    )
    ax.tick_params(axis="x", width=0.45, length=1.6, pad=1.0)
    ax.tick_params(axis="y", width=0.55, length=2.0, pad=1.2)

    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(0.72)


def main(output_stem: str | None = None, source_prefix: str = "fig3b") -> pd.DataFrame:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    fig_base = FIGURE_DIR / output_stem if output_stem is not None else FIG_BASE

    results, curtailment = load_inputs()
    annual_df = build_annual_curtailment_table(results, curtailment)
    best_combo = build_best_combo_table(results, annual_df)
    tech_orders, order_df = build_technology_orders(best_combo)

    plot_data = best_combo.copy()
    plot_data["capacity_gwh"] = plot_data["capacity_mwh"] / 1000.0
    plot_data.to_csv(SOURCE_DIR / f"{source_prefix}_source_tech_capacity_supply_points.csv", index=False)
    order_df.to_csv(SOURCE_DIR / f"{source_prefix}_source_province_order_by_technology.csv", index=False)

    max_capacity_gwh = float(plot_data["capacity_gwh"].max())
    norm = mcolors.Normalize(vmin=0.0, vmax=100.0)

    fig, axes = plt.subplots(4, 2, figsize=(15.2, 14.8))
    axes_flat = axes.ravel()

    for ax, tech_id in zip(axes_flat, TECH_ORDER):
        plot_technology_panel(
            ax=ax,
            tech_id=tech_id,
            best_combo=best_combo,
            province_order=tech_orders[tech_id],
            norm=norm,
            max_capacity_gwh=max_capacity_gwh,
        )

    for ax in axes[-1, :]:
        ax.set_xlabel("Province sorted by optimal capacity", fontsize=7.0, labelpad=2.0)
    for ax in axes[:, 0]:
        ax.set_ylabel("Duration (h)", fontsize=7.0, labelpad=2.0)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=SUPPLY_CONTRAST_CMAP)
    sm.set_array([])
    cax = fig.add_axes([0.920, 0.270, 0.014, 0.46])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.tick_params(labelsize=6.4, width=0.55, length=2.2)
    cbar.outline.set_linewidth(0.5)
    cbar.set_label("Maximum supply ratio (%)", fontsize=7.0, labelpad=8.0)
    cbar.set_ticks([0, 25, 50, 75, 100])

    add_capacity_legend(fig, max_capacity_gwh)

    fig.suptitle(
        "Technology-specific optimal capacity and maximum supply ratio",
        x=0.02,
        y=0.988,
        ha="left",
        fontsize=10.0,
        fontweight="semibold",
        color=PALETTE["neutral_black"],
    )
    fig.subplots_adjust(left=0.060, right=0.900, top=0.958, bottom=0.145, wspace=0.045, hspace=0.52)

    fig.savefig(f"{fig_base}.svg", bbox_inches="tight")
    fig.savefig(f"{fig_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{fig_base}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {fig_base}.svg/pdf/png")
    print(f"Source data saved to: {SOURCE_DIR}")
    return plot_data


if __name__ == "__main__":
    main()
