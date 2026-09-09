#!/usr/bin/env python3
"""Plot three-regime cross-duration heterogeneity for Extended Data Figure 7."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
FIGURE_WORK = HERE.parent
sys.path.insert(0, str(FIGURE_WORK))

from figure2_relayout_v62.plot_preview_figure2_world_callout_relayout import (  # noqa: E402
    MM,
    REGIME_COLORS,
    configure_publication,
    draw_map_panel,
)
from v60_figure_style import GRID, INK, MUTED, PALE, panel_label, quiet  # noqa: E402


SOURCE = HERE / "source_data"
MAP_SOURCE = FIGURE_WORK / "source_data_v60"
OUT = HERE / "outputs"
WIDTH_MM = 183.0
HEIGHT_MM = 210.0
REGIMES = ["Europe", "China", "Japan"]
REGIME_LABELS = {"Europe": "NVSB", "China": "PLRP", "Japan": "CLDB"}
CLUSTERS = [2, 4, 6]
STATUS_ORDER = ["left_censored", "observed", "right_censored"]
STATUS_COLORS = {
    "left_censored": "#A9C8D3",
    "observed": "#4C8B82",
    "right_censored": "#D97A64",
}
STATUS_LABELS = {
    "left_censored": "positive by 24 h",
    "observed": "crosses within 24–168 h",
    "right_censored": "not positive by 168 h",
}


def export_figure(fig: plt.Figure) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "ed_fig_v62_three_regime_cross_duration_heterogeneity"
    fig.savefig(stem.with_suffix(".pdf"), metadata={"Creator": "Python/matplotlib"})
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=360)
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )


def grouped_positions() -> dict[tuple[str, int], float]:
    positions: dict[tuple[str, int], float] = {}
    for regime_index, regime in enumerate(REGIMES):
        base = regime_index * 4.0
        for cluster_index, cluster in enumerate(CLUSTERS):
            positions[(regime, cluster)] = base + cluster_index
    return positions


def decorate_group_axis(ax: plt.Axes, positions: dict[tuple[str, int], float]) -> None:
    ticks = [positions[(regime, cluster)] for regime in REGIMES for cluster in CLUSTERS]
    ax.set_yticks(ticks, [f"{cluster} d" for _regime in REGIMES for cluster in CLUSTERS])
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0, pad=2)
    for boundary in [2.95, 6.95]:
        ax.axhline(boundary, color=GRID, lw=0.55, zorder=0)
    for regime in REGIMES:
        centre = np.mean([positions[(regime, cluster)] for cluster in CLUSTERS])
        ax.text(
            -0.09,
            centre,
            REGIME_LABELS[regime],
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            color=REGIME_COLORS[regime],
            fontsize=5.8,
            fontweight="bold",
            clip_on=False,
        )


def draw_status(ax: plt.Axes, status: pd.DataFrame) -> None:
    positions = grouped_positions()
    for regime in REGIMES:
        for cluster in CLUSTERS:
            subset = status.loc[
                status["regime"].eq(regime)
                & status["target_cluster_days"].eq(cluster)
            ].set_index("crossing_status")
            left = 0.0
            y = positions[(regime, cluster)]
            for crossing_status in STATUS_ORDER:
                share = float(subset.loc[crossing_status, "share_pct"]) if crossing_status in subset.index else 0.0
                ax.barh(
                    y,
                    share,
                    left=left,
                    height=0.62,
                    color=STATUS_COLORS[crossing_status],
                    edgecolor="white",
                    lw=0.45,
                )
                if share >= 15:
                    ax.text(
                        left + share / 2,
                        y,
                        f"{share:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=5.0,
                        color="white" if crossing_status != "left_censored" else INK,
                        fontweight="bold",
                    )
                left += share
    decorate_group_axis(ax, positions)
    ax.set_xlim(0, 100)
    ax.set_xlabel("system–years (%)")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.40)
    handles = [
        mpl.patches.Patch(facecolor=STATUS_COLORS[key], label=STATUS_LABELS[key])
        for key in STATUS_ORDER
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=3,
        frameon=False,
        handlelength=1.0,
        columnspacing=0.9,
        fontsize=5.0,
    )
    panel_label(ax, "b", -0.22, 1.08)


def draw_crossing_distribution(ax: plt.Axes, records: pd.DataFrame) -> None:
    positions = grouped_positions()
    rng = np.random.default_rng(6207)
    for regime in REGIMES:
        colour = REGIME_COLORS[regime]
        for cluster in CLUSTERS:
            y = positions[(regime, cluster)]
            subset = records.loc[
                records["regime"].eq(regime)
                & records["target_cluster_days"].eq(cluster)
            ]
            observed = subset.loc[subset["crossing_status"].eq("observed"), "crossing_duration_h"].to_numpy(float)
            if len(observed):
                jitter = rng.uniform(-0.18, 0.18, len(observed))
                ax.scatter(
                    observed,
                    y + jitter,
                    s=7.5,
                    facecolor=mpl.colors.to_rgba(colour, 0.34),
                    edgecolor="none",
                    rasterized=True,
                    zorder=2,
                )
                q25, median, q75 = np.quantile(observed, [0.25, 0.5, 0.75])
                ax.plot([q25, q75], [y, y], color=colour, lw=2.0, solid_capstyle="round", zorder=4)
                ax.scatter(median, y, s=22, marker="D", color=colour, edgecolor="white", lw=0.45, zorder=5)
            left_count = int(subset["crossing_status"].eq("left_censored").sum())
            right_count = int(subset["crossing_status"].eq("right_censored").sum())
            if left_count:
                ax.scatter(24, y, s=19 + left_count, marker="<", facecolor="white", edgecolor=colour, lw=0.8, zorder=6)
            if right_count:
                ax.scatter(168, y, s=19 + right_count, marker=">", facecolor="white", edgecolor=colour, lw=0.8, zorder=6)
    decorate_group_axis(ax, positions)
    ax.set_xlim(20, 172)
    ax.set_xticks([24, 48, 72, 96, 120, 144, 168])
    ax.set_xlabel("first sustained-positive duration (h)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.40)
    ax.plot([], [], marker="D", ms=3.5, color=MUTED, lw=1.7, label="median and IQR")
    ax.plot([], [], marker="<", ms=3.5, mfc="white", mec=MUTED, lw=0, label="positive by 24 h")
    ax.plot([], [], marker=">", ms=3.5, mfc="white", mec=MUTED, lw=0, label="not positive by 168 h")
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=3,
        frameon=False,
        handlelength=1.0,
        columnspacing=0.8,
        fontsize=5.0,
    )
    panel_label(ax, "c", -0.20, 1.08)


def draw_regime_relation(ax: plt.Axes, records: pd.DataFrame, regime: str, letter: str) -> None:
    frame = records.loc[
        records["regime"].eq(regime)
        & records["target_cluster_days"].eq(4)
    ].copy()
    colour = REGIME_COLORS[regime]
    observed = frame.loc[frame["crossing_status"].eq("observed")].copy()
    rng = np.random.default_rng(6207 + REGIMES.index(regime))
    x = observed["event_run_days"].to_numpy(float)
    y = observed["crossing_duration_h"].to_numpy(float)
    ax.scatter(
        x + rng.uniform(-0.12, 0.12, len(x)),
        y + rng.uniform(-1.5, 1.5, len(y)),
        s=13,
        facecolor=mpl.colors.to_rgba(colour, 0.48),
        edgecolor="white",
        lw=0.35,
        rasterized=True,
        zorder=3,
    )
    if len(observed) >= 3 and np.ptp(x) > 0:
        fit = np.polyfit(x, y, 1)
        line_x = np.linspace(x.min(), x.max(), 80)
        ax.plot(line_x, np.polyval(fit, line_x), color=colour, lw=0.95, zorder=4)
    left = frame.loc[frame["crossing_status"].eq("left_censored")]
    right = frame.loc[frame["crossing_status"].eq("right_censored")]
    if len(left):
        ax.scatter(left["event_run_days"], np.full(len(left), 24.0), marker="v", s=23, facecolor="white", edgecolor=colour, lw=0.75, zorder=5)
    if len(right):
        ax.scatter(right["event_run_days"], np.full(len(right), 168.0), marker="^", s=23, facecolor="white", edgecolor=colour, lw=0.75, zorder=5)
    rho = observed[["event_run_days", "crossing_duration_h"]].corr(method="spearman").iloc[0, 1]
    ax.text(0.04, 0.94, REGIME_LABELS[regime], transform=ax.transAxes, ha="left", va="top", color=colour, fontsize=6.2, fontweight="bold")
    ax.text(0.96, 0.94, rf"$\rho$ = {rho:.2f}", transform=ax.transAxes, ha="right", va="top", color=INK, fontsize=5.5)
    ax.set_ylim(20, 174)
    ax.set_yticks([24, 48, 72, 96, 120, 144, 168])
    ax.set_xlabel("longest observed event (d)")
    if regime == "Europe":
        ax.set_ylabel("four-day crossing duration (h)")
    else:
        ax.set_yticklabels([])
    ax.grid(color=GRID, lw=0.40)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    panel_label(ax, letter, -0.15, 1.04)


def main() -> None:
    configure_publication()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.labelsize": 5.8,
            "xtick.labelsize": 5.0,
            "ytick.labelsize": 5.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    records = pd.read_csv(SOURCE / "ed7_cross_duration_records.csv")
    status = pd.read_csv(SOURCE / "ed7_crossing_status.csv")
    europe = pd.read_csv(MAP_SOURCE / "figure2a_europe_map.csv")
    china = pd.read_csv(MAP_SOURCE / "figure2b_china_map.csv")
    japan = pd.read_csv(MAP_SOURCE / "figure2c_japan_map.csv")

    fig = plt.figure(figsize=(WIDTH_MM * MM, HEIGHT_MM * MM))
    grid = fig.add_gridspec(
        3,
        12,
        left=0.085,
        right=0.975,
        bottom=0.055,
        top=0.982,
        height_ratios=[1.30, 0.63, 0.60],
        hspace=0.24,
        wspace=0.92,
    )
    draw_map_panel(fig, grid[0, :], europe, china, japan)
    draw_status(fig.add_subplot(grid[1, 0:5]), status)
    draw_crossing_distribution(fig.add_subplot(grid[1, 6:12]), records)
    draw_regime_relation(fig.add_subplot(grid[2, 0:4]), records, "Europe", "d")
    draw_regime_relation(fig.add_subplot(grid[2, 4:8]), records, "China", "e")
    draw_regime_relation(fig.add_subplot(grid[2, 8:12]), records, "Japan", "f")
    export_figure(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
