#!/usr/bin/env python3
"""Standalone Figure 3 preview: chronology, recovery, cycling and value."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


HERE = Path(__file__).resolve().parent
FIGURE_WORK = HERE.parent
sys.path.insert(0, str(FIGURE_WORK))

from v60_figure_style import (  # noqa: E402
    BLUE,
    BLUE_LIGHT,
    CORAL,
    CORAL_LIGHT,
    GOLD,
    GRID,
    INK,
    MM,
    MUTED,
    PALE,
    REGIME_COLORS,
    REGIME_MARKERS,
    TEAL,
    configure,
    density_curve,
    deterministic_jitter,
    panel_label,
    quiet,
)


SOURCE = FIGURE_WORK / "source_data_v58"
PROFILE_SOURCE = HERE / "source_data"
OUT = HERE / "outputs"
WIDTH_MM = 183.0
HEIGHT_MM = 235.0

BASE_DISCHARGE = "#0F4D92"
TAIL_DISCHARGE = "#B64342"
BASE_CHARGE = "#B4C0E4"
TAIL_CHARGE = "#F6CFCB"
SOC_FILL = "#AEB6BC"
COMPONENT_COLORS = {
    "variable_om": "#8A9293",
    "charging": "#8E6AAE",
    "energy_service": "#55A58F",
    "firm_service": "#D2A33D",
}
REGIME_LABELS = {"Europe": "NVSB", "China": "PLRP", "Japan": "CLDB"}


def configure_publication() -> None:
    configure()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.3,
            "axes.labelsize": 6.4,
            "xtick.labelsize": 5.5,
            "ytick.labelsize": 5.5,
            "legend.fontsize": 5.3,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_bundle(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / stem
    fig.savefig(path.with_suffix(".pdf"), metadata={"Creator": "Python/matplotlib"})
    fig.savefig(path.with_suffix(".svg"))
    fig.savefig(path.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(path.with_suffix(".png"), dpi=360)
    rgba = plt.imread(path.with_suffix(".png"))
    grey = np.dot(rgba[..., :3], [0.2126, 0.7152, 0.0722])
    plt.imsave(
        path.with_name(path.name + "_grayscale").with_suffix(".png"),
        grey,
        cmap="gray",
        vmin=0,
        vmax=1,
    )


def elapsed_recovery(task: np.ndarray) -> np.ndarray:
    """Hours since the previous event day, capped at the analysis threshold."""
    event = np.asarray(task, float) > 1.0e-10
    elapsed = 0.0
    result = np.zeros(event.size, float)
    for index, active in enumerate(event):
        if active:
            elapsed = 0.0
        else:
            elapsed += 24.0
        result[index] = min(elapsed, 72.0)
    return result


def chronology_row(
    fig: plt.Figure,
    ax: plt.Axes,
    chronology: pd.DataFrame,
    name: str,
    colour: str,
    show_months: bool,
) -> None:
    days = chronology["day"].to_numpy(float) + 1.0
    task = chronology[f"{name}_task_normalized"].to_numpy(float)
    recharge = chronology[f"{name}_recharge_normalized"].to_numpy(float)
    recovery = elapsed_recovery(task)

    ax.fill_between(days, 0, task, color=CORAL_LIGHT, lw=0, alpha=0.95)
    ax.fill_between(days, 0, -recharge, color=BLUE_LIGHT, lw=0, alpha=0.98)
    ax.axhline(0, color=INK, lw=0.55)
    ax.set_xlim(1, 365)
    ax.set_ylim(-1.08, 1.08)
    ax.set_yticks([-0.75, 0, 0.75], ["recharge", "0", "task"])
    ax.tick_params(axis="y", length=0, pad=2.5)
    ax.spines[["left", "top", "right"]].set_visible(False)
    ax.text(
        0.008,
        0.86,
        name,
        transform=ax.transAxes,
        color=colour,
        fontweight="bold",
        ha="left",
        va="top",
    )

    recovery_ax = ax.twinx()
    recovery_ax.plot(days, recovery, color=INK, lw=0.52, alpha=0.72, zorder=4)
    recovery_ax.set_ylim(0, 76)
    recovery_ax.set_yticks([0, 72] if name == "clustered" else [])
    recovery_ax.tick_params(axis="y", colors=INK, length=1.8, pad=1.5)
    recovery_ax.spines[["left", "top", "bottom"]].set_visible(False)
    recovery_ax.spines["right"].set_color(INK)
    recovery_ax.spines["right"].set_linewidth(0.5)
    if name == "clustered":
        recovery_ax.set_ylabel("recovery opportunity (h)", rotation=270, labelpad=8.0)

    month_days = [1, 60, 121, 182, 244, 305, 365]
    month_labels = ["Jan", "Mar", "May", "Jul", "Sep", "Nov", "Dec"]
    ax.set_xticks(month_days)
    if show_months:
        ax.set_xticklabels(month_labels)
        ax.set_xlabel("day of operating year", labelpad=1.8)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)


def plot_chronology(fig: plt.Figure, spec, chronology: pd.DataFrame) -> None:
    nested = spec.subgridspec(2, 1, hspace=0.10)
    top = fig.add_subplot(nested[0, 0])
    bottom = fig.add_subplot(nested[1, 0], sharex=top)
    chronology_row(fig, top, chronology, "dispersed", BLUE, False)
    chronology_row(fig, bottom, chronology, "clustered", CORAL, True)
    top.tick_params(axis="x", labelbottom=False, length=0)
    panel_label(top, "a", -0.075, 1.10)

    legend = [
        Patch(facecolor=CORAL_LIGHT, edgecolor="none", label="balancing task"),
        Patch(facecolor=BLUE_LIGHT, edgecolor="none", label="available recharge"),
        Line2D([0], [0], color=INK, lw=0.7, alpha=0.72, label="elapsed recovery"),
    ]
    top.legend(
        handles=legend,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.55, 1.17),
        handlelength=1.4,
        columnspacing=1.1,
        borderaxespad=0,
    )


def maximum_activity_window(frame: pd.DataFrame, days: int = 14) -> pd.DataFrame:
    daily = (
        frame.assign(
            day=frame["model_hour"].to_numpy(int) // 24,
            storage_throughput=(
                frame["base_charge_mw"]
                + frame["tail_charge_mw"]
                + frame["base_discharge_mw"]
                + frame["tail_discharge_mw"]
            ),
        )
        .groupby("day", sort=True)["storage_throughput"]
        .sum()
    )
    end_day = int(daily.rolling(days, min_periods=days).sum().idxmax())
    start_hour = (end_day - days + 1) * 24
    result = frame.loc[
        frame["model_hour"].between(start_hour, start_hour + days * 24 - 1)
    ].copy()
    result["window_hour"] = np.arange(len(result), dtype=int)
    return result


def dispatch_small_multiple(
    ax: plt.Axes,
    frame: pd.DataFrame,
    power: float,
    regime: str,
    chronology: str,
    row: int,
) -> None:
    window = maximum_activity_window(frame, days=14)
    x = window["window_hour"].to_numpy(float)
    base_discharge = window["base_discharge_mw"].to_numpy(float) / power
    tail_discharge = window["tail_discharge_mw"].to_numpy(float) / power
    base_charge = window["base_charge_mw"].to_numpy(float) / power
    tail_charge = window["tail_charge_mw"].to_numpy(float) / power

    soc_ax = ax.twinx()
    soc_ax.fill_between(
        x,
        0,
        window["tail_soc_pct"].to_numpy(float),
        step="post",
        color=SOC_FILL,
        alpha=0.27,
        lw=0,
        zorder=0,
    )
    soc_ax.set_ylim(0, 102)
    soc_ax.set_zorder(ax.get_zorder() - 1)
    soc_ax.patch.set_visible(False)
    ax.patch.set_alpha(0)
    if row % 2 == 1:
        soc_ax.set_yticks([0, 50, 100])
    else:
        soc_ax.set_yticks([])
    soc_ax.spines[["left", "top", "bottom"]].set_visible(False)
    soc_ax.spines["right"].set_linewidth(0.5)

    ax.fill_between(
        x, 0, base_discharge, step="post", color=BASE_DISCHARGE, lw=0
    )
    ax.fill_between(
        x,
        base_discharge,
        base_discharge + tail_discharge,
        step="post",
        color=TAIL_DISCHARGE,
        lw=0,
    )
    ax.fill_between(
        x, 0, -base_charge, step="post", color=BASE_CHARGE, lw=0
    )
    ax.fill_between(
        x,
        -base_charge,
        -(base_charge + tail_charge),
        step="post",
        color=TAIL_CHARGE,
        lw=0,
    )
    ax.axhline(0, color=INK, lw=0.48)
    ax.set_xlim(0, 336)
    ax.set_ylim(-2.75, 2.20)
    ax.set_xticks(np.arange(0, 337, 24))
    ax.grid(axis="x", color="#E5E7E8", lw=0.35, zorder=-2)
    if row != 5:
        ax.tick_params(axis="x", labelbottom=False)
    if row == 5:
        ax.set_xlabel("hour within most active 14-day window", labelpad=1.8)
    ax.set_yticks([-2, 0, 2])
    chronology_label = (
        "dispersed"
        if chronology == "dispersed"
        else r"controlled clustering ($\leq$4 d)"
    )
    ax.text(
        0.004,
        0.96,
        f"{REGIME_LABELS[regime]} · {chronology_label}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="#4D4D4D",
        fontweight="bold",
        fontsize=5.4,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.80, "pad": 0.35},
        zorder=8,
    )
    quiet(ax)


def plot_dispatch_hero(
    fig: plt.Figure,
    spec,
    profiles: pd.DataFrame,
    selection: pd.DataFrame,
) -> None:
    nested = spec.subgridspec(
        8,
        1,
        hspace=0.22,
        height_ratios=[1, 1, 0.35, 1, 1, 0.35, 1, 1],
    )
    grid_rows = [0, 1, 3, 4, 6, 7]
    panel_order = [
        ("Europe", "dispersed"),
        ("Europe", "clustered"),
        ("China", "dispersed"),
        ("China", "clustered"),
        ("Japan", "dispersed"),
        ("Japan", "clustered"),
    ]
    axes: list[plt.Axes] = []
    for row, ((regime, chronology), grid_row) in enumerate(
        zip(panel_order, grid_rows)
    ):
        share = axes[0] if axes else None
        ax = fig.add_subplot(nested[grid_row, 0], sharex=share)
        axes.append(ax)
        frame = profiles.loc[
            profiles["regime"].eq(regime)
            & profiles["chronology"].eq(chronology)
        ]
        power = float(
            selection.loc[
                selection["regime"].eq(regime)
                & selection["chronology"].eq(chronology),
                "tail_power_mw",
            ].iloc[0]
        )
        dispatch_small_multiple(ax, frame, power, regime, chronology, row)
    bbox = spec.get_position(fig)
    fig.text(
        bbox.x0 - 0.046,
        0.5 * (bbox.y0 + bbox.y1),
        r"hourly normalized power  $P/P^*$",
        rotation=90,
        ha="center",
        va="center",
    )
    fig.text(
        bbox.x1 + 0.027,
        0.5 * (bbox.y0 + bbox.y1),
        "A-CAES inventory (%)",
        rotation=270,
        ha="center",
        va="center",
    )
    panel_label(axes[0], "a", -0.060, 1.28)
    fig.legend(
        handles=[
            Patch(facecolor=BASE_DISCHARGE, edgecolor="none", label="4 h LFP discharge"),
            Patch(facecolor=TAIL_DISCHARGE, edgecolor="none", label="24 h A-CAES discharge"),
            Patch(facecolor=BASE_CHARGE, edgecolor="none", label="4 h LFP charging"),
            Patch(facecolor=TAIL_CHARGE, edgecolor="none", label="24 h A-CAES charging"),
            Patch(
                facecolor=mpl.colors.to_rgba(SOC_FILL, 0.27),
                edgecolor="none",
                label="A-CAES inventory (hourly)",
            ),
        ],
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.992),
        handlelength=1.25,
        columnspacing=0.9,
        borderaxespad=0,
    )


def plot_cycles(ax: plt.Axes, records: pd.DataFrame) -> None:
    panel_label(ax, "b", -0.16, 1.08)
    order = ["Europe", "China", "Japan"]
    positions = [2.0, 1.0, 0.0]
    for y, regime in zip(positions, order):
        frame = records.loc[records["regime"].eq(regime)]
        before = frame["cycles_dispersed"].to_numpy(float)
        after = frame["cycles_clustered"].to_numpy(float)
        jitter = deterministic_jitter(len(frame), 0.055)
        ax.scatter(
            before,
            y + 0.13 + jitter,
            s=5.5,
            color=mpl.colors.to_rgba(BLUE, 0.25),
            edgecolor="none",
            rasterized=True,
        )
        ax.scatter(
            after,
            y - 0.13 + jitter,
            s=5.5,
            color=mpl.colors.to_rgba(CORAL, 0.28),
            edgecolor="none",
            rasterized=True,
        )
        med_before = float(np.median(before))
        med_after = float(np.median(after))
        ax.annotate(
            "",
            xy=(med_after, y),
            xytext=(med_before, y),
            arrowprops={"arrowstyle": "-|>", "color": REGIME_COLORS[regime], "lw": 0.9},
        )
        ax.scatter(med_before, y, s=28, facecolor="white", edgecolor=BLUE, lw=0.9, zorder=4)
        ax.scatter(med_after, y, s=30, facecolor=CORAL, edgecolor="white", lw=0.45, zorder=5)
        ax.text(
            0.99,
            (y + 0.28) / 2.85,
            f"{med_before:.1f} → {med_after:.1f}",
            transform=ax.transAxes,
            color=REGIME_COLORS[regime],
            ha="right",
            va="center",
            fontweight="bold",
            fontsize=5.3,
        )
    ax.set_yticks(positions, [REGIME_LABELS[regime] for regime in order])
    for tick, regime in zip(ax.get_yticklabels(), order):
        tick.set_color(REGIME_COLORS[regime])
        tick.set_fontweight("bold")
    ax.set_xlim(0, 65)
    ax.set_ylim(-0.48, 2.48)
    ax.set_xlabel("24 h A-CAES equivalent full cycles")
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", mfc="white", mec=BLUE, lw=0, label="dispersed"),
            Line2D([0], [0], marker="o", mfc=CORAL, mec="white", lw=0, label="clustered"),
        ],
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.55, 1.12),
        handletextpad=0.3,
        columnspacing=0.8,
    )
    quiet(ax, "x")


def plot_cycle_ledger(
    ax: plt.Axes,
    records: pd.DataFrame,
    label: str | None = "h",
) -> None:
    if label:
        panel_label(ax, label, -0.15, 1.08)
    order = ["Europe", "China", "Japan"]
    positions = [2, 1, 0]
    for y, regime in zip(positions, order):
        frame = records.loc[records["regime"].eq(regime)]
        dispersed = float(frame["cycles_dispersed"].median())
        clustered = float(frame["cycles_clustered"].median())
        lost = max(dispersed - clustered, 0.0)
        retained = 100.0 * clustered / dispersed
        ax.barh(y, clustered, height=0.46, color=TEAL, edgecolor="none")
        ax.barh(
            y,
            lost,
            left=clustered,
            height=0.46,
            color=CORAL_LIGHT,
            edgecolor="none",
        )
        ax.text(
            clustered * 0.5,
            y,
            f"{clustered:.1f}",
            color="white",
            fontweight="bold",
            ha="center",
            va="center",
            fontsize=5.2,
        )
        if lost > 3:
            ax.text(
                clustered + lost * 0.5,
                y,
                f"−{lost:.1f}",
                color=INK,
                ha="center",
                va="center",
                fontsize=5.1,
            )
        ax.text(
            75.0,
            y,
            f"{retained:.0f}% retained",
            color=REGIME_COLORS[regime],
            fontweight="bold",
            ha="right",
            va="center",
            fontsize=5.1,
        )
    ax.set_xlim(0, 78)
    ax.set_ylim(-0.55, 2.55)
    ax.set_yticks(positions, [REGIME_LABELS[regime] for regime in order])
    for tick, regime in zip(ax.get_yticklabels(), order):
        tick.set_color(REGIME_COLORS[regime])
        tick.set_fontweight("bold")
        tick.set_rotation(90)
        tick.set_horizontalalignment("center")
        tick.set_verticalalignment("center")
        tick.set_rotation_mode("anchor")
    ax.tick_params(axis="y", pad=7.0)
    ax.set_xlabel("24 h A-CAES equivalent full cycles")
    ax.legend(
        handles=[
            Patch(facecolor=TEAL, edgecolor="none", label="retained after clustering"),
            Patch(facecolor=CORAL_LIGHT, edgecolor="none", label="cycles lost"),
        ],
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.55, 1.13),
        handlelength=1.0,
        columnspacing=0.8,
    )
    quiet(ax, "x")


def plot_cycle_value(ax: plt.Axes, records: pd.DataFrame) -> None:
    panel_label(ax, "c", -0.10, 1.06)
    for regime in ["Europe", "China", "Japan"]:
        frame = records.loc[records["regime"].eq(regime)]
        colour = REGIME_COLORS[regime]
        ax.scatter(
            frame["cycle_effect"],
            frame["signed_sequencing_effect_eur_kw_year"],
            s=11.5,
            marker=REGIME_MARKERS[regime],
            color=mpl.colors.to_rgba(colour, 0.48),
            edgecolor="white",
            lw=0.20,
            rasterized=True,
        )
        rho = frame[["cycle_effect", "signed_sequencing_effect_eur_kw_year"]].corr(method="spearman").iloc[0, 1]
        y = {"Europe": 0.95, "China": 0.87, "Japan": 0.79}[regime]
        ax.text(
            0.025,
            y,
            rf"{REGIME_LABELS[regime]}  $\rho={rho:.2f}$",
            transform=ax.transAxes,
            color=colour,
            fontweight="bold",
            fontsize=5.3,
            va="top",
        )
    cost = float(records["matched_cost_movement_eur_kw_year"].median())
    ax.axhline(0, color=INK, lw=0.62)
    ax.axvline(0, color=INK, lw=0.62)
    ax.axhline(-cost, color="#7364A8", lw=0.75, ls=(0, (3, 2)))
    ax.annotate(
        f"2025–2035 cost scale\n−{cost:.2f} EUR kW$^{{-1}}$ yr$^{{-1}}$",
        xy=(-18.0, -cost),
        xytext=(-31.5, -22.0),
        textcoords="data",
        color="#7364A8",
        va="center",
        ha="left",
        fontsize=5.2,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 0.45},
        arrowprops={
            "arrowstyle": "-|>",
            "color": "#7364A8",
            "lw": 0.62,
            "mutation_scale": 6.0,
            "shrinkA": 1.5,
            "shrinkB": 1.5,
        },
        zorder=7,
    )
    ax.set_xlim(-48, 2)
    ax.set_ylim(-75, 9)
    ax.set_xlabel("change in equivalent full cycles")
    ax.set_ylabel("clustered − dispersed value\n(EUR kW$^{-1}$ yr$^{-1}$)")
    quiet(ax, "both")


def plot_value_strip(
    ax: plt.Axes,
    frame: pd.DataFrame,
    regime: str,
    label: str | None,
    xlabel: bool,
) -> None:
    values = frame["signed_sequencing_effect_eur_kw_year"].to_numpy(float)
    colour = REGIME_COLORS[regime]
    x, density = density_curve(values, bins=70, smooth=4.0)
    ax.fill_between(x, 0, density, color=mpl.colors.to_rgba(colour, 0.25), lw=0)
    ax.plot(x, density, color=colour, lw=0.95)
    ax.scatter(
        values,
        -0.11 + deterministic_jitter(values.size, 0.055),
        s=5.8,
        color=mpl.colors.to_rgba(colour, 0.32),
        edgecolor="white",
        lw=0.14,
        rasterized=True,
    )
    median = float(np.median(values))
    ax.axvline(0, color=INK, lw=0.65, ls=(0, (3, 2)))
    share = 100.0 * float(np.mean(values < 0))
    annotation_box = {
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.84,
        "pad": 0.35,
    }
    ax.text(
        0.01,
        0.95,
        REGIME_LABELS[regime],
        transform=ax.transAxes,
        color=colour,
        fontweight="bold",
        va="top",
        bbox=annotation_box,
    )
    ax.text(
        0.99,
        0.95,
        f"{share:.0f}% < 0  |  median {median:+.2f}",
        transform=ax.transAxes,
        color=MUTED,
        ha="right",
        va="top",
        fontsize=5.0,
        bbox=annotation_box,
    )
    ax.set_ylim(-0.24, 1.28)
    ax.set_yticks([])
    if xlabel:
        ax.set_xlabel("clustered − dispersed value (EUR kW$^{-1}$ yr$^{-1}$)")
    if label:
        panel_label(ax, label, -0.16, 1.06)
    quiet(ax)


def plot_cycle_retention(ax: plt.Axes, records: pd.DataFrame) -> None:
    panel_label(ax, "e", -0.10, 1.06)
    order = ["Europe", "China", "Japan"]
    positions = [2, 1, 0]
    for y, regime in zip(positions, order):
        frame = records.loc[records["regime"].eq(regime)]
        values = 100.0 * frame["cycles_clustered"].to_numpy(float) / frame["cycles_dispersed"].to_numpy(float)
        p05, q1, median, q3, p95 = np.quantile(values, [0.05, 0.25, 0.50, 0.75, 0.95])
        colour = REGIME_COLORS[regime]
        ax.barh(y, 100, left=0, height=0.42, color=PALE, edgecolor="none")
        ax.barh(y, median, left=0, height=0.42, color=mpl.colors.to_rgba(colour, 0.36), edgecolor="none")
        ax.plot([p05, p95], [y, y], color=colour, lw=0.85)
        ax.plot([q1, q3], [y, y], color=colour, lw=3.2, solid_capstyle="round")
        ax.scatter(median, y, s=31, marker=REGIME_MARKERS[regime], color=colour, edgecolor="white", lw=0.45, zorder=4)
        ax.text(
            119.0,
            y,
            f"{median:.1f}% retained",
            color=colour,
            va="center",
            ha="right",
            fontweight="bold",
            fontsize=5.4,
        )
    ax.axvline(100, color=INK, lw=0.65, ls=(0, (3, 2)))
    ax.set_xlim(0, 122)
    ax.set_ylim(-0.55, 2.55)
    ax.set_yticks(positions, [REGIME_LABELS[regime] for regime in order])
    for tick, regime in zip(ax.get_yticklabels(), order):
        tick.set_color(REGIME_COLORS[regime])
        tick.set_fontweight("bold")
    ax.set_xlabel("equivalent full cycles retained after clustering (%)")
    ax.text(0.82, 0.98, "dispersed reference", transform=ax.transAxes, color=MUTED, ha="center", va="top", fontsize=5.1)
    quiet(ax, "x")


def plot_value_decomposition(
    ax: plt.Axes,
    decomposition: pd.DataFrame,
    label: str | None = "f",
) -> None:
    if label:
        panel_label(ax, label, -0.13, 1.06)
    order = ["Europe", "China", "Japan"]
    positions = [2, 1, 0]
    components = ["variable_om", "charging", "energy_service", "firm_service"]
    labels = {
        "variable_om": "variable O&M",
        "charging": "charging",
        "energy_service": "energy service",
        "firm_service": "peak service",
    }
    for y, regime in zip(positions, order):
        medians = decomposition.loc[
            decomposition["regime"].eq(regime), [*components, "total"]
        ].median()
        denominator = abs(float(medians["total"]))
        component_ratio = 100.0 * medians[components] / denominator
        positive = 0.0
        negative = 0.0
        for component in components:
            value = float(component_ratio[component])
            if value >= 0:
                ax.barh(
                    y,
                    value,
                    left=positive,
                    height=0.48,
                    color=COMPONENT_COLORS[component],
                    edgecolor="white",
                    lw=0.35,
                )
                positive += value
            else:
                ax.barh(
                    y,
                    value,
                    left=negative,
                    height=0.48,
                    color=COMPONENT_COLORS[component],
                    edgecolor="white",
                    lw=0.35,
                )
                negative += value
        total_share = 100.0 if medians["total"] >= 0 else -100.0
        ax.scatter(
            total_share,
            y,
            s=30,
            marker=REGIME_MARKERS[regime],
            color=REGIME_COLORS[regime],
            edgecolor="white",
            lw=0.45,
            zorder=5,
        )
        ax.text(
            122,
            y,
            f"{float(medians['total']):+.2f}",
            color=REGIME_COLORS[regime],
            fontweight="bold",
            ha="right",
            va="center",
            fontsize=5.2,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.86,
                "pad": 0.25,
            },
        )
    ax.axvline(0, color=INK, lw=0.65)
    ax.set_xlim(-150, 128)
    ax.set_ylim(-0.55, 2.55)
    ax.set_yticks(positions, [REGIME_LABELS[regime] for regime in order])
    for tick, regime in zip(ax.get_yticklabels(), order):
        tick.set_color(REGIME_COLORS[regime])
        tick.set_fontweight("bold")
        tick.set_rotation(90)
        tick.set_horizontalalignment("center")
        tick.set_verticalalignment("center")
        tick.set_rotation_mode("anchor")
    ax.tick_params(axis="y", pad=7.0)
    ax.set_xlabel("component share of\n|sequencing effect| (%)")
    ax.legend(
        handles=[
            Patch(facecolor=COMPONENT_COLORS[key], edgecolor="none", label=labels[key])
            for key in components
        ],
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.50, 1.17),
        handlelength=1.0,
        columnspacing=0.8,
    )
    quiet(ax, "x")


def plot_recovery_gate(ax: plt.Axes, recovery: pd.DataFrame) -> None:
    panel_label(ax, "g", -0.10, 1.06)
    values = np.array([[72.0, np.nan, np.nan, np.nan], [72.0, 0.0, 0.0, 0.0]])
    cmap = LinearSegmentedColormap.from_list("window_gate", [CORAL, "#F5F2EE", TEAL])
    norm = Normalize(0, 72)

    for row in range(2):
        for col in range(4):
            value = values[row, col]
            face = "#ECEEEC" if np.isnan(value) else cmap(norm(value))
            rect = Rectangle((col + 0.04, row + 0.08), 0.92, 0.84, facecolor=face, edgecolor="white", lw=1.0)
            ax.add_patch(rect)
            label = "—" if np.isnan(value) else f"{value:.0f} h"
            text_colour = MUTED if np.isnan(value) else ("white" if value in (0, 72) else INK)
            ax.text(col + 0.50, row + 0.50, label, color=text_colour, ha="center", va="center", fontweight="bold")

    zero_share = 100.0 * float(
        recovery.loc[recovery["chronology"].eq("clustered"), "zero_recovery_window_share"].median()
    )
    ax.text(4.30, 0.50, "single-event run\n0% zero-window", color=BLUE, ha="left", va="center", fontweight="bold")
    ax.text(
        4.30,
        1.50,
        f"four-day cluster\n{zero_share:.1f}% zero-window",
        color=CORAL,
        ha="left",
        va="center",
        fontweight="bold",
    )
    ax.set_xlim(0, 6.0)
    ax.set_ylim(2.0, 0.0)
    ax.set_xticks(np.arange(4) + 0.5, ["first", "second", "third", "fourth"])
    ax.set_yticks([0.5, 1.5], ["dispersed", "clustered"])
    for tick, colour in zip(ax.get_yticklabels(), [BLUE, CORAL]):
        tick.set_color(colour)
        tick.set_fontweight("bold")
    ax.set_xlabel("event position within a run")
    ax.tick_params(axis="both", length=0)
    ax.spines[["left", "right", "top", "bottom"]].set_visible(False)
    ax.legend(
        handles=[
            Patch(facecolor=TEAL, edgecolor="none", label="open recovery window"),
            Patch(facecolor=CORAL, edgecolor="none", label="zero recovery window"),
            Patch(facecolor="#ECEEEC", edgecolor="none", label="no additional event"),
        ],
        ncol=3,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.10),
        handlelength=1.0,
        columnspacing=0.9,
        borderaxespad=0,
    )


def plot_soc_duration(fig: plt.Figure, spec, profiles: pd.DataFrame) -> None:
    nested = spec.subgridspec(1, 3, wspace=0.28)
    axes: list[plt.Axes] = []
    for index, regime in enumerate(["Europe", "China", "Japan"]):
        ax = fig.add_subplot(nested[0, index])
        axes.append(ax)
        curves = {}
        for chronology in ["dispersed", "clustered"]:
            values = profiles.loc[
                profiles["regime"].eq(regime)
                & profiles["chronology"].eq(chronology),
                "tail_soc_pct",
            ].to_numpy(float)
            curves[chronology] = np.sort(values)[::-1]
        x = np.linspace(0, 100, len(curves["dispersed"]))
        ax.fill_between(
            x,
            curves["dispersed"],
            curves["clustered"],
            color=mpl.colors.to_rgba(REGIME_COLORS[regime], 0.16),
            lw=0,
        )
        ax.plot(x, curves["dispersed"], color=BLUE, lw=0.82)
        ax.plot(x, curves["clustered"], color=CORAL, lw=0.82)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 102)
        ax.set_xticks([0, 50, 100])
        ax.set_xlabel("operating-year percentile")
        if index == 0:
            ax.set_ylabel("A-CAES inventory (%)")
        else:
            ax.set_yticklabels([])
        ax.text(
            0.04,
            0.94,
            REGIME_LABELS[regime],
            transform=ax.transAxes,
            color=REGIME_COLORS[regime],
            fontweight="bold",
            va="top",
        )
        quiet(ax, "both")
    panel_label(axes[0], "e", -0.27, 1.10)
    axes[1].legend(
        handles=[
            Line2D([0], [0], color=BLUE, lw=0.9, label="dispersed"),
            Line2D([0], [0], color=CORAL, lw=0.9, label="clustered"),
        ],
        ncol=1,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        handlelength=1.3,
        labelspacing=0.25,
        borderaxespad=0,
    )


def plot_value_triptych(
    fig: plt.Figure,
    spec,
    records: pd.DataFrame,
) -> None:
    nested = spec.subgridspec(1, 3, wspace=0.16)
    axes: list[plt.Axes] = []
    for index, regime in enumerate(["Europe", "China", "Japan"]):
        ax = fig.add_subplot(nested[0, index])
        axes.append(ax)
        plot_value_strip(
            ax,
            records.loc[records["regime"].eq(regime)],
            regime,
            None,
            index == 1,
        )
    panel_label(axes[0], "b", -0.16, 1.08)


def plot_economic_cycle_summary(
    fig: plt.Figure,
    spec,
    decomposition: pd.DataFrame,
    records: pd.DataFrame,
) -> None:
    nested = spec.subgridspec(
        2,
        1,
        height_ratios=[1.08, 0.92],
        hspace=0.62,
    )
    top = fig.add_subplot(nested[0, 0])
    bottom = fig.add_subplot(nested[1, 0])
    plot_value_decomposition(top, decomposition, None)
    plot_cycle_ledger(bottom, records, None)
    panel_label(top, "d", -0.035, 1.13)


def main() -> None:
    configure_publication()
    records = pd.read_csv(SOURCE / "figure3_24h_value_cycle_records.csv")
    profiles = pd.read_csv(
        PROFILE_SOURCE / "figure3_representative_dispatch_profiles.csv.gz"
    )
    selection = pd.read_csv(
        PROFILE_SOURCE / "figure3_representative_dispatch_selection.csv"
    )
    decomposition = pd.read_csv(
        PROFILE_SOURCE / "figure3_value_component_effects.csv.gz"
    )

    fig = plt.figure(figsize=(WIDTH_MM * MM, HEIGHT_MM * MM))
    grid = fig.add_gridspec(
        7,
        12,
        left=0.075,
        right=0.942,
        bottom=0.055,
        top=0.952,
        height_ratios=[10.50, 0.95, 2.50, 1.70, 5.00, 1.25, 3.20],
        hspace=0.0,
        wspace=0.96,
    )

    plot_dispatch_hero(fig, grid[0, :], profiles, selection)
    plot_value_triptych(fig, grid[2, :], records)
    plot_cycle_value(fig.add_subplot(grid[4, :8]), records)
    plot_soc_duration(fig, grid[6, :8], profiles)
    plot_economic_cycle_summary(fig, grid[4:, 8:], decomposition, records)

    save_bundle(fig, "fig_v62_preview_figure3_recovery_relayout")
    plt.close(fig)


if __name__ == "__main__":
    main()
