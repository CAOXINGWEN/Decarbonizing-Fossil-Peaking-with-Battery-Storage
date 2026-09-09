#!/usr/bin/env python3
"""Figure 3: recovery-window compression, cycling and portfolio value."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

from v60_figure_style import (
    BLUE,
    CORAL,
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


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source_data_v60"
OUT = HERE / "main_figures"
HEIGHT_MM = 184.0
WIDTH_MM = 183.0


def configure_publication() -> None:
    configure()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def export_figure(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / stem
    fig.savefig(path.with_suffix(".pdf"), metadata={"Creator": "Python/matplotlib"})
    fig.savefig(path.with_suffix(".svg"))
    fig.savefig(path.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(path.with_suffix(".png"), dpi=360)
    rgba = plt.imread(path.with_suffix(".png"))
    grey = np.dot(rgba[..., :3], [0.2126, 0.7152, 0.0722])
    plt.imsave(path.with_name(path.name + "_grayscale").with_suffix(".png"), grey, cmap="gray", vmin=0, vmax=1)


def elapsed_recovery(task: np.ndarray) -> np.ndarray:
    event = task > 1.0e-10
    elapsed = 0
    output = np.zeros(len(task), float)
    for index, active in enumerate(event):
        if active:
            elapsed = 0
        else:
            elapsed += 24
        output[index] = min(elapsed, 72)
    return output


def plot_recovery_calendar(fig: plt.Figure, spec, chronology: pd.DataFrame, recovery: pd.DataFrame) -> None:
    sub = spec.subgridspec(2, 18, height_ratios=[1.0, 0.16], hspace=0.48, wspace=0.15)
    ax = fig.add_subplot(sub[0, :])
    matrix = np.vstack(
        [
            elapsed_recovery(chronology["dispersed_task_normalized"].to_numpy(float)),
            elapsed_recovery(chronology["clustered_task_normalized"].to_numpy(float)),
        ]
    )
    cmap = LinearSegmentedColormap.from_list("recovery_window", [CORAL, "#F4F2EE", TEAL])
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, norm=Normalize(0, 72), extent=[1, 365, 1.5, -0.5], rasterized=True)
    ax.set_yticks([0, 1], ["dispersed", "clustered"])
    for tick, colour in zip(ax.get_yticklabels(), [BLUE, CORAL]):
        tick.set_color(colour)
        tick.set_fontweight("bold")
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_xticks([1, 62, 123, 184, 245, 306, 365], ["Jan", "Mar", "May", "Jul", "Sep", "Nov", "Dec"])
    ax.set_xlabel("")
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    panel_label(ax, "a", -0.055, 1.17)
    ax.text(0.0, 1.12, "Clustering closes inter-event recovery windows", transform=ax.transAxes, color=INK, fontweight="bold", va="bottom")
    grouped = recovery.groupby(["chronology"])[["median_recovery_hours", "zero_recovery_window_share"]].median()
    ax.text(0.01, 0.78, f"median {grouped.loc['dispersed', 'median_recovery_hours']:.0f} h", transform=ax.transAxes, color=INK, fontweight="bold")
    ax.text(
        0.99,
        0.18,
        f"median {grouped.loc['clustered', 'median_recovery_hours']:.0f} h  |  {100 * grouped.loc['clustered', 'zero_recovery_window_share']:.1f}% zero-window",
        transform=ax.transAxes,
        color=CORAL,
        ha="right",
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.7},
    )
    cax = fig.add_subplot(sub[1, :])
    cb = fig.colorbar(image, cax=cax, orientation="horizontal")
    cb.set_ticks([0, 24, 48, 72])
    cb.set_label("elapsed recovery opportunity (h; capped at 72 h)", fontsize=5.5, labelpad=1.0)
    cb.ax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=5.1, length=1.6, pad=1)


def split_ridge(ax: plt.Axes, values: np.ndarray, baseline: float, colour: str, direction: float) -> None:
    x, density = density_curve(values)
    y = baseline + direction * 0.33 * density
    ax.fill_between(x, baseline, y, color=mpl.colors.to_rgba(colour, 0.25), lw=0)
    ax.plot(x, y, color=colour, lw=0.9)
    scatter_y = baseline + direction * (0.05 + np.abs(deterministic_jitter(len(values), 0.07)))
    ax.scatter(values, scatter_y, s=6.5, color=mpl.colors.to_rgba(colour, 0.32), edgecolor="white", lw=0.15, rasterized=True)
    median = float(np.median(values))
    ax.scatter(median, baseline, s=27, color=colour, edgecolor="white", lw=0.45, zorder=4)


def plot_cycles(ax: plt.Axes, records: pd.DataFrame) -> None:
    panel_label(ax, "b", -0.13, 1.06)
    ax.text(0.0, 1.04, "Repeated cycling contracts in every regime", transform=ax.transAxes, color=INK, fontweight="bold", va="bottom")
    order = ["Europe", "China", "Japan"]
    positions = [2, 1, 0]
    for y, regime in zip(positions, order):
        frame = records.loc[records["regime"].eq(regime)]
        split_ridge(ax, frame["cycles_dispersed"].to_numpy(float), y, BLUE, +1)
        split_ridge(ax, frame["cycles_clustered"].to_numpy(float), y, CORAL, -1)
        before = float(frame["cycles_dispersed"].median())
        after = float(frame["cycles_clustered"].median())
        ax.text(63.5, y, f"{before:.1f} → {after:.1f}", color=REGIME_COLORS[regime], ha="right", va="center", fontweight="bold", fontsize=5.5)
    ax.set_yticks(positions, order)
    for tick, regime in zip(ax.get_yticklabels(), order):
        tick.set_color(REGIME_COLORS[regime])
        tick.set_fontweight("bold")
    ax.set_xlim(0, 65)
    ax.set_ylim(-0.55, 2.55)
    ax.set_xlabel("24 h A-CAES equivalent full cycles")
    ax.text(0.02, 0.02, "dispersed", transform=ax.transAxes, color=BLUE, fontweight="bold")
    ax.text(0.20, 0.02, "clustered", transform=ax.transAxes, color=CORAL, fontweight="bold")
    quiet(ax, "x")


def plot_transmission(ax: plt.Axes, records: pd.DataFrame) -> None:
    panel_label(ax, "c", -0.13, 1.06)
    ax.text(0.0, 1.04, "Cycle loss transmits into portfolio value", transform=ax.transAxes, color=INK, fontweight="bold", va="bottom")
    for regime in ["Europe", "China", "Japan"]:
        frame = records.loc[records["regime"].eq(regime)]
        colour = REGIME_COLORS[regime]
        ax.scatter(
            frame["cycle_effect"],
            frame["signed_sequencing_effect_eur_kw_year"],
            s=12,
            marker=REGIME_MARKERS[regime],
            color=mpl.colors.to_rgba(colour, 0.48),
            edgecolor="white",
            lw=0.22,
            rasterized=True,
        )
        rho = frame[["cycle_effect", "signed_sequencing_effect_eur_kw_year"]].corr(method="spearman").iloc[0, 1]
        y = {"Europe": 0.95, "China": 0.87, "Japan": 0.79}[regime]
        ax.text(0.04, y, rf"{regime}  $\rho={rho:.2f}$", transform=ax.transAxes, color=colour, fontweight="bold", fontsize=5.5)
    ax.axhline(0, color=INK, lw=0.6)
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_xlim(-48, 2)
    ax.set_ylim(-75, 9)
    ax.set_xlabel("change in equivalent full cycles")
    ax.set_ylabel("clustered − dispersed value\n(EUR kW$^{-1}$ yr$^{-1}$)")
    quiet(ax, "both")


def plot_value_distribution(ax: plt.Axes, frame: pd.DataFrame, regime: str, label: str) -> None:
    values = frame["signed_sequencing_effect_eur_kw_year"].to_numpy(float)
    colour = REGIME_COLORS[regime]
    x, density = density_curve(values)
    ax.fill_between(x, 0, density, color=mpl.colors.to_rgba(colour, 0.25), lw=0)
    ax.plot(x, density, color=colour, lw=1.0)
    ax.scatter(values, -0.12 + deterministic_jitter(len(values), 0.07), s=7.0, color=mpl.colors.to_rgba(colour, 0.35), edgecolor="white", lw=0.16, rasterized=True)
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    ax.plot([q1, q3], [1.12, 1.12], color=colour, lw=3.0, solid_capstyle="round")
    ax.scatter(median, 1.12, s=29, marker=REGIME_MARKERS[regime], color=colour, edgecolor="white", lw=0.5, zorder=4)
    ax.axvline(0, color=INK, lw=0.7, ls=(0, (3, 2)))
    loss_share = 100.0 * float(np.mean(values < 0))
    absolute_median = float(np.median(np.abs(values)))
    ax.text(0.03, 0.94, f"{loss_share:.0f}% negative", transform=ax.transAxes, color=colour, fontweight="bold", va="top")
    ax.text(0.97, 0.94, f"|Δ| median {absolute_median:.2f}", transform=ax.transAxes, color=MUTED, ha="right", va="top", fontsize=5.5)
    ax.set_title(regime, color=colour, fontweight="bold", pad=2)
    ax.set_ylim(-0.26, 1.33)
    ax.set_yticks([])
    ax.set_xlabel("clustered − dispersed value\n(EUR kW$^{-1}$ yr$^{-1}$)")
    panel_label(ax, label, -0.18, 1.06)
    quiet(ax)


def main() -> None:
    configure_publication()
    chronology = pd.read_csv(SOURCE / "figure2_conserved_chronology.csv")
    recovery = pd.read_csv(SOURCE / "figure3_recovery_window_records.csv")
    records = pd.read_csv(SOURCE / "figure3_24h_value_cycle_records.csv")

    fig = plt.figure(figsize=(WIDTH_MM * MM, HEIGHT_MM * MM))
    grid = fig.add_gridspec(
        3,
        12,
        left=0.075,
        right=0.985,
        bottom=0.065,
        top=0.97,
        height_ratios=[0.58, 1.02, 0.78],
        hspace=0.36,
        wspace=0.85,
    )
    plot_recovery_calendar(fig, grid[0, :], chronology, recovery)
    plot_cycles(fig.add_subplot(grid[1, :7]), records)
    plot_transmission(fig.add_subplot(grid[1, 8:]), records)
    for index, regime in enumerate(["Europe", "China", "Japan"]):
        plot_value_distribution(
            fig.add_subplot(grid[2, index * 4 : (index + 1) * 4]),
            records.loc[records["regime"].eq(regime)],
            regime,
            chr(ord("d") + index),
        )
    export_figure(fig, "fig_v60_figure3_recovery_value_mechanism")
    plt.close(fig)


if __name__ == "__main__":
    main()
