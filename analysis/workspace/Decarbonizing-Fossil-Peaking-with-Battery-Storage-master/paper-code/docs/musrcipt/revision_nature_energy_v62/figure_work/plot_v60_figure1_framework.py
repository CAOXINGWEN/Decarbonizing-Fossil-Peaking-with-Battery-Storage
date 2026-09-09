#!/usr/bin/env python3
"""Figure 1: visual framework for recovery-window sequencing parity."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from v60_figure_style import (
    BLUE,
    CORAL,
    GRID,
    INK,
    MM,
    MUTED,
    PURPLE,
    REGIME_COLORS,
    REGIME_MARKERS,
    TEAL,
    configure,
    panel_label,
    quiet,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = HERE / "source_data_v60"
OUT = HERE / "main_figures"
HEIGHT_MM = 170.0
WIDTH_MM = 183.0
THRESHOLD = 5.2960380996609935


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


def rgba_strip(rows: list[tuple[np.ndarray, str]]) -> np.ndarray:
    image = np.ones((len(rows), len(rows[0][0]), 4), float)
    for index, (values, colour) in enumerate(rows):
        values = np.clip(np.nan_to_num(values, nan=0.0), 0.0, 1.0)
        rgb = np.asarray(mpl.colors.to_rgb(colour))
        strength = 0.08 + 0.88 * values
        image[index, :, :3] = 1.0 - strength[:, None] * (1.0 - rgb[None, :])
    return image


def recovery_gap(task: np.ndarray) -> np.ndarray:
    event = task > 1.0e-10
    gap = np.zeros(len(task), float)
    elapsed = 0
    for i, active in enumerate(event):
        if active:
            elapsed = 0
        else:
            elapsed += 1
        gap[i] = min(elapsed * 24.0, 72.0) / 72.0
    return gap


def plot_calendar(fig: plt.Figure, spec, chronology: pd.DataFrame, summary: pd.DataFrame) -> None:
    sub = spec.subgridspec(2, 1, hspace=0.16)
    for index, chronology_name in enumerate(["dispersed", "clustered"]):
        ax = fig.add_subplot(sub[index, 0])
        task = chronology[f"{chronology_name}_task_normalized"].to_numpy(float)
        recharge = chronology[f"{chronology_name}_recharge_normalized"].to_numpy(float)
        gap = recovery_gap(task)
        image = rgba_strip([(task, CORAL), (recharge, BLUE), (gap, TEAL)])
        ax.imshow(image, aspect="auto", interpolation="nearest", extent=[1, 365, 2.5, -0.5], rasterized=True)
        ax.set_yticks([0, 1, 2], ["task", "recharge", "recovery"])
        ax.tick_params(axis="y", length=0, pad=3)
        for tick, colour in zip(ax.get_yticklabels(), [CORAL, BLUE, TEAL]):
            tick.set_color(colour)
            tick.set_fontweight("bold")
        ax.set_xlim(1, 365)
        ax.set_ylim(2.5, -0.5)
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        ax.tick_params(axis="x", length=0)
        if index == 0:
            ax.set_xticks([])
            panel_label(ax, "a", -0.055, 1.11)
            ax.text(0.0, 1.08, "Same annual task and recharge, different event spacing", transform=ax.transAxes, color=INK, fontweight="bold", va="bottom")
        else:
            ax.set_xticks([1, 62, 123, 184, 245, 306, 365], ["Jan", "Mar", "May", "Jul", "Sep", "Nov", "Dec"])
            ax.set_xlabel("day block in conserved annual sequence", labelpad=2)
        row = summary.loc[summary["regime"].eq("Europe")].iloc[0]
        recovery = float(row[f"median_recovery_{chronology_name}_h"])
        ax.text(0.005, 0.93, chronology_name, transform=ax.transAxes, color=BLUE if index == 0 else CORAL, fontweight="bold", fontsize=6.8, va="top")
        ax.text(0.995, 0.93, f"median recovery  {recovery:.0f} h", transform=ax.transAxes, color=INK, ha="right", va="top", fontweight="bold")
        if index == 1:
            zero = float(row["zero_window_clustered_pct"])
            ax.text(0.995, 0.08, f"{zero:.1f}% zero-window intervals", transform=ax.transAxes, color=CORAL, ha="right", va="bottom", fontweight="bold")


def plot_refill_ledger(ax: plt.Axes) -> None:
    panel_label(ax, "b", -0.08, 1.06)
    ax.text(0.0, 1.04, "Recovery time closes the refill aperture", transform=ax.transAxes, color=INK, fontweight="bold", va="bottom")
    ax.text(0.0, 0.89, r"$E_{\mathrm{rebuild},w}=\min\{R_w,\;P_{\mathrm{ch}}\Delta t_w,\;E_{\mathrm{room},w}\}$", transform=ax.transAxes, color=INK, fontsize=7.4)
    categories = ["available\nrecharge", "interface ×\ntime", "empty\ncapacity"]
    colours = [BLUE, TEAL, PURPLE]
    x = np.arange(3)
    for row, (window, values, rebuild, row_colour) in enumerate(
        [
            ("open window", [0.88, 0.74, 0.96], 0.74, TEAL),
            ("zero window", [0.88, 0.00, 0.96], 0.00, CORAL),
        ]
    ):
        base = 0.75 - row * 1.10
        for xi, value, colour in zip(x, values, colours):
            ax.bar(xi, value, width=0.56, bottom=base, color=mpl.colors.to_rgba(colour, 0.24), edgecolor=colour, lw=0.7)
        ax.hlines(base + rebuild, -0.34, 2.34, color=row_colour, lw=1.15, ls=(0, (3, 2)))
        ax.text(-0.45, base + 0.48, window, color=row_colour, ha="right", va="center", fontweight="bold")
        ax.text(2.45, base + rebuild, f"rebuild = {rebuild:.2f}", color=row_colour, ha="left", va="center", fontweight="bold")
    ax.set_xlim(-0.75, 3.15)
    ax.set_ylim(-0.45, 2.30)
    ax.set_xticks(x, categories)
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0, pad=3)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)


def plot_duration_ridges(ax: plt.Axes, duration: pd.DataFrame) -> None:
    panel_label(ax, "c", -0.10, 1.06)
    ax.text(0.0, 1.04, "Event-cluster scale moves the value trough", transform=ax.transAxes, color=INK, fontweight="bold", va="bottom")
    offsets = {"Europe": 2.0, "China": 1.0, "Japan": 0.0}
    cluster_days = {"Europe": 7, "China": 4, "Japan": 11}
    for regime in ["Europe", "China", "Japan"]:
        frame = duration.loc[duration["regime"].eq(regime)].sort_values("duration_h")
        x = frame["duration_h"].to_numpy(float)
        if np.any(x <= 0):
            raise ValueError("Storage durations must be strictly positive on the logarithmic axis.")
        y = frame["block_mean_effect_eur_kw_year"].to_numpy(float)
        scaled = y / max(float(np.max(np.abs(y))), 1.0)
        base = offsets[regime]
        colour = REGIME_COLORS[regime]
        ax.fill_between(x, base, base + 0.58 * scaled, color=mpl.colors.to_rgba(colour, 0.23), lw=0)
        ax.plot(x, base + 0.58 * scaled, color=colour, lw=1.0)
        minimum = int(np.argmin(y))
        ax.scatter(x[minimum], base + 0.58 * scaled[minimum], s=24, marker=REGIME_MARKERS[regime], color=colour, edgecolor="white", lw=0.45, zorder=4)
        ax.text(4.2, base + 0.16, regime, color=colour, fontweight="bold", ha="left", va="center")
        ax.text(690, base + 0.16, f"{cluster_days[regime]} d cluster  |  trough {x[minimum]:g} h", color=MUTED, ha="right", va="center", fontsize=5.5)
    ax.axhline(0, color=GRID, lw=0.5)
    ax.set_xscale("log")
    ax.set_xlim(4, 720)
    ax.set_ylim(-0.72, 2.58)
    ax.set_xticks([4, 8, 24, 48, 96, 240, 720], ["4", "8", "24", "48", "96", "240", "720"])
    ax.set_yticks([])
    ax.set_xlabel("storage duration (h)")
    ax.spines[["top", "right", "left"]].set_visible(False)


def plot_evidence(ax: plt.Axes, parity: pd.DataFrame, weather: dict[str, object]) -> None:
    panel_label(ax, "d", -0.055, 1.05)
    ax.text(0.0, 1.03, "Natural weather and three balancing regimes reach the same cost scale", transform=ax.transAxes, color=INK, fontweight="bold", va="bottom")
    labels = ["Natural weather", "Europe", "China", "Japan"]
    colours = [TEAL, REGIME_COLORS["Europe"], REGIME_COLORS["China"], REGIME_COLORS["Japan"]]
    y = np.arange(4)
    natural_central = float(weather["natural_q95_q05_span_eur_kw_year"])
    natural_full = float(weather["natural_full_span_eur_kw_year"])
    evidence_values = np.r_[
        [natural_central, natural_full, THRESHOLD],
        parity[["q25_sequencing_shift_eur_kw_year", "median_sequencing_shift_eur_kw_year", "q75_sequencing_shift_eur_kw_year"]].to_numpy(float).ravel(),
    ]
    if np.any(evidence_values <= 0):
        raise ValueError("Sequencing-value evidence must be strictly positive on the logarithmic axis.")
    ax.plot([natural_central, natural_full], [0, 0], color=TEAL, lw=3.0, solid_capstyle="round")
    ax.scatter([natural_central, natural_full], [0, 0], s=[29, 20], color=[TEAL, INK], edgecolor="white", lw=0.45, zorder=4)
    ax.text(natural_full * 1.08, 0, f"5--95% {natural_central:.2f}  |  full {natural_full:.2f}  |  161/2,775 pairs", color=TEAL, va="center", fontsize=5.5, fontweight="bold")
    for yi, regime in enumerate(["Europe", "China", "Japan"], start=1):
        row = parity.loc[parity["regime"].eq(regime)].iloc[0]
        q1 = float(row["q25_sequencing_shift_eur_kw_year"])
        med = float(row["median_sequencing_shift_eur_kw_year"])
        q3 = float(row["q75_sequencing_shift_eur_kw_year"])
        colour = REGIME_COLORS[regime]
        ax.plot([q1, q3], [yi, yi], color=colour, lw=3.0, solid_capstyle="round")
        ax.scatter(med, yi, s=31, marker=REGIME_MARKERS[regime], color=colour, edgecolor="white", lw=0.5, zorder=4)
        hits = int(row["above_parity_count"])
        total = int(row["n_system_years"])
        ax.text(q3 * 1.08, yi, f"median {med:.2f}  |  {hits}/{total}", color=colour, va="center", fontsize=5.5, fontweight="bold")
    ax.axvline(THRESHOLD, color=PURPLE, lw=0.9, ls=(0, (3, 2)))
    ax.text(THRESHOLD, -0.55, "2025--2035\ncost scale  5.30", color=PURPLE, ha="center", va="top", fontsize=5.5, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(1.2, 150)
    ax.set_ylim(3.55, -0.65)
    ax.set_xticks([2, 5, 10, 20, 50, 100], ["2", "5", "10", "20", "50", "100"])
    ax.set_yticks(y, labels)
    for tick, colour in zip(ax.get_yticklabels(), colours):
        tick.set_color(colour)
        tick.set_fontweight("bold")
    ax.set_xlabel("absolute sequencing-value displacement (EUR kW$^{-1}$ yr$^{-1}$; log scale)")
    quiet(ax, "x")


def main() -> None:
    configure_publication()
    chronology = pd.read_csv(SOURCE / "figure2_conserved_chronology.csv")
    summary = pd.read_csv(SOURCE / "figure2_regime_summary.csv")
    duration = pd.read_csv(SOURCE / "figure5_duration_response_summary.csv")
    parity = pd.read_csv(SOURCE / "harmonized_parity/three_regime_24h_summary.csv")
    weather = json.loads((ROOT / "weather_year_parity_v42/results/pecd_v42_75y_parity_summary.json").read_text(encoding="utf-8"))

    fig = plt.figure(figsize=(WIDTH_MM * MM, HEIGHT_MM * MM))
    grid = fig.add_gridspec(
        3,
        12,
        left=0.115,
        right=0.985,
        bottom=0.06,
        top=0.97,
        height_ratios=[1.00, 0.88, 0.82],
        hspace=0.47,
        wspace=0.90,
    )
    plot_calendar(fig, grid[0, :], chronology, summary)
    plot_refill_ledger(fig.add_subplot(grid[1, :6]))
    plot_duration_ridges(fig.add_subplot(grid[1, 7:]), duration)
    plot_evidence(fig.add_subplot(grid[2, :]), parity, weather)
    export_figure(fig, "fig_v60_figure1_sequencing_parity_framework")
    plt.close(fig)


if __name__ == "__main__":
    main()
