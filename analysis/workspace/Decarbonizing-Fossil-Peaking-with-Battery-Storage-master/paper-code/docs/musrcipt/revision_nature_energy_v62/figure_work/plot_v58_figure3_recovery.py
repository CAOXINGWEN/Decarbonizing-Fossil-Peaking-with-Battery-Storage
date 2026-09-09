#!/usr/bin/env python3
"""Figure 3: recovery-window compression, cycling and value across regimes."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source_data_v58"
OUT = HERE / "main_figures"
OUT.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4
WIDTH_MM = 183
HEIGHT_MM = 178

INK = "#2B383E"
MUTED = "#6D777B"
GRID = "#DDE2DF"
DISPERSED = "#4E8FAC"
CLUSTERED = "#D87863"
EUROPE = "#D87863"
CHINA = "#D0A13F"
JAPAN = "#4E8FAC"
COLORS = {"Europe": EUROPE, "China": CHINA, "Japan": JAPAN}


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.6,
            "axes.labelsize": 6.5,
            "axes.titlesize": 6.8,
            "xtick.labelsize": 5.6,
            "ytick.labelsize": 5.6,
            "legend.fontsize": 5.4,
            "legend.frameon": False,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.3,
            "ytick.major.size": 2.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.05) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=8.2, fontweight="bold", ha="left", va="bottom", color=INK)


def quiet(ax: plt.Axes, grid: str | None = None) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.45)
        ax.set_axisbelow(True)


def deterministic_jitter(n: int, scale: float = 0.08) -> np.ndarray:
    return scale * np.sin((np.arange(n) + 1) * 2.399963)


def density(values: np.ndarray, bins: int = 70) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    lo, hi = float(values.min()), float(values.max())
    if np.isclose(lo, hi):
        lo, hi = lo - 0.5, hi + 0.5
    hist, edges = np.histogram(values, bins=bins, range=(lo, hi), density=True)
    kernel_x = np.arange(-12, 13)
    kernel = np.exp(-0.5 * (kernel_x / 4.0) ** 2)
    kernel /= kernel.sum()
    smoothed = np.convolve(hist, kernel, mode="same")
    if smoothed.max() > 0:
        smoothed /= smoothed.max()
    return (edges[:-1] + edges[1:]) / 2, smoothed


def save_bundle(fig: plt.Figure, stem: str) -> None:
    path = OUT / stem
    fig.savefig(path.with_suffix(".pdf"), metadata={"Creator": "Python/matplotlib"})
    fig.savefig(path.with_suffix(".svg"))
    fig.savefig(path.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(path.with_suffix(".png"), dpi=360)
    rgba = plt.imread(path.with_suffix(".png"))
    grey = np.dot(rgba[..., :3], [0.2126, 0.7152, 0.0722])
    plt.imsave(path.with_name(path.name + "_grayscale").with_suffix(".png"), grey, cmap="gray", vmin=0, vmax=1)


def plot_recovery(ax: plt.Axes, recovery: pd.DataFrame) -> None:
    summary = (
        recovery.groupby(["regime", "chronology"], sort=False)
        .agg(
            median_recovery=("median_recovery_hours", "median"),
            median_zero=("zero_recovery_window_share", "median"),
            n=("median_recovery_hours", "size"),
        )
        .reset_index()
    )
    ax.set_xlim(-5, 82)
    ax.set_ylim(-0.75, 2.75)
    order = ["Europe", "China", "Japan"]
    for y, regime in enumerate(order):
        frame = summary.loc[summary["regime"].eq(regime)].set_index("chronology")
        start = float(frame.loc["dispersed", "median_recovery"])
        end = float(frame.loc["clustered", "median_recovery"])
        ax.annotate("", xy=(end, y), xytext=(start, y), arrowprops={"arrowstyle": "-|>", "color": COLORS[regime], "lw": 1.4})
        ax.scatter(start, y, s=31, color=DISPERSED, edgecolor="white", lw=0.5, zorder=3)
        ax.scatter(end, y, s=31, color=CLUSTERED, edgecolor="white", lw=0.5, zorder=3)
        zero = 100 * float(frame.loc["clustered", "median_zero"])
        ax.text(37, y + 0.18, f"{zero:.1f}% zero-window", color=COLORS[regime], fontweight="bold", ha="center", fontsize=5.5)
    ax.set_yticks(range(3), order)
    for tick, regime in zip(ax.get_yticklabels(), order):
        tick.set_color(COLORS[regime])
        tick.set_fontweight("bold")
    ax.set_xticks([0, 24, 48, 72])
    ax.set_xlabel("median inter-event recovery (h)")
    ax.legend(
        handles=[
            mpl.lines.Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=DISPERSED, markeredgecolor="white", label="dispersed"),
            mpl.lines.Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=CLUSTERED, markeredgecolor="white", label="clustered"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.52, 1.16),
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.25,
    )
    quiet(ax, "x")
    panel_label(ax, "a", -0.055, 1.08)


def plot_cycles(ax: plt.Axes, frame: pd.DataFrame, regime: str, label: str) -> None:
    frame = frame.sort_values(["inference_block", "system_id", "observation_year"]).reset_index(drop=True)
    colour = COLORS[regime]
    for _, row in frame.iterrows():
        ax.plot([0, 1], [row.cycles_dispersed, row.cycles_clustered], color=mpl.colors.to_rgba(colour, 0.13), lw=0.45, rasterized=True)
    ax.scatter(np.zeros(len(frame)), frame["cycles_dispersed"], s=8, color=mpl.colors.to_rgba(DISPERSED, 0.32), edgecolor="none", rasterized=True)
    ax.scatter(np.ones(len(frame)), frame["cycles_clustered"], s=8, color=mpl.colors.to_rgba(CLUSTERED, 0.32), edgecolor="none", rasterized=True)
    medians = [frame["cycles_dispersed"].median(), frame["cycles_clustered"].median()]
    ax.plot([0, 1], medians, color=INK, lw=1.3, zorder=4)
    ax.scatter([0, 1], medians, s=31, color=[DISPERSED, CLUSTERED], edgecolor="white", lw=0.55, zorder=5)
    ax.set_xlim(-0.35, 1.35)
    pad = 0.10 * max(
        float(np.ptp(frame[["cycles_dispersed", "cycles_clustered"]].to_numpy())),
        1.0,
    )
    lo = float(frame[["cycles_dispersed", "cycles_clustered"]].min().min())
    hi = float(frame[["cycles_dispersed", "cycles_clustered"]].max().max())
    ax.set_ylim(max(0, lo - pad), hi + pad)
    ax.set_xticks([0, 1], ["dispersed", "clustered"])
    ax.set_title(regime, color=colour, fontweight="bold", pad=3)
    ax.text(0.04, 0.94, f"median Δ = {frame.cycle_effect.median():.2f}", transform=ax.transAxes, ha="left", va="top", color=colour, fontweight="bold", fontsize=5.5)
    ax.text(0.96, 0.94, f"n = {len(frame)}", transform=ax.transAxes, ha="right", va="top", color=MUTED, fontsize=5.2)
    quiet(ax, "y")
    panel_label(ax, label, -0.18, 1.07)


def plot_value(ax: plt.Axes, frame: pd.DataFrame, regime: str, label: str) -> None:
    values = frame["signed_sequencing_effect_eur_kw_year"].to_numpy(float)
    colour = COLORS[regime]
    x, d = density(values)
    ax.fill_between(x, 0, 0.72 * d, color=mpl.colors.to_rgba(colour, 0.25), lw=0)
    ax.plot(x, 0.72 * d, color=colour, lw=0.95)
    ax.scatter(values, -0.08 + deterministic_jitter(len(values), 0.07), s=8.5, color=mpl.colors.to_rgba(colour, 0.42), edgecolor="white", lw=0.2, rasterized=True)
    q1, med, q3 = np.quantile(values, [0.25, 0.50, 0.75])
    ax.plot([q1, q3], [0.83, 0.83], color=colour, lw=3.2, solid_capstyle="round")
    ax.scatter(med, 0.83, s=30, color=colour, edgecolor="white", lw=0.55, zorder=4)
    ax.axvline(0, color=INK, lw=0.7, ls=(0, (3, 2)))
    ax.set_ylim(-0.23, 1.02)
    ax.set_yticks([])
    ax.set_xlabel("clustered − dispersed value\n(EUR kW$^{-1}$ yr$^{-1}$)")
    loss_share = 100 * np.mean(values < 0)
    absolute_median = float(np.median(np.abs(values)))
    ax.text(0.03, 0.94, f"{loss_share:.0f}% negative", transform=ax.transAxes, ha="left", va="top", color=colour, fontweight="bold", fontsize=5.5)
    ax.text(0.97, 0.94, f"|Δ| median {absolute_median:.2f}", transform=ax.transAxes, ha="right", va="top", color=MUTED, fontsize=5.2)
    quiet(ax)
    panel_label(ax, label, -0.18, 1.07)


def main() -> None:
    configure()
    recovery = pd.read_csv(SOURCE / "figure3_recovery_window_records.csv")
    records = pd.read_csv(SOURCE / "figure3_24h_value_cycle_records.csv")
    regimes = ["Europe", "China", "Japan"]

    fig = plt.figure(figsize=(WIDTH_MM * MM, HEIGHT_MM * MM))
    grid = fig.add_gridspec(
        3,
        3,
        left=0.075,
        right=0.985,
        bottom=0.07,
        top=0.945,
        height_ratios=[0.62, 1.10, 0.86],
        hspace=0.52,
        wspace=0.48,
    )
    plot_recovery(fig.add_subplot(grid[0, :]), recovery)
    for index, regime in enumerate(regimes):
        frame = records.loc[records["regime"].eq(regime)].copy()
        plot_cycles(fig.add_subplot(grid[1, index]), frame, regime, chr(ord("b") + index))
        plot_value(fig.add_subplot(grid[2, index]), frame, regime, chr(ord("e") + index))
    save_bundle(fig, "fig_v58_figure3_recovery_value_mechanism")
    plt.close(fig)


if __name__ == "__main__":
    main()
