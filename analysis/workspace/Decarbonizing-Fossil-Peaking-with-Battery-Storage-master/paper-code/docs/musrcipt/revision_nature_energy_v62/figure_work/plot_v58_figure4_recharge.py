#!/usr/bin/env python3
"""Figure 4: available recharge cannot replace missing recovery time."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source_data_v58"
OUT = HERE / "main_figures"
OUT.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4
WIDTH_MM = 183
HEIGHT_MM = 166

INK = "#2B383E"
MUTED = "#6D777B"
GRID = "#DDE2DF"
TASK = "#D87863"
RECHARGE = "#4E8FAC"
TIME = "#3E8A78"
CAPACITY = "#7065A0"
EUROPE = "#D87863"
CHINA = "#D0A13F"
JAPAN = "#4E8FAC"
COLORS = {"Europe": EUROPE, "China": CHINA, "Japan": JAPAN}
MARKERS = {"Europe": "o", "China": "s", "Japan": "^"}


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


def panel_label(ax: plt.Axes, label: str, x: float = -0.11, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=8.2, fontweight="bold", ha="left", va="bottom", color=INK)


def quiet(ax: plt.Axes, grid: str | None = None) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.45)
        ax.set_axisbelow(True)


def save_bundle(fig: plt.Figure, stem: str) -> None:
    path = OUT / stem
    fig.savefig(path.with_suffix(".pdf"), metadata={"Creator": "Python/matplotlib"})
    fig.savefig(path.with_suffix(".svg"))
    fig.savefig(path.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(path.with_suffix(".png"), dpi=360)
    rgba = plt.imread(path.with_suffix(".png"))
    grey = np.dot(rgba[..., :3], [0.2126, 0.7152, 0.0722])
    plt.imsave(path.with_name(path.name + "_grayscale").with_suffix(".png"), grey, cmap="gray", vmin=0, vmax=1)


def plot_identity(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "a", -0.03, 1.00)
    ax.text(0.00, 0.98, "Refill is jointly limited by energy, interface and time", color=INK, fontweight="bold", va="top")
    ax.text(
        0.05,
        0.76,
        r"$E_{\mathrm{rebuild},w}=\min\{R_w,\;P_{\mathrm{ch}}\Delta t_w,\;E_{\mathrm{room},w}\}$",
        color=INK,
        fontsize=8.5,
    )
    boxes = [
        (0.08, 0.45, 0.20, 0.14, RECHARGE, "available recharge\n$R_w$"),
        (0.40, 0.45, 0.20, 0.14, TIME, "charging interface × time\n$P_{ch}\Delta t_w$"),
        (0.72, 0.45, 0.20, 0.14, CAPACITY, "empty capacity\n$E_{room,w}$"),
    ]
    for x, y, width, height, colour, label in boxes:
        ax.add_patch(Rectangle((x, y), width, height, facecolor=mpl.colors.to_rgba(colour, 0.17), edgecolor=colour, lw=0.8))
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", color=colour, fontsize=5.8, linespacing=1.15)
    ax.add_patch(FancyArrowPatch((0.50, 0.40), (0.50, 0.26), arrowstyle="-|>", mutation_scale=8, color=INK, lw=0.8))
    ax.text(0.50, 0.17, r"zero window $\Rightarrow$ zero refill for every finite resource", ha="center", va="center", color=TASK, fontsize=6.7, fontweight="bold")


def plot_analytical_limit(ax: plt.Axes) -> None:
    window = np.linspace(0, 72, 220)
    for multiplier, colour in [(1, RECHARGE), (2, TIME), (4, CAPACITY)]:
        recovered = np.minimum(multiplier * window / 36.0, 1.0)
        ax.plot(window, recovered, color=colour, lw=1.25, label=f"{multiplier}× recharge ceiling")
        label_x = 0.58 * 36.0 / multiplier
        ax.text(label_x + 0.9, 0.58, f"{multiplier}×", color=colour, va="center", fontsize=5.5, fontweight="bold")
    ax.axvline(0, color=TASK, lw=1.0)
    ax.scatter([0, 0, 0], [0, 0, 0], s=22, color=[RECHARGE, TIME, CAPACITY], edgecolor="white", lw=0.4, zorder=4)
    ax.text(2.2, 0.08, "all resources meet at zero", color=TASK, fontsize=5.5, fontweight="bold")
    ax.set_xlim(0, 80)
    ax.set_ylim(-0.03, 1.08)
    ax.set_xlabel("recovery-window length (h)")
    ax.set_ylabel("maximum normalized refill")
    ax.set_title("Analytical refill envelope", loc="left", color=INK, fontweight="bold", pad=3)
    quiet(ax, "y")
    panel_label(ax, "b", -0.15, 1.05)


def plot_regime_intervention(ax: plt.Axes, frame: pd.DataFrame, regime: str, label: str) -> None:
    frame = frame.sort_values("recharge_multiplier")
    colour = COLORS[regime]
    x = frame["recharge_multiplier"].to_numpy(float)
    y = frame["country_block_mean_effect_eur_kw_year"].to_numpy(float)
    low = frame["ci95_low"].to_numpy(float)
    high = frame["ci95_high"].to_numpy(float)
    ax.fill_between(x, low, high, color=mpl.colors.to_rgba(colour, 0.16), lw=0)
    ax.plot(x, y, color=colour, marker=MARKERS[regime], ms=4.4, lw=1.25, mfc=colour, mec="white", mew=0.5)
    ax.axhline(0, color=INK, lw=0.7, ls=(0, (3, 2)))
    ax.set_xticks([1, 2, 4], ["1×", "2×", "4×"])
    ax.set_xlabel("available-recharge multiplier")
    if regime == "Europe":
        ax.set_ylabel("clustered − dispersed value\n(EUR kW$^{-1}$ yr$^{-1}$)")
    ax.set_title(regime, color=colour, fontweight="bold", pad=3)
    ax.text(0.04, 0.93, f"n = {int(frame.n_system_years.iloc[0])}", transform=ax.transAxes, color=MUTED, va="top", fontsize=5.3)
    ax.text(0.96, 0.93, f"4× retains {frame.effect_retained_pct.iloc[-1]:.1f}%", transform=ax.transAxes, color=colour, ha="right", va="top", fontweight="bold", fontsize=5.3)
    span = max(float(high.max() - low.min()), 1.0)
    ax.set_ylim(float(low.min() - 0.16 * span), max(float(high.max() + 0.16 * span), 0.12 * span))
    quiet(ax, "y")
    panel_label(ax, label, -0.18, 1.06)


def plot_retention(ax: plt.Axes, summary: pd.DataFrame) -> None:
    for regime in ["Europe", "China", "Japan"]:
        frame = summary.loc[summary["regime"].eq(regime)].sort_values("recharge_multiplier")
        ax.plot(
            frame["recharge_multiplier"],
            frame["effect_retained_pct"],
            color=COLORS[regime],
            marker=MARKERS[regime],
            ms=4.7,
            lw=1.25,
            mfc=COLORS[regime],
            mec="white",
            mew=0.5,
        )
        last = frame.iloc[-1]
        label_y = float(last.effect_retained_pct)
        if regime == "Europe":
            label_y -= 3.0
        elif regime == "China":
            label_y += 3.0
        ax.text(float(last.recharge_multiplier) + 0.08, label_y, regime, color=COLORS[regime], va="center", fontweight="bold", fontsize=5.7)
    ax.axhline(100, color=INK, lw=0.75, ls=(0, (3, 2)))
    ax.set_xlim(0.8, 4.75)
    ax.set_ylim(88, 166)
    ax.set_xticks([1, 2, 4], ["1×", "2×", "4×"])
    ax.set_yticks([100, 120, 140, 160])
    ax.set_xlabel("available-recharge multiplier")
    ax.set_ylabel("sequencing effect retained (%)")
    quiet(ax, "y")
    panel_label(ax, "f", -0.105, 1.04)


def main() -> None:
    configure()
    summary = pd.read_csv(SOURCE / "figure4_recharge_intervention_summary.csv")
    fig = plt.figure(figsize=(WIDTH_MM * MM, HEIGHT_MM * MM))
    grid = fig.add_gridspec(
        3,
        12,
        left=0.125,
        right=0.985,
        bottom=0.07,
        top=0.955,
        height_ratios=[0.90, 1.0, 0.62],
        hspace=0.52,
        wspace=0.82,
    )
    plot_identity(fig.add_subplot(grid[0, :7]))
    plot_analytical_limit(fig.add_subplot(grid[0, 8:]))
    for index, regime in enumerate(["Europe", "China", "Japan"]):
        plot_regime_intervention(
            fig.add_subplot(grid[1, index * 4 : (index + 1) * 4]),
            summary.loc[summary["regime"].eq(regime)],
            regime,
            chr(ord("c") + index),
        )
    plot_retention(fig.add_subplot(grid[2, :]), summary)
    save_bundle(fig, "fig_v58_figure4_recharge_time_constraint")
    plt.close(fig)


if __name__ == "__main__":
    main()
