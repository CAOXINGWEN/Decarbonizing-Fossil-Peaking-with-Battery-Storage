#!/usr/bin/env python3
"""Figure 1: sequencing-parity framework with parallel balancing regimes."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = HERE / "source_data_v58"
OUT = HERE / "main_figures"
OUT.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4
WIDTH_MM = 183
HEIGHT_MM = 146

INK = "#25343B"
MUTED = "#6B777B"
GRID = "#DCE1DE"
PALE = "#F3F5F2"
TASK = "#D97964"
RECHARGE = "#4D8FAE"
EUROPE = "#D97964"
CHINA = "#D3A341"
JAPAN = "#4D8FAE"
NATURAL = "#3B8877"
COST = "#7065A0"


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.6,
            "axes.labelsize": 6.6,
            "axes.titlesize": 7.0,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def panel_label(ax: plt.Axes, label: str, x: float = -0.025, y: float = 1.02) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8.2,
        fontweight="bold",
        ha="right",
        va="bottom",
        color=INK,
    )


def save_bundle(fig: plt.Figure, stem: str) -> None:
    path = OUT / stem
    fig.savefig(path.with_suffix(".pdf"), metadata={"Creator": "Python/matplotlib"})
    fig.savefig(path.with_suffix(".svg"))
    fig.savefig(path.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(path.with_suffix(".png"), dpi=360)
    rgba = plt.imread(path.with_suffix(".png"))
    grey = np.dot(rgba[..., :3], [0.2126, 0.7152, 0.0722])
    plt.imsave(path.with_name(path.name + "_grayscale").with_suffix(".png"), grey, cmap="gray", vmin=0, vmax=1)


def draw_sequence(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "a", -0.02, 1.00)
    ax.text(0.00, 0.98, "Same annual task, different event spacing", color=INK, fontweight="bold", va="top")

    event_height = np.array([0.15, 0.23, 0.18, 0.27, 0.20, 0.24])
    rows = [
        (0.64, "dispersed", np.array([0.10, 0.25, 0.40, 0.57, 0.73, 0.89]), RECHARGE),
        (0.29, "clustered", np.array([0.31, 0.39, 0.47, 0.55, 0.63, 0.71]), TASK),
    ]
    for y0, label, positions, colour in rows:
        ax.text(0.00, y0 + 0.10, label, color=colour, fontweight="bold", va="center")
        ax.plot([0.10, 0.94], [y0, y0], color=GRID, lw=0.8)
        xwave = np.linspace(0.10, 0.94, 260)
        ywave = y0 - 0.065 + 0.018 * np.sin(np.linspace(0, 13 * np.pi, len(xwave)))
        ax.plot(xwave, ywave, color=RECHARGE, lw=0.9)
        for xpos, height in zip(positions, event_height):
            ax.add_patch(
                Rectangle(
                    (xpos - 0.013, y0),
                    0.026,
                    height,
                    facecolor=TASK,
                    edgecolor="white",
                    lw=0.35,
                )
            )

    ax.annotate(
        "72 h recovery",
        xy=(0.25, 0.61),
        xytext=(0.40, 0.61),
        arrowprops={"arrowstyle": "<->", "color": NATURAL, "lw": 0.8},
        ha="center",
        va="center",
        color=NATURAL,
        fontsize=5.8,
    )
    ax.annotate(
        "0 h",
        xy=(0.47, 0.26),
        xytext=(0.52, 0.26),
        arrowprops={"arrowstyle": "<->", "color": NATURAL, "lw": 0.8},
        ha="center",
        va="center",
        color=NATURAL,
        fontsize=5.8,
    )
    ax.text(0.10, 0.03, "task energy", color=TASK, fontweight="bold")
    ax.text(0.30, 0.03, "+", color=MUTED)
    ax.text(0.35, 0.03, "recharge energy", color=RECHARGE, fontweight="bold")
    ax.text(0.61, 0.03, "held constant", color=INK)
    ax.plot([0.10, 0.26], [0.105, 0.105], color=TASK, lw=3.2, solid_capstyle="butt")
    ax.plot([0.35, 0.51], [0.105, 0.105], color=RECHARGE, lw=3.2, solid_capstyle="butt")


def draw_mechanism(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "b", -0.03, 1.00)
    ax.text(0.00, 0.98, "Recovery window sets the refill ceiling", color=INK, fontweight="bold", va="top")
    ax.text(
        0.04,
        0.76,
        r"$E_{\rm rebuild}=\min(R_{\rm window},\;P_{\rm ch}\Delta t,\;E_{\rm room})$",
        fontsize=8.0,
        color=INK,
    )
    for x, label, colour in [
        (0.16, "available\nrecharge", RECHARGE),
        (0.48, "interface ×\ntime", NATURAL),
        (0.78, "empty\ncapacity", COST),
    ]:
        ax.add_patch(Rectangle((x - 0.07, 0.53), 0.14, 0.09, facecolor=mpl.colors.to_rgba(colour, 0.18), edgecolor=colour, lw=0.8))
        ax.text(x, 0.575, label, ha="center", va="center", color=colour, fontsize=5.4, linespacing=1.05)
    ax.add_patch(FancyArrowPatch((0.49, 0.50), (0.49, 0.36), arrowstyle="-|>", mutation_scale=8, color=INK, lw=0.8))
    ax.text(0.49, 0.31, r"$\Delta t\rightarrow0\Rightarrow E_{\rm rebuild}\rightarrow0$", ha="center", color=TASK, fontsize=7.0, fontweight="bold")
    x = np.array([4, 8, 12, 18, 24, 36, 60, 96, 168, 240], float)
    y = np.array([0.0, -0.15, -0.42, -0.72, -0.64, -0.82, -0.30, 0.03, 0.45, 0.70])
    inset = ax.inset_axes([0.08, 0.02, 0.84, 0.20])
    inset.axhline(0, color=MUTED, lw=0.55)
    inset.fill_between(x, y, 0, where=y < 0, color=mpl.colors.to_rgba(TASK, 0.18), lw=0)
    inset.plot(x, y, color=TASK, lw=1.15)
    inset.set_xscale("log")
    inset.set_yticks([])
    inset.set_xticks([4, 24, 96, 240], ["4", "24", "96", "240"])
    inset.set_xlabel("storage duration (h)", labelpad=0.8)
    inset.spines[["left", "top", "right"]].set_visible(False)
    inset.tick_params(axis="x", pad=1)


def draw_regimes(ax: plt.Axes, task: pd.DataFrame, parity: pd.DataFrame) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "c", -0.012, 1.00)
    ax.text(0.00, 0.98, "Three balancing regimes test the same sequencing intervention", color=INK, fontweight="bold", va="top")
    colours = {"Europe": EUROPE, "China": CHINA, "Japan": JAPAN}
    labels = {
        "Europe": ("national", "gas peaking", "wind-solar surplus"),
        "China": ("provincial", "gas peaking", "wind-solar curtailment"),
        "Japan": ("community", "grid import", "distributed-PV reverse flow"),
    }
    for index, regime in enumerate(["Europe", "China", "Japan"]):
        x0 = index / 3
        x1 = (index + 1) / 3
        if index:
            ax.plot([x0, x0], [0.10, 0.88], color=GRID, lw=0.8)
        colour = colours[regime]
        contract = task.loc[task["regime"].eq(regime)].iloc[0]
        result = parity.loc[parity["regime"].eq(regime)].iloc[0]
        ax.scatter(x0 + 0.045, 0.80, s=42, color=colour, edgecolor="white", lw=0.5)
        ax.text(x0 + 0.082, 0.82, regime, color=colour, fontsize=7.2, fontweight="bold", va="center")
        ax.text(x0 + 0.082, 0.755, labels[regime][0], color=MUTED, va="center")
        ax.text(x0 + 0.045, 0.64, "task", color=MUTED, fontsize=5.3)
        ax.text(x0 + 0.045, 0.565, labels[regime][1], color=INK, fontweight="bold", fontsize=5.8)
        ax.text(x0 + 0.045, 0.45, "recharge", color=MUTED, fontsize=5.3)
        ax.text(x0 + 0.045, 0.375, labels[regime][2], color=INK, fontweight="bold", fontsize=5.6)
        ax.text(x1 - 0.045, 0.64, f"n = {int(contract.system_years)}", ha="right", color=INK, fontsize=5.6)
        ax.text(x0 + 0.045, 0.20, f"{result.median_sequencing_shift_eur_kw_year:.2f}", color=colour, fontsize=9.0, fontweight="bold")
        ax.text(x0 + 0.145, 0.205, "EUR kW$^{-1}$ yr$^{-1}$", color=MUTED, fontsize=5.1)
        ax.text(
            x0 + 0.045,
            0.105,
            f"{int(result.above_parity_count)}/{int(result.n_system_years)} reach cost scale",
            color=INK,
            fontsize=5.5,
        )


def draw_evidence_chain(ax: plt.Axes, weather: dict[str, object]) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "d", -0.012, 1.00)
    ax.text(0.00, 0.98, "Parity test: chronology value versus technology learning", color=INK, fontweight="bold", va="top")
    steps = [
        (0.10, "conserved\nsequencing", TASK),
        (0.36, "recovery-window\nmechanism", NATURAL),
        (0.62, "75 natural\noperating years", RECHARGE),
        (0.88, "2025–2035\ncost learning", COST),
    ]
    for index, (x, label, colour) in enumerate(steps):
        ax.scatter(x, 0.64, s=55, color=colour, edgecolor="white", lw=0.6, zorder=3)
        ax.text(x, 0.49, label, ha="center", va="top", color=INK, fontweight="bold", linespacing=1.15)
        if index < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + 0.055, 0.64), (steps[index + 1][0] - 0.055, 0.64), arrowstyle="-|>", mutation_scale=7, color=GRID, lw=1.2))
    ax.text(
        0.05,
        0.16,
        f"natural pairs  {weather['all_pairs_reaching_cost_scale']}/{weather['all_unordered_weather_pairs']}",
        color=RECHARGE,
        fontsize=6.2,
        fontweight="bold",
    )
    ax.text(
        0.39,
        0.16,
        f"adverse decades  {weather['matched_decades_erasing_learning']}/{weather['matched_ten_year_transitions']}",
        color=TASK,
        fontsize=6.2,
        fontweight="bold",
    )
    ax.text(
        0.72,
        0.16,
        f"cost scale  {weather['cost_improvement_2025_2035_eur_kw_year']:.2f}",
        color=COST,
        fontsize=6.2,
        fontweight="bold",
    )


def main() -> None:
    configure()
    task = pd.read_csv(SOURCE / "figure2_task_contract.csv")
    parity = pd.read_csv(SOURCE / "harmonized_parity/three_regime_24h_summary.csv")
    weather = json.loads(
        (ROOT / "weather_year_parity_v42/results/pecd_v42_75y_parity_summary.json").read_text(encoding="utf-8")
    )

    fig = plt.figure(figsize=(WIDTH_MM * MM, HEIGHT_MM * MM))
    grid = fig.add_gridspec(
        3,
        12,
        left=0.055,
        right=0.985,
        bottom=0.055,
        top=0.97,
        height_ratios=[1.03, 0.90, 0.70],
        hspace=0.22,
        wspace=0.65,
    )
    draw_sequence(fig.add_subplot(grid[0, :7]))
    draw_mechanism(fig.add_subplot(grid[0, 7:]))
    draw_regimes(fig.add_subplot(grid[1, :]), task, parity)
    draw_evidence_chain(fig.add_subplot(grid[2, :]), weather)
    save_bundle(fig, "fig_v58_figure1_sequencing_parity_framework")
    plt.close(fig)


if __name__ == "__main__":
    main()
