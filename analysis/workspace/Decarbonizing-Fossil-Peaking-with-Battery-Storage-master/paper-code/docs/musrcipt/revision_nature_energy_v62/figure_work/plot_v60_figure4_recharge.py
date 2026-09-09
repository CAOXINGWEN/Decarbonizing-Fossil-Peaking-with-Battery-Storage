#!/usr/bin/env python3
"""Figure 4: recharge abundance cannot replace a missing recovery window."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle, Wedge

from v60_figure_style import (
    BLUE,
    CORAL,
    GOLD,
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
SOURCE = HERE / "source_data_v60"
OUT = HERE / "main_figures"
HEIGHT_MM = 174.0
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


def plot_refill_phase(fig: plt.Figure, spec) -> None:
    sub = spec.subgridspec(1, 16, wspace=0.55)
    ax = fig.add_subplot(sub[0, :14])
    window = np.linspace(0.0, 72.0, 241)
    resource = np.linspace(0.8, 4.2, 171)
    refill = np.minimum(resource[:, None] * window[None, :] / 36.0, 1.0)
    cmap = LinearSegmentedColormap.from_list(
        "refill_phase",
        [CORAL, "#F1D8CF", "#F7F5F0", "#A8CEC2", TEAL],
    )
    image = ax.pcolormesh(
        window,
        resource,
        refill,
        shading="auto",
        cmap=cmap,
        norm=Normalize(0, 1),
        rasterized=True,
    )
    contours = ax.contour(
        window,
        resource,
        refill,
        levels=[0.25, 0.50, 0.75, 0.99],
        colors=INK,
        linewidths=0.48,
        alpha=0.72,
    )
    ax.text(
        0.98,
        0.035,
        "refill contours   25%   50%   75%   full",
        transform=ax.transAxes,
        color=INK,
        ha="right",
        va="bottom",
        fontsize=5.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.8},
    )
    ax.axvline(0, color=INK, lw=0.7)
    ax.text(
        2.2,
        3.88,
        "zero-hour gate:\nno refill at any multiplier",
        color=CORAL,
        fontweight="bold",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.8},
    )
    ax.text(70.2, 1.08, "inventory rebuilt", color=TEAL, ha="right", va="bottom", fontweight="bold")
    ax.set_xlim(0, 72)
    ax.set_ylim(0.8, 4.2)
    ax.set_xticks([0, 12, 24, 36, 48, 60, 72])
    ax.set_yticks([1, 2, 3, 4], ["1×", "2×", "3×", "4×"])
    ax.set_xlabel("inter-event recovery window (h)")
    ax.set_ylabel("available-recharge multiplier")
    panel_label(ax, "a", -0.12, 1.08)
    ax.text(
        0.0,
        1.055,
        "Refill phase space is bounded by time",
        transform=ax.transAxes,
        color=INK,
        fontweight="bold",
        va="bottom",
    )
    ax.spines[["top", "right"]].set_visible(False)

    cax = fig.add_subplot(sub[0, 15])
    cb = fig.colorbar(image, cax=cax, orientation="vertical")
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(["0", "50", "100"])
    cb.set_label("maximum inventory rebuilt (%)", fontsize=5.5, labelpad=2.0)
    cb.ax.tick_params(labelsize=5.5, length=1.6, pad=1)


def draw_unit_row(
    ax: plt.Axes,
    x0: float,
    y0: float,
    filled: int,
    colour: str,
    closed: bool = False,
) -> None:
    cell_w = 0.021
    gap = 0.004
    for index in range(12):
        x = x0 + index * (cell_w + gap)
        face = mpl.colors.to_rgba(colour, 0.72) if index < filled else "white"
        edge = colour if index < filled else GRID
        ax.add_patch(Rectangle((x, y0), cell_w, 0.075, facecolor=face, edgecolor=edge, lw=0.42))
    if closed:
        x1 = x0 + 12 * (cell_w + gap) - gap
        ax.plot([x0 - 0.014, x1 + 0.014], [y0 + 0.094, y0 - 0.018], color=CORAL, lw=1.0)
        ax.plot([x0 - 0.014, x1 + 0.014], [y0 - 0.018, y0 + 0.094], color=CORAL, lw=1.0)


def plot_resource_gate(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "b", -0.06, 1.01)
    ax.text(
        0.0,
        0.99,
        "More resource helps only while the gate is open",
        color=INK,
        fontweight="bold",
        va="top",
    )
    ax.text(0.28, 0.80, "12 h window", color=TEAL, ha="center", fontweight="bold")
    ax.text(0.78, 0.80, "0 h window", color=CORAL, ha="center", fontweight="bold")
    colours = [BLUE, GOLD, PURPLE]
    for row, (multiplier, filled, colour) in enumerate(zip([1, 2, 4], [4, 8, 12], colours)):
        y = 0.61 - row * 0.19
        ax.text(0.02, y + 0.038, f"{multiplier}×", color=colour, va="center", fontweight="bold")
        draw_unit_row(ax, 0.10, y, filled, colour)
        draw_unit_row(ax, 0.60, y, 0, colour, closed=True)
        ax.text(
            0.41,
            y + 0.038,
            f"{filled}/12",
            color=colour,
            ha="left",
            va="center",
            fontsize=5.5,
            fontweight="bold",
        )
        ax.text(
            0.91,
            y + 0.038,
            "0/12",
            color=CORAL,
            ha="left",
            va="center",
            fontsize=5.5,
            fontweight="bold",
        )
    ax.text(
        0.50,
        0.08,
        r"$E_{\mathrm{rebuild}}=\min\{R,\,P_{\mathrm{ch}}\Delta t,\,E_{\mathrm{room}}\}$",
        color=INK,
        ha="center",
        va="center",
        fontsize=6.7,
    )


def draw_window_waffle(
    ax: plt.Axes,
    x0: float,
    y0: float,
    zero_share: float,
) -> None:
    size = 0.031
    gap = 0.004
    zero_cells = int(round(100 * zero_share))
    for index in range(100):
        row, col = divmod(index, 10)
        face = CORAL if index < zero_cells else TEAL
        ax.add_patch(
            Rectangle(
                (x0 + col * (size + gap), y0 + row * (size + gap)),
                size,
                size,
                facecolor=mpl.colors.to_rgba(face, 0.82),
                edgecolor="white",
                lw=0.30,
            )
        )


def plot_window_states(ax: plt.Axes, recovery: pd.DataFrame) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "b", -0.06, 1.01)
    ax.text(
        0.0,
        0.99,
        "Clustering closes 74.7% of recovery windows",
        color=INK,
        fontweight="bold",
        va="top",
    )
    shares = (
        recovery.groupby("chronology")["zero_recovery_window_share"]
        .median()
        .reindex(["dispersed", "clustered"])
    )
    for chronology, x0, colour in zip(
        ["dispersed", "clustered"],
        [0.10, 0.58],
        [BLUE, CORAL],
    ):
        share = float(shares.loc[chronology])
        draw_window_waffle(ax, x0, 0.26, share)
        ax.text(
            x0 + 0.157,
            0.69,
            chronology,
            color=colour,
            ha="center",
            fontweight="bold",
        )
        ax.text(
            x0 + 0.157,
            0.17,
            f"{100 * share:.1f}% zero-window",
            color=colour,
            ha="center",
            fontsize=5.5,
            fontweight="bold",
        )
    ax.scatter(
        [0.16, 0.38],
        [0.07, 0.07],
        s=22,
        marker="s",
        color=[TEAL, CORAL],
        edgecolor="white",
        lw=0.35,
    )
    ax.text(0.19, 0.07, "open", va="center", color=TEAL, fontsize=5.5)
    ax.text(0.41, 0.07, "zero-window", va="center", color=CORAL, fontsize=5.5)
    ax.text(
        0.98,
        0.86,
        "185 conserved sequence pairs",
        ha="right",
        va="center",
        color=MUTED,
        fontsize=5.5,
    )


def plot_regime_interval(ax: plt.Axes, frame: pd.DataFrame, regime: str, label: str) -> None:
    frame = frame.sort_values("recharge_multiplier")
    colour = REGIME_COLORS[regime]
    y = np.arange(3)[::-1]
    means = frame["country_block_mean_effect_eur_kw_year"].to_numpy(float)
    low = frame["ci95_low"].to_numpy(float)
    high = frame["ci95_high"].to_numpy(float)
    for yi, lo, mean, hi in zip(y, low, means, high):
        ax.plot(
            [lo, hi],
            [yi, yi],
            color=mpl.colors.to_rgba(colour, 0.62),
            lw=1.5,
            solid_capstyle="round",
        )
        ax.scatter(
            mean,
            yi,
            s=34,
            marker=REGIME_MARKERS[regime],
            color=colour,
            edgecolor="white",
            lw=0.55,
            zorder=4,
        )
    ax.axvline(0, color=INK, lw=0.65, ls=(0, (3, 2)))
    ax.set_yticks(y, ["1×", "2×", "4×"])
    ax.set_xlabel("clustered − dispersed value\n(EUR kW$^{-1}$ yr$^{-1}$)")
    ax.set_title(regime, color=colour, fontweight="bold", pad=2)
    ax.text(
        0.03,
        0.04,
        f"n = {int(frame['n_system_years'].iloc[0])}",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=5.5,
    )
    ax.text(
        0.97,
        0.96,
        f"4×: {means[-1]:.2f}",
        transform=ax.transAxes,
        color=colour,
        ha="right",
        va="top",
        fontweight="bold",
        fontsize=5.5,
    )
    span = max(float(high.max() - low.min()), 1.0)
    ax.set_xlim(
        float(low.min() - 0.10 * span),
        max(float(high.max() + 0.12 * span), 0.05 * span),
    )
    ax.set_ylim(-0.55, 2.55)
    panel_label(ax, label, -0.20, 1.05)
    quiet(ax, "x")


def plot_retention_matrix(fig: plt.Figure, spec, summary: pd.DataFrame) -> None:
    ax = fig.add_subplot(spec)
    order = ["Europe", "China", "Japan"]
    matrix = np.vstack(
        [
            summary.loc[summary["regime"].eq(regime)]
            .sort_values("recharge_multiplier")["effect_retained_pct"]
            .to_numpy(float)
            for regime in order
        ]
    )
    cmap = LinearSegmentedColormap.from_list("retention", ["#F4F2EE", "#E7CBA7", CORAL])
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=100, vmax=160, interpolation="nearest")
    for row in range(3):
        for col in range(3):
            value = matrix[row, col]
            value_label = f"{value:.1f}%" if abs(value - 100) > 0.05 else "100%"
            ax.text(
                col,
                row,
                value_label,
                ha="center",
                va="center",
                color=INK if value < 145 else "white",
                fontweight="bold",
                fontsize=6.0,
            )
    ax.set_xticks([0, 1, 2], ["1×", "2×", "4×"])
    ax.set_yticks([0, 1, 2], order)
    for tick, regime in zip(ax.get_yticklabels(), order):
        tick.set_color(REGIME_COLORS[regime])
        tick.set_fontweight("bold")
    ax.tick_params(length=0)
    ax.set_xlabel("available-recharge multiplier")
    panel_label(ax, "f", -0.095, 1.10)
    ax.text(
        0.0,
        1.075,
        "Sequencing loss survives a fourfold recharge endowment",
        transform=ax.transAxes,
        color=INK,
        fontweight="bold",
        va="bottom",
    )
    ax.text(1.0, 1.075, "colour scale  100 → 160%", transform=ax.transAxes, color=MUTED, ha="right", va="bottom", fontsize=5.5)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)


def plot_retention_arcs(fig: plt.Figure, spec, summary: pd.DataFrame) -> None:
    sub = spec.subgridspec(1, 3, wspace=0.34)
    multipliers = [1, 2, 4]
    radii = [0.43, 0.32, 0.21]
    for index, regime in enumerate(["Europe", "China", "Japan"]):
        ax = fig.add_subplot(sub[0, index])
        frame = summary.loc[summary["regime"].eq(regime)].sort_values("recharge_multiplier")
        values = frame["effect_retained_pct"].to_numpy(float)
        colour = REGIME_COLORS[regime]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.axis("off")
        for ring_index, (multiplier, radius, value) in enumerate(
            zip(multipliers, radii, values)
        ):
            ax.add_patch(
                Wedge(
                    (0.38, 0.48),
                    radius,
                    90,
                    450,
                    width=0.065,
                    facecolor="#ECEDEA",
                    edgecolor="none",
                )
            )
            ax.add_patch(
                Wedge(
                    (0.38, 0.48),
                    radius,
                    90,
                    90 + 360 * min(value, 160) / 160,
                    width=0.065,
                    facecolor=mpl.colors.to_rgba(colour, 0.42 + 0.12 * ring_index),
                    edgecolor="none",
                )
            )
            ax.text(
                0.96,
                0.70 - 0.17 * ring_index,
                f"{multiplier}×  {value:.1f}%",
                ha="right",
                va="center",
                color=colour,
                fontweight="bold",
                fontsize=5.5,
            )
        ax.text(
            0.38,
            0.48,
            regime,
            ha="center",
            va="center",
            color=colour,
            fontweight="bold",
            fontsize=6.2,
        )
        ax.text(0.38, 0.05, "retained effect", ha="center", color=MUTED, fontsize=5.1)
        if index == 0:
            panel_label(ax, "f", -0.14, 1.07)
            ax.text(
                0.0,
                1.04,
                "Sequencing loss survives a fourfold recharge endowment",
                transform=ax.transAxes,
                color=INK,
                fontweight="bold",
                va="bottom",
            )


def main() -> None:
    configure_publication()
    summary = pd.read_csv(SOURCE / "figure4_recharge_intervention_summary.csv")
    recovery = pd.read_csv(SOURCE / "figure3_recovery_window_records.csv")
    fig = plt.figure(figsize=(WIDTH_MM * MM, HEIGHT_MM * MM))
    grid = fig.add_gridspec(
        3,
        12,
        left=0.105,
        right=0.985,
        bottom=0.065,
        top=0.965,
        height_ratios=[1.08, 0.78, 0.52],
        hspace=0.53,
        wspace=0.90,
    )
    plot_refill_phase(fig, grid[0, :7])
    plot_window_states(fig.add_subplot(grid[0, 7:]), recovery)
    for index, regime in enumerate(["Europe", "China", "Japan"]):
        plot_regime_interval(
            fig.add_subplot(grid[1, index * 4 : (index + 1) * 4]),
            summary.loc[summary["regime"].eq(regime)],
            regime,
            chr(ord("c") + index),
        )
    plot_retention_arcs(fig, grid[2, :], summary)
    export_figure(fig, "fig_v60_figure4_recharge_time_constraint")
    plt.close(fig)


if __name__ == "__main__":
    main()
