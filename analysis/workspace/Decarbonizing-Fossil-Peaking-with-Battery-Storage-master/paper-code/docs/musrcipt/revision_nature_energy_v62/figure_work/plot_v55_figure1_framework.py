#!/usr/bin/env python3
"""Build v55 Figure 1: natural discovery followed by mechanism validation."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parent
V55 = ROOT.parent
RESULTS = V55 / "weather_year_parity" / "results_europe"
OUT = ROOT / "main_figures"
SOURCE = ROOT / "source_data_v55"
OUT.mkdir(parents=True, exist_ok=True)
SOURCE.mkdir(parents=True, exist_ok=True)

MM = 1.0 / 25.4
INK = "#23343F"
MUTED = "#718087"
GRID = "#D7E0E3"
BLUE = "#2F7FAE"
TEAL = "#15988E"
CORAL = "#E56B2F"
PURPLE = "#7564A8"
GOLD = "#C99A28"
RED = "#C94B45"
GREEN = "#3D8F69"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 6.2,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.unicode_minus": False,
})


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = MUTED, lw: float = 0.8) -> None:
    ax.add_patch(FancyArrowPatch(start, end, transform=ax.transAxes, arrowstyle="-|>", mutation_scale=7, linewidth=lw, color=color))


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(0.015, 0.955, label, transform=ax.transAxes, fontsize=8.2, fontweight="bold", va="top")


def regime_icon(ax: plt.Axes, x: float, y: float, color: str, label: str) -> None:
    ax.add_patch(Circle((x, y), 0.034, transform=ax.transAxes, facecolor=mpl.colors.to_rgba(color, 0.12), edgecolor=color, linewidth=0.8))
    ax.add_patch(Rectangle((x - 0.018, y - 0.008), 0.009, 0.025, transform=ax.transAxes, facecolor=color, edgecolor="none"))
    ax.add_patch(Rectangle((x - 0.004, y - 0.008), 0.009, 0.037, transform=ax.transAxes, facecolor=color, edgecolor="none"))
    ax.add_patch(Rectangle((x + 0.010, y - 0.008), 0.009, 0.020, transform=ax.transAxes, facecolor=color, edgecolor="none"))
    ax.text(x, y - 0.052, label, transform=ax.transAxes, ha="center", va="top", color=color, fontsize=5.5, fontweight="bold")


def save_bundle(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), metadata={"Creator": "Python/matplotlib"})
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    rgba = plt.imread(stem.with_suffix(".png"))
    grey = np.dot(rgba[..., :3], [0.2126, 0.7152, 0.0722])
    plt.imsave(stem.with_name(stem.name + "_grayscale").with_suffix(".png"), grey, cmap="gray", vmin=0, vmax=1)


def build() -> None:
    values = pd.read_csv(RESULTS / "pecd_europe_34_window_relative_values_2025.csv")
    europe = values[values.scope.eq("Europe33_copperplate")].sort_values("window_start_year")
    risk = pd.read_csv(RESULTS / "pecd_europe_34_window_directional_risk.csv")
    decade = risk[
        risk.unit_id.eq("Europe33_copperplate")
        & risk.start_cost_year.eq(2025)
        & risk.end_cost_year.eq(2035)
    ].iloc[0]

    fig, axes = plt.subplots(1, 4, figsize=(183 * MM, 104 * MM), gridspec_kw={"wspace": 0.04})
    fig.subplots_adjust(left=0.012, right=0.988, bottom=0.035, top=0.975)
    backgrounds = ["#F4F7F8", "#F1F8F7", "#FBF6F1", "#F6F4F8"]
    for ax, face in zip(axes, backgrounds):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_facecolor(face)

    # a: source and temporal contract
    ax = axes[0]
    panel_label(ax, "a")
    ax.text(0.10, 0.89, "Natural weather record", transform=ax.transAxes, fontsize=6.9, fontweight="bold", color=INK)
    ax.text(0.10, 0.835, "PECD 2021.3", transform=ax.transAxes, fontsize=5.6, color=MUTED)
    years = np.arange(1982, 2017)
    heights = 0.025 + 0.045 * (0.5 + 0.5 * np.sin(np.arange(len(years)) * 1.7))
    x = np.linspace(0.10, 0.90, len(years))
    for xpos, height in zip(x, heights):
        ax.plot([xpos, xpos], [0.72, 0.72 + height], transform=ax.transAxes, color=BLUE, lw=1.05)
    ax.plot([0.10, 0.90], [0.72, 0.72], transform=ax.transAxes, color=INK, lw=0.65)
    ax.text(0.10, 0.682, "1982", transform=ax.transAxes, fontsize=5.2, color=MUTED)
    ax.text(0.90, 0.682, "2016", transform=ax.transAxes, fontsize=5.2, color=MUTED, ha="right")
    ax.text(0.10, 0.59, "35 climate years", transform=ax.transAxes, fontsize=7.4, fontweight="bold", color=BLUE)
    ax.text(0.10, 0.535, "34 complete July-June\noperating years", transform=ax.transAxes, fontsize=5.6, color=INK, linespacing=1.25)
    ax.add_patch(Circle((0.30, 0.345), 0.095, transform=ax.transAxes, facecolor="white", edgecolor=GRID, linewidth=0.8))
    nodes = np.array([[0.25, 0.38], [0.32, 0.40], [0.35, 0.32], [0.27, 0.29], [0.21, 0.33]])
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2), (1, 3)]:
        ax.plot(nodes[[i, j], 0], nodes[[i, j], 1], transform=ax.transAxes, color=GRID, lw=0.7)
    ax.scatter(nodes[:, 0], nodes[:, 1], transform=ax.transAxes, s=12, color=TEAL, edgecolor="white", linewidth=0.4)
    ax.text(0.45, 0.365, "Europe 33", transform=ax.transAxes, fontsize=6.5, color=TEAL, fontweight="bold")
    ax.text(0.45, 0.320, "synchronized\nwind-solar balance", transform=ax.transAxes, fontsize=5.4, color=INK, va="top")
    ax.text(0.10, 0.155, "Natural chronology varies", transform=ax.transAxes, fontsize=5.5, color=INK, fontweight="bold")
    ax.text(0.10, 0.110, "event shape, duration and\nrecovery geometry", transform=ax.transAxes, fontsize=5.2, color=MUTED, va="top")

    # b: natural parity result
    ax = axes[1]
    panel_label(ax, "b")
    ax.text(0.10, 0.89, "Natural sequencing parity", transform=ax.transAxes, fontsize=6.9, fontweight="bold", color=INK)
    ax.text(0.10, 0.835, "same system scale and annual energy", transform=ax.transAxes, fontsize=5.4, color=MUTED)
    gx = np.linspace(0.11, 0.89, len(europe))
    gy = 0.56 + 0.12 * (europe.value_2025_eur_kw_year.to_numpy() - europe.value_2025_eur_kw_year.min()) / europe.value_2025_eur_kw_year.max().__sub__(europe.value_2025_eur_kw_year.min())
    ax.plot(gx, gy, transform=ax.transAxes, color=TEAL, lw=1.0)
    ax.scatter(gx, gy, transform=ax.transAxes, s=6, color=TEAL, edgecolor="white", linewidth=0.25)
    for start, color in [(1988, GREEN), (1998, RED)]:
        idx = int(np.flatnonzero(europe.window_start_year.to_numpy() == start)[0])
        ax.scatter(gx[idx], gy[idx], transform=ax.transAxes, s=24, color=color, edgecolor="white", linewidth=0.6, zorder=5)
        ax.text(gx[idx], gy[idx] + 0.047, f"{start}/{str(start+1)[-2:]}", transform=ax.transAxes, ha="center", color=color, fontsize=5.0, fontweight="bold")
    i0 = int(np.flatnonzero(europe.window_start_year.to_numpy() == 1988)[0])
    i1 = int(np.flatnonzero(europe.window_start_year.to_numpy() == 1998)[0])
    ax.add_patch(FancyArrowPatch((gx[i0], gy[i0] - 0.03), (gx[i1], gy[i1] - 0.03), transform=ax.transAxes, arrowstyle="-|>", mutation_scale=7, linewidth=0.8, color=RED, connectionstyle="arc3,rad=0.18"))
    ax.text(0.50, 0.485, "natural decade shift", transform=ax.transAxes, ha="center", fontsize=5.2, color=MUTED)
    ax.text(0.50, 0.430, "-7.60", transform=ax.transAxes, ha="center", fontsize=10.5, color=RED, fontweight="bold")
    ax.text(0.50, 0.382, "EUR kW$^{-1}$ yr$^{-1}$", transform=ax.transAxes, ha="center", fontsize=5.2, color=INK)
    ax.plot([0.14, 0.86], [0.285, 0.285], transform=ax.transAxes, color=GRID, lw=6, solid_capstyle="butt")
    ax.plot([0.14, 0.14 + 0.72 * 5.296038099 / 7.597154652,], [0.285, 0.285], transform=ax.transAxes, color=PURPLE, lw=6, solid_capstyle="butt")
    ax.text(0.14, 0.330, "2025-2035 cost improvement", transform=ax.transAxes, fontsize=5.3, color=INK, fontweight="bold")
    ax.text(0.86, 0.273, "+5.30", transform=ax.transAxes, ha="right", va="top", fontsize=6.2, color=PURPLE, fontweight="bold")
    ax.text(0.10, 0.170, f"full span  {decade.full_natural_span_eur_kw_year:.2f}", transform=ax.transAxes, fontsize=5.5, color=INK)
    ax.text(0.10, 0.120, f"central 90%  {decade.q95_q05_natural_span_eur_kw_year:.2f}", transform=ax.transAxes, fontsize=5.5, color=INK)

    # c: cross-regime mechanism identification
    ax = axes[2]
    panel_label(ax, "c")
    ax.text(0.10, 0.89, "Mechanism across regimes", transform=ax.transAxes, fontsize=6.9, fontweight="bold", color=INK)
    regime_icon(ax, 0.23, 0.755, BLUE, "Europe")
    regime_icon(ax, 0.50, 0.755, CORAL, "China")
    regime_icon(ax, 0.77, 0.755, TEAL, "Japan")
    ax.text(0.10, 0.615, "observed peaking task + recharge", transform=ax.transAxes, fontsize=5.3, color=MUTED)
    for y0, clustered in [(0.505, False), (0.350, True)]:
        positions = np.array([0.17, 0.34, 0.51, 0.68, 0.85]) if not clustered else np.array([0.29, 0.39, 0.49, 0.59, 0.69])
        for j, xpos in enumerate(positions):
            height = [0.046, 0.070, 0.053, 0.076, 0.058][j]
            ax.add_patch(Rectangle((xpos - 0.018, y0), 0.036, height, transform=ax.transAxes, facecolor=CORAL, edgecolor="white", linewidth=0.3))
        wave_x = np.linspace(0.13, 0.89, 120)
        wave_y = y0 - 0.040 + 0.012 * np.sin(np.linspace(0, 8 * np.pi, 120))
        ax.plot(wave_x, wave_y, transform=ax.transAxes, color=TEAL, lw=0.9)
    ax.text(0.10, 0.560, "dispersed", transform=ax.transAxes, color=BLUE, fontsize=5.5, fontweight="bold")
    ax.text(0.10, 0.405, "clustered", transform=ax.transAxes, color=CORAL, fontsize=5.5, fontweight="bold")
    ax.annotate("72 h", xy=(0.32, 0.452), xytext=(0.48, 0.452), xycoords=ax.transAxes, textcoords=ax.transAxes, ha="center", va="center", fontsize=5.2, color=TEAL, arrowprops={"arrowstyle": "<->", "color": TEAL, "lw": 0.7})
    ax.annotate("0 h", xy=(0.45, 0.297), xytext=(0.51, 0.297), xycoords=ax.transAxes, textcoords=ax.transAxes, ha="center", va="center", fontsize=5.2, color=TEAL, arrowprops={"arrowstyle": "<->", "color": TEAL, "lw": 0.7})
    ax.text(0.10, 0.190, "intact 24 h blocks preserve", transform=ax.transAxes, fontsize=5.4, color=INK, fontweight="bold")
    ax.text(0.10, 0.142, "annual task, recharge and\nwithin-day task-recharge coupling", transform=ax.transAxes, fontsize=5.2, color=MUTED, va="top")

    # d: portfolio and implication
    ax = axes[3]
    panel_label(ax, "d")
    ax.text(0.10, 0.89, "Investment consequence", transform=ax.transAxes, fontsize=6.9, fontweight="bold", color=INK)
    ax.text(0.10, 0.835, "common intraday layer + duration tail", transform=ax.transAxes, fontsize=5.4, color=MUTED)
    # battery and storage tail
    ax.add_patch(Rectangle((0.12, 0.655), 0.29, 0.115, transform=ax.transAxes, facecolor=mpl.colors.to_rgba(BLUE, 0.10), edgecolor=BLUE, linewidth=0.9))
    for xpos in [0.15, 0.21, 0.27, 0.33]:
        ax.add_patch(Rectangle((xpos, 0.682), 0.045, 0.060, transform=ax.transAxes, facecolor=BLUE, edgecolor="none"))
    ax.text(0.265, 0.630, "common 4 h LFP", transform=ax.transAxes, ha="center", fontsize=5.3, color=BLUE, fontweight="bold")
    ax.add_patch(Polygon([(0.58, 0.655), (0.82, 0.655), (0.77, 0.770), (0.63, 0.770)], transform=ax.transAxes, closed=True, facecolor=mpl.colors.to_rgba(PURPLE, 0.12), edgecolor=PURPLE, linewidth=0.9))
    ax.text(0.70, 0.630, "LDES tail", transform=ax.transAxes, ha="center", fontsize=5.3, color=PURPLE, fontweight="bold")
    arrow(ax, (0.42, 0.712), (0.56, 0.712), color=TEAL, lw=0.9)
    # cross-duration response
    xcurve = np.linspace(0.12, 0.88, 180)
    u = (xcurve - xcurve.min()) / (xcurve.max() - xcurve.min())
    ycurve = 0.46 - 0.10 * np.exp(-((u - 0.48) / 0.25) ** 2) + 0.075 / (1 + np.exp(-(u - 0.76) * 20))
    ax.plot([0.12, 0.88], [0.45, 0.45], transform=ax.transAxes, color=MUTED, lw=0.55)
    ax.fill_between(xcurve, ycurve, 0.45, where=ycurve < 0.45, transform=ax.transAxes, color=mpl.colors.to_rgba(CORAL, 0.20), linewidth=0)
    ax.fill_between(xcurve, ycurve, 0.45, where=ycurve >= 0.45, transform=ax.transAxes, color=mpl.colors.to_rgba(PURPLE, 0.18), linewidth=0)
    ax.plot(xcurve, ycurve, transform=ax.transAxes, color=INK, lw=1.1)
    ax.text(0.43, 0.338, "intermediate-duration\nvalue loss", transform=ax.transAxes, ha="center", color=CORAL, fontsize=5.2, fontweight="bold")
    ax.text(0.79, 0.532, "long-duration\ncoverage", transform=ax.transAxes, ha="center", color=PURPLE, fontsize=5.2, fontweight="bold")
    ax.text(0.50, 0.245, "2035 Europe 33", transform=ax.transAxes, ha="center", fontsize=5.4, color=MUTED)
    ax.text(0.50, 0.182, "0 / 34", transform=ax.transAxes, ha="center", fontsize=10.5, color=RED, fontweight="bold")
    ax.text(0.50, 0.127, "weather years in which 24 h A-CAES\noutperforms an additional 4 h LFP tail", transform=ax.transAxes, ha="center", va="top", fontsize=5.1, color=INK)

    # Directional arrows between evidence stages.
    for left, right in zip(axes[:-1], axes[1:]):
        left_pos = left.get_position()
        right_pos = right.get_position()
        fig.add_artist(FancyArrowPatch((left_pos.x1 + 0.001, 0.50), (right_pos.x0 - 0.001, 0.50), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=7, linewidth=0.7, color="#9DACB2"))

    pd.DataFrame([
        ("source_climate_years", 35, "years"),
        ("complete_july_june_operating_years", 34, "years"),
        ("europe_scope", 33, "countries"),
        ("natural_decade_value_change", -7.597154652126802, "EUR kW-1 yr-1"),
        ("relative_cost_improvement_2025_2035", 5.2960380996609935, "EUR kW-1 yr-1"),
        ("natural_full_span", float(decade.full_natural_span_eur_kw_year), "EUR kW-1 yr-1"),
        ("natural_central_90_span", float(decade.q95_q05_natural_span_eur_kw_year), "EUR kW-1 yr-1"),
        ("caes24_outperformance_count_2035", 0, "of 34"),
    ], columns=["metric", "value", "unit"]).to_csv(SOURCE / "figure1_framework_metrics.csv", index=False)

    stem = OUT / "fig_v55_figure1_natural_sequencing_framework"
    save_bundle(fig, stem)
    plt.close(fig)


if __name__ == "__main__":
    build()
