#!/usr/bin/env python3
"""Render the final v62 Figure 6 with a large natural-year transition panel."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REVISION = HERE.parents[1]
OUT = HERE.parent / "main_figures"
SOURCE = HERE / "source_data"
OUT.mkdir(parents=True, exist_ok=True)
SOURCE.mkdir(parents=True, exist_ok=True)

MM = 1.0 / 25.4
WIDTH_MM = 183.0
HEIGHT_MM = 232.0


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load plotting module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module(
    "v61_preview_figure6_base",
    REVISION / "figure_work" / "plot_v60_figure6_natural.py",
)
transition = load_module(
    "v61_preview_natural_transition",
    REVISION / "natural_year_heatmap" / "plot_v60_natural_year_transition_heatmap.py",
)


def save_bundle(fig: plt.Figure, stem: str) -> None:
    path = OUT / stem
    fig.savefig(path.with_suffix(".pdf"), metadata={"Creator": "Python/matplotlib"})
    fig.savefig(path.with_suffix(".svg"))
    fig.savefig(
        path.with_suffix(".tiff"),
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(path.with_suffix(".png"), dpi=360)


def draw_transition_panel(fig: plt.Figure, spec) -> None:
    values, pairs, decades, prepared, summary = transition.load_data()
    threshold = float(summary["cost_improvement_2025_2035_eur_kw_year"])
    marginals, matrix = transition.prepare_source_data(
        values,
        pairs,
        decades,
        prepared,
        threshold,
    )
    pair_source = pd.read_csv(transition.SOURCE / "natural_year_transition_pairs.csv")

    sub = spec.subgridspec(
        3,
        2,
        width_ratios=[1.0, 0.17],
        height_ratios=[0.21, 1.0, 0.060],
        wspace=0.12,
        hspace=0.32,
    )
    chronology_ax = fig.add_subplot(sub[0, 0])
    matrix_ax = fig.add_subplot(sub[1, 0])
    recovery_ax = fig.add_subplot(sub[1, 1], sharey=matrix_ax)
    cax = fig.add_subplot(sub[2, 0])
    fig.add_subplot(sub[0, 1]).axis("off")
    fig.add_subplot(sub[2, 1]).axis("off")

    transition.draw_value_chronology(chronology_ax, marginals)
    chronology_ax.set_title("", loc="left")
    image = transition.draw_transition_heatmap(
        matrix_ax,
        matrix,
        marginals,
        pair_source,
        decades,
        threshold,
    )
    transition.draw_recovery_structure(recovery_ax, marginals)
    matrix_ax.set_title(
        "75 natural years | 2,775 transitions",
        loc="left",
        fontsize=6.2,
        fontweight="bold",
        pad=2.5,
    )
    recovery_ax.set_title("Natural clustering", loc="left", fontsize=6.0, pad=2.5)
    recovery_ax.set_xlabel("")
    recovery_legend = recovery_ax.get_legend()
    if recovery_legend is not None:
        recovery_legend.remove()

    fig.canvas.draw()
    chronology_box = chronology_ax.get_position()
    matrix_box = matrix_ax.get_position()
    recovery_box = recovery_ax.get_position()
    colorbar_box = cax.get_position()
    chronology_ax.set_position(
        [matrix_box.x0, chronology_box.y0, matrix_box.width, chronology_box.height]
    )
    recovery_ax.set_position(
        [matrix_box.x1 + 0.011, matrix_box.y0, recovery_box.width, matrix_box.height]
    )
    cax.set_position(
        [matrix_box.x0, colorbar_box.y0, matrix_box.width, colorbar_box.height]
    )
    base.panel_label(chronology_ax, "c", x=-0.16, y=1.05)

    colorbar = fig.colorbar(
        image,
        cax=cax,
        orientation="horizontal",
        ticks=[-2, -1, 0, 1, 2],
    )
    colorbar.outline.set_linewidth(0.55)
    colorbar.ax.tick_params(width=0.5, length=2.2, pad=1.4, labelsize=5.0)
    colorbar.ax.axvline(-1, color="white", lw=1.0)
    colorbar.set_label(
        "natural-year value transition / 2025-2035 cost scale  (later - earlier)",
        labelpad=1.8,
        fontsize=5.3,
    )


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.1,
            "axes.labelsize": 6.2,
            "axes.titlesize": 6.5,
            "xtick.labelsize": 5.2,
            "ytick.labelsize": 5.2,
            "legend.fontsize": 5.0,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    base.SOURCE = SOURCE / "legacy_panels"
    transition.SOURCE = SOURCE / "transition_panel"
    base.SOURCE.mkdir(parents=True, exist_ok=True)
    transition.SOURCE.mkdir(parents=True, exist_ok=True)

    data = base.load_data()
    values = data["values"]
    by_cost = data["by_cost"]
    decades = data["decades"]
    summary = data["summary"]
    assert isinstance(values, pd.DataFrame)
    assert isinstance(by_cost, pd.DataFrame)
    assert isinstance(decades, pd.DataFrame)
    assert isinstance(summary, dict)
    threshold = float(summary["cost_improvement_2025_2035_eur_kw_year"])
    daily = base.daily_contrast(values, decades)

    fig = plt.figure(figsize=(WIDTH_MM * MM, HEIGHT_MM * MM))
    outer = fig.add_gridspec(
        3,
        16,
        left=0.080,
        right=0.982,
        bottom=0.050,
        top=0.963,
        height_ratios=[0.55, 0.80, 2.48],
        hspace=0.22,
        wspace=0.42,
    )

    ax_a = fig.add_subplot(outer[0, :])
    daily_grid = outer[1, :].subgridspec(2, 1, hspace=0.08)
    ax_b1 = fig.add_subplot(daily_grid[0, 0])
    ax_b2 = fig.add_subplot(daily_grid[1, 0], sharex=ax_b1)
    base.plot_a(ax_a, values, by_cost, decades, threshold)
    base.plot_c((ax_b1, ax_b2), daily)

    draw_transition_panel(fig, outer[2, 0:11])

    right = outer[2, 11:16].subgridspec(
        4,
        1,
        height_ratios=[0.90, 1.0, 0.82, 1.0],
        hspace=0.58,
    )
    ax_c = fig.add_subplot(right[0, 0])
    ax_d = fig.add_subplot(right[1, 0])
    ax_f = fig.add_subplot(right[2, 0])
    ax_g = fig.add_subplot(right[3, 0])
    base.plot_b(ax_c, by_cost)
    base.plot_d(ax_d, decades, threshold)
    base.plot_f(ax_f, summary)
    base.plot_g(ax_g, values)

    for artist in list(ax_c.texts):
        if artist.get_text() == "c":
            artist.remove()
    base.panel_label(ax_c, "d", x=-0.18, y=1.04)

    for artist in list(ax_d.texts):
        if artist.get_text() != "d":
            artist.remove()
    for artist in list(ax_d.texts):
        if artist.get_text() == "d":
            artist.remove()
    base.panel_label(ax_d, "e", x=-0.18, y=1.04)
    ax_d.text(
        0.98,
        0.94,
        "3/65 reinforce",
        transform=ax_d.transAxes,
        ha="right",
        va="top",
        color=base.GREEN,
        fontsize=5.2,
        fontweight="bold",
    )
    ax_d.text(
        0.98,
        0.06,
        "3/65 erase",
        transform=ax_d.transAxes,
        ha="right",
        va="bottom",
        color=base.RED,
        fontsize=5.2,
        fontweight="bold",
    )
    for artist in list(ax_f.texts):
        if artist.get_text().startswith(("all pairs", "adverse decades")):
            artist.remove()

    ax_c.set_xlabel("cost year")
    ax_d.set_xlabel("ten-year start")
    ax_f.set_xlabel("natural value span")
    ax_g.set_xlabel("zero-recovery windows (%)")
    legend = ax_g.get_legend()
    if legend is not None:
        legend.remove()
    for x, year in zip([0.49, 0.68, 0.86], [1950, 1987, 2024]):
        norm = mpl.colors.Normalize(vmin=1950, vmax=2024)
        ax_g.scatter(
            x,
            0.10,
            transform=ax_g.transAxes,
            s=10,
            color=base.YEAR_CMAP(norm(year)),
            edgecolor="white",
            linewidth=0.25,
            zorder=5,
        )
        ax_g.text(
            x + 0.025,
            0.10,
            str(year),
            transform=ax_g.transAxes,
            ha="left",
            va="center",
            fontsize=5.0,
        )

    save_bundle(fig, "fig_v62_figure6_temporal_parity")
    plt.close(fig)


if __name__ == "__main__":
    main()
