#!/usr/bin/env python3
"""Figure 6: natural-weather transitions and controlled sequencing surfaces."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import SymLogNorm
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
REVISION = HERE.parent
OUT = HERE / "main_figures"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REVISION))

from natural_year_heatmap import plot_v60_natural_year_transition_heatmap as natural  # noqa: E402
from cluster_intensity_surface import plot_v60_cluster_intensity_surface as cluster  # noqa: E402


MM = 1.0 / 25.4
WIDTH_MM = 183.0
HEIGHT_MM = 196.0


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.2,
            "axes.labelsize": 6.4,
            "axes.titlesize": 6.5,
            "xtick.labelsize": 5.4,
            "ytick.labelsize": 5.4,
            "legend.fontsize": 5.2,
            "legend.frameon": False,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def relabel_panel(ax: plt.Axes, old: str, new: str) -> None:
    for artist in ax.texts:
        if artist.get_text() == old:
            artist.set_text(new)


def save_bundle(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), metadata={"Creator": "Python/matplotlib"})
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(stem.with_suffix(".png"), dpi=360)
    rgba = plt.imread(stem.with_suffix(".png"))
    grey = np.dot(rgba[..., :3], [0.2126, 0.7152, 0.0722])
    plt.imsave(
        stem.with_name(stem.name + "_grayscale").with_suffix(".png"),
        grey,
        cmap="gray",
        vmin=0,
        vmax=1,
    )


def main() -> None:
    configure_style()

    values, pairs, decades, prepared, summary = natural.load_data()
    threshold = float(summary["cost_improvement_2025_2035_eur_kw_year"])
    marginals, matrix = natural.prepare_source_data(values, pairs, decades, prepared, threshold)
    pair_source = pd.read_csv(natural.SOURCE / "natural_year_transition_pairs.csv")

    surface = pd.read_csv(cluster.SOURCE / "v60_cluster_intensity_surface.csv")
    intensity = pd.read_csv(cluster.SOURCE / "v60_cluster_intensity_axis.csv")
    boundaries = pd.read_csv(cluster.SOURCE / "v60_cluster_intensity_boundaries.csv")
    cluster.assert_complete(surface, intensity)
    absolute_max = float(np.nanmax(np.abs(surface["signed_parity_ratio"].to_numpy(float))))
    colour_limit = max(1.5, np.ceil(absolute_max * 2.0) / 2.0)
    cluster_norm = SymLogNorm(
        linthresh=0.20,
        linscale=0.65,
        vmin=-colour_limit,
        vmax=colour_limit,
        base=10,
    )

    fig = plt.figure(figsize=(WIDTH_MM * MM, HEIGHT_MM * MM), facecolor="white")

    # Natural chronology, transition matrix and clustering are aligned marginals of panel a.
    ax_chronology = fig.add_axes([0.17, 0.865, 0.45, 0.090])
    ax_matrix = fig.add_axes([0.17, 0.405, 0.45, 0.420])
    ax_clustering = fig.add_axes([0.660, 0.405, 0.130, 0.420], sharey=ax_matrix)
    natural_cax = fig.add_axes([0.810, 0.405, 0.012, 0.420])

    natural.draw_value_chronology(ax_chronology, marginals)
    matrix_image = natural.draw_transition_heatmap(
        ax_matrix,
        matrix,
        marginals,
        pair_source,
        decades,
        threshold,
    )
    natural.draw_recovery_structure(ax_clustering, marginals)
    for axis in (ax_chronology, ax_matrix, ax_clustering):
        axis.set_title("", loc="left")
        axis.set_title("", loc="center")
        axis.set_title("", loc="right")
    if ax_clustering.legend_ is not None:
        ax_clustering.legend_.remove()
    fig.text(0.130, 0.955, "a", fontsize=8.4, fontweight="bold", ha="left", va="top")

    natural_bar = fig.colorbar(
        matrix_image,
        cax=natural_cax,
        orientation="vertical",
        ticks=[-2, -1, 0, 1, 2],
    )
    natural_bar.outline.set_linewidth(0.55)
    natural_bar.ax.tick_params(width=0.5, length=2.1, pad=1.3, labelsize=5.2)
    natural_bar.ax.axhline(-1, color="white", lw=0.9)
    natural_bar.set_label(
        "Natural-year value transition / 2025-2035 cost scale\n(later - earlier)",
        labelpad=2.5,
        fontsize=5.4,
    )

    # The three controlled surfaces use a shared scale and appear as panels b-d.
    cluster_axes = [
        fig.add_axes([0.065, 0.062, 0.267, 0.170]),
        fig.add_axes([0.366, 0.062, 0.267, 0.170]),
        fig.add_axes([0.667, 0.062, 0.267, 0.170]),
    ]
    cluster_image = None
    for axis, regime, old_label, new_label in zip(
        cluster_axes,
        cluster.REGIMES,
        cluster.PANEL_LABELS,
        ("b", "c", "d"),
    ):
        cluster_image = cluster.draw_panel(
            axis,
            regime,
            old_label,
            surface,
            intensity,
            boundaries,
            cluster_norm,
        )
        axis.set_title("", loc="left")
        axis.set_title("", loc="center")
        axis.set_title("", loc="right")
        relabel_panel(axis, old_label, new_label)
    assert cluster_image is not None

    cluster_cax = fig.add_axes([0.075, 0.270, 0.50, 0.012])
    cluster_bar = fig.colorbar(cluster_image, cax=cluster_cax, orientation="horizontal")
    candidate_ticks = [-colour_limit, -4, -1, -0.2, 0, 0.2, 1, 4, colour_limit]
    ticks = sorted({value for value in candidate_ticks if -colour_limit <= value <= colour_limit})
    cluster_bar.set_ticks(ticks)
    cluster_bar.set_ticklabels([f"{value:g}" for value in ticks])
    cluster_bar.ax.tick_params(length=1.8, pad=1.2, labelsize=5.0)
    cluster_bar.set_label(
        "Sequencing effect / 2025-2035 cost scale (clustered - dispersed)",
        labelpad=1.2,
        fontsize=5.6,
    )

    legend_handles = [
        Line2D([0], [0], color=cluster.INK, lw=0.75, label="zero-effect boundary"),
        Line2D(
            [0],
            [0],
            color=cluster.MID_GREY,
            lw=1.0,
            ls="--",
            label="negative cost-scale boundary",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            markersize=4.0,
            markerfacecolor="#F1B83B",
            markeredgecolor=cluster.INK,
            markeredgewidth=0.35,
            label="sustained-positive duration",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="center right",
        bbox_to_anchor=(0.938, 0.276),
        frameon=False,
        ncol=2,
        handlelength=1.25,
        handletextpad=0.35,
        labelspacing=0.25,
        fontsize=5.0,
    )

    qa = {
        "figure_width_mm": WIDTH_MM,
        "figure_height_mm": HEIGHT_MM,
        "natural_operating_years": int(len(marginals)),
        "natural_transition_cells": int(np.isfinite(matrix).sum()),
        "matched_ten_year_transitions": int(len(decades)),
        "controlled_surface_rows": int(len(surface)),
        "controlled_regimes": list(cluster.REGIMES),
        "panel_titles_in_axes": 0,
        "cost_scale_eur_kw_year": threshold,
    }
    (HERE / "source_data_v61_figure6_qa.json").write_text(
        json.dumps(qa, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    save_bundle(fig, OUT / "fig_v61_figure6_temporal_parity")
    plt.close(fig)


if __name__ == "__main__":
    main()
