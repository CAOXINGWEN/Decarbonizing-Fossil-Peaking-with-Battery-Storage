#!/usr/bin/env python3
"""Preview Figure 5 with the three-regime duration-cluster phase map."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
REVISION = HERE.parents[1]
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

MM = 1.0 / 25.4
WIDTH_MM = 183.0
HEIGHT_MM = 222.0


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load plotting module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module(
    "v61_preview_figure5_base",
    REVISION / "figure_work" / "plot_v58_figure5_duration.py",
)
phase = load_module(
    "v61_preview_cluster_surface",
    REVISION / "cluster_intensity_surface" / "plot_v60_cluster_intensity_surface.py",
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


def remove_nested_panel_label(ax: plt.Axes, label: str) -> None:
    for artist in list(ax.texts):
        if artist.get_text() == label:
            artist.remove()


def draw_phase_map(fig: plt.Figure, spec) -> None:
    surface = pd.read_csv(phase.SOURCE / "v60_cluster_intensity_surface.csv")
    intensity = pd.read_csv(phase.SOURCE / "v60_cluster_intensity_axis.csv")
    boundaries = pd.read_csv(phase.SOURCE / "v60_cluster_intensity_boundaries.csv")
    phase.assert_complete(surface, intensity)

    absolute_max = float(np.nanmax(np.abs(surface["signed_parity_ratio"].to_numpy(float))))
    colour_limit = max(1.5, np.ceil(absolute_max * 2.0) / 2.0)
    norm = phase.SymLogNorm(
        linthresh=0.20,
        linscale=0.65,
        vmin=-colour_limit,
        vmax=colour_limit,
        base=10,
    )

    sub = spec.subgridspec(
        2,
        12,
        height_ratios=[1.0, 0.14],
        hspace=0.24,
        wspace=0.34,
    )
    axes = [
        fig.add_subplot(sub[0, 0:4]),
        fig.add_subplot(sub[0, 4:8]),
        fig.add_subplot(sub[0, 8:12]),
    ]
    image = None
    for ax, regime, nested_label in zip(axes, phase.REGIMES, phase.PANEL_LABELS):
        image = phase.draw_panel(
            ax,
            regime,
            nested_label,
            surface,
            intensity,
            boundaries,
            norm,
        )
        remove_nested_panel_label(ax, nested_label)
        ax.set_title(regime, loc="left", fontsize=6.4, fontweight="bold", pad=2.5)
    axes[-1].get_shared_y_axes()
    for candidate in fig.axes:
        if candidate is not axes[-1] and candidate.get_position().x0 >= axes[-1].get_position().x0:
            if candidate.get_ylabel() == "Median zero-recovery windows":
                candidate.set_ylabel("")
    assert image is not None
    base.panel_label(axes[0], "g", x=-0.28, y=1.07)

    cax = fig.add_subplot(sub[1, 0:7])
    colorbar = fig.colorbar(image, cax=cax, orientation="horizontal")
    candidate_ticks = [-colour_limit, -4, -1, -0.2, 0, 0.2, 1, 4, colour_limit]
    ticks = sorted(
        {value for value in candidate_ticks if -colour_limit <= value <= colour_limit}
    )
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([f"{value:g}" for value in ticks])
    colorbar.ax.tick_params(length=1.8, pad=1.2, labelsize=5.0)
    colorbar.set_label(
        "sequencing effect / 2025-2035 cost scale  (clustered - dispersed)",
        labelpad=1.6,
        fontsize=5.3,
    )

    legend_ax = fig.add_subplot(sub[1, 7:12])
    legend_ax.axis("off")
    legend_ax.legend(
        handles=[
            Line2D([0], [0], color=phase.INK, lw=0.75, label="zero-effect boundary"),
            Line2D(
                [0],
                [0],
                color=phase.MID_GREY,
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
                markeredgecolor=phase.INK,
                markeredgewidth=0.35,
                label="sustained-positive duration",
            ),
        ],
        loc="center left",
        frameon=False,
        fontsize=5.0,
        handlelength=1.3,
        labelspacing=0.35,
    )


def main() -> None:
    base.configure()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 6.2,
        }
    )

    duration = pd.read_csv(base.SOURCE / "figure5_duration_response_summary.csv")
    exposure = pd.read_csv(base.SOURCE / "figure5_system_duration_exposure.csv")
    records = pd.read_csv(base.SOURCE / "figure3_24h_value_records.csv")
    exposure_cmap = LinearSegmentedColormap.from_list(
        "parity_exposure",
        ["#F2F4F1", "#E5C778", "#C86153"],
    )

    fig = plt.figure(figsize=(WIDTH_MM * MM, HEIGHT_MM * MM))
    grid = fig.add_gridspec(
        3,
        12,
        left=0.080,
        right=0.988,
        bottom=0.070,
        top=0.962,
        height_ratios=[0.80, 1.04, 1.32],
        hspace=0.28,
        wspace=0.48,
    )

    for index, regime in enumerate(["Europe", "China", "Japan"]):
        base.plot_response(
            fig.add_subplot(grid[0, index * 4 : (index + 1) * 4]),
            duration.loc[duration["regime"].eq(regime)],
            regime,
            chr(ord("a") + index),
        )

    draw_phase_map(fig, grid[1, :9])
    parity_ax = fig.add_subplot(grid[1, 9:12])
    base.plot_parity(parity_ax, records)
    parity_ax.tick_params(axis="y", labelsize=5.2, pad=1.0)

    images = []
    for index, regime in enumerate(["Europe", "China", "Japan"]):
        images.append(
            base.plot_exposure_panel(
                fig,
                grid[2, index * 4 : (index + 1) * 4],
                exposure.loc[exposure["regime"].eq(regime)],
                regime,
                chr(ord("d") + index),
                exposure_cmap,
            )
        )

    fig.canvas.draw()
    heat_boxes = [image.axes.get_position() for image in images]
    heat_left = heat_boxes[0].x0
    heat_right = heat_boxes[-1].x1
    heat_bottom = min(box.y0 for box in heat_boxes)
    cbar_width = 0.42 * (heat_right - heat_left)
    cax = fig.add_axes(
        [
            heat_left + 0.30 * (heat_right - heat_left),
            heat_bottom - 0.043,
            cbar_width,
            0.007,
        ]
    )
    exposure_bar = fig.colorbar(images[0], cax=cax, orientation="horizontal")
    exposure_bar.set_ticks([0, 0.5, 1.0])
    exposure_bar.set_ticklabels(["0", "50", "100"])
    exposure_bar.set_label(
        "system-year share reaching matched cost scale (%)",
        labelpad=1.0,
        fontsize=5.2,
    )
    exposure_bar.ax.xaxis.set_label_position("top")
    exposure_bar.ax.tick_params(labelsize=5.0, length=1.4, pad=1)

    save_bundle(fig, "fig_v61_preview_figure5_relayout")
    plt.close(fig)


if __name__ == "__main__":
    main()
