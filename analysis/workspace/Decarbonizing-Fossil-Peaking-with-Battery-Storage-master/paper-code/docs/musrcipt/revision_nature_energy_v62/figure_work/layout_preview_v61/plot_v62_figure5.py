#!/usr/bin/env python3
"""Render the final v62 Figure 5 with v56-style parity-exposure matrices."""

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
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d


HERE = Path(__file__).resolve().parent
ORIGINAL_SCRIPT = HERE / "plot_preview_figure5.py"
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


layout = load_module("v62_figure5_layout_source", ORIGINAL_SCRIPT)
base = layout.base
OUT = HERE.parent / "main_figures"
OUT.mkdir(parents=True, exist_ok=True)


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


COMMON_DURATIONS = [
    4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 28,
    32, 36, 40, 44, 48, 60, 72, 96, 100, 120, 144, 168,
]

PHASE_CMAP = LinearSegmentedColormap.from_list(
    "sequencing_parity_nature_muted",
    ["#8A4F49", "#B8756B", "#D9ADA3", "#F5F3EE", "#BCD6D0", "#79AAA1", "#356F6B"],
    N=256,
)
REGIME_LABELS = {"Europe": "NVSB", "China": "PLRP", "Japan": "CLDB"}


def remove_nested_panel_label(ax: plt.Axes, label: str) -> None:
    for artist in list(ax.texts):
        if artist.get_text() == label:
            artist.remove()


def draw_phase_map(fig: plt.Figure, spec) -> None:
    surface = pd.read_csv(layout.phase.SOURCE / "v60_cluster_intensity_surface.csv")
    intensity = pd.read_csv(layout.phase.SOURCE / "v60_cluster_intensity_axis.csv")
    boundaries = pd.read_csv(layout.phase.SOURCE / "v60_cluster_intensity_boundaries.csv")
    layout.phase.assert_complete(surface, intensity)
    layout.phase.CMAP = PHASE_CMAP

    absolute_max = float(np.nanmax(np.abs(surface["signed_parity_ratio"].to_numpy(float))))
    colour_limit = max(1.5, np.ceil(absolute_max * 2.0) / 2.0)
    norm = layout.phase.SymLogNorm(
        linthresh=0.20,
        linscale=0.65,
        vmin=-colour_limit,
        vmax=colour_limit,
        base=10,
    )
    sub = spec.subgridspec(
        5,
        12,
        height_ratios=[1.0, 0.220, 0.045, 0.060, 0.060],
        hspace=0.03,
        wspace=0.34,
    )
    axes = [
        fig.add_subplot(sub[0, 0:4]),
        fig.add_subplot(sub[0, 4:8]),
        fig.add_subplot(sub[0, 8:12]),
    ]
    right_axes: list[plt.Axes] = []
    image = None
    for ax, regime, nested_label in zip(
        axes,
        layout.phase.REGIMES,
        layout.phase.PANEL_LABELS,
    ):
        image = layout.phase.draw_panel(
            ax,
            regime,
            nested_label,
            surface,
            intensity,
            boundaries,
            norm,
        )
        right_axes.append(fig.axes[-1])
        remove_nested_panel_label(ax, nested_label)
        ax.set_xlabel("")
        ax.set_title(REGIME_LABELS[regime], loc="left", fontsize=6.4, fontweight="bold", pad=2.5)
    for right_ax in right_axes[:2]:
        right_ax.set_yticks([])
        right_ax.set_ylabel("")
    right_axes[-1].set_ylabel("")
    right_axes[-1].tick_params(axis="y", labelsize=5.2, pad=1.0)
    if image is None:
        raise RuntimeError("The duration-cluster surface did not return a colour-mapped image.")
    base.panel_label(axes[0], "b", x=-0.22, y=1.07)

    xlabel_ax = fig.add_subplot(sub[1, :])
    xlabel_ax.axis("off")
    xlabel_ax.text(
        0.5,
        0.52,
        "storage duration (h)",
        transform=xlabel_ax.transAxes,
        ha="center",
        va="center",
        fontsize=5.9,
    )

    cax = fig.add_subplot(sub[2, :])
    colorbar = fig.colorbar(image, cax=cax, orientation="horizontal")
    candidate_ticks = [-colour_limit, -4, -1, -0.2, 0, 0.2, 1, 4, colour_limit]
    ticks = sorted(
        {value for value in candidate_ticks if -colour_limit <= value <= colour_limit}
    )
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([f"{value:g}" for value in ticks])
    colorbar.ax.tick_params(length=1.7, pad=0.9, labelsize=5.6)
    colorbar.set_label(
        "sequencing effect / 2025-2035 cost scale  (clustered - dispersed)",
        labelpad=0.7,
        fontsize=6.0,
    )
    colorbar.ax.xaxis.set_label_position("top")

    legend_ax = fig.add_subplot(sub[4, :])
    legend_ax.axis("off")
    legend_ax.legend(
        handles=[
            Line2D([0], [0], color=layout.phase.INK, lw=0.75, label="zero-effect boundary"),
            Line2D(
                [0],
                [0],
                color=layout.phase.MID_GREY,
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
                markeredgecolor=layout.phase.INK,
                markeredgewidth=0.35,
                label="sustained-positive duration",
            ),
        ],
        loc="center",
        ncol=3,
        frameon=False,
        fontsize=5.7,
        handlelength=1.5,
        columnspacing=3.0,
        borderaxespad=0,
    )


def short_system_label(regime: str, system_id: str) -> str:
    label = base.clean_system_label(regime, system_id)
    if regime == "China":
        return label.replace("Inner Mongolia", "Inner Mong.")
    if regime == "Japan" and "-" in label:
        prefecture, mesh = label.rsplit("-", 1)
        return f"{prefecture[:4]} {mesh[-3:]}"
    return label


def plot_exposure_matrix(
    fig: plt.Figure,
    spec,
    frame: pd.DataFrame,
    regime: str,
    label: str,
    cmap: LinearSegmentedColormap,
) -> tuple[mpl.collections.QuadMesh, plt.Axes]:
    frame = frame.loc[frame["duration_h"].isin(COMMON_DURATIONS)].copy()
    weighted = frame.assign(
        weighted_exposure=(
            frame["parity_exposure_frequency"] * frame["n_observation_years"]
        )
    )
    ranking = weighted.groupby("system_id").agg(
        weighted_sum=("weighted_exposure", "sum"),
        observation_sum=("n_observation_years", "sum"),
    )
    ranking["score"] = ranking["weighted_sum"] / ranking["observation_sum"]
    system_order = ranking["score"].sort_values(ascending=False, kind="mergesort").index
    matrix = frame.pivot(
        index="system_id",
        columns="duration_h",
        values="parity_exposure_frequency",
    ).reindex(index=system_order, columns=COMMON_DURATIONS)
    if matrix.isna().any().any():
        raise ValueError(f"Incomplete v56-style exposure matrix for {regime}.")
    values = matrix.to_numpy(float)
    if np.any((values < 0) | (values > 1)):
        raise ValueError("Exposure frequencies must lie within [0, 1].")

    sub = spec.subgridspec(2, 1, height_ratios=[0.48, 3.0], hspace=0.08)
    profile_ax = fig.add_subplot(sub[0, 0])
    heat_ax = fig.add_subplot(sub[1, 0])

    totals = (
        frame.groupby("duration_h")[["parity_years", "n_observation_years"]]
        .sum()
        .reindex(COMMON_DURATIONS)
    )
    profile = 100 * totals["parity_years"].to_numpy(float) / totals["n_observation_years"].to_numpy(float)
    x = np.arange(len(COMMON_DURATIONS))
    smooth_x = np.linspace(x.min(), x.max(), max(320, len(x) * 16))
    trend_nodes = gaussian_filter1d(profile, sigma=0.75, mode="nearest")
    smooth_profile = np.clip(PchipInterpolator(x, trend_nodes)(smooth_x), 0, 100)
    profile_ax.fill_between(
        smooth_x,
        0,
        smooth_profile,
        color=mpl.colors.to_rgba("#D87864", 0.23),
        lw=0,
    )
    profile_ax.plot(smooth_x, smooth_profile, color="#D87864", lw=0.9)
    profile_ax.axhline(0, color="#AEB6B3", lw=0.5, zorder=0)
    profile_ax.axhline(50, color="#D7DCDA", lw=0.5)
    profile_ax.set_xlim(-0.5, len(x) - 0.5)
    profile_ax.set_ylim(-10, 105)
    profile_ax.set_xticks([])
    if regime == "Europe":
        profile_ax.set_yticks([0, 50, 100])
        profile_ax.set_ylabel("systems (%)", fontsize=5.0, labelpad=1.5)
    else:
        profile_ax.set_yticks([])

    hit_row = frame.loc[frame["duration_h"].eq(24)]
    hits = int(hit_row["parity_years"].sum())
    total = int(hit_row["n_observation_years"].sum())
    profile_ax.set_title(
        f"{REGIME_LABELS[regime]}  |  {hits}/{total} at 24 h",
        color=base.INK,
        fontweight="bold",
        fontsize=5.25,
        pad=1.8,
    )
    base.quiet(profile_ax)
    base.panel_label(profile_ax, label, x=-0.27, y=1.20)

    x_edges = np.arange(values.shape[1] + 1) - 0.5
    y_edges = np.arange(values.shape[0] + 1) - 0.5
    image = heat_ax.pcolormesh(
        x_edges,
        y_edges,
        values,
        cmap=cmap,
        vmin=0,
        vmax=1,
        shading="flat",
        edgecolors="white",
        linewidth=0.08,
        antialiased=False,
        rasterized=False,
        snap=True,
    )
    labels = [short_system_label(regime, system_id) for system_id in system_order]
    heat_ax.set_xlim(-0.5, values.shape[1] - 0.5)
    heat_ax.set_ylim(values.shape[0] - 0.5, -0.5)
    heat_ax.set_yticks(np.arange(len(labels)), labels)
    heat_ax.set_xticks(np.arange(len(COMMON_DURATIONS)))
    shown = {4, 8, 12, 16, 24, 48, 96, 168}
    heat_ax.set_xticklabels(
        [str(value) if value in shown else "" for value in COMMON_DURATIONS]
    )
    heat_ax.tick_params(axis="both", length=0, pad=1.0, labelsize=5.0)
    heat_ax.set_xlabel("")
    for spine in heat_ax.spines.values():
        spine.set_visible(False)
    return image, heat_ax


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
        "parity_exposure_v56",
        ["#F3F1EC", "#B9D2D8", "#D1A13A", "#D87864"],
    )

    fig = plt.figure(figsize=(WIDTH_MM * MM, HEIGHT_MM * MM))
    grid = fig.add_gridspec(
        3,
        12,
        left=0.080,
        right=0.975,
        bottom=0.074,
        top=0.962,
        height_ratios=[0.80, 1.10, 1.28],
        hspace=0.15,
        wspace=0.60,
    )

    for index, regime in enumerate(["Europe", "China", "Japan"]):
        response_ax = fig.add_subplot(grid[0, index * 4 : (index + 1) * 4])
        base.plot_response(
            response_ax,
            duration.loc[duration["regime"].eq(regime)],
            regime,
            "a" if index == 0 else "",
        )
        response_ax.set_xticks(
            [4, 8, 24, 48, 96, 168, 720],
            ["4", "8", "24", "48", "96", "168", "720"],
        )

    draw_phase_map(fig, grid[1, :])

    mappables: list[mpl.collections.QuadMesh] = []
    heat_axes: list[plt.Axes] = []
    for index, regime in enumerate(["Europe", "China", "Japan"]):
        mappable, heat_ax = plot_exposure_matrix(
            fig,
            grid[2, index * 3 : (index + 1) * 3],
            exposure.loc[exposure["regime"].eq(regime)],
            regime,
            "c" if index == 0 else "",
            exposure_cmap,
        )
        mappables.append(mappable)
        heat_axes.append(heat_ax)

    parity_ax = fig.add_subplot(grid[2, 9:12])
    base.plot_parity(parity_ax, records)
    remove_nested_panel_label(parity_ax, "h")
    base.panel_label(parity_ax, "d", x=-0.18, y=1.04)
    parity_ax.tick_params(axis="y", labelsize=5.0, pad=1.0)

    fig.canvas.draw()
    heat_boxes = [axis.get_position() for axis in heat_axes]
    heat_left = heat_boxes[0].x0
    heat_right = heat_boxes[-1].x1
    heat_bottom = min(box.y0 for box in heat_boxes)
    span = heat_right - heat_left
    fig.text(
        0.5 * (heat_left + heat_right),
        heat_bottom - 0.017,
        "storage duration (h)",
        ha="center",
        va="top",
        fontsize=5.9,
    )
    cax = fig.add_axes([heat_left, heat_bottom - 0.060, span, 0.007])
    exposure_bar = fig.colorbar(mappables[0], cax=cax, orientation="horizontal")
    exposure_bar.set_ticks([0, 0.5, 1.0])
    exposure_bar.set_ticklabels(["0", "50", "100"])
    exposure_bar.set_label(
        "system-year share reaching matched cost scale (%)",
        labelpad=1.3,
        fontsize=5.9,
    )
    exposure_bar.ax.xaxis.set_label_position("top")
    exposure_bar.ax.tick_params(labelsize=5.7, length=1.5, pad=1.1)

    save_bundle(fig, "fig_v62_figure5_duration_parity_atlas")
    plt.close(fig)


if __name__ == "__main__":
    main()
