#!/usr/bin/env python3
"""Plot the V60 duration-by-clustering-intensity phase map."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source_data"
FIGURES = HERE / "figures"
PARITY_THRESHOLD = 5.2960380996609935
REGIMES = ("Europe", "China", "Japan")
PANEL_LABELS = ("a", "b", "c")
SYSTEM_YEAR_N = {"Europe": 132, "China": 26, "Japan": 27}
BLOCK_N = {"Europe": 15, "China": 26, "Japan": 5}
SAMPLE_LABEL = {
    "Europe": "n = 132 country-years",
    "China": "n = 26 province-years",
    "Japan": "n = 27 community-years",
}

MM = 1.0 / 25.4
WIDTH_MM = 183.0
HEIGHT_MM = 94.0

CORAL = "#CB6759"
TEAL = "#238B8D"
INK = "#262626"
MID_GREY = "#777777"
GRID_GREY = "#D5D5D5"

CMAP = LinearSegmentedColormap.from_list(
    "sequencing_parity",
    ["#7F1D1D", "#D96B5F", "#F2B6A8", "#F7F7F5", "#A9D9D2", "#3E9B98", "#145B63"],
    N=256,
)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6.0,
            "axes.titlesize": 7.0,
            "axes.labelsize": 6.5,
            "xtick.labelsize": 5.5,
            "ytick.labelsize": 5.5,
            "legend.fontsize": 5.5,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.3,
            "ytick.major.size": 2.3,
            "lines.linewidth": 0.8,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def centres_to_edges(values: np.ndarray, lower: float | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    middles = (values[:-1] + values[1:]) / 2.0
    edges = np.concatenate(
        ([values[0] - (middles[0] - values[0])], middles, [values[-1] + (values[-1] - middles[-1])])
    )
    if lower is not None:
        edges[0] = lower
    return edges


def assert_complete(surface: pd.DataFrame, intensity: pd.DataFrame) -> None:
    expected_clusters = list(range(1, 13))
    durations = sorted(surface["duration_h"].unique().tolist())
    if (surface["duration_h"] <= 0).any():
        raise RuntimeError("Storage duration must be positive for the logarithmic axis")
    if len(durations) != 14:
        raise RuntimeError(f"Expected 14 duration nodes, found {len(durations)}")
    for regime in REGIMES:
        frame = surface.loc[surface["regime"].eq(regime)]
        pivot = frame.pivot(index="target_cluster_days", columns="duration_h", values="signed_parity_ratio")
        if pivot.index.tolist() != expected_clusters or pivot.shape != (12, 14):
            raise RuntimeError(f"Incomplete surface for {regime}: {pivot.shape}")
        if not np.isfinite(pivot.to_numpy(float)).all():
            raise RuntimeError(f"Non-finite surface value for {regime}")
        axis = intensity.loc[intensity["regime"].eq(regime)].sort_values("target_cluster_days")
        if axis["target_cluster_days"].astype(int).tolist() != expected_clusters:
            raise RuntimeError(f"Incomplete intensity axis for {regime}")


def draw_panel(
    ax: plt.Axes,
    regime: str,
    panel_label: str,
    surface: pd.DataFrame,
    intensity: pd.DataFrame,
    boundaries: pd.DataFrame,
    norm: SymLogNorm,
) -> mpl.collections.QuadMesh:
    frame = surface.loc[surface["regime"].eq(regime)].copy()
    pivot = frame.pivot(
        index="target_cluster_days", columns="duration_h", values="signed_parity_ratio"
    ).sort_index().sort_index(axis=1)
    x = pivot.columns.to_numpy(float)
    x_plot = np.arange(len(x), dtype=float)
    position_by_duration = {float(duration): index for index, duration in enumerate(x)}
    y = pivot.index.to_numpy(float)
    z = pivot.to_numpy(float)
    x_edges = centres_to_edges(x_plot)
    y_edges = centres_to_edges(y, lower=0.5)
    image = ax.pcolormesh(
        x_edges,
        y_edges,
        z,
        cmap=CMAP,
        norm=norm,
        shading="flat",
        edgecolors=(1, 1, 1, 0.30),
        linewidth=0.18,
        rasterized=False,
    )

    # Contours connect evaluated nodes only; the raster itself is never smoothed.
    if np.nanmin(z) <= -1.0 <= np.nanmax(z):
        ax.contour(
            x_plot,
            y,
            z,
            levels=[-1.0],
            colors="white",
            linewidths=1.0,
            linestyles="--",
        )
    if np.nanmin(z) <= 0.0 <= np.nanmax(z):
        ax.contour(x_plot, y[1:], z[1:, :], levels=[0.0], colors=INK, linewidths=0.75)

    boundary = boundaries.loc[boundaries["regime"].eq(regime)].sort_values(
        "target_cluster_days"
    )
    finite = boundary["first_sustained_positive_duration_h"].notna()
    ax.scatter(
        [
            position_by_duration[float(duration)]
            for duration in boundary.loc[finite, "first_sustained_positive_duration_h"]
        ],
        boundary.loc[finite, "target_cluster_days"],
        s=12,
        marker="D",
        facecolor="#F1B83B",
        edgecolor=INK,
        linewidth=0.35,
        zorder=4,
    )

    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(0.5, 12.5)
    tick_durations = [24, 36, 48, 72, 96, 120, 168]
    ax.set_xticks([position_by_duration[float(duration)] for duration in tick_durations])
    ax.set_xticklabels(["24", "36", "48", "72", "96", "120", "168"])
    ax.set_yticks([1, 2, 4, 6, 8, 10, 12])
    ax.set_xlabel("Storage duration (h)")
    if panel_label == "a":
        ax.set_ylabel("Target event-cluster length (d)")
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.set_title(
        f"{regime}  |  {SAMPLE_LABEL[regime]}",
        loc="left",
        color=INK,
        pad=4.0,
        fontweight="bold",
    )
    panel_text = ax.text(
        0.018,
        0.982,
        panel_label,
        transform=ax.transAxes,
        fontsize=8.0,
        fontweight="bold",
        va="top",
        ha="left",
        color="white",
        clip_on=True,
        zorder=6,
    )
    panel_text.set_path_effects(
        [path_effects.withStroke(linewidth=0.9, foreground=INK)]
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK)

    axis = intensity.loc[intensity["regime"].eq(regime)].set_index("target_cluster_days")
    selected = [1, 4, 8, 12]
    labels = [f"{100 * axis.loc[value, 'median_zero_window_share']:.0f}%" for value in selected]
    right = ax.twinx()
    right.set_ylim(ax.get_ylim())
    right.set_yticks(selected)
    right.set_yticklabels(labels, color=MID_GREY)
    right.tick_params(axis="y", width=0, length=0, pad=1.5, labelsize=5.0)
    right.spines[:].set_visible(False)
    if panel_label == "c":
        right.set_ylabel("Median zero-recovery windows", color=MID_GREY, labelpad=4.0)
    return image


def save_bundle(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), metadata={"Creator": "Python/matplotlib"})
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(stem.with_suffix(".png"), dpi=360)

    colorbar_axes = [axis for axis in fig.axes if axis.get_label() == "<colorbar>"]
    colorbar_ticks = [
        (axis, axis.get_xticks(), [label.get_text() for label in axis.get_xticklabels()])
        for axis in colorbar_axes
    ]
    for collection in fig.findobj(mpl.collections.QuadMesh):
        collection.set_cmap("Greys")
    for axis, ticks, labels in colorbar_ticks:
        axis.set_xticks(ticks)
        axis.set_xticklabels(labels)
    fig.savefig(
        stem.with_name(stem.name + "_grayscale").with_suffix(".png"),
        dpi=360,
    )


def main() -> None:
    configure_style()
    surface = pd.read_csv(SOURCE / "v60_cluster_intensity_surface.csv")
    intensity = pd.read_csv(SOURCE / "v60_cluster_intensity_axis.csv")
    boundaries = pd.read_csv(SOURCE / "v60_cluster_intensity_boundaries.csv")
    assert_complete(surface, intensity)

    absolute_max = float(np.nanmax(np.abs(surface["signed_parity_ratio"].to_numpy(float))))
    colour_limit = max(1.5, np.ceil(absolute_max * 2.0) / 2.0)
    norm = SymLogNorm(linthresh=0.20, linscale=0.65, vmin=-colour_limit, vmax=colour_limit, base=10)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(WIDTH_MM * MM, HEIGHT_MM * MM),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.15, "left": 0.075, "right": 0.925, "top": 0.90, "bottom": 0.30},
    )
    image = None
    for ax, regime, panel_label in zip(axes, REGIMES, PANEL_LABELS):
        image = draw_panel(ax, regime, panel_label, surface, intensity, boundaries, norm)
    assert image is not None

    cax = fig.add_axes([0.08, 0.092, 0.52, 0.036])
    colorbar = fig.colorbar(image, cax=cax, orientation="horizontal")
    candidate_ticks = [-colour_limit, -4, -1, -0.2, 0, 0.2, 1, 4, colour_limit]
    ticks = sorted({value for value in candidate_ticks if -colour_limit <= value <= colour_limit})
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([f"{value:g}" for value in ticks])
    colorbar.ax.tick_params(length=2.0, pad=1.5)
    colorbar.set_label(
        "system-cost sequencing effect / 2025-2035 cost scale  (clustered - dispersed)",
        labelpad=2.5,
    )

    legend_handles = [
        Line2D([0], [0], color=INK, linewidth=0.75, label="Zero-effect boundary"),
        Line2D([0], [0], color=MID_GREY, linewidth=1.0, linestyle="--", label="Negative cost-scale boundary"),
        Line2D([0], [0], color="white", marker="D", markerfacecolor="#F1B83B", markeredgecolor=INK, markeredgewidth=0.35, markersize=4.2, label="Sustained-positive duration"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower right",
        bbox_to_anchor=(0.935, 0.075),
        frameon=False,
        ncol=2,
        handlelength=1.2,
        columnspacing=0.9,
        handletextpad=0.4,
    )
    save_bundle(fig, FIGURES / "fig_v60_duration_cluster_intensity_surface")
    plt.close(fig)


if __name__ == "__main__":
    main()
