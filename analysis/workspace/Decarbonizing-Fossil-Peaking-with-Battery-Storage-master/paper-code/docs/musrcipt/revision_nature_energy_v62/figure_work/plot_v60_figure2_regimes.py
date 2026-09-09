#!/usr/bin/env python3
"""Figure 2: three balancing regimes and the conserved sequencing design."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

from plot_v51_main_figures import CHINA_GEO, JAPAN_GEO_ROOT, add_geojson
from plot_v57_reference_style_figures import china_map, europe_map
from v60_figure_style import (
    BLUE,
    CORAL,
    GRID,
    INK,
    MM,
    MUTED,
    PALE,
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
HEIGHT_MM = 194.0
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


def south_china_sea_inset(ax: plt.Axes) -> None:
    """Add the nine-segment South China Sea boundary carried by the source map."""
    geo = json.loads(CHINA_GEO.read_text(encoding="utf-8"))
    inset = ax.inset_axes([0.735, 0.00, 0.245, 0.43])
    inset.set_facecolor("white")
    bounds = (105.0, 124.0, 2.5, 25.5)
    for feature in geo["features"]:
        if feature["geometry"]["type"] in {"Polygon", "MultiPolygon"}:
            add_geojson(
                inset,
                feature["geometry"],
                "#F0F1EE",
                edge="#D2D7D4",
                lw=0.18,
                bounds=bounds,
            )
    south_boundary = next(
        feature
        for feature in geo["features"]
        if feature["properties"].get("name") == "境界线"
        and min(point[1] for line in feature["geometry"]["coordinates"] for point in line) < 5
    )
    # The tenth segment in this source lies east of Taiwan; the requested
    # nine-dash representation uses the first nine South China Sea segments.
    for line in south_boundary["geometry"]["coordinates"][:9]:
        points = np.asarray(line, float)
        inset.plot(points[:, 0], points[:, 1], color=INK, lw=0.62, solid_capstyle="round")
    inset.set_xlim(bounds[:2])
    inset.set_ylim(bounds[2:])
    inset.set_aspect("equal")
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_color("#B8BFBC")
        spine.set_linewidth(0.38)
    inset.text(
        0.50,
        0.03,
        "South China Sea",
        transform=inset.transAxes,
        ha="center",
        va="bottom",
        color=MUTED,
        fontsize=5.0,
    )


def draw_japan_region(
    ax: plt.Axes,
    points: pd.DataFrame,
    bounds: tuple[float, float, float, float],
    cmap: LinearSegmentedColormap,
    norm: Normalize,
    label: str,
) -> None:
    for code in (11, 12, 13, 14, 22):
        path = JAPAN_GEO_ROOT / f"N03-20240101_{code}_GML/N03-20240101_{code}.geojson"
        for feature in json.loads(path.read_text(encoding="utf-8"))["features"]:
            add_geojson(
                ax,
                feature["geometry"],
                "#F0F1EE",
                edge="#C9CECB",
                lw=0.20,
                bounds=bounds,
            )
    inside = points.loc[
        points["longitude"].between(bounds[0], bounds[1])
        & points["latitude"].between(bounds[2], bounds[3])
    ]
    ax.scatter(
        inside["longitude"],
        inside["latitude"],
        c=inside["event_run_days"],
        cmap=cmap,
        norm=norm,
        s=42,
        marker="h",
        edgecolor="white",
        lw=0.48,
        zorder=5,
    )
    ax.set_xlim(bounds[:2])
    ax.set_ylim(bounds[2:])
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(
        0.03,
        0.04,
        label,
        transform=ax.transAxes,
        color=MUTED,
        fontsize=5.0,
        fontweight="bold",
    )


def japan_community_map(
    ax: plt.Axes,
    points: pd.DataFrame,
    cmap: LinearSegmentedColormap,
    norm: Normalize,
) -> None:
    """Use prefecture-scale zooms so individual community meshes remain visible."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    positions = [
        [0.00, 0.53, 0.31, 0.42],
        [0.345, 0.53, 0.31, 0.42],
        [0.69, 0.53, 0.31, 0.42],
        [0.17, 0.05, 0.31, 0.42],
        [0.515, 0.05, 0.31, 0.42],
    ]
    for prefecture, position in zip(
        ["Shizuoka", "Saitama", "Tokyo", "Kanagawa", "Chiba"],
        positions,
    ):
        local = points.loc[points["prefecture"].eq(prefecture)].copy()
        lon_span = float(local["longitude"].max() - local["longitude"].min())
        lat_span = float(local["latitude"].max() - local["latitude"].min())
        lon_margin = max(0.022, 0.65 * lon_span)
        lat_margin = max(0.018, 0.65 * lat_span)
        bounds = (
            float(local["longitude"].min() - lon_margin),
            float(local["longitude"].max() + lon_margin),
            float(local["latitude"].min() - lat_margin),
            float(local["latitude"].max() + lat_margin),
        )
        inset = ax.inset_axes(position)
        draw_japan_region(inset, local, bounds, cmap, norm, prefecture)


def rgba_strip(rows: list[tuple[np.ndarray, str]]) -> np.ndarray:
    image = np.ones((len(rows), len(rows[0][0]), 4), float)
    for index, (values, colour) in enumerate(rows):
        values = np.clip(np.nan_to_num(values, nan=0.0), 0.0, 1.0)
        rgb = np.asarray(mpl.colors.to_rgb(colour))
        strength = 0.07 + 0.90 * values
        image[index, :, :3] = 1.0 - strength[:, None] * (1.0 - rgb[None, :])
    return image


def plot_maps(fig: plt.Figure, spec) -> None:
    eu = pd.read_csv(SOURCE / "figure2a_europe_map.csv")
    cn = pd.read_csv(SOURCE / "figure2b_china_map.csv")
    jp = pd.read_csv(SOURCE / "figure2c_japan_map.csv")
    top = spec.subgridspec(2, 3, height_ratios=[1.0, 0.070], hspace=0.08, wspace=0.12)
    axes = [fig.add_subplot(top[0, index]) for index in range(3)]
    cmap = LinearSegmentedColormap.from_list("event_duration", [BLUE, "#F1F3F0", CORAL])
    norm = Normalize(3, 23)
    europe_map(axes[0], eu.set_index("gisco_code")["median_longest_event_days"].to_dict(), cmap, norm)
    china_map(axes[1], cn.set_index("province")["longest_extreme_peaking_run_days"].to_dict(), cmap, norm)
    south_china_sea_inset(axes[1])
    community = jp.groupby(
        ["prefecture", "mesh_id", "latitude", "longitude"],
        as_index=False,
    ).agg(event_run_days=("longest_extreme_import_run_days", "median"))
    japan_community_map(axes[2], community, cmap, norm)
    titles = [
        ("Europe | national", "15 countries · 132 country-years"),
        ("China | provincial", "26 provinces · 26 province-years"),
        ("Japan | community", "19 grids · 27 community-years"),
    ]
    for index, (ax, (title, subtitle), regime) in enumerate(zip(axes, titles, ["Europe", "China", "Japan"])):
        ax.set_title(f"{title}\n{subtitle}", color=REGIME_COLORS[regime], fontweight="bold", fontsize=6.3, pad=1.5, linespacing=1.25)
        ax.set_anchor("C")
        ax.set_box_aspect(0.74)
        panel_label(ax, chr(ord("a") + index), -0.06, 1.04)
    cax = fig.add_subplot(top[1, :])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_ticks([3, 7, 11, 15, 19, 23])
    cb.set_label("longest observed event (d)", fontsize=5.5, labelpad=1.0)
    cb.ax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=5.1, length=1.6, pad=1)


def plot_task_share(ax: plt.Axes, contract: pd.DataFrame) -> None:
    panel_label(ax, "d", -0.16, 1.05)
    ax.text(0.0, 1.03, "Extreme-day concentration", transform=ax.transAxes, color=INK, fontweight="bold", va="bottom")
    order = ["Europe", "China", "Japan"]
    for y, regime in enumerate(order):
        row = contract.loc[contract["regime"].eq(regime)].iloc[0]
        share = float(row["event_task_share_pct"])
        colour = REGIME_COLORS[regime]
        ax.barh(y, share, height=0.48, color=mpl.colors.to_rgba(colour, 0.86), edgecolor="white", lw=0.35)
        ax.barh(y, 100 - share, left=share, height=0.48, color=PALE, edgecolor="#C8CECB", lw=0.35, hatch="//")
        ax.text(max(share - 2.0, 4), y, f"{share:.1f}%", color="white", ha="right", va="center", fontweight="bold", fontsize=5.5)
        task = float(row["annual_task_value"])
        unit = str(row["annual_task_unit"])
        task_label = f"{task:,.1f} {unit}" if task >= 100 else f"{task:.2f} {unit}"
        ax.text(103, y, task_label, color=colour, va="center", fontweight="bold", fontsize=5.5)
    ax.set_yticks(range(3), order)
    for tick, regime in zip(ax.get_yticklabels(), order):
        tick.set_color(REGIME_COLORS[regime])
        tick.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 132)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("annual task on q75 event days (%)")
    ax.text(103, -0.70, "observed task", color=MUTED, fontsize=5.5, ha="left")
    quiet(ax, "x")


def plot_task_event(fig: plt.Figure, spec, records: pd.DataFrame, summary: pd.DataFrame) -> None:
    sub = spec.subgridspec(1, 3, wspace=0.18)
    axes = [fig.add_subplot(sub[0, index]) for index in range(3)]
    y_max = 24
    for index, (ax, regime) in enumerate(zip(axes, ["Europe", "China", "Japan"])):
        frame = records.loc[records["regime"].eq(regime)].copy()
        colour = REGIME_COLORS[regime]
        ax.scatter(
            frame["annual_task_percentile"],
            frame["event_run_days"],
            s=12 if regime == "Europe" else 16,
            marker=REGIME_MARKERS[regime],
            color=mpl.colors.to_rgba(colour, 0.47),
            edgecolor="white",
            lw=0.25,
            rasterized=True,
        )
        median = float(frame["event_run_days"].median())
        ax.axhline(median, color=colour, lw=0.65, ls=(0, (2, 2)), alpha=0.8)
        rho = float(summary.loc[summary["regime"].eq(regime), "spearman_rho_task_vs_event"].iloc[0])
        ax.text(0.04, 0.94, regime, transform=ax.transAxes, color=colour, va="top", fontweight="bold", fontsize=6.3)
        ax.text(0.96, 0.94, rf"$\rho={rho:+.2f}$", transform=ax.transAxes, color=colour, va="top", ha="right", fontweight="bold")
        ax.text(0.96, 0.08, f"median {median:.0f} d", transform=ax.transAxes, color=MUTED, va="bottom", ha="right", fontsize=5.5)
        ax.set_xlim(0, 103)
        ax.set_ylim(0, y_max)
        ax.set_xticks([0, 50, 100])
        ax.set_xlabel("task percentile")
        if index == 0:
            ax.set_ylabel("longest event (d)")
            panel_label(ax, "e", -0.24, 1.06)
            ax.text(0.0, 1.04, "Annual task does not determine event duration", transform=ax.transAxes, color=INK, fontweight="bold", va="bottom")
        else:
            ax.set_yticklabels([])
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)
        quiet(ax, "y")


def plot_conservation(fig: plt.Figure, spec, chronology: pd.DataFrame, summary: pd.DataFrame) -> None:
    sub = spec.subgridspec(1, 12, wspace=0.35)
    axes = [fig.add_subplot(sub[0, :5]), fig.add_subplot(sub[0, 5:10])]
    for index, (ax, chronology_name) in enumerate(zip(axes, ["dispersed", "clustered"])):
        task = chronology[f"{chronology_name}_task_normalized"].to_numpy(float)
        recharge = chronology[f"{chronology_name}_recharge_normalized"].to_numpy(float)
        ax.imshow(rgba_strip([(task, CORAL), (recharge, TEAL)]), aspect="auto", interpolation="nearest", extent=[1, 365, 1.5, -0.5], rasterized=True)
        ax.set_yticks([0, 1], ["task", "recharge"] if index == 0 else ["", ""])
        ax.tick_params(axis="y", length=0, pad=3)
        if index == 0:
            for tick, colour in zip(ax.get_yticklabels(), [CORAL, TEAL]):
                tick.set_color(colour)
                tick.set_fontweight("bold")
            panel_label(ax, "f", -0.18, 1.17)
            ax.text(0.0, 1.13, "Reorder intact 24 h blocks", transform=ax.transAxes, color=INK, fontweight="bold", va="bottom")
        ax.set_title(chronology_name, color=BLUE if index == 0 else CORAL, fontweight="bold", pad=2)
        ax.set_xlim(1, 365)
        ax.set_xticks([1, 92, 183, 274, 365], ["1", "92", "183", "274", "365"])
        ax.set_xlabel("day block")
        ax.spines[["top", "right", "left"]].set_visible(False)
    stats = fig.add_subplot(sub[0, 10:])
    stats.axis("off")
    row = summary.loc[summary["regime"].eq("Europe")].iloc[0]
    stats.text(0.0, 0.83, "task + recharge", color=INK, fontweight="bold")
    stats.text(0.0, 0.66, "exactly conserved", color=TEAL, fontweight="bold")
    stats.text(0.0, 0.42, "recovery", color=MUTED)
    stats.text(0.0, 0.27, f"{row.median_recovery_dispersed_h:.0f} → {row.median_recovery_clustered_h:.0f} h", color=CORAL, fontweight="bold", fontsize=6.6)
    stats.text(0.0, 0.06, f"{row.zero_window_clustered_pct:.1f}% zero-window", color=CORAL, fontweight="bold", fontsize=5.5)


def main() -> None:
    configure_publication()
    contract = pd.read_csv(SOURCE / "figure2_task_contract.csv")
    records = pd.read_csv(SOURCE / "figure2_task_event_records.csv")
    summary = pd.read_csv(SOURCE / "figure2_regime_summary.csv")
    chronology = pd.read_csv(SOURCE / "figure2_conserved_chronology.csv")

    fig = plt.figure(figsize=(WIDTH_MM * MM, HEIGHT_MM * MM))
    grid = fig.add_gridspec(
        3,
        12,
        left=0.075,
        right=0.985,
        bottom=0.06,
        top=0.97,
        height_ratios=[1.20, 0.78, 0.55],
        hspace=0.48,
        wspace=0.80,
    )
    plot_maps(fig, grid[0, :])
    plot_task_share(fig.add_subplot(grid[1, :4]), contract)
    plot_task_event(fig, grid[1, 5:], records, summary)
    plot_conservation(fig, grid[2, :], chronology, summary)
    export_figure(fig, "fig_v60_figure2_three_balancing_regimes")
    plt.close(fig)


if __name__ == "__main__":
    main()
