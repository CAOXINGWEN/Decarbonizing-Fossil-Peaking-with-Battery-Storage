"""
Nature-style Figure 2 for M1 event clustering.

This script converts the exploratory M1 plots into one publication-oriented
multi-panel figure:
  a. Provincial map of extreme-state persistence P22
  b. Run-length survival curves by clustering tercile
  c. Seasonal P22 heatmap
  d. Geometric vs discrete-Weibull AIC comparison for run-length distributions
  e. Distribution-shape correlations with P22

Primary output is SVG with editable text. PDF and PNG are secondary exports.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Nature figure defaults: keep SVG text editable.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.size"] = 8.0
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.9
plt.rcParams["legend.frameon"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


PAPER_CODE_DIR = Path(__file__).resolve().parent
MASTER_DIR = PAPER_CODE_DIR.parent
PROJECT_DIR = MASTER_DIR.parent
OUTPUT_DIR = PAPER_CODE_DIR / "output" / "m1_clustering"
FIGURE_DIR = OUTPUT_DIR / "nature_figure"
SOURCE_DIR = FIGURE_DIR / "source_data"
GEOJSON_PATH = PROJECT_DIR / "dataset" / "中国_省.geojson"


PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "blue_soft": "#AFC7E8",
    "teal": "#42949E",
    "red_strong": "#B64342",
    "red_soft": "#E9A6A1",
    "neutral_light": "#EDEDED",
    "neutral_mid": "#8F8F8F",
    "neutral_dark": "#4D4D4D",
    "neutral_black": "#272727",
}


P22_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "p22_blue",
    ["#F2F6FA", "#C9D8ED", "#6F9ED2", PALETTE["blue_main"]],
)


def apply_axis_style(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", labelsize=6.5, width=0.7, length=2.7)
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(0.9)


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=PALETTE["neutral_black"],
    )


def load_m1_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    annual = pd.read_csv(OUTPUT_DIR / "province_clustering_metrics.csv")
    seasonal = pd.read_csv(OUTPUT_DIR / "province_season_clustering.csv")
    runs = pd.read_csv(OUTPUT_DIR / "province_extreme_event_runs.csv")
    return annual, seasonal, runs


def iter_exterior_rings(geometry: dict):
    """Yield exterior rings from Polygon or MultiPolygon GeoJSON geometries."""
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates", [])
    if geom_type == "Polygon":
        if coords:
            yield coords[0]
    elif geom_type == "MultiPolygon":
        for polygon in coords:
            if polygon:
                yield polygon[0]


def collect_geometry_points(geometry: dict) -> np.ndarray:
    points = []
    for ring in iter_exterior_rings(geometry):
        points.extend(ring)
    return np.asarray(points, dtype=float) if points else np.empty((0, 2))


def plot_china_p22_map(ax: plt.Axes, annual: pd.DataFrame) -> None:
    with open(GEOJSON_PATH, encoding="utf-8") as f:
        geojson = json.load(f)

    p22_by_cn = dict(zip(annual["province_cn"], annual["p22"]))
    en_by_cn = dict(zip(annual["province_cn"], annual["province_en"]))
    norm = mcolors.Normalize(vmin=0.10, vmax=0.56)

    patches = []
    facecolors = []
    centroids = {}

    for feature in geojson["features"]:
        geometry = feature.get("geometry", {})
        if geometry.get("type") not in {"Polygon", "MultiPolygon"}:
            continue

        name_cn = feature.get("properties", {}).get("name")
        value = p22_by_cn.get(name_cn)
        fill = P22_CMAP(norm(value)) if value is not None else PALETTE["neutral_light"]

        points = collect_geometry_points(geometry)
        if points.size:
            centroids[name_cn] = points.mean(axis=0)

        for ring in iter_exterior_rings(geometry):
            arr = np.asarray(ring, dtype=float)
            if len(arr) < 3:
                continue
            patches.append(mpatches.Polygon(arr, closed=True))
            facecolors.append(fill)

    collection = PatchCollection(
        patches,
        facecolor=facecolors,
        edgecolor="white",
        linewidth=0.32,
        zorder=2,
    )
    ax.add_collection(collection)
    ax.set_xlim(73, 136)
    ax.set_ylim(17, 55)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=P22_CMAP)
    cbar = plt.colorbar(sm, ax=ax, fraction=0.036, pad=0.015)
    cbar.set_label("Annual $P_{22}$", fontsize=6.5, labelpad=3)
    cbar.ax.tick_params(labelsize=6, width=0.5, length=2)
    cbar.outline.set_linewidth(0.4)

    # Sparse callouts mark the strongest and weakest clustering examples.
    callouts = {
        "北京市": ("Beijing", (119.5, 43.5)),
        "内蒙古自治区": ("Inner Mongolia", (107.0, 49.2)),
        "新疆维吾尔自治区": ("Xinjiang", (84.5, 33.0)),
    }
    for name_cn, (label, xytext) in callouts.items():
        if name_cn not in centroids or name_cn not in p22_by_cn:
            continue
        xy = centroids[name_cn]
        ax.annotate(
            label,
            xy=xy,
            xytext=xytext,
            fontsize=5.8,
            ha="center",
            va="center",
            arrowprops={
                "arrowstyle": "-",
                "lw": 0.5,
                "color": PALETTE["neutral_dark"],
                "shrinkA": 1,
                "shrinkB": 1,
            },
            color=PALETTE["neutral_black"],
            zorder=5,
        )

    ax.set_title("Spatial heterogeneity in $P_{22}$", fontsize=7.4, pad=3)

    map_source = annual[
        [
            "province_en",
            "province_cn",
            "p22",
            "mean_run_length",
            "max_run_length",
            "frac_runs_ge3",
            "best_fit_model",
        ]
    ].copy()
    map_source.to_csv(SOURCE_DIR / "fig2_panel_a_map_p22.csv", index=False)


def assign_clustering_terciles(annual: pd.DataFrame) -> pd.DataFrame:
    ordered = annual.sort_values("p22", ascending=False).reset_index(drop=True).copy()
    n = len(ordered)
    labels = []
    for idx in range(n):
        if idx < math.ceil(n / 3):
            labels.append("High")
        elif idx < math.ceil(2 * n / 3):
            labels.append("Intermediate")
        else:
            labels.append("Low")
    ordered["clustering_group"] = labels
    return ordered


def make_survival_source(annual: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    grouped_annual = assign_clustering_terciles(annual)
    group_by_province = dict(zip(grouped_annual["province_en"], grouped_annual["clustering_group"]))

    annual_runs = runs[runs["season"] == "annual"].copy()
    max_len = int(max(annual_runs["run_length"].max(), 1))
    thresholds = np.arange(1, max_len + 1)

    province_rows = []
    for province_en, province_runs in annual_runs.groupby("province_en"):
        lengths = province_runs["run_length"].to_numpy()
        for threshold in thresholds:
            province_rows.append(
                {
                    "province_en": province_en,
                    "clustering_group": group_by_province[province_en],
                    "duration_threshold_days": threshold,
                    "survival_fraction": float(np.mean(lengths >= threshold)),
                }
            )

    province_survival = pd.DataFrame(province_rows)
    summary = (
        province_survival.groupby(["clustering_group", "duration_threshold_days"])
        .agg(
            mean_survival=("survival_fraction", "mean"),
            sd_survival=("survival_fraction", "std"),
            n_provinces=("province_en", "nunique"),
        )
        .reset_index()
    )
    summary["sem_survival"] = summary["sd_survival"] / np.sqrt(summary["n_provinces"])
    summary["sem_survival"] = summary["sem_survival"].fillna(0)
    province_survival.to_csv(SOURCE_DIR / "fig2_panel_b_province_survival.csv", index=False)
    summary.to_csv(SOURCE_DIR / "fig2_panel_b_survival_by_tercile.csv", index=False)
    return summary


def plot_run_length_survival(ax: plt.Axes, annual: pd.DataFrame, runs: pd.DataFrame) -> None:
    summary = make_survival_source(annual, runs)
    order = ["High", "Intermediate", "Low"]
    colors = {
        "High": PALETTE["blue_main"],
        "Intermediate": PALETTE["teal"],
        "Low": PALETTE["neutral_mid"],
    }
    labels = {
        "High": "High $P_{22}$ tercile",
        "Intermediate": "Intermediate",
        "Low": "Low $P_{22}$ tercile",
    }

    for group in order:
        data = summary[summary["clustering_group"] == group].sort_values("duration_threshold_days")
        x = data["duration_threshold_days"].to_numpy()
        y = data["mean_survival"].to_numpy()
        sem = data["sem_survival"].to_numpy()
        ax.plot(x, y, marker="o", markersize=3.0, lw=1.4, color=colors[group], label=labels[group])
        ax.fill_between(x, np.maximum(0, y - sem), np.minimum(1, y + sem), color=colors[group], alpha=0.14, lw=0)

    ax.set_xlim(1, int(summary["duration_threshold_days"].max()))
    ax.set_ylim(-0.02, 1.04)
    ax.set_xticks(np.arange(1, int(summary["duration_threshold_days"].max()) + 1))
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Minimum extreme-run duration (days)", fontsize=6.8)
    ax.set_ylabel("Run survival fraction", fontsize=6.8)
    ax.set_title("Run-length survival by clustering regime", fontsize=7.4, pad=3)
    ax.legend(loc="upper right", fontsize=5.8, handlelength=1.4, borderaxespad=0.2)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.35, alpha=0.45)
    apply_axis_style(ax)


def plot_seasonal_heatmap(ax: plt.Axes, annual: pd.DataFrame, seasonal: pd.DataFrame) -> None:
    order = annual.sort_values("p22", ascending=False)["province_en"].tolist()
    season_order = ["winter", "spring", "summer", "autumn"]
    season_labels = ["Winter", "Spring", "Summer", "Autumn"]

    pivot = seasonal.pivot_table(index="province_en", columns="season", values="p22")
    pivot = pivot.reindex(index=order, columns=season_order)

    norm = mcolors.Normalize(vmin=0.10, vmax=0.56)
    im = ax.imshow(pivot.values, aspect="auto", cmap=P22_CMAP, norm=norm)

    ax.set_xticks(np.arange(len(season_labels)))
    ax.set_xticklabels(season_labels, fontsize=6.2)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order, fontsize=5.2)
    ax.tick_params(axis="both", length=0)
    ax.tick_params(axis="y", pad=1.4)

    ax.set_xticks(np.arange(-0.5, len(season_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(order), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.35)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if pd.notna(value):
                color = "white" if value >= 0.42 else PALETTE["neutral_black"]
                weight = "bold" if value >= 0.50 else "normal"
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=4.1,
                    color=color,
                    fontweight=weight,
                )

    ax.set_title("Seasonal persistence of extreme peaking days", fontsize=7.4, pad=5)

    source = pivot.reset_index().rename(
        columns={
            "winter": "p22_winter",
            "spring": "p22_spring",
            "summer": "p22_summer",
            "autumn": "p22_autumn",
        }
    )
    source.to_csv(SOURCE_DIR / "fig2_panel_c_seasonal_p22_heatmap.csv", index=False)

    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_aic_comparison(ax: plt.Axes, annual: pd.DataFrame) -> None:
    required = ["aic_geom", "aic_dweibull"]
    aic = annual.dropna(subset=required).copy()
    aic["delta_aic_dweibull_minus_geom"] = aic["aic_dweibull"] - aic["aic_geom"]
    aic = aic.sort_values("delta_aic_dweibull_minus_geom", ascending=True).reset_index(drop=True)

    y = np.arange(len(aic))
    delta = aic["delta_aic_dweibull_minus_geom"].to_numpy(dtype=float)
    colors = np.where(delta < 0, PALETTE["teal"], PALETTE["neutral_dark"])
    ax.barh(y, delta, color=colors, height=0.64, edgecolor="none")
    ax.axvline(0, color=PALETTE["neutral_black"], linewidth=0.65)

    ax.set_yticks(y)
    ax.set_yticklabels(aic["province_en"].tolist(), fontsize=4.2)
    ax.invert_yaxis()
    ax.set_xlabel("Delta AIC: D-Weibull minus geometric", fontsize=6.2)
    ax.set_title("Run-length model comparison", fontsize=7.4, pad=3)
    ax.tick_params(axis="x", labelsize=5.2, width=0.55, length=2.0)
    ax.tick_params(axis="y", width=0.0, length=0.0)

    x_min = float(np.nanmin(delta))
    x_max = float(np.nanmax(delta))
    span = max(abs(x_min), abs(x_max), 1.0)
    ax.set_xlim(-span * 1.16, span * 1.16)

    for yi, value in enumerate(delta):
        if not np.isfinite(value):
            continue
        ha = "right" if value < 0 else "left"
        x_offset = -0.05 * span if value < 0 else 0.05 * span
        ax.text(
            value + x_offset,
            yi,
            f"{value:+.1f}",
            ha=ha,
            va="center",
            fontsize=3.9,
            color=PALETTE["neutral_black"],
        )

    ax.text(
        -0.98 * span,
        -1.0,
        "D-Weibull better",
        ha="left",
        va="center",
        fontsize=5.0,
        color=PALETTE["teal"],
        fontweight="bold",
    )
    ax.text(
        0.98 * span,
        -1.0,
        "Geometric better",
        ha="right",
        va="center",
        fontsize=5.0,
        color=PALETTE["neutral_dark"],
        fontweight="bold",
    )

    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    source_cols = [
        "province_en",
        "province_cn",
        "best_fit_model",
        "aic_geom",
        "aic_dweibull",
        "delta_aic_dweibull_minus_geom",
    ]
    source = aic[source_cols].copy()
    source.to_csv(SOURCE_DIR / "fig2_panel_d_aic_model_comparison.csv", index=False)


def plot_shape_correlation_panel(ax: plt.Axes, annual: pd.DataFrame) -> None:
    metrics = [
        ("demand_skewness", "Demand skewness", "Demand values"),
        ("demand_kurtosis", "Demand kurtosis", "Demand values"),
        ("extreme_skewness", "Extreme-day skewness", "Extreme values"),
        ("extreme_kurtosis", "Extreme-day kurtosis", "Extreme values"),
        ("runlen_skewness", "Run-length skewness", "Run lengths"),
        ("runlen_kurtosis", "Run-length kurtosis", "Run lengths"),
    ]

    rows = []
    for column, label, family in metrics:
        valid = annual[[column, "p22"]].dropna()
        r = valid[column].corr(valid["p22"])
        rows.append({"metric": label, "family": family, "pearson_r_with_p22": r})
    source = pd.DataFrame(rows)
    source.to_csv(SOURCE_DIR / "fig2_panel_e_shape_correlations.csv", index=False)

    plot_data = source.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(plot_data))
    family_colors = {
        "Demand values": PALETTE["neutral_mid"],
        "Extreme values": PALETTE["teal"],
        "Run lengths": PALETTE["red_strong"],
    }
    colors = [family_colors[f] for f in plot_data["family"]]

    ax.barh(y, plot_data["pearson_r_with_p22"], color=colors, edgecolor="white", linewidth=0.4, height=0.68)
    ax.axvline(0, color=PALETTE["neutral_black"], lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_data["metric"], fontsize=5.3)
    ax.set_xlim(-0.78, 0.22)
    ax.set_xticks([-0.75, -0.5, -0.25, 0, 0.25])
    ax.set_xlabel("Pearson r with annual $P_{22}$", fontsize=6.6)
    ax.set_title("Tail heaviness and temporal clustering are distinct", fontsize=7.4, pad=3)
    apply_axis_style(ax)

    for yi, val in enumerate(plot_data["pearson_r_with_p22"].to_numpy()):
        if np.isnan(val):
            continue
        if val >= 0:
            xpos = val + 0.018
            ha = "left"
        else:
            xpos = val - 0.018
            ha = "right"
        ax.text(
            xpos,
            yi,
            f"{val:+.2f}",
            va="center",
            ha=ha,
            fontsize=5.0,
            color=PALETTE["neutral_dark"],
        )

    legend_handles = [
        mpatches.Patch(facecolor=family_colors["Demand values"], edgecolor="none", label="Demand values"),
        mpatches.Patch(facecolor=family_colors["Extreme values"], edgecolor="none", label="Extreme values"),
        mpatches.Patch(facecolor=family_colors["Run lengths"], edgecolor="none", label="Run lengths"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=5.0, handlelength=1.0, borderaxespad=0.25)


def export_integrated_source_data(annual: pd.DataFrame, seasonal: pd.DataFrame, runs: pd.DataFrame) -> None:
    annual.to_csv(SOURCE_DIR / "fig2_source_annual_clustering_metrics.csv", index=False)
    seasonal.to_csv(SOURCE_DIR / "fig2_source_seasonal_clustering_metrics.csv", index=False)
    runs.to_csv(SOURCE_DIR / "fig2_source_extreme_event_runs.csv", index=False)


def make_figure() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    annual, seasonal, runs = load_m1_outputs()
    export_integrated_source_data(annual, seasonal, runs)

    fig = plt.figure(figsize=(7.55, 9.15))
    gs = fig.add_gridspec(
        nrows=3,
        ncols=2,
        width_ratios=[1.06, 0.94],
        height_ratios=[0.88, 0.82, 0.74],
        left=0.07,
        right=0.985,
        bottom=0.065,
        top=0.965,
        wspace=0.30,
        hspace=0.44,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1:, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, 1])

    plot_china_p22_map(ax_a, annual)
    plot_run_length_survival(ax_b, annual, runs)
    plot_seasonal_heatmap(ax_c, annual, seasonal)
    plot_aic_comparison(ax_d, annual)
    plot_shape_correlation_panel(ax_e, annual)

    add_panel_label(ax_a, "a", x=-0.03, y=1.03)
    add_panel_label(ax_b, "b", x=-0.11, y=1.03)
    add_panel_label(ax_c, "c", x=-0.10, y=1.03)
    add_panel_label(ax_d, "d", x=-0.11, y=1.03)
    add_panel_label(ax_e, "e", x=-0.11, y=1.03)

    out_base = FIGURE_DIR / "fig2_event_clustering_nature"
    fig.savefig(f"{out_base}.svg", bbox_inches="tight")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_base}.svg")
    print(f"Saved: {out_base}.pdf")
    print(f"Saved: {out_base}.png")
    print(f"Source data: {SOURCE_DIR}")


if __name__ == "__main__":
    make_figure()
