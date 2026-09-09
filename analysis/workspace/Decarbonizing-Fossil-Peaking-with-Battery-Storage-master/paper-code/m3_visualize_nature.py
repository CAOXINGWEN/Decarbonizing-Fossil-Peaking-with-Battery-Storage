"""
Nature-style stage figure for M3 dispatch results.

This script turns the M3 dispatch outputs into a compact, publication-style
overview figure for internal review:
  a. Provincial annual curtailment resource map
  b. Province-wise optimal capacity required by technology and duration
  c. Province-wise optimal equivalent full cycles (EFC) by technology and duration
  d. Province-level best outcome ranked by annual curtailment scale

Primary output is SVG with editable text. PDF and PNG are secondary exports.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# ---------------------------------------------------------------------------
# Nature-style defaults
# ---------------------------------------------------------------------------

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.size"] = 8.2
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.9
plt.rcParams["legend.frameon"] = False


PAPER_CODE_DIR = Path(__file__).resolve().parent
MASTER_DIR = PAPER_CODE_DIR.parent
PROJECT_DIR = MASTER_DIR.parent

OUTPUT_DIR = PAPER_CODE_DIR / "output" / "m3_dispatch"
FIGURE_DIR = OUTPUT_DIR / "nature_figure"
SOURCE_DIR = FIGURE_DIR / "source_data"

RESULTS_CSV = OUTPUT_DIR / "m3_dispatch_results.csv"
CURTAILMENT_CSV = OUTPUT_DIR / "daily_provincial_curtailment.csv"
GEOJSON_PATH = PROJECT_DIR / "dataset" / "中国_省.geojson"
FIG_BASE = FIGURE_DIR / "fig3_m3_dispatch_nature"


PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "teal": "#42949E",
    "green": "#8BCF8B",
    "lilac": "#B9A7E8",
    "violet": "#7C6CCF",
    "orange": "#E28E2C",
    "neutral_mid": "#767676",
    "neutral_dark": "#4D4D4D",
    "neutral_light": "#EDEDED",
    "neutral_black": "#272727",
}

TECH_ORDER = [
    "li_lfp",
    "vrfb",
    "fecr_flow",
    "gravity",
    "molten_salt_tes",
    "caes",
    "laes",
    "h2_cavern",
]

DURATION_ORDER = [4, 6, 8, 10, 12, 16, 24, 48, 72, 168]

TECH_LABELS = {
    "li_lfp": "LFP",
    "vrfb": "VRFB",
    "fecr_flow": "FeCr flow",
    "gravity": "Gravity",
    "molten_salt_tes": "Molten salt TES",
    "caes": "CAES",
    "laes": "LAES",
    "h2_cavern": "H2 cavern",
}

TECH_COLORS = {
    "li_lfp": PALETTE["blue_main"],
    "vrfb": PALETTE["blue_secondary"],
    "fecr_flow": PALETTE["teal"],
    "gravity": PALETTE["orange"],
    "molten_salt_tes": PALETTE["neutral_mid"],
    "caes": PALETTE["green"],
    "laes": PALETTE["lilac"],
    "h2_cavern": PALETTE["violet"],
}

CURTAILMENT_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "curtailment_blue",
    ["#F4F7FA", "#D7E3F3", "#8FB0DB", PALETTE["blue_main"]],
)

CAPACITY_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "capacity_blueviolet",
    ["#F5F7FB", "#E1EAF7", "#B8CDEA", "#7C9FD1", PALETTE["blue_main"]],
)

EFC_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "efc_green",
    ["#F5F8F2", "#E0EECB", "#B6DAA0", "#7BC07B", "#3D8E67"],
)

SUPPLY_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "supply_ratio_blue",
    ["#F6F7F9", "#D9E6F4", "#8FB5DF", PALETTE["blue_main"]],
)

SUPPLY_CONTRAST_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "nature_supply_ratio",
    [
        "#A9343A",
        "#E05D3C",
        "#EFEAE1",
        "#7EA9C9",
        "#2F78A8",
    ],
)

CAPACITY_BINS_GWH = [0.0, 10.0, 50.0, 150.0, 500.0, np.inf]
CAPACITY_BIN_LABELS = ["<10 GWh", "10-50 GWh", "50-150 GWh", "150-500 GWh", ">=500 GWh"]
CAPACITY_BIN_MARKERS = ["o", "^", "s", "D", "h"]
CAPACITY_BIN_SIZES = [17.0, 31.0, 50.0, 74.0, 106.0]

PROVINCE_ORDER = [
    "Beijing",
    "Tianjin",
    "Hebei",
    "Shanxi",
    "Inner Mongolia",
    "Liaoning",
    "Jilin",
    "Heilongjiang",
    "Shanghai",
    "Jiangsu",
    "Zhejiang",
    "Anhui",
    "Fujian",
    "Jiangxi",
    "Shandong",
    "Henan",
    "Hubei",
    "Hunan",
    "Guangdong",
    "Guangxi",
    "Hainan",
    "Chongqing",
    "Sichuan",
    "Guizhou",
    "Yunnan",
    "Shaanxi",
    "Gansu",
    "Qinghai",
    "Ningxia",
    "Xinjiang",
]

PROVINCE_SHORT_LABELS = {
    "Beijing": "Beijing",
    "Tianjin": "Tianjin",
    "Hebei": "Hebei",
    "Shanxi": "Shanxi",
    "Inner Mongolia": "Inner Mong.",
    "Liaoning": "Liaoning",
    "Jilin": "Jilin",
    "Jiangsu": "Jiangsu",
    "Zhejiang": "Zhejiang",
    "Anhui": "Anhui",
    "Fujian": "Fujian",
    "Jiangxi": "Jiangxi",
    "Shandong": "Shandong",
    "Henan": "Henan",
    "Hubei": "Hubei",
    "Hunan": "Hunan",
    "Guangdong": "Guangdong",
    "Guangxi": "Guangxi",
    "Sichuan": "Sichuan",
    "Guizhou": "Guizhou",
    "Yunnan": "Yunnan",
    "Shaanxi": "Shaanxi",
    "Gansu": "Gansu",
    "Qinghai": "Qinghai",
    "Ningxia": "Ningxia",
    "Xinjiang": "Xinjiang",
}


def q25(series: pd.Series) -> float:
    return float(series.quantile(0.25))


def q75(series: pd.Series) -> float:
    return float(series.quantile(0.75))


def apply_axis_style(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", labelsize=6.8, width=0.7, length=2.5)
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(0.9)


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.03) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.2,
        fontweight="bold",
        color=PALETTE["neutral_black"],
    )


def iter_exterior_rings(geometry: dict):
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


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    results = pd.read_csv(RESULTS_CSV)
    curtailment = pd.read_csv(CURTAILMENT_CSV, index_col=0)
    return results, curtailment


def build_annual_curtailment_table(results: pd.DataFrame, curtailment: pd.DataFrame) -> pd.DataFrame:
    prov_map = (
        results[["province_en", "province_cn"]]
        .drop_duplicates()
        .set_index("province_en")["province_cn"]
        .to_dict()
    )

    annual_kwh = curtailment.sum(axis=0)
    annual_df = pd.DataFrame(
        {
            "province_en": annual_kwh.index,
            "province_cn": [prov_map.get(p, p) for p in annual_kwh.index],
            "annual_curtailment_kwh": annual_kwh.values,
        }
    )
    annual_df["annual_curtailment_gwh"] = annual_df["annual_curtailment_kwh"] / 1_000_000.0
    annual_df["log10_annual_curtailment_gwh"] = np.log10(annual_df["annual_curtailment_gwh"])
    return annual_df


def build_best_combo_table(results: pd.DataFrame, annual_df: pd.DataFrame) -> pd.DataFrame:
    # For each province-tech-duration, keep the first capacity level that attains
    # the maximum supply ratio. The grid search is monotone in capacity, so this
    # corresponds to the minimum capacity needed to reach the best dispatch point.
    idx = results.groupby(["tech_id", "province_en", "duration_h"])["satisfy_rate_pct"].idxmax()
    best_combo = results.loc[idx].copy()

    best_combo = best_combo.merge(
        annual_df[["province_en", "annual_curtailment_kwh", "annual_curtailment_gwh"]],
        on="province_en",
        how="left",
    )
    best_combo["charge_share_of_curtailment_pct"] = np.where(
        best_combo["annual_curtailment_kwh"] > 0,
        best_combo["total_charge_kwh"] / best_combo["annual_curtailment_kwh"] * 100.0,
        np.nan,
    )
    best_combo["spillage_share_of_curtailment_pct"] = np.where(
        best_combo["annual_curtailment_kwh"] > 0,
        best_combo["total_spillage_kwh"] / best_combo["annual_curtailment_kwh"] * 100.0,
        np.nan,
    )
    best_combo["supply_share_of_curtailment_pct"] = np.where(
        best_combo["annual_curtailment_kwh"] > 0,
        best_combo["total_supply_kwh"] / best_combo["annual_curtailment_kwh"] * 100.0,
        np.nan,
    )
    return best_combo


def enrich_capacity_curve_metrics(results: pd.DataFrame) -> pd.DataFrame:
    """Add point-level efficiency and marginal-gain metrics to the M3 grid-search table."""
    pieces = []
    for _keys, group in results.groupby(["tech_id", "province_en", "duration_h"], sort=False):
        group = group.sort_values("capacity_mwh").copy()
        capacity_gwh = group["capacity_mwh"].astype(float) / 1000.0
        supply_ratio = group["satisfy_rate_pct"].astype(float)
        total_supply_gwh = group["total_supply_kwh"].astype(float) / 1_000_000.0

        group["capacity_gwh"] = capacity_gwh
        group["capacity_efficiency_pct_per_gwh"] = np.where(
            capacity_gwh > 0,
            supply_ratio / capacity_gwh,
            np.nan,
        )
        group["throughput_efficiency_gwh_per_gwh"] = np.where(
            capacity_gwh > 0,
            total_supply_gwh / capacity_gwh,
            np.nan,
        )

        delta_capacity_gwh = capacity_gwh.diff()
        delta_supply_ratio = supply_ratio.diff()
        group["delta_capacity_gwh"] = delta_capacity_gwh
        group["delta_supply_ratio_pctpt"] = delta_supply_ratio
        group["marginal_supply_gain_pct_per_gwh"] = np.where(
            delta_capacity_gwh > 0,
            delta_supply_ratio / delta_capacity_gwh,
            np.nan,
        )
        group["marginal_supply_gain_pct_per_mwh"] = np.where(
            delta_capacity_gwh > 0,
            delta_supply_ratio / (delta_capacity_gwh * 1000.0),
            np.nan,
        )
        pieces.append(group)

    return pd.concat(pieces, ignore_index=True)


def summarize_capacity_curve_metrics(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize each province-tech-duration curve with saturation-style engineering metrics."""
    rows = []
    for keys, group in results.groupby(["tech_id", "province_en", "duration_h"], sort=False):
        tech_id, province_en, duration_h = keys
        group = group.sort_values("capacity_mwh").copy()
        max_supply = float(group["satisfy_rate_pct"].max())
        if max_supply <= 0:
            continue

        threshold_95 = 0.95 * max_supply
        threshold_99 = 0.99 * max_supply

        c95_rows = group[group["satisfy_rate_pct"] >= threshold_95]
        c99_rows = group[group["satisfy_rate_pct"] >= threshold_99]
        c95_row = c95_rows.iloc[0] if not c95_rows.empty else group.iloc[-1]
        c99_row = c99_rows.iloc[0] if not c99_rows.empty else group.iloc[-1]

        marginal = group["marginal_supply_gain_pct_per_gwh"].astype(float)
        valid_marginal = group[marginal.notna()].copy()
        knee_row = group.iloc[-1]
        if not valid_marginal.empty:
            near_saturation = valid_marginal[
                (valid_marginal["marginal_supply_gain_pct_per_gwh"] <= 1.0)
                & (valid_marginal["satisfy_rate_pct"] >= 0.8 * max_supply)
            ]
            if not near_saturation.empty:
                knee_row = near_saturation.iloc[0]

        eff_idx = group["capacity_efficiency_pct_per_gwh"].astype(float).idxmax()
        eff_row = group.loc[eff_idx]

        rows.append(
            {
                "tech_id": tech_id,
                "province_en": province_en,
                "province_cn": group["province_cn"].iloc[0],
                "duration_h": int(duration_h),
                "max_supply_ratio_pct": max_supply,
                "max_supply_capacity_mwh": float(group.loc[group["satisfy_rate_pct"].idxmax(), "capacity_mwh"]),
                "c95_capacity_mwh": float(c95_row["capacity_mwh"]),
                "c95_supply_ratio_pct": float(c95_row["satisfy_rate_pct"]),
                "c99_capacity_mwh": float(c99_row["capacity_mwh"]),
                "c99_supply_ratio_pct": float(c99_row["satisfy_rate_pct"]),
                "knee_capacity_mwh": float(knee_row["capacity_mwh"]),
                "knee_supply_ratio_pct": float(knee_row["satisfy_rate_pct"]),
                "knee_marginal_gain_pct_per_gwh": float(knee_row["marginal_supply_gain_pct_per_gwh"])
                if pd.notna(knee_row["marginal_supply_gain_pct_per_gwh"])
                else np.nan,
                "peak_capacity_efficiency_pct_per_gwh": float(eff_row["capacity_efficiency_pct_per_gwh"]),
                "peak_capacity_efficiency_capacity_mwh": float(eff_row["capacity_mwh"]),
                "peak_throughput_efficiency_gwh_per_gwh": float(eff_row["throughput_efficiency_gwh_per_gwh"]),
            }
        )

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["technology_cn"] = summary["tech_id"].map(TECH_LABELS)
    return summary


def build_duration_summary(best_combo: pd.DataFrame) -> pd.DataFrame:
    summary = best_combo.groupby(["tech_id", "duration_h"]).agg(
        province_count=("province_en", "nunique"),
        satisfy_rate_median=("satisfy_rate_pct", "median"),
        satisfy_rate_q25=("satisfy_rate_pct", q25),
        satisfy_rate_q75=("satisfy_rate_pct", q75),
        capacity_mwh_median=("capacity_mwh", "median"),
        capacity_mwh_q25=("capacity_mwh", q25),
        capacity_mwh_q75=("capacity_mwh", q75),
        efc_median=("efc", "median"),
        efc_q25=("efc", q25),
        efc_q75=("efc", q75),
        charge_share_median=("charge_share_of_curtailment_pct", "median"),
        charge_share_q25=("charge_share_of_curtailment_pct", q25),
        charge_share_q75=("charge_share_of_curtailment_pct", q75),
        spillage_share_median=("spillage_share_of_curtailment_pct", "median"),
        spillage_share_q25=("spillage_share_of_curtailment_pct", q25),
        spillage_share_q75=("spillage_share_of_curtailment_pct", q75),
    ).reset_index()
    summary["technology_cn"] = summary["tech_id"].map(TECH_LABELS)
    return summary


def build_metric_matrix(summary: pd.DataFrame, metric_prefix: str) -> pd.DataFrame:
    matrix = (
        summary.pivot(index="tech_id", columns="duration_h", values=f"{metric_prefix}_median")
        .reindex(index=TECH_ORDER, columns=DURATION_ORDER)
    )
    return matrix


def build_best_overall_table(best_combo: pd.DataFrame) -> pd.DataFrame:
    # Choose the best row per province, preferring higher supply ratio and, under
    # ties, lower capacity and shorter duration.
    ordered = best_combo.sort_values(
        ["province_en", "satisfy_rate_pct", "capacity_mwh", "duration_h"],
        ascending=[True, False, True, True],
    )
    best_overall = ordered.groupby("province_en", as_index=False).first()
    best_overall["best_tech_label"] = best_overall["tech_id"].map(TECH_LABELS)
    return best_overall


def build_province_order_by_adequacy(best_combo: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    province_df = (
        best_combo[
            [
                "province_en",
                "province_cn",
                "annual_curtailment_kwh",
                "annual_curtailment_gwh",
                "total_demand_kwh",
            ]
        ]
        .drop_duplicates(subset=["province_en"])
        .copy()
    )
    province_df["demand_gwh"] = province_df["total_demand_kwh"] / 1_000_000.0
    province_df["curtailment_to_demand_ratio"] = np.where(
        province_df["total_demand_kwh"] > 0,
        province_df["annual_curtailment_kwh"] / province_df["total_demand_kwh"],
        np.nan,
    )
    province_df = province_df.sort_values(
        ["curtailment_to_demand_ratio", "demand_gwh", "province_en"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    return province_df["province_en"].tolist(), province_df


def build_province_order_by_capacity(best_combo: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    province_df = (
        best_combo[
            [
                "province_en",
                "province_cn",
                "annual_curtailment_kwh",
                "annual_curtailment_gwh",
                "total_demand_kwh",
                "capacity_mwh",
            ]
        ]
        .copy()
    )
    summary = (
        province_df.groupby(["province_en", "province_cn"], as_index=False)
        .agg(
            max_optimal_capacity_mwh=("capacity_mwh", "max"),
            median_optimal_capacity_mwh=("capacity_mwh", "median"),
            annual_curtailment_kwh=("annual_curtailment_kwh", "first"),
            annual_curtailment_gwh=("annual_curtailment_gwh", "first"),
            total_demand_kwh=("total_demand_kwh", "first"),
        )
    )
    summary["max_optimal_capacity_gwh"] = summary["max_optimal_capacity_mwh"] / 1000.0
    summary["median_optimal_capacity_gwh"] = summary["median_optimal_capacity_mwh"] / 1000.0
    summary["demand_gwh"] = summary["total_demand_kwh"] / 1_000_000.0
    summary = summary.sort_values(
        ["max_optimal_capacity_mwh", "median_optimal_capacity_mwh", "province_en"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return summary["province_en"].tolist(), summary


def get_province_colors(province_order: list[str]) -> dict[str, tuple[float, float, float, float]]:
    palettes = [plt.get_cmap("tab20"), plt.get_cmap("tab20b")]
    colors = []
    for cmap in palettes:
        colors.extend(cmap(np.linspace(0, 1, 20)))
    return {province: colors[i] for i, province in enumerate(province_order)}


def build_technology_orders_by_capacity(
    best_combo: pd.DataFrame,
) -> tuple[dict[str, list[str]], pd.DataFrame]:
    orders: dict[str, list[str]] = {}
    rows = []

    for tech_id in TECH_ORDER:
        tech_rows = best_combo[best_combo["tech_id"] == tech_id].copy()
        if tech_rows.empty:
            orders[tech_id] = []
            continue

        summary = (
            tech_rows.groupby(["province_en", "province_cn"], as_index=False)
            .agg(
                max_optimal_capacity_mwh=("capacity_mwh", "max"),
                median_optimal_capacity_mwh=("capacity_mwh", "median"),
                max_supply_ratio_pct=("satisfy_rate_pct", "max"),
            )
            .sort_values(
                ["max_optimal_capacity_mwh", "median_optimal_capacity_mwh", "province_en"],
                ascending=[False, False, True],
            )
            .reset_index(drop=True)
        )
        summary.insert(0, "tech_id", tech_id)
        summary.insert(1, "technology", TECH_LABELS[tech_id])
        summary["rank_within_technology"] = np.arange(1, len(summary) + 1)
        rows.append(summary)
        orders[tech_id] = summary["province_en"].tolist()

    order_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return orders, order_df


def assign_capacity_bin(capacity_gwh: pd.Series) -> pd.Series:
    return pd.cut(
        capacity_gwh,
        bins=CAPACITY_BINS_GWH,
        labels=range(len(CAPACITY_BIN_LABELS)),
        include_lowest=True,
        right=False,
    ).astype(int)


def plot_technology_capacity_supply_axis(
    ax: plt.Axes,
    tech_id: str,
    best_combo: pd.DataFrame,
    province_order: list[str],
    norm: mcolors.Normalize,
    x_label_fontsize: float = 5.05,
    y_label_fontsize: float = 5.0,
    title_fontsize: float = 6.8,
    marker_scale: float = 1.0,
) -> None:
    tech_rows = best_combo[best_combo["tech_id"] == tech_id].copy()
    tech_rows["duration_h"] = tech_rows["duration_h"].astype(int)
    tech_rows["capacity_gwh"] = tech_rows["capacity_mwh"] / 1000.0
    tech_rows["capacity_bin"] = assign_capacity_bin(tech_rows["capacity_gwh"])

    durations = [
        duration_h
        for duration_h in DURATION_ORDER
        if duration_h in set(tech_rows["duration_h"])
    ]
    duration_to_y = {duration_h: idx for idx, duration_h in enumerate(durations)}
    province_to_x = {province: idx for idx, province in enumerate(province_order)}

    tech_rows["x_pos"] = tech_rows["province_en"].map(province_to_x)
    tech_rows["y_pos"] = tech_rows["duration_h"].map(duration_to_y)
    tech_rows = tech_rows.dropna(subset=["x_pos", "y_pos"]).copy()

    for x in range(len(province_order)):
        ax.axvline(x, color="#F2F2F2", lw=0.25, zorder=0)
    for y in range(len(durations)):
        ax.axhline(y, color="#E8E8E8", lw=0.36, zorder=0)

    for bin_id, marker in enumerate(CAPACITY_BIN_MARKERS):
        marker_rows = tech_rows[tech_rows["capacity_bin"] == bin_id]
        if marker_rows.empty:
            continue
        ax.scatter(
            marker_rows["x_pos"],
            marker_rows["y_pos"],
            s=CAPACITY_BIN_SIZES[bin_id] * marker_scale,
            c=marker_rows["satisfy_rate_pct"],
            cmap=SUPPLY_CONTRAST_CMAP,
            norm=norm,
            marker=marker,
            alpha=0.90,
            edgecolor="white",
            linewidth=0.26,
            zorder=3,
        )

    ax.set_xlim(-0.65, len(province_order) - 0.35)
    ax.set_ylim(-0.55, len(durations) - 0.55)
    ax.set_xticks(np.arange(len(province_order)))
    ax.set_xticklabels(
        [PROVINCE_SHORT_LABELS.get(province, province) for province in province_order],
        fontsize=x_label_fontsize,
        rotation=90,
        ha="center",
        va="top",
    )
    for label in ax.get_xticklabels():
        label.set_color(PALETTE["neutral_black"])
        label.set_fontweight("medium")
    ax.set_yticks(range(len(durations)))
    ax.set_yticklabels([str(duration_h) for duration_h in durations], fontsize=y_label_fontsize)
    ax.set_title(
        TECH_LABELS[tech_id],
        fontsize=title_fontsize,
        color=PALETTE["neutral_black"],
        fontweight="semibold",
        pad=2.2,
    )
    ax.tick_params(axis="x", width=0.42, length=1.4, pad=0.8)
    ax.tick_params(axis="y", width=0.48, length=1.8, pad=1.0)
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(0.64)


def add_panel_b_capacity_legend(legend_ax: plt.Axes) -> None:
    legend_ax.set_axis_off()
    handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="none",
            markerfacecolor="#9A9A9A",
            markeredgecolor="white",
            markeredgewidth=0.45,
            alpha=0.90,
            markersize=float(np.sqrt(size)),
            label=label,
        )
        for marker, size, label in zip(
            CAPACITY_BIN_MARKERS,
            CAPACITY_BIN_SIZES,
            CAPACITY_BIN_LABELS,
        )
    ]
    legend_ax.legend(
        handles=handles,
        loc="center",
        title="Optimal capacity",
        fontsize=5.7,
        title_fontsize=6.0,
        ncol=5,
        frameon=False,
        borderpad=0.0,
        labelspacing=0.25,
        handlelength=0.95,
        handletextpad=0.38,
        columnspacing=0.85,
    )


def downsample_curve(df: pd.DataFrame, max_points: int = 120) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    indices = np.linspace(0, len(df) - 1, max_points).round().astype(int)
    indices = np.unique(np.r_[0, indices, len(df) - 1])
    return df.iloc[indices]


def save_source_data(
    annual_df: pd.DataFrame,
    best_combo: pd.DataFrame,
    summary: pd.DataFrame,
    best_overall: pd.DataFrame,
    curve_metrics: pd.DataFrame,
    curve_summary: pd.DataFrame,
) -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    annual_df.to_csv(SOURCE_DIR / "fig3_source_annual_curtailment.csv", index=False)
    best_combo.to_csv(SOURCE_DIR / "fig3_source_best_by_tech_province_duration.csv", index=False)
    summary.to_csv(SOURCE_DIR / "fig3_source_tech_duration_summary.csv", index=False)
    best_overall.to_csv(SOURCE_DIR / "fig3_source_best_by_province.csv", index=False)
    curve_metrics.to_csv(SOURCE_DIR / "fig3_source_capacity_curve_metrics_points.csv", index=False)
    curve_summary.to_csv(SOURCE_DIR / "fig3_source_capacity_curve_metrics_summary.csv", index=False)


def plot_china_curtailment_map(ax: plt.Axes, annual_df: pd.DataFrame) -> None:
    with open(GEOJSON_PATH, encoding="utf-8") as f:
        geojson = json.load(f)

    value_by_cn = dict(zip(annual_df["province_cn"], annual_df["annual_curtailment_gwh"]))
    centroids = {}
    patches = []
    facecolors = []

    positive = annual_df.loc[annual_df["annual_curtailment_gwh"] > 0, "annual_curtailment_gwh"]
    norm = mcolors.LogNorm(vmin=positive.min(), vmax=positive.max())

    for feature in geojson["features"]:
        geometry = feature.get("geometry", {})
        if geometry.get("type") not in {"Polygon", "MultiPolygon"}:
            continue

        name_cn = feature.get("properties", {}).get("name")
        value = value_by_cn.get(name_cn)
        fill = CURTAILMENT_CMAP(norm(value)) if value is not None else PALETTE["neutral_light"]

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

    sm = plt.cm.ScalarMappable(norm=norm, cmap=CURTAILMENT_CMAP)
    cbar = plt.colorbar(sm, ax=ax, fraction=0.036, pad=0.015)
    cbar.set_label("Annual curtailment (GWh)", fontsize=6.8, labelpad=3)
    cbar.ax.tick_params(labelsize=6.0, width=0.5, length=2)
    cbar.outline.set_linewidth(0.4)

    callouts = {
        "北京市": ("Beijing", (119.5, 43.5)),
        "江苏省": ("Jiangsu", (123.5, 33.5)),
        "内蒙古自治区": ("Inner Mongolia", (107.0, 49.2)),
        "新疆维吾尔自治区": ("Xinjiang", (84.5, 33.0)),
        "甘肃省": ("Gansu", (98.0, 42.3)),
    }
    for name_cn, (label, xytext) in callouts.items():
        if name_cn not in centroids or name_cn not in value_by_cn:
            continue
        ax.annotate(
            label,
            xy=centroids[name_cn],
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

    ax.set_title("Annual curtailment resource", fontsize=7.6, pad=3)


def plot_metric_heatmap(
    ax: plt.Axes,
    summary: pd.DataFrame,
    metric_prefix: str,
    cmap: mcolors.Colormap,
    title: str,
    cbar_label: str,
    log_scale: bool = False,
    cbar_ticks: list[float] | None = None,
    annotate: bool = False,
    annotate_fontsize: float = 4.4,
) -> None:
    matrix = build_metric_matrix(summary, metric_prefix)
    values = matrix.to_numpy(dtype=float)

    if log_scale:
        positive = values[np.isfinite(values) & (values > 0)]
        norm = mcolors.LogNorm(vmin=float(np.nanmin(positive)), vmax=float(np.nanmax(positive)))
    else:
        norm = mcolors.Normalize(vmin=float(np.nanmin(values)), vmax=float(np.nanmax(values)))

    im = ax.imshow(
        values,
        cmap=cmap,
        norm=norm,
        origin="upper",
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_title(title, fontsize=7.7, pad=3)
    ax.set_xlabel("Duration (h)", fontsize=7.1)
    ax.set_ylabel("Technology", fontsize=7.1)
    ax.set_xticks(np.arange(len(DURATION_ORDER)))
    ax.set_xticklabels([str(d) for d in DURATION_ORDER], fontsize=6.2)
    ax.set_yticks(np.arange(len(TECH_ORDER)))
    ax.set_yticklabels([TECH_LABELS[t] for t in TECH_ORDER], fontsize=6.2)
    ax.set_xlim(-0.5, len(DURATION_ORDER) - 0.5)
    ax.set_ylim(len(TECH_ORDER) - 0.5, -0.5)
    ax.tick_params(axis="both", which="major", length=0)
    ax.set_facecolor("white")

    ax.set_xticks(np.arange(-0.5, len(DURATION_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(TECH_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.85)
    ax.tick_params(which="minor", bottom=False, left=False)
    apply_axis_style(ax)
    ax.tick_params(axis="both", which="major", labelsize=6.2, width=0.7, length=0)

    if annotate:
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j]
                if not np.isfinite(value):
                    continue
                if abs(value - round(value)) < 1e-6:
                    label = f"{value:,.0f}"
                else:
                    label = f"{value:,.1f}"
                rgba = cmap(norm(value))
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                text_color = "white" if luminance < 0.56 else PALETTE["neutral_black"]
                ax.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    fontsize=annotate_fontsize,
                    color=text_color,
                    fontweight="semibold",
                    zorder=5,
                )

    cax = inset_axes(
        ax,
        width="3.2%",
        height="82%",
        loc="center right",
        borderpad=0.85,
    )
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=5.9, width=0.5, length=2)
    cbar.outline.set_linewidth(0.45)
    cbar.set_label(cbar_label, fontsize=6.5, labelpad=2.5)
    if cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks)


def plot_panel_b(
    container_ax: plt.Axes,
    best_combo: pd.DataFrame,
    results: pd.DataFrame,
) -> tuple[plt.Axes, plt.Axes]:
    container_ax.set_axis_off()

    inner = GridSpecFromSubplotSpec(
        8,
        1,
        subplot_spec=container_ax.get_subplotspec(),
        height_ratios=[1.0] * 8,
        hspace=1.32,
    )

    tech_orders, _order_df = build_technology_orders_by_capacity(best_combo)
    norm = mcolors.Normalize(vmin=0.0, vmax=100.0)
    panel_axes = []

    for idx, tech_id in enumerate(TECH_ORDER):
        ax = container_ax.figure.add_subplot(inner[idx, 0])
        plot_technology_capacity_supply_axis(
            ax=ax,
            tech_id=tech_id,
            best_combo=best_combo,
            province_order=tech_orders[tech_id],
            norm=norm,
            x_label_fontsize=9.4,
            y_label_fontsize=6.2,
            title_fontsize=7.4,
            marker_scale=0.88,
        )
        panel_axes.append(ax)

    panel_axes[-1].set_xlabel("Province sorted by optimal capacity", fontsize=7.2, labelpad=1.8)
    for ax in panel_axes:
        ax.set_ylabel("Duration (h)", fontsize=6.7, labelpad=1.8)

    container_ax.text(
        0.0,
        1.012,
        "Technology-specific optimal capacity and maximum supply ratio",
        transform=container_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.5,
        fontweight="semibold",
        color=PALETTE["neutral_black"],
    )

    return panel_axes[0], panel_axes[-1]


def plot_panel_b1(ax: plt.Axes, best_combo: pd.DataFrame, province_order: list[str]) -> None:
    data = best_combo.copy()
    data["capacity_gwh"] = data["capacity_mwh"] / 1000.0
    prov_to_idx = {p: i for i, p in enumerate(province_order)}

    x_positions = []
    x_labels = []
    tech_centers = []
    tech_bounds = []

    col_idx = 0
    tech_duration_indices: dict[str, dict[int, int]] = {}
    for tech_id in TECH_ORDER:
        tech_duration_indices[tech_id] = {}
        start_col = col_idx
        for duration_h in DURATION_ORDER:
            has_data = bool(
                ((data["tech_id"] == tech_id) & (data["duration_h"] == duration_h)).any()
            )
            if not has_data:
                continue
            tech_duration_indices[tech_id][duration_h] = col_idx
            x_positions.append(col_idx)
            x_labels.append(str(duration_h))
            col_idx += 1
        end_col = col_idx - 1
        tech_centers.append((start_col + end_col) / 2 if end_col >= start_col else start_col)
        tech_bounds.append(end_col + 0.5)
        col_idx += 1

    n_cols = max(x_positions) + 1 if x_positions else 0
    values = np.full((len(province_order), n_cols), np.nan)
    sizes = np.full((len(province_order), n_cols), np.nan)

    max_cap = float(data["capacity_gwh"].max())
    for row in data.itertuples(index=False):
        tech_id = getattr(row, "tech_id")
        province_en = getattr(row, "province_en")
        duration_h = getattr(row, "duration_h")
        if (
            tech_id not in tech_duration_indices
            or province_en not in prov_to_idx
            or duration_h not in tech_duration_indices[tech_id]
        ):
            continue
        matrix_row = prov_to_idx[province_en]
        matrix_col = tech_duration_indices[tech_id][duration_h]
        values[matrix_row, matrix_col] = float(getattr(row, "satisfy_rate_pct"))
        sizes[matrix_row, matrix_col] = float(getattr(row, "capacity_gwh")) / max_cap

    ax.set_facecolor("white")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if not np.isfinite(value):
                continue
            size_scale = sizes[i, j]
            bubble_area = 12 + 620 * (size_scale ** 0.7)
            ax.scatter(
                j,
                i,
                s=bubble_area,
                color=PALETTE["blue_secondary"],
                alpha=0.62,
                edgecolor=PALETTE["blue_main"],
                linewidth=0.28,
                zorder=3,
                clip_on=False,
            )

    for boundary in tech_bounds[:-1]:
        ax.axvline(boundary, color=PALETTE["neutral_mid"], lw=0.7, alpha=0.7, zorder=1)

    ax.set_xlim(-0.75, n_cols - 0.25)
    ax.set_ylim(values.shape[0] - 0.9, -0.9)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=4.9)
    ax.set_xlabel("Duration (h)", fontsize=6.0, labelpad=2.0)
    ax.set_yticks(np.arange(len(province_order)))
    ax.set_yticklabels(
        [PROVINCE_SHORT_LABELS.get(p, p) for p in province_order],
        fontsize=4.8,
    )
    ax.set_ylabel("Province, sorted by maximum optimal capacity", fontsize=6.0, labelpad=2.0)
    ax.tick_params(axis="y", length=0, pad=1.0)
    ax.tick_params(axis="x", labelsize=5.2, width=0.55, length=2.0)
    ax.grid(axis="both", color="#EAEAEA", lw=0.45, alpha=0.85)
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(0.8)

    ax.set_title(
        "b1  Province-level optimal capacity at each best dispatch point",
        fontsize=7.2,
        loc="left",
        pad=3.0,
    )

    for center, tech_id in zip(tech_centers, TECH_ORDER):
        ax.text(
            center,
            -1.58,
            TECH_LABELS[tech_id],
            ha="center",
            va="bottom",
            fontsize=5.6,
            color=TECH_COLORS[tech_id],
            fontweight="bold",
            clip_on=False,
        )

    legend_sizes = [1, 10, 50, 150, 500]
    handles = [
        ax.scatter(
            [],
            [],
            s=12 + 620 * ((s / max_cap) ** 0.7),
            color=PALETTE["blue_secondary"],
            alpha=0.62,
            edgecolor=PALETTE["blue_main"],
            linewidth=0.28,
        )
        for s in legend_sizes
    ]
    labels = [f"{s:.0f} GWh" for s in legend_sizes]
    ax.legend(
        handles,
        labels,
        title="Optimal capacity",
        fontsize=4.8,
        title_fontsize=5.0,
        loc="lower left",
        bbox_to_anchor=(1.004, 0.00),
        borderpad=0.2,
        labelspacing=0.3,
        handletextpad=0.5,
    )


def plot_panel_b2(
    container_ax: plt.Axes,
    legend_ax: plt.Axes,
    best_combo: pd.DataFrame,
    province_order: list[str],
) -> None:
    container_ax.set_axis_off()
    inner = GridSpecFromSubplotSpec(
        2,
        5,
        subplot_spec=container_ax.get_subplotspec(),
        wspace=0.13,
        hspace=0.22,
    )
    province_to_y = {province: idx for idx, province in enumerate(province_order)}
    y_ticks = np.arange(len(province_order))
    duration_set = set(best_combo["duration_h"].astype(int))

    for idx, duration_h in enumerate(DURATION_ORDER):
        if duration_h not in duration_set:
            continue
        row = idx // 5
        col = idx % 5
        ax = container_ax.figure.add_subplot(inner[row, col])
        dur_rows = best_combo[best_combo["duration_h"].astype(int) == duration_h].copy()
        present_techs = [tech_id for tech_id in TECH_ORDER if tech_id in set(dur_rows["tech_id"])]
        if len(present_techs) == 1:
            offsets = {present_techs[0]: 0.0}
        else:
            offsets = {
                tech_id: offset
                for tech_id, offset in zip(
                    present_techs,
                    np.linspace(-0.31, 0.31, len(present_techs)),
                )
            }

        for y in y_ticks:
            ax.axhline(y, color="#EDEDED", lw=0.34, zorder=0)

        for tech_id in present_techs:
            tech_rows = dur_rows[dur_rows["tech_id"] == tech_id].copy()
            tech_rows["y_pos"] = tech_rows["province_en"].map(province_to_y) + offsets[tech_id]
            tech_rows = tech_rows.dropna(subset=["y_pos"]).sort_values("y_pos")

            ax.scatter(
                tech_rows["satisfy_rate_pct"],
                tech_rows["y_pos"],
                s=8.8,
                color=TECH_COLORS[tech_id],
                alpha=0.78,
                edgecolor="white",
                linewidth=0.20,
                zorder=3,
                label=TECH_LABELS[tech_id],
            )

        if duration_h == 4:
            bj = dur_rows[dur_rows["province_en"] == "Beijing"].copy()
            bj["y_pos"] = bj["province_en"].map(province_to_y) + bj["tech_id"].map(offsets)
            for bj_row in bj.itertuples(index=False):
                ax.scatter(
                    getattr(bj_row, "satisfy_rate_pct"),
                    getattr(bj_row, "y_pos"),
                    s=18,
                    marker="D",
                    color=TECH_COLORS[getattr(bj_row, "tech_id")],
                    edgecolor=PALETTE["neutral_black"],
                    linewidth=0.35,
                    zorder=6,
                )

        ax.set_xlim(0, 100)
        ax.set_ylim(len(province_order) - 0.8, -0.8)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_yticks(y_ticks)
        ax.set_title(
            f"{duration_h} h",
            fontsize=6.2,
            pad=2.0,
            color=PALETTE["neutral_black"],
            fontweight="semibold",
        )
        ax.grid(axis="x", alpha=0.20, linewidth=0.42)
        ax.tick_params(axis="both", labelsize=4.6, width=0.5, length=1.7, pad=1.0)
        if row == 1:
            ax.set_xlabel("Max supply ratio (%)", fontsize=5.0, labelpad=1.0)
        else:
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelbottom=False)
        if col == 0:
            ax.set_yticklabels(
                [PROVINCE_SHORT_LABELS.get(p, p) for p in province_order],
                fontsize=4.0,
            )
            ax.set_ylabel("Province", fontsize=5.1, labelpad=1.0)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", labelleft=False)
        for side in ["left", "bottom"]:
            ax.spines[side].set_linewidth(0.7)

    container_ax.text(
        0.0,
        1.035,
        "b2  Maximum supply ratio at each duration; each point is one province-technology pair",
        transform=container_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.0,
        color=PALETTE["neutral_black"],
    )

    legend_ax.set_axis_off()
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=TECH_COLORS[tech_id],
            markeredgecolor="white",
            markeredgewidth=0.35,
            markersize=4.2,
            label=TECH_LABELS[tech_id],
        )
        for tech_id in TECH_ORDER
    ]
    legend_ax.legend(
        handles=handles,
        loc="center",
        ncol=8,
        fontsize=5.3,
        frameon=False,
        title="Technology color key",
        title_fontsize=5.6,
        handlelength=1.0,
        columnspacing=0.95,
        handletextpad=0.30,
        labelspacing=0.25,
    )


def plot_ranked_best_outcome(container_ax: plt.Axes, best_overall: pd.DataFrame) -> tuple[plt.Axes, plt.Axes]:
    container_ax.set_axis_off()

    sorted_df = best_overall.sort_values(
        ["annual_curtailment_gwh", "satisfy_rate_pct", "capacity_mwh", "duration_h"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    ypos = np.arange(len(sorted_df))

    inner = GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=container_ax.get_subplotspec(),
        width_ratios=[0.34, 0.66],
        wspace=0.08,
    )
    ax_left = container_ax.figure.add_subplot(inner[0, 0])
    ax_right = container_ax.figure.add_subplot(inner[0, 1])

    curtailment_values = sorted_df["annual_curtailment_gwh"].to_numpy(dtype=float)
    positive = curtailment_values[curtailment_values > 0]
    norm = mcolors.LogNorm(vmin=float(positive.min()), vmax=float(positive.max()))
    stem_base = float(positive.min()) * 0.82

    for y, value in zip(ypos, curtailment_values):
        color = CURTAILMENT_CMAP(norm(value))
        ax_left.hlines(y, stem_base, value, color=color, lw=4.2, alpha=0.95, zorder=2)
        ax_left.scatter(value, y, s=18, color=color, edgecolor="white", linewidth=0.35, zorder=3)

    ax_left.set_xscale("log")
    ax_left.set_xlim(stem_base, float(positive.max()) * 1.08)
    ax_left.set_ylim(len(sorted_df) - 0.5, -0.5)
    ax_left.set_yticks(ypos)
    ax_left.set_yticklabels(sorted_df["province_en"], fontsize=5.15)
    ax_left.set_xlabel("Annual curtailment (GWh, log scale)", fontsize=7.1)
    ax_left.set_title("Annual curtailment rank", fontsize=7.6, pad=3)
    ax_left.grid(axis="x", alpha=0.18, linewidth=0.5)
    ax_left.tick_params(axis="x", labelsize=6.2, width=0.7, length=2.5)
    ax_left.tick_params(axis="y", labelsize=5.15, pad=1.0, length=0)
    for side in ["left", "bottom"]:
        ax_left.spines[side].set_linewidth(0.9)

    for tech_id in TECH_ORDER:
        sub = sorted_df.loc[sorted_df["tech_id"] == tech_id]
        if sub.empty:
            continue

        y = sub.index.to_numpy(dtype=float)
        size = 16 + sub["duration_h"].to_numpy(dtype=float) * 0.72
        ax_right.scatter(
            sub["satisfy_rate_pct"],
            y,
            s=size,
            color=TECH_COLORS[tech_id],
            alpha=0.92,
            edgecolor="white",
            linewidth=0.55,
            label=TECH_LABELS[tech_id],
            zorder=3,
        )

    ax_right.axvline(100, color=PALETTE["neutral_light"], lw=0.7, zorder=1)
    ax_right.set_xlim(14, 102)
    ax_right.set_ylim(len(sorted_df) - 0.5, -0.5)
    ax_right.set_yticks(ypos)
    ax_right.set_yticklabels([])
    ax_right.set_xlabel("Best supply ratio (%)", fontsize=7.1)
    ax_right.set_title("Best supply ratio by province", fontsize=7.6, pad=3)
    ax_right.grid(axis="x", alpha=0.18, linewidth=0.5)
    ax_right.tick_params(axis="x", labelsize=6.2, width=0.7, length=2.5)
    ax_right.tick_params(axis="y", left=False, labelleft=False, length=0)
    for side in ["left", "bottom"]:
        ax_right.spines[side].set_linewidth(0.9)

    add_duration_size_legend(ax_right)
    return ax_left, ax_right


def plot_annual_curtailment_rank(ax: plt.Axes, best_overall: pd.DataFrame) -> None:
    sorted_df = best_overall.sort_values(
        ["annual_curtailment_gwh", "satisfy_rate_pct", "capacity_mwh", "duration_h"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    ypos = np.arange(len(sorted_df))

    curtailment_values = sorted_df["annual_curtailment_gwh"].to_numpy(dtype=float)
    positive = curtailment_values[curtailment_values > 0]
    norm = mcolors.LogNorm(vmin=float(positive.min()), vmax=float(positive.max()))
    stem_base = float(positive.min()) * 0.82

    for y, value in zip(ypos, curtailment_values):
        color = CURTAILMENT_CMAP(norm(value))
        ax.hlines(y, stem_base, value, color=color, lw=4.0, alpha=0.95, zorder=2)
        ax.scatter(value, y, s=17, color=color, edgecolor="white", linewidth=0.35, zorder=3)

    ax.set_xscale("log")
    ax.set_xlim(stem_base, float(positive.max()) * 1.08)
    ax.set_ylim(len(sorted_df) - 0.5, -0.5)
    ax.set_yticks(ypos)
    ax.set_yticklabels(sorted_df["province_en"], fontsize=5.7)
    ax.set_xlabel("Annual curtailment (GWh, log scale)", fontsize=7.1)
    ax.set_title("Annual curtailment rank", fontsize=7.6, pad=3)
    ax.grid(axis="x", alpha=0.18, linewidth=0.5)
    ax.tick_params(axis="x", labelsize=6.2, width=0.7, length=2.5)
    ax.tick_params(axis="y", labelsize=5.7, pad=1.0, length=0)
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(0.9)


def plot_best_supply_by_province(ax: plt.Axes, best_overall: pd.DataFrame) -> None:
    sorted_df = best_overall.sort_values(
        ["annual_curtailment_gwh", "satisfy_rate_pct", "capacity_mwh", "duration_h"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    ypos = np.arange(len(sorted_df))

    for tech_id in TECH_ORDER:
        sub = sorted_df.loc[sorted_df["tech_id"] == tech_id]
        if sub.empty:
            continue

        y = sub.index.to_numpy(dtype=float)
        size = 16 + sub["duration_h"].to_numpy(dtype=float) * 0.72
        ax.scatter(
            sub["satisfy_rate_pct"],
            y,
            s=size,
            color=TECH_COLORS[tech_id],
            alpha=0.92,
            edgecolor="white",
            linewidth=0.55,
            label=TECH_LABELS[tech_id],
            zorder=3,
        )

    ax.axvline(100, color=PALETTE["neutral_light"], lw=0.7, zorder=1)
    ax.set_xlim(14, 102)
    ax.set_ylim(len(sorted_df) - 0.5, -0.5)
    ax.set_yticks(ypos)
    ax.set_yticklabels(sorted_df["province_en"], fontsize=5.7)
    ax.set_xlabel("Best supply ratio (%)", fontsize=7.1)
    ax.set_title("Best supply ratio by province", fontsize=7.6, pad=3)
    ax.grid(axis="x", alpha=0.18, linewidth=0.5)
    ax.tick_params(axis="x", labelsize=6.2, width=0.7, length=2.5)
    ax.tick_params(axis="y", labelsize=5.7, pad=1.0, length=0)
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(0.9)

    add_duration_size_legend(ax, sorted_df["duration_h"].unique())


def add_duration_size_legend(ax: plt.Axes, durations: np.ndarray | list[int]) -> None:
    duration_values = sorted({int(duration) for duration in durations})
    handles = [
        plt.scatter(
            [],
            [],
            s=16 + 0.72 * duration,
            color=PALETTE["neutral_mid"],
            edgecolor="white",
        )
        for duration in duration_values
    ]
    labels = [f"{duration} h" for duration in duration_values]
    leg = ax.legend(
        handles,
        labels,
        title="Selected duration",
        fontsize=5.6,
        title_fontsize=5.8,
        loc="lower right",
        frameon=False,
        handletextpad=0.4,
        borderpad=0.2,
        labelspacing=0.35,
    )
    ax.add_artist(leg)


def add_technology_legend_below(fig: plt.Figure, anchor_ax: plt.Axes) -> None:
    bbox = anchor_ax.get_position()
    legend_ax = fig.add_axes([bbox.x0 + 0.04 * bbox.width, bbox.y0 - 0.062, bbox.width * 0.88, 0.032])
    legend_ax.set_axis_off()
    handles = [
        Line2D([0], [0], color=TECH_COLORS[tech_id], lw=1.8, marker="o", markersize=4)
        for tech_id in TECH_ORDER
    ]
    labels = [TECH_LABELS[tech_id] for tech_id in TECH_ORDER]
    legend_ax.legend(
        handles,
        labels,
        loc="center",
        ncol=4,
        fontsize=6.2,
        frameon=False,
        handlelength=1.55,
        columnspacing=0.85,
        handletextpad=0.35,
        labelspacing=0.32,
    )


def add_panel_b_bottom_guides(fig: plt.Figure, anchor_ax: plt.Axes) -> None:
    bbox = anchor_ax.get_position()
    capacity_ax = fig.add_axes([bbox.x0 + 0.03 * bbox.width, bbox.y0 - 0.074, bbox.width * 0.53, 0.034])
    add_panel_b_capacity_legend(capacity_ax)

    cbar_ax = fig.add_axes([bbox.x0 + 0.64 * bbox.width, bbox.y0 - 0.061, bbox.width * 0.31, 0.012])
    sm = plt.cm.ScalarMappable(
        norm=mcolors.Normalize(vmin=0.0, vmax=100.0),
        cmap=SUPPLY_CONTRAST_CMAP,
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=5.8, width=0.45, length=1.8, pad=1.0)
    cbar.outline.set_linewidth(0.45)
    cbar.set_ticks([0, 50, 100])
    cbar.set_label("Max supply ratio (%)", fontsize=6.1, labelpad=1.8)


def main() -> pd.DataFrame:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    results, curtailment = load_inputs()
    results = enrich_capacity_curve_metrics(results)
    annual_df = build_annual_curtailment_table(results, curtailment)
    best_combo = build_best_combo_table(results, annual_df)
    summary = build_duration_summary(best_combo)
    best_overall = build_best_overall_table(best_combo)
    curve_summary = summarize_capacity_curve_metrics(results)
    save_source_data(annual_df, best_combo, summary, best_overall, results, curve_summary)

    fig = plt.figure(figsize=(16.8, 16.2))
    gs = fig.add_gridspec(
        4,
        2,
        height_ratios=[1.02, 0.96, 1.03, 1.03],
        width_ratios=[0.84, 1.20],
        wspace=0.16,
        hspace=0.42,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d1 = fig.add_subplot(gs[2, 0])
    ax_d2 = fig.add_subplot(gs[3, 0])
    ax_b = fig.add_subplot(gs[:, 1])

    plot_china_curtailment_map(ax_a, annual_df)
    ax_b1, ax_b2 = plot_panel_b(ax_b, best_combo, results)
    plot_metric_heatmap(
        ax_c,
        summary,
        metric_prefix="efc",
        cmap=EFC_CMAP,
        title="Operational intensity at best dispatch point",
        cbar_label="EFC",
        log_scale=False,
        cbar_ticks=[5, 10, 15, 20, 25],
    )
    plot_annual_curtailment_rank(ax_d1, best_overall)
    plot_best_supply_by_province(ax_d2, best_overall)

    add_panel_label(ax_a, "a")
    add_panel_label(ax_b1, "b")
    add_panel_label(ax_c, "c")
    add_panel_label(ax_d1, "d1")
    add_panel_label(ax_d2, "d2")

    fig.subplots_adjust(left=0.045, right=0.988, top=0.975, bottom=0.090)

    pos_a = ax_a.get_position()
    ax_a.set_position([pos_a.x0 - 0.020, pos_a.y0, pos_a.width, pos_a.height])
    add_technology_legend_below(fig, ax_d2)
    add_panel_b_bottom_guides(fig, ax_b)

    fig.savefig(f"{FIG_BASE}.svg", bbox_inches="tight")
    fig.savefig(f"{FIG_BASE}.pdf", bbox_inches="tight")
    fig.savefig(f"{FIG_BASE}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {FIG_BASE}.svg/pdf/png")
    print(f"Source data saved to: {SOURCE_DIR}")
    return best_overall


if __name__ == "__main__":
    main()
