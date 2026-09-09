"""
Integrated CAES 8h province-level figure.

Panels:
  a. Capacity-supply curves for 26 provinces
  d. Marginal-gain curves
  e. Capacity-efficiency curves
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CJK_FONT_PATHS = [
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
]
for font_path in CJK_FONT_PATHS:
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Arial",
    "Noto Sans CJK JP",
    "WenQuanYi Zen Hei",
    "DejaVu Sans",
    "Liberation Sans",
]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.size"] = 8.2
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.9
plt.rcParams["legend.frameon"] = False


PAPER_CODE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PAPER_CODE_DIR / "output" / "m3_dispatch"
FIGURE_DIR = OUTPUT_DIR / "nature_figure"
SOURCE_DIR = FIGURE_DIR / "source_data"

RESULTS_CSV = OUTPUT_DIR / "m3_dispatch_results.csv"
CURVE_SUMMARY_CSV = SOURCE_DIR / "fig3_source_capacity_curve_metrics_summary.csv"
CURVE_POINTS_CSV = SOURCE_DIR / "fig3_source_capacity_curve_metrics_points.csv"

FIG_BASE = FIGURE_DIR / "fig3b_caes8_curve_plus_metrics"


def short_province(name: str) -> str:
    special = {
        "内蒙古自治区": "内蒙古",
        "广西壮族自治区": "广西",
        "宁夏回族自治区": "宁夏",
        "新疆维吾尔自治区": "新疆",
    }
    if name in special:
        return special[name]
    return str(name).replace("省", "").replace("市", "")


def downsample_curve(group: pd.DataFrame, max_points: int = 240) -> pd.DataFrame:
    group = group.sort_values("capacity_gwh")
    if len(group) <= max_points:
        return group
    keep = np.unique(np.linspace(0, len(group) - 1, max_points).round().astype(int))
    return group.iloc[keep].copy()


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(CURVE_SUMMARY_CSV)
    points = pd.read_csv(
        CURVE_POINTS_CSV,
        usecols=[
            "tech_id",
            "duration_h",
            "province_cn",
            "capacity_gwh",
            "satisfy_rate_pct",
            "marginal_supply_gain_pct_per_gwh",
            "capacity_efficiency_pct_per_gwh",
        ],
    )
    summary = summary[(summary["tech_id"] == "caes") & (summary["duration_h"] == 8)].copy()
    points = points[(points["tech_id"] == "caes") & (points["duration_h"] == 8)].copy()
    summary["province_short"] = summary["province_cn"].map(short_province)
    points["province_short"] = points["province_cn"].map(short_province)
    points = pd.concat(
        [downsample_curve(group, max_points=220) for _province, group in points.groupby("province_cn", sort=False)],
        ignore_index=True,
    )
    return summary, points


def draw_curve_panel(
    ax: plt.Axes,
    points: pd.DataFrame,
    order: list[str],
    color_map: dict[str, tuple[float, float, float, float]],
    value_col: str,
    title: str,
    ylabel: str,
    ylog: bool = False,
) -> None:
    for province in order:
        group = points[points["province_cn"] == province].sort_values("capacity_gwh")
        group = group[np.isfinite(group[value_col])]
        if group.empty:
            continue
        ax.plot(
            group["capacity_gwh"],
            group[value_col],
            color=color_map[province],
            lw=1.0,
            alpha=0.78,
            zorder=2,
        )

    ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    ax.set_xlim(0.02, 260)
    ax.set_xticks([0.03, 0.1, 1, 10, 100])
    ax.set_xticklabels(["0.03", "0.1", "1", "10", "100"])
    ax.set_title(title, fontsize=8.9, pad=4, loc="left")
    ax.set_xlabel("Capacity (GWh, log scale)", fontsize=7.2, labelpad=2)
    ax.set_ylabel(ylabel, fontsize=7.2, labelpad=2)
    ax.grid(axis="both", which="major", color="#ECECEC", lw=0.65)
    ax.tick_params(axis="both", labelsize=6.6, length=2.5)


def make_figure() -> None:
    summary, points = load_inputs()
    province_order = (
        summary.sort_values("c95_capacity_mwh", ascending=False)["province_cn"].tolist()
    )
    palette = plt.get_cmap("tab20", len(province_order))
    color_map = {province: palette(i) for i, province in enumerate(province_order)}
    summary["color"] = summary["province_cn"].map(color_map)

    fig = plt.figure(figsize=(13.8, 9.0))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.56, 1.0],
        height_ratios=[1.0, 1.0],
        left=0.065,
        right=0.985,
        top=0.90,
        bottom=0.13,
        wspace=0.22,
        hspace=0.30,
    )

    ax_curve = fig.add_subplot(gs[:, 0])
    ax_gain = fig.add_subplot(gs[0, 1])
    ax_eff = fig.add_subplot(gs[1, 1])

    for province in province_order:
        group = points[points["province_cn"] == province].sort_values("capacity_gwh")
        ax_curve.plot(
            group["capacity_gwh"],
            group["satisfy_rate_pct"],
            color=color_map[province],
            lw=1.1,
            alpha=0.92,
        )

    ax_curve.set_xscale("log")
    ax_curve.set_xlim(0.02, 260)
    ax_curve.set_ylim(0, 105)
    ax_curve.set_xticks([0.03, 0.1, 1, 10, 100])
    ax_curve.set_xticklabels(["0.03", "0.1", "1", "10", "100"])
    ax_curve.set_yticks([0, 20, 40, 60, 80, 100])
    ax_curve.set_xlabel("Energy capacity (GWh, log scale)", fontsize=7.6)
    ax_curve.set_ylabel("Peaking supply ratio (%)", fontsize=7.6)
    ax_curve.set_title("a  Capacity-supply curves", fontsize=9.4, pad=4, loc="left")
    ax_curve.grid(axis="both", which="major", color="#EAEAEA", lw=0.72)
    draw_curve_panel(
        ax_gain,
        points,
        province_order,
        color_map,
        "marginal_supply_gain_pct_per_gwh",
        "d  Marginal-gain curves",
        "pct-pt / GWh (log scale)",
        ylog=True,
    )
    draw_curve_panel(
        ax_eff,
        points,
        province_order,
        color_map,
        "capacity_efficiency_pct_per_gwh",
        "e  Capacity-efficiency curves",
        "Supply ratio per GWh (log scale)",
        ylog=True,
    )

    handles = [
        Line2D(
            [0],
            [0],
            color=color_map[province],
            lw=2.0,
            label=short_province(province),
        )
        for province in province_order
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.015),
        ncol=9,
        fontsize=6.2,
        handlelength=1.6,
        columnspacing=0.9,
    )

    fig.suptitle(
        "CAES 8h across 26 provinces: saturation curve and engineering capacity metrics",
        x=0.065,
        y=0.975,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.065,
        0.946,
        "Panel a shows province-specific capacity-supply trajectories; panels d and e show how marginal return and capacity efficiency decay as CAES 8h is oversized.",
        ha="left",
        va="top",
        fontsize=7.2,
        color="#555555",
    )

    for ext, kwargs in {
        ".svg": {},
        ".pdf": {},
        ".png": {"dpi": 320},
    }.items():
        fig.savefig(FIG_BASE.with_suffix(ext), bbox_inches="tight", **kwargs)
    plt.close(fig)
    print(f"Saved integrated CAES 8h metrics figure to {FIG_BASE.with_suffix('.png')}")


if __name__ == "__main__":
    make_figure()
