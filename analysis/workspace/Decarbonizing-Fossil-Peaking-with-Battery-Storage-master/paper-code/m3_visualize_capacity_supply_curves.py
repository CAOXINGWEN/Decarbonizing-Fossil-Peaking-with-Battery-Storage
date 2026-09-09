"""
Nature-style supplementary figure for M3 capacity-supply curves.

This script plots the full M3 capacity-search trajectories with installed
energy capacity on the x-axis and peaking supply ratio on the y-axis. It creates
a new figure and does not overwrite Fig. 3.
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


# ---------------------------------------------------------------------------
# Nature-style defaults
# ---------------------------------------------------------------------------

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
FIG_BASE = FIGURE_DIR / "fig3b_capacity_supply_curves"


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

DURATION_ORDER = [4, 6, 8, 10, 12, 16, 24, 48, 72, 168]
DURATION_COLORS = {
    4: "#2F78A8",
    6: "#5A94C2",
    8: "#3E8B72",
    10: "#7BAE69",
    12: "#C77C2E",
    16: "#D7A443",
    24: "#AD3A3E",
    48: "#9A83C8",
    72: "#6E64B7",
    168: "#4B3E87",
}

XTICKS_GWH = [0.03, 0.1, 1, 10, 100, 500]
XTICK_LABELS = ["0.03", "0.1", "1", "10", "100", "500"]


def load_curve_data() -> pd.DataFrame:
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"Missing M3 dispatch result: {RESULTS_CSV}")

    cols = [
        "tech_id",
        "technology_cn",
        "scenario",
        "duration_h",
        "province_en",
        "province_cn",
        "capacity_mwh",
        "satisfy_rate_pct",
        "total_supply_kwh",
        "total_demand_kwh",
    ]
    data = pd.read_csv(RESULTS_CSV, usecols=cols)
    data = data[data["scenario"].eq("baseline")].copy()
    data["duration_h"] = data["duration_h"].astype(int)
    data["capacity_mwh"] = data["capacity_mwh"].astype(float)
    data["capacity_gwh"] = data["capacity_mwh"] / 1000.0
    data["supply_ratio_pct"] = data["satisfy_rate_pct"].astype(float)
    data = data[data["capacity_gwh"] > 0].copy()
    return data


def downsample_curve(group: pd.DataFrame, max_points: int = 90) -> pd.DataFrame:
    group = group.sort_values("capacity_gwh")
    if len(group) <= max_points:
        return group
    keep = np.unique(np.linspace(0, len(group) - 1, max_points).round().astype(int))
    return group.iloc[keep].copy()


def build_plot_points(data: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for _keys, group in data.groupby(["tech_id", "province_en", "duration_h"], sort=False):
        pieces.append(downsample_curve(group, max_points=85))
    return pd.concat(pieces, ignore_index=True)


def build_median_curves(data: pd.DataFrame) -> pd.DataFrame:
    summary = (
        data.groupby(["tech_id", "duration_h", "capacity_mwh"], as_index=False)
        .agg(
            province_count=("province_en", "nunique"),
            capacity_gwh=("capacity_gwh", "first"),
            supply_median_pct=("supply_ratio_pct", "median"),
            supply_q25_pct=("supply_ratio_pct", lambda x: x.quantile(0.25)),
            supply_q75_pct=("supply_ratio_pct", lambda x: x.quantile(0.75)),
        )
        .sort_values(["tech_id", "duration_h", "capacity_gwh"])
    )
    return summary


def build_plot_medians(summary: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for _keys, group in summary.groupby(["tech_id", "duration_h"], sort=False):
        typical = group[group["province_count"] >= 5].copy()
        if typical.empty:
            typical = group.copy()
        pieces.append(downsample_curve(typical, max_points=120))
    return pd.concat(pieces, ignore_index=True)


def export_source_data(plot_points: pd.DataFrame, median_curves: pd.DataFrame) -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    plot_points.to_csv(
        SOURCE_DIR / "fig3b_source_capacity_supply_curve_points_sampled.csv",
        index=False,
    )
    median_curves.to_csv(
        SOURCE_DIR / "fig3b_source_capacity_supply_curve_medians.csv",
        index=False,
    )


def draw_tech_panel(
    ax: plt.Axes,
    tech_id: str,
    plot_points: pd.DataFrame,
    plot_medians: pd.DataFrame,
) -> None:
    tech_points = plot_points[plot_points["tech_id"].eq(tech_id)].copy()
    tech_medians = plot_medians[plot_medians["tech_id"].eq(tech_id)].copy()
    durations = sorted(tech_points["duration_h"].dropna().astype(int).unique())

    for _keys, group in tech_points.groupby(["province_en", "duration_h"], sort=False):
        group = group.sort_values("capacity_gwh")
        ax.plot(
            group["capacity_gwh"],
            group["supply_ratio_pct"],
            color="#AFAFAF",
            lw=0.45,
            alpha=0.18,
            zorder=1,
        )

    for duration in durations:
        group = tech_medians[tech_medians["duration_h"].eq(duration)].sort_values("capacity_gwh")
        if group.empty:
            continue
        color = DURATION_COLORS.get(duration, "#555555")
        ax.plot(
            group["capacity_gwh"],
            group["supply_median_pct"],
            color=color,
            lw=1.8,
            alpha=0.96,
            zorder=4,
        )
        ax.fill_between(
            group["capacity_gwh"],
            group["supply_q25_pct"],
            group["supply_q75_pct"],
            color=color,
            alpha=0.075,
            linewidth=0,
            zorder=2,
        )

    ax.set_title(TECH_LABELS[tech_id], fontsize=9.4, pad=3.0)
    ax.set_xscale("log")
    ax.set_xlim(0.02, 650)
    ax.set_ylim(-2, 102)
    ax.set_xticks(XTICKS_GWH)
    ax.set_xticklabels(XTICK_LABELS, fontsize=6.8)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="y", labelsize=6.8, length=2.5)
    ax.tick_params(axis="x", length=2.5)
    ax.grid(axis="both", which="major", color="#EDEDED", lw=0.65, zorder=0)

    duration_text = ", ".join(f"{d}h" for d in durations)
    ax.text(
        0.03,
        0.92,
        duration_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.9,
        color="#666666",
    )


def make_figure() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    data = load_curve_data()
    plot_points = build_plot_points(data)
    median_curves = build_median_curves(data)
    plot_medians = build_plot_medians(median_curves)
    export_source_data(plot_points, median_curves)

    fig, axes = plt.subplots(
        nrows=4,
        ncols=2,
        figsize=(11.2, 12.7),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    axes = axes.ravel()

    for ax, tech_id in zip(axes, TECH_ORDER):
        draw_tech_panel(ax, tech_id, plot_points, plot_medians)

    for i, ax in enumerate(axes):
        if i % 2 == 0:
            ax.set_ylabel("Peaking supply ratio (%)", fontsize=7.6)
        if i >= 6:
            ax.set_xlabel("Energy capacity (GWh, log scale)", fontsize=7.6, labelpad=2.0)

    duration_handles = [
        Line2D(
            [0],
            [0],
            color=DURATION_COLORS[duration],
            lw=1.8,
            label=f"{duration}h median",
        )
        for duration in DURATION_ORDER
    ]
    province_handle = Line2D(
        [0],
        [0],
        color="#AFAFAF",
        lw=1.0,
        alpha=0.55,
        label="Province-duration curves",
    )
    fig.legend(
        handles=[province_handle] + duration_handles,
        loc="lower center",
        ncol=6,
        bbox_to_anchor=(0.52, 0.018),
        fontsize=6.7,
        handlelength=1.8,
        columnspacing=1.0,
    )

    fig.suptitle(
        "Capacity-supply curves by technology and province",
        x=0.07,
        y=0.986,
        ha="left",
        fontsize=12.2,
        fontweight="bold",
    )
    fig.text(
        0.07,
        0.964,
        "Thin grey lines are province-duration trajectories; colored lines show the province-median supply curve for each rated duration.",
        ha="left",
        va="top",
        fontsize=7.4,
        color="#555555",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.925, bottom=0.095, hspace=0.36, wspace=0.16)

    for ext, kwargs in {
        ".svg": {},
        ".pdf": {},
        ".png": {"dpi": 320},
    }.items():
        fig.savefig(FIG_BASE.with_suffix(ext), bbox_inches="tight", **kwargs)

    plt.close(fig)
    print(f"Saved capacity-supply figure to {FIG_BASE.with_suffix('.png')}")
    print(f"Source data exported to {SOURCE_DIR}")


if __name__ == "__main__":
    make_figure()
