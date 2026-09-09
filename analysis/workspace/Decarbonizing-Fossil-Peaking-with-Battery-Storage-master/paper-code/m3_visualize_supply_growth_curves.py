"""
Nature-style supplementary figure for M3 maximum supply-ratio growth curves.

For each province, technology and available duration, the script keeps the
maximum peaking supply ratio across the capacity-search grid. It then computes
two growth metrics:
  - cumulative relative growth versus the shortest available duration of that
    technology, shown in the figure;
  - adjacent-duration growth, exported as source data for alternative plotting.

The figure is generated as a new file and does not overwrite Fig. 3.
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
EXISTING_BEST_CSV = SOURCE_DIR / "fig3_source_best_by_tech_province_duration.csv"
FIG_BASE = FIGURE_DIR / "fig3b_max_supply_growth_curves"


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

TECH_COLORS = {
    "li_lfp": "#2F78A8",
    "vrfb": "#5A94C2",
    "fecr_flow": "#3E8B72",
    "gravity": "#C77C2E",
    "molten_salt_tes": "#777777",
    "caes": "#68A95F",
    "laes": "#9A83C8",
    "h2_cavern": "#6E64B7",
}

HIGHLIGHT_COLORS = ["#AD3A3E", "#2F78A8", "#3E8B72", "#C77C2E"]


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


def load_best_supply_points() -> pd.DataFrame:
    if EXISTING_BEST_CSV.exists():
        best = pd.read_csv(EXISTING_BEST_CSV)
    else:
        if not RESULTS_CSV.exists():
            raise FileNotFoundError(f"Missing M3 dispatch result: {RESULTS_CSV}")
        cols = [
            "tech_id",
            "technology_cn",
            "scenario",
            "duration_h",
            "rte",
            "province_en",
            "province_cn",
            "power_mw",
            "capacity_mwh",
            "total_demand_kwh",
            "total_supply_kwh",
            "total_charge_kwh",
            "total_spillage_kwh",
            "satisfy_rate_pct",
            "efc",
        ]
        results = pd.read_csv(RESULTS_CSV, usecols=cols)
        idx = results.groupby(["tech_id", "province_en", "duration_h"])["satisfy_rate_pct"].idxmax()
        best = results.loc[idx].copy()

    if "scenario" in best.columns:
        best = best[best["scenario"].eq("baseline")].copy()

    needed = [
        "tech_id",
        "technology_cn",
        "province_en",
        "province_cn",
        "duration_h",
        "capacity_mwh",
        "satisfy_rate_pct",
        "total_supply_kwh",
        "total_demand_kwh",
    ]
    missing = [col for col in needed if col not in best.columns]
    if missing:
        raise ValueError(f"Missing required columns in best supply table: {missing}")

    best = best[needed].copy()
    best["duration_h"] = best["duration_h"].astype(int)
    best["max_supply_ratio_pct"] = best["satisfy_rate_pct"].astype(float)
    return best


def compute_growth(best: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in best.sort_values("duration_h").groupby(["tech_id", "province_en"], sort=False):
        tech_id, province_en = keys
        group = group.sort_values("duration_h").copy()
        base_row = group.iloc[0]
        baseline_supply = float(base_row["max_supply_ratio_pct"])
        baseline_duration = int(base_row["duration_h"])
        previous_supply = np.nan
        previous_duration = np.nan

        for _, row in group.iterrows():
            current_supply = float(row["max_supply_ratio_pct"])
            cumulative_gain_pp = current_supply - baseline_supply
            relative_growth_pct = (
                cumulative_gain_pp / baseline_supply * 100.0
                if baseline_supply > 0
                else np.nan
            )
            adjacent_gain_pp = (
                current_supply - previous_supply
                if np.isfinite(previous_supply)
                else 0.0
            )
            adjacent_growth_pct = (
                adjacent_gain_pp / previous_supply * 100.0
                if np.isfinite(previous_supply) and previous_supply > 0
                else 0.0
            )

            out = row.to_dict()
            out.update(
                {
                    "baseline_duration_h": baseline_duration,
                    "baseline_max_supply_ratio_pct": baseline_supply,
                    "previous_duration_h": previous_duration,
                    "previous_max_supply_ratio_pct": previous_supply,
                    "cumulative_gain_pp": cumulative_gain_pp,
                    "relative_growth_pct": relative_growth_pct,
                    "adjacent_gain_pp": adjacent_gain_pp,
                    "adjacent_growth_pct": adjacent_growth_pct,
                }
            )
            rows.append(out)
            previous_supply = current_supply
            previous_duration = int(row["duration_h"])

    growth = pd.DataFrame(rows)
    growth["tech_label"] = growth["tech_id"].map(TECH_LABELS)
    growth["province_short"] = growth["province_cn"].map(short_province)
    return growth


def build_summary(growth: pd.DataFrame) -> pd.DataFrame:
    summary = (
        growth.groupby(["tech_id", "tech_label", "duration_h"], as_index=False)
        .agg(
            province_count=("province_en", "nunique"),
            max_supply_ratio_median_pct=("max_supply_ratio_pct", "median"),
            relative_growth_median_pct=("relative_growth_pct", "median"),
            relative_growth_q25_pct=("relative_growth_pct", lambda x: x.quantile(0.25)),
            relative_growth_q75_pct=("relative_growth_pct", lambda x: x.quantile(0.75)),
            relative_growth_max_pct=("relative_growth_pct", "max"),
            cumulative_gain_median_pp=("cumulative_gain_pp", "median"),
            cumulative_gain_max_pp=("cumulative_gain_pp", "max"),
            adjacent_growth_median_pct=("adjacent_growth_pct", "median"),
            adjacent_growth_max_pct=("adjacent_growth_pct", "max"),
        )
        .sort_values(["tech_id", "duration_h"])
    )
    return summary


def export_source_data(growth: pd.DataFrame, summary: pd.DataFrame) -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    growth.to_csv(
        SOURCE_DIR / "fig3b_source_max_supply_growth_curve_points.csv",
        index=False,
    )
    summary.to_csv(
        SOURCE_DIR / "fig3b_source_max_supply_growth_curve_summary.csv",
        index=False,
    )


def draw_growth_panel(
    ax: plt.Axes,
    growth: pd.DataFrame,
    tech_id: str,
    global_ymin: float,
    global_ymax: float,
) -> None:
    tech_rows = growth[growth["tech_id"].eq(tech_id)].copy()
    durations = sorted(tech_rows["duration_h"].dropna().astype(int).unique())
    x_pos = {duration: i for i, duration in enumerate(durations)}
    color = TECH_COLORS.get(tech_id, "#555555")

    ax.axhline(0, color="#333333", lw=0.65, zorder=0)
    ax.grid(axis="y", color="#EDEDED", lw=0.7, zorder=0)

    for _, province_rows in tech_rows.groupby("province_en", sort=False):
        province_rows = province_rows.sort_values("duration_h")
        x = [x_pos[int(v)] for v in province_rows["duration_h"]]
        y = province_rows["relative_growth_pct"].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            color="#B8B8B8",
            lw=0.75,
            alpha=0.48,
            marker="o",
            markersize=2.0,
            markerfacecolor="white",
            markeredgewidth=0.35,
            zorder=1,
        )

    summary = (
        tech_rows.groupby("duration_h", as_index=False)
        .agg(
            median=("relative_growth_pct", "median"),
            q25=("relative_growth_pct", lambda x: x.quantile(0.25)),
            q75=("relative_growth_pct", lambda x: x.quantile(0.75)),
        )
        .sort_values("duration_h")
    )
    x_summary = [x_pos[int(v)] for v in summary["duration_h"]]
    ax.fill_between(
        x_summary,
        summary["q25"],
        summary["q75"],
        color=color,
        alpha=0.12,
        linewidth=0,
        zorder=2,
    )
    ax.plot(
        x_summary,
        summary["median"],
        color=color,
        lw=1.9,
        marker="o",
        markersize=3.4,
        zorder=4,
    )

    province_rank = (
        tech_rows.groupby(["province_en", "province_cn", "province_short"], as_index=False)
        .agg(max_growth=("relative_growth_pct", "max"), max_gain_pp=("cumulative_gain_pp", "max"))
        .sort_values(["max_growth", "max_gain_pp"], ascending=False)
    )
    highlight = province_rank[province_rank["max_growth"] > 0.02].head(4)

    for rank, (_, province_info) in enumerate(highlight.iterrows()):
        province_rows = tech_rows[tech_rows["province_en"].eq(province_info["province_en"])].sort_values("duration_h")
        x = [x_pos[int(v)] for v in province_rows["duration_h"]]
        y = province_rows["relative_growth_pct"].to_numpy(dtype=float)
        h_color = HIGHLIGHT_COLORS[rank % len(HIGHLIGHT_COLORS)]
        ax.plot(
            x,
            y,
            color=h_color,
            lw=1.45,
            marker="o",
            markersize=3.0,
            zorder=5,
        )
        max_idx = int(np.nanargmax(y))
        ax.text(
            x[max_idx] + 0.06,
            y[max_idx],
            f"{province_info['province_short']} {province_info['max_growth']:.2f}%",
            fontsize=5.8,
            color=h_color,
            ha="left",
            va="center",
        )

    decline_rank = (
        tech_rows.groupby(["province_en", "province_cn", "province_short"], as_index=False)
        .agg(min_growth=("relative_growth_pct", "min"), min_gain_pp=("cumulative_gain_pp", "min"))
        .sort_values(["min_growth", "min_gain_pp"], ascending=True)
    )
    decline = decline_rank[decline_rank["min_growth"] < -0.02].head(2)
    for _, province_info in decline.iterrows():
        province_rows = tech_rows[tech_rows["province_en"].eq(province_info["province_en"])].sort_values("duration_h")
        x = [x_pos[int(v)] for v in province_rows["duration_h"]]
        y = province_rows["relative_growth_pct"].to_numpy(dtype=float)
        h_color = "#555555"
        ax.plot(
            x,
            y,
            color=h_color,
            lw=1.3,
            linestyle="--",
            marker="o",
            markersize=2.8,
            zorder=5,
        )
        min_idx = int(np.nanargmin(y))
        ax.text(
            x[min_idx] + 0.06,
            y[min_idx],
            f"{province_info['province_short']} {province_info['min_growth']:.2f}%",
            fontsize=5.8,
            color=h_color,
            ha="left",
            va="center",
        )

    if province_rank["max_growth"].max() <= 0.02 and decline_rank["min_growth"].min() >= -0.02:
        ax.text(
            0.03,
            0.84,
            "all provinces ~0%",
            transform=ax.transAxes,
            fontsize=6.2,
            color="#777777",
            ha="left",
            va="center",
        )

    ax.set_title(TECH_LABELS[tech_id], fontsize=9.2, pad=3.0, color="#252525")
    ax.set_xlim(-0.28, len(durations) - 0.72)
    ax.set_ylim(global_ymin, global_ymax)
    ax.set_xticks(range(len(durations)))
    ax.set_xticklabels([str(v) for v in durations], fontsize=6.8)
    ax.tick_params(axis="y", labelsize=6.8, length=2.5)
    ax.tick_params(axis="x", length=2.5)


def make_figure() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    best = load_best_supply_points()
    growth = compute_growth(best)
    summary = build_summary(growth)
    export_source_data(growth, summary)

    growth_values = growth["relative_growth_pct"].to_numpy(dtype=float)
    max_growth = float(np.nanmax(growth_values))
    min_growth = float(np.nanmin(growth_values))
    global_ymax = max(0.25, max_growth * 1.18)
    global_ymin = min(-0.10, min_growth * 1.25)

    fig, axes = plt.subplots(
        nrows=4,
        ncols=2,
        figsize=(10.8, 12.4),
        sharey=True,
        constrained_layout=False,
    )
    axes = axes.ravel()

    for ax, tech_id in zip(axes, TECH_ORDER):
        draw_growth_panel(ax, growth, tech_id, global_ymin, global_ymax)

    for i, ax in enumerate(axes):
        if i % 2 == 0:
            ax.set_ylabel("Growth from shortest duration (%)", fontsize=7.5)
        if i >= 6:
            ax.set_xlabel("Rated duration (h)", fontsize=7.5, labelpad=2.0)

    legend_handles = [
        Line2D([0], [0], color="#B8B8B8", lw=1.1, marker="o", markersize=3, label="Province curve"),
        Line2D([0], [0], color="#2F78A8", lw=1.9, marker="o", markersize=4, label="Province median"),
        Line2D([0], [0], color=HIGHLIGHT_COLORS[0], lw=1.5, marker="o", markersize=4, label="Largest growth province"),
        Line2D([0], [0], color="#555555", lw=1.3, linestyle="--", marker="o", markersize=3, label="Largest decline province"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.52, 0.025),
        fontsize=7.0,
        handlelength=1.8,
        columnspacing=1.2,
    )

    fig.suptitle(
        "Province-level growth curves of maximum peaking supply ratio",
        x=0.07,
        y=0.985,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.07,
        0.963,
        "Growth is computed relative to each technology's shortest available duration; all province-level curves are shown.",
        ha="left",
        va="top",
        fontsize=7.4,
        color="#555555",
    )

    fig.subplots_adjust(left=0.075, right=0.985, top=0.925, bottom=0.075, hspace=0.36, wspace=0.18)

    for ext, kwargs in {
        ".svg": {},
        ".pdf": {},
        ".png": {"dpi": 320},
    }.items():
        fig.savefig(FIG_BASE.with_suffix(ext), bbox_inches="tight", **kwargs)

    plt.close(fig)
    print(f"Saved supply-growth figure to {FIG_BASE.with_suffix('.png')}")
    print(f"Source data exported to {SOURCE_DIR}")


if __name__ == "__main__":
    make_figure()
