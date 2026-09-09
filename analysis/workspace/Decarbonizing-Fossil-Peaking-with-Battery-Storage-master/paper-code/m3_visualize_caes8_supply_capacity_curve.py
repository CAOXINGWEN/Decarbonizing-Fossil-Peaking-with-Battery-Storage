"""
CAES 8h province-level capacity-supply curves.

The figure uses installed energy capacity on the x-axis and peaking supply
ratio on the y-axis for all 26 provinces. It is generated as a new file and
does not overwrite Fig. 3.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager
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
plt.rcParams["font.size"] = 8.4
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.95
plt.rcParams["legend.frameon"] = False


PAPER_CODE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PAPER_CODE_DIR / "output" / "m3_dispatch"
FIGURE_DIR = OUTPUT_DIR / "nature_figure"
SOURCE_DIR = FIGURE_DIR / "source_data"
RESULTS_CSV = OUTPUT_DIR / "m3_dispatch_results.csv"
FIG_BASE = FIGURE_DIR / "fig3b_caes8_capacity_supply_26provinces"


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


def load_caes8_data() -> pd.DataFrame:
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
    data = data[
        data["scenario"].eq("baseline")
        & data["tech_id"].eq("caes")
        & data["duration_h"].eq(8)
    ].copy()
    data["capacity_gwh"] = data["capacity_mwh"].astype(float) / 1000.0
    data["supply_ratio_pct"] = data["satisfy_rate_pct"].astype(float)
    data["province_short"] = data["province_cn"].map(short_province)
    data = data.sort_values(["province_cn", "supply_ratio_pct", "capacity_gwh"])
    return data


def build_summary(data: pd.DataFrame) -> pd.DataFrame:
    summary = (
        data.groupby(["province_en", "province_cn", "province_short"], as_index=False)
        .agg(
            point_count=("capacity_gwh", "size"),
            min_capacity_gwh=("capacity_gwh", "min"),
            max_capacity_gwh=("capacity_gwh", "max"),
            min_supply_ratio_pct=("supply_ratio_pct", "min"),
            max_supply_ratio_pct=("supply_ratio_pct", "max"),
            total_demand_gwh=("total_demand_kwh", lambda x: x.iloc[0] / 1_000_000.0),
        )
        .sort_values(["max_supply_ratio_pct", "max_capacity_gwh"], ascending=[False, False])
    )
    return summary


def downsample_curve(group: pd.DataFrame, max_points: int = 260) -> pd.DataFrame:
    group = group.sort_values("supply_ratio_pct")
    if len(group) <= max_points:
        return group
    keep = np.unique(np.linspace(0, len(group) - 1, max_points).round().astype(int))
    return group.iloc[keep].copy()


def export_source_data(data: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    sampled = pd.concat(
        [
            downsample_curve(group, max_points=260)
            for _province, group in data.groupby("province_cn", sort=False)
        ],
        ignore_index=True,
    )
    data.to_csv(SOURCE_DIR / "fig3b_source_caes8_capacity_supply_full.csv", index=False)
    sampled.to_csv(SOURCE_DIR / "fig3b_source_caes8_capacity_supply_sampled.csv", index=False)
    summary.to_csv(SOURCE_DIR / "fig3b_source_caes8_capacity_supply_summary.csv", index=False)
    return sampled


def label_position(group: pd.DataFrame) -> tuple[float, float]:
    group = group.sort_values("capacity_gwh")
    row = group.iloc[-1]
    return float(row["capacity_gwh"]), float(row["supply_ratio_pct"])


def make_figure() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    data = load_caes8_data()
    summary = build_summary(data)
    sampled = export_source_data(data, summary)

    order = summary.sort_values("max_capacity_gwh", ascending=False)["province_cn"].tolist()
    palette = plt.get_cmap("tab20", len(order))
    province_colors = {province: palette(i) for i, province in enumerate(order)}

    fig, ax = plt.subplots(figsize=(9.4, 7.4))

    for province in order:
        group = sampled[sampled["province_cn"].eq(province)].sort_values("capacity_gwh")
        color = province_colors[province]
        lw = 1.15
        alpha = 0.92
        zorder = 3
        ax.plot(
            group["capacity_gwh"],
            group["supply_ratio_pct"],
            color=color,
            lw=lw,
            alpha=alpha,
            zorder=zorder,
        )

    ax.set_xscale("log")
    ax.set_xlim(0.02, 260)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Energy capacity (GWh, log scale)", fontsize=8.3, labelpad=3)
    ax.set_ylabel("Peaking supply ratio (%)", fontsize=8.3, labelpad=3)
    ax.set_xticks([0.03, 0.1, 1, 10, 100])
    ax.set_xticklabels(["0.03", "0.1", "1", "10", "100"])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="both", labelsize=7.4, length=3)
    ax.grid(axis="both", which="major", color="#EAEAEA", lw=0.75)

    fig.suptitle(
        "CAES 8h: capacity-supply curves across 26 provinces",
        x=0.085,
        y=0.975,
        ha="left",
        fontsize=12.4,
        fontweight="bold",
    )
    fig.text(
        0.085,
        0.935,
        "Each colored line is one province. Province identities are intended to be matched through the shared figure legend in the integrated layout.",
        ha="left",
        va="top",
        fontsize=7.4,
        color="#555555",
    )

    note = (
        "Max supply ratio: Beijing 11.4%, Jiangsu 47.6%, "
        "Henan 90.1%; most other provinces approach 100%."
    )
    ax.text(
        0.02,
        0.04,
        note,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.9,
        color="#555555",
    )

    fig.subplots_adjust(left=0.085, right=0.84, top=0.88, bottom=0.105)

    for ext, kwargs in {
        ".svg": {},
        ".pdf": {},
        ".png": {"dpi": 340},
    }.items():
        fig.savefig(FIG_BASE.with_suffix(ext), bbox_inches="tight", **kwargs)

    plt.close(fig)
    print(f"Saved CAES 8h capacity-supply figure to {FIG_BASE.with_suffix('.png')}")
    print(f"Source data exported to {SOURCE_DIR}")


if __name__ == "__main__":
    make_figure()
