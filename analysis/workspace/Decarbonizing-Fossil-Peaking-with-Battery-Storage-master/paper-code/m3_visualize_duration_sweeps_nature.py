"""
Nature-style duration-sweep figures for all M3 technologies.

Outputs:
  - One stacked Nx3 figure per technology, where N is the number of available durations.
  - One CAES-stem compatibility export matching the previous CAES filename.
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

from m3_visualize_nature import enrich_capacity_curve_metrics

PAPER_CODE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PAPER_CODE_DIR / "output" / "m3_dispatch"
FIGURE_DIR = OUTPUT_DIR / "nature_figure"
RESULTS_CSV = OUTPUT_DIR / "m3_dispatch_results.csv"

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

TECH_OUTPUT_STEMS = {
    "li_lfp": "fig3b_lfp_duration_sweep_curve_plus_metrics_nature_5x3",
    "vrfb": "fig3b_vrfb_duration_sweep_curve_plus_metrics_nature_5x3",
    "fecr_flow": "fig3b_fecr_flow_duration_sweep_curve_plus_metrics_nature_5x3",
    "gravity": "fig3b_gravity_duration_sweep_curve_plus_metrics_nature_5x3",
    "molten_salt_tes": "fig3b_molten_salt_tes_duration_sweep_curve_plus_metrics_nature_5x3",
    "caes": "fig3b_caes_duration_sweep_curve_plus_metrics_nature_5x3",
    "laes": "fig3b_laes_duration_sweep_curve_plus_metrics_nature_5x3",
    "h2_cavern": "fig3b_h2_cavern_duration_sweep_curve_plus_metrics_nature_5x3",
}

PANEL_COLS = [
    ("satisfy_rate_pct", "Peaking supply ratio (%)", False, (0, 105)),
    ("marginal_supply_gain_pct_per_gwh", "Marginal gain (pct-pt GWh$^{-1}$)", True, (0.015, 700)),
    ("capacity_efficiency_pct_per_gwh", "Capacity efficiency (% GWh$^{-1}$)", True, (0.2, 900)),
]

PANEL_TITLES = [
    "Capacity-supply curve",
    "Marginal return",
    "Capacity efficiency",
]


def apply_publication_style() -> None:
    cjk_font_paths = [
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    ]
    for font_path in cjk_font_paths:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Arial",
        "DejaVu Sans",
        "Liberation Sans",
        "Noto Sans CJK JP",
        "WenQuanYi Zen Hei",
    ]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.size"] = 7.4
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 0.75
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["xtick.major.width"] = 0.75
    plt.rcParams["ytick.major.width"] = 0.75
    plt.rcParams["xtick.major.size"] = 2.5
    plt.rcParams["ytick.major.size"] = 2.5


def short_province(name: str) -> str:
    special = {
        "安徽省": "Anhui",
        "北京市": "Beijing",
        "福建省": "Fujian",
        "甘肃省": "Gansu",
        "广东省": "Guangdong",
        "内蒙古自治区": "Inner Mongolia",
        "贵州省": "Guizhou",
        "广西壮族自治区": "Guangxi",
        "河北省": "Hebei",
        "河南省": "Henan",
        "湖北省": "Hubei",
        "湖南省": "Hunan",
        "吉林省": "Jilin",
        "江苏省": "Jiangsu",
        "江西省": "Jiangxi",
        "辽宁省": "Liaoning",
        "宁夏回族自治区": "Ningxia",
        "青海省": "Qinghai",
        "陕西省": "Shaanxi",
        "山东省": "Shandong",
        "山西省": "Shanxi",
        "四川省": "Sichuan",
        "天津市": "Tianjin",
        "新疆维吾尔自治区": "Xinjiang",
        "云南省": "Yunnan",
        "浙江省": "Zhejiang",
    }
    if name in special:
        return special[name]
    return str(name).replace("省", "").replace("市", "")


def load_curve_points() -> pd.DataFrame:
    data = pd.read_csv(RESULTS_CSV)
    if "scenario" in data.columns:
        data = data[data["scenario"].eq("baseline")].copy()
    data = data[data["tech_id"].isin(TECH_ORDER)].copy()
    data["duration_h"] = data["duration_h"].astype(int)
    data = enrich_capacity_curve_metrics(data)
    keep_cols = [
        "tech_id",
        "duration_h",
        "province_cn",
        "capacity_gwh",
        "satisfy_rate_pct",
        "marginal_supply_gain_pct_per_gwh",
        "capacity_efficiency_pct_per_gwh",
    ]
    data = data[keep_cols].copy()
    data["province_short"] = data["province_cn"].map(short_province)
    data = data.sort_values(["tech_id", "duration_h", "province_cn", "capacity_gwh"])
    return data


def downsample_curve(group: pd.DataFrame, max_points: int = 180) -> pd.DataFrame:
    group = group.sort_values("capacity_gwh")
    if len(group) <= max_points:
        return group
    keep = np.unique(np.linspace(0, len(group) - 1, max_points).round().astype(int))
    return group.iloc[keep].copy()


def downsample_points(points: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [
            downsample_curve(group, max_points=180)
            for _keys, group in points.groupby(["tech_id", "duration_h", "province_cn"], sort=False)
        ],
        ignore_index=True,
    )


def build_province_color_map(tech_points: pd.DataFrame) -> dict[str, tuple[float, float, float, float]]:
    provinces = tech_points["province_cn"].drop_duplicates().tolist()
    colors = []
    for cmap_name in ["tab20", "tab20b", "tab20c"]:
        cmap = plt.get_cmap(cmap_name)
        colors.extend(cmap(np.linspace(0, 1, 20)))
    return {province: colors[i] for i, province in enumerate(provinces)}


def style_axis(ax: plt.Axes, ylog: bool, ylim: tuple[float, float]) -> None:
    ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    ax.set_xlim(0.02, 260)
    ax.set_ylim(*ylim)
    ax.set_xticks([0.03, 0.1, 1, 10, 100])
    ax.set_xticklabels(["0.03", "0.1", "1", "10", "100"])
    ax.grid(axis="both", which="major", color="#E6E6E6", lw=0.45)
    ax.grid(axis="both", which="minor", color="#F1F1F1", lw=0.25, alpha=0.55)
    ax.tick_params(axis="both", labelsize=6.6, pad=1.5)
    ax.set_xlabel("Energy capacity (GWh)", fontsize=7.0, labelpad=2)


def plot_metric_panel(
    ax: plt.Axes,
    points: pd.DataFrame,
    duration_h: int,
    color_map: dict[str, str],
    value_col: str,
    ylabel: str,
    ylog: bool,
    ylim: tuple[float, float],
    panel_title: str,
    show_ylabel: bool = True,
) -> None:
    duration_points = points[points["duration_h"].eq(duration_h)]

    for province, group in duration_points.groupby("province_cn", sort=False):
        group = group[np.isfinite(group[value_col])].sort_values("capacity_gwh")
        if group.empty:
            continue
        ax.plot(
            group["capacity_gwh"],
            group[value_col],
            color=color_map[province],
            lw=0.85,
            alpha=0.62,
            zorder=2,
        )

    style_axis(ax, ylog=ylog, ylim=ylim)
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=7.0, labelpad=2)
    else:
        ax.set_ylabel("")
    ax.set_title(panel_title, loc="left", fontsize=7.6, pad=3)

def add_shared_legend(fig: plt.Figure, province_order: list[str], color_map: dict[str, str]) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            color=color_map[province],
            lw=1.6,
            label=short_province(province),
        )
        for province in province_order
    ]
    legend_ax = fig.add_axes([0.112, 0.015, 0.88, 0.11])
    legend_ax.set_axis_off()
    legend_ax.legend(
        handles=handles,
        loc="upper left",
        ncol=9,
        fontsize=5.9,
        handlelength=1.35,
        columnspacing=0.95,
        handletextpad=0.38,
        borderaxespad=0.0,
        frameon=False,
    )


def save_figure(fig: plt.Figure, stem: Path, dpi: int = 450) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in {
        ".svg": {},
        ".pdf": {},
        ".png": {"dpi": dpi},
    }.items():
        fig.savefig(stem.with_suffix(ext), bbox_inches="tight", **kwargs)


def make_combined_duration_figure(points: pd.DataFrame, tech_id: str) -> None:
    tech_points = points[points["tech_id"].eq(tech_id)].copy()
    durations = sorted(tech_points["duration_h"].unique().tolist())
    if not durations:
        return
    province_order = tech_points["province_cn"].drop_duplicates().tolist()
    color_map = build_province_color_map(tech_points)

    fig, axes = plt.subplots(
        len(durations),
        3,
        figsize=(7.45, 1.72 * len(durations) + 1.25),
        sharex=False,
        constrained_layout=False,
    )
    if len(durations) == 1:
        axes = np.array([axes])

    for row, duration_h in enumerate(durations):
        for col, metric in enumerate(PANEL_COLS):
            value_col, ylabel, ylog, ylim = metric
            title = PANEL_TITLES[col] if row == 0 else ""
            plot_metric_panel(
                axes[row, col],
                tech_points,
                duration_h,
                color_map,
                value_col,
                ylabel,
                ylog,
                ylim,
                title,
                show_ylabel=True,
            )
            if col == 0:
                axes[row, col].text(
                    -0.26,
                    0.5,
                    f"{duration_h} h",
                    transform=axes[row, col].transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=7.6,
                    fontweight="bold",
                )

    tech_label = TECH_LABELS[tech_id]
    fig.suptitle(
        f"{tech_label} duration sweep: capacity response and engineering efficiency",
        x=0.06,
        y=0.995,
        ha="left",
        fontsize=9.2,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.975,
        "Rows show rated duration; all 26 provinces are displayed and color-coded consistently within each technology.",
        ha="left",
        va="top",
        fontsize=6.9,
        color="#4D4D4D",
    )
    add_shared_legend(fig, province_order, color_map)
    fig.subplots_adjust(left=0.112, right=0.995, top=0.935, bottom=0.18, wspace=0.33, hspace=0.42)

    stem = FIGURE_DIR / TECH_OUTPUT_STEMS[tech_id]
    save_figure(fig, stem)
    plt.close(fig)


def main() -> None:
    apply_publication_style()
    points = downsample_points(load_curve_points())
    for tech_id in TECH_ORDER:
        make_combined_duration_figure(points, tech_id)
    print("Saved Nature-style duration sweep figures to:")
    for tech_id in TECH_ORDER:
        print(f"  {FIGURE_DIR / (TECH_OUTPUT_STEMS[tech_id] + '.png')}")


if __name__ == "__main__":
    main()
