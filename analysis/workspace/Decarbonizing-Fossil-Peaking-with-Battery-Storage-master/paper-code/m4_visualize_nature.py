"""
Nature-style M4 economics figure with multi-technology median curves.

Panels:
  a. Median unit-cost ratio curves for multiple technologies
  b. Median unit-profit curves for multiple technologies
  c. Median marginal-cost ratio curves for multiple technologies
  d. National OSR by technology with cost-scenario overlays
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
plt.rcParams["font.size"] = 8.0
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.9
plt.rcParams["legend.frameon"] = False


PAPER_CODE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PAPER_CODE_DIR / "output" / "m4_economics"
FIGURE_DIR = OUTPUT_DIR / "nature_figure"
SOURCE_DIR = FIGURE_DIR / "source_data"

SUMMARY_ALL_CSV = OUTPUT_DIR / "m4_3r_summary_all_scenarios.csv"
NATIONAL_ALL_CSV = OUTPUT_DIR / "m4_national_3r_summary_all_scenarios.csv"
POINTS_BASELINE_CSV = OUTPUT_DIR / "m4_curve_points_baseline.csv"
SEGMENTS_BASELINE_CSV = OUTPUT_DIR / "m4_curve_segments_baseline.csv"

FIG_BASE = FIGURE_DIR / "fig4_m4_economics_nature"

TECH_LABELS = {
    "caes": "CAES",
    "h2_cavern": "H2 cavern",
    "laes": "LAES",
    "gravity": "Gravity",
    "fecr_flow": "FeCr flow",
    "molten_salt_tes": "Molten salt TES",
    "li_lfp": "LFP",
    "vrfb": "VRFB",
}

PANEL_A_TECHS = [
    "caes",
    "h2_cavern",
    "laes",
    "gravity",
    "fecr_flow",
    "molten_salt_tes",
    "li_lfp",
    "vrfb",
]
PANEL_BC_TECHS = PANEL_A_TECHS
SCENARIO_ORDER = ["high_cost", "baseline", "low_cost"]
SCENARIO_LABELS = {
    "high_cost": "High cost",
    "baseline": "Baseline",
    "low_cost": "Low cost",
}
SCENARIO_COLORS = {
    "high_cost": "#AD3A3E",
    "baseline": "#2F78A8",
    "low_cost": "#3E8B72",
}
TECH_COLORS = {
    "caes": "#16456A",
    "h2_cavern": "#7B68B3",
    "laes": "#C77C2E",
    "gravity": "#3E8B72",
    "fecr_flow": "#4E8B95",
    "molten_salt_tes": "#8A8A8A",
    "li_lfp": "#2F78A8",
    "vrfb": "#A05FB5",
}
PALETTE = {
    "province_line": "#D4D4D4",
    "fill_blue": "#D9E6F0",
    "fill_red": "#F4D9D8",
    "grid": "#E8E8E8",
    "red": "#9B3036",
    "black": "#262626",
    "grey": "#666666",
}


def require_input(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required M4 input is missing: {path}")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in [SUMMARY_ALL_CSV, NATIONAL_ALL_CSV, POINTS_BASELINE_CSV, SEGMENTS_BASELINE_CSV]:
        require_input(path)
    return (
        pd.read_csv(SUMMARY_ALL_CSV),
        pd.read_csv(NATIONAL_ALL_CSV),
        pd.read_csv(POINTS_BASELINE_CSV),
        pd.read_csv(SEGMENTS_BASELINE_CSV),
    )


def best_duration_by_tech(national: pd.DataFrame, tech_ids: list[str]) -> dict[str, int]:
    base = national[national["cost_scenario"].eq("baseline")].copy()
    out = {}
    for tech_id in tech_ids:
        sub = base[base["tech_id"].eq(tech_id)].sort_values("OSR_weighted_avg", ascending=False)
        if sub.empty:
            continue
        out[tech_id] = int(sub.iloc[0]["duration_h"])
    return out


def prepare_points(points: pd.DataFrame) -> pd.DataFrame:
    points = points.copy()
    points["cost_ratio_to_gas"] = points["unit_cost_cny_kwh"] / points["gas_unit_cost_cny_kwh"]
    points["unit_profit_cny_per_kwh_supply"] = np.where(
        points["total_supply_kwh"].astype(float) > 0,
        points["annual_net_profit_cny"].astype(float) / points["total_supply_kwh"].astype(float),
        np.nan,
    )
    return points


def prepare_segments(segments: pd.DataFrame) -> pd.DataFrame:
    segments = segments.copy()
    segments["marginal_ratio_to_gas"] = segments["marginal_cost_cny_kwh"] / segments["gas_unit_cost_cny_kwh"]
    return segments


def build_interpolated_median(df: pd.DataFrame, value_col: str, x_col: str) -> pd.DataFrame:
    x_grid = np.linspace(0, 100, 241)
    curves = []
    for _prov, group in df.groupby("province_cn", sort=False):
        group = group.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_col, value_col])
        group = group.sort_values(x_col)
        if len(group) < 2:
            continue
        x = group[x_col].to_numpy(dtype=float)
        y = group[value_col].to_numpy(dtype=float)
        x_unique, idx = np.unique(x, return_index=True)
        y_unique = y[idx]
        interp = np.full_like(x_grid, np.nan, dtype=float)
        mask = (x_grid >= x_unique.min()) & (x_grid <= x_unique.max())
        interp[mask] = np.interp(x_grid[mask], x_unique, y_unique)
        curves.append(interp)
    if not curves:
        return pd.DataFrame()
    arr = np.vstack(curves)
    median = np.nanmedian(arr, axis=0)
    out = pd.DataFrame({"supply_pct": x_grid, value_col: median})
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=[value_col])


def collect_curve_bundle(points: pd.DataFrame, tech_id: str, duration_h: int, value_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = points[(points["tech_id"].eq(tech_id)) & (points["duration_h"].eq(duration_h))].copy()
    median = build_interpolated_median(sub, value_col, "satisfy_rate_pct")
    return sub, median


def collect_segment_bundle(segments: pd.DataFrame, tech_id: str, duration_h: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = segments[(segments["tech_id"].eq(tech_id)) & (segments["duration_h"].eq(duration_h))].copy()
    median = build_interpolated_median(sub, "marginal_ratio_to_gas", "mid_satisfy_rate_pct")
    return sub, median


def export_source_data(
    panel_ab: pd.DataFrame,
    panel_c: pd.DataFrame,
    panel_d: pd.DataFrame,
) -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    panel_ab.to_csv(SOURCE_DIR / "fig4_panel_ab_multi_tech_curves.csv", index=False)
    panel_c.to_csv(SOURCE_DIR / "fig4_panel_c_multi_tech_marginal_curves.csv", index=False)
    panel_d.to_csv(SOURCE_DIR / "fig4_panel_d_national_osr_scenarios.csv", index=False)


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.10, y: float = 1.05) -> None:
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color=PALETTE["black"],
    )


def draw_panel_a(
    ax: plt.Axes,
    points: pd.DataFrame,
    best_durations: dict[str, int],
    tech_ids: list[str],
) -> pd.DataFrame:
    rows = []
    for tech_id in tech_ids:
        duration_h = best_durations[tech_id]
        thin, median = collect_curve_bundle(points, tech_id, duration_h, "cost_ratio_to_gas")
        thin["panel"] = "a"
        thin["tech_label"] = TECH_LABELS[tech_id]
        thin["selected_duration_h"] = duration_h
        rows.append(thin)
        for _prov, group in thin.groupby("province_cn", sort=False):
            ax.plot(
                group["satisfy_rate_pct"],
                group["cost_ratio_to_gas"],
                color=TECH_COLORS[tech_id],
                lw=0.55,
                alpha=0.22,
                zorder=1,
            )
        ax.plot(
            median["supply_pct"],
            median["cost_ratio_to_gas"],
            color=TECH_COLORS[tech_id],
            lw=2.2,
            zorder=3,
            label=f"{TECH_LABELS[tech_id]} {duration_h}h median",
        )
    ax.axhline(1.0, color=PALETTE["red"], lw=1.0, linestyle="--")
    ax.fill_between([0, 100], 0.12, 1.0, color=PALETTE["fill_blue"], alpha=0.18, zorder=0)
    ax.fill_between([0, 100], 1.0, 18.0, color=PALETTE["fill_red"], alpha=0.15, zorder=0)
    ax.set_yscale("log")
    ax.set_xlim(0, 100)
    ax.set_ylim(0.12, 18)
    ax.set_xlabel("Peaking supply ratio (%)", labelpad=2)
    ax.set_ylabel("Unit cost / gas cost", labelpad=2)
    ax.set_title("a  Unit-cost crossover", loc="left", fontsize=9.4, pad=4)
    ax.grid(axis="both", which="major", color=PALETTE["grid"], lw=0.6)
    ax.text(99, 1.08, "gas parity", ha="right", va="bottom", fontsize=6.8, color=PALETTE["red"])
    ax.legend(
        loc="upper left",
        fontsize=5.6,
        handlelength=1.2,
        borderaxespad=0.2,
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.5,
    )
    add_panel_label(ax, "a")
    return pd.concat(rows, ignore_index=True)


def draw_panel_b(ax: plt.Axes, points: pd.DataFrame, best_durations: dict[str, int]) -> pd.DataFrame:
    rows = []
    for tech_id in PANEL_BC_TECHS:
        duration_h = best_durations[tech_id]
        thin, median = collect_curve_bundle(points, tech_id, duration_h, "unit_profit_cny_per_kwh_supply")
        thin["panel"] = "b"
        thin["tech_label"] = TECH_LABELS[tech_id]
        thin["selected_duration_h"] = duration_h
        rows.append(thin)
        for _prov, group in thin.groupby("province_cn", sort=False):
            ax.plot(group["satisfy_rate_pct"], group["unit_profit_cny_per_kwh_supply"], color=PALETTE["province_line"], lw=0.6, alpha=0.12, zorder=1)
        ax.plot(median["supply_pct"], median["unit_profit_cny_per_kwh_supply"], color=TECH_COLORS[tech_id], lw=2.0, zorder=3, label=f"{TECH_LABELS[tech_id]} {duration_h}h")
    ax.axhline(0.0, color=PALETTE["red"], lw=1.0, linestyle="--")
    ax.fill_between([0, 100], -0.35, 0.0, color=PALETTE["fill_red"], alpha=0.15, zorder=0)
    ax.fill_between([0, 100], 0.0, 0.18, color=PALETTE["fill_blue"], alpha=0.18, zorder=0)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.35, 0.18)
    ax.set_xlabel("Peaking supply ratio (%)", labelpad=2)
    ax.set_ylabel("Unit profit (CNY kWh$^{-1}$)", labelpad=2)
    ax.set_title("b  Profitability collapse", loc="left", fontsize=9.4, pad=4)
    ax.grid(axis="both", which="major", color=PALETTE["grid"], lw=0.6)
    ax.text(99, 0.008, "zero profit", ha="right", va="bottom", fontsize=6.8, color=PALETTE["red"])
    ax.legend(
        loc="upper left",
        fontsize=5.6,
        handlelength=1.2,
        borderaxespad=0.2,
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.5,
    )
    add_panel_label(ax, "b")
    return pd.concat(rows, ignore_index=True)


def draw_panel_c(ax: plt.Axes, segments: pd.DataFrame, best_durations: dict[str, int]) -> pd.DataFrame:
    rows = []
    for tech_id in PANEL_BC_TECHS:
        duration_h = best_durations[tech_id]
        thin, median = collect_segment_bundle(segments, tech_id, duration_h)
        thin["panel"] = "c"
        thin["tech_label"] = TECH_LABELS[tech_id]
        thin["selected_duration_h"] = duration_h
        rows.append(thin)
        for _prov, group in thin.groupby("province_cn", sort=False):
            ax.plot(group["mid_satisfy_rate_pct"], group["marginal_ratio_to_gas"], color=PALETTE["province_line"], lw=0.6, alpha=0.12, zorder=1)
        ax.plot(median["supply_pct"], median["marginal_ratio_to_gas"], color=TECH_COLORS[tech_id], lw=2.0, zorder=3, label=f"{TECH_LABELS[tech_id]} {duration_h}h")
    ax.axhline(1.0, color=PALETTE["red"], lw=1.0, linestyle="--")
    ax.fill_between([0, 100], 0.12, 1.0, color=PALETTE["fill_blue"], alpha=0.18, zorder=0)
    ax.fill_between([0, 100], 1.0, 30.0, color=PALETTE["fill_red"], alpha=0.15, zorder=0)
    ax.set_yscale("log")
    ax.set_xlim(0, 100)
    ax.set_ylim(0.12, 30)
    ax.set_xlabel("Peaking supply ratio (%)", labelpad=2)
    ax.set_ylabel("Marginal cost / gas cost", labelpad=2)
    ax.set_title("c  Marginal-cost divergence", loc="left", fontsize=9.4, pad=4)
    ax.grid(axis="both", which="major", color=PALETTE["grid"], lw=0.6)
    ax.text(99, 1.08, "gas parity", ha="right", va="bottom", fontsize=6.8, color=PALETTE["red"])
    ax.legend(
        loc="upper left",
        fontsize=5.6,
        handlelength=1.2,
        borderaxespad=0.2,
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.5,
    )
    add_panel_label(ax, "c")
    return pd.concat(rows, ignore_index=True)


def draw_panel_d(ax: plt.Axes, national: pd.DataFrame) -> pd.DataFrame:
    keep = national[national["tech_id"].isin(TECH_LABELS)].copy()
    rows = []
    ordered = (
        keep[keep["cost_scenario"].eq("baseline")]
        .sort_values("OSR_weighted_avg", ascending=False)["tech_id"]
        .drop_duplicates()
        .tolist()
    )
    offsets = {"high_cost": -0.18, "baseline": 0.0, "low_cost": 0.18}
    for i, tech_id in enumerate(ordered):
        tech_rows = keep[keep["tech_id"].eq(tech_id)].copy()
        tech_rows["scenario_order"] = tech_rows["cost_scenario"].map({s: j for j, s in enumerate(SCENARIO_ORDER)})
        tech_rows = tech_rows.sort_values("scenario_order")
        xs = tech_rows["OSR_weighted_avg"].to_numpy(dtype=float)
        ys = np.array([i + offsets[row["cost_scenario"]] for _, row in tech_rows.iterrows()])
        if len(xs) > 1:
            ax.plot(xs, ys, color="#D3D3D3", lw=1.0, zorder=1)
        for _, row in tech_rows.iterrows():
            ax.scatter(
                row["OSR_weighted_avg"],
                i + offsets[row["cost_scenario"]],
                s=42,
                color=SCENARIO_COLORS[row["cost_scenario"]],
                edgecolors="white",
                linewidths=0.7,
                zorder=3,
            )
            rows.append(row)
        base_row = tech_rows[tech_rows["cost_scenario"].eq("baseline")].iloc[0]
        ax.text(base_row["OSR_weighted_avg"] + 0.8, i + 0.01, f"{int(base_row['duration_h'])}h", fontsize=6.2, color=PALETTE["grey"], va="center")
    ax.set_yticks(np.arange(len(ordered)))
    ax.set_yticklabels([TECH_LABELS[t] for t in ordered], fontsize=7.1)
    ax.invert_yaxis()
    ax.set_xlim(-1, max(5.0, keep["OSR_weighted_avg"].max() * 1.08))
    ax.set_xlabel("National OSR (%)", labelpad=2)
    ax.set_title("d  Best 3R boundary by technology", loc="left", fontsize=9.4, pad=4)
    ax.grid(axis="x", color=PALETTE["grid"], lw=0.6)
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=5.5, markerfacecolor=SCENARIO_COLORS[s], markeredgecolor="white", label=SCENARIO_LABELS[s])
        for s in SCENARIO_ORDER
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=6.4, handletextpad=0.4, borderaxespad=0.2)
    add_panel_label(ax, "d")
    return pd.DataFrame(rows)


def make_figure() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    summary, national, points, segments = load_inputs()
    points = prepare_points(points)
    segments = prepare_segments(segments)
    best_durations_a = best_duration_by_tech(national, PANEL_A_TECHS)
    best_durations_bc = best_duration_by_tech(national, PANEL_BC_TECHS)

    fig = plt.figure(figsize=(14.2, 10.0))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1.05, 1.0],
        height_ratios=[1.0, 1.0],
        left=0.065,
        right=0.985,
        top=0.94,
        bottom=0.085,
        wspace=0.28,
        hspace=0.32,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    panel_ab = draw_panel_a(ax_a, points, best_durations_a, PANEL_A_TECHS)
    panel_bb = draw_panel_b(ax_b, points, best_durations_bc)
    panel_c = draw_panel_c(ax_c, segments, best_durations_bc)
    panel_d = draw_panel_d(ax_d, national)

    export_source_data(
        panel_ab=pd.concat([panel_ab, panel_bb], ignore_index=True),
        panel_c=panel_c,
        panel_d=panel_d,
    )

    fig.text(
        0.065, 0.978,
        "M4 economics: cost crossover, profitability collapse and substitution boundaries",
        ha="left", va="top",
        fontsize=11, fontweight="bold", color=PALETTE["black"],
    )
    fig.text(
        0.065, 0.952,
        "Panels a–c follow the first-paper logic with provincial thin lines and multi-technology median curves; panel d summarizes national 3R ranking.",
        ha="left", va="top",
        fontsize=7.1, color=PALETTE["grey"],
    )

    for ext, kwargs in {
        ".svg": {},
        ".pdf": {},
        ".png": {"dpi": 340},
    }.items():
        fig.savefig(FIG_BASE.with_suffix(ext), bbox_inches="tight", **kwargs)

    plt.close(fig)
    print(f"Saved M4 Nature-style figure to {FIG_BASE.with_suffix('.png')}")
    print(f"Source data exported to {SOURCE_DIR}")


if __name__ == "__main__":
    make_figure()
