"""
Fig 4 (2yr 加强版) — Nature-style M4 economics figure 双年加强版本。

仿照 fig4_m4_economics_nature.png 的 a/b/c/d 四面板布局，
但数据源全部来自 M3 双年（730 天）+ 极端事件加强 + M4 经济性双年版本：
  a. Unit-cost crossover         （成本/燃气基准 vs 调峰满足率）
  b. Profitability collapse      （单位利润 vs 调峰满足率）
  c. Marginal-cost divergence    （边际成本/燃气基准 vs 调峰满足率）
  d. National OSR by tech × scenario（high/baseline/low_cost 三档对比）

输出：
  output/m4_economics/nature_figure/fig4_m4_economics_nature_2yr.{png,pdf,svg}
  source_data/fig4_2yr_panel_*.csv
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
for fp in CJK_FONT_PATHS:
    if fp.exists():
        font_manager.fontManager.addfont(str(fp))
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Noto Sans CJK JP",
                                    "WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 8.0
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.9
plt.rcParams["legend.frameon"] = False


PAPER_CODE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PAPER_CODE_DIR / "output" / "m4_economics"
FIGURE_DIR = OUTPUT_DIR / "nature_figure"
SOURCE_DIR = FIGURE_DIR / "source_data"

TAG = "2yr"

SUMMARY_BASELINE_CSV = OUTPUT_DIR / f"m4_3r_summary_baseline_{TAG}.csv"
SUMMARY_LOW_CSV = OUTPUT_DIR / f"m4_3r_summary_low_cost_{TAG}.csv"
NATIONAL_ALL_CSV = OUTPUT_DIR / f"m4_national_3r_summary_all_scenarios_{TAG}.csv"
POINTS_BASELINE_CSV = OUTPUT_DIR / f"m4_curve_points_baseline_{TAG}.csv"
POINTS_LOW_CSV = OUTPUT_DIR / f"m4_curve_points_low_cost_{TAG}.csv"
SEGMENTS_BASELINE_CSV = OUTPUT_DIR / f"m4_curve_segments_baseline_{TAG}.csv"
SEGMENTS_LOW_CSV = OUTPUT_DIR / f"m4_curve_segments_low_cost_{TAG}.csv"

FIG_BASE = FIGURE_DIR / f"fig4_m4_economics_nature_{TAG}"

TECH_LABELS = {
    "li_lfp": "LFP",
    "caes": "CAES",
    "h2_cavern": "H2 cavern",
    "laes": "LAES",
    "gravity": "Gravity",
    "fecr_flow": "FeCr flow",
    "molten_salt_tes": "Molten salt TES",
    "vrfb": "VRFB",
}
PANEL_TECHS = list(TECH_LABELS.keys())

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

SCENARIO_ORDER = ["baseline", "low_cost"]
SCENARIO_LABELS = {"baseline": "Baseline (2yr)", "low_cost": "Low cost (2yr)"}
SCENARIO_COLORS = {"baseline": "#2F78A8", "low_cost": "#3E8B72"}

PALETTE = {
    "province_line": "#D4D4D4",
    "fill_blue": "#D9E6F0",
    "fill_red": "#F4D9D8",
    "grid": "#E8E8E8",
    "red": "#9B3036",
    "black": "#262626",
    "grey": "#666666",
}


def best_duration_by_tech(national: pd.DataFrame, tech_ids: list[str],
                            scenario: str = "baseline") -> dict[str, int]:
    sub = national[national["cost_scenario"].eq(scenario)].copy()
    out = {}
    for tid in tech_ids:
        s = sub[sub["tech_id"].eq(tid)].sort_values(
            "OSR_weighted_avg", ascending=False)
        if s.empty:
            continue
        out[tid] = int(s.iloc[0]["duration_h"])
    return out


def prepare_points(points: pd.DataFrame) -> pd.DataFrame:
    p = points.copy()
    p["cost_ratio_to_gas"] = (p["unit_cost_cny_kwh"]
                                / p["gas_unit_cost_cny_kwh"])
    p["unit_profit_cny_per_kwh_supply"] = np.where(
        p["total_supply_kwh"].astype(float) > 0,
        p["annual_net_profit_cny"].astype(float)
            / p["total_supply_kwh"].astype(float),
        np.nan,
    )
    return p


def prepare_segments(segments: pd.DataFrame) -> pd.DataFrame:
    s = segments.copy()
    s["marginal_ratio_to_gas"] = (s["marginal_cost_cny_kwh"]
                                    / s["gas_unit_cost_cny_kwh"])
    return s


def build_interpolated_median(df: pd.DataFrame, value_col: str,
                                x_col: str) -> pd.DataFrame:
    x_grid = np.linspace(0, 100, 241)
    curves = []
    for _prov, group in df.groupby("province_cn", sort=False):
        g = group.replace([np.inf, -np.inf], np.nan).dropna(
            subset=[x_col, value_col]).sort_values(x_col)
        if len(g) < 2:
            continue
        x = g[x_col].to_numpy(dtype=float)
        y = g[value_col].to_numpy(dtype=float)
        x_u, idx = np.unique(x, return_index=True)
        y_u = y[idx]
        interp = np.full_like(x_grid, np.nan, dtype=float)
        m = (x_grid >= x_u.min()) & (x_grid <= x_u.max())
        interp[m] = np.interp(x_grid[m], x_u, y_u)
        curves.append(interp)
    if not curves:
        return pd.DataFrame()
    arr = np.vstack(curves)
    median = np.nanmedian(arr, axis=0)
    out = pd.DataFrame({"supply_pct": x_grid, value_col: median})
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=[value_col])


def add_panel_label(ax, label: str, x: float = -0.10, y: float = 1.05):
    ax.text(x, y, label, transform=ax.transAxes,
             ha="left", va="bottom",
             fontsize=12, fontweight="bold", color=PALETTE["black"])


# ---------------- Panel a ----------------
def draw_panel_a(ax, points: pd.DataFrame, best_durations: dict[str, int]):
    rows = []
    for tid in PANEL_TECHS:
        d = best_durations.get(tid)
        if d is None:
            continue
        sub = points[(points["tech_id"].eq(tid))
                      & (points["duration_h"].eq(d))].copy()
        if sub.empty:
            continue
        med = build_interpolated_median(sub, "cost_ratio_to_gas",
                                          "satisfy_rate_pct")
        sub["panel"] = "a"; sub["tech_label"] = TECH_LABELS[tid]
        sub["selected_duration_h"] = d
        rows.append(sub)
        for _p, g in sub.groupby("province_cn", sort=False):
            ax.plot(g["satisfy_rate_pct"], g["cost_ratio_to_gas"],
                     color=TECH_COLORS[tid], lw=0.55, alpha=0.22, zorder=1)
        ax.plot(med["supply_pct"], med["cost_ratio_to_gas"],
                 color=TECH_COLORS[tid], lw=2.2, zorder=3,
                 label=f"{TECH_LABELS[tid]} {d}h median")
    ax.axhline(1.0, color=PALETTE["red"], lw=1.0, linestyle="--")
    ax.fill_between([0, 100], 0.12, 1.0, color=PALETTE["fill_blue"],
                      alpha=0.18, zorder=0)
    ax.fill_between([0, 100], 1.0, 18.0, color=PALETTE["fill_red"],
                      alpha=0.15, zorder=0)
    ax.set_yscale("log")
    ax.set_xlim(0, 100); ax.set_ylim(0.12, 18)
    ax.set_xlabel("Peaking supply ratio (%)  [2yr augmented]", labelpad=2)
    ax.set_ylabel("Unit cost / gas cost", labelpad=2)
    ax.set_title("a  Unit-cost crossover (2yr)", loc="left",
                  fontsize=9.4, pad=4)
    ax.grid(axis="both", which="major", color=PALETTE["grid"], lw=0.6)
    ax.text(99, 1.08, "gas parity", ha="right", va="bottom",
             fontsize=6.8, color=PALETTE["red"])
    ax.legend(loc="upper left", fontsize=5.6, handlelength=1.2,
               borderaxespad=0.2, ncol=2, columnspacing=0.8,
               handletextpad=0.5)
    add_panel_label(ax, "a")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ---------------- Panel b ----------------
def draw_panel_b(ax, points: pd.DataFrame, best_durations: dict[str, int]):
    rows = []
    for tid in PANEL_TECHS:
        d = best_durations.get(tid)
        if d is None:
            continue
        sub = points[(points["tech_id"].eq(tid))
                      & (points["duration_h"].eq(d))].copy()
        if sub.empty:
            continue
        med = build_interpolated_median(sub,
                                          "unit_profit_cny_per_kwh_supply",
                                          "satisfy_rate_pct")
        sub["panel"] = "b"; sub["tech_label"] = TECH_LABELS[tid]
        sub["selected_duration_h"] = d
        rows.append(sub)
        for _p, g in sub.groupby("province_cn", sort=False):
            ax.plot(g["satisfy_rate_pct"], g["unit_profit_cny_per_kwh_supply"],
                     color=PALETTE["province_line"], lw=0.6, alpha=0.12, zorder=1)
        ax.plot(med["supply_pct"], med["unit_profit_cny_per_kwh_supply"],
                 color=TECH_COLORS[tid], lw=2.0, zorder=3,
                 label=f"{TECH_LABELS[tid]} {d}h")
    ax.axhline(0.0, color=PALETTE["red"], lw=1.0, linestyle="--")
    ax.fill_between([0, 100], -0.35, 0.0, color=PALETTE["fill_red"],
                      alpha=0.15, zorder=0)
    ax.fill_between([0, 100], 0.0, 0.18, color=PALETTE["fill_blue"],
                      alpha=0.18, zorder=0)
    ax.set_xlim(0, 100); ax.set_ylim(-0.35, 0.18)
    ax.set_xlabel("Peaking supply ratio (%)  [2yr augmented]", labelpad=2)
    ax.set_ylabel("Unit profit (CNY kWh$^{-1}$)", labelpad=2)
    ax.set_title("b  Profitability collapse (2yr)", loc="left",
                  fontsize=9.4, pad=4)
    ax.grid(axis="both", which="major", color=PALETTE["grid"], lw=0.6)
    ax.text(99, 0.008, "zero profit", ha="right", va="bottom",
             fontsize=6.8, color=PALETTE["red"])
    ax.legend(loc="upper left", fontsize=5.6, handlelength=1.2,
               borderaxespad=0.2, ncol=2, columnspacing=0.8,
               handletextpad=0.5)
    add_panel_label(ax, "b")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ---------------- Panel c ----------------
def draw_panel_c(ax, segments: pd.DataFrame, best_durations: dict[str, int]):
    rows = []
    for tid in PANEL_TECHS:
        d = best_durations.get(tid)
        if d is None:
            continue
        sub = segments[(segments["tech_id"].eq(tid))
                         & (segments["duration_h"].eq(d))].copy()
        if sub.empty:
            continue
        med = build_interpolated_median(sub, "marginal_ratio_to_gas",
                                          "mid_satisfy_rate_pct")
        sub["panel"] = "c"; sub["tech_label"] = TECH_LABELS[tid]
        sub["selected_duration_h"] = d
        rows.append(sub)
        for _p, g in sub.groupby("province_cn", sort=False):
            ax.plot(g["mid_satisfy_rate_pct"], g["marginal_ratio_to_gas"],
                     color=PALETTE["province_line"], lw=0.6, alpha=0.12, zorder=1)
        ax.plot(med["supply_pct"], med["marginal_ratio_to_gas"],
                 color=TECH_COLORS[tid], lw=2.0, zorder=3,
                 label=f"{TECH_LABELS[tid]} {d}h")
    ax.axhline(1.0, color=PALETTE["red"], lw=1.0, linestyle="--")
    ax.fill_between([0, 100], 0.12, 1.0, color=PALETTE["fill_blue"],
                      alpha=0.18, zorder=0)
    ax.fill_between([0, 100], 1.0, 30.0, color=PALETTE["fill_red"],
                      alpha=0.15, zorder=0)
    ax.set_yscale("log")
    ax.set_xlim(0, 100); ax.set_ylim(0.12, 30)
    ax.set_xlabel("Peaking supply ratio (%)  [2yr augmented]", labelpad=2)
    ax.set_ylabel("Marginal cost / gas cost", labelpad=2)
    ax.set_title("c  Marginal-cost divergence (2yr)", loc="left",
                  fontsize=9.4, pad=4)
    ax.grid(axis="both", which="major", color=PALETTE["grid"], lw=0.6)
    ax.text(99, 1.08, "gas parity", ha="right", va="bottom",
             fontsize=6.8, color=PALETTE["red"])
    ax.legend(loc="upper left", fontsize=5.6, handlelength=1.2,
               borderaxespad=0.2, ncol=2, columnspacing=0.8,
               handletextpad=0.5)
    add_panel_label(ax, "c")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ---------------- Panel d ----------------
def draw_panel_d(ax, national: pd.DataFrame):
    keep = national[national["tech_id"].isin(TECH_LABELS)
                     & national["cost_scenario"].isin(SCENARIO_ORDER)].copy()
    rows = []
    ordered = (
        keep[keep["cost_scenario"].eq("baseline")]
        .sort_values("OSR_weighted_avg", ascending=False)["tech_id"]
        .drop_duplicates().tolist()
    )
    # 补齐 baseline 中没有 OSR>0 的技术 (按 low_cost 顺序兜底)
    for t in PANEL_TECHS:
        if t not in ordered:
            ordered.append(t)

    offsets = {"baseline": -0.12, "low_cost": 0.12}
    for i, tid in enumerate(ordered):
        tr = keep[keep["tech_id"].eq(tid)].copy()
        if tr.empty:
            continue
        tr["scenario_order"] = tr["cost_scenario"].map(
            {s: j for j, s in enumerate(SCENARIO_ORDER)})
        tr = tr.sort_values("scenario_order")
        # 每个 scenario 取该 scenario 下 OSR 最优的 duration
        best_per_scen = (tr.sort_values("OSR_weighted_avg", ascending=False)
                          .groupby("cost_scenario", as_index=False).first())
        xs = best_per_scen["OSR_weighted_avg"].to_numpy(dtype=float)
        ys = np.array([i + offsets[s]
                        for s in best_per_scen["cost_scenario"]])
        if len(xs) > 1:
            ax.plot(xs, ys, color="#D3D3D3", lw=1.0, zorder=1)
        for _, row in best_per_scen.iterrows():
            ax.scatter(row["OSR_weighted_avg"],
                        i + offsets[row["cost_scenario"]],
                        s=42, color=SCENARIO_COLORS[row["cost_scenario"]],
                        edgecolors="white", linewidths=0.7, zorder=3)
            rows.append(row)
        # baseline duration 注释
        bsub = best_per_scen[best_per_scen["cost_scenario"].eq("baseline")]
        if not bsub.empty:
            br = bsub.iloc[0]
            ax.text(br["OSR_weighted_avg"] + 0.8, i + 0.01,
                     f"{int(br['duration_h'])}h",
                     fontsize=6.2, color=PALETTE["grey"], va="center")

    ax.set_yticks(np.arange(len(ordered)))
    ax.set_yticklabels([TECH_LABELS[t] for t in ordered], fontsize=7.1)
    ax.invert_yaxis()
    xmax = max(5.0, keep["OSR_weighted_avg"].max() * 1.08)
    ax.set_xlim(-1, xmax)
    ax.set_xlabel("National OSR (%)  [2yr augmented]", labelpad=2)
    ax.set_title("d  Best 3R boundary by technology (2yr)", loc="left",
                  fontsize=9.4, pad=4)
    ax.grid(axis="x", color=PALETTE["grid"], lw=0.6)
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=5.5,
                markerfacecolor=SCENARIO_COLORS[s], markeredgecolor="white",
                label=SCENARIO_LABELS[s]) for s in SCENARIO_ORDER
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=6.4,
               handletextpad=0.4, borderaxespad=0.2)
    add_panel_label(ax, "d")
    return pd.DataFrame(rows)


# ---------------- main ----------------
def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading 2yr inputs...")
    points_b = prepare_points(pd.read_csv(POINTS_BASELINE_CSV))
    points_l = prepare_points(pd.read_csv(POINTS_LOW_CSV))
    segs_b = prepare_segments(pd.read_csv(SEGMENTS_BASELINE_CSV))
    national = pd.read_csv(NATIONAL_ALL_CSV)
    print(f"  baseline points: {len(points_b)}, low_cost points: {len(points_l)}")
    print(f"  baseline segments: {len(segs_b)}")
    print(f"  national rows: {len(national)}")

    best_dur = best_duration_by_tech(national, PANEL_TECHS, "baseline")
    # 兜底：baseline 全 OSR=0 的技术从 low_cost 拿 best duration
    for t in PANEL_TECHS:
        if t not in best_dur:
            sub = national[(national["cost_scenario"].eq("low_cost"))
                            & (national["tech_id"].eq(t))]
            if not sub.empty:
                best_dur[t] = int(
                    sub.sort_values("OSR_weighted_avg", ascending=False)
                       .iloc[0]["duration_h"])

    fig = plt.figure(figsize=(14.2, 10.0))
    gs = fig.add_gridspec(2, 2,
                            width_ratios=[1.05, 1.0],
                            height_ratios=[1.0, 1.0],
                            left=0.065, right=0.985, top=0.93, bottom=0.075,
                            wspace=0.28, hspace=0.32)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    panel_a = draw_panel_a(ax_a, points_b, best_dur)
    panel_b = draw_panel_b(ax_b, points_b, best_dur)
    panel_c = draw_panel_c(ax_c, segs_b, best_dur)
    panel_d = draw_panel_d(ax_d, national)

    panel_a.to_csv(SOURCE_DIR / f"fig4_2yr_panel_a_unit_cost.csv", index=False)
    panel_b.to_csv(SOURCE_DIR / f"fig4_2yr_panel_b_unit_profit.csv",
                    index=False)
    panel_c.to_csv(SOURCE_DIR / f"fig4_2yr_panel_c_marginal.csv", index=False)
    panel_d.to_csv(SOURCE_DIR / f"fig4_2yr_panel_d_national.csv", index=False)

    fig.text(0.065, 0.978,
              "M4 economics (2-year augmented dispatch): cost crossover, "
              "profitability collapse and substitution boundaries",
              ha="left", va="top",
              fontsize=11, fontweight="bold", color=PALETTE["black"])
    fig.text(0.065, 0.952,
              "M3 730-day dispatch with Markov-chain virtual year + extreme-event "
              "augmentation; panel d covers baseline & low_cost scenarios.",
              ha="left", va="top",
              fontsize=7.1, color=PALETTE["grey"])

    for ext, kwargs in {".svg": {}, ".pdf": {}, ".png": {"dpi": 340}}.items():
        fig.savefig(FIG_BASE.with_suffix(ext), bbox_inches="tight", **kwargs)
        print(f"  saved: {FIG_BASE.with_suffix(ext)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
