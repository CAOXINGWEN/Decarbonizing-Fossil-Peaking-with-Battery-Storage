"""
Fig 5-progression ratio-sweep — LDES/LFP 容量比例约束下的 cost-supply 曲线。

每条曲线代表一种容量配比约束：
  ratio = LDES_cap / (LFP_cap + LDES_cap)
  采样点：[0%, 25%, 50%, 75%, 100%] + Pareto 前沿（无约束）共 6 条

用容差 ±5% 过滤：
  - ratio=0% → 仅 LFP-only 点（LDES_cap=0）
  - ratio=100% → 仅 LDES-only 点（LFP_cap=0）
  - ratio=50% → LDES_share ∈ [45%, 55%]
  ...

数据源：m4_progression_planA_Y0_real_caes.csv（已存在，无需重跑）
场景：Y0_real + CAES + 4 个 dw（8/12/16/24h）

输出：
  output/m4_economics/nature_figure/fig5_progression_planA_ratio_sweep.{png,svg,pdf}
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
for fp in CJK_FONT_PATHS:
    if fp.exists():
        font_manager.fontManager.addfont(str(fp))
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Noto Sans CJK JP",
                                    "WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7.6
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.85
plt.rcParams["legend.frameon"] = False


PAPER_CODE_DIR = Path(__file__).resolve().parent
M4_DIR = PAPER_CODE_DIR / "output" / "m4_economics"
FIGURE_DIR = M4_DIR / "nature_figure"
SOURCE_DIR = FIGURE_DIR / "source_data"
FIG_BASE = FIGURE_DIR / "fig5_progression_planA_ratio_sweep"

DW_VALUES = [8.0, 12.0, 16.0, 24.0]
SCENARIO = "baseline"
YEAR_NODE = "Y0_real"
LDES_TECH = "caes"
CSV_PATH = M4_DIR / f"m4_progression_planA_{YEAR_NODE}_{LDES_TECH}.csv"

# 5 个比例采样点 + Pareto 前沿
RATIO_SAMPLES = [0, 25, 50, 75, 100]   # %
RATIO_TOL = 5  # ±5% 容差

# 颜色：从 LFP 蓝到 LDES 深蓝渐变
RATIO_COLORS = {
    0:   "#2F78A8",   # 纯 LFP（蓝）
    25:  "#4A90A8",   # LDES 轻
    50:  "#5C9BAA",   # 平均
    75:  "#3E8B72",   # LDES 重（绿）
    100: "#16456A",   # 纯 CAES（深蓝）
}
PARETO_COLOR = "#9B3036"  # 红，Pareto 前沿（无约束）

PALETTE = {
    "fill_blue": "#D9E6F0",
    "fill_red": "#F4D9D8",
    "grid": "#E8E8E8",
    "red": "#9B3036",
    "black": "#262626",
    "grey": "#666666",
}


def load_data() -> pd.DataFrame:
    """加载 Y0_real CAES 数据。"""
    if not CSV_PATH.exists():
        raise SystemExit(f"找不到 {CSV_PATH}")
    needed = [
        "scenario", "dispatch_window_h",
        "lfp_duration_h", "ldes_duration_h",
        "province_en", "province_cn",
        "lfp_capacity_mwh", "ldes_capacity_mwh",
        "satisfy_rate_pct", "unit_cost_cny_kwh", "cost_ratio_to_gas",
    ]
    df = pd.read_csv(CSV_PATH, usecols=needed, low_memory=False)
    df = df[df["scenario"].eq(SCENARIO)].copy()
    print(f"  Y0_real CAES: {len(df):,} rows")
    # 计算 LDES 容量占比
    df["ldes_cap_share"] = (
        df["ldes_capacity_mwh"]
        / (df["lfp_capacity_mwh"] + df["ldes_capacity_mwh"])
        * 100.0
    ).fillna(0)
    return df


def filter_by_ratio(df: pd.DataFrame, ratio_pct: float,
                     tol_pct: float = RATIO_TOL) -> pd.DataFrame:
    """按 LDES 容量占比筛选。

    ratio=0 → 仅 LFP-only（LDES_cap=0）
    ratio=100 → 仅 LDES-only（LFP_cap=0）
    其他 → ratio ± tol 区间内
    """
    if ratio_pct == 0:
        return df[(df["ldes_capacity_mwh"].eq(0))
                   & (df["lfp_capacity_mwh"].gt(0))]
    if ratio_pct == 100:
        return df[(df["lfp_capacity_mwh"].eq(0))
                   & (df["ldes_capacity_mwh"].gt(0))]
    return df[(df["lfp_capacity_mwh"].gt(0))
              & (df["ldes_capacity_mwh"].gt(0))
              & (df["ldes_cap_share"].between(
                  ratio_pct - tol_pct, ratio_pct + tol_pct))]


def build_province_curves(sub):
    """每省 1% satisfy 桶取最低 cost。"""
    curves = {}
    for prov, g in sub.groupby("province_cn", sort=False):
        g = g[["satisfy_rate_pct", "cost_ratio_to_gas"]].dropna()
        if g.empty:
            continue
        g = g.copy()
        g["bin"] = (g["satisfy_rate_pct"] // 1.0).astype(int)
        g = (g.groupby("bin", as_index=False)
                .agg(satisfy_rate_pct=("satisfy_rate_pct", "mean"),
                      cost_ratio_to_gas=("cost_ratio_to_gas", "min"))
                .drop(columns="bin")
                .sort_values("satisfy_rate_pct")
                .reset_index(drop=True))
        curves[prov] = g
    return curves


def interpolate_to_grid(curves, x_grid):
    rows = []
    for _prov, g in curves.items():
        if len(g) < 2:
            continue
        x = g["satisfy_rate_pct"].to_numpy(dtype=float)
        y = g["cost_ratio_to_gas"].to_numpy(dtype=float)
        x_u, idx = np.unique(x, return_index=True)
        y_u = y[idx]
        if len(x_u) < 2:
            continue
        interp = np.full_like(x_grid, np.nan, dtype=float)
        m = (x_grid >= x_u.min()) & (x_grid <= x_u.max())
        interp[m] = np.interp(x_grid[m], x_u, y_u)
        rows.append(interp)
    return np.vstack(rows) if rows else np.empty((0, len(x_grid)))


def plot_median(ax, mat, x_grid, color, ls="-", lw=1.6,
                 label=None, marker=None, ms=0):
    if mat.size == 0:
        return None
    med = np.nanmedian(mat, axis=0)
    valid = ~np.isnan(med)
    if not valid.any():
        return None
    kw = dict(color=color, lw=lw, ls=ls, zorder=3, label=label)
    if marker is not None:
        kw.update(marker=marker, ms=ms, markevery=30)
    ax.plot(x_grid[valid], med[valid], **kw)
    return pd.DataFrame({"supply_pct": x_grid[valid],
                          "median_cost_ratio": med[valid]})


def add_panel_label(ax, label):
    ax.text(-0.13, 1.06, label, transform=ax.transAxes,
             ha="left", va="bottom", fontsize=10, fontweight="bold",
             color=PALETTE["black"])


def draw_cell(ax, df_full, dw_h, panel_letter,
                show_xlabel=True, show_ylabel=True, show_legend=True):
    x_grid = np.linspace(0, 100, 241)
    df_cell = df_full[df_full["dispatch_window_h"].eq(dw_h)]
    rows = []

    # 1) 5 个比例约束曲线（按容量占比筛选）
    for ratio in RATIO_SAMPLES:
        sub = filter_by_ratio(df_cell, ratio)
        if sub.empty:
            print(f"    dw={dw_h} ratio={ratio}%: empty")
            continue
        cur = build_province_curves(sub)
        mat = interpolate_to_grid(cur, x_grid)
        if ratio == 0:
            label = "LFP-only (0%)"
            ls, marker, ms = "--", "s", 3.5
            lw = 2.0
        elif ratio == 100:
            label = "CAES-only (100%)"
            ls, marker, ms = ":", "D", 3.5
            lw = 2.0
        else:
            label = f"LDES {ratio}% cap"
            ls, marker, ms = "-", "o", 3.0
            lw = 1.5
        med = plot_median(ax, mat, x_grid, RATIO_COLORS[ratio],
                            ls=ls, lw=lw, label=label,
                            marker=marker, ms=ms)
        if med is not None:
            med = med.assign(combo=label, dispatch_window_h=dw_h)
            rows.append(med)

    # 2) Pareto 前沿（无约束，所有 hybrid 组合）
    sub_all = df_cell[(df_cell["lfp_capacity_mwh"].gt(0))
                       & (df_cell["ldes_capacity_mwh"].gt(0))]
    if not sub_all.empty:
        cur = build_province_curves(sub_all)
        mat = interpolate_to_grid(cur, x_grid)
        med = plot_median(ax, mat, x_grid, PARETO_COLOR,
                            ls="-", lw=2.4,
                            label="Pareto (unconstrained)",
                            marker="*", ms=4.5)
        if med is not None:
            med = med.assign(combo="Pareto unconstrained",
                              dispatch_window_h=dw_h)
            rows.append(med)

    ax.axhline(1.0, color=PALETTE["red"], lw=0.9, ls="--",
                 alpha=0.5)
    ax.fill_between([0, 100], 0.05, 1.0, color=PALETTE["fill_blue"],
                      alpha=0.12, zorder=0)
    ax.fill_between([0, 100], 1.0, 100, color=PALETTE["fill_red"],
                      alpha=0.08, zorder=0)
    ax.set_yscale("log")
    ax.set_xlim(0, 100); ax.set_ylim(0.15, 20)
    if show_xlabel:
        ax.set_xlabel("Peaking supply ratio (%)", labelpad=2)
    else:
        ax.set_xlabel("")
    if show_ylabel:
        ax.set_ylabel("Unit cost / gas cost", labelpad=2)
    else:
        ax.set_ylabel("")
    ax.set_title(f"{panel_letter}  dw={int(dw_h)}h",
                  loc="left", fontsize=8.2, pad=2)
    ax.grid(axis="both", which="major", color=PALETTE["grid"], lw=0.5)
    ax.tick_params(labelsize=6.8)
    if show_legend:
        ax.legend(loc="upper left", fontsize=5.8, handlelength=1.4,
                   borderaxespad=0.2, ncol=1, columnspacing=0.5,
                   handletextpad=0.4, labelspacing=0.28)
    add_panel_label(ax, panel_letter)
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame()


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== 加载 {YEAR_NODE} {LDES_TECH} 数据 ===")
    df = load_data()
    print(f"  容量占比分布:")
    print(df.groupby(
        pd.cut(df["ldes_cap_share"], [-1, 5, 25, 50, 75, 95, 105])
    ).size().to_string())

    fig = plt.figure(figsize=(18.0, 4.8))
    gs = fig.add_gridspec(1, 4,
                            left=0.050, right=0.985,
                            top=0.84, bottom=0.18,
                            wspace=0.22)

    source_rows = []
    letters = "abcd"
    for c, dw_h in enumerate(DW_VALUES):
        ax = fig.add_subplot(gs[0, c])
        sd = draw_cell(ax, df, dw_h, letters[c],
                        show_xlabel=True,
                        show_ylabel=(c == 0),
                        show_legend=True)
        if not sd.empty:
            source_rows.append(sd)

    fig.text(0.050, 0.96,
              f"Fig 5 (Plan A hybrid, ratio sweep) — Y0_real CAES + LFP "
              f"capacity ratio constraint",
              ha="left", va="top",
              fontsize=11, fontweight="bold", color=PALETTE["black"])
    fig.text(0.050, 0.93,
              f"Each curve = LDES capacity share constraint at "
              f"[0, 25, 50, 75, 100]% (±5% tolerance) + Pareto "
              f"(unconstrained). 26-province median, panel a logic. "
              f"Anchors LFP/LDES = 50/50. Red dashed = gas parity.",
              ha="left", va="top",
              fontsize=7.4, color=PALETTE["black"])

    for ext, kwargs in {".svg": {}, ".pdf": {},
                          ".png": {"dpi": 300}}.items():
        path = FIG_BASE.with_suffix(ext)
        fig.savefig(path, bbox_inches="tight", **kwargs)
        print(f"  saved: {path.name}")

    if source_rows:
        sd = pd.concat(source_rows, ignore_index=True)
        sd["year_node"] = YEAR_NODE
        sd["ldes_tech"] = LDES_TECH
        sd.to_csv(SOURCE_DIR / "fig5_progression_planA_ratio_sweep.csv",
                    index=False)
        print(f"  source data: {len(sd)} rows")


if __name__ == "__main__":
    main()
