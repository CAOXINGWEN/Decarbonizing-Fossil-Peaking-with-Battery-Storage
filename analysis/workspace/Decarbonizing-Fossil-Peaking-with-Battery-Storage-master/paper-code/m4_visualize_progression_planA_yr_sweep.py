"""
Fig 5-progression yr-sweep — 多年限敏感性 4×4 矩阵图（Plan A hybrid）。

对每个仿真年限节点（Y0_real / Y1_yr1 / Y2_yr2 / Y3_yr3 / Y4_yr4 / Y5_yr5）
各出一幅 4×4 矩阵图，使用与 fig5_progression_planA_node_dw.png 同款逻辑：
  - 每条曲线 = 单一冠军 (lfp_dur, ldes_dur) 在 26 省的 bin-min 中位
  - 1% satisfy 桶取最低 cost（去除二维网格散点抖动）
  - 26 省插值取中位（panel a 同款）

注意：本文件中 4×4 矩阵的「行」不是 Y 节点（年限节点已固定），而是
M4_progression_search_planA 里原来的 NODE 维度（Y1/Y3/Y5/Y10 demand×N）。
但 yr_sweep 实验里所有数据都是 Y1 物理参数，**只有「年限」一个变量**。
所以 NODE 维度退化——本可视化用「dispatch_window_h」作为 X 轴单元，
4 行实际只显示 1 个 demand 节点（实际 demand 强度由年限决定，不再有
4 节点维度）。

修订设计：每年限节点出 1×4 矩阵（仅 4 个 dw），布局更紧凑。

数据源：
  output/m4_economics/m4_progression_planA_<year_node>_<tech>.csv
  output/m4_economics/m4_single_tech_li_lfp.csv (LFP-only 用 Y1 5yr 节点
  数据，**注意年限不一致**，仅作参考)

输出：
  output/m4_economics/nature_figure/fig5_progression_planA_<year_node>_dw.{png,svg,pdf}
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

DW_VALUES = [8.0, 12.0, 16.0, 24.0]
SCENARIO = "baseline"
CHAMP_SATISFY_MIN = 70.0

YR_NODES = ["Y0_real", "Y1_yr1", "Y2_yr2", "Y3_yr3", "Y4_yr4", "Y5_yr5"]
YR_LABELS = {
    "Y0_real": "Y0  1 year, real-only",
    "Y1_yr1":  "Y1  1 year, markov + extreme",
    "Y2_yr2":  "Y2  1 real + 1 markov (730 d)",
    "Y3_yr3":  "Y3  1 real + 2 markov (1095 d)",
    "Y4_yr4":  "Y4  1 real + 3 markov (1460 d)",
    "Y5_yr5":  "Y5  1 real + 4 markov (1825 d)",
}

LDES_TECHS = ["caes", "laes", "gravity", "fecr_flow",
                "molten_salt_tes", "vrfb", "h2_cavern"]
LDES_LABELS = {
    "caes": "CAES", "laes": "LAES",
    "gravity": "Gravity", "fecr_flow": "FeCr flow",
    "molten_salt_tes": "Molten salt TES",
    "vrfb": "VRFB", "h2_cavern": "H2 cavern",
}
TECH_COLORS = {
    "caes": "#16456A", "laes": "#C77C2E",
    "gravity": "#3E8B72", "fecr_flow": "#4E8B95",
    "molten_salt_tes": "#8A8A8A",
    "vrfb": "#A05FB5", "h2_cavern": "#7B68B3",
}
LFP_COLOR = "#2F78A8"

PALETTE = {
    "fill_blue": "#D9E6F0", "fill_red": "#F4D9D8",
    "grid": "#E8E8E8", "red": "#9B3036",
    "black": "#262626", "grey": "#666666",
}


def load_year_techs(year_node: str) -> dict[str, pd.DataFrame]:
    """加载指定年限节点的 7 LDES tech CSV。Y5_yr5 用现有
    m4_progression_planA_<tech>.csv（无 yr 后缀）。"""
    needed = ["scenario", "dispatch_window_h", "ldes_tech",
                "lfp_duration_h", "ldes_duration_h",
                "province_en", "province_cn",
                "lfp_capacity_mwh", "ldes_capacity_mwh",
                "satisfy_rate_pct", "unit_cost_cny_kwh",
                "cost_ratio_to_gas"]
    out = {}
    for tid in LDES_TECHS:
        if year_node == "Y5_yr5":
            fp = M4_DIR / f"m4_progression_planA_{tid}.csv"
        else:
            fp = M4_DIR / f"m4_progression_planA_{year_node}_{tid}.csv"
        if not fp.exists():
            print(f"  WARNING: 缺少 {fp.name}，跳过 {tid}")
            continue
        df = pd.read_csv(fp, usecols=needed, low_memory=False)
        df = df[df["scenario"].eq(SCENARIO)].copy()
        # Y5 数据有 node 列（Y1/Y3/Y5/Y10），yr_sweep 只跑了 Y1
        # 等价值；Y5_yr5 取 Y1（demand×1.0）以匹配其他年限（也用 Y1 物理参数）
        # 注意：Y5_yr5 的 node 列不在 usecols 里所以无 filter，直接用全部数据
        out[tid] = df
        print(f"  {tid}: {len(df):,} rows")
    return out


def load_lfp_only_for_year(year_node: str) -> pd.DataFrame:
    """LFP-only 数据。yr_sweep 没扫 LFP-only（节省时间），用 Y1 5yr 数据近似。"""
    fp = M4_DIR / "m4_single_tech_li_lfp.csv"
    if not fp.exists():
        return pd.DataFrame()
    needed = ["scenario", "node", "dispatch_window_h", "tech_id",
                "duration_h", "province_en", "province_cn",
                "capacity_mwh", "satisfy_rate_pct",
                "unit_cost_cny_kwh", "cost_ratio_to_gas"]
    df = pd.read_csv(fp, usecols=needed, low_memory=False)
    df = df[df["scenario"].eq(SCENARIO)].copy()
    # 取 Y1 节点（demand×1.0，与 yr_sweep 物理参数一致）
    df = df[df["node"].eq("Y1")].copy()
    print(f"  LFP-only (Y1): {len(df):,} rows")
    return df


def champion_hybrid(df_cell, satisfy_min=CHAMP_SATISFY_MIN):
    sub = df_cell[df_cell["satisfy_rate_pct"].ge(satisfy_min)
                    & df_cell["lfp_capacity_mwh"].gt(0)
                    & df_cell["ldes_capacity_mwh"].gt(0)]
    if sub.empty:
        return None
    med = (sub.groupby(["lfp_duration_h", "ldes_duration_h"])
              ["unit_cost_cny_kwh"].median().sort_values())
    if med.empty:
        return None
    lfp_d, ldes_d = med.index[0]
    return int(lfp_d), int(ldes_d)


def champion_lfp_only(df_lfp_cell, satisfy_min=CHAMP_SATISFY_MIN):
    sub = df_lfp_cell[df_lfp_cell["satisfy_rate_pct"].ge(satisfy_min)]
    if sub.empty:
        return None
    med = (sub.groupby("duration_h")["unit_cost_cny_kwh"]
              .median().sort_values())
    return int(med.index[0]) if not med.empty else None


def build_province_curves(sub):
    """每省 1% satisfy 桶取最低 cost（bin-min Pareto 前沿）。"""
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


def plot_thin(ax, curves, color, alpha=0.05):
    for _, g in curves.items():
        ax.plot(g["satisfy_rate_pct"], g["cost_ratio_to_gas"],
                 color=color, lw=0.5, alpha=alpha, zorder=1)


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


def draw_cell(ax, ldes_dfs, lfp_df, dw_h, panel_letter,
                show_xlabel=True, show_ylabel=True, show_legend=True):
    x_grid = np.linspace(0, 100, 241)
    rows = []
    champs = {}

    # 7 LDES + LFP hybrid 曲线先画（图例排前）
    for tid in LDES_TECHS:
        if tid not in ldes_dfs:
            continue
        df_cell = ldes_dfs[tid][ldes_dfs[tid]["dispatch_window_h"].eq(dw_h)]
        if df_cell.empty:
            continue
        ch = champion_hybrid(df_cell)
        if ch is None:
            continue
        lfp_d, ldes_d = ch
        sub = df_cell[(df_cell["lfp_duration_h"].eq(lfp_d))
                       & (df_cell["ldes_duration_h"].eq(ldes_d))
                       & (df_cell["lfp_capacity_mwh"].gt(0))
                       & (df_cell["ldes_capacity_mwh"].gt(0))]
        if sub.empty:
            continue
        cur = build_province_curves(sub)
        plot_thin(ax, cur, TECH_COLORS[tid], 0.05)
        mat = interpolate_to_grid(cur, x_grid)
        label = f"LFP{lfp_d}+{LDES_LABELS[tid]}{ldes_d}h"
        med = plot_median(ax, mat, x_grid, TECH_COLORS[tid], lw=1.5,
                            label=label, marker="o", ms=3.0)
        if med is not None:
            med = med.assign(combo=label, dispatch_window_h=dw_h)
            rows.append(med)
        champs[tid] = (lfp_d, ldes_d)

    # LFP-only 最后画（图例排末）
    if not lfp_df.empty:
        df_lfp_cell = lfp_df[lfp_df["dispatch_window_h"].eq(dw_h)]
        if not df_lfp_cell.empty:
            best = champion_lfp_only(df_lfp_cell)
            if best is not None:
                sub = df_lfp_cell[df_lfp_cell["duration_h"].eq(best)]
                cur = build_province_curves(
                    sub.rename(columns={"capacity_mwh": "_dummy"}))
                plot_thin(ax, cur, LFP_COLOR, 0.10)
                mat = interpolate_to_grid(cur, x_grid)
                med = plot_median(ax, mat, x_grid, LFP_COLOR, ls="--",
                                    lw=2.0,
                                    label=f"LFP {best}h only",
                                    marker="s", ms=3.5)
                if med is not None:
                    med = med.assign(combo=f"LFP{best}h only",
                                      dispatch_window_h=dw_h)
                    rows.append(med)
                champs["lfp_only_h"] = best

    ax.axhline(1.0, color=PALETTE["red"], lw=0.9, ls="--")
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
        ax.legend(loc="upper left", fontsize=5.4, handlelength=1.3,
                   borderaxespad=0.2, ncol=1, columnspacing=0.5,
                   handletextpad=0.4, labelspacing=0.25)
    add_panel_label(ax, panel_letter)
    if rows:
        return pd.concat(rows, ignore_index=True), champs
    return pd.DataFrame(), champs


def render_one_year(year_node: str):
    print(f"\n=== 渲染 {year_node} ===")
    ldes_dfs = load_year_techs(year_node)
    lfp_df = load_lfp_only_for_year(year_node)
    if not ldes_dfs:
        print(f"  跳过 {year_node}（无数据）")
        return

    # 1×4 矩阵图
    fig = plt.figure(figsize=(18.0, 4.8))
    gs = fig.add_gridspec(1, 4,
                            left=0.050, right=0.985,
                            top=0.84, bottom=0.18,
                            wspace=0.22)

    source_rows = []
    champ_summary = []
    letters = "abcd"
    for c, dw_h in enumerate(DW_VALUES):
        ax = fig.add_subplot(gs[0, c])
        sd, champs = draw_cell(
            ax, ldes_dfs, lfp_df, dw_h, letters[c],
            show_xlabel=True,
            show_ylabel=(c == 0),
            show_legend=True,
        )
        if not sd.empty:
            source_rows.append(sd)
        champ_summary.append({"dw_h": dw_h,
                                "lfp_only_h": champs.get("lfp_only_h"),
                                **{tid: f"LFP{a}+{LDES_LABELS[tid]}{b}h"
                                    for tid, val in champs.items()
                                    if tid in LDES_LABELS
                                    for a, b in [val]}})

    fig.text(0.050, 0.96,
              f"Fig 5 (Plan A hybrid, year sweep) — {YR_LABELS[year_node]}",
              ha="left", va="top",
              fontsize=11, fontweight="bold", color=PALETTE["black"])
    fig.text(0.050, 0.93,
              f"All 4 dispatch windows. Each curve = champion (LFP_dur, "
              f"LDES_dur) at satisfy>=70%. 26-province median (panel a logic). "
              f"Anchors LFP/LDES = 50/50 (step 2). Red dashed = gas parity.",
              ha="left", va="top",
              fontsize=7.4, color=PALETTE["black"])

    fig_base = FIGURE_DIR / f"fig5_progression_planA_{year_node}_dw"
    for ext, kwargs in {".svg": {}, ".pdf": {},
                          ".png": {"dpi": 300}}.items():
        path = fig_base.with_suffix(ext)
        fig.savefig(path, bbox_inches="tight", **kwargs)
        print(f"  saved: {path.name}")
    plt.close(fig)

    if source_rows:
        sd = pd.concat(source_rows, ignore_index=True)
        sd["year_node"] = year_node
        sd.to_csv(SOURCE_DIR / f"fig5_progression_planA_{year_node}_curves.csv",
                    index=False)
    if champ_summary:
        cs = pd.DataFrame(champ_summary)
        cs["year_node"] = year_node
        cs.to_csv(SOURCE_DIR / f"fig5_progression_planA_{year_node}_champions.csv",
                    index=False)
        print(f"  champions:\n{cs.to_string(index=False)}")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("year_node", nargs="?", default=None,
                        help=f"指定年限节点: {YR_NODES}")
    parser.add_argument("--all", action="store_true",
                        help="对所有 6 个年限节点出图")
    args = parser.parse_args()

    if args.all:
        for yn in YR_NODES:
            render_one_year(yn)
    elif args.year_node:
        render_one_year(args.year_node)
    else:
        parser.error("必须指定 year_node 或 --all")


if __name__ == "__main__":
    main()
