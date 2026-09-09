"""
Fig 5-progression hybrid (Plan A) — 4×4 矩阵图：每条曲线是一个 LDES tech +
LFP 混合的省级中位曲线（方案 A：LDES 优先放电）。

与 m4_visualize_progression_single_tech.py 同款画图逻辑：
  - 每条曲线 = 单一冠军 (lfp_dur, ldes_dur) 下，逐省按 satisfy_rate 升序
    线性插值到 0-100% 网格，再对 26 省取中位
  - 不做 cummin、不做滚动平滑

数据源：每 LDES tech 一个 CSV
  output/m4_economics/m4_progression_planA_<tech>.csv
  容量锚点：LFP 6 档 [5,20,40,60,80,95]%，LDES 21 档 [5,10,...,95,99]%

LFP-only 曲线：从 m4_single_tech_li_lfp.csv 读取（与 single-tech 图同源）

输出：output/m4_economics/nature_figure/fig5_progression_planA_node_dw.{png,svg,pdf}
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
FIG_BASE = FIGURE_DIR / "fig5_progression_planA_node_dw"

NODES = ["Y1", "Y3", "Y5", "Y10"]
DW_VALUES = [8.0, 12.0, 16.0, 24.0]
SCENARIO = "baseline"
CHAMP_SATISFY_MIN = 70.0

# 7 LDES tech 数据源
LDES_TECH_FILES = {
    "caes": "m4_progression_planA_caes.csv",
    "laes": "m4_progression_planA_laes.csv",
    "gravity": "m4_progression_planA_gravity.csv",
    "fecr_flow": "m4_progression_planA_fecr_flow.csv",
    "molten_salt_tes": "m4_progression_planA_molten_salt_tes.csv",
    "vrfb": "m4_progression_planA_vrfb.csv",
    "h2_cavern": "m4_progression_planA_h2_cavern.csv",
}
# LFP-only 单技术数据源（复用 single-tech 已扫数据）
LFP_ONLY_FILE = "m4_single_tech_li_lfp.csv"

LDES_LABELS = {
    "caes": "CAES",
    "laes": "LAES",
    "gravity": "Gravity",
    "fecr_flow": "FeCr flow",
    "molten_salt_tes": "Molten salt TES",
    "vrfb": "VRFB",
    "h2_cavern": "H2 cavern",
}
TECH_COLORS = {
    "caes": "#16456A",
    "laes": "#C77C2E",
    "gravity": "#3E8B72",
    "fecr_flow": "#4E8B95",
    "molten_salt_tes": "#8A8A8A",
    "vrfb": "#A05FB5",
    "h2_cavern": "#7B68B3",
}
LFP_COLOR = "#2F78A8"

PALETTE = {
    "fill_blue": "#D9E6F0",
    "fill_red": "#F4D9D8",
    "grid": "#E8E8E8",
    "red": "#9B3036",
    "black": "#262626",
    "grey": "#666666",
}


def load_ldes_techs() -> dict[str, pd.DataFrame]:
    needed = [
        "scenario", "node", "dispatch_window_h", "ldes_tech",
        "lfp_duration_h", "ldes_duration_h",
        "province_en", "province_cn",
        "lfp_capacity_mwh", "ldes_capacity_mwh",
        "satisfy_rate_pct", "unit_cost_cny_kwh", "cost_ratio_to_gas",
    ]
    out = {}
    for tid, fname in LDES_TECH_FILES.items():
        fp = M4_DIR / fname
        if not fp.exists():
            print(f"  WARNING: 缺少 {fp.name}，跳过 {tid}")
            continue
        df = pd.read_csv(fp, usecols=needed, low_memory=False)
        df = df[df["scenario"].eq(SCENARIO)].copy()
        out[tid] = df
        print(f"  {tid}: {len(df):,} rows")
    return out


def load_lfp_only() -> pd.DataFrame:
    fp = M4_DIR / LFP_ONLY_FILE
    if not fp.exists():
        print(f"  WARNING: 缺少 {fp.name}，LFP-only 曲线不画")
        return pd.DataFrame()
    needed = [
        "scenario", "node", "dispatch_window_h", "tech_id", "duration_h",
        "province_en", "province_cn", "capacity_mwh",
        "satisfy_rate_pct", "unit_cost_cny_kwh", "cost_ratio_to_gas",
    ]
    df = pd.read_csv(fp, usecols=needed, low_memory=False)
    df = df[df["scenario"].eq(SCENARIO)].copy()
    print(f"  LFP-only: {len(df):,} rows")
    return df


def champion_hybrid(df_cell: pd.DataFrame,
                     satisfy_min: float = CHAMP_SATISFY_MIN
                     ) -> tuple[int, int] | None:
    """选 (lfp_dur, ldes_dur) 冠军：satisfy>=70% 时省级中位单位成本最低。"""
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


def champion_lfp_only(df_lfp_cell: pd.DataFrame,
                       satisfy_min: float = CHAMP_SATISFY_MIN) -> int | None:
    sub = df_lfp_cell[df_lfp_cell["satisfy_rate_pct"].ge(satisfy_min)]
    if sub.empty:
        return None
    med = (sub.groupby("duration_h")["unit_cost_cny_kwh"]
              .median().sort_values())
    return int(med.index[0]) if not med.empty else None


def build_province_curves(sub: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """每省构建 (satisfy, cost) 曲线：仅 bin-min 去波。

    hybrid 是二维 (lfp_cap, ldes_cap) 网格，同 satisfy 可由多组合达成、
    cost 不同 → 直接 sort + interp 会得到锯齿。改为按 satisfy 1% 分桶
    取最小 cost（每个 supply 下保留最优 (lfp_cap, ldes_cap) 组合）。

    **不做 cummin**：cummin 会把曲线压成每省全局最低 cost 的水平线，
    丢失 U 形物理结构（左端摊薄、中段最优、右端饱和回升）。bin-min
    已经足够去除二维网格在同 satisfy 上的散点抖动。
    """
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


def plot_thin_lines(ax, curves, color, alpha=0.08):
    for _prov, g in curves.items():
        ax.plot(g["satisfy_rate_pct"], g["cost_ratio_to_gas"],
                 color=color, lw=0.5, alpha=alpha, zorder=1)


def plot_median(ax, mat, x_grid, color, ls="-", lw=1.6, label=None,
                 marker=None, ms=0):
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
             ha="left", va="bottom",
             fontsize=10, fontweight="bold", color=PALETTE["black"])


def draw_cell(ax, ldes_dfs: dict[str, pd.DataFrame],
                lfp_df: pd.DataFrame,
                node, dw_h, panel_letter,
                show_xlabel=True, show_ylabel=True,
                show_legend=True):
    x_grid = np.linspace(0, 100, 241)
    rows = []
    champs = {}

    # 7 LDES + LFP hybrid 曲线先画（图例排前）
    for tid in LDES_LABELS:
        if tid not in ldes_dfs:
            continue
        df_full = ldes_dfs[tid]
        df_cell = df_full[(df_full["node"] == node)
                           & (df_full["dispatch_window_h"].eq(dw_h))]
        if df_cell.empty:
            continue
        ch = champion_hybrid(df_cell)
        if ch is None:
            continue
        lfp_d, ldes_d = ch
        # 真正 hybrid 子集：LFP 和 LDES 容量都 > 0。
        # bin-min（每 1% satisfy 桶取最低 cost）已足够平滑二维网格散点，
        # 无需混入单边点（混入会让曲线退化为 LFP/LDES/hybrid 三者下包络，
        # 整体被 LFP-only 拉低，并在低段产生人造抛物线形状）。
        sub = df_cell[(df_cell["lfp_duration_h"].eq(lfp_d))
                       & (df_cell["ldes_duration_h"].eq(ldes_d))
                       & (df_cell["lfp_capacity_mwh"].gt(0))
                       & (df_cell["ldes_capacity_mwh"].gt(0))]
        if sub.empty:
            continue
        cur = build_province_curves(sub)
        plot_thin_lines(ax, cur, TECH_COLORS[tid], 0.05)
        mat = interpolate_to_grid(cur, x_grid)
        label = f"LFP{lfp_d}+{LDES_LABELS[tid]}{ldes_d}h"
        med = plot_median(ax, mat, x_grid, TECH_COLORS[tid], lw=1.5,
                            label=label, marker="o", ms=3.0)
        if med is not None:
            med = med.assign(combo=label,
                              node=node, dispatch_window_h=dw_h)
            rows.append(med)
        champs[tid] = (lfp_d, ldes_d)

    # LFP-only 曲线（最后画，图例排末位）
    if not lfp_df.empty:
        df_lfp_cell = lfp_df[(lfp_df["node"] == node)
                              & (lfp_df["dispatch_window_h"].eq(dw_h))]
        if not df_lfp_cell.empty:
            best = champion_lfp_only(df_lfp_cell)
            if best is not None:
                sub = df_lfp_cell[df_lfp_cell["duration_h"].eq(best)]
                cur = build_province_curves(
                    sub.rename(columns={"capacity_mwh": "_dummy"}))
                plot_thin_lines(ax, cur, LFP_COLOR, 0.10)
                mat = interpolate_to_grid(cur, x_grid)
                med = plot_median(ax, mat, x_grid, LFP_COLOR, ls="--",
                                    lw=2.0,
                                    label=f"LFP {best}h only",
                                    marker="s", ms=3.5)
                if med is not None:
                    med = med.assign(combo=f"LFP{best}h only",
                                      node=node, dispatch_window_h=dw_h)
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
    ax.set_title(f"{panel_letter}  {node} | dw={int(dw_h)}h",
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


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    print("=== 加载 7 LDES + LFP 数据 ===")
    ldes_dfs = load_ldes_techs()
    lfp_df = load_lfp_only()
    if not ldes_dfs:
        raise SystemExit("没有 plan A hybrid CSV，先跑 "
                          "m4_progression_planA_hybrid_per_tech.py")

    fig = plt.figure(figsize=(18.0, 16.5))
    gs = fig.add_gridspec(4, 4,
                            left=0.050, right=0.985,
                            top=0.945, bottom=0.045,
                            wspace=0.22, hspace=0.30)

    source_rows = []
    champ_summary = []
    letters = "abcdefghijklmnop"
    idx = 0
    for r, node in enumerate(NODES):
        for c, dw_h in enumerate(DW_VALUES):
            ax = fig.add_subplot(gs[r, c])
            sd, champs = draw_cell(
                ax, ldes_dfs, lfp_df, node, dw_h, letters[idx],
                show_xlabel=(r == 3),
                show_ylabel=(c == 0),
                show_legend=True,
            )
            if not sd.empty:
                source_rows.append(sd)
            champ_summary.append({
                "node": node, "dw_h": dw_h,
                "lfp_only_h": champs.get("lfp_only_h"),
                **{tid: f"LFP{a}+{LDES_LABELS[tid]}{b}h"
                    for tid, val in champs.items()
                    if tid in LDES_LABELS
                    for a, b in [val]},
            })
            idx += 1

    fig.text(0.050, 0.985,
              "Fig 5 (Plan A hybrid)  Decarbonization progression × dispatch-"
              "window — LDES-priority dispatch (LDES first discharge & charge), "
              "LFP backup",
              ha="left", va="top",
              fontsize=12, fontweight="bold", color=PALETTE["black"])
    fig.text(0.050, 0.967,
              "Rows: 4 progression nodes (demand×1.0/1.4/1.8/2.5 + extreme "
              "escalation). Columns: 4 dispatch windows (8/12/16/24 h). "
              "Each curve = (LFP_dur, LDES_dur) champion at satisfy>=70%, "
              "linearly interpolated then 26-province median (panel a logic). "
              "Red dashed = gas parity. LFP capacity anchors: 6, LDES anchors: "
              "20 (5/10/.../95/99%).",
              ha="left", va="top",
              fontsize=7.4, color=PALETTE["black"])

    for ext, kwargs in {".svg": {}, ".pdf": {}, ".png": {"dpi": 300}}.items():
        path = FIG_BASE.with_suffix(ext)
        fig.savefig(path, bbox_inches="tight", **kwargs)
        print(f"  saved: {path}")

    if source_rows:
        sd = pd.concat(source_rows, ignore_index=True)
        sd.to_csv(SOURCE_DIR / "fig5_progression_planA_hybrid_curves.csv",
                    index=False)
        print(f"  source data: {len(sd)} rows")

    if champ_summary:
        cs = pd.DataFrame(champ_summary)
        cs.to_csv(SOURCE_DIR / "fig5_progression_planA_hybrid_champions.csv",
                    index=False)
        print("\n=== Plan A hybrid champion (LFP_dur, LDES_dur) per (node, dw) ===")
        print(cs.to_string(index=False))


if __name__ == "__main__":
    main()
