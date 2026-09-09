"""
Fig 5-progression — 4×4 矩阵图：
  行 = 脱碳进程节点 Y1 / Y3 / Y5 / Y10
  列 = 放电窗口 dw = 8 / 12 / 16 / 24 h

每个子图沿用 fig4_hybrid_dw_sweep 的 panel a 风格：
  横轴：Peaking supply ratio (%)
  纵轴：Unit cost / gas cost (log)
  每条曲线：一种 LDES 技术在该 (node, dw) 下的冠军 (lfp_dur, ldes_dur)
  加 LFP-only 冠军作为基线对照

数据：output/m4_economics/m4_progression_results.csv
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
RESULTS_CSV = M4_DIR / "m4_progression_results.csv"
FIG_BASE = FIGURE_DIR / "fig5_progression_node_dw"

NODES = ["Y1", "Y3", "Y5", "Y10"]
NODE_LABELS = {
    "Y1":  "Y1  (baseline, demand×1.0)",
    "Y3":  "Y3  (中度脱碳, demand×1.4)",
    "Y5":  "Y5  (深度脱碳, demand×1.8)",
    "Y10": "Y10 (高度脱碳, demand×2.5)",
}
DW_VALUES = [8.0, 12.0, 16.0, 24.0]
SCENARIO = "baseline"
CHAMP_SATISFY_MIN = 70.0

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
LFP_ONLY_COLOR = "#2F78A8"

PALETTE = {
    "fill_blue": "#D9E6F0",
    "fill_red": "#F4D9D8",
    "grid": "#E8E8E8",
    "red": "#9B3036",
    "black": "#262626",
    "grey": "#666666",
}


# ========= 冠军时长识别（每个 cell 独立）=========

def identify_champions_for_cell(df_cell: pd.DataFrame,
                                  satisfy_min: float = CHAMP_SATISFY_MIN
                                  ) -> tuple[dict, int]:
    sub = df_cell[df_cell["satisfy_rate_pct"].ge(satisfy_min)]
    champs = {}
    for tid in LDES_LABELS:
        s = sub[(sub["ldes_tech"].eq(tid))
                 & (sub["lfp_capacity_mwh"].gt(0))
                 & (sub["ldes_capacity_mwh"].gt(0))]
        if s.empty:
            continue
        med = (s.groupby(["lfp_duration_h", "ldes_duration_h"])
                  ["unit_cost_cny_kwh"].median()
                  .reset_index()
                  .sort_values("unit_cost_cny_kwh"))
        if not med.empty:
            row = med.iloc[0]
            champs[tid] = (int(row["lfp_duration_h"]),
                            int(row["ldes_duration_h"]))

    lfp_only = sub[(sub["ldes_capacity_mwh"].eq(0))
                    & (sub["lfp_capacity_mwh"].gt(0))]
    lfp_best = 8
    if not lfp_only.empty:
        med = (lfp_only.groupby("lfp_duration_h")
                          ["unit_cost_cny_kwh"].median().sort_values())
        if not med.empty:
            lfp_best = int(med.index[0])
    return champs, lfp_best


# ========= 曲线工具（沿用 dw_sweep 实现）=========

def lfp_only_subset(df, lfp_dur):
    return df[df["ldes_capacity_mwh"].eq(0)
                & df["lfp_capacity_mwh"].gt(0)
                & df["lfp_duration_h"].eq(lfp_dur)]


def hybrid_subset(df, ldes_tech, lfp_dur, ldes_dur):
    return df[df["ldes_tech"].eq(ldes_tech)
                & df["ldes_capacity_mwh"].gt(0)
                & df["lfp_capacity_mwh"].gt(0)
                & df["lfp_duration_h"].eq(lfp_dur)
                & df["ldes_duration_h"].eq(ldes_dur)]


def build_province_curves(sub, x_col, y_col, smaller_better=True):
    curves = {}
    for prov, g in sub.groupby("province_cn", sort=False):
        g = g[[x_col, y_col]].dropna()
        if g.empty:
            continue
        agg = "min" if smaller_better else "max"
        g = (g.groupby(x_col, as_index=False)[y_col]
                .agg(agg).sort_values(x_col).reset_index(drop=True))
        curves[prov] = g
    return curves


def interpolate_to_grid(curves, x_col, y_col, x_grid):
    rows = []
    for _prov, g in curves.items():
        if len(g) < 2:
            continue
        x = g[x_col].to_numpy(dtype=float)
        y = g[y_col].to_numpy(dtype=float)
        x_u, idx = np.unique(x, return_index=True)
        y_u = y[idx]
        if len(x_u) < 2:
            continue
        interp = np.full_like(x_grid, np.nan, dtype=float)
        m = (x_grid >= x_u.min()) & (x_grid <= x_u.max())
        interp[m] = np.interp(x_grid[m], x_u, y_u)
        rows.append(interp)
    return np.vstack(rows) if rows else np.empty((0, len(x_grid)))


def plot_thin_lines(ax, curves, x_col, y_col, color, alpha=0.08):
    for _prov, g in curves.items():
        ax.plot(g[x_col], g[y_col], color=color, lw=0.5, alpha=alpha,
                 zorder=1)


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


# ========= 单元格绘制 =========

def draw_cell(ax, df_cell, node, dw_h, panel_letter,
                show_xlabel=True, show_ylabel=True,
                show_legend=True):
    champs, lfp_best = identify_champions_for_cell(df_cell)
    x_grid = np.linspace(0, 100, 241)
    rows = []

    # LFP-only 基线
    lfp = lfp_only_subset(df_cell, lfp_best)
    if not lfp.empty:
        cur = build_province_curves(
            lfp, "satisfy_rate_pct", "cost_ratio_to_gas", True)
        plot_thin_lines(ax, cur, "satisfy_rate_pct",
                          "cost_ratio_to_gas", LFP_ONLY_COLOR, 0.10)
        mat = interpolate_to_grid(cur, "satisfy_rate_pct",
                                     "cost_ratio_to_gas", x_grid)
        med = plot_median(ax, mat, x_grid, LFP_ONLY_COLOR, ls="--", lw=2.0,
                            label=f"LFP {lfp_best}h only",
                            marker="s", ms=3.5)
        if med is not None:
            med = med.assign(combo=f"LFP {lfp_best}h only",
                              node=node, dispatch_window_h=dw_h)
            rows.append(med)

    for tid in LDES_LABELS:
        if tid not in champs:
            continue
        lfp_d, ldes_d = champs[tid]
        sub = hybrid_subset(df_cell, tid, lfp_d, ldes_d)
        if sub.empty:
            continue
        cur = build_province_curves(
            sub, "satisfy_rate_pct", "cost_ratio_to_gas", True)
        plot_thin_lines(ax, cur, "satisfy_rate_pct",
                          "cost_ratio_to_gas", TECH_COLORS[tid], 0.06)
        mat = interpolate_to_grid(cur, "satisfy_rate_pct",
                                     "cost_ratio_to_gas", x_grid)
        med = plot_median(ax, mat, x_grid, TECH_COLORS[tid], lw=1.5,
                            label=f"LFP{lfp_d}+{LDES_LABELS[tid]}{ldes_d}h",
                            marker="o", ms=3.0)
        if med is not None:
            med = med.assign(
                combo=f"LFP{lfp_d}+{LDES_LABELS[tid]}{ldes_d}h",
                node=node, dispatch_window_h=dw_h)
            rows.append(med)

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
        return pd.concat(rows, ignore_index=True), champs, lfp_best
    return pd.DataFrame(), champs, lfp_best


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {RESULTS_CSV}")
    df = pd.read_csv(RESULTS_CSV)
    df = df[df["scenario"].eq(SCENARIO)].copy()
    print(f"  {len(df)} rows ({SCENARIO}); "
          f"nodes: {sorted(df['node'].unique())}, "
          f"dws: {sorted(df['dispatch_window_h'].unique())}")

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
            df_cell = df[(df["node"] == node)
                          & (df["dispatch_window_h"].eq(dw_h))]
            if df_cell.empty:
                ax.set_visible(False)
                idx += 1
                continue
            sd, champs, lfp_best = draw_cell(
                ax, df_cell, node, dw_h, letters[idx],
                show_xlabel=(r == 3),
                show_ylabel=(c == 0),
                show_legend=True,
            )
            if not sd.empty:
                source_rows.append(sd)
            champ_summary.append({
                "node": node, "dw_h": dw_h,
                "lfp_only_best_h": lfp_best,
                **{tid: f"LFP{a}h+{LDES_LABELS[tid]}{b}h"
                    for tid, (a, b) in champs.items()},
            })
            idx += 1

    fig.text(0.050, 0.985,
              "Fig 5  Decarbonization progression × dispatch-window sweep "
              "(baseline cost, 5-yr augmented demand)",
              ha="left", va="top",
              fontsize=12, fontweight="bold", color=PALETTE["black"])
    fig.text(0.050, 0.967,
              "Rows: 4 progression nodes (demand×1.0, ×1.4, ×1.8, ×2.5 + "
              "extreme escalation). Columns: 4 dispatch windows (8/12/16/24 h). "
              "Each cell re-selects champion (LFP_dur, LDES_dur) per technology. "
              "Curves: 26-province median; thin lines: provinces. "
              "Red dashed = gas parity.",
              ha="left", va="top",
              fontsize=7.4, color=PALETTE["black"])

    for ext, kwargs in {".svg": {}, ".pdf": {}, ".png": {"dpi": 300}}.items():
        path = FIG_BASE.with_suffix(ext)
        fig.savefig(path, bbox_inches="tight", **kwargs)
        print(f"  saved: {path}")

    if source_rows:
        sd = pd.concat(source_rows, ignore_index=True)
        sd.to_csv(SOURCE_DIR / "fig5_progression_node_dw_curves.csv",
                    index=False)
        print(f"  source data: {len(sd)} rows")

    if champ_summary:
        cs = pd.DataFrame(champ_summary)
        cs.to_csv(SOURCE_DIR / "fig5_progression_node_dw_champions.csv",
                    index=False)
        print("\n=== Champions per (node, dw) ===")
        print(cs.to_string(index=False))


if __name__ == "__main__":
    main()
