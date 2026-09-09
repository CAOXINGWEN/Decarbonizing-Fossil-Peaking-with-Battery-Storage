"""
Fig 5-progression (single-tech) — 4×4 矩阵图：8 条曲线（LFP-only + 7 LDES-only）。

与 fig4_m4_economics_nature_2yr panel a 完全同款逻辑：
  - 每条曲线 = 单一冠军 duration 下，逐省按 satisfy_rate 升序线性插值
    到 0-100% 网格，再对 26 省取中位
  - 不做 cummin、不做滚动平滑
  - 数据源：m4_progression_single_tech_sweep.py 的全量扫描结果
    （每 tech 一个 CSV，step=25 MWh）

数据源：output/m4_economics/m4_single_tech_<tech_id>.csv
  每文件含: node, scenario, dispatch_window_h, tech_id, duration_h,
            province_en, province_cn, capacity_mwh, satisfy_rate_pct,
            unit_cost_cny_kwh, cost_ratio_to_gas, ...

输出：output/m4_economics/nature_figure/fig5_progression_single_tech_node_dw.{png,svg,pdf}
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
FIG_BASE = FIGURE_DIR / "fig5_progression_single_tech_node_dw"

NODES = ["Y1", "Y3", "Y5", "Y10"]
DW_VALUES = [8.0, 12.0, 16.0, 24.0]
SCENARIO = "baseline"
CHAMP_SATISFY_MIN = 70.0

# 8 个 single-tech 数据源（每文件单独）
TECH_FILES = {
    "li_lfp": "m4_single_tech_li_lfp.csv",
    "caes": "m4_single_tech_caes.csv",
    "laes": "m4_single_tech_laes.csv",
    "gravity": "m4_single_tech_gravity.csv",
    "fecr_flow": "m4_single_tech_fecr_flow.csv",
    "molten_salt_tes": "m4_single_tech_molten_salt_tes.csv",
    "vrfb": "m4_single_tech_vrfb.csv",
    "h2_cavern": "m4_single_tech_h2_cavern.csv",
}
TECH_LABELS = {
    "li_lfp": "LFP",
    "caes": "CAES",
    "laes": "LAES",
    "gravity": "Gravity",
    "fecr_flow": "FeCr flow",
    "molten_salt_tes": "Molten salt TES",
    "vrfb": "VRFB",
    "h2_cavern": "H2 cavern",
}
TECH_COLORS = {
    "li_lfp": "#2F78A8",
    "caes": "#16456A",
    "laes": "#C77C2E",
    "gravity": "#3E8B72",
    "fecr_flow": "#4E8B95",
    "molten_salt_tes": "#8A8A8A",
    "vrfb": "#A05FB5",
    "h2_cavern": "#7B68B3",
}

PALETTE = {
    "fill_blue": "#D9E6F0",
    "fill_red": "#F4D9D8",
    "grid": "#E8E8E8",
    "red": "#9B3036",
    "black": "#262626",
    "grey": "#666666",
}


def load_all_techs() -> dict[str, pd.DataFrame]:
    """加载 8 个 single-tech CSV 到 dict[tech_id -> DataFrame]。

    只读必要列以控制内存（每文件可能数 GB）。
    """
    needed_cols = [
        "scenario", "node", "dispatch_window_h", "tech_id",
        "duration_h", "province_en", "province_cn",
        "capacity_mwh", "satisfy_rate_pct",
        "unit_cost_cny_kwh", "cost_ratio_to_gas",
    ]
    out = {}
    for tid, fname in TECH_FILES.items():
        fp = M4_DIR / fname
        if not fp.exists():
            print(f"  WARNING: 缺少 {fp.name}，跳过 {tid}")
            continue
        df = pd.read_csv(fp, usecols=needed_cols, low_memory=False)
        df = df[df["scenario"].eq(SCENARIO)].copy()
        out[tid] = df
        print(f"  {tid}: {len(df):,} rows")
    return out


def champion_duration(df_cell: pd.DataFrame,
                       satisfy_min: float = CHAMP_SATISFY_MIN) -> int | None:
    """在该 cell 内挑选冠军 duration：所有省 satisfy>=70% 时单位成本中位最低。"""
    sub = df_cell[df_cell["satisfy_rate_pct"].ge(satisfy_min)]
    if sub.empty:
        return None
    med = (sub.groupby("duration_h")["unit_cost_cny_kwh"]
              .median().sort_values())
    return int(med.index[0]) if not med.empty else None


def build_province_curves(sub: pd.DataFrame) -> dict[str, pd.DataFrame]:
    curves = {}
    for prov, g in sub.groupby("province_cn", sort=False):
        g = g[["satisfy_rate_pct", "cost_ratio_to_gas"]].dropna()
        if g.empty:
            continue
        g = (g.sort_values("satisfy_rate_pct").reset_index(drop=True))
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


def draw_cell(ax, tech_dfs: dict[str, pd.DataFrame],
                node, dw_h, panel_letter,
                show_xlabel=True, show_ylabel=True,
                show_legend=True):
    x_grid = np.linspace(0, 100, 241)
    rows = []
    champs = {}

    for tid in TECH_LABELS:
        if tid not in tech_dfs:
            continue
        df_full = tech_dfs[tid]
        df_cell = df_full[(df_full["node"] == node)
                           & (df_full["dispatch_window_h"].eq(dw_h))]
        if df_cell.empty:
            continue
        best = champion_duration(df_cell)
        if best is None:
            continue
        sub = df_cell[df_cell["duration_h"].eq(best)]
        if sub.empty:
            continue
        cur = build_province_curves(sub)
        plot_thin_lines(ax, cur, TECH_COLORS[tid],
                          0.10 if tid == "li_lfp" else 0.06)
        mat = interpolate_to_grid(cur, x_grid)
        ls = "--" if tid == "li_lfp" else "-"
        lw = 2.0 if tid == "li_lfp" else 1.5
        marker = "s" if tid == "li_lfp" else "o"
        ms = 3.5 if tid == "li_lfp" else 3.0
        med = plot_median(ax, mat, x_grid, TECH_COLORS[tid], ls=ls,
                            lw=lw, label=f"{TECH_LABELS[tid]} {best}h only",
                            marker=marker, ms=ms)
        if med is not None:
            med = med.assign(
                combo=f"{TECH_LABELS[tid]}{best}h",
                node=node, dispatch_window_h=dw_h)
            rows.append(med)
        champs[tid] = best

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

    print("=== 加载 8 个 single-tech CSV ===")
    tech_dfs = load_all_techs()
    if not tech_dfs:
        raise SystemExit("没有找到任何 single-tech CSV，先跑 "
                          "m4_progression_single_tech_sweep.py")

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
                ax, tech_dfs, node, dw_h, letters[idx],
                show_xlabel=(r == 3),
                show_ylabel=(c == 0),
                show_legend=True,
            )
            if not sd.empty:
                source_rows.append(sd)
            champ_summary.append({"node": node, "dw_h": dw_h, **champs})
            idx += 1

    fig.text(0.050, 0.985,
              "Fig 5 (Single-tech, full capacity sweep)  Decarbonization "
              "progression × dispatch-window — each curve is one technology "
              "operating alone, step=25 MWh",
              ha="left", va="top",
              fontsize=12, fontweight="bold", color=PALETTE["black"])
    fig.text(0.050, 0.967,
              "Rows: 4 progression nodes (demand×1.0/1.4/1.8/2.5 + extreme "
              "escalation). Columns: 4 dispatch windows (8/12/16/24 h). "
              "8 curves: LFP-only champion duration + 7 LDES-only champion "
              "durations. Province-level curves linearly interpolated then "
              "26-province median (panel a same logic). Red dashed = gas parity.",
              ha="left", va="top",
              fontsize=7.4, color=PALETTE["black"])

    for ext, kwargs in {".svg": {}, ".pdf": {}, ".png": {"dpi": 300}}.items():
        path = FIG_BASE.with_suffix(ext)
        fig.savefig(path, bbox_inches="tight", **kwargs)
        print(f"  saved: {path}")

    if source_rows:
        sd = pd.concat(source_rows, ignore_index=True)
        sd.to_csv(SOURCE_DIR / "fig5_progression_single_tech_curves.csv",
                    index=False)
        print(f"  source data: {len(sd)} rows")

    if champ_summary:
        cs = pd.DataFrame(champ_summary)
        cs.to_csv(SOURCE_DIR / "fig5_progression_single_tech_champions.csv",
                    index=False)
        print("\n=== Single-tech champion durations per (node, dw) ===")
        print(cs.to_string(index=False))


if __name__ == "__main__":
    main()
