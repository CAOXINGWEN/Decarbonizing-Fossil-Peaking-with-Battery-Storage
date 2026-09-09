"""
Fig 4-DW — 4 子图 (panel a 风格) 展示放电窗口 dw 灵敏度。

横轴：Peaking supply ratio (%)
纵轴：Unit cost / gas cost (log)

每个 dw (8h / 12h / 16h / 24h) 一个子图，子图内每条曲线一种 LDES 技术
（在该 dw 下重新挑选冠军 (lfp_dur, ldes_dur)），加 LFP-only 冠军时长基线。

数据：output/m4_economics/m4_hybrid_results_2yr_dwsweep.csv
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
plt.rcParams["font.size"] = 8.0
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.9
plt.rcParams["legend.frameon"] = False


PAPER_CODE_DIR = Path(__file__).resolve().parent
M4_DIR = PAPER_CODE_DIR / "output" / "m4_economics"
FIGURE_DIR = M4_DIR / "nature_figure"
SOURCE_DIR = FIGURE_DIR / "source_data"
RESULTS_CSV = M4_DIR / "m4_hybrid_results_2yr_dwsweep.csv"
FIG_BASE = FIGURE_DIR / "fig4_hybrid_dw_sweep"

DW_VALUES = [8.0, 12.0, 16.0, 24.0]
PANEL_LETTERS = ["a", "b", "c", "d"]
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


# ========= 冠军时长识别（每个 dw 独立）=========

def identify_champions_for_dw(df_dw: pd.DataFrame,
                                satisfy_min: float = CHAMP_SATISFY_MIN
                                ) -> tuple[dict[str, tuple[int, int]], int]:
    """对单一 dw 的子集，挑各 LDES 技术冠军 (lfp_dur, ldes_dur) + LFP-only 冠军 dur。

    判据：satisfy >= satisfy_min 子集内，按 (lfp_dur, ldes_dur) 分组取省级中位
    unit_cost_cny_kwh，最低者获胜。
    """
    sub = df_dw[df_dw["satisfy_rate_pct"].ge(satisfy_min)]
    champs: dict[str, tuple[int, int]] = {}
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
                          ["unit_cost_cny_kwh"].median()
                          .sort_values())
        if not med.empty:
            lfp_best = int(med.index[0])
    return champs, lfp_best


# ========= 曲线工具 =========

def lfp_only_subset(df: pd.DataFrame, lfp_dur: int) -> pd.DataFrame:
    return df[df["ldes_capacity_mwh"].eq(0)
                & df["lfp_capacity_mwh"].gt(0)
                & df["lfp_duration_h"].eq(lfp_dur)]


def hybrid_subset(df: pd.DataFrame, ldes_tech: str,
                    lfp_dur: int, ldes_dur: int) -> pd.DataFrame:
    return df[df["ldes_tech"].eq(ldes_tech)
                & df["ldes_capacity_mwh"].gt(0)
                & df["lfp_capacity_mwh"].gt(0)
                & df["lfp_duration_h"].eq(lfp_dur)
                & df["ldes_duration_h"].eq(ldes_dur)]


def build_province_curves(sub: pd.DataFrame, x_col: str, y_col: str,
                            smaller_better: bool = True
                            ) -> dict[str, pd.DataFrame]:
    """每省按 x 升序保留 (x, y) 曲线，不做累积单调化。

    与 m4_visualize_hybrid_2yr.py 内同名函数对齐。
    """
    curves = {}
    for prov, g in sub.groupby("province_cn", sort=False):
        g = g[[x_col, y_col]].dropna()
        if g.empty:
            continue
        agg = "min" if smaller_better else "max"
        g = (g.groupby(x_col, as_index=False)[y_col]
                .agg(agg)
                .sort_values(x_col)
                .reset_index(drop=True))
        curves[prov] = g
    return curves


def interpolate_to_grid(curves: dict[str, pd.DataFrame], x_col: str,
                          y_col: str, x_grid: np.ndarray) -> np.ndarray:
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


def add_panel_label(ax, label: str):
    ax.text(-0.10, 1.05, label, transform=ax.transAxes,
             ha="left", va="bottom",
             fontsize=12, fontweight="bold", color=PALETTE["black"])


def plot_thin_lines(ax, curves, x_col, y_col, color, alpha=0.10):
    for _prov, g in curves.items():
        ax.plot(g[x_col], g[y_col], color=color, lw=0.55, alpha=alpha,
                 zorder=1)


def plot_median(ax, mat, x_grid, color, ls="-", lw=2.0, label=None,
                 marker=None, ms=0):
    if mat.size == 0:
        return None
    med = np.nanmedian(mat, axis=0)
    valid = ~np.isnan(med)
    if not valid.any():
        return None
    kw = dict(color=color, lw=lw, ls=ls, zorder=3, label=label)
    if marker is not None:
        kw.update(marker=marker, ms=ms, markevery=20)
    ax.plot(x_grid[valid], med[valid], **kw)
    return pd.DataFrame({"supply_pct": x_grid[valid],
                          "median_cost_ratio": med[valid]})


# ========= 单 dw 子图 =========

def draw_dw_subplot(ax, df_dw, dw_h: float, panel_letter: str,
                      satisfy_min: float = CHAMP_SATISFY_MIN
                      ) -> pd.DataFrame:
    """画一个 dw 的 panel a 风格子图。返回该子图的 source data DataFrame。"""
    champs, lfp_best = identify_champions_for_dw(df_dw, satisfy_min)
    x_grid = np.linspace(0, 100, 241)

    rows = []

    # LFP-only 冠军
    lfp = lfp_only_subset(df_dw, lfp_best)
    if not lfp.empty:
        cur = build_province_curves(
            lfp, "satisfy_rate_pct", "cost_ratio_to_gas", True)
        plot_thin_lines(ax, cur, "satisfy_rate_pct",
                          "cost_ratio_to_gas", LFP_ONLY_COLOR, 0.10)
        mat = interpolate_to_grid(cur, "satisfy_rate_pct",
                                     "cost_ratio_to_gas", x_grid)
        med = plot_median(ax, mat, x_grid, LFP_ONLY_COLOR, ls="--", lw=2.4,
                            label=f"LFP {lfp_best}h only",
                            marker="s", ms=4)
        if med is not None:
            med = med.assign(combo=f"LFP {lfp_best}h only",
                              dispatch_window_h=dw_h)
            rows.append(med)

    # 各 LDES 冠军组合
    for tid in LDES_LABELS:
        if tid not in champs:
            continue
        lfp_d, ldes_d = champs[tid]
        sub = hybrid_subset(df_dw, tid, lfp_d, ldes_d)
        if sub.empty:
            continue
        cur = build_province_curves(
            sub, "satisfy_rate_pct", "cost_ratio_to_gas", True)
        plot_thin_lines(ax, cur, "satisfy_rate_pct",
                          "cost_ratio_to_gas", TECH_COLORS[tid], 0.08)
        mat = interpolate_to_grid(cur, "satisfy_rate_pct",
                                     "cost_ratio_to_gas", x_grid)
        med = plot_median(ax, mat, x_grid, TECH_COLORS[tid], lw=1.8,
                            label=f"LFP {lfp_d}h + {LDES_LABELS[tid]} {ldes_d}h",
                            marker="o", ms=3.5)
        if med is not None:
            med = med.assign(
                combo=f"LFP {lfp_d}h + {LDES_LABELS[tid]} {ldes_d}h",
                dispatch_window_h=dw_h)
            rows.append(med)

    ax.axhline(1.0, color=PALETTE["red"], lw=1.0, ls="--")
    ax.fill_between([0, 100], 0.1, 1.0, color=PALETTE["fill_blue"],
                      alpha=0.15, zorder=0)
    ax.fill_between([0, 100], 1.0, 50, color=PALETTE["fill_red"],
                      alpha=0.10, zorder=0)
    ax.set_yscale("log")
    ax.set_xlim(0, 100); ax.set_ylim(0.20, 12)
    ax.set_xlabel("Peaking supply ratio (%)", labelpad=2)
    ax.set_ylabel("Unit cost / gas cost", labelpad=2)
    ax.set_title(f"{panel_letter}  dw = {int(dw_h)}h  (champion durations)",
                  loc="left", fontsize=9.4, pad=4)
    ax.grid(axis="both", which="major", color=PALETTE["grid"], lw=0.6)
    ax.text(99, 1.05, "gas parity", ha="right", va="bottom",
             fontsize=6.8, color=PALETTE["red"])
    ax.legend(loc="upper left", fontsize=6.5, handlelength=1.6,
               borderaxespad=0.2, ncol=1, columnspacing=0.7,
               handletextpad=0.5)
    add_panel_label(ax, panel_letter)

    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame()


# ========= main =========

def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {RESULTS_CSV}")
    df = pd.read_csv(RESULTS_CSV)
    df = df[df["scenario"].eq(SCENARIO)].copy()
    print(f"  {len(df)} rows ({SCENARIO}); "
          f"dws: {sorted(df['dispatch_window_h'].unique())}")

    fig = plt.figure(figsize=(14.0, 11.5))
    gs = fig.add_gridspec(2, 2,
                            left=0.060, right=0.985,
                            top=0.93, bottom=0.060,
                            wspace=0.26, hspace=0.36)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
             fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]

    source_rows = []
    champ_summary = []

    for ax, dw_h, letter in zip(axes, DW_VALUES, PANEL_LETTERS):
        df_dw = df[df["dispatch_window_h"].eq(dw_h)]
        if df_dw.empty:
            ax.set_visible(False)
            continue
        sd = draw_dw_subplot(ax, df_dw, dw_h, letter)
        if not sd.empty:
            source_rows.append(sd)

        champs, lfp_best = identify_champions_for_dw(df_dw)
        champ_summary.append({"dw_h": dw_h,
                                "lfp_only_best_h": lfp_best,
                                **{f"{tid}": f"LFP{a}h+{LDES_LABELS[tid]}{b}h"
                                    for tid, (a, b) in champs.items()}})

    fig.text(0.060, 0.978,
              "M4 dispatch-window sensitivity (2yr augmented, baseline cost): "
              "LFP+LDES champion combinations vs LFP-only",
              ha="left", va="top",
              fontsize=11, fontweight="bold", color=PALETTE["black"])
    fig.text(0.060, 0.960,
              "Each panel re-selects the (LFP_dur, LDES_dur) champion pair for "
              "that dispatch window. Curves: 26-province median; thin shadows: "
              "individual provinces. Red dashed line = gas parity.",
              ha="left", va="top",
              fontsize=7.8, color=PALETTE["black"])

    for ext, kwargs in {".svg": {}, ".pdf": {}, ".png": {"dpi": 320}}.items():
        path = FIG_BASE.with_suffix(ext)
        fig.savefig(path, bbox_inches="tight", **kwargs)
        print(f"  saved: {path}")

    if source_rows:
        sd = pd.concat(source_rows, ignore_index=True)
        sd.to_csv(SOURCE_DIR / "fig4_hybrid_dw_sweep_curves.csv", index=False)
        print(f"  source data: {len(sd)} rows")

    if champ_summary:
        cs = pd.DataFrame(champ_summary)
        cs.to_csv(SOURCE_DIR / "fig4_hybrid_dw_sweep_champions.csv",
                    index=False)
        print("\n=== Champions per dw ===")
        print(cs.to_string(index=False))


if __name__ == "__main__":
    main()
