"""
Fig 4 (2yr Hybrid v3) — Nature-style 4 面板：LFP 基线 vs LFP+LDES 多技术组合。

v3 关键改进 vs v2:
  1) 每个 LDES 技术先确定"冠军时长组合 (LFP-dur, LDES-dur)"——baseline 下
     满足率 ≥70% 的省级中位单位成本最低的那一对时长。
  2) panel a/b/c 用冠军时长扫容量（LFP-cap × LDES-cap）画前沿，图例显式写明
     "LFP 8h + CAES 12h" 等。
  3) panel c 因为单一时长下的容量扫描是单调的，sat-cost 微分计算稳定，曲线
     不再为空。
  4) LFP-only 也找冠军时长（baseline 下 LFP-only 单位成本最低的那个 dur）。
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
M4_DIR = PAPER_CODE_DIR / "output" / "m4_economics"
FIGURE_DIR = M4_DIR / "nature_figure"
SOURCE_DIR = FIGURE_DIR / "source_data"
RESULTS_CSV = M4_DIR / "m4_hybrid_results_2yr.csv"
FIG_BASE = FIGURE_DIR / "fig4_hybrid_2yr"

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

SCENARIO_LABELS = {"baseline": "Baseline", "low_cost": "Low cost"}
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

CHAMP_SATISFY_MIN = 70.0


# ========= 冠军时长识别 =========

def identify_champion_durations(df: pd.DataFrame, scenario: str = "baseline",
                                  satisfy_min: float = CHAMP_SATISFY_MIN
                                  ) -> tuple[dict[str, tuple[int, int]], int]:
    """对每个 LDES 技术，找 (lfp_dur, ldes_dur) 使省级中位单位成本最低。
    返回 ({tid: (lfp_dur, ldes_dur)}, lfp_only_best_dur)。"""
    sub = df[(df["scenario"].eq(scenario))
              & (df["satisfy_rate_pct"] >= satisfy_min)
              & (df["lfp_capacity_mwh"] > 0)
              & (df["ldes_capacity_mwh"] > 0)]
    champ = (sub.groupby(["ldes_tech", "lfp_duration_h",
                            "ldes_duration_h"])["unit_cost_cny_kwh"]
                .median().reset_index()
                .sort_values("unit_cost_cny_kwh"))
    best_per_tech = champ.groupby("ldes_tech", as_index=False).first()
    out: dict[str, tuple[int, int]] = {}
    for _, r in best_per_tech.iterrows():
        out[r["ldes_tech"]] = (int(r["lfp_duration_h"]),
                                int(r["ldes_duration_h"]))

    sub_lfp = df[(df["scenario"].eq(scenario))
                  & (df["satisfy_rate_pct"] >= satisfy_min)
                  & (df["lfp_capacity_mwh"] > 0)
                  & (df["ldes_capacity_mwh"].eq(0))]
    lfp_med = (sub_lfp.groupby("lfp_duration_h")["unit_cost_cny_kwh"]
                       .median().sort_values())
    lfp_best = int(lfp_med.index[0]) if not lfp_med.empty else 8
    return out, lfp_best


# ========= 数据子集 =========

def lfp_only_subset(df: pd.DataFrame, lfp_dur: int) -> pd.DataFrame:
    """LFP-only 冠军时长：ldes_capacity=0 且 lfp_duration=lfp_dur。"""
    sub = df[df["ldes_capacity_mwh"].eq(0)
              & df["lfp_capacity_mwh"].gt(0)
              & df["lfp_duration_h"].eq(lfp_dur)]
    return sub


def hybrid_subset(df: pd.DataFrame, ldes_tech: str,
                    lfp_dur: int, ldes_dur: int) -> pd.DataFrame:
    """指定 (LDES 技术, LFP 时长, LDES 时长) 的容量扫描子集。"""
    return df[df["ldes_tech"].eq(ldes_tech)
                & df["ldes_capacity_mwh"].gt(0)
                & df["lfp_capacity_mwh"].gt(0)
                & df["lfp_duration_h"].eq(lfp_dur)
                & df["ldes_duration_h"].eq(ldes_dur)]


def build_province_frontiers(sub: pd.DataFrame, x_col: str, y_col: str,
                               smaller_better: bool = True
                               ) -> dict[str, pd.DataFrame]:
    """对每省按 x 升序构建 y 累积 min/max 前沿。"""
    frontiers = {}
    for prov, g in sub.groupby("province_cn", sort=False):
        g = g[[x_col, y_col]].dropna()
        if g.empty:
            continue
        g = g.sort_values(x_col).reset_index(drop=True)
        if smaller_better:
            g[y_col] = g[y_col].cummin()
        else:
            g[y_col] = g[y_col].cummax()
        frontiers[prov] = g
    return frontiers


def build_province_curves(sub: pd.DataFrame, x_col: str, y_col: str,
                            smaller_better: bool = True
                            ) -> dict[str, pd.DataFrame]:
    """每省按 x 升序保留原始 (x, y) 曲线，不做累积单调化。

    同一 x 上有多点（不同容量组合落到同一 satisfy）时，按 smaller_better 取
    min/max 聚合——这与 fig4_m4_economics_nature 的处理一致，保留 cost 随
    satisfy 上升的物理趋势（左低右高，穿越 gas parity）。
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


def interpolate_to_grid(frontiers: dict[str, pd.DataFrame], x_col: str,
                          y_col: str, x_grid: np.ndarray) -> np.ndarray:
    rows = []
    for prov, g in frontiers.items():
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


# ========= 绘图工具 =========

def add_panel_label(ax, label):
    ax.text(-0.10, 1.05, label, transform=ax.transAxes,
             ha="left", va="bottom",
             fontsize=12, fontweight="bold", color=PALETTE["black"])


def plot_province_thin_lines(ax, frontiers, x_col, y_col, color, alpha=0.16):
    for _prov, g in frontiers.items():
        ax.plot(g[x_col], g[y_col], color=color, lw=0.55, alpha=alpha,
                 zorder=1)


def plot_median_curve(ax, mat, x_grid, color, ls="-", lw=2.0, label=None,
                       marker=None, ms=0):
    if mat.size == 0:
        return
    med = np.nanmedian(mat, axis=0)
    valid = ~np.isnan(med)
    if not valid.any():
        return
    kw = dict(color=color, lw=lw, ls=ls, zorder=3, label=label)
    if marker is not None:
        kw.update(marker=marker, ms=ms, markevery=12)
    ax.plot(x_grid[valid], med[valid], **kw)


# ========= panel a: cost ratio =========
def draw_panel_a(ax, df_base, champs: dict[str, tuple[int, int]],
                  lfp_best: int):
    x_grid = np.linspace(0, 100, 201)

    lfp = lfp_only_subset(df_base, lfp_best)
    front_lfp = build_province_curves(
        lfp, "satisfy_rate_pct", "cost_ratio_to_gas", True)
    plot_province_thin_lines(ax, front_lfp, "satisfy_rate_pct",
                                "cost_ratio_to_gas", LFP_ONLY_COLOR, 0.10)
    mat_lfp = interpolate_to_grid(front_lfp, "satisfy_rate_pct",
                                     "cost_ratio_to_gas", x_grid)
    plot_median_curve(ax, mat_lfp, x_grid, LFP_ONLY_COLOR,
                       ls="--", lw=2.4, label=f"LFP {lfp_best}h only",
                       marker="s", ms=4)

    for tid in LDES_LABELS:
        if tid not in champs:
            continue
        lfp_d, ldes_d = champs[tid]
        sub = hybrid_subset(df_base, tid, lfp_d, ldes_d)
        if sub.empty:
            continue
        front = build_province_curves(
            sub, "satisfy_rate_pct", "cost_ratio_to_gas", True)
        plot_province_thin_lines(ax, front, "satisfy_rate_pct",
                                    "cost_ratio_to_gas", TECH_COLORS[tid], 0.10)
        mat = interpolate_to_grid(front, "satisfy_rate_pct",
                                     "cost_ratio_to_gas", x_grid)
        plot_median_curve(ax, mat, x_grid, TECH_COLORS[tid], lw=1.8,
                           label=f"LFP {lfp_d}h + {LDES_LABELS[tid]} {ldes_d}h",
                           marker="o", ms=3.5)

    ax.axhline(1.0, color=PALETTE["red"], lw=1.0, ls="--")
    ax.fill_between([0, 100], 0.1, 1.0, color=PALETTE["fill_blue"],
                      alpha=0.15, zorder=0)
    ax.fill_between([0, 100], 1.0, 50, color=PALETTE["fill_red"],
                      alpha=0.10, zorder=0)
    ax.set_yscale("log")
    ax.set_xlim(0, 100); ax.set_ylim(0.20, 12)
    ax.set_xlabel("Peaking supply ratio (%)  [2yr augmented]", labelpad=2)
    ax.set_ylabel("Unit cost / gas cost", labelpad=2)
    ax.set_title("a  Hybrid unit-cost crossover (baseline, champion durations)",
                  loc="left", fontsize=9.4, pad=4)
    ax.grid(axis="both", which="major", color=PALETTE["grid"], lw=0.6)
    ax.text(99, 1.05, "gas parity", ha="right", va="bottom",
             fontsize=6.8, color=PALETTE["red"])
    ax.legend(loc="upper left", fontsize=6.8, handlelength=1.6,
               borderaxespad=0.2, ncol=1, columnspacing=0.7,
               handletextpad=0.5)
    add_panel_label(ax, "a")


# ========= panel b: unit profit =========
def draw_panel_b(ax, df_low, champs_low: dict[str, tuple[int, int]],
                  lfp_best_low: int):
    x_grid = np.linspace(0, 100, 201)

    lfp = lfp_only_subset(df_low, lfp_best_low)
    front_lfp = build_province_curves(
        lfp, "satisfy_rate_pct",
        "unit_profit_cny_per_kwh_supply", smaller_better=False)
    plot_province_thin_lines(ax, front_lfp, "satisfy_rate_pct",
                                "unit_profit_cny_per_kwh_supply",
                                LFP_ONLY_COLOR, 0.10)
    mat_lfp = interpolate_to_grid(front_lfp, "satisfy_rate_pct",
                                     "unit_profit_cny_per_kwh_supply", x_grid)
    plot_median_curve(ax, mat_lfp, x_grid, LFP_ONLY_COLOR,
                       ls="--", lw=2.4, label=f"LFP {lfp_best_low}h only",
                       marker="s", ms=4)

    for tid in LDES_LABELS:
        if tid not in champs_low:
            continue
        lfp_d, ldes_d = champs_low[tid]
        sub = hybrid_subset(df_low, tid, lfp_d, ldes_d)
        if sub.empty:
            continue
        front = build_province_curves(
            sub, "satisfy_rate_pct",
            "unit_profit_cny_per_kwh_supply", smaller_better=False)
        plot_province_thin_lines(ax, front, "satisfy_rate_pct",
                                    "unit_profit_cny_per_kwh_supply",
                                    TECH_COLORS[tid], 0.10)
        mat = interpolate_to_grid(front, "satisfy_rate_pct",
                                     "unit_profit_cny_per_kwh_supply", x_grid)
        plot_median_curve(ax, mat, x_grid, TECH_COLORS[tid], lw=1.8,
                           label=f"LFP {lfp_d}h + {LDES_LABELS[tid]} {ldes_d}h",
                           marker="o", ms=3.5)

    ax.axhline(0.0, color=PALETTE["red"], lw=1.0, ls="--")
    ax.fill_between([0, 100], -0.5, 0.0, color=PALETTE["fill_red"],
                      alpha=0.10, zorder=0)
    ax.fill_between([0, 100], 0.0, 0.30, color=PALETTE["fill_blue"],
                      alpha=0.15, zorder=0)
    ax.set_xlim(0, 100); ax.set_ylim(-0.5, 0.30)
    ax.set_xlabel("Peaking supply ratio (%)  [2yr augmented]", labelpad=2)
    ax.set_ylabel("Unit profit (CNY kWh$^{-1}$)", labelpad=2)
    ax.set_title("b  Profitability under low_cost (champion durations)",
                  loc="left", fontsize=9.4, pad=4)
    ax.grid(axis="both", which="major", color=PALETTE["grid"], lw=0.6)
    ax.text(99, 0.005, "zero profit", ha="right", va="bottom",
             fontsize=6.8, color=PALETTE["red"])
    ax.legend(loc="lower left", fontsize=6.8, handlelength=1.6,
               borderaxespad=0.2, ncol=1, columnspacing=0.7,
               handletextpad=0.5)
    add_panel_label(ax, "b")


# ========= 非支配 Pareto 排序工具 =========
def pareto_frontier_max_x_min_y(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """二维非支配排序：x 越大越好（supply）、y 越小越好（cost）。

    返回 boolean mask，True = 该点位于 Pareto 前沿。算法迁移自
    pareto_frontier.py:265-276 的双重循环。O(n²)，对每省 ≤30 个组合点足够。
    """
    n = len(x)
    if n == 0:
        return np.zeros(0, dtype=bool)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            weakly_better = (x[j] >= x[i]) and (y[j] <= y[i])
            strictly_better = (x[j] > x[i]) or (y[j] < y[i])
            if weakly_better and strictly_better:
                dominated[i] = True
                break
    return ~dominated


# ========= panel c: marginal cost =========
def compute_marginal_segments(sub: pd.DataFrame, group_keys) -> pd.DataFrame:
    """对每个 (scenario, province, [ldes_tech]) 组：

    1) 在 (annual_supply_kwh ↑, annual_cost_cny ↓) 平面做非支配 Pareto 排序
    2) 沿前沿按 supply 升序，相邻点差分 → marginal cost = ΔC / ΔQ
    3) 用 mid_satisfy_rate_pct 作横轴展示

    这取代了旧版用 cummin/cummax 沿 satisfy_rate 构建"伪前沿"的做法
    （cummin 立即锁定最小成本点，导致 ΔC 全为 0，曲线全部被过滤）。
    """
    rows = []
    for _, g in sub.groupby(group_keys, sort=False):
        g = g.dropna(subset=["annual_supply_kwh", "annual_cost_cny",
                              "satisfy_rate_pct", "gas_unit_cost_cny_kwh"])
        # LFP-only 数据按 7 LDES × 4 LDES-dur 循环，物理重复，需去重
        g = g.drop_duplicates(
            subset=["annual_supply_kwh", "annual_cost_cny"]
        ).reset_index(drop=True)
        if len(g) < 2:
            continue

        x = g["annual_supply_kwh"].to_numpy(dtype=float)
        y = g["annual_cost_cny"].to_numpy(dtype=float)
        mask = pareto_frontier_max_x_min_y(x, y)
        if mask.sum() < 2:
            continue

        front = g.loc[mask].sort_values("annual_supply_kwh").reset_index(drop=True)
        ds = front["annual_supply_kwh"].diff()
        dc = front["annual_cost_cny"].diff()
        marg = dc / ds.replace(0, np.nan)
        ratio = marg / front["gas_unit_cost_cny_kwh"]
        mid = (front["satisfy_rate_pct"]
                + front["satisfy_rate_pct"].shift(1)) / 2.0

        seg = pd.DataFrame({
            "scenario": front["scenario"],
            "province_cn": front["province_cn"],
            "ldes_tech": front["ldes_tech"],
            "mid_satisfy_rate_pct": mid,
            "marginal_ratio_to_gas": ratio,
            "delta_supply_kwh": ds,
            "delta_cost_cny": dc,
        }).iloc[1:]
        seg = seg[(seg["marginal_ratio_to_gas"] > 0)
                  & np.isfinite(seg["marginal_ratio_to_gas"])]
        if not seg.empty:
            rows.append(seg)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def draw_panel_c(ax, df_base, champs: dict[str, tuple[int, int]],
                  lfp_best: int):
    x_grid = np.linspace(0, 100, 201)

    lfp = lfp_only_subset(df_base, lfp_best)
    seg_lfp = compute_marginal_segments(lfp, ["scenario", "province_cn"])
    front_lfp = build_province_frontiers(
        seg_lfp, "mid_satisfy_rate_pct", "marginal_ratio_to_gas", True)
    plot_province_thin_lines(ax, front_lfp, "mid_satisfy_rate_pct",
                                "marginal_ratio_to_gas",
                                LFP_ONLY_COLOR, 0.10)
    mat_lfp = interpolate_to_grid(front_lfp, "mid_satisfy_rate_pct",
                                     "marginal_ratio_to_gas", x_grid)
    plot_median_curve(ax, mat_lfp, x_grid, LFP_ONLY_COLOR,
                       ls="--", lw=2.4, label=f"LFP {lfp_best}h only",
                       marker="s", ms=4)

    for tid in LDES_LABELS:
        if tid not in champs:
            continue
        lfp_d, ldes_d = champs[tid]
        sub = hybrid_subset(df_base, tid, lfp_d, ldes_d)
        if sub.empty:
            continue
        seg = compute_marginal_segments(
            sub, ["scenario", "province_cn", "ldes_tech"])
        if seg.empty:
            continue
        front = build_province_frontiers(
            seg, "mid_satisfy_rate_pct", "marginal_ratio_to_gas", True)
        plot_province_thin_lines(ax, front, "mid_satisfy_rate_pct",
                                    "marginal_ratio_to_gas",
                                    TECH_COLORS[tid], 0.10)
        mat = interpolate_to_grid(front, "mid_satisfy_rate_pct",
                                     "marginal_ratio_to_gas", x_grid)
        plot_median_curve(ax, mat, x_grid, TECH_COLORS[tid], lw=1.8,
                           label=f"LFP {lfp_d}h + {LDES_LABELS[tid]} {ldes_d}h",
                           marker="o", ms=3.5)

    ax.axhline(1.0, color=PALETTE["red"], lw=1.0, ls="--")
    ax.fill_between([0, 100], 0.05, 1.0, color=PALETTE["fill_blue"],
                      alpha=0.15, zorder=0)
    ax.fill_between([0, 100], 1.0, 100, color=PALETTE["fill_red"],
                      alpha=0.10, zorder=0)
    ax.set_yscale("log")
    ax.set_xlim(0, 100); ax.set_ylim(0.05, 80)
    ax.set_xlabel("Peaking supply ratio (%)  [2yr augmented]", labelpad=2)
    ax.set_ylabel("Marginal cost / gas cost", labelpad=2)
    ax.set_title("c  Marginal cost of capacity expansion (champion durations)",
                  loc="left", fontsize=9.4, pad=4)
    ax.grid(axis="both", which="major", color=PALETTE["grid"], lw=0.6)
    ax.text(99, 1.05, "gas parity", ha="right", va="bottom",
             fontsize=6.8, color=PALETTE["red"])
    ax.legend(loc="upper left", fontsize=6.8, handlelength=1.6,
               borderaxespad=0.2, ncol=1, columnspacing=0.7,
               handletextpad=0.5)
    add_panel_label(ax, "c")


# ========= panel d =========
def draw_panel_d(ax, df_all, champs: dict[str, tuple[int, int]],
                  champs_low: dict[str, tuple[int, int]],
                  lfp_best: int, lfp_best_low: int):
    """每 (scenario, ldes_tech) 在冠军时长下、satisfy>=80% 中取省级最低单位
    成本组合，画分布的 25-75 IQR + 中位。"""
    rows = []
    for tid in LDES_LABELS:
        for scen, champ_dict in [("baseline", champs),
                                   ("low_cost", champs_low)]:
            if tid not in champ_dict:
                continue
            lfp_d, ldes_d = champ_dict[tid]
            s = df_all[df_all["ldes_tech"].eq(tid)
                        & df_all["scenario"].eq(scen)
                        & df_all["satisfy_rate_pct"].ge(80)
                        & df_all["lfp_duration_h"].eq(lfp_d)
                        & df_all["ldes_duration_h"].eq(ldes_d)
                        & df_all["lfp_capacity_mwh"].gt(0)
                        & df_all["ldes_capacity_mwh"].gt(0)]
            if s.empty:
                continue
            best = (s.sort_values("unit_cost_cny_kwh")
                      .groupby("province_cn", as_index=False).first())
            rows.append({
                "label": f"LFP {lfp_d}h + {LDES_LABELS[tid]} {ldes_d}h",
                "scenario": scen,
                "ldes_tech": tid,
                "unit_cost_median": best["unit_cost_cny_kwh"].median(),
                "unit_cost_q25": best["unit_cost_cny_kwh"].quantile(0.25),
                "unit_cost_q75": best["unit_cost_cny_kwh"].quantile(0.75),
                "lfp_share_median": best["lfp_supply_share"].median(),
                "lfp_cap_median": best["lfp_capacity_mwh"].median(),
                "ldes_cap_median": best["ldes_capacity_mwh"].median(),
                "lfp_dur": lfp_d,
                "ldes_dur": ldes_d,
                "n_prov": len(best),
            })

    for scen, lfp_d in [("baseline", lfp_best), ("low_cost", lfp_best_low)]:
        s = df_all[df_all["ldes_capacity_mwh"].eq(0)
                    & df_all["lfp_capacity_mwh"].gt(0)
                    & df_all["scenario"].eq(scen)
                    & df_all["satisfy_rate_pct"].ge(80)
                    & df_all["lfp_duration_h"].eq(lfp_d)]
        if s.empty:
            continue
        best = (s.sort_values("unit_cost_cny_kwh")
                  .groupby("province_cn", as_index=False).first())
        rows.append({
            "label": f"LFP {lfp_d}h only",
            "scenario": scen,
            "ldes_tech": "li_lfp_only",
            "unit_cost_median": best["unit_cost_cny_kwh"].median(),
            "unit_cost_q25": best["unit_cost_cny_kwh"].quantile(0.25),
            "unit_cost_q75": best["unit_cost_cny_kwh"].quantile(0.75),
            "lfp_share_median": 1.0,
            "lfp_cap_median": best["lfp_capacity_mwh"].median(),
            "ldes_cap_median": 0,
            "lfp_dur": lfp_d,
            "ldes_dur": 0,
            "n_prov": len(best),
        })

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    base_lbl_order = (summary[summary["scenario"].eq("baseline")]
                       .sort_values("unit_cost_median")["label"].tolist())
    # 把 LFP-only 永远放最上
    lfp_only_lbls = [l for l in base_lbl_order if "only" in l]
    others = [l for l in base_lbl_order if "only" not in l]
    order = lfp_only_lbls + others
    # 兜底：补齐 low_cost 独有 label
    for l in summary["label"].unique():
        if l not in order:
            order.append(l)
    y_idx = {l: i for i, l in enumerate(order)}

    offsets = {"baseline": -0.16, "low_cost": 0.16}
    for _, r in summary.iterrows():
        if r["label"] not in y_idx:
            continue
        y = y_idx[r["label"]] + offsets[r["scenario"]]
        ax.plot([r["unit_cost_q25"], r["unit_cost_q75"]], [y, y],
                 color=SCENARIO_COLORS[r["scenario"]], lw=1.0, zorder=2)
        ax.scatter(r["unit_cost_median"], y, s=44,
                    color=SCENARIO_COLORS[r["scenario"]],
                    edgecolors="white", linewidths=0.7, zorder=3)
        if r["scenario"] == "baseline":
            if r["ldes_tech"] == "li_lfp_only":
                tag = f"{int(r['lfp_cap_median'])} MWh"
            else:
                tag = (f"{int(r['lfp_cap_median'])} + "
                        f"{int(r['ldes_cap_median'])} MWh | "
                        f"LFP-share={r['lfp_share_median']*100:.0f}%")
            ax.text(r["unit_cost_q75"] + 0.02, y, tag,
                     fontsize=5.6, color=PALETTE["grey"], va="center")

    ax.axvline(1.0, color=PALETTE["grey"], ls=":", lw=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=6.8)
    ax.invert_yaxis()
    xmax = max(2.5, summary["unit_cost_q75"].max() * 1.15)
    ax.set_xlim(0, xmax)
    ax.set_xlabel("Unit cost (CNY kWh$^{-1}$, satisfy ≥ 80%)", labelpad=2)
    ax.set_title("d  Best hybrid combinations by unit cost",
                  loc="left", fontsize=9.4, pad=4)
    ax.grid(axis="x", color=PALETTE["grid"], lw=0.6)
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=6,
                markerfacecolor=SCENARIO_COLORS[s], markeredgecolor="white",
                label=SCENARIO_LABELS[s])
        for s in ["baseline", "low_cost"]
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=6.5,
               handletextpad=0.4, borderaxespad=0.2)
    add_panel_label(ax, "d")
    return summary


# ========= panel e: 省级技术-时长冠军分布 =========
def build_province_champion_table(df: pd.DataFrame,
                                    scenario: str = "baseline",
                                    satisfy_min: float = 80.0
                                    ) -> pd.DataFrame:
    """对每个 (省, LDES 技术)，在 satisfy ≥ satisfy_min 且 LFP/LDES 容量都 >0 的
    所有组合中，取 unit_cost_cny_kwh 最低的那一行作为「省级冠军」。

    Returns DataFrame，每行一个 (省, LDES 技术) 冠军记录，含
    province_cn / province_en / ldes_tech / lfp_dur / ldes_dur /
    lfp_cap / ldes_cap / unit_cost / cost_ratio_to_gas / lfp_supply_share。
    """
    sub = df[(df["scenario"].eq(scenario))
              & (df["satisfy_rate_pct"] >= satisfy_min)
              & (df["lfp_capacity_mwh"] > 0)
              & (df["ldes_capacity_mwh"] > 0)].copy()
    if sub.empty:
        return pd.DataFrame()
    best = (sub.sort_values("unit_cost_cny_kwh")
              .groupby(["province_cn", "ldes_tech"], as_index=False)
              .first())
    return best.rename(columns={
        "lfp_duration_h": "lfp_dur",
        "ldes_duration_h": "ldes_dur",
        "lfp_capacity_mwh": "lfp_cap",
        "ldes_capacity_mwh": "ldes_cap",
    })[["province_cn", "province_en", "ldes_tech",
        "lfp_dur", "ldes_dur", "lfp_cap", "ldes_cap",
        "satisfy_rate_pct", "unit_cost_cny_kwh",
        "cost_ratio_to_gas", "lfp_supply_share",
        "gas_unit_cost_cny_kwh"]]


def draw_panel_e(ax, df_all, scenario: str = "baseline",
                  satisfy_min: float = 80.0):
    """26 省 × 7 LDES 技术 的省级冠军热图。

    - 行：省份（按双年总需求降序）
    - 列：LDES 技术（按全国中位单位成本升序）
    - 颜色：unit_cost / gas_unit_cost（中心 1.0）
    - 单元格：标 "L{lfp}+T{ldes}"
    - 缺失：灰色斜线
    """
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Rectangle

    table = build_province_champion_table(df_all, scenario=scenario,
                                            satisfy_min=satisfy_min)
    if table.empty:
        ax.set_title(f"e  Province-level champion combinations (no data)",
                      loc="left", fontsize=9.4, pad=4)
        add_panel_label(ax, "e")
        return table

    # 列顺序：按全国中位 cost_ratio 升序（让最便宜的技术在最左）
    tech_order_df = (table.groupby("ldes_tech")["cost_ratio_to_gas"]
                       .median().sort_values())
    tech_order = [t for t in tech_order_df.index if t in LDES_LABELS]
    # 兜底：补全所有技术
    for t in LDES_LABELS:
        if t not in tech_order:
            tech_order.append(t)

    # 行顺序：所有 26 省。先按「冠军省最低 cost ratio」升序（最划算的省靠上），
    # 完全不可达的省（无任何技术达 satisfy_min）按 max satisfy_rate 降序排在底部，
    # 用全行 hatch 显示。
    sub_full = df_all[df_all["scenario"].eq(scenario)].copy()
    all_provs = sorted(sub_full["province_cn"].unique())
    best_per_prov = (table.sort_values("cost_ratio_to_gas")
                       .drop_duplicates(subset=["province_cn"], keep="first")
                       .set_index("province_cn")["cost_ratio_to_gas"])
    reachable = [p for p in all_provs if p in best_per_prov.index]
    reachable.sort(key=lambda p: best_per_prov[p])  # cost ratio 升序

    unreachable = [p for p in all_provs if p not in best_per_prov.index]
    if unreachable:
        max_sat = (sub_full[(sub_full["province_cn"].isin(unreachable))
                              & (sub_full["lfp_capacity_mwh"] > 0)
                              & (sub_full["ldes_capacity_mwh"] > 0)]
                     .groupby("province_cn")["satisfy_rate_pct"].max()
                     .reindex(unreachable, fill_value=0))
        unreachable.sort(key=lambda p: -max_sat.get(p, 0))
    prov_order = reachable + unreachable

    n_prov = len(prov_order)
    n_tech = len(tech_order)
    cost_ratio_mat = np.full((n_prov, n_tech), np.nan)
    label_mat = np.full((n_prov, n_tech), "", dtype=object)

    for i, prov in enumerate(prov_order):
        for j, tid in enumerate(tech_order):
            row = table[(table["province_cn"].eq(prov))
                         & (table["ldes_tech"].eq(tid))]
            if not row.empty:
                r = row.iloc[0]
                cost_ratio_mat[i, j] = float(r["cost_ratio_to_gas"])
                label_mat[i, j] = f"L{int(r['lfp_dur'])}+T{int(r['ldes_dur'])}"

    finite = cost_ratio_mat[np.isfinite(cost_ratio_mat)]
    if len(finite) == 0:
        return table
    vmin = max(0.5, float(np.nanmin(cost_ratio_mat)) * 0.95)
    vmax = max(2.0, float(np.nanmax(cost_ratio_mat)) * 1.02)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)

    im = ax.imshow(cost_ratio_mat, aspect="auto", cmap="RdYlGn_r", norm=norm)

    # 缺失格：灰色斜线
    for i in range(n_prov):
        for j in range(n_tech):
            if not np.isfinite(cost_ratio_mat[i, j]):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                          facecolor="#E0E0E0",
                                          edgecolor="#BDBDBD",
                                          hatch="///",
                                          linewidth=0.4))
                continue
            v = cost_ratio_mat[i, j]
            text_color = "white" if (v < 0.7 or v > 1.5) else "#1A1A1A"
            ax.text(j, i, label_mat[i, j],
                     ha="center", va="center",
                     fontsize=7.6, color=text_color, fontweight="bold")
            ax.text(j, i + 0.34, f"{v:.2f}",
                     ha="center", va="center",
                     fontsize=6.6, color=text_color, alpha=0.85)

    ax.set_xticks(range(n_tech))
    ax.set_xticklabels([LDES_LABELS.get(t, t) for t in tech_order],
                        fontsize=9.0, rotation=0)
    ax.set_yticks(range(n_prov))
    ax.set_yticklabels(prov_order, fontsize=8.0)
    ax.set_xlabel("LDES technology  (sorted by national median cost ratio)",
                   fontsize=8.5, labelpad=4)
    ax.tick_params(axis="x", which="both", length=0)
    ax.tick_params(axis="y", which="both", length=0)

    title = (f"e  Province-level champion combinations "
              f"(scenario = {scenario}, satisfy ≥ {int(satisfy_min)}%); "
              f"cell text = best LFP-h + LDES-h + cost / gas ratio")
    ax.set_title(title, loc="left", fontsize=9.4, pad=4)

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02,
                          aspect=35, shrink=0.95)
    cbar.set_label("Unit cost / gas cost", fontsize=8.0)
    cbar.ax.tick_params(labelsize=7)
    cbar.ax.axhline(1.0, color=PALETTE["red"], lw=1.0, ls="--")

    add_panel_label(ax, "e")
    return table


# ========= panel f: 26 省最优组合曲线 =========
def select_province_champion_combo(df: pd.DataFrame,
                                     scenario: str = "baseline",
                                     prefer_threshold_pct: float = 80.0,
                                     fallback_thresholds=(70, 60, 50, 40, 30, 20),
                                     ) -> pd.DataFrame:
    """对每省挑一个 (ldes_tech, lfp_dur, ldes_dur) 组合作为该省的"全局冠军":

    1) 在 satisfy ≥ prefer_threshold 的所有点中取 unit_cost 最低的那一行；
    2) 若该省没有任何点达到 prefer_threshold，就在 fallback 里逐档降级；
    3) 输出每省一行，记录该省冠军组合的标签和触发档位。
    """
    sub = df[df["scenario"].eq(scenario)
              & df["lfp_capacity_mwh"].gt(0)
              & df["ldes_capacity_mwh"].gt(0)].copy()
    rows = []
    for prov_cn, g in sub.groupby("province_cn"):
        threshold_used = np.nan
        best_row = None
        for t in [prefer_threshold_pct] + list(fallback_thresholds):
            cand = g[g["satisfy_rate_pct"] >= t]
            if not cand.empty:
                best_row = cand.sort_values("unit_cost_cny_kwh").iloc[0]
                threshold_used = t
                break
        if best_row is None:
            continue
        rows.append({
            "province_cn": prov_cn,
            "ldes_tech": best_row["ldes_tech"],
            "lfp_dur": int(best_row["lfp_duration_h"]),
            "ldes_dur": int(best_row["ldes_duration_h"]),
            "best_unit_cost": float(best_row["unit_cost_cny_kwh"]),
            "best_satisfy_pct": float(best_row["satisfy_rate_pct"]),
            "threshold_used_pct": threshold_used,
        })
    return pd.DataFrame(rows)


def draw_panel_f(ax, df_all,
                  scenario: str = "baseline",
                  prefer_threshold_pct: float = 80.0,
                  lfp_only_dur: int = 8):
    """26 省最优 (LDES_tech, LFP-h, LDES-h) 组合的成本-替代深度曲线。

    每省一条曲线（25 个混合 + 1 个江苏只到 60%），不同颜色 + 图例。
    叠加一条「LFP-only 8h」全国中位曲线作为单技术对照。

    - 横轴 satisfy_rate_pct (40-100%)，左留 40 %% 空白裁掉
    - 纵轴 unit_cost / gas_unit_cost (log scale)
    - 红色虚线 1.0 = gas parity
    - 图例显示 "省份名: LDES 技术 L?+T?h"
    """
    from matplotlib import cm

    table = select_province_champion_combo(
        df_all, scenario=scenario, prefer_threshold_pct=prefer_threshold_pct)
    if table.empty:
        ax.set_title("f  Province cost-vs-satisfy curves (no data)",
                      loc="left", fontsize=9.4, pad=4)
        add_panel_label(ax, "f")
        return table

    sub_all = df_all[df_all["scenario"].eq(scenario)
                       & df_all["lfp_capacity_mwh"].gt(0)
                       & df_all["ldes_capacity_mwh"].gt(0)]

    # ---- 25 条省级混合冠军曲线 ----
    curves = []
    for _, r in table.iterrows():
        prov_cn = r["province_cn"]
        prov_en_arr = sub_all[sub_all["province_cn"].eq(prov_cn)]["province_en"]
        prov_en = prov_en_arr.iloc[0] if not prov_en_arr.empty else prov_cn

        gp = sub_all[(sub_all["province_cn"].eq(prov_cn))
                      & (sub_all["ldes_tech"].eq(r["ldes_tech"]))
                      & (sub_all["lfp_duration_h"].eq(r["lfp_dur"]))
                      & (sub_all["ldes_duration_h"].eq(r["ldes_dur"]))]
        if len(gp) < 2:
            continue
        x_supply = gp["annual_supply_kwh"].to_numpy(dtype=float)
        y_cost = gp["annual_cost_cny"].to_numpy(dtype=float)
        mask = pareto_frontier_max_x_min_y(x_supply, y_cost)
        if mask.sum() < 2:
            continue
        front = gp.loc[mask].sort_values("satisfy_rate_pct").reset_index(drop=True)
        sat = front["satisfy_rate_pct"].to_numpy(dtype=float)
        ratio = front["cost_ratio_to_gas"].to_numpy(dtype=float)
        ratio = np.maximum.accumulate(ratio)
        curves.append({
            "province_cn": prov_cn,
            "province_en": prov_en,
            "label": (f"{LDES_LABELS.get(r['ldes_tech'], r['ldes_tech'])} "
                      f"L{r['lfp_dur']}+T{r['ldes_dur']}h"),
            "min_ratio": float(np.nanmin(ratio)),
            "max_ratio": float(np.nanmax(ratio)),
            "max_satisfy": float(np.nanmax(sat)),
            "sat": sat,
            "ratio": ratio,
            "ldes_tech": r["ldes_tech"],
        })
    if not curves:
        return table

    curves.sort(key=lambda d: d["min_ratio"])

    # ---- LFP-only 中位曲线 ----
    df_lfp_only = df_all[(df_all["scenario"].eq(scenario))
                          & (df_all["ldes_capacity_mwh"].eq(0))
                          & (df_all["lfp_capacity_mwh"].gt(0))
                          & (df_all["lfp_duration_h"].eq(lfp_only_dur))]
    # 同省同 LFP_cap 在 7 LDES × N LDES_dur 循环里重复了；按 (省, LFP_cap) 去重
    df_lfp_only = (df_lfp_only.drop_duplicates(
                        subset=["province_cn", "lfp_capacity_mwh"])
                                .sort_values(["province_cn", "satisfy_rate_pct"]))
    # 内插到固定 satisfy 网格再求中位
    sat_grid = np.linspace(40, 100, 121)
    rows_per_grid = []
    for prov, g in df_lfp_only.groupby("province_cn"):
        sat = g["satisfy_rate_pct"].to_numpy(dtype=float)
        ratio = g["cost_ratio_to_gas"].to_numpy(dtype=float)
        if len(sat) < 2:
            continue
        # 单调化
        ratio = np.maximum.accumulate(ratio)
        # 仅在该省覆盖的 satisfy 区间内插值
        in_range = (sat_grid >= sat.min()) & (sat_grid <= sat.max())
        interp = np.full_like(sat_grid, np.nan, dtype=float)
        if in_range.any():
            interp[in_range] = np.interp(sat_grid[in_range], sat, ratio)
        rows_per_grid.append(interp)
    lfp_median = np.nanmedian(np.vstack(rows_per_grid), axis=0) if rows_per_grid else None

    # ---- 颜色：用 tab20 + tab20b 拼出 26 种区分色 ----
    palette_a = cm.get_cmap("tab20").colors
    palette_b = cm.get_cmap("tab20b").colors
    palette = list(palette_a) + list(palette_b)

    # ---- 背景 + gas parity ----
    ax.fill_between([40, 100], 0.30, 1.0, color=PALETTE["fill_blue"],
                      alpha=0.18, zorder=0)
    ax.fill_between([40, 100], 1.0, 3.0, color=PALETTE["fill_red"],
                      alpha=0.10, zorder=0)
    ax.axhline(1.0, color=PALETTE["red"], lw=1.2, ls="--", zorder=2)

    # ---- 25 条省级曲线 ----
    legend_handles = []
    legend_labels = []
    for i, c in enumerate(curves):
        color = palette[i % len(palette)]
        # 只画 satisfy ≥ 40 的部分
        keep = c["sat"] >= 40.0
        if keep.sum() < 2:
            keep = c["sat"] >= c["sat"].min()
        sat_p = c["sat"][keep]
        ratio_p = c["ratio"][keep]
        ln, = ax.plot(sat_p, ratio_p,
                       color=color, lw=1.4, alpha=0.92, zorder=3,
                       marker="o", ms=2.6, mec="white", mew=0.3)
        legend_handles.append(ln)
        legend_labels.append(f"{c['province_en']}: {c['label']}")

    # ---- LFP-only 中位线（粗黑虚线） ----
    if lfp_median is not None:
        valid = ~np.isnan(lfp_median)
        ln_med, = ax.plot(sat_grid[valid], lfp_median[valid],
                            color="#1A1A1A", lw=2.6, ls=(0, (1.5, 1)),
                            zorder=5,
                            label=f"LFP {lfp_only_dur}h only (national median)")
        legend_handles.append(ln_med)
        legend_labels.append(f"LFP {lfp_only_dur}h only (national median)")

    # ---- 轴 / 标题 / 图例 ----
    ax.set_yscale("log")
    # ylim 上界压在 gas parity 上方 2x（即 cost ratio <=2.5），
    # 避免少数省在低 satisfy 处单位成本飙高把整张图压扁
    ax.set_xlim(40, 100)
    ax.set_ylim(0.35, 2.5)
    ax.set_xlabel("Peaking supply ratio (%)  [2yr augmented]",
                   fontsize=8.5, labelpad=2)
    ax.set_ylabel("Unit cost / gas cost", fontsize=8.5, labelpad=2)
    ax.set_title(
        f"f  Province-level lowest-cost LFP+LDES combination "
        f"vs LFP-only national median (scenario = {scenario}, "
        f"target satisfy = {int(prefer_threshold_pct)}%)",
        loc="left", fontsize=9.4, pad=4)
    ax.grid(axis="both", which="major", color=PALETTE["grid"], lw=0.5)
    ax.text(99.5, 1.05, "gas parity", ha="right", va="bottom",
             fontsize=7.0, color=PALETTE["red"])

    # 图例放在 panel 右侧外；2 列防止过长
    ax.legend(legend_handles, legend_labels,
                loc="center left", bbox_to_anchor=(1.02, 0.5),
                fontsize=6.4, ncol=1, frameon=False,
                handlelength=1.4, handletextpad=0.5,
                borderaxespad=0.2)

    add_panel_label(ax, "f")

    out = pd.DataFrame([
        {"province_cn": c["province_cn"],
         "province_en": c["province_en"],
         "ldes_tech": c["ldes_tech"],
         "label": c["label"],
         "min_cost_ratio_on_curve": c["min_ratio"],
         "max_cost_ratio_on_curve": c["max_ratio"],
         "max_satisfy_pct_on_curve": c["max_satisfy"],
         "n_points": len(c["sat"])}
        for c in curves
    ])
    return out


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {RESULTS_CSV}")
    df = pd.read_csv(RESULTS_CSV)
    print(f"  {len(df)} rows, scenarios: {df.scenario.unique()}")

    champs_b, lfp_best_b = identify_champion_durations(df, "baseline")
    champs_l, lfp_best_l = identify_champion_durations(df, "low_cost")
    print(f"  Champion durations (baseline): LFP-only={lfp_best_b}h")
    for tid, (a, b) in champs_b.items():
        print(f"    {tid}: LFP {a}h + {LDES_LABELS[tid]} {b}h")
    print(f"  Champion durations (low_cost): LFP-only={lfp_best_l}h")
    for tid, (a, b) in champs_l.items():
        print(f"    {tid}: LFP {a}h + {LDES_LABELS[tid]} {b}h")

    df_base = df[df["scenario"].eq("baseline")]
    df_low = df[df["scenario"].eq("low_cost")]

    fig = plt.figure(figsize=(15.0, 23.0))
    gs = fig.add_gridspec(4, 2,
                            width_ratios=[1.05, 1.0],
                            height_ratios=[1.0, 1.0, 1.55, 1.45],
                            left=0.060, right=0.965,
                            top=0.925, bottom=0.040,
                            wspace=0.30, hspace=0.42)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, :])
    # panel f 不跨满两列：ax 占左侧 70%，右侧 30% 留给图例
    ax_f = fig.add_subplot(gs[3, 0])
    ax_f.set_position([0.060, ax_f.get_position().y0,
                        0.620, ax_f.get_position().height])

    draw_panel_a(ax_a, df_base, champs_b, lfp_best_b)
    draw_panel_b(ax_b, df_low, champs_l, lfp_best_l)
    draw_panel_c(ax_c, df_base, champs_b, lfp_best_b)
    summary_d = draw_panel_d(ax_d, df, champs_b, champs_l,
                                lfp_best_b, lfp_best_l)
    table_e = draw_panel_e(ax_e, df, scenario="baseline", satisfy_min=60.0)
    table_f = draw_panel_f(ax_f, df, scenario="baseline",
                            prefer_threshold_pct=80.0)

    if summary_d is not None and not summary_d.empty:
        summary_d.to_csv(SOURCE_DIR / "fig4_hybrid_panel_d_summary.csv",
                          index=False)
    if table_e is not None and not table_e.empty:
        table_e.to_csv(SOURCE_DIR / "fig4_hybrid_panel_e_province_champions.csv",
                        index=False)
        print(f"  Province champions table built: "
              f"{table_e['province_cn'].nunique()} prov × "
              f"{table_e['ldes_tech'].nunique()} tech, "
              f"{len(table_e)} (province, tech) champion cells",
              flush=True)
    if table_f is not None and not table_f.empty:
        table_f.to_csv(SOURCE_DIR / "fig4_hybrid_panel_f_overall_champions.csv",
                        index=False)
        print(f"  Panel f curves: {len(table_f)} provinces drawn",
              flush=True)

    # 重新计算 panel c 的 source data：所有 (scenario, province, ldes_tech) 的真前沿段
    seg_all_rows = []
    for scen, champs in [("baseline", champs_b)]:
        df_s = df[df["scenario"].eq(scen)]
        # LFP-only 冠军
        lfp_only = lfp_only_subset(df_s, lfp_best_b)
        seg_lfp = compute_marginal_segments(
            lfp_only, ["scenario", "province_cn"])
        if not seg_lfp.empty:
            seg_lfp = seg_lfp.copy()
            seg_lfp["combo"] = f"LFP {lfp_best_b}h only"
            seg_all_rows.append(seg_lfp)
        # 各 LDES 冠军组合
        for tid, (lfp_d, ldes_d) in champs.items():
            sub = hybrid_subset(df_s, tid, lfp_d, ldes_d)
            if sub.empty:
                continue
            seg_t = compute_marginal_segments(
                sub, ["scenario", "province_cn", "ldes_tech"])
            if not seg_t.empty:
                seg_t = seg_t.copy()
                seg_t["combo"] = (
                    f"LFP {lfp_d}h + {LDES_LABELS[tid]} {ldes_d}h")
                seg_all_rows.append(seg_t)
    if seg_all_rows:
        seg_all = pd.concat(seg_all_rows, ignore_index=True)
        seg_all.to_csv(
            SOURCE_DIR / "fig4_hybrid_panel_c_marginal_pareto.csv",
            index=False)
        print(f"  Panel c marginal-cost frontier source data: "
              f"{len(seg_all)} segments across "
              f"{seg_all['combo'].nunique()} combos")

    fig.text(0.060, 0.978,
              "M4 hybrid configuration (2-year augmented dispatch): "
              "LFP baseline vs LFP+LDES combinations",
              ha="left", va="top",
              fontsize=11, fontweight="bold", color=PALETTE["black"])
    champ_strs = [f"LFP{a}h+{LDES_LABELS[t]}{b}h"
                    for t, (a, b) in champs_b.items()]
    fig.text(0.060, 0.960,
              f"Panels a-d: champion durations are the (LFP-h, LDES-h) pair "
              f"minimising the median unit cost across 26 provinces "
              f"(satisfy ≥ {int(CHAMP_SATISFY_MIN)}%). "
              f"Province-level optima may differ — see panel e.",
              ha="left", va="top",
              fontsize=7.8, color=PALETTE["black"])
    fig.text(0.060, 0.946,
              f"Baseline champions: LFP-only={lfp_best_b}h | "
              + ", ".join(champ_strs),
              ha="left", va="top",
              fontsize=7.0, color=PALETTE["grey"])

    for ext, kwargs in {".svg": {}, ".pdf": {}, ".png": {"dpi": 340}}.items():
        path = FIG_BASE.with_suffix(ext)
        fig.savefig(path, bbox_inches="tight", **kwargs)
        print(f"  saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
