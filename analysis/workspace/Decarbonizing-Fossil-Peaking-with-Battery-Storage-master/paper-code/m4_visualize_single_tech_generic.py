"""
Fig 5 实验 A — 复制年单技术 1×4 矩阵图（无 markov、无极端注入）。

5 个 demand 节点（Yr1_repeat / Yr2_repeat / ... / Yr5_repeat）各出一幅
1×4 矩阵图（4 dw × 8 单技术曲线），观察「单纯增加年限」是否影响经济性。

预期：因为 demand 是真实复制（无新极端事件），更多年只是放大噪声、
延长仿真时间。曲线形态应几乎不变 → 验证「5 年仿真」相对于「1 年真实
复制」带来的经济结论改变全部由 markov + 极端注入贡献。

数据源：m4_single_tech_yr<N>_repeat_<tech>.csv
输出：output/m4_economics/nature_figure/fig5_yr<N>_repeat_dw.{png,svg,pdf}
"""

from __future__ import annotations
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# OSR 选法用 metrics.compute_osr（与 fig4 panel a 完全一致）
from metrics import compute_osr as metrics_compute_osr
from config import CONFIG, GAS_POWER_PER_M3


CJK_FONT_PATHS = [
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
]
CJK_BOLD_FONT_PATHS = [
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
    Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
]
for fp in CJK_FONT_PATHS + CJK_BOLD_FONT_PATHS:
    if fp.exists():
        font_manager.fontManager.addfont(str(fp))


def _first_existing_font(paths):
    for fp in paths:
        if fp.exists():
            return fp
    return None


CJK_FONT_PATH = _first_existing_font(CJK_FONT_PATHS)
CJK_BOLD_FONT_PATH = _first_existing_font(CJK_BOLD_FONT_PATHS)
CJK_FONT_PROP = (
    font_manager.FontProperties(fname=str(CJK_FONT_PATH))
    if CJK_FONT_PATH is not None else None
)
CJK_BOLD_FONT_PROP = (
    font_manager.FontProperties(fname=str(CJK_BOLD_FONT_PATH), weight="bold")
    if CJK_BOLD_FONT_PATH is not None else CJK_FONT_PROP
)


def cjk_font_kwargs(bold: bool = False):
    """Return matplotlib kwargs for text that may contain Chinese glyphs."""
    prop = CJK_BOLD_FONT_PROP if bold else CJK_FONT_PROP
    return {"fontproperties": prop} if prop is not None else {}


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

TECH_FILES = {
    "li_lfp": ("LFP", "li_lfp"),
    "caes": ("CAES", "caes"),
    "laes": ("LAES", "laes"),
    "gravity": ("Gravity", "gravity"),
    "fecr_flow": ("FeCr flow", "fecr_flow"),
    "molten_salt_tes": ("Molten salt TES", "molten_salt_tes"),
    "vrfb": ("VRFB", "vrfb"),
    "h2_cavern": ("H2 cavern", "h2_cavern"),
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


def load_year_techs(stem: str, output_prefix: str) -> dict[str, pd.DataFrame]:
    needed = ["scenario", "dispatch_window_h", "tech_id", "duration_h",
                "province_en", "province_cn", "capacity_mwh",
                "satisfy_rate_pct", "annual_supply_kwh",
                "unit_cost_cny_kwh", "gas_unit_cost_cny_kwh",
                "cost_ratio_to_gas"]
    out = {}
    for tid in TECH_FILES:
        fp = M4_DIR / f"{output_prefix}_{stem}_{tid}.csv"
        if not fp.exists():
            print(f"  WARNING 缺少 {fp.name}")
            continue
        df = pd.read_csv(fp, usecols=needed, low_memory=False)
        df = df[df["scenario"].eq(SCENARIO)].copy()
        out[tid] = df
        print(f"  {tid}: {len(df):,} rows")
    return out


def champion_duration(df_cell, satisfy_min=CHAMP_SATISFY_MIN):
    """按 satisfy>=70% 下 unit_cost 中位最低选 dur（默认成本偏好选法）。"""
    sub = df_cell[df_cell["satisfy_rate_pct"].ge(satisfy_min)]
    if sub.empty:
        return None
    med = (sub.groupby("duration_h")["unit_cost_cny_kwh"]
              .median().sort_values())
    return int(med.index[0]) if not med.empty else None


def champion_duration_osr(df_cell):
    """按 OSR_weighted_avg 最高选 dur（与 fig4 panel a 同款选法）。

    直接调用 metrics.compute_osr 保证算法一致。
    OSR = 边际储能成本曲线与 gas parity 的交点位置（supply 百分比 0-100）。
    OSR_weighted_avg = 26 省 OSR 按 max satisfy_rate 加权平均
    （原版用 total_demand 加权，但本数据 total_demand 在 26 省差别同等于
    province pop，这里用 max satisfy 作合理替代）。
    """
    if df_cell.empty or "annual_supply_kwh" not in df_cell.columns:
        return None
    durations = sorted(df_cell["duration_h"].unique())
    weighted_osrs = {}
    for dur in durations:
        sub = df_cell[df_cell["duration_h"].eq(dur)]
        prov_data = []
        for prov, g in sub.groupby("province_cn"):
            osr = compute_osr_for_province(g)
            if osr is None:
                continue
            # 用该 prov-dur 第一行的 total_demand_kwh 类似量近似（gas-parity 不变）
            # 或者用 sum of supply 作为 weight
            weight = g["satisfy_rate_pct"].max()  # 用最大 satisfy 作权重近似
            prov_data.append((osr, weight))
        if not prov_data:
            continue
        osrs = np.array([p[0] for p in prov_data])
        weights = np.array([p[1] for p in prov_data])
        weighted_osr = np.average(osrs, weights=weights)
        weighted_osrs[dur] = weighted_osr
    if not weighted_osrs:
        return None
    # 取 OSR 最高者
    best_dur = max(weighted_osrs, key=weighted_osrs.get)
    return int(best_dur)


def compute_osr_for_province(prov_df):
    """对单 (prov, dur) 子集计算 OSR——直接调 metrics.compute_osr。

    需要列：satisfy_rate_pct, annual_supply_kwh, unit_cost_cny_kwh,
    gas_unit_cost_cny_kwh

    metrics.compute_osr 接口要求：
      - peak_supply_kwh   ≡ annual_supply_kwh
      - annual_cost       ≡ annual_supply_kwh × unit_cost_cny_kwh
      - satisfy_rate_pct
      - 三个标量参数：gas_price (per m3), gas_power_per_m3, generator_cost
        反推 fake_gas_price: (gas_unit - gen_cost) × GAS_POWER_PER_M3
    """
    if prov_df.empty or "annual_supply_kwh" not in prov_df.columns:
        return None
    df = prov_df.sort_values("satisfy_rate_pct").drop_duplicates(
        subset=["satisfy_rate_pct"], keep="first").reset_index(drop=True)
    if len(df) < 2:
        return None
    df = df.copy()
    df["peak_supply_kwh"] = df["annual_supply_kwh"]
    df["annual_cost"] = df["annual_supply_kwh"] * df["unit_cost_cny_kwh"]
    df["total_demand_kwh"] = 1.0
    gas_unit_cny = df["gas_unit_cost_cny_kwh"].iloc[0]
    if not np.isfinite(gas_unit_cny) or gas_unit_cny <= 0:
        return None
    gen_cost = float(CONFIG["gas"]["generator_cost"])
    fake_gas_price = (gas_unit_cny - gen_cost) * GAS_POWER_PER_M3
    try:
        return metrics_compute_osr(
            df, fake_gas_price, GAS_POWER_PER_M3, gen_cost)
    except Exception:
        return None



def build_province_curves(sub):
    curves = {}
    for prov, g in sub.groupby("province_cn", sort=False):
        g = g[["satisfy_rate_pct", "cost_ratio_to_gas"]].dropna()
        if g.empty:
            continue
        g = g.sort_values("satisfy_rate_pct").reset_index(drop=True)
        curves[prov] = g
    return curves


def interpolate_to_grid(curves, x_grid):
    rows = []
    for _, g in curves.items():
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


def plot_thin(ax, curves, color, alpha=0.06):
    for _, g in curves.items():
        ax.plot(g["satisfy_rate_pct"], g["cost_ratio_to_gas"],
                 color=color, lw=0.5, alpha=alpha, zorder=1)


def plot_median(ax, mat, x_grid, color, ls="-", lw=1.6,
                 label=None, marker=None, ms=0):
    if mat.size == 0:
        return None
    finite_cols = np.isfinite(mat).any(axis=0)
    med = np.full(mat.shape[1], np.nan, dtype=float)
    if finite_cols.any():
        med[finite_cols] = np.nanmedian(mat[:, finite_cols], axis=0)
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


def draw_cell(ax, tech_dfs, dw_h, panel_letter,
                show_xlabel=True, show_ylabel=True, show_legend=True,
                champion_by="unit_cost"):
    x_grid = np.linspace(0, 100, 241)
    rows = []
    champs = {}

    # 7 LDES + LFP（先 7 LDES，后 LFP-only）
    for tid in ["caes", "laes", "gravity", "fecr_flow",
                  "molten_salt_tes", "vrfb", "h2_cavern", "li_lfp"]:
        if tid not in tech_dfs:
            continue
        df_cell = tech_dfs[tid][tech_dfs[tid]["dispatch_window_h"].eq(dw_h)]
        if df_cell.empty:
            continue
        if champion_by == "osr":
            best = champion_duration_osr(df_cell)
        else:
            best = champion_duration(df_cell)
        if best is None:
            continue
        sub = df_cell[df_cell["duration_h"].eq(best)]
        cur = build_province_curves(sub)
        is_lfp = (tid == "li_lfp")
        plot_thin(ax, cur, TECH_COLORS[tid], 0.10 if is_lfp else 0.05)
        mat = interpolate_to_grid(cur, x_grid)
        ls = "--" if is_lfp else "-"
        lw = 2.0 if is_lfp else 1.5
        marker = "s" if is_lfp else "o"
        ms = 3.5 if is_lfp else 3.0
        label = f"{TECH_FILES[tid][0]} {best}h only"
        med = plot_median(ax, mat, x_grid, TECH_COLORS[tid],
                            ls=ls, lw=lw, label=label,
                            marker=marker, ms=ms)
        if med is not None:
            med = med.assign(combo=label, dispatch_window_h=dw_h)
            rows.append(med)
        champs[tid] = best

    ax.axhline(1.0, color=PALETTE["red"], lw=0.9, ls="--", alpha=0.5)
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


def render_one_node(stem: str, output_prefix: str = "m4_single_tech",
                     fig_prefix: str = "fig5",
                     title_prefix: str = "",
                     champion_by: str = "unit_cost"):
    print(f"\n=== 渲染 {stem} (champion_by={champion_by}) ===")
    tech_dfs = load_year_techs(stem, output_prefix)
    if not tech_dfs:
        print(f"  {stem} 无数据，跳过")
        return

    fig = plt.figure(figsize=(18.0, 4.8))
    gs = fig.add_gridspec(1, 4,
                            left=0.050, right=0.985,
                            top=0.84, bottom=0.18, wspace=0.22)

    source_rows = []
    champ_summary = []
    letters = "abcd"
    for c, dw_h in enumerate(DW_VALUES):
        ax = fig.add_subplot(gs[0, c])
        sd, champs = draw_cell(
            ax, tech_dfs, dw_h, letters[c],
            show_xlabel=True,
            show_ylabel=(c == 0),
            show_legend=True,
            champion_by=champion_by,
        )
        if not sd.empty:
            source_rows.append(sd)
        champ_summary.append({"dw_h": dw_h, **champs})

    fig.text(0.050, 0.96,
              f"{title_prefix} {stem}",
              ha="left", va="top",
              fontsize=11, fontweight="bold", color=PALETTE["black"])
    fig.text(0.050, 0.93,
              f"4 dispatch windows. Each curve = single-tech champion "
              f"duration. 26-province median (panel a logic). 25 MWh "
              f"capacity step (full sweep). Red dashed = gas parity.",
              ha="left", va="top",
              fontsize=7.4, color=PALETTE["black"])

    fig_base = FIGURE_DIR / f"{fig_prefix}_{stem}_dw"
    for ext, kw in {".svg": {}, ".pdf": {}, ".png": {"dpi": 300}}.items():
        path = fig_base.with_suffix(ext)
        fig.savefig(path, bbox_inches="tight", **kw)
        print(f"  saved: {path.name}")
    plt.close(fig)

    if source_rows:
        sd = pd.concat(source_rows, ignore_index=True)
        sd["stem"] = stem
        sd.to_csv(SOURCE_DIR / f"{fig_prefix}_{stem}_curves.csv",
                    index=False)
    if champ_summary:
        cs = pd.DataFrame(champ_summary)
        cs["stem"] = stem
        cs.to_csv(SOURCE_DIR / f"{fig_prefix}_{stem}_champions.csv",
                    index=False)
        print(f"  champions:")
        print(cs.to_string(index=False))


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--stems", nargs="+", default=None,
                        help="demand stems list, eg. yr1_repeat yr2_repeat")
    parser.add_argument("--all-yr-repeat", action="store_true",
                        help="渲染 5 个复制年实验图")
    parser.add_argument("--output-prefix", default="m4_single_tech",
                        help="输入 CSV 前缀（默认 m4_single_tech）")
    parser.add_argument("--fig-prefix", default="fig5",
                        help="输出 figure 前缀（默认 fig5）")
    parser.add_argument("--title-prefix", default="Fig5 (yr-repeat)",
                        help="图标题前缀")
    parser.add_argument("--champion-by", default="unit_cost",
                        choices=["unit_cost", "osr"],
                        help="冠军 dur 选法：unit_cost (默认) 或 osr "
                              "（与 fig4 panel a 对齐）")
    args = parser.parse_args()

    if args.all_yr_repeat:
        for n in [1, 2, 3, 4, 5]:
            render_one_node(f"yr{n}_repeat", args.output_prefix,
                              args.fig_prefix, args.title_prefix,
                              args.champion_by)
    elif args.stems:
        for stem in args.stems:
            render_one_node(stem, args.output_prefix,
                              args.fig_prefix, args.title_prefix,
                              args.champion_by)
    else:
        parser.error("必须指定 --stems 或 --all-yr-repeat")


if __name__ == "__main__":
    main()
