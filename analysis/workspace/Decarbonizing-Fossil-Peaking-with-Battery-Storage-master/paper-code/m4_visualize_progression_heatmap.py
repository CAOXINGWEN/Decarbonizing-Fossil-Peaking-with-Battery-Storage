"""
Fig 5 progression 热力图 — 8 tech × Y节点×dw 4×4 网格

每个 tech 一幅 4×4 热力图：
  行 = 脱碳节点 Y1 / Y3 / Y5 / Y10
  列 = 放电窗口 dw = 8h / 12h / 16h / 24h
  颜色 = 26 省 cost/gas 中位 @ satisfy=80%

8 幅热力图组合成 2×4 大图。

数据源：m4_single_tech_<tech_id>.csv（含 Y1/Y3/Y5/Y10 4 节点）
冠军 dur 选法：OSR_weighted_avg 最高（与 fig4 panel a 对齐）

输出: output/m4_economics/nature_figure/fig5_progression_heatmap_2x4.{png,svg,pdf}
"""

from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import m4_visualize_single_tech_generic as ms


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


PAPER_CODE_DIR = Path(__file__).resolve().parent
M4_DIR = PAPER_CODE_DIR / "output" / "m4_economics"
FIGURE_DIR = M4_DIR / "nature_figure"
SOURCE_DIR = FIGURE_DIR / "source_data"
FIG_BASE = FIGURE_DIR / "fig5_progression_heatmap_2x4"

NODES = ["Y1", "Y3", "Y5", "Y10"]
DWS = [8.0, 12.0, 16.0, 24.0]
SATISFY_TARGET = 80.0   # 在 satisfy=80% 处取 cost ratio

TECHS = ["caes", "laes", "gravity", "fecr_flow",
          "molten_salt_tes", "vrfb", "h2_cavern", "li_lfp"]
TECH_LABELS = {"caes": "CAES", "laes": "LAES",
                "gravity": "Gravity", "fecr_flow": "FeCr flow",
                "molten_salt_tes": "Molten salt TES",
                "vrfb": "VRFB", "h2_cavern": "H2 cavern", "li_lfp": "LFP"}


def cost_at_satisfy_for_node_dw(df_full: pd.DataFrame, node: str, dw: float,
                                  satisfy: float = SATISFY_TARGET,
                                  champion_by: str = "osr") -> float | None:
    """对单 (node, dw) 子集，按 OSR 选 dur，
    在 satisfy=80% 处取 26 省 cost_ratio_to_gas 中位。"""
    df_cell = df_full[(df_full["node"] == node)
                       & (df_full["dispatch_window_h"].eq(dw))]
    if df_cell.empty:
        return None
    if champion_by == "osr":
        best = ms.champion_duration_osr(df_cell)
    else:
        best = ms.champion_duration(df_cell)
    if best is None:
        return None
    sub = df_cell[df_cell["duration_h"].eq(best)]
    vals = []
    for prov, g in sub.groupby("province_cn"):
        g = g[["satisfy_rate_pct", "cost_ratio_to_gas"]].dropna()
        if len(g) < 2: continue
        g = g.sort_values("satisfy_rate_pct").reset_index(drop=True)
        x = g.satisfy_rate_pct.to_numpy(dtype=float)
        y = g.cost_ratio_to_gas.to_numpy(dtype=float)
        x_u, idx = np.unique(x, return_index=True); y_u = y[idx]
        if len(x_u) < 2: continue
        if satisfy < x_u.min() or satisfy > x_u.max(): continue
        vals.append(float(np.interp(satisfy, x_u, y_u)))
    if not vals:
        return None
    return float(np.median(vals))


def build_heatmap_for_tech(tid: str) -> tuple[np.ndarray, list[int]]:
    """返回 4×4 矩阵 [Y节点, dw] + 每行 (node) 的冠军 dur 列表"""
    fp = M4_DIR / f"m4_single_tech_{tid}.csv"
    needed = ["scenario", "node", "dispatch_window_h", "tech_id",
                "duration_h", "province_en", "province_cn",
                "capacity_mwh", "satisfy_rate_pct",
                "annual_supply_kwh", "unit_cost_cny_kwh",
                "gas_unit_cost_cny_kwh", "cost_ratio_to_gas"]
    df = pd.read_csv(fp, usecols=needed, low_memory=False)
    df = df[df["scenario"].eq("baseline")]
    print(f"  {tid}: {len(df):,} rows")

    mat = np.full((len(NODES), len(DWS)), np.nan)
    champs = {}
    for i, node in enumerate(NODES):
        for j, dw in enumerate(DWS):
            v = cost_at_satisfy_for_node_dw(df, node, dw)
            if v is not None:
                mat[i, j] = v
            # 冠军 dur 一致就好（只取一个 cell 的）
            if (node, dw) == (node, 12.0):
                df_cell = df[(df.node == node) & (df.dispatch_window_h.eq(dw))]
                if not df_cell.empty:
                    best = ms.champion_duration_osr(df_cell)
                    if best is not None:
                        champs[node] = int(best)
    return mat, champs


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== 构造 {len(TECHS)} 个 tech 的 4×4 热力图（行=Y节点，列=dw, "
          f"satisfy={SATISFY_TARGET}%）===")
    matrices = {}
    champ_table = {}
    rows = []
    for tid in TECHS:
        print(f"\n  {tid}:")
        mat, champs = build_heatmap_for_tech(tid)
        matrices[tid] = mat
        champ_table[tid] = champs
        print(f"    {mat}")
        print(f"    champion dur (dw=12h): {champs}")
        for i, node in enumerate(NODES):
            for j, dw in enumerate(DWS):
                rows.append({"tech_id": tid, "node": node,
                              "dw_h": dw,
                              "cost_ratio_at_satisfy_80": mat[i, j],
                              "champion_dur_h_dw12": champs.get(node)})
    src_df = pd.DataFrame(rows)
    src_df.to_csv(SOURCE_DIR / "fig5_progression_heatmap_2x4.csv",
                    index=False)

    # 全局 vmin/vmax
    all_vals = np.concatenate([m.flatten() for m in matrices.values()])
    all_vals = all_vals[np.isfinite(all_vals)]
    vmin = float(np.percentile(all_vals, 2))
    vmax = float(np.percentile(all_vals, 98))
    print(f"\n  色阶范围: vmin={vmin:.3f}, vmax={vmax:.3f}")

    fig, axes = plt.subplots(2, 4, figsize=(18, 9.5),
                                gridspec_kw={"wspace": 0.35, "hspace": 0.50})
    for k, tid in enumerate(TECHS):
        ax = axes[k // 4, k % 4]
        mat = matrices[tid]
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r",
                        vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(DWS)))
        ax.set_xticklabels([f"{int(d)}h" for d in DWS])
        ax.set_yticks(range(len(NODES)))
        ax.set_yticklabels(NODES)
        ax.set_xlabel("Dispatch window dw", labelpad=2)
        ax.set_ylabel("Decarbonization node", labelpad=2)
        ax.set_title(f"{TECH_LABELS[tid]}", fontsize=10, pad=6)
        for i in range(len(NODES)):
            for j in range(len(DWS)):
                v = mat[i, j]
                if np.isfinite(v):
                    text_color = "white" if v > (vmin + vmax) / 2 else "black"
                    ax.text(j, i, f"{v:.2f}",
                              ha="center", va="center",
                              color=text_color, fontsize=8)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(f"Cost / gas @ satisfy={SATISFY_TARGET:.0f}%",
                    fontsize=9)

    fig.suptitle(
        "Fig 5 progression 热力图 — 8 tech × 脱碳节点×放电窗口（OSR 选法 "
        "选冠军 dur, 26 省 cost/gas 中位 @ satisfy=80%）",
        fontsize=11, fontweight="bold", y=0.98)
    fig.text(0.06, 0.94,
              "色阶: 红=贵（>gas）, 绿=便宜（<gas）。行=Y1/Y3/Y5/Y10 progression "
              "脱碳节点（demand×1.0/1.4/1.8/2.5），列=dw 8/12/16/24h。"
              "数据源 m4_single_tech_<tech>.csv（5 年 markov + 极端注入）。",
              fontsize=8.0, ha="left", va="top")

    for ext, kw in {".svg": {}, ".pdf": {},
                      ".png": {"dpi": 300}}.items():
        path = FIG_BASE.with_suffix(ext)
        fig.savefig(path, bbox_inches="tight", **kw)
        print(f"  saved: {path.name}")
    plt.close(fig)


if __name__ == "__main__":
    main()
