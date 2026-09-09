"""
M4 progression — 省级热力图（每省一幅 8 tech × 16 cells = 4 Y节点 × 4 dw）

针对 fig5_progression_heatmap_2x4 的省级化版本。
对 26 省每省单独画一幅图，让用户看清不同省份下各 tech 的相对位置。

布局每幅图：2 行 × 4 列 = 8 子热力图（每子图一个 tech）：
  行 = Y1 / Y3 / Y5 / Y10 节点
  列 = dw = 8h / 12h / 16h / 24h
  颜色 = 该省 cost/gas @ satisfy=80%（不再做 26 省加权）

数据源：m4_single_tech_<tech>.csv
冠军 dur 选法：OSR（每省独立 OSR，不做加权平均）

输出：output/m4_economics/nature_figure/progression_per_province/
       fig5_progression_<province>_heatmap.{png,pdf}（共 26 张）
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
from event_clustering import EN_TO_CN

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
FIGURE_DIR = M4_DIR / "nature_figure" / "progression_per_province"
SOURCE_DIR = M4_DIR / "nature_figure" / "source_data"

NODES = ["Y1", "Y3", "Y5", "Y10"]
DWS = [8.0, 12.0, 16.0, 24.0]
SATISFY_TARGET = 80.0

TECHS = ["caes", "laes", "gravity", "fecr_flow",
          "molten_salt_tes", "vrfb", "h2_cavern", "li_lfp"]
TECH_LABELS = {"caes": "CAES", "laes": "LAES",
                "gravity": "Gravity", "fecr_flow": "FeCr flow",
                "molten_salt_tes": "Molten salt TES",
                "vrfb": "VRFB", "h2_cavern": "H2 cavern", "li_lfp": "LFP"}


def load_one_tech(tid: str) -> pd.DataFrame:
    """加载一个 tech 全量 CSV（含 4 节点）。仅必要列，并按 (province_cn,
    node, dispatch_window_h, duration_h) 排序，加速后续切片。"""
    fp = M4_DIR / f"m4_single_tech_{tid}.csv"
    needed = ["scenario", "node", "dispatch_window_h", "tech_id",
                "duration_h", "province_en", "province_cn",
                "satisfy_rate_pct", "annual_supply_kwh",
                "unit_cost_cny_kwh", "gas_unit_cost_cny_kwh",
                "cost_ratio_to_gas"]
    df = pd.read_csv(fp, usecols=needed, low_memory=False)
    df = df[df["scenario"].eq("baseline")].copy()
    print(f"  {tid}: {len(df):,} rows, {df.province_cn.nunique()} prov")
    return df


def build_tech_index(df: pd.DataFrame) -> dict:
    """对单 tech 全表，预先按 (province_cn, node, dw) 分组建索引，
    返回 dict[(prov, node, dw)] -> sub_df。这样每次查 O(1)。"""
    print(f"    building index... ", end="", flush=True)
    idx = {}
    for (prov, node, dw), sub in df.groupby(
            ["province_cn", "node", "dispatch_window_h"], sort=False):
        idx[(prov, node, dw)] = sub
    print(f"done: {len(idx)} groups")
    return idx


def cost_at_satisfy_one_prov(idx_dict: dict, prov_cn: str,
                                node: str, dw: float,
                                satisfy: float = SATISFY_TARGET) -> float | None:
    """对单省单 (node, dw)：用 OSR 选 dur 后取 satisfy=80% 处 cost ratio。

    idx_dict 由 build_tech_index 预先建好，O(1) 查找，不再做全表扫描。
    """
    df_cell = idx_dict.get((prov_cn, node, dw))
    if df_cell is None or df_cell.empty:
        return None
    # 该省每 dur 算 OSR，选最高
    best_dur = None; best_osr = -np.inf
    for dur, g in df_cell.groupby("duration_h", sort=False):
        if len(g) < 2: continue
        osr = ms.compute_osr_for_province(g)
        if osr is None: continue
        if osr > best_osr:
            best_osr = osr; best_dur = int(dur)
    if best_dur is None:
        return None
    sub = df_cell[df_cell["duration_h"].eq(best_dur)]
    g = sub[["satisfy_rate_pct", "cost_ratio_to_gas"]].dropna()
    if len(g) < 2: return None
    g = g.sort_values("satisfy_rate_pct").reset_index(drop=True)
    x = g.satisfy_rate_pct.to_numpy(dtype=float)
    y = g.cost_ratio_to_gas.to_numpy(dtype=float)
    x_u, idx = np.unique(x, return_index=True); y_u = y[idx]
    if len(x_u) < 2: return None
    if satisfy < x_u.min() or satisfy > x_u.max(): return None
    return float(np.interp(satisfy, x_u, y_u))


def build_prov_matrices(prov_cn: str, tech_indices: dict) -> dict[str, np.ndarray]:
    """对单省构造 8 个 tech 的 4×4 矩阵 [Y节点, dw]。"""
    out = {}
    for tid in TECHS:
        idx_dict = tech_indices[tid]
        mat = np.full((len(NODES), len(DWS)), np.nan)
        for i, node in enumerate(NODES):
            for j, dw in enumerate(DWS):
                v = cost_at_satisfy_one_prov(idx_dict, prov_cn, node, dw)
                if v is not None:
                    mat[i, j] = v
        out[tid] = mat
    return out


def render_one_province(prov_cn: str, prov_en: str,
                          tech_indices: dict,
                          vmin: float, vmax: float):
    matrices = build_prov_matrices(prov_cn, tech_indices)

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
        ax.set_ylabel("Decarb node", labelpad=2)
        ax.set_title(TECH_LABELS[tid], fontsize=10, pad=6)
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
    cbar.set_label(f"Cost / gas @ satisfy=80%", fontsize=9)

    fig.suptitle(
        f"Fig 5 progression 省级热力图 — {prov_cn}（{prov_en}）"
        f"\n8 tech × 4 Y节点 × 4 dw, 省级 OSR 选冠军 dur",
        fontsize=11, fontweight="bold", y=0.99)

    fig_base = FIGURE_DIR / f"fig5_progression_{prov_en}_heatmap"
    for ext, kw in {".pdf": {}, ".png": {"dpi": 300}}.items():
        path = fig_base.with_suffix(ext)
        fig.savefig(path, bbox_inches="tight", **kw)
    print(f"  saved: {fig_base.name}")
    plt.close(fig)
    return matrices


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== 加载 8 个 tech 并建索引 ===")
    tech_indices = {}
    sample_provs = None
    cn_to_en_map = {v: k for k, v in EN_TO_CN.items()}
    for tid in TECHS:
        df = load_one_tech(tid)
        tech_indices[tid] = build_tech_index(df)
        if sample_provs is None:
            sample_provs = sorted(df.province_cn.unique())
        del df  # 释放内存

    prov_cns = sample_provs
    print(f"\n=== 渲染 {len(prov_cns)} 省级图 ===")
    print("  估算全局色阶（前 8 省采样）...")
    all_vals = []
    for prov_cn in prov_cns[:8]:
        mats = build_prov_matrices(prov_cn, tech_indices)
        for m in mats.values():
            all_vals.extend(m[np.isfinite(m)])
    all_vals = np.array(all_vals)
    vmin = float(np.percentile(all_vals, 5))
    vmax = float(np.percentile(all_vals, 95))
    print(f"  色阶: vmin={vmin:.2f}, vmax={vmax:.2f}")

    all_rows = []
    for k, prov_cn in enumerate(prov_cns):
        prov_en = cn_to_en_map.get(prov_cn, prov_cn)
        prov_en_safe = prov_en.replace(" ", "_")
        print(f"  [{k+1}/{len(prov_cns)}] {prov_cn} ({prov_en_safe}):",
              end=" ", flush=True)
        try:
            mats = render_one_province(prov_cn, prov_en_safe, tech_indices,
                                          vmin, vmax)
            print("ok")
        except Exception as e:
            print(f"ERROR: {e}")
            continue
        for tid, mat in mats.items():
            for i, node in enumerate(NODES):
                for j, dw in enumerate(DWS):
                    all_rows.append({"province_cn": prov_cn,
                                       "province_en": prov_en_safe,
                                       "tech_id": tid, "node": node,
                                       "dw_h": dw,
                                       "cost_ratio_at_satisfy_80": mat[i, j]})
    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(SOURCE_DIR / "fig5_progression_per_province.csv", index=False)
        print(f"\n  saved source data: {len(df)} rows")


if __name__ == "__main__":
    main()
