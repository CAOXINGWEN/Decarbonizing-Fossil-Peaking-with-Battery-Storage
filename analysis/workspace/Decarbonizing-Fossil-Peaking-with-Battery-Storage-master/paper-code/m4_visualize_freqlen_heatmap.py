"""
M4 实验 D — 频率×长度热力图（8 tech × dw=12h）

每个 tech 一幅 4×4 热力图：
  行 = events ∈ {3, 5, 7, 10}/年
  列 = length ∈ {5, 7, 10, 14} 天
  颜色 = 该 (freq, len) 组合下 26 省 cost/gas 中位 @ satisfy=80%
        （插值 26 省曲线到 satisfy=80% 处取中位）

8 幅热力图组合成 2×4 大图。

用法：python m4_visualize_freqlen_heatmap.py
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

# q75 版（与 M1 模块对齐，基线 ≥3 天事件 6.0/省/年）
import os
QUANTILE = os.environ.get("FREQLEN_QUANTILE", "q75")
if QUANTILE == "q75":
    EVENTS = [4, 8, 12, 16]
    FIG_BASE = FIGURE_DIR / "fig5_freqlen_heatmap_2x4_q75"
else:
    EVENTS = [3, 5, 7, 10]
    FIG_BASE = FIGURE_DIR / "fig5_freqlen_heatmap_2x4"
LENGTHS = [5, 7, 10, 14]
DW_TARGET = 12.0
SATISFY_TARGET = 80.0   # 在 satisfy=80% 处取 cost ratio 作为热力图值

TECHS = ["caes", "laes", "gravity", "fecr_flow",
          "molten_salt_tes", "vrfb", "h2_cavern", "li_lfp"]
TECH_LABELS = {"caes":"CAES","laes":"LAES","gravity":"Gravity",
                "fecr_flow":"FeCr flow",
                "molten_salt_tes":"Molten salt TES",
                "vrfb":"VRFB","h2_cavern":"H2 cavern","li_lfp":"LFP"}


def cost_at_satisfy(stem: str, tid: str, dw: float = DW_TARGET,
                     satisfy: float = SATISFY_TARGET,
                     champion_by: str = "osr") -> float | None:
    """加载该 (stem, tid) CSV，按 OSR 选 dur，
    在 satisfy=80% 处取 26 省 cost_ratio_to_gas 中位。"""
    fp = M4_DIR / f"m4_single_tech_{stem}_{tid}.csv"
    if not fp.exists():
        return None
    needed = ["scenario","dispatch_window_h","tech_id","duration_h",
                "province_en","province_cn","capacity_mwh","satisfy_rate_pct",
                "annual_supply_kwh","unit_cost_cny_kwh",
                "gas_unit_cost_cny_kwh","cost_ratio_to_gas"]
    df = pd.read_csv(fp, usecols=needed, low_memory=False)
    df = df[df["scenario"].eq("baseline") & df["dispatch_window_h"].eq(dw)]
    if df.empty:
        return None
    if champion_by == "osr":
        best = ms.champion_duration_osr(df)
    else:
        best = ms.champion_duration(df)
    if best is None:
        return None
    sub = df[df["duration_h"].eq(best)]
    # 每省按 satisfy 升序，插值到 satisfy=80% 处
    vals = []
    for prov, g in sub.groupby("province_cn"):
        g = g[["satisfy_rate_pct","cost_ratio_to_gas"]].dropna()
        if len(g) < 2: continue
        g = g.sort_values("satisfy_rate_pct").reset_index(drop=True)
        x = g["satisfy_rate_pct"].to_numpy(dtype=float)
        y = g["cost_ratio_to_gas"].to_numpy(dtype=float)
        x_u, idx = np.unique(x, return_index=True); y_u = y[idx]
        if len(x_u) < 2: continue
        if satisfy < x_u.min() or satisfy > x_u.max():
            continue
        vals.append(float(np.interp(satisfy, x_u, y_u)))
    if not vals:
        return None
    return float(np.median(vals))


def build_heatmap(tid: str, champion_by: str = "osr") -> np.ndarray:
    """4×4 热力图矩阵 [freq, len]。"""
    mat = np.full((len(EVENTS), len(LENGTHS)), np.nan)
    for i, ev in enumerate(EVENTS):
        for j, ln in enumerate(LENGTHS):
            stem = f"freq{ev}_len{ln}"
            v = cost_at_satisfy(stem, tid, champion_by=champion_by)
            if v is not None:
                mat[i, j] = v
    return mat


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== 构造 {len(TECHS)} 个 tech 的 4×4 热力图（dw={DW_TARGET}h, "
          f"satisfy={SATISFY_TARGET}%）===")
    matrices = {}
    rows = []
    for tid in TECHS:
        print(f"  {tid}:")
        mat = build_heatmap(tid)
        matrices[tid] = mat
        print(f"    {mat}")
        for i, ev in enumerate(EVENTS):
            for j, ln in enumerate(LENGTHS):
                rows.append({"tech_id": tid, "events_per_yr": ev,
                              "length_days": ln,
                              "cost_ratio_at_satisfy_80": mat[i, j],
                              "dw_h": DW_TARGET})
    src_df = pd.DataFrame(rows)
    src_df.to_csv(SOURCE_DIR / f"{FIG_BASE.name}.csv", index=False)

    # 全局 vmin/vmax
    all_vals = np.concatenate([m.flatten() for m in matrices.values()])
    all_vals = all_vals[np.isfinite(all_vals)]
    vmin, vmax = float(np.percentile(all_vals, 5)), float(np.percentile(all_vals, 95))
    print(f"\n  色阶范围: vmin={vmin:.3f}, vmax={vmax:.3f}")

    # 2×4 子图布局
    fig, axes = plt.subplots(2, 4, figsize=(18, 9.5),
                                gridspec_kw={"wspace": 0.35, "hspace": 0.45})
    for k, tid in enumerate(TECHS):
        ax = axes[k // 4, k % 4]
        mat = matrices[tid]
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r",
                        vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(LENGTHS)))
        ax.set_xticklabels([f"{l}d" for l in LENGTHS])
        ax.set_yticks(range(len(EVENTS)))
        ax.set_yticklabels([f"{e}/yr" for e in EVENTS])
        ax.set_xlabel("Length (days)", labelpad=2)
        ax.set_ylabel("Events / year", labelpad=2)
        ax.set_title(f"{TECH_LABELS[tid]}", fontsize=10, pad=6)
        # 单元格内标注数值
        for i in range(len(EVENTS)):
            for j in range(len(LENGTHS)):
                v = mat[i, j]
                if np.isfinite(v):
                    text_color = "white" if v > (vmin + vmax) / 2 else "black"
                    ax.text(j, i, f"{v:.2f}",
                              ha="center", va="center",
                              color=text_color, fontsize=8)

    # 全局 colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(f"Cost / gas @ satisfy={SATISFY_TARGET:.0f}% "
                    f"(dw={DW_TARGET:.0f}h)",
                    fontsize=9)

    fig.suptitle(
        f"Fig 5 实验 D — 极端事件频率×长度热力图（8 tech × dw={DW_TARGET:.0f}h, "
        f"OSR 选法选冠军 dur, 26 省 cost/gas 中位 @ satisfy={SATISFY_TARGET:.0f}%）",
        fontsize=11, fontweight="bold", y=0.98)

    fig.text(0.06, 0.94,
              "色阶: 红=贵, 绿=便宜（gas parity = 1.0）。"
              f"行=极端事件频率（{EVENTS}/年），列=单事件长度（{LENGTHS} 天）。"
              "每单元格代表一种 (freq, length) 组合下该 tech 的省级中位单位成本。",
              fontsize=8.0, ha="left", va="top")

    for ext, kw in {".svg": {}, ".pdf": {}, ".png": {"dpi": 300}}.items():
        path = FIG_BASE.with_suffix(ext)
        fig.savefig(path, bbox_inches="tight", **kw)
        print(f"  saved: {path.name}")
    plt.close(fig)


if __name__ == "__main__":
    main()
