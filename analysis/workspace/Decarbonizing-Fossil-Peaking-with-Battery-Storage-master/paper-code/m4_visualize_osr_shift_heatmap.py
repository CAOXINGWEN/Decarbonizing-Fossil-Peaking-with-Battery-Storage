"""
Fig 4k: 26 省 × 8 技术 OSR 跃升幅度热图。

ΔOSR = OSR_2yr − OSR_1yr (pp，每个 (province, tech) 取所有 duration 上的最大 OSR)

布局：
  左：baseline 热图    右：low_cost 热图
  右侧附 1 列：mean_run_length_state2 (来自 m3_real_data_stats.csv)
  底部 1 行：每技术全国中位 ΔOSR

输出：
  output/m4_economics/nature_figure/fig4k_osr_shift_heatmap.{png,pdf,svg}
"""

from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PAPER_CODE_DIR = Path(__file__).resolve().parent
ECON_DIR = PAPER_CODE_DIR / "output" / "m4_economics"
M3_DIR = PAPER_CODE_DIR / "output" / "m3_dispatch"
OUT_DIR = ECON_DIR / "nature_figure"

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
plt.rcParams["pdf.fonttype"] = 42

TECH_ORDER = ["li_lfp", "caes", "gravity", "laes",
              "vrfb", "fecr_flow", "molten_salt_tes", "h2_cavern"]
TECH_LABELS = ["LFP", "CAES", "Gravity", "LAES",
                "VRFB", "FeCr", "Molten\nSalt", "H2"]


def best_osr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """对每个 (province_en, tech_id) 取所有 duration 上的最大 OSR。"""
    g = df.groupby(["province_en", "tech_id"])["osr_pct"].max().unstack("tech_id")
    g = g.reindex(columns=TECH_ORDER)
    return g


def draw_heatmap(ax, mat: pd.DataFrame, title: str,
                  cmap="RdBu_r", vmax=40):
    norm = TwoSlopeNorm(vmin=-vmax/4, vcenter=0, vmax=vmax)
    im = ax.imshow(mat.values, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(TECH_ORDER)))
    ax.set_xticklabels(TECH_LABELS, fontsize=8, rotation=0)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index, fontsize=7)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")

    # 标注 |Δ| > 5pp 的单元格
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.iat[i, j]
            if pd.isna(v) or abs(v) < 5:
                continue
            color = "white" if abs(v) > vmax * 0.5 else "#1a1a1a"
            ax.text(j, i, f"{v:+.0f}", ha="center", va="center",
                     fontsize=6.5, color=color)
    return im


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading 4 summary CSVs...")
    df_1b = pd.read_csv(ECON_DIR / "m4_3r_summary_baseline.csv")
    df_1l = pd.read_csv(ECON_DIR / "m4_3r_summary_low_cost.csv")
    df_2b = pd.read_csv(ECON_DIR / "m4_3r_summary_baseline_2yr.csv")
    df_2l = pd.read_csv(ECON_DIR / "m4_3r_summary_low_cost_2yr.csv")

    M_1b = best_osr_matrix(df_1b)
    M_1l = best_osr_matrix(df_1l)
    M_2b = best_osr_matrix(df_2b)
    M_2l = best_osr_matrix(df_2l)

    # 省份排序：按双年 baseline 总弃电量隐含的 demand 降序（用 total_demand_kwh 之和）
    prov_order = (df_2b.groupby("province_en")["total_demand_kwh"].first()
                   .sort_values(ascending=False).index.tolist())
    M_1b = M_1b.reindex(prov_order)
    M_1l = M_1l.reindex(prov_order)
    M_2b = M_2b.reindex(prov_order)
    M_2l = M_2l.reindex(prov_order)

    delta_b = (M_2b - M_1b)
    delta_l = (M_2l - M_1l)

    # mean_run_length 列
    stats = pd.read_csv(M3_DIR / "m3_real_data_stats.csv")
    mrl = stats.set_index("province")["mean_run_length_state2"].reindex(prov_order)

    fig = plt.figure(figsize=(14.5, 10))
    gs = fig.add_gridspec(2, 3, width_ratios=[5, 5, 0.9],
                           height_ratios=[10, 0.9],
                           hspace=0.05, wspace=0.18)

    ax_b = fig.add_subplot(gs[0, 0])
    ax_l = fig.add_subplot(gs[0, 1])
    ax_mrl = fig.add_subplot(gs[0, 2])
    ax_med_b = fig.add_subplot(gs[1, 0])
    ax_med_l = fig.add_subplot(gs[1, 1])
    ax_med_l2 = fig.add_subplot(gs[1, 2])
    ax_med_l2.axis("off")

    im_b = draw_heatmap(ax_b, delta_b,
                         "a. baseline  ΔOSR = OSR_2yr − OSR_1yr (pp)")
    im_l = draw_heatmap(ax_l, delta_l,
                         "b. low_cost  ΔOSR = OSR_2yr − OSR_1yr (pp)")
    ax_l.set_yticklabels([])

    # mean_run_length 列
    mrl_arr = mrl.values.reshape(-1, 1)
    im_mrl = ax_mrl.imshow(mrl_arr, aspect="auto", cmap="YlGn",
                            vmin=1.0, vmax=2.5)
    ax_mrl.set_xticks([0])
    ax_mrl.set_xticklabels(["MeanRun(d)"], fontsize=7.5, rotation=0)
    ax_mrl.set_yticks([])
    for i, v in enumerate(mrl.values):
        if not pd.isna(v):
            ax_mrl.text(0, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=6.5, color="#1a1a1a")
    ax_mrl.set_title("c. 真实年", loc="left", fontsize=10, fontweight="bold")

    # 底行：每技术中位 ΔOSR
    med_b = delta_b.median(axis=0).values.reshape(1, -1)
    med_l = delta_l.median(axis=0).values.reshape(1, -1)
    ax_med_b.imshow(med_b, aspect="auto", cmap="RdBu_r",
                     norm=TwoSlopeNorm(vmin=-10, vcenter=0, vmax=40))
    ax_med_b.set_xticks(range(len(TECH_LABELS)))
    ax_med_b.set_xticklabels(TECH_LABELS, fontsize=8)
    ax_med_b.set_yticks([0])
    ax_med_b.set_yticklabels(["全国中位"], fontsize=7.5)
    for j, v in enumerate(med_b.flatten()):
        if not pd.isna(v):
            ax_med_b.text(j, 0, f"{v:+.1f}", ha="center", va="center",
                           fontsize=7.5, fontweight="bold")
    ax_med_l.imshow(med_l, aspect="auto", cmap="RdBu_r",
                     norm=TwoSlopeNorm(vmin=-10, vcenter=0, vmax=40))
    ax_med_l.set_xticks(range(len(TECH_LABELS)))
    ax_med_l.set_xticklabels(TECH_LABELS, fontsize=8)
    ax_med_l.set_yticks([0])
    ax_med_l.set_yticklabels(["全国中位"], fontsize=7.5)
    for j, v in enumerate(med_l.flatten()):
        if not pd.isna(v):
            ax_med_l.text(j, 0, f"{v:+.1f}", ha="center", va="center",
                           fontsize=7.5, fontweight="bold")

    # 右侧 colorbar
    cbar = fig.colorbar(im_l, ax=[ax_b, ax_l], shrink=0.7,
                         orientation="vertical", pad=0.18,
                         location="right")
    cbar.set_label("ΔOSR (pp)", fontsize=9)

    fig.suptitle("Fig 4k  |  双年加强后 26 省 × 8 技术 OSR 跃升热图",
                  fontsize=13, fontweight="bold", y=0.995)

    for ext in ["png", "pdf", "svg"]:
        path = OUT_DIR / f"fig4k_osr_shift_heatmap.{ext}"
        fig.savefig(path, dpi=300 if ext == "png" else None,
                     bbox_inches="tight")
        print(f"  saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
