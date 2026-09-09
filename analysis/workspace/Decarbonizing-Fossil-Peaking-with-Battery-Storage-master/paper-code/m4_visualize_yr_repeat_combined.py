"""
Fig 5 实验 A — 5×4 复制年合图（行=年限 1/2/3/4/5，列=dw 8/12/16/24h）

把 5 张 1×4 矩阵图合并成一张 5×4 大图，便于直接对比仿真年限影响。
每行标注 Yr1/2/3/4/5 复制年。

数据源：m4_single_tech_yr<N>_repeat_<tech>.csv
输出：output/m4_economics/nature_figure/fig5_yr_repeat_combined_5x4.{png,svg,pdf}
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

import m4_visualize_single_tech_generic as ms

# 复用配置
DW_VALUES = ms.DW_VALUES
SCENARIO = ms.SCENARIO
TECH_FILES = ms.TECH_FILES
TECH_COLORS = ms.TECH_COLORS
PALETTE = ms.PALETTE


PAPER_CODE_DIR = Path(__file__).resolve().parent
M4_DIR = PAPER_CODE_DIR / "output" / "m4_economics"
FIGURE_DIR = M4_DIR / "nature_figure"
SOURCE_DIR = FIGURE_DIR / "source_data"
FIG_BASE = FIGURE_DIR / "fig5_yr_repeat_combined_5x4"


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--champion-by", default="unit_cost",
                        choices=["unit_cost", "osr"],
                        help="冠军 dur 选法：unit_cost 或 osr "
                              "（与 fig4 panel a 对齐）")
    args = parser.parse_args()
    champion_by = args.champion_by

    print(f"=== 加载 5 个复制年 × 8 tech 数据 (champion_by={champion_by}) ===")
    all_dfs = {}
    for n in [1, 2, 3, 4, 5]:
        stem = f"yr{n}_repeat"
        print(f"  Yr{n}_repeat:")
        all_dfs[n] = ms.load_year_techs(stem, "m4_single_tech")

    fig = plt.figure(figsize=(18.0, 19.0))
    gs = fig.add_gridspec(5, 4,
                            left=0.055, right=0.985,
                            top=0.955, bottom=0.04,
                            wspace=0.22, hspace=0.32)

    source_rows = []
    letters = "abcdefghijklmnopqrst"
    idx = 0
    for r, n in enumerate([1, 2, 3, 4, 5]):
        tech_dfs = all_dfs[n]
        for c, dw_h in enumerate(DW_VALUES):
            ax = fig.add_subplot(gs[r, c])
            sd, _ = ms.draw_cell(
                ax, tech_dfs, dw_h, letters[idx],
                show_xlabel=(r == 4),
                show_ylabel=(c == 0),
                show_legend=True,
                champion_by=champion_by,
            )
            if not sd.empty:
                sd["n_years"] = n
                source_rows.append(sd)
            # 每行最左侧加年限标注
            if c == 0:
                ax.text(-0.32, 0.5,
                         f"Yr{n}_repeat\n({n}×365 d)",
                         transform=ax.transAxes,
                         ha="center", va="center",
                         fontsize=11, fontweight="bold",
                         color=PALETTE["black"],
                         rotation=90)
            idx += 1

    suffix = f"_{champion_by}" if champion_by != "unit_cost" else ""
    fig.text(0.055, 0.985,
              f"Fig 5 实验 A — 复制年单技术合图（5×4：行=年限 1/2/3/4/5, "
              f"列=dw 8/12/16/24h，冠军 dur 选法={champion_by}）",
              ha="left", va="top",
              fontsize=12, fontweight="bold", color=PALETTE["black"])
    fig.text(0.055, 0.970,
              "demand = 1 年真实数据简单复制 N 次（无 markov、无极端注入）。"
              "每子图 8 条曲线为各技术单独充放电（25 MWh 全量扫描）的 26 省"
              "插值中位（panel a 同款逻辑）。预期：随年限增加曲线形态不变，"
              "证明仿真年限本身不影响 single-tech 经济性。"
              "Red dashed = gas parity.",
              ha="left", va="top",
              fontsize=7.6, color=PALETTE["black"])

    fig_base = FIGURE_DIR / f"fig5_yr_repeat_combined_5x4{suffix}"
    for ext, kw in {".svg": {}, ".pdf": {}, ".png": {"dpi": 300}}.items():
        path = fig_base.with_suffix(ext)
        fig.savefig(path, bbox_inches="tight", **kw)
        print(f"  saved: {path.name}")

    if source_rows:
        sd = pd.concat(source_rows, ignore_index=True)
        sd.to_csv(SOURCE_DIR / f"fig5_yr_repeat_combined_5x4{suffix}.csv",
                    index=False)
        print(f"  source data: {len(sd)} rows")


if __name__ == "__main__":
    main()
