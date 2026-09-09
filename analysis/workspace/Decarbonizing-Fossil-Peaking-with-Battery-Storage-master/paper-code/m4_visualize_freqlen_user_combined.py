"""
M4 实验 F + G — 用户定义的频率/长度扫描 4×4 合图

实验 F：频率扫描（60% / 70% / 80% / 90% 非零日比例）
实验 G：长度扫描（5 / 7 / 10 / 14 天平均段长）

每个实验出一张 4×4 矩阵图：
  行 = 4 个目标参数（频率或长度）
  列 = dw 8/12/16/24h
每子图 8 条曲线（LFP + 7 LDES，OSR 选冠军 dur，26 省插值取中位）

数据源：m4_single_tech_F<freq>_real_<tech>.csv 与 m4_single_tech_G<len>_real_<tech>.csv

用法：python m4_visualize_freqlen_user_combined.py [--champion-by osr|unit_cost]
"""

from __future__ import annotations
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import m4_visualize_single_tech_generic as ms

DW_VALUES = ms.DW_VALUES
PALETTE = ms.PALETTE


PAPER_CODE_DIR = Path(__file__).resolve().parent
M4_DIR = PAPER_CODE_DIR / "output" / "m4_economics"
FIGURE_DIR = M4_DIR / "nature_figure"
SOURCE_DIR = FIGURE_DIR / "source_data"


def render_combined(stems_with_labels, out_base, title, footer,
                     champion_by: str = "osr"):
    """4×4 合图：行=stems，列=dw。"""
    print(f"=== 加载 {len(stems_with_labels)} 个节点 × 8 tech ===")
    all_dfs = {}
    for stem, _ in stems_with_labels:
        print(f"  {stem}:")
        all_dfs[stem] = ms.load_year_techs(stem, "m4_single_tech")

    fig = plt.figure(figsize=(18.0, 16.0))
    gs = fig.add_gridspec(4, 4,
                            left=0.060, right=0.985,
                            top=0.945, bottom=0.045,
                            wspace=0.22, hspace=0.32)

    source_rows = []
    letters = "abcdefghijklmnop"
    idx = 0
    for r, (stem, label) in enumerate(stems_with_labels):
        tech_dfs = all_dfs[stem]
        for c, dw_h in enumerate(DW_VALUES):
            ax = fig.add_subplot(gs[r, c])
            sd, _ = ms.draw_cell(
                ax, tech_dfs, dw_h, letters[idx],
                show_xlabel=(r == 3),
                show_ylabel=(c == 0),
                show_legend=True,
                champion_by=champion_by,
            )
            if not sd.empty:
                sd["row_label"] = label
                sd["row_stem"] = stem
                source_rows.append(sd)
            if c == 0:
                ax.text(-0.32, 0.5, label,
                         transform=ax.transAxes,
                         ha="center", va="center",
                         fontsize=11, fontweight="bold",
                         color=PALETTE["black"], rotation=90,
                         **ms.cjk_font_kwargs(bold=True))
            idx += 1

    fig.text(0.060, 0.985, title,
              ha="left", va="top",
              fontsize=12, fontweight="bold", color=PALETTE["black"],
              **ms.cjk_font_kwargs(bold=True))
    fig.text(0.060, 0.970, footer,
              ha="left", va="top",
              fontsize=7.6, color=PALETTE["black"],
              **ms.cjk_font_kwargs())

    fig_base = FIGURE_DIR / out_base
    for ext, kw in {".svg": {}, ".pdf": {}, ".png": {"dpi": 300}}.items():
        path = fig_base.with_suffix(ext)
        fig.savefig(path, bbox_inches="tight", **kw)
        print(f"  saved: {path.name}")
    plt.close(fig)

    if source_rows:
        sd = pd.concat(source_rows, ignore_index=True)
        sd.to_csv(SOURCE_DIR / f"{out_base}.csv", index=False)
        print(f"  source data: {len(sd)} rows")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--champion-by", default="osr",
                        choices=["unit_cost", "osr"])
    args = parser.parse_args()

    suffix = f"_{args.champion_by}" if args.champion_by != "unit_cost" else ""

    # 实验 F：频率扫描
    freq_stems = [
        ("F60_real", "freq=60% (real distribution)"),
        ("F70_real", "freq=70% (real distribution)"),
        ("F80_real", "freq=80% (real distribution)"),
        ("F90_real", "freq=90% (real distribution)"),
    ]
    render_combined(
        freq_stems,
        out_base=f"fig5_F_freq_user_4x4{suffix}",
        title="Fig 5 实验 F — 用户定义频率扫描（4×4：行=非零调峰日比例 60/70/80/90%, "
               "列=dw 8/12/16/24h）",
        footer="基于真实 365 天数据，从每省真实非零调峰需求分布中重采样新增"
                "非零日，提高频率（实际真实均值=49%）。每子图 8 条单技术冠军 "
                "duration 曲线（OSR 选法）。25 MWh 全量扫描，26 省插值取中位。"
                "Red dashed = gas parity.",
        champion_by=args.champion_by,
    )

    # 实验 G：长度扫描
    len_stems = [
        ("G5_real", "avg_run_length=5d (~实际 4.9d)"),
        ("G7_real", "avg_run_length=7d (~实际 6.6d)"),
        ("G10_real", "avg_run_length=10d (~实际 8.7d)"),
        ("G14_real", "avg_run_length=14d (~实际 10.0d)"),
    ]
    render_combined(
        len_stems,
        out_base=f"fig5_G_len_user_4x4{suffix}",
        title="Fig 5 实验 G — 用户定义长度扫描（4×4：行=连续段平均长度 5/7/10/14 天, "
               "列=dw 8/12/16/24h）",
        footer="保持每省真实非零调峰天数（=真实），重组连续段使平均段长达到目标。"
                "数值从真实非零分布抽样。每子图 8 条单技术冠军 duration 曲线"
                "（OSR 选法）。Red dashed = gas parity.",
        champion_by=args.champion_by,
    )


if __name__ == "__main__":
    main()
