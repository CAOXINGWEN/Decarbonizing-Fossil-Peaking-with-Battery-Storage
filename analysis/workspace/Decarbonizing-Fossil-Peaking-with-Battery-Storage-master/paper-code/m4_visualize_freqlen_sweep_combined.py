"""
M4 实验 B 与 C 合图 — 频率扫描 4×4 与 长度扫描 4×4 矩阵图

按 fig5_yr_repeat_combined_5x4 的 OSR 同款逻辑。

实验 B（频率扫描）：
  行 = events ∈ {3, 5, 7, 10}/年（length=5 天固定）
  列 = dw 8/12/16/24h
  输出: fig5_freq_sweep_4x4{_osr,}.png

实验 C（长度扫描）：
  行 = length ∈ {5, 7, 10, 14} 天（events=5/年固定）
  列 = dw 8/12/16/24h
  输出: fig5_len_sweep_4x4{_osr,}.png

用法：
  python m4_visualize_freqlen_sweep_combined.py --champion-by osr
  python m4_visualize_freqlen_sweep_combined.py --champion-by unit_cost
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
    """渲染 4×4 合图：4 行（每行一个 stem）× 4 列 dw。"""
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
                         color=PALETTE["black"], rotation=90)
            idx += 1

    suffix = f"_{champion_by}" if champion_by != "unit_cost" else ""
    fig.text(0.060, 0.985, title,
              ha="left", va="top",
              fontsize=12, fontweight="bold", color=PALETTE["black"])
    fig.text(0.060, 0.970, footer,
              ha="left", va="top",
              fontsize=7.6, color=PALETTE["black"])

    fig_base = FIGURE_DIR / f"{out_base}{suffix}"
    for ext, kw in {".svg": {}, ".pdf": {}, ".png": {"dpi": 300}}.items():
        path = fig_base.with_suffix(ext)
        fig.savefig(path, bbox_inches="tight", **kw)
        print(f"  saved: {path.name}")
    plt.close(fig)

    if source_rows:
        sd = pd.concat(source_rows, ignore_index=True)
        sd.to_csv(SOURCE_DIR / f"{out_base}{suffix}.csv", index=False)
        print(f"  source data: {len(sd)} rows")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--champion-by", default="osr",
                        choices=["unit_cost", "osr"])
    parser.add_argument("--quantile", default="q75",
                        choices=["q90", "q75"],
                        help="选哪一组档位（q75=新版基线 6.0/省/年）")
    args = parser.parse_args()

    if args.quantile == "q75":
        # q75 标准（与 M1 模块对齐，基线 ≥3 天事件 6.0/省/年）
        # freq 4/8/12/16 对应基线 0.7×、1.3×、2.0×、2.7×
        EVENTS = [4, 8, 12, 16]
        suffix_q = "_q75"
    else:
        # q90 旧版
        EVENTS = [3, 5, 7, 10]
        suffix_q = ""

    # 实验 B：频率扫描（length=5 固定）
    freq_stems = [
        (f"freq{e}_len5", f"events={e}/yr (length=5d)")
        for e in EVENTS
    ]
    render_combined(
        freq_stems,
        out_base=f"fig5_freq_sweep_4x4{suffix_q}",
        title=f"Fig 5 实验 B — 极端事件频率扫描（4×4：行=events {'/'.join(map(str,EVENTS))}/年, "
               f"列=dw 8/12/16/24h, length=5d 固定, q75 基线 6.0/省/年）",
        footer="5 年 markov 仿真（1 真实 + 4 markov），Y1 物理参数（demand×1.0、"
                "extreme_boost=1.3、event_intensity=1.5）。每子图 8 条单技术冠军 "
                "duration 曲线（OSR 选法）。25 MWh 全量扫描，26 省插值取中位"
                "（panel a 同款逻辑）。Red dashed = gas parity.",
        champion_by=args.champion_by,
    )

    # 实验 C：长度扫描（events=基线第二档固定）
    base_freq = EVENTS[1]   # q75: 8, q90: 5
    len_stems = [
        (f"freq{base_freq}_len{l}", f"length={l}d (events={base_freq}/yr)")
        for l in [5, 7, 10, 14]
    ]
    render_combined(
        len_stems,
        out_base=f"fig5_len_sweep_4x4{suffix_q}",
        title=f"Fig 5 实验 C — 极端事件长度扫描（4×4：行=length 5/7/10/14 天, "
               f"列=dw 8/12/16/24h, events={base_freq}/年 固定）",
        footer="5 年 markov 仿真，Y1 物理参数。每子图 8 条单技术冠军 duration "
                "曲线（OSR 选法）。25 MWh 全量扫描，26 省插值取中位。"
                "Red dashed = gas parity.",
        champion_by=args.champion_by,
    )


if __name__ == "__main__":
    main()
