"""
Fig 4j: 8 类 LDES 单年 vs 双年加强 OSR 中位对比（每技术一个 panel，横轴 duration）。

每 panel 4 条曲线：
  - 1yr baseline (实线 + 圆点)
  - 1yr low_cost (实线 + 叉)
  - 2yr baseline (虚线 + 圆点)
  - 2yr low_cost (虚线 + 叉)
+ 每条曲线 26 省 IQR 阴影带

输出：
  output/m4_economics/nature_figure/fig4j_1yr_vs_2yr_osr.{png,pdf,svg}
"""

from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PAPER_CODE_DIR = Path(__file__).resolve().parent
ECON_DIR = PAPER_CODE_DIR / "output" / "m4_economics"
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

TECH_ORDER = [
    ("li_lfp", "LFP (lithium)"),
    ("caes", "CAES"),
    ("gravity", "Gravity"),
    ("laes", "LAES"),
    ("vrfb", "VRFB"),
    ("fecr_flow", "FeCr flow"),
    ("molten_salt_tes", "Molten salt TES"),
    ("h2_cavern", "H2 cavern"),
]

COLOR_BASE = "#1F4E79"
COLOR_LOW = "#C75B12"


def load_summary(scenario: str, tag: str | None) -> pd.DataFrame:
    suffix = f"_{tag}" if tag else ""
    path = ECON_DIR / f"m4_3r_summary_{scenario}{suffix}.csv"
    df = pd.read_csv(path)
    return df


def aggregate(df: pd.DataFrame, tech: str) -> pd.DataFrame:
    sub = df[df["tech_id"] == tech]
    g = sub.groupby("duration_h")["osr_pct"].agg(
        median="median", q25=lambda s: s.quantile(0.25),
        q75=lambda s: s.quantile(0.75),
    ).reset_index()
    return g


def draw_panel(ax, tech_id: str, label: str, dfs: dict):
    """dfs key: (n_yr, scenario) -> DataFrame."""
    styles = {
        (1, "baseline"): dict(color=COLOR_BASE, ls="-", marker="o",
                                label="1yr baseline"),
        (1, "low_cost"): dict(color=COLOR_LOW, ls="-", marker="x",
                                label="1yr low_cost"),
        (2, "baseline"): dict(color=COLOR_BASE, ls="--", marker="o",
                                label="2yr baseline", mfc="white"),
        (2, "low_cost"): dict(color=COLOR_LOW, ls="--", marker="x",
                                label="2yr low_cost"),
    }

    for (n_yr, scen), df in dfs.items():
        g = aggregate(df, tech_id)
        if g.empty:
            continue
        st = styles[(n_yr, scen)]
        ax.plot(g["duration_h"], g["median"], lw=1.6, ms=5, **st)
        ax.fill_between(g["duration_h"], g["q25"], g["q75"],
                         color=st["color"], alpha=0.08, lw=0)

    # Δmedian annotation: 2yr_baseline − 1yr_baseline at largest shared duration
    g1 = aggregate(dfs[(1, "baseline")], tech_id)
    g2 = aggregate(dfs[(2, "baseline")], tech_id)
    if not g1.empty and not g2.empty:
        common = sorted(set(g1.duration_h) & set(g2.duration_h))
        if common:
            d_anchor = common[len(common) // 2]
            v1 = g1.set_index("duration_h").loc[d_anchor, "median"]
            v2 = g2.set_index("duration_h").loc[d_anchor, "median"]
            delta = v2 - v1
            ax.annotate(
                f"Δ(2yr−1yr) baseline @ {d_anchor}h: {delta:+.1f} pp",
                xy=(0.02, 0.93), xycoords="axes fraction",
                fontsize=7.5, color="#333",
                bbox=dict(boxstyle="round,pad=0.25", fc="#F5F5F5",
                          ec="#999", lw=0.5),
            )

    ax.set_title(label, loc="left", fontsize=10, fontweight="bold")
    ax.set_xlabel("duration (h)", fontsize=8)
    ax.set_ylabel("OSR median (%)", fontsize=8)
    ax.set_ylim(-2, 100)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3, lw=0.5)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading 4 summary CSVs...")
    dfs = {
        (1, "baseline"): load_summary("baseline", None),
        (1, "low_cost"): load_summary("low_cost", None),
        (2, "baseline"): load_summary("baseline", "2yr"),
        (2, "low_cost"): load_summary("low_cost", "2yr"),
    }
    for k, v in dfs.items():
        print(f"  {k}: {len(v)} rows")

    fig, axes = plt.subplots(4, 2, figsize=(11.5, 13.5))
    for ax, (tid, label) in zip(axes.flat, TECH_ORDER):
        draw_panel(ax, tid, label, dfs)

    # global legend
    handles = [
        plt.Line2D([0], [0], color=COLOR_BASE, ls="-", marker="o",
                    label="1yr baseline"),
        plt.Line2D([0], [0], color=COLOR_LOW, ls="-", marker="x",
                    label="1yr low_cost"),
        plt.Line2D([0], [0], color=COLOR_BASE, ls="--", marker="o",
                    mfc="white", label="2yr baseline"),
        plt.Line2D([0], [0], color=COLOR_LOW, ls="--", marker="x",
                    label="2yr low_cost"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.005), fontsize=9, frameon=False)
    fig.suptitle(
        "Fig 4j  |  单年 vs 双年加强：8 类 LDES OSR 中位对比",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0.025, 1, 0.985])

    for ext in ["png", "pdf", "svg"]:
        path = OUT_DIR / f"fig4j_1yr_vs_2yr_osr.{ext}"
        fig.savefig(path, dpi=300 if ext == "png" else None,
                     bbox_inches="tight")
        print(f"  saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
