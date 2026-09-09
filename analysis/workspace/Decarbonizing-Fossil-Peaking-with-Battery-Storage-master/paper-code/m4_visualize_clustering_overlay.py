"""
Fig 4b: M4 经济边界叠加 M1 极端事件聚集度。

不覆盖原 fig4_m4_economics_nature。本图基于全年 M3/M4 结果，
将省级聚集度指标 (mean_run_length, p22) 作为分组/横轴维度，
展示聚集度如何影响不同 LDES 技术的经济性。

四 Panel:
  a. OSR vs mean_run_length 散点 (8 技术，各省点 + 线性拟合)
  b. OSR vs P(2→2) 散点 (聚焦多日峰持续概率)
  c. 高/低聚集度省份分组的 OSR 箱线图 (按技术)
  d. 26 省 × 8 技术 OSR 热力图 (省份按 mean_run_length 排序)
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
for font_path in CJK_FONT_PATHS:
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Arial",
    "Noto Sans CJK JP",
    "WenQuanYi Zen Hei",
    "DejaVu Sans",
]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.size"] = 8.0
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.9
plt.rcParams["legend.frameon"] = False


PAPER_CODE_DIR = Path(__file__).resolve().parent
M1_CSV = PAPER_CODE_DIR / "output" / "m1_clustering" / "province_clustering_metrics.csv"
M4_CSV = PAPER_CODE_DIR / "output" / "m4_economics" / "m4_3r_summary_all_scenarios.csv"
FIGURE_DIR = PAPER_CODE_DIR / "output" / "m4_economics" / "nature_figure"
SOURCE_DIR = FIGURE_DIR / "source_data"
FIG_BASE = FIGURE_DIR / "fig4b_m4_clustering_overlay"


TECH_ORDER = [
    "li_lfp", "vrfb", "fecr_flow", "gravity",
    "molten_salt_tes", "caes", "laes", "h2_cavern",
]
TECH_LABELS = {
    "li_lfp": "LFP", "vrfb": "VRFB", "fecr_flow": "FeCr flow",
    "gravity": "Gravity", "molten_salt_tes": "Molten salt TES",
    "caes": "CAES", "laes": "LAES", "h2_cavern": "H2 cavern",
}
TECH_COLORS = {
    "li_lfp": "#0F4D92", "vrfb": "#3775BA", "fecr_flow": "#42949E",
    "gravity": "#E28E2C", "molten_salt_tes": "#767676",
    "caes": "#8BCF8B", "laes": "#B9A7E8", "h2_cavern": "#7C6CCF",
}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    m1 = pd.read_csv(M1_CSV)
    m4 = pd.read_csv(M4_CSV)
    m4 = m4[m4["cost_scenario"].eq("baseline")].copy()
    # 每个 tech × province 取最优 duration（OSR 最大）
    best = (
        m4.sort_values("osr_pct", ascending=False)
          .groupby(["tech_id", "province_en"], as_index=False)
          .head(1)
    )
    merged = best.merge(
        m1[["province_en", "mean_run_length", "p22", "frac_runs_ge3",
            "max_run_length", "n_extreme_days"]],
        on="province_en", how="inner",
    )
    return m1, merged


def _linfit(x: np.ndarray, y: np.ndarray):
    """简单 OLS：返回 slope, intercept, r2。"""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, np.nan, np.nan
    x, y = x[mask], y[mask]
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return slope, intercept, r2


def draw_panel_a(ax, df: pd.DataFrame) -> None:
    """OSR vs mean_run_length，各技术散点 + 拟合线。"""
    for tech in TECH_ORDER:
        sub = df[df["tech_id"].eq(tech)]
        if sub.empty:
            continue
        c = TECH_COLORS[tech]
        ax.scatter(sub["mean_run_length"], sub["osr_pct"],
                   s=10, color=c, alpha=0.55, edgecolors="none")
        slope, intercept, r2 = _linfit(
            sub["mean_run_length"].to_numpy(),
            sub["osr_pct"].to_numpy(),
        )
        if np.isfinite(slope):
            xline = np.linspace(sub["mean_run_length"].min(),
                                sub["mean_run_length"].max(), 50)
            ax.plot(xline, slope * xline + intercept,
                    color=c, lw=1.1, alpha=0.85)
    ax.set_xlabel("Mean run length of extreme days (days)")
    ax.set_ylabel("OSR (%)")
    ax.set_title("a. OSR vs extreme-day clustering", loc="left",
                 fontsize=9, fontweight="bold")
    ax.grid(True, alpha=0.25, lw=0.5)


def draw_panel_b(ax, df: pd.DataFrame) -> None:
    """OSR vs p22 (P(2→2)) 散点。"""
    for tech in TECH_ORDER:
        sub = df[df["tech_id"].eq(tech)]
        if sub.empty:
            continue
        c = TECH_COLORS[tech]
        ax.scatter(sub["p22"], sub["osr_pct"],
                   s=10, color=c, alpha=0.55, edgecolors="none",
                   label=TECH_LABELS[tech])
        slope, intercept, r2 = _linfit(
            sub["p22"].to_numpy(),
            sub["osr_pct"].to_numpy(),
        )
        if np.isfinite(slope):
            xline = np.linspace(sub["p22"].min(), sub["p22"].max(), 50)
            ax.plot(xline, slope * xline + intercept,
                    color=c, lw=1.1, alpha=0.85)
    ax.set_xlabel("P(2→2): extreme-day persistence probability")
    ax.set_ylabel("OSR (%)")
    ax.set_title("b. OSR vs multi-day persistence", loc="left",
                 fontsize=9, fontweight="bold")
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.legend(loc="upper left", fontsize=6.5, ncol=2,
              handletextpad=0.4, columnspacing=0.7)


def draw_panel_c(ax, df: pd.DataFrame) -> None:
    """按聚集度高/低分组的 OSR 箱线图（每技术 2 箱）。"""
    median_rl = df.groupby("province_en")["mean_run_length"].first().median()
    df = df.copy()
    df["cluster_group"] = np.where(
        df["mean_run_length"] >= median_rl, "high", "low",
    )

    positions, labels = [], []
    width = 0.36
    for i, tech in enumerate(TECH_ORDER):
        for j, grp in enumerate(("low", "high")):
            sub = df[(df["tech_id"].eq(tech)) & (df["cluster_group"].eq(grp))]
            if sub.empty:
                continue
            pos = i + (j - 0.5) * width
            box = ax.boxplot(
                sub["osr_pct"].to_numpy(),
                positions=[pos], widths=width * 0.85,
                patch_artist=True, showfliers=False, whis=1.2,
            )
            face = TECH_COLORS[tech]
            for patch in box["boxes"]:
                patch.set_facecolor(face)
                patch.set_alpha(0.45 if grp == "low" else 0.85)
                patch.set_edgecolor(face)
                patch.set_linewidth(0.8)
            for med in box["medians"]:
                med.set_color("#262626")
                med.set_linewidth(1.0)
            for whisker in box["whiskers"]:
                whisker.set_color(face)
                whisker.set_linewidth(0.8)
            for cap in box["caps"]:
                cap.set_color(face)
                cap.set_linewidth(0.8)
        positions.append(i)
        labels.append(TECH_LABELS[tech])

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("OSR (%)")
    ax.set_title(
        f"c. OSR by clustering tier (split at median run length = {median_rl:.2f} d)",
        loc="left", fontsize=9, fontweight="bold",
    )
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc="#808080", alpha=0.45, ec="#404040"),
        plt.Rectangle((0, 0), 1, 1, fc="#808080", alpha=0.85, ec="#404040"),
    ]
    ax.legend(legend_handles, ["Low clustering", "High clustering"],
              loc="upper right", fontsize=7)
    ax.grid(True, axis="y", alpha=0.25, lw=0.5)


def draw_panel_d(ax, df: pd.DataFrame) -> None:
    """26 省 × 8 技术 OSR 热力图，省份按 mean_run_length 排序。"""
    pivot = (
        df.pivot_table(index="province_en", columns="tech_id",
                       values="osr_pct", aggfunc="max")
          .reindex(columns=TECH_ORDER)
    )
    rl = df.groupby("province_en")["mean_run_length"].first()
    order = rl.sort_values(ascending=True).index
    pivot = pivot.reindex(order)

    cn_map = df.drop_duplicates("province_en").set_index("province_en")["province_cn"]
    y_labels = [
        f"{cn_map.get(p, p)} ({rl[p]:.2f})" for p in pivot.index
    ]

    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="YlGnBu",
                   vmin=0, vmax=float(np.nanmax(pivot.to_numpy())))
    ax.set_xticks(range(len(TECH_ORDER)))
    ax.set_xticklabels([TECH_LABELS[t] for t in TECH_ORDER],
                       rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(y_labels, fontsize=6.5)
    ax.set_title(
        "d. OSR heatmap by province (sorted by mean run length, ↑)",
        loc="left", fontsize=9, fontweight="bold",
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("OSR (%)", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5)


def export_source_data(m1: pd.DataFrame, merged: pd.DataFrame) -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    keep = [
        "tech_id", "technology_cn", "province_en", "province_cn",
        "duration_h", "osr_pct", "csr_pct", "psr_pct",
        "optimal_unit_cost_cny_kwh", "optimal_capacity_mwh",
        "mean_run_length", "p22", "frac_runs_ge3",
        "max_run_length", "n_extreme_days",
    ]
    merged[keep].to_csv(SOURCE_DIR / "fig4b_source_osr_vs_clustering.csv",
                        index=False)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    m1, merged = load_data()

    fig = plt.figure(figsize=(11.2, 9.2))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28,
                          left=0.07, right=0.97, top=0.95, bottom=0.07)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    draw_panel_a(ax_a, merged)
    draw_panel_b(ax_b, merged)
    draw_panel_c(ax_c, merged)
    draw_panel_d(ax_d, merged)

    for ext in ("svg", "pdf", "png"):
        out_path = FIG_BASE.with_suffix(f".{ext}")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"saved: {out_path}")
    plt.close(fig)

    export_source_data(m1, merged)
    print(f"source data: {SOURCE_DIR / 'fig4b_source_osr_vs_clustering.csv'}")


if __name__ == "__main__":
    main()
