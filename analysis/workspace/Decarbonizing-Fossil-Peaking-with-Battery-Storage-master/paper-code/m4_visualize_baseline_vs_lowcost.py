"""
Fig 4e: M4 baseline vs low_cost 双面板 OSR 对比。

每个技术一张子图，横轴 duration_h，纵轴 OSR%（26 省中位 + IQR 阴影），
两条曲线（baseline 实线 / low_cost 虚线）。

DOE 2030 LDES Liftoff 目标区间通过侧边文本块标注：
  inter-day:   $1100-1400/kW + $20-30/kWh
  multi-day:   $1900-2500/kW + $10-15/kWh
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager
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
    "Arial", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "DejaVu Sans",
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
M4_DIR = PAPER_CODE_DIR / "output" / "m4_economics"
FIG_DIR = M4_DIR / "nature_figure"
SOURCE_DIR = FIG_DIR / "source_data"
FIG_BASE = FIG_DIR / "fig4e_baseline_vs_lowcost_osr"


TECH_ORDER = [
    "li_lfp", "caes", "vrfb", "fecr_flow",
    "laes", "gravity", "molten_salt_tes", "h2_cavern",
]
TECH_LABELS = {
    "li_lfp": "LFP", "vrfb": "VRFB", "fecr_flow": "FeCr flow",
    "gravity": "Gravity", "molten_salt_tes": "Molten salt TES",
    "caes": "CAES", "laes": "LAES", "h2_cavern": "H2 cavern",
}
TECH_COLORS = {
    "li_lfp": "#0F4D92", "vrfb": "#3775BA", "fecr_flow": "#42949E",
    "gravity": "#E28E2C", "molten_salt_tes": "#A55B17",
    "caes": "#2A8C2A", "laes": "#8466C2", "h2_cavern": "#7C6CCF",
}

# DOE 2030 LDES Liftoff targets (USD → CNY @ 7.1 fx)
DOE_INTER_DAY_CNY_KWH_AT_10H = (1100 * 7.1 / 10 + 20 * 7.1,
                                 1400 * 7.1 / 10 + 30 * 7.1)
DOE_MULTI_DAY_CNY_KWH_AT_100H = (1900 * 7.1 / 100 + 10 * 7.1,
                                  2500 * 7.1 / 100 + 15 * 7.1)
# inter-day @ 10h ≈ 923-1207 CNY/kWh
# multi-day @ 100h ≈ 206-284 CNY/kWh


def load_data() -> dict[str, pd.DataFrame]:
    out = {}
    for sc in ("baseline", "low_cost"):
        df = pd.read_csv(M4_DIR / f"m4_3r_summary_{sc}.csv")
        df = df[["tech_id", "duration_h", "province_en", "province_cn",
                 "osr_pct", "csr_pct", "psr_pct",
                 "optimal_unit_cost_cny_kwh"]].copy()
        df["scenario"] = sc
        out[sc] = df
    return out


def per_duration_stats(df: pd.DataFrame, tech: str) -> pd.DataFrame:
    """For one tech, aggregate OSR across 26 provinces per duration."""
    sub = df[df.tech_id.eq(tech)].copy()
    if sub.empty:
        return pd.DataFrame()
    g = sub.groupby("duration_h").agg(
        osr_median=("osr_pct", "median"),
        osr_q25=("osr_pct", lambda x: x.quantile(0.25)),
        osr_q75=("osr_pct", lambda x: x.quantile(0.75)),
        osr_max=("osr_pct", "max"),
        n_provinces_pos=("osr_pct", lambda x: int((x > 0).sum())),
    ).reset_index()
    return g.sort_values("duration_h").reset_index(drop=True)


def draw_panel(ax, tech: str, data: dict[str, pd.DataFrame],
               y_max: float = 95.0) -> None:
    c = TECH_COLORS[tech]

    # Track best-case (max across provinces) too, since many techs have median=0
    for sc, style, alpha_band in (
        ("baseline", dict(ls="-", marker="o", alpha=0.95, ms=4), 0.10),
        ("low_cost", dict(ls="--", marker="s", alpha=0.95, ms=4), 0.20),
    ):
        stats = per_duration_stats(data[sc], tech)
        if stats.empty:
            continue
        # IQR shaded band (typically near 0 for LDES)
        ax.fill_between(stats["duration_h"], stats["osr_q25"], stats["osr_q75"],
                        color=c, alpha=alpha_band)
        # max-province line (lighter)
        ax.plot(stats["duration_h"], stats["osr_max"],
                color=c, ls=style["ls"], lw=0.8, alpha=0.4)
        # median line + markers
        ax.plot(stats["duration_h"], stats["osr_median"],
                color=c, **style)

    # n_provinces_pos annotation (top-right of panel)
    base_stats = per_duration_stats(data["baseline"], tech)
    low_stats = per_duration_stats(data["low_cost"], tech)
    if not base_stats.empty and not low_stats.empty:
        # Use representative middle duration
        mid = base_stats.iloc[len(base_stats) // 2]
        mid_low = low_stats.iloc[len(low_stats) // 2]
        ax.text(0.97, 0.95,
                f"n_prov(OSR>0):\n  base={int(mid.n_provinces_pos)}\n"
                f"  low={int(mid_low.n_provinces_pos)}",
                transform=ax.transAxes, fontsize=6.2,
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="#bbb", lw=0.4, alpha=0.85))

    ax.set_title(TECH_LABELS[tech], loc="left", fontsize=9, fontweight="bold")
    ax.set_xlabel("Duration (h)", fontsize=7)
    ax.set_ylabel("OSR (%)", fontsize=7)
    ax.set_ylim(-3, y_max)
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.axhline(0, color="#444", lw=0.5, alpha=0.6)
    ax.tick_params(labelsize=7)


def fmt_doe_annotation(ax) -> None:
    """Side text annotation summarizing DOE 2030 target."""
    txt = (
        "DOE 2030 LDES Liftoff targets:\n"
        f"  Inter-day (4-24h): "
        f"{DOE_INTER_DAY_CNY_KWH_AT_10H[0]:.0f}-"
        f"{DOE_INTER_DAY_CNY_KWH_AT_10H[1]:.0f} CNY/kWh @ 10h\n"
        f"  Multi-day (24-168h): "
        f"{DOE_MULTI_DAY_CNY_KWH_AT_100H[0]:.0f}-"
        f"{DOE_MULTI_DAY_CNY_KWH_AT_100H[1]:.0f} CNY/kWh @ 100h"
    )
    ax.text(0.99, 0.985, txt, transform=ax.transAxes, fontsize=6.5,
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="#F8F4E3",
                      ec="#B89F4A", lw=0.6))


def draw_scenario_costs(ax, m2_csv: Path) -> None:
    """Bottom strip: show baseline vs low_cost total CAPEX per tech (4h or default)."""
    df = pd.read_csv(m2_csv)
    rows = []
    for tech in TECH_ORDER:
        r = df[df.tech_id.eq(tech)].iloc[0]
        dh = r["default_duration_h"]
        base_total = r["power_cost_cny_kw_base"] / dh + r["energy_cost_cny_kwh_base"]
        low_total = r["power_cost_cny_kw_low"] / dh + r["energy_cost_cny_kwh_low"]
        rows.append({
            "tech": tech, "duration_h": dh,
            "baseline_total_cny_kwh": base_total,
            "lowcost_total_cny_kwh": low_total,
            "drop_pct": (base_total - low_total) / base_total * 100,
        })
    cost_df = pd.DataFrame(rows)

    x = np.arange(len(TECH_ORDER))
    w = 0.36
    ax.bar(x - w/2, cost_df["baseline_total_cny_kwh"], w,
           color=[TECH_COLORS[t] for t in TECH_ORDER],
           alpha=0.55, edgecolor="none", label="baseline")
    ax.bar(x + w/2, cost_df["lowcost_total_cny_kwh"], w,
           color=[TECH_COLORS[t] for t in TECH_ORDER],
           alpha=0.95, edgecolor="none", label="low_cost")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{TECH_LABELS[t]}\n@{int(cost_df.iloc[i].duration_h)}h"
                        for i, t in enumerate(TECH_ORDER)],
                       fontsize=6.8)
    ax.set_yscale("log")
    ax.set_ylabel("Total CAPEX (CNY/kWh)", fontsize=7)
    ax.set_title("Total CAPEX comparison (baseline vs low_cost)",
                 loc="left", fontsize=9, fontweight="bold")
    ax.grid(True, axis="y", which="both", alpha=0.2, lw=0.4)
    # DOE target reference bands
    ax.axhspan(*DOE_INTER_DAY_CNY_KWH_AT_10H, color="#F4D35E", alpha=0.18,
               label="DOE 2030 inter-day @10h")
    ax.axhspan(*DOE_MULTI_DAY_CNY_KWH_AT_100H, color="#EE964B", alpha=0.18,
               label="DOE 2030 multi-day @100h")
    ax.legend(loc="upper left", fontsize=6.5, ncol=2)
    # drop pct labels
    for i, t in enumerate(TECH_ORDER):
        d = cost_df.iloc[i]["drop_pct"]
        ax.text(i, cost_df.iloc[i]["lowcost_total_cny_kwh"] * 0.55,
                f"−{d:.0f}%", ha="center", va="top", fontsize=6,
                color="#444", fontweight="bold")


def export_source_data(data: dict[str, pd.DataFrame]) -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for sc in ("baseline", "low_cost"):
        for tech in TECH_ORDER:
            stats = per_duration_stats(data[sc], tech)
            if stats.empty:
                continue
            stats["tech_id"] = tech
            stats["scenario"] = sc
            rows.append(stats)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(SOURCE_DIR / "fig4e_source_osr_per_duration.csv", index=False)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()

    # 8 tech subplots (2x4) + 1 bottom-row CAPEX strip → 3x4 grid with one row spanning
    fig = plt.figure(figsize=(14.0, 12.0))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.05],
                          hspace=0.50, wspace=0.30,
                          left=0.055, right=0.985, top=0.88, bottom=0.05)

    # 2 rows × 4 cols of tech panels
    for i, tech in enumerate(TECH_ORDER):
        row, col = divmod(i, 4)
        ax = fig.add_subplot(gs[row, col])
        draw_panel(ax, tech, data)

    # Global legend at figure level (top center) instead of cluttering LFP panel
    legend_handles = [
        plt.Line2D([0], [0], color="#444", ls="-", marker="o", ms=5,
                   label="baseline median"),
        plt.Line2D([0], [0], color="#444", ls="--", marker="s", ms=5,
                   label="low_cost median"),
        plt.Line2D([0], [0], color="#444", ls="-", lw=0.8, alpha=0.4,
                   label="best province (max)"),
        plt.Rectangle((0, 0), 1, 1, fc="#666", alpha=0.18,
                      label="IQR across 26 provinces"),
    ]
    fig.legend(handles=legend_handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.985), ncol=4, fontsize=8,
               frameon=False)

    # DOE 2030 target box: place as figure-level text in top-right corner
    doe_txt = (
        "DOE 2030 LDES Liftoff cost targets (USD → CNY @ 7.1):\n"
        f"  Inter-day (4–24h): $1100–1400/kW + $20–30/kWh "
        f"≈ {DOE_INTER_DAY_CNY_KWH_AT_10H[0]:.0f}–"
        f"{DOE_INTER_DAY_CNY_KWH_AT_10H[1]:.0f} CNY/kWh @10h\n"
        f"  Multi-day (24–168h): $1900–2500/kW + $10–15/kWh "
        f"≈ {DOE_MULTI_DAY_CNY_KWH_AT_100H[0]:.0f}–"
        f"{DOE_MULTI_DAY_CNY_KWH_AT_100H[1]:.0f} CNY/kWh @100h"
    )
    fig.text(0.99, 0.945, doe_txt, fontsize=7, ha="right", va="top",
             bbox=dict(boxstyle="round,pad=0.5", fc="#FFF6D5",
                       ec="#B89F4A", lw=0.6))

    # Bottom row: total CAPEX comparison
    ax_bottom = fig.add_subplot(gs[2, :])
    m2_csv = (PAPER_CODE_DIR / ".." / ".." / "dataset"
              / "M2_LDES_technology_parameter_library_v0_2.csv")
    draw_scenario_costs(ax_bottom, m2_csv)

    fig.suptitle("Fig 4e | Baseline vs Low-Cost LDES Scenarios — OSR shifts and CAPEX gap",
                 fontsize=12, fontweight="bold", y=0.995)

    for ext in ("svg", "pdf", "png"):
        out = FIG_BASE.with_suffix(f".{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"saved: {out}")
    plt.close(fig)

    export_source_data(data)
    print(f"source data: {SOURCE_DIR / 'fig4e_source_osr_per_duration.csv'}")


if __name__ == "__main__":
    main()
