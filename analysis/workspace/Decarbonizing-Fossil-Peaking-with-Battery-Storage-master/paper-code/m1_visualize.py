"""
M1 可视化：极端调峰需求聚集结构分析图。

生成 4 张图：
  Fig 1. 各省 P(2→2) 与平均运行长度散点图
  Fig 2. 26 省年度运行长度分布（箱线图）
  Fig 3. 季节 × 省份 P(2→2) 热力图
  Fig 4. 北京示例：全年 state 序列 + 极端运行标注
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Noto Sans CJK SC", "SimHei", "WenQuanYi Zen Hei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "m1_clustering")


def load_csvs():
    annual = pd.read_csv(os.path.join(OUTPUT_DIR, "province_clustering_metrics.csv"))
    seasonal = pd.read_csv(os.path.join(OUTPUT_DIR, "province_season_clustering.csv"))
    runs = pd.read_csv(os.path.join(OUTPUT_DIR, "province_extreme_event_runs.csv"))
    return annual, seasonal, runs


# ─── Fig 1: P(2→2) vs Mean Run-Length ───────────────────────────

def fig1_scatter(annual, fig_dir):
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = annual["best_fit_model"].map(
        {"geometric": "#2196F3", "negbin": "#FF5722"}
    )
    sizes = annual["n_runs"] * 3

    ax.scatter(annual["p22"], annual["mean_run_length"],
               c=colors, s=sizes, alpha=0.75, edgecolors="white", linewidths=0.5)

    for _, row in annual.iterrows():
        ax.annotate(row["province_cn"], (row["p22"], row["mean_run_length"]),
                    fontsize=7, ha="left", va="bottom",
                    xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("P(2→2) 状态自持概率", fontsize=11)
    ax.set_ylabel("平均运行长度（天）", fontsize=11)
    ax.set_title("各省极端调峰需求聚集度", fontsize=13)

    legend_elements = [
        mpatches.Patch(facecolor="#2196F3", label="几何分布"),
        mpatches.Patch(facecolor="#FF5722", label="负二项分布"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.text(0.13, 1.02, "独立到达基线", fontsize=7, color="gray")

    path = os.path.join(fig_dir, "fig1_p22_vs_mean_run.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


# ─── Fig 2: 运行长度箱线图 ──────────────────────────────────────

def fig2_boxplot(runs, annual, fig_dir):
    annual_runs = runs[runs["season"] == "annual"].copy()
    order = annual.sort_values("mean_run_length", ascending=False)["province_cn"].tolist()
    en_to_cn = dict(zip(annual["province_en"], annual["province_cn"]))
    annual_runs["province_cn"] = annual_runs["province_en"].map(en_to_cn)

    data_by_province = []
    labels = []
    for prov_cn in order:
        lengths = annual_runs.loc[annual_runs["province_cn"] == prov_cn, "run_length"].values
        if len(lengths) > 0:
            data_by_province.append(lengths)
            labels.append(prov_cn)

    fig, ax = plt.subplots(figsize=(14, 5))
    bp = ax.boxplot(data_by_province, vert=True, patch_artist=True, widths=0.6)
    for patch in bp["boxes"]:
        patch.set_facecolor("#90CAF9")
        patch.set_edgecolor("#1565C0")
    for median in bp["medians"]:
        median.set_color("#D32F2F")
        median.set_linewidth(1.5)

    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("运行长度（天）", fontsize=11)
    ax.set_title("各省极端调峰事件运行长度分布", fontsize=13)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    path = os.path.join(fig_dir, "fig2_run_length_boxplot.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


# ─── Fig 3: 季节 × 省份 P(2→2) 热力图 ──────────────────────────

def fig3_heatmap(seasonal, annual, fig_dir):
    order = annual.sort_values("p22", ascending=False)["province_en"].tolist()
    season_order = ["winter", "spring", "summer", "autumn"]
    season_cn = {"winter": "冬", "spring": "春", "summer": "夏", "autumn": "秋"}

    en_to_cn = dict(zip(annual["province_en"], annual["province_cn"]))

    pivot = seasonal.pivot_table(
        index="province_en", columns="season", values="p22"
    )
    pivot = pivot.reindex(index=order, columns=season_order)
    pivot_display = pivot.rename(index=en_to_cn, columns=season_cn)

    fig, ax = plt.subplots(figsize=(6, 10))
    im = ax.imshow(pivot_display.values, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=0.7)

    ax.set_xticks(range(len(pivot_display.columns)))
    ax.set_xticklabels(pivot_display.columns, fontsize=10)
    ax.set_yticks(range(len(pivot_display.index)))
    ax.set_yticklabels(pivot_display.index, fontsize=8)

    for i in range(pivot_display.shape[0]):
        for j in range(pivot_display.shape[1]):
            val = pivot_display.iloc[i, j]
            color = "white" if val > 0.4 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    ax.set_title("季节 × 省份 P(2→2) 热力图", fontsize=13, pad=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("P(2→2)", fontsize=10)

    path = os.path.join(fig_dir, "fig3_season_p22_heatmap.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


# ─── Fig 4: 北京全年 state 时间序列 ─────────────────────────────

def fig4_beijing_timeline(fig_dir):
    from event_clustering import load_demand_matrix, EN_TO_CN
    from monte_carlo_simulation import classify_states
    from config import MC_CONFIG

    data_path = os.path.join(
        os.path.dirname(__file__), "..", "data",
        "Daily Natural Gas Peaking Power Generation by Province in China.xlsx",
    )
    df = load_demand_matrix(data_path)
    values = df["Beijing"].values.astype(float)
    nonzero = values[values > 0]
    q75 = np.percentile(nonzero, MC_CONFIG["intensity_quantile"] * 100)
    states = classify_states(values, q75)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 5), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    days = np.arange(1, len(values) + 1)
    ax1.bar(days, values / 1e6, width=1.0, color="#BBDEFB", edgecolor="none")

    extreme_mask = states == 2
    ax1.bar(days[extreme_mask], values[extreme_mask] / 1e6,
            width=1.0, color="#E53935", edgecolor="none", alpha=0.85)

    ax1.axhline(y=q75 / 1e6, color="#FF9800", linestyle="--", linewidth=1)
    ax1.text(5, q75 / 1e6 * 1.05, f"Q75 = {q75/1e6:.1f} GWh", fontsize=8, color="#FF9800")

    ax1.set_ylabel("日调峰需求 (GWh)", fontsize=10)
    ax1.set_title("北京市全年调峰需求与极端事件聚集", fontsize=13)

    state_colors = {0: "#E0E0E0", 1: "#90CAF9", 2: "#E53935"}
    for i, s in enumerate(states):
        ax2.barh(0, 1, left=i + 1, height=0.6, color=state_colors[s], edgecolor="none")
    ax2.set_yticks([])
    ax2.set_xlabel("日序号（1月1日 = 1）", fontsize=10)
    ax2.set_xlim(1, 366)

    legend_elements = [
        mpatches.Patch(facecolor="#E0E0E0", label="State 0 (零需求)"),
        mpatches.Patch(facecolor="#90CAF9", label="State 1 (低需求)"),
        mpatches.Patch(facecolor="#E53935", label="State 2 (极端需求)"),
    ]
    ax2.legend(handles=legend_elements, loc="upper right", fontsize=8, ncol=3)

    path = os.path.join(fig_dir, "fig4_beijing_timeline.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


# ─── Fig 5: 需求值偏度-峰度散点图 ──────────────────────────────

def fig5_skewness_kurtosis(annual, fig_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    datasets = [
        ("demand_skewness", "demand_kurtosis", "调峰需求值（非零）"),
        ("extreme_skewness", "extreme_kurtosis", "极端需求值（State 2）"),
        ("runlen_skewness", "runlen_kurtosis", "运行长度"),
    ]

    for ax, (sk_col, ku_col, title) in zip(axes, datasets):
        sc = ax.scatter(annual[sk_col], annual[ku_col],
                        c=annual["p22"], cmap="YlOrRd", s=80,
                        edgecolors="white", linewidths=0.5, vmin=0.1, vmax=0.56)
        for _, row in annual.iterrows():
            ax.annotate(row["province_cn"], (row[sk_col], row[ku_col]),
                        fontsize=6, ha="left", va="bottom",
                        xytext=(2, 2), textcoords="offset points")
        ax.set_xlabel("偏度 (Skewness)", fontsize=10)
        ax.set_ylabel("峰度 (Kurtosis)", fontsize=10)
        ax.set_title(title, fontsize=11)

    cbar = fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.04)
    cbar.set_label("P(2→2)", fontsize=10)
    fig.suptitle("省级调峰需求分布形态与聚集度", fontsize=13, y=1.02)

    path = os.path.join(fig_dir, "fig5_skewness_kurtosis.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


# ─── Fig 6: 偏度/峰度 vs P(2→2) 相关性 ─────────────────────────

def fig6_shape_vs_clustering(annual, fig_dir):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    pairs = [
        ("demand_skewness", "需求偏度"),
        ("demand_kurtosis", "需求峰度"),
        ("extreme_skewness", "极端值偏度"),
        ("extreme_kurtosis", "极端值峰度"),
        ("runlen_skewness", "运行长度偏度"),
        ("runlen_kurtosis", "运行长度峰度"),
    ]

    for ax, (col, label) in zip(axes.ravel(), pairs):
        x = annual[col].values
        y = annual["p22"].values
        valid = np.isfinite(x) & np.isfinite(y)
        x_v, y_v = x[valid], y[valid]

        ax.scatter(x_v, y_v, c="#1565C0", s=60, alpha=0.7,
                   edgecolors="white", linewidths=0.5)

        if len(x_v) > 2:
            z = np.polyfit(x_v, y_v, 1)
            x_line = np.linspace(x_v.min(), x_v.max(), 50)
            ax.plot(x_line, np.polyval(z, x_line), color="#E53935",
                    linewidth=1.5, linestyle="--")
            r = np.corrcoef(x_v, y_v)[0, 1]
            ax.text(0.05, 0.92, f"r = {r:.3f}", transform=ax.transAxes,
                    fontsize=9, color="#E53935")

        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel("P(2→2)", fontsize=10)

    fig.suptitle("分布形态指标与聚集度的相关性", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    path = os.path.join(fig_dir, "fig6_shape_vs_clustering.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


# ─── 入口 ────────────────────────────────────────────────────────

def main():
    fig_dir = os.path.join(OUTPUT_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("加载数据...")
    annual, seasonal, runs = load_csvs()

    print("生成图 1: P(2→2) vs 平均运行长度散点图")
    fig1_scatter(annual, fig_dir)

    print("生成图 2: 运行长度箱线图")
    fig2_boxplot(runs, annual, fig_dir)

    print("生成图 3: 季节×省份 P(2→2) 热力图")
    fig3_heatmap(seasonal, annual, fig_dir)

    print("生成图 4: 北京全年 state 时间序列")
    fig4_beijing_timeline(fig_dir)

    print("生成图 5: 偏度-峰度散点图")
    fig5_skewness_kurtosis(annual, fig_dir)

    print("生成图 6: 分布形态 vs 聚集度相关性")
    fig6_shape_vs_clustering(annual, fig_dir)

    print(f"\n所有图已保存至 {fig_dir}/")


if __name__ == "__main__":
    main()
