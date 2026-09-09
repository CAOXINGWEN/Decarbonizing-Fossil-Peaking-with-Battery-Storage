"""
M3 真实 365 天数据统计描述。

目的：
  1) 给出 26 省 demand 序列的完整统计画像
  2) 验证 monte_carlo_simulation.py 的 Markov 链参数（state 阈值、转移矩阵）
  3) 输出统计报表 + PDF 报告，作为虚拟年生成的对照基准

输出:
  output/m3_dispatch/m3_real_data_stats.csv
  output/m3_dispatch/m3_real_data_stats_report.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monte_carlo_simulation import (
    classify_states, estimate_transition_matrix,
    compute_steady_state, load_peak_demand,
)


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


PAPER_CODE_DIR = Path(__file__).resolve().parent
DATA_PATH = PAPER_CODE_DIR.parent / "data" / \
    "Daily Natural Gas Peaking Power Generation by Province in China.xlsx"
OUTPUT_DIR = PAPER_CODE_DIR / "output" / "m3_dispatch"
STATS_CSV = OUTPUT_DIR / "m3_real_data_stats.csv"
TRANS_CSV = OUTPUT_DIR / "m3_real_data_transitions.csv"
PDF_PATH = OUTPUT_DIR / "m3_real_data_stats_report.pdf"


def per_province_stats(demand: np.ndarray, province: str) -> dict:
    """计算单省 demand 序列的关键统计量。"""
    nonzero = demand[demand > 0]
    if len(nonzero) == 0:
        return {"province": province, "n_days_zero": int((demand == 0).sum()),
                "total_kwh": 0.0}

    q75 = np.percentile(nonzero, 75)
    states = classify_states(demand, q75)
    T = estimate_transition_matrix(states, n_states=3)
    pi = compute_steady_state(T)

    # mean run length of state=2
    runs = []
    cur = 0
    for s in states:
        if s == 2:
            cur += 1
        else:
            if cur > 0:
                runs.append(cur)
            cur = 0
    if cur > 0:
        runs.append(cur)
    mean_run = float(np.mean(runs)) if runs else 0.0
    max_run = int(max(runs)) if runs else 0
    n_extreme = int((states == 2).sum())

    return {
        "province": province,
        "total_kwh": float(demand.sum()),
        "n_days_zero": int((demand == 0).sum()),
        "n_days_state1": int((states == 1).sum()),
        "n_days_state2": int(n_extreme),
        "mean_kwh": float(demand.mean()),
        "median_kwh": float(np.median(demand)),
        "q75_kwh": float(q75),
        "q90_kwh": float(np.percentile(nonzero, 90)),
        "q99_kwh": float(np.percentile(nonzero, 99)),
        "max_kwh": float(demand.max()),
        "p_00": float(T[0, 0]), "p_01": float(T[0, 1]), "p_02": float(T[0, 2]),
        "p_10": float(T[1, 0]), "p_11": float(T[1, 1]), "p_12": float(T[1, 2]),
        "p_20": float(T[2, 0]), "p_21": float(T[2, 1]), "p_22": float(T[2, 2]),
        "pi_0": float(pi[0]), "pi_1": float(pi[1]), "pi_2": float(pi[2]),
        "mean_run_length_state2": mean_run,
        "max_run_length_state2": max_run,
        "n_extreme_runs": len(runs),
    }


def make_pdf_report(stats_df: pd.DataFrame, demand_df: pd.DataFrame,
                    out_path: Path) -> None:
    """生成 26 省的统计可视化 PDF（多页）。"""
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(out_path) as pdf:
        # === 页 1：26 省总量 + 极端日数对比 ===
        fig, axes = plt.subplots(2, 1, figsize=(11, 10))
        ax1 = axes[0]
        order = stats_df.sort_values("total_kwh", ascending=False)
        ax1.barh(order["province"], order["total_kwh"] / 1e9,
                 color="#3775BA")
        ax1.set_xlabel("年总调峰量 (GWh)")
        ax1.set_title("26 省 2023 年燃气调峰总量",
                      loc="left", fontweight="bold", fontsize=11)
        ax1.invert_yaxis()
        ax1.grid(True, axis="x", alpha=0.3)

        ax2 = axes[1]
        order2 = stats_df.sort_values("n_days_state2", ascending=False)
        ax2.barh(order2["province"], order2["n_days_state2"], color="#E28E2C")
        ax2.set_xlabel("极端日数 (state=2，demand > Q75)")
        ax2.set_title("26 省极端日数分布", loc="left",
                      fontweight="bold", fontsize=11)
        ax2.invert_yaxis()
        ax2.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # === 页 2：mean_run_length 与 P(2->2) ===
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        ax1, ax2 = axes
        order3 = stats_df.sort_values("mean_run_length_state2", ascending=False)
        ax1.barh(order3["province"], order3["mean_run_length_state2"],
                 color="#42949E")
        ax1.set_xlabel("Mean run length of state=2 (days)")
        ax1.set_title("a. 极端日平均连续长度", loc="left",
                      fontweight="bold", fontsize=11)
        ax1.invert_yaxis()
        ax1.axvline(1.5, color="grey", ls="--", lw=0.8)
        ax1.grid(True, axis="x", alpha=0.3)

        order4 = stats_df.sort_values("p_22", ascending=False)
        ax2.barh(order4["province"], order4["p_22"], color="#7C6CCF")
        ax2.set_xlabel("P(2→2): 极端日次日仍是极端日的概率")
        ax2.set_title("b. 极端态自相关 P(2→2)", loc="left",
                      fontweight="bold", fontsize=11)
        ax2.invert_yaxis()
        ax2.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # === 页 3：6 个代表省的 demand 时间序列 ===
        rep_provs = ["Inner Mongolia", "Xinjiang", "Jiangsu",
                     "Hebei", "Beijing", "Fujian"]
        fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
        for ax, prov in zip(axes.flat, rep_provs):
            if prov not in demand_df.index:
                continue
            arr = demand_df.loc[prov].values
            day_idx = np.arange(len(arr))
            ax.fill_between(day_idx, 0, arr / 1e6, color="#3775BA", alpha=0.7)
            q75 = np.percentile(arr[arr > 0], 75) if (arr > 0).any() else 0
            ax.axhline(q75 / 1e6, color="red", ls="--", lw=0.8,
                       label=f"Q75 = {q75/1e6:.2f} GWh")
            ax.set_title(prov, loc="left", fontweight="bold", fontsize=10)
            ax.set_ylabel("Daily peaking (GWh)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=7)
        axes[-1, 0].set_xlabel("Day of year")
        axes[-1, 1].set_xlabel("Day of year")
        plt.suptitle("6 个代表省 365 天 demand 时间序列",
                     fontsize=12, fontweight="bold", y=0.995)
        plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # === 页 4：极端日数 vs 总量散点 ===
        fig, ax = plt.subplots(figsize=(9, 7))
        sc = ax.scatter(
            stats_df["total_kwh"] / 1e9,
            stats_df["mean_run_length_state2"],
            c=stats_df["n_days_state2"], cmap="YlOrRd",
            s=80, edgecolors="black", lw=0.5,
        )
        for _, r in stats_df.iterrows():
            ax.annotate(r["province"][:8],
                        (r["total_kwh"] / 1e9, r["mean_run_length_state2"]),
                        fontsize=6.5, alpha=0.7, xytext=(3, 2),
                        textcoords="offset points")
        ax.set_xlabel("年总调峰量 (GWh)")
        ax.set_ylabel("Mean run length state=2 (days)")
        ax.set_title("总量 × 聚集度 × 极端日数 三维分布",
                     loc="left", fontweight="bold", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("极端日数")
        plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # === 页 5：关键统计表 ===
        fig, ax = plt.subplots(figsize=(11.5, 13))
        ax.axis("off")
        cols = ["province", "total_kwh", "n_days_state2",
                "q75_kwh", "q99_kwh",
                "mean_run_length_state2", "max_run_length_state2",
                "p_22", "pi_2"]
        sub = stats_df[cols].copy()
        sub["total_kwh"] = (sub["total_kwh"] / 1e9).round(2)
        sub["q75_kwh"] = (sub["q75_kwh"] / 1e6).round(2)
        sub["q99_kwh"] = (sub["q99_kwh"] / 1e6).round(2)
        sub["mean_run_length_state2"] = sub["mean_run_length_state2"].round(2)
        sub["p_22"] = sub["p_22"].round(3)
        sub["pi_2"] = sub["pi_2"].round(3)
        sub.columns = ["省", "总量\n(GWh)", "极端\n日数",
                       "Q75\n(GWh)", "Q99\n(GWh)",
                       "MeanRun\n(d)", "MaxRun\n(d)",
                       "P(2→2)", "π2"]
        sub = sub.sort_values("总量\n(GWh)", ascending=False)
        tab = ax.table(cellText=sub.values, colLabels=sub.columns,
                       cellLoc="center", loc="center")
        tab.auto_set_font_size(False)
        tab.set_fontsize(8)
        tab.scale(1, 1.2)
        for i in range(len(sub.columns)):
            cell = tab[(0, i)]
            cell.set_facecolor("#1F4E79")
            cell.set_text_props(weight="bold", color="white")
        ax.set_title("26 省 demand 关键统计指标（2023 年真实数据）",
                     fontweight="bold", fontsize=12, pad=10)
        pdf.savefig(fig); plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"加载真实 demand 数据: {DATA_PATH}")
    raw = pd.read_excel(DATA_PATH, sheet_name=0)
    raw = raw.set_index(raw.columns[0])
    # 替换列名为整数日序
    raw.columns = list(range(1, len(raw.columns) + 1))
    print(f"  shape: {raw.shape}  provinces: {len(raw.index)}")

    # 逐省统计
    stats_rows = []
    for prov in raw.index:
        arr = raw.loc[prov].values.astype(float)
        stats_rows.append(per_province_stats(arr, prov))
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(STATS_CSV, index=False)
    print(f"  统计 CSV: {STATS_CSV}")

    # 转移矩阵单独保存
    trans_rows = []
    for _, r in stats_df.iterrows():
        for i in range(3):
            for j in range(3):
                trans_rows.append({
                    "province": r["province"],
                    "from_state": i, "to_state": j,
                    "p": r[f"p_{i}{j}"],
                })
    pd.DataFrame(trans_rows).to_csv(TRANS_CSV, index=False)
    print(f"  转移矩阵 CSV: {TRANS_CSV}")

    # PDF 报告
    make_pdf_report(stats_df, raw, PDF_PATH)
    print(f"  PDF 报告: {PDF_PATH}  ({PDF_PATH.stat().st_size / 1024:.1f} KB)")

    # 控制台关键摘要
    print("\n=== 关键摘要（按 mean_run_length 排序 Top 10） ===")
    cols = ["province", "total_kwh", "n_days_state2",
            "mean_run_length_state2", "max_run_length_state2", "p_22"]
    rep = stats_df[cols].sort_values("mean_run_length_state2",
                                       ascending=False).head(10)
    rep["total_kwh"] = (rep["total_kwh"] / 1e9).round(2)
    rep["mean_run_length_state2"] = rep["mean_run_length_state2"].round(2)
    rep["p_22"] = rep["p_22"].round(3)
    rep.columns = ["省", "总量(GWh)", "极端日数", "MeanRun(d)",
                   "MaxRun(d)", "P(2→2)"]
    print(rep.to_string(index=False))

    print("\n=== 全国汇总 ===")
    print(f"  26 省年总调峰量: {stats_df['total_kwh'].sum() / 1e9:.1f} GWh")
    print(f"  26 省平均极端日数: {stats_df['n_days_state2'].mean():.1f} 天/省")
    print(f"  26 省 mean_run_length 中位: "
          f"{stats_df['mean_run_length_state2'].median():.2f} 天")
    print(f"  26 省 P(2→2) 中位: {stats_df['p_22'].median():.3f}")
    print(f"  最长连峰: {stats_df['max_run_length_state2'].max()} 天 "
          f"({stats_df.loc[stats_df['max_run_length_state2'].idxmax(), 'province']})")


if __name__ == "__main__":
    main()
