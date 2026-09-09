"""
M4 progression — 生成 Y0/Y1/Y2/Y3/Y4 多年限 demand 数据
（Y5 已存在，本脚本只补 0/1/2/3/4 年版）

所有节点共享 Y1 物理参数（demand×1.0、extreme 3 ev × 5d × boost 1.3、
event_intensity 1.5）。**唯一变量是仿真年限**：
  Y0_real:  365 天纯真实 demand（无 markov、无极端注入）
  Y1_yr1:   365 天 markov 仿真 + 极端注入（无真实年）
  Y2_yr2:   1 年真实 + 1 年仿真（共 730 天）
  Y3_yr3:   1 年真实 + 2 年仿真（共 1095 天）
  Y4_yr4:   1 年真实 + 3 年仿真（共 1460 天）
  Y5_yr5（已有）: 1 年真实 + 4 年仿真（共 1825 天）

输出文件命名（避免与现有 Y1/Y3/Y5/Y10 冲突）：
  output/m3_dispatch/demand_yr<N>_real_only.csv         (Y0)
  output/m3_dispatch/demand_yr<N>_markov.csv            (Y1, 纯仿真)
  output/m3_dispatch/demand_yr<N>_real_plus_markov.csv  (Y2/Y3/Y4)
  output/m3_dispatch/curtailment_yr<N>.csv  (随 demand 同年限复制)
  output/m3_dispatch/charge_window_yr<N>.csv

用法：
  python build_yr_sweep_demand.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from config import CONFIG
from event_clustering import EN_TO_CN
from monte_carlo_simulation import fit_season_distributions
from build_2yr_demand import load_real_demand_long_format
import build_progression_demand as bpd


PAPER_CODE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PAPER_CODE_DIR / "output" / "m3_dispatch"

# Y1 共享配置（所有 yr_sweep 节点都用这套）
Y1_PARAMS = {
    "demand_multiplier": 1.0,
    "extreme_boost": 1.3,
    "extreme_events": 3,
    "event_length_days": 5,
    "event_intensity": 1.5,
    "intensity_multiplier": 1.05,
}


def build_yr_demand(n_years: int, use_real_year: bool,
                     real_df: pd.DataFrame, season_params: dict,
                     rng: np.random.Generator) -> pd.DataFrame:
    """以 Y1 物理参数 + 给定年限/真实年 flag 生成 demand。"""
    params = {**Y1_PARAMS,
                "use_real_year": use_real_year,
                "n_years": n_years,
                "label": f"yr={n_years}, real={use_real_year}"}
    return bpd.build_one_node_demand("yr_sweep", params,
                                       real_df, season_params, rng)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    real_df = load_real_demand_long_format()
    print(f"真实 demand: {real_df.shape}")
    season_params = fit_season_distributions(real_df, CONFIG)
    print(f"  季节模型完成: {len(season_params)} 省")
    rng = np.random.default_rng(42)

    # 节点定义：(label, n_years, use_real_year, demand 文件名)
    nodes = [
        ("Y0_real",   1, True,  "yr0_real_only"),       # 365 天纯真实
        ("Y1_yr1",    1, False, "yr1_markov"),          # 1 年 markov
        ("Y2_yr2",    2, True,  "yr2_real_plus_1mk"),   # 1 真 + 1 仿
        ("Y3_yr3",    3, True,  "yr3_real_plus_2mk"),   # 1 真 + 2 仿
        ("Y4_yr4",    4, True,  "yr4_real_plus_3mk"),   # 1 真 + 3 仿
    ]

    summary = []
    cn_to_en = {v: k for k, v in EN_TO_CN.items()}
    for node_key, n_years, use_real, fname in nodes:
        print(f"\n=== 生成 {node_key} (n_years={n_years}, "
              f"use_real_year={use_real}) ===")
        demand_df = build_yr_demand(n_years, use_real, real_df,
                                       season_params, rng)
        # 转换列名 中文 -> 英文（与 m3_dispatch 主流文件一致）
        demand_df.columns = [cn_to_en.get(c, c) for c in demand_df.columns]
        out_path = OUTPUT_DIR / f"demand_{fname}.csv"
        demand_df.to_csv(out_path)
        print(f"  saved demand: {out_path.name}, shape={demand_df.shape}, "
              f"total={demand_df.values.sum()/1e9:.2f} TWh")

        # 同步复制 curtailment 与 charge_window
        curt_df = bpd.replicate_curtailment_n_years(
            n_years, 0.05, rng)
        cw_df = bpd.replicate_charge_window_n_years(n_years)
        curt_df.to_csv(OUTPUT_DIR / f"curtailment_{fname}.csv")
        cw_df.to_csv(OUTPUT_DIR / f"charge_window_{fname}.csv")

        summary.append({
            "node": node_key,
            "n_years": n_years,
            "use_real_year": use_real,
            "demand_file": f"demand_{fname}.csv",
            "n_days": demand_df.shape[0],
            "total_demand_TWh": demand_df.values.sum() / 1e9,
            "max_single_day_GWh": demand_df.values.max() / 1e6,
        })

    sdf = pd.DataFrame(summary)
    sdf.to_csv(OUTPUT_DIR / "yr_sweep_demand_summary.csv", index=False)
    print(f"\n=== 摘要 ===")
    print(sdf.to_string(index=False))


if __name__ == "__main__":
    main()
