"""
M4 实验 A — 复制年 demand 生成器（无 markov、无极端注入）

5 个 demand：
  Yr1_repeat: 真实 365 天（与 fig4 panel a demand 一致）
  Yr2_repeat: 真实复制 2 次 = 730 天
  Yr3_repeat: 真实复制 3 次 = 1095 天
  Yr4_repeat: 真实复制 4 次 = 1460 天
  Yr5_repeat: 真实复制 5 次 = 1825 天

输出：
  output/m3_dispatch/demand_yr<N>_repeat.csv
  output/m3_dispatch/curtailment_yr<N>_repeat.csv (复制 N 次原始弃电)
  output/m3_dispatch/charge_window_yr<N>_repeat.csv

用法：python build_yr_repeat_demand.py
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd

from event_clustering import EN_TO_CN
from build_2yr_demand import load_real_demand_long_format

PAPER_CODE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PAPER_CODE_DIR / "output" / "m3_dispatch"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) 真实 demand 365 天 26 省（中文列）
    real_df = load_real_demand_long_format()
    print(f"真实 demand: {real_df.shape}")

    # 转列名 中文 -> 英文
    cn_to_en = {v: k for k, v in EN_TO_CN.items()}
    real_en = real_df.copy()
    real_en.columns = [cn_to_en.get(c, c) for c in real_en.columns]

    # 2) 真实弃电（已英文列）
    curt_real = pd.read_csv(
        OUTPUT_DIR / "daily_provincial_curtailment.csv", index_col=0)
    cw_real = pd.read_csv(
        OUTPUT_DIR / "daily_provincial_charge_window_h.csv", index_col=0)
    print(f"真实 curtailment: {curt_real.shape}, "
          f"charge_window: {cw_real.shape}")

    summary = []
    for n_years in [1, 2, 3, 4, 5]:
        # demand：复制 n 次
        demand = pd.concat([real_en] * n_years, axis=0, ignore_index=True)
        demand.index.name = "day_idx"
        out_demand = OUTPUT_DIR / f"demand_yr{n_years}_repeat.csv"
        demand.to_csv(out_demand)

        # curtailment：复制 n 次（不加扰动）
        curt = pd.concat([curt_real] * n_years, axis=0, ignore_index=True)
        curt.index.name = "day_idx"
        out_curt = OUTPUT_DIR / f"curtailment_yr{n_years}_repeat.csv"
        curt.to_csv(out_curt)

        # charge_window：复制 n 次
        cw = pd.concat([cw_real] * n_years, axis=0, ignore_index=True)
        cw.index.name = "day_idx"
        out_cw = OUTPUT_DIR / f"charge_window_yr{n_years}_repeat.csv"
        cw.to_csv(out_cw)

        total = demand.values.sum() / 1e9  # kWh -> TWh
        peak = demand.values.max() / 1e6   # kWh -> GWh
        print(f"  Yr{n_years}_repeat: demand {demand.shape}, "
              f"total={total:.2f} TWh, peak={peak:.1f} GWh")
        summary.append({
            "node": f"Yr{n_years}_repeat",
            "n_years": n_years,
            "n_days": demand.shape[0],
            "total_demand_TWh": total,
            "max_single_day_GWh": peak,
            "demand_file": out_demand.name,
        })

    sdf = pd.DataFrame(summary)
    sdf.to_csv(OUTPUT_DIR / "yr_repeat_demand_summary.csv", index=False)
    print(f"\n=== 摘要 ===")
    print(sdf.to_string(index=False))


if __name__ == "__main__":
    main()
