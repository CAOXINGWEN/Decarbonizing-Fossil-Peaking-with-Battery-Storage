"""
M3 双年仿真：加载 730 天 demand/curtailment/charge_window CSV，跑全部 LDES 技术。

复用 ldes_simulator.simulate_steady_state 内核（接受任意长度 1D 数组），
但绕过它默认的"加载 365 天 Excel"主流程。

输出: output/m3_dispatch/m3_dispatch_results_2yr.csv
"""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from config import CONFIG
from event_clustering import EN_TO_CN
from ldes_params import generate_duration_sweep
from ldes_simulator import (
    simulate_steady_state, capacity_search, run_tech_province_scan,
)
from storage_simulator import check_should_stop_numba

warnings.filterwarnings("ignore")


PAPER_CODE_DIR = Path(__file__).resolve().parent
M3_DIR = PAPER_CODE_DIR / "output" / "m3_dispatch"
DEMAND_2YR = M3_DIR / "demand_2yr_augmented.csv"
CURT_2YR = M3_DIR / "curtailment_2yr_augmented.csv"
CW_2YR = M3_DIR / "charge_window_2yr_augmented.csv"
OUT_CSV = M3_DIR / "m3_dispatch_results_2yr.csv"


def capacity_search_with_arrays(demand_arr, curt_arr, cw_arr, dw_arr,
                                  tech_params, step_mwh):
    """复制 capacity_search 但接受外部 dw_arr。"""
    step_kwh = step_mwh * 1000.0
    total_demand = float(np.sum(demand_arr))
    max_capacity_mwh = total_demand / 1000.0 * 10.0

    results = []
    capacity_kwh = step_kwh
    while capacity_kwh / 1000.0 <= max_capacity_mwh:
        discharge_pwr_kw = capacity_kwh / tech_params.duration_h
        charge_pwr_kw = discharge_pwr_kw * tech_params.charge_power_ratio

        (total_supply, total_charge, total_spillage,
         daily_supply, daily_spillage) = simulate_steady_state(
            demand_arr, curt_arr, cw_arr, dw_arr,
            capacity_kwh, charge_pwr_kw, discharge_pwr_kw, tech_params,
        )

        satisfy_rate = total_supply / total_demand * 100.0 if total_demand > 0 else 0.0
        efc = ((total_supply / tech_params.discharge_eff) / capacity_kwh
               if capacity_kwh > 0 and tech_params.discharge_eff > 0 else 0.0)

        results.append({
            "capacity_mwh": capacity_kwh / 1000.0,
            "capacity_kwh": capacity_kwh,
            "total_demand_kwh": total_demand,
            "total_supply_kwh": total_supply,
            "total_charge_kwh": total_charge,
            "total_spillage_kwh": total_spillage,
            "satisfy_rate_pct": satisfy_rate,
            "efc": efc,
        })

        if satisfy_rate >= 99.99:
            break
        if check_should_stop_numba(demand_arr, daily_supply, daily_spillage):
            break
        if capacity_kwh > total_demand * 10.0:
            break
        capacity_kwh += step_kwh

    return results


def main():
    M3_DIR.mkdir(parents=True, exist_ok=True)

    # 加载 730 天数据
    print(f"加载双年 demand: {DEMAND_2YR}")
    demand_df = pd.read_csv(DEMAND_2YR, index_col=0)
    print(f"  shape: {demand_df.shape}")
    print(f"加载双年 curtailment: {CURT_2YR}")
    curt_df = pd.read_csv(CURT_2YR, index_col=0)
    print(f"  shape: {curt_df.shape}")
    print(f"加载双年 charge_window: {CW_2YR}")
    cw_df = pd.read_csv(CW_2YR, index_col=0)
    print(f"  shape: {cw_df.shape}")

    n_days = demand_df.shape[0]
    baseline_dw = float(CONFIG["dispatch"]["baseline_discharge_window_h"])
    dw_arr = np.full(n_days, baseline_dw, dtype=np.float64)

    # 生成技术参数表
    print("生成技术参数组合...")
    params_list = generate_duration_sweep(scenario_name="baseline")
    print(f"  共 {len(params_list)} 个 (技术, 时长) 组合")

    all_results = []
    step_mwh = float(CONFIG["storage"]["step"])

    for i, tp in enumerate(params_list):
        t0 = time.time()
        print(f"  [{i+1}/{len(params_list)}] {tp.technology_cn} "
              f"({tp.duration_h}h, RTE={tp.rte:.2f})...")
        for prov in demand_df.columns:
            if prov not in EN_TO_CN:
                continue
            if prov not in curt_df.columns or prov not in cw_df.columns:
                continue
            demand_arr = demand_df[prov].values.astype(np.float64)
            curt_arr = curt_df[prov].values.astype(np.float64)
            cw_arr = cw_df[prov].values.astype(np.float64)
            results = capacity_search_with_arrays(
                demand_arr, curt_arr, cw_arr, dw_arr, tp, step_mwh,
            )
            for r in results:
                r["province_en"] = prov
                r["province_cn"] = EN_TO_CN.get(prov, prov)
                r["tech_id"] = tp.tech_id
                r["technology_cn"] = tp.technology_cn
                r["scenario"] = tp.scenario
                r["duration_h"] = tp.duration_h
                r["rte"] = tp.rte
                r["power_mw"] = r["capacity_mwh"] / tp.duration_h
                r["charge_power_ratio"] = tp.charge_power_ratio
                r["charge_power_mw"] = r["power_mw"] * tp.charge_power_ratio
                r["charge_window_method"] = CONFIG["dispatch"]["charge_window_method"]
                r["baseline_discharge_window_h"] = baseline_dw
            all_results.extend(results)
        print(f"    {len(all_results)} cumulative rows  ({time.time()-t0:.1f}s)")

    df = pd.DataFrame(all_results)
    col_order = [
        "tech_id", "technology_cn", "scenario", "duration_h", "rte",
        "charge_power_ratio", "charge_window_method", "baseline_discharge_window_h",
        "province_en", "province_cn", "power_mw", "charge_power_mw",
        "capacity_mwh", "total_demand_kwh", "total_supply_kwh",
        "total_charge_kwh", "total_spillage_kwh",
        "satisfy_rate_pct", "efc",
    ]
    df = df[[c for c in col_order if c in df.columns]]
    df.to_csv(OUT_CSV, index=False)

    print(f"\n=== M3 双年仿真完成 ===")
    print(f"输出: {OUT_CSV}")
    print(f"总行数: {len(df)}")
    summary = df.groupby(["tech_id", "province_en"])["satisfy_rate_pct"].max()
    summary = summary.unstack("tech_id")
    print("\n各技术各省最大调峰满足率 (%):")
    print(summary.round(1).to_string())


if __name__ == "__main__":
    main()
