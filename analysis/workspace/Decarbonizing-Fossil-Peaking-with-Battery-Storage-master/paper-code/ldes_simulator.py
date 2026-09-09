"""
M3 多技术多时长储能调度仿真。

泛化第一篇 4h 锂电仿真框架，新增：
  - 额定时长可变（4h→168h）
  - 日自放电损耗
  - 功率/容量双约束（充放电速率受功率限制）
  - 多技术循环调用（8 类 LDES 技术）

弃电数据：复用第一篇的真实站级弃电（30日弃电），
经 prepare_curtailment.py 预处理为省级日弃电 CSV。

输出：各技术×各省×各容量水平的年运行指标
"""

import argparse
import os
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd
from numba import njit

from config import CONFIG, MC_CONFIG
from ldes_params import (
    TechParams, generate_duration_sweep,
    PRIMARY_TECHS, calculate_crf, load_tech_library,
)
from event_clustering import load_demand_matrix, EN_TO_CN
from storage_simulator import check_should_stop_numba

# ───────────────────────────────────────────────────────────────────
# Numba 加速的 LDES 日调度仿真核心
# ───────────────────────────────────────────────────────────────────

@njit(cache=True, nogil=True)
def simulate_ldes_year(
    demand_arr,
    curtailment_arr,
    charge_window_h_arr,
    discharge_window_h_arr,
    capacity_kwh,
    charge_power_kw,
    discharge_power_kw,
    charge_eff,
    discharge_eff,
    soc_min_frac,
    soc_max_frac,
    self_discharge_rate,
    initial_soc_frac,
):
    """LDES 单年日步长调度仿真（Numba 加速）。

    Parameters
    ----------
    demand_arr : ndarray[float64]  每日调峰需求 (kWh)
    curtailment_arr : ndarray[float64]  每日可用弃电 (kWh)
    charge_window_h_arr : ndarray[float64]  每日有效充电窗口 (h)
    discharge_window_h_arr : ndarray[float64]  每日有效放电窗口 (h)
    capacity_kwh : float  额定能量容量 (kWh)
    charge_power_kw : float  充电额定功率 (kW)
    discharge_power_kw : float  放电额定功率 (kW)
    charge_eff : float  充电效率 (0-1)
    discharge_eff : float  放电效率 (0-1)
    soc_min_frac : float  最低 SOC 比例
    soc_max_frac : float  最高 SOC 比例
    self_discharge_rate : float  日自放电率 (0-1)
    initial_soc_frac : float  初始 SOC 比例

    Returns
    -------
    total_supply : float  年总调峰供给 (kWh)
    total_charge : float  年总充电量 (kWh)
    total_spillage : float  年弃电未利用量 (kWh)
    final_soc_frac : float  年末 SOC 比例
    daily_supply : ndarray  每日供给 (kWh)
    daily_spillage : ndarray  每日弃电未利用 (kWh)
    """
    n_days = demand_arr.shape[0]
    daily_supply = np.empty(n_days, dtype=np.float64)
    daily_spillage = np.empty(n_days, dtype=np.float64)

    soc = initial_soc_frac * capacity_kwh
    e_min = soc_min_frac * capacity_kwh
    e_max = soc_max_frac * capacity_kwh

    total_supply = 0.0
    total_charge = 0.0
    total_spillage = 0.0

    for i in range(n_days):
        # --- 自放电 ---
        if self_discharge_rate > 0.0:
            soc *= (1.0 - self_discharge_rate)
            if soc < e_min:
                soc = e_min

        demand = demand_arr[i]
        curtailment = curtailment_arr[i]
        charge_window_h = charge_window_h_arr[i]
        discharge_window_h = discharge_window_h_arr[i]
        daily_discharge_limit = discharge_power_kw * discharge_window_h
        daily_charge_limit_into_soc = charge_power_kw * charge_window_h * charge_eff

        # --- 放电满足需求 ---
        available_energy = (soc - e_min) * discharge_eff
        if available_energy < 0.0:
            available_energy = 0.0
        max_discharge = min(available_energy, daily_discharge_limit)
        supply = min(demand, max_discharge)
        if supply > 0.0:
            soc -= supply / discharge_eff
        total_supply += supply
        daily_supply[i] = supply

        # --- 充电从弃电 ---
        available_space = e_max - soc
        if available_space < 0.0:
            available_space = 0.0
        max_charge_power = min(curtailment * charge_eff, daily_charge_limit_into_soc)
        charge_energy = min(max_charge_power, available_space)
        if charge_eff > 0.0:
            actual_curtailment_used = charge_energy / charge_eff
        else:
            actual_curtailment_used = 0.0
        spillage = curtailment - actual_curtailment_used
        if spillage < 0.0:
            spillage = 0.0

        soc += charge_energy
        total_charge += actual_curtailment_used
        total_spillage += spillage
        daily_spillage[i] = spillage

    final_soc_frac = soc / capacity_kwh if capacity_kwh > 0.0 else 0.0
    return total_supply, total_charge, total_spillage, final_soc_frac, daily_supply, daily_spillage


# ───────────────────────────────────────────────────────────────────
# 稳态收敛仿真
# ───────────────────────────────────────────────────────────────────

def simulate_steady_state(demand_arr, curtailment_arr,
                          charge_window_h_arr, discharge_window_h_arr,
                          capacity_kwh, charge_power_kw, discharge_power_kw,
                          tech_params,
                          max_iterations=20, convergence_ratio=0.001):
    """多次迭代直到 SOC 收敛。"""
    initial_soc = CONFIG["storage"]["soc_min"]
    threshold = convergence_ratio * capacity_kwh

    for _ in range(max_iterations):
        (total_supply, total_charge, total_spillage,
         final_soc, daily_supply, daily_spillage) = simulate_ldes_year(
            demand_arr, curtailment_arr,
            charge_window_h_arr, discharge_window_h_arr,
            capacity_kwh, charge_power_kw, discharge_power_kw,
            tech_params.charge_eff, tech_params.discharge_eff,
            CONFIG["storage"]["soc_min"], CONFIG["storage"]["soc_max"],
            tech_params.self_discharge_rate_day,
            initial_soc,
        )
        if abs(final_soc - initial_soc) * capacity_kwh < threshold:
            break
        initial_soc = final_soc

    return total_supply, total_charge, total_spillage, daily_supply, daily_spillage


# ───────────────────────────────────────────────────────────────────
# 容量网格搜索
# ───────────────────────────────────────────────────────────────────

def capacity_search(demand_arr, curtailment_arr, tech_params,
                    charge_window_h_arr, discharge_window_h_arr,
                    step_mwh=25.0, max_capacity_mwh=None):
    """对单个技术-省份组合做容量网格搜索。

    功率随容量按额定时长计算：discharge_power = capacity / duration_h，
    charge_power = discharge_power × charge_power_ratio。

    Parameters
    ----------
    demand_arr : ndarray  日调峰需求 (kWh)
    curtailment_arr : ndarray  日弃电 (kWh)
    tech_params : TechParams  技术参数
    step_mwh : float  容量搜索步长 (MWh)
    max_capacity_mwh : float  最大搜索容量 (MWh)

    Returns
    -------
    list[dict]  每个容量水平的运行指标
    """
    step_kwh = step_mwh * 1000.0
    total_demand = float(np.sum(demand_arr))

    if max_capacity_mwh is None:
        max_capacity_mwh = total_demand / 1000.0 * 10.0

    results = []
    capacity_kwh = step_kwh

    while capacity_kwh / 1000.0 <= max_capacity_mwh:
        discharge_power_kw = capacity_kwh / tech_params.duration_h
        charge_power_kw = discharge_power_kw * tech_params.charge_power_ratio

        (total_supply, total_charge, total_spillage,
         daily_supply, daily_spillage) = simulate_steady_state(
            demand_arr, curtailment_arr,
            charge_window_h_arr, discharge_window_h_arr,
            capacity_kwh, charge_power_kw, discharge_power_kw, tech_params,
        )

        satisfy_rate = total_supply / total_demand * 100.0 if total_demand > 0 else 0.0
        # 与第一篇 yearly_analysis.py 口径一致：按放电侧吞吐量折算 EFC
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

        # 采用第一篇文章同口径的“弃电前缀和”停机条件：
        # 若任何仍未满足的日需求之前都没有可转移的弃电，则继续加容量也无效。
        if check_should_stop_numba(demand_arr, daily_supply, daily_spillage):
            break

        # 安全上限，和第一篇的实现保持一致
        if capacity_kwh > total_demand * 10.0:
            break

        capacity_kwh += step_kwh

    return results


# ───────────────────────────────────────────────────────────────────
# 省级弃电数据加载
# ───────────────────────────────────────────────────────────────────

_CURTAILMENT_DF = None
_CHARGE_WINDOW_DF = None

def _load_curtailment_df():
    """延迟加载省级日弃电 DataFrame（365 × 26 省，kWh）。"""
    global _CURTAILMENT_DF
    if _CURTAILMENT_DF is not None:
        return _CURTAILMENT_DF

    csv_path = os.path.join(
        os.path.dirname(__file__), "output", "m3_dispatch",
        "daily_provincial_curtailment.csv",
    )
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0)
        _CURTAILMENT_DF = df
        return df
    return None


def _load_charge_window_df():
    """延迟加载省级日有效充电窗口 DataFrame（365 × 26 省，h）。"""
    global _CHARGE_WINDOW_DF
    if _CHARGE_WINDOW_DF is not None:
        return _CHARGE_WINDOW_DF

    csv_path = os.path.join(
        os.path.dirname(__file__), "output", "m3_dispatch",
        "daily_provincial_charge_window_h.csv",
    )
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0)
        _CHARGE_WINDOW_DF = df
        return df
    return None


def load_curtailment_for_province(province_en):
    """加载省级弃电数据（365 天，kWh），优先使用真实弃电。

    真实数据来自 prepare_curtailment.py 生成的
    daily_provincial_curtailment.csv。
    若文件不存在或省份缺失，返回 None（调用方回退到简化假设）。
    """
    df = _load_curtailment_df()
    if df is not None and province_en in df.columns:
        return df[province_en].values.astype(np.float64)
    return None


def load_charge_window_for_province(province_en):
    """加载省级日有效充电窗口（365 天，h）。"""
    df = _load_charge_window_df()
    if df is not None and province_en in df.columns:
        return df[province_en].values.astype(np.float64)
    return None


# ───────────────────────────────────────────────────────────────────
# 主流程：单技术 × 全省扫描
# ───────────────────────────────────────────────────────────────────

def run_tech_province_scan(tech_params, demand_df,
                           step_mwh=25.0, provinces=None,
                           baseline_discharge_window_h: Optional[float] = None):
    """对一个技术参数组合，扫描所有省份的容量搜索。

    Returns
    -------
    list[dict]  所有省份 × 容量水平的结果
    """
    if provinces is None:
        provinces = [c for c in demand_df.columns if c in EN_TO_CN]
    if baseline_discharge_window_h is None:
        baseline_discharge_window_h = float(CONFIG["dispatch"]["baseline_discharge_window_h"])

    all_results = []
    for prov in provinces:
        demand_arr = demand_df[prov].values.astype(np.float64)

        # 弃电数据：加载真实数据，若缺失则报错终止
        curtailment = load_curtailment_for_province(prov)
        if curtailment is None:
            print(f"  WARNING: 省份 {prov} 无真实弃电数据，跳过")
            continue
        curtailment_arr = curtailment
        charge_window_h_arr = load_charge_window_for_province(prov)
        if charge_window_h_arr is None:
            print(f"  WARNING: 省份 {prov} 无有效充电窗口数据，跳过")
            continue
        discharge_window_h_arr = np.full(
            demand_arr.shape[0],
            float(baseline_discharge_window_h),
            dtype=np.float64,
        )

        results = capacity_search(
            demand_arr, curtailment_arr, tech_params,
            charge_window_h_arr, discharge_window_h_arr,
            step_mwh=step_mwh,
        )

        for r in results:
            r["province_en"] = prov
            r["province_cn"] = EN_TO_CN.get(prov, prov)
            r["tech_id"] = tech_params.tech_id
            r["technology_cn"] = tech_params.technology_cn
            r["scenario"] = tech_params.scenario
            r["duration_h"] = tech_params.duration_h
            r["rte"] = tech_params.rte
            r["power_mw"] = r["capacity_mwh"] / tech_params.duration_h
            r["charge_power_ratio"] = tech_params.charge_power_ratio
            r["charge_power_mw"] = r["power_mw"] * tech_params.charge_power_ratio
            r["charge_window_method"] = CONFIG["dispatch"]["charge_window_method"]
            r["baseline_discharge_window_h"] = baseline_discharge_window_h

        all_results.extend(results)

    return all_results


# ───────────────────────────────────────────────────────────────────
# 主入口
# ───────────────────────────────────────────────────────────────────

def main(output_dir=None, discharge_window_h: Optional[float] = None):
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(__file__), "output", "m3_dispatch"
        )
    os.makedirs(output_dir, exist_ok=True)
    if discharge_window_h is None:
        discharge_window_h = float(CONFIG["dispatch"]["baseline_discharge_window_h"])

    # 加载需求数据
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "data",
        "Daily Natural Gas Peaking Power Generation by Province in China.xlsx",
    )
    print("加载日调峰需求数据...")
    demand_df = load_demand_matrix(data_path)

    # 生成 4h→168h 时长扫描的 baseline 情景
    print("生成技术参数组合...")
    params_list = generate_duration_sweep(scenario_name="baseline")
    print(f"共 {len(params_list)} 个技术-情景组合")

    all_results = []
    for i, tp in enumerate(params_list):
        t0 = time.time()
        print(f"  [{i+1}/{len(params_list)}] {tp.technology_cn} "
              f"({tp.scenario}, {tp.duration_h}h, RTE={tp.rte:.2f})...")

        results = run_tech_province_scan(
            tp, demand_df,
            step_mwh=CONFIG["storage"]["step"],
            baseline_discharge_window_h=discharge_window_h,
        )
        all_results.extend(results)
        print(f"    {len(results)} 行, 耗时 {time.time()-t0:.1f}s")

    df = pd.DataFrame(all_results)

    # 输出列顺序
    col_order = [
        "tech_id", "technology_cn", "scenario", "duration_h", "rte",
        "charge_power_ratio", "charge_window_method", "baseline_discharge_window_h",
        "province_en", "province_cn", "power_mw", "charge_power_mw",
        "capacity_mwh", "total_demand_kwh", "total_supply_kwh",
        "total_charge_kwh", "total_spillage_kwh",
        "satisfy_rate_pct", "efc",
    ]
    df = df[[c for c in col_order if c in df.columns]]

    if abs(discharge_window_h - float(CONFIG["dispatch"]["baseline_discharge_window_h"])) < 1e-9:
        output_name = "m3_dispatch_results.csv"
    else:
        output_name = f"m3_dispatch_results_dw{int(discharge_window_h)}h.csv"
    out_path = os.path.join(output_dir, output_name)
    df.to_csv(out_path, index=False)
    print(f"\n=== M3 调度仿真完成 ===")
    print(f"输出: {out_path}")
    print(f"总行数: {len(df)}")
    print(f"技术数: {df['tech_id'].nunique()}")
    print(f"省份数: {df['province_en'].nunique()}")

    # 摘要：各技术在各省的最大满足率
    summary = df.groupby(["tech_id", "province_en"])["satisfy_rate_pct"].max()
    summary = summary.unstack("tech_id")
    print("\n各技术各省最大调峰满足率 (%):")
    print(summary.round(1).to_string())

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the M3 LDES dispatch module.")
    parser.add_argument(
        "--discharge-window-h",
        type=float,
        default=float(CONFIG["dispatch"]["baseline_discharge_window_h"]),
        help="Baseline discharge coverage window in hours for the daily-step model.",
    )
    args = parser.parse_args()
    main(discharge_window_h=args.discharge_window_h)
