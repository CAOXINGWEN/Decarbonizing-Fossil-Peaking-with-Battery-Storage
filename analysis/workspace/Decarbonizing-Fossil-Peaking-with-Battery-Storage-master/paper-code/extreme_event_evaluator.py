"""
M4-extreme: 极端事件区间内重算储能经济性。

物理调度（M3）不变——储能仍在全年充放电，但**经济性评估口径**切换为：
  - 供给视角：只把发生在极端日（state=2，demand>Q75）的放电算作"有价值供给"
  - 燃气替代视角：只对极端日的需求做替代深度核算

核心动机：长时储能的真正价值在于"为多日连峰储备能量"，而年度均摊会被
高弃电省份的低需求日稀释。极端日切片让聚集度结构进入经济评估。

输入：
  m3_dispatch_results.csv（保留全年调度轨迹）
  daily extreme-day mask（每省 365×bool，由 q75 分类得到）

输出（不覆盖原 M4）：
  output/m4_economics_extreme/extreme_evaluation_<scenario>.csv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from config import CONFIG
from event_clustering import load_demand_matrix
from ldes_params import calculate_crf, generate_duration_sweep, load_tech_library
from ldes_simulator import (
    load_curtailment_for_province,
    load_charge_window_for_province,
    simulate_steady_state,
    simulate_ldes_year,
)


PAPER_CODE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PAPER_CODE_DIR / "output" / "m4_economics_extreme"


# ───────────────────────────────────────────────────────────────────
# 极端日 mask 构建
# ───────────────────────────────────────────────────────────────────

def build_extreme_mask(demand_arr: np.ndarray, quantile: float = 0.75) -> np.ndarray:
    """对 365 天日需求按 q75 分类，返回 bool mask（state=2 即 True）。"""
    nonzero = demand_arr[demand_arr > 0]
    if len(nonzero) == 0:
        return np.zeros_like(demand_arr, dtype=bool)
    q = np.percentile(nonzero, quantile * 100)
    return demand_arr > q


# ───────────────────────────────────────────────────────────────────
# 极端日定向调度核心：非极端日禁止放电
# ───────────────────────────────────────────────────────────────────

@njit(cache=True, nogil=True)
def simulate_ldes_year_extreme_only(
    demand_arr, curtailment_arr,
    charge_window_h_arr, discharge_window_h_arr,
    extreme_mask,
    capacity_kwh, charge_power_kw, discharge_power_kw,
    charge_eff, discharge_eff,
    soc_min_frac, soc_max_frac,
    self_discharge_rate, initial_soc_frac,
):
    """与 simulate_ldes_year 行为一致，但只在 extreme_mask=True 的日子放电。

    充电仍然全年进行——储能在非极端日积累能量，等待连峰到来。
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

        # 放电：只有极端日才允许
        supply = 0.0
        if extreme_mask[i]:
            available_energy = (soc - e_min) * discharge_eff
            if available_energy < 0.0:
                available_energy = 0.0
            max_discharge = min(available_energy, daily_discharge_limit)
            supply = min(demand, max_discharge)
            if supply > 0.0:
                soc -= supply / discharge_eff
        total_supply += supply
        daily_supply[i] = supply

        # 充电（与原核心一致，全年都允许）
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


def simulate_steady_state_extreme(
    demand_arr, curtailment_arr,
    charge_window_h_arr, discharge_window_h_arr,
    extreme_mask, capacity_kwh,
    charge_power_kw, discharge_power_kw, tech_params,
    max_iterations: int = 20, convergence_ratio: float = 0.001,
):
    """极端日定向调度的稳态收敛。"""
    initial_soc = CONFIG["storage"]["soc_min"]
    threshold = convergence_ratio * capacity_kwh

    for _ in range(max_iterations):
        (total_supply, total_charge, total_spillage,
         final_soc, daily_supply, daily_spillage) = simulate_ldes_year_extreme_only(
            demand_arr, curtailment_arr,
            charge_window_h_arr, discharge_window_h_arr,
            extreme_mask,
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
# 重新调度并计算极端日切片指标
# ───────────────────────────────────────────────────────────────────

def evaluate_one_combo(tech_params, demand_arr, curtailment_arr,
                       charge_window_arr, discharge_window_arr,
                       capacity_kwh, extreme_mask):
    """单个 (技术, 容量) 组合：跑两套调度（全年 vs 极端日定向）。"""
    discharge_power_kw = capacity_kwh / tech_params.duration_h
    charge_power_kw = discharge_power_kw * tech_params.charge_power_ratio

    # 全年调度（与 M3 一致）
    (yr_supply, yr_charge, yr_spillage,
     yr_daily_supply, _) = simulate_steady_state(
        demand_arr, curtailment_arr,
        charge_window_arr, discharge_window_arr,
        capacity_kwh, charge_power_kw, discharge_power_kw, tech_params,
    )
    yr_extreme_supply = float(yr_daily_supply[extreme_mask].sum())

    # 极端日定向调度
    (ext_supply, ext_charge, ext_spillage,
     ext_daily_supply, _) = simulate_steady_state_extreme(
        demand_arr, curtailment_arr,
        charge_window_arr, discharge_window_arr,
        extreme_mask, capacity_kwh,
        charge_power_kw, discharge_power_kw, tech_params,
    )

    extreme_demand = float(demand_arr[extreme_mask].sum())

    return {
        "capacity_kwh": capacity_kwh,
        "discharge_power_kw": discharge_power_kw,
        "charge_power_kw": charge_power_kw,
        "total_demand_kwh": float(demand_arr.sum()),
        "extreme_demand_kwh": extreme_demand,
        "n_extreme_days": int(extreme_mask.sum()),
        # 全年调度结果
        "year_total_supply_kwh": float(yr_supply),
        "year_extreme_supply_kwh": yr_extreme_supply,
        "year_total_charge_kwh": float(yr_charge),
        "year_satisfy_total_pct": (yr_supply / float(demand_arr.sum()) * 100.0
                                   if demand_arr.sum() > 0 else 0.0),
        "year_satisfy_extreme_pct": (yr_extreme_supply / extreme_demand * 100.0
                                     if extreme_demand > 0 else 0.0),
        # 极端日定向调度结果
        "ext_total_supply_kwh": float(ext_supply),
        "ext_total_charge_kwh": float(ext_charge),
        "ext_satisfy_extreme_pct": (ext_supply / extreme_demand * 100.0
                                    if extreme_demand > 0 else 0.0),
    }


# ───────────────────────────────────────────────────────────────────
# 经济性指标计算（极端日口径）
# ───────────────────────────────────────────────────────────────────

def compute_extreme_economics(combo_result, tech_params, tech_row,
                              gas_unit_cost_cny_kwh, discount_rate=0.036):
    """根据物理结果 + 成本参数，计算两套口径的单位成本。

    返回三套单位成本：
      - unit_cost_year: 全年调度 / 全年总供给
      - unit_cost_extreme_yr: 全年调度 / 仅极端日供给（口径切片）
      - unit_cost_extreme_dispatch: 极端日定向调度 / 极端日供给（最优蓄水池模式）
    """
    capacity_kwh = combo_result["capacity_kwh"]
    discharge_power_kw = combo_result["discharge_power_kw"]
    rated_power_kw = max(combo_result["charge_power_kw"], discharge_power_kw)

    power_cost = float(tech_row["power_cost_cny_kw_base"])
    energy_cost = float(tech_row["energy_cost_cny_kwh_base"])
    fixed_om = float(tech_row["fixed_om_cny_kw_yr"])
    variable_om = float(tech_row["variable_om_cny_mwh"])
    project_life = int(round(float(tech_row["calendar_life_years"])))

    capex_power = power_cost * rated_power_kw
    capex_energy = energy_cost * capacity_kwh
    initial_capex = capex_power + capex_energy
    crf = calculate_crf(discount_rate, project_life)
    annualized_capex = initial_capex * crf
    annual_fixed_om = fixed_om * rated_power_kw

    # 全年调度的成本
    yr_supply = combo_result["year_total_supply_kwh"]
    yr_extreme_supply = combo_result["year_extreme_supply_kwh"]
    annual_var_om_yr = variable_om * (yr_supply / 1000.0)
    annual_cost_yr = annualized_capex + annual_fixed_om + annual_var_om_yr

    # 极端日定向调度的成本
    ext_supply = combo_result["ext_total_supply_kwh"]
    annual_var_om_ext = variable_om * (ext_supply / 1000.0)
    annual_cost_ext = annualized_capex + annual_fixed_om + annual_var_om_ext

    def _safe_div(a, b):
        return a / b if (b is not None and b > 0 and np.isfinite(b)) else np.nan

    unit_cost_year = _safe_div(annual_cost_yr, yr_supply)
    unit_cost_extreme_yr = _safe_div(annual_cost_yr, yr_extreme_supply)
    unit_cost_extreme_dispatch = _safe_div(annual_cost_ext, ext_supply)

    return {
        "initial_capex_cny": initial_capex,
        "annualized_capex_cny": annualized_capex,
        "annual_fixed_om_cny": annual_fixed_om,
        "annual_variable_om_year_cny": annual_var_om_yr,
        "annual_variable_om_ext_cny": annual_var_om_ext,
        "annual_cost_year_cny": annual_cost_yr,
        "annual_cost_ext_cny": annual_cost_ext,
        "gas_unit_cost_cny_kwh": gas_unit_cost_cny_kwh,
        "unit_cost_year_cny_kwh": unit_cost_year,
        "unit_cost_extreme_yr_cny_kwh": unit_cost_extreme_yr,
        "unit_cost_extreme_dispatch_cny_kwh": unit_cost_extreme_dispatch,
        "cost_ratio_year_to_gas": _safe_div(unit_cost_year, gas_unit_cost_cny_kwh),
        "cost_ratio_extreme_yr_to_gas": _safe_div(unit_cost_extreme_yr, gas_unit_cost_cny_kwh),
        "cost_ratio_extreme_dispatch_to_gas": _safe_div(unit_cost_extreme_dispatch, gas_unit_cost_cny_kwh),
    }


# ───────────────────────────────────────────────────────────────────
# 燃气价格加载
# ───────────────────────────────────────────────────────────────────

GAS_POWER_PER_M3 = 9.78  # kWh/m³


def load_provincial_gas_costs() -> dict[str, float]:
    """读取省级燃气价格 → CNY/kWh 单位。复用 M4 的 0.43 加成。"""
    gas_path = PAPER_CODE_DIR.parent.parent / "dataset" / "主要能源价格.xlsx"
    df = pd.read_excel(gas_path, sheet_name=0)
    gas_col = [c for c in df.columns if "天然气" in str(c) or "gas" in str(c).lower()]
    prov_col = df.columns[0]
    if gas_col:
        df = df[[prov_col, gas_col[0]]].dropna()
        df.columns = ["province", "gas_price_cny_m3"]
    else:
        df = df.iloc[:, :2].dropna()
        df.columns = ["province", "gas_price_cny_m3"]

    cn_to_en_path = PAPER_CODE_DIR / "event_clustering.py"
    from event_clustering import EN_TO_CN
    cn_to_en = {v: k for k, v in EN_TO_CN.items()}

    gen_cost = float(CONFIG["gas"]["generator_cost"])
    out = {}
    for _, row in df.iterrows():
        prov_cn = str(row["province"]).strip()
        en = cn_to_en.get(prov_cn)
        if en is None:
            for k, v in EN_TO_CN.items():
                if prov_cn in v or v in prov_cn:
                    en = k
                    break
        if en is None:
            continue
        price = float(row["gas_price_cny_m3"])
        out[en] = price / GAS_POWER_PER_M3 + gen_cost
    return out


# ───────────────────────────────────────────────────────────────────
# 主流程：每个 (技术, 时长, 省份) 取若干代表容量
# ───────────────────────────────────────────────────────────────────

def run_extreme_evaluation(scenario_name: str = "baseline",
                           target_satisfy_rates_pct=(20, 50, 80, 95),
                           output_dir: Path = OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)

    print("加载 M3 全年调度结果...")
    m3_path = PAPER_CODE_DIR / "output" / "m3_dispatch" / "m3_dispatch_results.csv"
    m3 = pd.read_csv(m3_path, usecols=[
        "tech_id", "duration_h", "province_en", "province_cn",
        "capacity_mwh", "satisfy_rate_pct", "scenario"
    ])
    m3 = m3[m3["scenario"].eq(scenario_name)].copy()

    data_path = PAPER_CODE_DIR.parent / "data" / \
        "Daily Natural Gas Peaking Power Generation by Province in China.xlsx"
    demand_df = load_demand_matrix(str(data_path))

    tech_library = load_tech_library()
    params_list = generate_duration_sweep(scenario_name=scenario_name)
    gas_costs = load_provincial_gas_costs()
    print(f"已加载 {len(gas_costs)} 省燃气成本")

    # 预计算每省的极端日 mask（与技术无关）
    extreme_masks = {}
    for prov in demand_df.columns:
        demand_arr = demand_df[prov].values.astype(np.float64)
        extreme_masks[prov] = build_extreme_mask(demand_arr, quantile=0.75)

    rows = []
    n_combos = len(params_list)

    for i, tp in enumerate(params_list):
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i+1}/{n_combos}] {tp.tech_id} {tp.duration_h:.0f}h ...")
        tech_row = tech_library.loc[tp.tech_id]
        for prov in demand_df.columns:
            if prov not in gas_costs:
                continue

            curt = load_curtailment_for_province(prov)
            cw = load_charge_window_for_province(prov)
            if curt is None or cw is None:
                continue

            demand_arr = demand_df[prov].values.astype(np.float64)
            dw_arr = np.full(demand_arr.shape[0],
                             float(CONFIG["dispatch"]["baseline_discharge_window_h"]))
            extreme_mask = extreme_masks[prov]

            sub_m3 = m3[(m3["tech_id"].eq(tp.tech_id)) &
                        (m3["duration_h"].eq(tp.duration_h)) &
                        (m3["province_en"].eq(prov))]
            if sub_m3.empty:
                continue

            for tgt in target_satisfy_rates_pct:
                idx = (sub_m3["satisfy_rate_pct"] - tgt).abs().idxmin()
                row_m3 = sub_m3.loc[idx]
                if abs(row_m3["satisfy_rate_pct"] - tgt) > 5.0:
                    continue
                cap_kwh = float(row_m3["capacity_mwh"]) * 1000.0

                phys = evaluate_one_combo(
                    tp, demand_arr, curt, cw, dw_arr, cap_kwh, extreme_mask,
                )
                econ = compute_extreme_economics(
                    phys, tp, tech_row, gas_costs[prov],
                )

                row = {
                    "scenario": scenario_name,
                    "tech_id": tp.tech_id,
                    "technology_cn": tp.technology_cn,
                    "duration_h": tp.duration_h,
                    "charge_power_ratio": tp.charge_power_ratio,
                    "rte": tp.rte,
                    "province_en": prov,
                    "province_cn": row_m3.get("province_cn", prov),
                    "target_satisfy_rate_pct": tgt,
                    "capacity_mwh": cap_kwh / 1000.0,
                }
                row.update(phys)
                row.update(econ)
                rows.append(row)

    df = pd.DataFrame(rows)
    out_path = output_dir / f"extreme_evaluation_{scenario_name}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n=== M4-extreme 评估完成 ===")
    print(f"输出: {out_path}")
    print(f"行数: {len(df)}")

    if not df.empty:
        print("\n=== 内蒙古 50% 满足率下 三套口径单位成本对比 ===")
        im_50 = df[(df["province_en"].eq("Inner Mongolia")) &
                   (df["target_satisfy_rate_pct"].eq(50))]
        if not im_50.empty:
            best = im_50.loc[im_50.groupby("tech_id")["unit_cost_extreme_dispatch_cny_kwh"].idxmin()]
            print(best[["tech_id", "duration_h",
                        "year_satisfy_extreme_pct", "ext_satisfy_extreme_pct",
                        "unit_cost_year_cny_kwh",
                        "unit_cost_extreme_yr_cny_kwh",
                        "unit_cost_extreme_dispatch_cny_kwh",
                        "cost_ratio_extreme_dispatch_to_gas"]].round(3).to_string(index=False))

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="baseline",
                        choices=["baseline", "low_cost", "high_cost"])
    args = parser.parse_args()
    run_extreme_evaluation(scenario_name=args.scenario)
