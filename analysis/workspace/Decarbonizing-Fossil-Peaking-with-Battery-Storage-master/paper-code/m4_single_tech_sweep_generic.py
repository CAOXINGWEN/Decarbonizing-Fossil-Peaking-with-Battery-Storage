"""
M4 single-tech sweep —— 通用 demand 路径版本。

与 m4_progression_single_tech_sweep.py 完全相同的扫描逻辑（25 MWh 全量、
单技术原生 simulate_steady_state、SOC 收敛早停），但 demand/curtailment/
charge_window 文件名通过参数指定。

用法：
  python m4_single_tech_sweep_generic.py \\
      --demand-stem yr1_repeat \\
      --tech caes
  → 读取 demand_yr1_repeat.csv / curtailment_yr1_repeat.csv /
    charge_window_yr1_repeat.csv
  → 输出 m4_single_tech_yr1_repeat_caes.csv

支持参数：
  --demand-stem <stem>:    文件名后缀（去掉 demand_/curtailment_/charge_window_ 前缀和 .csv）
  --tech <tid>:            li_lfp / caes / laes / gravity / fecr_flow /
                            molten_salt_tes / vrfb / h2_cavern
  --all-techs:             一次跑全部 8 tech
  --step-mwh <float>:      容量步长（默认 25 MWh）
  --output-prefix <str>:   输出文件前缀（默认 m4_single_tech_<demand_stem>）
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from event_clustering import EN_TO_CN
from ldes_params import load_tech_library
from ldes_economics import load_gas_prices, load_electricity_prices
from ldes_simulator import simulate_steady_state, check_should_stop_numba
from m4_hybrid_search import (
    get_tech_params, annualized_cost,
    LFP_DURATIONS_H, LDES_DURATIONS_H,
)
from config import CONFIG, GAS_POWER_PER_M3


warnings.filterwarnings("ignore")

PAPER_CODE_DIR = Path(__file__).resolve().parent
M3_DIR = PAPER_CODE_DIR / "output" / "m3_dispatch"
M4_DIR = PAPER_CODE_DIR / "output" / "m4_economics"

DISCHARGE_WINDOWS_H = [8.0, 12.0, 16.0, 24.0]
SCENARIOS = ["baseline"]
CAPACITY_STEP_MWH = 25.0

DISCOUNT_RATE = float(CONFIG["cost"]["discount_rate"])
GEN_COST = float(CONFIG["gas"]["generator_cost"])


def load_inputs(demand_stem: str, dw_value: float):
    """从 demand_<stem>.csv / curtailment_<stem>.csv / charge_window_<stem>.csv
    读取数据。"""
    demand_df = pd.read_csv(
        M3_DIR / f"demand_{demand_stem}.csv", index_col=0)
    curt_df = pd.read_csv(
        M3_DIR / f"curtailment_{demand_stem}.csv", index_col=0)
    cw_df = pd.read_csv(
        M3_DIR / f"charge_window_{demand_stem}.csv", index_col=0)
    n_days = len(demand_df)
    dw_arr = np.full(n_days, float(dw_value), dtype=np.float64)
    return demand_df, curt_df, cw_df, dw_arr, float(dw_value)


def load_prices():
    g = load_gas_prices()
    e = load_electricity_prices()
    return (g.set_index("province_cn")["gas_price_cny_m3"].to_dict(),
            e.set_index("province_cn")["electricity_price_cny_kwh"].to_dict())


def evaluate_single_tech(demand_arr, curt_arr, cw_arr, dw_arr,
                          tp, capacity_mwh, gas_price, elec_price):
    capacity_kwh = capacity_mwh * 1000.0
    discharge_pwr_kw = capacity_kwh / tp.duration_h
    charge_pwr_kw = discharge_pwr_kw * tp.charge_power_ratio

    (total_supply, total_charge, total_spillage,
     daily_supply, daily_spillage) = simulate_steady_state(
        demand_arr, curt_arr, cw_arr, dw_arr,
        capacity_kwh, charge_pwr_kw, discharge_pwr_kw, tp,
    )
    n_days = demand_arr.shape[0]
    year_factor = 365.0 / n_days
    annual_supply = total_supply * year_factor
    total_demand = float(np.sum(demand_arr))
    total_demand_annual = total_demand * year_factor
    satisfy_pct = (annual_supply / total_demand_annual * 100.0
                    if total_demand_annual > 0 else 0.0)
    capex, ann_fixed = annualized_cost(tp, capacity_mwh, DISCOUNT_RATE)
    annual_var_om = tp.variable_om_cny_mwh * annual_supply / 1000.0
    annual_cost = ann_fixed + annual_var_om
    unit_cost = (annual_cost / annual_supply
                  if annual_supply > 0 else np.nan)
    gas_unit_cost = gas_price / GAS_POWER_PER_M3 + GEN_COST
    cost_ratio = (unit_cost / gas_unit_cost
                   if pd.notna(unit_cost) and gas_unit_cost > 0 else np.nan)
    return {
        "satisfy_rate_pct": satisfy_pct,
        "annual_supply_kwh": annual_supply,
        "annual_charge_kwh": total_charge * year_factor,
        "annual_spillage_kwh": total_spillage * year_factor,
        "capex_cny": capex,
        "annual_cost_cny": annual_cost,
        "unit_cost_cny_kwh": unit_cost,
        "gas_unit_cost_cny_kwh": gas_unit_cost,
        "cost_ratio_to_gas": cost_ratio,
    }, daily_supply, daily_spillage


def sweep_capacities(demand_arr, curt_arr, cw_arr, dw_arr,
                      tp, gas_p, elec_p, step_mwh=CAPACITY_STEP_MWH):
    total_demand = float(np.sum(demand_arr))
    peak_daily = float(np.max(demand_arr))
    max_cap_mwh = peak_daily / 1000.0 * 100.0
    rows = []
    cap_mwh = step_mwh
    while cap_mwh <= max_cap_mwh:
        r, daily_sup, daily_spill = evaluate_single_tech(
            demand_arr, curt_arr, cw_arr, dw_arr,
            tp, cap_mwh, gas_p, elec_p,
        )
        r["capacity_mwh"] = cap_mwh
        rows.append(r)
        if r["satisfy_rate_pct"] >= 99.99:
            break
        if check_should_stop_numba(demand_arr, daily_sup, daily_spill):
            break
        cap_mwh += step_mwh
    return rows


def run_one_tech(demand_stem: str, tech_id: str,
                  output_prefix: str = None,
                  step_mwh: float = CAPACITY_STEP_MWH):
    M4_DIR.mkdir(parents=True, exist_ok=True)
    if output_prefix is None:
        output_prefix = f"m4_single_tech_{demand_stem}"
    out_csv = M4_DIR / f"{output_prefix}_{tech_id}.csv"
    if out_csv.exists():
        out_csv.unlink()

    if tech_id == "li_lfp":
        durations = LFP_DURATIONS_H
    elif tech_id in LDES_DURATIONS_H:
        durations = LDES_DURATIONS_H[tech_id]
    else:
        raise SystemExit(f"未知 tech_id: {tech_id}")

    print(f"=== Single-tech 全量扫描 [{demand_stem} | {tech_id}] ===",
          flush=True)
    print(f"  durations: {durations}, step: {step_mwh} MWh", flush=True)
    print(f"  输出: {out_csv}", flush=True)

    gas_map, elec_map = load_prices()
    tech_lib = load_tech_library()

    inputs_cache = {}
    for dw_value in DISCHARGE_WINDOWS_H:
        inputs_cache[dw_value] = load_inputs(demand_stem, dw_value)
    sample_dw = next(iter(inputs_cache))
    print(f"  loaded demand: shape={inputs_cache[sample_dw][0].shape}",
          flush=True)

    rows_buffer = []
    written_header = False
    total_rows = 0
    t_start = time.time()

    for scenario in SCENARIOS:
        tech_tps = {}
        for d in durations:
            tp = get_tech_params(tech_lib, tech_id, scenario, d)
            if tp is not None:
                tech_tps[d] = tp
        if not tech_tps:
            continue

        for dw_value in DISCHARGE_WINDOWS_H:
            demand_df, curt_df, cw_df, dw_arr, dw = inputs_cache[dw_value]
            t_cell = time.time()
            cell_rows = 0

            for prov_en in demand_df.columns:
                if prov_en not in EN_TO_CN:
                    continue
                prov_cn = EN_TO_CN[prov_en]
                if (prov_en not in curt_df.columns
                        or prov_en not in cw_df.columns
                        or prov_cn not in gas_map
                        or prov_cn not in elec_map):
                    continue
                gas_p = float(gas_map[prov_cn])
                elec_p = float(elec_map[prov_cn])
                d_arr = demand_df[prov_en].values.astype(np.float64)
                c_arr = curt_df[prov_en].values.astype(np.float64)
                w_arr = cw_df[prov_en].values.astype(np.float64)

                for dur, tp in tech_tps.items():
                    results = sweep_capacities(
                        d_arr, c_arr, w_arr, dw_arr,
                        tp, gas_p, elec_p, step_mwh,
                    )
                    for r in results:
                        r.update({
                            "demand_stem": demand_stem,
                            "scenario": scenario,
                            "dispatch_window_h": dw,
                            "tech_id": tech_id,
                            "duration_h": dur,
                            "province_en": prov_en,
                            "province_cn": prov_cn,
                        })
                    rows_buffer.extend(results)
                    cell_rows += len(results)

                    # per (prov, dur) flush，控制内存
                    if rows_buffer:
                        df_chunk = pd.DataFrame(rows_buffer)
                        df_chunk.to_csv(out_csv, mode="a",
                                         header=(not written_header),
                                         index=False)
                        written_header = True
                        total_rows += len(rows_buffer)
                        rows_buffer = []

            elapsed = time.time() - t_cell
            print(f"  [{demand_stem}|{tech_id}|dw={dw}h|{scenario}] "
                  f"{cell_rows} rows in {elapsed:.1f}s "
                  f"(累计 {total_rows:,})", flush=True)

    if rows_buffer:
        df_chunk = pd.DataFrame(rows_buffer)
        df_chunk.to_csv(out_csv, mode="a",
                         header=(not written_header), index=False)
        total_rows += len(rows_buffer)

    elapsed_total = time.time() - t_start
    print(f"\n=== [{demand_stem} | {tech_id}] 完成: 共 {total_rows:,} 行, "
          f"耗时 {elapsed_total/60:.1f} min ===", flush=True)
    return out_csv, total_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demand-stem", required=True,
                        help="文件名 stem，如 yr1_repeat 或 freq3_5yr")
    parser.add_argument("--tech", default=None,
                        help="li_lfp / caes / laes / gravity / fecr_flow / "
                              "molten_salt_tes / vrfb / h2_cavern")
    parser.add_argument("--all-techs", action="store_true",
                        help="串行跑所有 8 tech")
    parser.add_argument("--step-mwh", type=float, default=CAPACITY_STEP_MWH)
    parser.add_argument("--output-prefix", default=None)
    args = parser.parse_args()

    techs = ["li_lfp"] + list(LDES_DURATIONS_H.keys())
    if args.all_techs:
        for tid in techs:
            run_one_tech(args.demand_stem, tid,
                          args.output_prefix, args.step_mwh)
    elif args.tech:
        run_one_tech(args.demand_stem, args.tech,
                      args.output_prefix, args.step_mwh)
    else:
        parser.error("必须指定 --tech 或 --all-techs")


if __name__ == "__main__":
    main()
