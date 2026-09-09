"""
M4 progression — single-tech 全量容量扫描。

每个 tech 一个独立 Python 进程、独立 CSV：
  output/m4_economics/m4_single_tech_<tech_id>.csv

扫描维度：
  - node ∈ {Y1, Y3, Y5, Y10}            (4)
  - dispatch window dw ∈ {8, 12, 16, 24}h (4)
  - 该技术全部 duration                  (LFP 4-10h: 4; LDES 各自 3-5)
  - 26 省                                (26)
  - 容量 step = 25 MWh，从 step 起步，扫到 satisfy=99.99% 或停止条件
    （直接复用 ldes_simulator.simulate_steady_state 的 SOC 收敛早停）

仿真器：直接调 ldes_simulator.simulate_steady_state（单技术原生版，
SOC 收敛即早停，比 hybrid_simulator 固定 15 次迭代快 5-7×）。

成本计算：复用 m4_hybrid_search.annualized_cost（与 evaluate_combo 完全
等价的 capex + 固定 O&M 公式）+ var O&M + 单位成本 / gas 比率。

并行：
  - 单 tech：python m4_progression_single_tech_sweep.py <tech_id>
  - 推荐：8 个 tech 各起一个进程，机器 384 核扛得住
  - 全部串行：python m4_progression_single_tech_sweep.py --all
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

NODES = ["Y1", "Y3", "Y5", "Y10"]
DISCHARGE_WINDOWS_H = [8.0, 12.0, 16.0, 24.0]
SCENARIOS = ["baseline"]

CAPACITY_STEP_MWH = 25.0
DISCOUNT_RATE = float(CONFIG["cost"]["discount_rate"])
GEN_COST = float(CONFIG["gas"]["generator_cost"])


def load_node_inputs(node: str, dw_value: float):
    demand_df = pd.read_csv(M3_DIR / f"demand_5yr_{node}.csv", index_col=0)
    curt_df = pd.read_csv(M3_DIR / f"curtailment_5yr_{node}.csv", index_col=0)
    cw_df = pd.read_csv(M3_DIR / f"charge_window_5yr_{node}.csv", index_col=0)
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
    """评估单技术单容量点：调 simulate_steady_state + 算单位成本。"""
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
    """对单 (省, tech, dur) 做容量扫描，沿用 M3 capacity_search 的停止逻辑。

    上限策略（与 ldes_simulator.capacity_search 对齐）：
      - 启发式上限：依赖 satisfy 早停（>=99.99%）和 check_should_stop_numba。
      - 硬上限避免单 cell 失控（之前 max=total_demand/1000*10 把 kWh 错当 MWh，
        导致 Y10 上限可达 3.4 亿 MWh）：使用 5 年 demand 峰值 × 100 作为
        合理上限（约相当于一周连续放电的容量）。
    """
    total_demand = float(np.sum(demand_arr))  # 单位 kWh
    peak_daily = float(np.max(demand_arr))    # 单位 kWh
    # 经验上限：单日峰值 × 100（足够覆盖任何 14 天连峰极端事件）
    # 单位换算：peak_daily kWh / 1000 = MWh
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


def run_one_tech(tech_id: str, step_mwh: float = CAPACITY_STEP_MWH,
                  nodes_filter: list[str] = None, append_only: bool = False):
    M4_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = M4_DIR / f"m4_single_tech_{tech_id}.csv"
    if out_csv.exists() and not append_only:
        out_csv.unlink()
    written_header = out_csv.exists()  # append 模式下保留已有 header

    if tech_id == "li_lfp":
        durations = LFP_DURATIONS_H
    elif tech_id in LDES_DURATIONS_H:
        durations = LDES_DURATIONS_H[tech_id]
    else:
        raise SystemExit(f"未知 tech_id: {tech_id}")

    nodes_to_run = nodes_filter if nodes_filter else NODES

    print(f"=== Single-tech 全量扫描 [{tech_id}] ===", flush=True)
    print(f"  durations: {durations}, step: {step_mwh} MWh", flush=True)
    print(f"  scenarios: {SCENARIOS}, nodes: {nodes_to_run}, "
          f"append: {append_only}, 输出: {out_csv}", flush=True)

    gas_map, elec_map = load_prices()
    tech_lib = load_tech_library()

    inputs_cache = {}
    for node in nodes_to_run:
        for dw_value in DISCHARGE_WINDOWS_H:
            inputs_cache[(node, dw_value)] = load_node_inputs(node, dw_value)

    rows_buffer = []
    chunk_size = 200_000
    total_rows = 0
    written_header = False
    t_start = time.time()

    for scenario in SCENARIOS:
        tech_tps = {}
        for d in durations:
            tp = get_tech_params(tech_lib, tech_id, scenario, d)
            if tp is not None:
                tech_tps[d] = tp
        if not tech_tps:
            print(f"  [{scenario}] no params for {tech_id}, skip")
            continue

        for node in nodes_to_run:
            for dw_value in DISCHARGE_WINDOWS_H:
                demand_df, curt_df, cw_df, dw_arr, dw = inputs_cache[(node, dw_value)]
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
                                "node": node, "scenario": scenario,
                                "dispatch_window_h": dw,
                                "tech_id": tech_id,
                                "duration_h": dur,
                                "province_en": prov_en,
                                "province_cn": prov_cn,
                            })
                        rows_buffer.extend(results)
                        cell_rows += len(results)

                        # 每完成一个 (prov, dur) 立即 flush，彻底避免 buffer
                        # 累积。Y10 大省单 (prov, dur) 也只有几万行，安全。
                        if rows_buffer:
                            df_chunk = pd.DataFrame(rows_buffer)
                            df_chunk.to_csv(out_csv, mode="a",
                                             header=(not written_header),
                                             index=False)
                            written_header = True
                            total_rows += len(rows_buffer)
                            rows_buffer = []

                # 每完成一个 (node, dw) cell 末尾收尾 flush（理论上 buffer 已空）
                if rows_buffer:
                    df_chunk = pd.DataFrame(rows_buffer)
                    df_chunk.to_csv(out_csv, mode="a",
                                     header=(not written_header),
                                     index=False)
                    written_header = True
                    total_rows += len(rows_buffer)

                elapsed_cell = time.time() - t_cell
                print(f"  [{tech_id}|{node}|dw={dw}h|{scenario}] "
                      f"{cell_rows} rows in {elapsed_cell:.1f}s "
                      f"(累计 {total_rows + len(rows_buffer):,})",
                      flush=True)

    if rows_buffer:
        df_chunk = pd.DataFrame(rows_buffer)
        df_chunk.to_csv(out_csv, mode="a",
                         header=(not written_header), index=False)
        total_rows += len(rows_buffer)

    elapsed = time.time() - t_start
    print(f"\n=== [{tech_id}] 完成: 共 {total_rows:,} 行, "
          f"耗时 {elapsed/60:.1f} min ===", flush=True)
    return out_csv, total_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tech_id", nargs="?", default=None,
                        help="li_lfp / caes / laes / gravity / fecr_flow / "
                              "molten_salt_tes / vrfb / h2_cavern")
    parser.add_argument("--all", action="store_true",
                        help="串行跑所有 8 技术（不推荐，建议 8 进程并行）")
    parser.add_argument("--step-mwh", type=float, default=CAPACITY_STEP_MWH)
    parser.add_argument("--nodes", default=None,
                        help="逗号分隔节点列表，如 'Y10' 或 'Y5,Y10'，缺省跑全部")
    parser.add_argument("--append", action="store_true",
                        help="追加模式：不删除已有 CSV，直接续写（与 --nodes 配合补跑）")
    args = parser.parse_args()

    nodes_filter = args.nodes.split(",") if args.nodes else None

    if args.all:
        for tid in ["li_lfp"] + list(LDES_DURATIONS_H.keys()):
            run_one_tech(tid, args.step_mwh, nodes_filter, args.append)
    elif args.tech_id:
        run_one_tech(args.tech_id, args.step_mwh, nodes_filter, args.append)
    else:
        parser.error("必须指定 tech_id 或 --all")


if __name__ == "__main__":
    main()
