"""
M4 progression 搜索 —— 方案 A：LDES 优先放电 + LDES 优先充电。

与 m4_progression_search.py（方案 B：LFP 单日满放上限 + LDES 跨日补）独立，
只换 simulator 调度规则、其他参数完全一致，输出独立的结果文件:
  output/m4_economics/m4_progression_results_planA.csv
  output/m4_economics/_m4_prog_planA_{node}_dw{N}h.csv

通过 monkey-patch hybrid_simulator.simulate_hybrid_steady 注入方案 A 实现，
而无需复制 evaluate_combo 函数。
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# 先 patch：
#   1) 替换仿真器为方案 A
#   2) 加密 LDES 容量锚点（增加 5% 起点 + 中间分位），让 single-tech 曲线
#      在低满足率区段不再因稀疏采样而震荡，且能从供给率近 0 开始展示。
#      LFP 锚点同步加 5%。
import m4_hybrid_search
from hybrid_simulator_planA import simulate_hybrid_steady_planA
m4_hybrid_search.simulate_hybrid_steady = simulate_hybrid_steady_planA
m4_hybrid_search.LDES_GRID_ANCHORS_PCT = [5, 20, 35, 50, 65, 80, 95]
m4_hybrid_search.LFP_GRID_ANCHORS_PCT = [5, 20, 40, 60, 80, 95]

from event_clustering import EN_TO_CN
from ldes_params import load_tech_library
from ldes_economics import load_gas_prices, load_electricity_prices

from m4_hybrid_search import (
    _load_m3_2yr, build_province_grids,
    get_tech_params, evaluate_combo,
    LFP_DURATIONS_H, LDES_DURATIONS_H, LDES_TECHS, SCENARIOS,
    LFP_GRID_ANCHORS_PCT, LDES_GRID_ANCHORS_PCT,
)


warnings.filterwarnings("ignore")

PAPER_CODE_DIR = Path(__file__).resolve().parent
M3_DIR = PAPER_CODE_DIR / "output" / "m3_dispatch"
M4_DIR = PAPER_CODE_DIR / "output" / "m4_economics"

NODES = ["Y1", "Y3", "Y5", "Y10"]
DISCHARGE_WINDOWS_H = [8.0, 12.0, 16.0, 24.0]
OUT_CSV = M4_DIR / "m4_progression_results_planA.csv"

# 与 m4_progression_search.py 保持一致：Y3/Y5/Y10 容量网格按 demand 倍率放大
NODE_DEMAND_MULT = {"Y1": 1.0, "Y3": 1.4, "Y5": 1.8, "Y10": 2.5}


def load_node_inputs(node: str, dw_value: float):
    print(f"Loading {node} 5yr demand/curt/cw...")
    demand_df = pd.read_csv(M3_DIR / f"demand_5yr_{node}.csv", index_col=0)
    curt_df = pd.read_csv(M3_DIR / f"curtailment_5yr_{node}.csv", index_col=0)
    cw_df = pd.read_csv(M3_DIR / f"charge_window_5yr_{node}.csv", index_col=0)
    n_days = len(demand_df)
    print(f"  shape: {demand_df.shape}, n_days={n_days}")
    dw = float(dw_value)
    dw_arr = np.full(n_days, dw, dtype=np.float64)
    return demand_df, curt_df, cw_df, dw_arr, dw


def load_prices():
    g = load_gas_prices()
    e = load_electricity_prices()
    return (g.set_index("province_cn")["gas_price_cny_m3"].to_dict(),
            e.set_index("province_cn")["electricity_price_cny_kwh"].to_dict())


def run_one_node_dw(node: str, dw_value: float,
                      gas_map, elec_map, tech_lib) -> Path:
    demand_df, curt_df, cw_df, dw_arr, dw = load_node_inputs(node, dw_value)
    rows = []

    for scenario in SCENARIOS:
        lfp_tps = {}
        for ld in LFP_DURATIONS_H:
            tp = get_tech_params(tech_lib, "li_lfp", scenario, ld)
            if tp is not None:
                lfp_tps[ld] = tp
        if not lfp_tps:
            continue

        for ldes_id in LDES_TECHS:
            ldes_durs = LDES_DURATIONS_H.get(ldes_id, [])
            ldes_tps = {}
            for d in ldes_durs:
                tp = get_tech_params(tech_lib, ldes_id, scenario, d)
                if tp is not None:
                    ldes_tps[d] = tp
            if not ldes_tps:
                continue

            t0 = time.time()
            n_prov = 0
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

                for lfp_dur, lfp_tp in lfp_tps.items():
                    for ldes_dur, ldes_tp in ldes_tps.items():
                        lfp_caps, ldes_caps = build_province_grids(
                            prov_en, lfp_dur, ldes_id, ldes_dur)
                        mult = NODE_DEMAND_MULT.get(node, 1.0)
                        if mult != 1.0:
                            lfp_caps = sorted({0.0, *(c * mult for c in lfp_caps)})
                            ldes_caps = sorted({0.0, *(c * mult for c in ldes_caps)})
                        for lfp_cap in lfp_caps:
                            for ldes_cap in ldes_caps:
                                if lfp_cap == 0 and ldes_cap == 0:
                                    continue
                                r = evaluate_combo(
                                    d_arr, c_arr, w_arr, dw_arr,
                                    lfp_tp, lfp_cap, ldes_tp, ldes_cap,
                                    gas_p, elec_p, dw,
                                )
                                r.update({
                                    "node": node,
                                    "scenario": scenario,
                                    "dispatch_window_h": dw,
                                    "ldes_tech": ldes_id,
                                    "ldes_duration_h": ldes_dur,
                                    "lfp_duration_h": lfp_dur,
                                    "province_en": prov_en,
                                    "province_cn": prov_cn,
                                    "lfp_capacity_mwh": lfp_cap,
                                    "ldes_capacity_mwh": ldes_cap,
                                })
                                rows.append(r)
                n_prov += 1
            elapsed = time.time() - t0
            print(f"  [planA|{node}|dw={dw}h|{scenario}] {ldes_id} "
                  f"({len(ldes_tps)} dur, {n_prov} prov) "
                  f"done in {elapsed:.1f}s, rows: {len(rows)}",
                  flush=True)

    chunk_path = M4_DIR / f"_m4_prog_planA_{node}_dw{int(dw)}h.csv"
    pd.DataFrame(rows).to_csv(chunk_path, index=False)
    print(f"  -> {chunk_path.name} saved ({len(rows)} rows)", flush=True)
    del rows
    return chunk_path


def main():
    M4_DIR.mkdir(parents=True, exist_ok=True)
    gas_map, elec_map = load_prices()
    tech_lib = load_tech_library()
    print(f"=== Plan A (LDES priority discharge) ===")
    print(f"  gas: {len(gas_map)} prov, elec: {len(elec_map)} prov")
    print(f"Nodes: {NODES}")
    print(f"Dispatch windows: {DISCHARGE_WINDOWS_H} h")
    _load_m3_2yr()

    chunk_paths = []
    for node in NODES:
        for dw in DISCHARGE_WINDOWS_H:
            print(f"\n========== planA | {node} | dw = {dw}h ==========",
                  flush=True)
            p = run_one_node_dw(node, dw, gas_map, elec_map, tech_lib)
            chunk_paths.append(p)

    print("\n=== 合并 plan A 所有 (node, dw) chunk ===")
    parts = [pd.read_csv(p) for p in chunk_paths]
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"输出: {OUT_CSV}")
    print(f"总行数: {len(df)}")

    print("\n=== 摘要：每 (node, dw, scenario, LDES) 在 satisfy>=80% 下最优单位成本 ===")
    sub = df[df["satisfy_rate_pct"] >= 80].copy()
    if not sub.empty:
        best = (sub.sort_values("unit_cost_cny_kwh")
                  .groupby(["node", "dispatch_window_h", "scenario",
                              "ldes_tech"], as_index=False).first())
        print(best[["node", "dispatch_window_h", "scenario", "ldes_tech",
                      "lfp_duration_h", "lfp_capacity_mwh",
                      "ldes_duration_h", "ldes_capacity_mwh",
                      "satisfy_rate_pct", "unit_cost_cny_kwh",
                      "cost_ratio_to_gas", "lfp_supply_share"]
                  ].to_string(index=False))


if __name__ == "__main__":
    main()
