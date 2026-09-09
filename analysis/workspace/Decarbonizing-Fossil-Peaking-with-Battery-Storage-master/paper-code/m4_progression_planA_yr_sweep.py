"""
M4 progression — Plan A hybrid 多年限 sweep（按 tech 分文件）。

与 m4_progression_planA_hybrid_per_tech.py 完全相同的扫描逻辑（50/50
锚点、bin-min、方案 A 调度），唯一变量是「demand 仿真年限」：
  Y0_real:  365 天纯真实 demand（无 markov）
  Y1_yr1:   365 天 markov + 极端注入
  Y2_yr2:   1 真实 + 1 仿真 (730 天)
  Y3_yr3:   1 真实 + 2 仿真 (1095 天)
  Y4_yr4:   1 真实 + 3 仿真 (1460 天)
  Y5_yr5:   1 真实 + 4 仿真 (1825 天) — 已存在 m4_progression_planA_<tech>.csv

每年限 × 7 LDES tech，每个 tech 一个 CSV：
  output/m4_economics/m4_progression_planA_yr<N>_<tech>.csv

CLI：
  python m4_progression_planA_yr_sweep.py <year_node> <ldes_tech>
  python m4_progression_planA_yr_sweep.py --all-techs <year_node>
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# 注入方案 A 仿真器
import m4_hybrid_search
from hybrid_simulator_planA import simulate_hybrid_steady_planA
m4_hybrid_search.simulate_hybrid_steady = simulate_hybrid_steady_planA

# 50/50 锚点
import numpy as _np
_DENSE = sorted(set(_np.linspace(1, 99, 50).round().astype(int).tolist()))
m4_hybrid_search.LDES_GRID_ANCHORS_PCT = _DENSE
m4_hybrid_search.LFP_GRID_ANCHORS_PCT = _DENSE

from event_clustering import EN_TO_CN
from ldes_params import load_tech_library
from ldes_economics import load_gas_prices, load_electricity_prices
from m4_hybrid_search import (
    build_province_grids, get_tech_params, evaluate_combo,
    LFP_DURATIONS_H, LDES_DURATIONS_H,
)

warnings.filterwarnings("ignore")

PAPER_CODE_DIR = Path(__file__).resolve().parent
M3_DIR = PAPER_CODE_DIR / "output" / "m3_dispatch"
M4_DIR = PAPER_CODE_DIR / "output" / "m4_economics"

DISCHARGE_WINDOWS_H = [8.0, 12.0, 16.0, 24.0]
SCENARIOS = ["baseline"]

# 年限节点定义：(node_key, demand_filename_stem, n_years)
YR_NODES = {
    "Y0_real":  ("yr0_real_only", 1),
    "Y1_yr1":   ("yr1_markov", 1),
    "Y2_yr2":   ("yr2_real_plus_1mk", 2),
    "Y3_yr3":   ("yr3_real_plus_2mk", 3),
    "Y4_yr4":   ("yr4_real_plus_3mk", 4),
}


def load_yr_inputs(year_node: str, dw_value: float):
    if year_node not in YR_NODES:
        raise SystemExit(f"未知年限节点: {year_node}")
    fname_stem, _ = YR_NODES[year_node]
    demand_df = pd.read_csv(M3_DIR / f"demand_{fname_stem}.csv", index_col=0)
    curt_df = pd.read_csv(M3_DIR / f"curtailment_{fname_stem}.csv", index_col=0)
    cw_df = pd.read_csv(M3_DIR / f"charge_window_{fname_stem}.csv", index_col=0)
    n_days = len(demand_df)
    dw_arr = np.full(n_days, float(dw_value), dtype=np.float64)
    return demand_df, curt_df, cw_df, dw_arr, float(dw_value)


def load_prices():
    g = load_gas_prices()
    e = load_electricity_prices()
    return (g.set_index("province_cn")["gas_price_cny_m3"].to_dict(),
            e.set_index("province_cn")["electricity_price_cny_kwh"].to_dict())


def run_one_yr_tech(year_node: str, ldes_id: str):
    """对单个 (year_node, ldes_tech) 跑全 4 dw × 4 LFP_dur × N LDES_dur sweep。"""
    M4_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = M4_DIR / f"m4_progression_planA_{year_node}_{ldes_id}.csv"
    if out_csv.exists():
        out_csv.unlink()

    if ldes_id not in LDES_DURATIONS_H:
        raise SystemExit(f"未知 ldes tech: {ldes_id}")
    ldes_durations = LDES_DURATIONS_H[ldes_id]

    print(f"=== Plan A hybrid 加密扫描 [{year_node} | {ldes_id}] ===",
          flush=True)
    print(f"  LDES durations: {ldes_durations}, LFP: {LFP_DURATIONS_H}",
          flush=True)
    print(f"  LDES/LFP anchors (50/50): step=2", flush=True)
    print(f"  scenarios: {SCENARIOS}, 输出: {out_csv}", flush=True)

    gas_map, elec_map = load_prices()
    tech_lib = load_tech_library()

    inputs_cache = {}
    for dw_value in DISCHARGE_WINDOWS_H:
        inputs_cache[dw_value] = load_yr_inputs(year_node, dw_value)
    sample_dw = next(iter(inputs_cache))
    print(f"  loaded demand: shape={inputs_cache[sample_dw][0].shape}",
          flush=True)

    rows_buffer = []
    written_header = False
    total_rows = 0
    t_start = time.time()

    for scenario in SCENARIOS:
        lfp_tps = {}
        for ld in LFP_DURATIONS_H:
            tp = get_tech_params(tech_lib, "li_lfp", scenario, ld)
            if tp is not None:
                lfp_tps[ld] = tp
        ldes_tps = {}
        for d in ldes_durations:
            tp = get_tech_params(tech_lib, ldes_id, scenario, d)
            if tp is not None:
                ldes_tps[d] = tp
        if not lfp_tps or not ldes_tps:
            print(f"  [{scenario}] no params, skip"); continue

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

                for lfp_dur, lfp_tp in lfp_tps.items():
                    for ldes_dur, ldes_tp in ldes_tps.items():
                        lfp_caps, ldes_caps = build_province_grids(
                            prov_en, lfp_dur, ldes_id, ldes_dur)
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
                                    "year_node": year_node,
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
                                rows_buffer.append(r)
                                cell_rows += 1

                if rows_buffer:
                    df_chunk = pd.DataFrame(rows_buffer)
                    df_chunk.to_csv(out_csv, mode="a",
                                     header=(not written_header),
                                     index=False)
                    written_header = True
                    total_rows += len(rows_buffer)
                    rows_buffer = []

            elapsed = time.time() - t_cell
            print(f"  [{year_node}|{ldes_id}|dw={dw}h|{scenario}] "
                  f"{cell_rows} rows in {elapsed:.1f}s "
                  f"(累计 {total_rows:,})", flush=True)

    if rows_buffer:
        df_chunk = pd.DataFrame(rows_buffer)
        df_chunk.to_csv(out_csv, mode="a",
                         header=(not written_header), index=False)
        total_rows += len(rows_buffer)

    elapsed_total = time.time() - t_start
    print(f"\n=== [{year_node} | {ldes_id}] 完成: 共 {total_rows:,} 行, "
          f"耗时 {elapsed_total/60:.1f} min ===", flush=True)
    return out_csv, total_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("year_node",
                        help=f"年限节点: {list(YR_NODES.keys())}")
    parser.add_argument("ldes_tech", nargs="?", default=None,
                        help="caes/laes/gravity/fecr_flow/molten_salt_tes/"
                              "vrfb/h2_cavern")
    parser.add_argument("--all-techs", action="store_true",
                        help="串行跑所有 7 LDES tech")
    args = parser.parse_args()

    if args.all_techs:
        for tid in LDES_DURATIONS_H.keys():
            run_one_yr_tech(args.year_node, tid)
    elif args.ldes_tech:
        run_one_yr_tech(args.year_node, args.ldes_tech)
    else:
        parser.error("必须指定 ldes_tech 或 --all-techs")


if __name__ == "__main__":
    main()
