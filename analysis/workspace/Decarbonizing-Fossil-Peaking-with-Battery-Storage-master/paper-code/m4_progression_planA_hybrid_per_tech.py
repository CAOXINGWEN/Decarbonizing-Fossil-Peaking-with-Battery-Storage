"""
M4 progression — 方案 A hybrid 全量扫描（按 tech 分文件，加密锚点）。

每个 LDES tech 一个独立 Python 进程、独立 CSV：
  output/m4_economics/m4_progression_planA_<tech>.csv

调度规则：方案 A —— LDES 优先放电、LDES 优先充电（LFP 兜底）。

扫描维度（与 m4_progression_search_planA.py 同结构，但加密锚点 + 分文件存储）：
  - node ∈ {Y1, Y3, Y5, Y10}
  - dispatch window dw ∈ {8, 12, 16, 24}h
  - LFP duration ∈ {4, 6, 8, 10}h
  - LDES duration: 该 tech 全部
  - 26 省
  - LFP 容量锚点：[5, 20, 40, 60, 80, 95]% （6 档）
  - LDES 容量锚点：[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
                   65, 70, 75, 80, 85, 90, 95, 99]% （20 档）
  - 加 0 容量：LFP 7 点、LDES 21 点（不全交叉，过滤掉 lfp=0 且 ldes=0）

输出列：
  与 m4_progression_results_planA.csv 同结构（hybrid 格式）—— evaluate_combo
  返回的成本字段 + node/scenario/dispatch_window_h/ldes_tech 等坐标。

并行：
  - python m4_progression_planA_hybrid_per_tech.py <ldes_tech_id>
  - 推荐 7 进程并行（caes/laes/gravity/fecr_flow/molten_salt_tes/vrfb/h2_cavern）
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

# 50/50 锚点版：LFP / LDES 各 50 档，1/3/5/.../99 步长 2，密度大幅提升。
# 单 (lfp_dur, ldes_dur, 省) 组合数 = 51 × 51 - 1 = 2600（vs 18+23 锚点的 455）
# 数据量 ×5.7，预计 7 进程并行 ~1.5 小时，全套约 16 GB。
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

NODES = ["Y1", "Y3", "Y5", "Y10"]
DISCHARGE_WINDOWS_H = [8.0, 12.0, 16.0, 24.0]
SCENARIOS = ["baseline"]   # 只跑 baseline，节省时间

# Y3/Y5/Y10 容量网格按 demand 倍率放大（与 m4_progression_search_planA 一致）
NODE_DEMAND_MULT = {"Y1": 1.0, "Y3": 1.4, "Y5": 1.8, "Y10": 2.5}


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


def run_one_ldes_tech(ldes_id: str):
    M4_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = M4_DIR / f"m4_progression_planA_{ldes_id}.csv"
    if out_csv.exists():
        out_csv.unlink()

    if ldes_id not in LDES_DURATIONS_H:
        raise SystemExit(f"未知 ldes tech: {ldes_id}")
    ldes_durations = LDES_DURATIONS_H[ldes_id]

    print(f"=== Plan A hybrid 加密扫描 [{ldes_id}] ===", flush=True)
    print(f"  LDES durations: {ldes_durations}", flush=True)
    print(f"  LFP durations: {LFP_DURATIONS_H}", flush=True)
    print(f"  LDES anchors (20): {m4_hybrid_search.LDES_GRID_ANCHORS_PCT}",
          flush=True)
    print(f"  LFP anchors (6): {m4_hybrid_search.LFP_GRID_ANCHORS_PCT}",
          flush=True)
    print(f"  scenarios: {SCENARIOS}, 输出: {out_csv}", flush=True)

    gas_map, elec_map = load_prices()
    tech_lib = load_tech_library()

    # 缓存所有 (node, dw) 输入数据
    inputs_cache = {}
    for node in NODES:
        for dw_value in DISCHARGE_WINDOWS_H:
            inputs_cache[(node, dw_value)] = load_node_inputs(node, dw_value)

    rows_buffer = []
    written_header = False
    total_rows = 0
    t_start = time.time()

    for scenario in SCENARIOS:
        # 加载 LFP 4 dur + ldes_id 的所有 dur
        lfp_tps = {}
        for ld in LFP_DURATIONS_H:
            tp = get_tech_params(tech_lib, "li_lfp", scenario, ld)
            if tp is not None:
                lfp_tps[ld] = tp
        if not lfp_tps:
            print(f"  [{scenario}] no LFP params, skip")
            continue

        ldes_tps = {}
        for d in ldes_durations:
            tp = get_tech_params(tech_lib, ldes_id, scenario, d)
            if tp is not None:
                ldes_tps[d] = tp
        if not ldes_tps:
            print(f"  [{scenario}] no params for {ldes_id}, skip")
            continue

        for node in NODES:
            for dw_value in DISCHARGE_WINDOWS_H:
                demand_df, curt_df, cw_df, dw_arr, dw = inputs_cache[(node, dw_value)]
                t_cell = time.time()
                cell_rows = 0
                mult = NODE_DEMAND_MULT.get(node, 1.0)

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
                            if mult != 1.0:
                                lfp_caps = sorted(
                                    {0.0, *(c * mult for c in lfp_caps)})
                                ldes_caps = sorted(
                                    {0.0, *(c * mult for c in ldes_caps)})
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
                                        "node": node, "scenario": scenario,
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

                    # 每完成一个 (prov) 立即 flush，避免 Y10 大省累积爆 buffer
                    if rows_buffer:
                        df_chunk = pd.DataFrame(rows_buffer)
                        df_chunk.to_csv(out_csv, mode="a",
                                         header=(not written_header),
                                         index=False)
                        written_header = True
                        total_rows += len(rows_buffer)
                        rows_buffer = []

                elapsed_cell = time.time() - t_cell
                print(f"  [{ldes_id}|{node}|dw={dw}h|{scenario}] "
                      f"{cell_rows} rows in {elapsed_cell:.1f}s "
                      f"(累计 {total_rows:,})",
                      flush=True)

    # 收尾 flush
    if rows_buffer:
        df_chunk = pd.DataFrame(rows_buffer)
        df_chunk.to_csv(out_csv, mode="a",
                         header=(not written_header), index=False)
        total_rows += len(rows_buffer)

    elapsed = time.time() - t_start
    print(f"\n=== [{ldes_id}] 完成: 共 {total_rows:,} 行, "
          f"耗时 {elapsed/60:.1f} min ===", flush=True)
    return out_csv, total_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ldes_tech", nargs="?", default=None,
                        help="caes / laes / gravity / fecr_flow / "
                              "molten_salt_tes / vrfb / h2_cavern")
    parser.add_argument("--all", action="store_true",
                        help="串行跑所有 7 LDES（不推荐，建议 7 进程并行）")
    args = parser.parse_args()

    if args.all:
        for tid in LDES_DURATIONS_H.keys():
            run_one_ldes_tech(tid)
    elif args.ldes_tech:
        run_one_ldes_tech(args.ldes_tech)
    else:
        parser.error("必须指定 ldes_tech 或 --all")


if __name__ == "__main__":
    main()
