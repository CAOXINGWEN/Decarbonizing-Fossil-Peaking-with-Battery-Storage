"""
M4 多技术组合网格搜索：基线 LFP + 各类 LDES 容量组合的经济性评估。

输入：
  - 730 天 demand / curtailment / charge_window CSV（M3 双年加强版）
  - M2 LDES 参数库 v0.2

组合：
  LFP_DURATION_H = 4  (锂电固定 4h 时长)
  LFP_CAPACITIES_MWH  = [0, 25, 50, 100, 200, 400]
  LDES_CAPACITIES_MWH = [0, 50, 100, 250, 500, 1000]
  LDES_TECHS = ['caes', 'laes', 'gravity', 'fecr_flow',
                 'molten_salt_tes', 'vrfb', 'h2_cavern']
  scenarios = ['baseline', 'low_cost']

输出：
  output/m4_economics/m4_hybrid_results_2yr.csv
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from config import CONFIG, GAS_POWER_PER_M3, calculate_crf
from event_clustering import EN_TO_CN
from ldes_params import load_tech_library, extract_tech_params
from ldes_economics import load_gas_prices, load_electricity_prices
from hybrid_simulator import simulate_hybrid_steady


warnings.filterwarnings("ignore")

PAPER_CODE_DIR = Path(__file__).resolve().parent
M3_DIR = PAPER_CODE_DIR / "output" / "m3_dispatch"
M4_DIR = PAPER_CODE_DIR / "output" / "m4_economics"
DATA_DIR = PAPER_CODE_DIR.parent / "data"

DEMAND_2YR = M3_DIR / "demand_2yr_augmented.csv"
CURT_2YR = M3_DIR / "curtailment_2yr_augmented.csv"
CW_2YR = M3_DIR / "charge_window_2yr_augmented.csv"
OUT_CSV = M4_DIR / "m4_hybrid_results_2yr_dwsweep.csv"

GAS_PRICE_XLSX = DATA_DIR / "gas_prices.xlsx"
ELEC_PRICE_XLSX = DATA_DIR / "electricity_prices.xlsx"

LFP_DURATIONS_H = [4, 6, 8, 10]
LDES_DURATIONS_H = {
    "caes": [8, 10, 12, 16, 24],
    "laes": [8, 10, 12, 16, 24],
    "gravity": [8, 10, 12],
    "fecr_flow": [8, 10, 12],
    "molten_salt_tes": [8, 10, 12, 16],
    "vrfb": [4, 6, 8],
    "h2_cavern": [24, 48, 72, 168],
}
LDES_TECHS = list(LDES_DURATIONS_H.keys())
SCENARIOS = ["baseline", "low_cost"]
DISCHARGE_WINDOWS_H = [8.0, 12.0, 16.0, 24.0]

# 省级自适应网格：每省的 LFP / LDES 容量网格根据 M3 双年单技术结果反查
# 出该 (省, 技术, 时长) 达到不同 satisfy_rate 所需的最小容量作为 anchor。
# 这取代了旧版全国一律的 [0, 50, 100, 200, 400, 800] / [0, 100, 250, 500, 1000] MWh 写死网格——
# 旧网格对大需求省（江苏/河北/四川）严重低估、对小需求省（内蒙古/广西）严重浪费。
M3_2YR_RESULTS_CSV = M3_DIR / "m3_dispatch_results_2yr.csv"
LFP_GRID_ANCHORS_PCT = [20, 40, 60, 80, 95]   # LFP 5 个 satisfy anchor → 6 个容量点（含 0）
LDES_GRID_ANCHORS_PCT = [30, 60, 90]          # LDES 3 个 anchor → 4 个容量点（含 0），控制总评估量

_M3_CACHE = None


def _load_m3_2yr() -> pd.DataFrame:
    global _M3_CACHE
    if _M3_CACHE is None:
        if not M3_2YR_RESULTS_CSV.exists():
            raise FileNotFoundError(
                f"M3 2yr results not found at {M3_2YR_RESULTS_CSV}. "
                f"Run `python run_m3_2yr.py` first."
            )
        df = pd.read_csv(M3_2YR_RESULTS_CSV, usecols=[
            "tech_id", "duration_h", "province_en",
            "capacity_mwh", "satisfy_rate_pct", "scenario",
        ])
        # 按 (scenario, tech, dur, province) 建立 MultiIndex 加速反查
        df = df.sort_values(["scenario", "tech_id", "duration_h",
                              "province_en", "capacity_mwh"])
        df = df.set_index(["scenario", "tech_id", "duration_h", "province_en"])
        _M3_CACHE = df
    return _M3_CACHE


def province_capacity_anchors(prov_en: str, tech_id: str, duration_h: int,
                                anchors_pct,
                                scenario_for_m3: str = "baseline"
                                ) -> list[float]:
    """从 M3 2yr 结果反查 (省, 技术, 时长) 在每个 satisfy anchor 处的最小容量。

    返回包含 0 的升序容量列表（MWh），用作 hybrid 网格的该省专属取值。
    若该技术对该省最大满足率未到某个 anchor，则该 anchor 退化为 M3 搜索到的最大
    capacity（保证至少能尝试到该省的物理天花板）。
    """
    df = _load_m3_2yr()
    key = (scenario_for_m3, tech_id, int(duration_h), prov_en)
    try:
        sub = df.loc[key]
    except KeyError:
        return [0.0]
    if isinstance(sub, pd.Series):
        sub = sub.to_frame().T
    if sub.empty:
        return [0.0]

    cap_arr = sub["capacity_mwh"].to_numpy()
    sat_arr = sub["satisfy_rate_pct"].to_numpy()
    max_cap_searched = float(cap_arr.max())
    anchors = [0.0]
    for q in anchors_pct:
        ok = sat_arr >= q
        cap = float(cap_arr[ok].min()) if ok.any() else max_cap_searched
        anchors.append(cap)

    uniq = sorted({round(c, 1) for c in anchors})
    if uniq[0] != 0.0:
        uniq.insert(0, 0.0)
    return uniq


def build_province_grids(prov_en: str, lfp_dur: int,
                          ldes_tech: str, ldes_dur: int,
                          ) -> tuple[list[float], list[float]]:
    """对单个 (省, LFP_dur, LDES_tech, LDES_dur) 返回 (LFP 容量网格, LDES 容量网格)。

    LFP 网格固定取 [0, cap@20%, cap@40%, cap@60%, cap@80%, cap@95%]（LFP 自己达到该满足率所需）
    LDES 网格固定取 [0, cap@30%, cap@60%, cap@90%]（LDES 自己达到该满足率所需）
    """
    lfp_caps = province_capacity_anchors(
        prov_en, "li_lfp", lfp_dur, LFP_GRID_ANCHORS_PCT)
    ldes_caps = province_capacity_anchors(
        prov_en, ldes_tech, ldes_dur, LDES_GRID_ANCHORS_PCT)
    return lfp_caps, ldes_caps


def load_2yr_inputs(dw_value: float | None = None):
    print(f"Loading 2yr demand/curt/cw...")
    demand_df = pd.read_csv(DEMAND_2YR, index_col=0)
    curt_df = pd.read_csv(CURT_2YR, index_col=0)
    cw_df = pd.read_csv(CW_2YR, index_col=0)
    n_days = len(demand_df)
    print(f"  shape: {demand_df.shape}, n_days={n_days}")
    if dw_value is None:
        dw_value = float(CONFIG["dispatch"]["baseline_discharge_window_h"])
    dw = float(dw_value)
    dw_arr = np.full(n_days, dw, dtype=np.float64)
    return demand_df, curt_df, cw_df, dw_arr, dw


def load_prices():
    """复用 ldes_economics 的价格加载逻辑（中文省名）。"""
    g = load_gas_prices()
    e = load_electricity_prices()
    return (g.set_index("province_cn")["gas_price_cny_m3"].to_dict(),
            e.set_index("province_cn")["electricity_price_cny_kwh"].to_dict())


def get_tech_params(tech_lib, tech_id, scenario, duration_h):
    if tech_id not in tech_lib.index:
        return None
    row = tech_lib.loc[tech_id]
    dmin = float(row["duration_min_h"])
    dmax = float(row["duration_max_h"])
    if not (dmin <= duration_h <= dmax):
        # 取最近合法 duration
        if duration_h < dmin:
            duration_h = int(dmin)
        else:
            duration_h = int(dmax)
    return extract_tech_params(row, tech_id, scenario, duration_h)


def annualized_cost(tp, capacity_mwh, discount_rate=0.036):
    """tp.calendar_life_years 的年化 CAPEX + 固定 O&M。"""
    if capacity_mwh <= 0:
        return 0.0, 0.0
    power_mw = capacity_mwh / tp.duration_h
    discharge_pwr_kw = power_mw * 1000.0
    rated_pwr_kw = discharge_pwr_kw * max(tp.charge_power_ratio, 1.0)
    capacity_kwh = capacity_mwh * 1000.0
    capex = (tp.power_cost_cny_kw * rated_pwr_kw
              + tp.energy_cost_cny_kwh * capacity_kwh)
    crf = calculate_crf(discount_rate, tp.calendar_life_years)
    annual_capex = capex * crf
    annual_fom = tp.fixed_om_cny_kw_yr * rated_pwr_kw
    return capex, annual_capex + annual_fom


def evaluate_combo(demand_arr, curt_arr, cw_arr, dw_arr,
                    lfp_tp, lfp_cap_mwh,
                    ldes_tp, ldes_cap_mwh,
                    gas_price, elec_price, dw):
    """评估单个 (省, LDES_tech, LFP_cap, LDES_cap, scenario) 组合。"""
    soc_min = float(CONFIG["storage"]["soc_min"])
    soc_max = float(CONFIG["storage"]["soc_max"])
    discount = float(CONFIG["cost"]["discount_rate"])
    gen_cost = float(CONFIG["gas"]["generator_cost"])
    var_om_factor = 1.0 / 1000.0  # CNY/MWh -> CNY/kWh

    # 构造 LFP / LDES 各自的功率/容量参数
    lfp_cap_kwh = lfp_cap_mwh * 1000.0
    ldes_cap_kwh = ldes_cap_mwh * 1000.0
    lfp_disch_pwr_kw = (lfp_cap_kwh / lfp_tp.duration_h) if lfp_cap_mwh > 0 else 0.0
    lfp_ch_pwr_kw = lfp_disch_pwr_kw * lfp_tp.charge_power_ratio
    ldes_disch_pwr_kw = (ldes_cap_kwh / ldes_tp.duration_h) if ldes_cap_mwh > 0 else 0.0
    ldes_ch_pwr_kw = ldes_disch_pwr_kw * ldes_tp.charge_power_ratio

    # 0 容量保护：simulate 内除法，需用极小正数避免除零
    eps = 1e-6
    sim_lfp_cap = lfp_cap_kwh if lfp_cap_kwh > 0 else eps
    sim_ldes_cap = ldes_cap_kwh if ldes_cap_kwh > 0 else eps

    res = simulate_hybrid_steady(
        demand_arr, curt_arr, cw_arr, dw_arr,
        sim_lfp_cap, lfp_ch_pwr_kw, lfp_disch_pwr_kw,
        lfp_tp.charge_eff, lfp_tp.discharge_eff, lfp_tp.self_discharge_rate_day,
        sim_ldes_cap, ldes_ch_pwr_kw, ldes_disch_pwr_kw,
        ldes_tp.charge_eff, ldes_tp.discharge_eff, ldes_tp.self_discharge_rate_day,
        soc_min, soc_max,
    )
    (sup_lfp, sup_ldes, ch_lfp, ch_ldes, spill,
     daily_sup_lfp, daily_sup_ldes, daily_spill) = res

    # 730 天 → 年均
    n_days = demand_arr.shape[0]
    year_factor = 365.0 / n_days
    annual_supply_lfp = sup_lfp * year_factor
    annual_supply_ldes = sup_ldes * year_factor
    annual_supply_total = annual_supply_lfp + annual_supply_ldes
    total_demand_annual = float(np.sum(demand_arr)) * year_factor
    satisfy_pct = (annual_supply_total / total_demand_annual * 100.0
                    if total_demand_annual > 0 else 0.0)

    capex_lfp, ann_lfp_fixed = annualized_cost(lfp_tp, lfp_cap_mwh, discount)
    capex_ldes, ann_ldes_fixed = annualized_cost(ldes_tp, ldes_cap_mwh, discount)
    annual_var_om = (lfp_tp.variable_om_cny_mwh * annual_supply_lfp / 1000.0
                      + ldes_tp.variable_om_cny_mwh * annual_supply_ldes / 1000.0)
    annual_cost = ann_lfp_fixed + ann_ldes_fixed + annual_var_om
    unit_cost = (annual_cost / annual_supply_total
                  if annual_supply_total > 0 else np.nan)
    annual_revenue = annual_supply_total * elec_price
    annual_profit = annual_revenue - annual_cost
    gas_unit_cost = gas_price / GAS_POWER_PER_M3 + gen_cost

    return {
        "satisfy_rate_pct": satisfy_pct,
        "annual_supply_kwh": annual_supply_total,
        "annual_supply_lfp_kwh": annual_supply_lfp,
        "annual_supply_ldes_kwh": annual_supply_ldes,
        "lfp_supply_share": (annual_supply_lfp / annual_supply_total
                              if annual_supply_total > 0 else np.nan),
        "annual_charge_kwh": (ch_lfp + ch_ldes) * year_factor,
        "annual_spillage_kwh": spill * year_factor,
        "capex_lfp_cny": capex_lfp,
        "capex_ldes_cny": capex_ldes,
        "capex_total_cny": capex_lfp + capex_ldes,
        "annual_cost_cny": annual_cost,
        "annual_revenue_cny": annual_revenue,
        "annual_net_profit_cny": annual_profit,
        "unit_cost_cny_kwh": unit_cost,
        "gas_unit_cost_cny_kwh": gas_unit_cost,
        "cost_ratio_to_gas": (unit_cost / gas_unit_cost
                               if pd.notna(unit_cost) else np.nan),
        "unit_profit_cny_per_kwh_supply": (annual_profit / annual_supply_total
                                             if annual_supply_total > 0 else np.nan),
    }


def main():
    M4_DIR.mkdir(parents=True, exist_ok=True)

    gas_map, elec_map = load_prices()
    tech_lib = load_tech_library()
    print(f"  gas: {len(gas_map)} provinces, elec: {len(elec_map)} provinces")
    print(f"LFP durations: {LFP_DURATIONS_H}")
    print(f"LDES durations per tech: {LDES_DURATIONS_H}")
    print(f"Discharge windows (dw): {DISCHARGE_WINDOWS_H} h")
    print(f"Capacity grid mode: province-adaptive anchored on M3 2yr satisfy levels")
    print(f"  LFP anchors: {LFP_GRID_ANCHORS_PCT}% (5 anchors → 6 capacity points)")
    print(f"  LDES anchors: {LDES_GRID_ANCHORS_PCT}% (3 anchors → 4 capacity points)")
    _load_m3_2yr()  # 预热缓存

    # 流式：每个 dw 单独写 CSV，最后拼接，避免全部驻留内存
    dw_csv_paths = []

    for dw_value in DISCHARGE_WINDOWS_H:
        demand_df, curt_df, cw_df, dw_arr, dw = load_2yr_inputs(dw_value)
        print(f"\n=== dw = {dw}h ===", flush=True)
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
                                        "scenario": scenario,
                                        "dispatch_window_h": dw,
                                        "ldes_tech": ldes_id,
                                        "ldes_duration_h": ldes_dur,
                                        "lfp_duration_h": lfp_dur,
                                        "province_en": prov_en,
                                        "province_cn": prov_cn,
                                        "lfp_capacity_mwh": lfp_cap,
                                        "ldes_capacity_mwh": ldes_cap,
                                        "combo_label": (
                                            f"LFP{lfp_dur}h{lfp_cap:.0f}+{ldes_id.upper()}{ldes_dur}h{ldes_cap:.0f}"
                                            if (lfp_cap > 0 and ldes_cap > 0)
                                            else (f"LFP{lfp_dur}h{lfp_cap:.0f}-only" if lfp_cap > 0
                                                  else f"{ldes_id.upper()}{ldes_dur}h{ldes_cap:.0f}-only")
                                        ),
                                    })
                                    rows.append(r)
                    n_prov += 1
                elapsed = time.time() - t0
                print(f"  [dw={dw}h][{scenario}] {ldes_id} ({len(ldes_tps)} durations, "
                      f"{n_prov} prov) done in {elapsed:.1f}s, current dw rows: {len(rows)}",
                      flush=True)

        chunk_path = M4_DIR / f"_m4_hybrid_2yr_dw{int(dw)}h.csv"
        pd.DataFrame(rows).to_csv(chunk_path, index=False)
        dw_csv_paths.append(chunk_path)
        print(f"  -> dw={dw}h chunk saved: {chunk_path.name} ({len(rows)} rows)",
              flush=True)
        del rows  # 释放内存

    # 拼接所有 dw chunk
    print("\n=== 合并所有 dw chunk ===")
    parts = [pd.read_csv(p) for p in dw_csv_paths]
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"输出: {OUT_CSV}")
    print(f"总行数: {len(df)}")

    print("\n=== 摘要：每 (dw, scenario, LDES 技术) 在 satisfy>=80% 下最优单位成本 ===")
    sub = df[df["satisfy_rate_pct"] >= 80].copy()
    if not sub.empty:
        best = (sub.sort_values("unit_cost_cny_kwh")
                  .groupby(["dispatch_window_h", "scenario", "ldes_tech"],
                            as_index=False).first())
        print(best[["dispatch_window_h", "scenario", "ldes_tech",
                     "lfp_duration_h", "lfp_capacity_mwh",
                     "ldes_duration_h", "ldes_capacity_mwh", "satisfy_rate_pct",
                     "unit_cost_cny_kwh", "cost_ratio_to_gas",
                     "lfp_supply_share"]].to_string(index=False))


if __name__ == "__main__":
    main()
