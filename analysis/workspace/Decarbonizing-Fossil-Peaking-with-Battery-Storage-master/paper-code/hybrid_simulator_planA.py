"""
LFP + LDES 双储能日步长仿真 —— 方案 A：LDES 优先放电、LFP 兜底（Numba 加速）。

调度规则（方案 A）：
  - 放电：LDES 先放（被认为是"基底储备"），LFP 兜剩余
  - 充电：LDES 优先（让 LDES 有机会被填满），LFP 后充

注意：与 hybrid_simulator.py（方案 B：LFP 单日满放上限 + LDES 跨日补）独立，
避免互相覆盖；m4_progression_search_planA.py 专门 import 此模块。
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def simulate_hybrid_year_planA(
    demand_arr,
    curtailment_arr,
    charge_window_h_arr,
    discharge_window_h_arr,
    lfp_capacity_kwh, lfp_charge_pwr_kw, lfp_discharge_pwr_kw,
    lfp_charge_eff, lfp_discharge_eff, lfp_self_disch,
    ldes_capacity_kwh, ldes_charge_pwr_kw, ldes_discharge_pwr_kw,
    ldes_charge_eff, ldes_discharge_eff, ldes_self_disch,
    soc_min_frac, soc_max_frac,
):
    n_days = demand_arr.shape[0]
    daily_supply_lfp = np.empty(n_days, dtype=np.float64)
    daily_supply_ldes = np.empty(n_days, dtype=np.float64)
    daily_charge_lfp = np.empty(n_days, dtype=np.float64)
    daily_charge_ldes = np.empty(n_days, dtype=np.float64)
    daily_spillage = np.empty(n_days, dtype=np.float64)

    soc_lfp = soc_min_frac * lfp_capacity_kwh
    soc_ldes = soc_min_frac * ldes_capacity_kwh
    e_min_lfp = soc_min_frac * lfp_capacity_kwh
    e_max_lfp = soc_max_frac * lfp_capacity_kwh
    e_min_ldes = soc_min_frac * ldes_capacity_kwh
    e_max_ldes = soc_max_frac * ldes_capacity_kwh

    total_supply_lfp = 0.0
    total_supply_ldes = 0.0
    total_charge_lfp = 0.0
    total_charge_ldes = 0.0
    total_spillage = 0.0

    for i in range(n_days):
        if lfp_self_disch > 0.0:
            soc_lfp *= (1.0 - lfp_self_disch)
            if soc_lfp < e_min_lfp:
                soc_lfp = e_min_lfp
        if ldes_self_disch > 0.0:
            soc_ldes *= (1.0 - ldes_self_disch)
            if soc_ldes < e_min_ldes:
                soc_ldes = e_min_ldes

        demand = demand_arr[i]
        curt = curtailment_arr[i]
        cw = charge_window_h_arr[i]
        dw = discharge_window_h_arr[i]
        lim_disch_lfp = lfp_discharge_pwr_kw * dw
        lim_disch_ldes = ldes_discharge_pwr_kw * dw
        lim_ch_lfp = lfp_charge_pwr_kw * cw * lfp_charge_eff
        lim_ch_ldes = ldes_charge_pwr_kw * cw * ldes_charge_eff

        # === 放电：LDES 先放，LFP 兜底 ===
        avail_ldes = (soc_ldes - e_min_ldes) * ldes_discharge_eff
        if avail_ldes < 0.0:
            avail_ldes = 0.0
        max_disch_ldes = avail_ldes
        if max_disch_ldes > lim_disch_ldes:
            max_disch_ldes = lim_disch_ldes
        sup_ldes = demand if demand < max_disch_ldes else max_disch_ldes
        if sup_ldes > 0.0 and ldes_discharge_eff > 0.0:
            soc_ldes -= sup_ldes / ldes_discharge_eff

        remaining = demand - sup_ldes
        if remaining < 0.0:
            remaining = 0.0
        avail_lfp = (soc_lfp - e_min_lfp) * lfp_discharge_eff
        if avail_lfp < 0.0:
            avail_lfp = 0.0
        max_disch_lfp = avail_lfp
        if max_disch_lfp > lim_disch_lfp:
            max_disch_lfp = lim_disch_lfp
        sup_lfp = remaining if remaining < max_disch_lfp else max_disch_lfp
        if sup_lfp > 0.0 and lfp_discharge_eff > 0.0:
            soc_lfp -= sup_lfp / lfp_discharge_eff

        total_supply_lfp += sup_lfp
        total_supply_ldes += sup_ldes
        daily_supply_lfp[i] = sup_lfp
        daily_supply_ldes[i] = sup_ldes

        # === 充电：LDES 优先（让大库容先填满），LFP 后充 ===
        space_ldes = e_max_ldes - soc_ldes
        if space_ldes < 0.0:
            space_ldes = 0.0
        max_into_ldes = curt * ldes_charge_eff
        if max_into_ldes > lim_ch_ldes:
            max_into_ldes = lim_ch_ldes
        ch_ldes = max_into_ldes if max_into_ldes < space_ldes else space_ldes
        used_ldes = ch_ldes / ldes_charge_eff if ldes_charge_eff > 0.0 else 0.0
        soc_ldes += ch_ldes
        remaining_curt = curt - used_ldes
        if remaining_curt < 0.0:
            remaining_curt = 0.0

        space_lfp = e_max_lfp - soc_lfp
        if space_lfp < 0.0:
            space_lfp = 0.0
        max_into_lfp = remaining_curt * lfp_charge_eff
        if max_into_lfp > lim_ch_lfp:
            max_into_lfp = lim_ch_lfp
        ch_lfp = max_into_lfp if max_into_lfp < space_lfp else space_lfp
        used_lfp = ch_lfp / lfp_charge_eff if lfp_charge_eff > 0.0 else 0.0
        soc_lfp += ch_lfp
        spill = remaining_curt - used_lfp
        if spill < 0.0:
            spill = 0.0

        total_charge_lfp += used_lfp
        total_charge_ldes += used_ldes
        total_spillage += spill
        daily_charge_lfp[i] = used_lfp
        daily_charge_ldes[i] = used_ldes
        daily_spillage[i] = spill

    return (total_supply_lfp, total_supply_ldes,
            total_charge_lfp, total_charge_ldes, total_spillage,
            daily_supply_lfp, daily_supply_ldes, daily_spillage)


def simulate_hybrid_steady_planA(
    demand_arr, curtailment_arr, cw_arr, dw_arr,
    lfp_capacity_kwh, lfp_charge_pwr_kw, lfp_discharge_pwr_kw,
    lfp_charge_eff, lfp_discharge_eff, lfp_self_disch,
    ldes_capacity_kwh, ldes_charge_pwr_kw, ldes_discharge_pwr_kw,
    ldes_charge_eff, ldes_discharge_eff, ldes_self_disch,
    soc_min_frac, soc_max_frac,
    max_iter=15, conv_ratio=0.001,
):
    """方案 A 的 SOC 收敛迭代：直接重复 N 次取最后一次。"""
    last = None
    for _ in range(max_iter):
        last = simulate_hybrid_year_planA(
            demand_arr, curtailment_arr, cw_arr, dw_arr,
            lfp_capacity_kwh, lfp_charge_pwr_kw, lfp_discharge_pwr_kw,
            lfp_charge_eff, lfp_discharge_eff, lfp_self_disch,
            ldes_capacity_kwh, ldes_charge_pwr_kw, ldes_discharge_pwr_kw,
            ldes_charge_eff, ldes_discharge_eff, ldes_self_disch,
            soc_min_frac, soc_max_frac,
        )
    return last
