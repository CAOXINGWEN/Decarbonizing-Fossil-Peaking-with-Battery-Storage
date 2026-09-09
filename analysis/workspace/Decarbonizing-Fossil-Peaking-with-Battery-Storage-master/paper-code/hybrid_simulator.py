"""
LFP + LDES 双储能日步长仿真（Numba 加速）。

调度规则（方案 B：日内调峰 vs 日间转移分工）:
  - 放电：LFP 上限 = LFP 单日满放电量（lfp_capacity × discharge_eff）；
          单日 demand 超过该上限的部分必须由 LDES 跨日补。
          这把 LFP 限定在「日内调峰」，LDES 承担「日间/周间转移」。
  - 充电：LFP 优先（响应快、循环耐用），LFP 充满后剩余弃电充 LDES。
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def simulate_hybrid_year(
    demand_arr,
    curtailment_arr,
    charge_window_h_arr,
    discharge_window_h_arr,
    # LFP 参数
    lfp_capacity_kwh, lfp_charge_pwr_kw, lfp_discharge_pwr_kw,
    lfp_charge_eff, lfp_discharge_eff, lfp_self_disch,
    # LDES 参数
    ldes_capacity_kwh, ldes_charge_pwr_kw, ldes_discharge_pwr_kw,
    ldes_charge_eff, ldes_discharge_eff, ldes_self_disch,
    soc_min_frac, soc_max_frac,
):
    """LFP + LDES 双储能日步长仿真。返回每日各通道供给 / 充电 / 弃电。"""
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
        # 自放电
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

        # === 放电（方案 B）：LFP 单日满放上限 = 日内调峰；超出部分由 LDES 跨日补 ===
        # LFP 上限 = min(SOC 可放出量, 单日功率×dw, 单日满放容量)
        avail_lfp = (soc_lfp - e_min_lfp) * lfp_discharge_eff
        if avail_lfp < 0.0:
            avail_lfp = 0.0
        # 单日满放容量上限：整块 LFP 在一天内放光
        full_day_lfp = (e_max_lfp - e_min_lfp) * lfp_discharge_eff
        max_disch_lfp = avail_lfp
        if max_disch_lfp > lim_disch_lfp:
            max_disch_lfp = lim_disch_lfp
        if max_disch_lfp > full_day_lfp:
            max_disch_lfp = full_day_lfp
        sup_lfp = demand if demand < max_disch_lfp else max_disch_lfp
        if sup_lfp > 0.0 and lfp_discharge_eff > 0.0:
            soc_lfp -= sup_lfp / lfp_discharge_eff

        remaining = demand - sup_lfp
        if remaining < 0.0:
            remaining = 0.0
        avail_ldes = (soc_ldes - e_min_ldes) * ldes_discharge_eff
        if avail_ldes < 0.0:
            avail_ldes = 0.0
        max_disch_ldes = avail_ldes
        if max_disch_ldes > lim_disch_ldes:
            max_disch_ldes = lim_disch_ldes
        sup_ldes = remaining if remaining < max_disch_ldes else max_disch_ldes
        if sup_ldes > 0.0 and ldes_discharge_eff > 0.0:
            soc_ldes -= sup_ldes / ldes_discharge_eff

        total_supply_lfp += sup_lfp
        total_supply_ldes += sup_ldes
        daily_supply_lfp[i] = sup_lfp
        daily_supply_ldes[i] = sup_ldes

        # === 充电：LFP 优先（响应快、循环耐用） ===
        space_lfp = e_max_lfp - soc_lfp
        if space_lfp < 0.0:
            space_lfp = 0.0
        max_into_lfp = curt * lfp_charge_eff
        if max_into_lfp > lim_ch_lfp:
            max_into_lfp = lim_ch_lfp
        ch_lfp = max_into_lfp if max_into_lfp < space_lfp else space_lfp
        used_lfp = ch_lfp / lfp_charge_eff if lfp_charge_eff > 0.0 else 0.0
        soc_lfp += ch_lfp
        remaining_curt = curt - used_lfp
        if remaining_curt < 0.0:
            remaining_curt = 0.0

        space_ldes = e_max_ldes - soc_ldes
        if space_ldes < 0.0:
            space_ldes = 0.0
        max_into_ldes = remaining_curt * ldes_charge_eff
        if max_into_ldes > lim_ch_ldes:
            max_into_ldes = lim_ch_ldes
        ch_ldes = max_into_ldes if max_into_ldes < space_ldes else space_ldes
        used_ldes = ch_ldes / ldes_charge_eff if ldes_charge_eff > 0.0 else 0.0
        soc_ldes += ch_ldes
        spill = remaining_curt - used_ldes
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


def simulate_hybrid_steady(
    demand_arr, curtailment_arr, cw_arr, dw_arr,
    lfp_capacity_kwh, lfp_charge_pwr_kw, lfp_discharge_pwr_kw,
    lfp_charge_eff, lfp_discharge_eff, lfp_self_disch,
    ldes_capacity_kwh, ldes_charge_pwr_kw, ldes_discharge_pwr_kw,
    ldes_charge_eff, ldes_discharge_eff, ldes_self_disch,
    soc_min_frac, soc_max_frac,
    max_iter=15, conv_ratio=0.001,
):
    """对 LFP+LDES 一同做 SOC 收敛迭代。直接重复 N 次取最后一次结果。"""
    last = None
    for _ in range(max_iter):
        last = simulate_hybrid_year(
            demand_arr, curtailment_arr, cw_arr, dw_arr,
            lfp_capacity_kwh, lfp_charge_pwr_kw, lfp_discharge_pwr_kw,
            lfp_charge_eff, lfp_discharge_eff, lfp_self_disch,
            ldes_capacity_kwh, ldes_charge_pwr_kw, ldes_discharge_pwr_kw,
            ldes_charge_eff, ldes_discharge_eff, ldes_self_disch,
            soc_min_frac, soc_max_frac,
        )
    return last
