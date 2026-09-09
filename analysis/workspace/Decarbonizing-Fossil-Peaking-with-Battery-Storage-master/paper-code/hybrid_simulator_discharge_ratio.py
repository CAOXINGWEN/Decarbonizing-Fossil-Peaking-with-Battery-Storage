"""
LFP + CAES dispatch with fixed target discharge shares.

Two quota modes are supported:

  - hard_quota: each technology can only serve its assigned daily demand share.
  - soft_quota: technologies first serve assigned shares; any remaining demand
    can be served by residual deliverable energy from either technology.

Charging uses the same target share as a neutral first allocation, then lets
remaining curtailment fill any residual charge headroom. This keeps the
experiment focused on discharge shares without adding an LFP-first/CAES-first
charging bias.
"""

from __future__ import annotations

import numpy as np
from numba import njit


HARD_QUOTA = 0
SOFT_QUOTA = 1


@njit(cache=True, nogil=True)
def _allocate_residual_proportional(residual, extra_lfp, extra_ldes):
    add_lfp = 0.0
    add_ldes = 0.0
    if residual <= 0.0:
        return add_lfp, add_ldes
    if extra_lfp < 0.0:
        extra_lfp = 0.0
    if extra_ldes < 0.0:
        extra_ldes = 0.0
    extra_total = extra_lfp + extra_ldes
    if extra_total <= 0.0:
        return add_lfp, add_ldes
    if residual >= extra_total:
        return extra_lfp, extra_ldes
    add_lfp = residual * extra_lfp / extra_total
    add_ldes = residual - add_lfp
    return add_lfp, add_ldes


@njit(cache=True, nogil=True)
def simulate_hybrid_year_discharge_ratio(
    demand_arr,
    curtailment_arr,
    charge_window_h_arr,
    discharge_window_h_arr,
    lfp_capacity_kwh, lfp_charge_pwr_kw, lfp_discharge_pwr_kw,
    lfp_charge_eff, lfp_discharge_eff, lfp_self_disch,
    ldes_capacity_kwh, ldes_charge_pwr_kw, ldes_discharge_pwr_kw,
    ldes_charge_eff, ldes_discharge_eff, ldes_self_disch,
    soc_min_frac, soc_max_frac,
    lfp_discharge_share,
    quota_mode,
):
    n_days = demand_arr.shape[0]
    daily_supply_lfp = np.empty(n_days, dtype=np.float64)
    daily_supply_ldes = np.empty(n_days, dtype=np.float64)
    daily_charge_lfp = np.empty(n_days, dtype=np.float64)
    daily_charge_ldes = np.empty(n_days, dtype=np.float64)
    daily_spillage = np.empty(n_days, dtype=np.float64)

    if lfp_discharge_share < 0.0:
        lfp_discharge_share = 0.0
    if lfp_discharge_share > 1.0:
        lfp_discharge_share = 1.0
    ldes_discharge_share = 1.0 - lfp_discharge_share

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

        avail_lfp = (soc_lfp - e_min_lfp) * lfp_discharge_eff
        if avail_lfp < 0.0:
            avail_lfp = 0.0
        avail_ldes = (soc_ldes - e_min_ldes) * ldes_discharge_eff
        if avail_ldes < 0.0:
            avail_ldes = 0.0

        max_disch_lfp = avail_lfp
        if max_disch_lfp > lim_disch_lfp:
            max_disch_lfp = lim_disch_lfp
        max_disch_ldes = avail_ldes
        if max_disch_ldes > lim_disch_ldes:
            max_disch_ldes = lim_disch_ldes

        target_lfp = demand * lfp_discharge_share
        target_ldes = demand * ldes_discharge_share
        sup_lfp = target_lfp if target_lfp < max_disch_lfp else max_disch_lfp
        sup_ldes = target_ldes if target_ldes < max_disch_ldes else max_disch_ldes

        if quota_mode == SOFT_QUOTA:
            residual_demand = demand - sup_lfp - sup_ldes
            extra_lfp = max_disch_lfp - sup_lfp
            extra_ldes = max_disch_ldes - sup_ldes
            add_lfp, add_ldes = _allocate_residual_proportional(
                residual_demand, extra_lfp, extra_ldes
            )
            sup_lfp += add_lfp
            sup_ldes += add_ldes

        if sup_lfp > 0.0 and lfp_discharge_eff > 0.0:
            soc_lfp -= sup_lfp / lfp_discharge_eff
        if sup_ldes > 0.0 and ldes_discharge_eff > 0.0:
            soc_ldes -= sup_ldes / ldes_discharge_eff

        total_supply_lfp += sup_lfp
        total_supply_ldes += sup_ldes
        daily_supply_lfp[i] = sup_lfp
        daily_supply_ldes[i] = sup_ldes

        # Charge by the same target split first; residual curtailment can fill
        # either store's remaining charge headroom.
        max_input_lfp = 0.0
        if lfp_charge_eff > 0.0:
            max_input_lfp = lfp_charge_pwr_kw * cw
            space_input_lfp = (e_max_lfp - soc_lfp) / lfp_charge_eff
            if space_input_lfp < 0.0:
                space_input_lfp = 0.0
            if max_input_lfp > space_input_lfp:
                max_input_lfp = space_input_lfp
        max_input_ldes = 0.0
        if ldes_charge_eff > 0.0:
            max_input_ldes = ldes_charge_pwr_kw * cw
            space_input_ldes = (e_max_ldes - soc_ldes) / ldes_charge_eff
            if space_input_ldes < 0.0:
                space_input_ldes = 0.0
            if max_input_ldes > space_input_ldes:
                max_input_ldes = space_input_ldes

        target_input_lfp = curt * lfp_discharge_share
        target_input_ldes = curt * ldes_discharge_share
        used_lfp = target_input_lfp if target_input_lfp < max_input_lfp else max_input_lfp
        used_ldes = target_input_ldes if target_input_ldes < max_input_ldes else max_input_ldes
        residual_curt = curt - used_lfp - used_ldes
        extra_input_lfp = max_input_lfp - used_lfp
        extra_input_ldes = max_input_ldes - used_ldes
        add_input_lfp, add_input_ldes = _allocate_residual_proportional(
            residual_curt, extra_input_lfp, extra_input_ldes
        )
        used_lfp += add_input_lfp
        used_ldes += add_input_ldes

        ch_lfp = used_lfp * lfp_charge_eff
        ch_ldes = used_ldes * ldes_charge_eff
        soc_lfp += ch_lfp
        soc_ldes += ch_ldes
        spill = curt - used_lfp - used_ldes
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


def simulate_hybrid_steady_discharge_ratio(
    demand_arr, curtailment_arr, cw_arr, dw_arr,
    lfp_capacity_kwh, lfp_charge_pwr_kw, lfp_discharge_pwr_kw,
    lfp_charge_eff, lfp_discharge_eff, lfp_self_disch,
    ldes_capacity_kwh, ldes_charge_pwr_kw, ldes_discharge_pwr_kw,
    ldes_charge_eff, ldes_discharge_eff, ldes_self_disch,
    soc_min_frac, soc_max_frac,
    lfp_discharge_share,
    quota_mode,
    max_iter=15,
    conv_ratio=0.001,
):
    last = None
    for _ in range(max_iter):
        last = simulate_hybrid_year_discharge_ratio(
            demand_arr, curtailment_arr, cw_arr, dw_arr,
            lfp_capacity_kwh, lfp_charge_pwr_kw, lfp_discharge_pwr_kw,
            lfp_charge_eff, lfp_discharge_eff, lfp_self_disch,
            ldes_capacity_kwh, ldes_charge_pwr_kw, ldes_discharge_pwr_kw,
            ldes_charge_eff, ldes_discharge_eff, ldes_self_disch,
            soc_min_frac, soc_max_frac,
            lfp_discharge_share,
            quota_mode,
        )
    return last
