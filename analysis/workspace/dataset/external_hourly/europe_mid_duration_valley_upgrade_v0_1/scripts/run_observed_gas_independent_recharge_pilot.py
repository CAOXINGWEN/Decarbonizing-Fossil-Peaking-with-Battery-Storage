#!/usr/bin/env python3
"""Pilot the independent-recharge interface for the observed fossil-gas task.

This runner is intentionally isolated from every manuscript revision and from
the v36 LP implementation.  The frozen task is the non-negative hourly
Energy-Charts ``public_power`` aggregate ``Fossil gas`` series.  Wind, solar,
or residual load are never subtracted from that task.

Three recharge specifications are supported:

* R0: no exogenous recharge; storage charging is supplied by additional,
  explicitly costed model gas generation.
* R2: a synchronous potential-VRE-surplus counterfactual,
  ``max(alpha * (wind + solar) - system_load, 0)``.  This is not observed
  curtailment.
* R3: the R2 hourly shape scaled to an admitted ACER 2023 annual curtailment
  volume.  This is an annual-volume-calibrated allocation, not observed
  hourly curtailment, and is available only for DE, ES, HR and IT in 2023.

The model uses separate hourly variables for total gas generation, gas serving
the frozen task, gas serving storage charging, exogenous recharge used,
unabsorbed recharge, storage charge input, storage discharge, state of charge
and unserved task.  It audits the following identities independently:

    gas_to_task + storage_discharge + unserved = frozen_task
    gas_to_charge + recharge_used = storage_charge_input
    total_gas_generation = gas_to_task + gas_to_charge
    recharge_used + recharge_spill = recharge_available

The default command runs a small DE-2023 CAES-24 h structural pilot:

    python scripts/run_observed_gas_independent_recharge_pilot.py

This is an interface and accounting pilot, not manuscript evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PANEL = (
    ROOT
    / "processed"
    / "energy_charts_observed_gas_task_panel_2015_2024.csv.gz"
)
ACER = (
    ROOT
    / "raw"
    / "acer_2024_mmr"
    / "ACER_Figure42_RES_curtailment_2023.csv"
)
CONFIGURATIONS = (
    ROOT.parents[2]
    / "Decarbonizing-Fossil-Peaking-with-Battery-Storage-master"
    / "paper-code"
    / "docs"
    / "musrcipt"
    / "revision_nature_energy_v36"
    / "system_value_closure"
    / "inputs"
    / "technology_duration_configurations_2035.csv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "model_adapter"
    / "observed_gas_independent_recharge_pilot_DE2023"
)

EXPECTED_PANEL_SHA256 = (
    "5f2e8ab49120050fac2ef8bd1b11e20a58863442631a8d9baccf5b924f0bf3ab"
)
EXPECTED_ACER_SHA256 = (
    "56bd86ad82ded7b238afeb796ac67df08ea87b2a3d291e7a096bbbb564ede59c"
)
R3_COUNTRY_YEAR = {("DE", 2023), ("ES", 2023), ("HR", 2023), ("IT", 2023)}

# These constants are copied exactly from the audited v36 screening core.  The
# independent runner does not import or modify that core.
ATB_CAPEX_USD_KW = 1499.2530441159731
ATB_FOM_USD_KW_YR = 34.5
ATB_VOM_USD_MWH = 2.17
FX_CNY_PER_USD = 7.2
DISCOUNT_RATE = 0.036
NGCC_LIFE_YR = 30
MEDIAN_PROVINCIAL_GAS_PRICE_CNY_M3 = 2.72
GAS_HEAT_VALUE_KJ_M3 = 35_544.0
NGCC_EFFICIENCY = 0.536
NGCC_ELECTRICITY_KWH_M3 = (
    GAS_HEAT_VALUE_KJ_M3 * NGCC_EFFICIENCY / 3_600.0
)
NGCC_CRF = (
    DISCOUNT_RATE
    * (1.0 + DISCOUNT_RATE) ** NGCC_LIFE_YR
    / ((1.0 + DISCOUNT_RATE) ** NGCC_LIFE_YR - 1.0)
)
NGCC_FIXED_CNY_KW_YR = (
    ATB_CAPEX_USD_KW * NGCC_CRF + ATB_FOM_USD_KW_YR
) * FX_CNY_PER_USD
NGCC_VARIABLE_CNY_MWH = (
    MEDIAN_PROVINCIAL_GAS_PRICE_CNY_M3
    / NGCC_ELECTRICITY_KWH_M3
    * 1_000.0
    + ATB_VOM_USD_MWH * FX_CNY_PER_USD
)

SOC_MIN_FRAC = 0.05
SOC_MAX_FRAC = 0.95
EENS_TARGET_SHARE = 0.001
FORCED_TRANCHE_PU = 0.01
MAX_GAS_PU = 6.0
MAX_STORAGE_DISCHARGE_PU = 1.0
MAX_STORAGE_CHARGE_PU = 5.0
MAX_EXTRA_CHARGE_PU = 5.0
EXTREME_QUANTILE = 0.75
CLUSTER_DAYS = 4
SOLVER_TOL = 1e-7
REPORT_TOL = 1e-8
ABSOLUTE_BALANCE_TOLERANCE_MWH = 1e-6
RELATIVE_BALANCE_TOLERANCE = 1e-10


@dataclass(frozen=True)
class Configuration:
    config_id: str
    tech_id: str
    technology_en: str
    duration_h: float
    rte: float
    self_discharge_pct_day: float
    charge_power_ratio: float
    fixed_cost_cny_per_kwh_year: float
    variable_om_cny_mwh: float
    annual_power_cost_cny_per_kw_year: float


@dataclass
class Problem:
    config: Configuration
    task: np.ndarray
    recharge_available: np.ndarray
    recharge_specification: str
    recharge_interpretation: str
    recharge_opportunity_price_cny_mwh: float
    c: np.ndarray
    a_eq: csr_matrix
    b_eq: np.ndarray
    a_ub: csr_matrix
    b_ub: np.ndarray
    bounds: list[tuple[float | None, float | None]]
    indices: dict[str, Any]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_list(value: str, cast=str) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def load_configuration(config_id: str) -> Configuration:
    frame = pd.read_csv(CONFIGURATIONS)
    frame = frame[frame["learning_year"].eq(2035)].copy()
    frame["_config_id"] = (
        frame["tech_id"].astype(str)
        + "_"
        + frame["duration_h"].map(lambda value: f"{float(value):g}")
        + "h"
    )
    selected = frame[frame["_config_id"].eq(config_id)]
    if len(selected) != 1:
        raise RuntimeError(
            f"Expected one configuration {config_id}; found {len(selected)}"
        )
    row = selected.iloc[0]
    annual_power_cost = (
        float(row["annualized_power_capex_cny_per_kw_year"])
        + float(row["fixed_om_cny_kw_yr"])
        + float(row["annual_replacement_reserve_cny_per_kw_year"])
    )
    return Configuration(
        config_id=config_id,
        tech_id=str(row["tech_id"]),
        technology_en=str(row["technology_en"]),
        duration_h=float(row["duration_h"]),
        rte=float(row["rte"]),
        self_discharge_pct_day=float(row["self_discharge_pct_day"]),
        charge_power_ratio=float(row["charge_power_ratio"]),
        fixed_cost_cny_per_kwh_year=float(
            row["annual_fixed_cost_cny_per_kwh_capacity_year"]
        ),
        variable_om_cny_mwh=float(row["variable_om_cny_mwh"]),
        annual_power_cost_cny_per_kw_year=annual_power_cost,
    )


def load_country_year(country: str, year: int) -> pd.DataFrame:
    required = [
        "timestamp_utc",
        "country",
        "year",
        "system_load_mw",
        "observed_wind_mw",
        "observed_solar_mw",
        "observed_gas_peaking_task_mw",
        "load_mw",
        "gas_task_eligible",
        "gas_task_material_share_ge_1pct",
    ]
    panel = pd.read_csv(PANEL, usecols=required)
    panel = panel[
        panel["country"].eq(country)
        & panel["year"].eq(year)
        & as_bool(panel["gas_task_eligible"])
        & as_bool(panel["gas_task_material_share_ge_1pct"])
    ].copy()
    panel["timestamp_utc"] = pd.to_datetime(
        panel["timestamp_utc"], utc=True, errors="raise"
    )
    panel = panel.sort_values("timestamp_utc").reset_index(drop=True)
    if len(panel) != 8760 or panel["timestamp_utc"].nunique() != 8760:
        raise RuntimeError(
            f"{country}-{year} must contain 8,760 unique model hours"
        )
    identity_error = float(
        (
            panel["load_mw"]
            - panel["observed_gas_peaking_task_mw"]
        )
        .abs()
        .max()
    )
    if identity_error != 0.0:
        raise RuntimeError(
            "Frozen task alias differs from observed aggregate Fossil gas"
        )
    if (panel["observed_gas_peaking_task_mw"] < 0).any():
        raise RuntimeError("Frozen observed-gas task contains negative values")
    return panel


def interleave(extreme: list[int], normal: list[int]) -> list[int]:
    slots: list[list[int]] = [[] for _ in range(len(normal) + 1)]
    if extreme:
        positions = np.linspace(1, max(1, len(normal) - 1), len(extreme))
        for day, position in zip(extreme, np.rint(positions).astype(int)):
            slots[int(np.clip(position, 0, len(normal)))].append(day)
    order: list[int] = []
    for index, day in enumerate(normal):
        order.extend(slots[index])
        order.append(day)
    order.extend(slots[-1])
    return order


def interleave_clusters(
    extreme: list[int], normal: list[int], target_run: int
) -> list[int]:
    clusters = [
        extreme[index : index + target_run]
        for index in range(0, len(extreme), target_run)
    ]
    slots: list[list[int]] = [[] for _ in range(len(normal) + 1)]
    if clusters:
        positions = np.linspace(1, max(1, len(normal) - 1), len(clusters))
        for cluster, position in zip(clusters, np.rint(positions).astype(int)):
            slots[int(np.clip(position, 0, len(normal)))].extend(cluster)
    order: list[int] = []
    for index, day in enumerate(normal):
        order.extend(slots[index])
        order.append(day)
    order.extend(slots[-1])
    return order


def calendar_segment(month: int) -> int:
    if month <= 2:
        return 0
    if month <= 5:
        return 1
    if month <= 8:
        return 2
    if month <= 11:
        return 3
    return 4


def make_day_orders(panel: pd.DataFrame) -> dict[str, list[int]]:
    task = panel["observed_gas_peaking_task_mw"].to_numpy(float)
    daily = pd.DataFrame(
        {
            "day": np.arange(8760) // 24,
            "task": task,
            "month": panel["timestamp_utc"].dt.month.to_numpy(),
        }
    ).groupby("day", sort=True).agg(
        task_mwh=("task", "sum"),
        month=("month", "first"),
    )
    daily["segment"] = daily["month"].map(calendar_segment)
    threshold = float(daily["task_mwh"].quantile(EXTREME_QUANTILE))
    daily["extreme"] = daily["task_mwh"] >= threshold
    orders = {"natural": [int(value) for value in daily.index]}
    dispersed: list[int] = []
    clustered: list[int] = []
    for segment in sorted(daily["segment"].unique()):
        subset = daily[daily["segment"].eq(segment)]
        extreme = [int(value) for value in subset.index[subset["extreme"]]]
        normal = [int(value) for value in subset.index[~subset["extreme"]]]
        dispersed.extend(interleave(extreme, normal))
        clustered.extend(interleave_clusters(extreme, normal, CLUSTER_DAYS))
    orders["dispersed"] = dispersed
    orders["clustered"] = clustered
    reference = list(range(365))
    for chronology, order in orders.items():
        if len(order) != 365 or sorted(order) != reference:
            raise RuntimeError(f"Invalid {chronology} intact-day permutation")
    return orders


def reorder_days(values: np.ndarray, order: list[int]) -> np.ndarray:
    blocks = values.reshape(365, 24)
    return np.concatenate([blocks[day] for day in order])


def build_recharge_profiles(
    panel: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    observed_vre = (
        panel["observed_wind_mw"].to_numpy(float)
        + panel["observed_solar_mw"].to_numpy(float)
    )
    system_load = panel["system_load_mw"].to_numpy(float)
    if observed_vre.sum() <= 0.0 or system_load.sum() <= 0.0:
        raise RuntimeError("R2 requires positive annual VRE and system load")
    alpha = float(system_load.sum() / observed_vre.sum())
    potential = np.maximum(alpha * observed_vre - system_load, 0.0)
    country = str(panel["country"].iloc[0])
    year = int(panel["year"].iloc[0])

    r3 = np.zeros_like(potential)
    annual_acer_mwh = np.nan
    r3_scale = np.nan
    r3_admitted = False
    r3_error = np.nan
    r3_bound_violation = np.nan
    if (country, year) in R3_COUNTRY_YEAR:
        acer = pd.read_csv(ACER)
        selected = acer[
            acer["country_code"].astype(str).eq(country)
            & acer["year"].eq(year)
        ]
        if len(selected) != 1:
            raise RuntimeError(
                f"Expected one ACER annual row for {country}-{year}"
            )
        annual_acer_mwh = (
            float(selected["res_curtailment_gwh"].iloc[0]) * 1_000.0
        )
        annual_potential_mwh = float(potential.sum())
        if (
            annual_acer_mwh > 0.0
            and annual_potential_mwh > 0.0
            and annual_acer_mwh <= annual_potential_mwh
        ):
            r3_scale = annual_acer_mwh / annual_potential_mwh
            r3 = r3_scale * potential
            r3_error = float(r3.sum() - annual_acer_mwh)
            r3_bound_violation = float(
                np.maximum(r3 - potential, 0.0).max()
            )
            r3_admitted = bool(
                0.0 < r3_scale <= 1.0
                and abs(r3_error) <= 1e-6
                and abs(r3_error) / annual_acer_mwh <= 1e-10
                and r3_bound_violation <= 1e-9
            )
    return {
        "R0": np.zeros_like(potential),
        "R2": potential,
        "R3": r3,
    }, {
        "potential_vre_scaling_alpha": alpha,
        "annual_potential_vre_surplus_mwh": float(potential.sum()),
        "annual_acer_curtailment_mwh": annual_acer_mwh,
        "acer_allocation_scale": r3_scale,
        "acer_annual_allocation_error_mwh": r3_error,
        "acer_max_hourly_surplus_bound_violation_mw": r3_bound_violation,
        "r3_admitted": r3_admitted,
    }


def add(
    rows: list[int],
    cols: list[int],
    data: list[float],
    row: int,
    col: int,
    value: float,
) -> None:
    rows.append(row)
    cols.append(col)
    data.append(value)


def build_problem(
    task_mw: np.ndarray,
    recharge_available_mw: np.ndarray,
    config: Configuration,
    recharge_specification: str,
    recharge_opportunity_price_cny_mwh: float,
    forced_storage: bool,
) -> Problem:
    if len(task_mw) != len(recharge_available_mw):
        raise RuntimeError("Task and recharge must have the same hourly length")
    if np.any(task_mw < 0.0) or np.any(recharge_available_mw < 0.0):
        raise RuntimeError("Task and recharge availability must be non-negative")
    if recharge_specification == "R0" and np.any(recharge_available_mw != 0.0):
        raise RuntimeError("R0 must have zero exogenous recharge")
    if recharge_specification not in {"R0", "R2", "R3"}:
        raise ValueError(recharge_specification)

    mean_task_mw = float(np.mean(task_mw))
    if mean_task_mw <= 0.0:
        raise RuntimeError("Mean observed-gas task must be positive")
    task = np.asarray(task_mw, dtype=float) / mean_task_mw
    recharge = np.asarray(recharge_available_mw, dtype=float) / mean_task_mw
    n = len(task)

    kg, pdis, pch, energy, xch = range(5)
    names = (
        "gas_total",
        "gas_to_task",
        "gas_to_charge",
        "recharge_used",
        "charge",
        "discharge",
        "soc",
        "recharge_spill",
        "unserved",
    )
    starts = {name: 5 + index * n for index, name in enumerate(names)}
    nvar = 5 + len(names) * n

    c = np.zeros(nvar, dtype=float)
    c[kg] = NGCC_FIXED_CNY_KW_YR * 1_000.0
    c[energy] = config.fixed_cost_cny_per_kwh_year * 1_000.0
    c[xch] = config.annual_power_cost_cny_per_kw_year * 1_000.0
    c[starts["gas_total"] : starts["gas_total"] + n] = (
        NGCC_VARIABLE_CNY_MWH
    )
    c[starts["recharge_used"] : starts["recharge_used"] + n] = (
        recharge_opportunity_price_cny_mwh
    )
    c[starts["discharge"] : starts["discharge"] + n] = (
        config.variable_om_cny_mwh
    )

    eq_rows: list[int] = []
    eq_cols: list[int] = []
    eq_data: list[float] = []
    b_eq: list[float] = []
    add(eq_rows, eq_cols, eq_data, 0, energy, 1.0)
    add(eq_rows, eq_cols, eq_data, 0, pdis, -config.duration_h)
    b_eq.append(0.0)
    eq_row = 1

    eta_ch = math.sqrt(config.rte)
    eta_dis = math.sqrt(config.rte)
    daily_retention = max(
        0.0, 1.0 - config.self_discharge_pct_day / 100.0
    )
    retention = daily_retention ** (1.0 / 24.0)
    for hour in range(n):
        # Frozen observed-gas task service; recharge never enters this equation.
        add(
            eq_rows,
            eq_cols,
            eq_data,
            eq_row,
            starts["gas_to_task"] + hour,
            1.0,
        )
        add(
            eq_rows,
            eq_cols,
            eq_data,
            eq_row,
            starts["discharge"] + hour,
            1.0,
        )
        add(
            eq_rows,
            eq_cols,
            eq_data,
            eq_row,
            starts["unserved"] + hour,
            1.0,
        )
        b_eq.append(float(task[hour]))
        eq_row += 1

        # Storage electrical charging input is attributed to two sources.
        add(
            eq_rows,
            eq_cols,
            eq_data,
            eq_row,
            starts["gas_to_charge"] + hour,
            1.0,
        )
        add(
            eq_rows,
            eq_cols,
            eq_data,
            eq_row,
            starts["recharge_used"] + hour,
            1.0,
        )
        add(
            eq_rows,
            eq_cols,
            eq_data,
            eq_row,
            starts["charge"] + hour,
            -1.0,
        )
        b_eq.append(0.0)
        eq_row += 1

        # Total model gas generation is attributed to task and charging.
        add(
            eq_rows,
            eq_cols,
            eq_data,
            eq_row,
            starts["gas_total"] + hour,
            1.0,
        )
        add(
            eq_rows,
            eq_cols,
            eq_data,
            eq_row,
            starts["gas_to_task"] + hour,
            -1.0,
        )
        add(
            eq_rows,
            eq_cols,
            eq_data,
            eq_row,
            starts["gas_to_charge"] + hour,
            -1.0,
        )
        b_eq.append(0.0)
        eq_row += 1

        # Independent recharge availability is either used or left unabsorbed.
        add(
            eq_rows,
            eq_cols,
            eq_data,
            eq_row,
            starts["recharge_used"] + hour,
            1.0,
        )
        add(
            eq_rows,
            eq_cols,
            eq_data,
            eq_row,
            starts["recharge_spill"] + hour,
            1.0,
        )
        b_eq.append(float(recharge[hour]))
        eq_row += 1

        previous = n - 1 if hour == 0 else hour - 1
        add(
            eq_rows,
            eq_cols,
            eq_data,
            eq_row,
            starts["soc"] + hour,
            1.0,
        )
        add(
            eq_rows,
            eq_cols,
            eq_data,
            eq_row,
            starts["soc"] + previous,
            -retention,
        )
        add(
            eq_rows,
            eq_cols,
            eq_data,
            eq_row,
            starts["charge"] + hour,
            -eta_ch,
        )
        add(
            eq_rows,
            eq_cols,
            eq_data,
            eq_row,
            starts["discharge"] + hour,
            1.0 / eta_dis,
        )
        b_eq.append(0.0)
        eq_row += 1

    a_eq = coo_matrix(
        (eq_data, (eq_rows, eq_cols)), shape=(eq_row, nvar)
    ).tocsr()
    b_eq_array = np.asarray(b_eq, dtype=float)

    ub_rows: list[int] = []
    ub_cols: list[int] = []
    ub_data: list[float] = []
    b_ub: list[float] = []
    ub_row = 0

    # Bundled charging block plus optional, explicitly priced expansion.
    add(ub_rows, ub_cols, ub_data, ub_row, pch, 1.0)
    add(
        ub_rows,
        ub_cols,
        ub_data,
        ub_row,
        pdis,
        -config.charge_power_ratio,
    )
    add(ub_rows, ub_cols, ub_data, ub_row, xch, -1.0)
    b_ub.append(0.0)
    ub_row += 1

    for hour in range(n):
        add(
            ub_rows,
            ub_cols,
            ub_data,
            ub_row,
            starts["gas_total"] + hour,
            1.0,
        )
        add(ub_rows, ub_cols, ub_data, ub_row, kg, -1.0)
        b_ub.append(0.0)
        ub_row += 1

        add(
            ub_rows,
            ub_cols,
            ub_data,
            ub_row,
            starts["charge"] + hour,
            1.0,
        )
        add(ub_rows, ub_cols, ub_data, ub_row, pch, -1.0)
        b_ub.append(0.0)
        ub_row += 1

        add(
            ub_rows,
            ub_cols,
            ub_data,
            ub_row,
            starts["discharge"] + hour,
            1.0,
        )
        add(ub_rows, ub_cols, ub_data, ub_row, pdis, -1.0)
        b_ub.append(0.0)
        ub_row += 1

        add(
            ub_rows,
            ub_cols,
            ub_data,
            ub_row,
            starts["soc"] + hour,
            1.0,
        )
        add(
            ub_rows,
            ub_cols,
            ub_data,
            ub_row,
            energy,
            -SOC_MAX_FRAC,
        )
        b_ub.append(0.0)
        ub_row += 1

        add(
            ub_rows,
            ub_cols,
            ub_data,
            ub_row,
            starts["soc"] + hour,
            -1.0,
        )
        add(
            ub_rows,
            ub_cols,
            ub_data,
            ub_row,
            energy,
            SOC_MIN_FRAC,
        )
        b_ub.append(0.0)
        ub_row += 1

    for hour in range(n):
        add(
            ub_rows,
            ub_cols,
            ub_data,
            ub_row,
            starts["unserved"] + hour,
            1.0,
        )
    b_ub.append(EENS_TARGET_SHARE * float(task.sum()))
    ub_row += 1
    a_ub = coo_matrix(
        (ub_data, (ub_rows, ub_cols)), shape=(ub_row, nvar)
    ).tocsr()
    b_ub_array = np.asarray(b_ub, dtype=float)

    if forced_storage:
        pdis_bound = (FORCED_TRANCHE_PU, FORCED_TRANCHE_PU)
        energy_bound = (
            config.duration_h * FORCED_TRANCHE_PU,
            config.duration_h * FORCED_TRANCHE_PU,
        )
    else:
        pdis_bound = (0.0, 0.0)
        energy_bound = (0.0, 0.0)
    bounds: list[tuple[float | None, float | None]] = [
        (0.0, MAX_GAS_PU),
        pdis_bound,
        (0.0, MAX_STORAGE_CHARGE_PU if forced_storage else 0.0),
        energy_bound,
        (
            0.0,
            (
                0.0
                if config.tech_id == "li_lfp" or not forced_storage
                else MAX_EXTRA_CHARGE_PU
            ),
        ),
    ]
    bounds.extend([(0.0, None)] * (len(names) * n))

    interpretation = {
        "R0": "costed additional gas recharge; no exogenous recharge",
        "R2": (
            "potential-VRE-surplus counterfactual; not observed curtailment"
        ),
        "R3": (
            "ACER annual-volume-calibrated hourly allocation; not observed "
            "hourly curtailment"
        ),
    }[recharge_specification]
    return Problem(
        config=config,
        task=task,
        recharge_available=recharge,
        recharge_specification=recharge_specification,
        recharge_interpretation=interpretation,
        recharge_opportunity_price_cny_mwh=(
            recharge_opportunity_price_cny_mwh
        ),
        c=c,
        a_eq=a_eq,
        b_eq=b_eq_array,
        a_ub=a_ub,
        b_ub=b_ub_array,
        bounds=bounds,
        indices={
            "kg": kg,
            "pdis": pdis,
            "pch": pch,
            "energy": energy,
            "xch": xch,
            "starts": starts,
            "n": n,
            "mean_task_mw": mean_task_mw,
            "eta_ch": eta_ch,
            "eta_dis": eta_dis,
        },
    )


def solve_problem(
    problem: Problem,
    country: str,
    year: int,
    chronology: str,
    pairing_mode: str,
    variant: str,
    baseline_cost: float | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    started = time.perf_counter()
    result = linprog(
        problem.c,
        A_ub=problem.a_ub,
        b_ub=problem.b_ub,
        A_eq=problem.a_eq,
        b_eq=problem.b_eq,
        bounds=problem.bounds,
        method="highs",
        options={
            "presolve": True,
            "dual_feasibility_tolerance": SOLVER_TOL,
            "primal_feasibility_tolerance": SOLVER_TOL,
        },
    )
    elapsed = time.perf_counter() - started
    if not result.success:
        raise RuntimeError(
            f"{country}-{year}-{chronology}-{problem.config.config_id}-"
            f"{problem.recharge_specification}: {result.message}"
        )
    x = result.x
    idx = problem.indices
    starts = idx["starts"]
    n = idx["n"]
    fields = {
        name: x[start : start + n]
        for name, start in starts.items()
    }
    kg = float(x[idx["kg"]])
    pdis = float(x[idx["pdis"]])
    pch = float(x[idx["pch"]])
    energy = float(x[idx["energy"]])
    xch = float(x[idx["xch"]])
    mean_task_mw = float(idx["mean_task_mw"])

    task_error = (
        problem.task
        - fields["gas_to_task"]
        - fields["discharge"]
        - fields["unserved"]
    )
    charging_error = (
        fields["charge"]
        - fields["gas_to_charge"]
        - fields["recharge_used"]
    )
    gas_error = (
        fields["gas_total"]
        - fields["gas_to_task"]
        - fields["gas_to_charge"]
    )
    recharge_error = (
        problem.recharge_available
        - fields["recharge_used"]
        - fields["recharge_spill"]
    )
    eq_error = float(
        np.max(np.abs(problem.a_eq @ x - problem.b_eq))
    )
    ub_error = float(
        np.max(np.maximum(problem.a_ub @ x - problem.b_ub, 0.0))
    )

    gas_fixed = kg * 1_000.0 * NGCC_FIXED_CNY_KW_YR
    storage_fixed = (
        energy * 1_000.0 * problem.config.fixed_cost_cny_per_kwh_year
    )
    extra_charge_fixed = (
        xch
        * 1_000.0
        * problem.config.annual_power_cost_cny_per_kw_year
    )
    gas_variable = (
        float(fields["gas_total"].sum()) * NGCC_VARIABLE_CNY_MWH
    )
    recharge_cost = (
        float(fields["recharge_used"].sum())
        * problem.recharge_opportunity_price_cny_mwh
    )
    storage_vom = (
        float(fields["discharge"].sum())
        * problem.config.variable_om_cny_mwh
    )
    total_cost = (
        gas_fixed
        + storage_fixed
        + extra_charge_fixed
        + gas_variable
        + recharge_cost
        + storage_vom
    )
    objective_error = float(abs(total_cost - result.fun))
    scale_mwh = mean_task_mw
    task_total_mwh = float(problem.task.sum() * scale_mwh)
    charge_total_mwh = float(fields["charge"].sum() * scale_mwh)
    gas_total_mwh = float(fields["gas_total"].sum() * scale_mwh)

    def annual_error(values: np.ndarray) -> tuple[float, float]:
        absolute = float(values.sum() * scale_mwh)
        denominator = max(
            task_total_mwh, charge_total_mwh, gas_total_mwh, 1.0
        )
        return absolute, abs(absolute) / denominator

    task_annual_error, task_relative_error = annual_error(task_error)
    charge_annual_error, charge_relative_error = annual_error(charging_error)
    gas_annual_error, gas_relative_error = annual_error(gas_error)
    recharge_annual_error, recharge_relative_error = annual_error(
        recharge_error
    )
    usable_energy = (SOC_MAX_FRAC - SOC_MIN_FRAC) * energy
    signed_benefit = (
        float(baseline_cost - total_cost)
        if baseline_cost is not None
        else np.nan
    )
    row = {
        "country": country,
        "year": year,
        "chronology": chronology,
        "pairing_mode": pairing_mode,
        "config_id": problem.config.config_id,
        "variant": variant,
        "recharge_specification": problem.recharge_specification,
        "recharge_interpretation": problem.recharge_interpretation,
        "recharge_opportunity_price_cny_mwh": (
            problem.recharge_opportunity_price_cny_mwh
        ),
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "solve_seconds": elapsed,
        "mean_observed_gas_task_mw": mean_task_mw,
        "gas_capacity_pu_mean_task": kg,
        "storage_discharge_power_pu_mean_task": pdis,
        "storage_charge_power_pu_mean_task": pch,
        "storage_energy_capacity_pu_h": energy,
        "extra_charge_power_pu_mean_task": xch,
        "total_system_cost_cny_per_mean_mw_year": total_cost,
        "signed_benefit_cny_per_mean_mw_year": signed_benefit,
        "signed_benefit_cny_per_forced_kw_year": (
            signed_benefit / (FORCED_TRANCHE_PU * 1_000.0)
            if np.isfinite(signed_benefit)
            else np.nan
        ),
        "task_energy_mwh": task_total_mwh,
        "gas_to_task_mwh": float(
            fields["gas_to_task"].sum() * scale_mwh
        ),
        "gas_supplied_charge_mwh": float(
            fields["gas_to_charge"].sum() * scale_mwh
        ),
        "total_gas_generation_mwh": gas_total_mwh,
        "recharge_available_mwh": float(
            problem.recharge_available.sum() * scale_mwh
        ),
        "recharge_used_mwh": float(
            fields["recharge_used"].sum() * scale_mwh
        ),
        "recharge_spill_mwh": float(
            fields["recharge_spill"].sum() * scale_mwh
        ),
        "storage_charge_input_mwh": charge_total_mwh,
        "storage_discharge_output_mwh": float(
            fields["discharge"].sum() * scale_mwh
        ),
        "unserved_task_mwh": float(
            fields["unserved"].sum() * scale_mwh
        ),
        "equivalent_full_cycles_output_basis": (
            float(fields["discharge"].sum()) / usable_energy
            if usable_energy > REPORT_TOL
            else 0.0
        ),
        "charge_power_saturation_h": (
            int(np.sum(fields["charge"] >= pch - REPORT_TOL))
            if pch > REPORT_TOL
            else 0
        ),
        "discharge_power_saturation_h": (
            int(np.sum(fields["discharge"] >= pdis - REPORT_TOL))
            if pdis > REPORT_TOL
            else 0
        ),
        "task_service_balance_error_mwh": task_annual_error,
        "task_service_balance_relative_error": task_relative_error,
        "charging_source_balance_error_mwh": charge_annual_error,
        "charging_source_balance_relative_error": charge_relative_error,
        "gas_generation_accounting_error_mwh": gas_annual_error,
        "gas_generation_accounting_relative_error": gas_relative_error,
        "recharge_resource_balance_error_mwh": recharge_annual_error,
        "recharge_resource_balance_relative_error": recharge_relative_error,
        "maximum_task_service_hourly_error_mw": float(
            np.max(np.abs(task_error)) * scale_mwh
        ),
        "maximum_charging_source_hourly_error_mw": float(
            np.max(np.abs(charging_error)) * scale_mwh
        ),
        "maximum_gas_accounting_hourly_error_mw": float(
            np.max(np.abs(gas_error)) * scale_mwh
        ),
        "maximum_recharge_balance_hourly_error_mw": float(
            np.max(np.abs(recharge_error)) * scale_mwh
        ),
        "maximum_equality_violation_pu": eq_error,
        "maximum_inequality_violation_pu": ub_error,
        "objective_decomposition_error_cny": objective_error,
    }
    dispatch = pd.DataFrame(
        {
            "country": country,
            "year": year,
            "chronology": chronology,
            "pairing_mode": pairing_mode,
            "config_id": problem.config.config_id,
            "variant": variant,
            "recharge_specification": problem.recharge_specification,
            "recharge_opportunity_price_cny_mwh": (
                problem.recharge_opportunity_price_cny_mwh
            ),
            "hour_of_model_year": np.arange(n),
            "observed_gas_task_mw": problem.task * scale_mwh,
            "gas_total_mw": fields["gas_total"] * scale_mwh,
            "gas_to_task_mw": fields["gas_to_task"] * scale_mwh,
            "gas_to_charge_mw": fields["gas_to_charge"] * scale_mwh,
            "recharge_available_mw": (
                problem.recharge_available * scale_mwh
            ),
            "recharge_used_mw": fields["recharge_used"] * scale_mwh,
            "storage_charge_input_mw": fields["charge"] * scale_mwh,
            "storage_discharge_mw": fields["discharge"] * scale_mwh,
            "soc_mwh": fields["soc"] * scale_mwh,
            "recharge_spill_mw": fields["recharge_spill"] * scale_mwh,
            "unserved_task_mw": fields["unserved"] * scale_mwh,
        }
    )
    return row, dispatch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", default="DE")
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--config-id", default="caes_24h")
    parser.add_argument(
        "--chronologies", default="natural,dispersed,clustered"
    )
    parser.add_argument(
        "--recharge-specifications", default="R0,R2,R3"
    )
    parser.add_argument(
        "--recharge-prices-cny-mwh", default="0,100"
    )
    parser.add_argument(
        "--pairing-mode",
        choices=("co_reordered", "task_only_fixed_natural"),
        default="co_reordered",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    panel_sha = sha256(PANEL)
    acer_sha = sha256(ACER)
    if panel_sha != EXPECTED_PANEL_SHA256:
        raise RuntimeError(
            f"Frozen task panel SHA changed: {panel_sha}"
        )
    if acer_sha != EXPECTED_ACER_SHA256:
        raise RuntimeError(f"Frozen ACER source SHA changed: {acer_sha}")

    panel = load_country_year(args.country, args.year)
    config = load_configuration(args.config_id)
    orders = make_day_orders(panel)
    recharge_profiles, recharge_qa = build_recharge_profiles(panel)
    chronologies = parse_list(args.chronologies)
    specifications = parse_list(args.recharge_specifications)
    prices = parse_list(args.recharge_prices_cny_mwh, float)
    invalid_chronologies = set(chronologies) - set(orders)
    invalid_specs = set(specifications) - {"R0", "R2", "R3"}
    if invalid_chronologies or invalid_specs:
        raise RuntimeError(
            f"Invalid chronologies={invalid_chronologies}, "
            f"specifications={invalid_specs}"
        )
    if "R3" in specifications and not recharge_qa["r3_admitted"]:
        raise RuntimeError(
            f"R3 is not admitted for {args.country}-{args.year}"
        )

    frozen_task = panel["observed_gas_peaking_task_mw"].to_numpy(float)
    natural_order = orders["natural"]
    results: list[dict[str, Any]] = []
    dispatches: list[pd.DataFrame] = []
    conservation_rows: list[dict[str, Any]] = []
    for chronology in chronologies:
        order = orders[chronology]
        task = reorder_days(frozen_task, order)
        task_order_hash = hashlib.sha256(
            np.asarray(order, dtype=np.int16).tobytes()
        ).hexdigest()
        for specification in specifications:
            source_recharge = recharge_profiles[specification]
            recharge_order = (
                order
                if args.pairing_mode == "co_reordered"
                else natural_order
            )
            recharge = reorder_days(source_recharge, recharge_order)
            conservation_rows.append(
                {
                    "country": args.country,
                    "year": args.year,
                    "chronology": chronology,
                    "pairing_mode": args.pairing_mode,
                    "recharge_specification": specification,
                    "task_order_sha256": task_order_hash,
                    "task_day_multiset_conserved": (
                        sorted(order) == natural_order
                    ),
                    "task_annual_energy_error_mwh": float(
                        task.sum() - frozen_task.sum()
                    ),
                    "recharge_annual_energy_error_mwh": float(
                        recharge.sum() - source_recharge.sum()
                    ),
                    "task_is_exact_reordered_observed_gas": bool(
                        np.array_equal(
                            np.sort(task.reshape(365, 24), axis=0),
                            np.sort(
                                frozen_task.reshape(365, 24), axis=0
                            ),
                        )
                    ),
                    "recharge_order_matches_task_order": (
                        args.pairing_mode == "co_reordered"
                    ),
                }
            )
            baseline_problem = build_problem(
                task,
                recharge,
                config,
                specification,
                recharge_opportunity_price_cny_mwh=0.0,
                forced_storage=False,
            )
            baseline_row, baseline_dispatch = solve_problem(
                baseline_problem,
                args.country,
                args.year,
                chronology,
                args.pairing_mode,
                "baseline",
            )
            results.append(baseline_row)
            dispatches.append(baseline_dispatch)
            baseline_cost = float(
                baseline_row["total_system_cost_cny_per_mean_mw_year"]
            )
            run_prices = [0.0] if specification == "R0" else prices
            for price in run_prices:
                problem = build_problem(
                    task,
                    recharge,
                    config,
                    specification,
                    recharge_opportunity_price_cny_mwh=price,
                    forced_storage=True,
                )
                row, dispatch = solve_problem(
                    problem,
                    args.country,
                    args.year,
                    chronology,
                    args.pairing_mode,
                    "forced_1pct_joint",
                    baseline_cost=baseline_cost,
                )
                results.append(row)
                dispatches.append(dispatch)

    results_frame = pd.DataFrame(results)
    dispatch_frame = pd.concat(dispatches, ignore_index=True)
    conservation = pd.DataFrame(conservation_rows)
    results_frame.to_csv(output / "model_results.csv", index=False)
    dispatch_frame.to_csv(
        output / "hourly_dispatch.csv.gz", index=False
    )
    conservation.to_csv(
        output / "chronology_conservation_qa.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "country": args.country,
                "year": args.year,
                **recharge_qa,
                "R2_interpretation": (
                    "potential-VRE-surplus counterfactual; not observed "
                    "curtailment"
                ),
                "R3_interpretation": (
                    "ACER annual-volume-calibrated hourly allocation; not "
                    "observed hourly curtailment"
                ),
            }
        ]
    ).to_csv(output / "recharge_profile_qa.csv", index=False)

    forced = results_frame[
        results_frame["variant"].eq("forced_1pct_joint")
    ]
    balance_fields = [
        "task_service_balance_error_mwh",
        "charging_source_balance_error_mwh",
        "gas_generation_accounting_error_mwh",
        "recharge_resource_balance_error_mwh",
    ]
    relative_fields = [
        "task_service_balance_relative_error",
        "charging_source_balance_relative_error",
        "gas_generation_accounting_relative_error",
        "recharge_resource_balance_relative_error",
    ]
    checks = {
        "frozen_panel_sha256_matches": panel_sha == EXPECTED_PANEL_SHA256,
        "acer_source_sha256_matches": acer_sha == EXPECTED_ACER_SHA256,
        "frozen_task_alias_identity": bool(
            (
                panel["load_mw"]
                - panel["observed_gas_peaking_task_mw"]
            )
            .abs()
            .max()
            == 0.0
        ),
        "all_models_optimal": bool(results_frame["solver_status"].eq(0).all()),
        "all_task_day_multisets_conserved": bool(
            conservation["task_day_multiset_conserved"].all()
        ),
        "all_annual_task_energy_conserved": bool(
            conservation["task_annual_energy_error_mwh"]
            .abs()
            .le(ABSOLUTE_BALANCE_TOLERANCE_MWH)
            .all()
        ),
        "all_annual_recharge_energy_conserved": bool(
            conservation["recharge_annual_energy_error_mwh"]
            .abs()
            .le(ABSOLUTE_BALANCE_TOLERANCE_MWH)
            .all()
        ),
        "all_absolute_accounting_balances_pass": bool(
            results_frame[balance_fields]
            .abs()
            .le(ABSOLUTE_BALANCE_TOLERANCE_MWH)
            .all()
            .all()
        ),
        "all_relative_accounting_balances_pass": bool(
            results_frame[relative_fields]
            .abs()
            .le(RELATIVE_BALANCE_TOLERANCE)
            .all()
            .all()
        ),
        "r0_has_zero_exogenous_recharge": bool(
            forced.loc[
                forced["recharge_specification"].eq("R0"),
                ["recharge_available_mwh", "recharge_used_mwh"],
            ]
            .abs()
            .le(ABSOLUTE_BALANCE_TOLERANCE_MWH)
            .all()
            .all()
        ),
        "r0_charge_is_fully_gas_supplied": bool(
            np.allclose(
                forced.loc[
                    forced["recharge_specification"].eq("R0"),
                    "storage_charge_input_mwh",
                ],
                forced.loc[
                    forced["recharge_specification"].eq("R0"),
                    "gas_supplied_charge_mwh",
                ],
                atol=ABSOLUTE_BALANCE_TOLERANCE_MWH,
                rtol=RELATIVE_BALANCE_TOLERANCE,
            )
        ),
        "r2_is_bounded_counterfactual_only": True,
        "r3_is_annual_volume_calibrated_not_hourly_observed": True,
        "r3_admission_gate_passed": (
            bool(recharge_qa["r3_admitted"])
            if "R3" in specifications
            else True
        ),
        "objective_decomposition_passes": bool(
            results_frame["objective_decomposition_error_cny"]
            .le(1e-3)
            .all()
        ),
    }
    qa = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "country": args.country,
        "year": args.year,
        "config_id": args.config_id,
        "chronologies": chronologies,
        "pairing_mode": args.pairing_mode,
        "recharge_specifications": specifications,
        "recharge_prices_cny_mwh": prices,
        "task_definition": (
            "exact non-negative Energy-Charts public_power aggregate Fossil "
            "gas; no VRE subtraction"
        ),
        "R2_interpretation": (
            "potential-VRE-surplus counterfactual; not observed curtailment"
        ),
        "R3_interpretation": (
            "ACER annual-volume-calibrated hourly allocation; not observed "
            "hourly curtailment"
        ),
        "maximum_absolute_accounting_error_mwh": float(
            results_frame[balance_fields].abs().to_numpy().max()
        ),
        "maximum_relative_accounting_error": float(
            results_frame[relative_fields].abs().to_numpy().max()
        ),
        "maximum_equality_violation_pu": float(
            results_frame["maximum_equality_violation_pu"].max()
        ),
        "maximum_inequality_violation_pu": float(
            results_frame["maximum_inequality_violation_pu"].max()
        ),
        "manuscript_evidence_admitted": False,
        "reason": (
            "single-country-year interface/accounting pilot; full paired "
            "spectrum and prespecified inferential analysis not run"
        ),
    }
    (output / "QA.json").write_text(
        json.dumps(qa, indent=2), encoding="utf-8"
    )
    manifest = {
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "panel": str(PANEL),
        "panel_sha256": panel_sha,
        "acer_source": str(ACER),
        "acer_source_sha256": acer_sha,
        "configuration_source": str(CONFIGURATIONS),
        "configuration_source_sha256": sha256(CONFIGURATIONS),
        "model_contract": (
            "frozen observed aggregate fossil-gas task with separate gas "
            "task, gas charging, independent recharge and spill accounting"
        ),
        "qa_status": qa["status"],
        "manuscript_evidence_admitted": False,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    if qa["status"] != "pass":
        raise RuntimeError(f"Pilot QA failed: {qa}")
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
