#!/usr/bin/env python3
"""Stage-1 v0.2 smoke test for the observed-gas event-task experiment.

This runner is isolated from every previous observed-gas runner.  It consumes
the frozen Stage-0 event-task table and runs only two country-years by default:
DE-2023 and SI-2018.  The task is C3-q75, which is derived exclusively from
Energy-Charts actual aggregate ``Fossil gas``.  R2 potential VRE surplus is a
separate recharge input and is never subtracted from the task.

The storage model has:

* cyclic SOC and a common 5--95% SOC band;
* no gas-to-storage charging;
* exactly the same priced charge-power expansion rule for LFP and CAES;
* strict charge/discharge mutual exclusion;
* an exact McCormick strengthening for charge expansion times hourly mode;
* a sparse MILP with binary variables only in task/recharge overlap hours;
* selected 8,760-binary MILP validation runs;
* an endogenous, component-specific calendar-or-cycle LFP capital charge.

All outputs are non-evidential Stage-1 structural smoke tests.  This script
does not import, modify, or overwrite the previous R2/R3 runners or results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, csr_matrix


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STAGE0 = (
    ROOT
    / "processed"
    / "observed_gas_event_task_definitions_v0_1.csv.gz"
)
STAGE0_QA = (
    ROOT
    / "artifacts"
    / "research"
    / "OBSERVED_GAS_EVENT_TASK_STAGE0_QA.json"
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
    / "observed_gas_event_task_confirmatory_stage1_v0_2"
)

EXPECTED_STAGE0_SOURCE_PANEL_SHA256 = (
    "5f2e8ab49120050fac2ef8bd1b11e20a58863442631a8d9baccf5b924f0bf3ab"
)
EXPECTED_STAGE0_TABLE_SHA256 = (
    "2a901503a5a517723493b5bdfb3c26ee4ee8dc4b784d922bd26d419e8fd47d50"
)
EVIDENCE_STATUS = "non_evidential_stage1_structural_smoke"

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
FORCED_POWER_SHARE_OF_RAW_GAS_MEAN = 0.01
MAX_EXTRA_CHARGE_POWER_MULTIPLE = 5.0
RECHARGE_OPPORTUNITY_PRICE_CNY_MWH = 0.0
LFP_REPLACEABLE_FRACTION = 0.468106
LFP_REPLACEABLE_FRACTION_SENSITIVITY = 0.565037
PAIRED_SEED = "20260727-0"
CLUSTER_DAYS = 4

ABSOLUTE_BALANCE_TOLERANCE = 1e-6
RELATIVE_BALANCE_TOLERANCE = 1e-10
SIMULTANEOUS_HOURLY_TOLERANCE_MW = 1e-9
SIMULTANEOUS_RATIO_TOLERANCE = 1e-10
PWL_RELATIVE_GAP_TOLERANCE = 1e-4
FULL_BINARY_OBJECTIVE_RELATIVE_TOLERANCE = 1e-6
FULL_BINARY_AGGREGATE_RELATIVE_TOLERANCE = 1e-4
MIP_RELATIVE_GAP_TOLERANCE = 1e-5
MIP_ABSOLUTE_GAP_CNY_PER_FORCED_KW_YEAR = 0.01


@dataclass(frozen=True)
class Configuration:
    config_id: str
    tech_id: str
    technology_en: str
    duration_h: float
    rte: float
    self_discharge_pct_day: float
    charge_power_ratio: float
    calendar_life_years: float
    cycle_life_cycles: float
    total_capex_cny_kwh: float
    capital_recovery_factor: float
    annualized_capex_cny_per_kwh_capacity_year: float
    annualized_power_capex_cny_per_kw_year: float
    annualized_energy_capex_cny_per_kwh_year: float
    fixed_om_cny_kw_yr: float
    variable_om_cny_mwh: float
    annual_replacement_reserve_cny_per_kw_year: float
    annual_fixed_cost_cny_per_kw_discharge_year: float
    extra_charge_power_cost_cny_kw_year: float


@dataclass(frozen=True)
class Indices:
    kg: int
    xch: int
    capital: int
    starts: dict[str, int]
    binary_start: int
    product_start: int
    binary_hours: tuple[int, ...]
    nvar: int
    n: int


@dataclass
class Model:
    c: np.ndarray
    integrality: np.ndarray
    bounds: Bounds
    constraint: LinearConstraint
    indices: Indices
    task_mw: np.ndarray
    recharge_mw: np.ndarray
    pdis_mw: float
    energy_mwh: float
    native_charge_power_mw: float
    max_charge_power_mw: float
    config: Configuration
    binary_mode: str
    mutual_exclusion_formulation: str
    lfp_tangents: list[dict[str, float]]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(values: Iterable[int]) -> str:
    array = np.asarray(list(values), dtype=np.int16)
    return hashlib.sha256(array.tobytes()).hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def current_rss_mb() -> float:
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    return float("nan")


def peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if os.uname().sysname.lower() == "darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def crf(rate: float, years: float) -> float:
    if years <= 0.0:
        raise ValueError("Capital recovery life must be positive")
    return (
        rate
        * (1.0 + rate) ** years
        / ((1.0 + rate) ** years - 1.0)
    )


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
            f"Expected exactly one 2035 configuration {config_id}; "
            f"found {len(selected)}"
        )
    row = selected.iloc[0]
    extra_charge_cost = (
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
        calendar_life_years=float(row["calendar_life_years"]),
        cycle_life_cycles=float(row["cycle_life_cycles"]),
        total_capex_cny_kwh=float(row["total_capex_cny_kwh"]),
        capital_recovery_factor=float(row["capital_recovery_factor"]),
        annualized_capex_cny_per_kwh_capacity_year=float(
            row["annualized_capex_cny_per_kwh_capacity_year"]
        ),
        annualized_power_capex_cny_per_kw_year=float(
            row["annualized_power_capex_cny_per_kw_year"]
        ),
        annualized_energy_capex_cny_per_kwh_year=float(
            row["annualized_energy_capex_cny_per_kwh_year"]
        ),
        fixed_om_cny_kw_yr=float(row["fixed_om_cny_kw_yr"]),
        variable_om_cny_mwh=float(row["variable_om_cny_mwh"]),
        annual_replacement_reserve_cny_per_kw_year=float(
            row["annual_replacement_reserve_cny_per_kw_year"]
        ),
        annual_fixed_cost_cny_per_kw_discharge_year=float(
            row["annual_fixed_cost_cny_per_kw_discharge_year"]
        ),
        extra_charge_power_cost_cny_kw_year=extra_charge_cost,
    )


def parse_country_years(value: str) -> list[tuple[str, int]]:
    output: list[tuple[str, int]] = []
    for token in value.split(","):
        country, year = token.strip().split(":")
        output.append((country, int(year)))
    return output


def parse_validation_cells(
    value: str,
) -> set[tuple[str, int, str, str]]:
    output: set[tuple[str, int, str, str]] = set()
    for token in value.split(","):
        if not token.strip():
            continue
        country, year, config_id, cell = token.strip().split(":")
        output.add((country, int(year), config_id, cell))
    return output


def check_stage0_qa(path: Path, stage0_path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Stage-0 QA not found: {path}")
    qa = json.loads(path.read_text())
    if str(qa.get("status", "")).lower() != "pass":
        raise RuntimeError(
            f"Stage-0 QA status is not pass: {qa.get('status')}"
        )
    checks = qa.get("checks", {})
    if not isinstance(checks, dict) or not checks:
        raise RuntimeError("Stage-0 QA has no machine-readable checks")
    failed_checks = [
        key for key, value in checks.items() if value is not True
    ]
    if failed_checks:
        raise RuntimeError(
            f"Stage-0 QA contains failed checks: {failed_checks}"
        )
    boolean_candidates = [
        value
        for key, value in qa.items()
        if isinstance(value, bool)
        and (
            key.lower().endswith("pass")
            or key.lower().endswith("passed")
            or key.lower().startswith("all_")
        )
    ]
    if boolean_candidates and not all(boolean_candidates):
        failed = [
            key
            for key, value in qa.items()
            if isinstance(value, bool)
            and (
                key.lower().endswith("pass")
                or key.lower().endswith("passed")
                or key.lower().startswith("all_")
            )
            and not value
        ]
        raise RuntimeError(f"Stage-0 QA contains failed gates: {failed}")
    hashes = qa.get("hashes", {})
    source_sha = (
        qa.get("source_panel_sha256")
        or qa.get("input_panel_sha256")
        or qa.get("panel_sha256")
        or hashes.get("source_panel_sha256")
        or hashes.get("input_panel_sha256")
    )
    if (
        source_sha is not None
        and source_sha != EXPECTED_STAGE0_SOURCE_PANEL_SHA256
    ):
        raise RuntimeError(
            f"Stage-0 source-panel SHA changed: {source_sha}"
        )
    stage0_sha = sha256(stage0_path)
    qa_stage0_sha = hashes.get("hourly_output_sha256")
    if stage0_sha != EXPECTED_STAGE0_TABLE_SHA256:
        raise RuntimeError(
            "Frozen Stage-0 hourly table SHA changed: "
            f"{stage0_sha}"
        )
    if qa_stage0_sha != stage0_sha:
        raise RuntimeError(
            "Stage-0 QA/table SHA cross-check failed: "
            f"QA={qa_stage0_sha}, table={stage0_sha}"
        )
    return qa


def load_country_year(
    stage0_path: Path,
    country: str,
    year: int,
) -> pd.DataFrame:
    required = [
        "timestamp_utc",
        "country",
        "year",
        "source_hour_id",
        "source_day_id",
        "source_hour_of_day",
        "season_segment",
        "observed_gas_peaking_task_mw",
        "raw_gas_mean_mw",
        "r2_potential_recharge_mw",
        "event_flag_q75",
        "task_c3_q75_mw",
    ]
    frame = pd.read_csv(stage0_path, usecols=required)
    frame = frame[
        frame["country"].eq(country) & frame["year"].eq(year)
    ].copy()
    frame["timestamp_utc"] = pd.to_datetime(
        frame["timestamp_utc"], utc=True, errors="raise"
    )
    frame = frame.sort_values("source_hour_id").reset_index(drop=True)
    if len(frame) != 8760:
        raise RuntimeError(
            f"{country}-{year}: Stage-0 table must have 8,760 rows"
        )
    if not np.array_equal(
        frame["source_hour_id"].to_numpy(int), np.arange(8760)
    ):
        raise RuntimeError(
            f"{country}-{year}: source_hour_id is not canonical"
        )
    if not np.array_equal(
        frame["source_day_id"].to_numpy(int), np.arange(8760) // 24
    ):
        raise RuntimeError(
            f"{country}-{year}: source_day_id is not canonical"
        )
    if not np.array_equal(
        frame["source_hour_of_day"].to_numpy(int), np.arange(8760) % 24
    ):
        raise RuntimeError(
            f"{country}-{year}: source_hour_of_day is not canonical"
        )
    numeric = [
        "observed_gas_peaking_task_mw",
        "raw_gas_mean_mw",
        "r2_potential_recharge_mw",
        "task_c3_q75_mw",
    ]
    if not np.isfinite(frame[numeric].to_numpy(float)).all():
        raise RuntimeError(f"{country}-{year}: non-finite Stage-0 values")
    if (frame[numeric] < 0.0).any().any():
        raise RuntimeError(f"{country}-{year}: negative Stage-0 values")
    raw = frame["observed_gas_peaking_task_mw"].to_numpy(float)
    task = frame["task_c3_q75_mw"].to_numpy(float)
    if np.any(task > raw + 1e-9):
        raise RuntimeError(
            f"{country}-{year}: C3 task exceeds observed fossil gas"
        )
    if frame["raw_gas_mean_mw"].nunique() != 1:
        raise RuntimeError(
            f"{country}-{year}: raw_gas_mean_mw is not constant"
        )
    if abs(float(frame["raw_gas_mean_mw"].iloc[0]) - raw.mean()) > 1e-9:
        raise RuntimeError(
            f"{country}-{year}: raw gas mean is inconsistent"
        )
    for column in ["event_flag_q75"]:
        frame[column] = (
            frame[column]
            if frame[column].dtype == bool
            else frame[column]
            .astype(str)
            .str.lower()
            .isin({"true", "1", "yes", "y"})
        )
    event_positive_error = int(
        np.sum((task > 1e-12) & ~frame["event_flag_q75"].to_numpy(bool))
    )
    if event_positive_error:
        raise RuntimeError(
            f"{country}-{year}: positive C3 task outside q75 event days"
        )
    return frame


def seeded_permutation(
    values: list[int],
    country: str,
    year: int,
    segment: int,
    kind: str,
) -> list[int]:
    digest = hashlib.sha256(
        f"{PAIRED_SEED}|{country}|{year}|{segment}|{kind}".encode()
    ).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    return [
        int(value)
        for value in rng.permutation(np.asarray(values, dtype=int))
    ]


def interior_gap_positions(number: int, available: int) -> list[int]:
    if number == 0:
        return []
    if number > available:
        raise RuntimeError(
            f"Cannot place {number} groups in {available} interior gaps"
        )
    positions = np.floor(
        (np.arange(number, dtype=float) + 0.5) * available / number
    ).astype(int)
    if len(np.unique(positions)) != number:
        raise RuntimeError("Interior-gap allocator produced duplicates")
    return [int(value) for value in positions]


def place_groups(
    normal_days: list[int],
    event_groups: list[list[int]],
) -> list[int]:
    if len(normal_days) < 2:
        raise RuntimeError("Each segment needs two non-event boundary days")
    gap_count = len(normal_days) - 1
    positions = interior_gap_positions(len(event_groups), gap_count)
    by_gap = {
        gap: group for gap, group in zip(positions, event_groups)
    }
    order = [normal_days[0]]
    for gap in range(gap_count):
        order.extend(by_gap.get(gap, []))
        order.append(normal_days[gap + 1])
    return order


def make_day_orders(
    frame: pd.DataFrame,
    country: str,
    year: int,
) -> tuple[dict[str, list[int]], pd.DataFrame]:
    daily = (
        frame.groupby("source_day_id", sort=True)
        .agg(
            segment=("season_segment", "first"),
            segment_nunique=("season_segment", "nunique"),
            event=("event_flag_q75", "first"),
            event_nunique=("event_flag_q75", "nunique"),
            task_mwh=("task_c3_q75_mw", "sum"),
            recharge_mwh=("r2_potential_recharge_mw", "sum"),
        )
        .reset_index()
    )
    if len(daily) != 365:
        raise RuntimeError(f"{country}-{year}: daily table is not 365 rows")
    if not daily["segment_nunique"].eq(1).all():
        raise RuntimeError(f"{country}-{year}: segment changes within day")
    if not daily["event_nunique"].eq(1).all():
        raise RuntimeError(f"{country}-{year}: event flag changes within day")
    orders = {"D": [], "C": []}
    for segment in sorted(daily["segment"].unique()):
        subset = daily[daily["segment"].eq(segment)]
        event_days = [
            int(value)
            for value in subset.loc[subset["event"], "source_day_id"]
        ]
        normal_days = [
            int(value)
            for value in subset.loc[~subset["event"], "source_day_id"]
        ]
        event_days = seeded_permutation(
            event_days, country, year, int(segment), "event"
        )
        normal_days = seeded_permutation(
            normal_days, country, year, int(segment), "normal"
        )
        dispersed_groups = [[day] for day in event_days]
        clustered_groups = [
            event_days[index : index + CLUSTER_DAYS]
            for index in range(0, len(event_days), CLUSTER_DAYS)
        ]
        orders["D"].extend(place_groups(normal_days, dispersed_groups))
        orders["C"].extend(place_groups(normal_days, clustered_groups))
    reference = list(range(365))
    for label, order in orders.items():
        if len(order) != 365 or sorted(order) != reference:
            raise RuntimeError(
                f"{country}-{year}: invalid {label} day permutation"
            )
    return orders, daily


def hourly_order(day_order: list[int]) -> np.ndarray:
    return np.concatenate(
        [
            np.arange(day * 24, (day + 1) * 24, dtype=int)
            for day in day_order
        ]
    )


def event_run_statistics(flags: np.ndarray) -> dict[str, Any]:
    flags = np.asarray(flags, dtype=bool)
    if flags[0] or flags[-1]:
        raise RuntimeError(
            "Chronology construction must leave non-event boundary days"
        )
    lengths: list[int] = []
    start: int | None = None
    for index, flag in enumerate(flags):
        if flag and start is None:
            start = index
        if start is not None and (not flag or index == len(flags) - 1):
            end = index if not flag else index + 1
            lengths.append(end - start)
            start = None
    adjacent = int(np.sum(flags[:-1] & flags[1:]))
    return {
        "event_days": int(flags.sum()),
        "adjacent_event_day_pairs": adjacent,
        "mean_event_run_days": float(np.mean(lengths)),
        "maximum_event_run_days": int(max(lengths)),
        "event_runs": int(len(lengths)),
        "event_run_lengths_json": json.dumps(lengths),
    }


def lfp_capital_function(
    rated_efc: float,
    config: Configuration,
    energy_mwh: float,
    replaceable_fraction: float,
) -> tuple[float, float, float]:
    """Return annual capital CNY, d(cost)/d(discharge MWh), rep life."""
    calendar = config.calendar_life_years
    lifetime = config.cycle_life_cycles
    base_crf = crf(DISCOUNT_RATE, calendar)
    threshold = lifetime / calendar
    capex = config.total_capex_cny_kwh * energy_mwh * 1_000.0
    if rated_efc <= threshold or rated_efc <= 0.0:
        return capex * base_crf, 0.0, calendar
    replacement_life = lifetime / rated_efc
    replacement_crf = crf(DISCOUNT_RATE, replacement_life)
    annual = capex * (
        (1.0 - replaceable_fraction) * base_crf
        + replaceable_fraction * replacement_crf
    )
    log_factor = math.log1p(DISCOUNT_RATE)
    q = math.exp(-log_factor * replacement_life)
    derivative_crf_n = (
        DISCOUNT_RATE
        * log_factor
        * q
        * lifetime
        / (rated_efc**2 * (1.0 - q) ** 2)
    )
    derivative_discharge = (
        config.total_capex_cny_kwh
        * 1_000.0
        * replaceable_fraction
        * derivative_crf_n
    )
    return annual, derivative_discharge, replacement_life


def lfp_tangent(
    rated_efc: float,
    config: Configuration,
    energy_mwh: float,
    replaceable_fraction: float,
) -> dict[str, float]:
    annual, slope, life = lfp_capital_function(
        rated_efc, config, energy_mwh, replaceable_fraction
    )
    discharge = rated_efc * energy_mwh
    return {
        "anchor_rated_efc": float(rated_efc),
        "slope_cny_per_mwh": float(slope),
        "intercept_cny_year": float(annual - slope * discharge),
        "true_annual_capital_cny_year": float(annual),
        "replacement_life_years": float(life),
    }


class MatrixBuilder:
    def __init__(self, nvar: int) -> None:
        self.nvar = nvar
        self.rows: list[int] = []
        self.cols: list[int] = []
        self.data: list[float] = []
        self.lower: list[float] = []
        self.upper: list[float] = []

    def add(
        self,
        coefficients: Iterable[tuple[int, float]],
        lower: float,
        upper: float,
    ) -> None:
        row = len(self.lower)
        for column, value in coefficients:
            if value != 0.0:
                self.rows.append(row)
                self.cols.append(int(column))
                self.data.append(float(value))
        self.lower.append(float(lower))
        self.upper.append(float(upper))

    def build(self) -> LinearConstraint:
        matrix = coo_matrix(
            (self.data, (self.rows, self.cols)),
            shape=(len(self.lower), self.nvar),
        ).tocsr()
        return LinearConstraint(
            matrix,
            np.asarray(self.lower, dtype=float),
            np.asarray(self.upper, dtype=float),
        )


def build_model(
    task_mw: np.ndarray,
    recharge_mw: np.ndarray,
    config: Configuration,
    binary_mode: str,
    mutual_exclusion_formulation: str,
    lfp_tangents: list[dict[str, float]],
    replaceable_fraction: float,
) -> Model:
    task = np.asarray(task_mw, dtype=float)
    recharge = np.asarray(recharge_mw, dtype=float)
    if task.shape != (8760,) or recharge.shape != (8760,):
        raise RuntimeError("Stage-1 model requires 8,760 task/recharge hours")
    if np.any(task < 0.0) or np.any(recharge < 0.0):
        raise RuntimeError("Task and recharge must be non-negative")
    pdis = float(build_model.raw_gas_mean_mw) * (
        FORCED_POWER_SHARE_OF_RAW_GAS_MEAN
    )
    if pdis <= 0.0:
        raise RuntimeError("Forced discharge power must be positive")
    energy = config.duration_h * pdis
    native_pch = config.charge_power_ratio * pdis
    max_pch = native_pch + MAX_EXTRA_CHARGE_POWER_MULTIPLE * pdis
    if binary_mode == "sparse":
        binary_hours = tuple(
            int(value)
            for value in np.flatnonzero(
                (task > 1e-12) & (recharge > 1e-12)
            )
        )
    elif binary_mode == "full":
        binary_hours = tuple(range(8760))
    else:
        raise ValueError(binary_mode)
    if mutual_exclusion_formulation not in {
        "bigm",
        "indicator",
        "perspective_bigm",
    }:
        raise ValueError(mutual_exclusion_formulation)

    names = ("gas", "recharge_used", "charge", "discharge", "soc", "spill")
    starts = {
        name: 3 + index * len(task) for index, name in enumerate(names)
    }
    binary_start = 3 + len(names) * len(task)
    product_start = binary_start + len(binary_hours)
    product_count = (
        len(binary_hours)
        if mutual_exclusion_formulation == "perspective_bigm"
        else 0
    )
    nvar = product_start + product_count
    indices = Indices(
        kg=0,
        xch=1,
        capital=2,
        starts=starts,
        binary_start=binary_start,
        product_start=product_start,
        binary_hours=binary_hours,
        nvar=nvar,
        n=len(task),
    )

    c = np.zeros(nvar, dtype=float)
    c[indices.kg] = NGCC_FIXED_CNY_KW_YR * 1_000.0
    c[indices.xch] = (
        config.extra_charge_power_cost_cny_kw_year * 1_000.0
    )
    c[indices.capital] = 1.0
    c[starts["gas"] : starts["gas"] + len(task)] = (
        NGCC_VARIABLE_CNY_MWH
    )
    c[
        starts["recharge_used"] : starts["recharge_used"] + len(task)
    ] = RECHARGE_OPPORTUNITY_PRICE_CNY_MWH
    c[starts["discharge"] : starts["discharge"] + len(task)] = (
        config.variable_om_cny_mwh
    )

    lower = np.zeros(nvar, dtype=float)
    upper = np.full(nvar, np.inf, dtype=float)
    upper[indices.kg] = max(float(task.max()), 1e-9)
    upper[indices.xch] = MAX_EXTRA_CHARGE_POWER_MULTIPLE * pdis
    if config.tech_id == "li_lfp":
        upper[indices.capital] = np.inf
    else:
        fixed_capital = (
            config.annualized_capex_cny_per_kwh_capacity_year
            * energy
            * 1_000.0
        )
        lower[indices.capital] = fixed_capital
        upper[indices.capital] = fixed_capital
    upper[starts["gas"] : starts["gas"] + len(task)] = task
    upper[
        starts["recharge_used"] : starts["recharge_used"] + len(task)
    ] = recharge
    upper[starts["charge"] : starts["charge"] + len(task)] = recharge
    upper[starts["discharge"] : starts["discharge"] + len(task)] = (
        np.minimum(task, pdis)
    )
    lower[starts["soc"] : starts["soc"] + len(task)] = SOC_MIN_FRAC * energy
    upper[starts["soc"] : starts["soc"] + len(task)] = SOC_MAX_FRAC * energy
    upper[starts["spill"] : starts["spill"] + len(task)] = recharge
    if binary_hours:
        upper[
            binary_start : binary_start + len(binary_hours)
        ] = 1.0
    if product_count:
        upper[product_start:] = (
            MAX_EXTRA_CHARGE_POWER_MULTIPLE * pdis
        )

    integrality = np.zeros(nvar, dtype=np.uint8)
    if binary_hours:
        integrality[
            binary_start : binary_start + len(binary_hours)
        ] = 1

    builder = MatrixBuilder(nvar)
    eta_ch = math.sqrt(config.rte)
    eta_dis = math.sqrt(config.rte)
    retention = (
        max(0.0, 1.0 - config.self_discharge_pct_day / 100.0)
        ** (1.0 / 24.0)
    )
    for hour in range(len(task)):
        builder.add(
            [
                (starts["gas"] + hour, 1.0),
                (starts["discharge"] + hour, 1.0),
            ],
            task[hour],
            task[hour],
        )
        builder.add(
            [
                (starts["recharge_used"] + hour, 1.0),
                (starts["spill"] + hour, 1.0),
            ],
            recharge[hour],
            recharge[hour],
        )
        builder.add(
            [
                (starts["charge"] + hour, 1.0),
                (starts["recharge_used"] + hour, -1.0),
            ],
            0.0,
            0.0,
        )
        previous = (hour - 1) % len(task)
        builder.add(
            [
                (starts["soc"] + hour, 1.0),
                (starts["soc"] + previous, -retention),
                (starts["charge"] + hour, -eta_ch),
                (starts["discharge"] + hour, 1.0 / eta_dis),
            ],
            (1.0 - retention) * SOC_MIN_FRAC * energy,
            (1.0 - retention) * SOC_MIN_FRAC * energy,
        )
        builder.add(
            [
                (starts["gas"] + hour, 1.0),
                (indices.kg, -1.0),
            ],
            -np.inf,
            0.0,
        )
        builder.add(
            [
                (starts["charge"] + hour, 1.0),
                (indices.xch, -1.0),
            ],
            -np.inf,
            native_pch,
        )

    binary_position = {
        hour: binary_start + index
        for index, hour in enumerate(binary_hours)
    }
    product_position = {
        hour: product_start + index
        for index, hour in enumerate(binary_hours)
    }
    for hour, column in binary_position.items():
        if mutual_exclusion_formulation in {
            "bigm",
            "perspective_bigm",
        }:
            # These are the tight single-hour disjunctive bounds, not
            # technology-wide M values.
            charge_upper = min(float(recharge[hour]), max_pch)
            discharge_upper = min(float(task[hour]), pdis)
            builder.add(
                [
                    (starts["charge"] + hour, 1.0),
                    (column, -charge_upper),
                ],
                -np.inf,
                0.0,
            )
            builder.add(
                [
                    (starts["discharge"] + hour, 1.0),
                    (column, discharge_upper),
                ],
                -np.inf,
                discharge_upper,
            )
        if mutual_exclusion_formulation == "perspective_bigm":
            product = product_position[hour]
            extra_upper = MAX_EXTRA_CHARGE_POWER_MULTIPLE * pdis
            # Full McCormick envelope for w_h = xch * z_h, with
            # 0 <= xch <= extra_upper and z_h binary.
            builder.add([(product, 1.0)], 0.0, np.inf)
            builder.add(
                [(product, 1.0), (indices.xch, -1.0)],
                -np.inf,
                0.0,
            )
            builder.add(
                [(product, 1.0), (column, -extra_upper)],
                -np.inf,
                0.0,
            )
            builder.add(
                [
                    (product, 1.0),
                    (indices.xch, -1.0),
                    (column, -extra_upper),
                ],
                -extra_upper,
                np.inf,
            )
            builder.add(
                [
                    (starts["charge"] + hour, 1.0),
                    (column, -native_pch),
                    (product, -1.0),
                ],
                -np.inf,
                0.0,
            )

    if config.tech_id == "li_lfp":
        for tangent in lfp_tangents:
            coefficients = [
                (starts["discharge"] + hour, tangent["slope_cny_per_mwh"])
                for hour in range(len(task))
                if task[hour] > 0.0
            ]
            coefficients.append((indices.capital, -1.0))
            builder.add(
                coefficients,
                -np.inf,
                -tangent["intercept_cny_year"],
            )

    return Model(
        c=c,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraint=builder.build(),
        indices=indices,
        task_mw=task,
        recharge_mw=recharge,
        pdis_mw=pdis,
        energy_mwh=energy,
        native_charge_power_mw=native_pch,
        max_charge_power_mw=max_pch,
        config=config,
        binary_mode=binary_mode,
        mutual_exclusion_formulation=mutual_exclusion_formulation,
        lfp_tangents=list(lfp_tangents),
    )


# Set immediately before build_model; keeping it outside the matrix arguments
# makes the fixed normalization visibly independent of the event-task mean.
build_model.raw_gas_mean_mw = float("nan")


def solve_once(
    model: Model,
    time_limit_seconds: float,
    solver: str,
) -> tuple[Any, dict[str, float]]:
    rss_before = current_rss_mb()
    peak_before = peak_rss_mb()
    started = time.perf_counter()
    if solver == "scipy_highs":
        if model.mutual_exclusion_formulation != "bigm":
            raise RuntimeError(
                "SciPy/HiGHS requires the explicit big-M formulation"
            )
        result = milp(
            c=model.c,
            integrality=model.integrality,
            bounds=model.bounds,
            constraints=model.constraint,
            options={
                "presolve": True,
                "time_limit": float(time_limit_seconds),
                "mip_rel_gap": MIP_RELATIVE_GAP_TOLERANCE,
            },
        )
        result.objective_bound = float(
            getattr(result, "mip_dual_bound", np.nan)
        )
        result.absolute_gap_cny_per_forced_kw_year = (
            abs(float(result.fun) - result.objective_bound)
            / (model.pdis_mw * 1_000.0)
            if np.isfinite(getattr(result, "fun", np.nan))
            and np.isfinite(result.objective_bound)
            else float("nan")
        )
    elif solver == "gurobi":
        result = solve_once_gurobi(model, time_limit_seconds)
    else:
        raise ValueError(solver)
    elapsed = time.perf_counter() - started
    runtime = {
        "solve_seconds": float(elapsed),
        "rss_before_mb": rss_before,
        "rss_after_mb": current_rss_mb(),
        "peak_rss_before_mb": peak_before,
        "peak_rss_after_mb": peak_rss_mb(),
    }
    if not result.success:
        raise RuntimeError(
            f"MILP failed ({model.config.config_id}, "
            f"{model.binary_mode}, "
            f"{model.mutual_exclusion_formulation}): {result.message}; "
            f"incumbent={getattr(result, 'fun', np.nan)}, "
            f"best_bound={getattr(result, 'objective_bound', np.nan)}, "
            f"relative_gap={getattr(result, 'mip_gap', np.nan)}, "
            f"absolute_gap_cny_per_forced_kw_year="
            f"{getattr(result, 'absolute_gap_cny_per_forced_kw_year', np.nan)}, "
            f"nodes={getattr(result, 'mip_node_count', np.nan)}"
        )
    return result, runtime


def solve_once_gurobi(
    model: Model,
    time_limit_seconds: float,
) -> SimpleNamespace:
    """Solve an already assembled sparse model with Gurobi."""
    try:
        import gurobipy as gp
    except ImportError as exc:
        raise RuntimeError("gurobipy is not installed") from exc
    grb = gp.GRB
    gurobi_model = gp.Model()
    gurobi_model.Params.OutputFlag = 0
    gurobi_model.Params.LogToConsole = 0
    gurobi_model.Params.Presolve = 2
    gurobi_model.Params.MIPGap = MIP_RELATIVE_GAP_TOLERANCE
    gurobi_model.Params.MIPGapAbs = (
        MIP_ABSOLUTE_GAP_CNY_PER_FORCED_KW_YEAR
    )
    # The physical admission rule is much tighter than Gurobi's default
    # integer feasibility tolerance.  Without these settings, a nominally
    # binary mode can leak about M * 1e-5 MW into the forbidden direction.
    gurobi_model.Params.FeasibilityTol = 1e-9
    gurobi_model.Params.IntFeasTol = 1e-9
    gurobi_model.Params.OptimalityTol = 1e-9
    gurobi_model.Params.TimeLimit = float(time_limit_seconds)
    gurobi_model.Params.Threads = min(8, os.cpu_count() or 1)
    gurobi_model.Params.MIPFocus = int(
        os.environ.get("STAGE1_GUROBI_MIPFOCUS", "0")
    )
    log_path = os.environ.get("STAGE1_GUROBI_LOG", "").strip()
    if log_path:
        gurobi_model.Params.OutputFlag = 1
        gurobi_model.Params.LogToConsole = 0
        gurobi_model.Params.LogFile = log_path
    vtype = np.full(model.indices.nvar, grb.CONTINUOUS, dtype="U1")
    vtype[model.integrality.astype(bool)] = grb.BINARY
    variables = gurobi_model.addMVar(
        model.indices.nvar,
        lb=np.asarray(model.bounds.lb, dtype=float),
        ub=np.asarray(model.bounds.ub, dtype=float),
        vtype=vtype,
        name="x",
    )
    matrix = csr_matrix(model.constraint.A)
    lower = np.asarray(model.constraint.lb, dtype=float)
    upper = np.asarray(model.constraint.ub, dtype=float)
    equality = np.isfinite(lower) & np.isfinite(upper) & np.isclose(
        lower, upper, rtol=0.0, atol=1e-12
    )
    upper_only = np.isfinite(upper) & ~equality
    lower_only = np.isfinite(lower) & ~equality
    if equality.any():
        gurobi_model.addMConstr(
            matrix[equality], variables, "=", upper[equality], name="eq"
        )
    if upper_only.any():
        gurobi_model.addMConstr(
            matrix[upper_only],
            variables,
            "<",
            upper[upper_only],
            name="ub",
        )
    if lower_only.any():
        gurobi_model.addMConstr(
            matrix[lower_only],
            variables,
            ">",
            lower[lower_only],
            name="lb",
        )
    variable_list = variables.tolist()
    if model.mutual_exclusion_formulation == "indicator":
        for hour, binary_column in zip(
            model.indices.binary_hours,
            range(
                model.indices.binary_start,
                model.indices.binary_start
                + len(model.indices.binary_hours),
            ),
        ):
            binary = variable_list[binary_column]
            charge = variable_list[
                model.indices.starts["charge"] + hour
            ]
            discharge = variable_list[
                model.indices.starts["discharge"] + hour
            ]
            gurobi_model.addGenConstrIndicator(
                binary,
                0,
                charge == 0.0,
                name=f"no_charge_{hour}",
            )
            gurobi_model.addGenConstrIndicator(
                binary,
                1,
                discharge == 0.0,
                name=f"no_discharge_{hour}",
            )
    normalization = model.pdis_mw * 1_000.0
    gurobi_model.setObjective(
        (model.c @ variables) / normalization,
        grb.MINIMIZE,
    )
    gurobi_model.optimize()
    success = gurobi_model.Status == grb.OPTIMAL
    has_incumbent = gurobi_model.SolCount > 0
    if has_incumbent:
        vector = np.asarray(variables.X, dtype=float)
        objective = float(gurobi_model.ObjVal) * normalization
    else:
        vector = np.full(model.indices.nvar, np.nan, dtype=float)
        objective = float("nan")
    objective_bound = float(gurobi_model.ObjBound) * normalization
    absolute_gap_per_forced_kw_year = (
        abs(float(gurobi_model.ObjVal) - float(gurobi_model.ObjBound))
        if has_incumbent
        else float("nan")
    )
    return SimpleNamespace(
        success=success,
        status=0 if success else int(gurobi_model.Status),
        raw_solver_status=int(gurobi_model.Status),
        message=(
            f"Gurobi status {gurobi_model.Status}; "
            f"SolCount={gurobi_model.SolCount}; "
            f"ObjVal={objective}; ObjBound={objective_bound}; "
            f"MIPGap="
            f"{float(gurobi_model.MIPGap) if has_incumbent else np.nan}; "
            f"AbsGapPerForcedKWYear="
            f"{absolute_gap_per_forced_kw_year}; "
            f"NodeCount={float(gurobi_model.NodeCount)}"
        ),
        x=vector,
        fun=objective,
        objective_bound=objective_bound,
        absolute_gap_cny_per_forced_kw_year=(
            absolute_gap_per_forced_kw_year
        ),
        mip_gap=(
            float(gurobi_model.MIPGap)
            if has_incumbent
            else float("nan")
        ),
        mip_node_count=float(gurobi_model.NodeCount),
        incumbent_available=has_incumbent,
    )


def solve_with_lfp_outer_approximation(
    task_mw: np.ndarray,
    recharge_mw: np.ndarray,
    raw_gas_mean_mw: float,
    config: Configuration,
    binary_mode: str,
    mutual_exclusion_formulation: str,
    replaceable_fraction: float,
    time_limit_seconds: float,
    solver: str,
) -> tuple[Model, Any, dict[str, Any]]:
    build_model.raw_gas_mean_mw = raw_gas_mean_mw
    pdis = raw_gas_mean_mw * FORCED_POWER_SHARE_OF_RAW_GAS_MEAN
    energy = config.duration_h * pdis
    task_efc_ceiling = float(np.minimum(task_mw, pdis).sum() / energy)
    tangents: list[dict[str, float]] = []
    if config.tech_id == "li_lfp":
        tangents.append(
            lfp_tangent(0.0, config, energy, replaceable_fraction)
        )
        threshold = config.cycle_life_cycles / config.calendar_life_years
        if task_efc_ceiling > threshold + 1e-9:
            tangents.append(
                lfp_tangent(
                    task_efc_ceiling,
                    config,
                    energy,
                    replaceable_fraction,
                )
            )
    iterations: list[dict[str, Any]] = []
    total_runtime = {
        "solve_seconds": 0.0,
        "rss_before_mb": current_rss_mb(),
        "rss_after_mb": float("nan"),
        "peak_rss_before_mb": peak_rss_mb(),
        "peak_rss_after_mb": float("nan"),
    }
    for iteration in range(1, 11):
        model = build_model(
            task_mw,
            recharge_mw,
            config,
            binary_mode,
            mutual_exclusion_formulation,
            tangents,
            replaceable_fraction,
        )
        result, runtime = solve_once(model, time_limit_seconds, solver)
        total_runtime["solve_seconds"] += runtime["solve_seconds"]
        total_runtime["rss_after_mb"] = runtime["rss_after_mb"]
        total_runtime["peak_rss_after_mb"] = runtime["peak_rss_after_mb"]
        discharge = result.x[
            model.indices.starts["discharge"] :
            model.indices.starts["discharge"] + model.indices.n
        ]
        rated_efc = float(discharge.sum() / model.energy_mwh)
        capital_value = float(result.x[model.indices.capital])
        if config.tech_id == "li_lfp":
            true_capital, _, replacement_life = lfp_capital_function(
                rated_efc,
                config,
                model.energy_mwh,
                replaceable_fraction,
            )
            relative_gap = max(
                0.0,
                true_capital - capital_value,
            ) / max(true_capital, 1.0)
        else:
            true_capital = capital_value
            replacement_life = config.calendar_life_years
            relative_gap = 0.0
        iterations.append(
            {
                "iteration": iteration,
                "rated_output_efc": rated_efc,
                "epigraph_capital_cny_year": capital_value,
                "true_capital_cny_year": true_capital,
                "relative_gap": relative_gap,
                "replacement_life_years": replacement_life,
                "tangent_count": len(tangents),
                **runtime,
            }
        )
        if relative_gap <= PWL_RELATIVE_GAP_TOLERANCE:
            total_runtime["iterations"] = iterations
            total_runtime["pwl_relative_gap"] = relative_gap
            total_runtime["pwl_iterations"] = iteration
            total_runtime["task_efc_ceiling"] = task_efc_ceiling
            return model, result, total_runtime
        tangents.append(
            lfp_tangent(
                rated_efc,
                config,
                model.energy_mwh,
                replaceable_fraction,
            )
        )
    raise RuntimeError(
        f"LFP outer approximation did not converge in 10 iterations; "
        f"last gap={relative_gap}"
    )


def fields(model: Model, x: np.ndarray) -> dict[str, np.ndarray]:
    return {
        name: x[start : start + model.indices.n]
        for name, start in model.indices.starts.items()
    }


def annual_error(
    values: np.ndarray,
    denominator: float,
) -> tuple[float, float]:
    absolute = float(values.sum())
    relative = abs(absolute) / max(abs(denominator), 1.0)
    return absolute, relative


def summarize_solution(
    model: Model,
    result: Any,
    runtime: dict[str, Any],
    country: str,
    year: int,
    config: Configuration,
    cell: str,
    task_order: list[int],
    recharge_order: list[int],
    event_flags: np.ndarray,
    replaceable_fraction: float,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    x = np.asarray(result.x, dtype=float)
    f = fields(model, x)
    kg = float(x[model.indices.kg])
    xch = float(x[model.indices.xch])
    capital = float(x[model.indices.capital])
    pch = model.native_charge_power_mw + xch
    task_error = model.task_mw - f["gas"] - f["discharge"]
    charge_error = f["charge"] - f["recharge_used"]
    recharge_error = (
        model.recharge_mw - f["recharge_used"] - f["spill"]
    )
    previous_soc = np.roll(f["soc"], 1)
    retention = (
        max(0.0, 1.0 - config.self_discharge_pct_day / 100.0)
        ** (1.0 / 24.0)
    )
    eta_ch = math.sqrt(config.rte)
    eta_dis = math.sqrt(config.rte)
    soc_error = (
        f["soc"]
        - retention * previous_soc
        - eta_ch * f["charge"]
        + f["discharge"] / eta_dis
        - (1.0 - retention) * SOC_MIN_FRAC * model.energy_mwh
    )
    simultaneous = np.minimum(f["charge"], f["discharge"])
    simultaneous_ratio = float(
        simultaneous.sum() / max(f["discharge"].sum(), 1e-12)
    )
    gas_fixed = kg * 1_000.0 * NGCC_FIXED_CNY_KW_YR
    gas_variable = float(f["gas"].sum()) * NGCC_VARIABLE_CNY_MWH
    recharge_cost = (
        float(f["recharge_used"].sum())
        * RECHARGE_OPPORTUNITY_PRICE_CNY_MWH
    )
    storage_vom = (
        float(f["discharge"].sum()) * config.variable_om_cny_mwh
    )
    extra_charge = (
        xch
        * 1_000.0
        * config.extra_charge_power_cost_cny_kw_year
    )
    native_fom = (
        config.fixed_om_cny_kw_yr * model.pdis_mw * 1_000.0
    )
    rated_efc = float(f["discharge"].sum() / model.energy_mwh)
    usable_efc = float(
        f["discharge"].sum()
        / ((SOC_MAX_FRAC - SOC_MIN_FRAC) * model.energy_mwh)
    )
    true_capital, _, replacement_life = (
        lfp_capital_function(
            rated_efc,
            config,
            model.energy_mwh,
            replaceable_fraction,
        )
        if config.tech_id == "li_lfp"
        else (
            capital,
            0.0,
            config.calendar_life_years,
        )
    )
    solver_objective_reconstruction = (
        gas_fixed
        + gas_variable
        + recharge_cost
        + storage_vom
        + extra_charge
        + capital
    )
    true_cost_reconstruction = (
        gas_fixed
        + gas_variable
        + recharge_cost
        + storage_vom
        + extra_charge
        + true_capital
    )
    solver_total_cost = solver_objective_reconstruction + native_fom
    total_cost = true_cost_reconstruction + native_fom
    baseline_gas_capacity = float(model.task_mw.max())
    baseline_cost = (
        baseline_gas_capacity * 1_000.0 * NGCC_FIXED_CNY_KW_YR
        + float(model.task_mw.sum()) * NGCC_VARIABLE_CNY_MWH
    )
    benefit = baseline_cost - total_cost
    task_abs, task_rel = annual_error(
        task_error, float(model.task_mw.sum())
    )
    charge_abs, charge_rel = annual_error(
        charge_error, float(f["charge"].sum())
    )
    recharge_abs, recharge_rel = annual_error(
        recharge_error, float(model.recharge_mw.sum())
    )
    soc_abs, soc_rel = annual_error(soc_error, model.energy_mwh)
    run_stats = event_run_statistics(event_flags)
    row = {
        "country": country,
        "year": year,
        "config_id": config.config_id,
        "technology_en": config.technology_en,
        "task_definition": "C3_q75_event_gated_month_hour_q10_above_floor",
        "recharge_specification": "R2_potential_vre_surplus",
        "cell": cell,
        "task_chronology": cell[0],
        "recharge_chronology": cell[1],
        "binary_mode": model.binary_mode,
        "mutual_exclusion_formulation": (
            model.mutual_exclusion_formulation
        ),
        "binary_variables": len(model.indices.binary_hours),
        "perspective_product_variables": (
            len(model.indices.binary_hours)
            if model.mutual_exclusion_formulation
            == "perspective_bigm"
            else 0
        ),
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "mip_gap": float(getattr(result, "mip_gap", np.nan)),
        "solver_objective_bound_cny_year": float(
            getattr(result, "objective_bound", np.nan)
        ),
        "solver_objective_value_cny_year": float(result.fun),
        "solver_objective_bound_total_cost_cny_year": float(
            getattr(result, "objective_bound", np.nan) + native_fom
        ),
        "solver_absolute_gap_cny_per_forced_kw_year": float(
            getattr(
                result,
                "absolute_gap_cny_per_forced_kw_year",
                np.nan,
            )
        ),
        "mip_node_count": int(getattr(result, "mip_node_count", -1)),
        "solve_seconds": float(runtime["solve_seconds"]),
        "pwl_iterations": int(runtime["pwl_iterations"]),
        "pwl_relative_gap": float(runtime["pwl_relative_gap"]),
        "rss_after_mb": float(runtime["rss_after_mb"]),
        "peak_rss_after_mb": float(runtime["peak_rss_after_mb"]),
        "raw_gas_mean_mw": float(build_model.raw_gas_mean_mw),
        "forced_discharge_power_mw": model.pdis_mw,
        "rated_energy_capacity_mwh": model.energy_mwh,
        "native_charge_power_mw": model.native_charge_power_mw,
        "extra_charge_power_mw": xch,
        "total_charge_power_mw": pch,
        "gas_capacity_mw": kg,
        "task_energy_mwh": float(model.task_mw.sum()),
        "recharge_available_mwh": float(model.recharge_mw.sum()),
        "recharge_used_mwh": float(f["recharge_used"].sum()),
        "recharge_spill_mwh": float(f["spill"].sum()),
        "storage_charge_mwh": float(f["charge"].sum()),
        "storage_discharge_mwh": float(f["discharge"].sum()),
        "rated_output_efc": rated_efc,
        "usable_band_efc": usable_efc,
        "task_efc_ceiling": float(runtime["task_efc_ceiling"]),
        "maximum_simultaneous_charge_discharge_mw": float(
            simultaneous.max()
        ),
        "simultaneous_energy_ratio": simultaneous_ratio,
        "charge_power_saturation_h": int(
            np.sum(
                (f["charge"] >= 0.99 * pch)
                & (f["spill"] > 1e-9)
            )
        ),
        "discharge_power_saturation_h": int(
            np.sum(f["discharge"] >= 0.99 * model.pdis_mw)
        ),
        "task_balance_error_mwh": task_abs,
        "task_balance_relative_error": task_rel,
        "charge_source_error_mwh": charge_abs,
        "charge_source_relative_error": charge_rel,
        "recharge_balance_error_mwh": recharge_abs,
        "recharge_balance_relative_error": recharge_rel,
        "soc_balance_error_mwh": soc_abs,
        "soc_balance_relative_error": soc_rel,
        "maximum_task_balance_error_mw": float(
            np.max(np.abs(task_error))
        ),
        "maximum_charge_source_error_mw": float(
            np.max(np.abs(charge_error))
        ),
        "maximum_recharge_balance_error_mw": float(
            np.max(np.abs(recharge_error))
        ),
        "maximum_soc_balance_error_mwh": float(
            np.max(np.abs(soc_error))
        ),
        "cyclic_soc_difference_mwh": float(
            abs(
                f["soc"][0]
                - (
                    retention * f["soc"][-1]
                    + eta_ch * f["charge"][0]
                    - f["discharge"][0] / eta_dis
                    + (1.0 - retention)
                    * SOC_MIN_FRAC
                    * model.energy_mwh
                )
            )
        ),
        "gas_fixed_cost_cny_year": gas_fixed,
        "gas_variable_cost_cny_year": gas_variable,
        "storage_component_capital_cost_cny_year": capital,
        "storage_component_true_capital_cost_cny_year": true_capital,
        "storage_component_outer_approximation_gap_cny_year": (
            true_capital - capital
        ),
        "storage_native_fom_cny_year": native_fom,
        "extra_charge_interface_cost_cny_year": extra_charge,
        "extra_charge_interface_unit_cost_cny_kw_year": (
            config.extra_charge_power_cost_cny_kw_year
        ),
        "recharge_opportunity_cost_cny_year": recharge_cost,
        "storage_variable_om_cny_year": storage_vom,
        "solver_total_system_cost_cny_year": solver_total_cost,
        "total_system_cost_cny_year": total_cost,
        "baseline_gas_only_cost_cny_year": baseline_cost,
        "signed_benefit_cny_year": benefit,
        "signed_benefit_cny_per_forced_kw_year": (
            benefit / (model.pdis_mw * 1_000.0)
        ),
        "objective_reconstruction_error_cny": float(
            solver_objective_reconstruction - result.fun
        ),
        "cost_decomposition_error_cny": float(
            total_cost
            - (
                gas_fixed
                + gas_variable
                + true_capital
                + native_fom
                + extra_charge
                + recharge_cost
                + storage_vom
            )
        ),
        "lfp_replaceable_fraction": (
            replaceable_fraction
            if config.tech_id == "li_lfp"
            else np.nan
        ),
        "lfp_replacement_life_years": replacement_life,
        "lfp_fom_clean_status": (
            "unresolved_local_fom_treated_as_clean_structural_screen"
            if config.tech_id == "li_lfp"
            else "not_applicable"
        ),
        "task_day_order_sha256": stable_hash(task_order),
        "recharge_day_order_sha256": stable_hash(recharge_order),
        "paired_seed": PAIRED_SEED,
        **run_stats,
        "manuscript_evidence_admitted": False,
        "evidence_status": EVIDENCE_STATUS,
    }
    dispatch = pd.DataFrame(
        {
            "country": country,
            "year": year,
            "config_id": config.config_id,
            "cell": cell,
            "binary_mode": model.binary_mode,
            "mutual_exclusion_formulation": (
                model.mutual_exclusion_formulation
            ),
            "model_hour": np.arange(model.indices.n),
            "task_mw": model.task_mw,
            "recharge_available_mw": model.recharge_mw,
            "gas_to_task_mw": f["gas"],
            "recharge_used_mw": f["recharge_used"],
            "recharge_spill_mw": f["spill"],
            "storage_charge_mw": f["charge"],
            "storage_discharge_mw": f["discharge"],
            "soc_mwh": f["soc"],
            "task_event_day": np.repeat(event_flags, 24),
        }
    )
    cost = {
        key: row[key]
        for key in [
            "country",
            "year",
            "config_id",
            "cell",
            "binary_mode",
            "gas_fixed_cost_cny_year",
            "gas_variable_cost_cny_year",
            "storage_component_capital_cost_cny_year",
            "storage_component_true_capital_cost_cny_year",
            "storage_component_outer_approximation_gap_cny_year",
            "storage_native_fom_cny_year",
            "extra_charge_interface_cost_cny_year",
            "extra_charge_interface_unit_cost_cny_kw_year",
            "recharge_opportunity_cost_cny_year",
            "storage_variable_om_cny_year",
            "solver_total_system_cost_cny_year",
            "total_system_cost_cny_year",
            "baseline_gas_only_cost_cny_year",
            "signed_benefit_cny_year",
            "objective_reconstruction_error_cny",
            "cost_decomposition_error_cny",
            "lfp_replaceable_fraction",
            "lfp_replacement_life_years",
            "lfp_fom_clean_status",
        ]
    }
    return row, dispatch, cost


def event_ranges(flags: np.ndarray) -> list[tuple[int, int]]:
    flags = np.asarray(flags, dtype=bool)
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for day, flag in enumerate(flags):
        if flag and start is None:
            start = day
        if start is not None and not flag:
            ranges.append((start, day - 1))
            start = None
    if start is not None:
        ranges.append((start, len(flags) - 1))
    return ranges


def cyclic_indices(start: int, stop_exclusive: int, n: int) -> np.ndarray:
    if start < stop_exclusive:
        return np.arange(start, stop_exclusive, dtype=int)
    if start == stop_exclusive:
        return np.asarray([], dtype=int)
    return np.concatenate(
        [
            np.arange(start, n, dtype=int),
            np.arange(0, stop_exclusive, dtype=int),
        ]
    )


def event_diagnostics(
    dispatch: pd.DataFrame,
    row: dict[str, Any],
    event_flags: np.ndarray,
) -> pd.DataFrame:
    ranges = event_ranges(event_flags)
    if not ranges:
        raise RuntimeError("No q75 events found")
    output: list[dict[str, Any]] = []
    n = len(dispatch)
    soc = dispatch["soc_mwh"].to_numpy(float)
    recharge = dispatch["recharge_available_mw"].to_numpy(float)
    used = dispatch["recharge_used_mw"].to_numpy(float)
    spill = dispatch["recharge_spill_mw"].to_numpy(float)
    charge = dispatch["storage_charge_mw"].to_numpy(float)
    discharge = dispatch["storage_discharge_mw"].to_numpy(float)
    task = dispatch["task_mw"].to_numpy(float)
    pch = float(row["total_charge_power_mw"])
    for event_id, (start_day, end_day) in enumerate(ranges):
        next_start_day = ranges[(event_id + 1) % len(ranges)][0]
        start_hour = start_day * 24
        end_hour = (end_day + 1) * 24 - 1
        next_start_hour = next_start_day * 24
        within = np.arange(start_hour, end_hour + 1, dtype=int)
        gap_start = (end_hour + 1) % n
        gap = cyclic_indices(gap_start, next_start_hour, n)
        pre_soc = float(soc[(start_hour - 1) % n])
        next_pre_soc = float(soc[(next_start_hour - 1) % n])
        recovery_target = 0.99 * pre_soc
        recovery_h = np.nan
        recovered = False
        for offset, hour in enumerate(gap, start=1):
            if soc[hour] >= recovery_target - 1e-9:
                recovery_h = float(offset)
                recovered = True
                break
        output.append(
            {
                "country": row["country"],
                "year": row["year"],
                "config_id": row["config_id"],
                "cell": row["cell"],
                "binary_mode": row["binary_mode"],
                "event_id": event_id,
                "start_model_day": start_day,
                "end_model_day": end_day,
                "event_duration_days": end_day - start_day + 1,
                "pre_event_soc_mwh": pre_soc,
                "pre_event_soc_fraction": (
                    pre_soc / float(row["rated_energy_capacity_mwh"])
                ),
                "within_event_recharge_available_mwh": float(
                    recharge[within].sum()
                ),
                "within_event_recharge_used_mwh": float(
                    used[within].sum()
                ),
                "inter_event_recharge_available_mwh": float(
                    recharge[gap].sum()
                ),
                "inter_event_recharge_used_mwh": float(used[gap].sum()),
                "inter_event_recharge_spill_mwh": float(spill[gap].sum()),
                "inter_event_recharge_absorption_rate": (
                    float(used[gap].sum() / recharge[gap].sum())
                    if recharge[gap].sum() > 1e-12
                    else np.nan
                ),
                "charge_power_saturation_hours": int(
                    np.sum(
                        (charge[gap] >= 0.99 * pch)
                        & (spill[gap] > 1e-9)
                    )
                ),
                "recovery_time_to_pre_event_soc_h": recovery_h,
                "recovered_before_next_event": recovered,
                "recovery_right_censored": not recovered,
                "next_event_soc_mwh": next_pre_soc,
                "next_event_soc_fraction": (
                    next_pre_soc
                    / float(row["rated_energy_capacity_mwh"])
                ),
                "next_to_previous_pre_event_soc_ratio": (
                    next_pre_soc / pre_soc if pre_soc > 1e-12 else np.nan
                ),
                "event_task_mwh": float(task[within].sum()),
                "event_discharge_mwh": float(discharge[within].sum()),
                "event_task_coverage_fraction": (
                    float(discharge[within].sum() / task[within].sum())
                    if task[within].sum() > 1e-12
                    else np.nan
                ),
                "event_discharge_saturation_hours": int(
                    np.sum(
                        discharge[within]
                        >= 0.99 * float(row["forced_discharge_power_mw"])
                    )
                ),
                "event_rated_output_efc": float(
                    discharge[within].sum()
                    / float(row["rated_energy_capacity_mwh"])
                ),
                "manuscript_evidence_admitted": False,
                "evidence_status": EVIDENCE_STATUS,
            }
        )
    return pd.DataFrame(output)


def run_cell(
    frame: pd.DataFrame,
    country: str,
    year: int,
    config: Configuration,
    cell: str,
    orders: dict[str, list[int]],
    binary_mode: str,
    mutual_exclusion_formulation: str,
    replaceable_fraction: float,
    time_limit_seconds: float,
    solver: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    task_label, recharge_label = cell
    task_order = orders[task_label]
    recharge_order = orders[recharge_label]
    task_indices = hourly_order(task_order)
    recharge_indices = hourly_order(recharge_order)
    task = frame["task_c3_q75_mw"].to_numpy(float)[task_indices]
    recharge = frame["r2_potential_recharge_mw"].to_numpy(float)[
        recharge_indices
    ]
    source_event = (
        frame.groupby("source_day_id", sort=True)["event_flag_q75"]
        .first()
        .to_numpy(bool)
    )
    event_flags = source_event[np.asarray(task_order, dtype=int)]
    raw_mean = float(frame["raw_gas_mean_mw"].iloc[0])
    model, result, runtime = solve_with_lfp_outer_approximation(
        task,
        recharge,
        raw_mean,
        config,
        binary_mode,
        mutual_exclusion_formulation,
        replaceable_fraction,
        time_limit_seconds,
        solver,
    )
    row, dispatch, cost = summarize_solution(
        model,
        result,
        runtime,
        country,
        year,
        config,
        cell,
        task_order,
        recharge_order,
        event_flags,
        replaceable_fraction,
    )
    dispatch["task_source_hour_id"] = task_indices
    dispatch["task_source_day_id"] = task_indices // 24
    dispatch["task_source_hour_of_day"] = task_indices % 24
    dispatch["recharge_source_hour_id"] = recharge_indices
    dispatch["recharge_source_day_id"] = recharge_indices // 24
    dispatch["recharge_source_hour_of_day"] = recharge_indices % 24
    events = event_diagnostics(dispatch, row, event_flags)
    return row, dispatch, events, {
        "cost": cost,
        "runtime": runtime,
        "event_flags": event_flags,
    }


def relative_difference(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1.0)


def perspective_envelope_static_test() -> dict[str, Any]:
    """Verify the integer feasible set of the McCormick strengthening.

    The test covers both binary modes and zero, interior, and upper-bound
    values of the continuous charge-expansion decision.
    """
    extra_upper = 7.25
    native = 1.75
    tolerance = 1e-12
    cases: list[dict[str, Any]] = []
    for xch in [0.0, 0.37 * extra_upper, extra_upper]:
        for mode in [0.0, 1.0]:
            product = xch * mode
            envelope = {
                "w_nonnegative": product >= -tolerance,
                "w_le_xch": product <= xch + tolerance,
                "w_le_upper_z": (
                    product <= extra_upper * mode + tolerance
                ),
                "w_ge_xch_plus_upper_z_minus_upper": (
                    product
                    >= xch
                    + extra_upper * mode
                    - extra_upper
                    - tolerance
                ),
            }
            strengthened_charge_cap = native * mode + product
            original_integer_charge_cap = (
                native + xch if mode == 1.0 else 0.0
            )
            cap_equal = (
                abs(
                    strengthened_charge_cap
                    - original_integer_charge_cap
                )
                <= tolerance
            )
            cases.append(
                {
                    "xch": xch,
                    "mode": int(mode),
                    "product": product,
                    **envelope,
                    "integer_charge_capacity_equal": cap_equal,
                    "pass": bool(all(envelope.values()) and cap_equal),
                }
            )
    return {
        "test": "w_h_equals_xch_times_binary_charge_mode",
        "cases": cases,
        "all_cases_pass": bool(
            len(cases) == 6 and all(case["pass"] for case in cases)
        ),
    }


def validate_sparse_full(
    sparse: dict[str, Any],
    full: dict[str, Any],
) -> dict[str, Any]:
    metrics = [
        "total_system_cost_cny_year",
        "storage_charge_mwh",
        "storage_discharge_mwh",
        "rated_output_efc",
        "extra_charge_power_mw",
    ]
    differences = {
        f"{metric}_relative_difference": relative_difference(
            float(sparse[metric]), float(full[metric])
        )
        for metric in metrics
    }
    sparse_lower = float(
        sparse["solver_objective_bound_total_cost_cny_year"]
    )
    sparse_upper = float(sparse["solver_total_system_cost_cny_year"])
    full_lower = float(
        full["solver_objective_bound_total_cost_cny_year"]
    )
    full_upper = float(full["solver_total_system_cost_cny_year"])
    scale = max(
        abs(sparse_lower),
        abs(sparse_upper),
        abs(full_lower),
        abs(full_upper),
        1.0,
    )
    interval_separation = max(
        0.0,
        max(sparse_lower, full_lower)
        - min(sparse_upper, full_upper),
    )
    certified_point_tolerance = (
        max(0.0, sparse_upper - sparse_lower)
        + max(0.0, full_upper - full_lower)
        + FULL_BINARY_OBJECTIVE_RELATIVE_TOLERANCE * scale
    )
    point_difference = abs(sparse_upper - full_upper)
    return {
        "country": sparse["country"],
        "year": sparse["year"],
        "config_id": sparse["config_id"],
        "cell": sparse["cell"],
        **differences,
        "sparse_objective_lower_bound_cny_year": sparse_lower,
        "sparse_objective_upper_bound_cny_year": sparse_upper,
        "full_objective_lower_bound_cny_year": full_lower,
        "full_objective_upper_bound_cny_year": full_upper,
        "objective_interval_separation_cny_year": interval_separation,
        "objective_point_difference_cny_year": point_difference,
        "objective_certified_point_tolerance_cny_year": (
            certified_point_tolerance
        ),
        "objective_equivalent": (
            interval_separation
            <= FULL_BINARY_OBJECTIVE_RELATIVE_TOLERANCE * scale
            and point_difference <= certified_point_tolerance
        ),
        "aggregates_equivalent": all(
            differences[f"{metric}_relative_difference"]
            <= FULL_BINARY_AGGREGATE_RELATIVE_TOLERANCE
            for metric in metrics[1:]
        ),
        "sparse_no_simultaneous": (
            sparse["simultaneous_energy_ratio"]
            <= SIMULTANEOUS_RATIO_TOLERANCE
        ),
        "full_no_simultaneous": (
            full["simultaneous_energy_ratio"]
            <= SIMULTANEOUS_RATIO_TOLERANCE
        ),
        "sparse_hourly_balances_pass": max(
            abs(float(sparse[column]))
            for column in [
                "maximum_task_balance_error_mw",
                "maximum_charge_source_error_mw",
                "maximum_recharge_balance_error_mw",
                "maximum_soc_balance_error_mwh",
            ]
        )
        <= ABSOLUTE_BALANCE_TOLERANCE,
        "full_hourly_balances_pass": max(
            abs(float(full[column]))
            for column in [
                "maximum_task_balance_error_mw",
                "maximum_charge_source_error_mw",
                "maximum_recharge_balance_error_mw",
                "maximum_soc_balance_error_mwh",
            ]
        )
        <= ABSOLUTE_BALANCE_TOLERANCE,
        "sparse_cyclic_soc_pass": (
            abs(float(sparse["cyclic_soc_difference_mwh"]))
            <= ABSOLUTE_BALANCE_TOLERANCE
        ),
        "full_cyclic_soc_pass": (
            abs(float(full["cyclic_soc_difference_mwh"]))
            <= ABSOLUTE_BALANCE_TOLERANCE
        ),
        "sparse_objective_reconstruction_pass": (
            abs(float(sparse["objective_reconstruction_error_cny"]))
            <= 1e-4
        ),
        "full_objective_reconstruction_pass": (
            abs(float(full["objective_reconstruction_error_cny"]))
            <= 1e-4
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage0", type=Path, default=STAGE0)
    parser.add_argument("--stage0-qa", type=Path, default=STAGE0_QA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--country-years",
        default="DE:2023,SI:2018",
    )
    parser.add_argument(
        "--configs",
        default="li_lfp_4h,caes_24h",
    )
    parser.add_argument("--cells", default="DD,CD,DC,CC")
    parser.add_argument(
        "--full-binary-validation-cells",
        default="DE:2023:li_lfp_4h:DD,SI:2018:caes_24h:CC",
    )
    parser.add_argument(
        "--lfp-replaceable-fraction",
        type=float,
        default=LFP_REPLACEABLE_FRACTION,
    )
    parser.add_argument(
        "--time-limit-seconds",
        type=float,
        default=900.0,
    )
    parser.add_argument(
        "--solver",
        choices=["gurobi", "scipy_highs"],
        default="gurobi",
    )
    parser.add_argument(
        "--mutual-exclusion-formulation",
        choices=["indicator", "bigm", "perspective_bigm"],
        default="perspective_bigm",
    )
    parser.add_argument(
        "--charge-interface-cost-mode",
        choices=["technology_specific", "common_lfp", "common_caes"],
        default="technology_specific",
        help=(
            "The default preserves the source-library technology-specific "
            "screen. Common modes are structural cost sensitivities."
        ),
    )
    args = parser.parse_args()

    if not args.stage0.exists():
        raise FileNotFoundError(f"Stage-0 table not found: {args.stage0}")
    if (
        args.solver == "scipy_highs"
        and args.mutual_exclusion_formulation != "bigm"
        and args.mutual_exclusion_formulation != "perspective_bigm"
    ):
        raise RuntimeError(
            "scipy_highs cannot enforce native indicator constraints; "
            "use bigm or perspective_bigm"
        )
    perspective_static_test = perspective_envelope_static_test()
    if (
        args.mutual_exclusion_formulation == "perspective_bigm"
        and not perspective_static_test["all_cases_pass"]
    ):
        raise RuntimeError("Perspective-envelope static equivalence failed")
    stage0_qa = check_stage0_qa(args.stage0_qa, args.stage0)
    output = args.output
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(
            "Stage-1 output directory must be absent or empty to prevent "
            f"stale-file contamination: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    dispatch_dir = output / "dispatch"
    dispatch_dir.mkdir(parents=True, exist_ok=True)

    country_years = parse_country_years(args.country_years)
    config_ids = [
        token.strip() for token in args.configs.split(",") if token.strip()
    ]
    cells = [
        token.strip() for token in args.cells.split(",") if token.strip()
    ]
    if set(cells) != {"DD", "CD", "DC", "CC"}:
        raise RuntimeError(
            "Stage-1 frozen paired factorial requires DD, CD, DC and CC"
        )
    validation_cells = parse_validation_cells(
        args.full_binary_validation_cells
    )
    configurations = {
        config_id: load_configuration(config_id)
        for config_id in config_ids
    }
    if set(configurations) != {"li_lfp_4h", "caes_24h"}:
        raise RuntimeError(
            "Stage-1 requires exactly li_lfp_4h and caes_24h"
        )
    source_charge_costs = {
        config_id: config.extra_charge_power_cost_cny_kw_year
        for config_id, config in configurations.items()
    }
    if args.charge_interface_cost_mode != "technology_specific":
        anchor = (
            "li_lfp_4h"
            if args.charge_interface_cost_mode == "common_lfp"
            else "caes_24h"
        )
        common_cost = source_charge_costs[anchor]
        configurations = {
            config_id: replace(
                config,
                extra_charge_power_cost_cny_kw_year=common_cost,
            )
            for config_id, config in configurations.items()
        }

    result_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    event_frames: list[pd.DataFrame] = []
    chronology_rows: list[dict[str, Any]] = []
    sparse_lookup: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    frames: dict[tuple[str, int], pd.DataFrame] = {}
    orders_by_country_year: dict[
        tuple[str, int], dict[str, list[int]]
    ] = {}

    for country, year in country_years:
        frame = load_country_year(args.stage0, country, year)
        orders, daily = make_day_orders(frame, country, year)
        frames[(country, year)] = frame
        orders_by_country_year[(country, year)] = orders
        source_event = daily["event"].to_numpy(bool)
        source_task = frame["task_c3_q75_mw"].to_numpy(float).reshape(
            365, 24
        )
        source_recharge = frame[
            "r2_potential_recharge_mw"
        ].to_numpy(float).reshape(365, 24)
        for label, order in orders.items():
            flags = source_event[np.asarray(order, dtype=int)]
            chronology_rows.append(
                {
                    "country": country,
                    "year": year,
                    "chronology": label,
                    "day_order_sha256": stable_hash(order),
                    "task_energy_mwh": float(
                        source_task[np.asarray(order, dtype=int)].sum()
                    ),
                    "recharge_energy_mwh": float(
                        source_recharge[
                            np.asarray(order, dtype=int)
                        ].sum()
                    ),
                    **event_run_statistics(flags),
                }
            )
        if event_run_statistics(
            source_event[np.asarray(orders["D"], dtype=int)]
        )["maximum_event_run_days"] != 1:
            raise RuntimeError(f"{country}-{year}: dispersed gate failed")
        if event_run_statistics(
            source_event[np.asarray(orders["C"], dtype=int)]
        )["maximum_event_run_days"] > CLUSTER_DAYS:
            raise RuntimeError(f"{country}-{year}: clustered gate failed")

        for config_id, config in configurations.items():
            for cell in cells:
                row, dispatch, events, extra = run_cell(
                    frame,
                    country,
                    year,
                    config,
                    cell,
                    orders,
                    "sparse",
                    args.mutual_exclusion_formulation,
                    args.lfp_replaceable_fraction,
                    args.time_limit_seconds,
                    args.solver,
                )
                result_rows.append(row)
                cost_rows.append(extra["cost"])
                runtime_rows.append(
                    {
                        "country": country,
                        "year": year,
                        "config_id": config_id,
                        "cell": cell,
                        "binary_mode": "sparse",
                        **{
                            key: value
                            for key, value in extra["runtime"].items()
                            if key != "iterations"
                        },
                        "iterations_json": json.dumps(
                            json_ready(extra["runtime"]["iterations"]),
                            sort_keys=True,
                        ),
                    }
                )
                event_frames.append(events)
                sparse_lookup[(country, year, config_id, cell)] = row
                dispatch.to_csv(
                    dispatch_dir
                    / f"{country}_{year}_{config_id}_{cell}_sparse.csv.gz",
                    index=False,
                    compression="gzip",
                )

    validation_rows: list[dict[str, Any]] = []
    for key in sorted(validation_cells):
        country, year, config_id, cell = key
        if key not in sparse_lookup:
            raise RuntimeError(
                f"Full-binary validation cell not in sparse suite: {key}"
            )
        frame = frames[(country, year)]
        config = configurations[config_id]
        orders = orders_by_country_year[(country, year)]
        row, dispatch, events, extra = run_cell(
            frame,
            country,
            year,
            config,
            cell,
            orders,
            "full",
            args.mutual_exclusion_formulation,
            args.lfp_replaceable_fraction,
            args.time_limit_seconds,
            args.solver,
        )
        result_rows.append(row)
        cost_rows.append(extra["cost"])
        runtime_rows.append(
            {
                "country": country,
                "year": year,
                "config_id": config_id,
                "cell": cell,
                "binary_mode": "full",
                **{
                    item_key: value
                    for item_key, value in extra["runtime"].items()
                    if item_key != "iterations"
                },
                "iterations_json": json.dumps(
                    json_ready(extra["runtime"]["iterations"]),
                    sort_keys=True,
                ),
            }
        )
        event_frames.append(events)
        dispatch.to_csv(
            dispatch_dir
            / f"{country}_{year}_{config_id}_{cell}_full.csv.gz",
            index=False,
            compression="gzip",
        )
        validation_rows.append(
            validate_sparse_full(sparse_lookup[key], row)
        )

    results = pd.DataFrame(result_rows)
    costs = pd.DataFrame(cost_rows)
    runtimes = pd.DataFrame(runtime_rows)
    events = pd.concat(event_frames, ignore_index=True)
    chronologies = pd.DataFrame(chronology_rows)
    validations = pd.DataFrame(validation_rows)
    results.to_csv(output / "model_results.csv", index=False)
    costs.to_csv(output / "cost_decomposition.csv", index=False)
    runtimes.to_csv(output / "runtime_rss.csv", index=False)
    events.to_csv(
        output / "event_soc_recharge_diagnostics.csv.gz",
        index=False,
        compression="gzip",
    )
    chronologies.to_csv(
        output / "chronology_diagnostics.csv", index=False
    )
    validations.to_csv(
        output / "sparse_full_binary_validation.csv", index=False
    )

    sparse = results[results["binary_mode"].eq("sparse")]
    required_sparse = len(country_years) * len(config_ids) * len(cells)
    balance_columns = [
        "task_balance_error_mwh",
        "charge_source_error_mwh",
        "recharge_balance_error_mwh",
        "soc_balance_error_mwh",
    ]
    relative_columns = [
        "task_balance_relative_error",
        "charge_source_relative_error",
        "recharge_balance_relative_error",
        "soc_balance_relative_error",
    ]
    hourly_balance_columns = [
        "maximum_task_balance_error_mw",
        "maximum_charge_source_error_mw",
        "maximum_recharge_balance_error_mw",
        "maximum_soc_balance_error_mwh",
    ]
    qa_gates = {
        "stage0_qa_admitted": True,
        "exactly_132_stage0_country_years_expected_upstream": (
            int(
                stage0_qa.get("coverage", {}).get(
                    "main_country_years", -1
                )
            )
            == 132
        ),
        "two_stage1_country_years": len(country_years) == 2,
        "required_country_years_are_DE2023_and_SI2018": (
            set(country_years) == {("DE", 2023), ("SI", 2018)}
        ),
        "all_four_paired_cells_present": set(cells)
        == {"DD", "CD", "DC", "CC"},
        "all_sparse_cells_present": len(sparse) == required_sparse,
        "all_sparse_solver_success": sparse["solver_status"].eq(0).all(),
        "all_sparse_simultaneous_hourly_zero": (
            sparse["maximum_simultaneous_charge_discharge_mw"]
            <= SIMULTANEOUS_HOURLY_TOLERANCE_MW
        ).all(),
        "all_sparse_simultaneous_ratio_zero": (
            sparse["simultaneous_energy_ratio"]
            <= SIMULTANEOUS_RATIO_TOLERANCE
        ).all(),
        "all_absolute_balances_pass": (
            sparse[balance_columns].abs().max().max()
            <= ABSOLUTE_BALANCE_TOLERANCE
        ),
        "all_relative_balances_pass": (
            sparse[relative_columns].abs().max().max()
            <= RELATIVE_BALANCE_TOLERANCE
        ),
        "all_maximum_hourly_balances_pass": (
            sparse[hourly_balance_columns].abs().max().max()
            <= ABSOLUTE_BALANCE_TOLERANCE
        ),
        "all_cyclic_soc_pass": (
            sparse["cyclic_soc_difference_mwh"].abs().max()
            <= ABSOLUTE_BALANCE_TOLERANCE
        ),
        "all_lfp_piecewise_gaps_pass": (
            sparse.loc[
                sparse["config_id"].eq("li_lfp_4h"),
                "pwl_relative_gap",
            ].max()
            <= PWL_RELATIVE_GAP_TOLERANCE
        ),
        "all_objective_reconstructions_pass": (
            sparse["objective_reconstruction_error_cny"].abs().max()
            <= 1e-4
        ),
        "all_cost_decompositions_pass": (
            sparse["cost_decomposition_error_cny"].abs().max() <= 1e-6
        ),
        "full_binary_validation_cells_present": (
            len(validations) == len(validation_cells)
        ),
        "full_binary_objectives_equivalent": (
            not validations.empty
            and validations["objective_equivalent"].all()
        ),
        "full_binary_aggregates_equivalent": (
            not validations.empty
            and validations["aggregates_equivalent"].all()
        ),
        "full_binary_no_simultaneous": (
            not validations.empty
            and validations["full_no_simultaneous"].all()
        ),
        "full_binary_hourly_balances_pass": (
            not validations.empty
            and validations["full_hourly_balances_pass"].all()
        ),
        "full_binary_cyclic_soc_pass": (
            not validations.empty
            and validations["full_cyclic_soc_pass"].all()
        ),
        "full_binary_objective_reconstruction_pass": (
            not validations.empty
            and validations[
                "full_objective_reconstruction_pass"
            ].all()
        ),
        "paired_smoke_scope_explicit": True,
        "scope_does_not_claim_balanced_orthogonal_factorial": True,
        "perspective_static_equivalence_pass": (
            perspective_static_test["all_cases_pass"]
        ),
        "all_outputs_non_evidential": (
            (~results["manuscript_evidence_admitted"]).all()
            and results["evidence_status"].eq(EVIDENCE_STATUS).all()
        ),
    }
    qa = {
        "script_version": "observed_gas_event_task_stage1_v0_2",
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "qa_pass": bool(all(qa_gates.values())),
        "qa_gates": qa_gates,
        "stage0_path": str(args.stage0.resolve()),
        "stage0_sha256": sha256(args.stage0),
        "expected_stage0_sha256": EXPECTED_STAGE0_TABLE_SHA256,
        "stage0_qa_path": str(args.stage0_qa.resolve()),
        "stage0_qa_sha256": sha256(args.stage0_qa),
        "stage0_frozen_hashes": stage0_qa.get("hashes", {}),
        "configuration_path": str(CONFIGURATIONS.resolve()),
        "configuration_sha256": sha256(CONFIGURATIONS),
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": sha256(Path(__file__)),
        "country_years": [
            {"country": country, "year": year}
            for country, year in country_years
        ],
        "configurations": {
            key: asdict(value) for key, value in configurations.items()
        },
        "cells": cells,
        "full_binary_validation_cells": [
            list(key) for key in sorted(validation_cells)
        ],
        "lfp_replaceable_fraction": args.lfp_replaceable_fraction,
        "lfp_replaceable_fraction_sensitivity": (
            LFP_REPLACEABLE_FRACTION_SENSITIVITY
        ),
        "lfp_cost_interpretation": (
            "component-specific calendar-or-cycle capital recovery; "
            "local FOM is not augmentation-clean and all costs are a "
            "non-evidential structural screen"
        ),
        "task_contract": (
            "C3-q75 is a deterministic non-negative sub-quantity of "
            "Energy-Charts actual aggregate Fossil gas; R2 is separate"
        ),
        "gas_charging_allowed": False,
        "cyclic_soc": True,
        "common_charge_expansion_capacity_rule": True,
        "charge_interface_cost_mode": args.charge_interface_cost_mode,
        "source_charge_interface_unit_costs_cny_kw_year": (
            source_charge_costs
        ),
        "applied_charge_interface_unit_costs_cny_kw_year": {
            key: value.extra_charge_power_cost_cny_kw_year
            for key, value in configurations.items()
        },
        "charge_interface_cost_caveat": (
            "The source library gives LFP and CAES different empirical "
            "power-side cost evidence. This Stage-1 comparison is a "
            "non-evidential structural screen, not a claim of complete "
            "like-for-like charging-interface cost measurement."
        ),
        "mutual_exclusion": (
            f"{args.mutual_exclusion_formulation} constraints on sparse "
            "overlap-hour binaries with selected 8760-binary validation"
        ),
        "perspective_static_equivalence": perspective_static_test,
        "solver": args.solver,
        "mip_relative_gap_tolerance": MIP_RELATIVE_GAP_TOLERANCE,
        "mip_absolute_gap_cny_per_forced_kw_year": (
            MIP_ABSOLUTE_GAP_CNY_PER_FORCED_KW_YEAR
        ),
        "stage1_scope": (
            "paired chronology structural smoke only; the independently "
            "phase-balanced orthogonal factorial is deferred to Stage 3"
        ),
        "balanced_orthogonal_factorial_executed": False,
        "sparse_result_rows": int(len(sparse)),
        "full_validation_result_rows": int(
            results["binary_mode"].eq("full").sum()
        ),
        "event_diagnostic_rows": int(len(events)),
        "maximum_absolute_balance_error": float(
            sparse[balance_columns].abs().max().max()
        ),
        "maximum_relative_balance_error": float(
            sparse[relative_columns].abs().max().max()
        ),
        "maximum_hourly_balance_error": float(
            sparse[hourly_balance_columns].abs().max().max()
        ),
        "maximum_simultaneous_energy_ratio": float(
            results["simultaneous_energy_ratio"].max()
        ),
        "maximum_lfp_piecewise_relative_gap": float(
            sparse.loc[
                sparse["config_id"].eq("li_lfp_4h"),
                "pwl_relative_gap",
            ].max()
        ),
        "maximum_sparse_full_objective_relative_difference": (
            float(
                validations[
                    "total_system_cost_cny_year_relative_difference"
                ].max()
            )
            if not validations.empty
            else None
        ),
        "maximum_sparse_full_aggregate_relative_difference": (
            float(
                validations[
                    [
                        column
                        for column in validations.columns
                        if column.endswith("_relative_difference")
                        and not column.startswith("total_system_cost")
                    ]
                ]
                .max()
                .max()
            )
            if not validations.empty
            else None
        ),
        "maximum_peak_rss_mb": float(runtimes["peak_rss_after_mb"].max()),
        "maximum_single_solve_seconds": float(
            results["solve_seconds"].max()
        ),
        "evidence_status": EVIDENCE_STATUS,
        "manuscript_evidence_admitted": False,
    }
    (output / "QA.json").write_text(
        json.dumps(json_ready(qa), indent=2, ensure_ascii=False) + "\n"
    )
    manifest = {
        "output_files": {
            "model_results": "model_results.csv",
            "cost_decomposition": "cost_decomposition.csv",
            "runtime_rss": "runtime_rss.csv",
            "event_diagnostics": "event_soc_recharge_diagnostics.csv.gz",
            "chronology_diagnostics": "chronology_diagnostics.csv",
            "sparse_full_validation": "sparse_full_binary_validation.csv",
            "qa": "QA.json",
            "dispatch_directory": "dispatch",
        },
        "sha256": {
            str(path.relative_to(output)): sha256(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "run_manifest.json"
        },
        "evidence_status": EVIDENCE_STATUS,
        "manuscript_evidence_admitted": False,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(json_ready(manifest), indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(json_ready(qa), indent=2, ensure_ascii=False))
    if not qa["qa_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
