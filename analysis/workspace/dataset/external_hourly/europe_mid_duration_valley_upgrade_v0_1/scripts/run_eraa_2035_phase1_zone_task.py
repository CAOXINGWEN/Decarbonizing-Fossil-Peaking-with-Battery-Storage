#!/usr/bin/env python3
"""Run one auditable ERAA-2035 atomic-zone storage screen.

The runner is the smallest executable bridge from the standardized ERAA
weather shards to the confirmatory LP.  It deliberately requires the
Europe-wide synchronized day-order file even when only one atomic zone is
solved.  Thus independent-zone screens remain two components of the same
pan-European chronology intervention rather than country-specific treatments.

The current runner represents open-loop, pondage and reservoir hydropower as
one aggregate reservoir, and run-of-river inflow as spillable renewable
potential.  Closed-loop pumping and the released period-energy/minimum-hydro
constraints are recorded in the task manifest but are not yet endogenous.
Those omissions make a pre-EVA run a development check, never manuscript
evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


SCRIPT_VERSION = "0.1.0"
TARGET_YEAR = 2035
WEATHER_SCENARIOS = tuple(f"WS{number:02d}" for number in range(1, 37))
CHRONOLOGIES = ("dispersed", "clustered")
THERMAL_TECHNOLOGIES = (
    "gas_ccgt",
    "gas_ocgt",
    "hydrogen_ccgt",
    "hydrogen_ocgt",
    "nuclear",
    "hard_coal",
    "lignite",
    "light_oil",
    "heavy_oil",
)
RESERVOIR_CODES = ("HRI", "HOL", "HPI")
HYDRO_CODE_TO_CAPACITY_LABEL = {
    "HCL": "Closed loop pumping",
    "HOL": "Open loop pumping",
    "HPI": "Pondage",
    "HRI": "Reservoir",
    "HRR": "Run of river",
}
INCLUDED_STATUS = {
    "available on market",
    "inelastic supply / fixed profile",
    "out of market - for pv/battery dispatch optimization",
}

MODULE_PATH = Path(__file__).with_name("eraa_confirmatory_model_core.py")
EUROPE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = EUROPE_ROOT / "processed" / "eraa_2035_phase1_inputs_v0_1"
DEFAULT_CHRONOLOGY = (
    EUROPE_ROOT / "processed" / "eraa_2035_synchronized_chronologies_v0_1"
)
DEFAULT_DISPATCH = (
    EUROPE_ROOT / "processed" / "eraa_2035_dispatch_parameters_v0_1"
)
DEFAULT_CATALOG = (
    EUROPE_ROOT
    / "model_adapter"
    / "energy_charts_2015_2024_representative_pilot"
    / "configuration_catalog_53.csv"
)
DEFAULT_TECH_LIBRARY = (
    REPOSITORY_ROOT / "dataset" / "M2_LDES_technology_parameter_library_v0_2.csv"
)


def load_core():
    name = "eraa_confirmatory_model_core_for_zone_task"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(json_ready(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def write_csv(frame: pd.DataFrame, path: Path, *, compressed: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".csv.gz" if compressed else ".csv"
    if compressed and not str(path).endswith(suffix):
        raise ValueError(f"compressed CSV path must end in {suffix}: {path}")
    with tempfile.NamedTemporaryFile(
        mode="wb" if compressed else "w",
        encoding=None if compressed else "utf-8",
        newline=None if compressed else "",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        if compressed:
            handle.close()
            frame.to_csv(
                temporary,
                index=False,
                compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
                float_format="%.12g",
            )
        else:
            frame.to_csv(
                handle,
                index=False,
                lineterminator="\n",
                float_format="%.12g",
            )
    os.replace(temporary, path)


def prepare_output(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"output exists: {path}; pass --force")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def parse_config_ids(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        result.extend(part.strip() for part in value.split(",") if part.strip())
    if not result:
        raise ValueError("at least one --config-id is required")
    if len(set(result)) != len(result):
        raise ValueError("duplicate configuration IDs")
    return result


def parse_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def hash_order(order: Iterable[int]) -> str:
    payload = ",".join(str(int(value)) for value in order).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def load_day_order(
    chronology_root: Path,
    weather_scenario: str,
    chronology: str,
) -> tuple[np.ndarray, str]:
    path = chronology_root / "synchronized_day_orders.csv"
    frame = pd.read_csv(path)
    selected = frame.loc[
        frame["weather_scenario"].astype(str).eq(weather_scenario)
        & frame["chronology"].astype(str).eq(chronology)
    ].sort_values("new_day")
    if len(selected) != 365:
        raise ValueError(
            f"{path}: {weather_scenario}/{chronology} has {len(selected)} days"
        )
    new_day = pd.to_numeric(selected["new_day"], errors="raise").astype(int)
    source_day = pd.to_numeric(selected["source_day"], errors="raise").astype(int)
    if new_day.tolist() != list(range(365)):
        raise ValueError("synchronized order has a noncanonical new-day axis")
    if sorted(source_day.tolist()) != list(range(365)):
        raise ValueError("synchronized order is not a 365-day permutation")
    hashes = selected["order_sha256"].astype(str).unique()
    if len(hashes) != 1:
        raise ValueError("synchronized order carries more than one hash")
    calculated = hash_order(source_day)
    if calculated != hashes[0]:
        raise ValueError("synchronized order hash mismatch")
    if "order_scope" in selected and not selected["order_scope"].astype(str).eq(
        "same_order_for_all_52_demand_nodes"
    ).all():
        raise ValueError("order is not labelled as the all-52-node intervention")
    return source_day.to_numpy(dtype=int), calculated


def reorder_days(values: np.ndarray, order: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.shape != (8_760,):
        raise ValueError(f"expected 8760 values, observed {values.shape}")
    return values.reshape(365, 24)[order].reshape(8_760)


def read_zone_weather(
    input_root: Path,
    zone: str,
    weather_scenario: str,
) -> pd.DataFrame:
    path = (
        input_root
        / "hourly_profiles"
        / f"zone={zone}"
        / "hourly_profiles.csv.gz"
    )
    header = pd.read_csv(path, nrows=0)
    required = {
        "target_year",
        "weather_scenario",
        "zone",
        "hour_of_year",
        "load_mw",
        "wind_potential_mw",
        "solar_potential_mw",
    }
    missing = sorted(required.difference(header.columns))
    if missing:
        raise ValueError(f"{path}: missing {missing}")
    optional = {
        "csp_with_storage_dispatched_reference_mw",
        "csp_with_storage_predispatch_mw",
        "btm_surplus_potential_mw",
    }
    frame = pd.read_csv(path, usecols=sorted(required | (optional & set(header.columns))))
    frame = frame.loc[
        frame["weather_scenario"].astype(str).eq(weather_scenario)
    ].sort_values("hour_of_year")
    if len(frame) != 8_760:
        raise ValueError(f"{path}: expected 8760 selected rows, observed {len(frame)}")
    if set(frame["zone"].astype(str)) != {zone}:
        raise ValueError(f"{path}: zone mismatch")
    if set(pd.to_numeric(frame["target_year"], errors="raise").astype(int)) != {
        TARGET_YEAR
    }:
        raise ValueError(f"{path}: target-year mismatch")
    if not np.array_equal(
        pd.to_numeric(frame["hour_of_year"], errors="raise").astype(int),
        np.arange(8_760),
    ):
        raise ValueError(f"{path}: noncanonical hour axis")
    return frame.reset_index(drop=True)


def selected_capacity(input_root: Path, zone: str) -> pd.DataFrame:
    path = input_root / "capacity_2035_final.csv"
    frame = pd.read_csv(path)
    selected = frame.loc[
        frame["Market_Node"].astype(str).eq(zone)
        & pd.to_numeric(frame["Target year"], errors="raise").astype(int).eq(TARGET_YEAR)
        & frame["Operational_Status"].astype(str).str.strip().str.lower().isin(
            INCLUDED_STATUS
        )
    ].copy()
    selected["Value"] = pd.to_numeric(selected["Value"], errors="raise")
    return selected


def capacity_by_hydro_code(input_root: Path, zone: str) -> dict[str, float]:
    selected = selected_capacity(input_root, zone)
    by_label = selected.groupby("Technology")["Value"].sum().to_dict()
    return {
        code: float(by_label.get(label, 0.0))
        for code, label in HYDRO_CODE_TO_CAPACITY_LABEL.items()
    }


def hydro_storage_capacity_mwh(input_root: Path, zone: str) -> float:
    path = input_root / "hydro_capacity_2035_final.csv"
    frame = pd.read_csv(path)
    selected = frame.loc[
        frame["MARKET_NODE"].astype(str).eq(zone)
        & pd.to_numeric(frame["TARGET_YEAR"], errors="raise").astype(int).eq(
            TARGET_YEAR
        )
        & frame["PEMMDB_PLANT_TYPE"].astype(str).str.strip().str.lower().isin(
            {
                "reservoir",
                "open loop pumping",
                "pondage",
            }
        )
    ].copy()
    storage = pd.to_numeric(
        selected["Storage Capacity [TWh]"], errors="coerce"
    ).fillna(0.0)
    if (storage < 0).any():
        raise ValueError("negative released hydro storage capacity")
    return float(storage.sum() * 1_000_000.0)


def period_bounds(period: int, resolution: str) -> tuple[int, int]:
    if period <= 0:
        raise ValueError(f"non-positive hydro period {period}")
    resolution = resolution.strip().lower()
    hours = 24 if resolution == "day" else 168 if resolution == "week" else None
    if hours is None:
        raise ValueError(f"unknown hydro time resolution {resolution}")
    start = (period - 1) * hours
    stop = min(period * hours, 8_760)
    if start >= 8_760 or stop <= start:
        raise ValueError(
            f"hydro period {period}/{resolution} lies outside the 8760-hour year"
        )
    return start, stop


def hydro_periodic_arrays(
    input_root: Path,
    zone: str,
    weather_scenario: str,
    capacities: dict[str, float],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = (
        input_root
        / "hydro_periodic"
        / f"zone={zone}"
        / "hydro_periodic.csv.gz"
    )
    if not path.exists():
        zero = np.zeros(8_760, dtype=float)
        power = sum(capacities.get(code, 0.0) for code in RESERVOIR_CODES)
        return {
            "reservoir_inflow_mwh": zero.copy(),
            "ror_potential_mw": zero.copy(),
            "reservoir_available_power_mw": np.full(8_760, power),
        }, {
            "hydro_periodic_member_present": False,
            "period_energy_constraints_present_but_not_modelled": False,
            "minimum_constraints_present_but_not_modelled": False,
        }
    frame = pd.read_csv(path)
    frame = frame.loc[
        frame["weather_scenario"].astype(str).eq(weather_scenario)
    ].copy()
    inflow = np.zeros(8_760, dtype=float)
    ror = np.zeros(8_760, dtype=float)
    max_power_by_code: dict[str, np.ndarray] = {}
    constraint_seen: dict[str, bool] = {}
    for code in RESERVOIR_CODES:
        max_power_by_code[code] = np.full(
            8_760, float(capacities.get(code, 0.0)), dtype=float
        )
        constraint_seen[code] = False

    for row in frame.itertuples(index=False):
        code = str(row.hydro_code)
        start, stop = period_bounds(int(row.period), str(row.time_resolution))
        period_hours = stop - start
        raw_inflow = pd.to_numeric(
            pd.Series([getattr(row, "inflow_raw")]), errors="coerce"
        ).iloc[0]
        if pd.notna(raw_inflow):
            hourly = float(raw_inflow) / period_hours
            if hourly < 0:
                raise ValueError(f"{zone}/{weather_scenario}/{code}: negative inflow")
            if code == "HRR":
                ror[start:stop] += hourly
            elif code in RESERVOIR_CODES:
                inflow[start:stop] += hourly
        raw_max = pd.to_numeric(
            pd.Series([getattr(row, "max_turb_raw")]), errors="coerce"
        ).iloc[0]
        if code in RESERVOIR_CODES and pd.notna(raw_max):
            power_mw = float(raw_max) * 1_000.0
            if power_mw < 0:
                raise ValueError(f"{zone}/{weather_scenario}/{code}: negative MAX_TURB")
            max_power_by_code[code][start:stop] = power_mw
            constraint_seen[code] = True

    aggregate_power = np.sum(
        np.stack(list(max_power_by_code.values())), axis=0
    )
    period_energy_columns = [
        "max_turb_en_raw",
        "min_turb_en_raw",
    ]
    minimum_columns = [
        "min_turb_raw",
        "min_pump_raw",
        "min_res_raw",
    ]
    audit = {
        "hydro_periodic_member_present": True,
        "hydro_codes": sorted(frame["hydro_code"].astype(str).unique()),
        "hydro_constraint_seen_by_code": constraint_seen,
        "period_energy_constraints_present_but_not_modelled": bool(
            frame[period_energy_columns].notna().any().any()
        ),
        "minimum_constraints_present_but_not_modelled": bool(
            frame[minimum_columns].notna().any().any()
        ),
        "pumping_constraints_present_but_not_modelled": bool(
            frame[["max_pump_raw", "min_pump_raw"]].notna().any().any()
        ),
        "reservoir_level_constraints_present_but_not_modelled": bool(
            frame[["max_res_raw", "min_res_raw"]].notna().any().any()
        ),
        "closed_loop_pumping_capacity_mw_not_modelled": float(
            capacities.get("HCL", 0.0)
        ),
        "annual_reservoir_inflow_mwh": float(inflow.sum()),
        "annual_ror_potential_mwh": float(ror.sum()),
        "maximum_aggregate_turbine_power_mw": float(aggregate_power.max()),
    }
    return {
        "reservoir_inflow_mwh": inflow,
        "ror_potential_mw": ror,
        "reservoir_available_power_mw": aggregate_power,
    }, audit


def load_thermal_parameters(
    dispatch_root: Path,
    zone: str,
    eva_case: str,
    split_case: str,
) -> dict[str, np.ndarray | float]:
    capacity = pd.read_csv(dispatch_root / "thermal_capacity_split_cases.csv")
    selected = capacity.loc[
        capacity["zone"].astype(str).eq(zone)
        & capacity["eva_case"].astype(str).eq(eva_case)
        & capacity["split_case"].astype(str).eq(split_case)
    ].copy()
    if set(selected["technology"].astype(str)) != set(THERMAL_TECHNOLOGIES):
        raise ValueError(
            f"missing thermal capacity rows for {zone}/{eva_case}/{split_case}"
        )
    capacity_series = selected.set_index("technology")["capacity_mw"]
    existing = np.array(
        [float(capacity_series.loc[name]) for name in THERMAL_TECHNOLOGIES]
    )

    variable = pd.read_csv(
        dispatch_root / "thermal_variable_costs_2035_eur2024.csv"
    )
    selected_cost = variable.loc[variable["zone"].astype(str).eq(zone)]
    cost_series = selected_cost.set_index("technology")[
        "marginal_cost_eur2024_per_mwh"
    ]
    if not set(THERMAL_TECHNOLOGIES).issubset(cost_series.index):
        raise ValueError(f"missing thermal variable costs for {zone}")
    costs = np.array(
        [float(cost_series.loc[name]) for name in THERMAL_TECHNOLOGIES]
    )

    availability = pd.read_csv(
        dispatch_root / "thermal_expected_availability_daily.csv.gz"
    )
    selected_availability = availability.loc[
        availability["zone"].astype(str).eq(zone)
        & availability["eva_case"].astype(str).eq(eva_case)
        & availability["split_case"].astype(str).eq(split_case)
    ].copy()
    daily = np.empty((len(THERMAL_TECHNOLOGIES), 365), dtype=float)
    for index, technology in enumerate(THERMAL_TECHNOLOGIES):
        one = selected_availability.loc[
            selected_availability["technology"].astype(str).eq(technology)
        ].sort_values("day_of_year")
        if len(one) != 365 or not np.array_equal(
            pd.to_numeric(one["day_of_year"], errors="raise").astype(int),
            np.arange(365),
        ):
            raise ValueError(
                f"invalid availability axis for {zone}/{technology}"
            )
        daily[index] = pd.to_numeric(
            one["expected_available_fraction"], errors="raise"
        )
    hourly = np.repeat(daily, 24, axis=1)
    return {
        "existing_capacity_mw": existing,
        "variable_cost_eur_mwh": costs,
        "availability_fraction": hourly,
    }


def firm_addition_cost(dispatch_root: Path, technology: str) -> float:
    mapping = {
        "gas_ccgt": "gas_ccgt_new",
        "gas_ocgt": "gas_ocgt_new",
    }
    if technology not in mapping:
        raise ValueError("firm addition must be gas_ccgt or gas_ocgt")
    frame = pd.read_csv(dispatch_root / "firm_addition_costs_2035_eur2024.csv")
    selected = frame.loc[
        frame["firm_addition_technology"].astype(str).eq(mapping[technology])
    ]
    if len(selected) != 1:
        raise ValueError(f"missing firm-addition cost for {technology}")
    return float(selected.iloc[0]["annual_cost_eur2024_per_kw_year"])


def currency_rate(dispatch_root: Path, scenario: str) -> float:
    frame = pd.read_csv(dispatch_root / "currency_conversion_ledger.csv")
    selected = frame.loc[frame["scenario"].astype(str).eq(scenario)]
    if len(selected) != 1:
        raise ValueError(f"missing currency scenario {scenario}")
    rate = float(selected.iloc[0]["cny_per_eur"])
    if rate <= 0:
        raise ValueError("non-positive currency conversion")
    return rate


def build_storage_configurations(
    core,
    catalog_path: Path,
    technology_library_path: Path,
    config_ids: list[str],
    cny_per_eur: float,
    incremental_charge_cost_multiplier: float,
    lfp_degradation_multiplier: float,
    maximum_charge_to_discharge_ratio: float,
) -> list[Any]:
    if incremental_charge_cost_multiplier < 0:
        raise ValueError("incremental charge-cost multiplier must be non-negative")
    if lfp_degradation_multiplier < 0:
        raise ValueError("LFP degradation multiplier must be non-negative")
    if maximum_charge_to_discharge_ratio <= 0:
        raise ValueError("maximum charge-to-discharge ratio must be positive")
    catalog = pd.read_csv(catalog_path).set_index("config_id", drop=False)
    missing = sorted(set(config_ids).difference(catalog.index))
    if missing:
        raise ValueError(f"configuration IDs absent from catalog: {missing}")
    technology = pd.read_csv(technology_library_path).set_index("tech_id")
    result: list[Any] = []
    for config_id in config_ids:
        row = catalog.loc[config_id]
        tech_id = str(row["tech_id"])
        if tech_id not in technology.index:
            raise ValueError(f"{config_id}: technology parameters absent")
        tech = technology.loc[tech_id]
        annual_power_total = (
            float(row["annual_power_cost_cny_per_kw_year"]) / cny_per_eur
        )
        reference_charge_ratio = float(row["charge_power_ratio"])
        if maximum_charge_to_discharge_ratio <= reference_charge_ratio:
            raise ValueError(
                f"{config_id}: numerical charge-power ceiling must exceed "
                f"the released reference ratio {reference_charge_ratio:g}"
            )
        degradation = 0.0
        if tech_id == "li_lfp":
            cycle_life = float(tech["cycle_life_cycles"])
            if cycle_life <= 0:
                raise ValueError("LFP cycle life must be positive")
            degradation = (
                float(row["energy_cost_cny_kwh"])
                / cycle_life
                * 1_000.0
                / cny_per_eur
                * lfp_degradation_multiplier
            )
            # The core rejects a zero LFP degradation cost because accidental
            # omission was a reviewer-identified failure.  The explicit zero
            # sensitivity retains an infinitesimal audit marker.
            if degradation == 0:
                degradation = np.finfo(float).eps
        structural = parse_bool(row["structural_extrapolation"])
        result.append(
            core.StorageTechnology(
                config_id=config_id,
                tech_id=tech_id,
                duration_h=float(row["duration_h"]),
                round_trip_efficiency=float(row["rte"]),
                self_discharge_pct_day=float(row["self_discharge_pct_day"]),
                soc_min_fraction=0.05,
                soc_max_fraction=0.95,
                reference_charge_to_discharge_power_ratio=reference_charge_ratio,
                maximum_charge_to_discharge_power_ratio=(
                    maximum_charge_to_discharge_ratio
                ),
                # The duration-specific fixed-cost column already annualizes
                # the complete reference power-energy configuration.  Adding
                # the bundled power block again on Pdis would double count it.
                annual_discharge_power_cost_eur_kw=0.0,
                annual_charge_power_cost_eur_kw=(
                    annual_power_total * incremental_charge_cost_multiplier
                ),
                annual_energy_cost_eur_kwh=(
                    float(row["fixed_cost_cny_per_kwh_year"]) / cny_per_eur
                ),
                variable_om_eur_mwh=(
                    float(row["variable_om_cny_mwh"]) / cny_per_eur
                ),
                degradation_cost_eur_mwh_discharged=degradation,
                main_evidence_domain=bool(
                    not structural and float(row["duration_h"]) <= 168
                ),
                structural_extrapolation=structural,
            )
        )
    return result


def storage_event_diagnostics(
    dispatch: pd.DataFrame,
    chronology_root: Path,
    weather_scenario: str,
    chronology: str,
    order: np.ndarray,
    energy_capacity_mwh: float,
    charge_power_mw: float,
    quantile: float,
) -> dict[str, Any]:
    if dispatch.empty or energy_capacity_mwh <= 0:
        return {}
    daily_path = chronology_root / "europe_wide_daily_scarcity_scores.csv"
    if not daily_path.exists():
        return {
            "soc_event_recovery_mean_fraction": math.nan,
            "soc_event_recovery_event_count": 0,
            "soc_event_recovery_status": "Europe-wide daily scores unavailable",
        }
    daily = pd.read_csv(daily_path)
    daily = daily.loc[
        daily["weather_scenario"].astype(str).eq(weather_scenario)
    ].sort_values("day")
    if len(daily) != 365:
        raise ValueError("invalid Europe-wide daily scarcity score axis")
    original_scores = pd.to_numeric(daily["scarcity_score"], errors="raise").to_numpy()
    threshold = float(np.quantile(original_scores, quantile))
    extreme = original_scores[order] >= threshold
    events: list[tuple[int, int]] = []
    start: int | None = None
    for day, value in enumerate(extreme):
        if value and start is None:
            start = day
        if start is not None and (not value or day == len(extreme) - 1):
            stop = day + 1 if value and day == len(extreme) - 1 else day
            events.append((start, stop))
            start = None
    one = dispatch.sort_values("hour_of_year")
    soc = pd.to_numeric(one["soc_mwh"], errors="raise").to_numpy(dtype=float)
    charge = pd.to_numeric(one["charge_mw"], errors="raise").to_numpy(dtype=float)
    recoveries: list[float] = []
    event_charge = 0.0
    for start_day, stop_day in events:
        start_hour = start_day * 24
        stop_hour = min(stop_day * 24, len(soc))
        if start_hour >= len(soc) or stop_hour <= start_hour:
            continue
        pre = float(soc[(start_hour - 1) % len(soc)])
        event_min = float(soc[start_hour:stop_hour].min())
        recovery_stop = min(stop_hour + 72, len(soc))
        post_max = (
            float(soc[stop_hour:recovery_stop].max())
            if recovery_stop > stop_hour
            else float(soc[-1])
        )
        drawdown = pre - event_min
        if drawdown > 1e-9:
            recoveries.append((post_max - event_min) / drawdown)
        event_charge += float(charge[start_hour:stop_hour].sum())
    saturation = (
        float(np.mean(charge >= max(charge_power_mw - 1e-7, 0.0)))
        if charge_power_mw > 1e-9
        else 0.0
    )
    return {
        "soc_event_recovery_mean_fraction": (
            float(np.mean(recoveries)) if recoveries else math.nan
        ),
        "soc_event_recovery_median_fraction": (
            float(np.median(recoveries)) if recoveries else math.nan
        ),
        "soc_event_recovery_event_count": len(recoveries),
        "soc_event_recovery_definition": (
            "post-event 72h maximum SOC recovery relative to event drawdown"
        ),
        "minimum_soc_fraction_of_nameplate": float(soc.min() / energy_capacity_mwh),
        "mean_soc_fraction_of_nameplate": float(soc.mean() / energy_capacity_mwh),
        "charge_power_saturation_hour_fraction": saturation,
        "charge_during_europe_scarcity_events_mwh": event_charge,
    }


def solution_metric_row(
    solution,
    zone: str,
    weather_scenario: str,
    chronology: str,
    order_hash: str,
    capacity_case: str,
    split_case: str,
    evidence_status: str,
) -> dict[str, Any]:
    summary = dict(solution.summary)
    generation = summary.pop("expected_generation_by_technology_mwh", {})
    gas = float(generation.get("gas_ccgt", 0.0)) + float(
        generation.get("gas_ocgt", 0.0)
    )
    hydrogen = float(generation.get("hydrogen_ccgt", 0.0)) + float(
        generation.get("hydrogen_ocgt", 0.0)
    )
    return {
        "zone": zone,
        "weather_scenario": weather_scenario,
        "chronology": chronology,
        "order_sha256": order_hash,
        "capacity_case": capacity_case,
        "thermal_split_case": split_case,
        "evidence_status": evidence_status,
        **summary,
        "expected_gas_generation_mwh": gas,
        "expected_hydrogen_generation_mwh": hydrogen,
        **{
            f"expected_generation_{name}_mwh": value
            for name, value in generation.items()
        },
    }


def execute(args: argparse.Namespace) -> dict[str, Any]:
    core = load_core()
    config_ids = parse_config_ids(args.config_id)
    if args.weather_scenario not in WEATHER_SCENARIOS:
        raise ValueError("weather scenario must be WS01-WS36")
    if args.chronology not in CHRONOLOGIES:
        raise ValueError("chronology must be dispersed or clustered")
    if not 0 < args.tranche_share <= 0.1:
        raise ValueError("tranche share must lie in (0,0.1]")
    if args.development_hour_limit is not None:
        if (
            args.development_hour_limit <= 0
            or args.development_hour_limit > 8_760
            or args.development_hour_limit % 24
        ):
            raise ValueError("development hour limit must be a 24h multiple <=8760")

    input_root = args.input.resolve()
    chronology_root = args.chronology_root.resolve()
    dispatch_root = args.dispatch_root.resolve()
    output = args.output.resolve()
    prepare_output(output, args.force)

    order, order_hash = load_day_order(
        chronology_root, args.weather_scenario, args.chronology
    )
    profile = read_zone_weather(input_root, args.zone, args.weather_scenario)
    capacities = capacity_by_hydro_code(input_root, args.zone)
    hydro, hydro_audit = hydro_periodic_arrays(
        input_root, args.zone, args.weather_scenario, capacities
    )
    reservoir_power = float(
        sum(capacities.get(code, 0.0) for code in RESERVOIR_CODES)
    )
    available_power = hydro["reservoir_available_power_mw"]
    if reservoir_power <= 0 and float(np.max(available_power)) > 1e-9:
        raise ValueError(
            "released MAX_TURB is positive but mapped reservoir capacity is zero"
        )
    hydro_fraction = (
        available_power / reservoir_power
        if reservoir_power > 0
        else np.zeros(8_760, dtype=float)
    )

    load = reorder_days(
        pd.to_numeric(profile["load_mw"], errors="raise").to_numpy(), order
    )
    wind = reorder_days(
        pd.to_numeric(profile["wind_potential_mw"], errors="raise").to_numpy()
        + hydro["ror_potential_mw"],
        order,
    )
    solar_source = pd.to_numeric(
        profile["solar_potential_mw"], errors="raise"
    ).to_numpy(dtype=float)
    if "csp_with_storage_dispatched_reference_mw" in profile:
        solar_source += pd.to_numeric(
            profile["csp_with_storage_dispatched_reference_mw"], errors="raise"
        ).to_numpy(dtype=float)
    if "btm_surplus_potential_mw" in profile:
        solar_source += pd.to_numeric(
            profile["btm_surplus_potential_mw"], errors="raise"
        ).to_numpy(dtype=float)
    solar = reorder_days(solar_source, order)
    inflow = reorder_days(hydro["reservoir_inflow_mwh"], order)
    hydro_fraction = reorder_days(hydro_fraction, order)

    thermal = load_thermal_parameters(
        dispatch_root, args.zone, args.capacity_case, args.split_case
    )
    hours = args.development_hour_limit or 8_760
    load = load[:hours]
    wind = wind[:hours]
    solar = solar[:hours]
    inflow = inflow[:hours]
    hydro_fraction = hydro_fraction[:hours]
    thermal_availability = np.asarray(
        thermal["availability_fraction"], dtype=float
    )[:, :hours]

    firm_technology_index = THERMAL_TECHNOLOGIES.index(
        args.firm_addition_technology
    )
    case = core.StochasticTransportCase(
        case_id=(
            f"ERAA2035_{args.zone}_{args.weather_scenario}_"
            f"{args.chronology}_{args.capacity_case}_{args.split_case}"
        ),
        zones=(args.zone,),
        thermal_technologies=THERMAL_TECHNOLOGIES,
        scenario_ids=(args.weather_scenario,),
        scenario_probabilities=np.array([1.0]),
        load_mw=load[None, None, :],
        wind_potential_mw=wind[None, None, :],
        solar_potential_mw=solar[None, None, :],
        must_run_mw=np.zeros((1, 1, hours), dtype=float),
        thermal_existing_capacity_mw=np.asarray(
            thermal["existing_capacity_mw"], dtype=float
        )[None, :],
        thermal_available_fraction=thermal_availability[None, None, :, :],
        thermal_variable_cost_eur_mwh=np.asarray(
            thermal["variable_cost_eur_mwh"], dtype=float
        )[None, :],
        firm_addition_technology_index=firm_technology_index,
        firm_addition_annual_cost_eur_kw=np.array(
            [
                firm_addition_cost(
                    dispatch_root, args.firm_addition_technology
                )
            ]
        ),
        hydro_power_capacity_mw=np.array([reservoir_power]),
        hydro_available_fraction=hydro_fraction[None, None, :],
        hydro_energy_capacity_mwh=np.array(
            [hydro_storage_capacity_mwh(input_root, args.zone)]
        ),
        hydro_inflow_mwh=inflow[None, None, :],
        hydro_variable_cost_eur_mwh=np.array([args.hydro_variable_cost]),
        edges=(),
        network_capacity_mw=np.zeros((1, 0, hours), dtype=float),
        deployment_mask=np.array([True]),
    )
    mean_load = float(load.mean())
    settings = core.SolveSettings(
        candidate_total_discharge_power_mw=mean_load * args.tranche_share,
        value_of_lost_load_eur_mwh=args.value_of_lost_load,
        maximum_firm_addition_mw_per_zone=args.maximum_firm_addition_mw,
    )
    rate = currency_rate(dispatch_root, args.currency_scenario)
    storages = build_storage_configurations(
        core,
        args.catalog,
        args.technology_library,
        config_ids,
        rate,
        args.incremental_charge_cost_multiplier,
        args.lfp_degradation_multiplier,
        args.maximum_charge_to_discharge_ratio,
    )

    admission_gates = {
        "post_eva_fleet_applied": not args.capacity_case.startswith("pre_eva"),
        "official_utility_battery_modelled": False,
        "official_pumped_hydro_modelled": False,
        "official_explicit_dsr_modelled": False,
        "full_52_zone_network_context_modelled": False,
    }
    evidence_status = (
        "development_only_pre_eva_pending_annex3_transfer"
        if not admission_gates["post_eva_fleet_applied"]
        else "development_only_missing_existing_storage_dsr_and_network"
    )
    if hours < 8_760:
        evidence_status = "development_only_truncated_horizon"

    # The no-candidate baseline is invariant to storage configuration and is
    # solved only once for the full list.
    baseline = core.solve_stochastic_transport_lp(
        case, storages[0], settings, "baseline", keep_dispatch=True
    )
    baseline_firm = (
        baseline.first_stage.set_index("zone")
        .loc[[args.zone], "firm_addition_mw"]
        .to_numpy()
    )
    metric_rows = [
        solution_metric_row(
            baseline,
            args.zone,
            args.weather_scenario,
            args.chronology,
            order_hash,
            args.capacity_case,
            args.split_case,
            evidence_status,
        )
    ]
    first_rows: list[pd.DataFrame] = []
    base_first = baseline.first_stage.copy()
    base_first.insert(0, "config_id", "baseline_no_candidate")
    base_first.insert(0, "variant", "baseline")
    first_rows.append(base_first)
    diagnostic_rows: list[dict[str, Any]] = []
    dispatch_rows: list[pd.DataFrame] = []

    for storage in storages:
        for variant in ("joint", "turnover"):
            solution = core.solve_stochastic_transport_lp(
                case,
                storage,
                settings,
                variant,
                baseline_firm_addition_mw=(
                    baseline_firm if variant == "turnover" else None
                ),
                keep_dispatch=True,
            )
            if variant == "joint":
                solution.summary["signed_joint_benefit_eur_kw_year"] = (
                    baseline.objective_eur_year - solution.objective_eur_year
                ) / (
                    settings.candidate_total_discharge_power_mw * 1_000.0
                )
                solution.summary["equivalent_firm_capacity_mw"] = float(
                    baseline_firm.sum()
                    - solution.first_stage["firm_addition_mw"].sum()
                )
            else:
                solution.summary["signed_turnover_benefit_eur_kw_year"] = (
                    baseline.objective_eur_year - solution.objective_eur_year
                ) / (
                    settings.candidate_total_discharge_power_mw * 1_000.0
                )
            row = solution_metric_row(
                solution,
                args.zone,
                args.weather_scenario,
                args.chronology,
                order_hash,
                args.capacity_case,
                args.split_case,
                evidence_status,
            )
            metric_rows.append(row)
            first = solution.first_stage.copy()
            first.insert(0, "config_id", storage.config_id)
            first.insert(0, "variant", variant)
            first_rows.append(first)
            if variant == "joint":
                pdis = float(
                    solution.first_stage["candidate_discharge_power_mw"].sum()
                )
                pcharge = float(
                    solution.first_stage["candidate_charge_power_mw"].sum()
                )
                diagnostics = storage_event_diagnostics(
                    solution.dispatch,
                    chronology_root,
                    args.weather_scenario,
                    args.chronology,
                    order,
                    pdis * storage.duration_h,
                    pcharge,
                    args.scarcity_quantile,
                )
                diagnostic_rows.append(
                    {
                        "zone": args.zone,
                        "weather_scenario": args.weather_scenario,
                        "chronology": args.chronology,
                        "config_id": storage.config_id,
                        "variant": variant,
                        **diagnostics,
                    }
                )
            if args.keep_dispatch:
                one_dispatch = solution.dispatch.copy()
                one_dispatch.insert(0, "config_id", storage.config_id)
                one_dispatch.insert(0, "variant", variant)
                dispatch_rows.append(one_dispatch)

    metrics = pd.DataFrame(metric_rows)
    first_stage = pd.concat(first_rows, ignore_index=True)
    diagnostics = pd.DataFrame(diagnostic_rows)
    catalog_for_ledger = pd.read_csv(args.catalog).set_index("config_id")
    technology_for_ledger = pd.read_csv(args.technology_library).set_index(
        "tech_id"
    )
    ledger_rows: list[dict[str, Any]] = []
    for storage in storages:
        source = catalog_for_ledger.loc[storage.config_id]
        tech = technology_for_ledger.loc[storage.tech_id]
        cycle_life = (
            float(tech["cycle_life_cycles"])
            if pd.notna(tech["cycle_life_cycles"])
            else math.nan
        )
        ledger_rows.append(
            {
                "config_id": storage.config_id,
                "tech_id": storage.tech_id,
                "duration_h": storage.duration_h,
                "currency_scenario": args.currency_scenario,
                "cny_per_eur": rate,
                "base_fixed_cost_cny_per_kwh_year": float(
                    source["fixed_cost_cny_per_kwh_year"]
                ),
                "base_fixed_cost_eur_per_kwh_year": (
                    storage.annual_energy_cost_eur_kwh
                ),
                "bundled_power_cost_cny_per_kw_year": float(
                    source["annual_power_cost_cny_per_kw_year"]
                ),
                "base_discharge_power_cost_added_eur_per_kw_year": (
                    storage.annual_discharge_power_cost_eur_kw
                ),
                "incremental_charge_cost_multiplier": (
                    args.incremental_charge_cost_multiplier
                ),
                "incremental_charge_cost_eur_per_kw_year": (
                    storage.annual_charge_power_cost_eur_kw
                ),
                "reference_charge_to_discharge_ratio": (
                    storage.reference_charge_to_discharge_power_ratio
                ),
                "numerical_maximum_charge_to_discharge_ratio": (
                    storage.maximum_charge_to_discharge_power_ratio
                ),
                "energy_capex_cny_per_kwh_for_degradation": float(
                    source["energy_cost_cny_kwh"]
                ),
                "cycle_life_cycles": cycle_life,
                "lfp_degradation_multiplier": (
                    args.lfp_degradation_multiplier
                ),
                "degradation_cost_eur_per_mwh_discharged": (
                    storage.degradation_cost_eur_mwh_discharged
                ),
                "double_count_guard": (
                    "base duration-specific fixed cost includes the reference "
                    "integrated configuration; bundled power cost is applied "
                    "only to incremental charge kW"
                ),
            }
        )
    storage_cost_ledger = pd.DataFrame(ledger_rows)
    candidate_first_stage = first_stage.loc[
        pd.to_numeric(
            first_stage["candidate_discharge_power_mw"], errors="raise"
        ).gt(1e-9)
    ].copy()
    if candidate_first_stage.empty:
        maximum_charge_ratio_observed = 0.0
    else:
        maximum_charge_ratio_observed = float(
            (
                candidate_first_stage["candidate_charge_power_mw"]
                / candidate_first_stage["candidate_discharge_power_mw"]
            ).max()
        )
    charge_ratio_ceiling_binding = bool(
        maximum_charge_ratio_observed
        >= args.maximum_charge_to_discharge_ratio - 1e-6
    )
    write_csv(metrics, output / "task_metrics.csv")
    write_csv(first_stage, output / "first_stage.csv")
    write_csv(diagnostics, output / "storage_event_diagnostics.csv")
    write_csv(storage_cost_ledger, output / "storage_cost_ledger.csv")
    if args.keep_dispatch:
        write_csv(
            pd.concat(dispatch_rows, ignore_index=True),
            output / "candidate_dispatch.csv.gz",
            compressed=True,
        )

    objective_error = float(
        pd.to_numeric(
            metrics["objective_component_recalculation_error_eur"],
            errors="raise",
        ).max()
    )
    qa = {
        "pass": bool(
            metrics["maximum_balance_residual_mw"].max() <= 1e-6
            and metrics["maximum_hydro_stock_residual_mwh"].max() <= 1e-6
            and not charge_ratio_ceiling_binding
            and objective_error
            <= 1e-8 * max(float(metrics["objective_eur_year"].abs().max()), 1.0)
        ),
        "script_version": SCRIPT_VERSION,
        "evidence_status": evidence_status,
        "admission_gates": admission_gates,
        "zone": args.zone,
        "weather_scenario": args.weather_scenario,
        "chronology": args.chronology,
        "all_52_node_order_sha256": order_hash,
        "hours_solved": hours,
        "configurations": config_ids,
        "maximum_balance_residual_mw": float(
            metrics["maximum_balance_residual_mw"].max()
        ),
        "maximum_hydro_stock_residual_mwh": float(
            metrics["maximum_hydro_stock_residual_mwh"].max()
        ),
        "maximum_objective_component_error_eur": objective_error,
        "maximum_charge_to_discharge_ratio_observed": (
            maximum_charge_ratio_observed
        ),
        "charge_to_discharge_ratio_numerical_ceiling": (
            args.maximum_charge_to_discharge_ratio
        ),
        "charge_ratio_ceiling_binding": charge_ratio_ceiling_binding,
    }
    write_json(output / "QA.json", qa)
    manifest = {
        "script_version": SCRIPT_VERSION,
        "target_year": TARGET_YEAR,
        "input_root": input_root,
        "chronology_root": chronology_root,
        "dispatch_root": dispatch_root,
        "catalog": args.catalog.resolve(),
        "technology_library": args.technology_library.resolve(),
        "zone": args.zone,
        "weather_scenario": args.weather_scenario,
        "chronology": args.chronology,
        "order_sha256": order_hash,
        "order_scope": "same_order_for_all_52_demand_nodes",
        "capacity_case": args.capacity_case,
        "thermal_split_case": args.split_case,
        "firm_addition_technology": args.firm_addition_technology,
        "currency_scenario": args.currency_scenario,
        "cny_per_eur": rate,
        "tranche_share_of_zone_mean_load": args.tranche_share,
        "candidate_total_discharge_power_mw": (
            settings.candidate_total_discharge_power_mw
        ),
        "incremental_charge_cost_multiplier": (
            args.incremental_charge_cost_multiplier
        ),
        "charge_power_cost_interpretation": (
            "duration-specific fixed cost already contains the released "
            "reference integrated power-energy block; only "
            "max(0, Pcharge - r_ref*Pdis) is repriced at multiplier times "
            "the bundled EUR/kW-year cost"
        ),
        "maximum_charge_to_discharge_ratio_numerical_ceiling": (
            args.maximum_charge_to_discharge_ratio
        ),
        "lfp_degradation_multiplier": args.lfp_degradation_multiplier,
        "lfp_degradation_definition": (
            "energy-side CAPEX divided by released cycle life, per discharged MWh"
        ),
        "hours_solved": hours,
        "perfect_foresight": True,
        "network_case": "atomic_zone_zero_interconnection",
        "demand_scope": (
            "released ERAA demand after official BTM PV/battery net-output identity"
        ),
        "csp_rule": (
            "ordinary solar plus dispatched CSP reference plus BTM surplus; "
            "predispatch solar-field input excluded to avoid double count"
        ),
        "hydro_audit": hydro_audit,
        "hydro_model_scope": (
            "aggregate HRI/HOL/HPI reservoir plus spillable HRR; "
            "HCL pumping and period-energy/minimum constraints pending"
        ),
        "thermal_availability_scope": (
            "expected FOR x derating x calendar maintenance; maintenance not reordered"
        ),
        "evidence_status": evidence_status,
        "admission_gates": admission_gates,
        "admission_rule": (
            "post-EVA fleet, official utility batteries, pumped hydro, "
            "explicit DSR and the full 52-zone network must all pass before "
            "a run can be promoted from development to confirmatory evidence"
        ),
        "storage_configurations": [asdict(storage) for storage in storages],
        "qa_pass": qa["pass"],
    }
    write_json(output / "run_manifest.json", manifest)
    if not qa["pass"]:
        raise RuntimeError("zone-task QA failed")
    return qa


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    result.add_argument("--chronology-root", type=Path, default=DEFAULT_CHRONOLOGY)
    result.add_argument("--dispatch-root", type=Path, default=DEFAULT_DISPATCH)
    result.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    result.add_argument(
        "--technology-library", type=Path, default=DEFAULT_TECH_LIBRARY
    )
    result.add_argument("--zone", required=True)
    result.add_argument("--weather-scenario", required=True)
    result.add_argument("--chronology", required=True, choices=CHRONOLOGIES)
    result.add_argument("--config-id", action="append", required=True)
    result.add_argument(
        "--capacity-case",
        default="pre_eva_pending_unit_transfer",
    )
    result.add_argument("--split-case", default="all_ccgt")
    result.add_argument(
        "--firm-addition-technology",
        default="gas_ocgt",
        choices=("gas_ccgt", "gas_ocgt"),
    )
    result.add_argument(
        "--currency-scenario", default="central_cny_per_eur"
    )
    result.add_argument("--tranche-share", type=float, default=0.01)
    result.add_argument(
        "--incremental-charge-cost-multiplier", type=float, default=1.0
    )
    result.add_argument(
        "--maximum-charge-to-discharge-ratio", type=float, default=10.0
    )
    result.add_argument("--lfp-degradation-multiplier", type=float, default=1.0)
    result.add_argument("--scarcity-quantile", type=float, default=0.75)
    result.add_argument("--value-of-lost-load", type=float, default=15_000.0)
    result.add_argument("--hydro-variable-cost", type=float, default=5.0)
    result.add_argument(
        "--maximum-firm-addition-mw", type=float, default=100_000.0
    )
    result.add_argument("--development-hour-limit", type=int)
    result.add_argument("--keep-dispatch", action="store_true")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--force", action="store_true")
    return result


def main() -> None:
    execute(parser().parse_args())


if __name__ == "__main__":
    main()
