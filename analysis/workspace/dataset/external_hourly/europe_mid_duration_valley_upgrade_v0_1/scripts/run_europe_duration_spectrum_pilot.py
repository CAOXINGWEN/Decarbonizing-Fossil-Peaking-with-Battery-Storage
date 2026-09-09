#!/usr/bin/env python3
"""Run an isolated European storage-duration physical/value screen.

The adapter deliberately lives outside every manuscript revision directory.  It
reuses the audited v36 53-node technology catalogue and sparse LP equations,
but it does not read or write any v36 result.  Its purpose is to make the
larger OPSD panel executable before the full networked European model exists.

Three paired chronologies are produced from exactly the same intact 24-hour
load-wind-solar blocks:

* ``natural`` keeps source order;
* ``dispersed`` separates upper-quartile residual-demand days within seasons;
* ``clustered`` groups the same upper-quartile days in four-day runs.

The screen reports equivalent full cycles, charging, discharge, spill, EENS,
joint system benefit, turnover-only benefit, and a censoring-aware duration
boundary diagnostic.  The latter is a *screen*, not a confirmatory full-valley
test: ELCC intervals and a networked adequacy model are not created here.

Examples
--------
Validate an expanded panel and the 53-node catalogue without solving LPs::

    python scripts/run_europe_duration_spectrum_pilot.py \
        --mode validate --panel processed/eu_opsd_panel_2015_2019.csv.gz

Run a small executable pilot on one country-year::

    python scripts/run_europe_duration_spectrum_pilot.py \
        --mode pilot --panel processed/eu_opsd_panel_2015_2019.csv.gz \
        --unit country_year --countries DE --years 2019

Run the complete 53-node point-estimate screen for the same unit::

    python scripts/run_europe_duration_spectrum_pilot.py \
        --mode screen --panel processed/eu_opsd_panel_2015_2019.csv.gz \
        --unit country_year --countries DE --years 2019 --resume

For a copper-plate balanced-EU sensitivity, use
``--unit balanced_eu_multiyear``.  This preserves synchronous European day
vectors but is not a network-constrained European model.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
UPGRADE = HERE.parent
WORKSPACE = UPGRADE.parents[2]
V36 = (
    WORKSPACE
    / "Decarbonizing-Fossil-Peaking-with-Battery-Storage-master"
    / "paper-code"
    / "docs"
    / "musrcipt"
    / "revision_nature_energy_v36"
    / "system_value_closure"
)
CORE_PATH = V36 / "run_v36_system_value_closure.py"
CONFIGURATION_INPUT = (
    V36 / "inputs" / "technology_duration_configurations_2035.csv"
)
DEFAULT_RUN_ROOT = UPGRADE / "model_adapter"

EU27 = {
    "AT",
    "BE",
    "BG",
    "HR",
    "CY",
    "CZ",
    "DK",
    "EE",
    "FI",
    "FR",
    "DE",
    "GR",
    "HU",
    "IE",
    "IT",
    "LV",
    "LT",
    "LU",
    "MT",
    "NL",
    "PL",
    "PT",
    "RO",
    "SK",
    "SI",
    "ES",
    "SE",
}

CHRONOLOGIES = ("natural", "dispersed", "clustered")
PILOT_CONFIG_IDS = {
    "li_lfp_4h",
    "caes_12h",
    "caes_24h",
    "h2_cavern_48h",
    "h2_cavern_96h",
    "h2_cavern_168h",
    "h2_cavern_720h",
}
PANEL_CANDIDATES = (
    UPGRADE / "processed" / "eu_opsd_panel_2015_2019.csv.gz",
    UPGRADE / "processed" / "europe_opsd_hourly_panel_2015_2019.csv.gz",
    UPGRADE / "processed" / "hourly_load_wind_solar_2015_2019.csv.gz",
    V36 / "inputs" / "hourly_country_year_2016_2019.csv.gz",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_csv_list(value: str | None, cast=str) -> list[Any] | None:
    if value is None or not value.strip():
        return None
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def load_core():
    if not CORE_PATH.exists():
        raise FileNotFoundError(CORE_PATH)
    module_name = "europe_adapter_v36_lp_core"
    specification = importlib.util.spec_from_file_location(module_name, CORE_PATH)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot import LP core from {CORE_PATH}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def load_configurations(core) -> list[Any]:
    """Load v36 configurations without writing into the v36 prepared folder."""

    frame = pd.read_csv(CONFIGURATION_INPUT)
    frame = frame[frame["learning_year"] == 2035].copy()
    configurations = []
    for record in frame.to_dict("records"):
        duration = float(record["duration_h"])
        annual_power_cost = (
            float(record["annualized_power_capex_cny_per_kw_year"])
            + float(record["fixed_om_cny_kw_yr"])
            + float(record["annual_replacement_reserve_cny_per_kw_year"])
        )
        configurations.append(
            core.Configuration(
                config_id=f"{record['tech_id']}_{duration:g}h",
                tech_id=str(record["tech_id"]),
                technology_en=str(record["technology_en"]),
                duration_h=duration,
                rte=float(record["rte"]),
                self_discharge_pct_day=float(record["self_discharge_pct_day"]),
                charge_power_ratio=float(record["charge_power_ratio"]),
                fixed_cost_cny_per_kwh_year=float(
                    record["annual_fixed_cost_cny_per_kwh_capacity_year"]
                ),
                variable_om_cny_mwh=float(record["variable_om_cny_mwh"]),
                annual_power_cost_cny_per_kw_year=annual_power_cost,
                power_cost_cny_kw=float(record["power_cost_cny_kw"]),
                energy_cost_cny_kwh=float(record["energy_cost_cny_kwh"]),
                released_library_configuration=as_bool(
                    record["released_library_configuration"]
                ),
                direct_duration_anchor=as_bool(record["direct_duration_anchor"]),
                structural_extrapolation=as_bool(
                    record["structural_extrapolation"]
                ),
                configuration_status=str(record["configuration_status"]),
                interpolation_lower_h=float(record["interpolation_lower_h"]),
                interpolation_upper_h=(
                    float(record["interpolation_upper_h"])
                    if pd.notna(record["interpolation_upper_h"])
                    else np.nan
                ),
                source_key=str(record["source_key"]),
            )
        )
    configurations.sort(key=lambda item: (item.duration_h, item.tech_id))
    if len(configurations) != 53:
        raise RuntimeError(
            f"Expected the audited 53-node catalogue; found {len(configurations)}"
        )
    if len({item.config_id for item in configurations}) != len(configurations):
        raise RuntimeError("Configuration identifiers are not unique")
    return configurations


def choose_column(columns: Iterable[str], aliases: Sequence[str]) -> str | None:
    available = set(columns)
    return next((name for name in aliases if name in available), None)


def standardize_panel(
    path: Path, sample_flag: str | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Map a raw/processed OPSD panel to six model-facing columns."""

    raw = pd.read_csv(path)
    if raw.empty:
        raise RuntimeError(f"Panel is empty: {path}")
    sample_rows_before = len(raw)
    sample_flags: list[str] = []
    if sample_flag:
        sample_flags = [
            item.strip() for item in sample_flag.split(",") if item.strip()
        ]
        absent = [item for item in sample_flags if item not in raw.columns]
        if absent:
            raise RuntimeError(f"Sample flags absent from panel: {absent}")
        keep = pd.Series(True, index=raw.index)
        for flag in sample_flags:
            keep &= raw[flag].map(as_bool)
        raw = raw[keep].copy()
        if raw.empty:
            raise RuntimeError(
                f"--sample-flag {sample_flag!r} retained no panel rows"
            )
    country_column = choose_column(
        raw.columns, ("country", "country_code", "zone", "bidding_zone")
    )
    year_column = choose_column(raw.columns, ("year", "weather_year"))
    timestamp_column = choose_column(
        raw.columns,
        (
            "timestamp_utc",
            "utc_timestamp",
            "source_utc_timestamp",
            "datetime_utc",
            "timestamp",
        ),
    )
    load_column = choose_column(
        raw.columns,
        ("load_mw", "load", "load_actual_mw", "load_actual"),
    )
    solar_column = choose_column(
        raw.columns,
        ("solar_mw", "solar", "solar_generation_mw", "solar_generation_actual"),
    )
    wind_column = choose_column(
        raw.columns,
        ("wind_mw", "wind", "wind_generation_mw", "wind_generation_actual"),
    )
    if wind_column is None:
        wind_parts = [
            name
            for name in (
                "wind_onshore_mw",
                "wind_offshore_mw",
                "wind_onshore",
                "wind_offshore",
            )
            if name in raw.columns
        ]
        if wind_parts:
            raw["_adapter_wind_mw"] = raw[wind_parts].sum(axis=1, min_count=1)
            wind_column = "_adapter_wind_mw"
    required = {
        "country": country_column,
        "timestamp": timestamp_column,
        "load": load_column,
        "wind": wind_column,
        "solar": solar_column,
    }
    missing = [key for key, value in required.items() if value is None]
    if missing:
        raise RuntimeError(
            f"Panel lacks model-facing fields {missing}; columns={list(raw.columns)}"
        )
    timestamp = pd.to_datetime(raw[timestamp_column], utc=True, errors="coerce")
    year = (
        pd.to_numeric(raw[year_column], errors="coerce")
        if year_column
        else timestamp.dt.year
    )
    panel = pd.DataFrame(
        {
            "timestamp_utc": timestamp,
            "country": raw[country_column].astype(str).str.upper().str.strip(),
            "year": year,
            "load_mw": pd.to_numeric(raw[load_column], errors="coerce"),
            "wind_mw": pd.to_numeric(raw[wind_column], errors="coerce"),
            "solar_mw": pd.to_numeric(raw[solar_column], errors="coerce"),
        }
    )
    panel = panel.dropna(subset=["timestamp_utc", "country", "year"]).copy()
    panel["year"] = panel["year"].astype(int)
    field_map = {
        "source": str(path),
        "sample_flags_AND": sample_flags or ["none_panel_presumed_prequalified"],
        "sample_rows_before_flag": sample_rows_before,
        "sample_rows_after_flag": len(raw),
        "country": country_column,
        "year": year_column or "derived_from_timestamp_utc",
        "timestamp_utc": timestamp_column,
        "load_mw": load_column,
        "wind_mw": wind_column,
        "solar_mw": solar_column,
    }
    return panel, field_map


def maximum_missing_run(frame: pd.DataFrame, fields: Sequence[str]) -> int:
    missing = frame[list(fields)].isna().any(axis=1).to_numpy(bool)
    best = run = 0
    for value in missing:
        run = run + 1 if value else 0
        best = max(best, run)
    return int(best)


def clean_country_year(
    group: pd.DataFrame,
    max_interpolation_h: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    country = str(group["country"].iloc[0])
    year = int(group["year"].iloc[0])
    group = group.sort_values("timestamp_utc").copy()
    duplicate_hours = int(group["timestamp_utc"].duplicated().sum())
    if duplicate_hours:
        raise RuntimeError(f"Duplicate UTC hours for {country}-{year}")
    start = pd.Timestamp(f"{year}-01-01 00:00:00", tz="UTC")
    stop = pd.Timestamp(f"{year + 1}-01-01 00:00:00", tz="UTC")
    expected = pd.date_range(start, stop, freq="h", inclusive="left")
    values = (
        group.set_index("timestamp_utc")[["load_mw", "wind_mw", "solar_mw"]]
        .reindex(expected)
        .astype(float)
    )
    missing_before = values.isna().sum()
    longest = maximum_missing_run(values, ("load_mw", "wind_mw", "solar_mw"))
    if longest > max_interpolation_h:
        raise RuntimeError(
            f"{country}-{year} missing run {longest} h exceeds "
            f"{max_interpolation_h} h"
        )
    values = values.interpolate(
        method="time",
        limit=max_interpolation_h,
        limit_direction="both",
        limit_area="inside",
    )
    if values.isna().any().any():
        raise RuntimeError(
            f"{country}-{year} retains boundary/missing values after interpolation"
        )
    leap = (values.index.month == 2) & (values.index.day == 29)
    leap_hours_removed = int(leap.sum())
    values = values.loc[~leap].copy()
    if len(values) != 8760:
        raise RuntimeError(f"{country}-{year} has {len(values)} model hours")
    if (values < 0).any().any():
        raise RuntimeError(f"{country}-{year} contains negative load/generation")
    values.insert(0, "timestamp_utc", values.index)
    values.insert(1, "country", country)
    values.insert(2, "year", year)
    return values.reset_index(drop=True), {
        "country": country,
        "year": year,
        "source_rows": int(len(group)),
        "duplicate_utc_hours": duplicate_hours,
        "load_missing_hours_before_interpolation": int(missing_before["load_mw"]),
        "wind_missing_hours_before_interpolation": int(missing_before["wind_mw"]),
        "solar_missing_hours_before_interpolation": int(
            missing_before["solar_mw"]
        ),
        "maximum_missing_run_h": longest,
        "leap_day_hours_removed": leap_hours_removed,
        "model_hours": int(len(values)),
    }


def preprocess_panel(
    panel: pd.DataFrame,
    countries: list[str] | None,
    years: list[int] | None,
    max_interpolation_h: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if countries:
        panel = panel[panel["country"].isin(countries)].copy()
    if years:
        panel = panel[panel["year"].isin(years)].copy()
    if panel.empty:
        raise RuntimeError("No panel rows remain after country/year selection")
    cleaned = []
    coverage = []
    failures = []
    for (country, year), group in panel.groupby(["country", "year"], sort=True):
        try:
            output, record = clean_country_year(group, max_interpolation_h)
            cleaned.append(output)
            coverage.append(record)
        except Exception as error:  # retain auditable exclusions
            failures.append(
                {
                    "country": country,
                    "year": int(year),
                    "exclusion_reason": str(error),
                }
            )
    if not cleaned:
        raise RuntimeError(f"No eligible country-years; failures={failures}")
    coverage_frame = pd.DataFrame(coverage)
    if failures:
        failure_frame = pd.DataFrame(failures)
        coverage_frame = coverage_frame.merge(
            failure_frame, on=["country", "year"], how="outer"
        )
    return pd.concat(cleaned, ignore_index=True), coverage_frame


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


def interleave(extreme: list[int], normal: list[int]) -> list[int]:
    slots: list[list[int]] = [[] for _ in range(len(normal) + 1)]
    if extreme:
        positions = np.linspace(1, max(1, len(normal) - 1), len(extreme))
        for day, position in zip(extreme, np.rint(positions).astype(int)):
            slots[int(np.clip(position, 0, len(normal)))].append(day)
    order = []
    for index, day in enumerate(normal):
        order.extend(slots[index])
        order.append(day)
    order.extend(slots[-1])
    return order


def interleave_clusters(
    extreme: list[int], normal: list[int], cluster_days: int
) -> list[int]:
    clusters = [
        extreme[index : index + cluster_days]
        for index in range(0, len(extreme), cluster_days)
    ]
    slots: list[list[int]] = [[] for _ in range(len(normal) + 1)]
    if clusters:
        positions = np.linspace(1, max(1, len(normal) - 1), len(clusters))
        for cluster, position in zip(clusters, np.rint(positions).astype(int)):
            slots[int(np.clip(position, 0, len(normal)))].extend(cluster)
    order = []
    for index, day in enumerate(normal):
        order.extend(slots[index])
        order.append(day)
    order.extend(slots[-1])
    return order


def event_stats(extreme: np.ndarray) -> dict[str, float]:
    runs = []
    current = 0
    for value in extreme:
        if value:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return {
        "extreme_days": int(np.sum(extreme)),
        "extreme_events": int(len(runs)),
        "mean_extreme_run_d": float(np.mean(runs)) if runs else 0.0,
        "max_extreme_run_d": int(max(runs)) if runs else 0,
        "share_extreme_days_in_runs_ge3": (
            float(sum(run for run in runs if run >= 3) / sum(runs))
            if runs
            else 0.0
        ),
    }


def prepare_year_arrays(
    group: pd.DataFrame,
    vre_energy_share: float,
    extreme_quantile: float,
    cluster_days: int,
) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, Any]]]:
    """Normalize one system-year and return paired chronology arrays."""

    group = group.sort_values("timestamp_utc").reset_index(drop=True)
    if len(group) != 8760:
        raise RuntimeError("prepare_year_arrays requires exactly 8,760 hours")
    load = group["load_mw"].to_numpy(float)
    wind = group["wind_mw"].to_numpy(float)
    solar = group["solar_mw"].to_numpy(float)
    raw_vre = wind + solar
    if load.sum() <= 0 or raw_vre.sum() <= 0:
        raise RuntimeError("Non-positive load or wind-plus-solar annual energy")
    scale = vre_energy_share * float(load.sum() / raw_vre.sum())
    signed = load - raw_vre * scale
    mean_load = float(load.mean())
    base = pd.DataFrame(
        {
            "timestamp_utc": group["timestamp_utc"].to_numpy(),
            "model_day": np.arange(8760) // 24,
            "hour_of_day": np.arange(8760) % 24,
            "load": load / mean_load,
            "residual": np.maximum(signed, 0.0) / mean_load,
            "surplus": np.maximum(-signed, 0.0) / mean_load,
        }
    )
    base["date"] = pd.to_datetime(base["timestamp_utc"], utc=True).dt.floor("D")
    daily = (
        base.groupby("model_day", sort=True)
        .agg(
            date=("date", "first"),
            load=("load", "sum"),
            residual=("residual", "sum"),
            surplus=("surplus", "sum"),
        )
        .copy()
    )
    daily["month"] = pd.to_datetime(daily["date"], utc=True).dt.month
    daily["segment"] = daily["month"].map(calendar_segment)
    threshold = float(daily["residual"].quantile(extreme_quantile))
    daily["extreme"] = daily["residual"] >= threshold
    blocks = {
        int(day): block.sort_values("hour_of_day").copy()
        for day, block in base.groupby("model_day", sort=True)
    }
    orders: dict[str, list[int]] = {
        "natural": [int(day) for day in daily.index]
    }
    for chronology in ("dispersed", "clustered"):
        full_order = []
        for segment in sorted(daily["segment"].unique()):
            subset = daily[daily["segment"] == segment]
            extreme = [int(day) for day in subset.index[subset["extreme"]]]
            normal = [int(day) for day in subset.index[~subset["extreme"]]]
            if chronology == "dispersed":
                full_order.extend(interleave(extreme, normal))
            else:
                full_order.extend(interleave_clusters(extreme, normal, cluster_days))
        orders[chronology] = full_order

    reference = sorted(orders["natural"])
    arrays: dict[str, dict[str, np.ndarray]] = {}
    qa_rows = []
    for chronology, order in orders.items():
        if len(order) != 365 or sorted(order) != reference:
            raise RuntimeError(f"Invalid {chronology} day permutation")
        arranged = pd.concat([blocks[day] for day in order], ignore_index=True)
        arrays[chronology] = {
            field: arranged[field].to_numpy(float)
            for field in ("load", "residual", "surplus")
        }
        stats = event_stats(daily.loc[order, "extreme"].to_numpy(bool))
        qa_rows.append(
            {
                "chronology": chronology,
                "days": len(order),
                "unique_days": len(set(order)),
                "day_multiset_matches_natural": sorted(order) == reference,
                "annual_load_pu_h": float(arranged["load"].sum()),
                "annual_residual_pu_h": float(arranged["residual"].sum()),
                "annual_surplus_pu_h": float(arranged["surplus"].sum()),
                "vre_scale_factor": scale,
                "vre_energy_share": vre_energy_share,
                "q_threshold": extreme_quantile,
                "q_threshold_residual_pu_h": threshold,
                "ordered_day_sha256": hashlib.sha256(
                    np.asarray(order, dtype=np.int16).tobytes()
                ).hexdigest(),
                **stats,
            }
        )
    return arrays, qa_rows


def balanced_eu_panel(
    panel: pd.DataFrame,
    requested_years: list[int] | None,
    requested_countries: list[str] | None,
) -> tuple[pd.DataFrame, list[str], list[int]]:
    available = panel[panel["country"].isin(EU27)].copy()
    if requested_countries:
        available = available[available["country"].isin(requested_countries)]
    years = (
        sorted(requested_years)
        if requested_years
        else sorted(available["year"].unique().tolist())
    )
    counts = (
        available[available["year"].isin(years)]
        .groupby("country")["year"]
        .nunique()
    )
    countries = sorted(counts[counts == len(years)].index.tolist())
    if len(countries) < 2:
        raise RuntimeError(
            f"Balanced EU aggregate requires >=2 countries; got {countries}"
        )
    selected = available[
        available["country"].isin(countries) & available["year"].isin(years)
    ].copy()
    expected_rows = len(countries) * len(years) * 8760
    if len(selected) != expected_rows:
        raise RuntimeError(
            f"Unbalanced hourly panel: expected {expected_rows}, got {len(selected)}"
        )
    aggregate = (
        selected.groupby(["timestamp_utc", "year"], as_index=False)[
            ["load_mw", "wind_mw", "solar_mw"]
        ]
        .sum()
        .sort_values(["year", "timestamp_utc"])
    )
    aggregate.insert(1, "country", "BALANCED_EU")
    return aggregate, countries, years


def make_profile_units(
    core,
    panel: pd.DataFrame,
    unit: str,
    requested_years: list[int] | None,
    requested_countries: list[str] | None,
    vre_energy_share: float,
    extreme_quantile: float,
    cluster_days: int,
) -> tuple[dict[tuple[str, str], Any], pd.DataFrame, pd.DataFrame]:
    """Create core.Profile objects and chronology-conservation QA."""

    profile_sources: dict[str, list[tuple[int, pd.DataFrame]]] = {}
    unit_meta = []
    if unit == "country_year":
        for (country, year), group in panel.groupby(["country", "year"], sort=True):
            unit_id = f"{country}_{int(year)}"
            profile_sources[unit_id] = [(int(year), group)]
            unit_meta.append(
                {
                    "unit_id": unit_id,
                    "unit_type": unit,
                    "countries": country,
                    "years": str(int(year)),
                    "country_count": 1,
                    "weather_year_count": 1,
                    "spatial_model": "single_country",
                }
            )
    elif unit == "country_multiyear":
        for country, country_group in panel.groupby("country", sort=True):
            year_groups = [
                (int(year), group)
                for year, group in country_group.groupby("year", sort=True)
            ]
            if len(year_groups) < 2:
                continue
            profile_sources[country] = year_groups
            unit_meta.append(
                {
                    "unit_id": country,
                    "unit_type": unit,
                    "countries": country,
                    "years": ";".join(str(year) for year, _ in year_groups),
                    "country_count": 1,
                    "weather_year_count": len(year_groups),
                    "spatial_model": "single_country_shared_capacity",
                }
            )
    elif unit == "balanced_eu_multiyear":
        aggregate, countries, years = balanced_eu_panel(
            panel, requested_years, requested_countries
        )
        profile_sources["BALANCED_EU"] = [
            (int(year), group)
            for year, group in aggregate.groupby("year", sort=True)
        ]
        unit_meta.append(
            {
                "unit_id": "BALANCED_EU",
                "unit_type": unit,
                "countries": ";".join(countries),
                "years": ";".join(str(year) for year in years),
                "country_count": len(countries),
                "weather_year_count": len(years),
                "spatial_model": "synchronous_copper_plate_aggregate",
            }
        )
    else:
        raise ValueError(unit)
    if not profile_sources:
        raise RuntimeError(f"No profile units could be formed for {unit}")

    profiles: dict[tuple[str, str], Any] = {}
    qa_rows = []
    for unit_id, year_groups in profile_sources.items():
        per_chronology = {
            chronology: {"load": [], "residual": [], "surplus": []}
            for chronology in CHRONOLOGIES
        }
        years = []
        offsets = [0]
        for year, group in year_groups:
            arrays, records = prepare_year_arrays(
                group,
                vre_energy_share=vre_energy_share,
                extreme_quantile=extreme_quantile,
                cluster_days=cluster_days,
            )
            years.append(year)
            offsets.append(offsets[-1] + 8760)
            for record in records:
                record.update({"unit_id": unit_id, "year": year})
                qa_rows.append(record)
            for chronology in CHRONOLOGIES:
                for field in ("load", "residual", "surplus"):
                    per_chronology[chronology][field].append(
                        arrays[chronology][field]
                    )
        for chronology in CHRONOLOGIES:
            profiles[(unit_id, chronology)] = core.Profile(
                country=unit_id,
                persistence=chronology,
                years=tuple(years),
                offsets=tuple(offsets),
                load=np.concatenate(per_chronology[chronology]["load"]),
                residual=np.concatenate(per_chronology[chronology]["residual"]),
                surplus=np.concatenate(per_chronology[chronology]["surplus"]),
            )
    return profiles, pd.DataFrame(unit_meta), pd.DataFrame(qa_rows)


def select_configs(
    configurations: list[Any],
    mode: str,
    max_duration_h: float | None = None,
) -> list[Any]:
    if mode == "pilot":
        selected = [
            config for config in configurations if config.config_id in PILOT_CONFIG_IDS
        ]
        missing = PILOT_CONFIG_IDS - {config.config_id for config in selected}
        if missing:
            raise RuntimeError(f"Pilot configuration IDs missing: {sorted(missing)}")
        return selected
    if mode == "screen" and max_duration_h is not None:
        selected = [
            config
            for config in configurations
            if config.duration_h <= max_duration_h
        ]
        if not selected:
            raise RuntimeError(
                f"No configurations at or below {max_duration_h:g} h"
            )
        return selected
    return configurations


def write_incremental(records: list[dict[str, Any]], path: Path) -> None:
    if records:
        pd.DataFrame(records).to_csv(path, index=False)


def solve_profiles(
    core,
    profiles: dict[tuple[str, str], Any],
    configurations: list[Any],
    run_dir: Path,
    resume: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_path = run_dir / "node_results.csv"
    yearly_path = run_dir / "yearly_metrics.csv"
    results = (
        pd.read_csv(result_path).to_dict("records")
        if resume and result_path.exists()
        else []
    )
    yearly = (
        pd.read_csv(yearly_path).to_dict("records")
        if resume and yearly_path.exists()
        else []
    )
    completed = {
        (str(row["unit_id"]), str(row["chronology"]), str(row["config_id"]))
        for row in results
        if row.get("variant") == "forced_1pct_turnover"
    }
    baseline_rows = [
        row for row in results if row.get("variant") == "baseline"
    ]
    baseline_map = {
        (str(row["unit_id"]), str(row["chronology"])): row
        for row in baseline_rows
    }
    reference = next(
        config
        for config in configurations
        if config.tech_id == "li_lfp" and config.duration_h == 4.0
    )
    for (unit_id, chronology), profile in profiles.items():
        if (unit_id, chronology) in baseline_map:
            continue
        row, year_rows = core.solve_baseline(profile, reference)
        row.update(
            {
                "unit_id": unit_id,
                "chronology": chronology,
                "config_id": "no_storage",
                "tech_id": "no_storage",
                "duration_h": 0.0,
            }
        )
        for item in year_rows:
            item.update({"unit_id": unit_id, "chronology": chronology})
        results.append(row)
        yearly.extend(year_rows)
        baseline_map[(unit_id, chronology)] = row
        write_incremental(results, result_path)
        write_incremental(yearly, yearly_path)

    for (unit_id, chronology), profile in profiles.items():
        baseline = baseline_map[(unit_id, chronology)]
        baseline_cost = float(baseline["total_system_cost_cny_per_mean_mw_year"])
        baseline_kg = float(baseline["gas_capacity_pu_mean_load"])
        for config in configurations:
            key = (unit_id, chronology, config.config_id)
            if key in completed:
                continue
            problem = core.build_problem(profile, config)
            for variant in ("forced_1pct_joint", "forced_1pct_turnover"):
                row, year_rows, _ = core.solve_variant(
                    problem,
                    variant,
                    baseline_cost=baseline_cost,
                    baseline_kg=baseline_kg,
                )
                row.update(
                    {
                        "unit_id": unit_id,
                        "chronology": chronology,
                        "configuration_source": str(CONFIGURATION_INPUT),
                    }
                )
                for item in year_rows:
                    item.update(
                        {
                            "unit_id": unit_id,
                            "chronology": chronology,
                        }
                    )
                results.append(row)
                yearly.extend(year_rows)
            completed.add(key)
            write_incremental(results, result_path)
            write_incremental(yearly, yearly_path)
    return pd.DataFrame(results), pd.DataFrame(yearly)


def negative_runs(values: np.ndarray) -> list[tuple[int, int]]:
    runs = []
    index = 0
    while index < len(values):
        if values[index] >= 0:
            index += 1
            continue
        stop = index
        while stop + 1 < len(values) and values[stop + 1] < 0:
            stop += 1
        runs.append((index, stop))
        index = stop + 1
    return runs


def censored_boundaries(group: pd.DataFrame) -> dict[str, Any]:
    """Diagnose h_T, h_I and W without replacing censored values by endpoints."""

    ordered = group.sort_values("duration_h").reset_index(drop=True)
    duration = ordered["duration_h"].to_numpy(float)
    turnover = ordered[
        "turnover_signed_benefit_cny_per_forced_kw_year"
    ].to_numpy(float)
    joint = ordered["joint_signed_benefit_cny_per_forced_kw_year"].to_numpy(float)
    adequacy = ordered["adequacy_component_cny_per_forced_kw_year"].to_numpy(
        float
    )
    positive_turnover = turnover > 0
    prefix_end = -1
    for index, value in enumerate(positive_turnover):
        if value:
            prefix_end = index
        else:
            break
    if prefix_end < 0:
        h_t = np.nan
        h_t_status = "not_identified_no_short_positive_turnover_branch"
    elif prefix_end == len(duration) - 1:
        h_t = np.nan
        h_t_status = f"right_censored_above_{duration[-1]:g}h"
    else:
        h_t = float(duration[prefix_end])
        h_t_status = (
            f"discrete_bracket_{duration[prefix_end]:g}_"
            f"{duration[prefix_end + 1]:g}h"
        )

    candidate = None
    for start, stop in negative_runs(joint):
        if stop - start + 1 < 2:
            continue
        if prefix_end >= 0 and start <= prefix_end:
            continue
        candidate = (start, stop)
        break
    if candidate is None:
        gap_start = gap_end = h_i = np.nan
        h_i_status = "not_identified_no_two_node_negative_middle_interval"
    else:
        start, stop = candidate
        gap_start = float(duration[start])
        gap_end = float(duration[stop])
        reentry = np.where((joint[stop + 1 :] > 0) & (adequacy[stop + 1 :] > 0))[0]
        if len(reentry):
            position = stop + 1 + int(reentry[0])
            h_i = float(duration[position])
            h_i_status = (
                f"point_reentry_at_{h_i:g}h_adequacy_component_positive"
            )
        else:
            h_i = np.nan
            h_i_status = f"right_censored_above_{duration[-1]:g}h"
    complete = bool(np.isfinite(h_t) and np.isfinite(h_i) and h_i > h_t)
    width = float(math.log(h_i / h_t)) if complete else np.nan
    reasons = []
    if not np.isfinite(h_t):
        reasons.append(h_t_status)
    if candidate is None:
        reasons.append("no_adjacent_negative_joint_benefit_nodes")
    elif not np.isfinite(h_i):
        reasons.append(h_i_status)
    reasons.append(
        "confirmatory_ELCC_and_hierarchical_interval_rules_not_run_by_adapter"
    )
    return {
        "point_screen_complete_valley": complete,
        "confirmatory_complete_valley": False,
        "h_T_h": h_t,
        "h_T_censoring_status": h_t_status,
        "h_I_proxy_h": h_i,
        "h_I_censoring_status": h_i_status,
        "negative_interval_start_h": gap_start,
        "negative_interval_end_h": gap_end,
        "W_log_hI_over_hT": width,
        "duration_min_h": float(duration.min()),
        "duration_max_h": float(duration.max()),
        "duration_node_count": int(len(duration)),
        "nonconfirmation_reasons": ";".join(dict.fromkeys(reasons)),
    }


def postprocess(
    results: pd.DataFrame,
    yearly: pd.DataFrame,
    run_dir: Path,
    full_configuration_set: bool,
    forced_tranche_share: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_yearly = yearly[yearly["variant"] == "baseline"].copy()
    baseline_yearly = baseline_yearly[
        ["unit_id", "chronology", "year", "total_system_cost_cny_per_mean_mw_weather_year"]
    ].rename(
        columns={
            "total_system_cost_cny_per_mean_mw_weather_year": (
                "baseline_system_cost_cny_per_mean_mw_weather_year"
            )
        }
    )
    forced_yearly = yearly[yearly["variant"].str.startswith("forced_1pct")].copy()
    forced_yearly = forced_yearly.merge(
        baseline_yearly,
        on=["unit_id", "chronology", "year"],
        how="left",
        validate="many_to_one",
    )
    forced_yearly[
        "signed_benefit_cny_per_forced_kw_weather_year"
    ] = (
        forced_yearly[
            "baseline_system_cost_cny_per_mean_mw_weather_year"
        ]
        - forced_yearly["total_system_cost_cny_per_mean_mw_weather_year"]
    ) / (forced_tranche_share * 1_000.0)
    id_columns = [
        "unit_id",
        "chronology",
        "year",
        "config_id",
        "tech_id",
        "technology_en",
        "duration_h",
    ]
    year_value = (
        forced_yearly.pivot(
            index=id_columns,
            columns="variant",
            values="signed_benefit_cny_per_forced_kw_weather_year",
        )
        .reset_index()
        .rename_axis(columns=None)
        .rename(
            columns={
                "forced_1pct_joint": (
                    "joint_signed_benefit_cny_per_forced_kw_weather_year"
                ),
                "forced_1pct_turnover": (
                    "turnover_signed_benefit_cny_per_forced_kw_weather_year"
                ),
            }
        )
    )
    if not year_value.empty:
        year_value[
            "adequacy_component_cny_per_forced_kw_weather_year"
        ] = (
            year_value[
                "joint_signed_benefit_cny_per_forced_kw_weather_year"
            ]
            - year_value[
                "turnover_signed_benefit_cny_per_forced_kw_weather_year"
            ]
        )
    year_value.to_csv(run_dir / "country_year_node_value.csv", index=False)

    forced = results[results["variant"] == "forced_1pct_joint"].copy()
    turnover = results[results["variant"] == "forced_1pct_turnover"].copy()
    merged = forced.merge(
        turnover[
            [
                "unit_id",
                "chronology",
                "config_id",
                "signed_benefit_cny_per_forced_kw_year",
            ]
        ].rename(
            columns={
                "signed_benefit_cny_per_forced_kw_year": (
                    "turnover_signed_benefit_cny_per_forced_kw_year"
                )
            }
        ),
        on=["unit_id", "chronology", "config_id"],
        validate="one_to_one",
    )
    merged = merged.rename(
        columns={
            "signed_benefit_cny_per_forced_kw_year": (
                "joint_signed_benefit_cny_per_forced_kw_year"
            )
        }
    )
    merged["adequacy_component_cny_per_forced_kw_year"] = (
        merged["joint_signed_benefit_cny_per_forced_kw_year"]
        - merged["turnover_signed_benefit_cny_per_forced_kw_year"]
    )
    joint_yearly_diagnostics = (
        yearly[yearly["variant"] == "forced_1pct_joint"]
        .groupby(["unit_id", "chronology", "config_id"], as_index=False)
        .agg(
            expected_spill_mwh_per_mean_mw_year=(
                "spill_mwh_per_mean_mw",
                "mean",
            ),
            expected_gas_generation_mwh_per_mean_mw_year=(
                "gas_generation_mwh_per_mean_mw",
                "mean",
            ),
            mean_charge_power_saturation_h=(
                "charge_power_saturation_h",
                "mean",
            ),
            mean_discharge_power_saturation_h=(
                "discharge_power_saturation_h",
                "mean",
            ),
            mean_soc_min_observed_fraction=(
                "soc_min_observed_fraction",
                "mean",
            ),
            mean_soc_max_observed_fraction=(
                "soc_max_observed_fraction",
                "mean",
            ),
        )
    )
    merged = merged.merge(
        joint_yearly_diagnostics,
        on=["unit_id", "chronology", "config_id"],
        how="left",
        validate="one_to_one",
    )
    baseline_costs = (
        results[results["variant"] == "baseline"][
            ["unit_id", "chronology", "total_system_cost_cny_per_mean_mw_year"]
        ]
        .rename(
            columns={
                "total_system_cost_cny_per_mean_mw_year": (
                    "baseline_system_cost_cny_per_mean_mw_year"
                )
            }
        )
        .drop_duplicates(["unit_id", "chronology"])
    )
    merged = merged.merge(
        baseline_costs,
        on=["unit_id", "chronology"],
        how="left",
        validate="many_to_one",
    )
    merged.to_csv(run_dir / "configuration_screen_metrics.csv", index=False)
    envelope_rows = []
    for (unit_id, chronology, duration), group in merged.groupby(
        ["unit_id", "chronology", "duration_h"], sort=True
    ):
        winner = group.loc[
            group["joint_signed_benefit_cny_per_forced_kw_year"].idxmax()
        ]
        envelope_rows.append(
            {
                "unit_id": unit_id,
                "chronology": chronology,
                "duration_h": duration,
                "winning_config_id": winner["config_id"],
                "winning_tech_id": winner["tech_id"],
                "joint_signed_benefit_cny_per_forced_kw_year": winner[
                    "joint_signed_benefit_cny_per_forced_kw_year"
                ],
                "turnover_signed_benefit_cny_per_forced_kw_year": winner[
                    "turnover_signed_benefit_cny_per_forced_kw_year"
                ],
                "adequacy_component_cny_per_forced_kw_year": winner[
                    "adequacy_component_cny_per_forced_kw_year"
                ],
                "total_system_cost_cny_per_mean_mw_year": winner[
                    "total_system_cost_cny_per_mean_mw_year"
                ],
                "baseline_system_cost_cny_per_mean_mw_year": winner[
                    "baseline_system_cost_cny_per_mean_mw_year"
                ],
                "equivalent_full_cycles_output_basis": winner[
                    "expected_equivalent_full_cycles_output_basis"
                ],
                "charge_mwh_per_mean_mw_year": winner[
                    "expected_charge_mwh_per_mean_mw_year"
                ],
                "discharge_mwh_per_mean_mw_year": winner[
                    "expected_discharge_mwh_per_mean_mw_year"
                ],
                "eens_mwh_per_mean_mw_year": winner[
                    "expected_eens_mwh_per_mean_mw_year"
                ],
                "spill_mwh_per_mean_mw_year": winner[
                    "expected_spill_mwh_per_mean_mw_year"
                ],
                "gas_generation_mwh_per_mean_mw_year": winner[
                    "expected_gas_generation_mwh_per_mean_mw_year"
                ],
                "mean_charge_power_saturation_h": winner[
                    "mean_charge_power_saturation_h"
                ],
                "mean_discharge_power_saturation_h": winner[
                    "mean_discharge_power_saturation_h"
                ],
                "mean_soc_min_observed_fraction": winner[
                    "mean_soc_min_observed_fraction"
                ],
                "mean_soc_max_observed_fraction": winner[
                    "mean_soc_max_observed_fraction"
                ],
                "gas_capacity_avoided_pu_mean_load": winner[
                    "gas_capacity_avoided_pu_mean_load"
                ],
                "full_configuration_set": full_configuration_set,
            }
        )
    envelope = pd.DataFrame(envelope_rows)
    envelope.to_csv(run_dir / "duration_envelope.csv", index=False)

    boundary_rows = []
    for (unit_id, chronology), group in envelope.groupby(
        ["unit_id", "chronology"], sort=True
    ):
        record = {
            "unit_id": unit_id,
            "chronology": chronology,
            "full_configuration_set": full_configuration_set,
        }
        if full_configuration_set:
            record.update(censored_boundaries(group))
        else:
            record.update(
                {
                    "point_screen_complete_valley": False,
                    "confirmatory_complete_valley": False,
                    "h_T_h": np.nan,
                    "h_T_censoring_status": "not_evaluated_pilot_subset",
                    "h_I_proxy_h": np.nan,
                    "h_I_censoring_status": "not_evaluated_pilot_subset",
                    "negative_interval_start_h": np.nan,
                    "negative_interval_end_h": np.nan,
                    "W_log_hI_over_hT": np.nan,
                    "duration_min_h": float(group["duration_h"].min()),
                    "duration_max_h": float(group["duration_h"].max()),
                    "duration_node_count": int(group["duration_h"].nunique()),
                    "nonconfirmation_reasons": (
                        "pilot_configuration_subset_cannot_identify_boundaries;"
                        "confirmatory_ELCC_and_hierarchical_interval_rules_not_run"
                    ),
                }
            )
        boundary_rows.append(record)
    boundaries = pd.DataFrame(boundary_rows)

    paired_rows = []
    for unit_id, group in boundaries.groupby("unit_id", sort=True):
        low = group[group["chronology"] == "dispersed"]
        high = group[group["chronology"] == "clustered"]
        if len(low) and len(high):
            low_w = float(low["W_log_hI_over_hT"].iloc[0])
            high_w = float(high["W_log_hI_over_hT"].iloc[0])
            paired_rows.append(
                {
                    "unit_id": unit_id,
                    "W_dispersed": low_w,
                    "W_clustered": high_w,
                    "delta_W_clustered_minus_dispersed": (
                        high_w - low_w
                        if np.isfinite(low_w) and np.isfinite(high_w)
                        else np.nan
                    ),
                    "delta_W_status": (
                        "point_estimate_available_not_confirmatory"
                        if np.isfinite(low_w) and np.isfinite(high_w)
                        else "censored_or_valley_not_identified"
                    ),
                }
            )
    paired = pd.DataFrame(paired_rows)
    boundaries.to_csv(run_dir / "boundary_diagnostics.csv", index=False)
    paired.to_csv(run_dir / "paired_delta_W.csv", index=False)

    physical = (
        merged.groupby(["unit_id", "config_id", "duration_h"], as_index=False)
        .apply(
            lambda frame: pd.Series(
                {
                    "delta_efc_clustered_minus_dispersed": (
                        frame.loc[
                            frame["chronology"] == "clustered",
                            "expected_equivalent_full_cycles_output_basis",
                        ].mean()
                        - frame.loc[
                            frame["chronology"] == "dispersed",
                            "expected_equivalent_full_cycles_output_basis",
                        ].mean()
                    ),
                    "delta_joint_benefit_clustered_minus_dispersed": (
                        frame.loc[
                            frame["chronology"] == "clustered",
                            "joint_signed_benefit_cny_per_forced_kw_year",
                        ].mean()
                        - frame.loc[
                            frame["chronology"] == "dispersed",
                            "joint_signed_benefit_cny_per_forced_kw_year",
                        ].mean()
                    ),
                    "delta_turnover_benefit_clustered_minus_dispersed": (
                        frame.loc[
                            frame["chronology"] == "clustered",
                            "turnover_signed_benefit_cny_per_forced_kw_year",
                        ].mean()
                        - frame.loc[
                            frame["chronology"] == "dispersed",
                            "turnover_signed_benefit_cny_per_forced_kw_year",
                        ].mean()
                    ),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    physical.to_csv(run_dir / "paired_physical_value_changes.csv", index=False)
    negative_efc = int(
        (physical["delta_efc_clustered_minus_dispersed"] < 0).sum()
    )
    negative_turnover = int(
        (
            physical[
                "delta_turnover_benefit_clustered_minus_dispersed"
            ]
            < 0
        ).sum()
    )
    interpretation = [
        "# European duration-spectrum screen interpretation",
        "",
        "This file reports what the isolated adapter did and did not confirm.",
        "",
        f"- Configuration-unit comparisons with lower clustered EFC: "
        f"{negative_efc}/{len(physical)}.",
        f"- Configuration-unit comparisons with lower clustered turnover "
        f"benefit: {negative_turnover}/{len(physical)}.",
        f"- Point-estimate complete valleys: "
        f"{int(boundaries['point_screen_complete_valley'].sum())}/"
        f"{len(boundaries)} chronology-unit screens.",
        "- Confirmatory complete valleys: 0. The adapter does not estimate "
        "networked ELCC or hierarchical confidence intervals.",
        "",
        "## Censoring and non-confirmation by chronology",
        "",
    ]
    for row in boundaries.itertuples(index=False):
        interpretation.extend(
            [
                f"- `{row.unit_id}` / `{row.chronology}`: "
                f"h_T={row.h_T_h if pd.notna(row.h_T_h) else 'NA'} "
                f"({row.h_T_censoring_status}); "
                f"h_I={row.h_I_proxy_h if pd.notna(row.h_I_proxy_h) else 'NA'} "
                f"({row.h_I_censoring_status}); "
                f"W={row.W_log_hI_over_hT if pd.notna(row.W_log_hI_over_hT) else 'NA'}. "
                f"Reason: {row.nonconfirmation_reasons}.",
            ]
        )
    interpretation.extend(
        [
            "",
            "A negative clustered-minus-dispersed EFC or turnover response "
            "supports contraction of repeat-use value. It does not by itself "
            "establish a positive-negative-positive duration-value structure.",
        ]
    )
    (run_dir / "SCREEN_INTERPRETATION.md").write_text(
        "\n".join(interpretation) + "\n", encoding="utf-8"
    )
    return envelope, boundaries


def build_qa(
    results: pd.DataFrame | None,
    yearly: pd.DataFrame | None,
    chronology_qa: pd.DataFrame,
    configurations: list[Any],
    full_configuration_set: bool,
    mode: str,
    unit_meta: pd.DataFrame,
) -> dict[str, Any]:
    pivot = chronology_qa.pivot_table(
        index=["unit_id", "year"],
        columns="chronology",
        values="mean_extreme_run_d",
    )
    qa: dict[str, Any] = {
        "mode": mode,
        "profile_unit_count": int(len(unit_meta)),
        "profile_units": sorted(unit_meta["unit_id"].tolist()),
        "configuration_count_loaded": 53,
        "configuration_count_run": int(len(configurations)),
        "full_configuration_set": full_configuration_set,
        "chronology_rows": int(len(chronology_qa)),
        "all_day_multisets_conserved": bool(
            chronology_qa["day_multiset_matches_natural"].all()
        ),
        "maximum_annual_load_conservation_error_pu_h": 0.0,
        "maximum_annual_residual_conservation_error_pu_h": 0.0,
        "maximum_annual_surplus_conservation_error_pu_h": 0.0,
        "clustered_mean_run_exceeds_dispersed": bool(
            (pivot["clustered"] > pivot["dispersed"]).all()
        ),
        "screen_supports_confirmatory_valley_claim": False,
        "screen_scope": (
            "point-estimate copper-plate/single-country screening; "
            "no networked ELCC interval test"
        ),
    }
    for field, output in (
        ("annual_load_pu_h", "maximum_annual_load_conservation_error_pu_h"),
        (
            "annual_residual_pu_h",
            "maximum_annual_residual_conservation_error_pu_h",
        ),
        (
            "annual_surplus_pu_h",
            "maximum_annual_surplus_conservation_error_pu_h",
        ),
    ):
        values = chronology_qa.pivot_table(
            index=["unit_id", "year"], columns="chronology", values=field
        )
        qa[output] = float(
            (values.max(axis=1) - values.min(axis=1)).abs().max()
        )
    if results is not None and yearly is not None:
        qa.update(
            {
                "all_solver_optimal": bool((results["solver_status"] == 0).all()),
                "maximum_energy_balance_residual": float(
                    results["maximum_energy_balance_residual"].max()
                ),
                "maximum_equality_violation": float(
                    results["maximum_equality_violation"].max()
                ),
                "maximum_inequality_violation": float(
                    results["maximum_inequality_violation"].max()
                ),
                "maximum_objective_decomposition_error_cny": float(
                    results["objective_decomposition_error_cny"].max()
                ),
                "maximum_eens_share": float(yearly["eens_share_of_load"].max()),
                "duplicate_node_variant_rows": int(
                    results[
                        results["variant"].str.startswith("forced_1pct")
                    ].duplicated(
                        ["unit_id", "chronology", "config_id", "variant"]
                    ).sum()
                ),
            }
        )
    qa["passed"] = bool(
        qa["configuration_count_loaded"] == 53
        and qa["all_day_multisets_conserved"]
        and qa["maximum_annual_load_conservation_error_pu_h"] <= 1e-9
        and qa["maximum_annual_residual_conservation_error_pu_h"] <= 1e-9
        and qa["maximum_annual_surplus_conservation_error_pu_h"] <= 1e-9
        and qa["clustered_mean_run_exceeds_dispersed"]
        and (
            mode == "validate"
            or (
                qa["all_solver_optimal"]
                and qa["maximum_energy_balance_residual"] <= 2e-6
                and qa["maximum_equality_violation"] <= 2e-6
                and qa["maximum_inequality_violation"] <= 2e-6
                and qa["maximum_objective_decomposition_error_cny"] <= 1e-3
                and qa["maximum_eens_share"] <= 0.0010002
                and qa["duplicate_node_variant_rows"] == 0
            )
        )
    )
    return qa


def resolve_panel(argument: str | None) -> Path:
    if argument:
        path = Path(argument)
        if not path.is_absolute():
            local = UPGRADE / path
            path = local if local.exists() else Path.cwd() / path
        if not path.exists():
            raise FileNotFoundError(path)
        return path.resolve()
    for candidate in PANEL_CANDIDATES:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "No panel found. Pass --panel explicitly after the expanded OPSD "
        "panel has been built."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expanded-European storage-duration screen adapter"
    )
    parser.add_argument(
        "--mode", choices=("validate", "pilot", "screen"), default="validate"
    )
    parser.add_argument("--panel")
    parser.add_argument(
        "--sample-flag",
        help=(
            "Optional boolean panel column(s) defining the analysis sample. "
            "Comma-separated flags are combined with AND."
        ),
    )
    parser.add_argument(
        "--unit",
        choices=(
            "country_year",
            "country_multiyear",
            "balanced_eu_multiyear",
        ),
        default="country_year",
    )
    parser.add_argument("--countries", help="Comma-separated ISO2 country codes")
    parser.add_argument("--years", help="Comma-separated years")
    parser.add_argument(
        "--country-years",
        help="Exact comma-separated ISO2:year pairs, e.g. DK:2015,DE:2018",
    )
    parser.add_argument("--vre-energy-share", type=float, default=1.0)
    parser.add_argument("--extreme-quantile", type=float, default=0.75)
    parser.add_argument("--cluster-days", type=int, default=4)
    parser.add_argument(
        "--solve-chronologies",
        default="natural,dispersed,clustered",
        help=(
            "Comma-separated chronology variants to solve. Chronology "
            "construction and conservation QA always retain all three. Use "
            "dispersed,clustered for paired screens that do not require the "
            "natural chronology solution."
        ),
    )
    parser.add_argument("--max-interpolation-h", type=int, default=24)
    parser.add_argument(
        "--max-duration-h",
        type=float,
        help=(
            "Optional upper duration bound for screen mode. The evidence-"
            "supported 4-168 h catalogue uses --max-duration-h 168 and "
            "excludes the three structural H2 extrapolations."
        ),
    )
    parser.add_argument("--gas-fixed-cny-kw-year", type=float)
    parser.add_argument("--gas-variable-cny-mwh", type=float)
    parser.add_argument(
        "--max-gas-pu",
        type=float,
        help=(
            "Optional nonbinding numerical upper bound on gas capacity in "
            "units of the panel task's mean load. Observed task series with "
            "large peak-to-mean ratios require a value above the core "
            "default; the realized optimum must remain below this bound."
        ),
    )
    parser.add_argument("--forced-tranche-share", type=float, default=0.01)
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    panel_path = resolve_panel(args.panel)
    countries = parse_csv_list(args.countries, str)
    countries = [item.upper() for item in countries] if countries else None
    years = parse_csv_list(args.years, int)
    country_year_pairs = None
    if args.country_years:
        country_year_pairs = set()
        for item in args.country_years.split(","):
            country, separator, year = item.strip().partition(":")
            if not separator:
                raise ValueError(
                    "--country-years entries must use ISO2:year format"
                )
            country_year_pairs.add((country.upper(), int(year)))
    run_id = args.run_id or (
        f"{args.mode}_{args.unit}_"
        f"{hashlib.sha256(str(panel_path).encode()).hexdigest()[:8]}"
    )
    run_dir = DEFAULT_RUN_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    core = load_core()
    if args.gas_fixed_cny_kw_year is not None:
        core.NGCC_FIXED_CNY_KW_YR = float(args.gas_fixed_cny_kw_year)
    if args.gas_variable_cny_mwh is not None:
        core.NGCC_VARIABLE_CNY_MWH = float(args.gas_variable_cny_mwh)
    if args.max_gas_pu is not None:
        if not np.isfinite(args.max_gas_pu) or args.max_gas_pu <= 0:
            raise ValueError("--max-gas-pu must be finite and positive")
        core.MAX_GAS_PU = float(args.max_gas_pu)
    core.FORCED_TRANCHE_PU = float(args.forced_tranche_share)
    if not (0 < core.FORCED_TRANCHE_PU <= 0.1):
        raise ValueError("--forced-tranche-share must be in (0, 0.1]")

    configurations_all = load_configurations(core)
    configurations = select_configs(
        configurations_all,
        args.mode,
        max_duration_h=args.max_duration_h,
    )
    panel, field_map = standardize_panel(panel_path, sample_flag=args.sample_flag)
    if country_year_pairs:
        pair_index = pd.MultiIndex.from_frame(panel[["country", "year"]])
        panel = panel[pair_index.isin(country_year_pairs)].copy()
        if panel.empty:
            raise RuntimeError(
                f"No rows match --country-years {sorted(country_year_pairs)}"
            )
    panel, coverage = preprocess_panel(
        panel,
        countries=countries,
        years=years,
        max_interpolation_h=args.max_interpolation_h,
    )
    profiles, unit_meta, chronology_qa = make_profile_units(
        core,
        panel,
        unit=args.unit,
        requested_years=years,
        requested_countries=countries,
        vre_energy_share=args.vre_energy_share,
        extreme_quantile=args.extreme_quantile,
        cluster_days=args.cluster_days,
    )
    solve_chronologies = [
        item.strip()
        for item in args.solve_chronologies.split(",")
        if item.strip()
    ]
    invalid_chronologies = sorted(set(solve_chronologies) - set(CHRONOLOGIES))
    if invalid_chronologies or not solve_chronologies:
        raise ValueError(
            f"Invalid --solve-chronologies: {invalid_chronologies or 'empty'}"
        )
    profiles = {
        key: profile
        for key, profile in profiles.items()
        if key[1] in solve_chronologies
    }
    coverage.to_csv(run_dir / "country_year_input_coverage.csv", index=False)
    unit_meta.to_csv(run_dir / "profile_units.csv", index=False)
    chronology_qa.to_csv(run_dir / "chronology_conservation_qa.csv", index=False)
    pd.DataFrame([asdict(config) for config in configurations_all]).to_csv(
        run_dir / "configuration_catalog_53.csv", index=False
    )

    results = yearly = None
    full_configuration_set = args.mode == "screen"
    if args.mode != "validate":
        results, yearly = solve_profiles(
            core,
            profiles,
            configurations,
            run_dir=run_dir,
            resume=args.resume,
        )
        postprocess(
            results,
            yearly,
            run_dir=run_dir,
            full_configuration_set=full_configuration_set,
            forced_tranche_share=core.FORCED_TRANCHE_PU,
        )
    qa = build_qa(
        results,
        yearly,
        chronology_qa,
        configurations,
        full_configuration_set=full_configuration_set,
        mode=args.mode,
        unit_meta=unit_meta,
    )
    (run_dir / "QA.json").write_text(
        json.dumps(json_safe(qa), indent=2), encoding="utf-8"
    )
    manifest = {
        "adapter_version": "0.1.0",
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "elapsed_seconds": time.time() - started,
        "mode": args.mode,
        "unit": args.unit,
        "panel": str(panel_path),
        "panel_sha256": sha256(panel_path),
        "field_map": field_map,
        "configuration_catalog": str(CONFIGURATION_INPUT),
        "configuration_catalog_sha256": sha256(CONFIGURATION_INPUT),
        "v36_lp_core": str(CORE_PATH),
        "v36_lp_core_sha256": sha256(CORE_PATH),
        "countries_requested": countries,
        "years_requested": years,
        "country_year_pairs_requested": (
            sorted(country_year_pairs) if country_year_pairs else None
        ),
        "sample_flag": args.sample_flag,
        "profile_units": unit_meta.to_dict("records"),
        "chronologies_constructed_and_audited": list(CHRONOLOGIES),
        "chronologies_solved": solve_chronologies,
        "vre_energy_share": args.vre_energy_share,
        "extreme_quantile": args.extreme_quantile,
        "cluster_days": args.cluster_days,
        "maximum_duration_h": args.max_duration_h,
        "forced_tranche_share_of_mean_load": core.FORCED_TRANCHE_PU,
        "gas_fixed_cny_per_kw_year": core.NGCC_FIXED_CNY_KW_YR,
        "gas_variable_cny_per_mwh": core.NGCC_VARIABLE_CNY_MWH,
        "maximum_gas_capacity_pu_mean_task": core.MAX_GAS_PU,
        "cost_interpretation": (
            "controlled 2035 v36 scenario screen, not observed European "
            "country-specific price calibration"
        ),
        "spatial_interpretation": (
            "single-country or synchronous copper-plate aggregate; "
            "cross-border network is not represented"
        ),
        "confirmatory_valley_interpretation": (
            "not confirmatory until networked adequacy, ELCC intervals, "
            "hierarchical resampling, and uncensored h_T/h_I are available"
        ),
        "qa_passed": qa["passed"],
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(json_safe(manifest), indent=2), encoding="utf-8"
    )
    if not qa["passed"]:
        raise RuntimeError(f"Adapter QA failed; see {run_dir / 'QA.json'}")
    print(run_dir)


if __name__ == "__main__":
    main()
