#!/usr/bin/env python3
"""Build the frozen Stage-0 observed-gas event-task definitions.

This script implements only Stage 0 of
``OBSERVED_GAS_EVENT_TASK_CONFIRMATORY_DESIGN.md``.  It does not construct
counterfactual chronologies and does not call an optimizer.

The task-side quantities C0--C4 are deterministic functions of the sole task
source field ``observed_gas_peaking_task_mw``.  Load, wind and solar are used
only after all task quantities have been constructed, to form the independent
R2 potential-recharge upper bound.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

DEFAULT_PANEL = (
    ROOT
    / "processed"
    / "energy_charts_observed_gas_task_panel_2015_2024.csv.gz"
)
DEFAULT_SAMPLE = (
    ROOT
    / "artifacts"
    / "research"
    / "energy_charts_observed_gas_task_prespecified_sample.csv"
)
DEFAULT_DESIGN = (
    ROOT
    / "artifacts"
    / "research"
    / "OBSERVED_GAS_EVENT_TASK_CONFIRMATORY_DESIGN.md"
)
DEFAULT_OUTPUT = (
    ROOT
    / "processed"
    / "observed_gas_event_task_definitions_v0_1.csv.gz"
)
DEFAULT_COUNTRY_YEAR_QA = (
    ROOT
    / "artifacts"
    / "research"
    / "observed_gas_event_task_definitions_v0_1_country_year_qa.csv"
)
DEFAULT_MACHINE_QA = (
    ROOT
    / "artifacts"
    / "research"
    / "OBSERVED_GAS_EVENT_TASK_STAGE0_QA.json"
)
DEFAULT_MEMO = (
    ROOT
    / "artifacts"
    / "research"
    / "OBSERVED_GAS_EVENT_TASK_STAGE0_MEMO.md"
)

EXPECTED_PANEL_SHA256 = (
    "5f2e8ab49120050fac2ef8bd1b11e20a58863442631a8d9baccf5b924f0bf3ab"
)
EXPECTED_RAW_COUNTRY_YEARS = 155
EXPECTED_RAW_COUNTRIES = 17
EXPECTED_MAIN_COUNTRY_YEARS = 132
EXPECTED_MAIN_COUNTRIES = 15
EXPECTED_HOURS_PER_COUNTRY_YEAR = 8_760
EXPECTED_DAYS_PER_COUNTRY_YEAR = 365
EXPECTED_MAIN_COUNTRY_SET = {
    "AT",
    "BE",
    "BG",
    "CZ",
    "DE",
    "DK",
    "ES",
    "FR",
    "GB",
    "GR",
    "HR",
    "IT",
    "PT",
    "RO",
    "SI",
}

FLOOR_QUANTILE = 0.10
FORCED_POWER_SHARE = 0.01
EVENT_SELECTION_FRACTIONS = {
    "q70": Fraction(3, 10),
    "q75": Fraction(1, 4),
    "q80": Fraction(1, 5),
}
EVENT_QUANTILES = {
    "q70": 0.70,
    "q75": 0.75,
    "q80": 0.80,
}
SEASON_SEGMENTS = {
    1: {"label": "Jan-Feb", "months": (1, 2)},
    2: {"label": "Mar-May", "months": (3, 4, 5)},
    3: {"label": "Jun-Aug", "months": (6, 7, 8)},
    4: {"label": "Sep-Nov", "months": (9, 10, 11)},
    5: {"label": "Dec", "months": (12,)},
}
EXPECTED_SEGMENT_DAY_COUNTS = {1: 59, 2: 92, 3: 92, 4: 91, 5: 31}

TASK_COLUMNS = {
    "c0": "task_c0_mw",
    "c1": "task_c1_mw",
    "c2_q75": "task_c2_q75_mw",
    "c3_q70": "task_c3_q70_mw",
    "c3_q75": "task_c3_q75_mw",
    "c3_q80": "task_c3_q80_mw",
    "c4_q75": "task_c4_q75_mw",
}

OUTPUT_COLUMNS = [
    "timestamp_utc",
    "country",
    "year",
    "source_hour_id",
    "source_day_id",
    "source_hour_of_day",
    "calendar_date_utc",
    "calendar_month_utc",
    "season_segment",
    "season_segment_label",
    "season_days",
    "season_day_rank_by_above_floor_desc",
    "observed_gas_peaking_task_mw",
    "month_hour_q10_floor_mw",
    "above_floor_gas_mw",
    "daily_above_floor_gas_mwh",
    "season_daily_above_floor_q70_mwh",
    "season_daily_above_floor_q75_mwh",
    "season_daily_above_floor_q80_mwh",
    "event_target_days_q70",
    "event_target_days_q75",
    "event_target_days_q80",
    "event_flag_q70",
    "event_flag_q75",
    "event_flag_q80",
    "task_c0_mw",
    "task_c1_mw",
    "task_c2_q75_mw",
    "task_c3_q70_mw",
    "task_c3_q75_mw",
    "task_c3_q80_mw",
    "task_c4_q75_mw",
    "raw_gas_mean_mw",
    "forced_discharge_power_mw",
    "lfp4_nominal_energy_mwh",
    "caes24_nominal_energy_mwh",
    "r2_scaling_alpha",
    "r2_potential_recharge_mw",
    "task_efc_upper_c0_lfp4",
    "task_efc_upper_c0_caes24",
    "task_efc_upper_c1_lfp4",
    "task_efc_upper_c1_caes24",
    "task_efc_upper_c2_q75_lfp4",
    "task_efc_upper_c2_q75_caes24",
    "task_efc_upper_c3_q70_lfp4",
    "task_efc_upper_c3_q70_caes24",
    "task_efc_upper_c3_q75_lfp4",
    "task_efc_upper_c3_q75_caes24",
    "task_efc_upper_c3_q80_lfp4",
    "task_efc_upper_c3_q80_caes24",
    "task_efc_upper_c4_q75_lfp4",
    "task_efc_upper_c4_q75_caes24",
    "source_panel_sha256",
    "prespecified_sample_sha256",
    "confirmatory_design_sha256",
    "derivation_script_sha256",
    "definition_config_sha256",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def as_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes", "y"}
    )


def segment_from_month(month: pd.Series) -> pd.Series:
    conditions = [
        month.le(2),
        month.between(3, 5),
        month.between(6, 8),
        month.between(9, 11),
        month.eq(12),
    ]
    return pd.Series(
        np.select(conditions, [1, 2, 3, 4, 5], default=0),
        index=month.index,
        dtype="int8",
    )


def linear_quantile(values: pd.Series, quantile: float) -> float:
    return float(values.quantile(quantile, interpolation="linear"))


def task_efc_upper(
    task_mw: np.ndarray,
    discharge_power_mw: float,
    duration_h: float,
) -> float:
    numerator_mwh = float(
        np.minimum(task_mw, discharge_power_mw).sum(dtype=np.float64)
    )
    denominator_mwh = duration_h * discharge_power_mw
    if denominator_mwh <= 0.0:
        raise RuntimeError("Task-side EFC denominator must be positive")
    return numerator_mwh / denominator_mwh


def canonical_config() -> dict[str, Any]:
    return {
        "stage": 0,
        "version": "v0_1",
        "task_source_field": "observed_gas_peaking_task_mw",
        "floor": {
            "grouping": ["country", "year", "calendar_month_utc", "UTC_hour"],
            "quantile": FLOOR_QUANTILE,
            "interpolation": "linear",
        },
        "season_segments": SEASON_SEGMENTS,
        "event_quantiles": EVENT_QUANTILES,
        "event_selection": (
            "within each season select ceil((1-q)*N) highest daily "
            "above-floor energies; ties broken by ascending source_day_id"
        ),
        "forced_power_share_of_raw_gas_mean": FORCED_POWER_SHARE,
        "durations_h": {"lfp4": 4.0, "caes24": 24.0},
        "r2": "max(alpha*(observed_wind_mw+observed_solar_mw)-system_load_mw,0)",
        "r2_alpha": (
            "annual system_load_mwh divided by annual observed wind+solar mwh"
        ),
    }


def validate_input_coverage(
    panel: pd.DataFrame,
    sample: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    panel["timestamp_utc"] = pd.to_datetime(
        panel["timestamp_utc"], utc=True, errors="raise"
    )
    panel["year"] = pd.to_numeric(panel["year"], errors="raise").astype(int)
    panel["country"] = panel["country"].astype(str)

    panel_annual = (
        panel.groupby(["country", "year"], as_index=False)
        .agg(
            hours=("timestamp_utc", "size"),
            unique_hours=("timestamp_utc", "nunique"),
            first_hour=("timestamp_utc", "min"),
            last_hour=("timestamp_utc", "max"),
            eligible=("gas_task_eligible", lambda x: bool(as_bool(x).all())),
            material=(
                "gas_task_material_share_ge_1pct",
                lambda x: bool(as_bool(x).all()),
            ),
        )
        .sort_values(["country", "year"])
        .reset_index(drop=True)
    )
    sample = sample.copy()
    sample["country"] = sample["country"].astype(str)
    sample["year"] = pd.to_numeric(sample["year"], errors="raise").astype(int)
    sample["main_prespecified_sample"] = as_bool(
        sample["main_prespecified_sample"]
    )

    joined = panel_annual.merge(
        sample[
            [
                "country",
                "year",
                "main_prespecified_sample",
                "system_load_mwh",
                "raw_gas_mwh",
                "above_floor_gas_mwh",
            ]
        ],
        on=["country", "year"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    joined["panel_main_flags"] = joined["eligible"] & joined["material"]
    main = joined[joined["main_prespecified_sample"].fillna(False)].copy()

    checks = {
        "raw_panel_has_155_country_years": len(panel_annual)
        == EXPECTED_RAW_COUNTRY_YEARS,
        "raw_panel_has_17_countries": panel_annual["country"].nunique()
        == EXPECTED_RAW_COUNTRIES,
        "sample_has_155_country_years": len(sample)
        == EXPECTED_RAW_COUNTRY_YEARS,
        "sample_has_17_countries": sample["country"].nunique()
        == EXPECTED_RAW_COUNTRIES,
        "panel_and_sample_country_years_match": bool(
            joined["_merge"].eq("both").all()
        ),
        "all_raw_country_years_have_8760_hours": bool(
            panel_annual["hours"].eq(EXPECTED_HOURS_PER_COUNTRY_YEAR).all()
        ),
        "all_raw_country_years_have_unique_hours": bool(
            panel_annual["unique_hours"].eq(panel_annual["hours"]).all()
        ),
        "sample_rule_matches_hourly_flags": bool(
            joined["main_prespecified_sample"]
            .fillna(False)
            .eq(joined["panel_main_flags"].fillna(False))
            .all()
        ),
        "main_sample_has_132_country_years": len(main)
        == EXPECTED_MAIN_COUNTRY_YEARS,
        "main_sample_has_15_countries": main["country"].nunique()
        == EXPECTED_MAIN_COUNTRIES,
        "main_country_set_is_frozen": set(main["country"])
        == EXPECTED_MAIN_COUNTRY_SET,
        "ch_and_ee_excluded_from_main": not bool(
            main["country"].isin({"CH", "EE"}).any()
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "Input coverage contract failed:\n"
            + json.dumps(checks, indent=2, default=str)
        )
    return main.sort_values(["country", "year"]), checks


def build_country_year(
    group: pd.DataFrame,
    sample_row: pd.Series,
    hashes: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    group = group.sort_values("timestamp_utc").reset_index(drop=True).copy()
    country = str(group["country"].iloc[0])
    year = int(group["year"].iloc[0])
    if len(group) != EXPECTED_HOURS_PER_COUNTRY_YEAR:
        raise RuntimeError(f"{country}-{year} does not have 8,760 rows")
    if group["timestamp_utc"].nunique() != EXPECTED_HOURS_PER_COUNTRY_YEAR:
        raise RuntimeError(f"{country}-{year} has duplicate UTC hours")

    group["source_hour_id"] = np.arange(
        EXPECTED_HOURS_PER_COUNTRY_YEAR, dtype=np.int32
    )
    group["source_day_id"] = (
        group["source_hour_id"] // 24
    ).astype(np.int16)
    group["source_hour_of_day"] = group["timestamp_utc"].dt.hour.astype(
        np.int8
    )
    group["calendar_date_utc"] = group["timestamp_utc"].dt.strftime("%Y-%m-%d")
    group["calendar_month_utc"] = group["timestamp_utc"].dt.month.astype(
        np.int8
    )
    group["season_segment"] = segment_from_month(
        group["calendar_month_utc"]
    )
    group["season_segment_label"] = group["season_segment"].map(
        {key: value["label"] for key, value in SEASON_SEGMENTS.items()}
    )

    raw = pd.to_numeric(
        group["observed_gas_peaking_task_mw"], errors="raise"
    ).to_numpy(dtype=float)
    if not np.isfinite(raw).all() or (raw < 0.0).any():
        raise RuntimeError(f"{country}-{year} has invalid raw gas task")

    group["month_hour_q10_floor_mw"] = (
        group.groupby(
            ["calendar_month_utc", "source_hour_of_day"], sort=True
        )["observed_gas_peaking_task_mw"]
        .transform(lambda x: linear_quantile(x, FLOOR_QUANTILE))
        .astype(float)
    )
    floor = group["month_hour_q10_floor_mw"].to_numpy(dtype=float)
    above_floor = np.maximum(raw - floor, 0.0)
    group["above_floor_gas_mw"] = above_floor

    day_hour_counts = group.groupby("source_day_id").agg(
        hours=("timestamp_utc", "size"),
        dates=("calendar_date_utc", "nunique"),
        segment_count=("season_segment", "nunique"),
        season_segment=("season_segment", "first"),
        season_segment_label=("season_segment_label", "first"),
        daily_above_floor_gas_mwh=("above_floor_gas_mw", "sum"),
    )
    if len(day_hour_counts) != EXPECTED_DAYS_PER_COUNTRY_YEAR:
        raise RuntimeError(f"{country}-{year} does not have 365 model days")
    if not (
        day_hour_counts["hours"].eq(24).all()
        and day_hour_counts["dates"].eq(1).all()
        and day_hour_counts["segment_count"].eq(1).all()
    ):
        raise RuntimeError(f"{country}-{year} has an invalid intact UTC day")

    daily = day_hour_counts[
        [
            "season_segment",
            "season_segment_label",
            "daily_above_floor_gas_mwh",
        ]
    ].copy()
    daily["season_days"] = daily.groupby("season_segment")[
        "daily_above_floor_gas_mwh"
    ].transform("size").astype(int)
    daily["season_day_rank_by_above_floor_desc"] = 0
    for segment, subset in daily.groupby("season_segment", sort=True):
        ordered = (
            subset.reset_index()
            .sort_values(
                ["daily_above_floor_gas_mwh", "source_day_id"],
                ascending=[False, True],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )
        rank = pd.Series(
            np.arange(1, len(ordered) + 1, dtype=np.int16),
            index=ordered["source_day_id"].to_numpy(),
        )
        daily.loc[
            rank.index, "season_day_rank_by_above_floor_desc"
        ] = rank

        values = subset["daily_above_floor_gas_mwh"]
        for label, quantile in EVENT_QUANTILES.items():
            threshold = linear_quantile(values, quantile)
            daily.loc[
                subset.index,
                f"season_daily_above_floor_{label}_mwh",
            ] = threshold

        if int(segment) not in EXPECTED_SEGMENT_DAY_COUNTS:
            raise RuntimeError(f"Unexpected season segment {segment}")

    for label, selection_fraction in EVENT_SELECTION_FRACTIONS.items():
        targets = (
            daily["season_days"]
            .map(
                lambda n: math.ceil(
                    int(n)
                    * selection_fraction.numerator
                    / selection_fraction.denominator
                )
            )
            .astype(int)
        )
        daily[f"event_target_days_{label}"] = targets
        daily[f"event_flag_{label}"] = (
            daily["season_day_rank_by_above_floor_desc"] <= targets
        )

    daily_merge_columns = [
        "daily_above_floor_gas_mwh",
        "season_days",
        "season_day_rank_by_above_floor_desc",
        "season_daily_above_floor_q70_mwh",
        "season_daily_above_floor_q75_mwh",
        "season_daily_above_floor_q80_mwh",
        "event_target_days_q70",
        "event_target_days_q75",
        "event_target_days_q80",
        "event_flag_q70",
        "event_flag_q75",
        "event_flag_q80",
    ]
    group = group.merge(
        daily[daily_merge_columns],
        left_on="source_day_id",
        right_index=True,
        how="left",
        validate="many_to_one",
    )

    # Task definitions.  This block intentionally has no load, wind or solar
    # input: every candidate is a deterministic non-negative subquantity of G.
    event_q70 = group["event_flag_q70"].to_numpy(dtype=bool)
    event_q75 = group["event_flag_q75"].to_numpy(dtype=bool)
    event_q80 = group["event_flag_q80"].to_numpy(dtype=bool)
    daily_energy = group["daily_above_floor_gas_mwh"].to_numpy(dtype=float)
    daily_q75 = group[
        "season_daily_above_floor_q75_mwh"
    ].to_numpy(dtype=float)
    c4_fraction = np.divide(
        np.maximum(daily_energy - daily_q75, 0.0),
        daily_energy,
        out=np.zeros_like(daily_energy),
        where=daily_energy > 0.0,
    )
    tasks = {
        "c0": raw,
        "c1": above_floor,
        "c2_q75": raw * event_q75,
        "c3_q70": above_floor * event_q70,
        "c3_q75": above_floor * event_q75,
        "c3_q80": above_floor * event_q80,
        "c4_q75": above_floor * c4_fraction,
    }
    for key, column in TASK_COLUMNS.items():
        group[column] = tasks[key]

    raw_mean = float(raw.mean())
    discharge_power = FORCED_POWER_SHARE * raw_mean
    if discharge_power <= 0.0:
        raise RuntimeError(f"{country}-{year} has zero forced power")
    group["raw_gas_mean_mw"] = raw_mean
    group["forced_discharge_power_mw"] = discharge_power
    group["lfp4_nominal_energy_mwh"] = 4.0 * discharge_power
    group["caes24_nominal_energy_mwh"] = 24.0 * discharge_power

    # R2 is computed only after the task block.  It is an independent
    # potential-recharge upper bound and never enters C0--C4.
    system_load = pd.to_numeric(
        group["system_load_mw"], errors="raise"
    ).to_numpy(dtype=float)
    observed_wind = pd.to_numeric(
        group["observed_wind_mw"], errors="raise"
    ).to_numpy(dtype=float)
    observed_solar = pd.to_numeric(
        group["observed_solar_mw"], errors="raise"
    ).to_numpy(dtype=float)
    observed_vre = observed_wind + observed_solar
    if (
        not np.isfinite(system_load).all()
        or not np.isfinite(observed_vre).all()
        or system_load.sum() <= 0.0
        or observed_vre.sum() <= 0.0
    ):
        raise RuntimeError(f"{country}-{year} cannot form R2")
    alpha = float(system_load.sum() / observed_vre.sum())
    r2 = np.maximum(alpha * observed_vre - system_load, 0.0)
    group["r2_scaling_alpha"] = alpha
    group["r2_potential_recharge_mw"] = r2

    efc_values: dict[str, float] = {}
    for key, task in tasks.items():
        lfp_value = task_efc_upper(task, discharge_power, 4.0)
        caes_value = task_efc_upper(task, discharge_power, 24.0)
        group[f"task_efc_upper_{key}_lfp4"] = lfp_value
        group[f"task_efc_upper_{key}_caes24"] = caes_value
        efc_values[f"task_efc_upper_{key}_lfp4"] = lfp_value
        efc_values[f"task_efc_upper_{key}_caes24"] = caes_value

    for column, value in hashes.items():
        group[column] = value

    task_stack = np.column_stack(list(tasks.values()))
    candidate_nonnegative = bool((task_stack >= -1e-12).all())
    candidate_subquantity = bool(
        (task_stack <= raw[:, np.newaxis] + 1e-10).all()
    )
    c0_identity_error = float(np.max(np.abs(tasks["c0"] - raw)))
    c1_identity_error = float(
        np.max(np.abs(tasks["c1"] - np.maximum(raw - floor, 0.0)))
    )
    r2_formula_error = float(
        np.max(np.abs(r2 - np.maximum(alpha * observed_vre - system_load, 0.0)))
    )

    segment_counts = daily.groupby("season_segment").size().to_dict()
    segment_count_ok = segment_counts == EXPECTED_SEGMENT_DAY_COUNTS
    expected_event_counts: dict[str, int] = {}
    event_count_checks: dict[str, bool] = {}
    event_counts_by_segment: dict[str, int] = {}
    for label, selection_fraction in EVENT_SELECTION_FRACTIONS.items():
        expected = sum(
            math.ceil(
                days
                * selection_fraction.numerator
                / selection_fraction.denominator
            )
            for days in EXPECTED_SEGMENT_DAY_COUNTS.values()
        )
        actual = int(daily[f"event_flag_{label}"].sum())
        expected_event_counts[label] = expected
        event_count_checks[label] = actual == expected
        for segment, segment_days in EXPECTED_SEGMENT_DAY_COUNTS.items():
            expected_segment = math.ceil(
                segment_days
                * selection_fraction.numerator
                / selection_fraction.denominator
            )
            actual_segment = int(
                daily.loc[
                    daily["season_segment"].eq(segment),
                    f"event_flag_{label}",
                ].sum()
            )
            event_counts_by_segment[
                f"event_days_{label}_season_{segment}"
            ] = actual_segment
            event_count_checks[
                f"{label}_season_{segment}"
            ] = actual_segment == expected_segment

    annual_energy = {
        f"{key}_mwh": float(value.sum(dtype=np.float64))
        for key, value in tasks.items()
    }
    positive_hours = {
        f"{key}_positive_hours": int(np.count_nonzero(value > 0.0))
        for key, value in tasks.items()
    }
    row_checks = {
        "hours_8760": len(group) == EXPECTED_HOURS_PER_COUNTRY_YEAR,
        "unique_hours_8760": group["timestamp_utc"].nunique()
        == EXPECTED_HOURS_PER_COUNTRY_YEAR,
        "days_365": len(daily) == EXPECTED_DAYS_PER_COUNTRY_YEAR,
        "each_day_has_24_hours": bool(day_hour_counts["hours"].eq(24).all()),
        "five_segment_day_counts_exact": segment_count_ok,
        "q70_rank_count_exact": event_count_checks["q70"],
        "q75_rank_count_exact": event_count_checks["q75"],
        "q80_rank_count_exact": event_count_checks["q80"],
        "all_q70_segment_rank_counts_exact": all(
            event_count_checks[f"q70_season_{segment}"]
            for segment in EXPECTED_SEGMENT_DAY_COUNTS
        ),
        "all_q75_segment_rank_counts_exact": all(
            event_count_checks[f"q75_season_{segment}"]
            for segment in EXPECTED_SEGMENT_DAY_COUNTS
        ),
        "all_q80_segment_rank_counts_exact": all(
            event_count_checks[f"q80_season_{segment}"]
            for segment in EXPECTED_SEGMENT_DAY_COUNTS
        ),
        "raw_gas_nonnegative": bool((raw >= 0.0).all()),
        "all_candidates_nonnegative": candidate_nonnegative,
        "all_candidates_are_raw_gas_subquantities": candidate_subquantity,
        "c0_is_exact_raw_gas": c0_identity_error == 0.0,
        "c1_is_exact_above_floor": c1_identity_error <= 1e-12,
        "r2_nonnegative": bool((r2 >= 0.0).all()),
        "r2_formula_exact": r2_formula_error <= 1e-12,
        "raw_gas_energy_reconciles_sample": bool(
            np.isclose(
                annual_energy["c0_mwh"],
                float(sample_row["raw_gas_mwh"]),
                rtol=1e-12,
                atol=1e-6,
            )
        ),
        "above_floor_energy_reconciles_sample": bool(
            np.isclose(
                annual_energy["c1_mwh"],
                float(sample_row["above_floor_gas_mwh"]),
                rtol=1e-12,
                atol=1e-6,
            )
        ),
        "system_load_energy_reconciles_sample": bool(
            np.isclose(
                float(system_load.sum(dtype=np.float64)),
                float(sample_row["system_load_mwh"]),
                rtol=1e-12,
                atol=1e-6,
            )
        ),
        "c3_q80_subset_c3_q75": bool(
            (tasks["c3_q80"] <= tasks["c3_q75"] + 1e-10).all()
        ),
        "c3_q75_subset_c3_q70": bool(
            (tasks["c3_q75"] <= tasks["c3_q70"] + 1e-10).all()
        ),
        "c4_q75_subset_c3_q75": bool(
            (tasks["c4_q75"] <= tasks["c3_q75"] + 1e-10).all()
        ),
    }
    failed_checks = sorted(key for key, value in row_checks.items() if not value)
    qa_row: dict[str, Any] = {
        "country": country,
        "year": year,
        "hours": len(group),
        "unique_hours": int(group["timestamp_utc"].nunique()),
        "days": len(daily),
        "season_1_days": int(segment_counts.get(1, 0)),
        "season_2_days": int(segment_counts.get(2, 0)),
        "season_3_days": int(segment_counts.get(3, 0)),
        "season_4_days": int(segment_counts.get(4, 0)),
        "season_5_days": int(segment_counts.get(5, 0)),
        "event_days_q70": int(daily["event_flag_q70"].sum()),
        "event_days_q75": int(daily["event_flag_q75"].sum()),
        "event_days_q80": int(daily["event_flag_q80"].sum()),
        **event_counts_by_segment,
        "raw_gas_mean_mw": raw_mean,
        "forced_discharge_power_mw": discharge_power,
        "raw_gas_mwh": annual_energy["c0_mwh"],
        "above_floor_gas_mwh": annual_energy["c1_mwh"],
        "r2_potential_recharge_mwh": float(r2.sum(dtype=np.float64)),
        "r2_scaling_alpha": alpha,
        "c0_identity_max_abs_error_mw": c0_identity_error,
        "c1_identity_max_abs_error_mw": c1_identity_error,
        "r2_formula_max_abs_error_mw": r2_formula_error,
        **annual_energy,
        **positive_hours,
        **efc_values,
        "qa_pass": not failed_checks,
        "failed_checks": ";".join(failed_checks),
        **hashes,
    }
    if failed_checks:
        raise RuntimeError(
            f"{country}-{year} Stage-0 QA failed: {failed_checks}"
        )

    group["timestamp_utc"] = group["timestamp_utc"].dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    missing_output_columns = sorted(set(OUTPUT_COLUMNS) - set(group.columns))
    if missing_output_columns:
        raise RuntimeError(
            f"Missing output columns for {country}-{year}: "
            f"{missing_output_columns}"
        )
    return group[OUTPUT_COLUMNS], qa_row


def write_deterministic_csv_gz(
    output: Path,
    frames: list[pd.DataFrame],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_handle,
            compresslevel=6,
            mtime=0,
        ) as gzip_handle:
            with io.TextIOWrapper(
                gzip_handle, encoding="utf-8", newline=""
            ) as text_handle:
                for index, frame in enumerate(frames):
                    frame.to_csv(
                        text_handle,
                        index=False,
                        header=index == 0,
                        float_format="%.15g",
                        lineterminator="\n",
                    )
    temporary.replace(output)


def build_memo(
    qa: dict[str, Any],
    country_year_qa: pd.DataFrame,
) -> str:
    sums = {
        key: float(country_year_qa[key].sum())
        for key in [
            "raw_gas_mwh",
            "above_floor_gas_mwh",
            "c2_q75_mwh",
            "c3_q75_mwh",
            "c4_q75_mwh",
            "r2_potential_recharge_mwh",
        ]
    }
    return f"""# Observed-gas event task Stage 0 memo

版本为 v0.1，生成日期为 2026-07-27。Stage 0 只完成任务定义与数据合同审计，
没有构造反事实日序、调用求解器或产生效应估计。

## 冻结样本

- 原始覆盖为 {qa['coverage']['raw_countries']} 个国家和
  {qa['coverage']['raw_country_years']} 个国家—年份。
- 预注册主样本为 {qa['coverage']['main_countries']} 个国家和
  {qa['coverage']['main_country_years']} 个国家—年份，共
  {qa['coverage']['main_rows']:,} 个逐时观测。
- 每个主样本国家—年份均为 8,760 个唯一 UTC 小时和 365 个完整日块。
- CH 和 EE 保留在原始覆盖审计中，不进入 15 国、132 国家—年份主样本。

## 任务与补能合同

C0--C4 只由逐时实际聚合 `Fossil gas` 字段构造。month-hour q10 下界在
原始日序上计算。q70、q75 和 q80 事件日在五个固定季节段内按日 above-floor
能量降序选取，严格使用 `ceil((1-q)N_s)` 个最高排名日，并以
`source_day_id` 升序打破并列。主任务为 `task_c3_q75_mw`。

R2 为 `max(alpha*(wind+solar)-system_load,0)`，只写入
`r2_potential_recharge_mw` 作为高风光反事实下的潜在补能上界，不进入任何
任务公式，也不解释为观测弃电。

## 守恒与子量核验

- 132 个国家—年份均通过逐时非负、C0 原始任务恒等、C0--C4 不超过原始燃气、
  q70/q75/q80 季节内精确事件日数、年度源能量重构和 R2 公式审计。
- 各国—年固定事件日数为 q70={int(country_year_qa['event_days_q70'].iloc[0])}、
  q75={int(country_year_qa['event_days_q75'].iloc[0])} 和
  q80={int(country_year_qa['event_days_q80'].iloc[0])}。
- 主样本原始燃气合计为 {sums['raw_gas_mwh']:,.1f} MWh，above-floor
  合计为 {sums['above_floor_gas_mwh']:,.1f} MWh，C3 q75 合计为
  {sums['c3_q75_mwh']:,.1f} MWh。
- C2 q75 和 C4 q75 合计分别为 {sums['c2_q75_mwh']:,.1f} 和
  {sums['c4_q75_mwh']:,.1f} MWh。R2 潜在补能合计为
  {sums['r2_potential_recharge_mwh']:,.1f} MWh。

任务侧 EFC 上限均采用原始燃气年均值的 1% 作为共同放电功率基数，分别按
4 h LFP 和 24 h CAES 的名义能量容量计算。改变事件门控不会重新缩放资产。

## 产物与哈希

- 逐时定义表：`{qa['outputs']['hourly_table']}`
- 国家—年份 QA：`{qa['outputs']['country_year_qa']}`
- 机器 QA：`{qa['outputs']['machine_qa']}`
- 逐时表 SHA-256：`{qa['hashes']['hourly_output_sha256']}`
- 国家—年份 QA SHA-256：`{qa['hashes']['country_year_qa_sha256']}`
- 源 panel SHA-256：`{qa['hashes']['source_panel_sha256']}`
- 派生脚本 SHA-256：`{qa['hashes']['derivation_script_sha256']}`

本产物只准入 Stage 1 的结构 smoke test，不构成任务聚集、补能机制或技术经济
效果的结果证据。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--country-year-qa", type=Path, default=DEFAULT_COUNTRY_YEAR_QA
    )
    parser.add_argument("--machine-qa", type=Path, default=DEFAULT_MACHINE_QA)
    parser.add_argument("--memo", type=Path, default=DEFAULT_MEMO)
    args = parser.parse_args()

    for path in [args.panel, args.sample, args.design]:
        if not path.exists():
            raise FileNotFoundError(path)
    panel_sha = sha256(args.panel)
    if panel_sha != EXPECTED_PANEL_SHA256:
        raise RuntimeError(
            "Frozen source panel SHA-256 changed: "
            f"expected {EXPECTED_PANEL_SHA256}, observed {panel_sha}"
        )

    config_text = json.dumps(
        canonical_config(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    hashes = {
        "source_panel_sha256": panel_sha,
        "prespecified_sample_sha256": sha256(args.sample),
        "confirmatory_design_sha256": sha256(args.design),
        "derivation_script_sha256": sha256(Path(__file__).resolve()),
        "definition_config_sha256": sha256_text(config_text),
    }

    required_panel_columns = [
        "timestamp_utc",
        "country",
        "year",
        "system_load_mw",
        "observed_wind_mw",
        "observed_solar_mw",
        "observed_gas_peaking_task_mw",
        "gas_task_eligible",
        "gas_task_material_share_ge_1pct",
    ]
    panel = pd.read_csv(args.panel, usecols=required_panel_columns)
    sample = pd.read_csv(args.sample)
    main_coverage, coverage_checks = validate_input_coverage(panel, sample)
    main_keys = main_coverage[["country", "year"]].copy()
    main_panel = panel.merge(
        main_keys,
        on=["country", "year"],
        how="inner",
        validate="many_to_one",
    )
    main_sample = sample[
        as_bool(sample["main_prespecified_sample"])
    ].copy()
    sample_lookup = main_sample.set_index(["country", "year"])

    frames: list[pd.DataFrame] = []
    qa_rows: list[dict[str, Any]] = []
    for (country, year), group in main_panel.groupby(
        ["country", "year"], sort=True
    ):
        sample_row = sample_lookup.loc[(country, int(year))]
        derived, qa_row = build_country_year(group, sample_row, hashes)
        frames.append(derived)
        qa_rows.append(qa_row)

    country_year_qa = (
        pd.DataFrame(qa_rows)
        .sort_values(["country", "year"])
        .reset_index(drop=True)
    )
    prewrite_checks = {
        "country_year_qa_has_132_rows": len(country_year_qa)
        == EXPECTED_MAIN_COUNTRY_YEARS,
        "country_year_qa_has_15_countries": country_year_qa[
            "country"
        ].nunique()
        == EXPECTED_MAIN_COUNTRIES,
        "all_country_year_rows_pass": bool(country_year_qa["qa_pass"].all()),
        "derived_frame_count_is_132": len(frames)
        == EXPECTED_MAIN_COUNTRY_YEARS,
        "derived_rows_equal_132_times_8760": sum(map(len, frames))
        == EXPECTED_MAIN_COUNTRY_YEARS * EXPECTED_HOURS_PER_COUNTRY_YEAR,
        "all_derived_frames_have_expected_columns": all(
            list(frame.columns) == OUTPUT_COLUMNS for frame in frames
        ),
    }
    if not all(prewrite_checks.values()):
        raise RuntimeError(
            "Stage-0 prewrite QA failed:\n"
            + json.dumps(prewrite_checks, indent=2)
        )

    write_deterministic_csv_gz(args.output, frames)
    args.country_year_qa.parent.mkdir(parents=True, exist_ok=True)
    country_year_qa.to_csv(
        args.country_year_qa,
        index=False,
        float_format="%.15g",
        lineterminator="\n",
    )

    # Independent post-write structural read.  Only identifiers and core task
    # fields are reloaded, limiting memory while auditing the serialized file.
    audit_columns = [
        "country",
        "year",
        "source_hour_id",
        "source_day_id",
        "observed_gas_peaking_task_mw",
        "task_c0_mw",
        "task_c1_mw",
        "task_c2_q75_mw",
        "task_c3_q70_mw",
        "task_c3_q75_mw",
        "task_c3_q80_mw",
        "task_c4_q75_mw",
        "r2_potential_recharge_mw",
    ]
    serialized = pd.read_csv(args.output, usecols=audit_columns)
    serialized_counts = serialized.groupby(["country", "year"]).agg(
        hours=("source_hour_id", "size"),
        unique_hours=("source_hour_id", "nunique"),
        days=("source_day_id", "nunique"),
    )
    serialized_task_columns = [
        "task_c0_mw",
        "task_c1_mw",
        "task_c2_q75_mw",
        "task_c3_q70_mw",
        "task_c3_q75_mw",
        "task_c3_q80_mw",
        "task_c4_q75_mw",
    ]
    serialized_stack = serialized[serialized_task_columns].to_numpy(float)
    serialized_raw = serialized[
        "observed_gas_peaking_task_mw"
    ].to_numpy(float)
    postwrite_checks = {
        "serialized_rows_equal_1156320": len(serialized)
        == EXPECTED_MAIN_COUNTRY_YEARS * EXPECTED_HOURS_PER_COUNTRY_YEAR,
        "serialized_country_years_equal_132": len(serialized_counts)
        == EXPECTED_MAIN_COUNTRY_YEARS,
        "serialized_countries_equal_15": serialized["country"].nunique()
        == EXPECTED_MAIN_COUNTRIES,
        "serialized_all_country_years_have_8760_hours": bool(
            serialized_counts["hours"]
            .eq(EXPECTED_HOURS_PER_COUNTRY_YEAR)
            .all()
        ),
        "serialized_all_hours_unique_within_country_year": bool(
            serialized_counts["unique_hours"]
            .eq(EXPECTED_HOURS_PER_COUNTRY_YEAR)
            .all()
        ),
        "serialized_all_country_years_have_365_days": bool(
            serialized_counts["days"]
            .eq(EXPECTED_DAYS_PER_COUNTRY_YEAR)
            .all()
        ),
        "serialized_c0_equals_raw": bool(
            np.allclose(
                serialized["task_c0_mw"].to_numpy(float),
                serialized_raw,
                rtol=0.0,
                atol=1e-12,
            )
        ),
        "serialized_candidates_nonnegative": bool(
            (serialized_stack >= -1e-12).all()
        ),
        "serialized_candidates_are_raw_subquantities": bool(
            (serialized_stack <= serialized_raw[:, np.newaxis] + 1e-10).all()
        ),
        "serialized_r2_nonnegative": bool(
            (
                serialized["r2_potential_recharge_mw"].to_numpy(float)
                >= -1e-12
            ).all()
        ),
    }
    del serialized

    all_checks = {
        **coverage_checks,
        **prewrite_checks,
        **postwrite_checks,
    }
    status = "pass" if all(all_checks.values()) else "fail"
    machine_qa: dict[str, Any] = {
        "status": status,
        "stage": 0,
        "version": "v0_1",
        "scope": (
            "task construction and data-contract audit only; no chronology "
            "intervention and no optimization"
        ),
        "checks": all_checks,
        "coverage": {
            "raw_countries": int(panel["country"].nunique()),
            "raw_country_years": int(
                panel[["country", "year"]].drop_duplicates().shape[0]
            ),
            "raw_rows": int(len(panel)),
            "main_countries": int(country_year_qa["country"].nunique()),
            "main_country_years": int(len(country_year_qa)),
            "main_rows": int(
                country_year_qa["hours"].sum()
            ),
            "main_country_set": sorted(
                country_year_qa["country"].unique().tolist()
            ),
            "excluded_country_sets_retained_in_raw_audit": ["CH", "EE"],
        },
        "definitions": canonical_config(),
        "task_dependency_contract": {
            "task_source_field": "observed_gas_peaking_task_mw",
            "forbidden_task_inputs": [
                "system_load_mw",
                "observed_wind_mw",
                "observed_solar_mw",
                "residual_load",
                "simulated_gas",
            ],
            "r2_role": "recharge-side upper bound only",
        },
        "outputs": {
            "hourly_table": str(args.output.resolve()),
            "country_year_qa": str(args.country_year_qa.resolve()),
            "machine_qa": str(args.machine_qa.resolve()),
            "memo": str(args.memo.resolve()),
        },
        "hashes": {
            **hashes,
            "hourly_output_sha256": sha256(args.output),
            "country_year_qa_sha256": sha256(args.country_year_qa),
        },
        "output_columns": OUTPUT_COLUMNS,
        "country_year_qa_summary": {
            "rows": int(len(country_year_qa)),
            "passing_rows": int(country_year_qa["qa_pass"].sum()),
            "event_days_q70_unique": sorted(
                country_year_qa["event_days_q70"].unique().tolist()
            ),
            "event_days_q75_unique": sorted(
                country_year_qa["event_days_q75"].unique().tolist()
            ),
            "event_days_q80_unique": sorted(
                country_year_qa["event_days_q80"].unique().tolist()
            ),
        },
        "parquet_note": (
            "CSV.GZ is the frozen Stage-0 format because pyarrow is not "
            "installed in the execution environment."
        ),
    }
    if status != "pass":
        raise RuntimeError(
            "Stage-0 post-write QA failed:\n"
            + json.dumps(machine_qa, indent=2, default=str)
        )

    args.machine_qa.parent.mkdir(parents=True, exist_ok=True)
    args.machine_qa.write_text(
        json.dumps(machine_qa, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    args.memo.parent.mkdir(parents=True, exist_ok=True)
    args.memo.write_text(
        build_memo(machine_qa, country_year_qa),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": status,
                "hourly_output": str(args.output),
                "hourly_rows": machine_qa["coverage"]["main_rows"],
                "country_year_qa": str(args.country_year_qa),
                "country_year_rows": len(country_year_qa),
                "machine_qa": str(args.machine_qa),
                "memo": str(args.memo),
                "hourly_output_sha256": machine_qa["hashes"][
                    "hourly_output_sha256"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
