#!/usr/bin/env python3
"""Build the ERAA 2035 implicit-DSR and dispatchable-load input layer.

The builder reads the verified ERAA 2025 DemandData archive directly.  It
keeps the 52-node, 36-weather-scenario and 8,760-hour phase-1 grid unchanged,
applies each TSO's released price-sensitive share to the released EV and heat
pump component profiles, and exposes the price-sensitive share of the
behind-the-meter battery reference trajectory for explicit redispatch.

The Annex 2 treatment is deliberately narrow:

* EV energy may move only within two 6 h windows and one 12 h overnight
  window;
* heat-pump energy may move only within four 6 h windows;
* price-sensitive BTM batteries use released final power/energy dimensions,
  92% round-trip efficiency and cyclic 50% state of charge;
* electrolysers are optional buyers bounded by zero and nameplate;
* power-to-heat has no released hourly heat-availability profile, so only
  zero/nameplate bounds are published and no central value is admitted;
* the isolated NL0A_OFF wind and electrolyser are excluded as a pair in the
  central topology treatment.

This script prepares inputs only.  It does not modify the full-network runner
or make any manuscript-evidence claim.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable
import zipfile

import numpy as np
import pandas as pd


SCRIPT_VERSION = "0.1.0"
TARGET_YEAR = 2035
HOURS_PER_YEAR = 8_760
WEATHER_SCENARIOS = tuple(f"WS{index:02d}" for index in range(1, 37))
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / "raw" / "eraa_2025" / "ERAA_2025_DemandData.zip"
DEFAULT_ARCHIVE_QA = (
    ROOT
    / "raw"
    / "eraa_2025"
    / "ERAA_2025_DemandData.full_archive_qa.json"
)
DEFAULT_PHASE1 = ROOT / "processed" / "eraa_2035_phase1_inputs_v0_1"
DEFAULT_BATTERY_DATA = (
    ROOT
    / "raw"
    / "eraa_2025"
    / "extracted"
    / "ERAA_2025_DashboardRawData1"
    / "Dashboard Raw Data"
    / "Batteries additional information.csv"
)
DEFAULT_OUTPUT = (
    ROOT / "processed" / "eraa_2035_idsr_and_dispatchable_load_v0_1"
)

RATIO_MEMBER = "Demand data/iDSR_ratio.csv"
ADDITIONAL_MEMBER = "Demand data/Additional dispatchable consumption.csv"
EV_MEMBER = (
    "Demand data/Detailed demand/"
    "{zone}_Demand_EVpart_2035_National Trends.csv"
)
HP_MEMBER = (
    "Demand data/Detailed demand/"
    "{zone}_Demand_HPpart_2035_National Trends.csv"
)

HOURLY_COLUMNS = [
    "weather_scenario",
    "zone",
    "hour_of_year",
    "ev_price_sensitive_baseline_mw",
    "hp_price_sensitive_baseline_mw",
    "btm_price_sensitive_reference_net_output_mw",
]
WINDOW_COLUMNS = [
    "zone",
    "resource_type",
    "time_zone_utc_offset",
    "window_start_hours_utc",
    "window_durations_h",
]
BTM_COLUMNS = [
    "zone",
    "price_sensitive_ratio",
    "source_power_capacity_mw",
    "source_energy_capacity_mwh",
    "price_sensitive_charge_power_mw",
    "price_sensitive_discharge_power_mw",
    "price_sensitive_energy_capacity_mwh",
    "round_trip_efficiency",
    "initial_soc_fraction",
    "terminal_soc_fraction",
    "capacity_record_available",
    "released_reference_profile_available",
    "source_operational_status",
    "source_data_version",
    "phase1_reference_treatment",
]
ADDITIONAL_COLUMNS = [
    "zone",
    "resource_type",
    "nameplate_capacity_mw",
    "weighted_average_efficiency_fraction",
    "activation_price_eur_mwh",
    "availability_lower_bound_mw",
    "availability_upper_bound_mw",
    "hourly_availability_profile_status",
    "central_case_admitted",
    "central_case_treatment",
    "nl0a_paired_isolation",
    "source_country",
]
OUTPUT_FILENAMES = (
    "idsr_hourly_2035.csv.gz",
    "idsr_windows_2035.csv",
    "idsr_btm_battery_2035.csv",
    "additional_dispatchable_consumption_2035.csv",
    "QA.json",
)

# Annex 2 Table 8.  The two values separated by "/" in the source table are
# EV/HP starts.  Hours are retained exactly as stated in UTC for each reported
# standard-time offset.  Daylight-saving shifts are not introduced.
ANNEX2_WINDOWS: dict[int, dict[str, tuple[tuple[int, ...], tuple[int, ...]]]] = {
    -1: {
        "ev": ((6, 12, 18), (6, 6, 12)),
        "hp": ((4, 10, 16, 22), (6, 6, 6, 6)),
    },
    0: {
        "ev": ((7, 13, 19), (6, 6, 12)),
        "hp": ((5, 11, 17, 23), (6, 6, 6, 6)),
    },
    1: {
        "ev": ((8, 14, 20), (6, 6, 12)),
        "hp": ((6, 12, 18, 0), (6, 6, 6, 6)),
    },
    2: {
        "ev": ((9, 15, 21), (6, 6, 12)),
        "hp": ((1, 7, 13, 19), (6, 6, 6, 6)),
    },
}

UTC_ZERO_COUNTRIES = {"IE", "PT", "UK"}
UTC_PLUS_TWO_COUNTRIES = {
    "BG",
    "CY",
    "EE",
    "FI",
    "GR",
    "LT",
    "LV",
    "MD",
    "RO",
}
UTC_PLUS_ONE_COUNTRIES = {
    "AL",
    "AT",
    "BA",
    "BE",
    "CH",
    "CZ",
    "DE",
    "DK",
    "ES",
    "FR",
    "HR",
    "HU",
    "IT",
    "LU",
    "ME",
    "MK",
    "MT",
    "NL",
    "NO",
    "PL",
    "RS",
    "SE",
    "SI",
    "SK",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def temporary_path(path: Path) -> Path:
    """Return a process-unique sibling for safe concurrent regeneration."""

    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def atomic_json(data: dict[str, Any], path: Path) -> None:
    temporary = temporary_path(path)
    temporary.write_text(
        json.dumps(json_ready(data), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = temporary_path(path)
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def prepare_output(path: Path, force: bool) -> None:
    if path.exists() and not force:
        existing = [name for name in OUTPUT_FILENAMES if (path / name).exists()]
        if existing:
            raise FileExistsError(
                f"{path} contains {existing}; pass --force to replace them"
            )
    path.mkdir(parents=True, exist_ok=True)
    if force:
        for name in OUTPUT_FILENAMES:
            target = path / name
            if target.exists():
                target.unlink()


def validate_archive(
    archive: Path,
    archive_qa: Path,
) -> dict[str, Any]:
    qa = json.loads(archive_qa.read_text(encoding="utf-8"))
    if not bool(qa.get("pass", False)) or not bool(
        qa.get("zip_integrity_pass", False)
    ):
        raise RuntimeError(f"{archive_qa}: verified archive QA did not pass")
    observed_size = archive.stat().st_size
    if observed_size != int(qa["observed_size_bytes"]):
        raise ValueError(
            f"{archive}: size changed from verified archive QA "
            f"{qa['observed_size_bytes']} to {observed_size}"
        )
    observed_hash = sha256_file(archive)
    if observed_hash != str(qa["archive_sha256"]):
        raise ValueError(f"{archive}: SHA-256 differs from verified archive QA")
    return {
        "archive_bytes": observed_size,
        "archive_sha256": observed_hash,
        "archive_qa_pass": True,
        "archive_member_count": int(qa["zip_member_count"]),
    }


def validate_phase1(phase1: Path) -> dict[str, Any]:
    qa_path = phase1 / "QA.json"
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    if not bool(qa.get("release_eligible", False)):
        raise RuntimeError(f"{qa_path}: phase-1 inputs are not release eligible")
    if int(qa.get("demand_zone_contract_count", -1)) != 52:
        raise ValueError(f"{qa_path}: phase-1 demand-zone contract is not 52")
    if not bool(qa.get("all_selected_hourly_complete", False)):
        raise ValueError(f"{qa_path}: phase-1 hourly inputs are incomplete")
    return {
        "phase1_qa": str(qa_path.resolve()),
        "phase1_qa_sha256": sha256_file(qa_path),
        "phase1_release_eligible": True,
    }


def demand_zones(phase1: Path) -> list[str]:
    coverage_path = phase1 / "zone_coverage.csv"
    coverage = pd.read_csv(coverage_path)
    required = {
        "zone",
        "expected_demand_zone",
        "hourly_shard_written",
        "btm_battery_file_available",
    }
    missing = sorted(required.difference(coverage.columns))
    if missing:
        raise ValueError(f"{coverage_path}: missing columns {missing}")
    selected = coverage.loc[
        coverage["expected_demand_zone"].astype(bool)
    ].copy()
    if len(selected) != 52 or not selected["hourly_shard_written"].astype(
        bool
    ).all():
        raise ValueError("phase-1 demand-zone coverage is not complete")
    zones = sorted(selected["zone"].astype(str).tolist())
    if len(set(zones)) != 52:
        raise ValueError("phase-1 demand-zone IDs are not unique")
    return zones


def read_zip_csv(
    archive: zipfile.ZipFile,
    member: str,
    **kwargs: Any,
) -> pd.DataFrame:
    try:
        with archive.open(member) as handle:
            frame = pd.read_csv(handle, **kwargs)
    except KeyError as error:
        raise FileNotFoundError(f"archive member missing: {member}") from error
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def read_ratios(
    archive: zipfile.ZipFile,
    zones: Iterable[str],
) -> pd.DataFrame:
    source = read_zip_csv(archive, RATIO_MEMBER)
    rename = {
        "MARKET_NODE": "zone",
        "TARGET_YEAR": "target_year",
        "Ratio of price-sensitive consumer - Evs": "ev_ratio",
        "Ratio of price-sensitive consumer - Heat Pumps": "hp_ratio",
        (
            "Ratio of price-sensitive consumer - "
            "Non-Market Participating Batteries"
        ): "btm_ratio",
    }
    missing = sorted(set(rename).difference(source.columns))
    if missing:
        raise ValueError(f"{RATIO_MEMBER}: missing columns {missing}")
    frame = source.rename(columns=rename)[list(rename.values())].copy()
    frame["target_year"] = pd.to_numeric(
        frame["target_year"], errors="raise"
    ).astype(int)
    frame = frame.loc[frame["target_year"].eq(TARGET_YEAR)].copy()
    frame["zone"] = frame["zone"].astype(str)
    if frame["zone"].duplicated().any():
        raise ValueError(f"{RATIO_MEMBER}: duplicate 2035 zone ratios")
    expected = set(zones)
    observed = set(frame["zone"])
    if observed != expected:
        raise ValueError(
            f"{RATIO_MEMBER}: 2035 zone mismatch; "
            f"missing={sorted(expected-observed)}, extra={sorted(observed-expected)}"
        )
    for column in ("ev_ratio", "hp_ratio", "btm_ratio"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not frame[column].between(0.0, 1.0).all():
            raise ValueError(f"{RATIO_MEMBER}: {column} lies outside [0,1]")
    frame = frame.sort_values("zone").reset_index(drop=True)
    exact = frame.set_index("zone")
    if not np.isclose(exact.loc["AT00", "btm_ratio"], 0.134, atol=1e-12):
        raise ValueError("released AT00 2035 BTM ratio is not 13.4%")
    if not np.isclose(exact.loc["CH00", "btm_ratio"], 1.0, atol=1e-12):
        raise ValueError("released CH00 2035 BTM ratio is not 100%")
    return frame


def validate_calendar(frame: pd.DataFrame, member: str) -> None:
    required = {"Month", "Day", "Hour"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{member}: missing calendar columns {missing}")
    if len(frame) != HOURS_PER_YEAR:
        raise ValueError(
            f"{member}: expected {HOURS_PER_YEAR} rows, observed {len(frame)}"
        )
    canonical = pd.date_range(
        "2018-01-01 00:00:00",
        periods=HOURS_PER_YEAR,
        freq="h",
    )
    observed = {
        "Month": pd.to_numeric(frame["Month"], errors="raise").to_numpy(int),
        "Day": pd.to_numeric(frame["Day"], errors="raise").to_numpy(int),
        "Hour": pd.to_numeric(frame["Hour"], errors="raise").to_numpy(int),
    }
    expected = {
        "Month": canonical.month.to_numpy(),
        "Day": canonical.day.to_numpy(),
        "Hour": canonical.hour.to_numpy(),
    }
    for field in ("Month", "Day", "Hour"):
        if not np.array_equal(observed[field], expected[field]):
            raise ValueError(f"{member}: non-canonical non-leap {field} axis")


def read_component_matrix(
    archive: zipfile.ZipFile,
    member: str,
    *,
    allow_signed: bool = False,
) -> np.ndarray:
    frame = read_zip_csv(archive, member)
    required = {"Month", "Day", "Hour", *WEATHER_SCENARIOS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{member}: missing columns {missing}")
    validate_calendar(frame, member)
    values = frame.loc[:, list(WEATHER_SCENARIOS)].apply(
        pd.to_numeric, errors="raise"
    ).to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{member}: non-finite demand values")
    if not allow_signed and (values < -1e-12).any():
        raise ValueError(f"{member}: invalid non-negative demand values")
    return values


def timezone_offset(zone: str) -> int:
    country = str(zone)[:2]
    membership = [
        country in UTC_ZERO_COUNTRIES,
        country in UTC_PLUS_ONE_COUNTRIES,
        country in UTC_PLUS_TWO_COUNTRIES,
    ]
    if sum(membership) != 1:
        raise ValueError(f"{zone}: no unique Annex 2 standard-time mapping")
    if membership[0]:
        return 0
    if membership[1]:
        return 1
    return 2


def validate_window_partition(starts: tuple[int, ...], durations: tuple[int, ...]) -> None:
    if (
        len(starts) != len(durations)
        or len(set(starts)) != len(starts)
        or sum(durations) != 24
    ):
        raise ValueError("Annex 2 window definition is incomplete")
    ordered = sorted(zip(starts, durations))
    for index, (start, duration) in enumerate(ordered):
        next_start = ordered[(index + 1) % len(ordered)][0]
        if (next_start - start) % 24 != duration:
            raise ValueError(
                f"Annex 2 window {start}+{duration} does not meet {next_start}"
            )


def build_windows(zones: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for zone in zones:
        offset = timezone_offset(zone)
        for resource in ("ev", "hp"):
            starts, durations = ANNEX2_WINDOWS[offset][resource]
            validate_window_partition(starts, durations)
            rows.append(
                {
                    "zone": zone,
                    "resource_type": resource,
                    "time_zone_utc_offset": offset,
                    "window_start_hours_utc": "|".join(map(str, starts)),
                    "window_durations_h": "|".join(map(str, durations)),
                }
            )
    frame = pd.DataFrame(rows, columns=WINDOW_COLUMNS)
    if len(frame) != 104 or frame.duplicated(["zone", "resource_type"]).any():
        raise ValueError("expected one EV and one HP window row for 52 zones")
    return frame


def build_btm_parameters(
    zones: list[str],
    ratios: pd.DataFrame,
    battery_path: Path,
    phase1: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = pd.read_csv(battery_path)
    required = {
        "MARKET_NODE",
        "OP_STAT",
        "MAX LOAD CAP (MW)",
        "STORAGE CAPACITY (MWh)",
        "TARGET_YEAR",
        "data_version",
    }
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ValueError(f"{battery_path}: missing columns {missing}")
    status = "Out of market - for PV/battery dispatch optimization"
    source = source.loc[
        pd.to_numeric(source["TARGET_YEAR"], errors="raise").eq(TARGET_YEAR)
        & source["data_version"].astype(str).eq("ERAA 2025 final")
        & source["OP_STAT"].astype(str).eq(status)
    ].copy()
    source["MARKET_NODE"] = source["MARKET_NODE"].astype(str)
    if source["MARKET_NODE"].duplicated().any():
        raise ValueError(f"{battery_path}: duplicate final BTM battery records")
    by_zone = source.set_index("MARKET_NODE")
    ratio_by_zone = ratios.set_index("zone")
    coverage = pd.read_csv(phase1 / "zone_coverage.csv").set_index("zone")
    rows: list[dict[str, Any]] = []
    for zone in zones:
        ratio = float(ratio_by_zone.loc[zone, "btm_ratio"])
        available = zone in by_zone.index
        if available:
            record = by_zone.loc[zone]
            power = float(record["MAX LOAD CAP (MW)"])
            energy = float(record["STORAGE CAPACITY (MWh)"])
        else:
            power = 0.0
            energy = 0.0
        if power < 0 or energy < 0:
            raise ValueError(f"{zone}: negative BTM battery dimension")
        profile_available = bool(
            coverage.loc[zone, "btm_battery_file_available"]
        )
        rows.append(
            {
                "zone": zone,
                "price_sensitive_ratio": ratio,
                "source_power_capacity_mw": power,
                "source_energy_capacity_mwh": energy,
                "price_sensitive_charge_power_mw": power * ratio,
                "price_sensitive_discharge_power_mw": power * ratio,
                "price_sensitive_energy_capacity_mwh": energy * ratio,
                "round_trip_efficiency": 0.92,
                "initial_soc_fraction": 0.50,
                "terminal_soc_fraction": 0.50,
                "capacity_record_available": available,
                "released_reference_profile_available": profile_available,
                "source_operational_status": status if available else "",
                "source_data_version": "ERAA 2025 final" if available else "",
                "phase1_reference_treatment": (
                    "add price-sensitive reference net output to phase1 "
                    "signed balance (load minus BTM surplus) before explicit "
                    "battery redispatch"
                ),
            }
        )
    frame = pd.DataFrame(rows, columns=BTM_COLUMNS)
    if len(frame) != 52 or frame["zone"].duplicated().any():
        raise ValueError("BTM parameter table does not cover 52 zones")
    active = frame.loc[
        frame["price_sensitive_energy_capacity_mwh"].gt(0)
    ]
    if set(active["zone"]) != {"AT00", "CH00"}:
        raise ValueError(
            "released ratios and final BTM capacities should activate AT00 "
            "and CH00 only"
        )
    multiplication_error = max(
        float(
            (
                frame["source_power_capacity_mw"]
                * frame["price_sensitive_ratio"]
                - frame["price_sensitive_charge_power_mw"]
            )
            .abs()
            .max()
        ),
        float(
            (
                frame["source_energy_capacity_mwh"]
                * frame["price_sensitive_ratio"]
                - frame["price_sensitive_energy_capacity_mwh"]
            )
            .abs()
            .max()
        ),
    )
    qa = {
        "rows": len(frame),
        "active_price_sensitive_btm_zones": sorted(active["zone"].tolist()),
        "active_price_sensitive_charge_power_mw": float(
            active["price_sensitive_charge_power_mw"].sum()
        ),
        "active_price_sensitive_energy_capacity_mwh": float(
            active["price_sensitive_energy_capacity_mwh"].sum()
        ),
        "capacity_ratio_multiplication_max_abs_error": multiplication_error,
        "nonzero_ratio_without_released_final_capacity": sorted(
            frame.loc[
                frame["price_sensitive_ratio"].gt(0)
                & ~frame["capacity_record_available"].astype(bool),
                "zone",
            ].tolist()
        ),
    }
    return frame, qa


def build_additional_dispatchable(
    archive: zipfile.ZipFile,
    phase1: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = read_zip_csv(archive, ADDITIONAL_MEMBER)
    rename = {
        "MARKET_NODE": "zone",
        "PEMMDB_PLANT_TYPE": "source_resource_type",
        "MAX CAPACITY (MW)": "nameplate_capacity_mw",
        (
            "Weighted Average Electrolyser Efficiency"
        ): "weighted_average_efficiency_percent",
        "Weighted Average Activation Price": "activation_price_eur_mwh",
        "TARGET_YEAR": "target_year",
        "Country": "source_country",
    }
    missing = sorted(set(rename).difference(source.columns))
    if missing:
        raise ValueError(f"{ADDITIONAL_MEMBER}: missing columns {missing}")
    frame = source.rename(columns=rename)[list(rename.values())].copy()
    frame["target_year"] = pd.to_numeric(
        frame["target_year"], errors="raise"
    ).astype(int)
    frame = frame.loc[frame["target_year"].eq(TARGET_YEAR)].copy()
    frame["zone"] = frame["zone"].astype(str)
    frame["nameplate_capacity_mw"] = pd.to_numeric(
        frame["nameplate_capacity_mw"], errors="raise"
    )
    frame["weighted_average_efficiency_fraction"] = (
        pd.to_numeric(
            frame["weighted_average_efficiency_percent"], errors="raise"
        )
        / 100.0
    )
    frame["activation_price_eur_mwh"] = pd.to_numeric(
        frame["activation_price_eur_mwh"], errors="raise"
    )
    if (
        (frame["nameplate_capacity_mw"] < 0).any()
        or not frame["weighted_average_efficiency_fraction"]
        .between(0, 1)
        .all()
    ):
        raise ValueError(f"{ADDITIONAL_MEMBER}: invalid 2035 parameters")
    type_map = {
        "Electrolyser": "electrolyser_optional_buyer",
        "Power to heat": "power_to_heat",
    }
    unexpected = sorted(
        set(frame["source_resource_type"].astype(str)).difference(type_map)
    )
    if unexpected:
        raise ValueError(
            f"{ADDITIONAL_MEMBER}: unexpected resource types {unexpected}"
        )
    frame["resource_type"] = frame["source_resource_type"].map(type_map)
    frame["availability_lower_bound_mw"] = 0.0
    frame["availability_upper_bound_mw"] = frame["nameplate_capacity_mw"]
    electrolyser = frame["source_resource_type"].eq("Electrolyser")
    p2h = frame["source_resource_type"].eq("Power to heat")
    nl0a = frame["zone"].eq("NL0A_OFF")
    frame["hourly_availability_profile_status"] = np.where(
        electrolyser,
        "not_required_optional_buyer_0_to_nameplate",
        "missing_hourly_heat_availability_zero_nameplate_bounds_only",
    )
    frame["central_case_admitted"] = electrolyser & ~nl0a
    frame["central_case_treatment"] = np.select(
        [nl0a, p2h],
        [
            "paired_exclusion_with_isolated_50MW_NL0A_OFF_wind",
            "not_admitted_without_hourly_heat_availability",
        ],
        default="optional_buyer_dispatch_between_zero_and_nameplate",
    )
    frame["nl0a_paired_isolation"] = nl0a

    capacity = pd.read_csv(phase1 / "capacity_2035_final.csv")
    nl0a_capacity = capacity.loc[
        capacity["Market_Node"].astype(str).eq("NL0A_OFF")
    ]
    wind = float(
        nl0a_capacity.loc[
            nl0a_capacity["Technology"].astype(str).eq(
                "Wind offshore fixed"
            ),
            "Value",
        ].sum()
    )
    electrolyser_capacity = float(
        nl0a_capacity.loc[
            nl0a_capacity["Technology"].astype(str).eq("Electrolyser"),
            "Value",
        ].sum()
    )
    released_nl0a = frame.loc[
        nl0a & frame["source_resource_type"].eq("Electrolyser"),
        "nameplate_capacity_mw",
    ]
    if (
        len(released_nl0a) != 1
        or not np.isclose(float(released_nl0a.iloc[0]), 50.0)
        or not np.isclose(wind, 50.0)
        or not np.isclose(electrolyser_capacity, 50.0)
    ):
        raise ValueError("NL0A_OFF 50 MW wind/electrolyser pair is inconsistent")

    result = frame.loc[:, ADDITIONAL_COLUMNS].sort_values(
        ["zone", "resource_type"]
    ).reset_index(drop=True)
    if result.duplicated(["zone", "resource_type"]).any():
        raise ValueError("duplicate 2035 additional dispatchable resource")
    qa = {
        "rows": len(result),
        "electrolyser_rows": int(
            result["resource_type"].eq("electrolyser_optional_buyer").sum()
        ),
        "electrolyser_nameplate_mw": float(
            result.loc[
                result["resource_type"].eq("electrolyser_optional_buyer"),
                "nameplate_capacity_mw",
            ].sum()
        ),
        "power_to_heat_rows": int(
            result["resource_type"].eq("power_to_heat").sum()
        ),
        "power_to_heat_nameplate_mw": float(
            result.loc[
                result["resource_type"].eq("power_to_heat"),
                "nameplate_capacity_mw",
            ].sum()
        ),
        "power_to_heat_central_case_rows": int(
            result.loc[
                result["resource_type"].eq("power_to_heat"),
                "central_case_admitted",
            ].astype(bool).sum()
        ),
        "NL0A_OFF_wind_capacity_mw": wind,
        "NL0A_OFF_electrolyser_capacity_mw": electrolyser_capacity,
        "NL0A_OFF_paired_central_exclusion": bool(
            result.loc[result["zone"].eq("NL0A_OFF"), "nl0a_paired_isolation"]
            .astype(bool)
            .all()
        ),
        "all_zero_nameplate_bounds_exact": bool(
            result["availability_lower_bound_mw"].eq(0).all()
            and np.allclose(
                result["availability_upper_bound_mw"],
                result["nameplate_capacity_mw"],
                rtol=0,
                atol=0,
            )
        ),
    }
    return result, qa


def phase1_hourly(
    phase1: Path,
    zone: str,
) -> pd.DataFrame:
    path = (
        phase1
        / "hourly_profiles"
        / f"zone={zone}"
        / "hourly_profiles.csv.gz"
    )
    required = [
        "weather_scenario",
        "zone",
        "hour_of_year",
        "gross_demand_mw",
        "load_mw",
        "btm_surplus_potential_mw",
        "btm_battery_net_output_mw",
    ]
    columns = set(pd.read_csv(path, nrows=0).columns)
    missing = sorted(set(required).difference(columns))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    frame = pd.read_csv(path, usecols=required)
    expected_rows = HOURS_PER_YEAR * len(WEATHER_SCENARIOS)
    if len(frame) != expected_rows:
        raise ValueError(
            f"{path}: expected {expected_rows} rows, observed {len(frame)}"
        )
    expected_weather = np.tile(
        np.asarray(WEATHER_SCENARIOS, dtype=object), HOURS_PER_YEAR
    )
    expected_hours = np.repeat(
        np.arange(HOURS_PER_YEAR, dtype=np.int16),
        len(WEATHER_SCENARIOS),
    )
    if (
        not frame["zone"].astype(str).eq(zone).all()
        or not np.array_equal(
            frame["weather_scenario"].astype(str).to_numpy(),
            expected_weather,
        )
        or not np.array_equal(
            pd.to_numeric(
                frame["hour_of_year"], errors="raise"
            ).to_numpy(dtype=int),
            expected_hours,
        )
    ):
        raise ValueError(f"{path}: phase-1 key grid is not canonical")
    numeric = frame[
        [
            "gross_demand_mw",
            "load_mw",
            "btm_surplus_potential_mw",
            "btm_battery_net_output_mw",
        ]
    ].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{path}: non-finite phase-1 values")
    if (
        numeric[
            ["gross_demand_mw", "load_mw", "btm_surplus_potential_mw"]
        ]
        < -1e-12
    ).any().any():
        raise ValueError(f"{path}: negative phase-1 demand")
    frame.loc[
        :,
        [
            "gross_demand_mw",
            "load_mw",
            "btm_surplus_potential_mw",
            "btm_battery_net_output_mw",
        ],
    ] = numeric
    return frame


def write_hourly(
    archive: zipfile.ZipFile,
    phase1: Path,
    zones: list[str],
    ratios: pd.DataFrame,
    path: Path,
) -> dict[str, Any]:
    temporary = temporary_path(path)
    ratio_by_zone = ratios.set_index("zone")
    rows_per_zone = HOURS_PER_YEAR * len(WEATHER_SCENARIOS)
    total_rows = 0
    first = True
    totals = {
        "ev_price_sensitive_mwh_over_36_ws": 0.0,
        "hp_price_sensitive_mwh_over_36_ws": 0.0,
        "btm_price_sensitive_reference_net_output_mwh_over_36_ws": 0.0,
        "phase1_load_mwh_over_36_ws": 0.0,
        "phase1_gross_demand_mwh_over_36_ws": 0.0,
    }
    maximum_flexible_net_load_excess = -math.inf
    maximum_flexible_gross_load_excess = -math.inf
    flexible_net_load_conflict_hours = 0
    maximum_flexible_share = 0.0
    minimum_inflexible_signed_net_load = math.inf
    maximum_reference_reconstruction_error = 0.0
    maximum_annual_multiplication_error = 0.0
    hp_negative_source_value_count = 0
    hp_negative_source_sum_mwh = 0.0
    member_rows: list[dict[str, Any]] = []
    zone_rows: list[dict[str, Any]] = []

    with gzip.GzipFile(
        filename=str(temporary),
        mode="wb",
        compresslevel=6,
        mtime=0,
    ) as binary:
        with io.TextIOWrapper(binary, encoding="utf-8", newline="") as text:
            for zone in zones:
                ratio = ratio_by_zone.loc[zone]
                ev_ratio = float(ratio["ev_ratio"])
                hp_ratio = float(ratio["hp_ratio"])
                btm_ratio = float(ratio["btm_ratio"])
                ev_member = EV_MEMBER.format(zone=zone)
                hp_member = HP_MEMBER.format(zone=zone)
                ev = read_component_matrix(archive, ev_member)
                hp_signed = read_component_matrix(
                    archive, hp_member, allow_signed=True
                )
                # DFT can report a negative HP component where heat pumps
                # replace less-efficient electric heating.  This is an
                # inflexible load correction, not electricity consumption
                # that can be shifted inside an iDSR window.  Annex 2's
                # non-negative demand-shifting stock therefore applies to the
                # positive consumption part only.
                hp = np.maximum(hp_signed, 0.0)
                hp_negative_source_value_count += int((hp_signed < 0).sum())
                hp_negative_source_sum_mwh += float(
                    np.minimum(hp_signed, 0.0).sum()
                )
                phase = phase1_hourly(phase1, zone)
                ev_flexible = ev * ev_ratio
                hp_flexible = hp * hp_ratio
                flexible = (ev_flexible + hp_flexible).reshape(-1)
                load = phase["load_mw"].to_numpy(dtype=float)
                gross = phase["gross_demand_mw"].to_numpy(dtype=float)
                phase1_surplus = phase[
                    "btm_surplus_potential_mw"
                ].to_numpy(dtype=float)
                btm_reference = phase[
                    "btm_battery_net_output_mw"
                ].to_numpy(dtype=float)
                btm_flexible = btm_reference * btm_ratio
                net_excess = flexible - load
                gross_excess = flexible - gross
                maximum_flexible_net_load_excess = max(
                    maximum_flexible_net_load_excess,
                    float(net_excess.max()),
                )
                maximum_flexible_gross_load_excess = max(
                    maximum_flexible_gross_load_excess,
                    float(gross_excess.max()),
                )
                flexible_net_load_conflict_hours += int(
                    (net_excess > 1e-8).sum()
                )
                positive_gross = gross > 1e-12
                if positive_gross.any():
                    maximum_flexible_share = max(
                        maximum_flexible_share,
                        float(
                            (
                                flexible[positive_gross]
                                / gross[positive_gross]
                            ).max()
                        ),
                    )
                if (gross_excess > 1e-8).any():
                    position = int(np.argmax(gross_excess))
                    raise ValueError(
                        f"{zone}: price-sensitive EV+HP exceeds phase-1 "
                        f"gross demand "
                        f"at hour={position // 36}, "
                        f"weather={WEATHER_SCENARIOS[position % 36]}, "
                        f"excess_mw={gross_excess[position]}"
                    )
                # Phase 1 represents a signed BTM-adjusted balance N as
                # load-positive minus surplus-positive.  Removing the EV/HP
                # baseline F and reversing the flexible BTM reference R gives
                # the fixed signed balance N0=N-F+R.  It may legitimately be
                # negative in high-BTM-PV hours and must be split into fixed
                # load/surplus downstream rather than clipped.
                phase1_signed_net_load = load - phase1_surplus
                inflexible_signed_net_load = (
                    phase1_signed_net_load - flexible + btm_flexible
                )
                minimum_inflexible_signed_net_load = min(
                    minimum_inflexible_signed_net_load,
                    float(inflexible_signed_net_load.min()),
                )
                reconstruction = (
                    inflexible_signed_net_load
                    + flexible
                    - btm_flexible
                    - phase1_signed_net_load
                )
                maximum_reference_reconstruction_error = max(
                    maximum_reference_reconstruction_error,
                    float(np.abs(reconstruction).max()),
                )
                for raw, flexible_matrix, scalar in (
                    (ev, ev_flexible, ev_ratio),
                    (hp, hp_flexible, hp_ratio),
                ):
                    annual_error = np.abs(
                        flexible_matrix.sum(axis=0)
                        - raw.sum(axis=0) * scalar
                    )
                    maximum_annual_multiplication_error = max(
                        maximum_annual_multiplication_error,
                        float(annual_error.max()),
                    )

                output = pd.DataFrame(
                    {
                        "weather_scenario": np.tile(
                            np.asarray(WEATHER_SCENARIOS, dtype=object),
                            HOURS_PER_YEAR,
                        ),
                        "zone": zone,
                        "hour_of_year": np.repeat(
                            np.arange(HOURS_PER_YEAR, dtype=np.int16),
                            len(WEATHER_SCENARIOS),
                        ),
                        "ev_price_sensitive_baseline_mw": (
                            ev_flexible.reshape(-1)
                        ),
                        "hp_price_sensitive_baseline_mw": (
                            hp_flexible.reshape(-1)
                        ),
                        "btm_price_sensitive_reference_net_output_mw": (
                            btm_flexible
                        ),
                    },
                    columns=HOURLY_COLUMNS,
                )
                if len(output) != rows_per_zone:
                    raise AssertionError("hourly output chunk has wrong size")
                output.to_csv(
                    text,
                    index=False,
                    header=first,
                    float_format="%.10f",
                    lineterminator="\n",
                )
                first = False
                total_rows += len(output)
                zone_ev = float(ev_flexible.sum())
                zone_hp = float(hp_flexible.sum())
                zone_btm = float(btm_flexible.sum())
                totals["ev_price_sensitive_mwh_over_36_ws"] += zone_ev
                totals["hp_price_sensitive_mwh_over_36_ws"] += zone_hp
                totals[
                    "btm_price_sensitive_reference_net_output_mwh_over_36_ws"
                ] += zone_btm
                totals["phase1_load_mwh_over_36_ws"] += float(load.sum())
                totals["phase1_gross_demand_mwh_over_36_ws"] += float(
                    gross.sum()
                )
                zone_rows.append(
                    {
                        "zone": zone,
                        "ev_ratio": ev_ratio,
                        "hp_ratio": hp_ratio,
                        "btm_ratio": btm_ratio,
                        "ev_price_sensitive_mwh_over_36_ws": zone_ev,
                        "hp_price_sensitive_mwh_over_36_ws": zone_hp,
                        (
                            "btm_price_sensitive_reference_net_output_mwh_"
                            "over_36_ws"
                        ): zone_btm,
                        "maximum_ev_hp_minus_phase1_load_mw": float(
                            net_excess.max()
                        ),
                        "maximum_ev_hp_minus_phase1_gross_demand_mw": float(
                            gross_excess.max()
                        ),
                    }
                )
                for member in (ev_member, hp_member):
                    info = archive.getinfo(member)
                    member_rows.append(
                        {
                            "member": member,
                            "uncompressed_bytes": int(info.file_size),
                            "crc32": f"{info.CRC:08x}",
                        }
                    )
    temporary.replace(path)
    expected_rows = len(zones) * rows_per_zone
    if total_rows != expected_rows:
        raise ValueError(
            f"hourly output rows {total_rows} != expected {expected_rows}"
        )
    return {
        "rows": total_rows,
        "columns": HOURLY_COLUMNS,
        "zones": len(zones),
        "weather_scenarios": len(WEATHER_SCENARIOS),
        "hours_per_weather_scenario": HOURS_PER_YEAR,
        **totals,
        "maximum_ev_hp_flexible_minus_phase1_net_load_mw": (
            maximum_flexible_net_load_excess
        ),
        "maximum_ev_hp_flexible_minus_phase1_gross_demand_mw": (
            maximum_flexible_gross_load_excess
        ),
        "ev_hp_exceeds_nonnegative_phase1_net_load_hour_count": (
            flexible_net_load_conflict_hours
        ),
        "maximum_ev_hp_flexible_share_of_phase1_gross_demand": (
            maximum_flexible_share
        ),
        "minimum_inflexible_signed_net_load_mw": (
            minimum_inflexible_signed_net_load
        ),
        "signed_balance_reference_reconstruction_max_abs_error_mw": (
            maximum_reference_reconstruction_error
        ),
        "annual_ratio_multiplication_max_abs_error_mwh": (
            maximum_annual_multiplication_error
        ),
        "hp_negative_source_value_count_retained_as_inflexible_correction": (
            hp_negative_source_value_count
        ),
        "hp_negative_source_sum_mwh_retained_as_inflexible_correction": (
            hp_negative_source_sum_mwh
        ),
        "source_component_members": member_rows,
        "zone_energy_audit": zone_rows,
    }


def reread_hourly_output(
    path: Path,
    zones: list[str],
    write_qa: dict[str, Any],
) -> dict[str, Any]:
    rows_per_zone = HOURS_PER_YEAR * len(WEATHER_SCENARIOS)
    total_rows = 0
    totals = np.zeros(3, dtype=float)
    expected_columns = HOURLY_COLUMNS
    for chunk in pd.read_csv(path, chunksize=750_000):
        if list(chunk.columns) != expected_columns:
            raise ValueError(
                f"{path}: output columns changed during serialization"
            )
        start = total_rows
        stop = total_rows + len(chunk)
        positions = np.arange(start, stop, dtype=np.int64)
        zone_index = positions // rows_per_zone
        within_zone = positions % rows_per_zone
        expected_zone = np.asarray(zones, dtype=object)[zone_index]
        expected_hour = within_zone // len(WEATHER_SCENARIOS)
        expected_weather = np.asarray(
            WEATHER_SCENARIOS, dtype=object
        )[within_zone % len(WEATHER_SCENARIOS)]
        if (
            not np.array_equal(chunk["zone"].astype(str), expected_zone)
            or not np.array_equal(
                chunk["weather_scenario"].astype(str), expected_weather
            )
            or not np.array_equal(
                pd.to_numeric(
                    chunk["hour_of_year"], errors="raise"
                ).to_numpy(dtype=int),
                expected_hour,
            )
        ):
            raise ValueError(f"{path}: serialized key grid is not canonical")
        values = chunk[HOURLY_COLUMNS[3:]].apply(
            pd.to_numeric, errors="raise"
        ).to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{path}: non-finite serialized hourly values")
        if (values[:, :2] < -1e-12).any():
            raise ValueError(f"{path}: negative serialized EV/HP baseline")
        totals += values.sum(axis=0)
        total_rows = stop
    expected_rows = len(zones) * rows_per_zone
    if total_rows != expected_rows:
        raise ValueError(
            f"{path}: reread rows {total_rows} != expected {expected_rows}"
        )
    expected_totals = np.asarray(
        [
            write_qa["ev_price_sensitive_mwh_over_36_ws"],
            write_qa["hp_price_sensitive_mwh_over_36_ws"],
            write_qa[
                "btm_price_sensitive_reference_net_output_mwh_over_36_ws"
            ],
        ],
        dtype=float,
    )
    error = np.abs(totals - expected_totals)
    tolerance = np.maximum(1e-5, np.abs(expected_totals) * 1e-10)
    if (error > tolerance).any():
        raise ValueError(
            f"{path}: serialized energy totals differ from in-memory totals"
        )
    return {
        "pass": True,
        "rows": total_rows,
        "columns": list(expected_columns),
        "key_grid_exact": True,
        "serialized_totals_mwh": {
            column: float(value)
            for column, value in zip(HOURLY_COLUMNS[3:], totals)
        },
        "maximum_serialization_energy_error_mwh": float(error.max()),
    }


def execute(args: argparse.Namespace) -> dict[str, Any]:
    archive_path = args.archive.resolve()
    archive_qa_path = args.archive_qa.resolve()
    phase1 = args.phase1.resolve()
    battery_path = args.battery_data.resolve()
    output = args.output.resolve()
    prepare_output(output, args.force)
    archive_audit = validate_archive(archive_path, archive_qa_path)
    phase1_audit = validate_phase1(phase1)
    zones = demand_zones(phase1)

    with zipfile.ZipFile(archive_path) as archive:
        ratios = read_ratios(archive, zones)
        windows = build_windows(zones)
        btm, btm_qa = build_btm_parameters(
            zones, ratios, battery_path, phase1
        )
        additional, additional_qa = build_additional_dispatchable(
            archive, phase1
        )
        hourly_path = output / "idsr_hourly_2035.csv.gz"
        hourly_qa = write_hourly(
            archive, phase1, zones, ratios, hourly_path
        )

    atomic_csv(windows, output / "idsr_windows_2035.csv")
    atomic_csv(btm, output / "idsr_btm_battery_2035.csv")
    atomic_csv(
        additional,
        output / "additional_dispatchable_consumption_2035.csv",
    )
    reread_qa = (
        {"pass": None, "status": "skipped_by_explicit_cli_option"}
        if args.skip_output_reread
        else reread_hourly_output(hourly_path, zones, hourly_qa)
    )

    output_files = {}
    for name in OUTPUT_FILENAMES[:-1]:
        path = output / name
        output_files[name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    gates = {
        "verified_demand_archive": archive_audit["archive_qa_pass"],
        "phase1_release_eligible": phase1_audit["phase1_release_eligible"],
        "52_demand_nodes": len(zones) == 52,
        "36_weather_scenarios": len(WEATHER_SCENARIOS) == 36,
        "8760_hours_per_weather_scenario": (
            hourly_qa["hours_per_weather_scenario"] == HOURS_PER_YEAR
        ),
        "hourly_row_identity": (
            hourly_qa["rows"] == 52 * 36 * HOURS_PER_YEAR
        ),
        "fixed_hourly_columns_exact": (
            hourly_qa["columns"] == HOURLY_COLUMNS
        ),
        "ev_hp_never_exceeds_phase1_gross_demand": (
            hourly_qa[
                "maximum_ev_hp_flexible_minus_phase1_gross_demand_mw"
            ]
            <= 1e-8
        ),
        "signed_balance_reference_reconstruction_exact": (
            hourly_qa[
                "signed_balance_reference_reconstruction_max_abs_error_mw"
            ]
            <= 1e-8
        ),
        "annual_ratio_multiplication_exact": (
            hourly_qa["annual_ratio_multiplication_max_abs_error_mwh"]
            <= 1e-5
        ),
        "btm_capacity_multiplication_exact": (
            btm_qa["capacity_ratio_multiplication_max_abs_error"] <= 1e-10
        ),
        "annex2_windows_complete": (
            len(windows) == 104
            and windows.groupby("zone")["resource_type"].nunique().eq(2).all()
        ),
        "p2h_not_admitted_without_hourly_heat_availability": (
            additional_qa["power_to_heat_central_case_rows"] == 0
        ),
        "additional_consumption_bounds_are_zero_nameplate": (
            additional_qa["all_zero_nameplate_bounds_exact"]
        ),
        "NL0A_wind_electrolyser_paired_isolation": (
            additional_qa["NL0A_OFF_paired_central_exclusion"]
        ),
        "serialized_output_reread": bool(
            reread_qa["pass"] is True
        ),
    }
    passed = all(gates.values())
    qa = {
        "script": Path(__file__).name,
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now(),
        "target_year": TARGET_YEAR,
        "evidence_status": (
            "audited_standardized_model_input"
            if passed
            else "input_QA_failed"
        ),
        "pass": passed,
        "source_contract": {
            "demand_archive": str(archive_path),
            "archive_qa": str(archive_qa_path),
            "phase1_input": str(phase1),
            "dashboard_battery_data": str(battery_path),
            **archive_audit,
            **phase1_audit,
            "dashboard_battery_sha256": sha256_file(battery_path),
            "ratio_member": RATIO_MEMBER,
            "additional_dispatchable_member": ADDITIONAL_MEMBER,
        },
        "interpretation": {
            "idsr_hourly": (
                "released EV/HP component multiplied by each TSO's 2035 "
                "price-sensitive ratio; for the HP component only positive "
                "electricity consumption is shiftable, while negative DFT "
                "replacement corrections remain inflexible; BTM field is "
                "the price-sensitive share of the signed reference net "
                "output already embedded in phase-1 load"
            ),
            "btm_reverse_peel": (
                "with phase1 signed balance N=load_mw-"
                "btm_surplus_potential_mw, downstream fixed signed balance "
                "is N minus EV/HP flexible baseline plus BTM "
                "price-sensitive reference net output; split its positive "
                "and negative parts into fixed load and fixed surplus"
            ),
            "net_load_bound": (
                "EV/HP is an end-use component and is bounded against phase1 "
                "gross_demand_mw. It can exceed the non-negative net "
                "load_mw where BTM PV creates a simultaneous surplus; "
                "clipping it to load_mw would violate released TSO ratios "
                "and annual energy."
            ),
            "electrolyser": (
                "optional buyer with hourly consumption decision in "
                "[0,nameplate]"
            ),
            "power_to_heat": (
                "hourly heat availability is not released; zero/nameplate "
                "bounds are published but neither bound is a central case"
            ),
            "time_zone": (
                "Annex 2 Table 8 standard-time categories; no daylight-saving "
                "shift is introduced"
            ),
            "NL0A_OFF": (
                "central topology excludes the released 50 MW isolated "
                "offshore wind and 50 MW electrolyser together"
            ),
        },
        "hourly": hourly_qa,
        "hourly_reread": reread_qa,
        "windows": {
            "rows": len(windows),
            "columns": WINDOW_COLUMNS,
            "zone_offsets": {
                str(offset): sorted(
                    windows.loc[
                        windows["time_zone_utc_offset"].eq(offset), "zone"
                    ].unique().tolist()
                )
                for offset in sorted(
                    windows["time_zone_utc_offset"].unique()
                )
            },
        },
        "btm_battery": {
            "columns": BTM_COLUMNS,
            **btm_qa,
        },
        "additional_dispatchable_consumption": {
            "columns": ADDITIONAL_COLUMNS,
            **additional_qa,
        },
        "output_files": output_files,
        "gates": gates,
    }
    atomic_json(qa, output / "QA.json")
    if not passed:
        failed = sorted(name for name, value in gates.items() if not value)
        raise RuntimeError(f"output QA failed gates {failed}")
    return qa


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    result.add_argument("--archive-qa", type=Path, default=DEFAULT_ARCHIVE_QA)
    result.add_argument("--phase1", type=Path, default=DEFAULT_PHASE1)
    result.add_argument(
        "--battery-data",
        type=Path,
        default=DEFAULT_BATTERY_DATA,
    )
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--force", action="store_true")
    result.add_argument(
        "--skip-output-reread",
        action="store_true",
        help="skip the serialized 16.4M-row grid reread (development only)",
    )
    return result


def main() -> int:
    args = parser().parse_args()
    qa = execute(args)
    summary = {
        "pass": qa["pass"],
        "output": str(args.output.resolve()),
        "hourly_rows": qa["hourly"]["rows"],
        "hourly_columns": qa["hourly"]["columns"],
        "ev_price_sensitive_mwh_over_36_ws": qa["hourly"][
            "ev_price_sensitive_mwh_over_36_ws"
        ],
        "hp_price_sensitive_mwh_over_36_ws": qa["hourly"][
            "hp_price_sensitive_mwh_over_36_ws"
        ],
        "active_btm_zones": qa["btm_battery"][
            "active_price_sensitive_btm_zones"
        ],
        "additional_dispatchable_nameplate_mw": (
            qa["additional_dispatchable_consumption"][
                "electrolyser_nameplate_mw"
            ]
            + qa["additional_dispatchable_consumption"][
                "power_to_heat_nameplate_mw"
            ]
        ),
    }
    print(json.dumps(json_ready(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
