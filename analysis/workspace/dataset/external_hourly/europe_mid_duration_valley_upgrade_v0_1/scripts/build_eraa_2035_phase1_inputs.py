#!/usr/bin/env python3
"""Build auditable ERAA 2025-final 2035 phase-1 input shards.

The builder joins official ERAA demand, PECD renewable capacity factors and
ERAA-final installed capacities without replicating technology configurations
into the hourly table.  Hydrological inputs retain the released weekly
resolution.  Outputs are deterministic gzip CSV files because the project
environment does not require a Parquet engine.

This script prepares model inputs only.  It does not estimate a mid-duration
valley or produce manuscript results.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


SCRIPT_VERSION = "0.2.0"
TARGET_YEAR = 2035
EXPECTED_HOURS = 8_760
WEATHER_SCENARIOS = tuple(f"WS{index:02d}" for index in range(1, 37))
EXPECTED_SOURCE_COUNTS = {
    "demand": 52,
    "btm_pv": 4,
    "btm_battery": 4,
    "capacity_factor": 309,
    "hydro_constraint": 41,
    "hydro_inflow": 105,
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = (
    PROJECT_ROOT / "raw" / "eraa_2025" / "selective_2035"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "processed" / "eraa_2035_phase1_inputs_v0_1"
)
DEFAULT_CAPACITY_FILE = (
    PROJECT_ROOT
    / "raw"
    / "eraa_2025"
    / "extracted"
    / "ERAA_2025_DashboardRawData1"
    / "Dashboard Raw Data"
    / "GenerationCapacities.csv"
)
DEFAULT_HYDRO_CAPACITY_FILE = (
    PROJECT_ROOT
    / "raw"
    / "eraa_2025"
    / "extracted"
    / "ERAA_2025_DashboardRawData1"
    / "Dashboard Raw Data"
    / "Hydro additional information.csv"
)
DEFAULT_AGGREGATED_DEMAND_FILE = (
    PROJECT_ROOT
    / "raw"
    / "eraa_2025"
    / "extracted"
    / "ERAA_2025_DashboardRawData1"
    / "Dashboard Raw Data"
    / "Aggregated_Demand.csv"
)

EXPECTED_DEMAND_ZONES = frozenset(
    """
    AL00 AT00 BA00 BE00 BG00 CH00 CY00 CZ00 DE00 DKE1 DKW1 EE00 ES00
    FI00 FR00 GR00 GR03 HR00 HU00 IE00 ITCA ITCN ITCS ITN1 ITS1 ITSA
    ITSI LT00 LUG1 LV00 MD00 ME00 MK00 MT00 NL00 NOM1 NON1 NOS1 NOS2
    NOS3 PL00 PT00 RO00 RS00 SE01 SE02 SE03 SE04 SI00 SK00 UK00 UKNI
    """.split()
)
EXPECTED_BTM_ZONES = frozenset({"AT00", "CH00", "DE00", "HU00"})
EXCLUDED_AGGREGATE_ZONES = frozenset(
    {"DK00", "ISEM", "IT00", "LU00", "NO00", "SE00"}
)

DEMAND_RE = re.compile(
    r"^(?P<zone>.+?)_Demand_total_2035_National Trends\.csv$"
)
BTM_PV_RE = re.compile(
    r"^(?P<zone>.+?)_PV_2035_National Trends\.csv$"
)
BTM_BATTERY_RE = re.compile(
    r"^(?P<zone>.+?)_Batteries_2035_National Trends\.csv$"
)
CF_RE = re.compile(
    r"^(?P<zone>.+?)_CapacityFactors_(?P<profile>.+)_2035\.csv$"
)
HYDRO_CONSTRAINT_RE = re.compile(
    r"^(?P<zone>.+?)_Hydro_Constraints_"
    r"(?P<code>HCL|HOL|HPI|HRI|HRR)_2035\.csv$"
)
HYDRO_INFLOW_RE = re.compile(
    r"^(?P<zone>.+?)_Hydro_Inflows_"
    r"(?P<code>HCL|HOL|HPI|HRI|HRR)_2035\.csv$"
)

HYDRO_TYPES = {
    "HCL": "closed-loop pumping",
    "HOL": "open-loop pumping",
    "HPI": "pondage",
    "HRI": "reservoir",
    "HRR": "run of river",
}

# One installed-capacity technology can map to a PECD profile.  Multiple
# technologies sharing a profile (for example fixed and floating offshore
# wind) are summed before the profile is applied.
VRE_TECHNOLOGY_MAP = {
    "Wind onshore": ("Wind_Onshore", "wind_onshore"),
    "Wind offshore fixed": ("Wind_Offshore", "wind_offshore"),
    "Wind offshore floating": ("Wind_Offshore", "wind_offshore"),
    "Solar PV rooftop industrial": (
        "PV_industrial_rooftop",
        "solar_pv",
    ),
    "Solar PV rooftop residential": (
        "PV_residential_rooftop",
        "solar_pv",
    ),
    "Solar PV utility non-tracking": (
        "PV_utility_fixed",
        "solar_pv",
    ),
    "Solar PV utility tracking": (
        "PV_utility_tracking",
        "solar_pv",
    ),
    "Solar thermal without storage": (
        "CSP_noStorage_0h_dispatched",
        "csp_no_storage",
    ),
    "Solar thermal with storage": (
        "CSP_withStorage_7h_preDispatch",
        "csp_storage_input",
    ),
}

INCLUDED_STATUS_NORMALIZED = frozenset(
    {
        "available on market",
        "inelastic supply / fixed profile",
        "out of market - for pv/battery dispatch optimization",
    }
)

HOURLY_OUTPUT_COLUMNS = [
    "target_year",
    "climate_model",
    "climate_model_code",
    "climate_projection_year",
    "weather_scenario",
    "zone",
    "month",
    "day",
    "hour",
    "hour_of_year",
    "gross_demand_mw",
    "btm_pv_mw",
    "btm_battery_net_output_mw",
    "btm_net_output_mw",
    "net_load_after_btm_mw",
    "load_mw",
    "btm_surplus_potential_mw",
    "wind_onshore_potential_mw",
    "wind_offshore_potential_mw",
    "wind_potential_mw",
    "solar_pv_potential_mw",
    "csp_no_storage_potential_mw",
    "solar_potential_mw",
    "csp_with_storage_predispatch_mw",
    "csp_with_storage_dispatched_reference_mw",
    "demand_role",
    "load_definition",
]


@dataclass(frozen=True)
class WideHourlyProfile:
    """One 8,760-hour by 36-weather-scenario released profile."""

    calendar: pd.DataFrame
    values: np.ndarray
    source_path: Path
    weather_column_order: tuple[str, ...]

    @property
    def canonical_weather_column_order(self) -> bool:
        return self.weather_column_order == WEATHER_SCENARIOS


@dataclass(frozen=True)
class SourceInventory:
    demand: dict[str, Path]
    btm_pv: dict[str, Path]
    btm_battery: dict[str, Path]
    capacity_factor: dict[tuple[str, str], Path]
    hydro_constraint: dict[tuple[str, str], Path]
    hydro_inflow: dict[tuple[str, str], Path]
    duplicate_members: tuple[str, ...]

    @property
    def counts(self) -> dict[str, int]:
        return {
            "demand": len(self.demand),
            "btm_pv": len(self.btm_pv),
            "btm_battery": len(self.btm_battery),
            "capacity_factor": len(self.capacity_factor),
            "hydro_constraint": len(self.hydro_constraint),
            "hydro_inflow": len(self.hydro_inflow),
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_weather_scenario(value: object) -> str:
    match = re.fullmatch(r"\s*WS0*(\d{1,2})\s*", str(value), flags=re.I)
    if match is None:
        raise ValueError(f"invalid weather-scenario label: {value!r}")
    number = int(match.group(1))
    if not 1 <= number <= 36:
        raise ValueError(f"weather-scenario number outside 1..36: {value!r}")
    return f"WS{number:02d}"


def normalize_status(value: object) -> str:
    text = str(value).strip().lower()
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    return re.sub(r"\s+", " ", text)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        json_ready(payload),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.write("\n")
    os.replace(temporary, path)


def write_csv_atomic(
    frame: pd.DataFrame,
    path: Path,
    *,
    gzip_output: bool = False,
    float_format: str = "%.10g",
) -> None:
    """Write stable CSV bytes, including a zero-mtime gzip header."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as raw_handle:
        temporary = Path(raw_handle.name)
        if gzip_output:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw_handle,
                compresslevel=9,
                mtime=0,
            ) as compressed:
                with io.TextIOWrapper(
                    compressed, encoding="utf-8", newline=""
                ) as text_handle:
                    frame.to_csv(
                        text_handle,
                        index=False,
                        lineterminator="\n",
                        float_format=float_format,
                    )
        else:
            with io.TextIOWrapper(
                raw_handle, encoding="utf-8", newline=""
            ) as text_handle:
                frame.to_csv(
                    text_handle,
                    index=False,
                    lineterminator="\n",
                    float_format=float_format,
                )
    os.replace(temporary, path)


def detect_header_row(path: Path, required: Iterable[str], limit: int = 40) -> int:
    required_upper = {item.upper() for item in required}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for index, row in enumerate(reader):
            normalized = {str(item).strip().upper() for item in row}
            if required_upper.issubset(normalized):
                return index
            if index + 1 >= limit:
                break
    raise ValueError(
        f"{path}: did not find header containing {sorted(required_upper)} "
        f"within {limit} rows"
    )


def detect_hydro_header_row(path: Path, limit: int = 40) -> int:
    constraint_names = {
        "MAX_TURB_EN",
        "MAX_TURB",
        "MAX_PUMP",
        "MAX_RES",
        "MIN_TURB_EN",
        "MIN_TURB",
        "MIN_PUMP",
        "MIN_RES",
    }
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for index, row in enumerate(reader):
            normalized = {str(item).strip().upper() for item in row}
            has_time_key = bool({"WEEK", "DAY"}.intersection(normalized))
            has_values = bool(
                constraint_names.intersection(normalized)
                or "WS01" in normalized
            )
            if has_time_key and has_values:
                return index
            if index + 1 >= limit:
                break
    raise ValueError(f"{path}: did not find a valid hydro header")


def normalize_weather_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename: dict[object, str] = {}
    for column in frame.columns:
        text = str(column).strip()
        if re.fullmatch(r"WS0*\d{1,2}", text, flags=re.I):
            rename[column] = normalize_weather_scenario(text)
        else:
            rename[column] = text
    result = frame.rename(columns=rename)
    duplicated = result.columns[result.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(f"duplicate columns after normalization: {duplicated}")
    return result


def validate_weather_matrix(
    frame: pd.DataFrame,
    *,
    path: Path,
    minimum: float,
    maximum: float | None,
) -> np.ndarray:
    missing = sorted(set(WEATHER_SCENARIOS).difference(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing weather columns {missing}")
    values = frame.loc[:, WEATHER_SCENARIOS].apply(
        pd.to_numeric, errors="raise"
    )
    matrix = values.to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{path}: weather matrix contains non-finite values")
    if (matrix < minimum - 1e-12).any():
        raise ValueError(f"{path}: values below allowed minimum {minimum}")
    if maximum is not None and (matrix > maximum + 1e-8).any():
        actual = float(np.max(matrix))
        raise ValueError(
            f"{path}: values exceed allowed maximum {maximum}; max={actual}"
        )
    return matrix


def validate_calendar(calendar: pd.DataFrame, path: Path) -> None:
    if calendar.duplicated(["month", "day", "hour"]).any():
        raise ValueError(f"{path}: duplicate month-day-hour keys")
    if not calendar["hour"].between(0, 23).all():
        raise ValueError(f"{path}: hour is not in 0..23")
    if not calendar["month"].between(1, 12).all():
        raise ValueError(f"{path}: month outside 1..12")
    if not calendar["day"].between(1, 31).all():
        raise ValueError(f"{path}: day outside 1..31")


def read_demand_wide(
    path: Path, *, expected_hours: int = EXPECTED_HOURS
) -> WideHourlyProfile:
    frame = normalize_weather_columns(pd.read_csv(path))
    required = {"Date", "Month", "Day", "Hour", *WEATHER_SCENARIOS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing demand columns {missing}")
    if len(frame) != expected_hours:
        raise ValueError(
            f"{path}: expected {expected_hours} rows, observed {len(frame)}"
        )

    timestamps = pd.to_datetime(frame["Date"], errors="raise")
    month = pd.to_numeric(frame["Month"], errors="raise").astype(int)
    day = pd.to_numeric(frame["Day"], errors="raise").astype(int)
    hour = pd.to_numeric(frame["Hour"], errors="raise").astype(int)
    if not np.array_equal(timestamps.dt.month.to_numpy(), month.to_numpy()):
        raise ValueError(f"{path}: Date and Month fields disagree")
    if not np.array_equal(timestamps.dt.day.to_numpy(), day.to_numpy()):
        raise ValueError(f"{path}: Date and Day fields disagree")
    if not np.array_equal(timestamps.dt.hour.to_numpy(), hour.to_numpy()):
        raise ValueError(f"{path}: Date and Hour fields disagree")

    calendar = pd.DataFrame(
        {
            "month": month.to_numpy(),
            "day": day.to_numpy(),
            "hour": hour.to_numpy(),
            "hour_of_year": np.arange(expected_hours, dtype=np.int32),
        }
    )
    validate_calendar(calendar, path)
    source_weather_order = tuple(
        column for column in frame.columns if column in WEATHER_SCENARIOS
    )
    matrix = validate_weather_matrix(
        frame, path=path, minimum=0.0, maximum=None
    )
    return WideHourlyProfile(
        calendar, matrix, path, source_weather_order
    )


def read_btm_wide(
    path: Path,
    *,
    component: str,
    expected_hours: int = EXPECTED_HOURS,
) -> WideHourlyProfile:
    """Read released behind-the-meter additions on the demand calendar.

    PV is a non-negative output profile.  The released battery convention is
    signed: positive values reduce official net demand (discharge) and
    negative values increase it (charge).  The signed battery series must
    therefore never pass through the non-negative demand parser.
    """

    if component not in {"pv", "battery"}:
        raise ValueError(f"invalid BTM component {component!r}")
    frame = normalize_weather_columns(pd.read_csv(path))
    required = {"Date", "Month", "Day", "Hour", *WEATHER_SCENARIOS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing BTM columns {missing}")
    if len(frame) != expected_hours:
        raise ValueError(
            f"{path}: expected {expected_hours} rows, observed {len(frame)}"
        )

    timestamps = pd.to_datetime(frame["Date"], errors="raise")
    month = pd.to_numeric(frame["Month"], errors="raise").astype(int)
    day = pd.to_numeric(frame["Day"], errors="raise").astype(int)
    hour = pd.to_numeric(frame["Hour"], errors="raise").astype(int)
    if not np.array_equal(timestamps.dt.month.to_numpy(), month.to_numpy()):
        raise ValueError(f"{path}: Date and Month fields disagree")
    if not np.array_equal(timestamps.dt.day.to_numpy(), day.to_numpy()):
        raise ValueError(f"{path}: Date and Day fields disagree")
    if not np.array_equal(timestamps.dt.hour.to_numpy(), hour.to_numpy()):
        raise ValueError(f"{path}: Date and Hour fields disagree")
    calendar = pd.DataFrame(
        {
            "month": month.to_numpy(),
            "day": day.to_numpy(),
            "hour": hour.to_numpy(),
            "hour_of_year": np.arange(expected_hours, dtype=np.int32),
        }
    )
    validate_calendar(calendar, path)
    source_weather_order = tuple(
        column for column in frame.columns if column in WEATHER_SCENARIOS
    )
    values = frame.loc[:, WEATHER_SCENARIOS].apply(
        pd.to_numeric, errors="raise"
    )
    matrix = values.to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{path}: BTM weather matrix has non-finite values")
    if component == "pv" and (matrix < -1e-12).any():
        raise ValueError(f"{path}: BTM PV contains negative output")
    if component == "pv":
        matrix = np.maximum(matrix, 0.0)
    return WideHourlyProfile(
        calendar, matrix, path, source_weather_order
    )


def read_capacity_factor_wide(
    path: Path, *, expected_hours: int = EXPECTED_HOURS
) -> WideHourlyProfile:
    header = detect_header_row(path, {"Date", "Hour", "WS01"})
    frame = normalize_weather_columns(pd.read_csv(path, skiprows=header))
    required = {"Date", "Hour", *WEATHER_SCENARIOS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing capacity-factor columns {missing}")
    if len(frame) != expected_hours:
        raise ValueError(
            f"{path}: expected {expected_hours} rows, observed {len(frame)}"
        )

    date_parts = frame["Date"].astype(str).str.extract(
        r"^\s*(?P<day>\d{1,2})\.(?P<month>\d{1,2})\.\s*$"
    )
    if date_parts.isna().any().any():
        examples = frame.loc[date_parts.isna().any(axis=1), "Date"].head()
        raise ValueError(
            f"{path}: invalid DD.MM. date labels {examples.tolist()}"
        )
    hour_released = pd.to_numeric(frame["Hour"], errors="raise").astype(int)
    hour = hour_released - 1
    calendar = pd.DataFrame(
        {
            "month": date_parts["month"].astype(int).to_numpy(),
            "day": date_parts["day"].astype(int).to_numpy(),
            "hour": hour.to_numpy(),
            "hour_of_year": np.arange(expected_hours, dtype=np.int32),
        }
    )
    validate_calendar(calendar, path)
    source_weather_order = tuple(
        column for column in frame.columns if column in WEATHER_SCENARIOS
    )
    # CSP-with-storage pre-dispatch is the solar-field input and can exceed
    # one because the released plant has a solar multiple.  It is not a
    # conventional installed-capacity factor and must never be clipped.
    is_csp_storage_predispatch = (
        "CSP_withStorage_7h_preDispatch" in path.name
    )
    # The official ERAA 2025 release contains 36 WS20 values above one in
    # PT00_CapacityFactors_Wind_Onshore_2035.csv (maximum 1.051521884).
    # Preserve this narrowly identified source anomaly in the main
    # replication instead of silently clipping it.  A unit-capped PT00
    # profile is reserved for a documented sensitivity run.
    is_pt00_wind_release_overshoot = (
        path.name == "PT00_CapacityFactors_Wind_Onshore_2035.csv"
    )
    if is_csp_storage_predispatch:
        maximum = None
    elif is_pt00_wind_release_overshoot:
        maximum = 1.06
    else:
        maximum = 1.0
    matrix = validate_weather_matrix(
        frame,
        path=path,
        minimum=0.0,
        maximum=maximum,
    )
    if not (
        is_csp_storage_predispatch
        or is_pt00_wind_release_overshoot
    ):
        # Tiny floating-point overshoots below the declared tolerance are
        # clipped only for ordinary capacity-factor profiles.
        matrix = np.clip(matrix, 0.0, 1.0)
    return WideHourlyProfile(
        calendar, matrix, path, source_weather_order
    )


def assert_same_calendar(
    reference: pd.DataFrame, candidate: pd.DataFrame, path: Path
) -> None:
    columns = ["month", "day", "hour", "hour_of_year"]
    if len(reference) != len(candidate) or not np.array_equal(
        reference[columns].to_numpy(), candidate[columns].to_numpy()
    ):
        raise ValueError(f"{path}: calendar does not match demand calendar")


HYDRO_CONSTRAINT_COLUMNS = (
    "MAX_TURB_EN",
    "MAX_TURB",
    "MAX_PUMP",
    "MAX_RES",
    "MIN_TURB_EN",
    "MIN_TURB",
    "MIN_PUMP",
    "MIN_RES",
)


def read_hydro_constraints(path: Path) -> pd.DataFrame:
    header = detect_hydro_header_row(path)
    frame = pd.read_csv(path, skiprows=header)
    frame.columns = [str(item).strip().upper() for item in frame.columns]
    time_keys = [key for key in ("WEEK", "DAY") if key in frame]
    if len(time_keys) != 1:
        raise ValueError(f"{path}: expected exactly one WEEK/DAY key")
    time_key = time_keys[0]
    expected_periods = 53 if time_key == "WEEK" else 365
    if len(frame) != expected_periods:
        raise ValueError(
            f"{path}: expected {expected_periods} {time_key.lower()} rows, "
            f"observed {len(frame)}"
        )
    period = pd.to_numeric(frame[time_key], errors="raise").astype(int)
    if set(period) != set(range(1, expected_periods + 1)):
        raise ValueError(
            f"{path}: {time_key.lower()} values are not exactly "
            f"1..{expected_periods}"
        )
    released_constraints = [
        column for column in HYDRO_CONSTRAINT_COLUMNS if column in frame
    ]
    if not released_constraints:
        raise ValueError(f"{path}: no recognized constraint columns")
    result = pd.DataFrame(
        {
            "time_resolution": time_key.lower(),
            "period": period,
        }
    )
    for source in HYDRO_CONSTRAINT_COLUMNS:
        target = f"{source.lower()}_raw"
        result[target] = (
            pd.to_numeric(frame[source], errors="raise")
            if source in frame
            else np.nan
        )
    result["released_constraint_columns"] = "|".join(
        released_constraints
    )
    numeric = result[
        [f"{column.lower()}_raw" for column in HYDRO_CONSTRAINT_COLUMNS]
    ].to_numpy(dtype=float)
    finite_values = numeric[np.isfinite(numeric)]
    if finite_values.size and (finite_values < 0).any():
        raise ValueError(f"{path}: negative released hydro constraint")
    return result


def read_hydro_inflows(path: Path) -> pd.DataFrame:
    header = detect_hydro_header_row(path)
    frame = normalize_weather_columns(pd.read_csv(path, skiprows=header))
    time_keys = [key for key in ("WEEK", "DAY") if key in frame]
    if len(time_keys) != 1:
        raise ValueError(f"{path}: expected exactly one WEEK/DAY key")
    time_key = time_keys[0]
    expected_periods = 53 if time_key == "WEEK" else 365
    if len(frame) != expected_periods:
        raise ValueError(
            f"{path}: expected {expected_periods} {time_key.lower()} rows, "
            f"observed {len(frame)}"
        )
    period = pd.to_numeric(frame[time_key], errors="raise").astype(int)
    if set(period) != set(range(1, expected_periods + 1)):
        raise ValueError(
            f"{path}: {time_key.lower()} values are not exactly "
            f"1..{expected_periods}"
        )
    matrix = validate_weather_matrix(
        frame, path=path, minimum=0.0, maximum=None
    )
    result = pd.DataFrame(
        {
            "time_resolution": time_key.lower(),
            "period": np.repeat(
                period.to_numpy(), len(WEATHER_SCENARIOS)
            ),
            "weather_scenario": np.tile(
                np.asarray(WEATHER_SCENARIOS), len(frame)
            ),
            "inflow_raw": matrix.reshape(-1),
        }
    )
    return result


def discover_sources(source_root: Path) -> SourceInventory:
    groups: dict[str, dict[Any, Path]] = {
        "demand": {},
        "btm_pv": {},
        "btm_battery": {},
        "capacity_factor": {},
        "hydro_constraint": {},
        "hydro_inflow": {},
    }
    duplicates: list[str] = []
    patterns = (
        ("demand", DEMAND_RE),
        ("btm_pv", BTM_PV_RE),
        ("btm_battery", BTM_BATTERY_RE),
        ("capacity_factor", CF_RE),
        ("hydro_constraint", HYDRO_CONSTRAINT_RE),
        ("hydro_inflow", HYDRO_INFLOW_RE),
    )
    for path in sorted(source_root.rglob("*.csv")):
        if path.name.endswith(".part"):
            continue
        for group, pattern in patterns:
            match = pattern.fullmatch(path.name)
            if match is None:
                continue
            if group in {"demand", "btm_pv", "btm_battery"}:
                key: Any = match.group("zone")
            elif group == "capacity_factor":
                key = (match.group("zone"), match.group("profile"))
            else:
                key = (match.group("zone"), match.group("code"))
            if key in groups[group]:
                duplicates.append(
                    f"{group}:{key}:{groups[group][key]}:{path}"
                )
            else:
                groups[group][key] = path
            break
    return SourceInventory(
        demand=groups["demand"],
        btm_pv=groups["btm_pv"],
        btm_battery=groups["btm_battery"],
        capacity_factor=groups["capacity_factor"],
        hydro_constraint=groups["hydro_constraint"],
        hydro_inflow=groups["hydro_inflow"],
        duplicate_members=tuple(duplicates),
    )


def find_weather_mapping(source_root: Path) -> Path:
    matches = sorted(source_root.rglob("WeatherScenarios_Mapping.xlsx"))
    if len(matches) != 1:
        raise ValueError(
            f"expected one WeatherScenarios_Mapping.xlsx, found {len(matches)}"
        )
    return matches[0]


def read_weather_mapping(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path)
    required = {
        "Weather scenario ID",
        "C3S climate projection model",
        "C3S climate projection target year",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing mapping columns {missing}")
    result = pd.DataFrame(
        {
            "weather_scenario": frame["Weather scenario ID"].map(
                normalize_weather_scenario
            ),
            "climate_model": frame[
                "C3S climate projection model"
            ].astype(str),
            "climate_projection_year": pd.to_numeric(
                frame["C3S climate projection target year"],
                errors="raise",
            ).astype(int),
        }
    )
    result["climate_model_code"] = result["climate_model"].str.extract(
        r"\(([^()]+)\)\s*$", expand=False
    )
    if result["climate_model_code"].isna().any():
        raise ValueError(f"{path}: climate model code missing")
    result = result.sort_values("weather_scenario").reset_index(drop=True)
    if tuple(result["weather_scenario"]) != WEATHER_SCENARIOS:
        raise ValueError(f"{path}: weather scenarios are not exactly WS01..WS36")
    if result.duplicated("weather_scenario").any():
        raise ValueError(f"{path}: duplicate weather scenarios")
    return result


def audit_demand_weather_column_order(
    inventory: SourceInventory,
) -> pd.DataFrame:
    """Audit released column positions while freezing name-based ingestion."""

    rows: list[dict[str, Any]] = []
    expected = set(WEATHER_SCENARIOS)
    for zone, path in sorted(inventory.demand.items()):
        columns = normalize_weather_columns(
            pd.read_csv(path, nrows=0)
        ).columns
        observed = tuple(
            column for column in columns if column in expected
        )
        rows.append(
            {
                "zone": zone,
                "source_member": str(path),
                "weather_columns_observed": len(observed),
                "weather_column_order_canonical": (
                    observed == WEATHER_SCENARIOS
                ),
                "observed_weather_column_order": "|".join(observed),
                "missing_weather_columns": "|".join(
                    sorted(expected.difference(observed))
                ),
                "duplicate_weather_columns": (
                    len(observed) != len(set(observed))
                ),
                "ingestion_rule": "select_by_normalized_WS_name",
            }
        )
    return pd.DataFrame(rows).sort_values("zone").reset_index(drop=True)


def read_capacity_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "data_version",
        "Target year",
        "Market_Node",
        "Technology",
        "Technology_Simplified",
        "Operational_Status",
        "Value",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing capacity columns {missing}")
    selected = frame.loc[
        frame["data_version"].eq("ERAA 2025 final")
        & pd.to_numeric(frame["Target year"], errors="coerce").eq(TARGET_YEAR)
    ].copy()
    selected["Value"] = pd.to_numeric(selected["Value"], errors="raise")
    if selected.empty:
        raise ValueError(f"{path}: ERAA 2025 final 2035 rows are empty")
    if not np.isfinite(selected["Value"]).all() or selected["Value"].lt(0).any():
        raise ValueError(f"{path}: non-finite or negative installed capacity")
    selected = selected.sort_values(
        [
            "Market_Node",
            "Technology",
            "Operational_Status",
        ],
        na_position="last",
    ).reset_index(drop=True)
    return selected


def read_hydro_capacity_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    required = {
        "MARKET_NODE",
        "PEMMDB_PLANT_TYPE",
        "MAX PUMPING CAP (MW)",
        "Storage Capacity [TWh]",
        "TARGET_YEAR",
        "data_version",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing hydro-capacity columns {missing}")
    selected = frame.loc[
        frame["data_version"].eq("ERAA 2025 final")
        & pd.to_numeric(frame["TARGET_YEAR"], errors="coerce").eq(TARGET_YEAR)
    ].copy()
    return selected.sort_values(
        ["MARKET_NODE", "PEMMDB_PLANT_TYPE"]
    ).reset_index(drop=True)


def read_dashboard_annual_demand(path: Path) -> pd.DataFrame:
    """Read the dashboard comparison series without treating it as a target."""

    if not path.exists():
        return pd.DataFrame(
            columns=[
                "zone",
                "dashboard_avg_twh",
                "dashboard_min_twh",
                "dashboard_max_twh",
            ]
        )
    frame = pd.read_csv(path)
    required = {
        "data version",
        "Target_Year",
        "Market_Node",
        "Measure",
        "Measure_Type",
        "Value",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            f"{path}: missing Aggregated_Demand columns {missing}"
        )
    selected = frame.loc[
        frame["data version"].eq("ERAA 2025 final")
        & pd.to_numeric(
            frame["Target_Year"], errors="coerce"
        ).eq(TARGET_YEAR)
        & frame["Measure"].eq("Annual Demand (TWh)")
    ].copy()
    selected["Value"] = pd.to_numeric(selected["Value"], errors="raise")
    pivot = selected.pivot(
        index="Market_Node", columns="Measure_Type", values="Value"
    )
    expected_measures = {"AVG", "MIN", "MAX"}
    if not expected_measures.issubset(pivot.columns):
        raise ValueError(
            f"{path}: missing annual demand measures "
            f"{sorted(expected_measures.difference(pivot.columns))}"
        )
    return (
        pivot.loc[:, ["AVG", "MIN", "MAX"]]
        .rename(
            columns={
                "AVG": "dashboard_avg_twh",
                "MIN": "dashboard_min_twh",
                "MAX": "dashboard_max_twh",
            }
        )
        .rename_axis("zone")
        .reset_index()
    )


def build_vre_mapping(
    capacity: pd.DataFrame,
    inventory: SourceInventory,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in capacity.itertuples(index=False):
        technology = str(record.Technology)
        if technology not in VRE_TECHNOLOGY_MAP:
            continue
        profile, component = VRE_TECHNOLOGY_MAP[technology]
        status = str(record.Operational_Status)
        included = normalize_status(status) in INCLUDED_STATUS_NORMALIZED
        zone = str(record.Market_Node)
        source = inventory.capacity_factor.get((zone, profile))
        rows.append(
            {
                "zone": zone,
                "capacity_technology": technology,
                "operational_status": status,
                "status_included": included,
                "capacity_mw": float(record.Value),
                "profile_name": profile,
                "output_component": component,
                "profile_available": source is not None,
                "profile_source_member": str(source) if source else "",
            }
        )

        # The with-storage CSP release provides both the physical renewable
        # input and an official dispatched reference.  Preserve both, but do
        # not add both to the free solar potential.
        if technology == "Solar thermal with storage":
            reference_profile = "CSP_withStorage_7h_dispatched"
            reference = inventory.capacity_factor.get(
                (zone, reference_profile)
            )
            rows.append(
                {
                    "zone": zone,
                    "capacity_technology": technology,
                    "operational_status": status,
                    "status_included": included,
                    "capacity_mw": float(record.Value),
                    "profile_name": reference_profile,
                    "output_component": "csp_storage_reference",
                    "profile_available": reference is not None,
                    "profile_source_member": (
                        str(reference) if reference else ""
                    ),
                }
            )
    columns = [
        "zone",
        "capacity_technology",
        "operational_status",
        "status_included",
        "capacity_mw",
        "profile_name",
        "output_component",
        "profile_available",
        "profile_source_member",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        [
            "zone",
            "profile_name",
            "capacity_technology",
            "operational_status",
        ]
    ).reset_index(drop=True)


def required_profile_capacity(
    mapping: pd.DataFrame, zone: str
) -> pd.DataFrame:
    if mapping.empty:
        return pd.DataFrame(
            columns=["profile_name", "output_component", "capacity_mw"]
        )
    selected = mapping.loc[
        mapping["zone"].eq(zone)
        & mapping["status_included"]
        & mapping["capacity_mw"].gt(0)
    ]
    if selected.empty:
        return pd.DataFrame(
            columns=["profile_name", "output_component", "capacity_mw"]
        )
    return (
        selected.groupby(
            ["profile_name", "output_component"], as_index=False
        )["capacity_mw"]
        .sum()
        .sort_values(["profile_name", "output_component"])
        .reset_index(drop=True)
    )


def capacity_nodes(capacity: pd.DataFrame) -> list[str]:
    zones = sorted(set(capacity["Market_Node"].astype(str)))
    forbidden = sorted(EXCLUDED_AGGREGATE_ZONES.intersection(zones))
    if forbidden:
        raise ValueError(f"aggregate/alias capacity zones present: {forbidden}")
    return zones


def parse_zone_selection(values: Sequence[str] | None) -> set[str] | None:
    if not values:
        return None
    zones: set[str] = set()
    for item in values:
        zones.update(part.strip() for part in item.split(",") if part.strip())
    return zones


def make_zone_coverage(
    zones: Sequence[str],
    selected_zones: set[str],
    inventory: SourceInventory,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    constraint_counts: dict[str, int] = {}
    inflow_counts: dict[str, int] = {}
    for zone, _ in inventory.hydro_constraint:
        constraint_counts[zone] = constraint_counts.get(zone, 0) + 1
    for zone, _ in inventory.hydro_inflow:
        inflow_counts[zone] = inflow_counts.get(zone, 0) + 1

    rows: list[dict[str, Any]] = []
    for zone in zones:
        required = required_profile_capacity(mapping, zone)
        missing_profiles = [
            profile
            for profile in required["profile_name"].astype(str)
            if (zone, profile) not in inventory.capacity_factor
        ]
        expected_demand = zone in EXPECTED_DEMAND_ZONES
        demand_available = zone in inventory.demand
        expected_btm = zone in EXPECTED_BTM_ZONES
        btm_pv_available = zone in inventory.btm_pv
        btm_battery_available = zone in inventory.btm_battery
        issues: list[str] = []
        if expected_demand and not demand_available:
            issues.append("expected_demand_file_missing")
        if missing_profiles:
            issues.append("nonzero_vre_profile_missing")
        if not expected_demand and demand_available:
            issues.append("unexpected_demand_file_for_resource_node")
        if expected_btm and not btm_pv_available:
            issues.append("expected_btm_pv_file_missing")
        if expected_btm and not btm_battery_available:
            issues.append("expected_btm_battery_file_missing")
        if not expected_btm and (btm_pv_available or btm_battery_available):
            issues.append("unexpected_btm_member")
        rows.append(
            {
                "zone": zone,
                "selected": zone in selected_zones,
                "expected_demand_zone": expected_demand,
                "demand_file_available": demand_available,
                "expected_btm_zone": expected_btm,
                "btm_pv_file_available": btm_pv_available,
                "btm_battery_file_available": btm_battery_available,
                "demand_role": (
                    "modelled_demand"
                    if expected_demand
                    else "zero_load_resource_or_network_node"
                ),
                "required_nonzero_vre_profiles": len(required),
                "missing_nonzero_vre_profiles": "|".join(
                    sorted(missing_profiles)
                ),
                "hydro_constraint_members": constraint_counts.get(zone, 0),
                "hydro_inflow_members": inflow_counts.get(zone, 0),
                "hourly_input_complete": bool(
                    (demand_available or not expected_demand)
                    and (not expected_btm or btm_pv_available)
                    and (not expected_btm or btm_battery_available)
                    and not missing_profiles
                ),
                "issues": "|".join(issues),
                "hourly_shard_written": False,
                "hydro_shard_written": False,
            }
        )
    return pd.DataFrame(rows).sort_values("zone").reset_index(drop=True)


def build_hourly_zone(
    zone: str,
    calendar: pd.DataFrame,
    gross_demand_values: np.ndarray,
    btm_pv_values: np.ndarray,
    btm_battery_values: np.ndarray,
    mapping: pd.DataFrame,
    profile_capacity: pd.DataFrame,
    inventory: SourceInventory,
    used_sources: set[Path],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    shape = gross_demand_values.shape
    if shape != (len(calendar), len(WEATHER_SCENARIOS)):
        raise ValueError(f"{zone}: invalid demand matrix shape {shape}")
    for name, values in {
        "btm_pv": btm_pv_values,
        "btm_battery": btm_battery_values,
    }.items():
        if np.asarray(values).shape != shape:
            raise ValueError(f"{zone}: invalid {name} matrix shape")
        if not np.isfinite(values).all():
            raise ValueError(f"{zone}: non-finite {name} values")
    if (gross_demand_values < 0).any() or (btm_pv_values < 0).any():
        raise ValueError(f"{zone}: negative gross demand or BTM PV")
    btm_net_output = btm_pv_values + btm_battery_values
    signed_net_load = gross_demand_values - btm_net_output
    model_load = np.maximum(signed_net_load, 0.0)
    btm_surplus = np.maximum(-signed_net_load, 0.0)
    components = {
        "wind_onshore": np.zeros(shape, dtype=np.float64),
        "wind_offshore": np.zeros(shape, dtype=np.float64),
        "solar_pv": np.zeros(shape, dtype=np.float64),
        "csp_no_storage": np.zeros(shape, dtype=np.float64),
        "csp_storage_input": np.zeros(shape, dtype=np.float64),
        "csp_storage_reference": np.zeros(shape, dtype=np.float64),
    }
    profile_rows: list[dict[str, Any]] = []
    for record in profile_capacity.itertuples(index=False):
        path = inventory.capacity_factor.get((zone, record.profile_name))
        if path is None:
            raise ValueError(
                f"{zone}: required capacity factor missing for "
                f"{record.profile_name}"
            )
        profile = read_capacity_factor_wide(path)
        assert_same_calendar(calendar, profile.calendar, path)
        components[record.output_component] += (
            profile.values * float(record.capacity_mw)
        )
        used_sources.add(path)
        profile_rows.append(
            {
                "profile_name": record.profile_name,
                "component": record.output_component,
                "capacity_mw": float(record.capacity_mw),
                "mean_capacity_factor": float(profile.values.mean()),
                "maximum_capacity_factor": float(profile.values.max()),
            }
        )

    wind = components["wind_onshore"] + components["wind_offshore"]
    solar = components["solar_pv"] + components["csp_no_storage"]
    n_hours = len(calendar)
    n_weather = len(WEATHER_SCENARIOS)
    weather = np.tile(np.asarray(WEATHER_SCENARIOS), n_hours)
    weather_index = mapping.set_index("weather_scenario")
    climate_model = np.tile(
        weather_index.loc[
            list(WEATHER_SCENARIOS), "climate_model"
        ].to_numpy(),
        n_hours,
    )
    climate_code = np.tile(
        weather_index.loc[
            list(WEATHER_SCENARIOS), "climate_model_code"
        ].to_numpy(),
        n_hours,
    )
    projection_year = np.tile(
        weather_index.loc[
            list(WEATHER_SCENARIOS), "climate_projection_year"
        ].to_numpy(dtype=int),
        n_hours,
    )
    role = (
        "modelled_demand"
        if zone in EXPECTED_DEMAND_ZONES
        else "zero_load_resource_or_network_node"
    )
    frame = pd.DataFrame(
        {
            "target_year": np.full(
                n_hours * n_weather, TARGET_YEAR, dtype=np.int16
            ),
            "climate_model": climate_model,
            "climate_model_code": climate_code,
            "climate_projection_year": projection_year,
            "weather_scenario": weather,
            "zone": zone,
            "month": np.repeat(
                calendar["month"].to_numpy(dtype=np.int8), n_weather
            ),
            "day": np.repeat(
                calendar["day"].to_numpy(dtype=np.int8), n_weather
            ),
            "hour": np.repeat(
                calendar["hour"].to_numpy(dtype=np.int8), n_weather
            ),
            "hour_of_year": np.repeat(
                calendar["hour_of_year"].to_numpy(dtype=np.int16), n_weather
            ),
            "gross_demand_mw": gross_demand_values.reshape(-1),
            "btm_pv_mw": btm_pv_values.reshape(-1),
            "btm_battery_net_output_mw": btm_battery_values.reshape(-1),
            "btm_net_output_mw": btm_net_output.reshape(-1),
            "net_load_after_btm_mw": signed_net_load.reshape(-1),
            "load_mw": model_load.reshape(-1),
            "btm_surplus_potential_mw": btm_surplus.reshape(-1),
            "wind_onshore_potential_mw": components[
                "wind_onshore"
            ].reshape(-1),
            "wind_offshore_potential_mw": components[
                "wind_offshore"
            ].reshape(-1),
            "wind_potential_mw": wind.reshape(-1),
            "solar_pv_potential_mw": components["solar_pv"].reshape(-1),
            "csp_no_storage_potential_mw": components[
                "csp_no_storage"
            ].reshape(-1),
            "solar_potential_mw": solar.reshape(-1),
            "csp_with_storage_predispatch_mw": components[
                "csp_storage_input"
            ].reshape(-1),
            "csp_with_storage_dispatched_reference_mw": components[
                "csp_storage_reference"
            ].reshape(-1),
            "demand_role": role,
            "load_definition": (
                "positive_part_of_Demand_total_minus_BTM_PV_minus_"
                "BTM_battery;negative_part_retained_as_BTM_surplus"
            ),
        },
        columns=HOURLY_OUTPUT_COLUMNS,
    )
    key = ["target_year", "weather_scenario", "hour_of_year", "zone"]
    if frame.duplicated(key).any():
        raise ValueError(f"{zone}: duplicate hourly output keys")
    finite_numeric = frame[
        [
            "gross_demand_mw",
            "btm_pv_mw",
            "btm_battery_net_output_mw",
            "btm_net_output_mw",
            "net_load_after_btm_mw",
            "load_mw",
            "btm_surplus_potential_mw",
            "wind_onshore_potential_mw",
            "wind_offshore_potential_mw",
            "wind_potential_mw",
            "solar_pv_potential_mw",
            "csp_no_storage_potential_mw",
            "solar_potential_mw",
            "csp_with_storage_predispatch_mw",
            "csp_with_storage_dispatched_reference_mw",
        ]
    ].to_numpy(dtype=float)
    if not np.isfinite(finite_numeric).all():
        raise ValueError(f"{zone}: invalid numeric hourly output")
    nonnegative_numeric = frame[
        [
            "gross_demand_mw",
            "btm_pv_mw",
            "load_mw",
            "btm_surplus_potential_mw",
            "wind_onshore_potential_mw",
            "wind_offshore_potential_mw",
            "wind_potential_mw",
            "solar_pv_potential_mw",
            "csp_no_storage_potential_mw",
            "solar_potential_mw",
            "csp_with_storage_predispatch_mw",
            "csp_with_storage_dispatched_reference_mw",
        ]
    ].to_numpy(dtype=float)
    if (nonnegative_numeric < -1e-12).any():
        raise ValueError(f"{zone}: negative non-negative hourly output")
    balance_error = (
        frame["gross_demand_mw"]
        - frame["btm_net_output_mw"]
        - frame["load_mw"]
        + frame["btm_surplus_potential_mw"]
    )
    if float(balance_error.abs().max()) > 1e-8:
        raise ValueError(f"{zone}: BTM net-load decomposition failed")
    qa = {
        "zone": zone,
        "rows": len(frame),
        "weather_scenarios": int(frame["weather_scenario"].nunique()),
        "hours_per_weather_scenario": int(
            frame.groupby("weather_scenario")["hour_of_year"].nunique().min()
        ),
        "load_sum_mwh_over_weather_scenarios": float(
            frame["load_mw"].sum()
        ),
        "gross_demand_sum_mwh_over_weather_scenarios": float(
            frame["gross_demand_mw"].sum()
        ),
        "btm_pv_sum_mwh_over_weather_scenarios": float(
            frame["btm_pv_mw"].sum()
        ),
        "btm_battery_net_output_sum_mwh_over_weather_scenarios": float(
            frame["btm_battery_net_output_mw"].sum()
        ),
        "btm_surplus_sum_mwh_over_weather_scenarios": float(
            frame["btm_surplus_potential_mw"].sum()
        ),
        "minimum_signed_net_load_mw": float(
            frame["net_load_after_btm_mw"].min()
        ),
        "btm_decomposition_max_abs_error_mw": float(
            balance_error.abs().max()
        ),
        "wind_potential_sum_mwh_over_weather_scenarios": float(
            frame["wind_potential_mw"].sum()
        ),
        "solar_potential_sum_mwh_over_weather_scenarios": float(
            frame["solar_potential_mw"].sum()
        ),
        "profile_applications": profile_rows,
    }
    return frame, qa


def build_hydro_zone(
    zone: str,
    mapping: pd.DataFrame,
    inventory: SourceInventory,
    used_sources: set[Path],
) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    codes = sorted(
        {
            code
            for member_zone, code in (
                *inventory.hydro_constraint.keys(),
                *inventory.hydro_inflow.keys(),
            )
            if member_zone == zone
        }
    )
    if not codes:
        return None

    weather_mapping = mapping.set_index("weather_scenario")
    frames: list[pd.DataFrame] = []
    issues: list[str] = []
    for code in codes:
        constraint_path = inventory.hydro_constraint.get((zone, code))
        inflow_path = inventory.hydro_inflow.get((zone, code))
        if constraint_path is not None:
            constraint = read_hydro_constraints(constraint_path)
            used_sources.add(constraint_path)
            constraint_long = constraint.loc[
                constraint.index.repeat(len(WEATHER_SCENARIOS))
            ].reset_index(drop=True)
            constraint_long["weather_scenario"] = np.tile(
                np.asarray(WEATHER_SCENARIOS), len(constraint)
            )
        else:
            constraint_long = None
            issues.append(f"{code}:constraint_not_released")

        if inflow_path is not None:
            inflow = read_hydro_inflows(inflow_path)
            used_sources.add(inflow_path)
        else:
            inflow = None
            if code != "HCL":
                issues.append(f"{code}:inflow_not_released")

        if constraint_long is None and inflow is None:
            raise ValueError(
                f"{zone} {code}: neither constraints nor inflow released"
            )
        if constraint_long is not None and inflow is not None:
            resolutions = set(constraint_long["time_resolution"]).union(
                set(inflow["time_resolution"])
            )
            if len(resolutions) != 1:
                raise ValueError(
                    f"{zone} {code}: constraint and inflow resolutions differ"
                )
            merged = constraint_long.merge(
                inflow,
                on=[
                    "time_resolution",
                    "period",
                    "weather_scenario",
                ],
                how="outer",
                validate="one_to_one",
            )
        elif constraint_long is not None:
            merged = constraint_long.copy()
            # Closed-loop storage has no natural inflow.  Zero is a physical
            # non-applicability value here, not an imputation of missing data.
            merged["inflow_raw"] = 0.0 if code == "HCL" else np.nan
        else:
            merged = inflow.copy()
            for column in HYDRO_CONSTRAINT_COLUMNS:
                merged[f"{column.lower()}_raw"] = np.nan
            merged["released_constraint_columns"] = ""

        merged.insert(0, "target_year", TARGET_YEAR)
        merged.insert(1, "zone", zone)
        merged.insert(2, "hydro_code", code)
        merged.insert(3, "hydro_type", HYDRO_TYPES[code])
        merged["climate_model"] = merged["weather_scenario"].map(
            weather_mapping["climate_model"]
        )
        merged["climate_model_code"] = merged["weather_scenario"].map(
            weather_mapping["climate_model_code"]
        )
        merged["climate_projection_year"] = merged[
            "weather_scenario"
        ].map(weather_mapping["climate_projection_year"])
        merged["natural_inflow_applicable"] = code != "HCL"
        merged["constraint_source_member"] = (
            str(constraint_path) if constraint_path is not None else ""
        )
        merged["inflow_source_member"] = (
            str(inflow_path) if inflow_path is not None else ""
        )
        frames.append(merged)

    result = pd.concat(frames, ignore_index=True)
    result = result[
        [
            "target_year",
            "climate_model",
            "climate_model_code",
            "climate_projection_year",
            "weather_scenario",
            "zone",
            "hydro_code",
            "hydro_type",
            "time_resolution",
            "period",
            "inflow_raw",
            "max_turb_en_raw",
            "max_turb_raw",
            "max_pump_raw",
            "max_res_raw",
            "min_turb_en_raw",
            "min_turb_raw",
            "min_pump_raw",
            "min_res_raw",
            "released_constraint_columns",
            "natural_inflow_applicable",
            "constraint_source_member",
            "inflow_source_member",
        ]
    ].sort_values(
        [
            "hydro_code",
            "time_resolution",
            "weather_scenario",
            "period",
        ]
    ).reset_index(drop=True)
    key = [
        "target_year",
        "weather_scenario",
        "zone",
        "hydro_code",
        "time_resolution",
        "period",
    ]
    if result.duplicated(key).any():
        raise ValueError(f"{zone}: duplicate weekly hydro output keys")
    return result, {
        "zone": zone,
        "rows": len(result),
        "hydro_codes": codes,
        "issues": issues,
    }


def make_compute_scale_estimate() -> dict[str, Any]:
    zones = 67
    demand_zones = 52
    weather = 36
    hours = 8_760
    configurations = 50
    chronologies = 2
    outage_paths = 15
    estimands = 2
    base_hourly_rows = zones * weather * hours
    one_config_chronology_states = (
        zones * weather * outage_paths * hours
    )
    full_configuration_states = (
        one_config_chronology_states * configurations * chronologies
    )
    both_estimands_states = full_configuration_states * estimands
    return {
        "target_year": TARGET_YEAR,
        "atomic_capacity_nodes": zones,
        "demand_nodes": demand_zones,
        "synchronized_weather_scenarios": weather,
        "hours_per_weather_scenario": hours,
        "main_configurations": configurations,
        "chronology_endpoints": chronologies,
        "forced_outage_paths": outage_paths,
        "separately_dispatched_estimands": [
            "joint_value",
            "turnover_value_at_fixed_firm_capacity",
        ],
        "base_node_weather_hour_rows": base_hourly_rows,
        "rows_if_configs_were_wrongly_replicated_into_inputs": (
            base_hourly_rows * configurations
        ),
        "node_scenario_hours_one_config_one_chronology": (
            one_config_chronology_states
        ),
        "node_scenario_configuration_hours_50_configs_two_chronologies": (
            full_configuration_states
        ),
        "node_scenario_configuration_hours_joint_plus_turnover": (
            both_estimands_states
        ),
        "independent_weather_repeats_for_inference": weather,
        "weather_outage_subproblems_per_outer_task": weather * outage_paths,
        "outer_task_key": (
            "target_year__network_case__chronology__config_id__"
            "tranche__estimand"
        ),
        "decomposition": {
            "first_stage": [
                "shared firm capacity",
                "candidate storage siting",
                "charge power",
                "discharge power",
                "energy capacity",
                "predeclared network investment if enabled",
            ],
            "second_stage": (
                "one connected European dispatch subproblem per "
                "weather_scenario x forced_outage_path"
            ),
            "method": "L-shaped/Benders scenario decomposition",
            "zone_split_prohibited": True,
            "time_split_fallback": (
                "52 linked 168-h blocks plus one 24-h terminal block; "
                "SOC and reservoir states linked; one annual cyclic boundary"
            ),
            "baseline_reuse": (
                "solve each chronology baseline once; never repeat it for "
                "all 50 configurations"
            ),
            "elcc_scope": (
                "coarse bracketing and bisection only at preregistered "
                "key nodes and duration-boundary neighbours"
            ),
            "checkpoint_key_fields": [
                "target_year",
                "network_case",
                "chronology",
                "config_id",
                "tranche",
                "estimand",
                "weather_scenario",
                "forced_outage_path",
                "input_sha256",
                "solver_version",
                "random_seed",
            ],
        },
        "interpretation": (
            "State counts are computational load, not statistical sample "
            "sizes. The 36 synchronized European weather scenarios remain "
            "the weather repeats."
        ),
    }


def source_inventory_rows(
    source_root: Path,
    inventory: SourceInventory,
    used_sources: Iterable[Path],
    capacity_path: Path,
    hydro_capacity_path: Path,
    aggregated_demand_path: Path,
    weather_path: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    all_paths = set(used_sources)
    all_paths.add(capacity_path)
    all_paths.add(weather_path)
    if hydro_capacity_path.exists():
        all_paths.add(hydro_capacity_path)
    if aggregated_demand_path.exists():
        all_paths.add(aggregated_demand_path)
    for path in sorted(all_paths):
        try:
            relative = path.relative_to(source_root)
            role = "selective_release_member"
        except ValueError:
            relative = path
            role = "dashboard_or_mapping_source"
        rows.append(
            {
                "source_role": role,
                "path": str(relative),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(rows)


def prepare_output(output: Path, force: bool) -> None:
    if output.exists():
        if not force:
            raise FileExistsError(
                f"output already exists: {output}; use --force to replace it"
            )
        shutil.rmtree(output)
    output.mkdir(parents=True)


def execute_build(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root.resolve()
    output = args.output.resolve()
    capacity_path = args.capacity_file.resolve()
    hydro_capacity_path = args.hydro_capacity_file.resolve()
    aggregated_demand_path = args.aggregated_demand_file.resolve()
    if not source_root.exists():
        raise FileNotFoundError(source_root)
    if not capacity_path.exists():
        raise FileNotFoundError(capacity_path)

    inventory = discover_sources(source_root)
    if inventory.duplicate_members:
        raise ValueError(
            "duplicate ERAA source keys: "
            + "; ".join(inventory.duplicate_members[:10])
        )
    weather_path = find_weather_mapping(source_root)
    weather_mapping = read_weather_mapping(weather_path)
    capacity = read_capacity_table(capacity_path)
    hydro_capacity = read_hydro_capacity_table(hydro_capacity_path)
    dashboard_demand = read_dashboard_annual_demand(
        aggregated_demand_path
    )
    zones = capacity_nodes(capacity)
    if len(zones) != 67:
        raise ValueError(
            f"expected 67 ERAA-final 2035 capacity nodes, observed {len(zones)}"
        )
    if len(EXPECTED_DEMAND_ZONES) != 52:
        raise AssertionError("internal demand-zone contract is not 52 zones")

    requested = parse_zone_selection(args.zones)
    if requested is None:
        selected_zones = set(zones)
    else:
        unknown = sorted(requested.difference(zones))
        if unknown:
            raise ValueError(f"--zones contains unknown capacity nodes {unknown}")
        selected_zones = requested

    vre_mapping = build_vre_mapping(capacity, inventory)
    demand_column_audit = audit_demand_weather_column_order(inventory)
    coverage = make_zone_coverage(
        zones, selected_zones, inventory, vre_mapping
    )
    source_counts = inventory.counts
    source_count_complete = all(
        source_counts[key] == expected
        for key, expected in EXPECTED_SOURCE_COUNTS.items()
    )

    prepare_output(output, args.force)
    write_csv_atomic(
        capacity,
        output / "capacity_2035_final.csv",
    )
    if not hydro_capacity.empty:
        write_csv_atomic(
            hydro_capacity,
            output / "hydro_capacity_2035_final.csv",
        )
    write_csv_atomic(
        weather_mapping,
        output / "weather_scenario_mapping.csv",
    )
    write_csv_atomic(
        vre_mapping,
        output / "vre_capacity_profile_mapping.csv",
    )
    write_csv_atomic(
        demand_column_audit,
        output / "demand_weather_column_order_audit.csv",
    )
    write_json_atomic(
        output / "compute_scale_estimate.json",
        make_compute_scale_estimate(),
    )

    shard_rows: list[dict[str, Any]] = []
    zone_qa: list[dict[str, Any]] = []
    hydro_qa: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    demand_reconciliation_rows: list[dict[str, Any]] = []
    used_sources: set[Path] = {weather_path}

    if not args.audit_only:
        # A released demand calendar is the canonical non-leap axis for every
        # resource-only node.  Missing demand for an expected demand zone is
        # never replaced by this template.
        if not inventory.demand:
            raise ValueError("no ERAA 2035 demand files are available")
        template_zone = sorted(inventory.demand)[0]
        template_profile = read_demand_wide(inventory.demand[template_zone])
        template_calendar = template_profile.calendar
        used_sources.add(inventory.demand[template_zone])

        for zone in sorted(selected_zones):
            coverage_index = coverage.index[coverage["zone"].eq(zone)][0]
            row = coverage.loc[coverage_index]
            if not bool(row["hourly_input_complete"]):
                errors.append(
                    {
                        "zone": zone,
                        "stage": "hourly_precheck",
                        "error": str(row["issues"]),
                    }
                )
                if args.allow_partial:
                    continue
                raise ValueError(
                    f"{zone}: incomplete hourly input: {row['issues']}"
                )
            try:
                if zone in EXPECTED_DEMAND_ZONES:
                    demand_path = inventory.demand[zone]
                    demand = read_demand_wide(demand_path)
                    assert_same_calendar(
                        template_calendar, demand.calendar, demand_path
                    )
                    gross_demand_values = demand.values
                    btm_pv_values = np.zeros_like(gross_demand_values)
                    btm_battery_values = np.zeros_like(gross_demand_values)
                    if zone in EXPECTED_BTM_ZONES:
                        pv_path = inventory.btm_pv[zone]
                        battery_path = inventory.btm_battery[zone]
                        pv = read_btm_wide(pv_path, component="pv")
                        battery = read_btm_wide(
                            battery_path, component="battery"
                        )
                        assert_same_calendar(
                            demand.calendar, pv.calendar, pv_path
                        )
                        assert_same_calendar(
                            demand.calendar, battery.calendar, battery_path
                        )
                        btm_pv_values = pv.values
                        btm_battery_values = battery.values
                        used_sources.update({pv_path, battery_path})
                    calendar = demand.calendar
                    demand_column_order_canonical = (
                        demand.canonical_weather_column_order
                    )
                    demand_column_order = "|".join(
                        demand.weather_column_order
                    )
                    used_sources.add(demand_path)
                    net_load_after_btm = (
                        gross_demand_values
                        - btm_pv_values
                        - btm_battery_values
                    )
                    annual_twh = (
                        net_load_after_btm.sum(axis=0) / 1_000_000.0
                    )
                    gross_annual_twh = (
                        gross_demand_values.sum(axis=0) / 1_000_000.0
                    )
                    dashboard_match = dashboard_demand.loc[
                        dashboard_demand["zone"].eq(zone)
                    ]
                    if dashboard_match.empty:
                        dashboard_values = {
                            "dashboard_avg_twh": np.nan,
                            "dashboard_min_twh": np.nan,
                            "dashboard_max_twh": np.nan,
                        }
                    else:
                        dashboard_values = dashboard_match.iloc[0].to_dict()
                    demand_reconciliation_rows.append(
                        {
                            "zone": zone,
                            "hourly_mean_twh": float(annual_twh.mean()),
                            "hourly_min_twh": float(annual_twh.min()),
                            "hourly_max_twh": float(annual_twh.max()),
                            "gross_demand_mean_twh": float(
                                gross_annual_twh.mean()
                            ),
                            "btm_pv_mean_twh": float(
                                btm_pv_values.sum(axis=0).mean()
                                / 1_000_000.0
                            ),
                            "btm_battery_net_output_mean_twh": float(
                                btm_battery_values.sum(axis=0).mean()
                                / 1_000_000.0
                            ),
                            **{
                                key: dashboard_values[key]
                                for key in (
                                    "dashboard_avg_twh",
                                    "dashboard_min_twh",
                                    "dashboard_max_twh",
                                )
                            },
                            "mean_difference_twh_hourly_minus_dashboard": (
                                float(annual_twh.mean())
                                - float(
                                    dashboard_values[
                                        "dashboard_avg_twh"
                                    ]
                                )
                                if pd.notna(
                                    dashboard_values[
                                        "dashboard_avg_twh"
                                    ]
                                )
                                else np.nan
                            ),
                            "dashboard_used_to_modify_hourly_profiles": False,
                            "interpretation": (
                                "reconciliation_only_no_automatic_correction"
                            ),
                        }
                    )
                else:
                    calendar = template_calendar
                    gross_demand_values = np.zeros(
                        (EXPECTED_HOURS, len(WEATHER_SCENARIOS)),
                        dtype=np.float64,
                    )
                    btm_pv_values = np.zeros_like(gross_demand_values)
                    btm_battery_values = np.zeros_like(
                        gross_demand_values
                    )
                    demand_column_order_canonical = True
                    demand_column_order = "not_applicable_zero_load_node"

                profiles = required_profile_capacity(vre_mapping, zone)
                hourly, hourly_qa = build_hourly_zone(
                    zone,
                    calendar,
                    gross_demand_values,
                    btm_pv_values,
                    btm_battery_values,
                    weather_mapping,
                    profiles,
                    inventory,
                    used_sources,
                )
                hourly_qa["demand_weather_columns_read_by_name"] = True
                hourly_qa[
                    "released_demand_weather_column_order_canonical"
                ] = demand_column_order_canonical
                hourly_qa[
                    "released_demand_weather_column_order"
                ] = demand_column_order
                hourly_path = (
                    output
                    / "hourly_profiles"
                    / f"zone={zone}"
                    / "hourly_profiles.csv.gz"
                )
                write_csv_atomic(hourly, hourly_path, gzip_output=True)
                shard_rows.append(
                    {
                        "kind": "hourly_profiles",
                        "zone": zone,
                        "relative_path": str(hourly_path.relative_to(output)),
                        "rows": len(hourly),
                        "bytes": hourly_path.stat().st_size,
                        "sha256": sha256_file(hourly_path),
                    }
                )
                zone_qa.append(hourly_qa)
                coverage.loc[
                    coverage_index, "hourly_shard_written"
                ] = True
            except Exception as error:
                errors.append(
                    {
                        "zone": zone,
                        "stage": "hourly_build",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                if not args.allow_partial:
                    raise

            try:
                hydro_built = build_hydro_zone(
                    zone, weather_mapping, inventory, used_sources
                )
                if hydro_built is not None:
                    hydro, one_hydro_qa = hydro_built
                    hydro_path = (
                        output
                        / "hydro_periodic"
                        / f"zone={zone}"
                        / "hydro_periodic.csv.gz"
                    )
                    write_csv_atomic(hydro, hydro_path, gzip_output=True)
                    shard_rows.append(
                        {
                            "kind": "hydro_periodic",
                            "zone": zone,
                            "relative_path": str(
                                hydro_path.relative_to(output)
                            ),
                            "rows": len(hydro),
                            "bytes": hydro_path.stat().st_size,
                            "sha256": sha256_file(hydro_path),
                        }
                    )
                    hydro_qa.append(one_hydro_qa)
                    coverage.loc[
                        coverage_index, "hydro_shard_written"
                    ] = True
            except Exception as error:
                errors.append(
                    {
                        "zone": zone,
                        "stage": "hydro_build",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                if not args.allow_partial:
                    raise

    write_csv_atomic(coverage, output / "zone_coverage.csv")
    demand_reconciliation = pd.DataFrame(
        demand_reconciliation_rows,
        columns=[
            "zone",
            "hourly_mean_twh",
            "hourly_min_twh",
            "hourly_max_twh",
            "gross_demand_mean_twh",
            "btm_pv_mean_twh",
            "btm_battery_net_output_mean_twh",
            "dashboard_avg_twh",
            "dashboard_min_twh",
            "dashboard_max_twh",
            "mean_difference_twh_hourly_minus_dashboard",
            "dashboard_used_to_modify_hourly_profiles",
            "interpretation",
        ],
    )
    write_csv_atomic(
        demand_reconciliation,
        output / "demand_dashboard_reconciliation.csv",
    )
    shard_manifest = pd.DataFrame(
        shard_rows,
        columns=[
            "kind",
            "zone",
            "relative_path",
            "rows",
            "bytes",
            "sha256",
        ],
    )
    if not shard_manifest.empty:
        shard_manifest = shard_manifest.sort_values(
            ["kind", "zone"]
        ).reset_index(drop=True)
    write_csv_atomic(shard_manifest, output / "shard_manifest.csv")

    source_manifest = source_inventory_rows(
        source_root,
        inventory,
        used_sources,
        capacity_path,
        hydro_capacity_path,
        aggregated_demand_path,
        weather_path,
    )
    write_csv_atomic(source_manifest, output / "source_manifest.csv")

    selected_coverage = coverage.loc[coverage["selected"]]
    all_selected_hourly_complete = bool(
        selected_coverage["hourly_input_complete"].all()
    )
    expected_all_zones = selected_zones == set(zones)
    all_hourly_written = bool(
        (not args.audit_only)
        and selected_coverage["hourly_shard_written"].all()
    )
    release_eligible = bool(
        source_count_complete
        and expected_all_zones
        and all_selected_hourly_complete
        and all_hourly_written
        and not errors
    )
    qa = {
        "script_version": SCRIPT_VERSION,
        "target_year": TARGET_YEAR,
        "source_counts_observed": source_counts,
        "source_counts_expected": EXPECTED_SOURCE_COUNTS,
        "source_count_complete": source_count_complete,
        "capacity_rows": len(capacity),
        "capacity_nodes": len(zones),
        "demand_zone_contract_count": len(EXPECTED_DEMAND_ZONES),
        "demand_weather_column_order_anomaly_zones": (
            demand_column_audit.loc[
                ~demand_column_audit[
                    "weather_column_order_canonical"
                ],
                "zone",
            ].tolist()
        ),
        "demand_dashboard_reconciliation_rows": len(
            demand_reconciliation
        ),
        "selected_zone_count": len(selected_zones),
        "hourly_shards_written": int(
            coverage["hourly_shard_written"].sum()
        ),
        "hydro_shards_written": int(
            coverage["hydro_shard_written"].sum()
        ),
        "audit_only": bool(args.audit_only),
        "allow_partial": bool(args.allow_partial),
        "all_selected_hourly_complete": all_selected_hourly_complete,
        "release_eligible": release_eligible,
        "errors": errors,
        "zone_hourly_qa": zone_qa,
        "zone_hydro_qa": hydro_qa,
        "important_interpretation": [
            "A missing official demand member is never zero-filled.",
            (
                "Demand weather scenarios are selected by normalized WS "
                "column names, never by column position; released ordering "
                "anomalies are recorded rather than silently rearranged at "
                "source level."
            ),
            (
                "Dashboard Aggregated_Demand is an external reconciliation "
                "check only and is not used to overwrite released hourly "
                "profiles. For AT00, CH00, DE00 and HU00, the released "
                "identity Demand_total - BTM_PV - BTM_battery is applied "
                "before reconciliation."
            ),
            (
                "Negative net demand after BTM is preserved as a separate "
                "BTM surplus potential rather than clipped or discarded."
            ),
            (
                "The 36 WS20 values above one in the official PT00 onshore-"
                "wind profile (maximum 1.051521884) are preserved exactly "
                "in the main replication and are unit-capped only in a "
                "separately labelled sensitivity run."
            ),
            (
                "Only capacity nodes outside the frozen 52-zone demand "
                "contract receive zero load."
            ),
            (
                "Hydrological release fields retain their official weekly "
                "or daily resolution and raw suffixes; absent constraint "
                "columns remain missing rather than becoming zero."
            ),
            (
                "Zone shards are storage boundaries only; the model must "
                "solve the connected European network."
            ),
        ],
    }
    write_json_atomic(output / "QA.json", qa)
    run_manifest = {
        "script": Path(__file__).name,
        "script_version": SCRIPT_VERSION,
        "generated_at_utc": utc_now(),
        "source_root": source_root,
        "output": output,
        "capacity_file": capacity_path,
        "hydro_capacity_file": hydro_capacity_path,
        "aggregated_demand_file": aggregated_demand_path,
        "weather_mapping_file": weather_path,
        "selected_zones": sorted(selected_zones),
        "allow_partial": bool(args.allow_partial),
        "audit_only": bool(args.audit_only),
        "release_eligible": release_eligible,
        "source_manifest_sha256": sha256_file(
            output / "source_manifest.csv"
        ),
        "shard_manifest_sha256": sha256_file(
            output / "shard_manifest.csv"
        ),
        "qa_sha256": sha256_file(output / "QA.json"),
    }
    write_json_atomic(output / "run_manifest.json", run_manifest)

    if not args.allow_partial and not release_eligible:
        raise RuntimeError(
            "strict build completed without release eligibility; inspect QA.json"
        )
    return qa


def make_synthetic_cf_file(path: Path, hours: int = 48) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        metadata = [
            ("Country", "XX"),
            ("PECD Zone", "XX00"),
            ("Target Year", "2035"),
            ("Resolution", "Hourly"),
            ("Unit", "p.u."),
            ("Source", "synthetic parser test"),
            ("Metadata 7", ""),
            ("Metadata 8", ""),
            ("Metadata 9", ""),
            ("Metadata 10", ""),
        ]
        writer.writerows(metadata)
        writer.writerow(["Date", "Hour", *WEATHER_SCENARIOS])
        for hour_of_year in range(hours):
            day = hour_of_year // 24 + 1
            hour = hour_of_year % 24 + 1
            values = [
                round(
                    0.1
                    + 0.7
                    * ((hour_of_year + weather_index) % 48)
                    / 47.0,
                    8,
                )
                for weather_index in range(36)
            ]
            writer.writerow([f"{day:02d}.01.", hour, *values])


def make_synthetic_hydro_files(
    constraint_path: Path,
    inflow_path: Path,
    *,
    resolution: str = "week",
) -> None:
    time_key = resolution.upper()
    periods = 53 if resolution == "week" else 365
    with constraint_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([time_key, "MAX_TURB", "MIN_TURB"])
        for period in range(1, periods + 1):
            writer.writerow([period, 1.0, 0.1])
    with inflow_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([time_key, *WEATHER_SCENARIOS])
        for period in range(1, periods + 1):
            writer.writerow(
                [
                    period,
                    *[
                        period * 10 + index
                        for index in range(36)
                    ],
                ]
            )


def run_self_test(source_root: Path) -> dict[str, Any]:
    results: dict[str, Any] = {
        "script_version": SCRIPT_VERSION,
        "tests": {},
        "pass": False,
    }
    inventory = discover_sources(source_root) if source_root.exists() else None
    if inventory and inventory.demand:
        real_zone = sorted(inventory.demand)[0]
        real_demand = read_demand_wide(inventory.demand[real_zone])
        results["tests"]["real_demand_parser"] = {
            "pass": (
                real_demand.values.shape
                == (EXPECTED_HOURS, len(WEATHER_SCENARIOS))
            ),
            "zone": real_zone,
            "rows": len(real_demand.calendar),
            "weather_scenarios": real_demand.values.shape[1],
        }
    else:
        results["tests"]["real_demand_parser"] = {
            "pass": False,
            "reason": "no real selectively extracted demand member available",
        }

    with tempfile.TemporaryDirectory(
        prefix="eraa_2035_builder_selftest_"
    ) as temporary_text:
        temporary = Path(temporary_text)
        cf_path = temporary / (
            "XX00_CapacityFactors_Wind_Onshore_2035.csv"
        )
        make_synthetic_cf_file(cf_path)
        cf = read_capacity_factor_wide(cf_path, expected_hours=48)
        results["tests"]["official_like_cf_parser"] = {
            "pass": (
                cf.values.shape == (48, 36)
                and float(cf.values.min()) >= 0
                and float(cf.values.max()) <= 1
                and tuple(cf.calendar["hour"].head(24))
                == tuple(range(24))
            ),
            "shape": list(cf.values.shape),
            "minimum": float(cf.values.min()),
            "maximum": float(cf.values.max()),
        }
        csp_path = temporary / (
            "XX00_CapacityFactors_"
            "CSP_withStorage_7h_preDispatch_2035.csv"
        )
        make_synthetic_cf_file(csp_path)
        csp_text = csp_path.read_text(encoding="utf-8")
        csp_path.write_text(
            csp_text.replace(",0.8", ",1.6"),
            encoding="utf-8",
        )
        csp = read_capacity_factor_wide(csp_path, expected_hours=48)
        results["tests"]["csp_predispatch_solar_multiple"] = {
            "pass": float(csp.values.max()) > 1.0,
            "maximum": float(csp.values.max()),
            "interpretation": (
                "solar_field_input_not_installed_capacity_factor"
            ),
        }

        constraint_path = temporary / (
            "XX00_Hydro_Constraints_HRI_2035.csv"
        )
        inflow_path = temporary / "XX00_Hydro_Inflows_HRI_2035.csv"
        make_synthetic_hydro_files(constraint_path, inflow_path)
        constraint = read_hydro_constraints(constraint_path)
        inflow = read_hydro_inflows(inflow_path)
        daily_constraint_path = temporary / (
            "XX00_Hydro_Constraints_HRR_2035.csv"
        )
        daily_inflow_path = temporary / (
            "XX00_Hydro_Inflows_HRR_2035.csv"
        )
        make_synthetic_hydro_files(
            daily_constraint_path,
            daily_inflow_path,
            resolution="day",
        )
        daily_constraint = read_hydro_constraints(
            daily_constraint_path
        )
        daily_inflow = read_hydro_inflows(daily_inflow_path)
        results["tests"]["hydro_parser"] = {
            "pass": (
                len(constraint) == 53
                and len(inflow) == 53 * 36
                and len(daily_constraint) == 365
                and len(daily_inflow) == 365 * 36
                and constraint["max_res_raw"].isna().all()
                and daily_constraint["max_res_raw"].isna().all()
            ),
            "constraint_rows": len(constraint),
            "inflow_rows": len(inflow),
            "daily_constraint_rows": len(daily_constraint),
            "daily_inflow_rows": len(daily_inflow),
            "missing_constraints_preserved_as_nan": bool(
                constraint["max_res_raw"].isna().all()
            ),
        }

        deterministic = pd.DataFrame(
            {"a": [1, 2], "b": [0.125, 0.25]}
        )
        gzip_a = temporary / "a.csv.gz"
        gzip_b = temporary / "b.csv.gz"
        write_csv_atomic(deterministic, gzip_a, gzip_output=True)
        write_csv_atomic(deterministic, gzip_b, gzip_output=True)
        results["tests"]["deterministic_gzip"] = {
            "pass": sha256_file(gzip_a) == sha256_file(gzip_b),
            "sha256": sha256_file(gzip_a),
        }

    if inventory is not None:
        results["selective_source_readiness"] = {
            "observed_counts": inventory.counts,
            "expected_counts": EXPECTED_SOURCE_COUNTS,
            "complete": inventory.counts == EXPECTED_SOURCE_COUNTS,
            "duplicates": list(inventory.duplicate_members),
        }
    results["tests"]["compute_scale_arithmetic"] = {
        "pass": (
            make_compute_scale_estimate()[
                "base_node_weather_hour_rows"
            ]
            == 21_129_120
            and make_compute_scale_estimate()[
                "node_scenario_configuration_hours_joint_plus_turnover"
            ]
            == 63_387_360_000
        )
    }
    results["pass"] = all(
        bool(test.get("pass")) for test in results["tests"].values()
    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="selectively extracted ERAA 2035 root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="output directory",
    )
    parser.add_argument(
        "--capacity-file",
        type=Path,
        default=DEFAULT_CAPACITY_FILE,
        help="ERAA dashboard GenerationCapacities.csv",
    )
    parser.add_argument(
        "--hydro-capacity-file",
        type=Path,
        default=DEFAULT_HYDRO_CAPACITY_FILE,
        help="ERAA dashboard Hydro additional information.csv",
    )
    parser.add_argument(
        "--aggregated-demand-file",
        type=Path,
        default=DEFAULT_AGGREGATED_DEMAND_FILE,
        help=(
            "ERAA dashboard Aggregated_Demand.csv, used for "
            "reconciliation only"
        ),
    )
    parser.add_argument(
        "--zones",
        nargs="*",
        help="optional zones, separated by spaces or commas",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="skip incomplete selected zones and mark release_eligible=false",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="write readiness metadata without hourly/hydro shards",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run parser and determinism tests, then exit",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing output directory",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        result = run_self_test(args.source_root.resolve())
        print(
            json.dumps(
                json_ready(result),
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if result["pass"] else 1
    qa = execute_build(args)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "source_counts": qa["source_counts_observed"],
                "hourly_shards_written": qa["hourly_shards_written"],
                "release_eligible": qa["release_eligible"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
