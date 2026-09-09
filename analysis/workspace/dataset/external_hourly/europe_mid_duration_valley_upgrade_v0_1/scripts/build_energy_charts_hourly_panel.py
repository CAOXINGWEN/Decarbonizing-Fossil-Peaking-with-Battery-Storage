#!/usr/bin/env python3
"""Build an auditable Energy-Charts load-wind-solar hourly panel.

The input consists of country-year JSON responses archived by
``download_energy_charts_europe.py``.  API requests use each country's local
calendar.  This builder first combines adjacent source files on their genuine
UTC timestamps and only then slices UTC model years.  That avoids silently
labelling local-clock observations as UTC.

Eligibility follows the pre-specified exploratory physical-profile contract:

* load coverage >= 99%;
* joint wind-plus-solar coverage >= 99%;
* maximum consecutive gap in either series <= 24 h; and
* positive annual load and wind-plus-solar energy.

Eligible short gaps are linearly interpolated within a country-year.  A short
gap touching a UTC-year boundary is completed with the nearest observed value
and is counted separately in the QA table.  Leap-day hours are removed after
QA and imputation so every retained model year has exactly 8,760 hours.

The model-facing output deliberately contains no gas-generation input.
Energy-Charts ``Fossil gas`` is an aggregate generation response (including
different plant and operating roles), not an observed peaking-demand series.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


COUNTRY_METADATA = {
    "at": ("AT", "Austria", True),
    "be": ("BE", "Belgium", True),
    "bg": ("BG", "Bulgaria", True),
    "ch": ("CH", "Switzerland", False),
    "cz": ("CZ", "Czechia", True),
    "de": ("DE", "Germany", True),
    "dk": ("DK", "Denmark", True),
    "ee": ("EE", "Estonia", True),
    "es": ("ES", "Spain", True),
    "fr": ("FR", "France", True),
    "uk": ("GB", "United Kingdom", False),
    "gr": ("GR", "Greece", True),
    "hr": ("HR", "Croatia", True),
    "it": ("IT", "Italy", True),
    "pt": ("PT", "Portugal", True),
    "ro": ("RO", "Romania", True),
    "si": ("SI", "Slovenia", True),
}
SOURCE_NAME = "Energy-Charts.info"
SOURCE_ENDPOINT = "public_power"
SOURCE_API = "https://api.energy-charts.info/public_power"
SOURCE_LICENSE = "CC BY 4.0 unless an upstream source states otherwise"
MIN_COVERAGE = 0.99
MAX_GAP_H = 24
MIN_EXPANDED_YEARS = 3
TARGET_YEARS = tuple(range(2015, 2025))
RAW_PATTERN = re.compile(r"public_power_([a-z]+)_(\d{4})\.json$")
MODEL_FIELDS = ("load_mw", "wind_mw", "solar_mw")


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_to_csv(frame: pd.DataFrame, path: Path, **kwargs: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, **kwargs)
    temporary.replace(path)


def longest_missing_run(values: pd.Series | np.ndarray) -> int:
    missing = (
        values.isna().to_numpy()
        if isinstance(values, pd.Series)
        else np.asarray(values, dtype=bool)
    )
    if not missing.any():
        return 0
    changes = np.diff(np.r_[False, missing, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return int((ends - starts).max(initial=0))


def expected_utc_index(year: int) -> pd.DatetimeIndex:
    return pd.date_range(
        f"{year}-01-01 00:00:00+00:00",
        f"{year + 1}-01-01 00:00:00+00:00",
        freq="h",
        inclusive="left",
    )


def read_public_power(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    timestamps = pd.to_datetime(payload["unix_seconds"], unit="s", utc=True)
    data: dict[str, Any] = {"timestamp_utc": timestamps}
    duplicate_names: list[str] = []
    for series in payload["production_types"]:
        name = slug(str(series["name"]))
        if name in data:
            duplicate_names.append(name)
            continue
        values = pd.to_numeric(pd.Series(series["data"]), errors="coerce")
        if len(values) != len(timestamps):
            raise RuntimeError(f"Length mismatch in {path}: {series['name']}")
        data[name] = values.to_numpy(float)
    if duplicate_names:
        raise RuntimeError(
            f"Duplicate slugged fields in {path}: {sorted(set(duplicate_names))}"
        )
    frame = pd.DataFrame(data).set_index("timestamp_utc").sort_index()
    if frame.index.has_duplicates:
        frame = frame.groupby(level=0).mean()
    source_step_minutes = (
        float(frame.index.to_series().diff().dropna().dt.total_seconds().median())
        / 60.0
    )
    frame = frame.resample("1h").mean()

    if "load" not in frame:
        frame["load"] = np.nan
    wind_columns = [
        name
        for name in ("wind_onshore", "wind_offshore")
        if name in frame and frame[name].notna().any()
    ]
    if wind_columns:
        # Joint completeness is intentional when both components are reported.
        wind = frame[wind_columns].sum(axis=1, min_count=len(wind_columns))
    else:
        wind = pd.Series(np.nan, index=frame.index)
    solar = (
        frame["solar"]
        if "solar" in frame
        else pd.Series(np.nan, index=frame.index)
    )
    selected = pd.DataFrame(
        {
            "load_mw": frame["load"],
            "wind_mw": wind,
            "solar_mw": solar,
        },
        index=frame.index,
    )
    return selected, {
        "raw_timestamps": int(len(timestamps)),
        "hourly_timestamps": int(len(selected)),
        "source_step_minutes": source_step_minutes,
        "wind_source_fields": ";".join(wind_columns) if wind_columns else "",
        "has_load_field": "load" in data,
        "has_solar_field": "solar" in data,
    }


def combine_country_files(
    paths: list[Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pieces: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    for path in sorted(paths):
        match = RAW_PATTERN.search(path.name)
        if not match:
            continue
        api_code, source_year = match.groups()
        frame, record = read_public_power(path)
        pieces.append(frame)
        records.append(
            {
                "api_code": api_code,
                "source_local_calendar_year": int(source_year),
                "source_url": (
                    f"{SOURCE_API}?country={api_code}"
                    f"&start={source_year}-01-01"
                    f"&end={source_year}-12-31"
                ),
                "source_file": str(path),
                "source_bytes": int(path.stat().st_size),
                "source_sha256": sha256(path),
                **record,
            }
        )
    if not pieces:
        raise RuntimeError("No readable country files")
    combined = pd.concat(pieces).sort_index()
    duplicate_utc_hours = int(combined.index.duplicated(keep=False).sum())
    if combined.index.has_duplicates:
        combined = combined.groupby(level=0).mean()
    records_frame = pd.DataFrame(records)
    records_frame["duplicate_utc_hours_across_files"] = duplicate_utc_hours
    return combined, records_frame


def fill_eligible_short_gaps(
    raw: pd.DataFrame,
    max_gap_h: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    imputed = raw.isna()
    linear = raw.interpolate(
        method="time",
        limit=max_gap_h,
        limit_direction="both",
        limit_area="inside",
    )
    after_linear = linear.isna()
    # Remaining eligible gaps can only touch the UTC-year boundary.
    filled = linear.ffill(limit=max_gap_h).bfill(limit=max_gap_h)
    if filled.isna().any().any():
        raise RuntimeError("Eligible country-year retains unresolved values")
    linear_flags = imputed & ~after_linear
    boundary_flags = after_linear & ~filled.isna()
    counts = {
        f"{field}_linear_imputed_hours": int(linear_flags[field].sum())
        for field in MODEL_FIELDS
    }
    counts.update(
        {
            f"{field}_boundary_nearest_imputed_hours": int(
                boundary_flags[field].sum()
            )
            for field in MODEL_FIELDS
        }
    )
    return filled, imputed, counts


def field_dictionary() -> pd.DataFrame:
    definitions = {
        "timestamp_utc": (
            "Genuine UTC hour; leap-day hours are removed after source QA."
        ),
        "country": "ISO 3166-1 alpha-2 country code (GB for API code uk).",
        "country_name": "English country name.",
        "year": "UTC calendar year.",
        "is_current_eu27": "Country belongs to the current EU-27.",
        "physical_core_eligible": (
            "Country-year passes the 99% load/VRE coverage, <=24 h gap, "
            "and positive-energy contract."
        ),
        "expanded_europe_country": (
            "Country has at least three eligible years in 2015-2024."
        ),
        "balanced_eu_2015_2024_country": (
            "Current EU-27 country passes the physical contract in all ten "
            "target years."
        ),
        "source_energy_charts": "Always true; identifies the data provider.",
        "gas_is_response_only": (
            "Always true; aggregate Fossil gas is excluded from model inputs "
            "and may only be analysed as a system response."
        ),
        "load_mw": "Hourly mean national load after audited short-gap completion.",
        "wind_mw": (
            "Hourly mean onshore plus offshore wind generation after audited "
            "short-gap completion; available components are summed."
        ),
        "solar_mw": (
            "Hourly mean solar generation after audited short-gap completion."
        ),
        "load_imputed": "True where load_mw was absent before gap completion.",
        "wind_imputed": "True where wind_mw was absent before gap completion.",
        "solar_imputed": "True where solar_mw was absent before gap completion.",
    }
    units = {
        "timestamp_utc": "UTC",
        "year": "year",
        "load_mw": "MW",
        "wind_mw": "MW",
        "solar_mw": "MW",
    }
    return pd.DataFrame(
        [
            {
                "field": field,
                "definition": definition,
                "unit": units.get(field, "boolean/text"),
                "source_or_derivation": (
                    SOURCE_NAME
                    if field in {"load_mw", "wind_mw", "solar_mw"}
                    else "study-derived metadata/QA"
                ),
            }
            for field, definition in definitions.items()
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=list(TARGET_YEARS),
    )
    parser.add_argument("--coverage-threshold", type=float, default=MIN_COVERAGE)
    parser.add_argument("--max-gap-hours", type=int, default=MAX_GAP_H)
    parser.add_argument(
        "--min-expanded-years", type=int, default=MIN_EXPANDED_YEARS
    )
    parser.add_argument(
        "--output-name",
        default="energy_charts_europe_hourly_eligible_2015_2024.csv.gz",
    )
    args = parser.parse_args()

    years = tuple(sorted(set(args.years)))
    source_dir = args.root / "raw" / "energy_charts" / "public_power"
    processed_dir = args.root / "processed"
    research_dir = args.root / "artifacts" / "research"
    metadata_dir = args.root / "metadata"
    for directory in (processed_dir, research_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    by_country: dict[str, list[Path]] = {}
    for path in sorted(source_dir.glob("public_power_*_*.json")):
        match = RAW_PATTERN.search(path.name)
        if match and match.group(1) in COUNTRY_METADATA:
            by_country.setdefault(match.group(1), []).append(path)
    if not by_country:
        raise FileNotFoundError(f"No Energy-Charts files under {source_dir}")

    qa_rows: list[dict[str, Any]] = []
    file_rows: list[pd.DataFrame] = []
    candidate_groups: dict[tuple[str, int], dict[str, Any]] = {}
    for api_code, paths in sorted(by_country.items()):
        iso2, country_name, is_eu27 = COUNTRY_METADATA[api_code]
        combined, file_records = combine_country_files(paths)
        file_rows.append(file_records)
        downloaded_local_years = sorted(
            int(RAW_PATTERN.search(path.name).group(2)) for path in paths
        )
        for year in years:
            index = expected_utc_index(year)
            raw = combined.reindex(index)[list(MODEL_FIELDS)].astype(float)
            missing = raw.isna()
            vre = raw[["wind_mw", "solar_mw"]].sum(axis=1, min_count=2)
            load_coverage = float(raw["load_mw"].notna().mean())
            vre_coverage = float(vre.notna().mean())
            load_gap = longest_missing_run(raw["load_mw"])
            vre_gap = longest_missing_run(vre)
            annual_load = float(raw["load_mw"].sum(min_count=1))
            annual_vre = float(vre.sum(min_count=1))
            negative_load = int((raw["load_mw"] < 0).sum())
            negative_wind = int((raw["wind_mw"] < 0).sum())
            negative_solar = int((raw["solar_mw"] < 0).sum())
            eligible = bool(
                load_coverage >= args.coverage_threshold
                and vre_coverage >= args.coverage_threshold
                and load_gap <= args.max_gap_hours
                and vre_gap <= args.max_gap_hours
                and annual_load > 0
                and annual_vre > 0
                and negative_load == 0
                and negative_wind == 0
                and negative_solar == 0
            )
            reasons: list[str] = []
            if year not in downloaded_local_years:
                reasons.append("local_calendar_source_file_absent")
            if load_coverage < args.coverage_threshold:
                reasons.append("load_coverage_below_threshold")
            if vre_coverage < args.coverage_threshold:
                reasons.append("vre_coverage_below_threshold")
            if load_gap > args.max_gap_hours:
                reasons.append("load_gap_above_threshold")
            if vre_gap > args.max_gap_hours:
                reasons.append("vre_gap_above_threshold")
            if annual_load <= 0:
                reasons.append("nonpositive_annual_load")
            if annual_vre <= 0:
                reasons.append("nonpositive_annual_vre")
            if negative_load + negative_wind + negative_solar:
                reasons.append("negative_physical_input")

            imputation_counts = {
                f"{field}_linear_imputed_hours": 0 for field in MODEL_FIELDS
            }
            imputation_counts.update(
                {
                    f"{field}_boundary_nearest_imputed_hours": 0
                    for field in MODEL_FIELDS
                }
            )
            filled = None
            imputed = None
            if eligible:
                filled, imputed, imputation_counts = fill_eligible_short_gaps(
                    raw, args.max_gap_hours
                )
            leap = (index.month == 2) & (index.day == 29)
            qa_rows.append(
                {
                    "country": iso2,
                    "country_name": country_name,
                    "api_code": api_code,
                    "year": year,
                    "is_current_eu27": is_eu27,
                    "source_local_calendar_file_present": (
                        year in downloaded_local_years
                    ),
                    "expected_utc_hours_before_leap_day_removal": len(index),
                    "model_hours_after_leap_day_removal": int((~leap).sum()),
                    "load_observed_hours": int((~missing["load_mw"]).sum()),
                    "wind_observed_hours": int((~missing["wind_mw"]).sum()),
                    "solar_observed_hours": int((~missing["solar_mw"]).sum()),
                    "joint_vre_observed_hours": int(vre.notna().sum()),
                    "load_coverage": load_coverage,
                    "vre_coverage": vre_coverage,
                    "load_max_missing_gap_h": load_gap,
                    "vre_max_missing_gap_h": vre_gap,
                    "load_missing_hours": int(missing["load_mw"].sum()),
                    "wind_missing_hours": int(missing["wind_mw"].sum()),
                    "solar_missing_hours": int(missing["solar_mw"].sum()),
                    "negative_load_hours": negative_load,
                    "negative_wind_hours": negative_wind,
                    "negative_solar_hours": negative_solar,
                    "coverage_threshold": args.coverage_threshold,
                    "maximum_gap_threshold_h": args.max_gap_hours,
                    "physical_core_eligible": eligible,
                    "exclusion_reasons": ";".join(reasons),
                    **imputation_counts,
                }
            )
            if eligible and filled is not None and imputed is not None:
                candidate_groups[(iso2, year)] = {
                    "country_name": country_name,
                    "is_current_eu27": is_eu27,
                    "filled": filled.loc[~leap].copy(),
                    "imputed": imputed.loc[~leap].copy(),
                }

    qa = pd.DataFrame(qa_rows).sort_values(["country", "year"])
    local_audit_path = (
        research_dir / "energy_charts_country_year_coverage.csv"
    )
    if local_audit_path.exists():
        local_audit = pd.read_csv(local_audit_path)[
            [
                "country",
                "year",
                "physical_core_eligible",
                "load_coverage",
                "vre_coverage",
                "load_max_missing_gap_h",
                "vre_max_missing_gap_h",
            ]
        ].rename(
            columns={
                "physical_core_eligible": (
                    "source_local_calendar_physical_core_eligible"
                ),
                "load_coverage": "source_local_calendar_load_coverage",
                "vre_coverage": "source_local_calendar_vre_coverage",
                "load_max_missing_gap_h": (
                    "source_local_calendar_load_max_missing_gap_h"
                ),
                "vre_max_missing_gap_h": (
                    "source_local_calendar_vre_max_missing_gap_h"
                ),
            }
        )
        qa = qa.merge(
            local_audit,
            on=["country", "year"],
            how="left",
            validate="one_to_one",
        )
        qa["utc_vs_source_local_gate_disagreement"] = (
            qa["source_local_calendar_physical_core_eligible"].notna()
            & (
                qa["physical_core_eligible"]
                != qa[
                    "source_local_calendar_physical_core_eligible"
                ].astype(bool)
            )
        )
    else:
        qa["source_local_calendar_physical_core_eligible"] = np.nan
        qa["utc_vs_source_local_gate_disagreement"] = False
    eligible_counts = (
        qa.groupby("country")["physical_core_eligible"].sum().astype(int)
    )
    expanded = set(
        eligible_counts[
            eligible_counts >= args.min_expanded_years
        ].index.tolist()
    )
    balanced_eu = {
        country
        for country, count in eligible_counts.items()
        if count == len(years)
        and bool(
            qa.loc[qa["country"] == country, "is_current_eu27"].iloc[0]
        )
    }
    qa["eligible_years_for_country"] = qa["country"].map(eligible_counts)
    qa["expanded_europe_country"] = qa["country"].isin(expanded)
    qa["balanced_eu_2015_2024_country"] = qa["country"].isin(balanced_eu)

    panel_groups: list[pd.DataFrame] = []
    for (country, year), bundle in sorted(candidate_groups.items()):
        if country not in expanded:
            continue
        filled = bundle["filled"]
        imputed = bundle["imputed"]
        if len(filled) != 8760:
            raise RuntimeError(f"{country}-{year} has {len(filled)} model hours")
        panel_groups.append(
            pd.DataFrame(
                {
                    "timestamp_utc": filled.index,
                    "country": country,
                    "country_name": bundle["country_name"],
                    "year": year,
                    "is_current_eu27": bundle["is_current_eu27"],
                    "physical_core_eligible": True,
                    "expanded_europe_country": True,
                    "balanced_eu_2015_2024_country": (
                        country in balanced_eu
                    ),
                    "source_energy_charts": True,
                    "gas_is_response_only": True,
                    "load_mw": filled["load_mw"].to_numpy(float),
                    "wind_mw": filled["wind_mw"].to_numpy(float),
                    "solar_mw": filled["solar_mw"].to_numpy(float),
                    "load_imputed": imputed["load_mw"].to_numpy(bool),
                    "wind_imputed": imputed["wind_mw"].to_numpy(bool),
                    "solar_imputed": imputed["solar_mw"].to_numpy(bool),
                }
            )
        )
    if not panel_groups:
        raise RuntimeError("No country-years passed the physical-profile gate")
    panel = pd.concat(panel_groups, ignore_index=True).sort_values(
        ["country", "year", "timestamp_utc"]
    )
    if panel.duplicated(["country", "timestamp_utc"]).any():
        raise RuntimeError("Output contains duplicate country-UTC hours")
    if panel[list(MODEL_FIELDS)].isna().any().any():
        raise RuntimeError("Output retains missing physical inputs")
    if (panel[list(MODEL_FIELDS)] < 0).any().any():
        raise RuntimeError("Output contains negative physical inputs")

    output_path = processed_dir / args.output_name
    atomic_to_csv(
        panel,
        output_path,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        float_format="%.6f",
    )
    qa_path = research_dir / "energy_charts_panel_country_year_qa.csv"
    qa.to_csv(qa_path, index=False)
    source_files_path = metadata_dir / "energy_charts_panel_source_files.csv"
    pd.concat(file_rows, ignore_index=True).sort_values(
        ["api_code", "source_local_calendar_year"]
    ).to_csv(source_files_path, index=False)
    dictionary_path = research_dir / "energy_charts_panel_field_dictionary.csv"
    field_dictionary().to_csv(dictionary_path, index=False)

    inventory = (
        panel.groupby(
            [
                "country",
                "country_name",
                "is_current_eu27",
                "balanced_eu_2015_2024_country",
            ],
            as_index=False,
        )
        .agg(
            eligible_years=("year", "nunique"),
            first_year=("year", "min"),
            last_year=("year", "max"),
            country_hours=("timestamp_utc", "size"),
            imputed_load_hours=("load_imputed", "sum"),
            imputed_wind_hours=("wind_imputed", "sum"),
            imputed_solar_hours=("solar_imputed", "sum"),
        )
        .sort_values("country")
    )
    inventory_path = research_dir / "energy_charts_panel_country_inventory.csv"
    inventory.to_csv(inventory_path, index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "name": SOURCE_NAME,
            "endpoint": SOURCE_ENDPOINT,
            "api": SOURCE_API,
            "license": SOURCE_LICENSE,
            "gas_role": (
                "Fossil gas is excluded from the model-facing panel and may "
                "only be treated as an aggregate generation response."
            ),
        },
        "eligibility": {
            "target_years": list(years),
            "minimum_load_and_joint_vre_coverage": args.coverage_threshold,
            "maximum_consecutive_gap_hours": args.max_gap_hours,
            "minimum_eligible_years_for_expanded_europe": (
                args.min_expanded_years
            ),
            "interior_imputation": (
                "time-linear within the same country UTC calendar-year"
            ),
            "boundary_imputation": (
                "nearest observed value, <= maximum gap, same country-year"
            ),
            "leap_day": "removed after QA and imputation",
        },
        "sample": {
            "rows": int(len(panel)),
            "country_years": int(
                panel[["country", "year"]].drop_duplicates().shape[0]
            ),
            "countries": int(panel["country"].nunique()),
            "eu27_countries": int(
                panel.loc[panel["is_current_eu27"], "country"].nunique()
            ),
            "balanced_eu_countries": sorted(balanced_eu),
            "expanded_europe_countries": sorted(expanded),
        },
        "outputs": {
            "panel": {
                "path": str(output_path.relative_to(args.root)),
                "bytes": int(output_path.stat().st_size),
                "sha256": sha256(output_path),
            },
            "country_year_qa": str(qa_path.relative_to(args.root)),
            "country_inventory": str(inventory_path.relative_to(args.root)),
            "field_dictionary": str(dictionary_path.relative_to(args.root)),
            "source_files": str(source_files_path.relative_to(args.root)),
        },
    }
    manifest_path = metadata_dir / "energy_charts_panel_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["sample"], ensure_ascii=False, indent=2))
    print(output_path)


if __name__ == "__main__":
    main()
