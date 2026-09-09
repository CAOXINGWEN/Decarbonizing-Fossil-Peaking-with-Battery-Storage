#!/usr/bin/env python3
"""Build and audit the expanded European OPSD hourly-profile panel.

The source is the already archived, fixed Open Power System Data release
2020-10-06.  This script never alters that archive.  It keeps the reported
load, wind and solar series separate from a study-derived high-renewable
counterfactual in which annual wind-plus-solar energy is scaled to annual
load.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_VERSION = "2020-10-06"
SOURCE_DOI = "10.25832/time_series/2020-10-06"
SOURCE_PAGE = (
    "https://data.open-power-system-data.org/time_series/2020-10-06"
)
EXPECTED_SOURCE_BYTES = 130_339_665
EXPECTED_SOURCE_SHA256 = (
    "6a7f2bc571314cbf9c321cc03437691cd4be95c3a6f075e60ff99e8035c704c8"
)
YEARS = tuple(range(2015, 2020))
EXPLORATORY_MIN_COVERAGE = 0.99
EXPLORATORY_MAX_GAP_HOURS = 24
EXPLORATORY_MIN_EXPANDED_YEARS = 3
STRICT_MIN_COVERAGE = 0.995
STRICT_MAX_GAP_HOURS = 6
STRICT_MAX_ANNUAL_ANY_MISSING_HOURS = 24
STRICT_MIN_EXPANDED_YEARS = 3

CURRENT_EU27 = {
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

COUNTRIES = {
    "AT": {
        "name": "Austria",
        "load": "AT_load_actual_entsoe_transparency",
        "wind": "AT_wind_onshore_generation_actual",
        "wind_definition": "onshore wind",
        "solar": "AT_solar_generation_actual",
    },
    "BG": {
        "name": "Bulgaria",
        "load": "BG_load_actual_entsoe_transparency",
        "wind": "BG_wind_onshore_generation_actual",
        "wind_definition": "onshore wind",
        "solar": "BG_solar_generation_actual",
    },
    "CH": {
        "name": "Switzerland",
        "load": "CH_load_actual_entsoe_transparency",
        "wind": "CH_wind_onshore_generation_actual",
        "wind_definition": "onshore wind",
        "solar": "CH_solar_generation_actual",
    },
    "CZ": {
        "name": "Czechia",
        "load": "CZ_load_actual_entsoe_transparency",
        "wind": "CZ_wind_onshore_generation_actual",
        "wind_definition": "onshore wind",
        "solar": "CZ_solar_generation_actual",
    },
    "DE": {
        "name": "Germany",
        "load": "DE_load_actual_entsoe_transparency",
        "wind": "DE_wind_generation_actual",
        "wind_definition": "total wind",
        "solar": "DE_solar_generation_actual",
    },
    "DK": {
        "name": "Denmark",
        "load": "DK_load_actual_entsoe_transparency",
        "wind": "DK_wind_generation_actual",
        "wind_definition": "total wind",
        "solar": "DK_solar_generation_actual",
    },
    "EE": {
        "name": "Estonia",
        "load": "EE_load_actual_entsoe_transparency",
        "wind": "EE_wind_onshore_generation_actual",
        "wind_definition": "onshore wind",
        "solar": "EE_solar_generation_actual",
    },
    "ES": {
        "name": "Spain",
        "load": "ES_load_actual_entsoe_transparency",
        "wind": "ES_wind_onshore_generation_actual",
        "wind_definition": "onshore wind",
        "solar": "ES_solar_generation_actual",
    },
    "FR": {
        "name": "France",
        "load": "FR_load_actual_entsoe_transparency",
        "wind": "FR_wind_onshore_generation_actual",
        "wind_definition": "onshore wind",
        "solar": "FR_solar_generation_actual",
    },
    "GB": {
        "name": "United Kingdom",
        "load": "GB_UKM_load_actual_entsoe_transparency",
        "wind": "GB_UKM_wind_generation_actual",
        "wind_definition": "total wind, UK national aggregate",
        "solar": "GB_UKM_solar_generation_actual",
    },
    "GR": {
        "name": "Greece",
        "load": "GR_load_actual_entsoe_transparency",
        "wind": "GR_wind_onshore_generation_actual",
        "wind_definition": "onshore wind",
        "solar": "GR_solar_generation_actual",
    },
    "IT": {
        "name": "Italy",
        "load": "IT_load_actual_entsoe_transparency",
        "wind": "IT_wind_onshore_generation_actual",
        "wind_definition": "onshore wind",
        "solar": "IT_solar_generation_actual",
    },
    "RO": {
        "name": "Romania",
        "load": "RO_load_actual_entsoe_transparency",
        "wind": "RO_wind_onshore_generation_actual",
        "wind_definition": "onshore wind",
        "solar": "RO_solar_generation_actual",
    },
    "SI": {
        "name": "Slovenia",
        "load": "SI_load_actual_entsoe_transparency",
        "wind": "SI_wind_onshore_generation_actual",
        "wind_definition": "onshore wind",
        "solar": "SI_solar_generation_actual",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def longest_true_run(values: pd.Series | np.ndarray) -> int:
    best = run = 0
    for value in np.asarray(values, dtype=bool):
        run = run + 1 if value else 0
        best = max(best, run)
    return int(best)


def expected_index(year: int) -> pd.DatetimeIndex:
    """Return a uniform 8,760-hour UTC calendar; omit 29 February."""
    index = pd.date_range(
        f"{year}-01-01 00:00:00+00:00",
        f"{year}-12-31 23:00:00+00:00",
        freq="h",
    )
    return index[~((index.month == 2) & (index.day == 29))]


def file_record(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": int(path.stat().st_size),
        "sha256": sha256(path),
    }


def atomic_to_csv(
    frame: pd.DataFrame,
    path: Path,
    **kwargs: object,
) -> None:
    """Write a complete CSV beside its destination, then atomically replace."""
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, **kwargs)
    temporary.replace(path)


def main() -> None:
    upgrade_root = Path(__file__).resolve().parents[1]
    source_root = upgrade_root.parent / "opsd_time_series_2020-10-06"
    raw_path = source_root / "raw" / "time_series_60min_singleindex.csv"
    processed_dir = upgrade_root / "processed"
    research_dir = upgrade_root / "artifacts" / "research"
    processed_dir.mkdir(parents=True, exist_ok=True)
    research_dir.mkdir(parents=True, exist_ok=True)

    source_bytes = raw_path.stat().st_size
    source_hash = sha256(raw_path)
    if (
        source_bytes != EXPECTED_SOURCE_BYTES
        or source_hash != EXPECTED_SOURCE_SHA256
    ):
        raise RuntimeError(
            "The fixed OPSD source failed byte-size/SHA-256 verification: "
            f"{source_bytes} bytes, {source_hash}"
        )

    selected = ["utc_timestamp"]
    for specification in COUNTRIES.values():
        selected.extend(
            [
                specification["load"],
                specification["wind"],
                specification["solar"],
            ]
        )
    selected = list(dict.fromkeys(selected))
    header = list(pd.read_csv(raw_path, nrows=0).columns)
    absent = sorted(set(selected) - set(header))
    if absent:
        raise RuntimeError(f"Required OPSD columns are absent: {absent}")

    raw = pd.read_csv(
        raw_path,
        usecols=selected,
        parse_dates=["utc_timestamp"],
        low_memory=False,
    )
    raw["utc_timestamp"] = pd.to_datetime(raw["utc_timestamp"], utc=True)
    raw = raw.set_index("utc_timestamp").sort_index()
    duplicate_timestamps = int(raw.index.duplicated().sum())
    if duplicate_timestamps:
        raise RuntimeError(
            f"The OPSD UTC index contains {duplicate_timestamps} duplicates"
        )

    field_rows: list[dict[str, object]] = []
    for country, specification in COUNTRIES.items():
        for role in ("load", "wind", "solar"):
            field_rows.append(
                {
                    "country": country,
                    "country_name": specification["name"],
                    "current_eu27": country in CURRENT_EU27,
                    "variable_role": role,
                    "raw_column_name": specification[role],
                    "field_definition": (
                        specification["wind_definition"]
                        if role == "wind"
                        else (
                            "national actual load"
                            if role == "load"
                            else "solar generation"
                        )
                    ),
                    "unit": "MW",
                    "temporal_resolution": "60 min",
                    "official_release": SOURCE_VERSION,
                    "official_release_page": SOURCE_PAGE,
                    "doi": SOURCE_DOI,
                }
            )
    field_path = research_dir / "opsd_europe_field_dictionary.csv"
    pd.DataFrame(field_rows).to_csv(field_path, index=False)

    coverage_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    groups: dict[tuple[str, int], dict[str, object]] = {}

    for country, specification in COUNTRIES.items():
        source_columns = [
            specification["load"],
            specification["wind"],
            specification["solar"],
        ]
        role_by_column = dict(
            zip(source_columns, ("load", "wind", "solar"))
        )
        for year in YEARS:
            index = expected_index(year)
            source = raw.reindex(index)[source_columns].copy()
            source.columns = ["load", "wind", "solar"]
            missing = source.isna()
            any_missing = missing.any(axis=1)
            coverage = float((~any_missing).mean())
            longest_by_role = {
                role: longest_true_run(missing[role])
                for role in ("load", "wind", "solar")
            }
            longest_any = longest_true_run(any_missing)
            annual_any_missing_hours = int(any_missing.sum())
            exploratory_eligible = (
                coverage >= EXPLORATORY_MIN_COVERAGE
                and longest_any <= EXPLORATORY_MAX_GAP_HOURS
            )
            strict_confirmatory_eligible = (
                coverage >= STRICT_MIN_COVERAGE
                and longest_by_role["load"] <= STRICT_MAX_GAP_HOURS
                and longest_by_role["wind"] <= STRICT_MAX_GAP_HOURS
                and longest_by_role["solar"] <= STRICT_MAX_GAP_HOURS
                and longest_any <= STRICT_MAX_GAP_HOURS
                and annual_any_missing_hours
                <= STRICT_MAX_ANNUAL_ANY_MISSING_HOURS
            )

            for source_column, role in role_by_column.items():
                timestamps = source.index[source[role].isna()]
                missing_rows.extend(
                    {
                        "country": country,
                        "country_name": specification["name"],
                        "year": year,
                        "utc_timestamp": timestamp.isoformat(),
                        "variable_role": role,
                        "raw_column_name": source_column,
                    }
                    for timestamp in timestamps
                )

            filled = source.copy()
            if exploratory_eligible:
                # The input group contains exactly one country-year, so neither
                # interpolation nor nearest-value completion can cross a year
                # or country boundary.
                filled = filled.interpolate(
                    method="time",
                    limit=EXPLORATORY_MAX_GAP_HOURS,
                    limit_direction="both",
                )
            unresolved = int(filled.isna().sum().sum())
            if exploratory_eligible and unresolved:
                raise RuntimeError(
                    f"Eligible group {country}-{year} retained "
                    f"{unresolved} missing cells after interpolation"
                )

            coverage_rows.append(
                {
                    "country": country,
                    "country_name": specification["name"],
                    "current_eu27": country in CURRENT_EU27,
                    "year": year,
                    "expected_hours_after_leap_day_removal": len(index),
                    "load_observed_hours": int((~missing["load"]).sum()),
                    "wind_observed_hours": int((~missing["wind"]).sum()),
                    "solar_observed_hours": int((~missing["solar"]).sum()),
                    "complete_selected_hours": int((~any_missing).sum()),
                    "annual_any_selected_missing_hours": (
                        annual_any_missing_hours
                    ),
                    "complete_selected_coverage": coverage,
                    "complete_selected_coverage_pct": 100.0 * coverage,
                    "load_missing_hours": int(missing["load"].sum()),
                    "wind_missing_hours": int(missing["wind"].sum()),
                    "solar_missing_hours": int(missing["solar"].sum()),
                    "max_load_gap_hours": longest_by_role["load"],
                    "max_wind_gap_hours": longest_by_role["wind"],
                    "max_solar_gap_hours": longest_by_role["solar"],
                    "max_any_selected_gap_hours": longest_any,
                    "exploratory_coverage_threshold": (
                        EXPLORATORY_MIN_COVERAGE
                    ),
                    "exploratory_maximum_gap_threshold_hours": (
                        EXPLORATORY_MAX_GAP_HOURS
                    ),
                    "exploratory_eligible_country_year": (
                        exploratory_eligible
                    ),
                    # Backward-compatible alias.  It deliberately refers to
                    # the exploratory/expanded threshold, not to the strict
                    # confirmatory contract.
                    "eligible_country_year": exploratory_eligible,
                    "strict_coverage_threshold": STRICT_MIN_COVERAGE,
                    "strict_maximum_gap_threshold_hours": (
                        STRICT_MAX_GAP_HOURS
                    ),
                    "strict_maximum_annual_any_missing_hours": (
                        STRICT_MAX_ANNUAL_ANY_MISSING_HOURS
                    ),
                    "strict_confirmatory_eligible_country_year": (
                        strict_confirmatory_eligible
                    ),
                    "interpolation_rule": (
                        "time-linear, same country-year only, "
                        "exploratory-eligible groups only, maximum "
                        "missing run 24 h"
                    ),
                    "unresolved_cells_after_interpolation": unresolved,
                }
            )
            groups[(country, year)] = {
                "index": index,
                "raw": source,
                "filled": filled,
                "exploratory_eligible": exploratory_eligible,
                "strict_confirmatory_eligible": (
                    strict_confirmatory_eligible
                ),
            }

    coverage = pd.DataFrame(coverage_rows).sort_values(
        ["country", "year"]
    )
    exploratory_year_counts = (
        coverage.groupby("country")["exploratory_eligible_country_year"]
        .sum()
        .astype(int)
    )
    strict_year_counts = (
        coverage.groupby("country")[
            "strict_confirmatory_eligible_country_year"
        ]
        .sum()
        .astype(int)
    )
    exploratory_expanded_countries = set(
        exploratory_year_counts[
            exploratory_year_counts
            >= EXPLORATORY_MIN_EXPANDED_YEARS
        ].index
    )
    exploratory_balanced_eu_countries = {
        country
        for country, count in exploratory_year_counts.items()
        if count == len(YEARS) and country in CURRENT_EU27
    }
    exploratory_balanced_europe_countries = {
        country
        for country, count in exploratory_year_counts.items()
        if count == len(YEARS)
    }
    strict_expanded_countries = set(
        strict_year_counts[
            strict_year_counts >= STRICT_MIN_EXPANDED_YEARS
        ].index
    )
    strict_balanced_eu_countries = {
        country
        for country, count in strict_year_counts.items()
        if count == len(YEARS) and country in CURRENT_EU27
    }
    strict_balanced_europe_countries = {
        country
        for country, count in strict_year_counts.items()
        if count == len(YEARS)
    }

    coverage["exploratory_eligible_years_for_country"] = coverage[
        "country"
    ].map(exploratory_year_counts)
    coverage["strict_confirmatory_years_for_country"] = coverage[
        "country"
    ].map(strict_year_counts)
    coverage["exploratory_expanded_europe_country"] = coverage[
        "country"
    ].isin(exploratory_expanded_countries)
    coverage["exploratory_balanced_eu_country"] = coverage[
        "country"
    ].isin(exploratory_balanced_eu_countries)
    coverage["exploratory_balanced_europe_country"] = coverage[
        "country"
    ].isin(exploratory_balanced_europe_countries)
    coverage["strict_expanded_country"] = coverage["country"].isin(
        strict_expanded_countries
    )
    coverage["strict_balanced_eu_country"] = coverage["country"].isin(
        strict_balanced_eu_countries
    )
    coverage["strict_balanced_europe_country"] = coverage["country"].isin(
        strict_balanced_europe_countries
    )
    # Retain the original names as compatibility aliases, while the explicit
    # exploratory/strict names above determine interpretation.
    coverage["eligible_years_for_country"] = coverage["country"].map(
        exploratory_year_counts
    )
    coverage["expanded_europe_country"] = coverage["country"].isin(
        exploratory_expanded_countries
    )
    coverage["balanced_eu_primary_country"] = coverage["country"].isin(
        exploratory_balanced_eu_countries
    )
    coverage["balanced_europe_country"] = coverage["country"].isin(
        exploratory_balanced_europe_countries
    )
    coverage_path = (
        research_dir / "opsd_europe_country_year_coverage.csv"
    )
    coverage.to_csv(coverage_path, index=False)

    missing_path = research_dir / "opsd_europe_missing_hours.csv.gz"
    atomic_to_csv(
        pd.DataFrame(missing_rows).sort_values(
            ["country", "year", "utc_timestamp", "variable_role"]
        ),
        missing_path,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )

    sample_rows = []
    for country, specification in COUNTRIES.items():
        exploratory_years = coverage.loc[
            (coverage["country"] == country)
            & coverage["exploratory_eligible_country_year"],
            "year",
        ].astype(int).tolist()
        strict_years = coverage.loc[
            (coverage["country"] == country)
            & coverage["strict_confirmatory_eligible_country_year"],
            "year",
        ].astype(int).tolist()
        sample_rows.append(
            {
                "country": country,
                "country_name": specification["name"],
                "current_eu27": country in CURRENT_EU27,
                "exploratory_eligible_year_count": len(
                    exploratory_years
                ),
                "exploratory_eligible_years": " ".join(
                    map(str, exploratory_years)
                ),
                "exploratory_balanced_eu_country": (
                    country in exploratory_balanced_eu_countries
                ),
                "exploratory_balanced_europe_country": (
                    country in exploratory_balanced_europe_countries
                ),
                "exploratory_expanded_europe_country": (
                    country in exploratory_expanded_countries
                ),
                "strict_confirmatory_year_count": len(strict_years),
                "strict_confirmatory_years": " ".join(
                    map(str, strict_years)
                ),
                "strict_balanced_eu_country": (
                    country in strict_balanced_eu_countries
                ),
                "strict_balanced_europe_country": (
                    country in strict_balanced_europe_countries
                ),
                "strict_expanded_country": (
                    country in strict_expanded_countries
                ),
                # Backward-compatible aliases for the exploratory sample.
                "eligible_year_count": len(exploratory_years),
                "eligible_years": " ".join(map(str, exploratory_years)),
                "balanced_eu_primary_country": (
                    country in exploratory_balanced_eu_countries
                ),
                "balanced_europe_country": (
                    country in exploratory_balanced_europe_countries
                ),
                "expanded_europe_country": (
                    country in exploratory_expanded_countries
                ),
            }
        )
    sample_path = (
        research_dir / "opsd_europe_sample_definition.csv"
    )
    pd.DataFrame(sample_rows).sort_values("country").to_csv(
        sample_path, index=False
    )

    panel_groups: list[pd.DataFrame] = []
    annual_checks: list[dict[str, object]] = []
    for (country, year), bundle in groups.items():
        if (
            country not in exploratory_expanded_countries
            or not bool(bundle["exploratory_eligible"])
        ):
            continue
        source = bundle["raw"]
        filled = bundle["filled"]
        index = bundle["index"]
        load = filled["load"].to_numpy(float)
        wind = filled["wind"].to_numpy(float)
        solar = filled["solar"].to_numpy(float)
        observed_vre = wind + solar
        if observed_vre.sum() <= 0:
            raise RuntimeError(f"Non-positive annual VRE for {country}-{year}")
        scale = float(load.sum() / observed_vre.sum())
        scaled_vre = scale * observed_vre
        net_load = load - scaled_vre
        residual = np.maximum(net_load, 0.0)
        surplus = np.maximum(-net_load, 0.0)
        mean_load = float(load.mean())

        panel_groups.append(
            pd.DataFrame(
                {
                    "utc_timestamp": index,
                    "country": country,
                    "country_name": COUNTRIES[country]["name"],
                    "year": year,
                    "current_eu27": country in CURRENT_EU27,
                    "exploratory_balanced_eu_country": (
                        country in exploratory_balanced_eu_countries
                    ),
                    "exploratory_balanced_europe_country": (
                        country in exploratory_balanced_europe_countries
                    ),
                    "exploratory_expanded_europe_country": True,
                    "exploratory_eligible_country_year": True,
                    "strict_confirmatory_eligible_country_year": bool(
                        bundle["strict_confirmatory_eligible"]
                    ),
                    "strict_balanced_eu_country": (
                        country in strict_balanced_eu_countries
                    ),
                    "strict_balanced_europe_country": (
                        country in strict_balanced_europe_countries
                    ),
                    "strict_expanded_country": (
                        country in strict_expanded_countries
                    ),
                    # Backward-compatible exploratory aliases.
                    "balanced_eu_primary_country": (
                        country in exploratory_balanced_eu_countries
                    ),
                    "balanced_europe_country": (
                        country in exploratory_balanced_europe_countries
                    ),
                    "expanded_europe_country": True,
                    "eligible_country_year": True,
                    "load_mw_raw": source["load"].to_numpy(float),
                    "wind_mw_raw": source["wind"].to_numpy(float),
                    "solar_mw_raw": source["solar"].to_numpy(float),
                    "load_mw": load,
                    "wind_mw": wind,
                    "solar_mw": solar,
                    "load_imputed": source["load"].isna().to_numpy(),
                    "wind_imputed": source["wind"].isna().to_numpy(),
                    "solar_imputed": source["solar"].isna().to_numpy(),
                    "vre_observed_mw": observed_vre,
                    "vre_scale_factor": scale,
                    "scaled_vre_mw": scaled_vre,
                    "net_load_high_vre_mw": net_load,
                    "residual_mw": residual,
                    "surplus_mw": surplus,
                    "load_pu_mean": load / mean_load,
                    "residual_pu_mean": residual / mean_load,
                    "surplus_pu_mean": surplus / mean_load,
                }
            )
        )
        annual_checks.append(
            {
                "country": country,
                "year": year,
                "hours": len(index),
                "annual_load_mwh": float(load.sum()),
                "annual_observed_vre_mwh": float(observed_vre.sum()),
                "vre_scale_factor": scale,
                "annual_scaled_vre_mwh": float(scaled_vre.sum()),
                "scaled_vre_minus_load_mwh": float(
                    scaled_vre.sum() - load.sum()
                ),
                "minimum_load_mw": float(load.min()),
                "minimum_wind_mw": float(wind.min()),
                "minimum_solar_mw": float(solar.min()),
                "unresolved_cells": int(
                    pd.DataFrame(
                        {"load": load, "wind": wind, "solar": solar}
                    ).isna().sum().sum()
                ),
            }
        )

    panel = pd.concat(panel_groups, ignore_index=True)
    panel = panel.sort_values(
        ["country", "year", "utc_timestamp"]
    ).reset_index(drop=True)
    panel_path = (
        processed_dir
        / "opsd_europe_hourly_eligible_country_years_2015_2019.csv.gz"
    )
    atomic_to_csv(
        panel,
        panel_path,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )

    annual_check_path = (
        research_dir / "opsd_europe_annual_energy_checks.csv"
    )
    annual_checks_frame = pd.DataFrame(annual_checks).sort_values(
        ["country", "year"]
    )
    annual_checks_frame.to_csv(annual_check_path, index=False)

    country_years = (
        panel[["country", "year"]]
        .drop_duplicates()
        .sort_values(["country", "year"])
    )
    expected_panel_rows = len(country_years) * 8760
    maximum_energy_imbalance = float(
        annual_checks_frame["scaled_vre_minus_load_mwh"].abs().max()
    )
    qa = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "release": SOURCE_VERSION,
            "doi": SOURCE_DOI,
            "path": str(raw_path.relative_to(upgrade_root.parent.parent)),
            "bytes": source_bytes,
            "sha256": source_hash,
            "duplicate_utc_timestamps": duplicate_timestamps,
        },
        "eligibility": {
            "years": list(YEARS),
            "leap_day_rule": (
                "Remove 29 February so every retained country-year has "
                "8,760 UTC hours"
            ),
            "exploratory_expanded_contract": {
                "purpose": (
                    "Expanded/exploratory heterogeneity and sensitivity "
                    "analysis; not the confirmatory primary contract"
                ),
                "minimum_complete_selected_coverage": (
                    EXPLORATORY_MIN_COVERAGE
                ),
                "maximum_any_selected_missing_run_hours": (
                    EXPLORATORY_MAX_GAP_HOURS
                ),
                "minimum_eligible_years_for_expanded_europe": (
                    EXPLORATORY_MIN_EXPANDED_YEARS
                ),
            },
            "strict_confirmatory_contract": {
                "purpose": "Strict confirmatory country-year screen",
                "minimum_complete_selected_coverage": (
                    STRICT_MIN_COVERAGE
                ),
                "maximum_missing_run_for_each_variable_hours": (
                    STRICT_MAX_GAP_HOURS
                ),
                "maximum_any_selected_missing_run_hours": (
                    STRICT_MAX_GAP_HOURS
                ),
                "maximum_annual_any_selected_missing_hours": (
                    STRICT_MAX_ANNUAL_ANY_MISSING_HOURS
                ),
                "minimum_strict_years_for_expanded_country": (
                    STRICT_MIN_EXPANDED_YEARS
                ),
            },
            "interpolation": (
                "Time-linear within an exploratory-eligible country-year "
                "only; no country or year boundary is crossed"
            ),
        },
        "sample": {
            "candidate_countries": sorted(COUNTRIES),
            "candidate_country_count": len(COUNTRIES),
            "candidate_country_year_count": len(COUNTRIES) * len(YEARS),
            "exploratory_expanded_europe_countries": sorted(
                exploratory_expanded_countries
            ),
            "exploratory_expanded_europe_country_count": len(
                exploratory_expanded_countries
            ),
            "exploratory_eligible_country_year_count": len(country_years),
            "exploratory_balanced_eu_countries": sorted(
                exploratory_balanced_eu_countries
            ),
            "exploratory_balanced_eu_country_count": len(
                exploratory_balanced_eu_countries
            ),
            "exploratory_balanced_eu_country_year_count": (
                len(exploratory_balanced_eu_countries) * len(YEARS)
            ),
            "exploratory_balanced_europe_countries": sorted(
                exploratory_balanced_europe_countries
            ),
            "strict_confirmatory_country_year_count": int(
                coverage[
                    "strict_confirmatory_eligible_country_year"
                ].sum()
            ),
            "strict_expanded_countries": sorted(
                strict_expanded_countries
            ),
            "strict_expanded_country_count": len(
                strict_expanded_countries
            ),
            "strict_expanded_country_year_count": int(
                coverage[
                    coverage["country"].isin(strict_expanded_countries)
                ]["strict_confirmatory_eligible_country_year"].sum()
            ),
            "strict_balanced_eu_countries": sorted(
                strict_balanced_eu_countries
            ),
            "strict_balanced_eu_country_count": len(
                strict_balanced_eu_countries
            ),
            "strict_balanced_eu_country_year_count": (
                len(strict_balanced_eu_countries) * len(YEARS)
            ),
            "strict_balanced_europe_countries": sorted(
                strict_balanced_europe_countries
            ),
            # Backward-compatible aliases for the exploratory sample.
            "expanded_europe_countries": sorted(
                exploratory_expanded_countries
            ),
            "expanded_europe_country_count": len(
                exploratory_expanded_countries
            ),
            "eligible_country_year_count": len(country_years),
            "balanced_eu_primary_countries": sorted(
                exploratory_balanced_eu_countries
            ),
            "balanced_eu_primary_country_count": len(
                exploratory_balanced_eu_countries
            ),
            "balanced_eu_primary_country_year_count": (
                len(exploratory_balanced_eu_countries) * len(YEARS)
            ),
            "balanced_europe_countries": sorted(
                exploratory_balanced_europe_countries
            ),
        },
        "panel": {
            "rows": len(panel),
            "expected_rows": expected_panel_rows,
            "all_country_years_have_8760_hours": bool(
                country_years.merge(
                    panel.groupby(["country", "year"]).size().rename("n"),
                    on=["country", "year"],
                )["n"].eq(8760).all()
            ),
            "unresolved_filled_cells": int(
                panel[["load_mw", "wind_mw", "solar_mw"]]
                .isna()
                .sum()
                .sum()
            ),
            "maximum_absolute_annual_scaled_vre_minus_load_mwh": (
                maximum_energy_imbalance
            ),
            "observed_and_counterfactual_separated": True,
            "counterfactual_definition": (
                "Within each country-year, multiply observed wind and "
                "solar by one common factor so annual wind-plus-solar "
                "energy equals annual load; this is a high-VRE "
                "counterfactual, not an observed or forecast future."
            ),
        },
    }
    if len(panel) != expected_panel_rows:
        raise RuntimeError(
            f"Panel row mismatch: {len(panel)} != {expected_panel_rows}"
        )
    if qa["panel"]["unresolved_filled_cells"] != 0:
        raise RuntimeError("Processed panel contains unresolved missing values")
    if maximum_energy_imbalance > 1e-6:
        raise RuntimeError(
            "Annual scaled VRE and load do not balance within tolerance: "
            f"{maximum_energy_imbalance} MWh"
        )

    qa_path = research_dir / "opsd_europe_qa.json"
    qa_path.write_text(
        json.dumps(qa, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest_paths = [
        upgrade_root / "README.md",
        field_path,
        coverage_path,
        missing_path,
        sample_path,
        annual_check_path,
        panel_path,
        qa_path,
        Path(__file__).resolve(),
    ]
    manifest = pd.DataFrame(
        [file_record(path, upgrade_root) for path in manifest_paths]
    ).sort_values("path")
    manifest.to_csv(
        research_dir / "opsd_europe_file_manifest.csv", index=False
    )

    print(json.dumps(qa["sample"], ensure_ascii=False, indent=2))
    print(json.dumps(qa["panel"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
