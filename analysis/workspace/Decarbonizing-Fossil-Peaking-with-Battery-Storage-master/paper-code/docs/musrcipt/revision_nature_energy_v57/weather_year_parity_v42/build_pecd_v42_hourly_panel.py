#!/usr/bin/env python3
"""Build a 1950--2025 PECD v4.2 wind--solar--demand hourly panel.

PECD v4.2 supplies all renewable capacity factors and population-weighted
temperature. A pooled hourly demand-shape model is calibrated on observed
European public-grid load (2015--2024). The model removes each country-year's
annual mean before fitting, so it learns chronology rather than demand growth.
Fixed 2030 geographic load weights come from the verified ERAA 2025 DemandData
archive. No PECD 2021.3 hourly demand sequence enters the v56 panel.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import io
import json
import math
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge


HERE = Path(__file__).resolve().parent
EXTRACTED = HERE / "extracted"
PROCESSED = HERE / "processed"

EUROPE33 = [
    "AL", "AT", "BA", "BE", "BG", "CH", "CZ", "DE", "DK", "EE", "ES",
    "FI", "FR", "GR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "ME",
    "MK", "NL", "NO", "PL", "PT", "RO", "RS", "SE", "SI", "SK", "UK",
]
ALIASES = {"GB": "UK", "EL": "GR"}

PRODUCT_DIRS = {
    "onshore": "wind_onshore_tech35_regrb_peon_fv1",
    "offshore": "wind_offshore_tech21_regrb_peof_fv1",
    "solar": "solar_utility_fixed_tech62_peon_fv1",
    "temperature_fv2": "population_weighted_temperature_szon_fv2",
    "temperature_fv1": "population_weighted_temperature_szon_fv1",
}

PRODUCT_YEAR_RANGES = {
    "wind_onshore_tech35_regrb_peon_fv1": (1950, 2025),
    "wind_offshore_tech21_regrb_peof_fv1": (1950, 2025),
    "solar_utility_fixed_tech62_peon_fv1": (1950, 2025),
    "population_weighted_temperature_szon_fv2": (1950, 2021),
    "population_weighted_temperature_szon_fv1": (2022, 2025),
}


def workspace() -> Path:
    for parent in HERE.parents:
        if (parent / "dataset" / "external_hourly").is_dir():
            return parent
    raise RuntimeError("Cannot locate workspace dataset directory")


WORKSPACE = workspace()
DATA_ROOT = WORKSPACE / "dataset" / "external_hourly" / "europe_mid_duration_valley_upgrade_v0_1"
OBSERVED_LOAD = DATA_ROOT / "processed" / "energy_charts_europe_hourly_eligible_2015_2024.csv.gz"
ERAA_DEMAND = DATA_ROOT / "raw" / "eraa_2025" / "ERAA_2025_DemandData.zip"
ERAA_DEMAND_QA = DATA_ROOT / "raw" / "eraa_2025" / "ERAA_2025_DemandData.full_archive_qa.json"
ERAA_DASHBOARD = DATA_ROOT / "raw" / "eraa_2025" / "ERAA_2025_Dashboard_RawData.zip"
ERAA_CAPACITY_MEMBER = "ERAA_2025_DashboardRawData1/Dashboard Raw Data/GenerationCapacities.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def canonical_country(zone: str) -> str | None:
    match = re.match(r"^([A-Za-z]{2})", str(zone).strip())
    if not match:
        return None
    country = ALIASES.get(match.group(1).upper(), match.group(1).upper())
    return country if country in EUROPE33 else None


def canonical_zone(value: str) -> str | None:
    text = str(value).strip().upper()
    match = re.search(r"(?:^|[^A-Z0-9])([A-Z]{2}[A-Z0-9]{2,3})(?:$|[^A-Z0-9])", text)
    if match is None:
        return None
    zone = match.group(1)
    if zone.startswith("GB"):
        zone = "UK" + zone[2:]
    return zone


ZONE_WEIGHT_CROSSWALK = {
    ("LU", "LU00"): ("LUG1",),
    ("NO", "NOS0"): ("NOS1", "NOS2", "NOS3"),
}


def structural_weight_for_zone(
    zone_weights: pd.DataFrame,
    country: str,
    zone: str | None,
) -> float:
    if zone is None:
        return 0.0
    aliases = ZONE_WEIGHT_CROSSWALK.get((country, zone), (zone,))
    return float(
        zone_weights.loc[
            zone_weights.country.eq(country) & zone_weights.zone.isin(aliases),
            "weight_mw",
        ].sum()
    )


def column_weights_for_country(
    zone_weights: pd.DataFrame,
    country: str,
    columns: list[str],
) -> tuple[dict[str, float], str, float, float]:
    country_weights = zone_weights.loc[zone_weights.country.eq(country), ["zone", "weight_mw"]]
    country_total_weight = float(country_weights.weight_mw.sum())
    if country_total_weight <= 0.0:
        return {column: 1.0 for column in columns}, "unweighted_zero_structural_weight", 0.0, 0.0

    exact = {
        column: structural_weight_for_zone(zone_weights, country, canonical_zone(str(column)))
        for column in columns
    }
    exact_total = float(sum(exact.values()))
    residual = max(country_total_weight - exact_total, 0.0)
    distributed = residual / len(columns) if columns else 0.0
    weights = {column: exact[column] + distributed for column in columns}
    assigned = float(sum(weights.values()))
    if assigned <= 0.0:
        return {column: 1.0 for column in columns}, "unweighted_fallback", 0.0, country_total_weight
    if exact_total <= 0.0 and residual > 0.0:
        method = "country_total_distributed_to_pecd_regions"
    elif residual > 1.0e-9:
        method = "matched_zone_weighted_plus_country_residual"
    else:
        method = "matched_zone_weighted"
    return weights, method, min(assigned, country_total_weight), country_total_weight


def parse_timestamp(frame: pd.DataFrame, path: Path) -> tuple[pd.DatetimeIndex, str]:
    candidates = [
        column for column in frame.columns
        if str(column).strip().lower() in {"date", "time", "timestamp", "datetime"}
    ]
    if not candidates:
        candidates = [str(frame.columns[0])]
    column = candidates[0]
    timestamp = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if timestamp.isna().any():
        raise RuntimeError(f"Cannot parse all timestamps in {path}")
    return pd.DatetimeIndex(timestamp), column


def read_pecd_csv(path: Path) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    frame = pd.read_csv(path, comment="#")
    timestamp, date_column = parse_timestamp(frame, path)
    keep = ~((timestamp.month == 2) & (timestamp.day == 29))
    timestamp = timestamp[keep]
    values = frame.loc[keep, frame.columns != date_column].copy()
    usable = []
    for column in values.columns:
        if canonical_country(str(column)) is None:
            continue
        converted = pd.to_numeric(values[column], errors="coerce")
        if converted.notna().all():
            values[column] = converted.astype(np.float32)
            usable.append(column)
    if not usable:
        raise RuntimeError(f"No PECD regional value columns found in {path}")
    values = values[usable]
    if timestamp.duplicated().any():
        raise RuntimeError(f"Duplicate timestamps in {path}")
    return timestamp, values


def aggregate_file_to_country(
    path: Path,
    zone_weights: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    timestamp, values = read_pecd_csv(path)
    output = pd.DataFrame({"timestamp_utc": timestamp})
    inventory = []
    for country in EUROPE33:
        columns = [column for column in values if canonical_country(str(column)) == country]
        if not columns:
            continue
        weights, method, matched_weight, country_total_weight = column_weights_for_country(
            zone_weights, country, columns
        )
        ordered = list(weights)
        current_weights = np.asarray([weights[column] for column in ordered], dtype=float)
        current_weights /= current_weights.sum()
        output[country] = values[ordered].to_numpy(float).dot(current_weights).astype(np.float32)
        inventory.append({
            "source_file": path.name,
            "country": country,
            "regional_columns": len(columns),
            "regions": "|".join(map(str, columns)),
            "aggregation_method": method,
            "matched_weight_mw": matched_weight,
            "country_total_weight_mw": country_total_weight,
            "matched_weight_share": matched_weight / country_total_weight if country_total_weight > 0.0 else 1.0,
        })
    output["calendar_year"] = output.timestamp_utc.dt.year.astype(np.int16)
    return output, inventory


def consolidate_product(product_id: str, zone_weights: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = EXTRACTED / product_id
    files = sorted(root.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No extracted CSV files for {product_id}; run the downloader first")
    frames = []
    inventory_rows = []
    for index, path in enumerate(files, start=1):
        frame, inventory = aggregate_file_to_country(path, zone_weights)
        frames.append(frame)
        inventory_rows.extend(inventory)
        print(f"{product_id}: parsed {index}/{len(files)} {path.name}", flush=True)
    combined = pd.concat(frames, ignore_index=True)
    value_columns = [column for column in combined if column not in {"timestamp_utc", "calendar_year"}]
    combined = combined.groupby(["timestamp_utc", "calendar_year"], as_index=False)[value_columns].mean()
    combined = combined.sort_values("timestamp_utc").reset_index(drop=True)
    counts = combined.groupby("calendar_year").size()
    first_year, last_year = PRODUCT_YEAR_RANGES[product_id]
    expected_years = set(range(first_year, last_year + 1))
    if set(counts.index) != expected_years or not counts.eq(8760).all():
        raise RuntimeError(
            f"{product_id} calendar coverage is not {first_year}--{last_year} "
            f"with 8,760 h per year: {counts.to_dict()}"
        )
    return combined, pd.DataFrame(inventory_rows)


def consolidate_inputs() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    load_zone_weights = read_eraa_zone_load_weights().rename(columns={"mean_load_mw": "weight_mw"})
    vre_zone_weights = read_eraa_vre_zone_weights()
    weights_by_product = {
        "onshore": vre_zone_weights.loc[vre_zone_weights.technology.eq("onshore"), ["country", "zone", "capacity_mw"]].rename(columns={"capacity_mw": "weight_mw"}),
        "offshore": vre_zone_weights.loc[vre_zone_weights.technology.eq("offshore"), ["country", "zone", "capacity_mw"]].rename(columns={"capacity_mw": "weight_mw"}),
        "solar": vre_zone_weights.loc[vre_zone_weights.technology.eq("solar"), ["country", "zone", "capacity_mw"]].rename(columns={"capacity_mw": "weight_mw"}),
        "temperature_fv2": load_zone_weights,
        "temperature_fv1": load_zone_weights,
    }
    inventory = []
    for key in ("onshore", "offshore", "solar", "temperature_fv2", "temperature_fv1"):
        product_id = PRODUCT_DIRS[key]
        frame, current_inventory = consolidate_product(product_id, weights_by_product[key])
        frame.to_parquet(PROCESSED / f"{key}_country_hourly.parquet", index=False, compression="zstd")
        current_inventory.insert(0, "product", key)
        inventory.append(current_inventory)

    early = pd.read_parquet(PROCESSED / "temperature_fv2_country_hourly.parquet")
    late = pd.read_parquet(PROCESSED / "temperature_fv1_country_hourly.parquet")
    temperature = pd.concat([early, late], ignore_index=True).sort_values("timestamp_utc")
    if temperature.timestamp_utc.duplicated().any():
        raise RuntimeError("Temperature file revisions overlap")
    counts = temperature.groupby("calendar_year").size()
    if set(counts.index) != set(range(1950, 2026)) or not counts.eq(8760).all():
        raise RuntimeError("Combined PECD v4.2 temperature coverage failed")
    temperature.to_parquet(
        PROCESSED / "temperature_country_hourly.parquet", index=False, compression="zstd"
    )
    inventory_frame = pd.concat(inventory, ignore_index=True)
    inventory_frame.to_csv(PROCESSED / "pecd_v42_zone_inventory.csv", index=False)
    coverage_rows = []
    failed_countries = []
    for product, group in inventory_frame.groupby("product", sort=True):
        first_file = sorted(group.source_file.unique())[0]
        current = group[group.source_file.eq(first_file)]
        expected = (
            weights_by_product[product].groupby("country", as_index=False).weight_mw.sum()
            .rename(columns={"weight_mw": "expected_weight_mw"})
        )
        country_coverage = expected.merge(
            current[["country", "matched_weight_mw"]], on="country", how="left"
        )
        country_coverage["matched_weight_mw"] = country_coverage.matched_weight_mw.fillna(0.0)
        country_coverage["matched_weight_share"] = np.where(
            country_coverage.expected_weight_mw.gt(0.0),
            np.minimum(country_coverage.matched_weight_mw, country_coverage.expected_weight_mw)
            / country_coverage.expected_weight_mw,
            1.0,
        )
        current_failed = country_coverage.loc[
            country_coverage.expected_weight_mw.gt(0.0)
            & country_coverage.matched_weight_share.lt(0.95)
        ].copy()
        if not current_failed.empty:
            current_failed.insert(0, "product", product)
            failed_countries.append(current_failed)
        denominator = float(country_coverage.expected_weight_mw.sum())
        numerator = float(
            np.minimum(country_coverage.matched_weight_mw, country_coverage.expected_weight_mw).sum()
        )
        coverage_rows.append({
            "product": product,
            "audit_source_file": first_file,
            "matched_weight_mw": numerator,
            "total_weight_mw": denominator,
            "matched_weight_share": numerator / denominator if denominator > 0.0 else 1.0,
            "minimum_positive_country_weight_share": float(
                country_coverage.loc[
                    country_coverage.expected_weight_mw.gt(0.0), "matched_weight_share"
                ].min()
            ),
            "countries_below_95_percent": "|".join(current_failed.country.tolist()),
        })
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(PROCESSED / "pecd_v42_zone_weight_matching_qa.csv", index=False)
    failure_path = PROCESSED / "pecd_v42_zone_weight_matching_failures.csv"
    if failed_countries:
        failed = pd.concat(failed_countries, ignore_index=True)
        failed.to_csv(failure_path, index=False)
        raise RuntimeError(
            "PECD v4.2 zone matching retained less than 95% of structural weight "
            f"for one or more countries:\n{failed}"
        )
    # Refresh the ledger on successful reruns so an earlier failed attempt
    # cannot leave a stale non-empty failure file beside a passing QA table.
    pd.DataFrame(
        columns=[
            "product", "country", "expected_weight_mw", "matched_weight_mw",
            "matched_weight_share",
        ]
    ).to_csv(failure_path, index=False)


@functools.lru_cache(maxsize=1)
def read_eraa_zone_load_weights() -> pd.DataFrame:
    qa = json.loads(ERAA_DEMAND_QA.read_text(encoding="utf-8"))
    if not qa.get("zip_integrity_pass") or sha256(ERAA_DEMAND) != qa["archive_sha256"]:
        raise RuntimeError("ERAA 2025 DemandData archive failed source verification")
    pattern = re.compile(r"/([A-Z0-9]+)_Demand_total_2030_National Trends\.csv$")
    rows = []
    with zipfile.ZipFile(ERAA_DEMAND) as archive:
        names = [name for name in archive.namelist() if pattern.search(name)]
        for name in names:
            zone = pattern.search(name).group(1)
            country = canonical_country(zone)
            if country is None:
                continue
            frame = pd.read_csv(io.BytesIO(archive.read(name)))
            weather_columns = [column for column in frame if re.fullmatch(r"WS\d{2}", str(column))]
            if len(frame) != 8760 or len(weather_columns) != 36:
                raise RuntimeError(f"Unexpected ERAA demand shape in {name}")
            rows.append({
                "country": country,
                "zone": canonical_zone(zone),
                "mean_load_mw": float(frame[weather_columns].to_numpy(float).mean()),
            })
    zone = pd.DataFrame(rows).groupby(["country", "zone"], as_index=False).mean_load_mw.sum()
    if zone.zone.isna().any():
        raise RuntimeError("ERAA load zone normalization failed")
    return zone


def read_eraa_load_weights() -> pd.DataFrame:
    zone = read_eraa_zone_load_weights().copy()
    country = zone.groupby("country", as_index=False).agg(
        mean_2030_load_mw=("mean_load_mw", "sum"),
        demand_zones=("zone", "nunique"),
    )
    country = country[country.country.isin(EUROPE33)].copy()
    missing = sorted(set(EUROPE33) - set(country.country))
    if missing:
        raise RuntimeError(f"ERAA 2030 load weights missing for {missing}")
    country["load_share"] = country.mean_2030_load_mw / country.mean_2030_load_mw.sum()
    country.to_csv(PROCESSED / "eraa2025_2030_country_load_weights.csv", index=False)
    return country


@functools.lru_cache(maxsize=1)
def read_eraa_vre_zone_weights() -> pd.DataFrame:
    if not ERAA_DASHBOARD.exists():
        raise FileNotFoundError(f"ERAA 2025 dashboard archive missing: {ERAA_DASHBOARD}")
    with zipfile.ZipFile(ERAA_DASHBOARD) as archive:
        frame = pd.read_csv(io.BytesIO(archive.read(ERAA_CAPACITY_MEMBER)))
    frame = frame[
        frame.data_version.eq("ERAA 2025 final")
        & pd.to_numeric(frame["Target year"], errors="coerce").eq(2030)
    ].copy()
    frame["capacity_mw"] = pd.to_numeric(frame.Value, errors="coerce")
    frame["zone"] = frame.Market_Node.map(canonical_zone)
    frame["country"] = frame.zone.map(canonical_country)
    technology = frame.Technology.astype(str)
    frame["technology"] = np.select(
        [
            technology.eq("Wind onshore"),
            technology.str.startswith("Wind offshore"),
            technology.str.startswith("Solar PV"),
        ],
        ["onshore", "offshore", "solar"],
        default="",
    )
    frame = frame[
        frame.country.isin(EUROPE33)
        & frame.technology.ne("")
        & frame.capacity_mw.notna()
        & frame.capacity_mw.ge(0.0)
    ].copy()
    zone = frame.groupby(["country", "zone", "technology"], as_index=False).capacity_mw.sum()
    missing = sorted(set(EUROPE33) - set(zone.country))
    if missing:
        raise RuntimeError(f"ERAA 2025 final 2030 VRE capacities missing for {missing}")
    zone.to_csv(PROCESSED / "eraa2025_2030_zone_vre_capacities.csv", index=False)
    return zone


def read_capacity_weights() -> pd.DataFrame:
    zone = read_eraa_vre_zone_weights().copy()
    country = zone.groupby(["country", "technology"], as_index=False).capacity_mw.sum()
    wide = country.pivot(index="country", columns="technology", values="capacity_mw").fillna(0.0)
    wide = wide.rename(columns={
        "onshore": "onshore_capacity_mw",
        "offshore": "offshore_capacity_mw",
        "solar": "solar_capacity_mw",
    })
    wide = wide.reindex(EUROPE33).fillna(0.0).reset_index()
    wide.to_csv(PROCESSED / "fixed_2030_vre_capacity_weights.csv", index=False)
    return wide


def nonleap_day_of_year(timestamp: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    month = np.asarray(timestamp.month, dtype=np.int16)
    day = np.asarray(timestamp.day, dtype=np.int16)
    month_offsets = np.asarray([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334], dtype=np.int16)
    return (month_offsets[month - 1] + day).astype(float)


def continuous_features(timestamp: pd.DatetimeIndex, temperature_c: np.ndarray) -> np.ndarray:
    doy = nonleap_day_of_year(timestamp)
    hour = timestamp.hour.to_numpy(float)
    weekday = timestamp.dayofweek.to_numpy(int)
    workday = (weekday < 5).astype(float)
    hdd = np.maximum(15.5 - temperature_c, 0.0)
    cdd = np.maximum(temperature_c - 22.0, 0.0)
    angle_hour = 2.0 * np.pi * hour / 24.0
    columns = []
    for harmonic in range(1, 5):
        angle = 2.0 * np.pi * harmonic * (doy - 1.0) / 365.0
        columns.extend([np.sin(angle), np.cos(angle)])
    columns.extend([
        hdd,
        cdd,
        hdd * hdd / 100.0,
        cdd * cdd / 100.0,
        hdd * workday,
        cdd * workday,
        hdd * np.sin(angle_hour),
        hdd * np.cos(angle_hour),
        cdd * np.sin(angle_hour),
        cdd * np.cos(angle_hour),
        workday,
        (((timestamp.month == 12) & (timestamp.day >= 24)) | ((timestamp.month == 1) & (timestamp.day == 1))).astype(float),
    ])
    return np.column_stack(columns).astype(np.float64)


def design_matrix(
    timestamp: pd.DatetimeIndex,
    temperature_c: np.ndarray,
    continuous_mean: np.ndarray | None = None,
    continuous_scale: np.ndarray | None = None,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    n = len(timestamp)
    how = timestamp.dayofweek.to_numpy(int) * 24 + timestamp.hour.to_numpy(int)
    month = timestamp.month.to_numpy(int) - 1
    categorical = sparse.coo_matrix(
        (
            np.ones(2 * n),
            (
                np.concatenate([np.arange(n), np.arange(n)]),
                np.concatenate([how, 168 + month]),
            ),
        ),
        shape=(n, 180),
    ).tocsr()
    continuous = continuous_features(timestamp, temperature_c)
    if continuous_mean is None:
        continuous_mean = continuous.mean(axis=0)
    if continuous_scale is None:
        continuous_scale = continuous.std(axis=0)
        continuous_scale[continuous_scale < 1.0e-10] = 1.0
    continuous = (continuous - continuous_mean) / continuous_scale
    return sparse.hstack([categorical, sparse.csr_matrix(continuous)], format="csr"), continuous_mean, continuous_scale


def weighted_metrics(observed: np.ndarray, predicted: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    weights = weights / weights.sum()
    error = predicted - observed
    mean = float(np.sum(weights * observed))
    sse = float(np.sum(weights * error * error))
    sst = float(np.sum(weights * (observed - mean) ** 2))
    covariance = float(np.sum(weights * (observed - mean) * (predicted - np.sum(weights * predicted))))
    variance_predicted = float(np.sum(weights * (predicted - np.sum(weights * predicted)) ** 2))
    correlation = covariance / math.sqrt(max(sst * variance_predicted, 1.0e-30))
    return {
        "weighted_r2": 1.0 - sse / sst,
        "weighted_rmse_pu": math.sqrt(sse),
        "weighted_mae_pu": float(np.sum(weights * np.abs(error))),
        "weighted_pearson_r": correlation,
    }


def temperature_long_for_observations(temperature: pd.DataFrame, countries: list[str]) -> pd.DataFrame:
    available = [country for country in countries if country in temperature]
    long = temperature[["timestamp_utc", *available]].melt(
        id_vars="timestamp_utc", var_name="country", value_name="temperature_c"
    )
    long = long.sort_values(["country", "timestamp_utc"])
    long["temperature_24h_c"] = (
        long.groupby("country", sort=False).temperature_c
        .transform(lambda series: series.rolling(24, min_periods=1).mean())
    )
    return long


def fit_demand_model(temperature: pd.DataFrame) -> tuple[Ridge, np.ndarray, np.ndarray, dict[str, Any]]:
    observed = pd.read_csv(
        OBSERVED_LOAD,
        usecols=["timestamp_utc", "country", "year", "load_mw", "load_imputed"],
    )
    observed["country"] = observed.country.replace(ALIASES)
    observed["timestamp_utc"] = pd.to_datetime(observed.timestamp_utc, utc=True)
    observed = observed[~observed.load_imputed.astype(bool)].copy()
    observed["annual_mean_load_mw"] = observed.groupby(["country", "year"]).load_mw.transform("mean")
    observed["load_pu"] = observed.load_mw / observed.annual_mean_load_mw
    temperature_long = temperature_long_for_observations(temperature, sorted(observed.country.unique()))
    panel = observed.merge(temperature_long, on=["timestamp_utc", "country"], how="inner", validate="many_to_one")
    if len(panel) < 0.95 * len(observed):
        raise RuntimeError(f"Temperature merge retained only {len(panel)} of {len(observed)} observed rows")
    panel = panel.sort_values(["timestamp_utc", "country"]).reset_index(drop=True)
    timestamp = pd.DatetimeIndex(panel.timestamp_utc)
    train = panel.year.le(2022).to_numpy()
    validate = panel.year.ge(2023).to_numpy()
    x_train, mean, scale = design_matrix(timestamp[train], panel.loc[train, "temperature_24h_c"].to_numpy(float))
    x_validate, _, _ = design_matrix(
        timestamp[validate], panel.loc[validate, "temperature_24h_c"].to_numpy(float), mean, scale
    )
    y_train = panel.loc[train, "load_pu"].to_numpy(float)
    y_validate = panel.loc[validate, "load_pu"].to_numpy(float)
    weight_train = panel.loc[train, "annual_mean_load_mw"].to_numpy(float)
    weight_validate = panel.loc[validate, "annual_mean_load_mw"].to_numpy(float)
    weight_train /= np.mean(weight_train)
    weight_validate /= np.mean(weight_validate)

    candidates = []
    for alpha in (0.1, 1.0, 10.0, 100.0):
        model = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr", tol=1.0e-6)
        model.fit(x_train, y_train, sample_weight=weight_train)
        predicted = np.maximum(model.predict(x_validate), 0.05)
        metrics = weighted_metrics(y_validate, predicted, weight_validate)
        candidates.append((metrics["weighted_rmse_pu"], alpha, metrics))
    _, selected_alpha, holdout_metrics = min(candidates, key=lambda row: row[0])

    holdout_model = Ridge(alpha=selected_alpha, fit_intercept=True, solver="lsqr", tol=1.0e-6)
    holdout_model.fit(x_train, y_train, sample_weight=weight_train)
    predicted_holdout = np.maximum(holdout_model.predict(x_validate), 0.05)
    holdout = panel.loc[validate, ["timestamp_utc", "country", "year", "load_pu", "annual_mean_load_mw"]].copy()
    holdout["predicted_load_pu"] = predicted_holdout
    holdout["month"] = holdout.timestamp_utc.dt.month.astype(np.int8)
    holdout["hour"] = holdout.timestamp_utc.dt.hour.astype(np.int8)
    holdout["error_pu"] = holdout.predicted_load_pu - holdout.load_pu
    holdout_country_rows = []
    for (country, year), group in holdout.groupby(["country", "year"], sort=True):
        metrics = weighted_metrics(
            group.load_pu.to_numpy(float),
            group.predicted_load_pu.to_numpy(float),
            np.ones(len(group)),
        )
        holdout_country_rows.append({"country": country, "year": int(year), "hours": len(group), **metrics})
    pd.DataFrame(holdout_country_rows).to_csv(
        PROCESSED / "demand_model_holdout_country_year_fit.csv", index=False
    )
    holdout_surface = holdout.groupby(["month", "hour"], as_index=False).agg(
        observed_load_pu=("load_pu", "mean"),
        predicted_load_pu=("predicted_load_pu", "mean"),
        observations=("load_pu", "size"),
    )
    holdout_surface.to_csv(PROCESSED / "demand_model_holdout_month_hour_surface.csv", index=False)
    if len(holdout) > 120_000:
        holdout_sample = holdout.sample(120_000, random_state=5601).sort_values("timestamp_utc")
    else:
        holdout_sample = holdout
    holdout_sample.to_csv(
        PROCESSED / "demand_model_holdout_sample.csv.gz", index=False, compression="gzip"
    )

    x_all, mean_all, scale_all = design_matrix(timestamp, panel.temperature_24h_c.to_numpy(float))
    weight_all = panel.annual_mean_load_mw.to_numpy(float)
    weight_all /= np.mean(weight_all)
    final = Ridge(alpha=selected_alpha, fit_intercept=True, solver="lsqr", tol=1.0e-6)
    final.fit(x_all, panel.load_pu.to_numpy(float), sample_weight=weight_all)
    fitted = np.maximum(final.predict(x_all), 0.05)
    all_metrics = weighted_metrics(panel.load_pu.to_numpy(float), fitted, weight_all)

    country_rows = []
    panel["fitted_load_pu"] = fitted
    for (country, year), group in panel.groupby(["country", "year"], sort=True):
        metrics = weighted_metrics(
            group.load_pu.to_numpy(float), group.fitted_load_pu.to_numpy(float), np.ones(len(group))
        )
        country_rows.append({"country": country, "year": int(year), "hours": len(group), **metrics})
    pd.DataFrame(country_rows).to_csv(PROCESSED / "demand_model_country_year_fit.csv", index=False)
    pd.DataFrame([
        {"alpha": alpha, **metrics} for _, alpha, metrics in sorted(candidates, key=lambda row: row[1])
    ]).to_csv(PROCESSED / "demand_model_alpha_validation.csv", index=False)
    np.savez_compressed(
        PROCESSED / "demand_model_coefficients.npz",
        coefficient=final.coef_,
        intercept=np.asarray([final.intercept_]),
        alpha=np.asarray([selected_alpha]),
        continuous_mean=mean_all,
        continuous_scale=scale_all,
    )
    qa = {
        "status": "pass",
        "training_source": "Energy-Charts observed public-grid hourly load, 2015-2024",
        "observed_countries": sorted(panel.country.unique().tolist()),
        "observed_country_years": int(panel[["country", "year"]].drop_duplicates().shape[0]),
        "observed_hours": int(len(panel)),
        "validation_years": [2023, 2024],
        "holdout_hours": int(len(holdout)),
        "holdout_sample_hours": int(len(holdout_sample)),
        "selected_ridge_alpha": float(selected_alpha),
        "holdout": holdout_metrics,
        "all_observations_fit": all_metrics,
        "annual_normalization": "country-year mean load",
        "temperature_driver": "PECD v4.2 population-weighted temperature, trailing 24-h mean",
        "calendar_features": "hour-of-week, month, four annual harmonics, workday and year-end holiday",
    }
    if holdout_metrics["weighted_pearson_r"] < 0.70 or holdout_metrics["weighted_r2"] < 0.45:
        qa["status"] = "fail"
    write_json(PROCESSED / "demand_model_qa.json", qa)
    if qa["status"] != "pass":
        raise RuntimeError(f"Demand model validation failed: {qa}")
    return final, mean_all, scale_all, qa


def predict_country_load_shapes(
    temperature: pd.DataFrame,
    model: Ridge,
    continuous_mean: np.ndarray,
    continuous_scale: np.ndarray,
) -> pd.DataFrame:
    frames = []
    timestamp = pd.DatetimeIndex(temperature.timestamp_utc)
    for country in EUROPE33:
        if country not in temperature:
            raise RuntimeError(f"PECD v4.2 temperature missing for {country}")
        temp = pd.Series(temperature[country].to_numpy(float)).rolling(24, min_periods=1).mean().to_numpy()
        x, _, _ = design_matrix(timestamp, temp, continuous_mean, continuous_scale)
        predicted = np.maximum(model.predict(x), 0.05)
        years = timestamp.year.to_numpy(int)
        for year in range(1950, 2026):
            mask = years == year
            predicted[mask] /= predicted[mask].mean()
        frames.append(pd.DataFrame({
            "timestamp_utc": timestamp,
            "calendar_year": years.astype(np.int16),
            "country": country,
            "temperature_24h_c": temp.astype(np.float32),
            "predicted_load_pu": predicted.astype(np.float32),
        }))
        print(f"predicted demand shape for {country}", flush=True)
    output = pd.concat(frames, ignore_index=True)
    output.to_parquet(PROCESSED / "predicted_country_load_shapes.parquet", index=False, compression="zstd")
    return output


def build_europe_panel(load_shapes: pd.DataFrame) -> None:
    load_weights = read_eraa_load_weights().set_index("country").mean_2030_load_mw
    capacities = read_capacity_weights().set_index("country")
    generation = {
        key: pd.read_parquet(PROCESSED / f"{key}_country_hourly.parquet").set_index("timestamp_utc")
        for key in ("onshore", "offshore", "solar")
    }
    timestamp = pd.DatetimeIndex(generation["onshore"].index)
    panel = pd.DataFrame({"timestamp_utc": timestamp})
    panel["calendar_year"] = timestamp.year.astype(np.int16)

    predicted_wide = load_shapes.pivot(index="timestamp_utc", columns="country", values="predicted_load_pu")
    predicted_wide = predicted_wide.reindex(timestamp)
    panel["demand_mw"] = predicted_wide[EUROPE33].multiply(load_weights.reindex(EUROPE33), axis=1).sum(axis=1).to_numpy()

    technology_to_capacity = {
        "onshore": "onshore_capacity_mw",
        "offshore": "offshore_capacity_mw",
        "solar": "solar_capacity_mw",
    }
    for technology, capacity_column in technology_to_capacity.items():
        current = generation[technology]
        total = np.zeros(len(current), dtype=float)
        for country in EUROPE33:
            capacity = float(capacities.loc[country, capacity_column])
            if capacity == 0.0:
                continue
            if country not in current:
                raise RuntimeError(f"{technology} capacity factor missing for {country}")
            total += current[country].to_numpy(float) * capacity
        panel[f"{technology}_mw"] = total
    panel["vre_unscaled_mw"] = panel[["onshore_mw", "offshore_mw", "solar_mw"]].sum(axis=1)

    counts = panel.groupby("calendar_year").size()
    finite = np.isfinite(panel.select_dtypes(include=np.number).to_numpy()).all()
    if set(counts.index) != set(range(1950, 2026)) or not counts.eq(8760).all() or not finite:
        raise RuntimeError("Final PECD v4.2 Europe panel failed calendar or finite-value QA")
    if panel.demand_mw.le(0).any() or panel.vre_unscaled_mw.lt(0).any():
        raise RuntimeError("Final PECD v4.2 Europe panel has invalid power values")
    output = PROCESSED / "pecd_v42_europe33_hourly_1950_2025.parquet"
    panel.to_parquet(output, index=False, compression="zstd")
    qa = {
        "status": "pass",
        "calendar_years": [1950, 2025],
        "calendar_year_count": 76,
        "complete_july_june_operating_years": 75,
        "rows": int(len(panel)),
        "hours_per_calendar_year": 8760,
        "countries": EUROPE33,
        "demand_definition": "observed-load temperature-calendar model with ERAA 2025 2030 geographic weights",
        "residual_definition": "demand minus annual-energy-scaled PECD v4.2 onshore wind, offshore wind and solar",
        "february_29_treatment": "removed from every leap calendar year to retain 8,760-hour years",
        "output": output.name,
        "output_sha256": sha256(output),
    }
    write_json(PROCESSED / "pecd_v42_europe33_hourly_qa.json", qa)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["consolidate", "demand", "all"], nargs="?", default="all")
    args = parser.parse_args()
    if args.stage in {"consolidate", "all"}:
        consolidate_inputs()
    if args.stage in {"demand", "all"}:
        temperature = pd.read_parquet(PROCESSED / "temperature_country_hourly.parquet")
        model, mean, scale, _ = fit_demand_model(temperature)
        shapes = predict_country_load_shapes(temperature, model, mean, scale)
        build_europe_panel(shapes)


if __name__ == "__main__":
    main()
