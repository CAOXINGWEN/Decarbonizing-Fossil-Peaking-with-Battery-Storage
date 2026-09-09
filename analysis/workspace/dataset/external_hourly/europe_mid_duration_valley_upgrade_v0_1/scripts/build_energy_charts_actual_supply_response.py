#!/usr/bin/env python3
"""Build the observed hourly fossil-gas task and its system-response audit.

The primary historical task series is Energy-Charts ``Fossil gas`` generation
at hourly resolution.  It is the observed aggregate gas-dispatch task used to
measure event depth, duration, spacing and clustering.  The same files also
provide reservoir hydro, pumped-storage discharge and charging, other
dispatchable generation and physical cross-border exchange.  Those additional
series are retained as observed system responses and controls.

``Fossil gas`` is an aggregate technology category rather than a unit-level
OCGT/CCGT classification.  The output therefore calls it an observed
gas-dispatch or gas-peaking *task*, not a technology-specific peaker fleet.
Intact-day counterfactual reordering is performed downstream; this builder
only standardizes the observed chronology and describes natural events.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


RAW_PATTERN = re.compile(r"public_power_([a-z]+)_(\d{4})\.json$")
API_TO_ISO = {"uk": "GB"}
RESPONSE_MAP = {
    "fossil_gas": "fossil_gas_mw",
    "hydro_water_reservoir": "reservoir_hydro_mw",
    "hydro_run_of_river": "run_of_river_hydro_mw",
    "hydro_pumped_storage": "pumped_discharge_mw",
    "hydro_pumped_storage_consumption": (
        "pumped_storage_consumption_reported_mw"
    ),
    "cross_border_electricity_trading": "cross_border_signed_mw",
    "import_balance": "cross_border_signed_mw",
}
OTHER_DISPATCHABLE_FIELDS = (
    "biomass",
    "fossil_brown_coal_lignite",
    "fossil_coal_derived_gas",
    "fossil_hard_coal",
    "fossil_oil",
    "fossil_oil_shale",
    "fossil_peat",
    "geothermal",
    "nuclear",
    "others",
    "waste",
)
NON_GENERATION_FIELDS = {
    "load",
    "residual_load",
    "renewable_share_of_load",
    "renewable_share_of_generation",
    "cross_border_electricity_trading",
    "import_balance",
}
POWER_CHART_SIGN_DOCUMENTATION = (
    "https://www.energy-charts.info/charts/power/chart.htm"
)
API_DOCUMENTATION = "https://api.energy-charts.info/"
MIN_COVERAGE = 0.99
MAX_GAP_H = 24


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def longest_missing_run(values: pd.Series) -> int:
    missing = values.isna().to_numpy()
    if not missing.any():
        return 0
    changes = np.diff(np.r_[False, missing, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return int((ends - starts).max(initial=0))


def read_response_file(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    timestamps = pd.to_datetime(payload["unix_seconds"], unit="s", utc=True)
    data: dict[str, Any] = {"timestamp_utc": timestamps}
    for series in payload["production_types"]:
        raw_name = slug(str(series["name"]))
        values = pd.to_numeric(pd.Series(series["data"]), errors="coerce")
        if len(values) != len(timestamps):
            raise RuntimeError(f"Length mismatch: {path}: {series['name']}")
        data[raw_name] = values.to_numpy(float)
    frame = pd.DataFrame(data).set_index("timestamp_utc").sort_index()
    if frame.index.has_duplicates:
        frame = frame.groupby(level=0).mean()
    frame = frame.resample("1h").mean()
    output = pd.DataFrame(index=frame.index)
    for source, destination in RESPONSE_MAP.items():
        if destination == "cross_border_signed_mw":
            continue
        if source in frame:
            output[destination] = frame[source]
    if "cross_border_electricity_trading" in frame:
        output["cross_border_signed_mw"] = frame[
            "cross_border_electricity_trading"
        ]
    elif "import_balance" in frame:
        output["cross_border_signed_mw"] = frame["import_balance"]
    other_columns = [
        field
        for field in OTHER_DISPATCHABLE_FIELDS
        if field in frame and frame[field].notna().any()
    ]
    if other_columns:
        output["other_dispatchable_proxy_mw"] = frame[
            other_columns
        ].sum(axis=1, min_count=len(other_columns))
    generation_columns = [
        field
        for field in frame.columns
        if field not in NON_GENERATION_FIELDS and frame[field].notna().any()
    ]
    if generation_columns:
        output["balance_generation_without_cross_mw"] = frame[
            generation_columns
        ].sum(axis=1, min_count=len(generation_columns))
    return output


def build_country_response(paths: list[Path]) -> pd.DataFrame:
    frames = [read_response_file(path) for path in sorted(paths)]
    frame = pd.concat(frames).sort_index()
    if frame.index.has_duplicates:
        frame = frame.groupby(level=0).mean()
    for output in RESPONSE_MAP.values():
        if output not in frame:
            frame[output] = np.nan
    for output in (
        "other_dispatchable_proxy_mw",
        "balance_generation_without_cross_mw",
    ):
        if output not in frame:
            frame[output] = np.nan
    # The API reports storage consumption with a negative sign.
    frame["pumped_charge_mw"] = np.maximum(
        -frame["pumped_storage_consumption_reported_mw"], 0.0
    )
    return frame[
        [
            "fossil_gas_mw",
            "reservoir_hydro_mw",
            "run_of_river_hydro_mw",
            "pumped_discharge_mw",
            "pumped_charge_mw",
            "other_dispatchable_proxy_mw",
            "cross_border_signed_mw",
            "balance_generation_without_cross_mw",
        ]
    ]


def fill_component(
    values: pd.Series,
    minimum_coverage: float,
    maximum_gap_h: int,
) -> tuple[pd.Series, dict[str, Any]]:
    coverage = float(values.notna().mean())
    maximum_gap = longest_missing_run(values)
    eligible = bool(
        coverage >= minimum_coverage
        and maximum_gap <= maximum_gap_h
        and values.notna().any()
    )
    filled = values.copy()
    imputed = 0
    if eligible:
        missing = filled.isna()
        filled = filled.interpolate(
            method="time",
            limit=maximum_gap_h,
            limit_direction="both",
            limit_area="inside",
        )
        filled = filled.ffill(limit=maximum_gap_h).bfill(limit=maximum_gap_h)
        if filled.isna().any():
            raise RuntimeError("Eligible response retains missing values")
        imputed = int(missing.sum())
    return filled, {
        "coverage": coverage,
        "max_missing_gap_h": maximum_gap,
        "eligible": eligible,
        "imputed_hours": imputed,
    }


def label_runs(extreme: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    event_id = np.full(len(extreme), -1, dtype=int)
    duration = np.zeros(len(extreme), dtype=int)
    next_id = 0
    start = 0
    while start < len(extreme):
        if not extreme[start]:
            start += 1
            continue
        stop = start + 1
        while stop < len(extreme) and extreme[stop]:
            stop += 1
        event_id[start:stop] = next_id
        duration[start:stop] = stop - start
        next_id += 1
        start = stop
    return event_id, duration


def cluster_class(duration: int) -> str:
    if duration == 1:
        return "isolated_1d"
    if duration == 2:
        return "adjacent_2d"
    if duration >= 3:
        return "multiday_3plus"
    return "non_q75"


def weighted_mean_or_nan(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce")
    return float(values.mean()) if values.notna().any() else np.nan


def country_hierarchical_bootstrap(
    frame: pd.DataFrame,
    field: str,
    iterations: int = 2000,
    seed: int = 20260728,
) -> tuple[float, float]:
    sample = frame[["country", "year", field]].dropna()
    countries = np.asarray(sorted(sample["country"].unique()))
    grouped = {
        country: sample.loc[sample["country"] == country, field].to_numpy(
            float
        )
        for country in countries
    }
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations)
    for iteration in range(iterations):
        chosen = rng.choice(countries, size=len(countries), replace=True)
        country_means = []
        for country in chosen:
            values = grouped[country]
            country_means.append(
                float(
                    rng.choice(values, size=len(values), replace=True).mean()
                )
            )
        draws[iteration] = float(np.mean(country_means))
    return float(np.quantile(draws, 0.025)), float(
        np.quantile(draws, 0.975)
    )


def synchronized_year_block_bootstrap(
    frame: pd.DataFrame,
    field: str,
    iterations: int = 2000,
    seed: int = 20260729,
) -> tuple[float, float]:
    sample = frame[["country", "year", field]].dropna()
    year_means = sample.groupby("year")[field].mean().to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations)
    for iteration in range(iterations):
        draws[iteration] = float(
            rng.choice(
                year_means, size=len(year_means), replace=True
            ).mean()
        )
    return float(np.quantile(draws, 0.025)), float(
        np.quantile(draws, 0.975)
    )


def two_way_cluster_interval(
    frame: pd.DataFrame,
    field: str,
) -> tuple[float, float, float]:
    sample = frame[["country", "year", field]].dropna().copy()
    values = sample[field].to_numpy(float)
    mean = float(values.mean())
    residual = values - mean
    sample["_residual"] = residual
    n = len(sample)
    country_sums = sample.groupby("country")["_residual"].sum().to_numpy()
    year_sums = sample.groupby("year")["_residual"].sum().to_numpy()
    g_country = len(country_sums)
    g_year = len(year_sums)
    country_component = (
        g_country / (g_country - 1)
        * float(np.square(country_sums).sum())
        / n**2
    )
    year_component = (
        g_year / (g_year - 1)
        * float(np.square(year_sums).sum())
        / n**2
    )
    intersection_component = (
        n / (n - 1) * float(np.square(residual).sum()) / n**2
    )
    variance = max(
        country_component + year_component - intersection_component,
        0.0,
    )
    standard_error = float(np.sqrt(variance))
    degrees_freedom = max(1, min(g_country - 1, g_year - 1))
    critical = float(student_t.ppf(0.975, degrees_freedom))
    return (
        mean - critical * standard_error,
        mean + critical * standard_error,
        standard_error,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path(
            "processed/"
            "energy_charts_europe_hourly_eligible_2015_2024.csv.gz"
        ),
    )
    parser.add_argument("--q", type=float, default=0.75)
    parser.add_argument("--coverage-threshold", type=float, default=MIN_COVERAGE)
    parser.add_argument("--max-gap-hours", type=int, default=MAX_GAP_H)
    args = parser.parse_args()

    panel_path = (
        args.panel
        if args.panel.is_absolute()
        else args.root / args.panel
    )
    panel = pd.read_csv(
        panel_path,
        usecols=[
            "timestamp_utc",
            "country",
            "year",
            "load_mw",
            "wind_mw",
            "solar_mw",
        ],
    )
    panel["timestamp_utc"] = pd.to_datetime(
        panel["timestamp_utc"], utc=True
    )
    raw_dir = args.root / "raw" / "energy_charts" / "public_power"
    by_country: dict[str, list[Path]] = {}
    for path in sorted(raw_dir.glob("public_power_*_*.json")):
        match = RAW_PATTERN.search(path.name)
        if not match:
            continue
        api_code = match.group(1)
        iso2 = API_TO_ISO.get(api_code, api_code.upper())
        by_country.setdefault(iso2, []).append(path)

    response_cache = {
        country: build_country_response(paths)
        for country, paths in sorted(by_country.items())
    }
    hourly_outputs: list[pd.DataFrame] = []
    daily_outputs: list[pd.DataFrame] = []
    event_outputs: list[dict[str, Any]] = []
    qa_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    responses = (
        "fossil_gas_mw",
        "reservoir_hydro_mw",
        "run_of_river_hydro_mw",
        "pumped_discharge_mw",
        "pumped_charge_mw",
        "other_dispatchable_proxy_mw",
        "cross_border_signed_mw",
        "net_import_mw",
    )
    for (country, year), group in panel.groupby(
        ["country", "year"], sort=True
    ):
        group = group.sort_values("timestamp_utc").copy()
        response_source = response_cache[country].reindex(
            pd.DatetimeIndex(group["timestamp_utc"])
        )
        cross = response_source["cross_border_signed_mw"]
        generation = response_source[
            "balance_generation_without_cross_mw"
        ]
        load_for_balance = pd.Series(
            group["load_mw"].to_numpy(float),
            index=response_source.index,
        )
        balance_valid = (
            cross.notna() & generation.notna() & load_for_balance.notna()
        )
        if balance_valid.any():
            positive_import_error = (
                generation[balance_valid]
                + cross[balance_valid]
                - load_for_balance[balance_valid]
            )
            negative_import_error = (
                generation[balance_valid]
                - cross[balance_valid]
                - load_for_balance[balance_valid]
            )
            positive_import_mae = float(
                positive_import_error.abs().mean()
            )
            negative_import_mae = float(
                negative_import_error.abs().mean()
            )
            preferred_multiplier = (
                1.0
                if positive_import_mae <= negative_import_mae
                else -1.0
            )
            sign_status = (
                "official_positive_import_confirmed_by_lower_balance_MAE"
                if preferred_multiplier == 1.0
                else "official_sign_conflict_negative_import_lower_MAE"
            )
        else:
            positive_import_mae = np.nan
            negative_import_mae = np.nan
            preferred_multiplier = 1.0
            sign_status = (
                "insufficient_balance_hours_net_import_excluded"
            )
        response_source["net_import_mw"] = np.where(
            balance_valid.any(),
            preferred_multiplier * cross,
            np.nan,
        )
        response = pd.DataFrame(index=response_source.index)
        qa: dict[str, Any] = {
            "country": country,
            "year": int(year),
            "cross_border_official_power_chart_sign": (
                "positive import; negative export"
            ),
            "cross_border_balance_valid_hours": int(balance_valid.sum()),
            "cross_border_positive_import_balance_mae_mw": (
                positive_import_mae
            ),
            "cross_border_negative_import_balance_mae_mw": (
                negative_import_mae
            ),
            "cross_border_selected_multiplier_to_net_import": (
                preferred_multiplier
            ),
            "cross_border_sign_validation_status": sign_status,
            "cross_border_power_chart_documentation": (
                POWER_CHART_SIGN_DOCUMENTATION
            ),
            "energy_charts_api_documentation": API_DOCUMENTATION,
        }
        for field in responses:
            filled, record = fill_component(
                response_source[field],
                args.coverage_threshold,
                args.max_gap_hours,
            )
            response[field] = filled if record["eligible"] else np.nan
            for key, value in record.items():
                qa[f"{field}_{key}"] = value
        qa_rows.append(qa)

        load = group["load_mw"].to_numpy(float)
        wind = group["wind_mw"].to_numpy(float)
        solar = group["solar_mw"].to_numpy(float)
        vre = wind + solar
        scale = float(load.sum() / vre.sum())
        actual_residual = np.maximum(load - vre, 0.0)
        scaled_residual = np.maximum(load - scale * vre, 0.0)
        hourly = pd.DataFrame(
            {
                "timestamp_utc": group["timestamp_utc"].to_numpy(),
                "load_mw": load,
                "wind_mw": wind,
                "solar_mw": solar,
                "actual_residual_mw": actual_residual,
                "scaled_residual_mw": scaled_residual,
            }
        )
        for field in responses:
            hourly[field] = response[field].to_numpy(float)
        hourly.insert(0, "country", country)
        hourly.insert(1, "year", int(year))
        hourly["observed_gas_peaking_task_mw"] = np.maximum(
            hourly["fossil_gas_mw"], 0.0
        )
        hourly["gas_task_source"] = (
            "Energy-Charts public_power aggregate Fossil gas"
        )
        hourly["gas_task_hourly_observed"] = (
            hourly["fossil_gas_mw"].notna()
        )
        hourly_outputs.append(hourly.copy())
        hourly["date_utc"] = hourly["timestamp_utc"].dt.floor("D")
        daily = (
            hourly.groupby("date_utc", as_index=False)
            .agg(
                actual_residual_mwh=("actual_residual_mw", "sum"),
                scaled_residual_mwh=("scaled_residual_mw", "sum"),
                fossil_gas_mwh=("fossil_gas_mw", lambda x: x.sum(min_count=1)),
                reservoir_hydro_mwh=(
                    "reservoir_hydro_mw",
                    lambda x: x.sum(min_count=1),
                ),
                run_of_river_hydro_mwh=(
                    "run_of_river_hydro_mw",
                    lambda x: x.sum(min_count=1),
                ),
                pumped_discharge_mwh=(
                    "pumped_discharge_mw",
                    lambda x: x.sum(min_count=1),
                ),
                pumped_charge_mwh=(
                    "pumped_charge_mw",
                    lambda x: x.sum(min_count=1),
                ),
                other_dispatchable_proxy_mwh=(
                    "other_dispatchable_proxy_mw",
                    lambda x: x.sum(min_count=1),
                ),
                cross_border_signed_mwh=(
                    "cross_border_signed_mw",
                    lambda x: x.sum(min_count=1),
                ),
                net_import_mwh=(
                    "net_import_mw",
                    lambda x: x.sum(min_count=1),
                ),
            )
            .sort_values("date_utc")
        )
        daily.insert(0, "country", country)
        daily.insert(1, "year", int(year))
        daily["vre_scale_factor"] = scale
        label_specs = (
            (
                "actual_residual_q75_primary",
                "actual",
                "actual_residual_mwh",
            ),
            (
                "scaled_residual_q75_sensitivity",
                "scaled",
                "scaled_residual_mwh",
            ),
        )
        for label_name, prefix, label_field in label_specs:
            threshold = float(daily[label_field].quantile(args.q))
            scarcity_field = f"{prefix}_q75_scarcity_day"
            event_id_field = (
                f"{prefix}_event_id_within_country_year"
            )
            duration_field = f"{prefix}_event_duration_d"
            class_field = f"{prefix}_event_cluster_class"
            daily[scarcity_field] = daily[label_field] >= threshold
            event_id, event_duration = label_runs(
                daily[scarcity_field].to_numpy(bool)
            )
            daily[event_id_field] = event_id
            daily[duration_field] = event_duration
            daily[class_field] = [
                cluster_class(value) for value in event_duration
            ]
            daily[f"q_threshold_{label_field}"] = threshold

            extreme = daily[daily[scarcity_field]].copy()
            for event_id_value, event in extreme.groupby(
                event_id_field, sort=True
            ):
                row: dict[str, Any] = {
                    "country": country,
                    "year": int(year),
                    "scarcity_label": label_name,
                    "label_residual_field": label_field,
                    "is_primary_actual_dispatch_label": (
                        label_name == "actual_residual_q75_primary"
                    ),
                    "event_id_within_country_year": int(event_id_value),
                    "event_start_utc": event["date_utc"].min(),
                    "event_end_utc": event["date_utc"].max(),
                    "event_duration_d": int(len(event)),
                    "event_cluster_class": cluster_class(len(event)),
                    "scaled_residual_mwh": float(
                        event["scaled_residual_mwh"].sum()
                    ),
                    "actual_residual_mwh": float(
                        event["actual_residual_mwh"].sum()
                    ),
                }
                for field in (
                    "fossil_gas_mwh",
                    "reservoir_hydro_mwh",
                    "run_of_river_hydro_mwh",
                    "pumped_discharge_mwh",
                    "pumped_charge_mwh",
                    "other_dispatchable_proxy_mwh",
                    "cross_border_signed_mwh",
                    "net_import_mwh",
                ):
                    row[field] = (
                        float(event[field].sum(min_count=1))
                        if event[field].notna().any()
                        else np.nan
                    )
                event_outputs.append(row)

            comparison: dict[str, Any] = {
                "country": country,
                "year": int(year),
                "scarcity_label": label_name,
                "label_residual_field": label_field,
                "is_primary_actual_dispatch_label": (
                    label_name == "actual_residual_q75_primary"
                ),
                "response_intensity_denominator": (
                    "actual_residual_mwh"
                ),
                "q": args.q,
                "q_threshold_mwh": threshold,
                "q75_day_count": int(len(extreme)),
                "isolated_day_count": int(
                    (extreme[class_field] == "isolated_1d").sum()
                ),
                "multiday_3plus_day_count": int(
                    (extreme[class_field] == "multiday_3plus").sum()
                ),
            }
            isolated = extreme[
                extreme[class_field] == "isolated_1d"
            ]
            clustered = extreme[
                extreme[class_field] == "multiday_3plus"
            ]
            for field in (
                "actual_residual_mwh",
                "scaled_residual_mwh",
                "fossil_gas_mwh",
                "reservoir_hydro_mwh",
                "run_of_river_hydro_mwh",
                "pumped_discharge_mwh",
                "pumped_charge_mwh",
                "other_dispatchable_proxy_mwh",
                "cross_border_signed_mwh",
                "net_import_mwh",
            ):
                isolated_mean = weighted_mean_or_nan(isolated, field)
                clustered_mean = weighted_mean_or_nan(clustered, field)
                comparison[
                    f"isolated_mean_{field}_per_day"
                ] = isolated_mean
                comparison[
                    f"multiday_mean_{field}_per_day"
                ] = clustered_mean
                comparison[
                    f"multiday_minus_isolated_{field}_per_day"
                ] = clustered_mean - isolated_mean
            for field in (
                "fossil_gas_mwh",
                "reservoir_hydro_mwh",
                "run_of_river_hydro_mwh",
                "pumped_discharge_mwh",
                "pumped_charge_mwh",
                "other_dispatchable_proxy_mwh",
                "cross_border_signed_mwh",
                "net_import_mwh",
            ):
                isolated_intensity = (
                    float(isolated[field].sum(min_count=1))
                    / float(isolated["actual_residual_mwh"].sum())
                    if isolated[field].notna().any()
                    and len(isolated)
                    and float(isolated["actual_residual_mwh"].sum()) > 0
                    else np.nan
                )
                clustered_intensity = (
                    float(clustered[field].sum(min_count=1))
                    / float(clustered["actual_residual_mwh"].sum())
                    if clustered[field].notna().any()
                    and len(clustered)
                    and float(clustered["actual_residual_mwh"].sum()) > 0
                    else np.nan
                )
                comparison[
                    f"isolated_{field}_per_actual_residual_mwh"
                ] = isolated_intensity
                comparison[
                    f"multiday_{field}_per_actual_residual_mwh"
                ] = clustered_intensity
                comparison[
                    f"multiday_minus_isolated_{field}"
                    "_per_actual_residual_mwh"
                ] = clustered_intensity - isolated_intensity
            contrast_rows.append(comparison)
        daily_outputs.append(daily)

    processed = args.root / "processed"
    research = args.root / "artifacts" / "research"
    hourly_path = (
        processed
        / "energy_charts_actual_gas_peaking_hourly_2015_2024.csv.gz"
    )
    hourly_panel = pd.concat(hourly_outputs, ignore_index=True).sort_values(
        ["country", "year", "timestamp_utc"]
    )
    hourly_panel.to_csv(
        hourly_path,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        float_format="%.6f",
    )
    gas_task_path = (
        processed
        / "energy_charts_observed_gas_task_panel_2015_2024.csv.gz"
    )
    gas_task_panel = hourly_panel[
        [
            "timestamp_utc",
            "country",
            "year",
            "load_mw",
            "wind_mw",
            "solar_mw",
            "observed_gas_peaking_task_mw",
            "gas_task_source",
        ]
    ].copy()
    gas_task_panel = gas_task_panel.rename(
        columns={
            "load_mw": "system_load_mw",
            "wind_mw": "observed_wind_mw",
            "solar_mw": "observed_solar_mw",
        }
    )
    # Adapter-facing aliases.  With --vre-energy-share 0 the existing sparse
    # LP receives the observed gas task as residual demand and no exogenous
    # zero-cost surplus.  Recharge is supplied endogenously by additional
    # off-peak generation and therefore pays the gas variable cost and losses.
    gas_task_panel["load_mw"] = gas_task_panel[
        "observed_gas_peaking_task_mw"
    ]
    gas_task_panel["wind_mw"] = gas_task_panel["observed_wind_mw"]
    gas_task_panel["solar_mw"] = gas_task_panel["observed_solar_mw"]
    key = [gas_task_panel["country"], gas_task_panel["year"]]
    gas_task_panel["gas_task_eligible"] = (
        gas_task_panel["observed_gas_peaking_task_mw"]
        .notna()
        .groupby(key)
        .transform("all")
    )
    annual = (
        gas_task_panel.groupby(["country", "year"], sort=False)
        .agg(
            annual_gas_mwh=("observed_gas_peaking_task_mw", "sum"),
            annual_system_load_mwh=("system_load_mw", "sum"),
        )
    )
    annual["annual_gas_share_of_system_load"] = (
        annual["annual_gas_mwh"] / annual["annual_system_load_mwh"]
    )
    gas_task_panel = gas_task_panel.merge(
        annual[["annual_gas_share_of_system_load"]].reset_index(),
        on=["country", "year"],
        how="left",
        validate="many_to_one",
    )
    gas_task_panel["gas_task_material_share_ge_1pct"] = (
        gas_task_panel["annual_gas_share_of_system_load"] >= 0.01
    )
    gas_task_panel["gas_task_model_contract"] = (
        "observed Fossil gas is adapter load; run with vre_energy_share=0; "
        "charging energy is endogenously generated and explicitly costed"
    )
    gas_task_panel.to_csv(
        gas_task_path,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        float_format="%.6f",
    )
    gas_units = gas_task_panel[
        [
            "country",
            "year",
            "gas_task_eligible",
            "gas_task_material_share_ge_1pct",
        ]
    ].drop_duplicates(["country", "year"])
    main_gas_units = gas_units.loc[
        gas_units["gas_task_eligible"].astype(bool)
        & gas_units["gas_task_material_share_ge_1pct"].astype(bool)
    ]
    balanced_gas_countries = sorted(
        main_gas_units.groupby("country")["year"]
        .nunique()
        .loc[lambda values: values.eq(10)]
        .index.astype(str)
    )
    gas_task_qa = {
        "hourly_source_field": "Energy-Charts public_power: Fossil gas",
        "task_interpretation": (
            "observed aggregate hourly fossil-gas generation used as the "
            "gas-peaking task"
        ),
        "hourly_rows": int(len(gas_task_panel)),
        "country_years_in_parent_panel": int(len(gas_units)),
        "complete_hourly_gas_country_years": int(
            gas_units["gas_task_eligible"].astype(bool).sum()
        ),
        "main_country_years_complete_and_gas_share_ge_1pct": int(
            len(main_gas_units)
        ),
        "main_countries": int(main_gas_units["country"].nunique()),
        "balanced_2015_2024_countries": balanced_gas_countries,
        "balanced_2015_2024_country_years": (
            len(balanced_gas_countries) * 10
        ),
        "adapter_contract": (
            "set vre_energy_share=0 so observed gas is residual task; "
            "recharge is endogenously generated and pays fuel plus losses"
        ),
        "output": str(gas_task_path),
        "output_sha256": sha256(gas_task_path),
    }
    gas_task_qa["pass"] = bool(
        gas_task_qa["complete_hourly_gas_country_years"] == 141
        and gas_task_qa[
            "main_country_years_complete_and_gas_share_ge_1pct"
        ]
        >= 100
        and gas_task_qa["balanced_2015_2024_country_years"] >= 50
    )
    gas_task_qa_path = (
        research / "energy_charts_observed_gas_task_panel_qa.json"
    )
    gas_task_qa_path.write_text(
        json.dumps(gas_task_qa, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    daily_path = (
        processed
        / "energy_charts_actual_supply_response_daily_2015_2024.csv.gz"
    )
    event_path = (
        processed
        / "energy_charts_actual_supply_response_events_2015_2024.csv.gz"
    )
    pd.concat(daily_outputs, ignore_index=True).to_csv(
        daily_path,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        float_format="%.6f",
    )
    pd.DataFrame(event_outputs).sort_values(
        [
            "country",
            "year",
            "scarcity_label",
            "event_id_within_country_year",
        ]
    ).to_csv(
        event_path,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        float_format="%.6f",
    )
    qa_path = research / "energy_charts_actual_supply_response_qa.csv"
    pd.DataFrame(qa_rows).sort_values(["country", "year"]).to_csv(
        qa_path, index=False
    )
    contrast_path = (
        research
        / "energy_charts_natural_scarcity_supply_response_contrasts.csv"
    )
    contrasts = pd.DataFrame(contrast_rows).sort_values(
        ["country", "year", "scarcity_label"]
    )
    contrasts.to_csv(contrast_path, index=False)
    manifest = json.loads(
        (
            args.root / "metadata" / "energy_charts_panel_manifest.json"
        ).read_text(encoding="utf-8")
    )
    balanced = set(manifest["sample"]["balanced_eu_countries"])
    sample_rows: list[dict[str, Any]] = []
    contrast_fields = [
        column
        for column in contrasts.columns
        if column.startswith("multiday_minus_isolated_")
    ]
    for sample, countries in (
        ("expanded_europe_17c_155cy", None),
        ("balanced_eu_9c_90cy", balanced),
    ):
        selected = (
            contrasts
            if countries is None
            else contrasts[contrasts["country"].isin(countries)]
        )
        for label, label_frame in selected.groupby(
            "scarcity_label", sort=True
        ):
            for field in contrast_fields:
                values = pd.to_numeric(
                    label_frame[field], errors="coerce"
                ).dropna()
                inference_frame = label_frame.loc[
                    values.index, ["country", "year", field]
                ].copy()
                if len(values):
                    country_low, country_high = (
                        country_hierarchical_bootstrap(
                            inference_frame, field
                        )
                    )
                    year_low, year_high = (
                        synchronized_year_block_bootstrap(
                            inference_frame, field
                        )
                    )
                    two_way_low, two_way_high, two_way_se = (
                        two_way_cluster_interval(inference_frame, field)
                    )
                else:
                    country_low = country_high = np.nan
                    year_low = year_high = np.nan
                    two_way_low = two_way_high = two_way_se = np.nan
                sample_rows.append(
                    {
                        "sample": sample,
                        "scarcity_label": label,
                        "is_primary_actual_dispatch_label": bool(
                            label == "actual_residual_q75_primary"
                        ),
                        "metric": field,
                        "country_count_with_contrast": int(
                            label_frame.loc[
                                values.index, "country"
                            ].nunique()
                        ),
                        "country_year_count_with_contrast": int(len(values)),
                        "country_year_median": (
                            float(values.median()) if len(values) else np.nan
                        ),
                        "country_year_mean": (
                            float(values.mean()) if len(values) else np.nan
                        ),
                        "positive_country_year_fraction": (
                            float((values > 0).mean())
                            if len(values)
                            else np.nan
                        ),
                        "country_hierarchical_bootstrap_mean_ci_low": (
                            country_low
                        ),
                        "country_hierarchical_bootstrap_mean_ci_high": (
                            country_high
                        ),
                        "synchronized_year_block_bootstrap_mean_ci_low": (
                            year_low
                        ),
                        "synchronized_year_block_bootstrap_mean_ci_high": (
                            year_high
                        ),
                        "country_year_two_way_cluster_mean_ci_low": (
                            two_way_low
                        ),
                        "country_year_two_way_cluster_mean_ci_high": (
                            two_way_high
                        ),
                        "country_year_two_way_cluster_standard_error": (
                            two_way_se
                        ),
                        "dependence_robust_interval_envelope_low": (
                            min(country_low, year_low, two_way_low)
                            if len(values)
                            else np.nan
                        ),
                        "dependence_robust_interval_envelope_high": (
                            max(country_high, year_high, two_way_high)
                            if len(values)
                            else np.nan
                        ),
                        "empirical_unit": "country-year",
                        "interpretation": (
                            "descriptive natural-event response; countries "
                            "share synchronized calendar-year weather"
                        ),
                    }
                )
    sample_path = (
        research
        / "energy_charts_actual_supply_response_sample_summary.csv"
    )
    pd.DataFrame(sample_rows).to_csv(sample_path, index=False)
    dictionary_path = (
        research
        / "energy_charts_actual_supply_response_field_dictionary.csv"
    )
    pd.DataFrame(
        [
            {
                "field": "cross_border_signed_mw",
                "definition": (
                    "Raw public_power signed cross-border series. The power "
                    "chart convention is positive import and negative export."
                ),
                "unit": "MW",
                "source": POWER_CHART_SIGN_DOCUMENTATION,
            },
            {
                "field": "net_import_mw",
                "definition": (
                    "Cross-border series oriented so positive means net "
                    "import after country-year energy-balance sign audit."
                ),
                "unit": "MW",
                "source": (
                    "Energy-Charts public_power field plus study energy-"
                    "balance sign audit"
                ),
            },
            {
                "field": "fossil_gas_mw",
                "definition": (
                    "Observed aggregate hourly fossil-gas generation. Used "
                    "as the empirical gas-dispatch task; it does not split "
                    "unit-level OCGT from CCGT operation."
                ),
                "unit": "MW",
                "source": "Energy-Charts public_power: Fossil gas",
            },
            {
                "field": "observed_gas_peaking_task_mw",
                "definition": (
                    "Non-negative copy of observed aggregate fossil-gas "
                    "generation used for event geometry and downstream "
                    "paired intact-day chronology experiments."
                ),
                "unit": "MW",
                "source": "Energy-Charts public_power: Fossil gas",
            },
            {
                "field": "reservoir_hydro_mw",
                "definition": "Reported reservoir-hydro generation response.",
                "unit": "MW",
                "source": "Energy-Charts public_power",
            },
            {
                "field": "run_of_river_hydro_mw",
                "definition": "Reported run-of-river hydro generation.",
                "unit": "MW",
                "source": "Energy-Charts public_power",
            },
            {
                "field": "pumped_discharge_mw",
                "definition": "Reported pumped-storage generation.",
                "unit": "MW",
                "source": "Energy-Charts public_power",
            },
            {
                "field": "pumped_charge_mw",
                "definition": (
                    "Non-negative charging demand derived as the negative of "
                    "reported pumped-storage consumption."
                ),
                "unit": "MW",
                "source": "Energy-Charts public_power",
            },
            {
                "field": "other_dispatchable_proxy_mw",
                "definition": (
                    "Coverage-audit proxy summing reported biomass, coal, "
                    "oil, peat, geothermal, nuclear, other and waste series. "
                    "It is not treated as a homogeneous flexible fleet."
                ),
                "unit": "MW",
                "source": "Energy-Charts public_power component sum",
            },
            {
                "field": "actual_residual_mw",
                "definition": (
                    "max(load - observed wind - observed solar, 0); primary "
                    "label basis for actual-response q75 events."
                ),
                "unit": "MW",
                "source": "study-derived from observed hourly profiles",
            },
            {
                "field": "scaled_residual_mw",
                "definition": (
                    "High-VRE profile sensitivity after scaling annual "
                    "wind-plus-solar energy to annual load."
                ),
                "unit": "MW",
                "source": "study-derived sensitivity",
            },
        ]
    ).to_csv(dictionary_path, index=False)

    eligible_counts = {
        field: int(
            pd.DataFrame(qa_rows)[f"{field}_eligible"].astype(bool).sum()
        )
        for field in responses
    }
    interpretation = [
        "# Energy-Charts natural scarcity supply-response audit",
        "",
        (
            f"The load-wind-solar sample contains {len(qa_rows)} eligible "
            "country-years. Component-level response eligibility is "
            + ", ".join(
                f"{field} {count}/{len(qa_rows)}"
                for field, count in eligible_counts.items()
            )
            + "."
        ),
        "",
        (
            "The comparison is descriptive. It contrasts naturally isolated "
            "q75 actual-residual-demand days with days belonging to naturally "
            "observed runs of at least three such days within each "
            "country-year. This is the primary actual-dispatch label."
        ),
        "",
        (
            "A separate scaled-residual q75 label is retained only as a "
            "forward-looking profile sensitivity. It is never pooled with "
            "the primary label. All response-intensity denominators use "
            "actual residual demand, not scaled residual demand."
        ),
        "",
        (
            "The hourly Energy-Charts Fossil gas series is used downstream "
            "as the observed gas-peaking task. It is aggregate gas-fired "
            "generation; unit-level OCGT and CCGT operation are not split. "
            "Reservoir hydro, pumped storage and imports remain response "
            "controls rather than components of that task."
        ),
        "",
        (
            "The public power-chart cross-border convention is positive "
            "import and negative export. Each country-year is additionally "
            "checked against the hourly energy balance under both possible "
            "signs. The raw signed field and the resulting positive-import "
            "net_import field are both retained. Energy-Charts map rankings "
            "use a different display convention and are not used here."
        ),
        "",
        (
            "Reordering intact day blocks would preserve these recorded "
            "responses exactly. Therefore, the table is an external "
            "descriptive response check and not a causal test of the paired "
            "chronology intervention."
        ),
    ]
    interpretation_path = (
        research / "energy_charts_actual_supply_response_interpretation.md"
    )
    interpretation_path.write_text(
        "\n".join(interpretation) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "country_years": len(qa_rows),
                "response_eligible_country_years": eligible_counts,
                "hourly_gas_task_output": str(gas_task_path),
                "hourly_gas_task_qa": str(gas_task_qa_path),
                "daily_output": str(daily_path),
                "event_output": str(event_path),
                "contrast_output": str(contrast_path),
                "sample_summary": str(sample_path),
                "field_dictionary": str(dictionary_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
