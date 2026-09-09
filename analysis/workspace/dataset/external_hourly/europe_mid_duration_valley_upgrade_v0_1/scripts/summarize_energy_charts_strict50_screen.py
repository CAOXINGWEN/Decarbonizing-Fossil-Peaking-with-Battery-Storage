#!/usr/bin/env python3
"""Summarize the 4-168 h Energy-Charts screen without pseudo-replication.

Configuration-level results are first reduced within each country-year.  All
sample-level direction counts and hierarchical bootstrap intervals then use
country-years nested within countries; configuration-hour rows are never
treated as independent empirical observations.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


METRICS = {
    "efc": "expected_equivalent_full_cycles_output_basis",
    "charge_mwh_per_mean_mw_year": "expected_charge_mwh_per_mean_mw_year",
    "discharge_mwh_per_mean_mw_year": (
        "expected_discharge_mwh_per_mean_mw_year"
    ),
    "joint_benefit_cny_per_forced_kw_year": (
        "joint_signed_benefit_cny_per_forced_kw_year"
    ),
    "turnover_benefit_cny_per_forced_kw_year": (
        "turnover_signed_benefit_cny_per_forced_kw_year"
    ),
    "spill_mwh_per_mean_mw_year": "expected_spill_mwh_per_mean_mw_year",
    "gas_generation_mwh_per_mean_mw_year": (
        "expected_gas_generation_mwh_per_mean_mw_year"
    ),
    "charge_power_saturation_h": "mean_charge_power_saturation_h",
}
DELTA_FIELDS = tuple(f"delta_{name}" for name in METRICS)


def as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


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
    order: list[int] = []
    for index, day in enumerate(normal):
        order.extend(slots[index])
        order.append(day)
    order.extend(slots[-1])
    return order


def interleave_clusters(
    extreme: list[int],
    normal: list[int],
    cluster_days: int = 4,
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
    order: list[int] = []
    for index, day in enumerate(normal):
        order.extend(slots[index])
        order.append(day)
    order.extend(slots[-1])
    return order


def chronology_recharge_metrics(group: pd.DataFrame) -> list[dict[str, Any]]:
    country = str(group["country"].iloc[0])
    year = int(group["year"].iloc[0])
    group = group.sort_values("timestamp_utc")
    load = group["load_mw"].to_numpy(float)
    vre = (
        group["wind_mw"].to_numpy(float)
        + group["solar_mw"].to_numpy(float)
    )
    scale = float(load.sum() / vre.sum())
    mean_load = float(load.mean())
    residual = np.maximum(load - scale * vre, 0.0) / mean_load
    surplus = np.maximum(scale * vre - load, 0.0) / mean_load
    hourly = pd.DataFrame(
        {
            "model_day": np.arange(len(group)) // 24,
            "residual": residual,
            "surplus": surplus,
            "month": group["timestamp_utc"].dt.month.to_numpy(),
        }
    )
    daily = hourly.groupby("model_day", sort=True).agg(
        residual=("residual", "sum"),
        surplus=("surplus", "sum"),
        month=("month", "first"),
    )
    daily["segment"] = daily["month"].map(calendar_segment)
    threshold = float(daily["residual"].quantile(0.75))
    daily["extreme"] = daily["residual"] >= threshold
    orders = {"natural": [int(day) for day in daily.index]}
    for chronology in ("dispersed", "clustered"):
        order: list[int] = []
        for segment in sorted(daily["segment"].unique()):
            subset = daily[daily["segment"] == segment]
            extreme = [int(day) for day in subset.index[subset["extreme"]]]
            normal = [int(day) for day in subset.index[~subset["extreme"]]]
            if chronology == "dispersed":
                order.extend(interleave(extreme, normal))
            else:
                order.extend(interleave_clusters(extreme, normal))
        orders[chronology] = order

    records = []
    for chronology, order in orders.items():
        arranged = daily.loc[order].reset_index(drop=True)
        positions = np.flatnonzero(arranged["extreme"].to_numpy(bool))
        gaps = np.diff(positions) - 1
        intervening_surplus = np.array(
            [
                arranged.loc[left + 1 : right - 1, "surplus"].sum()
                if right > left + 1
                else 0.0
                for left, right in zip(positions[:-1], positions[1:])
            ],
            dtype=float,
        )
        records.append(
            {
                "country": country,
                "year": year,
                "chronology": chronology,
                "q75_day_count": int(len(positions)),
                "successive_q75_pair_count": int(len(gaps)),
                "zero_day_recharge_window_share": float((gaps == 0).mean()),
                "median_nonextreme_days_between_q75_days": float(
                    np.median(gaps)
                ),
                "q25_nonextreme_days_between_q75_days": float(
                    np.quantile(gaps, 0.25)
                ),
                "mean_nonextreme_days_between_q75_days": float(gaps.mean()),
                "zero_surplus_recharge_window_share": float(
                    (intervening_surplus <= 1e-12).mean()
                ),
                "median_surplus_between_q75_days_pu_h": float(
                    np.median(intervening_surplus)
                ),
                "q25_surplus_between_q75_days_pu_h": float(
                    np.quantile(intervening_surplus, 0.25)
                ),
                "annual_surplus_pu_h": float(arranged["surplus"].sum()),
                "annual_residual_pu_h": float(arranged["residual"].sum()),
            }
        )
    return records


def parse_unit_id(frame: pd.DataFrame) -> pd.DataFrame:
    split = frame["unit_id"].str.rsplit("_", n=1, expand=True)
    frame = frame.copy()
    frame["country"] = split[0]
    frame["year"] = pd.to_numeric(split[1], errors="raise").astype(int)
    return frame


def paired_configuration_changes(metrics: pd.DataFrame) -> pd.DataFrame:
    selected = metrics[
        metrics["chronology"].isin(["dispersed", "clustered"])
    ].copy()
    identifiers = [
        "unit_id",
        "config_id",
        "tech_id",
        "technology_en",
        "duration_h",
    ]
    output: pd.DataFrame | None = None
    for label, field in METRICS.items():
        wide = selected.pivot(
            index=identifiers,
            columns="chronology",
            values=field,
        )
        delta = (
            wide["clustered"] - wide["dispersed"]
        ).rename(f"delta_{label}").reset_index()
        output = (
            delta
            if output is None
            else output.merge(delta, on=identifiers, validate="one_to_one")
        )
    if output is None:
        raise RuntimeError("No paired configuration changes")
    return parse_unit_id(output)


def reduce_country_year(paired: pd.DataFrame) -> pd.DataFrame:
    counts = (
        paired.groupby(["country", "year", "unit_id"], as_index=False)
        .agg(
            configuration_count=("config_id", "nunique"),
            technology_count=("tech_id", "nunique"),
        )
    )
    per_technology = (
        paired.groupby(
            ["country", "year", "unit_id", "tech_id"],
            as_index=False,
        )[list(DELTA_FIELDS)]
        .median()
    )
    aggregations: dict[str, tuple[str, Any]] = {}
    for field in DELTA_FIELDS:
        aggregations[f"median_{field}"] = (field, "median")
        aggregations[f"negative_share_{field}"] = (
            field,
            lambda values: float((values < 0).mean()),
        )
    reduced = (
        per_technology.groupby(
            ["country", "year", "unit_id"], as_index=False
        )
        .agg(**aggregations)
    )
    return (
        counts.merge(
            reduced,
            on=["country", "year", "unit_id"],
            validate="one_to_one",
        )
        .sort_values(["country", "year"])
    )


def hierarchical_bootstrap_mean(
    frame: pd.DataFrame,
    field: str,
    iterations: int = 2000,
    seed: int = 20260726,
) -> tuple[float, float]:
    """Cluster-resample countries and years for a country-year mean.

    Every reported point estimate gives one equal weight to each observed
    country-year.  The bootstrap therefore concatenates the resampled
    country-year rows before taking their mean; averaging country means would
    target a different, country-weighted estimand when panels are unbalanced.
    """

    rng = np.random.default_rng(seed)
    countries = np.asarray(sorted(frame["country"].unique()))
    draws = np.empty(iterations, dtype=float)
    grouped = {
        country: frame.loc[frame["country"] == country, field]
        .dropna()
        .to_numpy(float)
        for country in countries
    }
    for iteration in range(iterations):
        sampled_countries = rng.choice(
            countries, size=len(countries), replace=True
        )
        sampled_rows = []
        for country in sampled_countries:
            values = grouped[country]
            sampled_rows.append(
                rng.choice(values, size=len(values), replace=True)
            )
        draws[iteration] = float(np.concatenate(sampled_rows).mean())
    return float(np.quantile(draws, 0.025)), float(
        np.quantile(draws, 0.975)
    )


def calendar_year_block_bootstrap_mean(
    frame: pd.DataFrame,
    field: str,
    iterations: int = 2000,
    seed: int = 20260727,
) -> tuple[float, float]:
    """Resample synchronized years for the same country-year mean estimand."""

    rng = np.random.default_rng(seed)
    sample = frame[["year", field]].dropna()
    grouped = {
        int(year): group[field].to_numpy(float)
        for year, group in sample.groupby("year", sort=True)
    }
    years = np.asarray(sorted(grouped))
    draws = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sampled_years = rng.choice(years, size=len(years), replace=True)
        draws[iteration] = float(
            np.concatenate([grouped[int(year)] for year in sampled_years])
            .mean()
        )
    return float(np.quantile(draws, 0.025)), float(
        np.quantile(draws, 0.975)
    )


def two_way_cluster_mean_interval(
    frame: pd.DataFrame,
    field: str,
) -> tuple[float, float, float]:
    """Two-way country and calendar-year cluster-robust mean interval."""

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
    standard_error = math.sqrt(variance)
    degrees_freedom = max(1, min(g_country - 1, g_year - 1))
    critical = float(student_t.ppf(0.975, degrees_freedom))
    return (
        mean - critical * standard_error,
        mean + critical * standard_error,
        standard_error,
    )


def sample_summary(
    country_year: pd.DataFrame,
    sample: str,
    countries: set[str] | None,
) -> list[dict[str, Any]]:
    selected = (
        country_year
        if countries is None
        else country_year[country_year["country"].isin(countries)]
    )
    rows = []
    for field in DELTA_FIELDS:
        reduced_field = f"median_{field}"
        values = selected[reduced_field].dropna()
        country_lower, country_upper = hierarchical_bootstrap_mean(
            selected, reduced_field
        )
        year_lower, year_upper = calendar_year_block_bootstrap_mean(
            selected, reduced_field
        )
        two_way_lower, two_way_upper, two_way_se = (
            two_way_cluster_mean_interval(selected, reduced_field)
        )
        rows.append(
            {
                "sample": sample,
                "country_count": int(selected["country"].nunique()),
                "country_year_count": int(
                    selected[["country", "year"]].drop_duplicates().shape[0]
                ),
                "metric": field,
                "country_year_mean": float(values.mean()),
                "country_year_median": float(values.median()),
                "negative_country_year_n": int((values < 0).sum()),
                "negative_country_year_fraction": float((values < 0).mean()),
                "country_hierarchical_bootstrap_mean_ci_low": (
                    country_lower
                ),
                "country_hierarchical_bootstrap_mean_ci_high": (
                    country_upper
                ),
                "synchronized_year_block_bootstrap_mean_ci_low": year_lower,
                "synchronized_year_block_bootstrap_mean_ci_high": year_upper,
                "country_year_two_way_cluster_mean_ci_low": two_way_lower,
                "country_year_two_way_cluster_mean_ci_high": two_way_upper,
                "country_year_two_way_cluster_standard_error": two_way_se,
                "dependence_robust_interval_envelope_low": min(
                    country_lower, year_lower, two_way_lower
                ),
                "dependence_robust_interval_envelope_high": max(
                    country_upper, year_upper, two_way_upper
                ),
                "empirical_unit": "country-year nested within country",
                "calendar_year_dependence": (
                    "countries sharing a calendar year are resampled as one "
                    "synchronized year block; a two-way country/year cluster "
                    "interval is also reported"
                ),
                "inference_status": (
                    "descriptive historical external validation; ERAA "
                    "synchronized weather scenarios remain confirmatory"
                ),
                "configuration_reduction": (
                    "median within each technology family across its fixed "
                    "duration nodes, then median across technology families "
                    "within each country-year"
                ),
            }
        )
    return rows


def technology_summary(
    paired: pd.DataFrame,
    sample: str,
    countries: set[str] | None,
) -> pd.DataFrame:
    selected = (
        paired
        if countries is None
        else paired[paired["country"].isin(countries)]
    )
    per_unit = (
        selected.groupby(
            ["country", "year", "tech_id", "technology_en"],
            as_index=False,
        )[list(DELTA_FIELDS)]
        .median()
    )
    rows = []
    for (tech_id, technology), group in per_unit.groupby(
        ["tech_id", "technology_en"], sort=True
    ):
        for field in DELTA_FIELDS:
            values = group[field].dropna()
            rows.append(
                {
                    "sample": sample,
                    "tech_id": tech_id,
                    "technology_en": technology,
                    "metric": field,
                    "country_year_count": int(len(values)),
                    "country_year_median": float(values.median()),
                    "negative_country_year_n": int((values < 0).sum()),
                    "negative_country_year_fraction": float(
                        (values < 0).mean()
                    ),
                    "within_country_year_reduction": (
                        "median across the technology's fixed duration nodes"
                    ),
                }
            )
    return pd.DataFrame(rows)


def duration_summary(
    paired: pd.DataFrame,
    sample: str,
    countries: set[str] | None,
) -> pd.DataFrame:
    selected = (
        paired
        if countries is None
        else paired[paired["country"].isin(countries)]
    )
    per_unit = (
        selected.groupby(
            ["country", "year", "duration_h"],
            as_index=False,
        )[list(DELTA_FIELDS)]
        .median()
    )
    rows = []
    for duration, group in per_unit.groupby("duration_h", sort=True):
        for field in DELTA_FIELDS:
            values = group[field].dropna()
            rows.append(
                {
                    "sample": sample,
                    "duration_h": float(duration),
                    "metric": field,
                    "country_year_count": int(len(values)),
                    "country_year_median": float(values.median()),
                    "negative_country_year_n": int((values < 0).sum()),
                    "negative_country_year_fraction": float(
                        (values < 0).mean()
                    ),
                    "within_country_year_reduction": (
                        "median across technologies represented at the fixed "
                        "duration node"
                    ),
                }
            )
    return pd.DataFrame(rows)


def country_summary(country_year: pd.DataFrame) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "eligible_years": ("year", "nunique")
    }
    for field in DELTA_FIELDS:
        reduced = f"median_{field}"
        aggregations[f"median_{field}_across_years"] = (reduced, "median")
        aggregations[f"negative_year_fraction_{field}"] = (
            reduced,
            lambda values: float((values < 0).mean()),
        )
    return (
        country_year.groupby("country", as_index=False)
        .agg(**aggregations)
        .sort_values("country")
    )


def recharge_pair_table(chronology: pd.DataFrame) -> pd.DataFrame:
    value_fields = [
        "zero_day_recharge_window_share",
        "median_nonextreme_days_between_q75_days",
        "q25_nonextreme_days_between_q75_days",
        "mean_nonextreme_days_between_q75_days",
        "zero_surplus_recharge_window_share",
        "median_surplus_between_q75_days_pu_h",
        "q25_surplus_between_q75_days_pu_h",
        "annual_surplus_pu_h",
        "annual_residual_pu_h",
    ]
    paired = chronology[
        chronology["chronology"].isin(["dispersed", "clustered"])
    ].pivot(
        index=["country", "year"],
        columns="chronology",
        values=value_fields,
    )
    output = paired.xs("clustered", axis=1, level=1).add_suffix(
        "_clustered"
    )
    dispersed = paired.xs("dispersed", axis=1, level=1).add_suffix(
        "_dispersed"
    )
    output = output.join(dispersed)
    for field in value_fields:
        output[f"delta_{field}"] = (
            output[f"{field}_clustered"]
            - output[f"{field}_dispersed"]
        )
    return output.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--batch-id", default="energy_charts_155cy_strict50_4_168h"
    )
    args = parser.parse_args()

    aggregate = args.root / "model_adapter" / f"{args.batch_id}_aggregate"
    metrics = pd.read_csv(aggregate / "all_configuration_screen_metrics.csv")
    paired = paired_configuration_changes(metrics)
    if int(paired["config_id"].nunique()) != 50:
        raise RuntimeError(
            f"Expected 50 configurations; found "
            f"{paired['config_id'].nunique()}"
        )
    if float(paired["duration_h"].max()) > 168:
        raise RuntimeError("Structural duration extrapolations entered screen")
    country_year = reduce_country_year(paired)

    manifest = json.loads(
        (
            args.root / "metadata" / "energy_charts_panel_manifest.json"
        ).read_text(encoding="utf-8")
    )
    balanced = set(manifest["sample"]["balanced_eu_countries"])
    expanded_summary = sample_summary(
        country_year, "expanded_europe_17c_155cy", None
    )
    balanced_summary = sample_summary(
        country_year, "balanced_eu_9c_90cy", balanced
    )
    sample_summary_frame = pd.DataFrame(
        expanded_summary + balanced_summary
    )
    consistency = (
        sample_summary_frame.pivot(
            index="metric",
            columns="sample",
            values=[
                "country_year_mean",
                "country_year_median",
                "negative_country_year_fraction",
            ],
        )
        .reset_index()
    )
    consistency.columns = [
        "_".join(str(item) for item in column if str(item))
        if isinstance(column, tuple)
        else str(column)
        for column in consistency.columns
    ]
    consistency["mean_direction_consistent"] = (
        np.sign(
            consistency[
                "country_year_mean_balanced_eu_9c_90cy"
            ]
        )
        == np.sign(
            consistency[
                "country_year_mean_expanded_europe_17c_155cy"
            ]
        )
    )
    consistency["median_direction_consistent"] = (
        np.sign(
            consistency[
                "country_year_median_balanced_eu_9c_90cy"
            ]
        )
        == np.sign(
            consistency[
                "country_year_median_expanded_europe_17c_155cy"
            ]
        )
    )

    technology = pd.concat(
        [
            technology_summary(
                paired, "expanded_europe_17c_155cy", None
            ),
            technology_summary(
                paired, "balanced_eu_9c_90cy", balanced
            ),
        ],
        ignore_index=True,
    )
    duration = pd.concat(
        [
            duration_summary(
                paired, "expanded_europe_17c_155cy", None
            ),
            duration_summary(
                paired, "balanced_eu_9c_90cy", balanced
            ),
        ],
        ignore_index=True,
    )
    country = country_summary(country_year)

    panel = pd.read_csv(
        args.root
        / "processed"
        / "energy_charts_europe_hourly_eligible_2015_2024.csv.gz",
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
    recharge_rows: list[dict[str, Any]] = []
    for _, group in panel.groupby(["country", "year"], sort=True):
        recharge_rows.extend(chronology_recharge_metrics(group))
    recharge = pd.DataFrame(recharge_rows)
    recharge_paired = recharge_pair_table(recharge)

    boundaries = pd.read_csv(aggregate / "all_boundary_diagnostics.csv")
    boundaries = parse_unit_id(boundaries)
    boundary_rows = []
    for sample, countries in (
        ("expanded_europe_17c_155cy", None),
        ("balanced_eu_9c_90cy", balanced),
    ):
        selected = (
            boundaries
            if countries is None
            else boundaries[boundaries["country"].isin(countries)]
        )
        for chronology, group in selected.groupby("chronology"):
            boundary_rows.append(
                {
                    "sample": sample,
                    "chronology": chronology,
                    "country_year_count": int(
                        group[["country", "year"]]
                        .drop_duplicates()
                        .shape[0]
                    ),
                    "point_screen_complete_valley_n": int(
                        group["point_screen_complete_valley"].map(as_bool).sum()
                    ),
                    "point_screen_partial_negative_interval_n": int(
                        group["negative_interval_start_h"].notna().sum()
                    ),
                    "point_screen_h_T_identified_n": int(
                        group["h_T_h"].notna().sum()
                    ),
                    "point_screen_h_I_proxy_identified_n": int(
                        group["h_I_proxy_h"].notna().sum()
                    ),
                    "point_screen_any_partial_boundary_n": int(
                        (
                            group[
                                [
                                    "negative_interval_start_h",
                                    "h_T_h",
                                    "h_I_proxy_h",
                                ]
                            ]
                            .notna()
                            .any(axis=1)
                            & ~group[
                                "point_screen_complete_valley"
                            ].map(as_bool)
                        ).sum()
                    ),
                    "h_T_right_censored_above_168h_n": int(
                        (
                            group["h_T_censoring_status"]
                            == "right_censored_above_168h"
                        ).sum()
                    ),
                    "h_I_not_identified_n": int(
                        group["h_I_proxy_h"].isna().sum()
                    ),
                    "confirmatory_complete_valley_n": 0,
                    "confirmatory_partial_valley_n": 0,
                    "confirmatory_rule_status": (
                        "not evaluated: networked ELCC intervals and "
                        "hierarchical joint sign criteria are outside this "
                        "historical screen"
                    ),
                }
            )
    valley = pd.DataFrame(boundary_rows)

    research = args.root / "artifacts" / "research"
    paired.to_csv(
        aggregate / "strict50_configuration_paired_changes.csv", index=False
    )
    country_year.to_csv(
        research / "energy_charts_strict50_country_year_summary.csv",
        index=False,
    )
    country.to_csv(
        research / "energy_charts_strict50_country_summary.csv", index=False
    )
    sample_summary_frame.to_csv(
        research
        / "energy_charts_strict50_expanded_vs_balanced_summary.csv",
        index=False,
    )
    consistency.to_csv(
        research / "energy_charts_strict50_sample_direction_consistency.csv",
        index=False,
    )
    technology.to_csv(
        research / "energy_charts_strict50_technology_direction_summary.csv",
        index=False,
    )
    duration.to_csv(
        research / "energy_charts_strict50_duration_direction_summary.csv",
        index=False,
    )
    recharge.to_csv(
        research / "energy_charts_chronology_recharge_window_metrics.csv",
        index=False,
    )
    recharge_paired.to_csv(
        research
        / "energy_charts_recharge_window_country_year_changes.csv",
        index=False,
    )
    valley.to_csv(
        research / "energy_charts_strict50_valley_status.csv", index=False
    )

    efc = sample_summary_frame[
        sample_summary_frame["metric"] == "delta_efc"
    ].set_index("sample")
    turnover = sample_summary_frame[
        sample_summary_frame["metric"]
        == "delta_turnover_benefit_cny_per_forced_kw_year"
    ].set_index("sample")
    recharge_zero = recharge_paired[
        "delta_zero_day_recharge_window_share"
    ]
    annual_surplus_error = float(
        recharge_paired["delta_annual_surplus_pu_h"].abs().max()
    )
    response_summary_path = (
        research / "energy_charts_actual_supply_response_sample_summary.csv"
    )
    response_summary = (
        pd.read_csv(response_summary_path)
        if response_summary_path.exists()
        else pd.DataFrame()
    )
    primary_gas = pd.DataFrame()
    primary_response = pd.DataFrame()
    if not response_summary.empty:
        primary_response = response_summary[
            (
                response_summary["scarcity_label"]
                == "actual_residual_q75_primary"
            )
        ].copy()
        primary_gas = primary_response[
            primary_response["metric"]
            == (
                "multiday_minus_isolated_fossil_gas_mwh"
                "_per_actual_residual_mwh"
            )
        ].set_index("sample")
    lines = [
        "# Energy-Charts 4–168 h historical screen decision memo",
        "",
        "## Empirical units",
        "",
        (
            "The expanded sample contains 17 countries and 155 country-years. "
            "The balanced EU sample contains 9 current EU countries and 90 "
            "country-years. Each metric is first reduced within technology "
            "across its fixed duration nodes and then across technology "
            "families within a country-year. Configuration nodes and hourly "
            "rows are not counted as independent empirical observations. "
            "Intervals are reported under country-hierarchical resampling, "
            "synchronized calendar-year block resampling and two-way "
            "country/year clustering; the envelope is retained because "
            "countries share real weather years."
        ),
        "",
        "## Direction",
        "",
        (
            "For EFC, clustered-minus-dispersed country-year medians are "
            f"negative in {int(efc.loc['expanded_europe_17c_155cy', 'negative_country_year_n'])}/"
            f"{int(efc.loc['expanded_europe_17c_155cy', 'country_year_count'])} "
            "expanded country-years and "
            f"{int(efc.loc['balanced_eu_9c_90cy', 'negative_country_year_n'])}/"
            f"{int(efc.loc['balanced_eu_9c_90cy', 'country_year_count'])} "
            "balanced-EU country-years."
        ),
        (
            "For turnover benefit, the corresponding negative counts are "
            f"{int(turnover.loc['expanded_europe_17c_155cy', 'negative_country_year_n'])}/"
            f"{int(turnover.loc['expanded_europe_17c_155cy', 'country_year_count'])} "
            "and "
            f"{int(turnover.loc['balanced_eu_9c_90cy', 'negative_country_year_n'])}/"
            f"{int(turnover.loc['balanced_eu_9c_90cy', 'country_year_count'])}."
        ),
        (
            "The signs of the country-year mean and median are compared "
            "explicitly in "
            "`energy_charts_strict50_sample_direction_consistency.csv`."
        ),
        "",
        "## Recharge opportunity",
        "",
        (
            "Clustering changes the distribution of intervals between q75 "
            "scarcity days. The country-year median change in the share of "
            "successive q75 pairs with no intervening non-scarcity day is "
            f"{recharge_zero.median():.3f}. Annual renewable surplus is "
            "conserved by construction; the maximum absolute paired "
            f"difference is {annual_surplus_error:.3e} p.u.-h."
        ),
        "",
        "## Valley status",
        "",
        (
            "This historical screen tests repeat-use compression across the "
            "evidence-supported 4–168 h technology-duration catalogue. It is "
            "not a networked ELCC confirmation test. Complete mid-duration "
            "valley status remains assigned to the ERAA confirmatory model."
        ),
    ]
    if not primary_gas.empty:
        expanded_gas = primary_gas.loc["expanded_europe_17c_155cy"]
        balanced_gas = primary_gas.loc["balanced_eu_9c_90cy"]
        lines.extend(
            [
                "",
                "## Actual system-response relevance",
                "",
                (
                    "Using each country-year's observed load-minus-observed-"
                    "wind-and-solar q75 days, aggregate fossil-gas response "
                    "per unit of actual residual demand is higher on natural "
                    "multi-day runs than on isolated q75 days in "
                    f"{round(expanded_gas['country_year_count_with_contrast'] * expanded_gas['positive_country_year_fraction'])}/"
                    f"{int(expanded_gas['country_year_count_with_contrast'])} "
                    "expanded-sample country-years and "
                    f"{round(balanced_gas['country_year_count_with_contrast'] * balanced_gas['positive_country_year_fraction'])}/"
                    f"{int(balanced_gas['country_year_count_with_contrast'])} "
                    "balanced-EU country-years. Median intensity differences "
                    f"are {expanded_gas['country_year_median']:.4f} and "
                    f"{balanced_gas['country_year_median']:.4f}, respectively. "
                    "The envelopes spanning country-hierarchical, synchronized "
                    "year-block and two-way country/year cluster intervals "
                    "for the mean are "
                    f"[{expanded_gas['dependence_robust_interval_envelope_low']:.4f}, "
                    f"{expanded_gas['dependence_robust_interval_envelope_high']:.4f}] "
                    "and "
                    f"[{balanced_gas['dependence_robust_interval_envelope_low']:.4f}, "
                    f"{balanced_gas['dependence_robust_interval_envelope_high']:.4f}]."
                ),
                (
                    "This is external descriptive relevance, not the paired "
                    "intervention. Fossil gas is an aggregate generation "
                    "response across plant and operating roles and is never "
                    "called gas peaking demand. The scaled-residual q75 "
                    "classification remains a separately reported "
                    "forward-looking sensitivity."
                ),
                (
                    "Observed cross-border exchange is retained as a system-"
                    "balance control variable. It is not used to construct "
                    "scarcity demand, renewable charging input or the paired "
                    "chronology intervention."
                ),
            ]
        )
        response_metrics = {
            "net imports": (
                "multiday_minus_isolated_net_import_mwh"
                "_per_actual_residual_mwh"
            ),
            "reservoir hydro": (
                "multiday_minus_isolated_reservoir_hydro_mwh"
                "_per_actual_residual_mwh"
            ),
            "pumped-storage discharge": (
                "multiday_minus_isolated_pumped_discharge_mwh"
                "_per_actual_residual_mwh"
            ),
        }
        lines.extend(["", "### Other observed responses", ""])
        for label, metric in response_metrics.items():
            rows = primary_response[
                primary_response["metric"] == metric
            ].set_index("sample")
            if rows.empty:
                continue
            expanded_row = rows.loc["expanded_europe_17c_155cy"]
            balanced_row = rows.loc["balanced_eu_9c_90cy"]
            lines.append(
                f"- {label}: median multi-day-minus-isolated response "
                f"intensity is {expanded_row['country_year_median']:.4f} "
                f"across {int(expanded_row['country_year_count_with_contrast'])} "
                "expanded-sample country-years and "
                f"{balanced_row['country_year_median']:.4f} across "
                f"{int(balanced_row['country_year_count_with_contrast'])} "
                "balanced-EU country-years."
            )
        response_qa_path = (
            research / "energy_charts_actual_supply_response_qa.csv"
        )
        if response_qa_path.exists():
            response_qa = pd.read_csv(response_qa_path)
            confirmed = int(
                response_qa[
                    "cross_border_sign_validation_status"
                ]
                .eq(
                    "official_positive_import_confirmed_by_lower_balance_MAE"
                )
                .sum()
            )
            lines.append(
                (
                    f"- Cross-border sign: the official power-chart "
                    f"positive-import convention is independently favoured "
                    f"by the hourly balance audit in {confirmed}/"
                    f"{len(response_qa)} country-years; unvalidated years are "
                    "excluded from net-import contrasts while their raw "
                    "signed field remains archived."
                )
            )
    lines.extend(
        [
            "",
            "## Day-ahead-price supplement",
            "",
            (
                "No price result is admitted to this decision memo. The first "
                "price download used the `country` query parameter, whereas "
                "the Energy-Charts `/price` endpoint requires `bzn` and "
                "otherwise defaults to DE-LU. All 54 downloaded files are "
                "therefore quarantined and all derived price tables are "
                "invalid for analysis. A future price supplement must use an "
                "audited country-to-bidding-zone mapping and remain an "
                "Extended Data sensitivity rather than part of the physical "
                "mechanism sample."
            ),
        ]
    )
    memo = research / "ENERGY_CHARTS_STRICT50_SCREEN_DECISION_MEMO.md"
    memo.write_text("\n".join(lines) + "\n", encoding="utf-8")
    combined_memo = (
        research
        / "ENERGY_CHARTS_MECHANISM_AND_RESPONSE_DECISION_MEMO.md"
    )
    combined_memo.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(sample_summary_frame.to_string(index=False))
    print(memo)


if __name__ == "__main__":
    main()
