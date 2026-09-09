#!/usr/bin/env python3
"""Audit whether observed aggregate fossil-gas dispatch tracks peaking need.

Two descriptive checks are computed independently from the hourly panel:

1. within-country-year Pearson correlation between observed ``Fossil gas`` and
   contemporaneous signed residual load (system load minus observed wind and
   solar);
2. actual gas response on natural q75 residual-load events, contrasting
   isolated one-day events with runs of at least three q75 days.

These checks support the relevance of aggregate fossil-gas dispatch as a
system task. They do not identify OCGT/CCGT units or turn the paired storage
screen into a causal estimate of historical dispatch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import summarize_energy_charts_strict50_screen as shared


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PANEL = (
    ROOT
    / "processed"
    / "energy_charts_observed_gas_task_panel_2015_2024.csv.gz"
)
RESEARCH = ROOT / "artifacts" / "research"
OUTPUT = RESEARCH / "energy_charts_observed_gas_task_validity_country_year.csv"
SUMMARY = RESEARCH / "energy_charts_observed_gas_task_validity_summary.csv"
QA = RESEARCH / "energy_charts_observed_gas_task_validity_qa.json"
EXISTING = (
    RESEARCH / "energy_charts_natural_scarcity_supply_response_contrasts.csv"
)
FLOOR_SAMPLE = (
    RESEARCH / "energy_charts_observed_gas_task_prespecified_sample.csv"
)
EU27 = {
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE",
    "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT",
    "RO", "SK", "SI", "ES", "SE",
}


def as_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def run_lengths(mask: pd.Series) -> pd.Series:
    groups = mask.ne(mask.shift(fill_value=False)).cumsum()
    return mask.groupby(groups).transform("sum").where(mask, 0).astype(int)


def audit_country_year(group: pd.DataFrame) -> dict[str, Any]:
    country = str(group["country"].iloc[0])
    year = int(group["year"].iloc[0])
    hourly_correlation = float(
        group["observed_gas_peaking_task_mw"].corr(
            group["signed_actual_residual_load_mw"]
        )
    )
    daily = (
        group.groupby("date", as_index=False)
        .agg(
            actual_residual_mwh=("actual_residual_load_mw", "sum"),
            fossil_gas_mwh=("observed_gas_peaking_task_mw", "sum"),
        )
        .sort_values("date")
    )
    threshold = float(daily["actual_residual_mwh"].quantile(0.75))
    daily["is_q75"] = daily["actual_residual_mwh"] >= threshold
    daily["q75_run_length_d"] = run_lengths(daily["is_q75"])
    isolated = daily[daily["is_q75"] & daily["q75_run_length_d"].eq(1)]
    multiday = daily[daily["is_q75"] & daily["q75_run_length_d"].ge(3)]
    if isolated.empty or multiday.empty:
        raise RuntimeError(
            f"{country}-{year} lacks isolated or >=3 d q75 events"
        )
    isolated_intensity = float(
        isolated["fossil_gas_mwh"].sum()
        / isolated["actual_residual_mwh"].sum()
    )
    multiday_intensity = float(
        multiday["fossil_gas_mwh"].sum()
        / multiday["actual_residual_mwh"].sum()
    )
    return {
        "country": country,
        "year": year,
        "hour_count": int(len(group)),
        "pearson_r_fossil_gas_vs_signed_actual_residual_load": (
            hourly_correlation
        ),
        "q75_actual_residual_threshold_mwh_per_day": threshold,
        "q75_day_count": int(daily["is_q75"].sum()),
        "isolated_q75_day_count": int(len(isolated)),
        "multiday_3plus_q75_day_count": int(len(multiday)),
        "isolated_fossil_gas_mwh_per_actual_residual_mwh": (
            isolated_intensity
        ),
        "multiday_fossil_gas_mwh_per_actual_residual_mwh": (
            multiday_intensity
        ),
        "multiday_minus_isolated_fossil_gas_response_intensity": (
            multiday_intensity - isolated_intensity
        ),
        "isolated_mean_fossil_gas_mwh_per_day": float(
            isolated["fossil_gas_mwh"].mean()
        ),
        "multiday_mean_fossil_gas_mwh_per_day": float(
            multiday["fossil_gas_mwh"].mean()
        ),
        "multiday_minus_isolated_fossil_gas_mwh_per_day": float(
            multiday["fossil_gas_mwh"].mean()
            - isolated["fossil_gas_mwh"].mean()
        ),
    }


def summarize_sample(
    frame: pd.DataFrame, name: str
) -> list[dict[str, Any]]:
    fields = (
        "pearson_r_fossil_gas_vs_signed_actual_residual_load",
        "multiday_minus_isolated_fossil_gas_response_intensity",
        "multiday_minus_isolated_fossil_gas_mwh_per_day",
    )
    rows = []
    for field in fields:
        values = frame[field].dropna()
        country_low, country_high = shared.hierarchical_bootstrap_mean(
            frame, field
        )
        year_low, year_high = shared.calendar_year_block_bootstrap_mean(
            frame, field
        )
        two_low, two_high, two_se = shared.two_way_cluster_mean_interval(
            frame, field
        )
        rows.append(
            {
                "sample": name,
                "country_count": int(frame["country"].nunique()),
                "country_year_count": int(len(frame)),
                "metric": field,
                "mean": float(values.mean()),
                "median": float(values.median()),
                "q25": float(values.quantile(0.25)),
                "q75": float(values.quantile(0.75)),
                "positive_n": int((values > 0).sum()),
                "positive_fraction": float((values > 0).mean()),
                "dependence_robust_interval_envelope_low": min(
                    country_low, year_low, two_low
                ),
                "dependence_robust_interval_envelope_high": max(
                    country_high, year_high, two_high
                ),
                "country_hierarchical_bootstrap_mean_ci_low": country_low,
                "country_hierarchical_bootstrap_mean_ci_high": country_high,
                "synchronized_year_block_mean_ci_low": year_low,
                "synchronized_year_block_mean_ci_high": year_high,
                "two_way_country_year_cluster_mean_ci_low": two_low,
                "two_way_country_year_cluster_mean_ci_high": two_high,
                "two_way_country_year_cluster_standard_error": two_se,
            }
        )
    return rows


def main() -> None:
    use = [
        "timestamp_utc",
        "country",
        "year",
        "system_load_mw",
        "observed_wind_mw",
        "observed_solar_mw",
        "observed_gas_peaking_task_mw",
        "load_mw",
        "gas_task_eligible",
        "gas_task_material_share_ge_1pct",
    ]
    panel = pd.read_csv(PANEL, usecols=use)
    panel = panel[
        as_bool(panel["gas_task_eligible"])
        & as_bool(panel["gas_task_material_share_ge_1pct"])
    ].copy()
    panel["timestamp_utc"] = pd.to_datetime(
        panel["timestamp_utc"], utc=True, errors="raise"
    )
    panel["date"] = panel["timestamp_utc"].dt.floor("D")
    panel["signed_actual_residual_load_mw"] = (
        panel["system_load_mw"]
        - panel["observed_wind_mw"]
        - panel["observed_solar_mw"]
    )
    panel["actual_residual_load_mw"] = panel[
        "signed_actual_residual_load_mw"
    ].clip(lower=0.0)
    rows = [
        audit_country_year(group)
        for _, group in panel.groupby(["country", "year"], sort=True)
    ]
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT, index=False)

    all_pairs = set(
        result[["country", "year"]].itertuples(index=False, name=None)
    )
    balanced = {
        country
        for country, group in result.groupby("country")
        if country in EU27 and set(group["year"]) == set(range(2015, 2025))
    }
    floor_sample = pd.read_csv(FLOOR_SAMPLE)
    floor_pairs = set(
        floor_sample.loc[
            as_bool(floor_sample["main_prespecified_sample"])
            & as_bool(floor_sample["above_floor_material_share_ge_1pct"]),
            ["country", "year"],
        ].itertuples(index=False, name=None)
    )
    sample_frames = {
        "prespecified_15c_132cy": result,
        "eu_only_14c_127cy": result[result["country"].isin(EU27)],
        "balanced_eu_8c_80cy": result[result["country"].isin(balanced)],
        "floor_material_15c_118cy": result[
            pd.MultiIndex.from_frame(result[["country", "year"]]).isin(
                floor_pairs
            )
        ],
    }
    summary = pd.DataFrame(
        [
            row
            for name, frame in sample_frames.items()
            for row in summarize_sample(frame, name)
        ]
    )
    summary.to_csv(SUMMARY, index=False)

    crosscheck_max = np.nan
    if EXISTING.exists():
        existing = pd.read_csv(EXISTING)
        existing = existing[
            existing["scarcity_label"].eq("actual_residual_q75_primary")
        ][
            [
                "country",
                "year",
                (
                    "multiday_minus_isolated_fossil_gas_mwh"
                    "_per_actual_residual_mwh"
                ),
            ]
        ]
        merged = result.merge(
            existing,
            on=["country", "year"],
            how="left",
            validate="one_to_one",
        )
        existing_field = (
            "multiday_minus_isolated_fossil_gas_mwh"
            "_per_actual_residual_mwh"
        )
        crosscheck_max = float(
            (
                merged[
                    "multiday_minus_isolated_fossil_gas_response_intensity"
                ]
                - merged[existing_field]
            )
            .abs()
            .max()
        )

    correlation = result[
        "pearson_r_fossil_gas_vs_signed_actual_residual_load"
    ]
    response = result[
        "multiday_minus_isolated_fossil_gas_response_intensity"
    ]
    checks = {
        "country_year_count_132": len(result) == 132,
        "country_count_15": result["country"].nunique() == 15,
        "all_country_years_have_8760_hours": bool(
            (result["hour_count"] == 8760).all()
        ),
        "all_correlations_finite": bool(np.isfinite(correlation).all()),
        "all_correlations_positive": bool((correlation > 0).all()),
        "adapter_load_exactly_equals_observed_fossil_gas": bool(
            np.allclose(
                panel["load_mw"],
                panel["observed_gas_peaking_task_mw"],
                rtol=0.0,
                atol=0.0,
            )
        ),
        "all_country_years_have_isolated_and_multiday_events": bool(
            (result["isolated_q75_day_count"] > 0).all()
            & (result["multiday_3plus_q75_day_count"] > 0).all()
        ),
        "existing_independent_response_crosscheck_matches": bool(
            np.isfinite(crosscheck_max) and crosscheck_max <= 1e-12
        ),
        "balanced_country_count_8": len(balanced) == 8,
        "floor_material_pair_count_118": len(floor_pairs & all_pairs) == 118,
    }
    qa = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "main_sample": {
            "countries": 15,
            "country_years": 132,
            "country_hours": 132 * 8760,
        },
        "hourly_correlation": {
            "definition": (
                "within-country-year Pearson r between observed aggregate "
                "Fossil gas and signed actual residual load = system load - "
                "observed wind - observed solar"
            ),
            "mean": float(correlation.mean()),
            "median": float(correlation.median()),
            "positive_n": int((correlation > 0).sum()),
            "minimum": float(correlation.min()),
            "maximum": float(correlation.max()),
        },
        "natural_event_response": {
            "definition": (
                "q75 daily non-negative actual residual-load days; "
                "hourly residual is max(system load - wind - solar, 0); "
                "ratio-of-sums "
                "gas/actual-residual intensity on >=3 d runs minus isolated "
                "one-day events; adjacent 2 d events excluded"
            ),
            "mean": float(response.mean()),
            "median": float(response.median()),
            "positive_n": int((response > 0).sum()),
            "negative_n": int((response < 0).sum()),
            "minimum": float(response.min()),
            "maximum": float(response.max()),
        },
        "independent_existing_response_crosscheck_max_abs_error": (
            crosscheck_max
        ),
        "interpretation": (
            "The two checks establish empirical relevance of aggregate gas "
            "dispatch to residual-load stress. They do not identify plant "
            "technology, causal peaking need, or a pure flexibility gap."
        ),
    }
    QA.write_text(json.dumps(qa, indent=2), encoding="utf-8")
    if qa["status"] != "pass":
        raise RuntimeError(json.dumps(qa, indent=2))
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
