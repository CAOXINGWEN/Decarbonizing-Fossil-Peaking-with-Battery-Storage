#!/usr/bin/env python3
"""Audit and summarize the observed-fossil-gas paired duration screens.

The empirical unit is a country-year. Configuration nodes are reduced within
country-year before sample-level direction counts or uncertainty intervals are
formed. The raw aggregate ``Fossil gas`` task is primary; the month-hour q10
above-floor task is a prespecified sensitivity for persistent CHP/baseload
output. Neither series identifies OCGT or CCGT units.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import summarize_energy_charts_observed_gas_key_screens as key_shared
import summarize_energy_charts_strict50_screen as shared


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL = ROOT / "model_adapter"
RESEARCH = ROOT / "artifacts" / "research"
RAW_PANEL = (
    ROOT
    / "processed"
    / "energy_charts_observed_gas_task_panel_2015_2024.csv.gz"
)
FLOOR_PANEL = (
    ROOT
    / "processed"
    / "energy_charts_observed_gas_above_floor_task_panel_2015_2024.csv.gz"
)
KEY_CONFIGS = ("li_lfp_4h", "caes_24h", "h2_cavern_72h", "h2_cavern_168h")
EU27 = {
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE",
    "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT",
    "RO", "SK", "SI", "ES", "SE",
}
FLOOR_SAMPLE = (
    RESEARCH / "energy_charts_observed_gas_task_prespecified_sample.csv"
)
SUPPLY_RESPONSE_QA = (
    RESEARCH / "energy_charts_actual_supply_response_qa.csv"
)
VALIDITY_QA = (
    RESEARCH / "energy_charts_observed_gas_task_validity_qa.json"
)
EXPECTED_RAW_PANEL_SHA256 = (
    "5f2e8ab49120050fac2ef8bd1b11e20a58863442631a8d9baccf5b924f0bf3ab"
)
EXPECTED_FLOOR_PANEL_SHA256 = (
    "b746c3d62ff41965b29a5851031be53112c2f58b7982386ca3ae52f3703ce984"
)
EXPECTED_SAMPLE_COUNTS = {
    "prespecified_15c_132cy": 132,
    "eu_only_14c_127cy": 127,
    "balanced_eu_8c_80cy": 80,
    "floor_material_15c_118cy": 118,
    "fully_observed_gas_14c_76cy": 76,
}


def as_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantile_summary(values: pd.Series) -> dict[str, Any]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return {
        "n": int(len(clean)),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "q25": float(clean.quantile(0.25)),
        "q75": float(clean.quantile(0.75)),
        "negative_n": int((clean < 0).sum()),
        "negative_fraction": float((clean < 0).mean()),
    }


def infer_balanced_countries(panel: pd.DataFrame) -> set[str]:
    """Return EU countries with all ten prespecified country-years.

    ``summarize_energy_charts_strict50_screen`` does not expose the EU set, so
    keep the ISO membership explicit here rather than accidentally admitting
    GB to the balanced-EU sensitivity.
    """

    pairs = panel[["country", "year"]].drop_duplicates()
    all_years = set(range(2015, 2025))
    return {
        str(country)
        for country, group in pairs.groupby("country")
        if str(country) in EU27
        and set(group["year"].astype(int)) == all_years
    }


def filter_empirical_sample(
    frame: pd.DataFrame,
    countries: set[str] | None = None,
    pairs: set[tuple[str, int]] | None = None,
) -> pd.DataFrame:
    selected = frame
    if countries is not None:
        selected = selected[selected["country"].isin(countries)]
    if pairs is not None:
        index = pd.MultiIndex.from_frame(selected[["country", "year"]])
        selected = selected[index.isin(pairs)]
    return selected.copy()


def load_panel_sample(path: Path, flags: tuple[str, ...]) -> pd.DataFrame:
    use = ["country", "year", "load_mw", *flags]
    frame = pd.read_csv(path, usecols=use)
    keep = pd.Series(True, index=frame.index)
    for flag in flags:
        keep &= as_bool(frame[flag])
    return frame.loc[keep, ["country", "year", "load_mw"]].copy()


def chronology_strength(
    chronology: pd.DataFrame, task_variant: str
) -> pd.DataFrame:
    identifiers = ["unit_id", "year"]
    fields = [
        "extreme_days",
        "mean_extreme_run_d",
        "max_extreme_run_d",
        "share_extreme_days_in_runs_ge3",
    ]
    paired = chronology[
        chronology["chronology"].isin(["dispersed", "clustered"])
    ].pivot(index=identifiers, columns="chronology", values=fields)
    rows: list[dict[str, Any]] = []
    for field in fields:
        low = paired[(field, "dispersed")]
        high = paired[(field, "clustered")]
        values = high - low
        rows.append(
            {
                "task_variant": task_variant,
                "metric": field,
                **{
                    f"dispersed_{key}": value
                    for key, value in quantile_summary(low).items()
                },
                **{
                    f"clustered_{key}": value
                    for key, value in quantile_summary(high).items()
                },
                **{
                    f"delta_clustered_minus_dispersed_{key}": value
                    for key, value in quantile_summary(values).items()
                },
            }
        )
    return pd.DataFrame(rows)


def key_configuration_summary(
    paired: pd.DataFrame,
    task_variant: str,
    sample_name: str,
    countries: set[str] | None,
    pairs: set[tuple[str, int]] | None = None,
) -> pd.DataFrame:
    selected = paired[paired["config_id"].isin(KEY_CONFIGS)].copy()
    selected = filter_empirical_sample(selected, countries, pairs)
    rows = []
    for config_id, group in selected.groupby("config_id", sort=True):
        for metric in shared.DELTA_FIELDS:
            stats = quantile_summary(group[metric])
            rows.append(
                {
                    "task_variant": task_variant,
                    "sample": sample_name,
                    "config_id": config_id,
                    "metric": metric,
                    **stats,
                }
            )
    return pd.DataFrame(rows)


def absolute_key_summary(
    metrics: pd.DataFrame,
    task_variant: str,
    sample_name: str,
    countries: set[str] | None,
    pairs: set[tuple[str, int]] | None = None,
) -> pd.DataFrame:
    selected = shared.parse_unit_id(
        metrics[
            metrics["config_id"].isin(KEY_CONFIGS)
            & metrics["chronology"].isin(["dispersed", "clustered"])
        ].copy()
    )
    selected = filter_empirical_sample(selected, countries, pairs)
    fields = {
        "equivalent_full_cycles": (
            "expected_equivalent_full_cycles_output_basis"
        ),
        "joint_incremental_system_benefit_cny_per_forced_kw_year": (
            "joint_signed_benefit_cny_per_forced_kw_year"
        ),
        "turnover_only_benefit_cny_per_forced_kw_year": (
            "turnover_signed_benefit_cny_per_forced_kw_year"
        ),
        "charge_mwh_per_mean_task_mw_year": (
            "expected_charge_mwh_per_mean_mw_year"
        ),
        "discharge_mwh_per_mean_task_mw_year": (
            "expected_discharge_mwh_per_mean_mw_year"
        ),
        "gas_capacity_avoided_pu_mean_task": (
            "gas_capacity_avoided_pu_mean_load"
        ),
    }
    rows = []
    for (config_id, chronology), group in selected.groupby(
        ["config_id", "chronology"], sort=True
    ):
        for label, field in fields.items():
            stats = quantile_summary(group[field])
            rows.append(
                {
                    "task_variant": task_variant,
                    "sample": sample_name,
                    "config_id": config_id,
                    "chronology": chronology,
                    "metric": label,
                    **stats,
                }
            )
    return pd.DataFrame(rows)


def valley_summary(
    boundaries: pd.DataFrame,
    task_variant: str,
    sample_name: str,
    countries: set[str] | None,
    pairs: set[tuple[str, int]] | None = None,
) -> pd.DataFrame:
    selected = shared.parse_unit_id(boundaries.copy())
    selected = filter_empirical_sample(selected, countries, pairs)
    rows = []
    for chronology, group in selected.groupby("chronology", sort=True):
        rows.append(
            {
                "task_variant": task_variant,
                "sample": sample_name,
                "chronology": chronology,
                "country_year_count": int(
                    group[["country", "year"]].drop_duplicates().shape[0]
                ),
                "point_screen_complete_valley_n": int(
                    as_bool(group["point_screen_complete_valley"]).sum()
                ),
                "partial_negative_interval_n": int(
                    group["negative_interval_start_h"].notna().sum()
                ),
                "h_T_identified_n": int(group["h_T_h"].notna().sum()),
                "h_T_right_censored_above_168h_n": int(
                    (
                        group["h_T_censoring_status"]
                        == "right_censored_above_168h"
                    ).sum()
                ),
                "h_T_no_short_positive_turnover_branch_n": int(
                    (
                        group["h_T_censoring_status"]
                        == "not_identified_no_short_positive_turnover_branch"
                    ).sum()
                ),
                "h_I_proxy_identified_n": int(
                    group["h_I_proxy_h"].notna().sum()
                ),
                "h_I_right_censored_above_168h_n": int(
                    (
                        group["h_I_censoring_status"]
                        == "right_censored_above_168h"
                    ).sum()
                ),
                "h_I_no_two_node_negative_middle_interval_n": int(
                    (
                        group["h_I_censoring_status"]
                        == "not_identified_no_two_node_negative_middle_interval"
                    ).sum()
                ),
                "confirmatory_valley_n": 0,
                "interpretation": (
                    "historical single-country point screen; no networked "
                    "ELCC interval confirmation"
                ),
            }
        )
    return pd.DataFrame(rows)


def load_run(
    batch_id: str,
    task_variant: str,
    panel_path: Path,
    flags: tuple[str, ...],
    floor_material_pairs: set[tuple[str, int]],
    fully_observed_pairs: set[tuple[str, int]],
) -> dict[str, Any]:
    aggregate = MODEL / f"{batch_id}_aggregate"
    qa = json.loads(
        (aggregate / "AGGREGATE_QA_AND_SUMMARY.json").read_text(
            encoding="utf-8"
        )
    )
    metrics = pd.read_csv(aggregate / "all_configuration_screen_metrics.csv")
    paired = shared.paired_configuration_changes(metrics)
    country_year = shared.reduce_country_year(paired)
    boundaries = pd.read_csv(aggregate / "all_boundary_diagnostics.csv")
    chronology = pd.read_csv(
        aggregate / "all_chronology_conservation_qa.csv"
    )
    panel = load_panel_sample(panel_path, flags)
    balanced = infer_balanced_countries(panel)
    eu_countries = set(panel["country"].unique()) & EU27
    peak_ratio = (
        panel.groupby(["country", "year"])["load_mw"].max()
        / panel.groupby(["country", "year"])["load_mw"].mean()
    )

    sample_specs = (
        ("prespecified_15c_132cy", None, None),
        ("eu_only_14c_127cy", eu_countries, None),
        ("balanced_eu_8c_80cy", balanced, None),
        ("floor_material_15c_118cy", None, floor_material_pairs),
        ("fully_observed_gas_14c_76cy", None, fully_observed_pairs),
    )
    overall = []
    for sample_name, countries, pairs in sample_specs:
        selected_country_year = filter_empirical_sample(
            country_year, countries, pairs
        )
        rows = shared.sample_summary(
            selected_country_year, sample_name, None
        )
        for row in rows:
            row["task_variant"] = task_variant
        overall.extend(rows)
    overall_frame = pd.DataFrame(overall)
    key = pd.concat(
        [
            key_configuration_summary(
                paired, task_variant, sample_name, countries, pairs
            )
            for sample_name, countries, pairs in sample_specs
        ],
        ignore_index=True,
    )
    absolute = pd.concat(
        [
            absolute_key_summary(
                metrics, task_variant, sample_name, countries, pairs
            )
            for sample_name, countries, pairs in sample_specs
        ],
        ignore_index=True,
    )
    did_frames = []
    metrics_with_country_year = shared.parse_unit_id(metrics.copy())
    for sample_name, countries, pairs in sample_specs:
        selected = filter_empirical_sample(
            metrics_with_country_year[
                metrics_with_country_year["config_id"].isin(
                    ["li_lfp_4h", "caes_24h"]
                )
                & metrics_with_country_year["chronology"].isin(
                    ["dispersed", "clustered"]
                )
            ],
            countries,
            pairs,
        )
        rows = key_shared.did_rows(
            selected,
            spec_id=task_variant,
            task_variant=task_variant,
            quantile=float(qa["extreme_quantile"]),
            sample_name=sample_name,
        )
        did_frames.append(pd.DataFrame(rows))
    did = pd.concat(did_frames, ignore_index=True)
    valley = pd.concat(
        [
            valley_summary(
                boundaries, task_variant, sample_name, countries, pairs
            )
            for sample_name, countries, pairs in sample_specs
        ],
        ignore_index=True,
    )
    return {
        "qa": qa,
        "metrics": metrics,
        "paired": paired,
        "country_year": country_year,
        "chronology": chronology,
        "overall": overall_frame,
        "key": key,
        "absolute": absolute,
        "did": did,
        "valley": valley,
        "balanced": balanced,
        "eu_countries": eu_countries,
        "peak_ratio": peak_ratio,
        "aggregate": aggregate,
    }


def value(
    frame: pd.DataFrame,
    *,
    sample: str,
    metric: str,
    config_id: str | None = None,
    field: str = "median",
) -> float:
    selected = frame[
        (frame["sample"] == sample) & (frame["metric"] == metric)
    ]
    if config_id is not None:
        selected = selected[selected["config_id"] == config_id]
    if len(selected) != 1:
        raise RuntimeError(
            f"Expected one row for {sample}, {metric}, {config_id}; "
            f"found {len(selected)}"
        )
    return float(selected.iloc[0][field])


def did_record(
    frame: pd.DataFrame,
    *,
    sample: str,
    metric: str,
) -> pd.Series:
    selected = frame[
        (frame["sample"] == sample) & (frame["metric"] == metric)
    ]
    if len(selected) != 1:
        raise RuntimeError(
            f"Expected one DID row for {sample}, {metric}; "
            f"found {len(selected)}"
        )
    return selected.iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-batch-id",
        default=(
            "energy_charts_observed_gas_raw_132cy_strict50_4_168h_pairedonly"
        ),
    )
    parser.add_argument(
        "--floor-batch-id",
        default=(
            "energy_charts_observed_gas_floor_132cy_strict50_4_168h_pairedonly"
        ),
    )
    args = parser.parse_args()
    floor_sample = pd.read_csv(FLOOR_SAMPLE)
    validity = json.loads(VALIDITY_QA.read_text(encoding="utf-8"))
    raw_panel_sha256 = sha256(RAW_PANEL)
    floor_panel_sha256 = sha256(FLOOR_PANEL)
    floor_material_pairs = set(
        floor_sample.loc[
            as_bool(floor_sample["main_prespecified_sample"])
            & as_bool(floor_sample["above_floor_material_share_ge_1pct"]),
            ["country", "year"],
        ].itertuples(index=False, name=None)
    )
    main_pairs = set(
        floor_sample.loc[
            as_bool(floor_sample["main_prespecified_sample"]),
            ["country", "year"],
        ].itertuples(index=False, name=None)
    )
    supply_qa = pd.read_csv(SUPPLY_RESPONSE_QA)
    fully_observed_pairs = (
        set(
            supply_qa.loc[
                supply_qa["fossil_gas_mw_imputed_hours"].eq(0),
                ["country", "year"],
            ].itertuples(index=False, name=None)
        )
        & main_pairs
    )
    raw = load_run(
        args.raw_batch_id,
        "raw_observed_fossil_gas_main",
        RAW_PANEL,
        ("gas_task_eligible", "gas_task_material_share_ge_1pct"),
        floor_material_pairs,
        fully_observed_pairs,
    )
    floor = load_run(
        args.floor_batch_id,
        "month_hour_q10_above_floor_sensitivity",
        FLOOR_PANEL,
        ("gas_task_main_prespecified_sample",),
        floor_material_pairs,
        fully_observed_pairs,
    )
    runs = (raw, floor)
    if raw["balanced"] != floor["balanced"]:
        raise RuntimeError("Balanced-country set differs across task variants")

    overall = pd.concat([run["overall"] for run in runs], ignore_index=True)
    key = pd.concat([run["key"] for run in runs], ignore_index=True)
    absolute = pd.concat([run["absolute"] for run in runs], ignore_index=True)
    did = pd.concat([run["did"] for run in runs], ignore_index=True)
    valley = pd.concat([run["valley"] for run in runs], ignore_index=True)
    strength = pd.concat(
        [
            chronology_strength(
                raw["chronology"], "raw_observed_fossil_gas_main"
            ),
            chronology_strength(
                floor["chronology"],
                "month_hour_q10_above_floor_sensitivity",
            ),
        ],
        ignore_index=True,
    )

    prefix = RESEARCH / "energy_charts_observed_gas_task"
    overall.to_csv(f"{prefix}_overall_direction_summary.csv", index=False)
    key.to_csv(f"{prefix}_key_configuration_changes.csv", index=False)
    absolute.to_csv(f"{prefix}_key_configuration_absolute_metrics.csv", index=False)
    did.to_csv(f"{prefix}_caes24_lfp4_did.csv", index=False)
    valley.to_csv(f"{prefix}_valley_status.csv", index=False)
    strength.to_csv(f"{prefix}_clustering_intervention_strength.csv", index=False)

    maximum_realized_gas_capacity = max(
        float(run["metrics"]["gas_capacity_pu_mean_load"].max())
        for run in runs
    )
    maximum_bound = max(
        float(run["qa"]["maximum_gas_capacity_pu_mean_task"])
        for run in runs
    )
    checks = {
        "raw_batch_qa_passed": bool(raw["qa"]["passed"]),
        "floor_batch_qa_passed": bool(floor["qa"]["passed"]),
        "raw_country_year_count_132": (
            int(raw["qa"]["country_year_count"]) == 132
        ),
        "floor_country_year_count_132": (
            int(floor["qa"]["country_year_count"]) == 132
        ),
        "both_configuration_counts_50": all(
            int(run["qa"]["configuration_count_per_country_year"]) == 50
            for run in runs
        ),
        "both_vre_energy_share_zero": all(
            float(run["qa"]["vre_energy_share"]) == 0.0 for run in runs
        ),
        "both_profiles_have_zero_exogenous_surplus": all(
            float(run["chronology"]["annual_surplus_pu_h"].abs().max())
            <= 1e-12
            for run in runs
        ),
        "both_profiles_use_task_itself_as_residual": all(
            float(
                (
                    run["chronology"]["annual_residual_pu_h"]
                    - run["chronology"]["annual_load_pu_h"]
                )
                .abs()
                .max()
            )
            <= 1e-12
            for run in runs
        ),
        "raw_panel_sha256_matches_frozen_input": (
            raw_panel_sha256 == EXPECTED_RAW_PANEL_SHA256
        ),
        "floor_panel_sha256_matches_frozen_input": (
            floor_panel_sha256 == EXPECTED_FLOOR_PANEL_SHA256
        ),
        "aggregate_records_match_panel_sha256": (
            raw["qa"]["panel_sha256"] == raw_panel_sha256
            and floor["qa"]["panel_sha256"] == floor_panel_sha256
        ),
        "independent_task_validity_audit_passed": (
            validity["status"] == "pass"
        ),
        "adapter_load_exactly_equals_observed_fossil_gas": bool(
            validity["checks"][
                "adapter_load_exactly_equals_observed_fossil_gas"
            ]
        ),
        "balanced_sample_8_countries": len(raw["balanced"]) == 8,
        "eu_only_sample_14_countries": len(raw["eu_countries"]) == 14,
        "floor_material_diagnostic_118_country_years": (
            len(floor_material_pairs) == 118
        ),
        "fully_observed_gas_diagnostic_14_countries_76_country_years": (
            len(fully_observed_pairs) == 76
            and len({country for country, _ in fully_observed_pairs}) == 14
        ),
        "gas_capacity_bound_nonbinding": (
            maximum_realized_gas_capacity < maximum_bound - 1e-6
        ),
        "raw_and_floor_same_empirical_sample": set(
            raw["country_year"][["country", "year"]].itertuples(
                index=False, name=None
            )
        )
        == set(
            floor["country_year"][["country", "year"]].itertuples(
                index=False, name=None
            )
        ),
        "caes24_lfp4_did_has_all_five_samples_and_seven_metrics": all(
            run["did"]["sample"].nunique() == 5
            and run["did"]["metric"].nunique() == 7
            for run in runs
        ),
        "all_caes24_lfp4_country_years_have_four_complete_cells": all(
            (
                run["metrics"][
                    run["metrics"]["config_id"].isin(
                        ["li_lfp_4h", "caes_24h"]
                    )
                    & run["metrics"]["chronology"].isin(
                        ["dispersed", "clustered"]
                    )
                ]
                .pipe(shared.parse_unit_id)
                .groupby(["country", "year"])
                .size()
                .eq(4)
                .all()
            )
            for run in runs
        ),
        "all_caes24_lfp4_metric_cells_nonmissing": all(
            run["metrics"].loc[
                run["metrics"]["config_id"].isin(
                    ["li_lfp_4h", "caes_24h"]
                )
                & run["metrics"]["chronology"].isin(
                    ["dispersed", "clustered"]
                ),
                list(key_shared.FIELDS.values()),
            ]
            .notna()
            .all()
            .all()
            for run in runs
        ),
        "all_did_rows_retain_expected_country_year_count": all(
            run["did"].loc[
                run["did"]["sample"].eq(sample),
                "country_year_count",
            ]
            .eq(expected)
            .all()
            for run in runs
            for sample, expected in EXPECTED_SAMPLE_COUNTS.items()
        ),
    }
    qa = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "main_sample": {
            "countries": 15,
            "country_years": 132,
            "hours": 132 * 8760,
            "rule": (
                "complete raw hourly Fossil gas AND annual raw Fossil gas "
                "energy >=1% of annual system load"
            ),
        },
        "balanced_eu_sensitivity": {
            "countries": sorted(raw["balanced"]),
            "country_years": 80,
        },
        "eu_only_sample": {
            "countries": sorted(raw["eu_countries"]),
            "country_years": 127,
        },
        "floor_material_diagnostic": {
            "country_years": len(floor_material_pairs),
            "role": (
                "additional diagnostic only; never replaces the fixed "
                "132-country-year floor sensitivity"
            ),
        },
        "fully_observed_gas_diagnostic": {
            "countries": len(
                {country for country, _ in fully_observed_pairs}
            ),
            "country_years": len(fully_observed_pairs),
            "rule": "zero completed fossil-gas hours",
            "role": (
                "source-observation sensitivity only; does not replace the "
                "fixed 132-country-year main analysis"
            ),
        },
        "configuration_count": 50,
        "configuration_count_interpretation": (
            "49 alternative technology-duration configurations plus the "
            "4 h LFP reference node"
        ),
        "duration_range_h": [4, 168],
        "panel_sha256": {
            "raw_observed_fossil_gas_main": raw_panel_sha256,
            "month_hour_q10_above_floor_sensitivity": floor_panel_sha256,
        },
        "model_task_field": "load_mw",
        "observed_source_field": "observed_gas_peaking_task_mw",
        "task_field_identity": (
            "load_mw is an exact row-wise copy of the non-negative hourly "
            "Energy-Charts public_power aggregate Fossil gas series"
        ),
        "chronology_contract": (
            "same 365 intact daily gas-task blocks and same annual gas task; "
            "q75 daily task blocks dispersed or grouped in four-day runs "
            "within calendar segments"
        ),
        "charging_contract": (
            "vre_energy_share=0; no free renewable-surplus charging. "
            "Additional charging energy is produced by the model gas "
            "generator and therefore pays gas variable cost, round-trip "
            "losses and any endogenous extra charging-power capacity cost."
        ),
        "economic_calibration": (
            "common controlled v36 2035 CNY technology and gas costs; not "
            "observed country-specific European cost estimates"
        ),
        "maximum_task_peak_to_mean_ratio": {
            "raw": float(raw["peak_ratio"].max()),
            "above_floor": float(floor["peak_ratio"].max()),
        },
        "maximum_gas_capacity_bound_pu_mean_task": maximum_bound,
        "maximum_realized_gas_capacity_pu_mean_task": (
            maximum_realized_gas_capacity
        ),
        "task_interpretation": (
            "Observed aggregate Fossil gas dispatch is the task input. It "
            "contains OCGT, CCGT, CHP and other gas operation and is not "
            "unit-level peaker generation. The above-floor task is a "
            "sensitivity, not a unit classifier."
        ),
        "external_validity_diagnostics": {
            "role": (
                "Residual load is not a model task. It is used only to test "
                "whether observed aggregate gas dispatch responds to "
                "contemporaneous system stress."
            ),
            "hourly_gas_vs_signed_residual_load_correlation": (
                validity["hourly_correlation"]
            ),
            "natural_multiday_event_response": (
                validity["natural_event_response"]
            ),
        },
        "valley_interpretation": (
            "A point-screen complete valley is reported separately from a "
            "confirmatory networked ELCC valley; the latter remains zero by "
            "design in this historical adapter."
        ),
        "inference_estimand": (
            "equal-country-year mean. Country-hierarchical, synchronized-"
            "year and two-way cluster procedures all target this same point "
            "estimand; their interval union is reported as the dependence-"
            "robust envelope."
        ),
    }
    (RESEARCH / "energy_charts_observed_gas_task_screen_qa.json").write_text(
        json.dumps(qa, indent=2), encoding="utf-8"
    )
    if qa["status"] != "pass":
        raise RuntimeError(json.dumps(qa, indent=2))

    raw_overall = raw["overall"]
    floor_overall = floor["overall"]
    sample = "prespecified_15c_132cy"
    raw_efc = value(raw_overall, sample=sample, metric="delta_efc",
                    field="country_year_mean")
    floor_efc = value(floor_overall, sample=sample, metric="delta_efc",
                      field="country_year_mean")
    raw_joint = value(
        raw_overall,
        sample=sample,
        metric="delta_joint_benefit_cny_per_forced_kw_year",
        field="country_year_mean",
    )
    floor_joint = value(
        floor_overall,
        sample=sample,
        metric="delta_joint_benefit_cny_per_forced_kw_year",
        field="country_year_mean",
    )
    raw_caes_efc = value(
        raw["key"],
        sample=sample,
        metric="delta_efc",
        config_id="caes_24h",
    )
    floor_caes_efc = value(
        floor["key"],
        sample=sample,
        metric="delta_efc",
        config_id="caes_24h",
    )
    raw_caes_joint = value(
        raw["key"],
        sample=sample,
        metric="delta_joint_benefit_cny_per_forced_kw_year",
        config_id="caes_24h",
    )
    floor_caes_joint = value(
        floor["key"],
        sample=sample,
        metric="delta_joint_benefit_cny_per_forced_kw_year",
        config_id="caes_24h",
    )
    raw_fractional_did = did_record(
        raw["did"], sample=sample, metric="fractional_efc_change"
    )
    raw_discharge_did = did_record(
        raw["did"],
        sample=sample,
        metric="discharge_mwh_per_mean_task_mw_year",
    )
    raw_joint_did = did_record(
        raw["did"],
        sample=sample,
        metric="joint_benefit_cny_per_forced_kw_year",
    )
    floor_fractional_did = did_record(
        floor["did"], sample=sample, metric="fractional_efc_change"
    )
    floor_discharge_did = did_record(
        floor["did"],
        sample=sample,
        metric="discharge_mwh_per_mean_task_mw_year",
    )
    floor_joint_did = did_record(
        floor["did"],
        sample=sample,
        metric="joint_benefit_cny_per_forced_kw_year",
    )
    strength_lookup = strength.set_index(["task_variant", "metric"])
    raw_run = strength_lookup.loc[
        ("raw_observed_fossil_gas_main", "mean_extreme_run_d")
    ]
    floor_run = strength_lookup.loc[
        (
            "month_hour_q10_above_floor_sensitivity",
            "mean_extreme_run_d",
        )
    ]
    raw_valley = valley[
        (valley["task_variant"] == "raw_observed_fossil_gas_main")
        & (valley["sample"] == sample)
        & (valley["chronology"].isin(["dispersed", "clustered"]))
    ]
    floor_valley = valley[
        (
            valley["task_variant"]
            == "month_hour_q10_above_floor_sensitivity"
        )
        & (valley["sample"] == sample)
        & (valley["chronology"].isin(["dispersed", "clustered"]))
    ]
    memo = f"""# Observed hourly fossil-gas task paired-screen evidence memo

## What is observed

The main task is the non-negative hourly Energy-Charts `public_power` aggregate
`Fossil gas` series. The prespecified sample requires a complete hourly series
and annual gas energy of at least 1% of system load, retaining **15 countries,
132 country-years and {132 * 8760:,} country-hours**. The balanced EU
sensitivity comprises {", ".join(sorted(raw["balanced"]))}, or 8 countries and
80 country-years.

The source archive supplies 1,155,877 of these hourly task values directly;
443 values (0.0383%) are audited short-gap completions. A separate zero-gap
diagnostic retains {len(fully_observed_pairs)} country-years in
{len({country for country, _ in fully_observed_pairs})} countries and is
reported alongside, but does not replace, the fixed main sample.

The frozen raw panel SHA-256 is `{raw_panel_sha256}`. The model-facing
`load_mw` field is an exact row-wise copy of
`observed_gas_peaking_task_mw`; residual load is never substituted for this
task. In the independent validity audit, the within-country-year Pearson
correlation between actual fossil-gas dispatch and contemporaneous signed
residual load is positive in all
{validity['hourly_correlation']['positive_n']} country-years (mean
{validity['hourly_correlation']['mean']:.2f}; median
{validity['hourly_correlation']['median']:.2f}). The response-intensity
difference between natural multi-day residual-load events and isolated
one-day events is positive in
{validity['natural_event_response']['positive_n']} of 132 country-years
(mean {validity['natural_event_response']['mean']:.3f}). These diagnostics
support empirical relevance to system stress; they do not turn aggregate gas
generation into unit-resolved peaker output.

This is observed aggregate gas dispatch, not unit-resolved peaking output.
It mixes OCGT, CCGT, CHP and other operating roles. Accordingly, the manuscript
may call it an observed hourly fossil-gas task or gas-dispatch task, but not
observed OCGT/CCGT peaking generation.

## Paired intervention and charging

Each country-year keeps exactly the same 365 intact daily gas-task blocks and
the same annual gas energy. Upper-quartile daily task blocks are separated in
the dispersed chronology and grouped in four-day runs in the clustered
chronology, within calendar segments. For the raw task, median mean event
duration rises from {raw_run['dispersed_median']:.2f} to
{raw_run['clustered_median']:.2f} d (median paired change
{raw_run['delta_clustered_minus_dispersed_median']:.2f} d). In the above-floor
sensitivity it rises from {floor_run['dispersed_median']:.2f} to
{floor_run['clustered_median']:.2f} d (change
{floor_run['delta_clustered_minus_dispersed_median']:.2f} d).

The adapter uses `vre_energy_share=0`. There is no free-curtailment charging.
Storage charging is supplied by additional model gas generation and pays the
gas variable cost, round-trip losses and endogenous extra charging-power cost.

Technology and gas costs use the common controlled v36 2035 CNY calibration,
so cross-country comparisons isolate task chronology under fixed economics.
They are not observed country-specific European cost estimates.

## Paired storage response

After screening the 49 alternative technology-duration configurations plus
the 4 h LFP reference within each country-year, the clustered-minus-dispersed
mean EFC change is {raw_efc:.3f}
cycles for the raw task and {floor_efc:.3f} cycles after removal of the
month-hour q10 floor. The corresponding mean change in joint incremental
system benefit is {raw_joint:.2f} and {floor_joint:.2f}
CNY per forced kW-year.

For 24 h CAES specifically, the median paired EFC change is
{raw_caes_efc:.3f} cycles in the raw task and {floor_caes_efc:.3f} cycles in
the floor sensitivity. Its median joint-benefit change is
{raw_caes_joint:.2f} and {floor_caes_joint:.2f} CNY per forced kW-year.
Configuration-specific values for 4 h LFP and 72/168 h cavern hydrogen are
reported in `energy_charts_observed_gas_task_key_configuration_changes.csv`.

The prespecified CAES24-minus-LFP4 difference-in-differences avoids treating
unequal absolute cycle counts as directly comparable. For the raw task, the
fractional-EFC DID is {raw_fractional_did['did_mean']:.3f}
[{raw_fractional_did['dependence_robust_interval_low']:.3f},
{raw_fractional_did['dependence_robust_interval_high']:.3f}], the common-unit
discharge DID is {raw_discharge_did['did_mean']:.4f}
[{raw_discharge_did['dependence_robust_interval_low']:.4f},
{raw_discharge_did['dependence_robust_interval_high']:.4f}] MWh per
mean-task MW-year, and the joint-benefit DID is
{raw_joint_did['did_mean']:.2f}
[{raw_joint_did['dependence_robust_interval_low']:.2f},
{raw_joint_did['dependence_robust_interval_high']:.2f}] CNY per forced
kW-year. The above-floor counterparts are
{floor_fractional_did['did_mean']:.3f}
[{floor_fractional_did['dependence_robust_interval_low']:.3f},
{floor_fractional_did['dependence_robust_interval_high']:.3f}],
{floor_discharge_did['did_mean']:.4f}
[{floor_discharge_did['dependence_robust_interval_low']:.4f},
{floor_discharge_did['dependence_robust_interval_high']:.4f}] and
{floor_joint_did['did_mean']:.2f}
[{floor_joint_did['dependence_robust_interval_low']:.2f},
{floor_joint_did['dependence_robust_interval_high']:.2f}], respectively.
All intervals are envelopes over country-hierarchical, synchronized-year and
two-way country/year cluster procedures. The full table is
`energy_charts_observed_gas_task_caes24_lfp4_did.csv`.

## Mid-duration-valley status

The raw point screen identifies
{int(raw_valley['point_screen_complete_valley_n'].sum())} complete
chronology-country-year valleys across dispersed and clustered chronologies;
the above-floor sensitivity identifies
{int(floor_valley['point_screen_complete_valley_n'].sum())}. These are
historical single-country point screens. They do not satisfy the networked
ELCC and joint interval rule required for confirmatory valley evidence, so the
confirmatory valley count remains zero.

The raw screen contains
{int(raw_valley['partial_negative_interval_n'].sum())} chronology-country-year
profiles with at least one adjacent negative middle interval, while
{int(raw_valley['h_T_no_short_positive_turnover_branch_n'].sum())} do not
identify a positive short-duration turnover branch. The corresponding
above-floor counts are
{int(floor_valley['partial_negative_interval_n'].sum())} and
{int(floor_valley['h_T_no_short_positive_turnover_branch_n'].sum())}.
Accordingly, a negative middle interval is not reported as a complete valley
unless both the short-duration turnover boundary and a longer-duration
adequacy re-entry are observed.

## Decision

The raw observed task is admissible as the historical main specification. The
month-hour q10 subtraction is retained only as a robustness test against
persistent gas output. Agreement in response direction supports robustness to
the aggregate-gas composition; disagreement must be reported rather than
resolved by relabelling the above-floor remainder as a peaker technology.
"""
    memo_path = RESEARCH / "ENERGY_CHARTS_OBSERVED_GAS_TASK_SCREEN_MEMO.md"
    memo_path.write_text(memo, encoding="utf-8")
    print(json.dumps(qa, indent=2))
    print(memo_path)


if __name__ == "__main__":
    main()
