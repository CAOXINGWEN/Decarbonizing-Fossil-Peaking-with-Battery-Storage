#!/usr/bin/env python3
"""Summarize q70/q75/q80 observed-gas key configuration screens.

The script treats country-years, not configuration nodes, as empirical units.
It reports 4 h LFP and 24 h CAES in common throughput and economic units,
retains EFC for within-technology interpretation, and explicitly computes
CAES-minus-LFP paired difference-in-differences with dependence-robust
interval envelopes.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import summarize_energy_charts_strict50_screen as shared


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL = ROOT / "model_adapter"
RESEARCH = ROOT / "artifacts" / "research"
FLOOR_SAMPLE = (
    RESEARCH / "energy_charts_observed_gas_task_prespecified_sample.csv"
)
SUPPLY_RESPONSE_QA = (
    RESEARCH / "energy_charts_actual_supply_response_qa.csv"
)
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
EU27 = {
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE",
    "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT",
    "RO", "SK", "SI", "ES", "SE",
}
SPECS = {
    "raw_q70": (
        "energy_charts_observed_gas_raw_132cy_key7_q70_pairedonly",
        "raw_observed_fossil_gas_main",
        0.70,
    ),
    "raw_q75": (
        "energy_charts_observed_gas_raw_132cy_key7_q75_pairedonly",
        "raw_observed_fossil_gas_main",
        0.75,
    ),
    "raw_q80": (
        "energy_charts_observed_gas_raw_132cy_key7_q80_pairedonly",
        "raw_observed_fossil_gas_main",
        0.80,
    ),
    "floor_q75": (
        "energy_charts_observed_gas_floor_132cy_key7_q75_pairedonly",
        "month_hour_q10_above_floor_sensitivity",
        0.75,
    ),
}
CONFIGS = ("li_lfp_4h", "caes_24h")
FIELDS = {
    "equivalent_full_cycles": (
        "expected_equivalent_full_cycles_output_basis"
    ),
    "charge_mwh_per_mean_task_mw_year": (
        "expected_charge_mwh_per_mean_mw_year"
    ),
    "discharge_mwh_per_mean_task_mw_year": (
        "expected_discharge_mwh_per_mean_mw_year"
    ),
    "joint_benefit_cny_per_forced_kw_year": (
        "joint_signed_benefit_cny_per_forced_kw_year"
    ),
    "turnover_benefit_cny_per_forced_kw_year": (
        "turnover_signed_benefit_cny_per_forced_kw_year"
    ),
    "gas_capacity_avoided_pu_mean_task": (
        "gas_capacity_avoided_pu_mean_load"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def attach_country_year(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    split = result["unit_id"].str.rsplit("_", n=1, expand=True)
    result["country"] = split[0]
    result["year"] = split[1].astype(int)
    return result


def filter_pairs(
    frame: pd.DataFrame,
    countries: set[str] | None,
    pairs: set[tuple[str, int]] | None,
) -> pd.DataFrame:
    selected = frame
    if countries is not None:
        selected = selected[selected["country"].isin(countries)]
    if pairs is not None:
        index = pd.MultiIndex.from_frame(selected[["country", "year"]])
        selected = selected[index.isin(pairs)]
    return selected.copy()


def interval(frame: pd.DataFrame, field: str) -> dict[str, float]:
    country_low, country_high = shared.hierarchical_bootstrap_mean(
        frame, field
    )
    year_low, year_high = shared.calendar_year_block_bootstrap_mean(
        frame, field
    )
    two_low, two_high, two_se = shared.two_way_cluster_mean_interval(
        frame, field
    )
    return {
        "dependence_robust_interval_low": min(
            country_low, year_low, two_low
        ),
        "dependence_robust_interval_high": max(
            country_high, year_high, two_high
        ),
        "country_hierarchical_ci_low": country_low,
        "country_hierarchical_ci_high": country_high,
        "synchronized_year_block_ci_low": year_low,
        "synchronized_year_block_ci_high": year_high,
        "two_way_country_year_cluster_ci_low": two_low,
        "two_way_country_year_cluster_ci_high": two_high,
        "two_way_cluster_standard_error": two_se,
    }


def configuration_rows(
    selected: pd.DataFrame,
    spec_id: str,
    task_variant: str,
    quantile: float,
    sample_name: str,
) -> list[dict[str, Any]]:
    rows = []
    for config_id in CONFIGS:
        config = selected[selected["config_id"].eq(config_id)]
        for label, field in FIELDS.items():
            paired = config.pivot(
                index=["country", "year"],
                columns="chronology",
                values=field,
            ).reset_index()
            paired["delta"] = paired["clustered"] - paired["dispersed"]
            delta = paired["delta"]
            rows.append(
                {
                    "spec_id": spec_id,
                    "task_variant": task_variant,
                    "extreme_quantile": quantile,
                    "sample": sample_name,
                    "config_id": config_id,
                    "metric": label,
                    "country_count": int(paired["country"].nunique()),
                    "country_year_count": int(len(paired)),
                    "dispersed_mean": float(paired["dispersed"].mean()),
                    "dispersed_median": float(paired["dispersed"].median()),
                    "clustered_mean": float(paired["clustered"].mean()),
                    "clustered_median": float(paired["clustered"].median()),
                    "pooled_clustered_over_dispersed": float(
                        paired["clustered"].sum()
                        / paired["dispersed"].sum()
                    )
                    if abs(float(paired["dispersed"].sum())) > 1e-12
                    else np.nan,
                    "delta_mean": float(delta.mean()),
                    "delta_median": float(delta.median()),
                    "delta_negative_n": int((delta < -1e-8).sum()),
                    "delta_positive_n": int((delta > 1e-8).sum()),
                    "delta_zero_n": int((delta.abs() <= 1e-8).sum()),
                    **interval(paired, "delta"),
                }
            )
    return rows


def did_rows(
    selected: pd.DataFrame,
    spec_id: str,
    task_variant: str,
    quantile: float,
    sample_name: str,
) -> list[dict[str, Any]]:
    rows = []
    for label, field in FIELDS.items():
        wide = selected.pivot(
            index=["country", "year"],
            columns=["config_id", "chronology"],
            values=field,
        ).reset_index()
        wide["did"] = (
            wide[("caes_24h", "clustered")]
            - wide[("caes_24h", "dispersed")]
            - wide[("li_lfp_4h", "clustered")]
            + wide[("li_lfp_4h", "dispersed")]
        )
        # Flatten the two identifier columns produced by a MultiIndex pivot.
        audit = pd.DataFrame(
            {
                "country": wide[("country", "")],
                "year": wide[("year", "")].astype(int),
                "did": wide["did"],
            }
        )
        rows.append(
            {
                "spec_id": spec_id,
                "task_variant": task_variant,
                "extreme_quantile": quantile,
                "sample": sample_name,
                "metric": label,
                "contrast": (
                    "(clustered-dispersed)_CAES24 - "
                    "(clustered-dispersed)_LFP4"
                ),
                "country_count": int(audit["country"].nunique()),
                "country_year_count": int(len(audit)),
                "did_mean": float(audit["did"].mean()),
                "did_median": float(audit["did"].median()),
                "did_negative_n": int((audit["did"] < -1e-8).sum()),
                "did_positive_n": int((audit["did"] > 1e-8).sum()),
                "did_zero_n": int((audit["did"].abs() <= 1e-8).sum()),
                **interval(audit, "did"),
            }
        )

    efc = selected.pivot(
        index=["country", "year"],
        columns=["config_id", "chronology"],
        values=FIELDS["equivalent_full_cycles"],
    )
    fractional = (
        efc[("caes_24h", "clustered")]
        / efc[("caes_24h", "dispersed")]
        - 1.0
        - efc[("li_lfp_4h", "clustered")]
        / efc[("li_lfp_4h", "dispersed")]
        + 1.0
    ).replace([np.inf, -np.inf], np.nan)
    audit = fractional.dropna().rename("did").reset_index()
    rows.append(
        {
            "spec_id": spec_id,
            "task_variant": task_variant,
            "extreme_quantile": quantile,
            "sample": sample_name,
            "metric": "fractional_efc_change",
            "contrast": (
                "fractional EFC change CAES24 minus fractional EFC change "
                "LFP4"
            ),
            "country_count": int(audit["country"].nunique()),
            "country_year_count": int(len(audit)),
            "did_mean": float(audit["did"].mean()),
            "did_median": float(audit["did"].median()),
            "did_negative_n": int((audit["did"] < -1e-8).sum()),
            "did_positive_n": int((audit["did"] > 1e-8).sum()),
            "did_zero_n": int((audit["did"].abs() <= 1e-8).sum()),
            **interval(audit, "did"),
        }
    )
    return rows


def geometry_rows(
    chronology: pd.DataFrame,
    spec_id: str,
    task_variant: str,
    quantile: float,
    sample_name: str,
) -> list[dict[str, Any]]:
    fields = (
        "extreme_days",
        "mean_extreme_run_d",
        "max_extreme_run_d",
        "share_extreme_days_in_runs_ge3",
    )
    rows = []
    for field in fields:
        paired = chronology[
            chronology["chronology"].isin(["dispersed", "clustered"])
        ].pivot(
            index=["country", "year"],
            columns="chronology",
            values=field,
        ).reset_index()
        paired["delta"] = paired["clustered"] - paired["dispersed"]
        rows.append(
            {
                "spec_id": spec_id,
                "task_variant": task_variant,
                "extreme_quantile": quantile,
                "sample": sample_name,
                "metric": field,
                "country_count": int(paired["country"].nunique()),
                "country_year_count": int(len(paired)),
                "dispersed_mean": float(paired["dispersed"].mean()),
                "dispersed_median": float(paired["dispersed"].median()),
                "clustered_mean": float(paired["clustered"].mean()),
                "clustered_median": float(paired["clustered"].median()),
                "delta_mean": float(paired["delta"].mean()),
                "delta_median": float(paired["delta"].median()),
            }
        )
    return rows


def pick(
    frame: pd.DataFrame,
    spec_id: str,
    metric: str,
    *,
    config_id: str | None = None,
) -> pd.Series:
    selected = frame[
        frame["spec_id"].eq(spec_id)
        & frame["sample"].eq("prespecified_15c_132cy")
        & frame["metric"].eq(metric)
    ]
    if config_id is not None:
        selected = selected[selected["config_id"].eq(config_id)]
    if len(selected) != 1:
        raise RuntimeError(
            f"Expected one row: {spec_id}, {metric}, {config_id}; "
            f"found {len(selected)}"
        )
    return selected.iloc[0]


def main() -> None:
    sample_frame = pd.read_csv(FLOOR_SAMPLE)
    sample_frame = sample_frame[
        as_bool(sample_frame["main_prespecified_sample"])
    ]
    all_pairs = set(
        sample_frame[["country", "year"]].itertuples(index=False, name=None)
    )
    floor_pairs = set(
        sample_frame.loc[
            as_bool(sample_frame["above_floor_material_share_ge_1pct"]),
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
        & all_pairs
    )
    countries = set(sample_frame["country"])
    eu = countries & EU27
    balanced = {
        country
        for country, group in sample_frame.groupby("country")
        if country in EU27 and set(group["year"]) == set(range(2015, 2025))
    }
    samples = (
        ("prespecified_15c_132cy", None, all_pairs),
        ("eu_only_14c_127cy", eu, all_pairs),
        ("balanced_eu_8c_80cy", balanced, all_pairs),
        ("floor_material_15c_118cy", None, floor_pairs),
        ("fully_observed_gas_14c_76cy", None, fully_observed_pairs),
    )

    configuration_output = []
    did_output = []
    geometry_output = []
    run_qa = {}
    chronology_by_spec: dict[str, pd.DataFrame] = {}
    metrics_by_spec: dict[str, pd.DataFrame] = {}
    for spec_id, (batch_id, task_variant, quantile) in SPECS.items():
        aggregate = MODEL / f"{batch_id}_aggregate"
        qa = json.loads(
            (aggregate / "AGGREGATE_QA_AND_SUMMARY.json").read_text(
                encoding="utf-8"
            )
        )
        run_qa[spec_id] = qa
        metrics = attach_country_year(
            pd.read_csv(aggregate / "all_configuration_screen_metrics.csv")
        )
        metrics = metrics[metrics["config_id"].isin(CONFIGS)]
        metrics_by_spec[spec_id] = metrics
        chronology = attach_country_year(
            pd.read_csv(aggregate / "all_chronology_conservation_qa.csv")
        )
        chronology_by_spec[spec_id] = chronology
        for sample_name, country_set, pair_set in samples:
            selected = filter_pairs(metrics, country_set, pair_set)
            selected_chronology = filter_pairs(
                chronology, country_set, pair_set
            )
            configuration_output.extend(
                configuration_rows(
                    selected,
                    spec_id,
                    task_variant,
                    quantile,
                    sample_name,
                )
            )
            did_output.extend(
                did_rows(
                    selected,
                    spec_id,
                    task_variant,
                    quantile,
                    sample_name,
                )
            )
            geometry_output.extend(
                geometry_rows(
                    selected_chronology,
                    spec_id,
                    task_variant,
                    quantile,
                    sample_name,
                )
            )

    configurations = pd.DataFrame(configuration_output)
    did = pd.DataFrame(did_output)
    geometry = pd.DataFrame(geometry_output)
    configurations.to_csv(
        RESEARCH
        / "energy_charts_observed_gas_key_configuration_threshold_summary.csv",
        index=False,
    )
    did.to_csv(
        RESEARCH / "energy_charts_observed_gas_key_did_threshold_summary.csv",
        index=False,
    )
    geometry.to_csv(
        RESEARCH / "energy_charts_observed_gas_key_geometry_threshold_summary.csv",
        index=False,
    )

    checks = {
        "all_four_batch_qa_passed": all(
            bool(qa["passed"]) for qa in run_qa.values()
        ),
        "all_batches_132_country_years": all(
            int(qa["country_year_count"]) == 132 for qa in run_qa.values()
        ),
        "all_batches_seven_key_nodes": all(
            int(qa["configuration_count_per_country_year"]) == 7
            for qa in run_qa.values()
        ),
        "all_batches_vre_share_zero": all(
            float(qa["vre_energy_share"]) == 0.0
            for qa in run_qa.values()
        ),
        "all_batches_have_zero_exogenous_surplus": all(
            float(frame["annual_surplus_pu_h"].abs().max()) <= 1e-12
            for frame in chronology_by_spec.values()
        ),
        "all_batches_use_observed_task_itself_as_residual": all(
            float(
                (
                    frame["annual_residual_pu_h"]
                    - frame["annual_load_pu_h"]
                )
                .abs()
                .max()
            )
            <= 1e-12
            for frame in chronology_by_spec.values()
        ),
        "quantiles_match_contract": all(
            abs(float(run_qa[spec_id]["extreme_quantile"]) - quantile)
            <= 1e-12
            for spec_id, (_, _, quantile) in SPECS.items()
        ),
        "fixed_main_sample_132": len(all_pairs) == 132,
        "eu_only_sample_127": int(
            sample_frame["country"].isin(EU27).sum()
        )
        == 127,
        "balanced_sample_80": int(
            sample_frame["country"].isin(balanced).sum()
        )
        == 80,
        "floor_material_diagnostic_118": len(floor_pairs) == 118,
        "fully_observed_gas_diagnostic_76": (
            len(fully_observed_pairs) == 76
            and len({country for country, _ in fully_observed_pairs}) == 14
        ),
        "raw_q70_q75_q80_share_identical_input_sha": len(
            {
                run_qa[spec_id]["panel_sha256"]
                for spec_id in ("raw_q70", "raw_q75", "raw_q80")
            }
        )
        == 1,
        "raw_input_sha_matches_current_panel": (
            run_qa["raw_q75"]["panel_sha256"] == sha256(RAW_PANEL)
        ),
        "floor_input_sha_matches_current_panel": (
            run_qa["floor_q75"]["panel_sha256"] == sha256(FLOOR_PANEL)
        ),
        "every_country_year_has_all_four_config_chronology_cells": all(
            metrics.groupby(["country", "year"]).size().eq(4).all()
            and metrics.groupby(["country", "year"])["config_id"]
            .nunique()
            .eq(2)
            .all()
            and metrics.groupby(["country", "year"])["chronology"]
            .nunique()
            .eq(2)
            .all()
            for metrics in metrics_by_spec.values()
        ),
        "all_caes24_lfp4_metric_cells_nonmissing": all(
            metrics[list(FIELDS.values())].notna().all().all()
            for metrics in metrics_by_spec.values()
        ),
        "all_configuration_summary_rows_retain_expected_sample_size": all(
            group["country_year_count"].eq(expected).all()
            for sample, expected in {
                "prespecified_15c_132cy": 132,
                "eu_only_14c_127cy": 127,
                "balanced_eu_8c_80cy": 80,
                "floor_material_15c_118cy": 118,
                "fully_observed_gas_14c_76cy": 76,
            }.items()
            for _, group in configurations[
                configurations["sample"].eq(sample)
            ].groupby("spec_id")
        ),
        "all_did_rows_retain_expected_sample_size": all(
            group["country_year_count"].eq(expected).all()
            for sample, expected in {
                "prespecified_15c_132cy": 132,
                "eu_only_14c_127cy": 127,
                "balanced_eu_8c_80cy": 80,
                "floor_material_15c_118cy": 118,
                "fully_observed_gas_14c_76cy": 76,
            }.items()
            for _, group in did[did["sample"].eq(sample)].groupby("spec_id")
        ),
    }
    qa = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "primary_quantile": 0.75,
        "sensitivity_quantiles": [0.70, 0.80],
        "configurations_interpreted": list(CONFIGS),
        "additional_pilot_nodes_solved_but_not_used_in_primary_contrast": [
            "caes_12h",
            "h2_cavern_48h",
            "h2_cavern_96h",
            "h2_cavern_168h",
            "h2_cavern_720h",
        ],
        "sample_contract": {
            "primary": "15 countries, 132 country-years",
            "eu_only": "14 countries, 127 country-years",
            "balanced_eu": "8 countries, 80 country-years",
            "floor_material": (
                "118-country-year diagnostic only; does not replace the "
                "fixed 132-country-year floor sensitivity"
            ),
            "fully_observed_gas": (
                "14 countries, 76 country-years with zero completed fossil-"
                "gas hours; diagnostic only"
            ),
        },
        "metric_contract": (
            "absolute cycle DID is not used for cross-duration inference; "
            "common-unit discharge throughput, fractional EFC change and "
            "joint economic benefit are primary. Point estimates and all "
            "resampling intervals target the same equal-country-year mean."
        ),
        "task_input_contract": {
            "main_source_field": "observed_gas_peaking_task_mw",
            "adapter_field": "load_mw",
            "main_field_identity": (
                "load_mw is an exact copy of observed Energy-Charts "
                "public_power aggregate Fossil gas"
            ),
            "raw_panel": str(RAW_PANEL),
            "raw_panel_sha256": sha256(RAW_PANEL),
            "floor_panel": str(FLOOR_PANEL),
            "floor_panel_sha256": sha256(FLOOR_PANEL),
            "residual_load_role": (
                "correlation and external event-response diagnostics only; "
                "never the storage-screen task input"
            ),
        },
        "valley_status": (
            "not identifiable from seven pilot nodes; full 50-node "
            "4-168 h screen required"
        ),
    }
    (RESEARCH / "energy_charts_observed_gas_key_screen_qa.json").write_text(
        json.dumps(qa, indent=2), encoding="utf-8"
    )
    if qa["status"] != "pass":
        raise RuntimeError(json.dumps(qa, indent=2))

    lines = [
        "# Observed hourly fossil-gas key-screen memo",
        "",
        "## Contract",
        "",
        (
            "The primary sample contains 15 countries and 132 country-years. "
            "The task input is `observed_gas_peaking_task_mw`, an exact copy "
            "of observed Energy-Charts `public_power` aggregate `Fossil gas`; "
            "the adapter reads that series through `load_mw`. The raw panel "
            f"SHA-256 is `{sha256(RAW_PANEL)}`. The raw hourly aggregate "
            "Fossil gas task is paired under q75; "
            "q70 and q80 test threshold sensitivity. The month-hour q10 "
            "above-floor task is run on the same 132 country-years."
        ),
        (
            "Residual load is used only for task-validity correlation and "
            "natural-event response diagnostics. It is never substituted "
            "for observed aggregate fossil-gas generation in q70, q75 or "
            "q80 storage screens."
        ),
        (
            "Charging uses additional costed gas generation because "
            "`vre_energy_share=0`; no free-curtailment charging enters these "
            "screens."
        ),
        (
            "Technology and gas costs are the common controlled v36 2035 "
            "CNY calibration. They hold economics fixed while historical "
            "task chronology changes; they are not observed country-specific "
            "European costs."
        ),
        "",
        "## Main and threshold sensitivity",
        "",
    ]
    for spec_id in ("raw_q70", "raw_q75", "raw_q80", "floor_q75"):
        lfp = pick(
            configurations,
            spec_id,
            "equivalent_full_cycles",
            config_id="li_lfp_4h",
        )
        caes = pick(
            configurations,
            spec_id,
            "equivalent_full_cycles",
            config_id="caes_24h",
        )
        discharge_did = pick(
            did,
            spec_id,
            "discharge_mwh_per_mean_task_mw_year",
        )
        benefit_did = pick(
            did,
            spec_id,
            "joint_benefit_cny_per_forced_kw_year",
        )
        fractional = pick(did, spec_id, "fractional_efc_change")
        lines.append(
            (
                f"- {spec_id}: pooled EFC retention is "
                f"{lfp['pooled_clustered_over_dispersed']:.2%} for 4 h LFP "
                f"and {caes['pooled_clustered_over_dispersed']:.2%} for "
                f"24 h CAES. CAES-minus-LFP fractional-EFC DID is "
                f"{fractional['did_mean']:.3f} "
                f"[{fractional['dependence_robust_interval_low']:.3f}, "
                f"{fractional['dependence_robust_interval_high']:.3f}]. "
                f"Common-unit discharge DID is {discharge_did['did_mean']:.4f} "
                f"[{discharge_did['dependence_robust_interval_low']:.4f}, "
                f"{discharge_did['dependence_robust_interval_high']:.4f}], "
                f"and joint-benefit DID is {benefit_did['did_mean']:.2f} "
                f"[{benefit_did['dependence_robust_interval_low']:.2f}, "
                f"{benefit_did['dependence_robust_interval_high']:.2f}]."
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "EFC is meaningful within a technology but an absolute "
                "difference in cycle counts is not comparable across 4 h "
                "and 24 h configurations. The common-unit throughput and "
                "economic DIDs therefore carry the cross-technology claim."
            ),
            (
                "Every DID point estimate and every dependence-aware "
                "resampling interval targets the same equal-country-year "
                "mean; the displayed interval is the union of country-"
                "hierarchical, synchronized-year and two-way cluster "
                "procedures."
            ),
            (
                "Turnover-only benefit is approximately unchanged when "
                "charging is supplied by costed fossil generation. This "
                "historical screen supports a capacity/peak channel; it is "
                "not evidence of free-recharge arbitrage."
            ),
            (
                "The seven-node screen cannot identify a complete "
                "mid-duration valley. Valley status remains pending the "
                "full 50-node 4-168 h screen."
            ),
        ]
    )
    memo = RESEARCH / "ENERGY_CHARTS_OBSERVED_GAS_KEY_SCREEN_MEMO.md"
    memo.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(qa, indent=2))
    print(memo)


if __name__ == "__main__":
    main()
