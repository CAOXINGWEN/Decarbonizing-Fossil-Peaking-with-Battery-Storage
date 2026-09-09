#!/usr/bin/env python3
"""Summarize the v42 shared-recharge-power intervention.

The experiment keeps the same 4 h LFP + LDES versus 4 h LFP + extra 4 h LFP
portfolio comparison and changes only the shared recharge-power multiplier
(r1, r2 and r4). All economic values are reconstructed from the directly
solved candidate and comparator arms. Scenario 20 is the central main-text
case; scenario 31 is retained as a separately labelled conditional case.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from postprocess_v42_portfolio_results import (
    DEFAULT_SCENARIO_TABLE,
    FOCAL_CONFIGS,
    add_derived_metrics,
    country_block_bootstrap,
    infer_arm_inputs,
    load_arm_results,
    load_cells,
    load_valuation_scenarios,
    quantiles,
    sha256,
    utc_now,
    wait_for_inputs,
    write_csv,
    write_csv_gz,
    write_json,
)


HERE = Path(__file__).resolve().parent
V42_ROOT = HERE.parent
DEFAULT_INPUTS = {
    "fixed_s20_central:r1": (
        HERE
        / "full_132cy_fixed_s20_central"
        / "ldes_vs_extra_lfp4_results.csv.gz"
    ),
    "fixed_s20_central:r2": (
        HERE
        / "recharge_intervention_fixed_s20_r2"
        / "ldes_vs_extra_lfp4_results.csv.gz"
    ),
    "fixed_s20_central:r4": (
        HERE
        / "recharge_intervention_fixed_s20_r4"
        / "ldes_vs_extra_lfp4_results.csv.gz"
    ),
    "fixed_s31_conditional:r1": (
        HERE / "full_132cy_fixed" / "ldes_vs_extra_lfp4_results.csv.gz"
    ),
    "fixed_s31_conditional:r2": (
        HERE
        / "recharge_intervention_fixed_r2"
        / "ldes_vs_extra_lfp4_results.csv.gz"
    ),
    "fixed_s31_conditional:r4": (
        HERE
        / "recharge_intervention_fixed_r4"
        / "ldes_vs_extra_lfp4_results.csv.gz"
    ),
}
INTERVENTIONS = {
    "r1": {
        "quantity_multiplier": 1.0,
        "profile_mode": "observed",
        "intervention_axis": "reference",
    },
    "r2": {
        "quantity_multiplier": 2.0,
        "profile_mode": "observed",
        "intervention_axis": "quantity",
    },
    "r4": {
        "quantity_multiplier": 4.0,
        "profile_mode": "observed",
        "intervention_axis": "quantity",
    },
    "uniform_annual_total": {
        "quantity_multiplier": 1.0,
        "profile_mode": "uniform_8760",
        "intervention_axis": "timing",
    },
}
REQUIRED_INTERVENTIONS = {"r1", "r2", "r4"}
SOURCE_SCENARIOS = {
    "fixed_s20_central": 20,
    "fixed_s31_conditional": 31,
}
DEFAULT_OUTPUT = (
    V42_ROOT / "source_data" / "v42_recharge_multiplier_paired_summary.csv"
)
DEFAULT_RAW_OUTPUT = (
    V42_ROOT
    / "source_data"
    / "v42_recharge_multiplier_country_year_effects.csv.gz"
)
DEFAULT_AUDIT_OUTPUT = (
    V42_ROOT / "source_data" / "v42_recharge_multiplier_audit.json"
)
DEFAULT_DRAWS = 5_000
DEFAULT_SEED = 20260728

METRICS = {
    "tail_charge_mwh": "MWh yr^-1",
    "tail_discharge_mwh": "MWh yr^-1",
    "tail_equivalent_full_cycles": "cycles yr^-1",
    "shared_recharge_binding_hours": "h yr^-1",
    "base_charge_power_binding_hours": "h yr^-1",
    "tail_charge_power_binding_hours": "h yr^-1",
    "unserved_eens_mwh": "MWh yr^-1",
    "portfolio_efc_mw": "MW",
    "mix_eens_reduction_vs_extra_lfp4_mwh": "MWh yr^-1",
    "mix_efc_gain_vs_extra_lfp4_mw": "MW",
    "mix_benefit_eur_per_kw_common_power_year": "EUR kW^-1 yr^-1",
    "mix_benefit_eur_per_mwh_task": "EUR MWh^-1 task",
    "tail_charge_share_of_task": "fraction",
    "tail_discharge_share_of_task": "fraction",
    "unserved_eens_share_of_task": "fraction",
    "portfolio_efc_ratio_to_common_power": "fraction",
    "mix_eens_reduction_share_of_task": "fraction",
    "mix_efc_gain_ratio_to_common_power": "fraction",
}


def parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--input must use SOURCE:r1=/path, SOURCE:r2=/path or "
            "SOURCE:r4=/path"
        )
    run_label, raw_path = value.split("=", maxsplit=1)
    run_label = run_label.strip()
    if ":" not in run_label:
        raise argparse.ArgumentTypeError(
            "Input label must use SOURCE:r1, SOURCE:r2 or SOURCE:r4"
        )
    source_label, intervention = run_label.rsplit(":", maxsplit=1)
    if source_label not in SOURCE_SCENARIOS:
        raise argparse.ArgumentTypeError(
            "SOURCE must be fixed_s20_central or fixed_s31_conditional"
        )
    if intervention not in INTERVENTIONS:
        raise argparse.ArgumentTypeError(
            "Intervention must be r1, r2, r4 or uniform_annual_total"
        )
    return run_label, Path(raw_path).expanduser()


def split_run_label(run_label: str) -> tuple[str, str]:
    source_label, intervention = run_label.rsplit(":", maxsplit=1)
    return source_label, intervention


def build_long_cells(cells: pd.DataFrame) -> pd.DataFrame:
    focal = cells[cells["config_id"].isin(FOCAL_CONFIGS)].copy()
    focal["recharge_quantity_multiplier"] = focal["intervention_id"].map(
        lambda value: INTERVENTIONS[str(value)]["quantity_multiplier"]
    )
    focal["recharge_power_multiplier"] = focal[
        "recharge_quantity_multiplier"
    ]
    focal["recharge_profile_mode"] = focal["intervention_id"].map(
        lambda value: INTERVENTIONS[str(value)]["profile_mode"]
    )
    focal["intervention_axis"] = focal["intervention_id"].map(
        lambda value: INTERVENTIONS[str(value)]["intervention_axis"]
    )
    if focal["recharge_power_multiplier"].isna().any():
        raise RuntimeError("Unknown recharge intervention label")
    focal["tail_charge_share_of_task"] = (
        focal["tail_charge_mwh"] / focal["base_task_mwh"]
    )
    focal["tail_discharge_share_of_task"] = (
        focal["tail_discharge_mwh"] / focal["base_task_mwh"]
    )
    focal["unserved_eens_share_of_task"] = (
        focal["unserved_eens_mwh"] / focal["base_task_mwh"]
    )
    focal["portfolio_efc_ratio_to_common_power"] = (
        focal["portfolio_efc_mw"] / focal["base_power_mw"]
    )
    focal["mix_eens_reduction_share_of_task"] = (
        focal["mix_eens_reduction_vs_extra_lfp4_mwh"]
        / focal["base_task_mwh"]
    )
    focal["mix_efc_gain_ratio_to_common_power"] = (
        focal["mix_efc_gain_vs_extra_lfp4_mw"]
        / focal["base_power_mw"]
    )
    identity = [
        "source_label",
        "run_label",
        "intervention_id",
        "recharge_power_multiplier",
        "recharge_quantity_multiplier",
        "recharge_profile_mode",
        "intervention_axis",
        "model_mode",
        "valuation_scenario_id",
        "country",
        "year",
        "country_year_id",
        "chronology",
        "config_id",
        "tech_id",
        "technology_en",
        "duration_h",
    ]
    long = focal.melt(
        id_vars=identity,
        value_vars=list(METRICS),
        var_name="metric",
        value_name="effect_value",
    )
    long["metric_unit"] = long["metric"].map(METRICS)
    long["analysis"] = "chronology_level"
    long["comparison"] = long["chronology"]
    long["reference_recharge_power_multiplier"] = np.nan
    long["reference_intervention_id"] = ""
    return long


def chronology_pairs(long: pd.DataFrame) -> pd.DataFrame:
    index = [
        "source_label",
        "run_label",
        "intervention_id",
        "recharge_power_multiplier",
        "recharge_quantity_multiplier",
        "recharge_profile_mode",
        "intervention_axis",
        "model_mode",
        "valuation_scenario_id",
        "country",
        "year",
        "country_year_id",
        "config_id",
        "tech_id",
        "technology_en",
        "duration_h",
        "metric",
        "metric_unit",
    ]
    wide = long.pivot(
        index=index, columns="chronology", values="effect_value"
    ).reset_index()
    if (
        "clustered" not in wide
        or "dispersed" not in wide
        or wide[["clustered", "dispersed"]].isna().any().any()
    ):
        raise RuntimeError("Incomplete clustered/dispersed intervention pair")
    wide["effect_value"] = wide["clustered"] - wide["dispersed"]
    wide["analysis"] = "clustered_minus_dispersed"
    wide["comparison"] = "clustered_minus_dispersed"
    wide["chronology"] = "clustered_minus_dispersed"
    wide["reference_recharge_power_multiplier"] = np.nan
    wide["reference_intervention_id"] = ""
    return wide


def intervention_contrasts(long: pd.DataFrame) -> pd.DataFrame:
    index = [
        "source_label",
        "model_mode",
        "valuation_scenario_id",
        "country",
        "year",
        "country_year_id",
        "chronology",
        "config_id",
        "tech_id",
        "technology_en",
        "duration_h",
        "metric",
        "metric_unit",
    ]
    wide = long.pivot(
        index=index,
        columns="intervention_id",
        values="effect_value",
    ).reset_index()
    if any(value not in wide for value in REQUIRED_INTERVENTIONS):
        raise RuntimeError("The recharge intervention requires r1, r2 and r4")
    interventions = [
        value
        for value in ("r2", "r4", "uniform_annual_total")
        if value in wide
    ]
    required_columns = ["r1"] + interventions
    if wide[required_columns].isna().any().any():
        raise RuntimeError("Incomplete country-year recharge multiplier pair")
    frames: list[pd.DataFrame] = []
    for intervention in interventions:
        specification = INTERVENTIONS[intervention]
        multiplier = float(specification["quantity_multiplier"])
        result = wide[index].copy()
        result["run_label"] = (
            result["source_label"]
            + f":{intervention}_minus_r1"
        )
        result["intervention_id"] = f"{intervention}_minus_r1"
        result["recharge_power_multiplier"] = multiplier
        result["recharge_quantity_multiplier"] = multiplier
        result["recharge_profile_mode"] = specification["profile_mode"]
        result["intervention_axis"] = specification["intervention_axis"]
        result["reference_recharge_power_multiplier"] = 1.0
        result["reference_intervention_id"] = "r1"
        result["effect_value"] = wide[intervention] - wide["r1"]
        result["analysis"] = "recharge_intervention_minus_r1"
        result["comparison"] = f"{intervention}_minus_r1"
        frames.append(result)
    return pd.concat(frames, ignore_index=True)


def difference_in_differences(pairs: pd.DataFrame) -> pd.DataFrame:
    index = [
        "source_label",
        "model_mode",
        "valuation_scenario_id",
        "country",
        "year",
        "country_year_id",
        "config_id",
        "tech_id",
        "technology_en",
        "duration_h",
        "metric",
        "metric_unit",
    ]
    wide = pairs.pivot(
        index=index,
        columns="intervention_id",
        values="effect_value",
    ).reset_index()
    if any(value not in wide for value in REQUIRED_INTERVENTIONS):
        raise RuntimeError("Cannot form recharge difference-in-differences")
    interventions = [
        value
        for value in ("r2", "r4", "uniform_annual_total")
        if value in wide
    ]
    required_columns = ["r1"] + interventions
    if wide[required_columns].isna().any().any():
        raise RuntimeError("Incomplete recharge difference-in-differences")
    frames: list[pd.DataFrame] = []
    for intervention in interventions:
        specification = INTERVENTIONS[intervention]
        multiplier = float(specification["quantity_multiplier"])
        result = wide[index].copy()
        result["run_label"] = (
            result["source_label"]
            + f":{intervention}_minus_r1"
        )
        result["intervention_id"] = f"{intervention}_minus_r1"
        result["recharge_power_multiplier"] = multiplier
        result["recharge_quantity_multiplier"] = multiplier
        result["recharge_profile_mode"] = specification["profile_mode"]
        result["intervention_axis"] = specification["intervention_axis"]
        result["reference_recharge_power_multiplier"] = 1.0
        result["reference_intervention_id"] = "r1"
        result["effect_value"] = wide[intervention] - wide["r1"]
        result["analysis"] = "chronology_difference_in_differences"
        result["comparison"] = (
            f"(clustered-dispersed)_{intervention}"
            "-(clustered-dispersed)_r1"
        )
        result["chronology"] = "clustered_minus_dispersed"
        frames.append(result)
    return pd.concat(frames, ignore_index=True)


def summarize_effects(
    effects: pd.DataFrame,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    keys = [
        "analysis",
        "source_label",
        "run_label",
        "intervention_id",
        "recharge_power_multiplier",
        "recharge_quantity_multiplier",
        "recharge_profile_mode",
        "intervention_axis",
        "reference_recharge_power_multiplier",
        "reference_intervention_id",
        "model_mode",
        "valuation_scenario_id",
        "config_id",
        "tech_id",
        "technology_en",
        "duration_h",
        "chronology",
        "comparison",
        "metric",
        "metric_unit",
    ]
    groups = list(effects.groupby(keys, sort=True, dropna=False))
    children = np.random.SeedSequence(seed).spawn(len(groups))
    rows: list[dict[str, Any]] = []
    for (key, group), child in zip(groups, children):
        distribution = quantiles(group["effect_value"])
        bootstrap = country_block_bootstrap(
            group,
            "effect_value",
            draws,
            np.random.default_rng(child),
            country_aggregation="mean",
        )
        rows.append(
            {
                **dict(zip(keys, key)),
                "n_country_years": int(
                    group[["country", "year"]].drop_duplicates().shape[0]
                ),
                "n_countries": int(group["country"].nunique()),
                **distribution,
                "country_block_mean": bootstrap["point_estimate"],
                "country_block_ci_low": bootstrap["ci_low"],
                "country_block_ci_high": bootstrap["ci_high"],
                "bootstrap_draws": draws,
                "bootstrap_independent_block": "country",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        type=parse_input,
        help=(
            "Repeat SOURCE:r1=/path, SOURCE:r2=/path and SOURCE:r4=/path. "
            "Each included source must contain the complete r1/r2/r4 set. "
            "SOURCE:uniform_annual_total=/path is an optional timing test."
        ),
    )
    parser.add_argument(
        "--scenario-table", type=Path, default=DEFAULT_SCENARIO_TABLE
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW_OUTPUT)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT_OUTPUT)
    parser.add_argument("--bootstrap-draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--expected-country-years", type=int, default=132)
    parser.add_argument("--wait-for-inputs", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-seconds", type=float, default=86400.0)
    args = parser.parse_args()
    if args.bootstrap_draws <= 0:
        raise ValueError("--bootstrap-draws must be positive")
    inputs = (
        args.input
        if args.input
        else [(label, path) for label, path in DEFAULT_INPUTS.items()]
    )
    run_labels = [label for label, _ in inputs]
    if len(run_labels) != len(set(run_labels)):
        raise RuntimeError("Recharge run labels must be unique")
    source_interventions: dict[str, set[str]] = {}
    for run_label in run_labels:
        source_label, intervention = split_run_label(run_label)
        source_interventions.setdefault(source_label, set()).add(intervention)
    incomplete_sources = {
        source: sorted(interventions)
        for source, interventions in source_interventions.items()
        if not REQUIRED_INTERVENTIONS.issubset(interventions)
    }
    if incomplete_sources:
        raise RuntimeError(
            "Each source requires r1, r2 and r4: "
            + json.dumps(incomplete_sources, sort_keys=True)
        )
    arm_inputs = infer_arm_inputs(inputs)
    paths = [path for _, path in inputs + arm_inputs]
    if args.wait_for_inputs:
        wait_for_inputs(paths, args.poll_seconds, args.timeout_seconds)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing recharge inputs: {missing}")

    started = time.perf_counter()
    scenarios = {
        int(row["scenario_id"]): row
        for row in load_valuation_scenarios(
            args.scenario_table,
            sorted(
                SOURCE_SCENARIOS[source]
                for source in source_interventions
            ),
        )
    }
    raw_cells = load_cells(inputs)
    raw_arms = load_arm_results(arm_inputs)
    derived_frames: list[pd.DataFrame] = []
    for source_label in sorted(source_interventions):
        scenario_id = SOURCE_SCENARIOS[source_label]
        source_runs = [
            run_label
            for run_label in run_labels
            if split_run_label(run_label)[0] == source_label
        ]
        source_cells = raw_cells[
            raw_cells["source_label"].isin(source_runs)
        ].copy()
        source_arms = raw_arms[
            raw_arms["source_label"].isin(source_runs)
        ].copy()
        dispatch_scenarios = set(
            int(value) for value in source_cells["scenario_id"].unique()
        )
        if dispatch_scenarios != {scenario_id}:
            raise RuntimeError(
                f"{source_label} expected dispatch scenario {scenario_id}, "
                f"observed {sorted(dispatch_scenarios)}"
            )
        derived = add_derived_metrics(
            source_cells,
            source_arms,
            scenarios[scenario_id],
        )
        derived = derived.rename(columns={"source_label": "run_label"})
        derived["source_label"] = source_label
        derived["intervention_id"] = derived["run_label"].map(
            lambda value: split_run_label(str(value))[1]
        )
        derived_frames.append(derived)
    cells = pd.concat(derived_frames, ignore_index=True)
    focal_counts = (
        cells[cells["config_id"].isin(FOCAL_CONFIGS)]
        .groupby(
            ["source_label", "run_label", "intervention_id"],
            sort=True,
        )
        .agg(
            n_country_years=("country_year_id", "nunique"),
            n_rows=("country_year_id", "size"),
            n_configs=("config_id", "nunique"),
            n_chronologies=("chronology", "nunique"),
        )
        .reset_index()
    )
    failures: list[str] = []
    if set(focal_counts["source_label"]) != set(source_interventions):
        failures.append("source_coverage")
    if args.expected_country_years > 0 and not focal_counts[
        "n_country_years"
    ].eq(args.expected_country_years).all():
        failures.append("country_year_coverage")
    if not focal_counts["n_configs"].eq(len(FOCAL_CONFIGS)).all():
        failures.append("focal_configuration_coverage")
    if not focal_counts["n_chronologies"].eq(2).all():
        failures.append("chronology_coverage")
    if failures:
        raise RuntimeError(
            "Recharge intervention input QA failed: " + ", ".join(failures)
        )

    long = build_long_cells(cells)
    chronology_effects = chronology_pairs(long)
    multiplier_effects = intervention_contrasts(long)
    did_effects = difference_in_differences(chronology_effects)
    effects = pd.concat(
        [long, chronology_effects, multiplier_effects, did_effects],
        ignore_index=True,
        sort=False,
    )
    summary = summarize_effects(
        effects,
        args.bootstrap_draws,
        args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(summary, args.output)
    write_csv_gz(effects, args.raw_output)
    audit = {
        "analysis_id": "v42_recharge_multiplier_joint_portfolio",
        "created_utc": utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "status": "pass",
        "comparison": (
            "common 4 h LFP + CAES24/H2-48 tail versus common 4 h LFP "
            "+ additional 4 h LFP"
        ),
        "source_scenarios": {
            source: SOURCE_SCENARIOS[source]
            for source in sorted(source_interventions)
        },
        "interventions_by_source": {
            source: sorted(interventions)
            for source, interventions in source_interventions.items()
        },
        "recharge_power_multipliers": [1.0, 2.0, 4.0],
        "optional_uniform_annual_total_role": (
            "implementation-only timing sensitivity; excluded from the "
            "default production workflow and not interpreted as one-way "
            "mechanism evidence"
        ),
        "focal_configurations": list(FOCAL_CONFIGS),
        "country_block_bootstrap": {
            "draws": args.bootstrap_draws,
            "seed": args.seed,
            "independent_block": "country",
            "within_block": "all country-years",
        },
        "input_coverage": focal_counts.to_dict("records"),
        "inputs": [
            {
                "label": label,
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for label, path in inputs + arm_inputs
        ],
        "outputs": [
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in (args.output, args.raw_output)
        ],
    }
    write_json(args.audit_output, audit)
    print(
        json.dumps(
            {
                "status": "pass",
                "summary": str(args.output),
                "raw_effects": str(args.raw_output),
                "rows": int(len(summary)),
                "elapsed_seconds": audit["elapsed_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
