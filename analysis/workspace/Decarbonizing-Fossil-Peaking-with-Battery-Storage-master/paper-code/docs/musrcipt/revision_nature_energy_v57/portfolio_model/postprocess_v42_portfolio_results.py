#!/usr/bin/env python3
"""Post-process the v42 European 4 h LFP + LDES portfolio experiment.

The script consumes only the *new* v42 joint-portfolio outputs.  It never
reads or copies v41 headline values.  Its primary comparison is the annual
system-cost advantage of a common 4 h LFP layer plus one LDES tail option
relative to the same common layer plus an additional 4 h LFP tail option.

For every country-year, chronology and supported duration, the script:

1. normalizes the mix advantage by the common base-LFP power and annual task;
2. selects the best candidate technology at that duration;
3. identifies a strictly bounded positive--negative--positive valley on
   the prespecified 4--168 h domain;
4. forms paired clustered-minus-dispersed effects;
5. reports country-block bootstrap intervals; and
6. exports figure-ready source data plus a machine-readable audit summary.

The default inputs are the central fixed-scale run and two prespecified
scenario-31 sensitivity runs (fixed scale and bounded tail power).  For every
candidate table, the corresponding directly solved portfolio-arm table is
also required.  With ``--wait-for-inputs`` the process waits until all six
gzip files are present and stable, allowing it to be launched before the
Gurobi runs finish.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
V42_ROOT = HERE.parent
V40_ROOT = V42_ROOT.parent / "revision_nature_energy_v40"
DEFAULT_INPUTS = {
    "fixed_s20_central": (
        HERE
        / "full_132cy_fixed_s20_central"
        / "ldes_vs_extra_lfp4_results.csv.gz"
    ),
    "fixed_s31_conditional": (
        HERE / "full_132cy_fixed" / "ldes_vs_extra_lfp4_results.csv.gz"
    ),
    "optimized_s31_robustness": (
        HERE
        / "full_132cy_optimized"
        / "ldes_vs_extra_lfp4_results.csv.gz"
    ),
}
DEFAULT_SCENARIO_TABLE = (
    V40_ROOT
    / "currency_work"
    / "value_sensitivity_gap_summary_source_aware_eur.csv"
)
DEFAULT_OUTPUT = V42_ROOT / "source_data"
DEFAULT_BOOTSTRAP_DRAWS = 5_000
DEFAULT_SEED = 20260728
DEFAULT_ZERO_TOLERANCE = 1.0e-6
VALLEY_MIN_H = 4.0
VALLEY_MAX_H = 168.0
FOCAL_CONFIGS = ("caes_24h", "h2_cavern_48h")
EXPECTED_CHRONOLOGIES = ("dispersed", "clustered")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            json_ready(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.10g")


def write_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(
        path,
        index=False,
        compression={"method": "gzip", "mtime": 0},
        float_format="%.10g",
    )


def parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--input must use LABEL=/path/to/results.csv.gz"
        )
    label, path = value.split("=", maxsplit=1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Input label cannot be empty")
    return label, Path(path).expanduser()


def wait_for_inputs(
    paths: Iterable[Path],
    poll_seconds: float,
    timeout_seconds: float,
) -> None:
    paths = list(paths)
    started = time.monotonic()
    prior_sizes: dict[Path, int] = {}
    stable_polls: dict[Path, int] = {path: 0 for path in paths}
    while True:
        ready = True
        for path in paths:
            if not path.exists():
                ready = False
                stable_polls[path] = 0
                continue
            size = path.stat().st_size
            if size <= 0 or prior_sizes.get(path) != size:
                ready = False
                stable_polls[path] = 0
            else:
                stable_polls[path] += 1
                if stable_polls[path] < 1:
                    ready = False
            prior_sizes[path] = size
        if ready:
            return
        if timeout_seconds > 0 and time.monotonic() - started > timeout_seconds:
            missing = [str(path) for path in paths if not path.exists()]
            raise TimeoutError(
                "Timed out while waiting for v42 portfolio inputs; "
                f"missing={missing}"
            )
        time.sleep(poll_seconds)


def required_columns() -> set[str]:
    return {
        "country",
        "year",
        "chronology",
        "arm_type",
        "config_id",
        "tech_id",
        "technology_en",
        "duration_h",
        "tail_power_mode",
        "base_power_mw",
        "tail_power_mw",
        "base_energy_capacity_mwh",
        "tail_energy_capacity_mwh",
        "base_task_mwh",
        "base_charge_mwh",
        "tail_charge_mwh",
        "total_charge_mwh",
        "base_discharge_mwh",
        "tail_discharge_mwh",
        "total_discharge_mwh",
        "unserved_eens_mwh",
        "portfolio_efc_mw",
        "base_equivalent_full_cycles",
        "tail_equivalent_full_cycles",
        "base_fixed_cost_eur",
        "tail_fixed_cost_eur",
        "base_vom_cost_eur",
        "tail_vom_cost_eur",
        "charge_opportunity_cost_eur",
        "system_cost_eur",
        "energy_value_eur_mwh",
        "firm_value_eur_kw_year",
        "charge_price_eur_mwh",
        "shared_recharge_binding_hours",
        "base_charge_power_binding_hours",
        "tail_charge_power_binding_hours",
        "base_discharge_power_binding_hours",
        "tail_discharge_power_binding_hours",
        "base_only_total_discharge_mwh",
        "base_only_total_charge_mwh",
        "base_only_unserved_eens_mwh",
        "base_only_portfolio_efc_mw",
        "extra_lfp4_system_cost_eur",
        "extra_lfp4_unserved_eens_mwh",
        "extra_lfp4_portfolio_efc_mw",
        "extra_lfp4_firm_peak_reduction_mw",
        "extra_lfp4_total_discharge_mwh",
        "extra_lfp4_total_charge_mwh",
        "extra_lfp4_tail_power_mw",
        "extra_lfp4_tail_fixed_cost_eur",
        "mix_net_benefit_vs_extra_lfp4_eur",
        "mix_eens_reduction_vs_extra_lfp4_mwh",
        "mix_efc_gain_vs_extra_lfp4_mw",
        "mix_firm_peak_gain_vs_extra_lfp4_mw",
        "mix_discharge_gain_vs_extra_lfp4_mwh",
        "mix_charge_change_vs_extra_lfp4_mwh",
        "solver_status_name",
    }


def load_cells(inputs: list[tuple[str, Path]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for label, path in inputs:
        frame = pd.read_csv(path)
        missing = sorted(required_columns() - set(frame.columns))
        if missing:
            raise RuntimeError(f"{path} is missing required columns: {missing}")
        if not frame["arm_type"].eq("ldes_candidate").all():
            raise RuntimeError(
                f"{path} must contain candidate-comparison rows only"
            )
        if not frame["solver_status_name"].eq("optimal").all():
            bad = frame.loc[
                ~frame["solver_status_name"].eq("optimal"),
                ["country", "year", "chronology", "config_id"],
            ]
            raise RuntimeError(
                f"{path} contains non-optimal rows: {bad.head().to_dict('records')}"
            )
        observed_modes = frame["tail_power_mode"].dropna().unique().tolist()
        if len(observed_modes) != 1:
            raise RuntimeError(
                f"{path} has non-unique tail_power_mode={observed_modes}"
            )
        frame.insert(0, "source_label", label)
        frame.insert(1, "model_mode", str(observed_modes[0]))
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    duplicated = combined.duplicated(
        [
            "source_label",
            "model_mode",
            "country",
            "year",
            "chronology",
            "config_id",
        ]
    )
    if duplicated.any():
        raise RuntimeError(
            "Duplicate source/model/country/year/chronology/config cells "
            "detected"
        )
    if not set(combined["chronology"]).issubset(EXPECTED_CHRONOLOGIES):
        raise RuntimeError("Unexpected chronology label")
    return combined


def infer_arm_inputs(inputs: list[tuple[str, Path]]) -> list[tuple[str, Path]]:
    return [
        (label, path.parent / "portfolio_arm_results.csv.gz")
        for label, path in inputs
    ]


def load_arm_results(inputs: list[tuple[str, Path]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for label, path in inputs:
        frame = pd.read_csv(path)
        required = {
            "country",
            "year",
            "chronology",
            "arm_type",
            "config_id",
            "tail_power_mode",
            "total_discharge_mwh",
            "total_charge_mwh",
            "portfolio_efc_mw",
            "unserved_eens_mwh",
            "firm_peak_reduction_mw",
            "base_fixed_cost_eur",
            "tail_fixed_cost_eur",
            "base_vom_cost_eur",
            "tail_vom_cost_eur",
            "system_cost_eur",
            "solver_status_name",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise RuntimeError(
                f"{path} is missing direct-arm columns: {missing}"
            )
        observed_modes = frame["tail_power_mode"].dropna().unique().tolist()
        if len(observed_modes) != 1:
            raise RuntimeError(
                f"{path} has non-unique tail_power_mode={observed_modes}"
            )
        if not frame["solver_status_name"].eq("optimal").all():
            bad = frame.loc[
                ~frame["solver_status_name"].eq("optimal"),
                ["country", "year", "chronology", "arm_type", "config_id"],
            ]
            raise RuntimeError(
                f"{path} contains non-optimal direct arms: "
                f"{bad.head().to_dict('records')}"
            )
        frame.insert(0, "source_label", label)
        frame.insert(1, "model_mode", str(observed_modes[0]))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_valuation_scenarios(
    path: Path, scenario_ids: list[int]
) -> list[dict[str, float]]:
    frame = pd.read_csv(path)
    columns = [
        "scenario_id",
        "energy_value_eur_mwh",
        "firm_value_eur_kw_year",
        "charge_price_eur_mwh",
    ]
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f"{path} lacks scenario columns: {missing}")
    rows: list[dict[str, float]] = []
    for scenario_id in scenario_ids:
        unique = (
            frame.loc[frame["scenario_id"].eq(scenario_id), columns]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        if len(unique) != 1:
            raise RuntimeError(
                f"Scenario {scenario_id} is absent or non-unique in {path}"
            )
        row = unique.iloc[0]
        rows.append(
            {
                "scenario_id": int(row["scenario_id"]),
                "energy_value_eur_mwh": float(
                    row["energy_value_eur_mwh"]
                ),
                "firm_value_eur_kw_year": float(
                    row["firm_value_eur_kw_year"]
                ),
                "charge_price_eur_mwh": float(
                    row["charge_price_eur_mwh"]
                ),
            }
        )
    return rows


def parse_integer_list(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError(
            "Scenario IDs must be a non-empty unique comma list"
        )
    return result


def parse_float_list(value: str) -> list[float]:
    result = [
        float(item.strip()) for item in value.split(",") if item.strip()
    ]
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError(
            "Values must be a non-empty unique comma list"
        )
    return result


def add_direct_comparator(
    frame: pd.DataFrame, arms: pd.DataFrame
) -> pd.DataFrame:
    keys = [
        "source_label",
        "model_mode",
        "country",
        "year",
        "chronology",
    ]
    comparator_columns = keys + [
        "total_discharge_mwh",
        "total_charge_mwh",
        "portfolio_efc_mw",
        "unserved_eens_mwh",
        "firm_peak_reduction_mw",
        "base_fixed_cost_eur",
        "tail_fixed_cost_eur",
        "base_vom_cost_eur",
        "tail_vom_cost_eur",
        "system_cost_eur",
    ]
    comparator = arms[
        arms["arm_type"].eq("extra_lfp4_comparator")
    ][comparator_columns].copy()
    if comparator.duplicated(keys).any():
        raise RuntimeError("Direct extra-LFP comparator is not unique")
    comparator = comparator.rename(
        columns={
            column: f"direct_extra_lfp4_{column}"
            for column in comparator_columns
            if column not in keys
        }
    )
    result = frame.merge(
        comparator, on=keys, how="left", validate="many_to_one"
    )
    if result["direct_extra_lfp4_system_cost_eur"].isna().any():
        raise RuntimeError("Some candidate rows lack a direct comparator arm")
    candidate_value_columns = [
        "total_discharge_mwh",
        "total_charge_mwh",
        "portfolio_efc_mw",
        "unserved_eens_mwh",
        "base_fixed_cost_eur",
        "tail_fixed_cost_eur",
        "base_vom_cost_eur",
        "tail_vom_cost_eur",
        "system_cost_eur",
    ]
    direct_candidate = arms[arms["arm_type"].eq("ldes_candidate")][
        keys + ["config_id"] + candidate_value_columns
    ].copy()
    if direct_candidate.duplicated(keys + ["config_id"]).any():
        raise RuntimeError("Direct LDES candidate arm is not unique")
    direct_candidate = direct_candidate.rename(
        columns={
            column: f"direct_candidate_{column}"
            for column in candidate_value_columns
        }
    )
    result = result.merge(
        direct_candidate,
        on=keys + ["config_id"],
        how="left",
        validate="one_to_one",
    )
    candidate_errors = {
        column: (
            result[f"direct_candidate_{column}"] - result[column]
        ).abs().max()
        for column in candidate_value_columns
    }
    comparator_error = (
        result["direct_extra_lfp4_system_cost_eur"]
        - result["extra_lfp4_system_cost_eur"]
    ).abs()
    if (
        max(candidate_errors.values()) > 1.0e-3
        or comparator_error.max() > 1.0e-3
    ):
        raise RuntimeError(
            "Direct-arm and candidate-comparison records disagree"
        )
    return result


def add_derived_metrics(
    frame: pd.DataFrame,
    arms: pd.DataFrame,
    primary_scenario: dict[str, float],
) -> pd.DataFrame:
    result = add_direct_comparator(frame.copy(), arms)
    result["country_year_id"] = (
        result["country"].astype(str)
        + "-"
        + result["year"].astype(int).astype(str)
    )
    common_power_kw = result["base_power_mw"] * 1000.0
    result["common_base_power_kw"] = common_power_kw
    result["dispatch_scenario_id"] = result["scenario_id"]
    result["valuation_scenario_id"] = int(primary_scenario["scenario_id"])
    energy_value = float(primary_scenario["energy_value_eur_mwh"])
    firm_value = float(primary_scenario["firm_value_eur_kw_year"])
    charge_price = float(primary_scenario["charge_price_eur_mwh"])
    result["valuation_energy_value_eur_mwh"] = energy_value
    result["valuation_firm_value_eur_kw_year"] = firm_value
    result["valuation_charge_price_eur_mwh"] = charge_price

    # Reconstruct every candidate-minus-comparator service delta from the
    # directly solved arms. The candidate-comparison table also carries these
    # fields, but the direct reconstruction makes the main estimand auditable
    # and prevents a stale helper column from determining the v42 envelope.
    direct_deltas = {
        "mix_discharge_gain_vs_extra_lfp4_mwh": (
            result["total_discharge_mwh"]
            - result["direct_extra_lfp4_total_discharge_mwh"]
        ),
        "mix_charge_change_vs_extra_lfp4_mwh": (
            result["total_charge_mwh"]
            - result["direct_extra_lfp4_total_charge_mwh"]
        ),
        "mix_efc_gain_vs_extra_lfp4_mw": (
            result["portfolio_efc_mw"]
            - result["direct_extra_lfp4_portfolio_efc_mw"]
        ),
        "mix_eens_reduction_vs_extra_lfp4_mwh": (
            result["direct_extra_lfp4_unserved_eens_mwh"]
            - result["unserved_eens_mwh"]
        ),
    }
    for column, direct_value in direct_deltas.items():
        residual_column = f"{column}_direct_arm_residual"
        result[residual_column] = result[column] - direct_value
        if result[residual_column].abs().max() > 1.0e-6:
            raise RuntimeError(
                f"Direct-arm reconstruction disagrees with {column}"
            )
        result[column] = direct_value

    result["portfolio_net_value_efc_eur"] = (
        energy_value * result["total_discharge_mwh"]
        + firm_value * 1000.0 * result["portfolio_efc_mw"]
        - result["base_fixed_cost_eur"]
        - result["tail_fixed_cost_eur"]
        - result["base_vom_cost_eur"]
        - result["tail_vom_cost_eur"]
        - charge_price * result["total_charge_mwh"]
    )
    result["extra_lfp4_net_value_efc_eur"] = (
        energy_value
        * result["direct_extra_lfp4_total_discharge_mwh"]
        + firm_value
        * 1000.0
        * result["direct_extra_lfp4_portfolio_efc_mw"]
        - result["direct_extra_lfp4_base_fixed_cost_eur"]
        - result["direct_extra_lfp4_tail_fixed_cost_eur"]
        - result["direct_extra_lfp4_base_vom_cost_eur"]
        - result["direct_extra_lfp4_tail_vom_cost_eur"]
        - charge_price * result["direct_extra_lfp4_total_charge_mwh"]
    )
    result["mix_net_benefit_residual_peak_vs_extra_lfp4_eur"] = result[
        "mix_net_benefit_vs_extra_lfp4_eur"
    ]
    result[
        "mix_benefit_residual_peak_eur_per_kw_common_power_year"
    ] = (
        result["mix_net_benefit_residual_peak_vs_extra_lfp4_eur"]
        / common_power_kw
    )
    result["mix_benefit_residual_peak_eur_per_mwh_task"] = (
        result["mix_net_benefit_residual_peak_vs_extra_lfp4_eur"]
        / result["base_task_mwh"]
    )
    result["mix_net_benefit_efc_vs_extra_lfp4_eur"] = (
        result["portfolio_net_value_efc_eur"]
        - result["extra_lfp4_net_value_efc_eur"]
    )
    # Generic mix-benefit columns are intentionally aliases of the EFC-valued
    # main estimand. Residual-peak system cost is retained under an explicit
    # sensitivity name and never drives the duration envelope.
    result["mix_net_benefit_vs_extra_lfp4_eur"] = result[
        "mix_net_benefit_efc_vs_extra_lfp4_eur"
    ]
    result["mix_benefit_eur_per_kw_common_power_year"] = (
        result["mix_net_benefit_efc_vs_extra_lfp4_eur"] / common_power_kw
    )
    result["mix_benefit_eur_per_mwh_task"] = (
        result["mix_net_benefit_efc_vs_extra_lfp4_eur"]
        / result["base_task_mwh"]
    )
    result["mix_benefit_eur_per_kw_selected_tail_year"] = np.where(
        result["tail_power_mw"] > 0,
        result["mix_net_benefit_efc_vs_extra_lfp4_eur"]
        / (result["tail_power_mw"] * 1000.0),
        np.nan,
    )
    result["portfolio_task_service_mwh"] = (
        result["base_task_mwh"] - result["unserved_eens_mwh"]
    )
    result["extra_lfp4_task_service_mwh"] = (
        result["base_task_mwh"]
        - result["direct_extra_lfp4_unserved_eens_mwh"]
    )
    result["base_only_task_service_mwh"] = (
        result["base_task_mwh"] - result["base_only_unserved_eens_mwh"]
    )
    result["tail_discharge_share"] = np.where(
        result["total_discharge_mwh"] > 0,
        result["tail_discharge_mwh"] / result["total_discharge_mwh"],
        np.nan,
    )
    result["tail_charge_share"] = np.where(
        result["total_charge_mwh"] > 0,
        result["tail_charge_mwh"] / result["total_charge_mwh"],
        np.nan,
    )
    result["tail_power_share"] = np.where(
        result["base_power_mw"] + result["tail_power_mw"] > 0,
        result["tail_power_mw"]
        / (result["base_power_mw"] + result["tail_power_mw"]),
        np.nan,
    )
    result["tail_energy_share"] = np.where(
        result["base_energy_capacity_mwh"]
        + result["tail_energy_capacity_mwh"]
        > 0,
        result["tail_energy_capacity_mwh"]
        / (
            result["base_energy_capacity_mwh"]
            + result["tail_energy_capacity_mwh"]
        ),
        np.nan,
    )
    result["marginal_extra_lfp4_discharge_mwh"] = (
        result["direct_extra_lfp4_total_discharge_mwh"]
        - result["base_only_total_discharge_mwh"]
    )
    result["marginal_extra_lfp4_charge_mwh"] = (
        result["direct_extra_lfp4_total_charge_mwh"]
        - result["base_only_total_charge_mwh"]
    )
    result["marginal_extra_lfp4_efc"] = np.where(
        result["extra_lfp4_tail_power_mw"] > 0,
        result["marginal_extra_lfp4_discharge_mwh"]
        / (4.0 * result["extra_lfp4_tail_power_mw"]),
        np.nan,
    )

    result["extra_lfp4_vom_cost_eur"] = (
        result["direct_extra_lfp4_base_vom_cost_eur"]
        + result["direct_extra_lfp4_tail_vom_cost_eur"]
    )
    result["fixed_cost_contribution_eur"] = (
        result["direct_extra_lfp4_base_fixed_cost_eur"]
        + result["direct_extra_lfp4_tail_fixed_cost_eur"]
        - result["base_fixed_cost_eur"]
        - result["tail_fixed_cost_eur"]
    )
    result["variable_om_contribution_eur"] = (
        result["extra_lfp4_vom_cost_eur"]
        - result["base_vom_cost_eur"]
        - result["tail_vom_cost_eur"]
    )
    result["charging_cost_contribution_eur"] = (
        charge_price
        * (
            result["direct_extra_lfp4_total_charge_mwh"]
            - result["total_charge_mwh"]
        )
    )
    result["energy_service_contribution_eur"] = (
        energy_value * result["mix_discharge_gain_vs_extra_lfp4_mwh"]
    )
    result["firm_service_contribution_eur"] = (
        firm_value
        * 1000.0
        * result["mix_efc_gain_vs_extra_lfp4_mw"]
    )
    component_columns = [
        "fixed_cost_contribution_eur",
        "variable_om_contribution_eur",
        "charging_cost_contribution_eur",
        "energy_service_contribution_eur",
        "firm_service_contribution_eur",
    ]
    result["value_component_sum_eur"] = result[component_columns].sum(axis=1)
    result["value_component_identity_residual_eur"] = (
        result["value_component_sum_eur"]
        - result["mix_net_benefit_vs_extra_lfp4_eur"]
    )
    for column in component_columns:
        result[
            column.replace("_eur", "_eur_per_kw_common_power_year")
        ] = result[column] / common_power_kw
        result[
            column.replace("_eur", "_eur_per_mwh_task")
        ] = result[column] / result["base_task_mwh"]
    return result


def choose_duration_envelope(
    cells: pd.DataFrame,
    min_duration_h: float,
    max_duration_h: float,
    tolerance: float,
) -> pd.DataFrame:
    domain = cells[
        cells["duration_h"].between(
            min_duration_h, max_duration_h, inclusive="both"
        )
    ].copy()
    group_keys = [
        "source_label",
        "model_mode",
        "valuation_scenario_id",
        "country",
        "year",
        "country_year_id",
        "chronology",
        "duration_h",
    ]
    # Stable deterministic tie-break: benefit, then lower fixed cost, then ID.
    domain = domain.sort_values(
        group_keys
        + [
            "mix_benefit_eur_per_kw_common_power_year",
            "fixed_cost_eur_kw_year",
            "config_id",
        ],
        ascending=[True] * len(group_keys) + [False, True, True],
        kind="stable",
    )
    envelope = domain.drop_duplicates(group_keys, keep="first").copy()
    envelope = envelope.rename(
        columns={
            "config_id": "dominant_config_id",
            "tech_id": "dominant_tech_id",
            "technology_en": "dominant_technology_en",
            "mix_net_benefit_vs_extra_lfp4_eur": "best_mix_benefit_eur_year",
            "mix_benefit_eur_per_kw_common_power_year": (
                "best_mix_benefit_eur_per_kw_common_power_year"
            ),
            "mix_benefit_eur_per_mwh_task": (
                "best_mix_benefit_eur_per_mwh_task"
            ),
        }
    )
    benefit = envelope[
        "best_mix_benefit_eur_per_kw_common_power_year"
    ]
    envelope["distance_to_zero_eur_per_kw_common_power_year"] = benefit
    envelope["positive_mix_advantage"] = benefit > tolerance
    envelope["negative_mix_advantage"] = benefit < -tolerance
    envelope["numerically_ambiguous_mix_advantage"] = (
        benefit.abs() <= tolerance
    )
    envelope["choice_envelope_eur_per_kw_common_power_year"] = benefit.clip(
        lower=0.0
    )
    envelope = envelope.sort_values(
        [
            "source_label",
            "model_mode",
            "valuation_scenario_id",
            "country",
            "year",
            "chronology",
            "duration_h",
        ],
        kind="stable",
    ).reset_index(drop=True)
    return envelope


def find_valley(
    frame: pd.DataFrame,
    tolerance: float,
) -> dict[str, Any]:
    ordered = frame.sort_values("duration_h", kind="stable").reset_index(
        drop=True
    )
    values = ordered[
        "best_mix_benefit_eur_per_kw_common_power_year"
    ].to_numpy(float)
    positive = values > tolerance
    negative = values < -tolerance
    candidates: list[dict[str, Any]] = []
    index = 0
    while index < len(ordered):
        if not negative[index]:
            index += 1
            continue
        start = index
        while index + 1 < len(ordered) and negative[index + 1]:
            index += 1
        end = index
        negative_count = end - start + 1
        if (
            negative_count >= 2
            and start > 0
            and end < len(ordered) - 1
        ):
            left = ordered.iloc[start - 1]
            right = ordered.iloc[end + 1]
            gap = ordered.iloc[start : end + 1]
            if positive[start - 1] and positive[end + 1]:
                minimum_index = gap[
                    "best_mix_benefit_eur_per_kw_common_power_year"
                ].idxmin()
                minimum = ordered.loc[minimum_index]
                left_h = float(left["duration_h"])
                right_h = float(right["duration_h"])
                candidates.append(
                    {
                        "left_boundary_h": left_h,
                        "right_boundary_h": right_h,
                        "valley_start_nonpositive_h": float(
                            gap["duration_h"].iloc[0]
                        ),
                        "valley_end_nonpositive_h": float(
                            gap["duration_h"].iloc[-1]
                        ),
                        "valley_width_h": right_h - left_h,
                        "valley_width_log_ratio": math.log(
                            right_h / left_h
                        ),
                        "negative_node_count": int(len(gap)),
                        "nonpositive_node_count": int(len(gap)),
                        "minimum_gap_depth_eur_per_kw_common_power_year": (
                            float(
                                minimum[
                                    "best_mix_benefit_eur_per_kw_common_power_year"
                                ]
                            )
                        ),
                        "mean_gap_depth_eur_per_kw_common_power_year": (
                            float(
                                gap[
                                    "best_mix_benefit_eur_per_kw_common_power_year"
                                ].mean()
                            )
                        ),
                        "minimum_gap_duration_h": float(
                            minimum["duration_h"]
                        ),
                        "left_boundary_config_id": str(
                            left["dominant_config_id"]
                        ),
                        "left_boundary_tech_id": str(
                            left["dominant_tech_id"]
                        ),
                        "right_boundary_config_id": str(
                            right["dominant_config_id"]
                        ),
                        "right_boundary_tech_id": str(
                            right["dominant_tech_id"]
                        ),
                        "negative_durations_h": "|".join(
                            f"{float(value):g}"
                            for value in gap["duration_h"]
                        ),
                        "nonpositive_durations_h": "|".join(
                            f"{float(value):g}"
                            for value in gap["duration_h"]
                        ),
                    }
                )
        index += 1
    # A strict classification is withheld if any tested node lies inside the
    # declared accounting/solver tolerance. This prevents a numerically
    # unresolved sign elsewhere on the same envelope from being ignored.
    if candidates and np.any(np.abs(values) <= tolerance):
        candidates = []
    if not candidates:
        if len(values) and np.all(positive):
            envelope_shape = "full_ldes_entry"
        elif len(values) and np.all(negative):
            envelope_shape = "full_ldes_exit"
        elif (
            len(values)
            and positive[0]
            and not positive[-1]
            and np.sum(np.diff(positive.astype(int)) != 0) == 1
        ):
            envelope_shape = "positive_to_exit_frontier"
        elif (
            len(values)
            and not positive[0]
            and positive[-1]
            and np.sum(np.diff(positive.astype(int)) != 0) == 1
        ):
            envelope_shape = "entry_frontier"
        else:
            envelope_shape = "complex_nonvalley"
        return {
            "strict_valley": False,
            "envelope_shape": envelope_shape,
            "strict_valley_count": 0,
            "left_boundary_h": np.nan,
            "right_boundary_h": np.nan,
            "valley_start_nonpositive_h": np.nan,
            "valley_end_nonpositive_h": np.nan,
            "valley_width_h": 0.0,
            "valley_width_log_ratio": 0.0,
            "nonpositive_node_count": 0,
            "negative_node_count": 0,
            "minimum_gap_depth_eur_per_kw_common_power_year": np.nan,
            "mean_gap_depth_eur_per_kw_common_power_year": np.nan,
            "minimum_gap_duration_h": np.nan,
            "left_boundary_config_id": "",
            "left_boundary_tech_id": "",
            "right_boundary_config_id": "",
            "right_boundary_tech_id": "",
            "nonpositive_durations_h": "",
            "negative_durations_h": "",
            "right_censored": bool(
                len(ordered)
                and values[-1] <= tolerance
                and np.any(positive)
            ),
            "ambiguous_node_count": int(
                np.sum(np.abs(values) <= tolerance)
            ),
            "positive_node_count": int(np.sum(positive)),
            "negative_node_count_total": int(np.sum(negative)),
            "minimum_absolute_sign_margin_eur_per_kw_common_power_year": (
                float(np.min(np.abs(values))) if len(values) else np.nan
            ),
        }
    # The strict valley is the widest bounded non-positive interval.  Ties
    # are resolved by the deeper minimum, then the earlier left boundary.
    candidates.sort(
        key=lambda item: (
            -item["valley_width_h"],
            item["minimum_gap_depth_eur_per_kw_common_power_year"],
            item["left_boundary_h"],
        )
    )
    selected = candidates[0]
    selected["strict_valley"] = True
    selected["envelope_shape"] = "strict_valley"
    selected["strict_valley_count"] = len(candidates)
    selected["right_censored"] = False
    selected["ambiguous_node_count"] = int(
        np.sum(np.abs(values) <= tolerance)
    )
    selected[
        "minimum_absolute_sign_margin_eur_per_kw_common_power_year"
    ] = float(np.min(np.abs(values))) if len(values) else np.nan
    selected["positive_node_count"] = int(np.sum(positive))
    selected["negative_node_count_total"] = int(np.sum(negative))
    return selected


def build_valley_records(
    envelope: pd.DataFrame,
    tolerance: float,
) -> pd.DataFrame:
    keys = [
        "source_label",
        "model_mode",
        "valuation_scenario_id",
        "country",
        "year",
        "country_year_id",
        "chronology",
    ]
    rows: list[dict[str, Any]] = []
    for key, group in envelope.groupby(keys, sort=True, dropna=False):
        record = dict(zip(keys, key))
        record.update(find_valley(group, tolerance))
        record["tested_duration_count"] = int(group["duration_h"].nunique())
        record["minimum_tested_duration_h"] = float(group["duration_h"].min())
        record["maximum_tested_duration_h"] = float(group["duration_h"].max())
        record["zero_tolerance_eur_per_kw_common_power_year"] = tolerance
        rows.append(record)
    return pd.DataFrame(rows)


def pivot_paired(
    frame: pd.DataFrame,
    index: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    subset = frame[index + ["chronology"] + metrics].copy()
    wide = subset.pivot(
        index=index, columns="chronology", values=metrics
    ).reset_index()
    wide.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in wide.columns
    ]
    for metric in metrics:
        dispersed = f"{metric}_dispersed"
        clustered = f"{metric}_clustered"
        if dispersed not in wide or clustered not in wide:
            raise RuntimeError(
                f"Cannot form complete chronology pair for {metric}"
            )
        if wide[[dispersed, clustered]].isna().any().any():
            raise RuntimeError(
                f"Incomplete dispersed/clustered chronology pair for {metric}"
            )
        wide[f"clustered_minus_dispersed_{metric}"] = (
            wide[clustered] - wide[dispersed]
        )
    return wide


def build_config_pairs(cells: pd.DataFrame) -> pd.DataFrame:
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
    ]
    metrics = [
        "mix_net_benefit_vs_extra_lfp4_eur",
        "mix_benefit_eur_per_kw_common_power_year",
        "mix_benefit_eur_per_mwh_task",
        "mix_eens_reduction_vs_extra_lfp4_mwh",
        "mix_efc_gain_vs_extra_lfp4_mw",
        "mix_firm_peak_gain_vs_extra_lfp4_mw",
        "mix_discharge_gain_vs_extra_lfp4_mwh",
        "mix_charge_change_vs_extra_lfp4_mwh",
        "base_charge_mwh",
        "tail_charge_mwh",
        "total_charge_mwh",
        "base_discharge_mwh",
        "tail_discharge_mwh",
        "total_discharge_mwh",
        "base_equivalent_full_cycles",
        "tail_equivalent_full_cycles",
        "unserved_eens_mwh",
        "portfolio_efc_mw",
        "tail_power_mw",
        "tail_discharge_share",
        "tail_charge_share",
        "shared_recharge_binding_hours",
        "base_charge_power_binding_hours",
        "tail_charge_power_binding_hours",
        "base_discharge_power_binding_hours",
        "tail_discharge_power_binding_hours",
        "fixed_cost_contribution_eur",
        "variable_om_contribution_eur",
        "charging_cost_contribution_eur",
        "energy_service_contribution_eur",
        "firm_service_contribution_eur",
    ]
    return pivot_paired(cells, index, metrics)


def build_envelope_pairs(envelope: pd.DataFrame) -> pd.DataFrame:
    index = [
        "source_label",
        "model_mode",
        "valuation_scenario_id",
        "country",
        "year",
        "country_year_id",
        "duration_h",
    ]
    metrics = [
        "best_mix_benefit_eur_year",
        "best_mix_benefit_eur_per_kw_common_power_year",
        "best_mix_benefit_eur_per_mwh_task",
        "mix_eens_reduction_vs_extra_lfp4_mwh",
        "mix_efc_gain_vs_extra_lfp4_mw",
        "mix_discharge_gain_vs_extra_lfp4_mwh",
        "mix_charge_change_vs_extra_lfp4_mwh",
        "tail_equivalent_full_cycles",
        "tail_discharge_mwh",
        "tail_charge_mwh",
        "tail_discharge_share",
        "tail_charge_share",
    ]
    pairs = pivot_paired(envelope, index, metrics)
    tech = envelope.pivot(
        index=index,
        columns="chronology",
        values=["dominant_config_id", "dominant_tech_id"],
    ).reset_index()
    tech.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in tech.columns
    ]
    return pairs.merge(tech, on=index, how="left", validate="one_to_one")


def build_valley_pairs(records: pd.DataFrame) -> pd.DataFrame:
    index = [
        "source_label",
        "model_mode",
        "valuation_scenario_id",
        "country",
        "year",
        "country_year_id",
    ]
    metrics = [
        "strict_valley",
        "left_boundary_h",
        "right_boundary_h",
        "valley_width_h",
        "valley_width_log_ratio",
        "negative_node_count",
        "nonpositive_node_count",
        "minimum_gap_depth_eur_per_kw_common_power_year",
        "mean_gap_depth_eur_per_kw_common_power_year",
        "right_censored",
    ]
    wide = records.pivot(
        index=index, columns="chronology", values=metrics
    ).reset_index()
    wide.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in wide.columns
    ]
    for chronology in EXPECTED_CHRONOLOGIES:
        column = f"strict_valley_{chronology}"
        if column not in wide or wide[column].isna().any():
            raise RuntimeError(
                "Incomplete dispersed/clustered chronology pair in "
                "strict-valley records"
            )
        wide[column] = wide[column].astype(bool)
    wide["delta_valley_width_h"] = (
        wide["valley_width_h_clustered"]
        - wide["valley_width_h_dispersed"]
    )
    wide["delta_valley_width_log_ratio"] = (
        wide["valley_width_log_ratio_clustered"]
        - wide["valley_width_log_ratio_dispersed"]
    )
    dispersed = wide["strict_valley_dispersed"]
    clustered = wide["strict_valley_clustered"]
    wide["cluster_induced_valley"] = ~dispersed & clustered
    wide["valley_retained"] = dispersed & clustered
    wide["valley_dissolved"] = dispersed & ~clustered
    wide["valley_never_formed"] = ~dispersed & ~clustered
    wide["valley_transition"] = np.select(
        [
            wide["cluster_induced_valley"],
            wide["valley_retained"],
            wide["valley_dissolved"],
        ],
        ["cluster_induced", "retained", "dissolved"],
        default="never",
    )
    wide["left_boundary_shift_h"] = (
        wide["left_boundary_h_clustered"]
        - wide["left_boundary_h_dispersed"]
    )
    wide["right_boundary_shift_h"] = (
        wide["right_boundary_h_clustered"]
        - wide["right_boundary_h_dispersed"]
    )
    return wide


def summarize_valley_shapes(records: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "source_label",
        "model_mode",
        "valuation_scenario_id",
        "chronology",
    ]
    totals = (
        records.groupby(keys, sort=True)
        .agg(
            n_country_years=("country_year_id", "size"),
            n_countries=("country", "nunique"),
            strict_valley_count=("strict_valley", "sum"),
            strict_valley_rate=("strict_valley", "mean"),
            right_censored_count=("right_censored", "sum"),
        )
        .reset_index()
    )
    shapes = (
        records.groupby(keys + ["envelope_shape"], sort=True)
        .size()
        .rename("shape_count")
        .reset_index()
    )
    shapes = shapes.merge(totals, on=keys, how="left", validate="many_to_one")
    shapes["shape_rate"] = shapes["shape_count"] / shapes["n_country_years"]
    return shapes


def quantiles(values: pd.Series) -> dict[str, float]:
    clean = values.dropna().to_numpy(float)
    if clean.size == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "q05": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "q95": np.nan,
        }
    q05, q25, median, q75, q95 = np.quantile(
        clean, [0.05, 0.25, 0.5, 0.75, 0.95]
    )
    return {
        "mean": float(np.mean(clean)),
        "median": float(median),
        "q05": float(q05),
        "q25": float(q25),
        "q75": float(q75),
        "q95": float(q95),
    }


def summarize_duration(envelope: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "source_label",
        "model_mode",
        "valuation_scenario_id",
        "chronology",
        "duration_h",
    ]
    metrics = [
        "best_mix_benefit_eur_per_kw_common_power_year",
        "best_mix_benefit_eur_per_mwh_task",
        "mix_eens_reduction_vs_extra_lfp4_mwh",
        "mix_efc_gain_vs_extra_lfp4_mw",
        "tail_equivalent_full_cycles",
        "tail_discharge_mwh",
        "tail_charge_mwh",
        "tail_discharge_share",
        "tail_charge_share",
    ]
    rows: list[dict[str, Any]] = []
    for key, group in envelope.groupby(keys, sort=True, dropna=False):
        row = dict(zip(keys, key))
        row["n_country_years"] = int(
            group[["country", "year"]].drop_duplicates().shape[0]
        )
        row["n_countries"] = int(group["country"].nunique())
        row["positive_mix_share"] = float(
            (
                group[
                    "best_mix_benefit_eur_per_kw_common_power_year"
                ]
                > 0.0
            ).mean()
        )
        counts = group["dominant_tech_id"].value_counts()
        row["modal_dominant_tech_id"] = (
            str(counts.index[0]) if len(counts) else ""
        )
        row["dominant_technology_counts"] = "|".join(
            f"{technology}:{int(count)}"
            for technology, count in counts.items()
        )
        for metric in metrics:
            for statistic, value in quantiles(group[metric]).items():
                row[f"{metric}_{statistic}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_configurations(cells: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "source_label",
        "model_mode",
        "valuation_scenario_id",
        "chronology",
        "config_id",
        "tech_id",
        "technology_en",
        "duration_h",
    ]
    metrics = [
        "mix_benefit_eur_per_kw_common_power_year",
        "mix_benefit_eur_per_mwh_task",
        "mix_eens_reduction_vs_extra_lfp4_mwh",
        "mix_efc_gain_vs_extra_lfp4_mw",
        "mix_discharge_gain_vs_extra_lfp4_mwh",
        "mix_charge_change_vs_extra_lfp4_mwh",
        "base_equivalent_full_cycles",
        "tail_equivalent_full_cycles",
        "base_discharge_mwh",
        "tail_discharge_mwh",
        "base_charge_mwh",
        "tail_charge_mwh",
        "tail_power_mw",
        "tail_discharge_share",
        "tail_charge_share",
        "shared_recharge_binding_hours",
    ]
    rows: list[dict[str, Any]] = []
    for key, group in cells.groupby(keys, sort=True, dropna=False):
        row = dict(zip(keys, key))
        row["n_country_years"] = int(
            group[["country", "year"]].drop_duplicates().shape[0]
        )
        row["n_countries"] = int(group["country"].nunique())
        row["positive_mix_share"] = float(
            (
                group["mix_benefit_eur_per_kw_common_power_year"] > 0.0
            ).mean()
        )
        for metric in metrics:
            for statistic, value in quantiles(group[metric]).items():
                row[f"{metric}_{statistic}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def country_block_bootstrap(
    frame: pd.DataFrame,
    value_column: str,
    draws: int,
    rng: np.random.Generator,
    country_aggregation: str = "mean",
) -> dict[str, float]:
    if country_aggregation not in {"mean", "median"}:
        raise ValueError(country_aggregation)
    country_values = (
        frame.groupby("country", sort=True)[value_column]
        .agg(country_aggregation)
        .dropna()
        .to_numpy(float)
    )
    if len(country_values) == 0:
        return {
            "point_estimate": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n_countries": 0,
        }
    point = (
        float(np.mean(country_values))
        if country_aggregation == "mean"
        else float(np.median(country_values))
    )
    indices = rng.integers(
        0, len(country_values), size=(draws, len(country_values))
    )
    sampled = country_values[indices]
    estimates = (
        sampled.mean(axis=1)
        if country_aggregation == "mean"
        else np.median(sampled, axis=1)
    )
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "point_estimate": point,
        "ci_low": float(low),
        "ci_high": float(high),
        "n_countries": int(len(country_values)),
    }


def add_bootstrap_rows(
    rows: list[dict[str, Any]],
    frame: pd.DataFrame,
    analysis: str,
    group_keys: list[str],
    metrics: list[str],
    draws: int,
    seed_sequence: np.random.SeedSequence,
    country_aggregation: str = "mean",
) -> None:
    groups = list(frame.groupby(group_keys, sort=True, dropna=False))
    children = seed_sequence.spawn(len(groups) * len(metrics))
    child_index = 0
    for key, group in groups:
        if not isinstance(key, tuple):
            key = (key,)
        identity = dict(zip(group_keys, key))
        for metric in metrics:
            rng = np.random.default_rng(children[child_index])
            child_index += 1
            estimate = country_block_bootstrap(
                group,
                metric,
                draws,
                rng,
                country_aggregation=country_aggregation,
            )
            rows.append(
                {
                    "analysis": analysis,
                    **identity,
                    "metric": metric,
                    "country_aggregation": country_aggregation,
                    "bootstrap_draws": draws,
                    "n_country_years": int(
                        group[["country", "year"]]
                        .drop_duplicates()
                        .shape[0]
                    ),
                    **estimate,
                }
            )


def build_bootstrap_summary(
    config_pairs: pd.DataFrame,
    envelope_pairs: pd.DataFrame,
    valley_pairs: pd.DataFrame,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    valley_pairs = valley_pairs.copy()
    seed_sequence = np.random.SeedSequence(seed)
    focal = config_pairs[config_pairs["config_id"].isin(FOCAL_CONFIGS)]
    focal_metrics = [
        "clustered_minus_dispersed_mix_benefit_eur_per_kw_common_power_year",
        "clustered_minus_dispersed_mix_benefit_eur_per_mwh_task",
        "clustered_minus_dispersed_mix_eens_reduction_vs_extra_lfp4_mwh",
        "clustered_minus_dispersed_mix_efc_gain_vs_extra_lfp4_mw",
        "clustered_minus_dispersed_tail_equivalent_full_cycles",
        "clustered_minus_dispersed_tail_discharge_mwh",
        "clustered_minus_dispersed_tail_charge_mwh",
    ]
    add_bootstrap_rows(
        rows,
        focal,
        "focal_configuration_paired_effect",
        [
            "source_label",
            "model_mode",
            "valuation_scenario_id",
            "config_id",
            "tech_id",
            "duration_h",
        ],
        focal_metrics,
        draws,
        seed_sequence,
    )
    envelope_metrics = [
        (
            "clustered_minus_dispersed_"
            "best_mix_benefit_eur_per_kw_common_power_year"
        ),
        (
            "clustered_minus_dispersed_"
            "best_mix_benefit_eur_per_mwh_task"
        ),
        "clustered_minus_dispersed_mix_eens_reduction_vs_extra_lfp4_mwh",
        "clustered_minus_dispersed_mix_efc_gain_vs_extra_lfp4_mw",
        "clustered_minus_dispersed_mix_discharge_gain_vs_extra_lfp4_mwh",
        "clustered_minus_dispersed_mix_charge_change_vs_extra_lfp4_mwh",
        "clustered_minus_dispersed_tail_equivalent_full_cycles",
        "clustered_minus_dispersed_tail_discharge_mwh",
        "clustered_minus_dispersed_tail_charge_mwh",
        "clustered_minus_dispersed_tail_discharge_share",
        "clustered_minus_dispersed_tail_charge_share",
    ]
    add_bootstrap_rows(
        rows,
        envelope_pairs,
        "duration_envelope_paired_effect",
        [
            "source_label",
            "model_mode",
            "valuation_scenario_id",
            "duration_h",
        ],
        envelope_metrics,
        draws,
        seed_sequence,
    )
    valley_metrics = [
        "delta_valley_width_h",
        "delta_valley_width_log_ratio",
        "cluster_induced_valley",
        "valley_retained",
        "valley_dissolved",
        "valley_never_formed",
    ]
    for column in valley_metrics[2:]:
        valley_pairs[column] = valley_pairs[column].astype(float)
    add_bootstrap_rows(
        rows,
        valley_pairs,
        "strict_valley_paired_effect",
        ["source_label", "model_mode", "valuation_scenario_id"],
        valley_metrics,
        draws,
        seed_sequence,
        country_aggregation="mean",
    )
    # The prespecified central tendency for valley widening is the median of
    # country-level median paired widths; report its cluster bootstrap too.
    add_bootstrap_rows(
        rows,
        valley_pairs,
        "strict_valley_paired_effect_median",
        ["source_label", "model_mode", "valuation_scenario_id"],
        ["delta_valley_width_h", "delta_valley_width_log_ratio"],
        draws,
        seed_sequence,
        country_aggregation="median",
    )
    return pd.DataFrame(rows)


def build_focal_summary(
    cells: pd.DataFrame,
    config_pairs: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    focal = cells[cells["config_id"].isin(FOCAL_CONFIGS)]
    metrics = [
        "mix_benefit_eur_per_kw_common_power_year",
        "mix_benefit_eur_per_mwh_task",
        "mix_eens_reduction_vs_extra_lfp4_mwh",
        "mix_efc_gain_vs_extra_lfp4_mw",
        "tail_equivalent_full_cycles",
        "tail_discharge_mwh",
        "tail_charge_mwh",
        "tail_discharge_share",
        "tail_charge_share",
    ]
    keys = [
        "source_label",
        "model_mode",
        "valuation_scenario_id",
        "chronology",
        "config_id",
        "tech_id",
        "technology_en",
        "duration_h",
    ]
    for key, group in focal.groupby(keys, sort=True, dropna=False):
        identity = dict(zip(keys, key))
        for metric in metrics:
            summary = quantiles(group[metric])
            rows.append(
                {
                    **identity,
                    "estimand_type": "chronology_level",
                    "metric": metric,
                    "n_country_years": int(
                        group[["country", "year"]]
                        .drop_duplicates()
                        .shape[0]
                    ),
                    "n_countries": int(group["country"].nunique()),
                    **summary,
                    "bootstrap_ci_low": np.nan,
                    "bootstrap_ci_high": np.nan,
                }
            )
    paired_bootstrap = bootstrap[
        bootstrap["analysis"].eq("focal_configuration_paired_effect")
    ]
    for record in paired_bootstrap.to_dict("records"):
        rows.append(
            {
                "source_label": record["source_label"],
                "model_mode": record["model_mode"],
                "valuation_scenario_id": record[
                    "valuation_scenario_id"
                ],
                "chronology": "clustered_minus_dispersed",
                "config_id": record["config_id"],
                "tech_id": record["tech_id"],
                "technology_en": "",
                "duration_h": record["duration_h"],
                "estimand_type": "paired_country_block_bootstrap",
                "metric": record["metric"],
                "n_country_years": record["n_country_years"],
                "n_countries": record["n_countries"],
                "mean": record["point_estimate"],
                "median": np.nan,
                "q05": np.nan,
                "q25": np.nan,
                "q75": np.nan,
                "q95": np.nan,
                "bootstrap_ci_low": record["ci_low"],
                "bootstrap_ci_high": record["ci_high"],
            }
        )
    return pd.DataFrame(rows)


def select_layer_metrics(envelope: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "source_label",
        "model_mode",
        "valuation_scenario_id",
        "country",
        "year",
        "country_year_id",
        "chronology",
        "duration_h",
        "dominant_config_id",
        "dominant_tech_id",
        "dominant_technology_en",
        "base_power_mw",
        "tail_power_mw",
        "base_energy_capacity_mwh",
        "tail_energy_capacity_mwh",
        "base_discharge_mwh",
        "tail_discharge_mwh",
        "total_discharge_mwh",
        "base_charge_mwh",
        "tail_charge_mwh",
        "total_charge_mwh",
        "base_equivalent_full_cycles",
        "tail_equivalent_full_cycles",
        "marginal_extra_lfp4_discharge_mwh",
        "marginal_extra_lfp4_charge_mwh",
        "marginal_extra_lfp4_efc",
        "tail_power_share",
        "tail_energy_share",
        "tail_discharge_share",
        "tail_charge_share",
        "shared_recharge_binding_hours",
        "base_charge_power_binding_hours",
        "tail_charge_power_binding_hours",
        "base_discharge_power_binding_hours",
        "tail_discharge_power_binding_hours",
        "portfolio_task_service_mwh",
        "extra_lfp4_task_service_mwh",
        "mix_eens_reduction_vs_extra_lfp4_mwh",
        "portfolio_efc_mw",
        "extra_lfp4_portfolio_efc_mw",
        "mix_efc_gain_vs_extra_lfp4_mw",
        "best_mix_benefit_eur_per_kw_common_power_year",
        "best_mix_benefit_eur_per_mwh_task",
    ]
    return envelope[columns].copy()


def select_value_components(envelope: pd.DataFrame) -> pd.DataFrame:
    base_columns = [
        "source_label",
        "model_mode",
        "valuation_scenario_id",
        "country",
        "year",
        "country_year_id",
        "chronology",
        "duration_h",
        "dominant_config_id",
        "dominant_tech_id",
        "best_mix_benefit_eur_year",
        "best_mix_benefit_eur_per_kw_common_power_year",
        "best_mix_benefit_eur_per_mwh_task",
        "value_component_identity_residual_eur",
    ]
    components = [
        "fixed_cost_contribution",
        "variable_om_contribution",
        "charging_cost_contribution",
        "energy_service_contribution",
        "firm_service_contribution",
    ]
    frames: list[pd.DataFrame] = []
    for component in components:
        value = envelope[base_columns].copy()
        value["component"] = component
        value["component_value_eur_year"] = envelope[f"{component}_eur"]
        value[
            "component_value_eur_per_kw_common_power_year"
        ] = envelope[f"{component}_eur_per_kw_common_power_year"]
        value["component_value_eur_per_mwh_task"] = envelope[
            f"{component}_eur_per_mwh_task"
        ]
        frames.append(value)
    return pd.concat(frames, ignore_index=True)


def build_value_scan(
    cells: pd.DataFrame,
    energy_values: list[float],
    charge_values: list[float],
    firm_values: list[float],
    cost_multipliers: list[float],
    tolerance: float,
) -> dict[str, pd.DataFrame]:
    """Scan conditional EFC value structures without reusing v41 results."""
    domain = cells[
        cells["duration_h"].between(
            VALLEY_MIN_H, VALLEY_MAX_H, inclusive="both"
        )
    ].copy()
    identity = [
        "source_label",
        "model_mode",
        "country",
        "year",
        "country_year_id",
        "chronology",
        "duration_h",
        "config_id",
        "tech_id",
        "technology_en",
        "fixed_cost_eur_kw_year",
        "common_base_power_kw",
        "base_task_mwh",
        "mix_discharge_gain_vs_extra_lfp4_mwh",
        "mix_efc_gain_vs_extra_lfp4_mw",
        "mix_charge_change_vs_extra_lfp4_mwh",
        "fixed_cost_contribution_eur",
        "variable_om_contribution_eur",
        "tail_fixed_cost_eur",
    ]
    domain = domain[identity].drop_duplicates(
        [
            "source_label",
            "model_mode",
            "country",
            "year",
            "chronology",
            "config_id",
        ]
    )
    envelope_keys = [
        "source_label",
        "model_mode",
        "country",
        "year",
        "country_year_id",
        "chronology",
        "duration_h",
    ]
    valley_rows: list[dict[str, Any]] = []
    duration_rows: list[dict[str, Any]] = []
    for energy_value in energy_values:
        for charge_value in charge_values:
            for firm_value in firm_values:
                for cost_multiplier in cost_multipliers:
                    scan_id = (
                        f"E{energy_value:g}_Q{charge_value:g}_"
                        f"F{firm_value:g}_M{cost_multiplier:g}"
                    )
                    work = domain.copy()
                    work[
                        "best_mix_benefit_eur_per_kw_common_power_year"
                    ] = (
                        work["fixed_cost_contribution_eur"]
                        + (1.0 - cost_multiplier)
                        * work["tail_fixed_cost_eur"]
                        + work["variable_om_contribution_eur"]
                        + energy_value
                        * work["mix_discharge_gain_vs_extra_lfp4_mwh"]
                        + firm_value
                        * 1000.0
                        * work["mix_efc_gain_vs_extra_lfp4_mw"]
                        - charge_value
                        * work["mix_charge_change_vs_extra_lfp4_mwh"]
                    ) / work["common_base_power_kw"]
                    work = work.sort_values(
                        envelope_keys
                        + [
                            "best_mix_benefit_eur_per_kw_common_power_year",
                            "fixed_cost_eur_kw_year",
                            "config_id",
                        ],
                        ascending=[True] * len(envelope_keys)
                        + [False, True, True],
                        kind="stable",
                    )
                    selected = work.drop_duplicates(
                        envelope_keys, keep="first"
                    ).rename(
                        columns={
                            "config_id": "dominant_config_id",
                            "tech_id": "dominant_tech_id",
                        }
                    )
                    selected["scan_id"] = scan_id
                    selected["energy_value_eur_mwh"] = energy_value
                    selected["charge_price_eur_mwh"] = charge_value
                    selected["firm_value_eur_kw_year"] = firm_value
                    selected[
                        "ldes_tail_fixed_cost_multiplier"
                    ] = cost_multiplier
                    valley_group_keys = [
                        "source_label",
                        "model_mode",
                        "country",
                        "year",
                        "country_year_id",
                        "chronology",
                    ]
                    for key, group in selected.groupby(
                        valley_group_keys,
                        sort=True,
                        dropna=False,
                    ):
                        record = dict(zip(valley_group_keys, key))
                        record.update(
                            {
                                "scan_id": scan_id,
                                "energy_value_eur_mwh": energy_value,
                                "charge_price_eur_mwh": charge_value,
                                "firm_value_eur_kw_year": firm_value,
                                "ldes_tail_fixed_cost_multiplier": (
                                    cost_multiplier
                                ),
                            }
                        )
                        record.update(find_valley(group, tolerance))
                        valley_rows.append(record)
                    for key, group in selected.groupby(
                        [
                            "source_label",
                            "model_mode",
                            "chronology",
                            "duration_h",
                        ],
                        sort=True,
                        dropna=False,
                    ):
                        source_label, mode, chronology, duration_h = key
                        values = group[
                            "best_mix_benefit_eur_per_kw_common_power_year"
                        ]
                        summary = quantiles(values)
                        technology_counts = group[
                            "dominant_tech_id"
                        ].value_counts()
                        duration_rows.append(
                            {
                                "scan_id": scan_id,
                                "source_label": source_label,
                                "model_mode": mode,
                                "chronology": chronology,
                                "duration_h": duration_h,
                                "energy_value_eur_mwh": energy_value,
                                "charge_price_eur_mwh": charge_value,
                                "firm_value_eur_kw_year": firm_value,
                                "ldes_tail_fixed_cost_multiplier": (
                                    cost_multiplier
                                ),
                                "n_country_years": int(len(group)),
                                "positive_mix_share": float(
                                    (values > tolerance).mean()
                                ),
                                "negative_mix_share": float(
                                    (values < -tolerance).mean()
                                ),
                                "modal_dominant_tech_id": (
                                    str(technology_counts.index[0])
                                    if len(technology_counts)
                                    else ""
                                ),
                                **summary,
                            }
                        )
    valley_records = pd.DataFrame(valley_rows)
    duration_summary = pd.DataFrame(duration_rows)
    pair_index = [
        "scan_id",
        "energy_value_eur_mwh",
        "charge_price_eur_mwh",
        "firm_value_eur_kw_year",
        "ldes_tail_fixed_cost_multiplier",
        "source_label",
        "model_mode",
        "country",
        "year",
        "country_year_id",
    ]
    valley_pairs = valley_records.pivot(
        index=pair_index,
        columns="chronology",
        values=["strict_valley", "valley_width_h"],
    ).reset_index()
    valley_pairs.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in valley_pairs.columns
    ]
    for chronology in EXPECTED_CHRONOLOGIES:
        column = f"strict_valley_{chronology}"
        if column not in valley_pairs or valley_pairs[column].isna().any():
            raise RuntimeError(
                "Incomplete dispersed/clustered chronology pair in "
                "conditional value scan"
            )
        valley_pairs[column] = valley_pairs[column].astype(bool)
    dispersed = valley_pairs["strict_valley_dispersed"]
    clustered = valley_pairs["strict_valley_clustered"]
    valley_pairs["cluster_induced_valley"] = ~dispersed & clustered
    valley_pairs["valley_retained"] = dispersed & clustered
    valley_pairs["valley_dissolved"] = dispersed & ~clustered
    valley_pairs["valley_never_formed"] = ~dispersed & ~clustered
    valley_pairs["delta_valley_width_h"] = (
        valley_pairs["valley_width_h_clustered"]
        - valley_pairs["valley_width_h_dispersed"]
    )

    rate_rows: list[dict[str, Any]] = []
    rate_keys = [
        "scan_id",
        "source_label",
        "model_mode",
        "chronology",
        "energy_value_eur_mwh",
        "charge_price_eur_mwh",
        "firm_value_eur_kw_year",
        "ldes_tail_fixed_cost_multiplier",
    ]
    for key, group in valley_records.groupby(
        rate_keys, sort=True, dropna=False
    ):
        record = dict(zip(rate_keys, key))
        present = group[group["strict_valley"]]
        record.update(
            {
                "n_country_years": int(len(group)),
                "n_countries": int(group["country"].nunique()),
                "country_years_with_strict_valley": int(
                    group["strict_valley"].sum()
                ),
                "strict_valley_rate": float(
                    group["strict_valley"].mean()
                ),
                "mean_valley_width_h_when_present": (
                    float(present["valley_width_h"].mean())
                    if len(present)
                    else np.nan
                ),
                "median_valley_width_h_when_present": (
                    float(present["valley_width_h"].median())
                    if len(present)
                    else np.nan
                ),
                "mean_left_boundary_h_when_present": (
                    float(present["left_boundary_h"].mean())
                    if len(present)
                    else np.nan
                ),
                "mean_right_boundary_h_when_present": (
                    float(present["right_boundary_h"].mean())
                    if len(present)
                    else np.nan
                ),
            }
        )
        rate_rows.append(record)
    rate_curve = pd.DataFrame(rate_rows)
    paired_rate_keys = [
        "scan_id",
        "source_label",
        "model_mode",
        "energy_value_eur_mwh",
        "charge_price_eur_mwh",
        "firm_value_eur_kw_year",
        "ldes_tail_fixed_cost_multiplier",
    ]
    paired_rate = (
        valley_pairs.groupby(paired_rate_keys, sort=True)
        .agg(
            n_country_years=("country_year_id", "size"),
            cluster_induced_rate=("cluster_induced_valley", "mean"),
            retained_rate=("valley_retained", "mean"),
            dissolved_rate=("valley_dissolved", "mean"),
            never_rate=("valley_never_formed", "mean"),
            mean_delta_valley_width_h=("delta_valley_width_h", "mean"),
            median_delta_valley_width_h=("delta_valley_width_h", "median"),
        )
        .reset_index()
    )

    threshold_keys = [
        "source_label",
        "model_mode",
        "country",
        "year",
        "country_year_id",
        "chronology",
        "energy_value_eur_mwh",
        "charge_price_eur_mwh",
        "ldes_tail_fixed_cost_multiplier",
    ]
    threshold_universe = valley_records[threshold_keys].drop_duplicates()
    thresholds = (
        valley_records[valley_records["strict_valley"]]
        .groupby(threshold_keys, sort=True)["firm_value_eur_kw_year"]
        .min()
        .rename("minimum_firm_value_for_strict_valley_eur_kw_year")
        .reset_index()
    )
    thresholds = threshold_universe.merge(
        thresholds,
        on=threshold_keys,
        how="left",
        validate="one_to_one",
    )
    thresholds["strict_valley_reached_by_scan_max"] = thresholds[
        "minimum_firm_value_for_strict_valley_eur_kw_year"
    ].notna()
    induced_keys = [
        "source_label",
        "model_mode",
        "country",
        "year",
        "country_year_id",
        "energy_value_eur_mwh",
        "charge_price_eur_mwh",
        "ldes_tail_fixed_cost_multiplier",
    ]
    induced_universe = valley_pairs[induced_keys].drop_duplicates()
    induced_threshold = (
        valley_pairs[valley_pairs["cluster_induced_valley"]]
        .groupby(induced_keys, sort=True)["firm_value_eur_kw_year"]
        .min()
        .rename(
            "minimum_firm_value_for_cluster_induced_valley_eur_kw_year"
        )
        .reset_index()
    )
    induced_threshold = induced_universe.merge(
        induced_threshold,
        on=induced_keys,
        how="left",
        validate="one_to_one",
    )
    induced_threshold[
        "cluster_induced_valley_reached_by_scan_max"
    ] = induced_threshold[
        "minimum_firm_value_for_cluster_induced_valley_eur_kw_year"
    ].notna()
    cost_threshold_keys = [
        "source_label",
        "model_mode",
        "country",
        "year",
        "country_year_id",
        "chronology",
        "energy_value_eur_mwh",
        "charge_price_eur_mwh",
        "firm_value_eur_kw_year",
    ]
    cost_threshold_universe = valley_records[
        cost_threshold_keys
    ].drop_duplicates()
    cost_thresholds = (
        valley_records[valley_records["strict_valley"]]
        .groupby(cost_threshold_keys, sort=True)[
            "ldes_tail_fixed_cost_multiplier"
        ]
        .max()
        .rename(
            "first_valley_cost_multiplier_scanning_from_1p2_downward"
        )
        .reset_index()
    )
    cost_thresholds = cost_threshold_universe.merge(
        cost_thresholds,
        on=cost_threshold_keys,
        how="left",
        validate="one_to_one",
    )
    induced_cost_keys = [
        "source_label",
        "model_mode",
        "country",
        "year",
        "country_year_id",
        "energy_value_eur_mwh",
        "charge_price_eur_mwh",
        "firm_value_eur_kw_year",
    ]
    induced_cost_universe = valley_pairs[
        induced_cost_keys
    ].drop_duplicates()
    induced_cost_thresholds = (
        valley_pairs[valley_pairs["cluster_induced_valley"]]
        .groupby(induced_cost_keys, sort=True)[
            "ldes_tail_fixed_cost_multiplier"
        ]
        .max()
        .rename(
            "first_cluster_induced_valley_cost_multiplier_"
            "scanning_from_1p2_downward"
        )
        .reset_index()
    )
    induced_cost_thresholds = induced_cost_universe.merge(
        induced_cost_thresholds,
        on=induced_cost_keys,
        how="left",
        validate="one_to_one",
    )
    return {
        "valley_records": valley_records,
        "valley_pairs": valley_pairs,
        "rate_curve": rate_curve,
        "paired_rate_curve": paired_rate,
        "minimum_thresholds": thresholds,
        "minimum_cluster_induced_thresholds": induced_threshold,
        "first_cost_thresholds": cost_thresholds,
        "first_cluster_induced_cost_thresholds": (
            induced_cost_thresholds
        ),
        "duration_summary": duration_summary,
    }


def summarize_mode(
    source_label: str,
    mode: str,
    valuation_scenario_id: int,
    cells: pd.DataFrame,
    valley_records: pd.DataFrame,
    valley_pairs: pd.DataFrame,
    focal_summary: pd.DataFrame,
) -> dict[str, Any]:
    subset_cells = cells[
        cells["source_label"].eq(source_label)
        & cells["model_mode"].eq(mode)
        & cells["valuation_scenario_id"].eq(valuation_scenario_id)
    ]
    subset_records = valley_records[
        valley_records["source_label"].eq(source_label)
        & valley_records["model_mode"].eq(mode)
        & valley_records["valuation_scenario_id"].eq(
            valuation_scenario_id
        )
    ]
    subset_pairs = valley_pairs[
        valley_pairs["source_label"].eq(source_label)
        & valley_pairs["model_mode"].eq(mode)
        & valley_pairs["valuation_scenario_id"].eq(
            valuation_scenario_id
        )
    ]
    result: dict[str, Any] = {
        "country_years": int(
            subset_cells[["country", "year"]].drop_duplicates().shape[0]
        ),
        "countries": int(subset_cells["country"].nunique()),
        "candidate_configurations": int(subset_cells["config_id"].nunique()),
        "tested_durations_4_168_h": sorted(
            float(value)
            for value in subset_cells.loc[
                subset_cells["duration_h"].between(4, 168), "duration_h"
            ].unique()
        ),
        "strict_valley_by_chronology": {},
        "envelope_shape_counts_by_chronology": {},
        "valley_transitions": (
            subset_pairs["valley_transition"].value_counts().to_dict()
        ),
        "delta_valley_width_h": quantiles(
            subset_pairs["delta_valley_width_h"]
        ),
        "focal_configurations": {},
    }
    for chronology in EXPECTED_CHRONOLOGIES:
        records = subset_records[
            subset_records["chronology"].eq(chronology)
        ]
        positive = records[records["strict_valley"]]
        result["strict_valley_by_chronology"][chronology] = {
            "count": int(records["strict_valley"].sum()),
            "denominator": int(len(records)),
            "rate": (
                float(records["strict_valley"].mean())
                if len(records)
                else None
            ),
            "width_h_when_present": quantiles(
                positive["valley_width_h"]
            ),
            "left_boundary_h_when_present": quantiles(
                positive["left_boundary_h"]
            ),
            "right_boundary_h_when_present": quantiles(
                positive["right_boundary_h"]
            ),
        }
        result["envelope_shape_counts_by_chronology"][chronology] = (
            records["envelope_shape"].value_counts().to_dict()
        )
    for config_id in FOCAL_CONFIGS:
        focal = focal_summary[
            focal_summary["source_label"].eq(source_label)
            & focal_summary["model_mode"].eq(mode)
            & focal_summary["valuation_scenario_id"].eq(
                valuation_scenario_id
            )
            & focal_summary["config_id"].eq(config_id)
        ]
        result["focal_configurations"][config_id] = focal.to_dict("records")
    return result


def validate_output(
    cells: pd.DataFrame,
    envelope: pd.DataFrame,
    records: pd.DataFrame,
    expected_country_years: int,
) -> dict[str, Any]:
    cases = (
        cells[["source_label", "model_mode"]]
        .drop_duplicates()
        .sort_values(["source_label", "model_mode"])
    )
    country_year_counts = {
        f"{row.source_label}|{row.model_mode}": int(
            cells.loc[
                cells["source_label"].eq(row.source_label)
                & cells["model_mode"].eq(row.model_mode),
                ["country", "year"],
            ]
            .drop_duplicates()
            .shape[0]
        )
        for row in cases.itertuples(index=False)
    }
    configuration_counts = {
        f"{row.source_label}|{row.model_mode}": int(
            cells.loc[
                cells["source_label"].eq(row.source_label)
                & cells["model_mode"].eq(row.model_mode),
                "config_id",
            ].nunique()
        )
        for row in cases.itertuples(index=False)
    }
    expected_envelope = 0
    expected_records = 0
    for row in cases.itertuples(index=False):
        mode_cells = cells[
            cells["source_label"].eq(row.source_label)
            & cells["model_mode"].eq(row.model_mode)
        ]
        expected_envelope += (
            mode_cells[["country", "year"]].drop_duplicates().shape[0]
            * len(EXPECTED_CHRONOLOGIES)
            * mode_cells["valuation_scenario_id"].nunique()
            * mode_cells.loc[
                mode_cells["duration_h"].between(4, 168), "duration_h"
            ].nunique()
        )
        expected_records += (
            mode_cells[["country", "year"]].drop_duplicates().shape[0]
            * len(EXPECTED_CHRONOLOGIES)
            * mode_cells["valuation_scenario_id"].nunique()
        )
    checks = {
        "all_solver_status_optimal": bool(
            cells["solver_status_name"].eq("optimal").all()
        ),
        "chronologies": sorted(cells["chronology"].unique().tolist()),
        "country_year_counts": country_year_counts,
        "configuration_counts": configuration_counts,
        "expected_envelope_rows": int(expected_envelope),
        "actual_envelope_rows": int(len(envelope)),
        "expected_valley_record_rows": int(expected_records),
        "valley_record_rows": int(len(records)),
        "maximum_value_component_identity_residual_eur": float(
            cells["value_component_identity_residual_eur"].abs().max()
        ),
        "maximum_task_balance_residual_mw": (
            float(cells["maximum_task_balance_residual_mw"].abs().max())
            if "maximum_task_balance_residual_mw" in cells
            else None
        ),
        "maximum_shared_recharge_violation_mw": (
            float(
                cells["maximum_shared_recharge_violation_mw"].abs().max()
            )
            if "maximum_shared_recharge_violation_mw" in cells
            else None
        ),
    }
    failures: list[str] = []
    if not checks["all_solver_status_optimal"]:
        failures.append("solver_status")
    if checks["chronologies"] != sorted(EXPECTED_CHRONOLOGIES):
        failures.append("chronology_coverage")
    if checks["expected_envelope_rows"] != checks["actual_envelope_rows"]:
        failures.append("envelope_row_count")
    if checks["expected_valley_record_rows"] != checks["valley_record_rows"]:
        failures.append("valley_record_row_count")
    if (
        checks["maximum_value_component_identity_residual_eur"]
        > 1.0e-3
    ):
        failures.append("value_component_identity")
    if expected_country_years > 0 and any(
        value != expected_country_years
        for value in country_year_counts.values()
    ):
        failures.append("country_year_coverage")
    return {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        type=parse_input,
        help=(
            "Repeatable LABEL=/path/to/ldes_vs_extra_lfp4_results.csv.gz. "
            "Defaults to the three prespecified v42 production cases."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--scenario-table", type=Path, default=DEFAULT_SCENARIO_TABLE
    )
    parser.add_argument(
        "--valuation-scenario-ids",
        type=parse_integer_list,
        default=[20, 31, 34],
        help="Exact published value cases used for direct EFC revaluation.",
    )
    parser.add_argument(
        "--primary-valuation-scenario-id", type=int, default=20
    )
    parser.add_argument(
        "--primary-source-label",
        type=str,
        default="fixed_s20_central",
    )
    parser.add_argument(
        "--central-valuation-scenario-id", type=int, default=20
    )
    parser.add_argument(
        "--scan-energy-values",
        type=parse_float_list,
        default=[37.0, 65.0, 98.54031839922592],
    )
    parser.add_argument(
        "--scan-charge-values",
        type=parse_float_list,
        default=[0.0, 6.16, 12.32],
    )
    parser.add_argument("--scan-firm-min", type=float, default=0.0)
    parser.add_argument("--scan-firm-max", type=float, default=600.0)
    parser.add_argument("--scan-firm-step", type=float, default=25.0)
    parser.add_argument(
        "--scan-cost-multiplier-min", type=float, default=0.4
    )
    parser.add_argument(
        "--scan-cost-multiplier-max", type=float, default=1.2
    )
    parser.add_argument(
        "--scan-cost-multiplier-step", type=float, default=0.05
    )
    parser.add_argument("--skip-value-scan", action="store_true")
    parser.add_argument(
        "--bootstrap-draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--zero-tolerance-eur-kw-year",
        type=float,
        default=DEFAULT_ZERO_TOLERANCE,
    )
    parser.add_argument(
        "--expected-country-years",
        type=int,
        default=0,
        help="Set to 132 for the manuscript production run; 0 disables.",
    )
    parser.add_argument("--wait-for-inputs", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-seconds", type=float, default=86400.0)
    args = parser.parse_args()
    if args.bootstrap_draws <= 0:
        raise ValueError("--bootstrap-draws must be positive")
    if args.zero_tolerance_eur_kw_year < 0:
        raise ValueError("--zero-tolerance-eur-kw-year cannot be negative")
    if args.scan_firm_step <= 0 or args.scan_firm_max < args.scan_firm_min:
        raise ValueError("Invalid firm-value scan bounds")
    if (
        args.scan_cost_multiplier_step <= 0
        or args.scan_cost_multiplier_max < args.scan_cost_multiplier_min
    ):
        raise ValueError("Invalid cost-multiplier scan bounds")
    inputs = (
        args.input
        if args.input
        else [(label, path) for label, path in DEFAULT_INPUTS.items()]
    )
    arm_inputs = infer_arm_inputs(inputs)
    paths = [path for _, path in inputs] + [
        path for _, path in arm_inputs
    ]
    if args.wait_for_inputs:
        wait_for_inputs(paths, args.poll_seconds, args.timeout_seconds)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing v42 portfolio result files: {missing}"
        )

    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = load_valuation_scenarios(
        args.scenario_table, args.valuation_scenario_ids
    )
    scenario_by_id = {
        int(scenario["scenario_id"]): scenario for scenario in scenarios
    }
    if args.primary_valuation_scenario_id not in scenario_by_id:
        raise RuntimeError(
            "Primary valuation scenario is not in --valuation-scenario-ids"
        )
    if args.central_valuation_scenario_id not in scenario_by_id:
        raise RuntimeError(
            "Central valuation scenario is not in --valuation-scenario-ids"
        )
    raw_cells = load_cells(inputs)
    if args.primary_source_label not in set(raw_cells["source_label"]):
        raise RuntimeError(
            f"Primary source label {args.primary_source_label!r} is absent"
        )
    arms = load_arm_results(arm_inputs)
    discrete_cells = pd.concat(
        [
            add_derived_metrics(raw_cells, arms, scenario)
            for scenario in scenarios
        ],
        ignore_index=True,
    )
    cells = discrete_cells[
        discrete_cells["source_label"].eq(args.primary_source_label)
        & discrete_cells["valuation_scenario_id"].eq(
            args.primary_valuation_scenario_id
        )
    ].copy()
    envelope = choose_duration_envelope(
        cells,
        VALLEY_MIN_H,
        VALLEY_MAX_H,
        args.zero_tolerance_eur_kw_year,
    )
    valley_records = build_valley_records(
        envelope, args.zero_tolerance_eur_kw_year
    )
    config_pairs = build_config_pairs(cells)
    envelope_pairs = build_envelope_pairs(envelope)
    valley_pairs = build_valley_pairs(valley_records)
    primary_valley_shape_summary = summarize_valley_shapes(valley_records)
    bootstrap = build_bootstrap_summary(
        config_pairs,
        envelope_pairs,
        valley_pairs,
        args.bootstrap_draws,
        args.seed,
    )
    focal_summary = build_focal_summary(cells, config_pairs, bootstrap)
    duration_summary = summarize_duration(envelope)
    configuration_summary = summarize_configurations(cells)
    layer_metrics = select_layer_metrics(envelope)
    value_components = select_value_components(envelope)
    discrete_envelope = choose_duration_envelope(
        discrete_cells,
        VALLEY_MIN_H,
        VALLEY_MAX_H,
        args.zero_tolerance_eur_kw_year,
    )
    discrete_valley_records = build_valley_records(
        discrete_envelope, args.zero_tolerance_eur_kw_year
    )
    discrete_valley_pairs = build_valley_pairs(discrete_valley_records)
    discrete_config_pairs = build_config_pairs(discrete_cells)
    discrete_envelope_pairs = build_envelope_pairs(discrete_envelope)
    discrete_bootstrap = build_bootstrap_summary(
        discrete_config_pairs,
        discrete_envelope_pairs,
        discrete_valley_pairs,
        args.bootstrap_draws,
        args.seed + 1,
    )
    discrete_focal_summary = build_focal_summary(
        discrete_cells,
        discrete_config_pairs,
        discrete_bootstrap,
    )
    discrete_valley_shape_summary = summarize_valley_shapes(
        discrete_valley_records
    )
    central_valley_shape_summary = discrete_valley_shape_summary[
        discrete_valley_shape_summary["source_label"].eq(
            args.primary_source_label
        )
        & discrete_valley_shape_summary["valuation_scenario_id"].eq(
            args.central_valuation_scenario_id
        )
    ].copy()
    discrete_duration_summary = summarize_duration(discrete_envelope)
    discrete_configuration_summary = summarize_configurations(
        discrete_cells
    )
    value_scan: dict[str, pd.DataFrame] | None = None
    cost_firm_scan: dict[str, pd.DataFrame] | None = None
    firm_values = np.round(
        np.arange(
            args.scan_firm_min,
            args.scan_firm_max + args.scan_firm_step * 0.5,
            args.scan_firm_step,
        ),
        10,
    ).tolist()
    cost_multipliers = np.round(
        np.arange(
            args.scan_cost_multiplier_min,
            args.scan_cost_multiplier_max
            + args.scan_cost_multiplier_step * 0.5,
            args.scan_cost_multiplier_step,
        ),
        10,
    ).tolist()
    primary_scenario = scenario_by_id[
        args.primary_valuation_scenario_id
    ]
    # One valuation copy per physical production source is sufficient for the
    # deterministic value scans because service deltas and cost components do
    # not change across the exact post-hoc valuation copies. Keeping all
    # source labels here puts the central fixed case and the two scenario-31
    # power-mode robustness cases on exactly the same value grid.
    scan_cells = discrete_cells[
        discrete_cells["valuation_scenario_id"].eq(
            args.primary_valuation_scenario_id
        )
    ].copy()
    if not args.skip_value_scan:
        value_scan = build_value_scan(
            scan_cells,
            args.scan_energy_values,
            args.scan_charge_values,
            firm_values,
            [1.0],
            args.zero_tolerance_eur_kw_year,
        )
        cost_firm_scan = build_value_scan(
            scan_cells,
            [float(primary_scenario["energy_value_eur_mwh"])],
            [float(primary_scenario["charge_price_eur_mwh"])],
            firm_values,
            cost_multipliers,
            args.zero_tolerance_eur_kw_year,
        )

    outputs = {
        "cells": args.output_dir / "v42_fixed_scale_mix_cells.csv.gz",
        "envelope": (
            args.output_dir / "v42_fixed_scale_mix_envelope.csv"
        ),
        "valley_records": (
            args.output_dir / "v42_fixed_scale_mix_valley_records.csv"
        ),
        "valley_pairs": (
            args.output_dir / "v42_fixed_scale_mix_valley_pairs.csv"
        ),
        "config_pairs": (
            args.output_dir
            / "v42_fixed_scale_mix_config_paired_effects.csv.gz"
        ),
        "envelope_pairs": (
            args.output_dir
            / "v42_fixed_scale_mix_envelope_paired_effects.csv"
        ),
        "duration_summary": (
            args.output_dir / "v42_fixed_scale_mix_duration_summary.csv"
        ),
        "configuration_summary": (
            args.output_dir
            / "v42_fixed_scale_mix_configuration_summary.csv"
        ),
        "layer_metrics": (
            args.output_dir / "v42_fixed_scale_mix_layer_metrics.csv"
        ),
        "value_components": (
            args.output_dir / "v42_fixed_scale_mix_value_components.csv"
        ),
        "focal_summary": (
            args.output_dir
            / "v42_fixed_scale_mix_focal_config_summary.csv"
        ),
        "bootstrap_summary": (
            args.output_dir
            / "v42_fixed_scale_mix_bootstrap_summary.csv"
        ),
        "discrete_scenario_duration_summary": (
            args.output_dir
            / "v42_fixed_scale_mix_efc_discrete_scenario_"
            "duration_summary.csv"
        ),
        "discrete_scenario_cells": (
            args.output_dir
            / "v42_all_cases_mix_efc_discrete_scenario_cells.csv.gz"
        ),
        "discrete_scenario_config_pairs": (
            args.output_dir
            / "v42_all_cases_mix_efc_discrete_scenario_"
            "config_paired_effects.csv.gz"
        ),
        "discrete_scenario_bootstrap_summary": (
            args.output_dir
            / "v42_all_cases_mix_efc_discrete_scenario_"
            "bootstrap_summary.csv"
        ),
        "discrete_scenario_focal_summary": (
            args.output_dir
            / "v42_all_cases_mix_efc_discrete_scenario_"
            "focal_config_summary.csv"
        ),
        "discrete_scenario_envelope": (
            args.output_dir
            / "v42_fixed_scale_mix_efc_discrete_scenario_"
            "envelope.csv.gz"
        ),
        "discrete_scenario_envelope_pairs": (
            args.output_dir
            / "v42_all_cases_mix_efc_discrete_scenario_"
            "envelope_paired_effects.csv.gz"
        ),
        "discrete_scenario_configuration_summary": (
            args.output_dir
            / "v42_fixed_scale_mix_efc_discrete_scenario_"
            "configuration_summary.csv"
        ),
        "discrete_scenario_valley_records": (
            args.output_dir
            / "v42_fixed_scale_mix_efc_discrete_scenario_"
            "valley_records.csv"
        ),
        "discrete_scenario_valley_pairs": (
            args.output_dir
            / "v42_fixed_scale_mix_efc_discrete_scenario_"
            "valley_pairs.csv"
        ),
        "primary_valley_shape_summary": (
            args.output_dir
            / "v42_fixed_scale_mix_efc_primary_scenario_"
            "valley_shape_summary.csv"
        ),
        "discrete_scenario_valley_shape_summary": (
            args.output_dir
            / "v42_fixed_scale_mix_efc_discrete_scenario_"
            "valley_shape_summary.csv"
        ),
        "central_scenario_valley_shape_summary": (
            args.output_dir
            / "v42_fixed_scale_mix_efc_central_scenario_"
            "valley_shape_summary.csv"
        ),
    }
    write_csv_gz(cells, outputs["cells"])
    write_csv(envelope, outputs["envelope"])
    write_csv(valley_records, outputs["valley_records"])
    write_csv(valley_pairs, outputs["valley_pairs"])
    write_csv_gz(config_pairs, outputs["config_pairs"])
    write_csv(envelope_pairs, outputs["envelope_pairs"])
    write_csv(duration_summary, outputs["duration_summary"])
    write_csv(configuration_summary, outputs["configuration_summary"])
    write_csv(layer_metrics, outputs["layer_metrics"])
    write_csv(value_components, outputs["value_components"])
    write_csv(focal_summary, outputs["focal_summary"])
    write_csv(bootstrap, outputs["bootstrap_summary"])
    write_csv(
        discrete_duration_summary,
        outputs["discrete_scenario_duration_summary"],
    )
    write_csv_gz(
        discrete_cells,
        outputs["discrete_scenario_cells"],
    )
    write_csv_gz(
        discrete_config_pairs,
        outputs["discrete_scenario_config_pairs"],
    )
    write_csv(
        discrete_bootstrap,
        outputs["discrete_scenario_bootstrap_summary"],
    )
    write_csv(
        discrete_focal_summary,
        outputs["discrete_scenario_focal_summary"],
    )
    write_csv_gz(
        discrete_envelope,
        outputs["discrete_scenario_envelope"],
    )
    write_csv_gz(
        discrete_envelope_pairs,
        outputs["discrete_scenario_envelope_pairs"],
    )
    write_csv(
        discrete_configuration_summary,
        outputs["discrete_scenario_configuration_summary"],
    )
    write_csv(
        discrete_valley_records,
        outputs["discrete_scenario_valley_records"],
    )
    write_csv(
        discrete_valley_pairs,
        outputs["discrete_scenario_valley_pairs"],
    )
    write_csv(
        primary_valley_shape_summary,
        outputs["primary_valley_shape_summary"],
    )
    write_csv(
        discrete_valley_shape_summary,
        outputs["discrete_scenario_valley_shape_summary"],
    )
    write_csv(
        central_valley_shape_summary,
        outputs["central_scenario_valley_shape_summary"],
    )
    if value_scan is not None and cost_firm_scan is not None:
        scan_outputs = {
            "value_scan_valley_records": (
                args.output_dir
                / "v42_fixed_scale_mix_efc_value_scan_"
                "valley_records.csv.gz"
            ),
            "value_scan_rate_curve": (
                args.output_dir
                / "v42_fixed_scale_mix_efc_value_scan_"
                "formation_rate_curve.csv"
            ),
            "value_scan_paired_rate_curve": (
                args.output_dir
                / "v42_fixed_scale_mix_efc_value_scan_"
                "paired_formation_rate_curve.csv"
            ),
            "value_scan_minimum_firm_thresholds": (
                args.output_dir
                / "v42_fixed_scale_mix_efc_value_scan_"
                "minimum_firm_thresholds.csv"
            ),
            "value_scan_minimum_cluster_induced_thresholds": (
                args.output_dir
                / "v42_fixed_scale_mix_efc_value_scan_"
                "minimum_cluster_induced_firm_thresholds.csv"
            ),
            "value_scan_duration_summary": (
                args.output_dir
                / "v42_fixed_scale_mix_efc_value_scan_"
                "duration_summary.csv.gz"
            ),
            "cost_firm_valley_records": (
                args.output_dir
                / "v42_fixed_scale_mix_efc_cost_firm_phase_"
                "valley_records.csv.gz"
            ),
            "cost_firm_rate_curve": (
                args.output_dir
                / "v42_fixed_scale_mix_efc_cost_firm_phase_"
                "formation_rate.csv"
            ),
            "cost_firm_paired_rate_curve": (
                args.output_dir
                / "v42_fixed_scale_mix_efc_cost_firm_phase_"
                "paired_formation_rate.csv"
            ),
            "cost_firm_first_cost_thresholds": (
                args.output_dir
                / "v42_fixed_scale_mix_efc_cost_firm_phase_"
                "first_cost_thresholds.csv"
            ),
            "cost_firm_first_cluster_induced_cost_thresholds": (
                args.output_dir
                / "v42_fixed_scale_mix_efc_cost_firm_phase_"
                "first_cluster_induced_cost_thresholds.csv"
            ),
            "cost_firm_duration_summary": (
                args.output_dir
                / "v42_fixed_scale_mix_efc_cost_firm_phase_"
                "duration_summary.csv.gz"
            ),
        }
        write_csv_gz(
            value_scan["valley_records"],
            scan_outputs["value_scan_valley_records"],
        )
        write_csv(
            value_scan["rate_curve"],
            scan_outputs["value_scan_rate_curve"],
        )
        write_csv(
            value_scan["paired_rate_curve"],
            scan_outputs["value_scan_paired_rate_curve"],
        )
        write_csv(
            value_scan["minimum_thresholds"],
            scan_outputs["value_scan_minimum_firm_thresholds"],
        )
        write_csv(
            value_scan["minimum_cluster_induced_thresholds"],
            scan_outputs[
                "value_scan_minimum_cluster_induced_thresholds"
            ],
        )
        write_csv_gz(
            value_scan["duration_summary"],
            scan_outputs["value_scan_duration_summary"],
        )
        write_csv_gz(
            cost_firm_scan["valley_records"],
            scan_outputs["cost_firm_valley_records"],
        )
        write_csv(
            cost_firm_scan["rate_curve"],
            scan_outputs["cost_firm_rate_curve"],
        )
        write_csv(
            cost_firm_scan["paired_rate_curve"],
            scan_outputs["cost_firm_paired_rate_curve"],
        )
        write_csv(
            cost_firm_scan["first_cost_thresholds"],
            scan_outputs["cost_firm_first_cost_thresholds"],
        )
        write_csv(
            cost_firm_scan["first_cluster_induced_cost_thresholds"],
            scan_outputs[
                "cost_firm_first_cluster_induced_cost_thresholds"
            ],
        )
        write_csv_gz(
            cost_firm_scan["duration_summary"],
            scan_outputs["cost_firm_duration_summary"],
        )
        outputs.update(scan_outputs)

    primary_qa = validate_output(
        cells, envelope, valley_records, args.expected_country_years
    )
    qa = validate_output(
        discrete_cells,
        discrete_envelope,
        discrete_valley_records,
        args.expected_country_years,
    )
    qa["primary_case"] = primary_qa
    if primary_qa["status"] != "pass":
        qa["status"] = "fail"
        qa["failures"] = sorted(
            set(qa["failures"] + ["primary_case_" + item for item in primary_qa["failures"]])
        )
    summary = {
        "analysis_id": "v42_europe_joint_lfp_ldes_portfolio",
        "created_utc": utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "evidence_status": (
            "full_132_country_year_manuscript_evidence"
            if all(
                value == 132
                for value in qa["checks"]["country_year_counts"].values()
            )
            else "smoke_or_partial_not_manuscript_evidence"
        ),
        "comparison": (
            "common 4 h LFP + candidate LDES tail versus common 4 h "
            "LFP + additional 4 h LFP tail"
        ),
        "normalization": {
            "primary": (
                "EFC-valued candidate portfolio net value minus directly "
                "solved extra-LFP portfolio net value, divided by "
                "base_power_mw * 1000; unit EUR kW^-1 yr^-1"
            ),
            "secondary": (
                "mix_net_benefit_vs_extra_lfp4_eur divided by "
                "base_task_mwh; unit EUR MWh^-1 annual task"
            ),
        },
        "strict_valley_definition": {
            "domain_h": [VALLEY_MIN_H, VALLEY_MAX_H],
            "rule": (
                "at least two adjacent tested nodes strictly below the "
                "negative tolerance, immediately bounded by nodes strictly "
                "above the positive tolerance; the widest bounded interval "
                "is reported"
            ),
            "width_h": "right positive boundary minus left positive boundary",
            "zero_tolerance_eur_per_kw_common_power_year": (
                args.zero_tolerance_eur_kw_year
            ),
        },
        "bootstrap": {
            "independent_block": "country",
            "within_block_observations": "all available years",
            "country_aggregation": (
                "country mean for primary paired effects; country median "
                "also reported for valley-width change"
            ),
            "draws": args.bootstrap_draws,
            "seed": args.seed,
            "interval": "percentile 95%",
        },
        "valuation": {
            "primary_source_label": args.primary_source_label,
            "primary_scenario_id": args.primary_valuation_scenario_id,
            "central_scenario_id": args.central_valuation_scenario_id,
            "scenario_roles": {
                "20": "central value case",
                "31": "high-energy, zero-charge-price conditional case",
                "34": "higher task-firm-value conditional case",
            },
            "exact_scenario_ids": args.valuation_scenario_ids,
            "efc_definition": (
                "energy_value * total_discharge + firm_value * 1000 * "
                "EENS-equivalent portfolio EFC - fixed cost - VOM - "
                "charging opportunity cost"
            ),
            "residual_peak_system_cost_role": (
                "retained only as an explicitly named sensitivity; it does "
                "not determine the main mix envelope or valley"
            ),
            "value_scan": {
                "physical_source_labels": sorted(
                    scan_cells["source_label"].unique().tolist()
                ),
                "common_grid_role": (
                    "identical post-hoc EFC value grid applied separately "
                    "to each physical production source"
                ),
                "energy_value_eur_mwh": args.scan_energy_values,
                "charge_price_eur_mwh": args.scan_charge_values,
                "firm_value_eur_kw_year": firm_values,
                "ldes_tail_fixed_cost_multiplier_phase_scan": (
                    cost_multipliers
                ),
                "cost_multiplier_interpretation": (
                    "deterministic reversal distance applied only to the "
                    "LDES tail fixed cost; it is not a probability"
                ),
            },
        },
        "inputs": [
            {
                "label": label,
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for label, path in inputs + arm_inputs
        ],
        "scenario_table": {
            "path": str(args.scenario_table.resolve()),
            "sha256": sha256(args.scenario_table),
        },
        "outputs": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for name, path in outputs.items()
        },
        "qa": qa,
        "results_by_source_mode_and_valuation_scenario": {
            f"{source_label}|{mode}|scenario_{scenario_id}": summarize_mode(
                source_label,
                mode,
                int(scenario_id),
                discrete_cells,
                discrete_valley_records,
                discrete_valley_pairs,
                focal_summary,
            )
            for source_label, mode in (
                discrete_cells[["source_label", "model_mode"]]
                .drop_duplicates()
                .itertuples(index=False, name=None)
            )
            for scenario_id in args.valuation_scenario_ids
        },
        "provenance_note": (
            "All numerical results were recomputed from the listed v42 "
            "joint-portfolio files; no v41 numerical result was imported."
        ),
    }
    summary_path = (
        args.output_dir / "v42_fixed_scale_mix_summary.json"
    )
    write_json(summary_path, summary)
    if qa["status"] != "pass":
        raise RuntimeError(
            "Post-processing QA failed: " + ", ".join(qa["failures"])
        )
    print(
        json.dumps(
            {
                "status": "pass",
                "summary": str(summary_path),
                "country_year_counts": qa["checks"][
                    "country_year_counts"
                ],
                "elapsed_seconds": summary["elapsed_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
