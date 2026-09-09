#!/usr/bin/env python3
"""Independent audit of the v41 full perfect-foresight experiment.

The audit is deliberately separate from the experiment runner. It does not
alter any model output. It verifies file integrity, the frozen experiment
contract, LP status and dimensions, chronology conservation, economic
identities, the matched-greedy comparison, the duration-envelope construction,
and the strict-valley classification. A compact JSON report is written beside
the v41 figure scripts for manuscript traceability.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "perfect_foresight_valley_full_132cy"
REPORT = HERE / "perfect_foresight_valley_independent_audit_v41.json"

KEYS = ["country", "year", "chronology", "config_id"]
EXPECTED_COUNTRIES = 15
EXPECTED_COUNTRY_YEARS = 132
EXPECTED_CONFIGURATIONS = 53
EXPECTED_CHRONOLOGIES = {"dispersed", "clustered"}
EXPECTED_ROWS = (
    EXPECTED_COUNTRY_YEARS
    * EXPECTED_CONFIGURATIONS
    * len(EXPECTED_CHRONOLOGIES)
)
TOL_CONSERVATION_MWH = 1e-6
TOL_SOC_MWH = 1e-8
TOL_SERVICE_SHARE = 1e-12
TOL_ECONOMIC_IDENTITY = 1e-8
TOL_ENVELOPE = 1e-8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def strict_valley_from_envelope(group: pd.DataFrame) -> dict[str, object]:
    """Reproduce the prespecified widest bracketed non-positive interval."""

    ordered = (
        group.sort_values("duration_h")
        .drop_duplicates("duration_h", keep="first")
        .reset_index(drop=True)
    )
    durations = ordered["duration_h"].to_numpy(float)
    values = ordered[
        "scenario31_net_benefit_eur_per_kw_year"
    ].to_numpy(float)
    nonpositive = values <= 0.0
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    for index, flag in enumerate(nonpositive):
        if flag and start is None:
            start = index
        if start is not None and (not flag or index == len(nonpositive) - 1):
            end = index - 1 if not flag else index
            left = start - 1
            right = end + 1
            if left >= 0 and right < len(values):
                if values[left] > 0.0 and values[right] > 0.0:
                    intervals.append((start, end))
            start = None
    if not intervals:
        return {
            "has_strict_valley": False,
            "candidate_nonpositive_intervals": 0,
            "gap_width_h": 0.0,
            "left_positive_duration_h": np.nan,
            "nonpositive_start_duration_h": np.nan,
            "nonpositive_end_duration_h": np.nan,
            "right_reentry_duration_h": np.nan,
            "left_positive_net_eur_per_kw_year": np.nan,
            "minimum_gap_net_eur_per_kw_year": np.nan,
            "right_positive_net_eur_per_kw_year": np.nan,
            "nonpositive_duration_nodes": 0,
        }
    start, end = max(
        intervals,
        key=lambda interval: (
            durations[interval[1] + 1] - durations[interval[0] - 1]
        ),
    )
    left = start - 1
    right = end + 1
    return {
        "has_strict_valley": True,
        "candidate_nonpositive_intervals": len(intervals),
        "gap_width_h": float(durations[right] - durations[left]),
        "left_positive_duration_h": float(durations[left]),
        "nonpositive_start_duration_h": float(durations[start]),
        "nonpositive_end_duration_h": float(durations[end]),
        "right_reentry_duration_h": float(durations[right]),
        "left_positive_net_eur_per_kw_year": float(values[left]),
        "minimum_gap_net_eur_per_kw_year": float(
            np.min(values[start : end + 1])
        ),
        "right_positive_net_eur_per_kw_year": float(values[right]),
        "nonpositive_duration_nodes": int(end - start + 1),
    }


def numeric_equal(
    left: pd.Series,
    right: pd.Series,
    tolerance: float,
) -> bool:
    return bool(
        np.allclose(
            left.to_numpy(float),
            right.to_numpy(float),
            rtol=0.0,
            atol=tolerance,
            equal_nan=True,
        )
    )


def main() -> None:
    manifest = json.loads(
        (RESULTS / "run_manifest.json").read_text(encoding="utf-8")
    )
    contract = json.loads(
        (RESULTS / "run_contract.json").read_text(encoding="utf-8")
    )

    output_hashes_match = True
    output_sizes_match = True
    for filename, metadata in manifest["output_files"].items():
        path = RESULTS / filename
        require(path.exists(), f"Missing recorded output: {filename}")
        output_hashes_match &= sha256(path) == metadata["sha256"]
        output_sizes_match &= path.stat().st_size == metadata["bytes"]
    input_hashes_match = all(
        Path(filename).exists() and sha256(Path(filename)) == digest
        for filename, digest in manifest["inputs"].items()
    )

    perfect = pd.read_csv(
        RESULTS / "configuration_results_perfect_foresight.csv.gz"
    )
    matched = pd.read_csv(
        RESULTS / "configuration_results_matched_greedy.csv.gz"
    )
    comparison = pd.read_csv(
        RESULTS / "heuristic_vs_perfect_foresight.csv.gz"
    )
    conservation = pd.read_csv(RESULTS / "chronology_conservation.csv")
    envelope = pd.read_csv(
        RESULTS / "scenario31_duration_envelope.csv.gz"
    )
    records = pd.read_csv(RESULTS / "strict_valley_records.csv")

    for frame, name in [(perfect, "perfect"), (matched, "matched")]:
        require(len(frame) == EXPECTED_ROWS, f"{name} row count")
        require(
            not frame.duplicated(KEYS).any(),
            f"{name} has duplicate experiment keys",
        )
        require(
            set(frame["chronology"]) == EXPECTED_CHRONOLOGIES,
            f"{name} chronology coverage",
        )
        require(
            frame["config_id"].nunique() == EXPECTED_CONFIGURATIONS,
            f"{name} configuration coverage",
        )
        require(
            frame[["country", "year"]].drop_duplicates().shape[0]
            == EXPECTED_COUNTRY_YEARS,
            f"{name} country-year coverage",
        )
        require(
            frame["country"].nunique() == EXPECTED_COUNTRIES,
            f"{name} country coverage",
        )

    require(len(comparison) == EXPECTED_ROWS, "comparison row count")
    require(not comparison.duplicated(KEYS).any(), "comparison duplicate keys")
    require(
        perfect["solver_status_name"].eq("optimal").all(),
        "not all perfect-foresight models are optimal",
    )
    require(
        perfect["solver_binary_variables"].eq(0).all()
        and perfect["solver_integer_variables"].eq(0).all(),
        "perfect-foresight model is not a continuous LP",
    )
    require(
        perfect["solver_variables"].eq(26280).all()
        and perfect["solver_constraints"].eq(8760).all(),
        "unexpected LP dimensions",
    )

    max_task_error = float(
        conservation["task_original_conservation_error_mwh"].abs().max()
    )
    max_recharge_error = float(
        conservation[
            "recharge_original_conservation_error_mwh"
        ].abs().max()
    )
    max_soc_balance = float(
        perfect["maximum_soc_balance_residual_mwh"].abs().max()
    )
    max_cyclic_soc = float(
        perfect["cyclic_soc_residual_mwh"].abs().max()
    )
    require(
        max_task_error <= TOL_CONSERVATION_MWH
        and max_recharge_error <= TOL_CONSERVATION_MWH,
        "chronology reordering did not conserve annual inputs",
    )
    require(
        max_soc_balance <= TOL_SOC_MWH
        and max_cyclic_soc <= TOL_SOC_MWH,
        "LP SOC residual exceeds tolerance",
    )
    # This field counts hours in which the *raw potential* task and recharge
    # series are both positive. It is not simultaneous charging/discharging:
    # the frozen LP contract sets the charge upper bound to zero in every
    # positive-task hour. The count must therefore be an input property and
    # remain identical across all configurations for a chronology.
    overlap_nunique = perfect.groupby(
        ["country", "year", "chronology"]
    )["task_recharge_overlap_hours"].nunique()
    require(
        overlap_nunique.eq(1).all(),
        "potential task/recharge overlap changed across configurations",
    )

    reconstructed_net = (
        perfect["energy_value_component_eur_per_kw_year"]
        + perfect["firm_value_component_eur_per_kw_year"]
        - perfect["annual_fixed_cost_eur_per_kw_year_source_aware"]
        - perfect["storage_vom_component_eur_per_kw_year"]
        - perfect["charge_cost_component_eur_per_kw_year"]
    )
    max_economic_residual = float(
        (
            reconstructed_net
            - perfect["scenario31_net_benefit_eur_per_kw_year"]
        )
        .abs()
        .max()
    )
    require(
        max_economic_residual <= TOL_ECONOMIC_IDENTITY,
        "net-benefit accounting identity failed",
    )

    max_service_loss_share = float(
        (
            comparison["matched_served_share"]
            - comparison["perfect_served_share"]
        ).max()
    )
    max_abs_discharge_difference = float(
        comparison["perfect_minus_matched_discharge_mwh"].abs().max()
    )
    max_abs_net_difference = float(
        comparison[
            "perfect_minus_matched_scenario31_net_benefit_eur_per_kw_year"
        ]
        .abs()
        .max()
    )
    require(
        max_service_loss_share <= TOL_SERVICE_SHARE,
        "perfect foresight supplied less task energy than matched greedy",
    )
    require(
        max_abs_discharge_difference <= TOL_SOC_MWH,
        "perfect foresight changed task supply beyond numerical tolerance",
    )
    total_charge_avoided_mwh = float(
        comparison["matched_charge_mwh"].sum()
        - comparison["perfect_charge_mwh"].sum()
    )
    require(
        total_charge_avoided_mwh >= -TOL_SOC_MWH,
        "perfect foresight used more aggregate charging",
    )

    # Independently reconstruct the best-technology envelope from all
    # configuration outputs for the two newly solved dispatch modes.
    reconstructed_parts: list[pd.DataFrame] = []
    for mode, frame in [
        ("matched_greedy", matched),
        ("perfect_foresight", perfect),
    ]:
        limited = frame.loc[frame["duration_h"].le(168.0)].copy()
        group_columns = ["country", "year", "chronology", "duration_h"]
        best_indices = limited.groupby(group_columns)[
            "scenario31_net_benefit_eur_per_kw_year"
        ].idxmax()
        best = limited.loc[
            best_indices,
            group_columns
            + ["config_id", "scenario31_net_benefit_eur_per_kw_year"],
        ].copy()
        best["dispatch_mode"] = mode
        reconstructed_parts.append(best)
    reconstructed_envelope = pd.concat(
        reconstructed_parts,
        ignore_index=True,
    )
    recorded_envelope = envelope.loc[
        envelope["dispatch_mode"].isin(
            ["matched_greedy", "perfect_foresight"]
        ),
        [
            "country",
            "year",
            "chronology",
            "duration_h",
            "config_id",
            "scenario31_net_benefit_eur_per_kw_year",
            "dispatch_mode",
        ],
    ].copy()
    envelope_keys = [
        "country",
        "year",
        "chronology",
        "duration_h",
        "dispatch_mode",
    ]
    envelope_check = reconstructed_envelope.merge(
        recorded_envelope,
        on=envelope_keys,
        suffixes=("_audit", "_recorded"),
        validate="one_to_one",
    )
    require(
        len(envelope_check) == len(recorded_envelope),
        "duration envelope row coverage",
    )
    require(
        envelope_check["config_id_audit"].eq(
            envelope_check["config_id_recorded"]
        ).all(),
        "duration envelope did not select the net-value maximum",
    )
    max_envelope_residual = float(
        (
            envelope_check[
                "scenario31_net_benefit_eur_per_kw_year_audit"
            ]
            - envelope_check[
                "scenario31_net_benefit_eur_per_kw_year_recorded"
            ]
        )
        .abs()
        .max()
    )
    require(
        max_envelope_residual <= TOL_ENVELOPE,
        "recorded duration-envelope values differ from reconstruction",
    )

    # Independently classify every strict valley from the recorded 168 h
    # envelope and compare all structural fields with the runner's records.
    audit_records: list[dict[str, object]] = []
    for keys, group in envelope.groupby(
        ["dispatch_mode", "country", "year", "chronology"],
        sort=False,
    ):
        classified = strict_valley_from_envelope(group)
        audit_records.append(
            dict(
                zip(
                    ["dispatch_mode", "country", "year", "chronology"],
                    keys,
                ),
                **classified,
            )
        )
    audited = pd.DataFrame(audit_records)
    record_keys = ["dispatch_mode", "country", "year", "chronology"]
    compared = audited.merge(
        records,
        on=record_keys,
        suffixes=("_audit", "_recorded"),
        validate="one_to_one",
    )
    require(len(compared) == len(records), "strict-valley record coverage")
    structural_columns = [
        "candidate_nonpositive_intervals",
        "gap_width_h",
        "left_positive_duration_h",
        "nonpositive_start_duration_h",
        "nonpositive_end_duration_h",
        "right_reentry_duration_h",
        "nonpositive_duration_nodes",
    ]
    require(
        compared["has_strict_valley_audit"].eq(
            compared["has_strict_valley_recorded"]
        ).all(),
        "strict-valley Boolean classification mismatch",
    )
    for column in structural_columns:
        require(
            numeric_equal(
                compared[f"{column}_audit"],
                compared[f"{column}_recorded"],
                TOL_ENVELOPE,
            ),
            f"strict-valley field mismatch: {column}",
        )

    recomputed_summary = (
        audited.groupby(["dispatch_mode", "chronology"])[
            ["has_strict_valley", "gap_width_h"]
        ]
        .agg(
            strict_valleys=("has_strict_valley", "sum"),
            mean_width_h=("gap_width_h", "mean"),
            median_width_h=("gap_width_h", "median"),
        )
        .reset_index()
    )
    perfect_clustered = audited.loc[
        audited["dispatch_mode"].eq("perfect_foresight")
        & audited["chronology"].eq("clustered")
    ]
    perfect_dispersed = audited.loc[
        audited["dispatch_mode"].eq("perfect_foresight")
        & audited["chronology"].eq("dispersed")
    ]
    require(
        perfect_dispersed["has_strict_valley"].sum() == 0
        and perfect_clustered["has_strict_valley"].sum()
        == EXPECTED_COUNTRY_YEARS,
        "perfect-foresight paired valley result",
    )

    minimum_abs_envelope_value = float(
        envelope["scenario31_net_benefit_eur_per_kw_year"].abs().min()
    )
    report = {
        "audit_status": "pass",
        "independent_audit": True,
        "raw_results_modified": False,
        "file_integrity": {
            "recorded_output_hashes_match": bool(output_hashes_match),
            "recorded_output_sizes_match": bool(output_sizes_match),
            "recorded_input_hashes_match": bool(input_hashes_match),
        },
        "scope": {
            "countries": int(perfect["country"].nunique()),
            "country_years": int(
                perfect[["country", "year"]].drop_duplicates().shape[0]
            ),
            "configurations": int(perfect["config_id"].nunique()),
            "chronologies": sorted(perfect["chronology"].unique().tolist()),
            "perfect_foresight_lp_models": int(len(perfect)),
            "matched_greedy_cells": int(len(matched)),
            "lp_variables_per_model": int(
                perfect["solver_variables"].iloc[0]
            ),
            "lp_constraints_per_model": int(
                perfect["solver_constraints"].iloc[0]
            ),
            "integer_variables_per_model": 0,
            "binary_variables_per_model": 0,
            "scenario_id": int(perfect["scenario_id"].iloc[0]),
            "envelope_max_duration_h": float(
                envelope["duration_h"].max()
            ),
        },
        "frozen_value_contract": {
            "energy_value_eur_mwh": float(
                perfect["energy_value_eur_mwh"].iloc[0]
            ),
            "firm_value_eur_kw_year": float(
                perfect["firm_value_eur_kw_year"].iloc[0]
            ),
            "charge_price_eur_mwh": float(
                perfect["charge_price_eur_mwh"].iloc[0]
            ),
            "soc_min": float(perfect["soc_min_fraction"].min()),
            "soc_max": float(perfect["soc_max_fraction"].max()),
        },
        "numerical_checks": {
            "all_solver_status_optimal": True,
            "all_models_continuous_lp": True,
            "max_task_conservation_error_mwh": max_task_error,
            "max_recharge_conservation_error_mwh": max_recharge_error,
            "max_soc_balance_residual_mwh": max_soc_balance,
            "max_cyclic_soc_residual_mwh": max_cyclic_soc,
            "max_net_benefit_identity_residual_eur_per_kw_year": (
                max_economic_residual
            ),
            "max_envelope_reconstruction_residual_eur_per_kw_year": (
                max_envelope_residual
            ),
            "minimum_absolute_envelope_value_eur_per_kw_year": (
                minimum_abs_envelope_value
            ),
        },
        "paired_dispatch_check": {
            "max_perfect_minus_matched_service_loss_share": (
                max_service_loss_share
            ),
            "max_absolute_discharge_difference_mwh": (
                max_abs_discharge_difference
            ),
            "max_absolute_net_benefit_difference_eur_per_kw_year": (
                max_abs_net_difference
            ),
            "total_charge_avoided_mwh": total_charge_avoided_mwh,
            "aggregate_charge_avoided_pct": float(
                100
                * total_charge_avoided_mwh
                / comparison["matched_charge_mwh"].sum()
            ),
        },
        "recomputed_valley_summary": recomputed_summary.to_dict(
            orient="records"
        ),
        "confirmatory_result": {
            "perfect_foresight_dispersed_strict_valleys": int(
                perfect_dispersed["has_strict_valley"].sum()
            ),
            "perfect_foresight_clustered_strict_valleys": int(
                perfect_clustered["has_strict_valley"].sum()
            ),
            "country_years": EXPECTED_COUNTRY_YEARS,
            "clustered_mean_gap_width_h": float(
                perfect_clustered["gap_width_h"].mean()
            ),
            "clustered_median_gap_width_h": float(
                perfect_clustered["gap_width_h"].median()
            ),
            "clustered_modal_gap_width_h": float(
                perfect_clustered["gap_width_h"].mode().iloc[0]
            ),
        },
        "manuscript_evidence_admitted_after_independent_audit": True,
    }
    require(output_hashes_match, "recorded output hash mismatch")
    require(output_sizes_match, "recorded output size mismatch")
    require(input_hashes_match, "recorded input hash mismatch")
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
