#!/usr/bin/env python3
"""Merge QA-passed v62 capacity-expansion shards at one planning resolution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


TABLES = (
    "design_summary.csv",
    "capacity_mix.csv",
    "scenario_metrics.csv.gz",
    "chronology_conservation.csv",
    "diversification_comparison.csv",
)
ACTIVE_TOLERANCE = 1.0e-6
MATERIAL_POWER_CHANGE_PP = 1.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.force:
        raise RuntimeError("Output directory is non-empty; pass --force to replace")
    output.mkdir(parents=True, exist_ok=True)

    qa_rows: list[tuple[Path, dict[str, Any]]] = []
    countries_seen: set[str] = set()
    table_frames: dict[str, list[pd.DataFrame]] = {name: [] for name in TABLES}
    for input_dir in args.inputs:
        root = input_dir.resolve()
        qa = json.loads((root / "qa.json").read_text(encoding="utf-8"))
        if qa.get("status") != "pass":
            raise RuntimeError(f"Shard has not passed QA: {root}")
        countries = set(map(str, qa.get("countries", [])))
        overlap = countries_seen & countries
        if overlap:
            raise RuntimeError(f"Countries appear in multiple shards: {sorted(overlap)}")
        countries_seen |= countries
        qa_rows.append((root, qa))
        for name in TABLES:
            table_frames[name].append(pd.read_csv(root / name))

    planning_steps = {int(qa["planning_step_hours"]) for _, qa in qa_rows}
    input_contracts = {
        json.dumps(qa["input_sha256"], sort_keys=True) for _, qa in qa_rows
    }
    risk_contracts = {
        tuple(float(value) for value in qa["risk_weights"]) for _, qa in qa_rows
    }
    mode_contracts = {tuple(map(str, qa["modes"])) for _, qa in qa_rows}
    if not (
        len(planning_steps) == len(input_contracts) == len(risk_contracts)
        == len(mode_contracts) == 1
    ):
        raise RuntimeError("Shard inputs, planning resolution, risks or modes differ")

    merged: dict[str, pd.DataFrame] = {}
    for name, frames in table_frames.items():
        frame = pd.concat(frames, ignore_index=True)
        sort_columns = [
            column
            for column in (
                "country", "risk_weight", "mode", "year", "chronology", "config_id"
            )
            if column in frame.columns
        ]
        frame = frame.sort_values(sort_columns, kind="stable").reset_index(drop=True)
        frame.to_csv(output / name, index=False)
        merged[name] = frame

    summaries = merged["design_summary.csv"]
    capacities = merged["capacity_mix.csv"]
    conservation = merged["chronology_conservation.csv"]
    comparisons = merged["diversification_comparison.csv"]
    risks = next(iter(risk_contracts))
    modes = next(iter(mode_contracts))
    expected_rows = len(countries_seen) * len(risks) * len(modes)
    if summaries.duplicated(["country", "risk_weight", "mode"]).any():
        raise RuntimeError("Duplicate design summaries across shards")
    if len(summaries) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} designs, observed {len(summaries)}")

    diversified = capacities.loc[capacities["mode"].eq("diversified")].copy()
    diversified["active"] = diversified["capacity_per_normalized_mw"].gt(
        ACTIVE_TOLERANCE
    )
    shift_rows: list[dict[str, Any]] = []
    for country, group in diversified.groupby("country", sort=True):
        by_risk = {
            float(weight): subset.set_index("config_id")
            for weight, subset in group.groupby("risk_weight")
        }
        expected = by_risk[0.0]
        risk_aware = by_risk[0.5]
        expected_active = tuple(sorted(expected.index[expected["active"]]))
        risk_active = tuple(sorted(risk_aware.index[risk_aware["active"]]))
        expected_total = float(expected["capacity_per_normalized_mw"].sum())
        risk_total = float(risk_aware["capacity_per_normalized_mw"].sum())
        shift_rows.append(
            {
                "country": country,
                "expected_active_set": "+".join(expected_active) or "none",
                "risk_aware_active_set": "+".join(risk_active) or "none",
                "expected_selected_technology_count": len(expected_active),
                "risk_aware_selected_technology_count": len(risk_active),
                "expected_dominant_technology": str(
                    expected["capacity_per_normalized_mw"].idxmax()
                ),
                "risk_aware_dominant_technology": str(
                    risk_aware["capacity_per_normalized_mw"].idxmax()
                ),
                "technology_set_change": expected_active != risk_active,
                "dominant_technology_change": (
                    expected["capacity_per_normalized_mw"].idxmax()
                    != risk_aware["capacity_per_normalized_mw"].idxmax()
                ),
                "expected_total_power_percent_mean_gas": expected_total * 100.0,
                "risk_aware_total_power_percent_mean_gas": risk_total * 100.0,
                "risk_aware_minus_expected_power_percentage_points": (
                    risk_total - expected_total
                ) * 100.0,
                "material_power_change": abs(risk_total - expected_total) * 100.0
                >= MATERIAL_POWER_CHANGE_PP,
            }
        )
    shifts = pd.DataFrame(shift_rows)
    shifts.to_csv(output / "procurement_design_shifts.csv", index=False)

    expected_comparison = comparisons.loc[comparisons["risk_weight"].eq(0.0)]
    risk_comparison = comparisons.loc[comparisons["risk_weight"].eq(0.5)]
    scenario_count = int(
        conservation[["country", "year", "chronology"]].drop_duplicates().shape[0]
    )
    headline = {
        "country_count": len(countries_seen),
        "natural_country_year_count": int(
            conservation[["country", "year"]].drop_duplicates().shape[0]
        ),
        "chronology_scenario_count": scenario_count,
        "planning_step_hours": next(iter(planning_steps)),
        "expected_cost_mixed_portfolio_countries": int(
            shifts["expected_selected_technology_count"].gt(1).sum()
        ),
        "risk_aware_mixed_portfolio_countries": int(
            shifts["risk_aware_selected_technology_count"].gt(1).sum()
        ),
        "technology_set_change_countries": int(shifts["technology_set_change"].sum()),
        "dominant_technology_change_countries": int(
            shifts["dominant_technology_change"].sum()
        ),
        "material_power_change_countries": int(shifts["material_power_change"].sum()),
        "expected_cost_diversification_improvement_countries": int(
            expected_comparison[
                "diversified_minus_best_single_mean_cost_eur_per_normalized_mw"
            ].lt(-1.0e-6).sum()
        ),
        "risk_aware_cvar90_diversification_improvement_countries": int(
            risk_comparison[
                "diversified_minus_best_single_cvar90_cost_eur_per_normalized_mw"
            ].lt(-1.0e-6).sum()
        ),
        "median_expected_cost_diversification_change_eur_per_normalized_mw": float(
            expected_comparison[
                "diversified_minus_best_single_mean_cost_eur_per_normalized_mw"
            ].median()
        ),
        "median_risk_aware_cvar90_diversification_change_eur_per_normalized_mw": float(
            risk_comparison[
                "diversified_minus_best_single_cvar90_cost_eur_per_normalized_mw"
            ].median()
        ),
        "median_absolute_procurement_power_shift_percentage_points": float(
            shifts["risk_aware_minus_expected_power_percentage_points"].abs().median()
        ),
        "maximum_objective_identity_error_eur_per_normalized_mw": float(
            summaries["objective_identity_error_eur_per_normalized_mw"].max()
        ),
        "maximum_task_balance_residual": float(
            summaries["maximum_task_balance_residual"].max()
        ),
        "maximum_shared_recharge_violation": float(
            summaries["maximum_shared_recharge_violation"].max()
        ),
        "maximum_state_balance_residual_mwh": float(
            summaries["maximum_state_balance_residual_mwh"].max()
        ),
        "minimum_capacity_upper_bound_slack": float(
            summaries["maximum_capacity_upper_bound_slack"].min()
        ),
    }
    write_json(output / "headline_metrics.json", headline)

    energy_ranges = conservation.groupby(["country", "year"])[
        ["task_energy_normalized_mwh", "recharge_energy_normalized_mwh"]
    ].agg(lambda values: float(values.max() - values.min()))
    qa = {
        "status": "pass",
        "evidence_status": "merged_multiyear_capacity_expansion_validation",
        "countries": sorted(countries_seen),
        "planning_step_hours": next(iter(planning_steps)),
        **headline,
        "all_solver_statuses_optimal": bool(summaries["solver_status"].eq("optimal").all()),
        "maximum_chronology_task_energy_residual": float(
            energy_ranges["task_energy_normalized_mwh"].max()
        ),
        "maximum_chronology_recharge_energy_residual": float(
            energy_ranges["recharge_energy_normalized_mwh"].max()
        ),
        "input_sha256": json.loads(next(iter(input_contracts))),
        "source_shard_qa_sha256": {
            root.name: sha256(root / "qa.json") for root, _ in qa_rows
        },
    }
    qa["status"] = (
        "pass"
        if qa["all_solver_statuses_optimal"]
        and qa["maximum_objective_identity_error_eur_per_normalized_mw"] <= 0.02
        and qa["maximum_task_balance_residual"] <= 1.0e-6
        and qa["maximum_shared_recharge_violation"] <= 1.0e-6
        and qa["maximum_state_balance_residual_mwh"] <= 1.0e-6
        and qa["minimum_capacity_upper_bound_slack"] >= -1.0e-8
        and qa["maximum_chronology_task_energy_residual"] <= 1.0e-10
        and qa["maximum_chronology_recharge_energy_residual"] <= 1.0e-10
        else "fail"
    )
    qa["output_sha256"] = {
        name: sha256(output / name)
        for name in (*TABLES, "procurement_design_shifts.csv", "headline_metrics.json")
    }
    write_json(output / "qa.json", qa)
    print(json.dumps(qa, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if qa["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
