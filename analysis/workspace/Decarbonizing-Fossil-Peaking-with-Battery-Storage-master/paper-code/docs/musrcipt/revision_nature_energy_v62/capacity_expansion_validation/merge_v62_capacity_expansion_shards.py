#!/usr/bin/env python3
"""Merge and audit the three v62 capacity-expansion result shards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SHARDS = [HERE / "shard1", HERE / "shard2", HERE / "shard3"]
OUTPUT = HERE / "results"
ACTIVE_TOLERANCE = 1.0e-6
MATERIAL_POWER_CHANGE_PERCENTAGE_POINTS = 1.0


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


def concatenate(name: str) -> pd.DataFrame:
    frames = [pd.read_csv(shard / name) for shard in SHARDS]
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    for shard in SHARDS:
        qa = json.loads((shard / "qa.json").read_text(encoding="utf-8"))
        if qa.get("status") != "pass":
            raise RuntimeError(f"Shard did not pass: {shard}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summaries = concatenate("design_summary.csv")
    capacities = concatenate("capacity_mix.csv")
    scenarios = concatenate("scenario_metrics.csv.gz")
    conservation = concatenate("chronology_conservation.csv")
    comparisons = concatenate("diversification_comparison.csv")

    key = ["country", "risk_weight", "mode"]
    if summaries.duplicated(key).any():
        raise RuntimeError("Duplicate design summaries across shards")
    expected_rows = summaries["country"].nunique() * 2 * 4
    if len(summaries) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} design rows, observed {len(summaries)}"
        )
    expected_scenarios = (
        conservation[["country", "year", "chronology"]]
        .drop_duplicates()
        .shape[0]
    )
    if expected_scenarios != 264:
        raise RuntimeError(f"Expected 264 chronology scenarios, found {expected_scenarios}")

    diversified_capacity = capacities.loc[
        capacities["mode"].eq("diversified")
    ].copy()
    diversified_capacity["active"] = diversified_capacity[
        "capacity_per_normalized_mw"
    ].gt(ACTIVE_TOLERANCE)
    mix_records: list[dict[str, Any]] = []
    for country, group in diversified_capacity.groupby("country", sort=True):
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
        expected_dominant = str(expected["capacity_per_normalized_mw"].idxmax())
        risk_dominant = str(risk_aware["capacity_per_normalized_mw"].idxmax())
        mix_records.append(
            {
                "country": country,
                "expected_active_set": "+".join(expected_active) or "none",
                "risk_aware_active_set": "+".join(risk_active) or "none",
                "expected_selected_technology_count": len(expected_active),
                "risk_aware_selected_technology_count": len(risk_active),
                "expected_dominant_technology": expected_dominant,
                "risk_aware_dominant_technology": risk_dominant,
                "technology_set_change": expected_active != risk_active,
                "dominant_technology_change": expected_dominant != risk_dominant,
                "expected_total_power_percent_mean_gas": expected_total * 100.0,
                "risk_aware_total_power_percent_mean_gas": risk_total * 100.0,
                "risk_aware_minus_expected_power_percentage_points": (
                    risk_total - expected_total
                ) * 100.0,
                "material_power_change": abs(risk_total - expected_total) * 100.0
                >= MATERIAL_POWER_CHANGE_PERCENTAGE_POINTS,
            }
        )
    procurement_shifts = pd.DataFrame(mix_records)

    risk_comparison = comparisons.loc[comparisons["risk_weight"].eq(0.5)].copy()
    expected_comparison = comparisons.loc[comparisons["risk_weight"].eq(0.0)].copy()
    headline = {
        "country_count": int(summaries["country"].nunique()),
        "natural_country_year_count": int(
            conservation[["country", "year"]].drop_duplicates().shape[0]
        ),
        "chronology_scenario_count": expected_scenarios,
        "expected_cost_mixed_portfolio_countries": int(
            procurement_shifts["expected_selected_technology_count"].gt(1).sum()
        ),
        "risk_aware_mixed_portfolio_countries": int(
            procurement_shifts["risk_aware_selected_technology_count"].gt(1).sum()
        ),
        "technology_set_change_countries": int(
            procurement_shifts["technology_set_change"].sum()
        ),
        "dominant_technology_change_countries": int(
            procurement_shifts["dominant_technology_change"].sum()
        ),
        "material_power_change_countries": int(
            procurement_shifts["material_power_change"].sum()
        ),
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
            procurement_shifts[
                "risk_aware_minus_expected_power_percentage_points"
            ].abs().median()
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

    summaries.to_csv(OUTPUT / "design_summary.csv", index=False)
    capacities.to_csv(OUTPUT / "capacity_mix.csv", index=False)
    scenarios.to_csv(OUTPUT / "scenario_metrics.csv.gz", index=False)
    conservation.to_csv(OUTPUT / "chronology_conservation.csv", index=False)
    comparisons.to_csv(OUTPUT / "diversification_comparison.csv", index=False)
    procurement_shifts.to_csv(OUTPUT / "procurement_design_shifts.csv", index=False)
    write_json(OUTPUT / "headline_metrics.json", headline)

    qa = {
        "status": "pass",
        "evidence_status": "merged_multiyear_capacity_expansion_validation",
        **headline,
        "all_solver_statuses_optimal": bool(
            summaries["solver_status"].eq("optimal").all()
        ),
        "source_shard_qa_sha256": {
            shard.name: sha256(shard / "qa.json") for shard in SHARDS
        },
        "output_sha256": {
            name: sha256(OUTPUT / name)
            for name in (
                "design_summary.csv",
                "capacity_mix.csv",
                "scenario_metrics.csv.gz",
                "chronology_conservation.csv",
                "diversification_comparison.csv",
                "procurement_design_shifts.csv",
                "headline_metrics.json",
            )
        },
    }
    write_json(OUTPUT / "qa.json", qa)
    print(json.dumps(qa, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
