#!/usr/bin/env python3
"""Merge country shards from the v64 2 h diversified-capacity run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


EXPECTED_COUNTRIES = {
    "AT", "BE", "BG", "CZ", "DE", "DK", "ES", "FR", "GB", "GR",
    "HR", "IT", "PT", "RO", "SI",
}
TABLES = (
    "design_summary.csv",
    "capacity_mix.csv",
    "scenario_metrics.csv.gz",
    "chronology_conservation.csv",
    "procurement_design_shifts.csv",
)


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

    countries_seen: set[str] = set()
    qas: list[tuple[Path, dict[str, Any]]] = []
    frames: dict[str, list[pd.DataFrame]] = {name: [] for name in TABLES}
    for input_path in args.inputs:
        root = input_path.resolve()
        qa = json.loads((root / "qa.json").read_text(encoding="utf-8"))
        if qa.get("status") != "pass":
            raise RuntimeError(f"Shard has not passed QA: {root}")
        countries = set(map(str, qa["countries"]))
        overlap = countries_seen & countries
        if overlap:
            raise RuntimeError(f"Duplicate countries across shards: {sorted(overlap)}")
        countries_seen |= countries
        qas.append((root, qa))
        for name in TABLES:
            frames[name].append(pd.read_csv(root / name))

    if countries_seen != EXPECTED_COUNTRIES:
        raise RuntimeError(
            f"Expected {sorted(EXPECTED_COUNTRIES)}, observed {sorted(countries_seen)}"
        )
    step_values = {int(qa["planning_step_hours"]) for _, qa in qas}
    input_contracts = {
        json.dumps(qa["input_sha256"], sort_keys=True) for _, qa in qas
    }
    if step_values != {2} or len(input_contracts) != 1:
        raise RuntimeError("Shard resolution or input contracts differ")

    merged: dict[str, pd.DataFrame] = {}
    for name, parts in frames.items():
        frame = pd.concat(parts, ignore_index=True)
        sort_columns = [
            column for column in (
                "country", "risk_weight", "mode", "year", "chronology", "config_id"
            ) if column in frame.columns
        ]
        frame = frame.sort_values(sort_columns, kind="stable").reset_index(drop=True)
        frame.to_csv(output / name, index=False)
        merged[name] = frame

    summaries = merged["design_summary.csv"]
    capacities = merged["capacity_mix.csv"]
    shifts = merged["procurement_design_shifts.csv"]
    conservation = merged["chronology_conservation.csv"]
    if len(summaries) != 30 or summaries.duplicated(["country", "risk_weight"]).any():
        raise RuntimeError("The merged 2 h design panel is incomplete")
    if len(capacities) != 90:
        raise RuntimeError("The merged 2 h technology-capacity panel is incomplete")
    if len(shifts) != 15 or shifts["country"].duplicated().any():
        raise RuntimeError("The merged 2 h risk-response panel is incomplete")

    energy_ranges = conservation.groupby(["country", "year"])[
        ["task_energy_normalized_mwh", "recharge_energy_normalized_mwh"]
    ].agg(lambda values: float(values.max() - values.min()))
    qa = {
        "status": "pass",
        "evidence_status": "merged_v64_two_hour_diversified_resolution_convergence",
        "countries": sorted(countries_seen),
        "country_count": len(countries_seen),
        "planning_step_hours": 2,
        "risk_weights": [0.0, 0.5],
        "modes": ["diversified"],
        "design_count": len(summaries),
        "all_solver_statuses_optimal": bool(summaries["solver_status"].eq("optimal").all()),
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
        "maximum_chronology_task_energy_residual": float(
            energy_ranges["task_energy_normalized_mwh"].max()
        ),
        "maximum_chronology_recharge_energy_residual": float(
            energy_ranges["recharge_energy_normalized_mwh"].max()
        ),
        "input_sha256": json.loads(next(iter(input_contracts))),
        "source_shard_qa_sha256": {
            root.name: sha256(root / "qa.json") for root, _ in qas
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
    qa["output_sha256"] = {name: sha256(output / name) for name in TABLES}
    write_json(output / "qa.json", qa)
    print(json.dumps(qa, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if qa["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
