#!/usr/bin/env python3
"""Isolate chronology effects in the v62 multi-year procurement model.

The frozen capacity-expansion experiment pools dispersed and clustered
chronologies.  This companion experiment holds the technology catalogue,
cost year, value scenario, planning resolution and risk objective fixed while
solving four chronology ensembles independently:

* observed natural chronology only;
* conserved dispersed chronology only;
* conserved clustered chronology only; and
* the pooled dispersed-plus-clustered ensemble used by Supplementary Fig. 5.

Only the diversified three-technology design is needed for the chronology
contrast.  The pooled design is reused from the frozen, QA-passed output so
the published Supplementary Fig. 5 result is not recomputed or changed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
BASE_RUNNER_PATH = HERE / "run_v62_multiyear_capacity_expansion.py"
DEFAULT_COMBINED_ROOT = HERE / "results"
ENSEMBLES = ("observed", "dispersed", "clustered", "combined")
RISK_WEIGHTS = (0.0, 0.5)
ACTIVE_TOLERANCE = 1.0e-6
MATERIAL_POWER_THRESHOLD_PP = 1.0


def load_base_runner():
    specification = importlib.util.spec_from_file_location(
        "v62_capacity_expansion_base", BASE_RUNNER_PATH
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("Cannot import the v62 capacity-expansion runner")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    def convert(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): convert(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(val) for val in item]
        if isinstance(item, (np.integer,)):
            return int(item)
        if isinstance(item, (np.floating,)):
            value_float = float(item)
            return value_float if np.isfinite(value_float) else None
        if isinstance(item, (np.bool_,)):
            return bool(item)
        if isinstance(item, Path):
            return str(item)
        return item

    path.write_text(
        json.dumps(convert(value), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def prepare_one_chronology(
    base,
    task_hourly: np.ndarray,
    recharge_hourly: np.ndarray,
    step_hours: int,
) -> tuple[np.ndarray, np.ndarray]:
    usable_recharge_hourly = np.where(
        task_hourly > base.NUMERICAL_ZERO, 0.0, recharge_hourly
    )
    task = base.aggregate_blocks(task_hourly, step_hours, "mean")
    usable_recharge = base.aggregate_blocks(
        usable_recharge_hourly, step_hours, "strict_recharge"
    )
    return task, usable_recharge


def build_ensemble_scenarios(
    base,
    frame: pd.DataFrame,
    chronology_runner,
    step_hours: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    ensembles: dict[str, list[dict[str, Any]]] = {
        "observed": [],
        "dispersed": [],
        "clustered": [],
        "combined": [],
    }
    conservation: list[dict[str, Any]] = []
    for year, annual in frame.groupby("year", sort=True):
        annual = annual.sort_values("timestamp_utc", kind="stable").reset_index(
            drop=True
        )
        if len(annual) != 8760:
            raise RuntimeError(f"Year {year} does not contain 8,760 rows")
        scale_mw = float(annual["raw_gas_mean_mw"].iloc[0])
        if scale_mw <= 0:
            raise RuntimeError("The normalization scale must be positive")

        task_original = annual[base.TASK_COLUMN].to_numpy(dtype=float) / scale_mw
        recharge_original = (
            annual[base.RECHARGE_COLUMN].to_numpy(dtype=float) / scale_mw
        )
        orders, _ = chronology_runner.day_orders(annual)
        chronology_inputs = {
            "observed": (task_original, recharge_original),
            "dispersed": (
                chronology_runner.reorder_days(task_original, orders["dispersed"]),
                chronology_runner.reorder_days(
                    recharge_original, orders["dispersed"]
                ),
            ),
            "clustered": (
                chronology_runner.reorder_days(task_original, orders["clustered"]),
                chronology_runner.reorder_days(
                    recharge_original, orders["clustered"]
                ),
            ),
        }
        for chronology, (task_hourly, recharge_hourly) in chronology_inputs.items():
            task, usable_recharge = prepare_one_chronology(
                base, task_hourly, recharge_hourly, step_hours
            )
            scenario = {
                "scenario_id": f"{int(year)}_{chronology}",
                "year": int(year),
                "chronology": chronology,
                "normalization_scale_mw": scale_mw,
                "task": task,
                "usable_recharge": usable_recharge,
            }
            ensembles[chronology].append(scenario)
            if chronology != "observed":
                ensembles["combined"].append(scenario)
            conservation.append(
                {
                    "year": int(year),
                    "chronology": chronology,
                    "task_energy_normalized_mwh": float(task.sum() * step_hours),
                    "recharge_energy_normalized_mwh": float(
                        recharge_hourly.sum()
                    ),
                    "usable_recharge_energy_normalized_mwh": float(
                        usable_recharge.sum() * step_hours
                    ),
                }
            )
    return ensembles, conservation


def read_reused_combined(
    root: Path,
    countries: list[str],
    planning_step_hours: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    qa = json.loads((root / "qa.json").read_text(encoding="utf-8"))
    if qa.get("status") != "pass":
        raise RuntimeError("The pooled chronology output has not passed QA")
    summaries = pd.read_csv(root / "design_summary.csv")
    capacities = pd.read_csv(root / "capacity_mix.csv")
    scenarios = pd.read_csv(root / "scenario_metrics.csv.gz")
    summaries = summaries.loc[
        summaries["country"].isin(countries)
        & summaries["mode"].eq("diversified")
        & summaries["risk_weight"].isin(RISK_WEIGHTS)
    ].copy()
    capacities = capacities.loc[
        capacities["country"].isin(countries)
        & capacities["mode"].eq("diversified")
        & capacities["risk_weight"].isin(RISK_WEIGHTS)
    ].copy()
    scenarios = scenarios.loc[
        scenarios["country"].isin(countries)
        & scenarios["mode"].eq("diversified")
        & scenarios["risk_weight"].isin(RISK_WEIGHTS)
    ].copy()
    if not summaries["planning_step_hours"].eq(planning_step_hours).all():
        raise RuntimeError("Pooled chronology resolution does not match this run")
    expected_rows = len(countries) * len(RISK_WEIGHTS)
    if len(summaries) != expected_rows:
        raise RuntimeError("Pooled chronology output is incomplete for this shard")
    for frame in (summaries, capacities, scenarios):
        frame.insert(1, "ensemble", "combined")
    return summaries, capacities, scenarios


def active_set(group: pd.DataFrame) -> str:
    active = group.loc[
        group["capacity_per_normalized_mw"].gt(ACTIVE_TOLERANCE), "config_id"
    ].astype(str)
    return "+".join(sorted(active)) if len(active) else "none"


def dominant_technology(group: pd.DataFrame) -> str:
    ordered = group.sort_values(
        ["capacity_per_normalized_mw", "config_id"],
        ascending=[False, True],
        kind="stable",
    )
    if ordered.empty or float(ordered.iloc[0]["capacity_per_normalized_mw"]) <= ACTIVE_TOLERANCE:
        return "none"
    return str(ordered.iloc[0]["config_id"])


def build_comparisons(
    summaries: pd.DataFrame,
    capacities: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for (country, risk_weight, ensemble), summary_group in summaries.groupby(
        ["country", "risk_weight", "ensemble"], sort=True
    ):
        summary = summary_group.iloc[0]
        capacity = capacities.loc[
            capacities["country"].eq(country)
            & capacities["risk_weight"].eq(risk_weight)
            & capacities["ensemble"].eq(ensemble)
        ].copy()
        row: dict[str, Any] = {
            "country": country,
            "risk_weight": float(risk_weight),
            "ensemble": ensemble,
            "scenario_count": int(summary["scenario_count"]),
            "total_power_capacity_per_normalized_mw": float(
                summary["total_power_capacity_per_normalized_mw"]
            ),
            "total_power_capacity_percent_mean_gas": 100.0
            * float(summary["total_power_capacity_per_normalized_mw"]),
            "active_set": active_set(capacity),
            "dominant_technology": dominant_technology(capacity),
        }
        for config_id in base_config_ids(capacities):
            selected = capacity.loc[capacity["config_id"].astype(str).eq(config_id)]
            row[f"{config_id}_power_capacity_per_normalized_mw"] = (
                float(selected.iloc[0]["capacity_per_normalized_mw"])
                if len(selected)
                else 0.0
            )
            row[f"{config_id}_capacity_share"] = (
                float(selected.iloc[0]["capacity_share"]) if len(selected) else 0.0
            )
        records.append(row)
    ensemble_frame = pd.DataFrame(records)

    direct_records: list[dict[str, Any]] = []
    config_ids = base_config_ids(capacities)
    for (country, risk_weight), group in ensemble_frame.groupby(
        ["country", "risk_weight"], sort=True
    ):
        indexed = group.set_index("ensemble")
        if set(indexed.index) != set(ENSEMBLES):
            raise RuntimeError(f"Incomplete ensemble comparison for {country}")
        observed = indexed.loc["observed"]
        dispersed = indexed.loc["dispersed"]
        clustered = indexed.loc["clustered"]
        combined = indexed.loc["combined"]
        row = {
            "country": country,
            "risk_weight": float(risk_weight),
            "clustered_minus_dispersed_total_power_pp": float(
                clustered["total_power_capacity_percent_mean_gas"]
                - dispersed["total_power_capacity_percent_mean_gas"]
            ),
            "clustered_vs_dispersed_material_power_change": bool(
                abs(
                    clustered["total_power_capacity_percent_mean_gas"]
                    - dispersed["total_power_capacity_percent_mean_gas"]
                )
                > MATERIAL_POWER_THRESHOLD_PP
            ),
            "clustered_vs_dispersed_technology_set_change": bool(
                clustered["active_set"] != dispersed["active_set"]
            ),
            "clustered_vs_dispersed_dominant_technology_change": bool(
                clustered["dominant_technology"]
                != dispersed["dominant_technology"]
            ),
            "combined_minus_observed_total_power_pp": float(
                combined["total_power_capacity_percent_mean_gas"]
                - observed["total_power_capacity_percent_mean_gas"]
            ),
            "combined_vs_observed_material_power_change": bool(
                abs(
                    combined["total_power_capacity_percent_mean_gas"]
                    - observed["total_power_capacity_percent_mean_gas"]
                )
                > MATERIAL_POWER_THRESHOLD_PP
            ),
            "combined_vs_observed_technology_set_change": bool(
                combined["active_set"] != observed["active_set"]
            ),
            "combined_vs_observed_dominant_technology_change": bool(
                combined["dominant_technology"]
                != observed["dominant_technology"]
            ),
            "clustered_vs_dispersed_mix_total_variation_pp": 50.0
            * float(
                sum(
                    abs(
                        clustered[f"{config_id}_capacity_share"]
                        - dispersed[f"{config_id}_capacity_share"]
                    )
                    for config_id in config_ids
                )
            ),
            "combined_vs_observed_mix_total_variation_pp": 50.0
            * float(
                sum(
                    abs(
                        combined[f"{config_id}_capacity_share"]
                        - observed[f"{config_id}_capacity_share"]
                    )
                    for config_id in config_ids
                )
            ),
        }
        for config_id in config_ids:
            row[f"clustered_minus_dispersed_{config_id}_power_pp"] = 100.0 * float(
                clustered[f"{config_id}_power_capacity_per_normalized_mw"]
                - dispersed[f"{config_id}_power_capacity_per_normalized_mw"]
            )
            row[f"combined_minus_observed_{config_id}_power_pp"] = 100.0 * float(
                combined[f"{config_id}_power_capacity_per_normalized_mw"]
                - observed[f"{config_id}_power_capacity_per_normalized_mw"]
            )
        direct_records.append(row)
    direct_frame = pd.DataFrame(direct_records)

    headline: dict[str, Any] = {
        "country_count": int(direct_frame["country"].nunique()),
        "ensemble_count": len(ENSEMBLES),
        "risk_weights": list(RISK_WEIGHTS),
        "material_power_threshold_percentage_points": MATERIAL_POWER_THRESHOLD_PP,
    }
    for risk_weight, label in ((0.0, "expected"), (0.5, "mean_cvar90")):
        subset = direct_frame.loc[direct_frame["risk_weight"].eq(risk_weight)]
        headline[label] = {
            "clustered_vs_dispersed_material_power_change_countries": int(
                subset["clustered_vs_dispersed_material_power_change"].sum()
            ),
            "clustered_vs_dispersed_technology_set_change_countries": int(
                subset["clustered_vs_dispersed_technology_set_change"].sum()
            ),
            "clustered_vs_dispersed_dominant_technology_change_countries": int(
                subset["clustered_vs_dispersed_dominant_technology_change"].sum()
            ),
            "clustered_vs_dispersed_median_absolute_power_change_pp": float(
                subset["clustered_minus_dispersed_total_power_pp"].abs().median()
            ),
            "clustered_vs_dispersed_median_mix_total_variation_pp": float(
                subset["clustered_vs_dispersed_mix_total_variation_pp"].median()
            ),
            "combined_vs_observed_material_power_change_countries": int(
                subset["combined_vs_observed_material_power_change"].sum()
            ),
            "combined_vs_observed_technology_set_change_countries": int(
                subset["combined_vs_observed_technology_set_change"].sum()
            ),
            "combined_vs_observed_median_absolute_power_change_pp": float(
                subset["combined_minus_observed_total_power_pp"].abs().median()
            ),
        }
    return ensemble_frame, direct_frame, headline


def base_config_ids(capacities: pd.DataFrame) -> list[str]:
    present = set(capacities["config_id"].astype(str))
    ordered = [
        "li_lfp_4h",
        "caes_24h",
        "h2_cavern_168h",
    ]
    return [item for item in ordered if item in present]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--countries", default="")
    parser.add_argument("--planning-step-hours", type=int, default=6)
    parser.add_argument("--cost-year", type=int, default=2035)
    parser.add_argument("--value-scenario", type=int, default=20)
    parser.add_argument(
        "--reuse-combined-root", type=Path, default=DEFAULT_COMBINED_ROOT
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.planning_step_hours <= 0 or 8760 % args.planning_step_hours:
        raise ValueError("Planning step must be a positive divisor of 8,760")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.force:
        raise RuntimeError("Output directory is non-empty; pass --force to replace")
    output.mkdir(parents=True, exist_ok=True)

    base = load_base_runner()
    chronology_runner = base.load_registered_runner()
    usecols = [
        "country",
        "year",
        "timestamp_utc",
        "source_day_id",
        "season_segment",
        "event_flag_q75",
        "task_c3_q75_mw",
        "raw_gas_mean_mw",
        base.TASK_COLUMN,
        base.RECHARGE_COLUMN,
    ]
    panel = pd.read_csv(base.PANEL_PATH, usecols=usecols)
    available = sorted(panel["country"].astype(str).unique())
    countries = parse_list(args.countries) or available
    missing = sorted(set(countries) - set(available))
    if missing:
        raise RuntimeError(f"Countries absent from the panel: {missing}")
    technologies = base.load_technologies(args.cost_year)
    value = base.load_value_scenario(args.value_scenario)

    summary_rows: list[dict[str, Any]] = []
    capacity_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    conservation_rows: list[dict[str, Any]] = []
    started_at = utc_now()
    for country in countries:
        country_frame = panel.loc[panel["country"].astype(str).eq(country)].copy()
        ensembles, conservation = build_ensemble_scenarios(
            base, country_frame, chronology_runner, args.planning_step_hours
        )
        conservation_rows.extend(
            [{"country": country, **row} for row in conservation]
        )
        for ensemble in ("observed", "dispersed", "clustered"):
            for risk_weight in RISK_WEIGHTS:
                summary, capacities, scenario_metrics = base.solve_country_design_highs(
                    country=country,
                    scenarios=ensembles[ensemble],
                    technologies=technologies,
                    value=value,
                    risk_weight=risk_weight,
                    mode="diversified",
                    step_hours=args.planning_step_hours,
                    threads=1,
                )
                summary_rows.append({**summary, "ensemble": ensemble})
                capacity_rows.extend(
                    [{**row, "ensemble": ensemble} for row in capacities]
                )
                scenario_rows.extend(
                    [{**row, "ensemble": ensemble} for row in scenario_metrics]
                )
        pd.DataFrame(summary_rows).to_csv(
            output / "design_summary_checkpoint.csv", index=False
        )

    reused_summaries, reused_capacities, reused_scenarios = read_reused_combined(
        args.reuse_combined_root.resolve(), countries, args.planning_step_hours
    )
    summaries = pd.concat(
        [pd.DataFrame(summary_rows), reused_summaries], ignore_index=True
    )
    capacities = pd.concat(
        [pd.DataFrame(capacity_rows), reused_capacities], ignore_index=True
    )
    scenario_metrics = pd.concat(
        [pd.DataFrame(scenario_rows), reused_scenarios], ignore_index=True
    )
    conservation = pd.DataFrame(conservation_rows)

    summaries.to_csv(output / "design_summary.csv", index=False)
    capacities.to_csv(output / "capacity_mix.csv", index=False)
    scenario_metrics.to_csv(output / "scenario_metrics.csv.gz", index=False)
    conservation.to_csv(output / "chronology_conservation.csv", index=False)
    ensemble_comparison, direct_effect, headline = build_comparisons(
        summaries, capacities
    )
    ensemble_comparison.to_csv(output / "ensemble_comparison.csv", index=False)
    direct_effect.to_csv(output / "direct_chronology_effect.csv", index=False)
    write_json(output / "headline_metrics.json", headline)

    energy_ranges = conservation.groupby(["country", "year"])[
        ["task_energy_normalized_mwh", "recharge_energy_normalized_mwh"]
    ].agg(lambda values: float(values.max() - values.min()))
    qa = {
        "status": "pass",
        "evidence_status": "chronology_isolated_multiyear_procurement",
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "countries": countries,
        "country_count": len(countries),
        "ensembles": list(ENSEMBLES),
        "risk_weights": list(RISK_WEIGHTS),
        "planning_step_hours": args.planning_step_hours,
        "all_solver_statuses_optimal": bool(
            summaries["solver_status"].eq("optimal").all()
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
        "maximum_chronology_task_energy_residual": float(
            energy_ranges["task_energy_normalized_mwh"].max()
        ),
        "maximum_chronology_recharge_energy_residual": float(
            energy_ranges["recharge_energy_normalized_mwh"].max()
        ),
        "input_sha256": {
            "panel": sha256(base.PANEL_PATH),
            "base_runner": sha256(BASE_RUNNER_PATH),
            "registered_chronology_runner": sha256(base.REGISTERED_RUNNER_PATH),
            "cost_catalogue": sha256(base.COST_CATALOGUE_PATH),
            "value_scenarios": sha256(base.VALUE_SCENARIO_PATH),
            "reused_combined_design_summary": sha256(
                args.reuse_combined_root.resolve() / "design_summary.csv"
            ),
        },
    }
    qa["status"] = (
        "pass"
        if qa["all_solver_statuses_optimal"]
        and qa["maximum_objective_identity_error_eur_per_normalized_mw"] <= 0.02
        and qa["maximum_task_balance_residual"] <= 1.0e-6
        and qa["maximum_shared_recharge_violation"] <= 1.0e-6
        and qa["maximum_state_balance_residual_mwh"] <= 1.0e-6
        and qa["maximum_chronology_task_energy_residual"] <= 1.0e-10
        and qa["maximum_chronology_recharge_energy_residual"] <= 1.0e-10
        else "fail"
    )
    qa["output_sha256"] = {
        name: sha256(output / name)
        for name in (
            "design_summary.csv",
            "capacity_mix.csv",
            "scenario_metrics.csv.gz",
            "chronology_conservation.csv",
            "ensemble_comparison.csv",
            "direct_chronology_effect.csv",
            "headline_metrics.json",
        )
    }
    write_json(output / "qa.json", qa)
    (output / "design_summary_checkpoint.csv").unlink(missing_ok=True)
    print(json.dumps(qa, ensure_ascii=False, indent=2))
    return 0 if qa["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
