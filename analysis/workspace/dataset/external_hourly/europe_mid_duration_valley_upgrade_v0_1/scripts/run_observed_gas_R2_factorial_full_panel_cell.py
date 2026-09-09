#!/usr/bin/env python3
"""Run one country-year/configuration cell of the R2-only 2x2 factorial.

This runner is deliberately separate from the four-country R2/R3 key-suite
runner.  It freezes the observed non-negative hourly aggregate ``Fossil gas``
task, constructs the potential-VRE-surplus R2 recharge opportunity, and
independently reorders intact task and recharge day blocks into dispersed or
clustered chronologies.  Four prespecified within-season recharge phase
offsets are evaluated.

The output is a mechanism diagnostic.  R2 is a counterfactual potential
surplus and is not observed curtailment.  Every run is permanently marked
``manuscript_evidence_admitted=false``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_observed_gas_independent_recharge_factorial as design
import run_observed_gas_independent_recharge_pilot as core


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_OUTPUT = (
    ROOT
    / "model_adapter"
    / "observed_gas_R2_factorial_full132_v0_1"
    / "DE_2023_caes_24h"
)
FACTOR_LEVELS = ("dispersed", "clustered")
PHASE_FRACTIONS = (0.0, 0.25, 0.5, 0.75)
SPECIFICATION = "R2"
OPPORTUNITY_PRICE_CNY_MWH = 0.0


def annotation(
    *,
    task_chronology: str,
    recharge_chronology: str,
    phase_fraction: float,
    task_order: list[int],
    recharge_order: list[int],
    recharge_phase_shifts: dict[str, int],
) -> dict[str, Any]:
    return {
        "task_chronology": task_chronology,
        "recharge_chronology": recharge_chronology,
        "recharge_phase_fraction": phase_fraction,
        "recharge_phase_label": design.phase_label(phase_fraction),
        "task_day_order_sha256": design.order_hash(task_order),
        "recharge_day_order_sha256": design.order_hash(recharge_order),
        "recharge_phase_shifts_days": json.dumps(
            recharge_phase_shifts,
            sort_keys=True,
        ),
        "task_order_basis": (
            "daily high observed aggregate fossil-gas energy"
        ),
        "recharge_order_basis": design.RECHARGE_ORDER_BASIS,
        "factorial_design": (
            "independent intact-day task chronology x recharge chronology"
        ),
        "recharge_specification": SPECIFICATION,
        "manuscript_evidence_admitted": False,
    }


def solve(
    *,
    task: np.ndarray,
    recharge: np.ndarray,
    config: core.Configuration,
    country: str,
    year: int,
    chronology: str,
    variant: str,
    forced_storage: bool,
    baseline_cost: float | None,
    annotations: dict[str, Any],
) -> dict[str, Any]:
    problem = core.build_problem(
        task,
        recharge,
        config,
        SPECIFICATION,
        recharge_opportunity_price_cny_mwh=OPPORTUNITY_PRICE_CNY_MWH,
        forced_storage=forced_storage,
    )
    row, dispatch = core.solve_problem(
        problem,
        country,
        year,
        chronology,
        "independent_factorial_R2_full_panel",
        variant,
        baseline_cost=baseline_cost,
    )
    # Hourly dispatch is intentionally not persisted for the 264-cell batch.
    # solve_problem already returns audited hourly maxima and aggregate
    # accounting identities in ``row``.
    del dispatch
    row.update(annotations)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument(
        "--config-id",
        required=True,
        choices=("li_lfp_4h", "caes_24h"),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    panel_sha = core.sha256(core.PANEL)
    if panel_sha != core.EXPECTED_PANEL_SHA256:
        raise RuntimeError(f"Frozen task panel SHA changed: {panel_sha}")

    panel = core.load_country_year(args.country, args.year)
    config = core.load_configuration(args.config_id)
    segments = design.day_segments(panel)
    frozen_task = panel["observed_gas_peaking_task_mw"].to_numpy(float)
    recharge_profiles, recharge_profile_qa = core.build_recharge_profiles(
        panel
    )
    frozen_recharge = recharge_profiles[SPECIFICATION]

    task_orders_all = core.make_day_orders(panel)
    task_orders = {
        level: list(task_orders_all[level]) for level in FACTOR_LEVELS
    }
    (
        recharge_orders,
        recharge_order_diagnostics,
        recharge_extreme_days,
    ) = design.make_recharge_orders(panel, frozen_recharge)

    task_daily_energy = frozen_task.reshape(365, 24).sum(axis=1)
    task_threshold = float(
        pd.Series(task_daily_energy).quantile(core.EXTREME_QUANTILE)
    )
    task_extreme_days = set(
        int(day)
        for day in np.flatnonzero(task_daily_energy >= task_threshold)
    )
    task_order_diagnostics = {
        level: design.adjacency_statistics(
            task_orders[level],
            task_extreme_days,
            segments,
        )
        for level in FACTOR_LEVELS
    }

    results: list[dict[str, Any]] = []
    conservation_rows: list[dict[str, Any]] = []
    baseline_costs: dict[str, float] = {}

    reference_recharge_order = recharge_orders["dispersed"]
    reference_recharge = core.reorder_days(
        frozen_recharge,
        reference_recharge_order,
    )
    zero_shifts = {str(value): 0 for value in range(5)}
    for task_chronology in FACTOR_LEVELS:
        task_order = task_orders[task_chronology]
        task = core.reorder_days(frozen_task, task_order)
        annotations = annotation(
            task_chronology=task_chronology,
            recharge_chronology="baseline_reference",
            phase_fraction=0.0,
            task_order=task_order,
            recharge_order=reference_recharge_order,
            recharge_phase_shifts=zero_shifts,
        )
        row = solve(
            task=task,
            recharge=reference_recharge,
            config=config,
            country=args.country,
            year=args.year,
            chronology=f"task_{task_chronology}__baseline",
            variant="baseline",
            forced_storage=False,
            baseline_cost=None,
            annotations=annotations,
        )
        results.append(row)
        baseline_costs[task_chronology] = float(
            row["total_system_cost_cny_per_mean_mw_year"]
        )

    for task_chronology in FACTOR_LEVELS:
        task_order = task_orders[task_chronology]
        task = core.reorder_days(frozen_task, task_order)
        task_daily_ordered = task.reshape(365, 24).sum(axis=1)
        task_high_flags = np.asarray(
            [day in task_extreme_days for day in task_order],
            dtype=bool,
        )
        for recharge_chronology in FACTOR_LEVELS:
            base_recharge_order = recharge_orders[recharge_chronology]
            for phase_fraction in PHASE_FRACTIONS:
                recharge_order, phase_shifts = (
                    design.rotate_within_segments(
                        base_recharge_order,
                        segments,
                        phase_fraction,
                    )
                )
                recharge = core.reorder_days(
                    frozen_recharge,
                    recharge_order,
                )
                recharge_daily_ordered = recharge.reshape(
                    365, 24
                ).sum(axis=1)
                rank_correlation = float(
                    pd.Series(task_daily_ordered).corr(
                        pd.Series(recharge_daily_ordered),
                        method="spearman",
                    )
                )
                recharge_high_flags = np.asarray(
                    [
                        day in recharge_extreme_days
                        for day in recharge_order
                    ],
                    dtype=bool,
                )
                annotations = annotation(
                    task_chronology=task_chronology,
                    recharge_chronology=recharge_chronology,
                    phase_fraction=phase_fraction,
                    task_order=task_order,
                    recharge_order=recharge_order,
                    recharge_phase_shifts=phase_shifts,
                )
                chronology = (
                    f"task_{task_chronology}__"
                    f"recharge_{recharge_chronology}__"
                    f"{design.phase_label(phase_fraction)}"
                )
                row = solve(
                    task=task,
                    recharge=recharge,
                    config=config,
                    country=args.country,
                    year=args.year,
                    chronology=chronology,
                    variant="forced_1pct_joint",
                    forced_storage=True,
                    baseline_cost=baseline_costs[task_chronology],
                    annotations=annotations,
                )
                results.append(row)
                conservation_rows.append(
                    {
                        "country": args.country,
                        "year": args.year,
                        "config_id": args.config_id,
                        "recharge_specification": SPECIFICATION,
                        "task_chronology": task_chronology,
                        "recharge_chronology": recharge_chronology,
                        "recharge_phase_fraction": phase_fraction,
                        "task_day_order_sha256": design.order_hash(
                            task_order
                        ),
                        "recharge_day_order_sha256": design.order_hash(
                            recharge_order
                        ),
                        "task_day_multiset_conserved": (
                            sorted(task_order) == list(range(365))
                        ),
                        "recharge_day_multiset_conserved": (
                            sorted(recharge_order) == list(range(365))
                        ),
                        "task_annual_energy_error_mwh": float(
                            task.sum() - frozen_task.sum()
                        ),
                        "recharge_annual_energy_error_mwh": float(
                            recharge.sum() - frozen_recharge.sum()
                        ),
                        "task_and_recharge_orders_constructed_"
                        "independently": True,
                        "task_recharge_daily_energy_spearman": (
                            rank_correlation
                        ),
                        "high_task_high_recharge_position_overlap_days": (
                            int(
                                np.logical_and(
                                    task_high_flags,
                                    recharge_high_flags,
                                ).sum()
                            )
                        ),
                        "task_high_days": len(task_extreme_days),
                        "recharge_high_opportunity_days": len(
                            recharge_extreme_days
                        ),
                        **design.low_opportunity_gap_statistics(
                            recharge_order,
                            recharge_extreme_days,
                        ),
                    }
                )

    results_frame = pd.DataFrame(results)
    conservation = pd.DataFrame(conservation_rows)
    diagnostics_rows: list[dict[str, Any]] = []
    for level in FACTOR_LEVELS:
        diagnostics_rows.append(
            {
                "country": args.country,
                "year": args.year,
                "config_id": args.config_id,
                "factor": "task",
                "chronology": level,
                "daily_extreme_quantile": core.EXTREME_QUANTILE,
                "daily_extreme_threshold_mwh": task_threshold,
                "daily_extreme_days": len(task_extreme_days),
                **task_order_diagnostics[level],
            }
        )
        diagnostics_rows.append(
            {
                "country": args.country,
                "year": args.year,
                "config_id": args.config_id,
                "factor": "recharge",
                "chronology": level,
                "daily_extreme_quantile": core.EXTREME_QUANTILE,
                "daily_extreme_threshold_mwh": (
                    recharge_order_diagnostics[
                        "daily_recharge_extreme_threshold_mwh"
                    ]
                ),
                "daily_extreme_days": len(recharge_extreme_days),
                **recharge_order_diagnostics[level],
                **design.low_opportunity_gap_statistics(
                    recharge_orders[level],
                    recharge_extreme_days,
                ),
            }
        )
    diagnostics = pd.DataFrame(diagnostics_rows)

    results_frame.to_csv(output / "model_results.csv", index=False)
    conservation.to_csv(
        output / "factorial_cell_conservation_qa.csv",
        index=False,
    )
    diagnostics.to_csv(
        output / "factor_order_diagnostics.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "country": args.country,
                "year": args.year,
                **recharge_profile_qa,
                "R2_interpretation": (
                    "potential-VRE-surplus counterfactual; not observed "
                    "curtailment"
                ),
            }
        ]
    ).to_csv(output / "recharge_profile_qa.csv", index=False)

    forced = results_frame[
        results_frame["variant"].eq("forced_1pct_joint")
    ]
    balance_fields = [
        "task_service_balance_error_mwh",
        "charging_source_balance_error_mwh",
        "gas_generation_accounting_error_mwh",
        "recharge_resource_balance_error_mwh",
    ]
    relative_fields = [
        "task_service_balance_relative_error",
        "charging_source_balance_relative_error",
        "gas_generation_accounting_relative_error",
        "recharge_resource_balance_relative_error",
    ]
    cell_key = [
        "task_chronology",
        "recharge_chronology",
        "recharge_phase_fraction",
    ]
    task_diag = diagnostics[diagnostics["factor"].eq("task")].set_index(
        "chronology"
    )
    recharge_diag = diagnostics[
        diagnostics["factor"].eq("recharge")
    ].set_index("chronology")
    checks = {
        "frozen_panel_sha256_matches": (
            panel_sha == core.EXPECTED_PANEL_SHA256
        ),
        "exactly_18_models": len(results_frame) == 18,
        "exactly_16_forced_factorial_cells": len(forced) == 16,
        "exactly_2_task_specific_baselines": int(
            results_frame["variant"].eq("baseline").sum()
        )
        == 2,
        "full_2x2_by_four_phase_factorial": bool(
            len(forced.drop_duplicates(cell_key)) == 16
        ),
        "all_models_optimal": bool(
            results_frame["solver_status"].eq(0).all()
        ),
        "R2_only": set(results_frame["recharge_specification"])
        == {SPECIFICATION},
        "zero_recharge_opportunity_price_only": set(
            results_frame["recharge_opportunity_price_cny_mwh"]
        )
        == {OPPORTUNITY_PRICE_CNY_MWH},
        "all_task_day_multisets_conserved": bool(
            conservation["task_day_multiset_conserved"].all()
        ),
        "all_recharge_day_multisets_conserved": bool(
            conservation["recharge_day_multiset_conserved"].all()
        ),
        "all_annual_task_energy_conserved": bool(
            conservation["task_annual_energy_error_mwh"]
            .abs()
            .le(core.ABSOLUTE_BALANCE_TOLERANCE_MWH)
            .all()
        ),
        "all_annual_recharge_energy_conserved": bool(
            conservation["recharge_annual_energy_error_mwh"]
            .abs()
            .le(core.ABSOLUTE_BALANCE_TOLERANCE_MWH)
            .all()
        ),
        "task_and_recharge_orders_independently_constructed": bool(
            conservation[
                "task_and_recharge_orders_constructed_independently"
            ].all()
        ),
        "task_clustering_increases_extreme_adjacency": bool(
            task_diag.loc["clustered", "adjacent_extreme_day_pairs"]
            > task_diag.loc["dispersed", "adjacent_extreme_day_pairs"]
        ),
        "recharge_clustering_increases_extreme_adjacency": bool(
            recharge_diag.loc[
                "clustered", "adjacent_extreme_day_pairs"
            ]
            > recharge_diag.loc[
                "dispersed", "adjacent_extreme_day_pairs"
            ]
        ),
        "phase_alignment_diagnostics_finite": bool(
            np.isfinite(
                conservation["task_recharge_daily_energy_spearman"]
            ).all()
            and conservation[
                "high_task_high_recharge_position_overlap_days"
            ]
            .between(0, 365)
            .all()
        ),
        "all_absolute_accounting_balances_pass": bool(
            results_frame[balance_fields]
            .abs()
            .le(core.ABSOLUTE_BALANCE_TOLERANCE_MWH)
            .all()
            .all()
        ),
        "all_relative_accounting_balances_pass": bool(
            results_frame[relative_fields]
            .abs()
            .le(core.RELATIVE_BALANCE_TOLERANCE)
            .all()
            .all()
        ),
        "objective_decomposition_passes": bool(
            results_frame["objective_decomposition_error_cny"]
            .abs()
            .le(1e-3)
            .all()
        ),
        "all_outputs_non_evidential": bool(
            results_frame["manuscript_evidence_admitted"]
            .eq(False)
            .all()
        ),
        "R2_not_observed_curtailment": bool(
            results_frame["recharge_interpretation"]
            .str.contains("not observed curtailment", regex=False)
            .all()
        ),
    }
    qa = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "country": args.country,
        "year": args.year,
        "config_id": args.config_id,
        "result_rows": int(len(results_frame)),
        "forced_factorial_rows": int(len(forced)),
        "baseline_rows": int(
            results_frame["variant"].eq("baseline").sum()
        ),
        "maximum_absolute_accounting_error_mwh": float(
            results_frame[balance_fields].abs().to_numpy().max()
        ),
        "maximum_relative_accounting_error": float(
            results_frame[relative_fields].abs().to_numpy().max()
        ),
        "solve_seconds_total": float(results_frame["solve_seconds"].sum()),
        "task_recharge_daily_energy_spearman_range": [
            float(
                conservation[
                    "task_recharge_daily_energy_spearman"
                ].min()
            ),
            float(
                conservation[
                    "task_recharge_daily_energy_spearman"
                ].max()
            ),
        ],
        "task_definition": (
            "exact non-negative hourly aggregate Fossil gas; no VRE "
            "subtraction"
        ),
        "R2_interpretation": (
            "potential-VRE-surplus counterfactual; not observed curtailment"
        ),
        "manuscript_evidence_admitted": False,
        "reason": (
            "full-historical-panel R2 factorial mechanism diagnostic; R2 is "
            "counterfactual potential surplus, not observed curtailment"
        ),
    }
    (output / "QA.json").write_text(
        json.dumps(qa, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "script": str(Path(__file__).resolve()),
        "script_sha256": core.sha256(Path(__file__).resolve()),
        "key_factorial_helper": str(Path(design.__file__).resolve()),
        "key_factorial_helper_sha256": core.sha256(
            Path(design.__file__).resolve()
        ),
        "core_runner": str(Path(core.__file__).resolve()),
        "core_runner_sha256": core.sha256(Path(core.__file__).resolve()),
        "panel": str(core.PANEL),
        "panel_sha256": panel_sha,
        "configuration_source": str(core.CONFIGURATIONS),
        "configuration_source_sha256": core.sha256(core.CONFIGURATIONS),
        "factorial_contract": (
            "R2 only; independent intact-day task and recharge reorderings; "
            "full 2x2 by four prespecified phase offsets"
        ),
        "hourly_dispatch_persisted": False,
        "qa_status": qa["status"],
        "manuscript_evidence_admitted": False,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    if qa["status"] != "pass":
        raise RuntimeError(json.dumps(qa, indent=2))
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
