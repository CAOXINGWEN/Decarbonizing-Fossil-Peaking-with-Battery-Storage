#!/usr/bin/env python3
"""Run a strict task-chronology × recharge-chronology 2×2 factorial.

The frozen task is the exact non-negative hourly aggregate ``Fossil gas``
series.  Task days and recharge days are reordered independently.  For each
factor, high-energy intact days are either dispersed or arranged in four-day
clusters within broad calendar segments.  Four within-segment circular phase
offsets marginalize the factorial effects over arbitrary task/recharge phase
alignment.

The default diagnostic covers R2 and R3 at zero recharge opportunity price.
R2 is a potential-VRE-surplus counterfactual.  R3 is an ACER annual-volume-
calibrated allocation over the R2 shape; it is not observed hourly
curtailment.  Every output is explicitly non-evidential.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_observed_gas_independent_recharge_pilot as core


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_OUTPUT = (
    ROOT
    / "model_adapter"
    / "observed_gas_independent_recharge_factorial_DE2023_caes24"
)
FACTOR_LEVELS = ("dispersed", "clustered")
DEFAULT_PHASE_FRACTIONS = (0.0, 0.25, 0.5, 0.75)
RECHARGE_ORDER_BASIS = "daily high recharge-opportunity energy"


def parse_float_list(value: str) -> list[float]:
    output = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not output:
        raise ValueError("At least one phase fraction is required")
    if any(item < 0.0 or item >= 1.0 for item in output):
        raise ValueError("Phase fractions must be in [0, 1)")
    if len(set(output)) != len(output):
        raise ValueError("Phase fractions must be unique")
    return output


def day_segments(panel: pd.DataFrame) -> pd.Series:
    return (
        pd.DataFrame(
            {
                "day": np.arange(len(panel)) // 24,
                "month": panel["timestamp_utc"].dt.month.to_numpy(),
            }
        )
        .groupby("day", sort=True)["month"]
        .first()
        .map(core.calendar_segment)
    )


def adjacency_statistics(
    order: list[int],
    extreme_days: set[int],
    segments: pd.Series,
) -> dict[str, float | int]:
    adjacent_pairs = 0
    run_lengths: list[int] = []
    for segment in sorted(segments.unique()):
        sequence = [
            day for day in order if int(segments.loc[day]) == int(segment)
        ]
        flags = [day in extreme_days for day in sequence]
        adjacent_pairs += sum(
            int(left and right)
            for left, right in zip(flags[:-1], flags[1:])
        )
        current = 0
        for flag in flags + [False]:
            if flag:
                current += 1
            elif current:
                run_lengths.append(current)
                current = 0
    return {
        "adjacent_extreme_day_pairs": int(adjacent_pairs),
        "maximum_extreme_day_run": int(max(run_lengths, default=0)),
        "mean_extreme_day_run": float(
            np.mean(run_lengths) if run_lengths else 0.0
        ),
    }


def make_recharge_orders(
    panel: pd.DataFrame,
    recharge: np.ndarray,
) -> tuple[dict[str, list[int]], dict[str, Any], set[int]]:
    if len(recharge) != 8760 or np.any(recharge < 0.0):
        raise RuntimeError("Recharge profile must contain 8,760 non-negative hours")
    segments = day_segments(panel)
    daily_energy = pd.Series(
        recharge.reshape(365, 24).sum(axis=1),
        index=np.arange(365),
        name="recharge_mwh",
    )
    threshold = float(daily_energy.quantile(core.EXTREME_QUANTILE))
    extreme_days = set(
        int(day) for day in daily_energy.index[daily_energy >= threshold]
    )
    if len(extreme_days) == 0 or len(extreme_days) == 365:
        raise RuntimeError(
            "Recharge chronology requires a non-degenerate high-opportunity "
            "day classification"
        )
    orders: dict[str, list[int]] = {}
    for level in FACTOR_LEVELS:
        order: list[int] = []
        for segment in sorted(segments.unique()):
            subset = [
                int(day)
                for day in segments.index[segments.eq(segment)]
            ]
            extreme = [day for day in subset if day in extreme_days]
            normal = [day for day in subset if day not in extreme_days]
            if level == "dispersed":
                order.extend(core.interleave(extreme, normal))
            else:
                order.extend(
                    core.interleave_clusters(
                        extreme,
                        normal,
                        core.CLUSTER_DAYS,
                    )
                )
        if len(order) != 365 or sorted(order) != list(range(365)):
            raise RuntimeError(f"Invalid recharge {level} day permutation")
        orders[level] = order
    diagnostics = {
        "recharge_order_basis": RECHARGE_ORDER_BASIS,
        "daily_recharge_extreme_quantile": core.EXTREME_QUANTILE,
        "daily_recharge_extreme_threshold_mwh": threshold,
        "daily_recharge_extreme_days": len(extreme_days),
        "dispersed": adjacency_statistics(
            orders["dispersed"], extreme_days, segments
        ),
        "clustered": adjacency_statistics(
            orders["clustered"], extreme_days, segments
        ),
    }
    if (
        diagnostics["clustered"]["adjacent_extreme_day_pairs"]
        <= diagnostics["dispersed"]["adjacent_extreme_day_pairs"]
    ):
        raise RuntimeError(
            "Recharge clustering manipulation did not increase adjacent "
            "high-opportunity day pairs"
        )
    return orders, diagnostics, extreme_days


def rotate_within_segments(
    order: list[int],
    segments: pd.Series,
    phase_fraction: float,
) -> tuple[list[int], dict[str, int]]:
    rotated: list[int] = []
    shifts: dict[str, int] = {}
    for segment in sorted(segments.unique()):
        sequence = [
            day for day in order if int(segments.loc[day]) == int(segment)
        ]
        shift = int(round(phase_fraction * len(sequence))) % len(sequence)
        shifts[str(int(segment))] = shift
        rotated.extend(sequence[shift:] + sequence[:shift])
    if len(rotated) != 365 or sorted(rotated) != list(range(365)):
        raise RuntimeError("Phase rotation broke the intact-day permutation")
    return rotated, shifts


def order_hash(order: list[int]) -> str:
    return hashlib.sha256(
        np.asarray(order, dtype=np.int16).tobytes()
    ).hexdigest()


def phase_label(value: float) -> str:
    return f"p{int(round(100 * value)):03d}"


def low_opportunity_gap_statistics(
    order: list[int],
    high_opportunity_days: set[int],
) -> dict[str, float | int]:
    positions = np.asarray(
        [
            index
            for index, day in enumerate(order)
            if day in high_opportunity_days
        ],
        dtype=int,
    )
    if len(positions) == 0:
        raise RuntimeError("No high-opportunity days in recharge order")
    gaps = np.concatenate(
        [
            np.asarray([positions[0]], dtype=int),
            np.diff(positions) - 1,
            np.asarray([364 - positions[-1]], dtype=int),
        ]
    )
    return {
        "maximum_low_recharge_gap_days": int(gaps.max()),
        "median_low_recharge_gap_days": float(np.median(gaps)),
    }


def annotate(
    row: dict[str, Any],
    dispatch: pd.DataFrame,
    *,
    task_chronology: str,
    recharge_chronology: str,
    phase_fraction: float,
    task_order: list[int],
    recharge_order: list[int],
    recharge_phase_shifts: dict[str, int],
    recharge_specification: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    values: dict[str, Any] = {
        "task_chronology": task_chronology,
        "recharge_chronology": recharge_chronology,
        "recharge_phase_fraction": phase_fraction,
        "recharge_phase_label": phase_label(phase_fraction),
        "task_day_order_sha256": order_hash(task_order),
        "recharge_day_order_sha256": order_hash(recharge_order),
        "recharge_phase_shifts_days": json.dumps(
            recharge_phase_shifts,
            sort_keys=True,
        ),
        "task_order_basis": "daily high observed aggregate fossil-gas energy",
        "recharge_order_basis": RECHARGE_ORDER_BASIS,
        "factorial_design": (
            "independent intact-day task chronology x recharge chronology"
        ),
        "recharge_specification": recharge_specification,
    }
    row.update(values)
    for key, value in values.items():
        dispatch[key] = value
    return row, dispatch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", default="DE")
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--config-id", default="caes_24h")
    parser.add_argument(
        "--recharge-specifications",
        default="R2,R3",
    )
    parser.add_argument(
        "--recharge-opportunity-price-cny-mwh",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--phase-fractions",
        default=",".join(str(value) for value in DEFAULT_PHASE_FRACTIONS),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    panel_sha = core.sha256(core.PANEL)
    acer_sha = core.sha256(core.ACER)
    if panel_sha != core.EXPECTED_PANEL_SHA256:
        raise RuntimeError(f"Frozen task panel SHA changed: {panel_sha}")
    if acer_sha != core.EXPECTED_ACER_SHA256:
        raise RuntimeError(f"Frozen ACER source SHA changed: {acer_sha}")

    specifications = core.parse_list(args.recharge_specifications)
    if set(specifications) - {"R2", "R3"}:
        raise RuntimeError("Factorial diagnostic supports only R2 and R3")
    if set(specifications) != {"R2", "R3"}:
        raise RuntimeError("Prespecified key factorial requires both R2 and R3")
    phase_fractions = parse_float_list(args.phase_fractions)
    panel = core.load_country_year(args.country, args.year)
    config = core.load_configuration(args.config_id)
    task_orders_all = core.make_day_orders(panel)
    task_orders = {
        level: task_orders_all[level] for level in FACTOR_LEVELS
    }
    recharge_profiles, recharge_profile_qa = core.build_recharge_profiles(
        panel
    )
    if not recharge_profile_qa["r3_admitted"]:
        raise RuntimeError(f"R3 not admitted for {args.country}-{args.year}")
    segments = day_segments(panel)
    frozen_task = panel["observed_gas_peaking_task_mw"].to_numpy(float)
    task_daily_energy = frozen_task.reshape(365, 24).sum(axis=1)
    task_extreme_threshold = float(
        pd.Series(task_daily_energy).quantile(core.EXTREME_QUANTILE)
    )
    task_extreme_days = set(
        int(day)
        for day in np.flatnonzero(
            task_daily_energy >= task_extreme_threshold
        )
    )

    recharge_orders_by_spec: dict[str, dict[str, list[int]]] = {}
    recharge_order_diagnostics: list[dict[str, Any]] = []
    (
        frozen_recharge_orders,
        frozen_recharge_diagnostics,
        frozen_recharge_extreme_days,
    ) = make_recharge_orders(panel, recharge_profiles["R2"])
    for specification in specifications:
        # R3 is a positive scalar multiple of R2. Reusing the exact R2
        # permutation prevents specification-specific phase drift.
        orders = {
            level: list(frozen_recharge_orders[level])
            for level in FACTOR_LEVELS
        }
        diagnostics = frozen_recharge_diagnostics
        recharge_orders_by_spec[specification] = orders
        specification_daily = recharge_profiles[specification].reshape(
            365, 24
        ).sum(axis=1)
        specification_threshold = float(
            pd.Series(specification_daily).quantile(
                core.EXTREME_QUANTILE
            )
        )
        for level in FACTOR_LEVELS:
            recharge_order_diagnostics.append(
                {
                    "country": args.country,
                    "year": args.year,
                    "config_id": args.config_id,
                    "recharge_specification": specification,
                    "recharge_chronology": level,
                    "recharge_order_basis": diagnostics[
                        "recharge_order_basis"
                    ],
                    "daily_recharge_extreme_quantile": diagnostics[
                        "daily_recharge_extreme_quantile"
                    ],
                    "daily_recharge_extreme_threshold_mwh": (
                        specification_threshold
                    ),
                    "daily_recharge_extreme_days": diagnostics[
                        "daily_recharge_extreme_days"
                    ],
                    "recharge_order_frozen_from_specification": "R2",
                    "same_recharge_order_reused_for_R2_and_R3": True,
                    **diagnostics[level],
                    **low_opportunity_gap_statistics(
                        orders[level],
                        frozen_recharge_extreme_days,
                    ),
                }
            )

    results: list[dict[str, Any]] = []
    dispatches: list[pd.DataFrame] = []
    conservation_rows: list[dict[str, Any]] = []
    baseline_costs: dict[tuple[str, str], float] = {}

    # One baseline per task chronology and recharge specification.  With no
    # forced storage, recharge is entirely spilled and its chronology is
    # economically inactive.
    for specification in specifications:
        source_recharge = recharge_profiles[specification]
        reference_recharge_order = recharge_orders_by_spec[specification][
            "dispersed"
        ]
        recharge = core.reorder_days(
            source_recharge,
            reference_recharge_order,
        )
        for task_chronology in FACTOR_LEVELS:
            task_order = task_orders[task_chronology]
            task = core.reorder_days(frozen_task, task_order)
            problem = core.build_problem(
                task,
                recharge,
                config,
                specification,
                recharge_opportunity_price_cny_mwh=0.0,
                forced_storage=False,
            )
            row, dispatch = core.solve_problem(
                problem,
                args.country,
                args.year,
                f"task_{task_chronology}__baseline",
                "independent_factorial",
                "baseline",
            )
            row, dispatch = annotate(
                row,
                dispatch,
                task_chronology=task_chronology,
                recharge_chronology="baseline_reference",
                phase_fraction=0.0,
                task_order=task_order,
                recharge_order=reference_recharge_order,
                recharge_phase_shifts={str(value): 0 for value in range(5)},
                recharge_specification=specification,
            )
            results.append(row)
            dispatches.append(dispatch)
            baseline_costs[(task_chronology, specification)] = float(
                row["total_system_cost_cny_per_mean_mw_year"]
            )

    for specification in specifications:
        source_recharge = recharge_profiles[specification]
        for task_chronology in FACTOR_LEVELS:
            task_order = task_orders[task_chronology]
            task = core.reorder_days(frozen_task, task_order)
            for recharge_chronology in FACTOR_LEVELS:
                base_recharge_order = recharge_orders_by_spec[
                    specification
                ][recharge_chronology]
                for phase_fraction in phase_fractions:
                    recharge_order, phase_shifts = rotate_within_segments(
                        base_recharge_order,
                        segments,
                        phase_fraction,
                    )
                    recharge = core.reorder_days(
                        source_recharge,
                        recharge_order,
                    )
                    task_daily_ordered = task.reshape(365, 24).sum(axis=1)
                    recharge_daily_ordered = recharge.reshape(
                        365, 24
                    ).sum(axis=1)
                    daily_rank_correlation = float(
                        pd.Series(task_daily_ordered).corr(
                            pd.Series(recharge_daily_ordered),
                            method="spearman",
                        )
                    )
                    task_high_flags = np.asarray(
                        [day in task_extreme_days for day in task_order],
                        dtype=bool,
                    )
                    recharge_high_flags = np.asarray(
                        [
                            day in frozen_recharge_extreme_days
                            for day in recharge_order
                        ],
                        dtype=bool,
                    )
                    recharge_gap_statistics = (
                        low_opportunity_gap_statistics(
                            recharge_order,
                            frozen_recharge_extreme_days,
                        )
                    )
                    chronology = (
                        f"task_{task_chronology}__recharge_"
                        f"{recharge_chronology}__"
                        f"{phase_label(phase_fraction)}"
                    )
                    problem = core.build_problem(
                        task,
                        recharge,
                        config,
                        specification,
                        recharge_opportunity_price_cny_mwh=(
                            args.recharge_opportunity_price_cny_mwh
                        ),
                        forced_storage=True,
                    )
                    row, dispatch = core.solve_problem(
                        problem,
                        args.country,
                        args.year,
                        chronology,
                        "independent_factorial",
                        "forced_1pct_joint",
                        baseline_cost=baseline_costs[
                            (task_chronology, specification)
                        ],
                    )
                    row, dispatch = annotate(
                        row,
                        dispatch,
                        task_chronology=task_chronology,
                        recharge_chronology=recharge_chronology,
                        phase_fraction=phase_fraction,
                        task_order=task_order,
                        recharge_order=recharge_order,
                        recharge_phase_shifts=phase_shifts,
                        recharge_specification=specification,
                    )
                    results.append(row)
                    dispatches.append(dispatch)
                    conservation_rows.append(
                        {
                            "country": args.country,
                            "year": args.year,
                            "config_id": args.config_id,
                            "recharge_specification": specification,
                            "task_chronology": task_chronology,
                            "recharge_chronology": recharge_chronology,
                            "recharge_phase_fraction": phase_fraction,
                            "task_day_order_sha256": order_hash(task_order),
                            "recharge_day_order_sha256": order_hash(
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
                                recharge.sum() - source_recharge.sum()
                            ),
                            "task_and_recharge_orders_constructed_"
                            "independently": True,
                            "task_recharge_daily_energy_spearman": (
                                daily_rank_correlation
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
                                frozen_recharge_extreme_days
                            ),
                            **recharge_gap_statistics,
                        }
                    )

    results_frame = pd.DataFrame(results)
    dispatch_frame = pd.concat(dispatches, ignore_index=True)
    conservation = pd.DataFrame(conservation_rows)
    diagnostics_frame = pd.DataFrame(recharge_order_diagnostics)
    results_frame.to_csv(output / "model_results.csv", index=False)
    dispatch_frame.to_csv(
        output / "hourly_dispatch.csv.gz",
        index=False,
    )
    conservation.to_csv(
        output / "factorial_cell_conservation_qa.csv",
        index=False,
    )
    diagnostics_frame.to_csv(
        output / "recharge_order_diagnostics.csv",
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
                "R3_interpretation": (
                    "ACER annual-volume-calibrated hourly allocation; not "
                    "observed hourly curtailment"
                ),
            }
        ]
    ).to_csv(output / "recharge_profile_qa.csv", index=False)

    forced = results_frame[
        results_frame["variant"].eq("forced_1pct_joint")
    ]
    expected_forced_rows = (
        len(specifications)
        * len(FACTOR_LEVELS)
        * len(FACTOR_LEVELS)
        * len(phase_fractions)
    )
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
    cell_counts = forced.groupby(
        [
            "recharge_specification",
            "task_chronology",
            "recharge_chronology",
            "recharge_phase_fraction",
        ]
    ).size()
    checks = {
        "frozen_panel_sha256_matches": (
            panel_sha == core.EXPECTED_PANEL_SHA256
        ),
        "acer_source_sha256_matches": (
            acer_sha == core.EXPECTED_ACER_SHA256
        ),
        "all_models_optimal": bool(
            results_frame["solver_status"].eq(0).all()
        ),
        "full_2x2_factorial_per_specification_and_phase": bool(
            len(forced) == expected_forced_rows
            and len(cell_counts) == expected_forced_rows
            and cell_counts.eq(1).all()
        ),
        "both_task_factor_levels_present": (
            set(forced["task_chronology"]) == set(FACTOR_LEVELS)
        ),
        "both_recharge_factor_levels_present": (
            set(forced["recharge_chronology"]) == set(FACTOR_LEVELS)
        ),
        "all_phase_offsets_present": (
            set(forced["recharge_phase_fraction"])
            == set(phase_fractions)
        ),
        "task_and_recharge_orders_independently_constructed": bool(
            conservation[
                "task_and_recharge_orders_constructed_independently"
            ].all()
        ),
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
        "recharge_clustering_manipulation_passes": bool(
            diagnostics_frame.groupby("recharge_specification")
            .apply(
                lambda group: float(
                    group.loc[
                        group["recharge_chronology"].eq("clustered"),
                        "adjacent_extreme_day_pairs",
                    ].iloc[0]
                )
                > float(
                    group.loc[
                        group["recharge_chronology"].eq("dispersed"),
                        "adjacent_extreme_day_pairs",
                    ].iloc[0]
                ),
                include_groups=False,
            )
            .all()
        ),
        "r2_r3_recharge_orders_are_identical": bool(
            diagnostics_frame.groupby("recharge_chronology")[
                [
                    "adjacent_extreme_day_pairs",
                    "maximum_extreme_day_run",
                    "maximum_low_recharge_gap_days",
                ]
            ]
            .nunique()
            .le(1)
            .all()
            .all()
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
            and conservation["maximum_low_recharge_gap_days"]
            .ge(0)
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
        "r3_admission_gate_passed": bool(
            recharge_profile_qa["r3_admitted"]
        ),
        "r2_not_observed_curtailment": True,
        "r3_not_observed_hourly_curtailment": True,
        "objective_decomposition_passes": bool(
            results_frame["objective_decomposition_error_cny"]
            .abs()
            .le(1e-3)
            .all()
        ),
    }
    qa = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "country": args.country,
        "year": args.year,
        "config_id": args.config_id,
        "recharge_specifications": specifications,
        "recharge_opportunity_price_cny_mwh": (
            args.recharge_opportunity_price_cny_mwh
        ),
        "task_chronology_levels": list(FACTOR_LEVELS),
        "recharge_chronology_levels": list(FACTOR_LEVELS),
        "recharge_phase_fractions": phase_fractions,
        "recharge_order_basis": RECHARGE_ORDER_BASIS,
        "recharge_order_frozen_from_specification": "R2",
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
        "high_task_high_recharge_overlap_days_range": [
            int(
                conservation[
                    "high_task_high_recharge_position_overlap_days"
                ].min()
            ),
            int(
                conservation[
                    "high_task_high_recharge_position_overlap_days"
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
        "R3_interpretation": (
            "ACER annual-volume-calibrated counterfactual allocation; not "
            "observed hourly curtailment"
        ),
        "manuscript_evidence_admitted": False,
        "reason": (
            "prespecified key-configuration factorial mechanism diagnostic; "
            "not full-spectrum or confirmatory evidence"
        ),
    }
    (output / "QA.json").write_text(
        json.dumps(qa, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "script": str(Path(__file__).resolve()),
        "script_sha256": core.sha256(Path(__file__).resolve()),
        "core_runner": str(Path(core.__file__).resolve()),
        "core_runner_sha256": core.sha256(Path(core.__file__).resolve()),
        "panel": str(core.PANEL),
        "panel_sha256": panel_sha,
        "acer_source": str(core.ACER),
        "acer_source_sha256": acer_sha,
        "configuration_source": str(core.CONFIGURATIONS),
        "configuration_source_sha256": core.sha256(core.CONFIGURATIONS),
        "factorial_contract": (
            "independent intact-day task and recharge reorderings; full 2x2 "
            "per specification and phase offset"
        ),
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
