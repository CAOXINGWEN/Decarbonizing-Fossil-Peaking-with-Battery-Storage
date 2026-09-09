#!/usr/bin/env python3
"""Summarize the strict observed-gas task × recharge 2×2 factorial."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUN_ROOT = (
    ROOT
    / "model_adapter"
    / "observed_gas_independent_recharge_factorial_key_suite_v0_1"
)
ARTIFACT_ROOT = ROOT / "artifacts" / "research"
OUTPUT_STEM = ARTIFACT_ROOT / (
    "OBSERVED_GAS_INDEPENDENT_RECHARGE_FACTORIAL_KEY_SUITE"
)
OUTPUT_CSV = OUTPUT_STEM.with_suffix(".csv")
OUTPUT_QA = ARTIFACT_ROOT / (
    "OBSERVED_GAS_INDEPENDENT_RECHARGE_FACTORIAL_KEY_SUITE_QA.json"
)
OUTPUT_MD = OUTPUT_STEM.with_suffix(".md")

COUNTRIES = ("DE", "ES", "HR", "IT")
CONFIGS = ("li_lfp_4h", "caes_24h")
SPECIFICATIONS = ("R2", "R3")
LEVELS = ("dispersed", "clustered")
PHASES = (0.0, 0.25, 0.5, 0.75)
EFFECTS = ("task_main", "recharge_main", "interaction")

RAW_METRICS = (
    "storage_charge_input_mwh",
    "storage_discharge_output_mwh",
    "gas_supplied_charge_mwh",
    "recharge_used_mwh",
    "unserved_task_mwh",
    "total_system_cost_cny_per_mean_mw_year",
    "signed_benefit_cny_per_forced_kw_year",
    "equivalent_full_cycles_output_basis",
    "charge_power_saturation_h",
    "discharge_power_saturation_h",
)
NORMALIZED_ENERGY_METRICS = (
    "storage_charge_input_mwh",
    "storage_discharge_output_mwh",
    "gas_supplied_charge_mwh",
    "recharge_used_mwh",
    "unserved_task_mwh",
)
BALANCE_FIELDS = (
    "task_service_balance_error_mwh",
    "charging_source_balance_error_mwh",
    "gas_generation_accounting_error_mwh",
    "recharge_resource_balance_error_mwh",
)
RELATIVE_BALANCE_FIELDS = (
    "task_service_balance_relative_error",
    "charging_source_balance_relative_error",
    "gas_generation_accounting_relative_error",
    "recharge_resource_balance_relative_error",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_runs() -> tuple[pd.DataFrame, list[dict[str, Any]], list[str]]:
    frames: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    failures: list[str] = []
    for country in COUNTRIES:
        for config in CONFIGS:
            directory = RUN_ROOT / f"{country}_{config}"
            result_path = directory / "model_results.csv"
            qa_path = directory / "QA.json"
            manifest_path = directory / "run_manifest.json"
            conservation_path = (
                directory / "factorial_cell_conservation_qa.csv"
            )
            required = (
                result_path,
                qa_path,
                manifest_path,
                conservation_path,
            )
            missing = [str(path) for path in required if not path.exists()]
            if missing:
                failures.append(
                    f"{directory.name}: missing {', '.join(missing)}"
                )
                continue
            qa = json.loads(qa_path.read_text(encoding="utf-8"))
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            failed_checks = [
                name
                for name, value in qa.get("checks", {}).items()
                if not bool(value)
            ]
            if qa.get("status") != "pass" or failed_checks:
                failures.append(
                    f"{directory.name}: QA={qa.get('status')}, "
                    f"failed={failed_checks}"
                )
            if manifest.get("manuscript_evidence_admitted") is not False:
                failures.append(
                    f"{directory.name}: evidence admission must be false"
                )
            frame = pd.read_csv(result_path)
            conservation = pd.read_csv(conservation_path)
            frame["run_directory"] = str(directory)
            frames.append(frame)
            audits.append(
                {
                    "country": country,
                    "config_id": config,
                    "directory": str(directory),
                    "qa_status": qa.get("status"),
                    "failed_checks": failed_checks,
                    "result_rows": int(len(frame)),
                    "forced_rows": int(
                        frame["variant"].eq("forced_1pct_joint").sum()
                    ),
                    "baseline_rows": int(
                        frame["variant"].eq("baseline").sum()
                    ),
                    "conservation_rows": int(len(conservation)),
                    "model_results_sha256": sha256(result_path),
                    "qa_sha256": sha256(qa_path),
                    "manifest_sha256": sha256(manifest_path),
                    "panel_sha256": manifest.get("panel_sha256"),
                    "acer_source_sha256": manifest.get(
                        "acer_source_sha256"
                    ),
                    "configuration_source_sha256": manifest.get(
                        "configuration_source_sha256"
                    ),
                    "runner_sha256": manifest.get("script_sha256"),
                    "core_runner_sha256": manifest.get(
                        "core_runner_sha256"
                    ),
                    "manuscript_evidence_admitted": manifest.get(
                        "manuscript_evidence_admitted"
                    ),
                    "minimum_task_recharge_rank_correlation": float(
                        conservation[
                            "task_recharge_daily_energy_spearman"
                        ].min()
                    ),
                    "maximum_task_recharge_rank_correlation": float(
                        conservation[
                            "task_recharge_daily_energy_spearman"
                        ].max()
                    ),
                    "minimum_high_day_overlap": int(
                        conservation[
                            "high_task_high_recharge_position_overlap_days"
                        ].min()
                    ),
                    "maximum_high_day_overlap": int(
                        conservation[
                            "high_task_high_recharge_position_overlap_days"
                        ].max()
                    ),
                }
            )
    if not frames:
        return pd.DataFrame(), audits, failures
    return pd.concat(frames, ignore_index=True), audits, failures


def normalize(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    output = frame.copy()
    denominator = output["mean_observed_gas_task_mw"]
    normalized_columns: list[str] = []
    for metric in NORMALIZED_ENERGY_METRICS:
        column = f"{metric}_per_mean_task_mw"
        output[column] = output[metric] / denominator
        normalized_columns.append(column)
    cost_column = "total_system_cost_cny_per_mean_task_kw_year"
    output[cost_column] = (
        output["total_system_cost_cny_per_mean_mw_year"] / 1000.0
    )
    normalized_columns.append(cost_column)
    metric_columns = list(RAW_METRICS) + normalized_columns
    return output, metric_columns


def factorial_effects(
    forced: pd.DataFrame,
    metric_columns: list[str],
) -> pd.DataFrame:
    group_columns = [
        "country",
        "year",
        "config_id",
        "recharge_specification",
        "recharge_opportunity_price_cny_mwh",
        "recharge_phase_fraction",
    ]
    rows: list[dict[str, Any]] = []
    coefficients = {
        "task_main": {
            ("dispersed", "dispersed"): -0.5,
            ("dispersed", "clustered"): -0.5,
            ("clustered", "dispersed"): 0.5,
            ("clustered", "clustered"): 0.5,
        },
        "recharge_main": {
            ("dispersed", "dispersed"): -0.5,
            ("dispersed", "clustered"): 0.5,
            ("clustered", "dispersed"): -0.5,
            ("clustered", "clustered"): 0.5,
        },
        "interaction": {
            ("dispersed", "dispersed"): 1.0,
            ("dispersed", "clustered"): -1.0,
            ("clustered", "dispersed"): -1.0,
            ("clustered", "clustered"): 1.0,
        },
    }
    for key, group in forced.groupby(group_columns, sort=True):
        cells = group.set_index(
            ["task_chronology", "recharge_chronology"]
        )
        if set(cells.index) != {
            (task, recharge)
            for task in LEVELS
            for recharge in LEVELS
        }:
            continue
        base = dict(zip(group_columns, key, strict=True))
        for effect, effect_coefficients in coefficients.items():
            row = {
                **base,
                "row_type": "technology_factorial_effect_phase",
                "effect_type": effect,
            }
            for metric in metric_columns:
                row[metric] = float(
                    sum(
                        coefficient * cells.loc[cell, metric]
                        for cell, coefficient in effect_coefficients.items()
                    )
                )
            rows.append(row)
    return pd.DataFrame(rows)


def phase_means(
    effects: pd.DataFrame,
    metric_columns: list[str],
    row_type: str,
) -> pd.DataFrame:
    group_columns = [
        "country",
        "year",
        "config_id",
        "recharge_specification",
        "recharge_opportunity_price_cny_mwh",
        "effect_type",
    ]
    output = (
        effects.groupby(group_columns, as_index=False)[metric_columns]
        .mean()
        .copy()
    )
    output["recharge_phase_fraction"] = np.nan
    output["row_type"] = row_type
    return output


def technology_contrasts(
    effects: pd.DataFrame,
    metric_columns: list[str],
    row_type: str,
) -> pd.DataFrame:
    group_columns = [
        "country",
        "year",
        "recharge_specification",
        "recharge_opportunity_price_cny_mwh",
        "recharge_phase_fraction",
        "effect_type",
    ]
    rows: list[dict[str, Any]] = []
    for key, group in effects.groupby(group_columns, sort=True):
        by_config = group.set_index("config_id")
        if set(by_config.index) != set(CONFIGS):
            continue
        row = dict(zip(group_columns, key, strict=True))
        row["config_id"] = "caes_24h_minus_li_lfp_4h"
        row["row_type"] = row_type
        for metric in metric_columns:
            row[metric] = float(
                by_config.loc["caes_24h", metric]
                - by_config.loc["li_lfp_4h", metric]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def add_cell_rows(
    forced: pd.DataFrame,
    metric_columns: list[str],
) -> pd.DataFrame:
    columns = [
        "country",
        "year",
        "config_id",
        "recharge_specification",
        "recharge_opportunity_price_cny_mwh",
        "recharge_phase_fraction",
        "task_chronology",
        "recharge_chronology",
    ] + metric_columns
    output = forced[columns].copy()
    output["row_type"] = "factorial_cell"
    output["effect_type"] = ""
    return output


def fmt(value: float, digits: int = 2) -> str:
    if not np.isfinite(value):
        return "NA"
    threshold = 0.5 * 10 ** (-digits)
    if abs(value) < threshold:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def mean_range(group: pd.DataFrame, column: str) -> str:
    return (
        f"{fmt(float(group[column].mean()))} "
        f"[{fmt(float(group[column].min()))}, "
        f"{fmt(float(group[column].max()))}]"
    )


def build_markdown(
    technology_phase_mean: pd.DataFrame,
    contrasts: pd.DataFrame,
    contrast_phase_mean: pd.DataFrame,
    qa: dict[str, Any],
) -> str:
    metrics = {
        "charge": "storage_charge_input_mwh_per_mean_task_mw",
        "discharge": "storage_discharge_output_mwh_per_mean_task_mw",
        "gas_charge": "gas_supplied_charge_mwh_per_mean_task_mw",
        "recharge_used": "recharge_used_mwh_per_mean_task_mw",
        "unserved": "unserved_task_mwh_per_mean_task_mw",
        "cost": "total_system_cost_cny_per_mean_task_kw_year",
        "benefit": "signed_benefit_cny_per_forced_kw_year",
        "efc": "equivalent_full_cycles_output_basis",
    }
    lines = [
        "# Observed-gas independent-recharge strict factorial",
        "",
        "## Design and evidence status",
        "",
        (
            "The frozen task is the exact non-negative hourly aggregate "
            "`Fossil gas` series. Task and recharge intact-day blocks are "
            "reordered independently in a strict 2×2 design: task chronology "
            "(dispersed or clustered) × recharge-opportunity chronology "
            "(dispersed or clustered). R2 and R3 are evaluated at 0 CNY "
            "MWh⁻¹ for LFP 4 h and CAES 24 h in DE, ES, HR and IT (2023)."
        ),
        "",
        (
            "Recharge effects are averaged over four prespecified "
            "within-season circular phase offsets (0, 0.25, 0.50 and 0.75) "
            "so the factorial estimate is not tied to one arbitrary alignment "
            "between task and recharge clusters. R2 is a potential-VRE-"
            "surplus counterfactual. R3 reuses the same R2 day ordering and "
            "allocates an ACER annual curtailment volume over that shape; it "
            "is not observed hourly curtailment."
        ),
        "",
        (
            "**Evidence status:** diagnostic only; all runs and aggregate "
            "outputs remain `manuscript_evidence_admitted=false`."
        ),
        "",
        "## QA",
        "",
        f"- Runs: {qa['observed_run_count']} / {qa['expected_run_count']}.",
        f"- Forced factorial cells: {qa['forced_factorial_rows']}.",
        f"- Aggregate QA status: **{qa['status']}**.",
        f"- Maximum absolute accounting residual: "
        f"{qa['maximum_absolute_accounting_error_mwh']:.3e} MWh.",
        f"- Maximum relative accounting residual: "
        f"{qa['maximum_relative_accounting_error']:.3e}.",
        "",
        "## Result synopsis",
        "",
    ]
    synopsis_rows: list[str] = []
    for specification in SPECIFICATIONS:
        selected = contrast_phase_mean[
            contrast_phase_mean["recharge_specification"].eq(
                specification
            )
        ].set_index("effect_type")
        task_discharge = float(
            selected.loc["task_main", metrics["discharge"]].mean()
        )
        task_cost = float(
            selected.loc["task_main", metrics["cost"]].mean()
        )
        recharge_discharge = float(
            selected.loc["recharge_main", metrics["discharge"]].mean()
        )
        recharge_cost = float(
            selected.loc["recharge_main", metrics["cost"]].mean()
        )
        interaction_discharge = float(
            selected.loc["interaction", metrics["discharge"]].mean()
        )
        interaction_cost = float(
            selected.loc["interaction", metrics["cost"]].mean()
        )
        synopsis_rows.append(
            f"- **{specification}:** CAES24−LFP4 task-clustering main "
            f"effect is {fmt(task_discharge)} annual MWh per mean-task MW "
            f"for discharge and {fmt(task_cost)} CNY per mean-task kW-year "
            f"for system cost. The recharge-opportunity clustering main "
            f"effect is {fmt(recharge_discharge)} and "
            f"{fmt(recharge_cost)}, respectively. The four-cell interaction "
            f"is {fmt(interaction_discharge)} and "
            f"{fmt(interaction_cost)}."
        )
    lines.extend(
        synopsis_rows
        + [
            "",
            (
                "Thus the strict orthogonal design does not reproduce a "
                "material task-clustering main effect. The dominant response "
                "loads on concentration of recharge opportunities, which "
                "creates longer intervening low-recharge gaps. The four-cell "
                "interaction is much smaller and is phase-sensitive."
            ),
            "",
            "## Technology-specific factorial effects",
            "",
            (
                "Values are phase-marginalized four-country means [minimum, "
                "maximum across country means]. The task main effect averages "
                "clustered-minus-dispersed task responses over both recharge "
                "chronologies. The recharge main effect does the converse. The "
                "interaction is Y_CC − Y_CD − Y_DC + Y_DD, where the first "
                "subscript is task chronology and the second is recharge "
                "chronology."
            ),
            "",
            "| Recharge | Technology | Effect | Charge throughput | Discharge "
            "throughput | Gas-to-charge | Recharge used | Unserved | System cost "
            "| Incremental benefit | EFC |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for (
        specification,
        config,
        effect,
    ), group in technology_phase_mean.groupby(
        ["recharge_specification", "config_id", "effect_type"],
        sort=True,
    ):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(specification),
                    str(config),
                    str(effect),
                    mean_range(group, metrics["charge"]),
                    mean_range(group, metrics["discharge"]),
                    mean_range(group, metrics["gas_charge"]),
                    mean_range(group, metrics["recharge_used"]),
                    mean_range(group, metrics["unserved"]),
                    mean_range(group, metrics["cost"]),
                    mean_range(group, metrics["benefit"]),
                    mean_range(group, metrics["efc"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## CAES 24 h minus LFP 4 h effect contrasts",
            "",
            (
                "The interaction row is the requested CAES–LFP four-cell "
                "factorial contrast: "
                "(Y_CC − Y_CD − Y_DC + Y_DD)_CAES minus the same expression "
                "for LFP. A positive system-cost contrast indicates a larger "
                "CAES cost penalty; a negative throughput or incremental-"
                "benefit contrast indicates a larger CAES loss."
            ),
            "",
            "| Recharge | Effect | Charge throughput | Discharge throughput "
            "| Gas-to-charge | Recharge used | Unserved | System cost | "
            "Incremental benefit | EFC |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for (
        specification,
        effect,
    ), group in contrast_phase_mean.groupby(
        ["recharge_specification", "effect_type"],
        sort=True,
    ):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(specification),
                    str(effect),
                    mean_range(group, metrics["charge"]),
                    mean_range(group, metrics["discharge"]),
                    mean_range(group, metrics["gas_charge"]),
                    mean_range(group, metrics["recharge_used"]),
                    mean_range(group, metrics["unserved"]),
                    mean_range(group, metrics["cost"]),
                    mean_range(group, metrics["benefit"]),
                    mean_range(group, metrics["efc"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Phase sensitivity of CAES–LFP contrasts",
            "",
            (
                "Counts below refer to the 16 country × phase diagnostic "
                "cells. They are sensitivity counts, not independent "
                "replicates or inferential sample sizes."
            ),
            "",
            "| Recharge | Effect | Discharge negative / positive | System "
            "cost negative / positive | Incremental benefit negative / "
            "positive |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for (
        specification,
        effect,
    ), group in contrasts.groupby(
        ["recharge_specification", "effect_type"],
        sort=True,
    ):
        def counts(column: str) -> str:
            negative = int((group[column] < -1e-8).sum())
            positive = int((group[column] > 1e-8).sum())
            return f"{negative} / {positive}"

        lines.append(
            "| "
            + " | ".join(
                [
                    str(specification),
                    str(effect),
                    counts(metrics["discharge"]),
                    counts(metrics["cost"]),
                    counts(metrics["benefit"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            (
                "Units: energy quantities are annual MWh per mean-task MW "
                "(equivalent h yr⁻¹); system cost is CNY per mean-task "
                "kW-year; incremental benefit is CNY per forced-storage "
                "kW-year; EFC is cycles per year."
            ),
            "",
            "## Interpretation boundary",
            "",
            (
                "This factorial separates clustering of the frozen gas task "
                "from clustering of modeled recharge opportunities. It does "
                "not convert R2 or R3 into observed hourly curtailment, does "
                "not test the full duration spectrum, and does not by itself "
                "establish a complete mid-duration valley."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    all_results, audits, failures = load_runs()
    if all_results.empty:
        raise RuntimeError("No factorial results found")
    forced = all_results[
        all_results["variant"].eq("forced_1pct_joint")
    ].copy()
    forced, metric_columns = normalize(forced)

    expected_runs = len(COUNTRIES) * len(CONFIGS)
    expected_forced = (
        expected_runs
        * len(SPECIFICATIONS)
        * len(LEVELS)
        * len(LEVELS)
        * len(PHASES)
    )
    expected_key = [
        "country",
        "config_id",
        "recharge_specification",
        "task_chronology",
        "recharge_chronology",
        "recharge_phase_fraction",
    ]
    if len(audits) != expected_runs:
        failures.append(
            f"expected {expected_runs} runs; found {len(audits)}"
        )
    if len(forced) != expected_forced:
        failures.append(
            f"expected {expected_forced} forced cells; found {len(forced)}"
        )
    if forced.duplicated(expected_key).any():
        failures.append("duplicate factorial cells")

    effects = factorial_effects(forced, metric_columns)
    technology_phase_mean = phase_means(
        effects,
        metric_columns,
        "technology_factorial_effect_phase_mean",
    )
    contrasts = technology_contrasts(
        effects,
        metric_columns,
        "caes_minus_lfp_factorial_effect_phase",
    )
    contrast_phase_mean = phase_means(
        contrasts,
        metric_columns,
        "caes_minus_lfp_factorial_effect_phase_mean",
    )
    cells = add_cell_rows(forced, metric_columns)
    combined = pd.concat(
        [
            cells,
            effects,
            technology_phase_mean,
            contrasts,
            contrast_phase_mean,
        ],
        ignore_index=True,
        sort=False,
    )
    combined = combined.sort_values(
        [
            "row_type",
            "country",
            "recharge_specification",
            "config_id",
            "effect_type",
            "recharge_phase_fraction",
            "task_chronology",
            "recharge_chronology",
        ],
        kind="stable",
    )
    combined.to_csv(OUTPUT_CSV, index=False)

    maximum_absolute = float(
        all_results[list(BALANCE_FIELDS)].abs().to_numpy().max()
    )
    maximum_relative = float(
        all_results[list(RELATIVE_BALANCE_FIELDS)]
        .abs()
        .to_numpy()
        .max()
    )
    source_hashes = {
        field: sorted(
            {
                str(audit[field])
                for audit in audits
                if audit.get(field) is not None
            }
        )
        for field in (
            "panel_sha256",
            "acer_source_sha256",
            "configuration_source_sha256",
            "runner_sha256",
            "core_runner_sha256",
        )
    }
    checks = {
        "all_8_runs_present": len(audits) == expected_runs,
        "all_run_QA_pass": all(
            audit["qa_status"] == "pass"
            and not audit["failed_checks"]
            for audit in audits
        ),
        "all_runs_non_evidential": all(
            audit["manuscript_evidence_admitted"] is False
            for audit in audits
        ),
        "each_run_has_32_forced_and_4_baseline_rows": all(
            audit["result_rows"] == 36
            and audit["forced_rows"] == 32
            and audit["baseline_rows"] == 4
            and audit["conservation_rows"] == 32
            for audit in audits
        ),
        "all_256_forced_factorial_cells_present": len(forced)
        == expected_forced,
        "all_factorial_cell_keys_unique": not bool(
            forced.duplicated(expected_key).any()
        ),
        "all_models_optimal": bool(
            all_results["solver_status"].eq(0).all()
        ),
        "all_countries_present": set(forced["country"])
        == set(COUNTRIES),
        "both_configurations_present": set(forced["config_id"])
        == set(CONFIGS),
        "both_recharge_specifications_present": set(
            forced["recharge_specification"]
        )
        == set(SPECIFICATIONS),
        "both_task_levels_present": set(forced["task_chronology"])
        == set(LEVELS),
        "both_recharge_levels_present": set(
            forced["recharge_chronology"]
        )
        == set(LEVELS),
        "all_phase_offsets_present": set(
            forced["recharge_phase_fraction"]
        )
        == set(PHASES),
        "zero_recharge_opportunity_price_only": set(
            forced["recharge_opportunity_price_cny_mwh"]
        )
        == {0.0},
        "technology_effect_phase_rows_complete": len(effects)
        == (
            len(COUNTRIES)
            * len(CONFIGS)
            * len(SPECIFICATIONS)
            * len(PHASES)
            * len(EFFECTS)
        ),
        "technology_effect_phase_mean_rows_complete": len(
            technology_phase_mean
        )
        == (
            len(COUNTRIES)
            * len(CONFIGS)
            * len(SPECIFICATIONS)
            * len(EFFECTS)
        ),
        "caes_lfp_effect_phase_rows_complete": len(contrasts)
        == (
            len(COUNTRIES)
            * len(SPECIFICATIONS)
            * len(PHASES)
            * len(EFFECTS)
        ),
        "caes_lfp_effect_phase_mean_rows_complete": len(
            contrast_phase_mean
        )
        == (
            len(COUNTRIES) * len(SPECIFICATIONS) * len(EFFECTS)
        ),
        "all_absolute_accounting_balances_pass": maximum_absolute <= 1e-6,
        "all_relative_accounting_balances_pass": maximum_relative <= 1e-10,
        "all_source_hashes_frozen_within_suite": all(
            len(values) == 1 for values in source_hashes.values()
        ),
        "r2_not_labeled_observed_curtailment": bool(
            forced.loc[
                forced["recharge_specification"].eq("R2"),
                "recharge_interpretation",
            ]
            .str.contains("not observed curtailment", regex=False)
            .all()
        ),
        "r3_not_labeled_observed_hourly_curtailment": bool(
            forced.loc[
                forced["recharge_specification"].eq("R3"),
                "recharge_interpretation",
            ]
            .str.contains(
                "not observed hourly curtailment",
                regex=False,
            )
            .all()
        ),
    }
    if not all(checks.values()):
        failures.extend(
            f"failed aggregate check: {name}"
            for name, value in checks.items()
            if not value
        )
    qa = {
        "status": "pass"
        if all(checks.values()) and not failures
        else "fail",
        "checks": checks,
        "expected_run_count": expected_runs,
        "observed_run_count": len(audits),
        "all_result_rows": int(len(all_results)),
        "forced_factorial_rows": int(len(forced)),
        "technology_effect_phase_rows": int(len(effects)),
        "technology_effect_phase_mean_rows": int(
            len(technology_phase_mean)
        ),
        "caes_lfp_effect_phase_rows": int(len(contrasts)),
        "caes_lfp_effect_phase_mean_rows": int(
            len(contrast_phase_mean)
        ),
        "maximum_absolute_accounting_error_mwh": maximum_absolute,
        "maximum_relative_accounting_error": maximum_relative,
        "task_recharge_rank_correlation_range": [
            min(
                audit["minimum_task_recharge_rank_correlation"]
                for audit in audits
            ),
            max(
                audit["maximum_task_recharge_rank_correlation"]
                for audit in audits
            ),
        ],
        "high_task_high_recharge_overlap_days_range": [
            min(audit["minimum_high_day_overlap"] for audit in audits),
            max(audit["maximum_high_day_overlap"] for audit in audits),
        ],
        "source_hashes": source_hashes,
        "failures": failures,
        "runs": audits,
        "task_main_effect_formula": (
            "(Y_clustered,dispersed + Y_clustered,clustered - "
            "Y_dispersed,dispersed - Y_dispersed,clustered) / 2"
        ),
        "recharge_main_effect_formula": (
            "(Y_dispersed,clustered + Y_clustered,clustered - "
            "Y_dispersed,dispersed - Y_clustered,dispersed) / 2"
        ),
        "interaction_formula": (
            "Y_clustered,clustered - Y_clustered,dispersed - "
            "Y_dispersed,clustered + Y_dispersed,dispersed"
        ),
        "caes_lfp_four_cell_contrast": (
            "interaction_CA​ES24 - interaction_LFP4"
        ).replace("\u200b", ""),
        "R2_interpretation": (
            "potential-VRE-surplus counterfactual; not observed curtailment"
        ),
        "R3_interpretation": (
            "ACER annual-volume-calibrated counterfactual allocation; not "
            "observed hourly curtailment"
        ),
        "manuscript_evidence_admitted": False,
        "reason": (
            "four-country two-configuration factorial mechanism diagnostic; "
            "not full-spectrum or confirmatory inference"
        ),
        "summary_csv": str(OUTPUT_CSV),
        "summary_csv_sha256": sha256(OUTPUT_CSV),
    }
    OUTPUT_QA.write_text(
        json.dumps(qa, indent=2),
        encoding="utf-8",
    )
    OUTPUT_MD.write_text(
        build_markdown(
            technology_phase_mean,
            contrasts,
            contrast_phase_mean,
            qa,
        ),
        encoding="utf-8",
    )
    if qa["status"] != "pass":
        raise RuntimeError(json.dumps(qa, indent=2))
    print(
        json.dumps(
            {
                "status": qa["status"],
                "runs": qa["observed_run_count"],
                "forced_factorial_rows": qa[
                    "forced_factorial_rows"
                ],
                "technology_effect_phase_rows": qa[
                    "technology_effect_phase_rows"
                ],
                "caes_lfp_effect_phase_rows": qa[
                    "caes_lfp_effect_phase_rows"
                ],
                "maximum_absolute_accounting_error_mwh": (
                    maximum_absolute
                ),
                "output_csv": str(OUTPUT_CSV),
                "output_qa": str(OUTPUT_QA),
                "output_md": str(OUTPUT_MD),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
