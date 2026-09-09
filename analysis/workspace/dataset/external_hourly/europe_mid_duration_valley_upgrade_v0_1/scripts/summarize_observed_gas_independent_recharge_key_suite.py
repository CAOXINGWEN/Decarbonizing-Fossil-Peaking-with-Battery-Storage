#!/usr/bin/env python3
"""Summarize the prespecified observed-gas independent-recharge key suite.

The suite is deliberately small and diagnostic:

* DE, ES, HR and IT in 2023;
* LFP 4 h and CAES 24 h;
* co-reordered and task-only/fixed-natural recharge pairing;
* R0 at 0 CNY MWh-1 and R2/R3 at 0 and 100 CNY MWh-1.

R2 is a potential-VRE-surplus counterfactual.  R3 allocates an observed ACER
annual curtailment volume over the R2 shape; it is not observed hourly
curtailment.  The output is therefore a mechanism diagnostic and is never
admitted as manuscript evidence by this script.
"""

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
    / "observed_gas_independent_recharge_key_suite_v0_1"
)
ARTIFACT_ROOT = ROOT / "artifacts" / "research"
OUTPUT_CSV = (
    ARTIFACT_ROOT / "OBSERVED_GAS_INDEPENDENT_RECHARGE_KEY_SUITE.csv"
)
OUTPUT_QA = (
    ARTIFACT_ROOT / "OBSERVED_GAS_INDEPENDENT_RECHARGE_KEY_SUITE_QA.json"
)
OUTPUT_MD = (
    ARTIFACT_ROOT / "OBSERVED_GAS_INDEPENDENT_RECHARGE_KEY_SUITE.md"
)

COUNTRIES = ("DE", "ES", "HR", "IT")
CONFIGS = ("li_lfp_4h", "caes_24h")
PAIRINGS = ("co_reordered", "task_only_fixed_natural")
SCENARIOS = (
    ("R0", 0.0),
    ("R2", 0.0),
    ("R2", 100.0),
    ("R3", 0.0),
    ("R3", 100.0),
)
CHRONOLOGIES = ("natural", "dispersed", "clustered")

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


def expected_directory(country: str, config: str, pairing: str) -> Path:
    return RUN_ROOT / f"{country}_{config}_{pairing}"


def load_suite() -> tuple[pd.DataFrame, list[dict[str, Any]], list[str]]:
    frames: list[pd.DataFrame] = []
    run_audits: list[dict[str, Any]] = []
    failures: list[str] = []
    for country in COUNTRIES:
        for config in CONFIGS:
            for pairing in PAIRINGS:
                directory = expected_directory(country, config, pairing)
                result_path = directory / "model_results.csv"
                qa_path = directory / "QA.json"
                manifest_path = directory / "run_manifest.json"
                required = (result_path, qa_path, manifest_path)
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
                        f"failed_checks={failed_checks}"
                    )
                if manifest.get("manuscript_evidence_admitted") is not False:
                    failures.append(
                        f"{directory.name}: manifest evidence admission "
                        "must be false"
                    )
                frame = pd.read_csv(result_path)
                frame["run_directory"] = str(directory)
                frames.append(frame)
                run_audits.append(
                    {
                        "country": country,
                        "config_id": config,
                        "pairing_mode": pairing,
                        "directory": str(directory),
                        "qa_status": qa.get("status"),
                        "failed_checks": failed_checks,
                        "model_results_rows": int(len(frame)),
                        "baseline_result_rows": int(
                            frame["variant"].eq("baseline").sum()
                        ),
                        "forced_result_rows": int(
                            frame["variant"].eq(
                                "forced_1pct_joint"
                            ).sum()
                        ),
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
                        "manuscript_evidence_admitted": manifest.get(
                            "manuscript_evidence_admitted"
                        ),
                    }
                )
    if not frames:
        return pd.DataFrame(), run_audits, failures
    return pd.concat(frames, ignore_index=True), run_audits, failures


def add_normalized_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    denominator = output["mean_observed_gas_task_mw"]
    for metric in NORMALIZED_ENERGY_METRICS:
        output[f"{metric}_per_mean_task_mw"] = output[metric] / denominator
    output["total_system_cost_cny_per_mean_task_kw_year"] = (
        output["total_system_cost_cny_per_mean_mw_year"] / 1000.0
    )
    return output


def paired_deltas(forced: pd.DataFrame) -> pd.DataFrame:
    index = [
        "country",
        "year",
        "pairing_mode",
        "recharge_specification",
        "recharge_opportunity_price_cny_mwh",
        "config_id",
    ]
    metric_columns = list(RAW_METRICS)
    metric_columns.extend(
        f"{metric}_per_mean_task_mw"
        for metric in NORMALIZED_ENERGY_METRICS
    )
    metric_columns.append(
        "total_system_cost_cny_per_mean_task_kw_year"
    )
    subset = forced[
        forced["chronology"].isin(["dispersed", "clustered"])
    ].copy()
    wide = subset.pivot(
        index=index,
        columns="chronology",
        values=metric_columns,
    )
    rows: list[dict[str, Any]] = []
    for key, values in wide.iterrows():
        row = dict(zip(index, key, strict=True))
        row["row_type"] = "technology_delta"
        for metric in metric_columns:
            dispersed = float(values[(metric, "dispersed")])
            clustered = float(values[(metric, "clustered")])
            row[f"{metric}_dispersed"] = dispersed
            row[f"{metric}_clustered"] = clustered
            row[f"{metric}_clustered_minus_dispersed"] = (
                clustered - dispersed
            )
        rows.append(row)
    return pd.DataFrame(rows)


def add_did_rows(delta: pd.DataFrame) -> pd.DataFrame:
    id_columns = [
        "country",
        "year",
        "pairing_mode",
        "recharge_specification",
        "recharge_opportunity_price_cny_mwh",
    ]
    delta_columns = [
        column
        for column in delta.columns
        if column.endswith("_clustered_minus_dispersed")
    ]
    rows: list[dict[str, Any]] = []
    for key, group in delta.groupby(id_columns, sort=True):
        by_config = group.set_index("config_id")
        if set(by_config.index) != set(CONFIGS):
            continue
        row = dict(zip(id_columns, key, strict=True))
        row["config_id"] = "caes_24h_minus_li_lfp_4h"
        row["row_type"] = "caes24_minus_lfp4_did"
        for column in delta_columns:
            row[column] = float(
                by_config.loc["caes_24h", column]
                - by_config.loc["li_lfp_4h", column]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def add_mean_did_rows(did: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "year",
        "pairing_mode",
        "recharge_specification",
        "recharge_opportunity_price_cny_mwh",
        "config_id",
    ]
    metric_columns = [
        column
        for column in did.columns
        if column.endswith("_clustered_minus_dispersed")
    ]
    mean = (
        did.groupby(group_columns, as_index=False)[metric_columns]
        .mean()
        .copy()
    )
    mean.insert(0, "country", "MEAN_4_COUNTRIES")
    mean["row_type"] = "caes24_minus_lfp4_did_country_mean"
    return mean


def add_pairing_contrast_rows(did: pd.DataFrame) -> pd.DataFrame:
    """Compute co-reordered minus task-only CAES-LFP DID."""
    id_columns = [
        "country",
        "year",
        "recharge_specification",
        "recharge_opportunity_price_cny_mwh",
        "config_id",
    ]
    metric_columns = [
        column
        for column in did.columns
        if column.endswith("_clustered_minus_dispersed")
    ]
    rows: list[dict[str, Any]] = []
    for key, group in did.groupby(id_columns, sort=True):
        by_pairing = group.set_index("pairing_mode")
        if set(by_pairing.index) != set(PAIRINGS):
            continue
        row = dict(zip(id_columns, key, strict=True))
        row["pairing_mode"] = (
            "co_reordered_minus_task_only_fixed_natural"
        )
        row["row_type"] = "pairing_contrast_of_caes_lfp_did"
        for column in metric_columns:
            row[column] = float(
                by_pairing.loc["co_reordered", column]
                - by_pairing.loc["task_only_fixed_natural", column]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def add_mean_pairing_contrast_rows(
    contrast: pd.DataFrame,
) -> pd.DataFrame:
    group_columns = [
        "year",
        "pairing_mode",
        "recharge_specification",
        "recharge_opportunity_price_cny_mwh",
        "config_id",
    ]
    metric_columns = [
        column
        for column in contrast.columns
        if column.endswith("_clustered_minus_dispersed")
    ]
    mean = (
        contrast.groupby(group_columns, as_index=False)[metric_columns]
        .mean()
        .copy()
    )
    mean.insert(0, "country", "MEAN_4_COUNTRIES")
    mean["row_type"] = "pairing_contrast_country_mean"
    return mean


def fmt(value: float, digits: int = 2) -> str:
    if not np.isfinite(value):
        return "NA"
    threshold = 0.5 * 10 ** (-digits)
    if abs(value) < threshold:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def summarize_metric(
    did: pd.DataFrame, group: pd.DataFrame, column: str, digits: int = 2
) -> str:
    value = float(group[column].mean())
    minimum = float(group[column].min())
    maximum = float(group[column].max())
    return (
        f"{fmt(value, digits)} "
        f"[{fmt(minimum, digits)}, {fmt(maximum, digits)}]"
    )


def build_markdown(
    delta: pd.DataFrame,
    did: pd.DataFrame,
    contrast: pd.DataFrame,
    qa: dict[str, Any],
) -> str:
    did_metric_columns = {
        "charge": (
            "storage_charge_input_mwh_per_mean_task_mw_"
            "clustered_minus_dispersed"
        ),
        "discharge": (
            "storage_discharge_output_mwh_per_mean_task_mw_"
            "clustered_minus_dispersed"
        ),
        "gas_charge": (
            "gas_supplied_charge_mwh_per_mean_task_mw_"
            "clustered_minus_dispersed"
        ),
        "recharge_used": (
            "recharge_used_mwh_per_mean_task_mw_"
            "clustered_minus_dispersed"
        ),
        "unserved": (
            "unserved_task_mwh_per_mean_task_mw_"
            "clustered_minus_dispersed"
        ),
        "cost": (
            "total_system_cost_cny_per_mean_task_kw_year_"
            "clustered_minus_dispersed"
        ),
        "benefit": (
            "signed_benefit_cny_per_forced_kw_year_"
            "clustered_minus_dispersed"
        ),
        "efc": (
            "equivalent_full_cycles_output_basis_"
            "clustered_minus_dispersed"
        ),
    }
    lines = [
        "# Observed-gas independent-recharge key suite",
        "",
        "## Status and scope",
        "",
        (
            "This prespecified diagnostic uses the exact non-negative hourly "
            "aggregate `Fossil gas` series as the frozen task. Wind and solar "
            "generation are never subtracted from that task. The suite covers "
            "DE, ES, HR and IT in 2023, LFP 4 h and CAES 24 h, and two "
            "recharge-pairing interventions."
        ),
        "",
        (
            "**Evidence status:** diagnostic only. Every run and this aggregate "
            "remain `manuscript_evidence_admitted=false`."
        ),
        "",
        "Recharge specifications:",
        "",
        "- **R0** has no exogenous recharge. All storage charging is supplied "
        "by additional model gas and is explicitly costed.",
        "- **R2** is a synchronous potential-VRE-surplus counterfactual "
        "`max(alpha*(wind+solar)-load, 0)`. It is not observed curtailment.",
        "- **R3** allocates an ACER observed annual curtailment volume over the "
        "R2 hourly shape. It is an annual-volume-calibrated counterfactual "
        "allocation, not observed hourly curtailment.",
        "",
        (
            "`co_reordered` moves the task and recharge profile together. "
            "`task_only_fixed_natural` reorders only the task and holds the "
            "recharge profile in its natural calendar order."
        ),
        "",
        "## QA",
        "",
        f"- Expected runs found: {qa['observed_run_count']} / "
        f"{qa['expected_run_count']}.",
        f"- Forced paired comparisons: {qa['forced_result_rows']} rows.",
        f"- Aggregate QA status: **{qa['status']}**.",
        f"- Maximum absolute accounting residual: "
        f"{qa['maximum_absolute_accounting_error_mwh']:.3e} MWh.",
        f"- Maximum relative accounting residual: "
        f"{qa['maximum_relative_accounting_error']:.3e}.",
        "",
        "## CAES 24 h minus LFP 4 h chronology response",
        "",
        (
            "Each cell is the four-country mean [minimum, maximum] of the "
            "difference-in-differences "
            "`(CAES clustered - dispersed) - (LFP clustered - dispersed)`. "
            "Energy quantities are normalized by each country's mean hourly "
            "gas task. A negative discharge or EFC value means clustering "
            "reduces CAES throughput more than LFP. A positive system-cost "
            "value means clustering raises CAES cost relative to LFP."
        ),
        "",
        "| Pairing | Recharge | Price (CNY MWh⁻¹) | Charge throughput | Discharge "
        "throughput | Gas-to-charge | Recharge used | Unserved | System cost "
        "| Incremental benefit | EFC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (
        pairing,
        specification,
        price,
    ), group in did.groupby(
        [
            "pairing_mode",
            "recharge_specification",
            "recharge_opportunity_price_cny_mwh",
        ],
        sort=True,
    ):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(pairing),
                    str(specification),
                    f"{float(price):.0f}",
                    summarize_metric(
                        did, group, did_metric_columns["charge"]
                    ),
                    summarize_metric(
                        did, group, did_metric_columns["discharge"]
                    ),
                    summarize_metric(
                        did, group, did_metric_columns["gas_charge"]
                    ),
                    summarize_metric(
                        did, group, did_metric_columns["recharge_used"]
                    ),
                    summarize_metric(
                        did, group, did_metric_columns["unserved"]
                    ),
                    summarize_metric(
                        did, group, did_metric_columns["cost"]
                    ),
                    summarize_metric(
                        did, group, did_metric_columns["benefit"]
                    ),
                    summarize_metric(
                        did, group, did_metric_columns["efc"]
                    ),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            (
                "Units: charge, discharge, gas-to-charge, recharge-used and "
                "unserved are annual MWh per mean-task MW "
                "(equivalent h yr⁻¹); system cost is CNY "
                "per mean-task kW-year; incremental benefit is CNY per forced "
                "storage kW-year; EFC is cycles per year."
            ),
            "",
            "## Recharge-pairing contrast",
            "",
            (
                "This contrast subtracts the task-only/fixed-natural "
                "CAES-LFP difference-in-differences from its co-reordered "
                "counterpart. Negative throughput and benefit values therefore "
                "show the additional CAES-relative loss when recharge timing "
                "is moved with the clustered task. Values are four-country "
                "means [minimum, maximum]."
            ),
            "",
            "| Recharge | Price (CNY MWh⁻¹) | Charge throughput | Discharge throughput | "
            "Gas-to-charge | Recharge used | Unserved | System cost | "
            "Incremental benefit | EFC |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for (
        specification,
        price,
    ), group in contrast.groupby(
        [
            "recharge_specification",
            "recharge_opportunity_price_cny_mwh",
        ],
        sort=True,
    ):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(specification),
                    f"{float(price):.0f}",
                    summarize_metric(
                        contrast, group, did_metric_columns["charge"]
                    ),
                    summarize_metric(
                        contrast, group, did_metric_columns["discharge"]
                    ),
                    summarize_metric(
                        contrast, group, did_metric_columns["gas_charge"]
                    ),
                    summarize_metric(
                        contrast,
                        group,
                        did_metric_columns["recharge_used"],
                    ),
                    summarize_metric(
                        contrast, group, did_metric_columns["unserved"]
                    ),
                    summarize_metric(
                        contrast, group, did_metric_columns["cost"]
                    ),
                    summarize_metric(
                        contrast, group, did_metric_columns["benefit"]
                    ),
                    summarize_metric(
                        contrast, group, did_metric_columns["efc"]
                    ),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Technology-specific clustered-minus-dispersed responses",
            "",
            (
                "The table below reports four-country means. Country-level "
                "values, dispersed and clustered endpoints, all absolute "
                "metrics and all difference-in-differences are retained in "
                "`OBSERVED_GAS_INDEPENDENT_RECHARGE_KEY_SUITE.csv`."
            ),
            "",
            "| Pairing | Recharge | Price (CNY MWh⁻¹) | Technology | Charge throughput | "
            "Discharge throughput | Gas-to-charge | Recharge used | Unserved "
            "| System cost | Incremental benefit | EFC |",
            "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    tech_metric_columns = {
        key: value for key, value in did_metric_columns.items()
    }
    for (
        pairing,
        specification,
        price,
        config,
    ), group in delta.groupby(
        [
            "pairing_mode",
            "recharge_specification",
            "recharge_opportunity_price_cny_mwh",
            "config_id",
        ],
        sort=True,
    ):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(pairing),
                    str(specification),
                    f"{float(price):.0f}",
                    str(config),
                    fmt(float(group[tech_metric_columns["charge"]].mean())),
                    fmt(
                        float(
                            group[tech_metric_columns["discharge"]].mean()
                        )
                    ),
                    fmt(
                        float(
                            group[tech_metric_columns["gas_charge"]].mean()
                        )
                    ),
                    fmt(
                        float(
                            group[
                                tech_metric_columns["recharge_used"]
                            ].mean()
                        )
                    ),
                    fmt(
                        float(
                            group[tech_metric_columns["unserved"]].mean()
                        )
                    ),
                    fmt(float(group[tech_metric_columns["cost"]].mean())),
                    fmt(float(group[tech_metric_columns["benefit"]].mean())),
                    fmt(float(group[tech_metric_columns["efc"]].mean())),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            (
                "This suite tests whether the chronology response changes when "
                "recharge is moved with the task or held on the natural "
                "calendar. It does not identify observed hourly curtailment, "
                "does not constitute a full duration-spectrum analysis, and "
                "does not by itself establish the mid-duration valley. Any "
                "missing or failed run is reported in the QA JSON and blocks a "
                "pass status."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    all_results, run_audits, failures = load_suite()
    if all_results.empty:
        raise RuntimeError("No suite results found")

    forced = all_results[
        all_results["variant"].eq("forced_1pct_joint")
    ].copy()
    forced = add_normalized_metrics(forced)
    expected_scenarios = set(SCENARIOS)
    observed_scenarios = set(
        forced[
            [
                "recharge_specification",
                "recharge_opportunity_price_cny_mwh",
            ]
        ]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if observed_scenarios != expected_scenarios:
        failures.append(
            f"scenario mismatch: observed={sorted(observed_scenarios)}"
        )
    expected_runs = len(COUNTRIES) * len(CONFIGS) * len(PAIRINGS)
    expected_forced_rows = (
        expected_runs * len(CHRONOLOGIES) * len(SCENARIOS)
    )
    if len(run_audits) != expected_runs:
        failures.append(
            f"expected {expected_runs} runs; found {len(run_audits)}"
        )
    if len(forced) != expected_forced_rows:
        failures.append(
            f"expected {expected_forced_rows} forced rows; found "
            f"{len(forced)}"
        )
    forced_key = [
        "country",
        "year",
        "pairing_mode",
        "config_id",
        "chronology",
        "recharge_specification",
        "recharge_opportunity_price_cny_mwh",
    ]
    if forced.duplicated(forced_key).any():
        failures.append("duplicate forced result keys")
    if not forced["solver_status"].eq(0).all():
        failures.append("one or more forced models are non-optimal")
    if set(forced["country"]) != set(COUNTRIES):
        failures.append("country set mismatch")
    if set(forced["config_id"]) != set(CONFIGS):
        failures.append("configuration set mismatch")
    if set(forced["pairing_mode"]) != set(PAIRINGS):
        failures.append("pairing set mismatch")
    if set(forced["chronology"]) != set(CHRONOLOGIES):
        failures.append("chronology set mismatch")
    if not all(
        forced.loc[
            forced["recharge_specification"].eq("R3"),
            "recharge_interpretation",
        ]
        .str.contains("not observed hourly curtailment", regex=False)
        .all()
        for _ in [0]
    ):
        failures.append("R3 interpretation boundary missing")
    if not forced.loc[
        forced["recharge_specification"].eq("R2"),
        "recharge_interpretation",
    ].str.contains("not observed curtailment", regex=False).all():
        failures.append("R2 interpretation boundary missing")

    delta = paired_deltas(forced)
    did = add_did_rows(delta)
    mean_did = add_mean_did_rows(did)
    contrast = add_pairing_contrast_rows(did)
    mean_contrast = add_mean_pairing_contrast_rows(contrast)
    combined = pd.concat(
        [delta, did, mean_did, contrast, mean_contrast],
        ignore_index=True,
        sort=False,
    )
    combined = combined.sort_values(
        [
            "row_type",
            "country",
            "pairing_mode",
            "recharge_specification",
            "recharge_opportunity_price_cny_mwh",
            "config_id",
        ],
        kind="stable",
    )
    combined.to_csv(OUTPUT_CSV, index=False)

    max_absolute = float(
        all_results[list(BALANCE_FIELDS)].abs().to_numpy().max()
    )
    max_relative = float(
        all_results[list(RELATIVE_BALANCE_FIELDS)]
        .abs()
        .to_numpy()
        .max()
    )
    source_hashes = {
        field: sorted(
            {
                str(run[field])
                for run in run_audits
                if run.get(field) is not None
            }
        )
        for field in (
            "panel_sha256",
            "acer_source_sha256",
            "configuration_source_sha256",
            "runner_sha256",
        )
    }
    for field, values in source_hashes.items():
        if len(values) != 1:
            failures.append(
                f"{field} must be identical across runs; found {values}"
            )
    checks = {
        "all_16_prespecified_runs_present": len(run_audits)
        == expected_runs,
        "all_run_QA_pass": all(
            run["qa_status"] == "pass" and not run["failed_checks"]
            for run in run_audits
        ),
        "all_runs_not_admitted_as_manuscript_evidence": all(
            run["manuscript_evidence_admitted"] is False
            for run in run_audits
        ),
        "all_240_forced_rows_present": len(forced)
        == expected_forced_rows,
        "all_384_model_rows_present": len(all_results)
        == expected_runs * 24,
        "each_run_has_9_baseline_and_15_forced_rows": all(
            run["model_results_rows"] == 24
            and run["baseline_result_rows"] == 9
            and run["forced_result_rows"] == 15
            for run in run_audits
        ),
        "all_forced_models_optimal": bool(
            forced["solver_status"].eq(0).all()
        ),
        "all_forced_keys_unique": not bool(
            forced.duplicated(forced_key).any()
        ),
        "r0_price_deduplicated_to_zero": set(
            forced.loc[
                forced["recharge_specification"].eq("R0"),
                "recharge_opportunity_price_cny_mwh",
            ]
        )
        == {0.0},
        "r2_and_r3_prices_are_zero_and_100": all(
            set(
                forced.loc[
                    forced["recharge_specification"].eq(specification),
                    "recharge_opportunity_price_cny_mwh",
                ]
            )
            == {0.0, 100.0}
            for specification in ("R2", "R3")
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
                "not observed hourly curtailment", regex=False
            )
            .all()
        ),
        "all_absolute_accounting_balances_pass": max_absolute <= 1e-6,
        "all_relative_accounting_balances_pass": max_relative <= 1e-10,
        "technology_delta_rows_complete": len(delta)
        == len(COUNTRIES) * len(CONFIGS) * len(PAIRINGS) * len(SCENARIOS),
        "did_rows_complete": len(did)
        == len(COUNTRIES) * len(PAIRINGS) * len(SCENARIOS),
        "pairing_contrast_rows_complete": len(contrast)
        == len(COUNTRIES) * len(SCENARIOS),
        "all_source_hashes_frozen_within_suite": all(
            len(values) == 1 for values in source_hashes.values()
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
        "observed_run_count": len(run_audits),
        "all_model_result_rows": int(len(all_results)),
        "forced_result_rows": int(len(forced)),
        "technology_delta_rows": int(len(delta)),
        "caes24_minus_lfp4_did_rows": int(len(did)),
        "country_mean_did_rows": int(len(mean_did)),
        "pairing_contrast_rows": int(len(contrast)),
        "country_mean_pairing_contrast_rows": int(
            len(mean_contrast)
        ),
        "maximum_absolute_accounting_error_mwh": max_absolute,
        "maximum_relative_accounting_error": max_relative,
        "source_hashes": source_hashes,
        "failures": failures,
        "runs": run_audits,
        "task_definition": (
            "exact non-negative hourly aggregate Fossil gas; VRE is never "
            "subtracted from the task"
        ),
        "R2_interpretation": (
            "potential-VRE-surplus counterfactual; not observed curtailment"
        ),
        "R3_interpretation": (
            "ACER annual-volume-calibrated counterfactual hourly allocation; "
            "not observed hourly curtailment"
        ),
        "manuscript_evidence_admitted": False,
        "reason": (
            "prespecified four-country key-configuration mechanism "
            "diagnostic; not a full spectrum or confirmatory inference"
        ),
        "summary_csv": str(OUTPUT_CSV),
        "summary_csv_sha256": sha256(OUTPUT_CSV),
    }
    OUTPUT_QA.write_text(
        json.dumps(qa, indent=2), encoding="utf-8"
    )
    OUTPUT_MD.write_text(
        build_markdown(delta, did, contrast, qa),
        encoding="utf-8",
    )
    if qa["status"] != "pass":
        raise RuntimeError(json.dumps(qa, indent=2))
    print(
        json.dumps(
            {
                "status": qa["status"],
                "runs": qa["observed_run_count"],
                "forced_rows": qa["forced_result_rows"],
                "technology_delta_rows": qa[
                    "technology_delta_rows"
                ],
                "did_rows": qa["caes24_minus_lfp4_did_rows"],
                "maximum_absolute_accounting_error_mwh": max_absolute,
                "maximum_relative_accounting_error": max_relative,
                "output_csv": str(OUTPUT_CSV),
                "output_qa": str(OUTPUT_QA),
                "output_md": str(OUTPUT_MD),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
