#!/usr/bin/env python3
"""Audit and summarize the complete 132-country-year R2 factorial panel."""

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
    / "observed_gas_R2_factorial_full132_v0_1"
)
PANEL = (
    ROOT
    / "processed"
    / "energy_charts_observed_gas_task_panel_2015_2024.csv.gz"
)
ARTIFACT_ROOT = ROOT / "artifacts" / "research"
OUTPUT_STEM = ARTIFACT_ROOT / "OBSERVED_GAS_R2_FACTORIAL_FULL132"
OUTPUT_EFFECTS = ARTIFACT_ROOT / (
    "OBSERVED_GAS_R2_FACTORIAL_FULL132_EFFECTS.csv.gz"
)
OUTPUT_COUNTRY_YEAR = ARTIFACT_ROOT / (
    "OBSERVED_GAS_R2_FACTORIAL_FULL132_COUNTRY_YEAR_EFFECTS.csv"
)
OUTPUT_DISTRIBUTIONS = ARTIFACT_ROOT / (
    "OBSERVED_GAS_R2_FACTORIAL_FULL132_DISTRIBUTIONS.csv"
)
OUTPUT_QA = ARTIFACT_ROOT / (
    "OBSERVED_GAS_R2_FACTORIAL_FULL132_QA.json"
)
OUTPUT_MD = OUTPUT_STEM.with_suffix(".md")

CONFIGS = ("li_lfp_4h", "caes_24h")
LEVELS = ("dispersed", "clustered")
PHASES = (0.0, 0.25, 0.5, 0.75)
EFFECTS = ("task_main", "recharge_main", "interaction")
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
NORMALIZED_ENERGY_METRICS = (
    "storage_charge_input_mwh",
    "storage_discharge_output_mwh",
    "gas_supplied_charge_mwh",
    "recharge_used_mwh",
    "unserved_task_mwh",
)
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def expected_country_years() -> list[tuple[str, int]]:
    frame = pd.read_csv(
        PANEL,
        usecols=[
            "country",
            "year",
            "gas_task_eligible",
            "gas_task_material_share_ge_1pct",
        ],
    )
    frame = frame[
        as_bool(frame["gas_task_eligible"])
        & as_bool(frame["gas_task_material_share_ge_1pct"])
    ][["country", "year"]].drop_duplicates()
    pairs = sorted(
        {
            (str(row.country), int(row.year))
            for row in frame.itertuples(index=False)
        }
    )
    if len(pairs) != 132:
        raise RuntimeError(
            f"Frozen complete sample must contain 132 country-years; "
            f"found {len(pairs)}"
        )
    return pairs


def load_runs(
    expected_pairs: list[tuple[str, int]],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[str]]:
    frames: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    failures: list[str] = []
    for country, year in expected_pairs:
        for config in CONFIGS:
            key = f"{country}_{year}_{config}"
            directory = RUN_ROOT / key
            paths = {
                "results": directory / "model_results.csv",
                "qa": directory / "QA.json",
                "manifest": directory / "run_manifest.json",
                "conservation": (
                    directory / "factorial_cell_conservation_qa.csv"
                ),
                "diagnostics": directory / "factor_order_diagnostics.csv",
            }
            missing = [
                name for name, path in paths.items() if not path.exists()
            ]
            if missing:
                failures.append(f"{key}: missing {', '.join(missing)}")
                continue
            try:
                qa = json.loads(
                    paths["qa"].read_text(encoding="utf-8")
                )
                manifest = json.loads(
                    paths["manifest"].read_text(encoding="utf-8")
                )
                result = pd.read_csv(paths["results"])
                conservation = pd.read_csv(paths["conservation"])
                diagnostics = pd.read_csv(paths["diagnostics"])
            except Exception as error:
                failures.append(f"{key}: read failure {error}")
                continue
            failed_checks = [
                name
                for name, value in qa.get("checks", {}).items()
                if not bool(value)
            ]
            if qa.get("status") != "pass" or failed_checks:
                failures.append(
                    f"{key}: QA={qa.get('status')}, failed={failed_checks}"
                )
            if manifest.get("manuscript_evidence_admitted") is not False:
                failures.append(f"{key}: manifest admission is not false")
            result["run_directory"] = str(directory)
            frames.append(result)
            audits.append(
                {
                    "key": key,
                    "country": country,
                    "year": year,
                    "config_id": config,
                    "qa_status": qa.get("status"),
                    "failed_checks": failed_checks,
                    "result_rows": int(len(result)),
                    "forced_rows": int(
                        result["variant"].eq("forced_1pct_joint").sum()
                    ),
                    "baseline_rows": int(
                        result["variant"].eq("baseline").sum()
                    ),
                    "conservation_rows": int(len(conservation)),
                    "diagnostic_rows": int(len(diagnostics)),
                    "result_sha256": sha256(paths["results"]),
                    "qa_sha256": sha256(paths["qa"]),
                    "manifest_sha256": sha256(paths["manifest"]),
                    "panel_sha256": manifest.get("panel_sha256"),
                    "configuration_source_sha256": manifest.get(
                        "configuration_source_sha256"
                    ),
                    "runner_sha256": manifest.get("script_sha256"),
                    "factorial_helper_sha256": manifest.get(
                        "key_factorial_helper_sha256"
                    ),
                    "core_runner_sha256": manifest.get(
                        "core_runner_sha256"
                    ),
                    "manuscript_evidence_admitted": manifest.get(
                        "manuscript_evidence_admitted"
                    ),
                }
            )
    if not frames:
        return pd.DataFrame(), audits, failures
    return pd.concat(frames, ignore_index=True), audits, failures


def normalize(
    forced: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    output = forced.copy()
    denominator = output["mean_observed_gas_task_mw"]
    normalized: list[str] = []
    for metric in NORMALIZED_ENERGY_METRICS:
        column = f"{metric}_per_mean_task_mw"
        output[column] = output[metric] / denominator
        normalized.append(column)
    cost_column = "total_system_cost_cny_per_mean_task_kw_year"
    output[cost_column] = (
        output["total_system_cost_cny_per_mean_mw_year"] / 1000.0
    )
    normalized.append(cost_column)
    return output, list(RAW_METRICS) + normalized


def factorial_effects(
    forced: pd.DataFrame,
    metric_columns: list[str],
) -> pd.DataFrame:
    grouping = [
        "country",
        "year",
        "config_id",
        "recharge_phase_fraction",
    ]
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
    rows: list[dict[str, Any]] = []
    expected_cells = {
        (task, recharge) for task in LEVELS for recharge in LEVELS
    }
    for key, group in forced.groupby(grouping, sort=True):
        cells = group.set_index(
            ["task_chronology", "recharge_chronology"]
        )
        if set(cells.index) != expected_cells:
            continue
        base = dict(zip(grouping, key))
        for effect, weights in coefficients.items():
            row: dict[str, Any] = {
                **base,
                "row_type": "technology_effect_phase",
                "effect_type": effect,
                "manuscript_evidence_admitted": False,
            }
            for metric in metric_columns:
                row[metric] = float(
                    sum(
                        weight * cells.loc[cell, metric]
                        for cell, weight in weights.items()
                    )
                )
            rows.append(row)
    return pd.DataFrame(rows)


def phase_means(
    frame: pd.DataFrame,
    metric_columns: list[str],
    group_columns: list[str],
    row_type: str,
) -> pd.DataFrame:
    output = (
        frame.groupby(group_columns, as_index=False)[metric_columns]
        .mean()
        .copy()
    )
    output["recharge_phase_fraction"] = np.nan
    output["row_type"] = row_type
    output["manuscript_evidence_admitted"] = False
    return output


def technology_contrasts(
    effects: pd.DataFrame,
    metric_columns: list[str],
) -> pd.DataFrame:
    grouping = [
        "country",
        "year",
        "recharge_phase_fraction",
        "effect_type",
    ]
    rows: list[dict[str, Any]] = []
    for key, group in effects.groupby(grouping, sort=True):
        values = group.set_index("config_id")
        if set(values.index) != set(CONFIGS):
            continue
        row: dict[str, Any] = {
            **dict(zip(grouping, key)),
            "config_id": "caes_24h_minus_li_lfp_4h",
            "row_type": "caes_minus_lfp_effect_phase",
            "manuscript_evidence_admitted": False,
        }
        for metric in metric_columns:
            row[metric] = float(
                values.loc["caes_24h", metric]
                - values.loc["li_lfp_4h", metric]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def descriptive_distributions(
    country_year: pd.DataFrame,
    metric_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (
        row_type,
        config,
        effect,
    ), group in country_year.groupby(
        ["row_type", "config_id", "effect_type"],
        sort=True,
    ):
        for metric in metric_columns:
            values = group[metric].to_numpy(float)
            tolerance = 1e-8
            rows.append(
                {
                    "row_type": row_type,
                    "config_id": config,
                    "effect_type": effect,
                    "metric": metric,
                    "country_year_n": int(len(values)),
                    "country_n": int(group["country"].nunique()),
                    "mean": float(np.mean(values)),
                    "standard_deviation": float(
                        np.std(values, ddof=1)
                    ),
                    "minimum": float(np.min(values)),
                    "q025": float(np.quantile(values, 0.025)),
                    "q25": float(np.quantile(values, 0.25)),
                    "median": float(np.median(values)),
                    "q75": float(np.quantile(values, 0.75)),
                    "q975": float(np.quantile(values, 0.975)),
                    "maximum": float(np.max(values)),
                    "negative_count": int(
                        np.sum(values < -tolerance)
                    ),
                    "near_zero_count": int(
                        np.sum(np.abs(values) <= tolerance)
                    ),
                    "positive_count": int(
                        np.sum(values > tolerance)
                    ),
                    "inference_status": (
                        "descriptive country-year distribution; repeated "
                        "years within countries are not independent"
                    ),
                    "manuscript_evidence_admitted": False,
                }
            )
    return pd.DataFrame(rows)


def fmt(value: float, digits: int = 2) -> str:
    if abs(value) < 0.5 * 10 ** (-digits):
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def build_markdown(
    distributions: pd.DataFrame,
    qa: dict[str, Any],
) -> str:
    focus = {
        "Discharge throughput": (
            "storage_discharge_output_mwh_per_mean_task_mw"
        ),
        "Equivalent full cycles": "equivalent_full_cycles_output_basis",
        "System cost": (
            "total_system_cost_cny_per_mean_task_kw_year"
        ),
        "Incremental benefit": "signed_benefit_cny_per_forced_kw_year",
    }
    contrast = distributions[
        distributions["row_type"].eq(
            "caes_minus_lfp_effect_phase_mean"
        )
    ]
    lines = [
        "# Full historical R2 task × recharge factorial",
        "",
        "## Frozen design",
        "",
        (
            "The analysis freezes 132 complete country-years of actual "
            "hourly aggregate `Fossil gas` generation and compares LFP 4 h "
            "with CAES 24 h. Intact task days and R2 potential-surplus days "
            "are independently arranged as dispersed or clustered sequences. "
            "Four prespecified within-season recharge phase offsets are "
            "averaged within each country-year."
        ),
        "",
        (
            "**Evidence status:** mechanism diagnostic only. R2 is modeled "
            "potential VRE surplus, not observed curtailment; every cell and "
            "aggregate remains `manuscript_evidence_admitted=false`."
        ),
        "",
        "## QA",
        "",
        f"- Country-years: {qa['observed_country_year_count']} / 132.",
        f"- Technology runs: {qa['observed_run_count']} / 264.",
        (
            f"- Forced factorial cells: {qa['forced_factorial_rows']} / "
            "4,224."
        ),
        f"- Aggregate QA status: **{qa['status']}**.",
        (
            "- Maximum absolute accounting residual: "
            f"{qa['maximum_absolute_accounting_error_mwh']:.3e} MWh."
        ),
        (
            "- Maximum relative accounting residual: "
            f"{qa['maximum_relative_accounting_error']:.3e}."
        ),
        "",
        "## CAES 24 h minus LFP 4 h country-year distributions",
        "",
        (
            "The task main effect is the clustered-minus-dispersed task "
            "contrast averaged over both recharge chronologies. The recharge "
            "main effect is defined analogously. The interaction is "
            "Y_CC − Y_CD − Y_DC + Y_DD. Values below are descriptive means "
            "[2.5th, 97.5th country-year percentiles], followed by negative / "
            "positive country-year counts."
        ),
        "",
        "| Effect | Metric | Mean [2.5%, 97.5%] | Negative / positive |",
        "|---|---|---:|---:|",
    ]
    for effect in EFFECTS:
        for label, metric in focus.items():
            selected = contrast[
                contrast["effect_type"].eq(effect)
                & contrast["metric"].eq(metric)
            ]
            if len(selected) != 1:
                continue
            row = selected.iloc[0]
            lines.append(
                f"| {effect} | {label} | "
                f"{fmt(float(row['mean']))} "
                f"[{fmt(float(row['q025']))}, "
                f"{fmt(float(row['q975']))}] | "
                f"{int(row['negative_count'])} / "
                f"{int(row['positive_count'])} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            (
                "Country-year percentiles describe heterogeneity and are not "
                "confidence intervals because repeated years within a country "
                "and spatially correlated European systems are not "
                "independent. The design isolates task and modeled recharge "
                "chronology; it does not turn R2 into observed hourly "
                "curtailment or establish a complete mid-duration valley."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    expected_pairs = expected_country_years()
    all_results, audits, failures = load_runs(expected_pairs)
    if all_results.empty:
        raise RuntimeError("No full-panel R2 factorial results found")
    expected_run_count = len(expected_pairs) * len(CONFIGS)
    if failures or len(audits) != expected_run_count:
        raise RuntimeError(
            "Strict full-panel aggregation refused: expected "
            f"{expected_run_count} completed technology runs, found "
            f"{len(audits)}; missing or failed entries={len(failures)}. "
            "No partial aggregate was written."
        )

    forced = all_results[
        all_results["variant"].eq("forced_1pct_joint")
    ].copy()
    forced, metric_columns = normalize(forced)
    effects = factorial_effects(forced, metric_columns)
    technology_phase_mean = phase_means(
        effects,
        metric_columns,
        ["country", "year", "config_id", "effect_type"],
        "technology_effect_phase_mean",
    )
    contrasts = technology_contrasts(effects, metric_columns)
    contrast_phase_mean = phase_means(
        contrasts,
        metric_columns,
        ["country", "year", "config_id", "effect_type"],
        "caes_minus_lfp_effect_phase_mean",
    )
    country_year = pd.concat(
        [technology_phase_mean, contrast_phase_mean],
        ignore_index=True,
        sort=False,
    ).sort_values(
        ["row_type", "country", "year", "config_id", "effect_type"],
        kind="stable",
    )
    distributions = descriptive_distributions(
        country_year,
        metric_columns,
    )

    effects_out = pd.concat(
        [effects, contrasts],
        ignore_index=True,
        sort=False,
    ).sort_values(
        [
            "row_type",
            "country",
            "year",
            "config_id",
            "effect_type",
            "recharge_phase_fraction",
        ],
        kind="stable",
    )
    effects_out.to_csv(OUTPUT_EFFECTS, index=False)
    country_year.to_csv(OUTPUT_COUNTRY_YEAR, index=False)
    distributions.to_csv(OUTPUT_DISTRIBUTIONS, index=False)

    expected_forced = (
        len(expected_pairs)
        * len(CONFIGS)
        * len(LEVELS)
        * len(LEVELS)
        * len(PHASES)
    )
    expected_result_rows = expected_run_count * 18
    expected_key = [
        "country",
        "year",
        "config_id",
        "task_chronology",
        "recharge_chronology",
        "recharge_phase_fraction",
    ]
    observed_pairs = sorted(
        {
            (str(row.country), int(row.year))
            for row in forced[["country", "year"]]
            .drop_duplicates()
            .itertuples(index=False)
        }
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
            "configuration_source_sha256",
            "runner_sha256",
            "factorial_helper_sha256",
            "core_runner_sha256",
        )
    }
    maximum_absolute = float(
        all_results[list(BALANCE_FIELDS)].abs().to_numpy().max()
    )
    maximum_relative = float(
        all_results[list(RELATIVE_BALANCE_FIELDS)]
        .abs()
        .to_numpy()
        .max()
    )
    order_hash_nunique = (
        forced.groupby(
            [
                "country",
                "year",
                "task_chronology",
                "recharge_chronology",
                "recharge_phase_fraction",
            ],
            sort=True,
        )[
            ["task_day_order_sha256", "recharge_day_order_sha256"]
        ]
        .nunique()
        .to_numpy()
        .max()
    )
    checks = {
        "all_132_country_years_present": observed_pairs == expected_pairs,
        "all_264_runs_present": len(audits) == expected_run_count,
        "all_run_QA_pass": all(
            audit["qa_status"] == "pass"
            and not audit["failed_checks"]
            for audit in audits
        ),
        "all_runs_non_evidential": all(
            audit["manuscript_evidence_admitted"] is False
            for audit in audits
        ),
        "each_run_has_18_results_16_forced_2_baseline": all(
            audit["result_rows"] == 18
            and audit["forced_rows"] == 16
            and audit["baseline_rows"] == 2
            and audit["conservation_rows"] == 16
            and audit["diagnostic_rows"] == 4
            for audit in audits
        ),
        "all_4752_result_rows_present": (
            len(all_results) == expected_result_rows
        ),
        "all_4224_forced_cells_present": len(forced) == expected_forced,
        "all_factorial_keys_unique": not bool(
            forced.duplicated(expected_key).any()
        ),
        "all_models_optimal": bool(
            all_results["solver_status"].eq(0).all()
        ),
        "both_technologies_present": set(forced["config_id"])
        == set(CONFIGS),
        "R2_only": set(forced["recharge_specification"]) == {"R2"},
        "zero_opportunity_price_only": set(
            forced["recharge_opportunity_price_cny_mwh"]
        )
        == {0.0},
        "all_four_phases_present": set(
            forced["recharge_phase_fraction"]
        )
        == set(PHASES),
        "technology_effect_phase_rows_complete": len(effects)
        == len(expected_pairs) * len(CONFIGS) * len(PHASES) * len(EFFECTS),
        "technology_effect_phase_mean_rows_complete": len(
            technology_phase_mean
        )
        == len(expected_pairs) * len(CONFIGS) * len(EFFECTS),
        "caes_lfp_effect_phase_rows_complete": len(contrasts)
        == len(expected_pairs) * len(PHASES) * len(EFFECTS),
        "caes_lfp_effect_phase_mean_rows_complete": len(
            contrast_phase_mean
        )
        == len(expected_pairs) * len(EFFECTS),
        "cross_technology_factor_orders_identical": (
            int(order_hash_nunique) == 1
        ),
        "all_absolute_accounting_balances_pass": (
            maximum_absolute <= 1e-6
        ),
        "all_relative_accounting_balances_pass": (
            maximum_relative <= 1e-10
        ),
        "all_source_hashes_frozen_within_suite": all(
            len(values) == 1 for values in source_hashes.values()
        ),
        "frozen_panel_hash_matches_outputs": source_hashes[
            "panel_sha256"
        ]
        == [sha256(PANEL)],
        "R2_not_labeled_observed_curtailment": bool(
            forced["recharge_interpretation"]
            .str.contains("not observed curtailment", regex=False)
            .all()
        ),
        "aggregate_outputs_non_evidential": bool(
            country_year["manuscript_evidence_admitted"]
            .eq(False)
            .all()
            and distributions["manuscript_evidence_admitted"]
            .eq(False)
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
        "status": (
            "pass" if all(checks.values()) and not failures else "fail"
        ),
        "checks": checks,
        "expected_country_year_count": len(expected_pairs),
        "observed_country_year_count": len(observed_pairs),
        "expected_run_count": expected_run_count,
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
        "source_hashes": source_hashes,
        "failures": failures,
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
        "caes_lfp_contrast": (
            "each factorial effect for CAES24 minus the same effect for LFP4"
        ),
        "R2_interpretation": (
            "potential-VRE-surplus counterfactual; not observed curtailment"
        ),
        "distribution_interpretation": (
            "descriptive country-year distributions; not confidence "
            "intervals or independent replicates"
        ),
        "manuscript_evidence_admitted": False,
        "reason": (
            "full historical R2 factorial remains a mechanism diagnostic "
            "because R2 is counterfactual potential surplus"
        ),
        "effects_file": str(OUTPUT_EFFECTS),
        "country_year_file": str(OUTPUT_COUNTRY_YEAR),
        "distributions_file": str(OUTPUT_DISTRIBUTIONS),
        "effects_file_sha256": sha256(OUTPUT_EFFECTS),
        "country_year_file_sha256": sha256(OUTPUT_COUNTRY_YEAR),
        "distributions_file_sha256": sha256(OUTPUT_DISTRIBUTIONS),
    }
    OUTPUT_QA.write_text(
        json.dumps(qa, indent=2),
        encoding="utf-8",
    )
    OUTPUT_MD.write_text(
        build_markdown(distributions, qa),
        encoding="utf-8",
    )
    if qa["status"] != "pass":
        raise RuntimeError(json.dumps(qa, indent=2))
    print(
        json.dumps(
            {
                "status": qa["status"],
                "country_years": qa["observed_country_year_count"],
                "runs": qa["observed_run_count"],
                "forced_factorial_rows": qa[
                    "forced_factorial_rows"
                ],
                "maximum_absolute_accounting_error_mwh": qa[
                    "maximum_absolute_accounting_error_mwh"
                ],
                "maximum_relative_accounting_error": qa[
                    "maximum_relative_accounting_error"
                ],
                "manuscript_evidence_admitted": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
