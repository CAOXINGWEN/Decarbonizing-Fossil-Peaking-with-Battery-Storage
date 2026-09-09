#!/usr/bin/env python3
"""Shared, non-solver utilities for the ERAA 2035 adaptive workflow.

This module deliberately does not import or alter the optimization runner.
It freezes a 12-node screen, creates baseline-cache-aware task registries,
validates runner outputs, triggers the remaining 4--168 h catalogue
conservatively, and applies the preregistered boundary rules.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


SCRIPT_VERSION = "0.1.0"
SCHEMA_VERSION = "eraa_adaptive_results_v0.1.0"
TARGET_YEAR = 2035
WEATHER_SCENARIOS = tuple(f"WS{number:02d}" for number in range(1, 37))
CHRONOLOGIES = ("dispersed", "clustered")
RUNNER_VARIANTS = ("joint", "repeat_use_fixed_firm")

# The screen covers every technology family, all six preregistered critical
# durations and both sides of the prospective intermediate-duration region.
# The identifiers are frozen before looking at ERAA model results.
KEY_CONFIG_IDS = (
    "li_lfp_4h",
    "vrfb_4h",
    "caes_8h",
    "fecr_flow_8h",
    "gravity_8h",
    "molten_salt_tes_8h",
    "caes_24h",
    "laes_24h",
    "h2_cavern_24h",
    "h2_cavern_48h",
    "h2_cavern_72h",
    "h2_cavern_168h",
)
KEY_DURATIONS_H = (4.0, 8.0, 24.0, 48.0, 72.0, 168.0)

RUNNER_PRIMARY_KEY = (
    "weather_scenario",
    "chronology",
    "deployment_scope",
    "capacity_case",
    "split_case",
    "variant",
    "config_id",
)
BASELINE_STATE_KEY = RUNNER_PRIMARY_KEY[:5]
TRIGGER_REQUIRED_COLUMNS = {
    *RUNNER_PRIMARY_KEY,
    "technology",
    "duration_h",
    "objective_eur_year",
    "signed_joint_benefit_eur_kw_year",
    "baseline_cache_key",
}
FINAL_REQUIRED_COLUMNS = {
    *TRIGGER_REQUIRED_COLUMNS,
    "signed_repeat_use_benefit_eur_kw_year",
    "residual_adequacy_interaction_benefit_eur_kw_year",
    "expected_eens_mwh",
    "expected_lole_h",
    "elcc_mw",
    "candidate_charge_power_mw",
    "candidate_discharge_power_mw",
    "equivalent_full_cycles_output_basis",
    "gas_generation_mwh",
    "vre_spill_mwh",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = Path(__file__).with_name("eraa_confirmatory_model_core.py")
DEFAULT_CATALOG = (
    PROJECT_ROOT
    / "model_adapter"
    / "energy_charts_2015_2024_representative_pilot"
    / "configuration_catalog_53.csv"
)
DEFAULT_WEATHER_MAPPING = (
    PROJECT_ROOT
    / "artifacts"
    / "research"
    / "ERAA_2025_WEATHER_SCENARIO_MAPPING.csv"
)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        json_ready(payload),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.write("\n")
    os.replace(temporary, path)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(
            handle,
            index=False,
            lineterminator="\n",
            float_format="%.12g",
        )
    os.replace(temporary, path)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def stable_id(prefix: str, fields: Iterable[object]) -> str:
    payload = json.dumps(
        [str(value) for value in fields],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_{digest}"


def parse_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    allowed = {"true", "false", "1", "0", "yes", "no", "y", "n"}
    unexpected = sorted(set(normalized).difference(allowed))
    if unexpected:
        raise ValueError(f"cannot parse boolean values {unexpected}")
    return normalized.isin({"true", "1", "yes", "y"})


def load_catalog(path: Path = DEFAULT_CATALOG) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "config_id",
        "tech_id",
        "duration_h",
        "structural_extrapolation",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing catalogue columns {missing}")
    frame = frame.copy()
    frame["config_id"] = frame["config_id"].astype(str)
    frame["tech_id"] = frame["tech_id"].astype(str)
    frame["duration_h"] = pd.to_numeric(
        frame["duration_h"], errors="raise"
    ).astype(float)
    frame["structural_extrapolation"] = parse_bool_series(
        frame["structural_extrapolation"]
    )
    if frame["config_id"].duplicated().any():
        duplicate = sorted(
            frame.loc[frame["config_id"].duplicated(False), "config_id"].unique()
        )
        raise ValueError(f"{path}: duplicate config_id values {duplicate}")
    main = frame.loc[
        ~frame["structural_extrapolation"]
        & frame["duration_h"].between(4.0, 168.0, inclusive="both")
    ]
    if len(main) != 50:
        raise ValueError(
            f"{path}: expected 50 non-structural 4-168 h configurations, "
            f"observed {len(main)}"
        )
    key_missing = sorted(set(KEY_CONFIG_IDS).difference(frame["config_id"]))
    if key_missing:
        raise ValueError(f"{path}: missing frozen key configurations {key_missing}")
    key = frame.set_index("config_id").loc[list(KEY_CONFIG_IDS)].reset_index()
    if key["structural_extrapolation"].any():
        raise ValueError("a frozen key configuration is structural")
    if not key["duration_h"].between(4.0, 168.0).all():
        raise ValueError("a frozen key configuration lies outside 4-168 h")
    if set(key["duration_h"]) != set(KEY_DURATIONS_H):
        raise ValueError(
            "frozen key configurations do not cover all critical durations"
        )
    if key["tech_id"].nunique() != 8:
        raise ValueError("frozen key screen must cover eight technology families")
    return frame.sort_values(
        ["duration_h", "tech_id", "config_id"]
    ).reset_index(drop=True)


def result_schema_contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "fail-closed adapter aligned to "
            "run_eraa_2035_full_network_gurobi.py v0.1.0, including direct "
            "ELCC and equivalent-cycle outputs"
        ),
        "runner_result_filename": "run_metrics.csv",
        "runner_primary_key": list(RUNNER_PRIMARY_KEY),
        "trigger_required_columns": sorted(TRIGGER_REQUIRED_COLUMNS),
        "final_required_columns": sorted(FINAL_REQUIRED_COLUMNS),
        "allowed_variants": [
            "baseline",
            "joint",
            "repeat_use_fixed_firm",
        ],
        "baseline_rule": {
            "config_id": "__baseline__",
            "variant": "baseline",
            "solve_frequency": (
                "one physical solve per weather_scenario x chronology x "
                "capacity_case x split_case x physical input payload; the "
                "runner may repeat an identical cached-baseline reference row "
                "in each candidate run_metrics.csv"
            ),
            "cache_key": (
                "computed by the runner from its physical cache payload; it "
                "must be identical for every baseline reference and candidate "
                "row in the same manifest baseline_cache_group_id"
            ),
            "benefit_fields": "null or zero; never used as candidate values",
        },
        "variant_field_rules": {
            "joint": {
                "non_null": [
                    "signed_joint_benefit_eur_kw_year",
                    "residual_adequacy_interaction_benefit_eur_kw_year",
                    "expected_eens_mwh",
                    "expected_lole_h",
                    "elcc_mw",
                ]
            },
            "repeat_use_fixed_firm": {
                "non_null": [
                    "signed_repeat_use_benefit_eur_kw_year",
                ]
            },
        },
        "elcc_postprocessing_table": {
            "status": (
                "optional backward-compatible override/audit only; the "
                "current full runner writes elcc_mw directly"
            ),
            "primary_key": [
                "weather_scenario",
                "chronology",
                "deployment_scope",
                "capacity_case",
                "split_case",
                "config_id",
            ],
            "value_column": "elcc_mw",
            "join_rule": (
                "when supplied, complete coverage of every candidate "
                "configuration is required and values must agree with any "
                "runner-reported ELCC within 1e-6 MW; no interpolation or "
                "zero filling"
            ),
        },
        "benefit_identity": (
            "residual_adequacy_interaction_benefit_eur_kw_year must equal "
            "signed_joint_benefit_eur_kw_year minus "
            "signed_repeat_use_benefit_eur_kw_year after variant merge, "
            "within the declared numerical tolerance"
        ),
        "qa_contract": {
            "per_result_file": "sibling QA.json with pass=true",
            "evidence_status": (
                "full_network_confirmatory_screen and "
                "manuscript_evidence_admitted=true for manuscript release"
            ),
            "failed_or_missing_qa": "aggregation stops; no adaptive decision",
        },
        "unknown_columns": "allowed and preserved only in raw runner outputs",
        "manifest_context_fields_not_repeated_in_run_metrics": [
            "target_year",
            "network_case",
            "model_input_fingerprint",
        ],
        "remaining_external_field": [],
        "runner_execution_rule": (
            "final confirmatory runs must not use --skip-elcc; missing or "
            "non-finite candidate elcc_mw fails validation"
        ),
    }


def baseline_cache_key(
    *,
    target_year: int,
    weather_scenario: str,
    chronology: str,
    deployment_scope: str,
    capacity_case: str,
    split_case: str,
    network_case: str,
    model_input_fingerprint: str,
) -> str:
    fields = (
        target_year,
        weather_scenario,
        chronology,
        deployment_scope,
        capacity_case,
        split_case,
        network_case,
        model_input_fingerprint,
    )
    return stable_id("baseline", fields)


def build_adaptive_manifest(
    catalog: pd.DataFrame,
    *,
    output_root: Path,
    deployment_scope: str = "eu27",
    capacity_case: str = "amended_annex3_risk_averse_enhanced_hurdle",
    split_case: str = "retirement_feasible_max_ccgt",
    network_case: str = "full_75node_269edge_ntc",
    model_input_fingerprint: str = "UNFROZEN_INPUT_FINGERPRINT",
) -> pd.DataFrame:
    main = catalog.loc[
        ~catalog["structural_extrapolation"]
        & catalog["duration_h"].between(4.0, 168.0, inclusive="both")
    ].copy()
    key = main.loc[main["config_id"].isin(KEY_CONFIG_IDS)].copy()
    dense = main.loc[~main["config_id"].isin(KEY_CONFIG_IDS)].copy()
    if len(key) != 12 or len(dense) != 38:
        raise AssertionError(
            f"adaptive split must be 12+38, observed {len(key)}+{len(dense)}"
        )

    rows: list[dict[str, Any]] = []
    for weather in WEATHER_SCENARIOS:
        for chronology in CHRONOLOGIES:
            cache_group_id = baseline_cache_key(
                target_year=TARGET_YEAR,
                weather_scenario=weather,
                chronology=chronology,
                deployment_scope=deployment_scope,
                capacity_case=capacity_case,
                split_case=split_case,
                network_case=network_case,
                model_input_fingerprint=model_input_fingerprint,
            )
            cache_warm_task = stable_id(
                "task",
                (
                    "key_screen_cache_warm",
                    cache_group_id,
                    KEY_CONFIG_IDS[0],
                ),
            )
            for stage, stage_order, activation, configs in (
                ("key_screen", 1, "required", key),
                (
                    "dense_followup",
                    2,
                    "conditional_on_key_joint_sign_change",
                    dense,
                ),
            ):
                for config in configs.itertuples(index=False):
                    is_cache_warm = (
                        stage == "key_screen"
                        and config.config_id == KEY_CONFIG_IDS[0]
                    )
                    adaptive_task = (
                        cache_warm_task
                        if is_cache_warm
                        else stable_id(
                            "task",
                            (
                                stage,
                                TARGET_YEAR,
                                weather,
                                chronology,
                                deployment_scope,
                                capacity_case,
                                split_case,
                                network_case,
                                config.config_id,
                                model_input_fingerprint,
                            ),
                        )
                    )
                    rows.append(
                        {
                            "adaptive_task_id": adaptive_task,
                            "stage": stage,
                            "stage_order": stage_order,
                            "task_type": (
                                "candidate_cache_warm"
                                if is_cache_warm
                                else "candidate"
                            ),
                            "activation_state": activation,
                            "target_year": TARGET_YEAR,
                            "weather_scenario": weather,
                            "chronology": chronology,
                            "deployment_scope": deployment_scope,
                            "capacity_case": capacity_case,
                            "split_case": split_case,
                            "network_case": network_case,
                            "config_id": config.config_id,
                            "technology": config.tech_id,
                            "duration_h": float(config.duration_h),
                            "structural_extrapolation": bool(
                                config.structural_extrapolation
                            ),
                            "runner_variants": "|".join(RUNNER_VARIANTS),
                            "baseline_cache_group_id": cache_group_id,
                            "baseline_cache_key": (
                                "runner_computed_from_physical_payload"
                            ),
                            "depends_on_task_id": (
                                "" if is_cache_warm else cache_warm_task
                            ),
                            "result_relative_directory": str(
                                Path(stage) / adaptive_task
                            ),
                            "runner_script": (
                                "scripts/run_eraa_2035_full_network_gurobi.py"
                            ),
                            "runner_cli_arguments_json": json.dumps(
                                {
                                    "--weather-scenario": weather,
                                    "--chronology": chronology,
                                    "--deployment-scope": deployment_scope,
                                    "--capacity-case": capacity_case,
                                    "--split-case": split_case,
                                    "--config-id": config.config_id,
                                    "--cache-root": str(
                                        output_root / "baseline_cache"
                                    ),
                                    "--output": str(
                                        output_root
                                        / "results"
                                        / stage
                                        / adaptive_task
                                    ),
                                },
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                            "expected_result_file": "run_metrics.csv",
                            "expected_qa_file": "QA.json",
                            "status": "pending",
                        }
                    )
    result = pd.DataFrame(rows).sort_values(
        [
            "stage_order",
            "weather_scenario",
            "chronology",
            "duration_h",
            "technology",
            "config_id",
        ]
    )
    result = result.reset_index(drop=True)
    if result["adaptive_task_id"].duplicated().any():
        raise AssertionError("adaptive task IDs are not unique")
    if len(result.loc[result["stage"] == "key_screen"]) != 864:
        raise AssertionError("expected 864 key-screen candidate tasks")
    if len(result.loc[result["stage"] == "dense_followup"]) != 2_736:
        raise AssertionError("expected 2736 dense candidate tasks")
    if result["baseline_cache_group_id"].nunique() != 72:
        raise AssertionError("expected 72 baseline cache groups")
    warm = result.loc[result["task_type"].eq("candidate_cache_warm")]
    if len(warm) != 72:
        raise AssertionError("expected one cache-warm task per cache group")
    dependent = result.loc[~result["task_type"].eq("candidate_cache_warm")]
    if dependent["depends_on_task_id"].eq("").any():
        raise AssertionError("non-warm tasks must depend on a cache-warm task")
    return result


def compute_scale_estimate(manifest: pd.DataFrame) -> dict[str, Any]:
    counts = manifest["stage"].value_counts().to_dict()
    baseline_tasks = int(manifest["baseline_cache_group_id"].nunique())
    key_candidates = int(counts.get("key_screen", 0))
    dense_candidates = int(counts.get("dense_followup", 0))
    candidate_variants = len(RUNNER_VARIANTS)
    key_solves = baseline_tasks + key_candidates * candidate_variants
    maximum_solves = (
        baseline_tasks
        + (key_candidates + dense_candidates) * candidate_variants
    )
    naive_baseline_solves = key_candidates + dense_candidates
    cache_savings = naive_baseline_solves - baseline_tasks
    return {
        "schema_version": SCHEMA_VERSION,
        "target_year": TARGET_YEAR,
        "weather_scenarios": 36,
        "chronologies": 2,
        "cached_baseline_tasks": baseline_tasks,
        "key_configurations": 12,
        "key_candidate_tasks": key_candidates,
        "dense_remaining_configurations": 38,
        "conditional_dense_candidate_tasks": dense_candidates,
        "candidate_variants_per_task": candidate_variants,
        "initial_key_stage_optimization_solves": key_solves,
        "maximum_full_spectrum_optimization_solves": maximum_solves,
        "naive_baseline_solves_without_cache": naive_baseline_solves,
        "baseline_solves_avoided_at_full_spectrum": cache_savings,
        "maximum_runner_task_rows": len(manifest),
        "maximum_candidate_metric_rows": (
            key_candidates + dense_candidates
        )
        * candidate_variants,
        "independent_inference_units": 36,
        "interpretation": (
            "Task and solve counts are compute scale, not statistical sample "
            "sizes. Weather scenarios remain the inferential units."
        ),
        "wall_time_formula": (
            "(72 * baseline_minutes + 7200 * candidate_variant_minutes) "
            "/ parallel_workers at the full-spectrum upper bound"
        ),
    }


def collect_result_files(paths: Sequence[Path]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    files: list[Path] = []
    for raw in paths:
        path = raw.resolve()
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("run_metrics.csv")))
        else:
            raise FileNotFoundError(path)
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError("no run_metrics.csv files found")
    frames: list[pd.DataFrame] = []
    qa_rows: list[dict[str, Any]] = []
    for path in files:
        qa_path = path.parent / "QA.json"
        if not qa_path.exists():
            raise FileNotFoundError(
                f"{path}: sibling QA.json is required by the fail-closed contract"
            )
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        if not bool(qa.get("pass", False)):
            raise RuntimeError(f"{path}: QA.json does not have pass=true")
        frame = pd.read_csv(path)
        frame.insert(0, "_source_file", str(path))
        frames.append(frame)
        qa_rows.append(
            {
                "result_file": str(path),
                "qa_file": str(qa_path),
                "qa_pass": True,
                "evidence_status": qa.get("evidence_status", ""),
                "manuscript_evidence_admitted": bool(
                    qa.get("manuscript_evidence_admitted", False)
                ),
                "baseline_cache_hit": qa.get("baseline_cache_hit"),
            }
        )
    return pd.concat(frames, ignore_index=True), qa_rows


def validate_results(
    results: pd.DataFrame,
    *,
    mode: str,
) -> pd.DataFrame:
    required = (
        TRIGGER_REQUIRED_COLUMNS
        if mode == "trigger"
        else FINAL_REQUIRED_COLUMNS
    )
    missing = sorted(required.difference(results.columns))
    if missing:
        raise ValueError(
            f"{mode} result schema is missing required columns {missing}"
        )
    frame = results.copy()
    for column in (
        "weather_scenario",
        "chronology",
        "deployment_scope",
        "capacity_case",
        "split_case",
        "variant",
        "config_id",
        "technology",
        "baseline_cache_key",
    ):
        frame[column] = frame[column].astype(str)
    if not set(frame["weather_scenario"]).issubset(WEATHER_SCENARIOS):
        invalid = sorted(set(frame["weather_scenario"]).difference(WEATHER_SCENARIOS))
        raise ValueError(f"invalid weather scenarios {invalid}")
    if not set(frame["chronology"]).issubset(CHRONOLOGIES):
        invalid = sorted(set(frame["chronology"]).difference(CHRONOLOGIES))
        raise ValueError(f"invalid chronologies {invalid}")
    allowed_variants = {"baseline", *RUNNER_VARIANTS}
    if not set(frame["variant"]).issubset(allowed_variants):
        invalid = sorted(set(frame["variant"]).difference(allowed_variants))
        raise ValueError(f"invalid variants {invalid}")
    frame["duration_h"] = pd.to_numeric(
        frame["duration_h"], errors="raise"
    ).astype(float)
    numeric = {
        "objective_eur_year",
        "signed_joint_benefit_eur_kw_year",
    }
    if mode == "final":
        numeric.update(
            {
                "signed_repeat_use_benefit_eur_kw_year",
                "residual_adequacy_interaction_benefit_eur_kw_year",
                "expected_eens_mwh",
                "expected_lole_h",
                "elcc_mw",
                "candidate_charge_power_mw",
                "candidate_discharge_power_mw",
                "equivalent_full_cycles_output_basis",
                "gas_generation_mwh",
                "vre_spill_mwh",
            }
        )
    for column in sorted(numeric):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    baseline_mask = frame["variant"].eq("baseline")
    joint_mask = frame["variant"].eq("joint")
    turnover_mask = frame["variant"].eq("repeat_use_fixed_firm")
    finite_rules: dict[str, pd.Series] = {
        "baseline objective_eur_year": (
            baseline_mask
            & ~np.isfinite(frame["objective_eur_year"].to_numpy(dtype=float))
        ),
        "joint objective_eur_year": (
            joint_mask
            & ~np.isfinite(frame["objective_eur_year"].to_numpy(dtype=float))
        ),
        "joint signed_joint_benefit_eur_kw_year": (
            joint_mask
            & ~np.isfinite(
                frame["signed_joint_benefit_eur_kw_year"].to_numpy(dtype=float)
            )
        ),
    }
    if mode == "final":
        for label, mask, column in (
            (
                "joint residual_adequacy_interaction_benefit_eur_kw_year",
                joint_mask,
                "residual_adequacy_interaction_benefit_eur_kw_year",
            ),
            ("joint expected_eens_mwh", joint_mask, "expected_eens_mwh"),
            ("joint expected_lole_h", joint_mask, "expected_lole_h"),
            ("joint elcc_mw", joint_mask, "elcc_mw"),
            (
                "joint candidate_charge_power_mw",
                joint_mask,
                "candidate_charge_power_mw",
            ),
            (
                "joint candidate_discharge_power_mw",
                joint_mask,
                "candidate_discharge_power_mw",
            ),
            (
                "joint equivalent_full_cycles_output_basis",
                joint_mask,
                "equivalent_full_cycles_output_basis",
            ),
            (
                "joint gas_generation_mwh",
                joint_mask,
                "gas_generation_mwh",
            ),
            ("joint vre_spill_mwh", joint_mask, "vre_spill_mwh"),
            (
                "turnover signed_repeat_use_benefit_eur_kw_year",
                turnover_mask,
                "signed_repeat_use_benefit_eur_kw_year",
            ),
        ):
            finite_rules[label] = mask & ~np.isfinite(
                frame[column].to_numpy(dtype=float)
            )
    failed_finite = {
        label: int(mask.sum())
        for label, mask in finite_rules.items()
        if int(mask.sum())
    }
    if failed_finite:
        raise ValueError(
            f"variant-specific non-finite required values: {failed_finite}"
        )

    # Candidate duplicates are forbidden. The runner intentionally writes an
    # identical cached-baseline reference into every candidate result file,
    # so baseline references are audited and collapsed by physical state.
    candidate_frame = frame.loc[~baseline_mask].copy()
    duplicated = candidate_frame.duplicated(
        list(RUNNER_PRIMARY_KEY), keep=False
    )
    if duplicated.any():
        keys = (
            candidate_frame.loc[duplicated, list(RUNNER_PRIMARY_KEY)]
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            f"duplicate candidate runner primary keys, examples: {keys}"
        )

    baseline = frame.loc[frame["variant"].eq("baseline")]
    if not baseline.empty:
        if not baseline["config_id"].eq("__baseline__").all():
            raise ValueError("baseline rows must use config_id=__baseline__")
        consistency_columns = [
            column
            for column in (
                "baseline_cache_key",
                "objective_eur_year",
                "expected_eens_mwh",
                "expected_lole_h",
                "firm_addition_mw",
                "gas_generation_mwh",
                "vre_spill_mwh",
            )
            if column in baseline.columns
        ]
        inconsistent: list[dict[str, Any]] = []
        for state, group in baseline.groupby(
            list(BASELINE_STATE_KEY), sort=True, dropna=False
        ):
            bad = [
                column
                for column in consistency_columns
                if group[column].nunique(dropna=False) > 1
            ]
            if bad:
                inconsistent.append(
                    {
                        "state": list(state),
                        "inconsistent_columns": bad,
                    }
                )
        if inconsistent:
            raise ValueError(
                "cached baseline references disagree within a physical "
                f"state: {inconsistent[:10]}"
            )
        if "_source_file" in baseline.columns:
            baseline = baseline.sort_values("_source_file")
        baseline = baseline.drop_duplicates(
            list(BASELINE_STATE_KEY), keep="first"
        )
    candidates = frame.loc[frame["variant"].isin(RUNNER_VARIANTS)]
    if candidates["config_id"].eq("__baseline__").any():
        raise ValueError("candidate variants cannot use __baseline__")
    return pd.concat([baseline, candidates], ignore_index=True)


def audit_baseline_cache(
    results: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    require_all_manifest_baselines: bool = True,
) -> dict[str, Any]:
    state_columns = list(BASELINE_STATE_KEY)
    expected = manifest[
        state_columns + ["baseline_cache_group_id"]
    ].drop_duplicates()
    baseline = results.loc[
        results["variant"].eq("baseline"),
        state_columns + ["baseline_cache_key"],
    ].drop_duplicates()
    merged = expected.merge(
        baseline,
        how="left",
        on=state_columns,
        validate="one_to_one",
    )
    missing = int(merged["baseline_cache_key"].isna().sum())
    candidates = results.loc[results["variant"].isin(RUNNER_VARIANTS)]
    candidate_audit = candidates.merge(
        merged[state_columns + ["baseline_cache_key"]],
        on=state_columns,
        how="left",
        suffixes=("_candidate", "_baseline"),
        validate="many_to_one",
    )
    candidate_key_mismatch = candidate_audit[
        "baseline_cache_key_baseline"
    ].notna() & (
        candidate_audit["baseline_cache_key_candidate"]
        != candidate_audit["baseline_cache_key_baseline"]
    )
    mismatch = int(candidate_key_mismatch.sum())
    expected_state_keys = set(
        zip(
            expected["weather_scenario"],
            expected["chronology"],
            expected["deployment_scope"],
            expected["capacity_case"],
            expected["split_case"],
        )
    )
    observed_state_keys = set(
        zip(
            baseline["weather_scenario"],
            baseline["chronology"],
            baseline["deployment_scope"],
            baseline["capacity_case"],
            baseline["split_case"],
        )
    )
    extra_baseline_states = sorted(observed_state_keys.difference(expected_state_keys))
    passed = (
        (not require_all_manifest_baselines or missing == 0)
        and mismatch == 0
        and not extra_baseline_states
    )
    return {
        "pass": passed,
        "expected_baselines": len(expected),
        "observed_baselines": len(baseline),
        "missing_baselines": missing,
        "candidate_rows_with_mismatched_baseline_keys": mismatch,
        "extra_baseline_states": extra_baseline_states,
        "baseline_group_count": int(
            expected["baseline_cache_group_id"].nunique()
        ),
        "cache_key_interpretation": (
            "manifest freezes the physical group; the runner computes the "
            "payload hash and every candidate in the group must reference it"
        ),
    }


def _sign(value: float, tolerance: float) -> int:
    if not np.isfinite(value) or abs(value) <= tolerance:
        return 0
    return 1 if value > 0 else -1


def evaluate_dense_trigger(
    results: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    sign_tolerance_eur_kw_year: float = 1e-6,
) -> tuple[dict[str, Any], pd.DataFrame]:
    expected = manifest.loc[
        manifest["stage"].eq("key_screen"),
        [
            "weather_scenario",
            "chronology",
            "deployment_scope",
            "capacity_case",
            "split_case",
            "config_id",
            "technology",
            "duration_h",
        ],
    ].copy()
    joint = results.loc[
        results["variant"].eq("joint"),
        [
            "weather_scenario",
            "chronology",
            "deployment_scope",
            "capacity_case",
            "split_case",
            "config_id",
            "signed_joint_benefit_eur_kw_year",
            "baseline_cache_key",
        ],
    ].copy()
    merged = expected.merge(
        joint,
        how="left",
        on=[
            "weather_scenario",
            "chronology",
            "deployment_scope",
            "capacity_case",
            "split_case",
            "config_id",
        ],
        validate="one_to_one",
    )
    value = pd.to_numeric(
        merged["signed_joint_benefit_eur_kw_year"], errors="coerce"
    )
    missing = int(value.isna().sum())
    if missing:
        decision = {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked_incomplete_or_mismatched_key_screen",
            "activate_dense_followup": False,
            "key_rows_expected": len(expected),
            "key_rows_observed": int(value.notna().sum()),
            "missing_or_nonfinite_key_rows": missing,
            "cache_key_mismatches": 0,
            "trigger_rule": "not evaluated until the complete key screen passes",
            "trigger_event_count": 0,
        }
        return decision, pd.DataFrame()

    merged["joint_sign"] = [
        _sign(number, sign_tolerance_eur_kw_year) for number in value
    ]
    envelope = (
        merged.groupby(
            ["weather_scenario", "chronology", "duration_h"],
            as_index=False,
        )["signed_joint_benefit_eur_kw_year"]
        .max()
        .sort_values(["weather_scenario", "chronology", "duration_h"])
    )
    envelope["joint_sign"] = [
        _sign(number, sign_tolerance_eur_kw_year)
        for number in envelope["signed_joint_benefit_eur_kw_year"]
    ]
    events: list[dict[str, Any]] = []
    # A near-zero or changing non-envelope technology is also sufficient to
    # trigger the dense spectrum. This fail-open rule prevents the sparse
    # screen from hiding a technology-consistent curve behind another
    # technology that wins the cross-technology envelope.
    for (weather, chronology, technology), group in merged.groupby(
        ["weather_scenario", "chronology", "technology"], sort=True
    ):
        group = (
            group.sort_values("duration_h")
            .drop_duplicates("duration_h")
            .reset_index(drop=True)
        )
        records = list(group.itertuples(index=False))
        for record in records:
            if int(record.joint_sign) == 0:
                events.append(
                    {
                        "trigger_type": "near_zero_technology_node",
                        "weather_scenario": weather,
                        "chronology": chronology,
                        "duration_left_h": float(record.duration_h),
                        "duration_right_h": float(record.duration_h),
                        "value_left": float(
                            record.signed_joint_benefit_eur_kw_year
                        ),
                        "value_right": float(
                            record.signed_joint_benefit_eur_kw_year
                        ),
                        "sign_left": 0,
                        "sign_right": 0,
                    }
                )
        for left, right in zip(records[:-1], records[1:]):
            if int(left.joint_sign) != int(right.joint_sign):
                events.append(
                    {
                        "trigger_type": "technology_curve_sign_change",
                        "weather_scenario": weather,
                        "chronology": chronology,
                        "duration_left_h": float(left.duration_h),
                        "duration_right_h": float(right.duration_h),
                        "value_left": float(
                            left.signed_joint_benefit_eur_kw_year
                        ),
                        "value_right": float(
                            right.signed_joint_benefit_eur_kw_year
                        ),
                        "sign_left": int(left.joint_sign),
                        "sign_right": int(right.joint_sign),
                    }
                )
    config_pivot = merged.pivot_table(
        index=["weather_scenario", "config_id", "duration_h"],
        columns="chronology",
        values=[
            "signed_joint_benefit_eur_kw_year",
            "joint_sign",
        ],
        aggfunc="first",
    )
    for (weather, _config, duration), row in config_pivot.iterrows():
        dispersed_sign = int(row[("joint_sign", "dispersed")])
        clustered_sign = int(row[("joint_sign", "clustered")])
        if dispersed_sign != clustered_sign:
            events.append(
                {
                    "trigger_type": "technology_node_cross_chronology_sign_change",
                    "weather_scenario": weather,
                    "chronology": "paired",
                    "duration_left_h": float(duration),
                    "duration_right_h": float(duration),
                    "value_left": float(
                        row[
                            (
                                "signed_joint_benefit_eur_kw_year",
                                "dispersed",
                            )
                        ]
                    ),
                    "value_right": float(
                        row[
                            (
                                "signed_joint_benefit_eur_kw_year",
                                "clustered",
                            )
                        ]
                    ),
                    "sign_left": dispersed_sign,
                    "sign_right": clustered_sign,
                }
            )
    for (weather, chronology), group in envelope.groupby(
        ["weather_scenario", "chronology"], sort=True
    ):
        group = group.sort_values("duration_h")
        records = list(group.itertuples(index=False))
        for left, right in zip(records[:-1], records[1:]):
            if (
                left.joint_sign == 0
                or right.joint_sign == 0
                or left.joint_sign != right.joint_sign
            ):
                events.append(
                    {
                        "trigger_type": (
                            "near_zero_key_envelope"
                            if left.joint_sign == 0 or right.joint_sign == 0
                            else "adjacent_duration_sign_change"
                        ),
                        "weather_scenario": weather,
                        "chronology": chronology,
                        "duration_left_h": float(left.duration_h),
                        "duration_right_h": float(right.duration_h),
                        "value_left": float(
                            left.signed_joint_benefit_eur_kw_year
                        ),
                        "value_right": float(
                            right.signed_joint_benefit_eur_kw_year
                        ),
                        "sign_left": int(left.joint_sign),
                        "sign_right": int(right.joint_sign),
                    }
                )
    pivot = envelope.pivot_table(
        index=["weather_scenario", "duration_h"],
        columns="chronology",
        values=[
            "signed_joint_benefit_eur_kw_year",
            "joint_sign",
        ],
        aggfunc="first",
    )
    for (weather, duration), row in pivot.iterrows():
        dispersed_sign = int(row[("joint_sign", "dispersed")])
        clustered_sign = int(row[("joint_sign", "clustered")])
        if (
            dispersed_sign == 0
            or clustered_sign == 0
            or dispersed_sign != clustered_sign
        ):
            events.append(
                {
                    "trigger_type": (
                        "near_zero_cross_chronology"
                        if dispersed_sign == 0 or clustered_sign == 0
                        else "cross_chronology_sign_change"
                    ),
                    "weather_scenario": weather,
                    "chronology": "paired",
                    "duration_left_h": float(duration),
                    "duration_right_h": float(duration),
                    "value_left": float(
                        row[
                            (
                                "signed_joint_benefit_eur_kw_year",
                                "dispersed",
                            )
                        ]
                    ),
                    "value_right": float(
                        row[
                            (
                                "signed_joint_benefit_eur_kw_year",
                                "clustered",
                            )
                        ]
                    ),
                    "sign_left": dispersed_sign,
                    "sign_right": clustered_sign,
                }
            )
    event_frame = pd.DataFrame(events)
    activate = not event_frame.empty
    decision = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "dense_followup_activated"
            if activate
            else "dense_followup_not_activated_no_key_sign_change"
        ),
        "activate_dense_followup": activate,
        "key_rows_expected": len(expected),
        "key_rows_observed": len(merged),
        "missing_or_nonfinite_key_rows": 0,
        "cache_key_mismatches": 0,
        "joint_sign_tolerance_eur_kw_year": sign_tolerance_eur_kw_year,
        "trigger_event_count": len(event_frame),
        "trigger_rule": (
            "activate all remaining non-structural 4-168 h configurations "
            "if any key technology node or weather-scenario envelope is near "
            "zero, any technology curve or envelope changes sign between "
            "adjacent key durations, or a fixed node/envelope changes sign "
            "between dispersed and clustered chronologies"
        ),
        "nonactivation_interpretation": (
            "no boundary detected at key-screen resolution; hT/hI/W/DeltaW "
            "remain not evaluated rather than being set to zero"
        ),
    }
    return decision, event_frame


def resolve_manifest(
    manifest: pd.DataFrame,
    decision: dict[str, Any],
) -> pd.DataFrame:
    result = manifest.copy()
    dense = result["stage"].eq("dense_followup")
    status = decision.get("status", "")
    if status.startswith("blocked"):
        result.loc[dense, "activation_state"] = "blocked_key_screen_incomplete"
        result.loc[dense, "status"] = "blocked"
    elif bool(decision.get("activate_dense_followup", False)):
        result.loc[dense, "activation_state"] = "activated"
        result.loc[dense, "status"] = "pending"
    else:
        result.loc[dense, "activation_state"] = "not_activated"
        result.loc[dense, "status"] = "skipped_no_key_sign_change"
    return result


def load_weather_mapping(path: Path = DEFAULT_WEATHER_MAPPING) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "weather_scenario",
        "climate_model",
        "projection_year",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing weather mapping columns {missing}")
    frame = frame.copy()
    frame["weather_scenario"] = frame["weather_scenario"].astype(str)
    frame["climate_model"] = frame["climate_model"].astype(str)
    frame["projection_year"] = pd.to_numeric(
        frame["projection_year"], errors="raise"
    ).astype(int)
    if set(frame["weather_scenario"]) != set(WEATHER_SCENARIOS):
        raise ValueError(f"{path}: mapping must contain exactly WS01-WS36")
    counts = frame.groupby("climate_model")["weather_scenario"].size()
    if len(counts) != 3 or not counts.eq(12).all():
        raise ValueError(f"{path}: expected three climate models x 12 years")
    return frame.sort_values(
        ["climate_model", "projection_year", "weather_scenario"]
    ).reset_index(drop=True)


def hierarchical_weather_resamples(
    mapping: pd.DataFrame,
    *,
    draws: int,
    block_length: int,
    seed: int,
) -> list[list[str]]:
    if draws <= 0:
        raise ValueError("draws must be positive")
    models = sorted(mapping["climate_model"].unique())
    if block_length <= 0 or block_length > 12:
        raise ValueError("block_length must lie in 1..12")
    by_model = {
        model: (
            mapping.loc[mapping["climate_model"].eq(model)]
            .sort_values("projection_year")["weather_scenario"]
            .tolist()
        )
        for model in models
    }
    rng = np.random.default_rng(seed)
    result: list[list[str]] = []
    for _ in range(draws):
        selected_models = rng.choice(models, size=len(models), replace=True)
        sample: list[str] = []
        for model in selected_models:
            values = by_model[str(model)]
            one: list[str] = []
            while len(one) < len(values):
                start = int(rng.integers(0, len(values)))
                one.extend(
                    values[(start + offset) % len(values)]
                    for offset in range(block_length)
                )
            sample.extend(one[: len(values)])
        if len(sample) != 36:
            raise AssertionError("hierarchical resample does not contain 36 rows")
        result.append(sample)
    return result


def merge_candidate_variants(
    results: pd.DataFrame,
    catalog: pd.DataFrame,
    *,
    adequacy_identity_tolerance: float = 1e-6,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidates = results.loc[results["variant"].isin(RUNNER_VARIANTS)].copy()
    key = [
        "weather_scenario",
        "chronology",
        "deployment_scope",
        "capacity_case",
        "split_case",
        "config_id",
    ]
    joint_columns = key + [
        "technology",
        "duration_h",
        "signed_joint_benefit_eur_kw_year",
        "residual_adequacy_interaction_benefit_eur_kw_year",
        "expected_eens_mwh",
        "expected_lole_h",
        "elcc_mw",
        "candidate_charge_power_mw",
        "candidate_discharge_power_mw",
        "equivalent_full_cycles_output_basis",
        "gas_generation_mwh",
        "vre_spill_mwh",
        "baseline_cache_key",
    ]
    turnover_columns = key + [
        "signed_repeat_use_benefit_eur_kw_year",
        "baseline_cache_key",
    ]
    joint = candidates.loc[
        candidates["variant"].eq("joint"), joint_columns
    ].copy()
    turnover = candidates.loc[
        candidates["variant"].eq("repeat_use_fixed_firm"), turnover_columns
    ].copy()
    merged = joint.merge(
        turnover,
        on=key,
        how="outer",
        suffixes=("_joint", "_turnover"),
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        missing = merged.loc[~merged["_merge"].eq("both"), key + ["_merge"]]
        raise ValueError(
            "candidate joint/turnover variants are incomplete: "
            f"{missing.head(10).to_dict('records')}"
        )
    merged = merged.drop(columns="_merge")
    if not (
        merged["baseline_cache_key_joint"]
        == merged["baseline_cache_key_turnover"]
    ).all():
        raise ValueError("joint and turnover rows reference different baselines")
    merged["bt"] = pd.to_numeric(
        merged["signed_repeat_use_benefit_eur_kw_year"], errors="raise"
    )
    merged["bj"] = pd.to_numeric(
        merged["signed_joint_benefit_eur_kw_year"], errors="raise"
    )
    merged["ba_derived"] = merged["bj"] - merged["bt"]
    merged["ba_reported"] = pd.to_numeric(
        merged["residual_adequacy_interaction_benefit_eur_kw_year"],
        errors="raise",
    )
    error = (merged["ba_reported"] - merged["ba_derived"]).abs()
    scale = np.maximum(
        1.0,
        np.maximum(merged["ba_reported"].abs(), merged["ba_derived"].abs()),
    )
    relative = error / scale
    if (relative > adequacy_identity_tolerance).any():
        bad = merged.loc[
            relative > adequacy_identity_tolerance,
            key + ["ba_reported", "ba_derived"],
        ].head(10)
        raise ValueError(
            "reported adequacy interaction violates BJ-BT identity: "
            f"{bad.to_dict('records')}"
        )
    merged["ba"] = merged["ba_derived"]
    merged["elcc"] = pd.to_numeric(merged["elcc_mw"], errors="raise")
    merged["baseline_cache_key"] = merged["baseline_cache_key_joint"]
    metadata = catalog[
        [
            "config_id",
            "tech_id",
            "duration_h",
            "structural_extrapolation",
        ]
    ].rename(
        columns={
            "duration_h": "catalog_duration_h",
            "structural_extrapolation": "main_structural_extrapolation",
        }
    )
    merged = merged.merge(
        metadata,
        on="config_id",
        how="left",
        validate="many_to_one",
    )
    if merged["tech_id"].isna().any():
        missing = sorted(merged.loc[merged["tech_id"].isna(), "config_id"].unique())
        raise ValueError(f"results include configurations absent from catalog {missing}")
    if not np.allclose(
        merged["duration_h"].astype(float),
        merged["catalog_duration_h"].astype(float),
        atol=1e-9,
        rtol=0,
    ):
        raise ValueError("runner duration_h does not match configuration catalogue")
    qa = {
        "candidate_unit_rows": len(merged),
        "maximum_adequacy_identity_absolute_error": float(error.max()),
        "maximum_adequacy_identity_relative_error": float(relative.max()),
        "adequacy_identity_tolerance": adequacy_identity_tolerance,
        "pass": True,
    }
    return merged, qa


def _load_boundary_core():
    name = "eraa_confirmatory_model_core_for_adaptive_aggregator"
    spec = importlib.util.spec_from_file_location(name, CORE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(CORE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _metric_envelope(
    unit_values: pd.DataFrame,
    sampled_weather: Sequence[str],
) -> pd.DataFrame:
    counts = Counter(sampled_weather)
    weighted = unit_values.copy()
    weighted["_weight"] = weighted["weather_scenario"].map(counts).fillna(0)
    weighted = weighted.loc[weighted["_weight"].gt(0)].copy()
    rows: list[dict[str, Any]] = []
    for (chronology, duration), group in weighted.groupby(
        ["chronology", "duration_h"], sort=True
    ):
        config_rows: list[dict[str, Any]] = []
        for (config, technology), one in group.groupby(
            ["config_id", "tech_id"], sort=True
        ):
            weight = one["_weight"].to_numpy(dtype=float)
            config_rows.append(
                {
                    "config_id": config,
                    "technology": technology,
                    "bt": float(np.average(one["bt"], weights=weight)),
                    "bj": float(np.average(one["bj"], weights=weight)),
                    "ba": float(np.average(one["ba"], weights=weight)),
                    "elcc": float(np.average(one["elcc"], weights=weight)),
                    "main_evidence_domain": bool(
                        (~one["main_structural_extrapolation"]).all()
                        and float(duration) <= 168.0
                    ),
                }
            )
        config_frame = pd.DataFrame(config_rows)
        row: dict[str, Any] = {
            "chronology": chronology,
            "duration_h": float(duration),
            "main_evidence_domain": bool(
                config_frame["main_evidence_domain"].all()
            ),
        }
        for metric in ("bt", "bj", "ba", "elcc"):
            winner = config_frame.loc[config_frame[metric].idxmax()]
            row[metric] = float(winner[metric])
            row[f"{metric}_winner_config_id"] = winner["config_id"]
            row[f"{metric}_winner_technology"] = winner["technology"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["chronology", "duration_h"]
    ).reset_index(drop=True)


def _metric_technology_curves(
    unit_values: pd.DataFrame,
    sampled_weather: Sequence[str],
) -> pd.DataFrame:
    """Return within-technology duration curves for one weather resample."""

    counts = Counter(sampled_weather)
    weighted = unit_values.copy()
    weighted["_weight"] = weighted["weather_scenario"].map(counts).fillna(0)
    weighted = weighted.loc[weighted["_weight"].gt(0)].copy()
    rows: list[dict[str, Any]] = []
    for (chronology, technology, duration), group in weighted.groupby(
        ["chronology", "tech_id", "duration_h"], sort=True
    ):
        config_rows: list[dict[str, Any]] = []
        for config, one in group.groupby("config_id", sort=True):
            weight = one["_weight"].to_numpy(dtype=float)
            config_rows.append(
                {
                    "config_id": config,
                    "bt": float(np.average(one["bt"], weights=weight)),
                    "bj": float(np.average(one["bj"], weights=weight)),
                    "ba": float(np.average(one["ba"], weights=weight)),
                    "elcc": float(np.average(one["elcc"], weights=weight)),
                    "main_evidence_domain": bool(
                        (~one["main_structural_extrapolation"]).all()
                        and float(duration) <= 168.0
                    ),
                }
            )
        config_frame = pd.DataFrame(config_rows)
        row: dict[str, Any] = {
            "chronology": chronology,
            "technology": technology,
            "duration_h": float(duration),
            "main_evidence_domain": bool(
                config_frame["main_evidence_domain"].all()
            ),
        }
        for metric in ("bt", "bj", "ba", "elcc"):
            winner = config_frame.loc[config_frame[metric].idxmax()]
            row[metric] = float(winner[metric])
            row[f"{metric}_winner_config_id"] = winner["config_id"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["chronology", "technology", "duration_h"]
    ).reset_index(drop=True)


def _boundary_input_from_points(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "duration_h": frame["duration_h"],
            "bt_lower": frame["bt"],
            "bj_upper": frame["bj"],
            "bj_lower": frame["bj"],
            "ba_lower": frame["ba"],
            "elcc_lower": frame["elcc"],
            "main_evidence_domain": frame["main_evidence_domain"],
        }
    )


def boundary_analysis(
    unit_values: pd.DataFrame,
    weather_mapping: pd.DataFrame,
    *,
    bootstrap_draws: int = 2_000,
    block_length: int = 3,
    seed: int = 2026072603,
    minimum_valid_delta_draw_fraction: float = 0.95,
) -> dict[str, Any]:
    core = _load_boundary_core()
    expected_configs = set(
        unit_values.loc[
            ~unit_values["main_structural_extrapolation"],
            "config_id",
        ]
    )
    if len(expected_configs) != 50:
        raise ValueError(
            "final boundary analysis requires all 50 non-structural "
            f"configurations; observed {len(expected_configs)}"
        )
    expected_rows = 36 * 2 * 50
    if len(unit_values.loc[
        ~unit_values["main_structural_extrapolation"]
    ]) != expected_rows:
        raise ValueError(
            "final boundary analysis requires exactly one row per "
            f"36 weather x 2 chronology x 50 config; expected "
            f"{expected_rows}"
        )
    main_values = unit_values.loc[
        ~unit_values["main_structural_extrapolation"]
        & unit_values["duration_h"].between(4.0, 168.0, inclusive="both")
    ].copy()

    full_sample = list(WEATHER_SCENARIOS)
    point_envelope = _metric_envelope(main_values, full_sample)
    point_technology_curves = _metric_technology_curves(
        main_values, full_sample
    )
    resamples = hierarchical_weather_resamples(
        weather_mapping,
        draws=bootstrap_draws,
        block_length=block_length,
        seed=seed,
    )
    draw_frames: list[pd.DataFrame] = []
    technology_draw_frames: list[pd.DataFrame] = []
    boundary_draw_rows: list[dict[str, Any]] = []
    for draw_id, sample in enumerate(resamples):
        envelope = _metric_envelope(main_values, sample)
        envelope.insert(0, "bootstrap_draw", draw_id)
        draw_frames.append(envelope)
        technology_curves = _metric_technology_curves(main_values, sample)
        technology_curves.insert(0, "bootstrap_draw", draw_id)
        technology_draw_frames.append(technology_curves)
        for chronology in CHRONOLOGIES:
            one = envelope.loc[envelope["chronology"].eq(chronology)]
            boundary = core.evaluate_valley_for_chronology(
                _boundary_input_from_points(one)
            )
            boundary_draw_rows.append(
                {
                    "bootstrap_draw": draw_id,
                    "chronology": chronology,
                    **boundary,
                }
            )
    draws = pd.concat(draw_frames, ignore_index=True)
    technology_draws = pd.concat(
        technology_draw_frames, ignore_index=True
    )
    boundary_draws = pd.DataFrame(boundary_draw_rows)

    interval_rows: list[dict[str, Any]] = []
    for (chronology, duration), group in draws.groupby(
        ["chronology", "duration_h"], sort=True
    ):
        point = point_envelope.loc[
            point_envelope["chronology"].eq(chronology)
            & point_envelope["duration_h"].eq(duration)
        ].iloc[0]
        row: dict[str, Any] = {
            "chronology": chronology,
            "duration_h": float(duration),
            "main_evidence_domain": bool(point["main_evidence_domain"]),
        }
        for metric in ("bt", "bj", "ba", "elcc"):
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_point"] = float(point[metric])
            row[f"{metric}_lower"] = float(np.quantile(values, 0.025))
            row[f"{metric}_upper"] = float(np.quantile(values, 0.975))
            row[f"{metric}_winner_config_id"] = point[
                f"{metric}_winner_config_id"
            ]
            row[f"{metric}_winner_technology"] = point[
                f"{metric}_winner_technology"
            ]
        interval_rows.append(row)
    intervals = pd.DataFrame(interval_rows).sort_values(
        ["chronology", "duration_h"]
    )

    technology_interval_rows: list[dict[str, Any]] = []
    for (chronology, technology, duration), group in technology_draws.groupby(
        ["chronology", "technology", "duration_h"], sort=True
    ):
        point = point_technology_curves.loc[
            point_technology_curves["chronology"].eq(chronology)
            & point_technology_curves["technology"].eq(technology)
            & point_technology_curves["duration_h"].eq(duration)
        ].iloc[0]
        row: dict[str, Any] = {
            "chronology": chronology,
            "technology": technology,
            "duration_h": float(duration),
            "main_evidence_domain": bool(point["main_evidence_domain"]),
        }
        for metric in ("bt", "bj", "ba", "elcc"):
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_point"] = float(point[metric])
            row[f"{metric}_lower"] = float(np.quantile(values, 0.025))
            row[f"{metric}_upper"] = float(np.quantile(values, 0.975))
            row[f"{metric}_winner_config_id"] = point[
                f"{metric}_winner_config_id"
            ]
        technology_interval_rows.append(row)
    technology_intervals = pd.DataFrame(technology_interval_rows).sort_values(
        ["chronology", "technology", "duration_h"]
    )
    technology_boundary_rows: list[dict[str, Any]] = []
    for (chronology, technology), one in technology_intervals.groupby(
        ["chronology", "technology"], sort=True
    ):
        boundary = core.evaluate_valley_for_chronology(
            pd.DataFrame(
                {
                    "duration_h": one["duration_h"],
                    "bt_lower": one["bt_lower"],
                    "bj_upper": one["bj_upper"],
                    "bj_lower": one["bj_lower"],
                    "ba_lower": one["ba_lower"],
                    "elcc_lower": one["elcc_lower"],
                    "main_evidence_domain": one["main_evidence_domain"],
                }
            )
        )
        technology_boundary_rows.append(
            {
                "chronology": chronology,
                "technology": technology,
                **boundary,
            }
        )
    technology_boundaries = pd.DataFrame(technology_boundary_rows)

    population_rows: list[dict[str, Any]] = []
    for chronology in CHRONOLOGIES:
        one = intervals.loc[intervals["chronology"].eq(chronology)]
        boundary = core.evaluate_valley_for_chronology(
            pd.DataFrame(
                {
                    "duration_h": one["duration_h"],
                    "bt_lower": one["bt_lower"],
                    "bj_upper": one["bj_upper"],
                    "bj_lower": one["bj_lower"],
                    "ba_lower": one["ba_lower"],
                    "elcc_lower": one["elcc_lower"],
                    "main_evidence_domain": one["main_evidence_domain"],
                }
            )
        )
        population_rows.append({"chronology": chronology, **boundary})
    population = pd.DataFrame(population_rows)

    weather_boundary_rows: list[dict[str, Any]] = []
    weather_delta_rows: list[dict[str, Any]] = []
    for weather in WEATHER_SCENARIOS:
        one_weather = main_values.loc[
            main_values["weather_scenario"].eq(weather)
        ]
        one_envelope = _metric_envelope(one_weather, [weather])
        pair: dict[str, dict[str, Any]] = {}
        for chronology in CHRONOLOGIES:
            one = one_envelope.loc[
                one_envelope["chronology"].eq(chronology)
            ]
            boundary = core.evaluate_valley_for_chronology(
                _boundary_input_from_points(one)
            )
            pair[chronology] = boundary
            weather_boundary_rows.append(
                {
                    "weather_scenario": weather,
                    "chronology": chronology,
                    **boundary,
                }
            )
        delta = core.evaluate_delta_w(
            pair["dispersed"], pair["clustered"]
        )
        weather_delta_rows.append(
            {"weather_scenario": weather, **delta}
        )
    weather_boundaries = pd.DataFrame(weather_boundary_rows)
    weather_delta = pd.DataFrame(weather_delta_rows)

    delta_draw_rows: list[dict[str, Any]] = []
    for draw_id, group in boundary_draws.groupby("bootstrap_draw", sort=True):
        by_chronology = {
            row.chronology: row._asdict()
            for row in group.itertuples(index=False)
        }
        delta = core.evaluate_delta_w(
            by_chronology["dispersed"],
            by_chronology["clustered"],
        )
        delta_draw_rows.append({"bootstrap_draw": draw_id, **delta})
    delta_draws = pd.DataFrame(delta_draw_rows)
    valid_delta = pd.to_numeric(
        delta_draws["delta_W"], errors="coerce"
    ).dropna()
    valid_fraction = len(valid_delta) / bootstrap_draws
    population_by_chronology = {
        row.chronology: row._asdict()
        for row in population.itertuples(index=False)
    }
    both_population_boundaries = all(
        bool(population_by_chronology[chronology]["confirmed"])
        for chronology in CHRONOLOGIES
    )
    if (
        both_population_boundaries
        and valid_fraction >= minimum_valid_delta_draw_fraction
    ):
        delta_point = float(
            population_by_chronology["clustered"]["W_log_ratio"]
            - population_by_chronology["dispersed"]["W_log_ratio"]
        )
        delta_lower = float(np.quantile(valid_delta, 0.025))
        delta_upper = float(np.quantile(valid_delta, 0.975))
        delta_status = (
            "primary_case_widening_interval_above_zero"
            if delta_lower > 0
            else "primary_case_delta_interval_crosses_or_below_zero"
        )
    else:
        delta_point = math.nan
        delta_lower = math.nan
        delta_upper = math.nan
        delta_status = (
            "population_boundary_unidentified_or_censored"
            if not both_population_boundaries
            else "bootstrap_boundary_identification_fraction_below_threshold"
        )
    delta_summary = {
        "schema_version": SCHEMA_VERSION,
        "bootstrap_draws": bootstrap_draws,
        "bootstrap_seed": seed,
        "climate_model_resampling": (
            "resample three climate models with replacement; within each "
            f"selected model use circular moving blocks of {block_length} "
            "projection years to reconstruct 12 years"
        ),
        "minimum_valid_delta_draw_fraction": (
            minimum_valid_delta_draw_fraction
        ),
        "valid_delta_draws": len(valid_delta),
        "valid_delta_draw_fraction": valid_fraction,
        "delta_W_point": delta_point,
        "delta_W_ci_lower": delta_lower,
        "delta_W_ci_upper": delta_upper,
        "primary_case_status": delta_status,
        "technology_curves_confirmed": (
            technology_boundaries.loc[
                technology_boundaries["confirmed"].astype(bool),
                ["chronology", "technology"],
            ]
            .groupby("chronology")["technology"]
            .apply(list)
            .to_dict()
        ),
        "same_technology_supports_both_chronologies": bool(
            set(
                technology_boundaries.loc[
                    technology_boundaries["chronology"].eq("dispersed")
                    & technology_boundaries["confirmed"].astype(bool),
                    "technology",
                ]
            )
            & set(
                technology_boundaries.loc[
                    technology_boundaries["chronology"].eq("clustered")
                    & technology_boundaries["confirmed"].astype(bool),
                    "technology",
                ]
            )
        ),
        "cost_consistent_family_curve_status": (
            "not_evaluated_no_frozen_curve_family_mapping"
        ),
        "full_confirmatory_status": (
            "not_evaluated_pending_prespecified_threshold_cluster_tranche_"
            "network_cost_and_charging_sensitivities"
        ),
    }
    censor_rows: list[dict[str, Any]] = []
    for table_name, frame, unit_column in (
        ("weather_scenario", weather_boundaries, "weather_scenario"),
        ("bootstrap", boundary_draws, "bootstrap_draw"),
    ):
        for chronology in CHRONOLOGIES:
            one = frame.loc[frame["chronology"].eq(chronology)]
            for field in ("h_T_status", "h_I_status"):
                for status, count in one[field].value_counts(
                    dropna=False
                ).items():
                    censor_rows.append(
                        {
                            "analysis_level": table_name,
                            "chronology": chronology,
                            "boundary": field.replace("_status", ""),
                            "status": str(status),
                            "count": int(count),
                            "total_units": int(one[unit_column].nunique()),
                        }
                    )
    censoring = pd.DataFrame(censor_rows)
    statistical_rows: list[dict[str, Any]] = []
    for level, frame, total_units in (
        ("weather_scenario", weather_boundaries, 36),
        ("hierarchical_bootstrap", boundary_draws, bootstrap_draws),
    ):
        for chronology in CHRONOLOGIES:
            one = frame.loc[frame["chronology"].eq(chronology)]
            for metric in ("h_T_h", "h_I_h", "W_log_ratio"):
                values = pd.to_numeric(one[metric], errors="coerce").dropna()
                statistical_rows.append(
                    {
                        "analysis_level": level,
                        "chronology": chronology,
                        "metric": metric,
                        "identified_n": len(values),
                        "total_units": total_units,
                        "identified_fraction": len(values) / total_units,
                        "mean": (
                            float(values.mean()) if len(values) else math.nan
                        ),
                        "median": (
                            float(values.median()) if len(values) else math.nan
                        ),
                        "q025": (
                            float(values.quantile(0.025))
                            if len(values)
                            else math.nan
                        ),
                        "q975": (
                            float(values.quantile(0.975))
                            if len(values)
                            else math.nan
                        ),
                        "conditioning_note": (
                            "summaries use identified values only; censoring "
                            "counts are reported separately"
                        ),
                    }
                )
    for level, frame, total_units in (
        ("weather_scenario", weather_delta, 36),
        ("hierarchical_bootstrap", delta_draws, bootstrap_draws),
    ):
        values = pd.to_numeric(frame["delta_W"], errors="coerce").dropna()
        statistical_rows.append(
            {
                "analysis_level": level,
                "chronology": "clustered_minus_dispersed",
                "metric": "delta_W",
                "identified_n": len(values),
                "total_units": total_units,
                "identified_fraction": len(values) / total_units,
                "mean": float(values.mean()) if len(values) else math.nan,
                "median": float(values.median()) if len(values) else math.nan,
                "q025": (
                    float(values.quantile(0.025)) if len(values) else math.nan
                ),
                "q975": (
                    float(values.quantile(0.975)) if len(values) else math.nan
                ),
                "conditioning_note": (
                    "DeltaW is defined only when both chronology-specific "
                    "boundaries are identified; no censored value is filled"
                ),
            }
        )
    statistical_summary = pd.DataFrame(statistical_rows)
    return {
        "point_envelope": point_envelope,
        "bootstrap_envelope_intervals": intervals,
        "technology_curve_intervals": technology_intervals,
        "technology_curve_boundaries": technology_boundaries,
        "population_boundaries": population,
        "weather_scenario_boundaries": weather_boundaries,
        "weather_scenario_delta_w": weather_delta,
        "bootstrap_boundary_draws": boundary_draws,
        "bootstrap_delta_w_draws": delta_draws,
        "boundary_censoring_summary": censoring,
        "boundary_statistical_summary": statistical_summary,
        "delta_w_summary": delta_summary,
    }
