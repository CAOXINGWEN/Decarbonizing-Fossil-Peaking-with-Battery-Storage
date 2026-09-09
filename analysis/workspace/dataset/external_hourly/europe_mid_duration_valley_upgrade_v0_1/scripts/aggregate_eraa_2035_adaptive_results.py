#!/usr/bin/env python3
"""Resolve the adaptive trigger or aggregate the complete ERAA spectrum.

This post-processor never launches or changes the model runner. It requires
runner outputs and sibling QA files that satisfy the explicit schema
contract. Trigger mode resolves the conditional 38-node task set. Final mode
computes weather-scenario and hierarchical-bootstrap hT/hI/W/DeltaW results
without filling censored boundaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import pandas as pd

from eraa_2035_adaptive_workflow import (
    DEFAULT_CATALOG,
    DEFAULT_WEATHER_MAPPING,
    SCHEMA_VERSION,
    SCRIPT_VERSION,
    audit_baseline_cache,
    boundary_analysis,
    collect_result_files,
    evaluate_dense_trigger,
    load_catalog,
    load_weather_mapping,
    merge_candidate_variants,
    resolve_manifest,
    validate_results,
    write_csv,
    write_json,
)


def prepare_output(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"output exists: {path}; pass --force")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def attach_elcc_results(
    results: pd.DataFrame,
    path: Path | None,
) -> pd.DataFrame:
    """Optionally join a legacy/external ELCC table and audit direct values."""

    frame = results.copy()
    if path is None:
        return frame
    elcc = pd.read_csv(path.resolve())
    keys = [
        "weather_scenario",
        "chronology",
        "deployment_scope",
        "capacity_case",
        "split_case",
        "config_id",
    ]
    required = {*keys, "elcc_mw"}
    missing = sorted(required.difference(elcc.columns))
    if missing:
        raise ValueError(f"{path}: ELCC table missing columns {missing}")
    elcc = elcc[keys + ["elcc_mw"]].copy()
    if elcc.duplicated(keys).any():
        duplicate = elcc.loc[
            elcc.duplicated(keys, keep=False), keys
        ].head(10)
        raise ValueError(
            f"{path}: duplicate ELCC keys {duplicate.to_dict('records')}"
        )
    elcc["elcc_mw"] = pd.to_numeric(elcc["elcc_mw"], errors="raise")
    if not elcc["elcc_mw"].map(pd.notna).all():
        raise ValueError(f"{path}: ELCC contains missing values")
    had_elcc = "elcc_mw" in frame.columns
    if had_elcc:
        frame = frame.rename(columns={"elcc_mw": "_runner_elcc_mw"})
    frame = frame.merge(elcc, on=keys, how="left", validate="many_to_one")
    candidate = frame["variant"].astype(str).isin(
        ("joint", "repeat_use_fixed_firm")
    )
    if frame.loc[candidate, "elcc_mw"].isna().any():
        missing_keys = frame.loc[
            candidate & frame["elcc_mw"].isna(), keys
        ].drop_duplicates().head(10)
        raise ValueError(
            f"{path}: ELCC table does not cover candidate keys "
            f"{missing_keys.to_dict('records')}"
        )
    if had_elcc:
        both = candidate & frame["_runner_elcc_mw"].notna()
        error = (
            pd.to_numeric(frame.loc[both, "_runner_elcc_mw"], errors="raise")
            - frame.loc[both, "elcc_mw"]
        ).abs()
        if (error > 1e-6).any():
            raise ValueError(
                "runner and external ELCC values disagree by more than 1e-6 MW"
            )
        frame = frame.drop(columns="_runner_elcc_mw")
    frame.loc[frame["variant"].astype(str).eq("baseline"), "elcc_mw"] = 0.0
    return frame


def execute_trigger(
    results: pd.DataFrame,
    manifest: pd.DataFrame,
    output: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    validated = validate_results(results, mode="trigger")
    baseline_qa = audit_baseline_cache(validated, manifest)
    if not baseline_qa["pass"]:
        raise RuntimeError(f"baseline cache audit failed: {baseline_qa}")
    decision, diagnostics = evaluate_dense_trigger(
        validated,
        manifest,
        sign_tolerance_eur_kw_year=args.sign_tolerance,
    )
    resolved = resolve_manifest(manifest, decision)
    write_json(output / "adaptive_trigger_decision.json", decision)
    write_csv(diagnostics, output / "adaptive_trigger_diagnostics.csv")
    write_csv(resolved, output / "adaptive_run_manifest_resolved.csv")
    write_json(output / "baseline_cache_QA.json", baseline_qa)
    return {
        "script_version": SCRIPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "mode": "trigger",
        "status": decision["status"],
        "activate_dense_followup": decision["activate_dense_followup"],
        "trigger_event_count": decision["trigger_event_count"],
        "baseline_cache_pass": True,
        "release_status": "adaptive_compute_decision_only_no_manuscript_result",
    }


def execute_final(
    results: pd.DataFrame,
    manifest: pd.DataFrame,
    output: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    validated = validate_results(results, mode="final")
    baseline_qa = audit_baseline_cache(validated, manifest)
    if not baseline_qa["pass"]:
        raise RuntimeError(f"baseline cache audit failed: {baseline_qa}")
    catalog = load_catalog(args.catalog.resolve())
    unit_values, identity_qa = merge_candidate_variants(
        validated,
        catalog,
        adequacy_identity_tolerance=args.adequacy_identity_tolerance,
    )
    write_csv(unit_values, output / "weather_config_chronology_values.csv")
    mapping = load_weather_mapping(args.weather_mapping.resolve())
    analysis = boundary_analysis(
        unit_values,
        mapping,
        bootstrap_draws=args.bootstrap_draws,
        block_length=args.block_length,
        seed=args.seed,
        minimum_valid_delta_draw_fraction=(
            args.minimum_valid_delta_draw_fraction
        ),
    )
    for name, value in analysis.items():
        if isinstance(value, pd.DataFrame):
            write_csv(value, output / f"{name}.csv")
        else:
            write_json(output / f"{name}.json", value)
    write_json(output / "baseline_cache_QA.json", baseline_qa)
    write_json(output / "benefit_identity_QA.json", identity_qa)
    delta = analysis["delta_w_summary"]
    summary: dict[str, object] = {
        "script_version": SCRIPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "mode": "final",
        "status": "boundary_aggregation_complete",
        "baseline_cache_pass": True,
        "benefit_identity_pass": True,
        "weather_scenarios": 36,
        "chronologies": 2,
        "nonstructural_configurations": 50,
        "bootstrap_draws": args.bootstrap_draws,
        "delta_W_point": delta["delta_W_point"],
        "delta_W_ci_lower": delta["delta_W_ci_lower"],
        "delta_W_ci_upper": delta["delta_W_ci_upper"],
        "primary_case_status": delta["primary_case_status"],
        "full_confirmatory_status": delta["full_confirmatory_status"],
        "release_status": (
            "primary_case_boundary_result_only; full confirmation remains "
            "conditional on all prespecified sensitivities and release QA"
        ),
    }
    return summary


def execute(args: argparse.Namespace) -> dict[str, object]:
    output = args.output.resolve()
    prepare_output(output, args.force)
    manifest = pd.read_csv(args.manifest.resolve())
    required_manifest = {
        "adaptive_task_id",
        "stage",
        "weather_scenario",
        "chronology",
        "config_id",
        "baseline_cache_key",
        "baseline_cache_group_id",
    }
    missing = sorted(required_manifest.difference(manifest.columns))
    if missing:
        raise ValueError(f"adaptive manifest missing columns {missing}")
    results, qa_rows = collect_result_files(args.results)
    if args.mode == "final":
        results = attach_elcc_results(results, args.elcc_results)
    qa_index = pd.DataFrame(qa_rows)
    write_csv(qa_index, output / "input_result_QA_index.csv")
    all_network_confirmatory = bool(
        not qa_index.empty
        and qa_index["evidence_status"].astype(str).eq(
            "full_network_confirmatory_screen"
        ).all()
        and qa_index["manuscript_evidence_admitted"].astype(bool).all()
    )
    if args.require_network_confirmatory and not all_network_confirmatory:
        statuses = sorted(
            qa_index["evidence_status"].astype(str).unique().tolist()
        )
        raise RuntimeError(
            "results are not all admitted full_network_confirmatory_screen "
            "runs; "
            f"observed evidence_status values {statuses}"
        )
    if args.mode == "trigger":
        summary = execute_trigger(results, manifest, output, args)
    else:
        summary = execute_final(results, manifest, output, args)
    summary["input_result_files"] = len(qa_rows)
    summary["adaptive_manifest"] = str(args.manifest.resolve())
    summary["all_input_runs_network_confirmatory"] = all_network_confirmatory
    summary["manuscript_release_eligible_from_evidence_status"] = (
        all_network_confirmatory
    )
    write_json(output / "aggregation_manifest.json", summary)
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=("trigger", "final"), required=True)
    result.add_argument(
        "--results",
        type=Path,
        nargs="+",
        required=True,
        help="run_metrics.csv files or directories recursively containing them",
    )
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    result.add_argument(
        "--elcc-results",
        type=Path,
        help=(
            "optional backward-compatible ELCC table keyed by weather "
            "scenario, chronology, deployment/capacity/split case and "
            "config_id; when supplied it must cover every candidate and "
            "agree with direct runner elcc_mw values within 1e-6 MW"
        ),
    )
    result.add_argument(
        "--weather-mapping",
        type=Path,
        default=DEFAULT_WEATHER_MAPPING,
    )
    result.add_argument("--sign-tolerance", type=float, default=1e-6)
    result.add_argument("--adequacy-identity-tolerance", type=float, default=1e-6)
    result.add_argument("--bootstrap-draws", type=int, default=2_000)
    result.add_argument("--block-length", type=int, default=3)
    result.add_argument("--seed", type=int, default=2026072603)
    result.add_argument(
        "--minimum-valid-delta-draw-fraction",
        type=float,
        default=0.95,
    )
    result.add_argument(
        "--require-network-confirmatory",
        action="store_true",
        help=(
            "stop unless every sibling QA.json is labelled "
            "evidence_status=full_network_confirmatory_screen and "
            "manuscript_evidence_admitted=true"
        ),
    )
    result.add_argument("--force", action="store_true")
    return result


def main() -> int:
    summary = execute(parser().parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
