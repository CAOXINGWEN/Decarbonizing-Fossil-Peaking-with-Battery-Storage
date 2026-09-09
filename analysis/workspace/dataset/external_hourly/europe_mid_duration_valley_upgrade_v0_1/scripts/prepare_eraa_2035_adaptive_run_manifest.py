#!/usr/bin/env python3
"""Create the baseline-cached ERAA 2035 adaptive task registry.

The registry contains all tasks, but only cached baselines and the frozen
12-node key screen are initially active. The remaining 38 non-structural
4--168 h nodes require a machine-generated trigger decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from eraa_2035_adaptive_workflow import (
    DEFAULT_CATALOG,
    KEY_CONFIG_IDS,
    SCHEMA_VERSION,
    SCRIPT_VERSION,
    build_adaptive_manifest,
    compute_scale_estimate,
    load_catalog,
    result_schema_contract,
    sha256_file,
    write_csv,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "runs" / "eraa_2035_adaptive_manifest_v0_1"
)
PLANNING_FILENAMES = {
    "adaptive_run_manifest.csv",
    "key_12_configuration_catalog.csv",
    "conditional_dense_38_configuration_catalog.csv",
    "compute_scale_estimate.json",
    "compute_scale_estimate.csv",
    "result_schema_contract.json",
    "run_manifest.json",
}


def execute(args: argparse.Namespace) -> dict[str, object]:
    output = args.output.resolve()
    if output.exists():
        if not args.force:
            raise FileExistsError(f"output exists: {output}; pass --force")
        # A production result tree may live below ``output/results``. Never
        # delete it while regenerating the deterministic planning registry.
        for name in PLANNING_FILENAMES:
            path = output / name
            if path.exists():
                path.unlink()
    output.mkdir(parents=True, exist_ok=True)

    catalog = load_catalog(args.catalog.resolve())
    manifest = build_adaptive_manifest(
        catalog,
        output_root=output,
        deployment_scope=args.deployment_scope,
        capacity_case=args.capacity_case,
        split_case=args.split_case,
        network_case=args.network_case,
        model_input_fingerprint=args.model_input_fingerprint,
    )
    write_csv(manifest, output / "adaptive_run_manifest.csv")
    key_catalog = catalog.loc[
        catalog["config_id"].isin(KEY_CONFIG_IDS)
    ].copy()
    dense_catalog = catalog.loc[
        ~catalog["structural_extrapolation"]
        & catalog["duration_h"].between(4.0, 168.0, inclusive="both")
        & ~catalog["config_id"].isin(KEY_CONFIG_IDS)
    ].copy()
    write_csv(
        key_catalog.sort_values(["duration_h", "tech_id", "config_id"]),
        output / "key_12_configuration_catalog.csv",
    )
    write_csv(
        dense_catalog.sort_values(["duration_h", "tech_id", "config_id"]),
        output / "conditional_dense_38_configuration_catalog.csv",
    )
    scale = compute_scale_estimate(manifest)
    write_json(output / "compute_scale_estimate.json", scale)
    write_csv(
        pd.DataFrame(
            [
                {"quantity": key, "value": value}
                for key, value in scale.items()
                if isinstance(value, (int, float, str, bool))
            ]
        ),
        output / "compute_scale_estimate.csv",
    )
    schema = result_schema_contract()
    write_json(output / "result_schema_contract.json", schema)
    input_frozen = (
        args.model_input_fingerprint != "UNFROZEN_INPUT_FINGERPRINT"
    )
    summary: dict[str, object] = {
        "script_version": SCRIPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "evidence_status": "adaptive_task_graph_only_no_model_results",
        "configuration_catalog": str(args.catalog.resolve()),
        "configuration_catalog_sha256": sha256_file(args.catalog.resolve()),
        "model_input_fingerprint": args.model_input_fingerprint,
        "model_input_fingerprint_frozen": input_frozen,
        "execution_ready": input_frozen,
        "execution_blocker": (
            ""
            if input_frozen
            else "provide the immutable full-network model input fingerprint"
        ),
        "deployment_scope": args.deployment_scope,
        "capacity_case": args.capacity_case,
        "split_case": args.split_case,
        "network_case": args.network_case,
        "manifest_rows": len(manifest),
        "cached_baselines": int(
            manifest["baseline_cache_group_id"].nunique()
        ),
        "key_screen_candidate_tasks": int(
            manifest["stage"].eq("key_screen").sum()
        ),
        "conditional_dense_candidate_tasks": int(
            manifest["stage"].eq("dense_followup").sum()
        ),
        "frozen_key_config_ids": list(KEY_CONFIG_IDS),
        "runner_contract_status": (
            "aligned to full-network runner v0.1.0 direct ELCC and "
            "equivalent-cycle outputs; no runner code modified"
        ),
    }
    write_json(output / "run_manifest.json", summary)
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--deployment-scope", default="eu27")
    result.add_argument(
        "--capacity-case",
        default="amended_annex3_risk_averse_enhanced_hurdle",
    )
    result.add_argument(
        "--split-case", default="retirement_feasible_max_ccgt"
    )
    result.add_argument(
        "--network-case",
        default="full_75node_269edge_ntc",
        help=(
            "manifest label only; the current full runner has the complete "
            "75-node 269-edge NTC graph fixed internally"
        ),
    )
    result.add_argument(
        "--model-input-fingerprint",
        default="UNFROZEN_INPUT_FINGERPRINT",
        help=(
            "immutable hash or bundle ID shared by all tasks; the default "
            "creates an auditable plan but blocks execution"
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
