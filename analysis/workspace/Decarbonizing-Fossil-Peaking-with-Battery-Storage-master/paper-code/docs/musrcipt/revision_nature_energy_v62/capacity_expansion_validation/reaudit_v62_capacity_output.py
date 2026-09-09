#!/usr/bin/env python3
"""Re-audit a completed v62 capacity-expansion output directory.

The model uses the maximum normalized peaking task as a redundant physical
upper bound on each discharge-power decision.  A design may validly attain
that bound: capacity above the task peak cannot supply additional task and is
strictly dominated by positive fixed cost.  This audit therefore tests that
the bound is not violated, rather than requiring an arbitrary positive slack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


OUTPUT_NAMES = (
    "design_summary.csv",
    "capacity_mix.csv",
    "scenario_metrics.csv.gz",
    "chronology_conservation.csv",
    "diversification_comparison.csv",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    qa_path = output / "qa.json"
    qa = json.loads(qa_path.read_text(encoding="utf-8"))

    old_hashes = qa.get("output_sha256", {})
    current_hashes = {name: sha256(output / name) for name in OUTPUT_NAMES}
    if old_hashes and current_hashes != old_hashes:
        raise RuntimeError("Persisted result hashes differ from the original audit")

    summaries = pd.read_csv(output / "design_summary.csv")
    slacks = summaries["maximum_capacity_upper_bound_slack"]
    qa["minimum_capacity_upper_bound_slack"] = float(slacks.min())
    qa["capacity_upper_bound_binding_design_count"] = int(slacks.le(1.0e-6).sum())
    qa["capacity_upper_bound_definition"] = (
        "maximum normalized peaking task across the fitted scenarios"
    )
    qa["capacity_upper_bound_validation"] = (
        "equality is an admissible full-coverage optimum; negative slack is a violation"
    )
    qa["output_sha256"] = current_hashes

    tolerances_pass = (
        qa["all_solver_statuses_optimal"]
        and qa["maximum_objective_identity_error_eur_per_normalized_mw"] <= 0.02
        and qa["maximum_task_balance_residual"] <= 1.0e-6
        and qa["maximum_shared_recharge_violation"] <= 1.0e-6
        and qa["maximum_state_balance_residual_mwh"] <= 1.0e-6
        and qa["minimum_capacity_upper_bound_slack"] >= -1.0e-8
        and qa["maximum_chronology_task_energy_residual"] <= 1.0e-10
        and qa["maximum_chronology_recharge_energy_residual"] <= 1.0e-10
    )
    qa["status"] = "pass" if tolerances_pass else "fail"
    qa["audit_revision"] = "task_peak_bound_equality_admitted"
    qa_path.write_text(
        json.dumps(qa, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(qa, ensure_ascii=False, indent=2))
    return 0 if tolerances_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
