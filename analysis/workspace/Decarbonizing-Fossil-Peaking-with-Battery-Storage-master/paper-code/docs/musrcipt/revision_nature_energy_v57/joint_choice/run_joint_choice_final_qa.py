#!/usr/bin/env python3
"""Fail-closed QA for the v44 joint-choice and natural-FE artefacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    summary = pd.read_csv(HERE / "joint_choice_flip_summary.csv")
    fixed = summary[
        summary.model_mode.eq("fixed")
        & summary.account.eq("efc_value")
        & summary.decision_scope.eq("voluntary_tail")
    ].set_index("cost_year")
    require(fixed.canonical_identity_flips.to_dict() == {2025: 0, 2030: 3, 2035: 22}, "Unexpected central fixed-power flip counts")
    require(int(fixed.loc[2035, "tail_to_no_tail_flips"]) == 22, "2035 flips are not all tail-to-no-tail")
    require(int(fixed.loc[2035, "margin_within_cost_path_span_count"]) == 130, "Unexpected cost-path margin count")

    mandatory = summary[
        summary.model_mode.eq("fixed")
        & summary.account.eq("efc_value")
        & summary.decision_scope.eq("mandatory_tail")
    ].set_index("cost_year")
    require(mandatory.canonical_identity_flips.to_dict() == {2025: 0, 2030: 1, 2035: 1}, "Unexpected LDES-LFP forced-tail flip counts")

    natural_qa = json.loads((HERE / "natural_fixed_effect_qa.json").read_text())
    solver_qa = json.loads((HERE / "natural_caes24_s20_fixed_highs" / "full_qa.json").read_text())
    smoke_qa = json.loads((HERE / "natural_caes24_s20_fixed_highs" / "smoke_qa.json").read_text())
    require(natural_qa["country_years"] == 132 and natural_qa["countries"] == 15, "Natural FE coverage mismatch")
    require(natural_qa["countries_with_natural_caes_lfp_sign_switch"] == 0, "Natural decision switch count changed")
    require(solver_qa["status"] == "pass" and solver_qa["all_optimal"], "Natural HiGHS solve failed")
    require(smoke_qa["status"] == "pass" and smoke_qa["maximum_relative_difference"] < 1e-7, "HiGHS-Gurobi equivalence failed")

    coefficients = pd.read_csv(HERE / "natural_fixed_effect_coefficients.csv")
    zero = coefficients[
        coefficients.model_id.eq("zero_window_only")
        & coefficients.predictor.eq("natural_zero_recovery_window_share")
    ].iloc[0]
    require(zero.bootstrap_ci95_high < 0.0, "Natural zero-window interval no longer excludes zero")

    report = {
        "status": "pass",
        "central_fixed_power_voluntary_flips": fixed.canonical_identity_flips.astype(int).to_dict(),
        "central_fixed_power_mandatory_tail_flips": mandatory.canonical_identity_flips.astype(int).to_dict(),
        "natural_country_years": 132,
        "natural_country_blocks": 15,
        "highs_gurobi_max_relative_difference": smoke_qa["maximum_relative_difference"],
        "natural_zero_window_coefficient": float(zero.coefficient),
        "natural_zero_window_ci95": [float(zero.bootstrap_ci95_low), float(zero.bootstrap_ci95_high)],
        "central_bounded_power_status": "pending_gurobi_token_service_recovery",
    }
    (HERE / "FINAL_QA.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
