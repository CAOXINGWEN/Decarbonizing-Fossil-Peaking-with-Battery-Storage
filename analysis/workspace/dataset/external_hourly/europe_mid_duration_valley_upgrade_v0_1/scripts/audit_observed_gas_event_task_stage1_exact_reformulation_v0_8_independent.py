#!/usr/bin/env python3
"""Independent, non-evidential audit of the Stage 1 v0.8 toy verifier.

This audit does not modify v0.8 and does not load or solve an 8,760 h case.
It independently rebuilds certificates, attacks the public OBBT consumer,
checks the frozen 512-row toy inventory, and audits the external-output
manifest produced by a separate v0.8 toy rerun.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.optimize import linprog


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import run_observed_gas_event_task_stage1_exact_reformulation_toy_v0_8 as v08  # noqa: E402


V08_RUNNER = (
    HERE
    / "run_observed_gas_event_task_stage1_exact_reformulation_toy_v0_8.py"
)
V08_DESIGN = (
    ROOT
    / "artifacts"
    / "research"
    / "OBSERVED_GAS_EVENT_TASK_STAGE1_EXACT_REFORMULATION_DESIGN_V0_8.md"
)
V08_OUTPUT = (
    ROOT
    / "artifacts"
    / "research"
    / "stage1_exact_reformulation_toy_v0_8"
)
EXTERNAL_RECHECK = Path("/tmp/stage1_v08_independent_external_recheck")
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts"
    / "research"
    / "stage1_exact_reformulation_toy_v0_8_independent_audit"
    / "independent_audit_results.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return "NaN"
        if math.isinf(number):
            return "+Infinity" if number > 0 else "-Infinity"
        return number
    return value


def rehash(certificate: dict[str, Any]) -> None:
    model = certificate["model"]
    objective = certificate["objective"]
    constraints = {
        key: model[key]
        for key in (
            "variable_names",
            "inequality_names",
            "equality_names",
            "A_ub",
            "b_ub",
            "A_eq",
            "b_eq",
        )
    }
    bounds = {
        key: model[key]
        for key in ("variable_names", "lower_bounds", "upper_bounds")
    }
    certificate["model_sha256"] = canonical_sha256(model)
    certificate["constraints_sha256"] = canonical_sha256(constraints)
    certificate["variable_bounds_sha256"] = canonical_sha256(bounds)
    certificate["objective_sha256"] = canonical_sha256(objective)


def exact_three_variable_certificate(
    hour: int,
    mode: int,
    *,
    objective_constant: float,
    cutoff: float = 5.0,
) -> dict[str, Any]:
    """Build a certificate with active inequality, equality and both bound types."""

    model = {
        "schema_version": "lp-proof-model-v0.8",
        "model_name": f"independent-three-variable-{hour}-{mode}",
        "mode_hour": int(hour),
        "fixed_mode": int(mode),
        "variable_names": ["x", "y", "z"],
        "inequality_names": ["minimum_y_plus_z"],
        "equality_names": ["x_plus_z"],
        "A_ub": [[0.0, -1.0, -1.0]],
        "b_ub": [-3.0],
        "A_eq": [[1.0, 0.0, 1.0]],
        "b_eq": [2.0],
        "lower_bounds": [1.0, 0.0, -1.0],
        "upper_bounds": [4.0, 2.0, 3.0],
    }
    objective = {
        "schema_version": "lp-proof-objective-v0.8",
        "sense": "min",
        "variable_names": ["x", "y", "z"],
        "coefficients": [1.0, 0.0, 0.0],
        "constant": float(objective_constant),
    }
    optimum = float(objective_constant + 1.0)
    certificate = {
        "schema_version": "dual-primal-lp-certificate-v0.8",
        "source": "independent_exact_rational",
        "solver": "independent_analytic",
        "solver_version": "1.0",
        "certificate_id": f"independent-{hour}-{mode}-{objective_constant:g}",
        "mode_hour": int(hour),
        "fixed_mode": int(mode),
        "objective_cutoff": float(cutoff),
        "solver_status": {
            "status": "optimal",
            "code": 0,
            "message": "exact rational KKT construction",
        },
        "model": model,
        "objective": objective,
        "primal": {
            "values": [1.0, 2.0, 1.0],
            "objective": optimum,
            "max_primal_residual": 0.0,
        },
        "dual": {
            "inequality_multipliers": [1.0],
            "equality_multipliers": [1.0],
            "lower_bound_multipliers": [2.0, 0.0, 0.0],
            "upper_bound_multipliers": [0.0, 1.0, 0.0],
            "objective": optimum,
            "dual_feasibility_residual": 0.0,
        },
    }
    rehash(certificate)
    return certificate


def expected_hashes(
    data: dict[tuple[int, int], dict[str, Any]]
) -> tuple[dict[str, str], dict[str, str]]:
    return (
        {
            f"{hour}:{mode}": certificate["model_sha256"]
            for (hour, mode), certificate in data.items()
        },
        {
            f"{hour}:{mode}": certificate["objective_sha256"]
            for (hour, mode), certificate in data.items()
        },
    )


def branch_attack(
    name: str,
    mutate: Callable[[dict[str, Any]], None],
    *,
    rehash_after: bool = False,
) -> dict[str, Any]:
    hour = 1000 + sum(ord(character) for character in name)
    supplied = {
        (hour, 0): exact_three_variable_certificate(
            hour, 0, objective_constant=5.0
        ),
        (hour, 1): exact_three_variable_certificate(
            hour, 1, objective_constant=3.0
        ),
    }
    mutate(supplied[(hour, 0)])
    if rehash_after:
        rehash(supplied[(hour, 0)])
    model_hashes, objective_hashes = expected_hashes(supplied)
    try:
        decision = v08.objective_cutoff_obbt(
            mode_hours=(hour,),
            incumbent_upper_bound=5.0,
            expected_model_hashes=model_hashes,
            expected_objective_hashes=objective_hashes,
            certificate_provider=lambda h, m: supplied[(h, m)],
            allowed_sources={"independent_exact_rational"},
        )[0]
        return {
            "safe_no_fixing": decision.fixed_mode is None,
            "decision": asdict(decision),
        }
    except Exception as exc:
        return {
            "safe_no_fixing": True,
            "hard_stop_before_return": True,
            "exception": f"{type(exc).__name__}: {exc}",
        }


def overflow_certificate(hour: int, mode: int) -> dict[str, Any]:
    """All payload scalars are finite, but matrix products overflow to NaN."""

    model = {
        "schema_version": "lp-proof-model-v0.8",
        "model_name": "finite-input-overflow",
        "mode_hour": int(hour),
        "fixed_mode": int(mode),
        "variable_names": ["x", "y"],
        "inequality_names": ["overflowing_inequality"],
        "equality_names": [],
        "A_ub": [[1.0e308, -1.0e308]],
        "b_ub": [0.0],
        "A_eq": [],
        "b_eq": [],
        "lower_bounds": [0.0, 0.0],
        "upper_bounds": [2.0, 2.0],
    }
    objective = {
        "schema_version": "lp-proof-objective-v0.8",
        "sense": "min",
        "variable_names": ["x", "y"],
        "coefficients": [0.0, 0.0],
        "constant": 6.0,
    }
    certificate = {
        "schema_version": "dual-primal-lp-certificate-v0.8",
        "source": "independent_exact_rational",
        "solver": "malicious_finite_payload",
        "solver_version": "1.0",
        "certificate_id": "finite-input-overflow",
        "mode_hour": int(hour),
        "fixed_mode": int(mode),
        "objective_cutoff": 5.0,
        "solver_status": {
            "status": "optimal",
            "code": 0,
            "message": "malicious payload",
        },
        "model": model,
        "objective": objective,
        "primal": {
            "values": [2.0, 1.9],
            "objective": 6.0,
            "max_primal_residual": 0.0,
        },
        "dual": {
            "inequality_multipliers": [0.0],
            "equality_multipliers": [],
            "lower_bound_multipliers": [0.0, 0.0],
            "upper_bound_multipliers": [0.0, 0.0],
            "objective": 6.0,
            "dual_feasibility_residual": 0.0,
        },
    }
    rehash(certificate)
    return certificate


def main() -> None:
    output = DEFAULT_OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)

    # Independent formula and finite-bound test.
    valid = exact_three_variable_certificate(7, 0, objective_constant=5.0)
    verification = v08.verify_dual_primal_certificate(
        valid,
        expected_mode_hour=7,
        expected_fixed_mode=0,
        expected_model_sha256=valid["model_sha256"],
        expected_objective_sha256=valid["objective_sha256"],
        expected_objective_cutoff=5.0,
        allowed_sources={"independent_exact_rational"},
        primal_tolerance=1.0e-9,
        dual_tolerance=1.0e-9,
        objective_tolerance=1.0e-9,
        optimality_tolerance=1.0e-8,
    )
    model = valid["model"]
    objective = valid["objective"]
    scipy_solution = linprog(
        c=np.asarray(objective["coefficients"]),
        A_ub=np.asarray(model["A_ub"]),
        b_ub=np.asarray(model["b_ub"]),
        A_eq=np.asarray(model["A_eq"]),
        b_eq=np.asarray(model["b_eq"]),
        bounds=list(zip(model["lower_bounds"], model["upper_bounds"])),
        method="highs",
    )
    scipy_raw_objective = float(
        objective["constant"] + scipy_solution.fun
    )
    lam = np.asarray(valid["dual"]["inequality_multipliers"])
    mu = np.asarray(valid["dual"]["equality_multipliers"])
    alpha = np.asarray(valid["dual"]["lower_bound_multipliers"])
    beta = np.asarray(valid["dual"]["upper_bound_multipliers"])
    c = np.asarray(objective["coefficients"])
    a_ub = np.asarray(model["A_ub"])
    a_eq = np.asarray(model["A_eq"])
    b_ub = np.asarray(model["b_ub"])
    b_eq = np.asarray(model["b_eq"])
    lower = np.asarray(model["lower_bounds"])
    upper = np.asarray(model["upper_bounds"])
    q = c + a_ub.T @ lam + a_eq.T @ mu
    base = float(objective["constant"] - b_ub @ lam - b_eq @ mu)
    independent_box_bound = float(
        base + np.minimum(q * lower, q * upper).sum()
    )
    independent_kkt_bound = float(
        base + lower @ alpha - upper @ beta
    )

    attacks: dict[str, dict[str, Any]] = {}
    attacks["missing_dual_payload"] = branch_attack(
        "missing_dual_payload", lambda cert: cert.pop("dual")
    )
    attacks["empty_solver"] = branch_attack(
        "empty_solver", lambda cert: cert.__setitem__("solver", "")
    )
    attacks["empty_solver_version"] = branch_attack(
        "empty_solver_version",
        lambda cert: cert.__setitem__("solver_version", ""),
    )
    attacks["empty_certificate_id"] = branch_attack(
        "empty_certificate_id",
        lambda cert: cert.__setitem__("certificate_id", ""),
    )
    attacks["negative_primal_residual"] = branch_attack(
        "negative_primal_residual",
        lambda cert: cert["primal"].__setitem__(
            "max_primal_residual", -1.0
        ),
    )
    attacks["nan_primal_residual"] = branch_attack(
        "nan_primal_residual",
        lambda cert: cert["primal"].__setitem__(
            "max_primal_residual", float("nan")
        ),
    )
    attacks["negative_dual_residual"] = branch_attack(
        "negative_dual_residual",
        lambda cert: cert["dual"].__setitem__(
            "dual_feasibility_residual", -1.0
        ),
    )
    attacks["nan_dual_residual"] = branch_attack(
        "nan_dual_residual",
        lambda cert: cert["dual"].__setitem__(
            "dual_feasibility_residual", float("nan")
        ),
    )
    attacks["wrong_declared_model_hash"] = branch_attack(
        "wrong_declared_model_hash",
        lambda cert: cert.__setitem__("model_sha256", "f" * 64),
    )
    attacks["matrix_payload_changed_without_rehash"] = branch_attack(
        "matrix_payload_changed_without_rehash",
        lambda cert: cert["model"]["A_eq"][0].__setitem__(0, 2.0),
    )
    attacks["wrong_constraints_hash"] = branch_attack(
        "wrong_constraints_hash",
        lambda cert: cert.__setitem__("constraints_sha256", "e" * 64),
    )
    attacks["wrong_bounds_hash"] = branch_attack(
        "wrong_bounds_hash",
        lambda cert: cert.__setitem__("variable_bounds_sha256", "d" * 64),
    )
    attacks["stationarity_infeasible"] = branch_attack(
        "stationarity_infeasible",
        lambda cert: cert["dual"].__setitem__(
            "inequality_multipliers", [0.0]
        ),
    )
    attacks["negative_inequality_multiplier"] = branch_attack(
        "negative_inequality_multiplier",
        lambda cert: cert["dual"].__setitem__(
            "inequality_multipliers", [-1.0]
        ),
    )
    attacks["negative_lower_bound_multiplier"] = branch_attack(
        "negative_lower_bound_multiplier",
        lambda cert: cert["dual"].__setitem__(
            "lower_bound_multipliers", [-2.0, 0.0, 0.0]
        ),
    )
    attacks["negative_upper_bound_multiplier"] = branch_attack(
        "negative_upper_bound_multiplier",
        lambda cert: cert["dual"].__setitem__(
            "upper_bound_multipliers", [0.0, -1.0, 0.0]
        ),
    )
    attacks["complementarity_infeasible"] = branch_attack(
        "complementarity_infeasible",
        lambda cert: (
            cert["dual"].__setitem__(
                "lower_bound_multipliers", [3.0, 0.0, 0.0]
            ),
            cert["dual"].__setitem__(
                "upper_bound_multipliers", [1.0, 1.0, 0.0]
            ),
        ),
    )
    attacks["fraudulent_dual_objective"] = branch_attack(
        "fraudulent_dual_objective",
        lambda cert: cert["dual"].__setitem__("objective", 1.0e12),
    )
    attacks["primal_infeasible"] = branch_attack(
        "primal_infeasible",
        lambda cert: cert["primal"].__setitem__(
            "values", [1.5, 2.0, 0.5]
        ),
    )
    attacks["lower_bound_above_upper_bound"] = branch_attack(
        "lower_bound_above_upper_bound",
        lambda cert: cert["model"]["lower_bounds"].__setitem__(0, 5.0),
        rehash_after=True,
    )
    attacks["nonfinite_upper_bound"] = branch_attack(
        "nonfinite_upper_bound",
        lambda cert: cert["model"]["upper_bounds"].__setitem__(
            0, float("inf")
        ),
    )
    attacks["nan_matrix_entry"] = branch_attack(
        "nan_matrix_entry",
        lambda cert: cert["model"]["A_ub"][0].__setitem__(
            0, float("nan")
        ),
    )
    attacks["duplicate_variable_names"] = branch_attack(
        "duplicate_variable_names",
        lambda cert: (
            cert["model"].__setitem__(
                "variable_names", ["x", "x", "z"]
            ),
            cert["objective"].__setitem__(
                "variable_names", ["x", "x", "z"]
            ),
        ),
        rehash_after=True,
    )
    attacks["objective_variable_order_mismatch"] = branch_attack(
        "objective_variable_order_mismatch",
        lambda cert: cert["objective"].__setitem__(
            "variable_names", ["y", "x", "z"]
        ),
        rehash_after=True,
    )
    attacks["missing_equality_multiplier"] = branch_attack(
        "missing_equality_multiplier",
        lambda cert: cert["dual"].__setitem__(
            "equality_multipliers", []
        ),
    )
    attacks["nonoptimal_status"] = branch_attack(
        "nonoptimal_status",
        lambda cert: cert["solver_status"].__setitem__("status", "feasible"),
    )
    attacks["unallowlisted_source"] = branch_attack(
        "unallowlisted_source",
        lambda cert: cert.__setitem__("source", "unknown"),
    )
    attacks["wrong_cutoff"] = branch_attack(
        "wrong_cutoff",
        lambda cert: cert.__setitem__("objective_cutoff", 4.0),
    )

    baseline = {
        (30, 0): exact_three_variable_certificate(
            30, 0, objective_constant=5.0
        ),
        (30, 1): exact_three_variable_certificate(
            30, 1, objective_constant=3.0
        ),
    }
    baseline_models, baseline_objectives = expected_hashes(baseline)
    valid_decision = v08.objective_cutoff_obbt(
        mode_hours=(30,),
        incumbent_upper_bound=5.0,
        expected_model_hashes=baseline_models,
        expected_objective_hashes=baseline_objectives,
        certificate_provider=lambda h, m: baseline[(h, m)],
        allowed_sources={"independent_exact_rational"},
    )[0]

    guard_results: dict[str, Any] = {}
    for name, value in (
        ("negative_proof_guard", -1.0),
        ("nan_proof_guard", float("nan")),
    ):
        try:
            v08.objective_cutoff_obbt(
                mode_hours=(30,),
                incumbent_upper_bound=5.0,
                expected_model_hashes=baseline_models,
                expected_objective_hashes=baseline_objectives,
                certificate_provider=lambda h, m: baseline[(h, m)],
                allowed_sources={"independent_exact_rational"},
                proof_guard=value,
            )
            guard_results[name] = {"hard_stop": False}
        except Exception as exc:
            guard_results[name] = {
                "hard_stop": True,
                "exception": f"{type(exc).__name__}: {exc}",
            }

    both = {
        (31, mode): exact_three_variable_certificate(
            31, mode, objective_constant=5.0 + mode
        )
        for mode in (0, 1)
    }
    both_models, both_objectives = expected_hashes(both)
    try:
        v08.objective_cutoff_obbt(
            mode_hours=(31,),
            incumbent_upper_bound=5.0,
            expected_model_hashes=both_models,
            expected_objective_hashes=both_objectives,
            certificate_provider=lambda h, m: both[(h, m)],
            allowed_sources={"independent_exact_rational"},
        )
        both_excluded = {"hard_stop": False}
    except Exception as exc:
        both_excluded = {
            "hard_stop": True,
            "exception": f"{type(exc).__name__}: {exc}",
        }

    spoof = {
        (32, mode): exact_three_variable_certificate(
            32, mode, objective_constant=3.0
        )
        for mode in (0, 1)
    }
    for certificate in spoof.values():
        certificate["claimed_lower_bound"] = 1.0e12
    spoof_models, spoof_objectives = expected_hashes(spoof)
    spoof_decision = v08.objective_cutoff_obbt(
        mode_hours=(32,),
        incumbent_upper_bound=5.0,
        expected_model_hashes=spoof_models,
        expected_objective_hashes=spoof_objectives,
        certificate_provider=lambda h, m: spoof[(h, m)],
        allowed_sources={"independent_exact_rational"},
    )[0]

    # Minimal exploit: finite payloads create NaN during matrix multiplication.
    overflow = {
        (40, 0): overflow_certificate(40, 0),
        (40, 1): exact_three_variable_certificate(
            40, 1, objective_constant=3.0
        ),
    }
    overflow_models, overflow_objectives = expected_hashes(overflow)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        overflow_verification = v08.verify_dual_primal_certificate(
            overflow[(40, 0)],
            expected_mode_hour=40,
            expected_fixed_mode=0,
            expected_model_sha256=overflow[(40, 0)]["model_sha256"],
            expected_objective_sha256=overflow[(40, 0)][
                "objective_sha256"
            ],
            expected_objective_cutoff=5.0,
            allowed_sources={"independent_exact_rational"},
            primal_tolerance=1.0e-9,
            dual_tolerance=1.0e-9,
            objective_tolerance=1.0e-9,
            optimality_tolerance=1.0e-8,
        )
        overflow_decision = v08.objective_cutoff_obbt(
            mode_hours=(40,),
            incumbent_upper_bound=5.0,
            expected_model_hashes=overflow_models,
            expected_objective_hashes=overflow_objectives,
            certificate_provider=lambda h, m: overflow[(h, m)],
            allowed_sources={"independent_exact_rational"},
        )[0]
    overflow_result = {
        "all_payload_scalars_finite": True,
        "verification": asdict(overflow_verification),
        "decision": asdict(overflow_decision),
        "runtime_warnings": [
            f"{item.category.__name__}: {item.message}" for item in captured
        ],
        "fail_closed": (
            not overflow_verification.admitted
            and overflow_decision.fixed_mode is None
        ),
    }

    results_path = V08_OUTPUT / "toy_pattern_equivalence.csv"
    results = pd.read_csv(results_path)
    expected_by_horizon = {4: 16, 6: 16, 8: 32, 12: 64, 16: 128, 24: 256}
    inventory: dict[str, Any] = {}
    inventory_pass = True
    for horizon, expected_count in expected_by_horizon.items():
        subset = results.loc[results["horizon_h"] == horizon]
        observed = [int(value) for value in subset["pattern_index"]]
        exact = (
            len(observed) == expected_count
            and len(set(observed)) == expected_count
            and set(observed) == set(range(expected_count))
        )
        inventory_pass = inventory_pass and exact
        inventory[str(horizon)] = {
            "expected": expected_count,
            "observed": len(observed),
            "unique": len(set(observed)),
            "exact_index_set": exact,
        }

    manifest_path = V08_OUTPUT / "manifest.json"
    qa_path = V08_OUTPUT / "QA.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    external_manifest_path = EXTERNAL_RECHECK / "manifest.json"
    external_manifest = json.loads(
        external_manifest_path.read_text(encoding="utf-8")
    )
    external_files = external_manifest["files"]
    external_manifest_checks = {
        "manifest_exists": external_manifest_path.exists(),
        "runner_hash_matches_current": (
            external_files["runner"]["sha256"] == sha256(V08_RUNNER)
        ),
        "result_path_is_absolute_external": (
            external_files["results"]["path_kind"] == "absolute_external"
        ),
        "qa_path_is_absolute_external": (
            external_files["qa"]["path_kind"] == "absolute_external"
        ),
        "external_results_hash_matches_disk": (
            external_files["results"]["sha256"]
            == sha256(EXTERNAL_RECHECK / "toy_pattern_equivalence.csv")
        ),
        "external_results_byte_identical_to_frozen": (
            sha256(EXTERNAL_RECHECK / "toy_pattern_equivalence.csv")
            == sha256(results_path)
        ),
    }

    source = V08_RUNNER.read_text(encoding="utf-8")
    source_checks = {
        "no_gurobi_mip_start_assignment": (
            ".Start" not in source
            and "MIPStart" not in source
            and "setAttr(\"Start\"" not in source
        ),
        "production_backend_pending": (
            qa["production_obbt_backend_status"]
            == "pending_not_implemented"
        ),
        "optimization_not_authorized": (
            qa["optimization_authorized"] is False
            and manifest["optimization_authorized"] is False
        ),
        "manuscript_evidence_not_admitted": (
            qa["manuscript_evidence_admitted"] is False
            and manifest["manuscript_evidence_admitted"] is False
        ),
        "full_year_not_run": (
            qa["full_year_benchmark_status"] == "full_year_not_run"
            and manifest["full_year_benchmark_status"]
            == "full_year_not_run"
            and int(results["horizon_h"].max()) <= 24
        ),
    }

    all_requested_attacks_safe = all(
        item["safe_no_fixing"] for item in attacks.values()
    )
    core_checks = {
        "valid_multivariable_certificate_admitted": verification.admitted,
        "scipy_optimum_matches_six": abs(scipy_raw_objective - 6.0)
        <= 1.0e-9,
        "independent_box_bound_matches_six": abs(
            independent_box_bound - 6.0
        )
        <= 1.0e-12,
        "independent_kkt_bound_matches_six": abs(
            independent_kkt_bound - 6.0
        )
        <= 1.0e-12,
        "verifier_bounds_match_independent_formula": (
            abs(
                float(verification.recomputed_box_dual_lower_bound)
                - independent_box_bound
            )
            <= 1.0e-12
            and abs(
                float(verification.recomputed_kkt_dual_objective)
                - independent_kkt_bound
            )
            <= 1.0e-12
        ),
        "valid_certificate_can_fix_expected_opposite_mode": (
            valid_decision.fixed_mode == 1
        ),
        "all_requested_declared_payload_attacks_safe": all_requested_attacks_safe,
        "negative_and_nan_proof_guard_hard_stop": all(
            item["hard_stop"] for item in guard_results.values()
        ),
        "both_excluded_hard_stop": both_excluded["hard_stop"],
        "caller_scalar_lower_bound_ignored": (
            spoof_decision.fixed_mode is None
            and spoof_decision.branch_zero_recomputed_lower_bound == 4.0
            and spoof_decision.branch_one_recomputed_lower_bound == 4.0
        ),
        "complete_pattern_inventory_512": (
            inventory_pass and len(results) == 512
        ),
        "toy_csv_byte_identical_external_recheck": (
            sha256(results_path)
            == sha256(EXTERNAL_RECHECK / "toy_pattern_equivalence.csv")
        ),
        "external_manifest_current_runner": all(
            external_manifest_checks.values()
        ),
        "environment_ledger_complete": all(
            key in manifest["environment"]
            for key in (
                "python",
                "numpy",
                "pandas",
                "scipy",
                "highs",
                "gurobi",
            )
        ),
        "mip_start_not_fixed": source_checks[
            "no_gurobi_mip_start_assignment"
        ],
        "production_backend_pending": source_checks[
            "production_backend_pending"
        ],
        "no_full_year_and_no_evidence_admission": all(
            source_checks[key]
            for key in (
                "optimization_not_authorized",
                "manuscript_evidence_not_admitted",
                "full_year_not_run",
            )
        ),
        "recomputed_nonfinite_values_fail_closed": overflow_result[
            "fail_closed"
        ],
    }
    overall_status = (
        "pass" if all(core_checks.values()) else "fail"
    )
    payload = {
        "audit_scope": (
            "Stage1 v0.8 toy verifier only; no 8,760 h instance loaded or solved"
        ),
        "status": overall_status,
        "authorization": {
            "de_2023_0x_1x_2x_obbt_disabled_non_evidential_benchmark": True,
            "de_2023_certificate_enabled_benchmark": False,
            "country_year_132_formal_optimization": False,
            "manuscript_evidence": False,
        },
        "blocking_finding": {
            "id": "V08-NONFINITE-RECOMPUTATION-FAIL-OPEN",
            "description": (
                "Finite payload scalars can overflow during matrix products; "
                "NaN recomputed primal/complementarity residuals are admitted "
                "because the verifier does not explicitly reject non-finite "
                "recomputed diagnostics."
            ),
            "minimal_reproduction": overflow_result,
        },
        "core_checks": core_checks,
        "formula_recheck": {
            "scipy_success": bool(scipy_solution.success),
            "scipy_raw_objective": scipy_raw_objective,
            "independent_box_bound": independent_box_bound,
            "independent_kkt_bound": independent_kkt_bound,
            "verification": asdict(verification),
        },
        "attacks": attacks,
        "guard_tests": guard_results,
        "both_excluded_test": both_excluded,
        "claimed_scalar_test": asdict(spoof_decision),
        "pattern_inventory": inventory,
        "external_manifest_checks": external_manifest_checks,
        "source_and_status_checks": source_checks,
        "environment": manifest["environment"],
        "frozen_hashes": {
            "runner": sha256(V08_RUNNER),
            "design": sha256(V08_DESIGN),
            "toy_results": sha256(results_path),
            "adversarial_tests": sha256(
                V08_OUTPUT / "obbt_adversarial_unit_tests.json"
            ),
            "qa": sha256(qa_path),
            "manifest": sha256(manifest_path),
            "external_recheck_results": sha256(
                EXTERNAL_RECHECK / "toy_pattern_equivalence.csv"
            ),
            "external_recheck_qa": sha256(EXTERNAL_RECHECK / "QA.json"),
            "external_recheck_manifest": sha256(external_manifest_path),
        },
    }
    output.write_text(
        json.dumps(
            json_ready(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_ready(payload["core_checks"]), indent=2))
    print(f"status={overall_status}")
    print(f"output={output}")
    if overall_status != "fail":
        raise SystemExit(
            "Independent audit expected the documented fail-open exploit"
        )


if __name__ == "__main__":
    main()
