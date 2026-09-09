#!/usr/bin/env python3
"""Stage 1 v0.9 fail-closed numerical certificate-consumer toy audit.

This append-only repair imports the SHA-locked v0.8/v0.7 toy formulation.  It
wraps every v0.8 certificate recomputation in a NumPy floating-point exception
boundary and rejects every admitted certificate unless every recomputed
numeric field is finite.  It exercises only 4--24 h toy instances and never
loads or solves an 8,760 h country-year.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import scipy

import run_observed_gas_event_task_stage1_exact_reformulation_toy_v0_8 as v08


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VERSION = "0.9.0"
EVIDENCE_STATUS = "non_evidential_stage1_exact_reformulation_toy"

V08_RUNNER = (
    HERE
    / "run_observed_gas_event_task_stage1_exact_reformulation_toy_v0_8.py"
)
V07_RUNNER = (
    HERE
    / "run_observed_gas_event_task_stage1_exact_reformulation_toy_v0_7.py"
)
V08_AUDIT = (
    ROOT
    / "artifacts"
    / "research"
    / "OBSERVED_GAS_EVENT_TASK_STAGE1_EXACT_REFORMULATION_V0_8_INDEPENDENT_AUDIT.md"
)
V08_RESULTS = (
    ROOT
    / "artifacts"
    / "research"
    / "stage1_exact_reformulation_toy_v0_8"
    / "toy_pattern_equivalence.csv"
)
V07_RESULTS = (
    ROOT
    / "artifacts"
    / "research"
    / "stage1_exact_reformulation_toy_v0_7"
    / "toy_pattern_equivalence.csv"
)
DESIGN = (
    ROOT
    / "artifacts"
    / "research"
    / "OBSERVED_GAS_EVENT_TASK_STAGE1_EXACT_REFORMULATION_DESIGN_V0_9.md"
)
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts"
    / "research"
    / "stage1_exact_reformulation_toy_v0_9"
)

V08_RUNNER_EXPECTED_SHA256 = (
    "71f0ee2b78225f916ecb953b24f9604dc5858e023c893dde5b8f04148f12c771"
)
V07_RUNNER_EXPECTED_SHA256 = (
    "6ad07e6fb63eae15275a54155730a82f63d22bec8ba30d6f5363ec89fbe9c677"
)
V08_RESULTS_EXPECTED_SHA256 = (
    "494d0476c7d3c004bba30d94f982496044671a87944000f51a9cf9f6ff9c97bf"
)
V07_RESULTS_EXPECTED_SHA256 = V08_RESULTS_EXPECTED_SHA256

_V08_VERIFY = v08.verify_dual_primal_certificate
_RECOMPUTED_FINITE_FIELDS = (
    "recomputed_primal_residual",
    "recomputed_dual_feasibility_residual",
    "recomputed_complementarity_residual",
    "recomputed_primal_objective",
    "recomputed_kkt_dual_objective",
    "recomputed_box_dual_lower_bound",
    "recomputed_primal_dual_gap",
)


def _finite_array(value: Any, context: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise v08.CertificateError(
            f"{context} cannot be represented as a numeric array"
        ) from exc
    if not bool(np.all(np.isfinite(array))):
        raise v08.CertificateError(f"{context} contains a nonfinite value")
    return array


def _finite_result(value: Any, context: str) -> Any:
    array = np.asarray(value)
    if not bool(np.all(np.isfinite(array))):
        raise v08.CertificateError(
            f"{context} recomputation produced a nonfinite value"
        )
    return value


def _finite_recomputation_preflight(certificate: Any) -> None:
    """Explicitly finite-audit every numerical intermediate used by v0.8."""

    if not isinstance(certificate, Mapping):
        raise v08.CertificateError("certificate must be a mapping")
    model = certificate.get("model")
    objective = certificate.get("objective")
    primal = certificate.get("primal")
    dual = certificate.get("dual")
    if not all(
        isinstance(item, Mapping)
        for item in (model, objective, primal, dual)
    ):
        raise v08.CertificateError(
            "model, objective, primal and dual must be mappings"
        )

    variable_names = model.get("variable_names")
    if not isinstance(variable_names, (list, tuple)):
        raise v08.CertificateError("model.variable_names must be a sequence")
    n = len(variable_names)
    if n == 0:
        raise v08.CertificateError("model must contain variables")

    a_ub = _finite_array(model.get("A_ub"), "model.A_ub")
    b_ub = _finite_array(model.get("b_ub"), "model.b_ub")
    a_eq = _finite_array(model.get("A_eq"), "model.A_eq")
    b_eq = _finite_array(model.get("b_eq"), "model.b_eq")
    if a_ub.size == 0:
        a_ub = np.zeros((0, n), dtype=float)
    if a_eq.size == 0:
        a_eq = np.zeros((0, n), dtype=float)
    if a_ub.ndim != 2 or a_ub.shape[1] != n:
        raise v08.CertificateError("model.A_ub has invalid dimensions")
    if a_eq.ndim != 2 or a_eq.shape[1] != n:
        raise v08.CertificateError("model.A_eq has invalid dimensions")
    if b_ub.shape != (a_ub.shape[0],):
        raise v08.CertificateError("model.b_ub has invalid dimensions")
    if b_eq.shape != (a_eq.shape[0],):
        raise v08.CertificateError("model.b_eq has invalid dimensions")

    lower = _finite_array(model.get("lower_bounds"), "model.lower_bounds")
    upper = _finite_array(model.get("upper_bounds"), "model.upper_bounds")
    coefficients = _finite_array(
        objective.get("coefficients"), "objective.coefficients"
    )
    x = _finite_array(primal.get("values"), "primal.values")
    lam_raw = _finite_array(
        dual.get("inequality_multipliers"),
        "dual.inequality_multipliers",
    )
    mu = _finite_array(
        dual.get("equality_multipliers"),
        "dual.equality_multipliers",
    )
    alpha_raw = _finite_array(
        dual.get("lower_bound_multipliers"),
        "dual.lower_bound_multipliers",
    )
    beta_raw = _finite_array(
        dual.get("upper_bound_multipliers"),
        "dual.upper_bound_multipliers",
    )
    for array, context, expected in (
        (lower, "model.lower_bounds", n),
        (upper, "model.upper_bounds", n),
        (coefficients, "objective.coefficients", n),
        (x, "primal.values", n),
        (lam_raw, "dual.inequality_multipliers", a_ub.shape[0]),
        (mu, "dual.equality_multipliers", a_eq.shape[0]),
        (alpha_raw, "dual.lower_bound_multipliers", n),
        (beta_raw, "dual.upper_bound_multipliers", n),
    ):
        if array.shape != (expected,):
            raise v08.CertificateError(f"{context} has invalid dimensions")

    constant = v08._finite_float(
        objective.get("constant"), "objective.constant"
    )
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        ub_activity = _finite_result(a_ub @ x, "A_ub @ x")
        ub_signed = _finite_result(
            np.subtract(ub_activity, b_ub), "A_ub @ x - b_ub"
        )
        ub_positive = _finite_result(
            np.maximum(ub_signed, 0.0), "inequality positive violation"
        )
        ub_violation = _finite_result(
            np.max(ub_positive) if ub_positive.size else np.float64(0.0),
            "maximum inequality violation",
        )

        eq_activity = _finite_result(a_eq @ x, "A_eq @ x")
        eq_signed = _finite_result(
            np.subtract(eq_activity, b_eq), "A_eq @ x - b_eq"
        )
        eq_absolute = _finite_result(
            np.abs(eq_signed), "absolute equality violation"
        )
        eq_violation = _finite_result(
            np.max(eq_absolute) if eq_absolute.size else np.float64(0.0),
            "maximum equality violation",
        )

        lower_signed = _finite_result(
            np.subtract(lower, x), "lower - x"
        )
        upper_signed = _finite_result(
            np.subtract(x, upper), "x - upper"
        )
        lower_violation = _finite_result(
            np.max(np.maximum(lower_signed, 0.0)),
            "maximum lower-bound violation",
        )
        upper_violation = _finite_result(
            np.max(np.maximum(upper_signed, 0.0)),
            "maximum upper-bound violation",
        )
        _finite_result(
            np.max(
                np.asarray(
                    [
                        ub_violation,
                        eq_violation,
                        lower_violation,
                        upper_violation,
                    ],
                    dtype=float,
                )
            ),
            "primal residual",
        )

        objective_dot = _finite_result(
            coefficients @ x, "objective coefficients @ x"
        )
        _finite_result(
            np.add(np.float64(constant), objective_dot),
            "primal objective",
        )

        lam = _finite_result(
            np.maximum(lam_raw, 0.0), "nonnegative inequality multipliers"
        )
        alpha = _finite_result(
            np.maximum(alpha_raw, 0.0), "nonnegative lower multipliers"
        )
        beta = _finite_result(
            np.maximum(beta_raw, 0.0), "nonnegative upper multipliers"
        )
        sign_terms = _finite_result(
            np.concatenate(
                (
                    np.maximum(-lam_raw, 0.0),
                    np.maximum(-alpha_raw, 0.0),
                    np.maximum(-beta_raw, 0.0),
                )
            ),
            "dual sign residual terms",
        )
        _finite_result(
            np.max(sign_terms) if sign_terms.size else np.float64(0.0),
            "dual sign residual",
        )

        q = coefficients.copy()
        _finite_result(q, "initial stationarity vector")
        if a_ub.shape[0]:
            ub_dual_term = _finite_result(
                a_ub.T @ lam, "A_ub.T @ lambda"
            )
            q = _finite_result(
                np.add(q, ub_dual_term), "stationarity after inequalities"
            )
        if a_eq.shape[0]:
            eq_dual_term = _finite_result(a_eq.T @ mu, "A_eq.T @ mu")
            q = _finite_result(
                np.add(q, eq_dual_term), "stationarity after equalities"
            )
        stationarity = _finite_result(
            np.add(np.subtract(q, alpha), beta),
            "stationarity vector",
        )
        _finite_result(
            np.max(np.abs(stationarity)), "stationarity residual"
        )

        slack_ub = _finite_result(
            np.subtract(b_ub, ub_activity), "inequality slack"
        )
        ub_complementarity = _finite_result(
            np.multiply(lam, slack_ub), "inequality complementarity"
        )
        lower_distance = _finite_result(
            np.subtract(x, lower), "lower-bound distance"
        )
        upper_distance = _finite_result(
            np.subtract(upper, x), "upper-bound distance"
        )
        lower_complementarity = _finite_result(
            np.multiply(alpha, lower_distance),
            "lower-bound complementarity",
        )
        upper_complementarity = _finite_result(
            np.multiply(beta, upper_distance),
            "upper-bound complementarity",
        )
        complementarity_terms = _finite_result(
            np.concatenate(
                (
                    np.abs(ub_complementarity),
                    np.abs(lower_complementarity),
                    np.abs(upper_complementarity),
                )
            ),
            "complementarity residual terms",
        )
        _finite_result(
            np.max(complementarity_terms),
            "complementarity residual",
        )

        ub_dual_dot = _finite_result(b_ub @ lam, "b_ub @ lambda")
        eq_dual_dot = _finite_result(b_eq @ mu, "b_eq @ mu")
        base_dual = _finite_result(
            np.subtract(
                np.subtract(np.float64(constant), ub_dual_dot),
                eq_dual_dot,
            ),
            "base dual objective",
        )
        lower_dual_dot = _finite_result(lower @ alpha, "lower @ alpha")
        upper_dual_dot = _finite_result(upper @ beta, "upper @ beta")
        kkt_dual = _finite_result(
            np.subtract(np.add(base_dual, lower_dual_dot), upper_dual_dot),
            "KKT dual objective",
        )
        q_lower = _finite_result(np.multiply(q, lower), "q * lower")
        q_upper = _finite_result(np.multiply(q, upper), "q * upper")
        box_terms = _finite_result(
            np.minimum(q_lower, q_upper), "box-infimum terms"
        )
        box_infimum = _finite_result(
            np.sum(box_terms), "box infimum"
        )
        box_dual = _finite_result(
            np.add(base_dual, box_infimum), "box dual lower bound"
        )
        _finite_result(
            np.subtract(kkt_dual, box_dual),
            "KKT versus box-dual difference",
        )
        # Recompute the primal objective once more here so the gap is audited
        # directly from finite components rather than from caller scalars.
        primal_objective = _finite_result(
            np.add(np.float64(constant), objective_dot),
            "primal objective for gap",
        )
        _finite_result(
            np.subtract(primal_objective, box_dual),
            "primal-dual gap",
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            json_ready(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def portable_path(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        value = str(resolved.relative_to(ROOT))
        kind = "repository_relative"
    except ValueError:
        value = str(resolved)
        kind = "absolute_external"
    return {"path": value, "path_kind": kind}


def manifest_path_entry(path: Path) -> dict[str, str]:
    entry = portable_path(path)
    entry["sha256"] = sha256(path.resolve())
    return entry


def _package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed_in_runtime"


def detect_highs_version() -> dict[str, str]:
    code = (
        "from scipy.optimize import linprog;"
        "linprog([1.0],bounds=[(0.0,1.0)],method='highs',"
        "options={'disp':True})"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        banner = completed.stdout + "\n" + completed.stderr
        match = re.search(r"Running HiGHS\s+([^\s]+)", banner)
        version = match.group(1) if match else "runtime_banner_not_detected"
        first_line = next(
            (
                line.strip()
                for line in banner.splitlines()
                if "Running HiGHS" in line
            ),
            "runtime_banner_not_detected",
        )
        return {"version": version, "runtime_banner": first_line}
    except Exception as exc:
        return {
            "version": "runtime_detection_failed",
            "runtime_banner": f"{type(exc).__name__}: {exc}",
        }


def detect_gurobi_version() -> dict[str, str]:
    package = _package_version("gurobipy")
    try:
        import gurobipy as gp

        version = ".".join(str(part) for part in gp.gurobi.version())
        return {"gurobipy": package, "gurobi": version}
    except Exception as exc:
        executable = shutil.which("gurobi_cl")
        if executable:
            try:
                completed = subprocess.run(
                    [executable, "--version"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                banner = (completed.stdout + "\n" + completed.stderr).strip()
                match = re.search(
                    r"Gurobi Optimizer version\s+([^\s]+)", banner
                )
                if match:
                    return {
                        "gurobipy": package,
                        "gurobi": match.group(1),
                        "gurobi_cl": executable,
                        "runtime_banner": banner.splitlines()[0],
                        "python_binding_detail": f"{type(exc).__name__}: {exc}",
                    }
            except Exception:
                pass
        return {
            "gurobipy": package,
            "gurobi": "not_available_in_runtime",
            "detail": f"{type(exc).__name__}: {exc}",
        }


def verify_dual_primal_certificate(
    certificate: Any,
    **kwargs: Any,
) -> v08.VerifiedCertificate:
    """Fail closed on every floating-point exception or nonfinite recomputation.

    The complete v0.8 verifier is executed inside one numerical exception
    boundary.  This covers all matrix products, dot products, residuals,
    objectives, dual bounds and gaps performed by the frozen verifier.  The
    final field audit is deliberately redundant: no admitted result can carry
    a None, NaN or infinity in any recomputed numeric field.
    """

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            _finite_recomputation_preflight(certificate)
            verified = _V08_VERIFY(certificate, **kwargs)
    except Exception as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return v08._failed_verification(
            f"fail-closed numerical exception: {type(exc).__name__}: {exc}"
        )

    if not verified.admitted:
        return verified
    for field in _RECOMPUTED_FINITE_FIELDS:
        value = getattr(verified, field)
        if value is None:
            return v08._failed_verification(
                f"admitted certificate has missing recomputed field {field}"
            )
        try:
            finite = bool(np.isfinite(float(value)))
        except (TypeError, ValueError, OverflowError):
            finite = False
        if not finite:
            return v08._failed_verification(
                f"admitted certificate has nonfinite recomputed field {field}"
            )
    return verified


# The frozen v0.8 objective-cutoff routine resolves this module-global symbol
# at call time.  Replacing it here hardens all v0.9 OBBT calls without changing
# the SHA-locked v0.8 file.
v08.verify_dual_primal_certificate = verify_dual_primal_certificate


def _rehash_certificate(certificate: dict[str, Any]) -> None:
    model = certificate["model"]
    objective = certificate["objective"]
    constraints_payload = {
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
    bounds_payload = {
        key: model[key]
        for key in ("variable_names", "lower_bounds", "upper_bounds")
    }
    certificate["model_sha256"] = v08.canonical_sha256(model)
    certificate["constraints_sha256"] = v08.canonical_sha256(
        constraints_payload
    )
    certificate["variable_bounds_sha256"] = v08.canonical_sha256(
        bounds_payload
    )
    certificate["objective_sha256"] = v08.canonical_sha256(objective)


def _two_variable_certificate(
    hour: int,
    mode: int,
    *,
    model_name: str,
    a_ub: list[list[float]],
    b_ub: list[float],
    a_eq: list[list[float]],
    b_eq: list[float],
    lower: list[float],
    upper: list[float],
    coefficients: list[float],
    constant: float,
    x: list[float],
    lam: list[float],
    mu: list[float],
    alpha: list[float],
    beta: list[float],
    declared_primal_objective: float,
    declared_dual_objective: float,
) -> dict[str, Any]:
    """Construct a finite-payload adversarial proof and lock its hashes."""

    model = {
        "schema_version": "lp-proof-model-v0.8",
        "model_name": model_name,
        "mode_hour": int(hour),
        "fixed_mode": int(mode),
        "variable_names": ["x", "y"],
        "inequality_names": [
            f"ub_{index}" for index in range(len(a_ub))
        ],
        "equality_names": [
            f"eq_{index}" for index in range(len(a_eq))
        ],
        "A_ub": a_ub,
        "b_ub": b_ub,
        "A_eq": a_eq,
        "b_eq": b_eq,
        "lower_bounds": lower,
        "upper_bounds": upper,
    }
    objective = {
        "schema_version": "lp-proof-objective-v0.8",
        "sense": "min",
        "variable_names": ["x", "y"],
        "coefficients": coefficients,
        "constant": constant,
    }
    certificate = {
        "schema_version": "dual-primal-lp-certificate-v0.8",
        "source": "analytic_toy_dual",
        "solver": "adversarial_finite_payload",
        "solver_version": "1.0",
        "certificate_id": f"{model_name}-{hour}-{mode}",
        "mode_hour": int(hour),
        "fixed_mode": int(mode),
        "objective_cutoff": 5.0,
        "solver_status": {
            "status": "optimal",
            "code": 0,
            "message": "adversarial finite payload",
        },
        "model": model,
        "objective": objective,
        "primal": {
            "values": x,
            "objective": declared_primal_objective,
            "max_primal_residual": 0.0,
        },
        "dual": {
            "inequality_multipliers": lam,
            "equality_multipliers": mu,
            "lower_bound_multipliers": alpha,
            "upper_bound_multipliers": beta,
            "objective": declared_dual_objective,
            "dual_feasibility_residual": 0.0,
        },
    }
    _rehash_certificate(certificate)
    return certificate


def _finite_overflow_certificates() -> dict[str, dict[str, Any]]:
    """Return finite scalar payloads that overflow at distinct recomputations."""

    return {
        "finite_input_aub_primal_overflow": _two_variable_certificate(
            200,
            0,
            model_name="finite-input-aub-primal-overflow",
            a_ub=[[1.0e308, -1.0e308]],
            b_ub=[0.0],
            a_eq=[],
            b_eq=[],
            lower=[0.0, 0.0],
            upper=[2.0, 2.0],
            coefficients=[0.0, 0.0],
            constant=6.0,
            x=[2.0, 1.9],
            lam=[0.0],
            mu=[],
            alpha=[0.0, 0.0],
            beta=[0.0, 0.0],
            declared_primal_objective=6.0,
            declared_dual_objective=6.0,
        ),
        "finite_input_aeq_primal_overflow": _two_variable_certificate(
            201,
            0,
            model_name="finite-input-aeq-primal-overflow",
            a_ub=[],
            b_ub=[],
            a_eq=[[1.0e308, -1.0e308]],
            b_eq=[0.0],
            lower=[0.0, 0.0],
            upper=[2.0, 2.0],
            coefficients=[0.0, 0.0],
            constant=6.0,
            x=[2.0, 1.9],
            lam=[],
            mu=[0.0],
            alpha=[0.0, 0.0],
            beta=[0.0, 0.0],
            declared_primal_objective=6.0,
            declared_dual_objective=6.0,
        ),
        "finite_input_objective_dot_overflow": _two_variable_certificate(
            202,
            0,
            model_name="finite-input-objective-dot-overflow",
            a_ub=[],
            b_ub=[],
            a_eq=[],
            b_eq=[],
            lower=[0.0, 0.0],
            upper=[2.0, 2.0],
            coefficients=[1.0e308, 1.0e308],
            constant=0.0,
            x=[2.0, 2.0],
            lam=[],
            mu=[],
            alpha=[0.0, 0.0],
            beta=[0.0, 0.0],
            declared_primal_objective=6.0,
            declared_dual_objective=6.0,
        ),
        "finite_input_stationarity_overflow": _two_variable_certificate(
            203,
            0,
            model_name="finite-input-stationarity-overflow",
            a_ub=[[1.0e308, 0.0], [1.0e308, 0.0]],
            b_ub=[0.0, 0.0],
            a_eq=[],
            b_eq=[],
            lower=[0.0, 0.0],
            upper=[2.0, 2.0],
            coefficients=[0.0, 0.0],
            constant=6.0,
            x=[0.0, 0.0],
            lam=[2.0, 2.0],
            mu=[],
            alpha=[0.0, 0.0],
            beta=[0.0, 0.0],
            declared_primal_objective=6.0,
            declared_dual_objective=6.0,
        ),
        "finite_input_complementarity_overflow": _two_variable_certificate(
            204,
            0,
            model_name="finite-input-complementarity-overflow",
            a_ub=[[0.0, 0.0]],
            b_ub=[2.0],
            a_eq=[],
            b_eq=[],
            lower=[0.0, 0.0],
            upper=[2.0, 2.0],
            coefficients=[0.0, 0.0],
            constant=6.0,
            x=[0.0, 0.0],
            lam=[1.0e308],
            mu=[],
            alpha=[0.0, 0.0],
            beta=[0.0, 0.0],
            declared_primal_objective=6.0,
            declared_dual_objective=6.0,
        ),
        "finite_input_dual_bound_overflow": _two_variable_certificate(
            205,
            0,
            model_name="finite-input-dual-bound-overflow",
            a_ub=[[1.0, 0.0]],
            b_ub=[2.0],
            a_eq=[],
            b_eq=[],
            lower=[2.0, 0.0],
            upper=[2.0, 2.0],
            coefficients=[0.0, 0.0],
            constant=6.0,
            x=[2.0, 0.0],
            lam=[1.0e308],
            mu=[],
            alpha=[1.0e308, 0.0],
            beta=[0.0, 0.0],
            declared_primal_objective=6.0,
            declared_dual_objective=6.0,
        ),
    }


def _direct_and_decision_attack(
    name: str,
    malicious: dict[str, Any],
    *,
    mutate_without_rehash: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    hour = int(malicious["mode_hour"])
    supplied = {
        (hour, 0): copy.deepcopy(malicious),
        (hour, 1): v08.make_analytic_certificate(
            hour, 1, 4.0, objective_cutoff=5.0
        ),
    }
    if mutate_without_rehash is not None:
        mutate_without_rehash(supplied[(hour, 0)])
    expected_models, expected_objectives = v08._certificate_hash_contract(
        supplied
    )
    direct = verify_dual_primal_certificate(
        supplied[(hour, 0)],
        expected_mode_hour=hour,
        expected_fixed_mode=0,
        expected_model_sha256=expected_models[v08._branch_key(hour, 0)],
        expected_objective_sha256=expected_objectives[
            v08._branch_key(hour, 0)
        ],
        expected_objective_cutoff=5.0,
        allowed_sources={"analytic_toy_dual"},
        primal_tolerance=v08.PRIMAL_FEASIBILITY_TOLERANCE,
        dual_tolerance=v08.DUAL_FEASIBILITY_TOLERANCE,
        objective_tolerance=v08.OBJECTIVE_ACCOUNTING_TOLERANCE,
        optimality_tolerance=v08.OPTIMALITY_TOLERANCE,
    )
    try:
        decision = v08.objective_cutoff_obbt(
            mode_hours=(hour,),
            incumbent_upper_bound=5.0,
            expected_model_hashes=expected_models,
            expected_objective_hashes=expected_objectives,
            certificate_provider=lambda h, m: supplied[(h, m)],
            allowed_sources={"analytic_toy_dual"},
        )[0]
        safe = not direct.admitted and decision.fixed_mode is None
        return {
            "name": name,
            "direct_rejected": not direct.admitted,
            "safe_unfixed": decision.fixed_mode is None,
            "pass": safe,
            "verification": asdict(direct),
            "decision": asdict(decision),
        }
    except Exception as exc:
        return {
            "name": name,
            "direct_rejected": not direct.admitted,
            "hard_stop_before_fixing": True,
            "pass": not direct.admitted,
            "verification": asdict(direct),
            "exception": f"{type(exc).__name__}: {exc}",
        }


def run_numerical_fail_closed_tests() -> dict[str, Any]:
    tests: dict[str, dict[str, Any]] = {}
    finite_payloads = _finite_overflow_certificates()
    for name, certificate in finite_payloads.items():
        tests[name] = _direct_and_decision_attack(name, certificate)

    base_inf = v08.make_analytic_certificate(
        220, 0, 6.0, objective_cutoff=5.0
    )
    tests["explicit_inf_matrix_payload"] = _direct_and_decision_attack(
        "explicit_inf_matrix_payload",
        base_inf,
        mutate_without_rehash=lambda cert: cert["model"]["A_ub"][0].__setitem__(
            0, float("inf")
        ),
    )
    base_nan = v08.make_analytic_certificate(
        221, 0, 6.0, objective_cutoff=5.0
    )
    tests["explicit_nan_primal_payload"] = _direct_and_decision_attack(
        "explicit_nan_primal_payload",
        base_nan,
        mutate_without_rehash=lambda cert: cert["primal"]["values"].__setitem__(
            0, float("nan")
        ),
    )
    base_inf_dual = v08.make_analytic_certificate(
        222, 0, 6.0, objective_cutoff=5.0
    )
    tests["explicit_inf_dual_payload"] = _direct_and_decision_attack(
        "explicit_inf_dual_payload",
        base_inf_dual,
        mutate_without_rehash=lambda cert: cert["dual"][
            "inequality_multipliers"
        ].__setitem__(0, float("inf")),
    )

    finite_overflow_tests = [
        value
        for key, value in tests.items()
        if key.startswith("finite_input_")
    ]
    direct_nonfinite_tests = [
        value
        for key, value in tests.items()
        if key.startswith("explicit_")
    ]
    return {
        "status": (
            "pass" if tests and all(item["pass"] for item in tests.values())
            else "fail"
        ),
        "np_errstate_contract": {
            "over": "raise",
            "invalid": "raise",
            "divide": "raise",
        },
        "finite_input_overflow_attack_count": len(finite_overflow_tests),
        "direct_nonfinite_attack_count": len(direct_nonfinite_tests),
        "all_finite_input_overflow_attacks_fail_closed": all(
            item["pass"] for item in finite_overflow_tests
        ),
        "all_explicit_nonfinite_attacks_fail_closed": all(
            item["pass"] for item in direct_nonfinite_tests
        ),
        "tests": tests,
    }


def pattern_inventory_check(results: pd.DataFrame) -> dict[str, Any]:
    details: dict[str, Any] = {}
    passed = True
    for horizon in v08.v07.HORIZONS:
        subset = results.loc[results["horizon_h"] == horizon]
        expected_count = 2 ** v08.v07.OVERLAP_COUNTS[horizon]
        expected = set(range(expected_count))
        observed_list = [
            int(value) for value in subset["pattern_index"].tolist()
        ]
        observed = set(observed_list)
        item_pass = (
            len(observed_list) == expected_count
            and len(observed) == expected_count
            and observed == expected
        )
        passed = passed and item_pass
        details[str(horizon)] = {
            "expected_count": expected_count,
            "observed_count": len(observed_list),
            "unique_count": len(observed),
            "missing": sorted(expected - observed),
            "unexpected": sorted(observed - expected),
            "pass": item_pass,
        }
    return {"pass": passed, "by_horizon": details}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(
            f"Append-only v0.9 output already exists and is non-empty: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for horizon in v08.v07.HORIZONS:
        instance = v08.v07.generate_toy_instance(horizon)
        for pattern_index, pattern in enumerate(
            v08.v07.mode_patterns(len(instance.overlap_hours))
        ):
            rows.append(
                v08.v07.audit_pattern(instance, pattern_index, pattern)
            )
    results = pd.DataFrame(rows)
    results_path = output / "toy_pattern_equivalence.csv"
    results.to_csv(results_path, index=False, float_format="%.12g")

    inherited_obbt = v08.run_obbt_unit_tests()
    inherited_obbt_path = output / "inherited_obbt_adversarial_tests.json"
    write_json(inherited_obbt_path, inherited_obbt)
    numerical = run_numerical_fail_closed_tests()
    numerical_path = output / "nonfinite_overflow_adversarial_tests.json"
    write_json(numerical_path, numerical)

    maxima = {
        column: float(pd.to_numeric(results[column]).max())
        for column in (
            "objective_absolute_difference",
            "reference_to_hull_max_residual",
            "hull_to_reference_max_residual",
            "reference_simultaneous_mw",
            "hull_simultaneous_mw",
            "lex_charge_max_difference_mw",
            "lex_discharge_max_difference_mw",
            "lex_shifted_soc_max_difference_mwh",
            "lex_absolute_soc_max_difference_mwh",
            "raw_centered_ledger_difference",
        )
    }
    expected_patterns = sum(
        2 ** v08.v07.OVERLAP_COUNTS[n] for n in v08.v07.HORIZONS
    )
    pattern_inventory = pattern_inventory_check(results)
    path_probe = manifest_path_entry(results_path)
    outside_probe_path = Path("/tmp/stage1_v0_9_manifest_probe.json")
    outside_probe_payload = portable_path(outside_probe_path)
    external_path_logic_pass = (
        not str(outside_probe_path.resolve()).startswith(str(ROOT) + "/")
        and outside_probe_payload["path_kind"] == "absolute_external"
    )
    checks = {
        "v08_runner_dependency_sha_locked": (
            sha256(V08_RUNNER) == V08_RUNNER_EXPECTED_SHA256
        ),
        "v07_runner_dependency_sha_locked": (
            sha256(V07_RUNNER) == V07_RUNNER_EXPECTED_SHA256
        ),
        "v08_results_sha_locked": (
            sha256(V08_RESULTS) == V08_RESULTS_EXPECTED_SHA256
        ),
        "v07_results_sha_locked": (
            sha256(V07_RESULTS) == V07_RESULTS_EXPECTED_SHA256
        ),
        "only_toy_horizons_4_to_24": (
            sorted(results["horizon_h"].unique().tolist())
            == list(v08.v07.HORIZONS)
        ),
        "all_expected_patterns_enumerated": (
            len(results) == expected_patterns
        ),
        "pattern_sets_exact_by_horizon": pattern_inventory["pass"],
        "all_reference_models_solved": bool(
            results["reference_success"].all()
        ),
        "all_hull_models_solved": bool(results["hull_success"].all()),
        "all_pattern_rows_pass": bool(results["pass"].all()),
        "objective_equivalence": (
            maxima["objective_absolute_difference"]
            <= v08.v07.OBJECTIVE_TOLERANCE
        ),
        "reference_to_hull_mapping_feasible": (
            maxima["reference_to_hull_max_residual"]
            <= v08.v07.FEASIBILITY_TOLERANCE
        ),
        "hull_to_reference_mapping_feasible": (
            maxima["hull_to_reference_max_residual"]
            <= v08.v07.FEASIBILITY_TOLERANCE
        ),
        "reference_no_simultaneous_flow": (
            maxima["reference_simultaneous_mw"]
            <= v08.v07.SIMULTANEOUS_TOLERANCE
        ),
        "hull_no_simultaneous_flow": (
            maxima["hull_simultaneous_mw"]
            <= v08.v07.SIMULTANEOUS_TOLERANCE
        ),
        "deterministic_tiebreak_trajectory_equivalence": (
            max(
                maxima["lex_charge_max_difference_mw"],
                maxima["lex_discharge_max_difference_mw"],
                maxima["lex_shifted_soc_max_difference_mwh"],
                maxima["lex_absolute_soc_max_difference_mwh"],
            )
            <= v08.v07.TRAJECTORY_TOLERANCE
        ),
        "raw_centered_objective_ledger_equivalence": (
            maxima["raw_centered_ledger_difference"]
            <= v08.v07.OBJECTIVE_TOLERANCE
        ),
        "toy_results_byte_identical_to_v08": (
            sha256(results_path) == V08_RESULTS_EXPECTED_SHA256
        ),
        "toy_results_byte_identical_to_v07": (
            sha256(results_path) == V07_RESULTS_EXPECTED_SHA256
        ),
        "inherited_obbt_adversarial_tests": (
            inherited_obbt["status"] == "pass"
        ),
        "nonfinite_and_overflow_attacks_fail_closed": (
            numerical["status"] == "pass"
        ),
        "external_output_manifest_path_supported": external_path_logic_pass,
        "current_output_manifest_path_kind_valid": (
            path_probe["path_kind"]
            in {"repository_relative", "absolute_external"}
        ),
        "no_8760_hour_instance": int(results["horizon_h"].max()) <= 24,
        "mip_start_pattern_fixed_as_constraint": False,
    }
    status = (
        "pass"
        if all(
            value is True
            for key, value in checks.items()
            if key != "mip_start_pattern_fixed_as_constraint"
        )
        and checks["mip_start_pattern_fixed_as_constraint"] is False
        else "fail"
    )
    qa = {
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "evidence_status": EVIDENCE_STATUS,
        "toy_equivalence_status": (
            "toy_equivalence_pass" if status == "pass" else "fail"
        ),
        "optimization_authorized": False,
        "certificate_enabled_benchmark_authorized": False,
        "manuscript_evidence_admitted": False,
        "production_obbt_backend_status": "pending_not_implemented",
        "full_year_benchmark_status": "full_year_not_run",
        "horizons_h": list(v08.v07.HORIZONS),
        "expected_pattern_count": expected_patterns,
        "observed_pattern_count": len(results),
        "pattern_inventory": pattern_inventory,
        "maxima": maxima,
        "tolerances": {
            "primal_feasibility": v08.PRIMAL_FEASIBILITY_TOLERANCE,
            "dual_feasibility": v08.DUAL_FEASIBILITY_TOLERANCE,
            "objective_accounting": v08.OBJECTIVE_ACCOUNTING_TOLERANCE,
            "optimality": v08.OPTIMALITY_TOLERANCE,
            "obbt_proof_guard": v08.OBBT_PROOF_GUARD,
            "inherited_v07_feasibility": v08.v07.FEASIBILITY_TOLERANCE,
            "inherited_v07_objective": v08.v07.OBJECTIVE_TOLERANCE,
            "inherited_v07_trajectory": v08.v07.TRAJECTORY_TOLERANCE,
        },
        "checks": checks,
        "notes": [
            "The exact-projection and state-hull toy kernel is imported "
            "unchanged from SHA-locked v0.8/v0.7 dependencies.",
            "All v0.8 certificate matrix products, dot products, residuals, "
            "objectives, dual bounds and gaps execute inside np.errstate "
            "(over='raise', invalid='raise', divide='raise').",
            "Every admitted certificate receives a redundant finite-field "
            "audit immediately before it can reach OBBT.",
            "The independent-audit finite A_ub=[1e308,-1e308], "
            "x=[2,1.9] overflow is a frozen regression test.",
            "The production certificate exporter/backend remains pending.",
            "No 8,760-hour country-year was loaded or solved.",
        ],
    }
    qa_path = output / "QA.json"
    write_json(qa_path, qa)

    script_path = Path(__file__).resolve()
    manifest = {
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "evidence_status": EVIDENCE_STATUS,
        "optimization_authorized": False,
        "certificate_enabled_benchmark_authorized": False,
        "manuscript_evidence_admitted": False,
        "full_year_benchmark_status": "full_year_not_run",
        "files": {
            "runner": manifest_path_entry(script_path),
            "v08_runner_dependency": manifest_path_entry(V08_RUNNER),
            "v07_runner_dependency": manifest_path_entry(V07_RUNNER),
            "v08_independent_audit_basis": manifest_path_entry(V08_AUDIT),
            "design": manifest_path_entry(DESIGN),
            "results": manifest_path_entry(results_path),
            "inherited_obbt_adversarial_tests": manifest_path_entry(
                inherited_obbt_path
            ),
            "nonfinite_overflow_adversarial_tests": manifest_path_entry(
                numerical_path
            ),
            "qa": manifest_path_entry(qa_path),
        },
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "highs": detect_highs_version(),
            "gurobi": detect_gurobi_version(),
        },
    }
    manifest_path = output / "manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps(json_ready(qa), indent=2, ensure_ascii=False))
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
