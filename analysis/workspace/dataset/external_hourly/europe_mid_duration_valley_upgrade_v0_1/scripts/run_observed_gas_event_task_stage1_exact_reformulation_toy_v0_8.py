#!/usr/bin/env python3
"""Stage 1 v0.8 toy audit with independently verified OBBT certificates.

This append-only repair imports the frozen v0.7 toy formulation and reruns its
4--24 h equivalence suite.  It replaces only the OBBT certificate consumer and
the reproducibility ledger.  It never loads or solves an 8,760 h instance.
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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy

import run_observed_gas_event_task_stage1_exact_reformulation_toy_v0_7 as v07


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VERSION = "0.8.0"
EVIDENCE_STATUS = "non_evidential_stage1_exact_reformulation_toy"
V07_RUNNER = (
    HERE
    / "run_observed_gas_event_task_stage1_exact_reformulation_toy_v0_7.py"
)
V07_AUDIT = (
    ROOT
    / "artifacts"
    / "research"
    / "OBSERVED_GAS_EVENT_TASK_STAGE1_EXACT_REFORMULATION_V0_7_INDEPENDENT_AUDIT.md"
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
    / "OBSERVED_GAS_EVENT_TASK_STAGE1_EXACT_REFORMULATION_DESIGN_V0_8.md"
)
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts"
    / "research"
    / "stage1_exact_reformulation_toy_v0_8"
)

V07_RUNNER_EXPECTED_SHA256 = (
    "6ad07e6fb63eae15275a54155730a82f63d22bec8ba30d6f5363ec89fbe9c677"
)
V07_AUDIT_EXPECTED_SHA256 = (
    "d45df7eebb3c1b0eca62e4a397a76f1d629834deb742ea5b480097641dd910dd"
)
V07_RESULTS_EXPECTED_SHA256 = (
    "494d0476c7d3c004bba30d94f982496044671a87944000f51a9cf9f6ff9c97bf"
)

PRIMAL_FEASIBILITY_TOLERANCE = 1e-9
DUAL_FEASIBILITY_TOLERANCE = 1e-9
OBJECTIVE_ACCOUNTING_TOLERANCE = 1e-9
OPTIMALITY_TOLERANCE = 1e-8
OBBT_PROOF_GUARD = 1e-7


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


def manifest_path_entry(path: Path) -> dict[str, str]:
    entry = portable_path(path)
    entry["sha256"] = sha256(path.resolve())
    return entry


def portable_path(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        value = str(resolved.relative_to(ROOT))
        kind = "repository_relative"
    except ValueError:
        value = str(resolved)
        kind = "absolute_external"
    return {
        "path": value,
        "path_kind": kind,
    }


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
    except Exception as exc:  # fail-safe environment inventory
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
                        "python_binding_detail": (
                            f"{type(exc).__name__}: {exc}"
                        ),
                    }
            except Exception:
                pass
        return {
            "gurobipy": package,
            "gurobi": "not_available_in_runtime",
            "detail": f"{type(exc).__name__}: {exc}",
        }


class CertificateError(ValueError):
    """A proof bundle cannot be independently admitted."""


def _required(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise CertificateError(f"{context} missing required field {key!r}")
    return mapping[key]


def _as_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CertificateError(f"{context} must be a mapping")
    return value


def _nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CertificateError(f"{context} must be a non-empty string")
    return value.strip()


def _finite_float(value: Any, context: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise CertificateError(f"{context} must be numeric, not boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CertificateError(f"{context} must be numeric") from exc
    if not np.isfinite(number):
        raise CertificateError(f"{context} must be finite")
    return number


def _nonnegative_float(value: Any, context: str) -> float:
    number = _finite_float(value, context)
    if number < 0.0:
        raise CertificateError(f"{context} must be non-negative")
    return number


def _integer(value: Any, context: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise CertificateError(f"{context} must be an integer, not boolean")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise CertificateError(f"{context} must be an integer") from exc
    if isinstance(value, (float, np.floating)) and float(value) != number:
        raise CertificateError(f"{context} must be an exact integer")
    return number


def _names(value: Any, expected: int | None, context: str) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise CertificateError(f"{context} must be a sequence")
    names = [_nonempty_string(item, context) for item in value]
    if expected is not None and len(names) != expected:
        raise CertificateError(
            f"{context} length {len(names)} does not match {expected}"
        )
    if len(set(names)) != len(names):
        raise CertificateError(f"{context} contains duplicate names")
    return names


def _vector(value: Any, length: int, context: str) -> np.ndarray:
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise CertificateError(f"{context} must be a numeric sequence")
    if len(value) != length:
        raise CertificateError(
            f"{context} length {len(value)} does not match {length}"
        )
    return np.asarray(
        [_finite_float(item, f"{context}[{index}]") for index, item in enumerate(value)],
        dtype=float,
    )


def _matrix(
    value: Any, rows: int, columns: int, context: str
) -> np.ndarray:
    if not isinstance(value, (list, tuple)):
        raise CertificateError(f"{context} must be a sequence of rows")
    if len(value) != rows:
        raise CertificateError(
            f"{context} row count {len(value)} does not match {rows}"
        )
    if rows == 0:
        return np.zeros((0, columns), dtype=float)
    parsed = [
        _vector(row, columns, f"{context}[{index}]")
        for index, row in enumerate(value)
    ]
    return np.vstack(parsed)


def _hash_string(value: Any, context: str) -> str:
    text = _nonempty_string(value, context).lower()
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise CertificateError(f"{context} must be a SHA-256 hex digest")
    return text


@dataclass(frozen=True)
class VerifiedCertificate:
    admitted: bool
    reason: str
    certificate_id: str | None
    solver: str | None
    recomputed_model_sha256: str | None
    recomputed_objective_sha256: str | None
    recomputed_primal_residual: float | None
    recomputed_dual_feasibility_residual: float | None
    recomputed_complementarity_residual: float | None
    recomputed_primal_objective: float | None
    recomputed_kkt_dual_objective: float | None
    recomputed_box_dual_lower_bound: float | None
    recomputed_primal_dual_gap: float | None


def _failed_verification(reason: str) -> VerifiedCertificate:
    return VerifiedCertificate(
        admitted=False,
        reason=reason,
        certificate_id=None,
        solver=None,
        recomputed_model_sha256=None,
        recomputed_objective_sha256=None,
        recomputed_primal_residual=None,
        recomputed_dual_feasibility_residual=None,
        recomputed_complementarity_residual=None,
        recomputed_primal_objective=None,
        recomputed_kkt_dual_objective=None,
        recomputed_box_dual_lower_bound=None,
        recomputed_primal_dual_gap=None,
    )


def verify_dual_primal_certificate(
    certificate: Any,
    *,
    expected_mode_hour: int,
    expected_fixed_mode: int,
    expected_model_sha256: str,
    expected_objective_sha256: str,
    expected_objective_cutoff: float,
    allowed_sources: set[str],
    primal_tolerance: float,
    dual_tolerance: float,
    objective_tolerance: float,
    optimality_tolerance: float,
) -> VerifiedCertificate:
    """Recompute a complete LP certificate; never consume claimed bounds."""

    try:
        cert = _as_mapping(certificate, "certificate")
        schema = _nonempty_string(
            _required(cert, "schema_version", "certificate"),
            "certificate.schema_version",
        )
        if schema != "dual-primal-lp-certificate-v0.8":
            raise CertificateError("unsupported certificate schema")
        source = _nonempty_string(
            _required(cert, "source", "certificate"), "certificate.source"
        )
        if source not in allowed_sources:
            raise CertificateError("certificate source is not allowed")
        solver = _nonempty_string(
            _required(cert, "solver", "certificate"), "certificate.solver"
        )
        _nonempty_string(
            _required(cert, "solver_version", "certificate"),
            "certificate.solver_version",
        )
        certificate_id = _nonempty_string(
            _required(cert, "certificate_id", "certificate"),
            "certificate.certificate_id",
        )
        hour = _integer(
            _required(cert, "mode_hour", "certificate"),
            "certificate.mode_hour",
        )
        mode = _integer(
            _required(cert, "fixed_mode", "certificate"),
            "certificate.fixed_mode",
        )
        if hour != expected_mode_hour or mode != expected_fixed_mode:
            raise CertificateError("certificate branch identity mismatch")
        if mode not in (0, 1):
            raise CertificateError("fixed_mode must be 0 or 1")
        declared_cutoff = _finite_float(
            _required(cert, "objective_cutoff", "certificate"),
            "certificate.objective_cutoff",
        )
        if abs(declared_cutoff - expected_objective_cutoff) > objective_tolerance:
            raise CertificateError(
                "certificate objective cutoff does not match consumer cutoff"
            )

        status = _as_mapping(
            _required(cert, "solver_status", "certificate"),
            "certificate.solver_status",
        )
        if (
            _nonempty_string(
                _required(status, "status", "solver_status"),
                "solver_status.status",
            ).lower()
            != "optimal"
        ):
            raise CertificateError("solver status is not globally optimal")
        status_code = _required(status, "code", "solver_status")
        if isinstance(status_code, (str, bytes)):
            _nonempty_string(status_code, "solver_status.code")
        else:
            _finite_float(status_code, "solver_status.code")
        _nonempty_string(
            _required(status, "message", "solver_status"),
            "solver_status.message",
        )

        declared_model_hash = _hash_string(
            _required(cert, "model_sha256", "certificate"),
            "certificate.model_sha256",
        )
        declared_objective_hash = _hash_string(
            _required(cert, "objective_sha256", "certificate"),
            "certificate.objective_sha256",
        )
        expected_model_hash = _hash_string(
            expected_model_sha256, "expected_model_sha256"
        )
        expected_objective_hash = _hash_string(
            expected_objective_sha256, "expected_objective_sha256"
        )
        if declared_model_hash != expected_model_hash:
            raise CertificateError("declared model hash does not match expected")
        if declared_objective_hash != expected_objective_hash:
            raise CertificateError(
                "declared objective hash does not match expected"
            )

        model = _as_mapping(
            _required(cert, "model", "certificate"), "certificate.model"
        )
        objective = _as_mapping(
            _required(cert, "objective", "certificate"),
            "certificate.objective",
        )
        recomputed_model_hash = canonical_sha256(model)
        recomputed_objective_hash = canonical_sha256(objective)
        if recomputed_model_hash != declared_model_hash:
            raise CertificateError("model payload SHA-256 mismatch")
        if recomputed_objective_hash != declared_objective_hash:
            raise CertificateError("objective payload SHA-256 mismatch")
        constraints_payload = {
            key: _required(model, key, "model")
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
            key: _required(model, key, "model")
            for key in ("variable_names", "lower_bounds", "upper_bounds")
        }
        recomputed_constraints_hash = canonical_sha256(constraints_payload)
        recomputed_bounds_hash = canonical_sha256(bounds_payload)
        if _hash_string(
            _required(cert, "constraints_sha256", "certificate"),
            "certificate.constraints_sha256",
        ) != recomputed_constraints_hash:
            raise CertificateError("constraint payload SHA-256 mismatch")
        if _hash_string(
            _required(cert, "variable_bounds_sha256", "certificate"),
            "certificate.variable_bounds_sha256",
        ) != recomputed_bounds_hash:
            raise CertificateError("variable-bound payload SHA-256 mismatch")

        if (
            _nonempty_string(
                _required(model, "schema_version", "model"),
                "model.schema_version",
            )
            != "lp-proof-model-v0.8"
        ):
            raise CertificateError("unsupported model payload schema")
        _nonempty_string(
            _required(model, "model_name", "model"), "model.model_name"
        )
        if (
            _integer(_required(model, "mode_hour", "model"), "model.mode_hour")
            != hour
            or _integer(
                _required(model, "fixed_mode", "model"), "model.fixed_mode"
            )
            != mode
        ):
            raise CertificateError("model payload branch identity mismatch")

        variable_names = _names(
            _required(model, "variable_names", "model"),
            None,
            "model.variable_names",
        )
        n = len(variable_names)
        if n == 0:
            raise CertificateError("model must contain at least one variable")
        inequality_names = _names(
            _required(model, "inequality_names", "model"),
            None,
            "model.inequality_names",
        )
        equality_names = _names(
            _required(model, "equality_names", "model"),
            None,
            "model.equality_names",
        )
        m_ub = len(inequality_names)
        m_eq = len(equality_names)
        a_ub = _matrix(
            _required(model, "A_ub", "model"), m_ub, n, "model.A_ub"
        )
        b_ub = _vector(
            _required(model, "b_ub", "model"), m_ub, "model.b_ub"
        )
        a_eq = _matrix(
            _required(model, "A_eq", "model"), m_eq, n, "model.A_eq"
        )
        b_eq = _vector(
            _required(model, "b_eq", "model"), m_eq, "model.b_eq"
        )
        lower = _vector(
            _required(model, "lower_bounds", "model"),
            n,
            "model.lower_bounds",
        )
        upper = _vector(
            _required(model, "upper_bounds", "model"),
            n,
            "model.upper_bounds",
        )
        if np.any(lower > upper):
            raise CertificateError("model contains lower bound above upper bound")

        if (
            _nonempty_string(
                _required(objective, "schema_version", "objective"),
                "objective.schema_version",
            )
            != "lp-proof-objective-v0.8"
        ):
            raise CertificateError("unsupported objective payload schema")
        if (
            _nonempty_string(
                _required(objective, "sense", "objective"),
                "objective.sense",
            ).lower()
            != "min"
        ):
            raise CertificateError("only minimization certificates are allowed")
        objective_names = _names(
            _required(objective, "variable_names", "objective"),
            n,
            "objective.variable_names",
        )
        if objective_names != variable_names:
            raise CertificateError("objective variable order mismatch")
        coefficients = _vector(
            _required(objective, "coefficients", "objective"),
            n,
            "objective.coefficients",
        )
        constant = _finite_float(
            _required(objective, "constant", "objective"),
            "objective.constant",
        )

        primal = _as_mapping(
            _required(cert, "primal", "certificate"), "certificate.primal"
        )
        x = _vector(
            _required(primal, "values", "primal"), n, "primal.values"
        )
        declared_primal_objective = _finite_float(
            _required(primal, "objective", "primal"), "primal.objective"
        )
        declared_primal_residual = _nonnegative_float(
            _required(primal, "max_primal_residual", "primal"),
            "primal.max_primal_residual",
        )

        ub_violation = (
            float(np.max(np.maximum(a_ub @ x - b_ub, 0.0)))
            if m_ub
            else 0.0
        )
        eq_violation = (
            float(np.max(np.abs(a_eq @ x - b_eq))) if m_eq else 0.0
        )
        lower_violation = float(np.max(np.maximum(lower - x, 0.0)))
        upper_violation = float(np.max(np.maximum(x - upper, 0.0)))
        primal_residual = max(
            ub_violation, eq_violation, lower_violation, upper_violation
        )
        primal_objective = float(constant + coefficients @ x)
        if primal_residual > primal_tolerance:
            raise CertificateError("recomputed primal feasibility residual exceeds tolerance")
        if abs(declared_primal_residual - primal_residual) > primal_tolerance:
            raise CertificateError("declared primal residual does not match recomputation")
        if (
            abs(declared_primal_objective - primal_objective)
            > objective_tolerance
        ):
            raise CertificateError("declared primal objective does not match recomputation")

        dual = _as_mapping(
            _required(cert, "dual", "certificate"), "certificate.dual"
        )
        lam_raw = _vector(
            _required(dual, "inequality_multipliers", "dual"),
            m_ub,
            "dual.inequality_multipliers",
        )
        mu = _vector(
            _required(dual, "equality_multipliers", "dual"),
            m_eq,
            "dual.equality_multipliers",
        )
        alpha_raw = _vector(
            _required(dual, "lower_bound_multipliers", "dual"),
            n,
            "dual.lower_bound_multipliers",
        )
        beta_raw = _vector(
            _required(dual, "upper_bound_multipliers", "dual"),
            n,
            "dual.upper_bound_multipliers",
        )
        declared_dual_objective = _finite_float(
            _required(dual, "objective", "dual"), "dual.objective"
        )
        declared_dual_residual = _nonnegative_float(
            _required(dual, "dual_feasibility_residual", "dual"),
            "dual.dual_feasibility_residual",
        )

        sign_residual = max(
            float(np.max(np.maximum(-lam_raw, 0.0))) if m_ub else 0.0,
            float(np.max(np.maximum(-alpha_raw, 0.0))),
            float(np.max(np.maximum(-beta_raw, 0.0))),
        )
        lam = np.maximum(lam_raw, 0.0)
        alpha = np.maximum(alpha_raw, 0.0)
        beta = np.maximum(beta_raw, 0.0)
        q = coefficients.copy()
        if m_ub:
            q = q + a_ub.T @ lam
        if m_eq:
            q = q + a_eq.T @ mu
        stationarity = q - alpha + beta
        stationarity_residual = float(np.max(np.abs(stationarity)))
        dual_residual = max(sign_residual, stationarity_residual)
        if dual_residual > dual_tolerance:
            raise CertificateError("recomputed dual feasibility residual exceeds tolerance")
        if abs(declared_dual_residual - dual_residual) > dual_tolerance:
            raise CertificateError("declared dual residual does not match recomputation")

        slack_ub = b_ub - a_ub @ x if m_ub else np.zeros(0)
        complementarity_residual = max(
            float(np.max(np.abs(lam * slack_ub))) if m_ub else 0.0,
            float(np.max(np.abs(alpha * (x - lower)))),
            float(np.max(np.abs(beta * (upper - x)))),
        )
        if complementarity_residual > optimality_tolerance:
            raise CertificateError("recomputed complementarity residual exceeds tolerance")

        base_dual = constant
        if m_ub:
            base_dual -= float(b_ub @ lam)
        if m_eq:
            base_dual -= float(b_eq @ mu)
        kkt_dual_objective = float(
            base_dual + lower @ alpha - upper @ beta
        )
        box_infimum = float(
            np.sum(np.minimum(q * lower, q * upper))
        )
        box_dual_lower_bound = float(base_dual + box_infimum)
        if not np.isfinite(kkt_dual_objective) or not np.isfinite(
            box_dual_lower_bound
        ):
            raise CertificateError("recomputed dual objective is not finite")
        if (
            abs(declared_dual_objective - kkt_dual_objective)
            > objective_tolerance
        ):
            raise CertificateError("declared dual objective does not match recomputation")
        if (
            abs(kkt_dual_objective - box_dual_lower_bound)
            > optimality_tolerance
        ):
            raise CertificateError("KKT and box dual lower bounds disagree")
        gap = float(primal_objective - box_dual_lower_bound)
        if gap < -objective_tolerance:
            raise CertificateError("recomputed dual lower bound exceeds primal objective")
        if gap > optimality_tolerance:
            raise CertificateError("recomputed primal-dual gap exceeds tolerance")

        return VerifiedCertificate(
            admitted=True,
            reason="complete primal/dual payload independently verified",
            certificate_id=certificate_id,
            solver=solver,
            recomputed_model_sha256=recomputed_model_hash,
            recomputed_objective_sha256=recomputed_objective_hash,
            recomputed_primal_residual=primal_residual,
            recomputed_dual_feasibility_residual=dual_residual,
            recomputed_complementarity_residual=complementarity_residual,
            recomputed_primal_objective=primal_objective,
            recomputed_kkt_dual_objective=kkt_dual_objective,
            recomputed_box_dual_lower_bound=box_dual_lower_bound,
            recomputed_primal_dual_gap=gap,
        )
    except Exception as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return _failed_verification(f"{type(exc).__name__}: {exc}")


@dataclass(frozen=True)
class OBBTDecision:
    mode_hour: int
    decision: str
    fixed_mode: int | None
    branch_zero_admitted: bool
    branch_one_admitted: bool
    branch_zero_recomputed_lower_bound: float | None
    branch_one_recomputed_lower_bound: float | None
    branch_zero_reason: str
    branch_one_reason: str
    reason: str


def _branch_key(hour: int, mode: int) -> str:
    return f"{int(hour)}:{int(mode)}"


def objective_cutoff_obbt(
    mode_hours: Sequence[int],
    incumbent_upper_bound: float,
    expected_model_hashes: Mapping[str, str],
    expected_objective_hashes: Mapping[str, str],
    certificate_provider: Callable[[int, int], Any],
    allowed_sources: set[str],
    *,
    primal_tolerance: float = PRIMAL_FEASIBILITY_TOLERANCE,
    dual_tolerance: float = DUAL_FEASIBILITY_TOLERANCE,
    objective_tolerance: float = OBJECTIVE_ACCOUNTING_TOLERANCE,
    optimality_tolerance: float = OPTIMALITY_TOLERANCE,
    proof_guard: float = OBBT_PROOF_GUARD,
) -> list[OBBTDecision]:
    """Return only fixes supported by independently recomputed LP proofs."""

    incumbent = _finite_float(
        incumbent_upper_bound, "incumbent_upper_bound"
    )
    for value, name in (
        (primal_tolerance, "primal_tolerance"),
        (dual_tolerance, "dual_tolerance"),
        (objective_tolerance, "objective_tolerance"),
        (optimality_tolerance, "optimality_tolerance"),
        (proof_guard, "proof_guard"),
    ):
        _nonnegative_float(value, name)
    if not isinstance(allowed_sources, set) or any(
        not isinstance(value, str) or not value.strip()
        for value in allowed_sources
    ):
        raise ValueError("allowed_sources must be a set of non-empty strings")

    decisions: list[OBBTDecision] = []
    for raw_hour in mode_hours:
        hour = _integer(raw_hour, "mode_hour")
        verified: dict[int, VerifiedCertificate] = {}
        for mode in (0, 1):
            key = _branch_key(hour, mode)
            try:
                expected_model_hash = expected_model_hashes[key]
                expected_objective_hash = expected_objective_hashes[key]
                certificate = certificate_provider(hour, mode)
                verified[mode] = verify_dual_primal_certificate(
                    certificate,
                    expected_mode_hour=hour,
                    expected_fixed_mode=mode,
                    expected_model_sha256=expected_model_hash,
                    expected_objective_sha256=expected_objective_hash,
                    expected_objective_cutoff=incumbent,
                    allowed_sources=allowed_sources,
                    primal_tolerance=primal_tolerance,
                    dual_tolerance=dual_tolerance,
                    objective_tolerance=objective_tolerance,
                    optimality_tolerance=optimality_tolerance,
                )
            except Exception as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                verified[mode] = _failed_verification(
                    f"{type(exc).__name__}: {exc}"
                )

        excluded = {
            mode: bool(
                verified[mode].admitted
                and verified[mode].recomputed_box_dual_lower_bound is not None
                and float(
                    verified[mode].recomputed_box_dual_lower_bound
                )
                > incumbent + proof_guard
            )
            for mode in (0, 1)
        }
        if excluded[0] and excluded[1]:
            raise RuntimeError(
                f"Both modes at hour {hour} exceed the independently "
                "verified cutoff; certificate/model inconsistency"
            )
        if excluded[0]:
            decision = "fixed"
            fixed_mode = 1
            reason = "verified z=0 LP lower bound exceeds incumbent cutoff"
        elif excluded[1]:
            decision = "fixed"
            fixed_mode = 0
            reason = "verified z=1 LP lower bound exceeds incumbent cutoff"
        else:
            decision = "unfixed"
            fixed_mode = None
            if not all(verified[mode].admitted for mode in (0, 1)):
                reason = "one or both complete branch proofs were not admitted"
            else:
                reason = "neither recomputed branch lower bound exceeds cutoff"
        decisions.append(
            OBBTDecision(
                mode_hour=hour,
                decision=decision,
                fixed_mode=fixed_mode,
                branch_zero_admitted=verified[0].admitted,
                branch_one_admitted=verified[1].admitted,
                branch_zero_recomputed_lower_bound=verified[
                    0
                ].recomputed_box_dual_lower_bound,
                branch_one_recomputed_lower_bound=verified[
                    1
                ].recomputed_box_dual_lower_bound,
                branch_zero_reason=verified[0].reason,
                branch_one_reason=verified[1].reason,
                reason=reason,
            )
        )
    return decisions


def make_analytic_certificate(
    hour: int,
    mode: int,
    lower_bound: float,
    objective_cutoff: float = 100.0,
) -> dict[str, Any]:
    """Build an exact certificate for min x subject to x >= lower_bound."""

    lower_value = float(lower_bound)
    model = {
        "schema_version": "lp-proof-model-v0.8",
        "model_name": f"analytic-branch-{hour}-{mode}",
        "mode_hour": int(hour),
        "fixed_mode": int(mode),
        "variable_names": ["x"],
        "inequality_names": ["minimum_x"],
        "equality_names": [],
        "A_ub": [[-1.0]],
        "b_ub": [-lower_value],
        "A_eq": [],
        "b_eq": [],
        "lower_bounds": [0.0],
        "upper_bounds": [200.0],
    }
    objective = {
        "schema_version": "lp-proof-objective-v0.8",
        "sense": "min",
        "variable_names": ["x"],
        "coefficients": [1.0],
        "constant": 0.0,
    }
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
    return {
        "schema_version": "dual-primal-lp-certificate-v0.8",
        "source": "analytic_toy_dual",
        "solver": "analytic_enumeration",
        "solver_version": "1.0",
        "certificate_id": f"analytic-{hour}-{mode}-{lower_value:g}",
        "mode_hour": int(hour),
        "fixed_mode": int(mode),
        "objective_cutoff": float(objective_cutoff),
        "solver_status": {
            "status": "optimal",
            "code": 0,
            "message": "exact analytic optimum",
        },
        "model_sha256": canonical_sha256(model),
        "constraints_sha256": canonical_sha256(constraints_payload),
        "variable_bounds_sha256": canonical_sha256(bounds_payload),
        "objective_sha256": canonical_sha256(objective),
        "model": model,
        "objective": objective,
        "primal": {
            "values": [lower_value],
            "objective": lower_value,
            "max_primal_residual": 0.0,
        },
        "dual": {
            "inequality_multipliers": [1.0],
            "equality_multipliers": [],
            "lower_bound_multipliers": [0.0],
            "upper_bound_multipliers": [0.0],
            "objective": lower_value,
            "dual_feasibility_residual": 0.0,
        },
    }


def _certificate_hash_contract(
    data: Mapping[tuple[int, int], Mapping[str, Any]]
) -> tuple[dict[str, str], dict[str, str]]:
    model_hashes = {
        _branch_key(hour, mode): str(certificate["model_sha256"])
        for (hour, mode), certificate in data.items()
    }
    objective_hashes = {
        _branch_key(hour, mode): str(certificate["objective_sha256"])
        for (hour, mode), certificate in data.items()
    }
    return model_hashes, objective_hashes


def run_obbt_unit_tests() -> dict[str, Any]:
    allowed = {"analytic_toy_dual"}
    normal_data: dict[tuple[int, int], dict[str, Any]] = {}
    values = {
        0: (101.0, 99.0),
        1: (98.0, 102.0),
        2: (99.2, 99.1),
    }
    for hour, bounds in values.items():
        for mode, lower in enumerate(bounds):
            normal_data[(hour, mode)] = make_analytic_certificate(
                hour, mode, lower
            )
    model_hashes, objective_hashes = _certificate_hash_contract(normal_data)
    normal_decisions = objective_cutoff_obbt(
        mode_hours=tuple(values),
        incumbent_upper_bound=100.0,
        expected_model_hashes=model_hashes,
        expected_objective_hashes=objective_hashes,
        certificate_provider=lambda hour, mode: normal_data[(hour, mode)],
        allowed_sources=allowed,
    )
    expected = {0: 1, 1: 0, 2: None}
    normal_pass = all(
        decision.fixed_mode == expected[decision.mode_hour]
        for decision in normal_decisions
    )

    attack_results: dict[str, dict[str, Any]] = {}

    def attack(
        name: str, mutate: Callable[[dict[str, Any]], None]
    ) -> None:
        hour = 100 + len(attack_results)
        original = {
            (hour, 0): make_analytic_certificate(hour, 0, 101.0),
            (hour, 1): make_analytic_certificate(hour, 1, 99.0),
        }
        expected_models, expected_objectives = _certificate_hash_contract(
            original
        )
        supplied = copy.deepcopy(original)
        mutate(supplied[(hour, 0)])
        try:
            decision = objective_cutoff_obbt(
                mode_hours=(hour,),
                incumbent_upper_bound=100.0,
                expected_model_hashes=expected_models,
                expected_objective_hashes=expected_objectives,
                certificate_provider=lambda h, m: supplied[(h, m)],
                allowed_sources=allowed,
            )[0]
            safe = decision.fixed_mode is None
            attack_results[name] = {
                "safe_unfixed": safe,
                "decision": asdict(decision),
            }
        except Exception as exc:
            attack_results[name] = {
                "safe_unfixed": True,
                "hard_stop_before_fixing": True,
                "exception": f"{type(exc).__name__}: {exc}",
            }

    attack("missing_dual_payload", lambda cert: cert.pop("dual"))
    attack("empty_solver", lambda cert: cert.__setitem__("solver", ""))
    attack(
        "empty_solver_version",
        lambda cert: cert.__setitem__("solver_version", ""),
    )
    attack(
        "empty_certificate_id",
        lambda cert: cert.__setitem__("certificate_id", ""),
    )
    attack(
        "negative_dual_residual",
        lambda cert: cert["dual"].__setitem__(
            "dual_feasibility_residual", -1.0
        ),
    )
    attack(
        "nonfinite_dual_residual",
        lambda cert: cert["dual"].__setitem__(
            "dual_feasibility_residual", float("nan")
        ),
    )
    attack(
        "negative_primal_residual",
        lambda cert: cert["primal"].__setitem__(
            "max_primal_residual", -1.0
        ),
    )
    attack(
        "wrong_declared_model_hash",
        lambda cert: cert.__setitem__("model_sha256", "c" * 64),
    )
    attack(
        "wrong_declared_bounds_hash",
        lambda cert: cert.__setitem__("variable_bounds_sha256", "d" * 64),
    )
    attack(
        "wrong_objective_cutoff",
        lambda cert: cert.__setitem__("objective_cutoff", 99.0),
    )
    attack(
        "missing_objective_cutoff",
        lambda cert: cert.pop("objective_cutoff"),
    )

    def alter_model_payload(cert: dict[str, Any]) -> None:
        cert["model"]["b_ub"][0] -= 1.0

    attack("model_payload_hash_mismatch", alter_model_payload)
    attack(
        "dual_stationarity_infeasible",
        lambda cert: cert["dual"].__setitem__(
            "inequality_multipliers", [0.0]
        ),
    )
    attack(
        "negative_dual_multiplier",
        lambda cert: cert["dual"].__setitem__(
            "inequality_multipliers", [-1.0]
        ),
    )
    attack(
        "fraudulent_dual_objective",
        lambda cert: cert["dual"].__setitem__("objective", 10000.0),
    )
    attack(
        "primal_infeasible",
        lambda cert: cert["primal"].__setitem__("values", [0.0]),
    )
    attack(
        "source_not_allowlisted",
        lambda cert: cert.__setitem__("source", "untrusted"),
    )
    attack(
        "missing_objective_coefficients",
        lambda cert: cert["objective"].pop("coefficients"),
    )

    negative_guard_raises = False
    try:
        objective_cutoff_obbt(
            mode_hours=(0,),
            incumbent_upper_bound=100.0,
            expected_model_hashes=model_hashes,
            expected_objective_hashes=objective_hashes,
            certificate_provider=lambda hour, mode: normal_data[(hour, mode)],
            allowed_sources=allowed,
            proof_guard=-1.0,
        )
    except (ValueError, CertificateError):
        negative_guard_raises = True

    inconsistent = {
        (5, mode): make_analytic_certificate(5, mode, 103.0 + mode)
        for mode in (0, 1)
    }
    inconsistent_models, inconsistent_objectives = (
        _certificate_hash_contract(inconsistent)
    )
    both_excluded_raises = False
    try:
        objective_cutoff_obbt(
            mode_hours=(5,),
            incumbent_upper_bound=100.0,
            expected_model_hashes=inconsistent_models,
            expected_objective_hashes=inconsistent_objectives,
            certificate_provider=lambda hour, mode: inconsistent[(hour, mode)],
            allowed_sources=allowed,
        )
    except RuntimeError:
        both_excluded_raises = True

    spoof = {
        (6, mode): make_analytic_certificate(6, mode, 99.0)
        for mode in (0, 1)
    }
    for certificate in spoof.values():
        certificate["claimed_lower_bound"] = 1.0e12
    spoof_models, spoof_objectives = _certificate_hash_contract(spoof)
    spoof_decision = objective_cutoff_obbt(
        mode_hours=(6,),
        incumbent_upper_bound=100.0,
        expected_model_hashes=spoof_models,
        expected_objective_hashes=spoof_objectives,
        certificate_provider=lambda hour, mode: spoof[(hour, mode)],
        allowed_sources=allowed,
    )[0]
    scalar_lower_bound_ignored = (
        spoof_decision.fixed_mode is None
        and spoof_decision.branch_zero_admitted
        and spoof_decision.branch_one_admitted
        and spoof_decision.branch_zero_recomputed_lower_bound == 99.0
        and spoof_decision.branch_one_recomputed_lower_bound == 99.0
    )

    all_attacks_safe = all(
        value["safe_unfixed"] for value in attack_results.values()
    )
    status = (
        "pass"
        if normal_pass
        and all_attacks_safe
        and negative_guard_raises
        and both_excluded_raises
        and scalar_lower_bound_ignored
        else "fail"
    )
    return {
        "status": status,
        "normal_case_pass": normal_pass,
        "all_malicious_certificates_fail_closed": all_attacks_safe,
        "negative_proof_guard_hard_stops": negative_guard_raises,
        "both_branches_excluded_hard_stops": both_excluded_raises,
        "caller_scalar_lower_bound_ignored": scalar_lower_bound_ignored,
        "production_dual_backend_status": "pending_not_implemented",
        "incumbent_pattern_argument_exists": False,
        "mip_start_pattern_fixed_as_constraint": False,
        "normal_decisions": [
            asdict(decision) for decision in normal_decisions
        ],
        "malicious_certificate_tests": attack_results,
    }


def pattern_inventory_check(results: pd.DataFrame) -> dict[str, Any]:
    details: dict[str, Any] = {}
    passed = True
    for horizon in v07.HORIZONS:
        subset = results.loc[results["horizon_h"] == horizon]
        expected_count = 2 ** v07.OVERLAP_COUNTS[horizon]
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
            f"Append-only v0.8 output already exists and is non-empty: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for horizon in v07.HORIZONS:
        instance = v07.generate_toy_instance(horizon)
        for pattern_index, pattern in enumerate(
            v07.mode_patterns(len(instance.overlap_hours))
        ):
            rows.append(v07.audit_pattern(instance, pattern_index, pattern))
    results = pd.DataFrame(rows)
    results_path = output / "toy_pattern_equivalence.csv"
    results.to_csv(results_path, index=False, float_format="%.12g")

    obbt = run_obbt_unit_tests()
    obbt_path = output / "obbt_adversarial_unit_tests.json"
    write_json(obbt_path, obbt)

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
        2 ** v07.OVERLAP_COUNTS[n] for n in v07.HORIZONS
    )
    pattern_inventory = pattern_inventory_check(results)
    path_probe = manifest_path_entry(results_path)
    outside_probe_path = Path("/tmp/stage1_v0_8_manifest_probe.json")
    outside_probe_payload = portable_path(outside_probe_path)
    external_path_logic_pass = (
        not str(outside_probe_path.resolve()).startswith(str(ROOT) + "/")
        and outside_probe_payload["path_kind"] == "absolute_external"
    )
    checks = {
        "v07_runner_dependency_sha_locked": sha256(V07_RUNNER)
        == V07_RUNNER_EXPECTED_SHA256,
        "v07_independent_audit_sha_locked": sha256(V07_AUDIT)
        == V07_AUDIT_EXPECTED_SHA256,
        "v07_results_sha_locked": sha256(V07_RESULTS)
        == V07_RESULTS_EXPECTED_SHA256,
        "only_toy_horizons_4_to_24": sorted(
            results["horizon_h"].unique().tolist()
        )
        == list(v07.HORIZONS),
        "all_expected_patterns_enumerated": len(results)
        == expected_patterns,
        "pattern_sets_exact_by_horizon": pattern_inventory["pass"],
        "all_reference_models_solved": bool(results["reference_success"].all()),
        "all_hull_models_solved": bool(results["hull_success"].all()),
        "all_pattern_rows_pass": bool(results["pass"].all()),
        "objective_equivalence": maxima["objective_absolute_difference"]
        <= v07.OBJECTIVE_TOLERANCE,
        "reference_to_hull_mapping_feasible": maxima[
            "reference_to_hull_max_residual"
        ]
        <= v07.FEASIBILITY_TOLERANCE,
        "hull_to_reference_mapping_feasible": maxima[
            "hull_to_reference_max_residual"
        ]
        <= v07.FEASIBILITY_TOLERANCE,
        "reference_no_simultaneous_flow": maxima[
            "reference_simultaneous_mw"
        ]
        <= v07.SIMULTANEOUS_TOLERANCE,
        "hull_no_simultaneous_flow": maxima["hull_simultaneous_mw"]
        <= v07.SIMULTANEOUS_TOLERANCE,
        "deterministic_tiebreak_trajectory_equivalence": max(
            maxima["lex_charge_max_difference_mw"],
            maxima["lex_discharge_max_difference_mw"],
            maxima["lex_shifted_soc_max_difference_mwh"],
            maxima["lex_absolute_soc_max_difference_mwh"],
        )
        <= v07.TRAJECTORY_TOLERANCE,
        "raw_centered_objective_ledger_equivalence": maxima[
            "raw_centered_ledger_difference"
        ]
        <= v07.OBJECTIVE_TOLERANCE,
        "toy_results_byte_identical_to_v07": sha256(results_path)
        == V07_RESULTS_EXPECTED_SHA256,
        "obbt_complete_proof_adversarial_tests": obbt["status"] == "pass",
        "external_output_manifest_path_supported": external_path_logic_pass,
        "current_output_manifest_path_kind_valid": path_probe["path_kind"]
        in {"repository_relative", "absolute_external"},
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
        "manuscript_evidence_admitted": False,
        "production_obbt_backend_status": "pending_not_implemented",
        "full_year_benchmark_status": "full_year_not_run",
        "horizons_h": list(v07.HORIZONS),
        "expected_pattern_count": expected_patterns,
        "observed_pattern_count": len(results),
        "pattern_inventory": pattern_inventory,
        "maxima": maxima,
        "tolerances": {
            "primal_feasibility": PRIMAL_FEASIBILITY_TOLERANCE,
            "dual_feasibility": DUAL_FEASIBILITY_TOLERANCE,
            "objective_accounting": OBJECTIVE_ACCOUNTING_TOLERANCE,
            "optimality": OPTIMALITY_TOLERANCE,
            "obbt_proof_guard": OBBT_PROOF_GUARD,
            "inherited_v07_feasibility": v07.FEASIBILITY_TOLERANCE,
            "inherited_v07_objective": v07.OBJECTIVE_TOLERANCE,
            "inherited_v07_trajectory": v07.TRAJECTORY_TOLERANCE,
        },
        "checks": checks,
        "notes": [
            "The exact-projection and state-hull toy kernel is imported "
            "unchanged from the SHA-locked v0.7 runner.",
            "The OBBT consumer recomputes primal feasibility, dual "
            "feasibility, both dual objectives and the primal-dual gap from "
            "the complete matrix payload.",
            "No caller-provided scalar lower bound is consumed.",
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
        "manuscript_evidence_admitted": False,
        "full_year_benchmark_status": "full_year_not_run",
        "files": {
            "runner": manifest_path_entry(script_path),
            "v07_runner_dependency": manifest_path_entry(V07_RUNNER),
            "v07_independent_audit_basis": manifest_path_entry(V07_AUDIT),
            "design": manifest_path_entry(DESIGN),
            "results": manifest_path_entry(results_path),
            "obbt_adversarial_unit_tests": manifest_path_entry(obbt_path),
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
