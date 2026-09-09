#!/usr/bin/env python3
"""Exact-projection and state-hull toy audit for Stage 1 v0.7.

This append-only prototype never loads or solves an 8,760-hour country-year.
It compares an unprojected reference LP with a projected, objective-centred
Balas state-transition formulation on exhaustively enumerated 4--24 h toy
mode patterns.  All outputs are non-evidential.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linprog


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VERSION = "0.7.0"
EVIDENCE_STATUS = "non_evidential_stage1_exact_reformulation_toy"
DESIGN = (
    ROOT
    / "artifacts"
    / "research"
    / "OBSERVED_GAS_EVENT_TASK_STAGE1_EXACT_REFORMULATION_DESIGN_V0_7.md"
)
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts"
    / "research"
    / "stage1_exact_reformulation_toy_v0_7"
)

HORIZONS = (4, 6, 8, 12, 16, 24)
OVERLAP_COUNTS = {4: 4, 6: 4, 8: 5, 12: 6, 16: 7, 24: 8}
FEASIBILITY_TOLERANCE = 2e-7
OBJECTIVE_TOLERANCE = 2e-7
TRAJECTORY_TOLERANCE = 2e-6
SIMULTANEOUS_TOLERANCE = 1e-8
DUAL_FEASIBILITY_TOLERANCE = 1e-8
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


@dataclass(frozen=True)
class ToyInstance:
    instance_id: str
    task_mw: tuple[float, ...]
    recharge_mw: tuple[float, ...]
    overlap_hours: tuple[int, ...]
    pdis_mw: float
    native_pch_mw: float
    max_extra_pch_mw: float
    energy_mwh: float
    soc_min_fraction: float
    soc_max_fraction: float
    eta_ch: float
    eta_dis: float
    retention: float
    gas_fixed_cost: float
    gas_variable_cost: float
    extra_pch_cost: float
    recharge_price: float
    discharge_vom: float
    fixed_capital_cost: float

    @property
    def n(self) -> int:
        return len(self.task_mw)

    @property
    def usable_soc_mwh(self) -> float:
        return (
            (self.soc_max_fraction - self.soc_min_fraction)
            * self.energy_mwh
        )

    @property
    def soc_floor_mwh(self) -> float:
        return self.soc_min_fraction * self.energy_mwh

    @property
    def raw_objective_constant(self) -> float:
        return self.fixed_capital_cost

    @property
    def centered_objective_constant(self) -> float:
        return (
            self.fixed_capital_cost
            + self.gas_variable_cost * sum(self.task_mw)
        )


@dataclass
class LinearProgram:
    formulation: str
    instance: ToyInstance
    pattern: tuple[int, ...]
    names: tuple[str, ...]
    name_to_index: dict[str, int]
    objective: np.ndarray
    objective_constant: float
    bounds: tuple[tuple[float | None, float | None], ...]
    a_eq: np.ndarray
    b_eq: np.ndarray
    a_ub: np.ndarray
    b_ub: np.ndarray

    @property
    def nvar(self) -> int:
        return len(self.names)

    def model_hash(self) -> str:
        payload = {
            "formulation": self.formulation,
            "instance": asdict(self.instance),
            "pattern": self.pattern,
            "names": self.names,
            "objective": self.objective.tolist(),
            "objective_constant": self.objective_constant,
            "bounds": self.bounds,
            "a_eq": self.a_eq.tolist(),
            "b_eq": self.b_eq.tolist(),
            "a_ub": self.a_ub.tolist(),
            "b_ub": self.b_ub.tolist(),
        }
        return canonical_sha256(payload)

    def value(self, vector: np.ndarray, name: str) -> float:
        return float(vector[self.name_to_index[name]])

    def values(
        self, vector: np.ndarray, stem: str, n: int | None = None
    ) -> np.ndarray:
        length = self.instance.n if n is None else n
        return np.asarray(
            [self.value(vector, f"{stem}[{hour}]") for hour in range(length)],
            dtype=float,
        )

    def raw_objective(self, vector: np.ndarray) -> float:
        return float(self.objective @ vector + self.objective_constant)

    def residuals(self, vector: np.ndarray) -> dict[str, float]:
        x = np.asarray(vector, dtype=float)
        equality = (
            float(np.max(np.abs(self.a_eq @ x - self.b_eq)))
            if self.a_eq.size
            else 0.0
        )
        inequality = (
            float(np.max(np.maximum(self.a_ub @ x - self.b_ub, 0.0)))
            if self.a_ub.size
            else 0.0
        )
        bound = 0.0
        for value, (lower, upper) in zip(x, self.bounds):
            if lower is not None:
                bound = max(bound, float(max(lower - value, 0.0)))
            if upper is not None:
                bound = max(bound, float(max(value - upper, 0.0)))
        return {
            "equality": equality,
            "inequality": inequality,
            "bound": bound,
            "maximum": max(equality, inequality, bound),
        }


class LPBuilder:
    def __init__(
        self,
        formulation: str,
        instance: ToyInstance,
        pattern: tuple[int, ...],
    ) -> None:
        self.formulation = formulation
        self.instance = instance
        self.pattern = pattern
        self.names: list[str] = []
        self.name_to_index: dict[str, int] = {}
        self.objective: list[float] = []
        self.bounds: list[tuple[float | None, float | None]] = []
        self.equalities: list[tuple[dict[int, float], float]] = []
        self.inequalities: list[tuple[dict[int, float], float]] = []

    def add_variable(
        self,
        name: str,
        lower: float | None = 0.0,
        upper: float | None = None,
        objective: float = 0.0,
    ) -> int:
        if name in self.name_to_index:
            raise RuntimeError(f"Duplicate LP variable: {name}")
        index = len(self.names)
        self.names.append(name)
        self.name_to_index[name] = index
        self.objective.append(float(objective))
        self.bounds.append((lower, upper))
        return index

    def row(self, coefficients: Mapping[str, float]) -> dict[int, float]:
        output: dict[int, float] = {}
        for name, coefficient in coefficients.items():
            if coefficient == 0.0:
                continue
            index = self.name_to_index[name]
            output[index] = output.get(index, 0.0) + float(coefficient)
        return output

    def add_eq(
        self, coefficients: Mapping[str, float], right_hand_side: float
    ) -> None:
        self.equalities.append((self.row(coefficients), float(right_hand_side)))

    def add_le(
        self, coefficients: Mapping[str, float], right_hand_side: float
    ) -> None:
        self.inequalities.append(
            (self.row(coefficients), float(right_hand_side))
        )

    def build(self, objective_constant: float) -> LinearProgram:
        nvar = len(self.names)

        def matrix(
            rows: list[tuple[dict[int, float], float]]
        ) -> tuple[np.ndarray, np.ndarray]:
            coefficients = np.zeros((len(rows), nvar), dtype=float)
            rhs = np.zeros(len(rows), dtype=float)
            for row_index, (terms, value) in enumerate(rows):
                for column, coefficient in terms.items():
                    coefficients[row_index, column] = coefficient
                rhs[row_index] = value
            return coefficients, rhs

        a_eq, b_eq = matrix(self.equalities)
        a_ub, b_ub = matrix(self.inequalities)
        return LinearProgram(
            formulation=self.formulation,
            instance=self.instance,
            pattern=self.pattern,
            names=tuple(self.names),
            name_to_index=dict(self.name_to_index),
            objective=np.asarray(self.objective, dtype=float),
            objective_constant=float(objective_constant),
            bounds=tuple(self.bounds),
            a_eq=a_eq,
            b_eq=b_eq,
            a_ub=a_ub,
            b_ub=b_ub,
        )


def add_hourly_core_variables(
    builder: LPBuilder,
    include_unprojected: bool,
    centered: bool,
) -> None:
    instance = builder.instance
    builder.add_variable(
        "kg",
        0.0,
        max(instance.task_mw),
        instance.gas_fixed_cost,
    )
    builder.add_variable(
        "xch",
        0.0,
        instance.max_extra_pch_mw,
        instance.extra_pch_cost,
    )
    for hour in range(instance.n):
        if include_unprojected:
            builder.add_variable(
                f"gas[{hour}]",
                0.0,
                instance.task_mw[hour],
                instance.gas_variable_cost,
            )
            builder.add_variable(
                f"recharge_used[{hour}]",
                0.0,
                instance.recharge_mw[hour],
                instance.recharge_price,
            )
        builder.add_variable(
            f"charge[{hour}]",
            0.0,
            instance.recharge_mw[hour],
            instance.recharge_price if centered else 0.0,
        )
        builder.add_variable(
            f"discharge[{hour}]",
            0.0,
            min(instance.task_mw[hour], instance.pdis_mw),
            (
                instance.discharge_vom - instance.gas_variable_cost
                if centered
                else instance.discharge_vom
            ),
        )
        builder.add_variable(
            f"x[{hour}]",
            0.0,
            instance.usable_soc_mwh,
            0.0,
        )
        if include_unprojected:
            builder.add_variable(
                f"spill[{hour}]",
                0.0,
                instance.recharge_mw[hour],
                0.0,
            )
    for position, hour in enumerate(instance.overlap_hours):
        mode = float(builder.pattern[position])
        builder.add_variable(f"z[{hour}]", mode, mode, 0.0)


def build_unprojected_reference(
    instance: ToyInstance, pattern: tuple[int, ...]
) -> LinearProgram:
    if len(pattern) != len(instance.overlap_hours):
        raise ValueError("Pattern length does not match overlap-hour count")
    builder = LPBuilder("unprojected_reference", instance, pattern)
    add_hourly_core_variables(
        builder, include_unprojected=True, centered=False
    )
    mode_by_hour = dict(zip(instance.overlap_hours, pattern))
    for hour in range(instance.n):
        previous = (hour - 1) % instance.n
        builder.add_eq(
            {
                f"gas[{hour}]": 1.0,
                f"discharge[{hour}]": 1.0,
            },
            instance.task_mw[hour],
        )
        builder.add_eq(
            {
                f"recharge_used[{hour}]": 1.0,
                f"spill[{hour}]": 1.0,
            },
            instance.recharge_mw[hour],
        )
        builder.add_eq(
            {
                f"charge[{hour}]": 1.0,
                f"recharge_used[{hour}]": -1.0,
            },
            0.0,
        )
        builder.add_eq(
            {
                f"x[{hour}]": 1.0,
                f"x[{previous}]": -instance.retention,
                f"charge[{hour}]": -instance.eta_ch,
                f"discharge[{hour}]": 1.0 / instance.eta_dis,
            },
            0.0,
        )
        builder.add_le(
            {f"gas[{hour}]": 1.0, "kg": -1.0},
            0.0,
        )
        builder.add_le(
            {f"charge[{hour}]": 1.0, "xch": -1.0},
            instance.native_pch_mw,
        )
        if hour in mode_by_hour:
            mode = mode_by_hour[hour]
            charge_cap = min(
                instance.recharge_mw[hour],
                instance.native_pch_mw + instance.max_extra_pch_mw,
            )
            discharge_cap = min(instance.task_mw[hour], instance.pdis_mw)
            builder.add_le(
                {
                    f"charge[{hour}]": 1.0,
                    f"z[{hour}]": -charge_cap,
                },
                0.0,
            )
            builder.add_le(
                {
                    f"discharge[{hour}]": 1.0,
                    f"z[{hour}]": discharge_cap,
                },
                discharge_cap,
            )
            if mode not in (0, 1):
                raise ValueError("Toy pattern must be binary")
    return builder.build(instance.raw_objective_constant)


def build_projected_state_hull(
    instance: ToyInstance, pattern: tuple[int, ...]
) -> LinearProgram:
    if len(pattern) != len(instance.overlap_hours):
        raise ValueError("Pattern length does not match overlap-hour count")
    builder = LPBuilder("projected_state_hull", instance, pattern)
    add_hourly_core_variables(
        builder, include_unprojected=False, centered=True
    )
    overlap = set(instance.overlap_hours)
    for hour in instance.overlap_hours:
        for branch in ("c", "d"):
            builder.add_variable(
                f"xprev_{branch}[{hour}]",
                0.0,
                instance.usable_soc_mwh,
            )
            builder.add_variable(
                f"xnext_{branch}[{hour}]",
                0.0,
                instance.usable_soc_mwh,
            )
            builder.add_variable(
                f"xch_{branch}[{hour}]",
                0.0,
                instance.max_extra_pch_mw,
            )

    for hour in range(instance.n):
        previous = (hour - 1) % instance.n
        builder.add_le(
            {
                f"discharge[{hour}]": -1.0,
                "kg": -1.0,
            },
            -instance.task_mw[hour],
        )
        builder.add_le(
            {f"charge[{hour}]": 1.0, "xch": -1.0},
            instance.native_pch_mw,
        )
        if hour not in overlap:
            builder.add_eq(
                {
                    f"x[{hour}]": 1.0,
                    f"x[{previous}]": -instance.retention,
                    f"charge[{hour}]": -instance.eta_ch,
                    f"discharge[{hour}]": 1.0 / instance.eta_dis,
                },
                0.0,
            )

    for position, hour in enumerate(instance.overlap_hours):
        previous = (hour - 1) % instance.n
        z = f"z[{hour}]"
        charge_cap = min(
            instance.recharge_mw[hour],
            instance.native_pch_mw + instance.max_extra_pch_mw,
        )
        discharge_cap = min(instance.task_mw[hour], instance.pdis_mw)
        builder.add_eq(
            {
                f"x[{previous}]": 1.0,
                f"xprev_c[{hour}]": -1.0,
                f"xprev_d[{hour}]": -1.0,
            },
            0.0,
        )
        builder.add_eq(
            {
                f"x[{hour}]": 1.0,
                f"xnext_c[{hour}]": -1.0,
                f"xnext_d[{hour}]": -1.0,
            },
            0.0,
        )
        builder.add_eq(
            {
                f"xnext_c[{hour}]": 1.0,
                f"xprev_c[{hour}]": -instance.retention,
                f"charge[{hour}]": -instance.eta_ch,
            },
            0.0,
        )
        builder.add_eq(
            {
                f"xnext_d[{hour}]": 1.0,
                f"xprev_d[{hour}]": -instance.retention,
                f"discharge[{hour}]": 1.0 / instance.eta_dis,
            },
            0.0,
        )
        builder.add_eq(
            {
                "xch": 1.0,
                f"xch_c[{hour}]": -1.0,
                f"xch_d[{hour}]": -1.0,
            },
            0.0,
        )
        for name in (
            f"xprev_c[{hour}]",
            f"xnext_c[{hour}]",
        ):
            builder.add_le(
                {name: 1.0, z: -instance.usable_soc_mwh}, 0.0
            )
        for name in (
            f"xprev_d[{hour}]",
            f"xnext_d[{hour}]",
        ):
            builder.add_le(
                {name: 1.0, z: instance.usable_soc_mwh},
                instance.usable_soc_mwh,
            )
        builder.add_le(
            {
                f"xch_c[{hour}]": 1.0,
                z: -instance.max_extra_pch_mw,
            },
            0.0,
        )
        builder.add_le(
            {
                f"xch_d[{hour}]": 1.0,
                z: instance.max_extra_pch_mw,
            },
            instance.max_extra_pch_mw,
        )
        builder.add_le(
            {
                f"charge[{hour}]": 1.0,
                f"xch_c[{hour}]": -1.0,
                z: -instance.native_pch_mw,
            },
            0.0,
        )
        builder.add_le(
            {f"charge[{hour}]": 1.0, z: -charge_cap},
            0.0,
        )
        builder.add_le(
            {
                f"discharge[{hour}]": 1.0,
                z: discharge_cap,
            },
            discharge_cap,
        )
        if pattern[position] not in (0, 1):
            raise ValueError("Toy pattern must be binary")
    return builder.build(instance.centered_objective_constant)


@dataclass(frozen=True)
class LPSolution:
    success: bool
    status: int
    message: str
    vector: np.ndarray
    variable_objective: float
    raw_objective: float


def solve_lp(
    model: LinearProgram,
    objective: np.ndarray | None = None,
    primary_optimum: float | None = None,
    primary_tolerance: float = 0.0,
) -> LPSolution:
    c = model.objective if objective is None else np.asarray(objective, float)
    a_ub = model.a_ub
    b_ub = model.b_ub
    if primary_optimum is not None:
        tolerance = max(float(primary_tolerance), 1e-10)
        a_ub = np.vstack(
            [a_ub, model.objective, -model.objective]
        )
        b_ub = np.concatenate(
            [
                b_ub,
                [primary_optimum + tolerance, -primary_optimum + tolerance],
            ]
        )
    result = linprog(
        c=c,
        A_ub=a_ub if a_ub.size else None,
        b_ub=b_ub if b_ub.size else None,
        A_eq=model.a_eq if model.a_eq.size else None,
        b_eq=model.b_eq if model.b_eq.size else None,
        bounds=model.bounds,
        method="highs",
        options={
            "primal_feasibility_tolerance": 1e-9,
            "dual_feasibility_tolerance": 1e-9,
            "ipm_optimality_tolerance": 1e-10,
        },
    )
    if result.success:
        vector = np.asarray(result.x, dtype=float)
        variable_objective = float(model.objective @ vector)
        raw_objective = model.raw_objective(vector)
    else:
        vector = np.full(model.nvar, np.nan)
        variable_objective = float("nan")
        raw_objective = float("nan")
    return LPSolution(
        success=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        vector=vector,
        variable_objective=variable_objective,
        raw_objective=raw_objective,
    )


def fingerprint_objective(model: LinearProgram) -> np.ndarray:
    weights = np.zeros(model.nvar, dtype=float)
    weights[model.name_to_index["kg"]] = 0.113
    weights[model.name_to_index["xch"]] = 0.127
    n = model.instance.n
    for hour in range(n):
        weights[model.name_to_index[f"charge[{hour}]"]] = (
            1e-3 * (hour + 1)
        )
        weights[model.name_to_index[f"discharge[{hour}]"]] = (
            1.3e-3 * (n - hour)
        )
        weights[model.name_to_index[f"x[{hour}]"]] = (
            1e-5 * (hour + 1) ** 2
        )
    return weights


def map_unprojected_to_hull(
    reference: LinearProgram,
    hull: LinearProgram,
    reference_vector: np.ndarray,
) -> np.ndarray:
    mapped = np.zeros(hull.nvar, dtype=float)
    for name in ("kg", "xch"):
        mapped[hull.name_to_index[name]] = reference.value(
            reference_vector, name
        )
    for hour in range(reference.instance.n):
        for stem in ("charge", "discharge", "x"):
            name = f"{stem}[{hour}]"
            mapped[hull.name_to_index[name]] = reference.value(
                reference_vector, name
            )
    for position, hour in enumerate(reference.instance.overlap_hours):
        mode = int(reference.pattern[position])
        previous = (hour - 1) % reference.instance.n
        mapped[hull.name_to_index[f"z[{hour}]"]] = float(mode)
        x_previous = reference.value(reference_vector, f"x[{previous}]")
        x_next = reference.value(reference_vector, f"x[{hour}]")
        xch = reference.value(reference_vector, "xch")
        if mode == 1:
            mapped[hull.name_to_index[f"xprev_c[{hour}]"]] = x_previous
            mapped[hull.name_to_index[f"xnext_c[{hour}]"]] = x_next
            mapped[hull.name_to_index[f"xch_c[{hour}]"]] = xch
        else:
            mapped[hull.name_to_index[f"xprev_d[{hour}]"]] = x_previous
            mapped[hull.name_to_index[f"xnext_d[{hour}]"]] = x_next
            mapped[hull.name_to_index[f"xch_d[{hour}]"]] = xch
    return mapped


def map_hull_to_unprojected(
    hull: LinearProgram,
    reference: LinearProgram,
    hull_vector: np.ndarray,
) -> np.ndarray:
    instance = hull.instance
    mapped = np.zeros(reference.nvar, dtype=float)
    for name in ("kg", "xch"):
        mapped[reference.name_to_index[name]] = hull.value(hull_vector, name)
    for hour in range(instance.n):
        charge = hull.value(hull_vector, f"charge[{hour}]")
        discharge = hull.value(hull_vector, f"discharge[{hour}]")
        shifted_soc = hull.value(hull_vector, f"x[{hour}]")
        values = {
            f"gas[{hour}]": instance.task_mw[hour] - discharge,
            f"recharge_used[{hour}]": charge,
            f"charge[{hour}]": charge,
            f"discharge[{hour}]": discharge,
            f"x[{hour}]": shifted_soc,
            f"spill[{hour}]": instance.recharge_mw[hour] - charge,
        }
        for name, value in values.items():
            mapped[reference.name_to_index[name]] = value
    for position, hour in enumerate(instance.overlap_hours):
        mapped[reference.name_to_index[f"z[{hour}]"]] = float(
            reference.pattern[position]
        )
    return mapped


def physical_projection(
    model: LinearProgram, vector: np.ndarray
) -> dict[str, np.ndarray | float]:
    return {
        "kg": model.value(vector, "kg"),
        "xch": model.value(vector, "xch"),
        "charge": model.values(vector, "charge"),
        "discharge": model.values(vector, "discharge"),
        "shifted_soc": model.values(vector, "x"),
        "absolute_soc": (
            model.values(vector, "x") + model.instance.soc_floor_mwh
        ),
    }


def max_projection_difference(
    left: dict[str, np.ndarray | float],
    right: dict[str, np.ndarray | float],
) -> dict[str, float]:
    output: dict[str, float] = {}
    for key in left:
        lhs = np.asarray(left[key], dtype=float)
        rhs = np.asarray(right[key], dtype=float)
        output[key] = float(np.max(np.abs(lhs - rhs)))
    return output


def generate_toy_instance(horizon: int) -> ToyInstance:
    if horizon not in OVERLAP_COUNTS:
        raise ValueError(horizon)
    overlap_count = OVERLAP_COUNTS[horizon]
    overlap = tuple(
        sorted(
            set(
                int(value)
                for value in np.linspace(
                    0, horizon - 1, overlap_count, dtype=int
                )
            )
        )
    )
    if len(overlap) != overlap_count:
        raise RuntimeError("Toy overlap construction is not unique")
    task: list[float] = []
    recharge: list[float] = []
    for hour in range(horizon):
        if hour in overlap:
            task.append(1.1 + 0.17 * ((3 * hour + horizon) % 7))
            recharge.append(0.8 + 0.13 * ((5 * hour + horizon) % 9))
        elif hour % 4 == 0:
            task.append(1.35 + 0.09 * (hour % 5))
            recharge.append(0.0)
        elif hour % 4 == 1:
            task.append(0.0)
            recharge.append(1.05 + 0.08 * (hour % 6))
        elif hour % 4 == 2:
            task.append(0.75 + 0.05 * (hour % 7))
            recharge.append(0.0)
        else:
            task.append(0.0)
            recharge.append(0.55 + 0.07 * (hour % 8))
    rte = 0.81
    return ToyInstance(
        instance_id=f"toy_{horizon}h_k{overlap_count}",
        task_mw=tuple(task),
        recharge_mw=tuple(recharge),
        overlap_hours=overlap,
        pdis_mw=2.2,
        native_pch_mw=0.85,
        max_extra_pch_mw=1.4,
        energy_mwh=6.5,
        soc_min_fraction=0.05,
        soc_max_fraction=0.95,
        eta_ch=math.sqrt(rte),
        eta_dis=math.sqrt(rte),
        retention=0.997,
        gas_fixed_cost=19.0,
        gas_variable_cost=13.0,
        extra_pch_cost=4.2,
        recharge_price=0.7,
        discharge_vom=1.5,
        fixed_capital_cost=11.0,
    )


def mode_patterns(count: int) -> Iterable[tuple[int, ...]]:
    for integer in range(2**count):
        yield tuple((integer >> position) & 1 for position in range(count))


def audit_pattern(
    instance: ToyInstance,
    pattern_index: int,
    pattern: tuple[int, ...],
) -> dict[str, Any]:
    reference = build_unprojected_reference(instance, pattern)
    hull = build_projected_state_hull(instance, pattern)
    reference_solution = solve_lp(reference)
    hull_solution = solve_lp(hull)
    row: dict[str, Any] = {
        "instance_id": instance.instance_id,
        "horizon_h": instance.n,
        "overlap_hours": "|".join(map(str, instance.overlap_hours)),
        "overlap_count": len(instance.overlap_hours),
        "pattern_index": pattern_index,
        "pattern": "".join(map(str, pattern)),
        "reference_success": reference_solution.success,
        "hull_success": hull_solution.success,
        "reference_status": reference_solution.status,
        "hull_status": hull_solution.status,
        "reference_nvar": reference.nvar,
        "hull_nvar": hull.nvar,
        "reference_raw_objective": reference_solution.raw_objective,
        "hull_reconstructed_raw_objective": hull_solution.raw_objective,
        "objective_absolute_difference": float("nan"),
        "reference_to_hull_max_residual": float("nan"),
        "hull_to_reference_max_residual": float("nan"),
        "reference_simultaneous_mw": float("nan"),
        "hull_simultaneous_mw": float("nan"),
        "lex_charge_max_difference_mw": float("nan"),
        "lex_discharge_max_difference_mw": float("nan"),
        "lex_shifted_soc_max_difference_mwh": float("nan"),
        "lex_absolute_soc_max_difference_mwh": float("nan"),
        "raw_centered_ledger_difference": float("nan"),
        "reference_model_sha256": reference.model_hash(),
        "hull_model_sha256": hull.model_hash(),
        "pass": False,
    }
    if not reference_solution.success or not hull_solution.success:
        return row

    objective_difference = abs(
        reference_solution.raw_objective - hull_solution.raw_objective
    )
    reference_mapped = map_unprojected_to_hull(
        reference, hull, reference_solution.vector
    )
    hull_mapped = map_hull_to_unprojected(
        hull, reference, hull_solution.vector
    )
    reference_to_hull_residual = hull.residuals(reference_mapped)["maximum"]
    hull_to_reference_residual = reference.residuals(hull_mapped)["maximum"]
    reference_physical = physical_projection(
        reference, reference_solution.vector
    )
    hull_physical = physical_projection(hull, hull_solution.vector)
    reference_simultaneous = float(
        np.max(
            np.minimum(
                reference_physical["charge"],
                reference_physical["discharge"],
            )
        )
    )
    hull_simultaneous = float(
        np.max(
            np.minimum(
                hull_physical["charge"],
                hull_physical["discharge"],
            )
        )
    )

    primary_tolerance = 5e-9 * max(
        1.0,
        abs(reference_solution.variable_objective),
        abs(hull_solution.variable_objective),
    )
    reference_lex = solve_lp(
        reference,
        objective=fingerprint_objective(reference),
        primary_optimum=reference_solution.variable_objective,
        primary_tolerance=primary_tolerance,
    )
    hull_lex = solve_lp(
        hull,
        objective=fingerprint_objective(hull),
        primary_optimum=hull_solution.variable_objective,
        primary_tolerance=primary_tolerance,
    )
    if reference_lex.success and hull_lex.success:
        lex_difference = max_projection_difference(
            physical_projection(reference, reference_lex.vector),
            physical_projection(hull, hull_lex.vector),
        )
    else:
        lex_difference = {
            "charge": float("inf"),
            "discharge": float("inf"),
            "shifted_soc": float("inf"),
            "absolute_soc": float("inf"),
        }

    mapped_raw_objective = reference.raw_objective(hull_mapped)
    ledger_difference = abs(
        hull_solution.raw_objective - mapped_raw_objective
    )
    passed = (
        objective_difference <= OBJECTIVE_TOLERANCE
        and reference_to_hull_residual <= FEASIBILITY_TOLERANCE
        and hull_to_reference_residual <= FEASIBILITY_TOLERANCE
        and reference_simultaneous <= SIMULTANEOUS_TOLERANCE
        and hull_simultaneous <= SIMULTANEOUS_TOLERANCE
        and lex_difference["charge"] <= TRAJECTORY_TOLERANCE
        and lex_difference["discharge"] <= TRAJECTORY_TOLERANCE
        and lex_difference["shifted_soc"] <= TRAJECTORY_TOLERANCE
        and lex_difference["absolute_soc"] <= TRAJECTORY_TOLERANCE
        and ledger_difference <= OBJECTIVE_TOLERANCE
    )
    row.update(
        {
            "objective_absolute_difference": objective_difference,
            "reference_to_hull_max_residual": reference_to_hull_residual,
            "hull_to_reference_max_residual": hull_to_reference_residual,
            "reference_simultaneous_mw": reference_simultaneous,
            "hull_simultaneous_mw": hull_simultaneous,
            "lex_charge_max_difference_mw": lex_difference["charge"],
            "lex_discharge_max_difference_mw": lex_difference["discharge"],
            "lex_shifted_soc_max_difference_mwh": lex_difference[
                "shifted_soc"
            ],
            "lex_absolute_soc_max_difference_mwh": lex_difference[
                "absolute_soc"
            ],
            "raw_centered_ledger_difference": ledger_difference,
            "pass": passed,
        }
    )
    return row


@dataclass(frozen=True)
class DualBoundCertificate:
    model_sha256: str
    objective_sha256: str
    mode_hour: int
    fixed_mode: int
    status: str
    lower_bound: float | None
    dual_feasibility_residual: float | None
    source: str
    solver: str
    certificate_id: str


@dataclass(frozen=True)
class OBBTDecision:
    mode_hour: int
    decision: str
    fixed_mode: int | None
    branch_zero_admitted: bool
    branch_one_admitted: bool
    branch_zero_lower_bound: float | None
    branch_one_lower_bound: float | None
    reason: str


def objective_cutoff_obbt(
    mode_hours: Sequence[int],
    incumbent_upper_bound: float,
    model_sha256: str,
    objective_sha256: str,
    certificate_provider: Callable[[int, int], DualBoundCertificate],
    allowed_sources: set[str],
    dual_tolerance: float = DUAL_FEASIBILITY_TOLERANCE,
    proof_guard: float = OBBT_PROOF_GUARD,
) -> list[OBBTDecision]:
    """Consume dual-bound certificates and return only proof-safe fixes.

    This interface deliberately has no MIP-start or incumbent-pattern input.
    The incumbent supplies only a scalar objective cutoff.
    """

    if not np.isfinite(incumbent_upper_bound):
        raise ValueError("Incumbent objective cutoff must be finite")

    def admitted(
        certificate: DualBoundCertificate, hour: int, mode: int
    ) -> bool:
        return bool(
            certificate.model_sha256 == model_sha256
            and certificate.objective_sha256 == objective_sha256
            and certificate.mode_hour == hour
            and certificate.fixed_mode == mode
            and certificate.status.lower() == "optimal"
            and certificate.source in allowed_sources
            and certificate.lower_bound is not None
            and np.isfinite(certificate.lower_bound)
            and certificate.dual_feasibility_residual is not None
            and np.isfinite(certificate.dual_feasibility_residual)
            and certificate.dual_feasibility_residual <= dual_tolerance
        )

    decisions: list[OBBTDecision] = []
    for hour in mode_hours:
        certificates = {
            mode: certificate_provider(hour, mode) for mode in (0, 1)
        }
        accepted = {
            mode: admitted(certificates[mode], hour, mode) for mode in (0, 1)
        }
        excluded = {
            mode: bool(
                accepted[mode]
                and float(certificates[mode].lower_bound)
                > incumbent_upper_bound + proof_guard
            )
            for mode in (0, 1)
        }
        if excluded[0] and excluded[1]:
            raise RuntimeError(
                f"Both modes at hour {hour} exceed the certified cutoff; "
                "certificate/model inconsistency"
            )
        if excluded[0]:
            decision = "fixed"
            fixed_mode = 1
            reason = "certified z=0 LP lower bound exceeds incumbent cutoff"
        elif excluded[1]:
            decision = "fixed"
            fixed_mode = 0
            reason = "certified z=1 LP lower bound exceeds incumbent cutoff"
        else:
            decision = "unfixed"
            fixed_mode = None
            if not all(accepted.values()):
                reason = "one or both branch certificates were not admitted"
            else:
                reason = "neither admitted branch lower bound exceeds cutoff"
        decisions.append(
            OBBTDecision(
                mode_hour=int(hour),
                decision=decision,
                fixed_mode=fixed_mode,
                branch_zero_admitted=accepted[0],
                branch_one_admitted=accepted[1],
                branch_zero_lower_bound=certificates[0].lower_bound,
                branch_one_lower_bound=certificates[1].lower_bound,
                reason=reason,
            )
        )
    return decisions


def run_obbt_unit_tests() -> dict[str, Any]:
    model_hash = "a" * 64
    objective_hash = "b" * 64
    allowed = {"analytic_toy_dual"}
    data: dict[tuple[int, int], DualBoundCertificate] = {}

    def certificate(
        hour: int,
        mode: int,
        lower: float,
        *,
        status: str = "optimal",
        residual: float = 0.0,
        source: str = "analytic_toy_dual",
        supplied_model_hash: str = model_hash,
    ) -> None:
        data[(hour, mode)] = DualBoundCertificate(
            model_sha256=supplied_model_hash,
            objective_sha256=objective_hash,
            mode_hour=hour,
            fixed_mode=mode,
            status=status,
            lower_bound=lower,
            dual_feasibility_residual=residual,
            source=source,
            solver="analytic_enumeration",
            certificate_id=f"toy-{hour}-{mode}",
        )

    certificate(0, 0, 101.0)
    certificate(0, 1, 99.0)
    certificate(1, 0, 98.0)
    certificate(1, 1, 102.0)
    certificate(2, 0, 101.0, residual=1e-4)
    certificate(2, 1, 99.0)
    certificate(3, 0, 101.0, supplied_model_hash="c" * 64)
    certificate(3, 1, 99.0)
    certificate(4, 0, 99.5)
    certificate(4, 1, 99.7)

    decisions = objective_cutoff_obbt(
        mode_hours=(0, 1, 2, 3, 4),
        incumbent_upper_bound=100.0,
        model_sha256=model_hash,
        objective_sha256=objective_hash,
        certificate_provider=lambda hour, mode: data[(hour, mode)],
        allowed_sources=allowed,
    )
    expected = {0: 1, 1: 0, 2: None, 3: None, 4: None}
    normal_pass = all(
        decision.fixed_mode == expected[decision.mode_hour]
        for decision in decisions
    )

    inconsistent_data = dict(data)
    for mode in (0, 1):
        inconsistent_data[(5, mode)] = DualBoundCertificate(
            model_sha256=model_hash,
            objective_sha256=objective_hash,
            mode_hour=5,
            fixed_mode=mode,
            status="optimal",
            lower_bound=103.0 + mode,
            dual_feasibility_residual=0.0,
            source="analytic_toy_dual",
            solver="analytic_enumeration",
            certificate_id=f"inconsistent-5-{mode}",
        )
    inconsistency_raised = False
    try:
        objective_cutoff_obbt(
            mode_hours=(5,),
            incumbent_upper_bound=100.0,
            model_sha256=model_hash,
            objective_sha256=objective_hash,
            certificate_provider=lambda hour, mode: inconsistent_data[
                (hour, mode)
            ],
            allowed_sources=allowed,
        )
    except RuntimeError:
        inconsistency_raised = True
    return {
        "status": (
            "pass" if normal_pass and inconsistency_raised else "fail"
        ),
        "normal_case_pass": normal_pass,
        "both_branches_excluded_raises": inconsistency_raised,
        "incumbent_pattern_argument_exists": False,
        "mip_start_pattern_fixed_as_constraint": False,
        "production_dual_backend_status": "pending_not_implemented",
        "decisions": [asdict(value) for value in decisions],
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            json_ready(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(
            f"Append-only v0.7 output already exists and is non-empty: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        instance = generate_toy_instance(horizon)
        for pattern_index, pattern in enumerate(
            mode_patterns(len(instance.overlap_hours))
        ):
            rows.append(audit_pattern(instance, pattern_index, pattern))
    results = pd.DataFrame(rows)
    results_path = output / "toy_pattern_equivalence.csv"
    results.to_csv(results_path, index=False, float_format="%.12g")

    obbt = run_obbt_unit_tests()
    obbt_path = output / "obbt_fail_closed_unit_tests.json"
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
    expected_patterns = sum(2 ** OVERLAP_COUNTS[n] for n in HORIZONS)
    checks = {
        "only_toy_horizons_4_to_24": sorted(
            results["horizon_h"].unique().tolist()
        )
        == list(HORIZONS),
        "all_expected_patterns_enumerated": len(results)
        == expected_patterns,
        "all_reference_models_solved": bool(results["reference_success"].all()),
        "all_hull_models_solved": bool(results["hull_success"].all()),
        "all_pattern_rows_pass": bool(results["pass"].all()),
        "objective_equivalence": maxima["objective_absolute_difference"]
        <= OBJECTIVE_TOLERANCE,
        "reference_to_hull_mapping_feasible": maxima[
            "reference_to_hull_max_residual"
        ]
        <= FEASIBILITY_TOLERANCE,
        "hull_to_reference_mapping_feasible": maxima[
            "hull_to_reference_max_residual"
        ]
        <= FEASIBILITY_TOLERANCE,
        "reference_no_simultaneous_flow": maxima[
            "reference_simultaneous_mw"
        ]
        <= SIMULTANEOUS_TOLERANCE,
        "hull_no_simultaneous_flow": maxima["hull_simultaneous_mw"]
        <= SIMULTANEOUS_TOLERANCE,
        "lexicographic_physical_trajectory_equivalence": max(
            maxima["lex_charge_max_difference_mw"],
            maxima["lex_discharge_max_difference_mw"],
            maxima["lex_shifted_soc_max_difference_mwh"],
            maxima["lex_absolute_soc_max_difference_mwh"],
        )
        <= TRAJECTORY_TOLERANCE,
        "raw_centered_objective_ledger_equivalence": maxima[
            "raw_centered_ledger_difference"
        ]
        <= OBJECTIVE_TOLERANCE,
        "obbt_fail_closed_unit_tests": obbt["status"] == "pass",
        "no_8760_hour_instance": int(results["horizon_h"].max()) <= 24,
        "mip_start_pattern_fixed_as_constraint": False,
    }
    status = "pass" if all(
        value is True
        for key, value in checks.items()
        if key != "mip_start_pattern_fixed_as_constraint"
    ) and checks["mip_start_pattern_fixed_as_constraint"] is False else "fail"
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
        "full_year_benchmark_status": "not_run",
        "horizons_h": HORIZONS,
        "expected_pattern_count": expected_patterns,
        "observed_pattern_count": len(results),
        "maxima": maxima,
        "tolerances": {
            "feasibility": FEASIBILITY_TOLERANCE,
            "objective": OBJECTIVE_TOLERANCE,
            "trajectory": TRAJECTORY_TOLERANCE,
            "simultaneous": SIMULTANEOUS_TOLERANCE,
            "dual_feasibility": DUAL_FEASIBILITY_TOLERANCE,
            "obbt_proof_guard": OBBT_PROOF_GUARD,
        },
        "checks": checks,
        "notes": [
            "The audit exhaustively enumerates selected toy overlap modes, "
            "not all 24 hourly modes.",
            "The production dual-certified OBBT solve backend is deliberately "
            "not implemented or exercised in v0.7.",
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
        "files": {
            "runner": {
                "path": str(script_path.relative_to(ROOT)),
                "sha256": sha256(script_path),
            },
            "design": {
                "path": str(DESIGN.relative_to(ROOT)),
                "sha256": sha256(DESIGN),
            },
            "results": {
                "path": str(results_path.relative_to(ROOT)),
                "sha256": sha256(results_path),
            },
            "obbt_unit_tests": {
                "path": str(obbt_path.relative_to(ROOT)),
                "sha256": sha256(obbt_path),
            },
            "qa": {
                "path": str(qa_path.relative_to(ROOT)),
                "sha256": sha256(qa_path),
            },
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    manifest_path = output / "manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps(json_ready(qa), indent=2, ensure_ascii=False))
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
