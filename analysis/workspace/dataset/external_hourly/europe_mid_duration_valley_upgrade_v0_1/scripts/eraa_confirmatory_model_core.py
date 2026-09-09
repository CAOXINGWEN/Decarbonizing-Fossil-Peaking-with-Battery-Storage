#!/usr/bin/env python3
"""Core utilities for the ERAA 2025 confirmatory model.

The module provides four deliberately separable pieces:

1. synchronized 24-hour block rearrangement across every model zone;
2. exact conservation audits for paired chronologies;
3. a small stochastic transport-and-storage LP used as an executable model
   skeleton; and
4. machine-readable hT/hI/W/DeltaW decision rules.

The LP is suitable for phase-1 country/region pilots and synthetic integrity
tests.  The full ERAA network run must retain the same equations and output
contract while adding the released fleet detail, sequential outage paths and
flow-based network sensitivities described in the frozen model specification.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix


REPORT_TOL = 1e-7
CONSERVATION_RTOL = 1e-10
BALANCE_TOL = 1e-6
OBJECTIVE_RTOL = 1e-8
EXCLUDED_AGGREGATE_ZONE_CODES = {
    "DK00",
    "ISEM",
    "IT00",
    "LU00",
    "NO00",
    "SE00",
}

HOURLY_CONSERVATION_COLUMNS = (
    "load_mw",
    "wind_potential_mw",
    "solar_potential_mw",
    "ror_potential_mw",
    "hydro_inflow_mwh",
    "must_run_mw",
)


@dataclass(frozen=True)
class StorageTechnology:
    """One evidence-labelled technology-duration configuration."""

    config_id: str
    tech_id: str
    duration_h: float
    round_trip_efficiency: float
    self_discharge_pct_day: float
    soc_min_fraction: float
    soc_max_fraction: float
    reference_charge_to_discharge_power_ratio: float
    maximum_charge_to_discharge_power_ratio: float
    annual_discharge_power_cost_eur_kw: float
    annual_charge_power_cost_eur_kw: float
    annual_energy_cost_eur_kwh: float
    variable_om_eur_mwh: float
    degradation_cost_eur_mwh_discharged: float
    main_evidence_domain: bool
    structural_extrapolation: bool

    @property
    def eta_charge(self) -> float:
        return math.sqrt(self.round_trip_efficiency)

    @property
    def eta_discharge(self) -> float:
        return math.sqrt(self.round_trip_efficiency)

    @property
    def hourly_retention(self) -> float:
        daily = 1.0 - self.self_discharge_pct_day / 100.0
        return float(max(daily, 0.0) ** (1.0 / 24.0))

    def validate(self) -> None:
        if not (0 < self.round_trip_efficiency <= 1):
            raise ValueError(f"{self.config_id}: invalid round-trip efficiency")
        if not (0 <= self.soc_min_fraction < self.soc_max_fraction <= 1):
            raise ValueError(f"{self.config_id}: invalid SOC bounds")
        if self.duration_h <= 0:
            raise ValueError(f"{self.config_id}: non-positive duration")
        if self.maximum_charge_to_discharge_power_ratio <= 0:
            raise ValueError(f"{self.config_id}: invalid charging-power ratio")
        if not (
            0
            <= self.reference_charge_to_discharge_power_ratio
            <= self.maximum_charge_to_discharge_power_ratio
        ):
            raise ValueError(
                f"{self.config_id}: invalid reference charging-power ratio"
            )
        costs = (
            self.annual_discharge_power_cost_eur_kw,
            self.annual_charge_power_cost_eur_kw,
            self.annual_energy_cost_eur_kwh,
            self.variable_om_eur_mwh,
            self.degradation_cost_eur_mwh_discharged,
        )
        if any(value < 0 for value in costs):
            raise ValueError(f"{self.config_id}: negative cost parameter")
        if self.tech_id == "li_lfp" and self.degradation_cost_eur_mwh_discharged <= 0:
            raise ValueError(
                f"{self.config_id}: LFP throughput degradation must be positive"
            )
        if self.main_evidence_domain and self.duration_h > 168:
            raise ValueError(
                f"{self.config_id}: >168 h cannot enter the main evidence domain"
            )
        if self.structural_extrapolation and self.main_evidence_domain:
            raise ValueError(
                f"{self.config_id}: structural node cannot be confirmatory"
            )


@dataclass(frozen=True)
class NetworkEdge:
    edge_id: str
    from_zone: str
    to_zone: str
    loss_fraction: float = 0.0

    def validate(self) -> None:
        if self.from_zone == self.to_zone:
            raise ValueError(f"{self.edge_id}: self-loop")
        if not (0 <= self.loss_fraction < 1):
            raise ValueError(f"{self.edge_id}: invalid loss fraction")


@dataclass(frozen=True)
class StochasticTransportCase:
    """Dense arrays for a phase-1 stochastic transport LP.

    Scenario is a flattened weather × outage key.  First-stage firm and
    candidate-storage capacities are shared over this dimension.
    """

    case_id: str
    zones: tuple[str, ...]
    thermal_technologies: tuple[str, ...]
    scenario_ids: tuple[str, ...]
    scenario_probabilities: np.ndarray  # [S]
    load_mw: np.ndarray  # [S, Z, T]
    wind_potential_mw: np.ndarray  # [S, Z, T]
    solar_potential_mw: np.ndarray  # [S, Z, T]
    must_run_mw: np.ndarray  # [S, Z, T]
    thermal_existing_capacity_mw: np.ndarray  # [Z, K]
    thermal_available_fraction: np.ndarray  # [S, Z, K, T]
    thermal_variable_cost_eur_mwh: np.ndarray  # [Z, K]
    firm_addition_technology_index: int
    firm_addition_annual_cost_eur_kw: np.ndarray  # [Z]
    hydro_power_capacity_mw: np.ndarray  # [Z]
    hydro_available_fraction: np.ndarray  # [S, Z, T], may exceed 1 in ERAA
    hydro_energy_capacity_mwh: np.ndarray  # [Z]
    hydro_inflow_mwh: np.ndarray  # [S, Z, T]
    hydro_variable_cost_eur_mwh: np.ndarray  # [Z]
    edges: tuple[NetworkEdge, ...]
    network_capacity_mw: np.ndarray  # [S, E, T]
    deployment_mask: np.ndarray  # [Z], bool

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.load_mw.shape

    @property
    def n_scenarios(self) -> int:
        return self.shape[0]

    @property
    def n_zones(self) -> int:
        return self.shape[1]

    @property
    def n_hours(self) -> int:
        return self.shape[2]

    def with_load_increment(self, total_increment_mw: float) -> "StochasticTransportCase":
        """Add a flat synchronous load increment, allocated by mean load."""

        if total_increment_mw < 0:
            raise ValueError("load increment must be non-negative")
        zone_mean = self.load_mw.mean(axis=(0, 2))
        if zone_mean.sum() <= 0:
            raise ValueError("case has no positive mean load")
        shares = zone_mean / zone_mean.sum()
        increment = shares[None, :, None] * float(total_increment_mw)
        return replace(self, load_mw=self.load_mw + increment)

    def validate(self) -> None:
        s, z, t = self.shape
        if s == 0 or z == 0 or t == 0:
            raise ValueError("empty stochastic transport case")
        if len(self.zones) != z or len(set(self.zones)) != z:
            raise ValueError("zone labels are not unique or do not match arrays")
        if EXCLUDED_AGGREGATE_ZONE_CODES.intersection(self.zones):
            raise ValueError(
                "aggregate/alias zone codes entered an atomic-zone model: "
                f"{sorted(EXCLUDED_AGGREGATE_ZONE_CODES.intersection(self.zones))}"
            )
        if len(self.scenario_ids) != s or len(set(self.scenario_ids)) != s:
            raise ValueError("scenario labels are not unique or do not match arrays")
        probabilities = np.asarray(self.scenario_probabilities, dtype=float)
        if probabilities.shape != (s,) or (probabilities < 0).any():
            raise ValueError("invalid scenario probabilities")
        if not np.isclose(probabilities.sum(), 1.0, rtol=0, atol=1e-12):
            raise ValueError("scenario probabilities must sum to one")

        sz_t_arrays = {
            "wind_potential_mw": self.wind_potential_mw,
            "solar_potential_mw": self.solar_potential_mw,
            "must_run_mw": self.must_run_mw,
            "hydro_inflow_mwh": self.hydro_inflow_mwh,
            "hydro_available_fraction": self.hydro_available_fraction,
        }
        for name, array in sz_t_arrays.items():
            if np.asarray(array).shape != (s, z, t):
                raise ValueError(f"{name} shape mismatch")
        if self.thermal_available_fraction.shape != (
            s,
            z,
            len(self.thermal_technologies),
            t,
        ):
            raise ValueError("thermal availability shape mismatch")
        if self.thermal_existing_capacity_mw.shape != (
            z,
            len(self.thermal_technologies),
        ):
            raise ValueError("thermal capacity shape mismatch")
        if self.thermal_variable_cost_eur_mwh.shape != (
            z,
            len(self.thermal_technologies),
        ):
            raise ValueError("thermal variable-cost shape mismatch")
        if not 0 <= self.firm_addition_technology_index < len(
            self.thermal_technologies
        ):
            raise ValueError("firm addition technology index is invalid")
        for name, array in {
            "firm_addition_annual_cost_eur_kw": self.firm_addition_annual_cost_eur_kw,
            "hydro_power_capacity_mw": self.hydro_power_capacity_mw,
            "hydro_energy_capacity_mwh": self.hydro_energy_capacity_mwh,
            "hydro_variable_cost_eur_mwh": self.hydro_variable_cost_eur_mwh,
            "deployment_mask": self.deployment_mask,
        }.items():
            if np.asarray(array).shape != (z,):
                raise ValueError(f"{name} shape mismatch")
        if len(self.edges) != self.network_capacity_mw.shape[1]:
            raise ValueError("network edge count mismatch")
        if self.network_capacity_mw.shape != (s, len(self.edges), t):
            raise ValueError("network capacity shape mismatch")
        for edge in self.edges:
            edge.validate()
            if edge.from_zone not in self.zones or edge.to_zone not in self.zones:
                raise ValueError(f"{edge.edge_id}: endpoint absent from zones")
        nonnegative = {
            "load": self.load_mw,
            "wind": self.wind_potential_mw,
            "solar": self.solar_potential_mw,
            "must_run": self.must_run_mw,
            "thermal_capacity": self.thermal_existing_capacity_mw,
            "thermal_availability": self.thermal_available_fraction,
            "thermal_variable_cost": self.thermal_variable_cost_eur_mwh,
            "firm_addition_annual_cost": self.firm_addition_annual_cost_eur_kw,
            "hydro_power": self.hydro_power_capacity_mw,
            "hydro_availability": self.hydro_available_fraction,
            "hydro_energy": self.hydro_energy_capacity_mwh,
            "hydro_inflow": self.hydro_inflow_mwh,
            "hydro_variable_cost": self.hydro_variable_cost_eur_mwh,
            "network_capacity": self.network_capacity_mw,
        }
        for name, array in nonnegative.items():
            if not np.isfinite(array).all() or (np.asarray(array) < 0).any():
                raise ValueError(f"{name} contains negative or non-finite values")
        if (self.thermal_available_fraction > 1 + 1e-9).any():
            raise ValueError("thermal available fraction exceeds one")
        # ERAA publishes MAX_TURB trajectories in absolute GW.  A few
        # reported constraint/capacity ratios exceed one; retaining those
        # values is part of the source audit, so hydro availability is not
        # clipped at one here.


@dataclass(frozen=True)
class SolveSettings:
    candidate_total_discharge_power_mw: float
    value_of_lost_load_eur_mwh: float = 15_000.0
    renewable_spill_penalty_eur_mwh: float = 1e-6
    hydro_spill_penalty_eur_mwh: float = 1e-7
    flow_regularization_eur_mwh: float = 1e-8
    maximum_firm_addition_mw_per_zone: float = 100_000.0
    solver_tolerance: float = 1e-8


@dataclass
class ModelSolution:
    variant: str
    objective_eur_year: float
    first_stage: pd.DataFrame
    summary: dict[str, Any]
    dispatch: pd.DataFrame
    balance_duals: pd.DataFrame
    raw_status: int
    raw_message: str


@dataclass(frozen=True)
class IndexLayout:
    nvar: int
    firm_add: np.ndarray
    pdis: np.ndarray
    pcharge: np.ndarray
    pcharge_incremental: np.ndarray
    thermal: np.ndarray
    hydro: np.ndarray
    reservoir: np.ndarray
    hydro_spill: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray
    soc: np.ndarray
    vre_spill: np.ndarray
    unserved: np.ndarray
    flow: np.ndarray


def allocate_indices(
    n_scenarios: int,
    n_zones: int,
    n_thermal: int,
    n_hours: int,
    n_edges: int,
) -> IndexLayout:
    cursor = 0
    arrays: dict[str, np.ndarray] = {}

    def allocate(name: str, shape: tuple[int, ...]) -> None:
        nonlocal cursor
        size = int(np.prod(shape))
        arrays[name] = np.arange(cursor, cursor + size, dtype=int).reshape(shape)
        cursor += size

    allocate("firm_add", (n_zones,))
    allocate("pdis", (n_zones,))
    allocate("pcharge", (n_zones,))
    allocate("pcharge_incremental", (n_zones,))
    allocate("thermal", (n_scenarios, n_zones, n_thermal, n_hours))
    for name in (
        "hydro",
        "reservoir",
        "hydro_spill",
        "charge",
        "discharge",
        "soc",
        "vre_spill",
        "unserved",
    ):
        allocate(name, (n_scenarios, n_zones, n_hours))
    allocate("flow", (n_scenarios, n_edges, n_hours))
    return IndexLayout(nvar=cursor, **arrays)


def _append(
    rows: list[int],
    cols: list[int],
    data: list[float],
    row: int,
    column: int,
    value: float,
) -> None:
    rows.append(row)
    cols.append(int(column))
    data.append(float(value))


def solve_stochastic_transport_lp(
    case: StochasticTransportCase,
    storage: StorageTechnology,
    settings: SolveSettings,
    variant: str,
    baseline_firm_addition_mw: np.ndarray | None = None,
    keep_dispatch: bool = True,
) -> ModelSolution:
    """Solve a phase-1 stochastic transport and candidate-storage LP.

    Variants:

    - ``baseline``: no candidate storage; firm additions endogenous.
    - ``joint``: a signed forced storage tranche; firm additions endogenous.
    - ``turnover``: the same tranche with firm additions fixed to the
      no-storage baseline.
    """

    case.validate()
    storage.validate()
    if variant not in {"baseline", "joint", "turnover"}:
        raise ValueError(f"unknown variant {variant}")
    if variant == "turnover":
        if baseline_firm_addition_mw is None:
            raise ValueError("turnover variant requires baseline firm additions")
        if np.asarray(baseline_firm_addition_mw).shape != (case.n_zones,):
            raise ValueError("baseline firm-addition vector shape mismatch")

    s_count, z_count, t_count = case.shape
    k_count = len(case.thermal_technologies)
    e_count = len(case.edges)
    idx = allocate_indices(s_count, z_count, k_count, t_count, e_count)

    c = np.zeros(idx.nvar, dtype=float)
    c[idx.firm_add] = case.firm_addition_annual_cost_eur_kw * 1_000.0
    if variant != "baseline":
        c[idx.pdis] = storage.annual_discharge_power_cost_eur_kw * 1_000.0
        c[idx.pcharge_incremental] = (
            storage.annual_charge_power_cost_eur_kw * 1_000.0
        )
        c[idx.pdis] += (
            storage.duration_h
            * storage.annual_energy_cost_eur_kwh
            * 1_000.0
        )
    for scenario in range(s_count):
        weight = float(case.scenario_probabilities[scenario])
        for zone in range(z_count):
            for technology in range(k_count):
                c[idx.thermal[scenario, zone, technology, :]] = (
                    weight * case.thermal_variable_cost_eur_mwh[zone, technology]
                )
            c[idx.hydro[scenario, zone, :]] = (
                weight * case.hydro_variable_cost_eur_mwh[zone]
            )
            c[idx.hydro_spill[scenario, zone, :]] = (
                weight * settings.hydro_spill_penalty_eur_mwh
            )
            c[idx.discharge[scenario, zone, :]] = weight * (
                storage.variable_om_eur_mwh
                + storage.degradation_cost_eur_mwh_discharged
            )
            c[idx.vre_spill[scenario, zone, :]] = (
                weight * settings.renewable_spill_penalty_eur_mwh
            )
            c[idx.unserved[scenario, zone, :]] = (
                weight * settings.value_of_lost_load_eur_mwh
            )
        if e_count:
            c[idx.flow[scenario, :, :]] = (
                weight * settings.flow_regularization_eur_mwh
            )

    eq_rows: list[int] = []
    eq_cols: list[int] = []
    eq_data: list[float] = []
    b_eq: list[float] = []
    balance_row_lookup = np.empty((s_count, z_count, t_count), dtype=int)
    eq_row = 0
    zone_lookup = {zone: index for index, zone in enumerate(case.zones)}
    outgoing: dict[int, list[int]] = {zone: [] for zone in range(z_count)}
    incoming: dict[int, list[int]] = {zone: [] for zone in range(z_count)}
    for edge_index, edge in enumerate(case.edges):
        outgoing[zone_lookup[edge.from_zone]].append(edge_index)
        incoming[zone_lookup[edge.to_zone]].append(edge_index)

    # Nodal balances.
    for scenario in range(s_count):
        for zone in range(z_count):
            for hour in range(t_count):
                balance_row_lookup[scenario, zone, hour] = eq_row
                for technology in range(k_count):
                    _append(
                        eq_rows,
                        eq_cols,
                        eq_data,
                        eq_row,
                        idx.thermal[scenario, zone, technology, hour],
                        1.0,
                    )
                for variable, coefficient in (
                    (idx.hydro[scenario, zone, hour], 1.0),
                    (idx.discharge[scenario, zone, hour], 1.0),
                    (idx.charge[scenario, zone, hour], -1.0),
                    (idx.vre_spill[scenario, zone, hour], -1.0),
                    (idx.unserved[scenario, zone, hour], 1.0),
                ):
                    _append(
                        eq_rows,
                        eq_cols,
                        eq_data,
                        eq_row,
                        variable,
                        coefficient,
                    )
                for edge_index in outgoing[zone]:
                    _append(
                        eq_rows,
                        eq_cols,
                        eq_data,
                        eq_row,
                        idx.flow[scenario, edge_index, hour],
                        -1.0,
                    )
                for edge_index in incoming[zone]:
                    edge = case.edges[edge_index]
                    _append(
                        eq_rows,
                        eq_cols,
                        eq_data,
                        eq_row,
                        idx.flow[scenario, edge_index, hour],
                        1.0 - edge.loss_fraction,
                    )
                b_eq.append(
                    float(
                        case.load_mw[scenario, zone, hour]
                        - case.wind_potential_mw[scenario, zone, hour]
                        - case.solar_potential_mw[scenario, zone, hour]
                        - case.must_run_mw[scenario, zone, hour]
                    )
                )
                eq_row += 1

    # Candidate-storage cyclic SOC.
    for scenario in range(s_count):
        for zone in range(z_count):
            for hour in range(t_count):
                previous = (hour - 1) % t_count
                for variable, coefficient in (
                    (idx.soc[scenario, zone, hour], 1.0),
                    (
                        idx.soc[scenario, zone, previous],
                        -storage.hourly_retention,
                    ),
                    (
                        idx.charge[scenario, zone, hour],
                        -storage.eta_charge,
                    ),
                    (
                        idx.discharge[scenario, zone, hour],
                        1.0 / storage.eta_discharge,
                    ),
                ):
                    _append(
                        eq_rows,
                        eq_cols,
                        eq_data,
                        eq_row,
                        variable,
                        coefficient,
                    )
                b_eq.append(0.0)
                eq_row += 1

    # Reservoir cyclic stock.  Inflow, generation and explicit hydro spill
    # are all energy quantities over one hourly interval.
    for scenario in range(s_count):
        for zone in range(z_count):
            for hour in range(t_count):
                previous = (hour - 1) % t_count
                for variable, coefficient in (
                    (idx.reservoir[scenario, zone, hour], 1.0),
                    (idx.reservoir[scenario, zone, previous], -1.0),
                    (idx.hydro[scenario, zone, hour], 1.0),
                    (idx.hydro_spill[scenario, zone, hour], 1.0),
                ):
                    _append(
                        eq_rows,
                        eq_cols,
                        eq_data,
                        eq_row,
                        variable,
                        coefficient,
                    )
                b_eq.append(float(case.hydro_inflow_mwh[scenario, zone, hour]))
                eq_row += 1

    # Forced total candidate tranche.  The baseline has zero via bounds.
    if variant != "baseline":
        for zone in range(z_count):
            _append(
                eq_rows,
                eq_cols,
                eq_data,
                eq_row,
                idx.pdis[zone],
                1.0,
            )
        b_eq.append(float(settings.candidate_total_discharge_power_mw))
        eq_row += 1

    a_eq = coo_matrix(
        (eq_data, (eq_rows, eq_cols)), shape=(eq_row, idx.nvar)
    ).tocsr()
    b_eq_array = np.asarray(b_eq, dtype=float)

    ub_rows: list[int] = []
    ub_cols: list[int] = []
    ub_data: list[float] = []
    b_ub: list[float] = []
    ub_row = 0

    for scenario in range(s_count):
        for zone in range(z_count):
            for technology in range(k_count):
                for hour in range(t_count):
                    availability = float(
                        case.thermal_available_fraction[
                            scenario, zone, technology, hour
                        ]
                    )
                    _append(
                        ub_rows,
                        ub_cols,
                        ub_data,
                        ub_row,
                        idx.thermal[scenario, zone, technology, hour],
                        1.0,
                    )
                    if technology == case.firm_addition_technology_index:
                        _append(
                            ub_rows,
                            ub_cols,
                            ub_data,
                            ub_row,
                            idx.firm_add[zone],
                            -availability,
                        )
                    b_ub.append(
                        availability
                        * case.thermal_existing_capacity_mw[zone, technology]
                    )
                    ub_row += 1
            for hour in range(t_count):
                # Hydro power.
                _append(
                    ub_rows,
                    ub_cols,
                    ub_data,
                    ub_row,
                    idx.hydro[scenario, zone, hour],
                    1.0,
                )
                b_ub.append(float(case.hydro_power_capacity_mw[zone]))
                b_ub[-1] *= float(
                    case.hydro_available_fraction[scenario, zone, hour]
                )
                ub_row += 1

                # Candidate charge/discharge interfaces.
                _append(
                    ub_rows,
                    ub_cols,
                    ub_data,
                    ub_row,
                    idx.charge[scenario, zone, hour],
                    1.0,
                )
                _append(
                    ub_rows,
                    ub_cols,
                    ub_data,
                    ub_row,
                    idx.pcharge[zone],
                    -1.0,
                )
                b_ub.append(0.0)
                ub_row += 1

                _append(
                    ub_rows,
                    ub_cols,
                    ub_data,
                    ub_row,
                    idx.discharge[scenario, zone, hour],
                    1.0,
                )
                _append(
                    ub_rows,
                    ub_cols,
                    ub_data,
                    ub_row,
                    idx.pdis[zone],
                    -1.0,
                )
                b_ub.append(0.0)
                ub_row += 1

                # SOC upper and lower bounds with E = duration * Pdis.
                _append(
                    ub_rows,
                    ub_cols,
                    ub_data,
                    ub_row,
                    idx.soc[scenario, zone, hour],
                    1.0,
                )
                _append(
                    ub_rows,
                    ub_cols,
                    ub_data,
                    ub_row,
                    idx.pdis[zone],
                    -storage.soc_max_fraction * storage.duration_h,
                )
                b_ub.append(0.0)
                ub_row += 1

                _append(
                    ub_rows,
                    ub_cols,
                    ub_data,
                    ub_row,
                    idx.soc[scenario, zone, hour],
                    -1.0,
                )
                _append(
                    ub_rows,
                    ub_cols,
                    ub_data,
                    ub_row,
                    idx.pdis[zone],
                    storage.soc_min_fraction * storage.duration_h,
                )
                b_ub.append(0.0)
                ub_row += 1

                # Spill cannot exceed weather-driven wind plus solar potential.
                _append(
                    ub_rows,
                    ub_cols,
                    ub_data,
                    ub_row,
                    idx.vre_spill[scenario, zone, hour],
                    1.0,
                )
                b_ub.append(
                    float(
                        case.wind_potential_mw[scenario, zone, hour]
                        + case.solar_potential_mw[scenario, zone, hour]
                    )
                )
                ub_row += 1

    for zone in range(z_count):
        # Charging power is an endogenous design variable.
        _append(
            ub_rows,
            ub_cols,
            ub_data,
            ub_row,
            idx.pcharge[zone],
            1.0,
        )
        _append(
            ub_rows,
            ub_cols,
            ub_data,
            ub_row,
            idx.pdis[zone],
            -storage.maximum_charge_to_discharge_power_ratio,
        )
        b_ub.append(0.0)
        ub_row += 1

        # The integrated power block is paid once through Pdis.  Only charge
        # interface capacity above the source-library reference ratio is
        # charged as an incremental design.  This prevents counting a bundled
        # turbine+compressor+BOP block once on each interface.
        _append(
            ub_rows,
            ub_cols,
            ub_data,
            ub_row,
            idx.pcharge[zone],
            1.0,
        )
        _append(
            ub_rows,
            ub_cols,
            ub_data,
            ub_row,
            idx.pdis[zone],
            -storage.reference_charge_to_discharge_power_ratio,
        )
        _append(
            ub_rows,
            ub_cols,
            ub_data,
            ub_row,
            idx.pcharge_incremental[zone],
            -1.0,
        )
        b_ub.append(0.0)
        ub_row += 1

    a_ub = coo_matrix(
        (ub_data, (ub_rows, ub_cols)), shape=(ub_row, idx.nvar)
    ).tocsr()
    b_ub_array = np.asarray(b_ub, dtype=float)

    bounds: list[tuple[float | None, float | None]] = [
        (0.0, None)
    ] * idx.nvar
    for zone in range(z_count):
        if variant == "turnover":
            fixed = float(np.asarray(baseline_firm_addition_mw)[zone])
            bounds[int(idx.firm_add[zone])] = (fixed, fixed)
        else:
            bounds[int(idx.firm_add[zone])] = (
                0.0,
                settings.maximum_firm_addition_mw_per_zone,
            )
        if variant == "baseline" or not bool(case.deployment_mask[zone]):
            bounds[int(idx.pdis[zone])] = (0.0, 0.0)
            bounds[int(idx.pcharge[zone])] = (0.0, 0.0)
            bounds[int(idx.pcharge_incremental[zone])] = (0.0, 0.0)
    for scenario in range(s_count):
        for zone in range(z_count):
            for hour in range(t_count):
                bounds[int(idx.reservoir[scenario, zone, hour])] = (
                    0.0,
                    float(case.hydro_energy_capacity_mwh[zone]),
                )
        for edge_index in range(e_count):
            for hour in range(t_count):
                bounds[int(idx.flow[scenario, edge_index, hour])] = (
                    0.0,
                    float(case.network_capacity_mw[scenario, edge_index, hour]),
                )

    result = linprog(
        c,
        A_ub=a_ub,
        b_ub=b_ub_array,
        A_eq=a_eq,
        b_eq=b_eq_array,
        bounds=bounds,
        method="highs",
        options={
            "presolve": True,
            "dual_feasibility_tolerance": settings.solver_tolerance,
            "primal_feasibility_tolerance": settings.solver_tolerance,
        },
    )
    if not result.success:
        raise RuntimeError(
            f"{case.case_id}-{storage.config_id}-{variant}: {result.message}"
        )

    x = result.x
    equality_residual = a_eq @ x - b_eq_array
    inequality_residual = np.maximum(a_ub @ x - b_ub_array, 0.0)
    maximum_balance_residual = float(
        np.max(
            np.abs(
                equality_residual[
                    balance_row_lookup.reshape(-1)
                ]
            )
        )
    )
    maximum_eq_residual = float(np.max(np.abs(equality_residual)))
    maximum_ub_residual = float(np.max(inequality_residual))
    objective_recalculated = float(c @ x)
    objective_error = abs(objective_recalculated - float(result.fun))
    soc_values = x[idx.soc]
    soc_transition_residual = (
        soc_values
        - storage.hourly_retention * np.roll(soc_values, 1, axis=2)
        - storage.eta_charge * x[idx.charge]
        + x[idx.discharge] / storage.eta_discharge
    )
    reservoir_values = x[idx.reservoir]
    hydro_stock_residual = (
        reservoir_values
        - np.roll(reservoir_values, 1, axis=2)
        + x[idx.hydro]
        + x[idx.hydro_spill]
        - case.hydro_inflow_mwh
    )
    maximum_soc_transition_residual = float(
        np.max(np.abs(soc_transition_residual))
    )
    maximum_hydro_stock_residual = float(
        np.max(np.abs(hydro_stock_residual))
    )
    if maximum_eq_residual > BALANCE_TOL or maximum_ub_residual > BALANCE_TOL:
        raise RuntimeError(
            "constraint audit failed: "
            f"eq={maximum_eq_residual}, ub={maximum_ub_residual}"
        )
    if objective_error > max(
        1e-6, OBJECTIVE_RTOL * max(abs(objective_recalculated), 1.0)
    ):
        raise RuntimeError(
            f"objective decomposition failed: error={objective_error}"
        )

    first_stage = pd.DataFrame(
        {
            "zone": case.zones,
            "firm_addition_mw": x[idx.firm_add],
            "candidate_discharge_power_mw": x[idx.pdis],
            "candidate_charge_power_mw": x[idx.pcharge],
            "candidate_incremental_charge_power_mw": (
                x[idx.pcharge_incremental]
            ),
            "candidate_energy_capacity_mwh": (
                x[idx.pdis] * storage.duration_h
            ),
        }
    )

    dispatch_rows: list[pd.DataFrame] = []
    dual_rows: list[pd.DataFrame] = []
    balance_marginals = np.asarray(result.eqlin.marginals)[
        balance_row_lookup
    ]
    for scenario in range(s_count):
        for zone in range(z_count):
            frame = pd.DataFrame(
                {
                    "scenario_id": case.scenario_ids[scenario],
                    "scenario_probability": case.scenario_probabilities[scenario],
                    "zone": case.zones[zone],
                    "hour_of_year": np.arange(t_count),
                    "load_mw": case.load_mw[scenario, zone],
                    "wind_potential_mw": case.wind_potential_mw[scenario, zone],
                    "solar_potential_mw": case.solar_potential_mw[scenario, zone],
                    "hydro_generation_mw": x[idx.hydro[scenario, zone]],
                    "hydro_reservoir_mwh": x[idx.reservoir[scenario, zone]],
                    "hydro_spill_mwh": x[idx.hydro_spill[scenario, zone]],
                    "charge_mw": x[idx.charge[scenario, zone]],
                    "discharge_mw": x[idx.discharge[scenario, zone]],
                    "soc_mwh": x[idx.soc[scenario, zone]],
                    "vre_spill_mw": x[idx.vre_spill[scenario, zone]],
                    "unserved_mw": x[idx.unserved[scenario, zone]],
                }
            )
            for technology, name in enumerate(case.thermal_technologies):
                frame[f"thermal_{name}_mw"] = x[
                    idx.thermal[scenario, zone, technology]
                ]
            dispatch_rows.append(frame)
            dual_rows.append(
                pd.DataFrame(
                    {
                        "scenario_id": case.scenario_ids[scenario],
                        "scenario_probability": case.scenario_probabilities[
                            scenario
                        ],
                        "zone": case.zones[zone],
                        "hour_of_year": np.arange(t_count),
                        "balance_dual_eur_mwh": balance_marginals[
                            scenario, zone
                        ],
                        "charge_mw": x[idx.charge[scenario, zone]],
                    }
                )
            )

    dispatch = pd.concat(dispatch_rows, ignore_index=True)
    balance_duals = pd.concat(dual_rows, ignore_index=True)
    dispatch["weighted_unserved_mwh"] = (
        dispatch["unserved_mw"] * dispatch["scenario_probability"]
    )
    dispatch["weighted_discharge_mwh"] = (
        dispatch["discharge_mw"] * dispatch["scenario_probability"]
    )
    dispatch["weighted_charge_mwh"] = (
        dispatch["charge_mw"] * dispatch["scenario_probability"]
    )
    dispatch["weighted_spill_mwh"] = (
        dispatch["vre_spill_mw"] * dispatch["scenario_probability"]
    )
    balance_duals["weighted_charging_opportunity_cost_eur"] = (
        balance_duals["charge_mw"]
        * balance_duals["balance_dual_eur_mwh"]
        * balance_duals["scenario_probability"]
    )

    # LOLE counts an hour once per scenario if any model zone has unserved
    # energy.  It is not the sum of zone-level loss-of-load hours.
    scenario_hour = (
        dispatch.groupby(
            ["scenario_id", "scenario_probability", "hour_of_year"],
            as_index=False,
        )["unserved_mw"]
        .sum()
    )
    expected_lole = float(
        scenario_hour.loc[
            scenario_hour["unserved_mw"].gt(REPORT_TOL),
            "scenario_probability",
        ].sum()
    )
    expected_eens = float(dispatch["weighted_unserved_mwh"].sum())
    total_pdis = float(first_stage["candidate_discharge_power_mw"].sum())
    usable_energy = (
        total_pdis
        * storage.duration_h
        * (storage.soc_max_fraction - storage.soc_min_fraction)
    )
    rated_energy = total_pdis * storage.duration_h
    expected_discharge = float(dispatch["weighted_discharge_mwh"].sum())
    expected_charge = float(dispatch["weighted_charge_mwh"].sum())
    firm_capacity_cost = float(
        np.sum(
            x[idx.firm_add]
            * case.firm_addition_annual_cost_eur_kw
            * 1_000.0
        )
    )
    storage_discharge_power_cost = float(
        np.sum(x[idx.pdis])
        * storage.annual_discharge_power_cost_eur_kw
        * 1_000.0
    )
    storage_charge_power_cost = float(
        np.sum(x[idx.pcharge_incremental])
        * storage.annual_charge_power_cost_eur_kw
        * 1_000.0
    )
    storage_energy_cost = float(
        np.sum(x[idx.pdis])
        * storage.duration_h
        * storage.annual_energy_cost_eur_kwh
        * 1_000.0
    )
    thermal_variable_cost = 0.0
    expected_generation_by_technology: dict[str, float] = {}
    for technology, name in enumerate(case.thermal_technologies):
        generation = 0.0
        technology_cost = 0.0
        for scenario in range(s_count):
            weight = float(case.scenario_probabilities[scenario])
            values = x[idx.thermal[scenario, :, technology, :]]
            generation += weight * float(values.sum())
            technology_cost += weight * float(
                np.sum(
                    values
                    * case.thermal_variable_cost_eur_mwh[:, technology, None]
                )
            )
        expected_generation_by_technology[name] = generation
        thermal_variable_cost += technology_cost
    hydro_variable_cost = 0.0
    hydro_spill_cost = 0.0
    storage_variable_cost = 0.0
    renewable_spill_cost = 0.0
    unserved_energy_cost = 0.0
    flow_regularization_cost = 0.0
    for scenario in range(s_count):
        weight = float(case.scenario_probabilities[scenario])
        hydro_variable_cost += weight * float(
            np.sum(
                x[idx.hydro[scenario]]
                * case.hydro_variable_cost_eur_mwh[:, None]
            )
        )
        hydro_spill_cost += (
            weight
            * settings.hydro_spill_penalty_eur_mwh
            * float(x[idx.hydro_spill[scenario]].sum())
        )
        storage_variable_cost += weight * (
            storage.variable_om_eur_mwh
            + storage.degradation_cost_eur_mwh_discharged
        ) * float(x[idx.discharge[scenario]].sum())
        renewable_spill_cost += (
            weight
            * settings.renewable_spill_penalty_eur_mwh
            * float(x[idx.vre_spill[scenario]].sum())
        )
        unserved_energy_cost += (
            weight
            * settings.value_of_lost_load_eur_mwh
            * float(x[idx.unserved[scenario]].sum())
        )
        if e_count:
            flow_regularization_cost += (
                weight
                * settings.flow_regularization_eur_mwh
                * float(x[idx.flow[scenario]].sum())
            )
    objective_components = {
        "firm_capacity_cost_eur_year": firm_capacity_cost,
        "storage_discharge_power_cost_eur_year": storage_discharge_power_cost,
        "storage_charge_power_cost_eur_year": storage_charge_power_cost,
        "storage_energy_cost_eur_year": storage_energy_cost,
        "thermal_variable_cost_eur_year": thermal_variable_cost,
        "hydro_variable_cost_eur_year": hydro_variable_cost,
        "hydro_spill_cost_eur_year": hydro_spill_cost,
        "storage_variable_and_degradation_cost_eur_year": storage_variable_cost,
        "renewable_spill_cost_eur_year": renewable_spill_cost,
        "unserved_energy_cost_eur_year": unserved_energy_cost,
        "flow_regularization_cost_eur_year": flow_regularization_cost,
    }
    component_total = float(sum(objective_components.values()))
    component_error = abs(component_total - float(result.fun))
    if component_error > max(
        1e-5, OBJECTIVE_RTOL * max(abs(float(result.fun)), 1.0)
    ):
        raise RuntimeError(
            "objective component audit failed: "
            f"components={component_total}, solver={result.fun}, "
            f"error={component_error}"
        )
    simultaneous = np.minimum(
        dispatch["charge_mw"].to_numpy(),
        dispatch["discharge_mw"].to_numpy(),
    )
    expected_simultaneous = float(
        np.sum(
            simultaneous
            * dispatch["scenario_probability"].to_numpy()
        )
    )
    charging_opportunity_cost = float(
        balance_duals["weighted_charging_opportunity_cost_eur"].sum()
    )
    summary = {
        "case_id": case.case_id,
        "config_id": storage.config_id,
        "tech_id": storage.tech_id,
        "duration_h": storage.duration_h,
        "variant": variant,
        "objective_eur_year": float(result.fun),
        "expected_eens_mwh": expected_eens,
        "expected_lole_h": expected_lole,
        "expected_charge_mwh": expected_charge,
        "expected_discharge_mwh": expected_discharge,
        "expected_vre_spill_mwh": float(dispatch["weighted_spill_mwh"].sum()),
        "equivalent_full_cycles_output_basis": (
            expected_discharge / rated_energy
            if rated_energy > REPORT_TOL
            else 0.0
        ),
        "candidate_efc_rated_energy": (
            expected_discharge / rated_energy
            if rated_energy > REPORT_TOL
            else 0.0
        ),
        "candidate_efc_usable_soc_window": (
            expected_discharge / usable_energy
            if usable_energy > REPORT_TOL
            else 0.0
        ),
        "candidate_efc_internal_withdrawal_usable_window": (
            expected_discharge
            / storage.eta_discharge
            / usable_energy
            if usable_energy > REPORT_TOL
            else 0.0
        ),
        "equivalent_full_cycles_definition": (
            "delivered_discharge_mwh_divided_by_rated_energy_mwh"
        ),
        "charging_opportunity_cost_from_balance_duals_eur": (
            charging_opportunity_cost
        ),
        "expected_simultaneous_charge_discharge_mwh": expected_simultaneous,
        "expected_generation_by_technology_mwh": (
            expected_generation_by_technology
        ),
        **objective_components,
        "objective_component_recalculation_error_eur": component_error,
        "lfp_degradation_cost_eur_mwh": (
            storage.degradation_cost_eur_mwh_discharged
        ),
        "maximum_balance_residual_mw": maximum_balance_residual,
        "maximum_equality_residual": maximum_eq_residual,
        "maximum_inequality_residual": maximum_ub_residual,
        "maximum_soc_transition_residual_mwh": (
            maximum_soc_transition_residual
        ),
        "maximum_hydro_stock_residual_mwh": maximum_hydro_stock_residual,
        "objective_recalculation_error_eur": objective_error,
        "solver_iterations": int(result.nit),
    }
    return ModelSolution(
        variant=variant,
        objective_eur_year=float(result.fun),
        first_stage=first_stage,
        summary=summary,
        dispatch=dispatch if keep_dispatch else pd.DataFrame(),
        balance_duals=balance_duals if keep_dispatch else pd.DataFrame(),
        raw_status=int(result.status),
        raw_message=str(result.message),
    )


def solve_signed_value_variants(
    case: StochasticTransportCase,
    storage: StorageTechnology,
    settings: SolveSettings,
    keep_dispatch: bool = True,
) -> dict[str, ModelSolution]:
    baseline = solve_stochastic_transport_lp(
        case,
        storage,
        settings,
        "baseline",
        keep_dispatch=keep_dispatch,
    )
    baseline_firm = (
        baseline.first_stage.set_index("zone")
        .loc[list(case.zones), "firm_addition_mw"]
        .to_numpy()
    )
    joint = solve_stochastic_transport_lp(
        case,
        storage,
        settings,
        "joint",
        keep_dispatch=keep_dispatch,
    )
    turnover = solve_stochastic_transport_lp(
        case,
        storage,
        settings,
        "turnover",
        baseline_firm_addition_mw=baseline_firm,
        keep_dispatch=keep_dispatch,
    )
    tranche = settings.candidate_total_discharge_power_mw
    if tranche <= 0:
        raise ValueError("signed value variants require a positive tranche")
    joint.summary["signed_joint_benefit_eur_kw_year"] = (
        baseline.objective_eur_year - joint.objective_eur_year
    ) / (tranche * 1_000.0)
    turnover.summary["signed_turnover_benefit_eur_kw_year"] = (
        baseline.objective_eur_year - turnover.objective_eur_year
    ) / (tranche * 1_000.0)
    joint.summary["residual_capacity_and_interaction_benefit_eur_kw_year"] = (
        joint.summary["signed_joint_benefit_eur_kw_year"]
        - turnover.summary["signed_turnover_benefit_eur_kw_year"]
    )
    return {"baseline": baseline, "joint": joint, "turnover": turnover}


def expected_reliability(dispatch: pd.DataFrame) -> dict[str, float]:
    required = {
        "scenario_id",
        "scenario_probability",
        "hour_of_year",
        "zone",
        "unserved_mw",
    }
    missing = required.difference(dispatch.columns)
    if missing:
        raise ValueError(f"dispatch missing reliability columns {sorted(missing)}")
    expected_eens = float(
        (
            dispatch["unserved_mw"]
            * dispatch["scenario_probability"]
        ).sum()
    )
    by_hour = (
        dispatch.groupby(
            ["scenario_id", "scenario_probability", "hour_of_year"],
            as_index=False,
        )["unserved_mw"]
        .sum()
    )
    expected_lole = float(
        by_hour.loc[
            by_hour["unserved_mw"].gt(REPORT_TOL),
            "scenario_probability",
        ].sum()
    )
    return {"expected_eens_mwh": expected_eens, "expected_lole_h": expected_lole}


def elcc_bisection(
    case: StochasticTransportCase,
    storage: StorageTechnology,
    settings: SolveSettings,
    baseline_solution: ModelSolution,
    baseline_firm_addition_mw: np.ndarray,
    upper_bound_mw: float,
    tolerance_mw: float,
    maximum_iterations: int = 30,
) -> dict[str, Any]:
    """Find a conservative ELCC satisfying both EENS and LOLE criteria."""

    if upper_bound_mw <= 0 or tolerance_mw <= 0:
        raise ValueError("ELCC upper bound and tolerance must be positive")
    reference_eens = float(baseline_solution.summary["expected_eens_mwh"])
    reference_lole = float(baseline_solution.summary["expected_lole_h"])

    def eligible(increment: float) -> tuple[bool, ModelSolution]:
        augmented = case.with_load_increment(increment)
        candidate = solve_stochastic_transport_lp(
            augmented,
            storage,
            settings,
            "turnover",
            baseline_firm_addition_mw=baseline_firm_addition_mw,
            keep_dispatch=False,
        )
        eens_ok = candidate.summary["expected_eens_mwh"] <= (
            reference_eens + max(1e-8, 1e-8 * max(reference_eens, 1.0))
        )
        lole_ok = candidate.summary["expected_lole_h"] <= (
            reference_lole + 1e-8
        )
        return bool(eens_ok and lole_ok), candidate

    low = 0.0
    high = float(upper_bound_mw)
    high_ok, high_solution = eligible(high)
    if high_ok:
        return {
            "elcc_lower_mw": high,
            "elcc_upper_mw": math.inf,
            "right_censored": True,
            "iterations": 1,
            "binding_metric": "upper_bound_not_binding",
            "endpoint_eens_mwh": high_solution.summary["expected_eens_mwh"],
            "endpoint_lole_h": high_solution.summary["expected_lole_h"],
        }
    last_solution = high_solution
    iterations = 0
    while high - low > tolerance_mw and iterations < maximum_iterations:
        middle = 0.5 * (low + high)
        ok, solution = eligible(middle)
        last_solution = solution
        if ok:
            low = middle
        else:
            high = middle
        iterations += 1
    eens_ratio = (
        last_solution.summary["expected_eens_mwh"] / reference_eens
        if reference_eens > REPORT_TOL
        else math.inf
    )
    lole_ratio = (
        last_solution.summary["expected_lole_h"] / reference_lole
        if reference_lole > REPORT_TOL
        else math.inf
    )
    binding = "EENS" if eens_ratio >= lole_ratio else "LOLE"
    return {
        "elcc_lower_mw": low,
        "elcc_upper_mw": high,
        "right_censored": False,
        "iterations": iterations,
        "binding_metric": binding,
        "endpoint_eens_mwh": last_solution.summary["expected_eens_mwh"],
        "endpoint_lole_h": last_solution.summary["expected_lole_h"],
    }


def _season_from_day(day: np.ndarray) -> np.ndarray:
    """Map a non-leap 365-day index to meteorological seasons."""

    timestamp = pd.Timestamp("2001-01-01") + pd.to_timedelta(day, unit="D")
    month = pd.DatetimeIndex(timestamp).month.to_numpy()
    return np.select(
        [
            np.isin(month, [12, 1, 2]),
            np.isin(month, [3, 4, 5]),
            np.isin(month, [6, 7, 8]),
        ],
        [0, 1, 2],
        default=3,
    ).astype(int)


def daily_scarcity_scores(hourly: pd.DataFrame) -> pd.DataFrame:
    """Calculate synchronized Europe-wide daily residual-load peak scores."""

    required = {
        "zone",
        "hour_of_year",
        "load_mw",
        "wind_potential_mw",
        "solar_potential_mw",
    }
    missing = required.difference(hourly.columns)
    if missing:
        raise ValueError(f"hourly panel missing {sorted(missing)}")
    frame = hourly.copy()
    frame["day"] = frame["hour_of_year"].astype(int) // 24
    frame["within_day_hour"] = frame["hour_of_year"].astype(int) % 24
    residual = (
        frame["load_mw"]
        - frame["wind_potential_mw"]
        - frame["solar_potential_mw"]
    )
    frame["residual_mw"] = residual
    system_hourly = (
        frame.groupby(["day", "within_day_hour"], as_index=False)
        .agg(residual_mw=("residual_mw", "sum"), load_mw=("load_mw", "sum"))
    )
    normalizer = float(system_hourly["load_mw"].mean())
    if normalizer <= 0:
        raise ValueError("non-positive system mean load")
    daily = (
        system_hourly.groupby("day", as_index=False)
        .agg(
            scarcity_score=("residual_mw", lambda values: float(values.max())),
            daily_residual_energy_mwh=("residual_mw", "sum"),
        )
    )
    daily["scarcity_score"] /= normalizer
    daily["season"] = _season_from_day(daily["day"].to_numpy())
    return daily


def _interleave_singletons(
    extreme_days: list[int], normal_days: list[int]
) -> list[int]:
    slots: list[list[int]] = [[] for _ in range(len(normal_days) + 1)]
    if extreme_days:
        positions = np.linspace(0, len(normal_days), len(extreme_days))
        for day, position in zip(extreme_days, np.rint(positions).astype(int)):
            slots[int(np.clip(position, 0, len(normal_days)))].append(day)
    order: list[int] = []
    for index, normal in enumerate(normal_days):
        order.extend(slots[index])
        order.append(normal)
    order.extend(slots[-1])
    return order


def _interleave_clusters(
    extreme_days: list[int],
    normal_days: list[int],
    cluster_days: int,
) -> list[int]:
    clusters = [
        extreme_days[index : index + cluster_days]
        for index in range(0, len(extreme_days), cluster_days)
    ]
    if not clusters:
        return list(normal_days)
    slots: list[list[list[int]]] = [[] for _ in range(len(normal_days) + 1)]
    positions = np.linspace(0, len(normal_days), len(clusters))
    for cluster, position in zip(clusters, np.rint(positions).astype(int)):
        slots[int(np.clip(position, 0, len(normal_days)))].append(cluster)
    order: list[int] = []
    for index, normal in enumerate(normal_days):
        for cluster in slots[index]:
            order.extend(cluster)
        order.append(normal)
    for cluster in slots[-1]:
        order.extend(cluster)
    return order


def make_day_order(
    daily: pd.DataFrame,
    chronology: str,
    quantile: float = 0.75,
    cluster_days: int = 4,
    seed: int = 2026072601,
    randomized: bool = False,
) -> list[int]:
    """Return one season-constrained day order for a paired chronology."""

    if chronology not in {"natural", "dispersed", "clustered"}:
        raise ValueError(f"unknown chronology {chronology}")
    if not 0 < quantile < 1:
        raise ValueError("quantile must be between zero and one")
    if cluster_days <= 0:
        raise ValueError("cluster_days must be positive")
    daily = daily.sort_values("day").copy()
    expected = list(range(int(daily["day"].max()) + 1))
    if daily["day"].astype(int).tolist() != expected:
        raise ValueError("days must be consecutive and start at zero")
    if chronology == "natural":
        return expected
    threshold = float(daily["scarcity_score"].quantile(quantile))
    daily["extreme"] = daily["scarcity_score"].ge(threshold)
    rng = np.random.default_rng(seed)
    order: list[int] = []
    for season in sorted(daily["season"].unique()):
        group = daily.loc[daily["season"].eq(season)]
        extreme = group.loc[group["extreme"], "day"].astype(int).tolist()
        normal = group.loc[~group["extreme"], "day"].astype(int).tolist()
        if randomized:
            rng.shuffle(extreme)
            rng.shuffle(normal)
        else:
            extreme = (
                group.loc[group["extreme"]]
                .sort_values(["scarcity_score", "day"], ascending=[False, True])[
                    "day"
                ]
                .astype(int)
                .tolist()
            )
            normal = sorted(normal)
        if chronology == "dispersed":
            season_order = _interleave_singletons(extreme, normal)
        else:
            season_order = _interleave_clusters(extreme, normal, cluster_days)
        order.extend(season_order)
    if sorted(order) != expected or len(order) != len(expected):
        raise RuntimeError("day-order construction violated permutation identity")
    return order


def apply_synchronized_day_order(
    hourly: pd.DataFrame,
    order: Sequence[int],
    chronology: str,
) -> pd.DataFrame:
    """Move complete 24-hour, all-zone vectors using one shared day order."""

    required = {"zone", "hour_of_year"}
    missing = required.difference(hourly.columns)
    if missing:
        raise ValueError(f"hourly panel missing {sorted(missing)}")
    frame = hourly.copy()
    frame["source_hour_of_year"] = frame["hour_of_year"].astype(int)
    frame["source_day"] = frame["source_hour_of_year"] // 24
    frame["within_day_hour"] = frame["source_hour_of_year"] % 24
    expected_days = sorted(frame["source_day"].unique())
    if list(sorted(int(value) for value in order)) != expected_days:
        raise ValueError("day order does not contain exactly the source days")
    pieces: list[pd.DataFrame] = []
    for new_day, source_day in enumerate(order):
        block = frame.loc[frame["source_day"].eq(int(source_day))].copy()
        block["day"] = new_day
        block["hour_of_year"] = new_day * 24 + block["within_day_hour"]
        block["chronology"] = chronology
        pieces.append(block)
    result = pd.concat(pieces, ignore_index=True)
    result = result.sort_values(["zone", "hour_of_year"]).reset_index(drop=True)
    return result


def _digest_frame(frame: pd.DataFrame, columns: Iterable[str]) -> str:
    selected = frame.loc[:, list(columns)].copy()
    payload = selected.to_csv(index=False, float_format="%.12g").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def chronology_conservation_audit(
    original: pd.DataFrame,
    transformed: pd.DataFrame,
) -> dict[str, Any]:
    """Audit energy sums, day-block identity and all-zone synchronization."""

    columns = [
        column
        for column in HOURLY_CONSERVATION_COLUMNS
        if column in original.columns and column in transformed.columns
    ]
    if not columns:
        raise ValueError("no conservation columns are shared")
    if len(original) != len(transformed):
        raise ValueError("row count changed during chronology transformation")
    original_sums = original[columns].sum()
    transformed_sums = transformed[columns].sum()
    denominator = np.maximum(np.abs(original_sums.to_numpy()), 1.0)
    relative_errors = np.abs(
        transformed_sums.to_numpy() - original_sums.to_numpy()
    ) / denominator

    source_key = ["zone", "source_hour_of_year"]
    original_for_hash = original.copy()
    original_for_hash["source_hour_of_year"] = original_for_hash[
        "hour_of_year"
    ].astype(int)
    original_sorted = original_for_hash.sort_values(source_key)
    transformed_sorted = transformed.sort_values(source_key)
    hash_columns = [*source_key, *columns]
    original_hash = _digest_frame(original_sorted, hash_columns)
    transformed_hash = _digest_frame(transformed_sorted, hash_columns)

    source_day_by_new_hour = (
        transformed.groupby("hour_of_year")["source_day"].nunique()
    )
    spatial_sync = bool(source_day_by_new_hour.eq(1).all())
    source_use = transformed[
        ["zone", "source_hour_of_year"]
    ].value_counts()
    each_source_once = bool(source_use.eq(1).all())

    audit = {
        "row_count_original": int(len(original)),
        "row_count_transformed": int(len(transformed)),
        "conservation_columns": columns,
        "maximum_relative_annual_sum_error": float(relative_errors.max()),
        "annual_sum_relative_errors": {
            column: float(error)
            for column, error in zip(columns, relative_errors)
        },
        "source_block_hash_original": original_hash,
        "source_block_hash_transformed": transformed_hash,
        "source_block_identity": original_hash == transformed_hash,
        "spatial_day_mapping_synchronized": spatial_sync,
        "each_zone_source_hour_used_once": each_source_once,
    }
    audit["pass"] = bool(
        audit["maximum_relative_annual_sum_error"] <= CONSERVATION_RTOL
        and audit["source_block_identity"]
        and spatial_sync
        and each_source_once
    )
    return audit


def event_statistics(
    hourly: pd.DataFrame,
    quantile: float = 0.75,
) -> dict[str, float]:
    daily = daily_scarcity_scores(hourly)
    threshold = float(daily["scarcity_score"].quantile(quantile))
    extreme = daily["scarcity_score"].ge(threshold).to_numpy()
    runs: list[int] = []
    gap_runs: list[int] = []
    current = 0
    gap = 0
    seen_extreme = False
    for value in extreme:
        if value:
            if current == 0 and seen_extreme and gap:
                gap_runs.append(gap)
            current += 1
            gap = 0
            seen_extreme = True
        else:
            if current:
                runs.append(current)
                current = 0
            if seen_extreme:
                gap += 1
    if current:
        runs.append(current)
    return {
        "extreme_day_share": float(extreme.mean()),
        "number_of_extreme_events": float(len(runs)),
        "mean_event_duration_d": float(np.mean(runs)) if runs else 0.0,
        "maximum_event_duration_d": float(max(runs)) if runs else 0.0,
        "mean_interevent_gap_d": (
            float(np.mean(gap_runs)) if gap_runs else math.nan
        ),
    }


def validate_standardized_hourly_contract(hourly: pd.DataFrame) -> dict[str, Any]:
    """Validate a standardized ERAA hourly table before any expensive solve."""

    required = {
        "target_year",
        "climate_model",
        "weather_scenario",
        "hour_of_year",
        "zone",
        "load_mw",
        "wind_potential_mw",
        "solar_potential_mw",
    }
    missing = required.difference(hourly.columns)
    if missing:
        raise ValueError(f"standardized hourly table missing {sorted(missing)}")
    forbidden = EXCLUDED_AGGREGATE_ZONE_CODES.intersection(
        set(hourly["zone"].astype(str).unique())
    )
    if forbidden:
        raise ValueError(f"aggregate/alias zones present: {sorted(forbidden)}")
    key = [
        "target_year",
        "weather_scenario",
        "hour_of_year",
        "zone",
    ]
    if hourly.duplicated(key).any():
        raise ValueError("duplicate standardized hourly primary keys")
    for column in (
        column
        for column in HOURLY_CONSERVATION_COLUMNS
        if column in hourly.columns
    ):
        values = pd.to_numeric(hourly[column], errors="raise")
        if not np.isfinite(values).all() or values.lt(0).any():
            raise ValueError(f"{column} contains negative or non-finite values")

    rows: list[dict[str, Any]] = []
    for (target_year, weather), group in hourly.groupby(
        ["target_year", "weather_scenario"], sort=True
    ):
        zones = sorted(group["zone"].astype(str).unique())
        counts = group.groupby("zone")["hour_of_year"].nunique()
        hour_sets = group.groupby("zone")["hour_of_year"].apply(
            lambda values: tuple(sorted(pd.to_numeric(values).astype(int)))
        )
        synchronized = len(set(hour_sets)) == 1
        rows.append(
            {
                "target_year": int(target_year),
                "weather_scenario": str(weather),
                "zone_count": len(zones),
                "minimum_hours_per_zone": int(counts.min()),
                "maximum_hours_per_zone": int(counts.max()),
                "synchronized_hour_axis": synchronized,
            }
        )
    coverage = pd.DataFrame(rows)
    scenario_counts = (
        coverage.groupby("target_year")["weather_scenario"].nunique().to_dict()
    )
    return {
        "rows": int(len(hourly)),
        "zones": int(hourly["zone"].nunique()),
        "target_years": sorted(
            int(value) for value in hourly["target_year"].unique()
        ),
        "weather_scenarios_per_target_year": {
            str(int(year)): int(count)
            for year, count in scenario_counts.items()
        },
        "minimum_hours_per_zone_scenario": int(
            coverage["minimum_hours_per_zone"].min()
        ),
        "maximum_hours_per_zone_scenario": int(
            coverage["maximum_hours_per_zone"].max()
        ),
        "all_hour_axes_synchronized": bool(
            coverage["synchronized_hour_axis"].all()
        ),
        "coverage": coverage,
    }


def evaluate_valley_for_chronology(
    envelope: pd.DataFrame,
) -> dict[str, Any]:
    """Apply the frozen hT/hI/W rule to one chronology envelope.

    Required columns are interval bounds after envelope reselection:
    ``duration_h, bt_lower, bj_upper, bj_lower, ba_lower, elcc_lower,
    main_evidence_domain``.
    """

    required = {
        "duration_h",
        "bt_lower",
        "bj_upper",
        "bj_lower",
        "ba_lower",
        "elcc_lower",
        "main_evidence_domain",
    }
    missing = required.difference(envelope.columns)
    if missing:
        raise ValueError(f"envelope missing {sorted(missing)}")
    frame = envelope.sort_values("duration_h").drop_duplicates("duration_h")
    frame = frame.reset_index(drop=True)
    duration = frame["duration_h"].to_numpy(dtype=float)
    bt_positive = frame["bt_lower"].to_numpy(dtype=float) > 0
    joint_negative = frame["bj_upper"].to_numpy(dtype=float) < 0
    stable_long = (
        (frame["bj_lower"].to_numpy(dtype=float) > 0)
        & (frame["ba_lower"].to_numpy(dtype=float) > 0)
        & (frame["elcc_lower"].to_numpy(dtype=float) > 0)
    )
    main = frame["main_evidence_domain"].astype(bool).to_numpy()

    # hT belongs to the connected turnover-supported branch that begins at
    # the shortest tested duration.  A later isolated positive island cannot
    # be relabelled as the left wall after inspecting the curve.
    connected_positive_end: int | None = None
    for index, supported in enumerate(bt_positive):
        if supported:
            connected_positive_end = index
        else:
            break
    h_t_index: int | None = (
        connected_positive_end
        if connected_positive_end is not None and connected_positive_end >= 1
        else None
    )
    negative_pair_start: int | None = None
    if h_t_index is not None:
        for index in range(h_t_index + 1, len(frame) - 1):
            if joint_negative[index] and joint_negative[index + 1]:
                negative_pair_start = index
                break
    if h_t_index is None or negative_pair_start is None:
        return {
            "confirmed": False,
            "h_T_h": math.nan,
            "h_I_h": math.nan,
            "W_log_ratio": math.nan,
            "h_T_status": "not_identified_or_right_censored",
            "h_I_status": "not_tested_without_hT",
            "failure_reasons": [
                "no connected positive turnover branch followed by two "
                "adjacent joint-negative nodes"
            ],
        }

    h_i_index: int | None = None
    structural_reentry = False
    for index in range(negative_pair_start + 2, len(frame) - 1):
        if stable_long[index] and stable_long[index + 1]:
            if main[index] and main[index + 1]:
                h_i_index = index
                break
            structural_reentry = True
    if h_i_index is None:
        reason = (
            "long-duration re-entry occurs only in structural >168 h nodes"
            if structural_reentry
            else "no stable long-duration joint/capacity/ELCC re-entry"
        )
        return {
            "confirmed": False,
            "h_T_h": float(duration[h_t_index]),
            "h_I_h": math.nan,
            "W_log_ratio": math.nan,
            "h_T_status": "identified",
            "h_I_status": (
                "right_censored_structural_only"
                if structural_reentry
                else "not_identified"
            ),
            "negative_pair_start_h": float(duration[negative_pair_start]),
            "failure_reasons": [reason],
        }

    h_t = float(duration[h_t_index])
    h_i = float(duration[h_i_index])
    return {
        "confirmed": True,
        "h_T_h": h_t,
        "h_I_h": h_i,
        "W_log_ratio": float(math.log(h_i / h_t)),
        "h_T_status": "identified",
        "h_I_status": "identified",
        "negative_pair_start_h": float(duration[negative_pair_start]),
        "failure_reasons": [],
    }


def evaluate_delta_w(
    dispersed: dict[str, Any],
    clustered: dict[str, Any],
    delta_w_ci_lower: float | None = None,
) -> dict[str, Any]:
    """Pair two boundary results without filling censored values."""

    if not dispersed.get("confirmed") or not clustered.get("confirmed"):
        return {
            "delta_W": math.nan,
            "confirmed_widening": False,
            "status": "not_estimable_due_to_unidentified_or_censored_boundary",
        }
    delta = float(clustered["W_log_ratio"] - dispersed["W_log_ratio"])
    interval_supported = (
        delta_w_ci_lower is not None and float(delta_w_ci_lower) > 0
    )
    return {
        "delta_W": delta,
        "delta_W_ci_lower": delta_w_ci_lower,
        "confirmed_widening": bool(delta > 0 and interval_supported),
        "status": (
            "confirmed"
            if delta > 0 and interval_supported
            else "point_widening_without_positive_interval"
            if delta > 0
            else "not_wider"
        ),
    }


def synthetic_hourly_panel(
    days: int = 28,
    zones: tuple[str, ...] = ("Z1", "Z2"),
    seed: int = 2026072601,
) -> pd.DataFrame:
    """Create a synchronized all-zone panel for chronology smoke tests."""

    if days < 12:
        raise ValueError("synthetic panel needs at least 12 days")
    rng = np.random.default_rng(seed)
    hours = np.arange(days * 24)
    day = hours // 24
    hod = hours % 24
    scarcity_days = {3, 4, 10, 11, 17, 18, 24}
    rows: list[pd.DataFrame] = []
    for zone_index, zone in enumerate(zones):
        daily = 0.95 + 0.08 * np.sin(2 * np.pi * hod / 24 - 0.5)
        event = np.array([0.34 if int(value) in scarcity_days else 0.0 for value in day])
        load = (80 + zone_index * 25) * (daily + event)
        solar = (
            (26 + zone_index * 8)
            * np.maximum(np.sin(np.pi * (hod - 6) / 12), 0)
            * (0.9 + 0.08 * rng.random(len(hours)))
        )
        wind = (
            (30 + zone_index * 5)
            * (0.65 + 0.22 * np.sin(2 * np.pi * day / 7 + zone_index))
        )
        wind *= np.where(np.isin(day, list(scarcity_days)), 0.18, 1.0)
        frame = pd.DataFrame(
            {
                "zone": zone,
                "hour_of_year": hours,
                "load_mw": load,
                "wind_potential_mw": np.maximum(wind, 0),
                "solar_potential_mw": solar,
                "ror_potential_mw": 3.0 + zone_index,
                "hydro_inflow_mwh": 1.6 + 0.2 * zone_index,
                "must_run_mw": 0.0,
            }
        )
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def synthetic_transport_case(
    chronology_panel: pd.DataFrame,
    case_id: str,
) -> StochasticTransportCase:
    """Convert a two-zone synthetic chronology into a two-scenario LP case."""

    zones = tuple(sorted(chronology_panel["zone"].unique()))
    hours = sorted(chronology_panel["hour_of_year"].unique())
    z_count = len(zones)
    t_count = len(hours)
    zone_lookup = {zone: index for index, zone in enumerate(zones)}
    arrays = {
        name: np.zeros((2, z_count, t_count), dtype=float)
        for name in (
            "load_mw",
            "wind_potential_mw",
            "solar_potential_mw",
            "must_run_mw",
            "hydro_inflow_mwh",
        )
    }
    for zone, group in chronology_panel.groupby("zone"):
        z = zone_lookup[zone]
        group = group.sort_values("hour_of_year")
        for name in arrays:
            arrays[name][0, z] = group[name].to_numpy(dtype=float)
            arrays[name][1, z] = group[name].to_numpy(dtype=float) * (
                1.0 if name != "load_mw" else 1.015
            )
    thermal_technologies = ("CCGT", "OCGT")
    existing = np.array([[80.0, 8.0], [95.0, 10.0]])
    availability = np.ones((2, z_count, 2, t_count), dtype=float)
    availability[1, :, 0, 80:104] = 0.72
    availability[1, :, 1, 80:104] = 0.55
    edges = (
        NetworkEdge("Z1_to_Z2", zones[0], zones[1], 0.02),
        NetworkEdge("Z2_to_Z1", zones[1], zones[0], 0.02),
    )
    network = np.full((2, 2, t_count), 18.0, dtype=float)
    return StochasticTransportCase(
        case_id=case_id,
        zones=zones,
        thermal_technologies=thermal_technologies,
        scenario_ids=("weather01_outage01", "weather01_outage02"),
        scenario_probabilities=np.array([0.5, 0.5]),
        load_mw=arrays["load_mw"],
        wind_potential_mw=arrays["wind_potential_mw"],
        solar_potential_mw=arrays["solar_potential_mw"],
        must_run_mw=arrays["must_run_mw"],
        thermal_existing_capacity_mw=existing,
        thermal_available_fraction=availability,
        thermal_variable_cost_eur_mwh=np.array(
            [[74.0, 126.0], [76.0, 130.0]]
        ),
        firm_addition_technology_index=1,
        firm_addition_annual_cost_eur_kw=np.array([82.0, 86.0]),
        hydro_power_capacity_mw=np.array([12.0, 15.0]),
        hydro_available_fraction=np.ones((2, z_count, t_count)),
        hydro_energy_capacity_mwh=np.array([420.0, 520.0]),
        hydro_inflow_mwh=arrays["hydro_inflow_mwh"],
        hydro_variable_cost_eur_mwh=np.array([5.0, 5.0]),
        edges=edges,
        network_capacity_mw=network,
        deployment_mask=np.ones(z_count, dtype=bool),
    )


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
