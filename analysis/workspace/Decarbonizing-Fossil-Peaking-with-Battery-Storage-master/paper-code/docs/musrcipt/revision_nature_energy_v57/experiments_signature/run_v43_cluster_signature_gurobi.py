#!/usr/bin/env python3
"""Cluster-length mechanism-signature experiment for manuscript v43.

The model separates the storage portfolio into two operational layers:

* a common, fixed-power 4 h LFP layer that may serve the complete C1
  above-floor gas-peaking task; and
* a tail layer, either additional 4 h LFP or one candidate LDES
  configuration, that may discharge only into the preregistered C3-q75
  event-tail task.  Its power is either fixed at the common comparison scale
  or selected continuously between zero and that common upper bound.

The two assets re-dispatch jointly, share the same R2 recharge resource and
retain separate cyclic state-of-charge equations.  Each country-year is solved
for the frozen dispersed and clustered intact-day chronologies.  The objective
minimizes annual storage plus residual peaking-system cost, followed
lexicographically by total charging energy.

This is a frozen copy of the v42 portfolio runner with one intentional
extension: the target length of the controlled event-day clusters is an
explicit command-line parameter.  All dispatch, cost, state-of-charge and
comparison definitions are otherwise unchanged.  Keeping the extension in a
separate runner protects the manuscript's primary run from accidental drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB


HERE = Path(__file__).resolve().parent
MUSRIPT_ROOT = HERE.parents[1]
WORKSPACE_ROOT = HERE.parents[5]
DATASET_ROOT = (
    WORKSPACE_ROOT
    / "dataset"
    / "external_hourly"
    / "europe_mid_duration_valley_upgrade_v0_1"
)
V40_ROOT = MUSRIPT_ROOT / "revision_nature_energy_v40"

DEFAULT_PANEL = (
    DATASET_ROOT
    / "processed"
    / "observed_gas_event_task_definitions_v0_2.csv.gz"
)
DEFAULT_CATALOGUE = (
    V40_ROOT
    / "currency_work"
    / "configuration_catalog_53_source_aware_eur.csv"
)
DEFAULT_SCENARIO = (
    V40_ROOT
    / "currency_work"
    / "value_sensitivity_gap_summary_source_aware_eur.csv"
)
DEFAULT_OUTPUT = HERE / "smoke_de_2023"

BASE_CONFIG_ID = "li_lfp_4h"
BASE_TASK_COLUMN = "task_c1_mw"
TAIL_TASK_COLUMN = "task_c3_q75_mw"
RECHARGE_COLUMN = "r2_potential_recharge_mw"
DEFAULT_SCENARIO_ID = 31
SOC_MIN = 0.05
SOC_MAX = 0.95
USABLE_SOC_FRACTION = SOC_MAX - SOC_MIN
FORCED_POWER_SHARE = 0.01
DEFAULT_CLUSTER_DAYS = 4
NUMERICAL_ZERO = 1.0e-10
SOC_TOLERANCE_MWH = 1.0e-6
BALANCE_TOLERANCE_MW = 1.0e-6
PRIMARY_OBJECTIVE_ABS_TOLERANCE_EUR = 0.01
EXPECTED_CHRONOLOGIES = ("dispersed", "clustered")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            json_ready(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def output_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(
        path,
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )


def parse_csv_list(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    if len(items) != len(set(items)):
        raise ValueError("Duplicate comma-separated entries are not allowed")
    return items


def parse_country_years(value: str | None) -> list[tuple[str, int]] | None:
    items = parse_csv_list(value)
    if items is None:
        return None
    result: list[tuple[str, int]] = []
    for item in items:
        country, year = item.split(":", maxsplit=1)
        result.append((country.strip().upper(), int(year)))
    return result


def interleave(extreme: list[int], normal: list[int]) -> list[int]:
    slots: list[list[int]] = [[] for _ in range(len(normal) + 1)]
    if extreme:
        positions = np.linspace(1, max(1, len(normal) - 1), len(extreme))
        for day, position in zip(extreme, np.rint(positions).astype(int)):
            slots[int(np.clip(position, 0, len(normal)))].append(day)
    order: list[int] = []
    for index, day in enumerate(normal):
        order.extend(slots[index])
        order.append(day)
    order.extend(slots[-1])
    return order


def interleave_clusters(
    extreme: list[int], normal: list[int], target_run: int
) -> list[int]:
    clusters = [
        extreme[index : index + target_run]
        for index in range(0, len(extreme), target_run)
    ]
    slots: list[list[int]] = [[] for _ in range(len(normal) + 1)]
    if clusters:
        positions = np.linspace(1, max(1, len(normal) - 1), len(clusters))
        for cluster, position in zip(clusters, np.rint(positions).astype(int)):
            slots[int(np.clip(position, 0, len(normal)))].extend(cluster)
    order: list[int] = []
    for index, day in enumerate(normal):
        order.extend(slots[index])
        order.append(day)
    order.extend(slots[-1])
    return order


def day_orders(
    frame: pd.DataFrame, cluster_days: int = DEFAULT_CLUSTER_DAYS
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    """Reproduce the intact-day construction at a chosen cluster length."""
    if cluster_days <= 0:
        raise ValueError("cluster_days must be a positive integer")
    daily = (
        frame.groupby("source_day_id", sort=True)
        .agg(
            segment=("season_segment", "first"),
            event=("event_flag_q75", "max"),
            tail_task_mwh=(TAIL_TASK_COLUMN, "sum"),
        )
        .reset_index()
    )
    if len(daily) != 365:
        raise RuntimeError("Each country-year must contain 365 complete days")
    dispersed: list[int] = []
    clustered: list[int] = []
    for segment in sorted(daily["segment"].unique()):
        subset = daily[daily["segment"].eq(segment)]
        extreme = [
            int(value)
            for value in subset.loc[subset["event"].astype(bool), "source_day_id"]
        ]
        normal = [
            int(value)
            for value in subset.loc[~subset["event"].astype(bool), "source_day_id"]
        ]
        dispersed.extend(interleave(extreme, normal))
        clustered.extend(interleave_clusters(extreme, normal, cluster_days))
    orders = {"dispersed": dispersed, "clustered": clustered}
    for chronology, order in orders.items():
        if len(order) != 365 or sorted(order) != list(range(365)):
            raise RuntimeError(f"Invalid {chronology} intact-day order")

    event_days = set(
        int(value)
        for value in daily.loc[daily["event"].astype(bool), "source_day_id"]
    )
    segment_by_day = daily.set_index("source_day_id")["segment"].to_dict()

    def adjacency(order: list[int]) -> tuple[int, int]:
        adjacent = 0
        longest = 0
        run = 0
        previous_segment: int | None = None
        previous_event = False
        for day in order:
            segment = int(segment_by_day[day])
            flag = day in event_days
            if segment != previous_segment:
                run = 0
                previous_event = False
            if flag:
                run += 1
                longest = max(longest, run)
                if previous_event:
                    adjacent += 1
            else:
                run = 0
            previous_segment = segment
            previous_event = flag
        return adjacent, longest

    dispersed_adjacent, dispersed_longest = adjacency(dispersed)
    clustered_adjacent, clustered_longest = adjacency(clustered)
    return orders, {
        "target_cluster_days": int(cluster_days),
        "event_days": int(daily["event"].sum()),
        "dispersed_adjacent_event_pairs": dispersed_adjacent,
        "clustered_adjacent_event_pairs": clustered_adjacent,
        "dispersed_longest_event_run_days": dispersed_longest,
        "clustered_longest_event_run_days": clustered_longest,
    }


def reorder_days(values: np.ndarray, order: list[int]) -> np.ndarray:
    return values.reshape(365, 24)[np.asarray(order, dtype=int)].reshape(-1)


def equivalent_firm_capacity(
    task_mw: np.ndarray, target_unserved_mwh: float
) -> float:
    """Firm flat capacity with the same EENS as the storage portfolio."""
    total = float(task_mw.sum())
    if target_unserved_mwh >= total - 1.0e-9:
        return 0.0
    low = 0.0
    high = float(task_mw.max())
    for _ in range(70):
        midpoint = 0.5 * (low + high)
        unserved = float(np.maximum(task_mw - midpoint, 0.0).sum())
        if unserved > target_unserved_mwh:
            low = midpoint
        else:
            high = midpoint
    return 0.5 * (low + high)


def select_panel(
    panel: pd.DataFrame,
    requested: list[tuple[str, int]] | None,
    max_country_years: int | None,
) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    available = [
        (str(row.country), int(row.year))
        for row in (
            panel[["country", "year"]]
            .drop_duplicates()
            .sort_values(["country", "year"])
            .itertuples(index=False)
        )
    ]
    chosen = available if requested is None else requested
    missing = sorted(set(chosen) - set(available))
    if missing:
        raise RuntimeError(f"Requested country-years are absent: {missing}")
    if max_country_years is not None:
        if max_country_years <= 0:
            raise ValueError("--max-country-years must be positive")
        chosen = chosen[:max_country_years]
    chosen_set = set(chosen)
    selected = panel[
        [
            (str(country), int(year)) in chosen_set
            for country, year in zip(panel["country"], panel["year"])
        ]
    ].copy()
    return selected, chosen


def load_scenario(path: Path, scenario_id: int) -> dict[str, float]:
    frame = pd.read_csv(path)
    subset = frame[frame["scenario_id"].eq(scenario_id)]
    if set(subset["chronology"]) != set(EXPECTED_CHRONOLOGIES):
        raise RuntimeError(
            f"Scenario {scenario_id} must contain both chronologies"
        )
    columns = [
        "energy_value_eur_mwh",
        "firm_value_eur_kw_year",
        "charge_price_eur_mwh",
    ]
    unique = subset[columns].drop_duplicates()
    if len(unique) != 1:
        raise RuntimeError("Scenario 31 parameters are not unique")
    return {column: float(unique.iloc[0][column]) for column in columns}


def select_catalogue(
    catalogue: pd.DataFrame, requested_config_ids: list[str] | None
) -> tuple[pd.Series, pd.DataFrame]:
    if catalogue["config_id"].duplicated().any():
        raise RuntimeError("Catalogue config_id values must be unique")
    base_rows = catalogue[catalogue["config_id"].eq(BASE_CONFIG_ID)]
    if len(base_rows) != 1:
        raise RuntimeError(f"Catalogue must contain exactly one {BASE_CONFIG_ID}")
    base = base_rows.iloc[0]
    candidates = catalogue[~catalogue["tech_id"].eq("li_lfp")].copy()
    if requested_config_ids is not None:
        missing = sorted(
            set(requested_config_ids) - set(catalogue["config_id"].astype(str))
        )
        if missing:
            raise RuntimeError(f"Unknown candidate config IDs: {missing}")
        invalid = sorted(
            set(requested_config_ids)
            - set(candidates["config_id"].astype(str))
        )
        if invalid:
            raise RuntimeError(
                "Candidate subset cannot include Li-ion LFP; the 4 h LFP "
                f"comparator is added automatically: {invalid}"
            )
        candidates = candidates[
            candidates["config_id"].isin(requested_config_ids)
        ].copy()
    candidates = candidates.sort_values(
        ["duration_h", "tech_id"], kind="stable"
    ).reset_index(drop=True)
    return base, candidates


def model_parameters(config: pd.Series) -> dict[str, float]:
    return {
        "duration_h": float(config["duration_h"]),
        "rte": float(config["rte"]),
        "self_discharge_pct_day": float(
            config["self_discharge_pct_day"]
        ),
        "charge_power_ratio": float(config["charge_power_ratio"]),
        "fixed_cost_eur_kw_year": float(
            config[
                "annual_fixed_cost_eur_per_kw_year_source_aware"
            ]
        ),
        "vom_eur_mwh": float(
            config["variable_om_eur_mwh_source_aware"]
        ),
    }


def config_identity(config: pd.Series) -> dict[str, Any]:
    return {
        "config_id": str(config["config_id"]),
        "tech_id": str(config["tech_id"]),
        "technology_en": str(config["technology_en"]),
        "duration_h": float(config["duration_h"]),
        "rte": float(config["rte"]),
        "self_discharge_pct_day": float(
            config["self_discharge_pct_day"]
        ),
        "charge_power_ratio": float(config["charge_power_ratio"]),
        "fixed_cost_eur_kw_year": float(
            config[
                "annual_fixed_cost_eur_per_kw_year_source_aware"
            ]
        ),
        "vom_eur_mwh": float(
            config["variable_om_eur_mwh_source_aware"]
        ),
    }


def soc_diagnostics(
    state: np.ndarray,
    charge: np.ndarray,
    discharge: np.ndarray,
    retention: float,
    eta: float,
) -> tuple[float, float, float]:
    previous = np.roll(state, 1)
    residual = (
        state - retention * previous - eta * charge + discharge / eta
    )
    return (
        float(residual[0]),
        float(np.max(np.abs(residual))),
        float(np.min(state)),
    )


def solve_portfolio_arm(
    base_task_mw: np.ndarray,
    tail_task_mw: np.ndarray,
    recharge_mw: np.ndarray,
    base_power_mw: float,
    tail_power_upper_mw: float,
    tail_power_mode: str,
    base_config: pd.Series,
    tail_config: pd.Series,
    scenario: dict[str, float],
    env: gp.Env,
    threads: int,
    method: int,
    write_hourly: bool,
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    """Solve one annual joint-capacity/dispatch LP."""
    hours = len(base_task_mw)
    if not (
        len(tail_task_mw) == hours
        and len(recharge_mw) == hours
        and hours == 8760
    ):
        raise RuntimeError("Portfolio arrays must contain exactly 8,760 hours")
    if np.any(tail_task_mw - base_task_mw > BALANCE_TOLERANCE_MW):
        raise RuntimeError("C3 tail task must be a subset of the C1 task")

    base = model_parameters(base_config)
    tail = model_parameters(tail_config)
    base_eta = float(np.sqrt(base["rte"]))
    tail_eta = float(np.sqrt(tail["rte"]))
    base_retention = float(
        (1.0 - base["self_discharge_pct_day"] / 100.0)
        ** (1.0 / 24.0)
    )
    tail_retention = float(
        (1.0 - tail["self_discharge_pct_day"] / 100.0)
        ** (1.0 / 24.0)
    )
    base_usable_capacity_mwh = (
        USABLE_SOC_FRACTION * base["duration_h"] * base_power_mw
    )
    # A recharge hour is available only outside the C1 peaking task.  This
    # prevents simultaneous charging and peaking-service discharge.
    usable_recharge_mw = np.where(
        base_task_mw > NUMERICAL_ZERO, 0.0, recharge_mw
    )

    model = gp.Model(env=env)
    model.Params.OutputFlag = 0
    model.Params.Threads = threads
    model.Params.Method = method
    model.Params.MultiObjMethod = method
    model.Params.Presolve = 2
    model.Params.FeasibilityTol = 1.0e-8
    model.Params.OptimalityTol = 1.0e-8

    q_base = model.addMVar(
        hours,
        lb=0.0,
        ub=np.minimum(
            usable_recharge_mw,
            base["charge_power_ratio"] * base_power_mw,
        ),
        name="charge_base_mw",
    )
    d_base = model.addMVar(
        hours,
        lb=0.0,
        ub=np.minimum(base_task_mw, base_power_mw),
        name="discharge_base_mw",
    )
    s_base = model.addMVar(
        hours,
        lb=0.0,
        ub=base_usable_capacity_mwh,
        name="usable_soc_base_mwh",
    )
    tail_power_lower_mw = (
        tail_power_upper_mw
        if tail_power_mode == "fixed"
        and tail_power_upper_mw > NUMERICAL_ZERO
        else 0.0
    )
    tail_power = model.addVar(
        lb=tail_power_lower_mw,
        ub=tail_power_upper_mw,
        name="tail_discharge_power_mw",
    )
    q_tail = model.addMVar(hours, lb=0.0, name="charge_tail_mw")
    d_tail = model.addMVar(hours, lb=0.0, name="discharge_tail_mw")
    s_tail = model.addMVar(hours, lb=0.0, name="usable_soc_tail_mwh")
    unserved = model.addMVar(
        hours, lb=0.0, ub=base_task_mw, name="unserved_task_mw"
    )
    residual_peak = model.addVar(lb=0.0, name="residual_peak_mw")

    model.addConstr(
        q_base + q_tail <= usable_recharge_mw,
        name="shared_recharge_resource",
    )
    model.addConstr(
        q_tail <= tail["charge_power_ratio"] * tail_power,
        name="tail_charge_power",
    )
    model.addConstr(d_tail <= tail_power, name="tail_discharge_power")
    model.addConstr(d_tail <= tail_task_mw, name="tail_task_eligibility")
    model.addConstr(
        s_tail
        <= USABLE_SOC_FRACTION * tail["duration_h"] * tail_power,
        name="tail_energy_capacity",
    )
    model.addConstr(
        d_base + d_tail + unserved == base_task_mw,
        name="joint_task_balance",
    )
    model.addConstr(residual_peak >= unserved, name="residual_peak")

    model.addConstr(
        s_base[0]
        == base_retention * s_base[-1]
        + base_eta * q_base[0]
        - d_base[0] / base_eta,
        name="base_soc_cyclic",
    )
    model.addConstr(
        s_base[1:]
        == base_retention * s_base[:-1]
        + base_eta * q_base[1:]
        - d_base[1:] / base_eta,
        name="base_soc_balance",
    )
    model.addConstr(
        s_tail[0]
        == tail_retention * s_tail[-1]
        + tail_eta * q_tail[0]
        - d_tail[0] / tail_eta,
        name="tail_soc_cyclic",
    )
    model.addConstr(
        s_tail[1:]
        == tail_retention * s_tail[:-1]
        + tail_eta * q_tail[1:]
        - d_tail[1:] / tail_eta,
        name="tail_soc_balance",
    )

    fixed_base = (
        base_power_mw * 1000.0 * base["fixed_cost_eur_kw_year"]
    )
    fixed_tail = (
        tail_power * 1000.0 * tail["fixed_cost_eur_kw_year"]
    )
    vom_base = base["vom_eur_mwh"] * d_base.sum()
    vom_tail = tail["vom_eur_mwh"] * d_tail.sum()
    charge_cost = scenario["charge_price_eur_mwh"] * (
        q_base.sum() + q_tail.sum()
    )
    residual_energy_cost = (
        scenario["energy_value_eur_mwh"] * unserved.sum()
    )
    residual_firm_cost = (
        scenario["firm_value_eur_kw_year"]
        * 1000.0
        * residual_peak
    )
    system_cost = (
        fixed_base
        + fixed_tail
        + vom_base
        + vom_tail
        + charge_cost
        + residual_energy_cost
        + residual_firm_cost
    )
    model.ModelSense = GRB.MINIMIZE
    model.setObjectiveN(
        system_cost,
        index=0,
        priority=2,
        weight=1.0,
        abstol=PRIMARY_OBJECTIVE_ABS_TOLERANCE_EUR,
        reltol=0.0,
        name="minimum_annual_system_cost",
    )
    model.setObjectiveN(
        q_base.sum() + q_tail.sum(),
        index=1,
        priority=1,
        weight=1.0,
        abstol=1.0e-8,
        reltol=0.0,
        name="minimum_charge_tiebreak",
    )
    model.optimize()
    status = int(model.Status)
    if status != GRB.OPTIMAL:
        summary = {
            "solver_status": status,
            "solver_status_name": {
                GRB.INFEASIBLE: "infeasible",
                GRB.INF_OR_UNBD: "infeasible_or_unbounded",
                GRB.UNBOUNDED: "unbounded",
                GRB.NUMERIC: "numeric",
            }.get(status, f"status_{status}"),
            "solver_runtime_seconds": float(model.Runtime),
        }
        model.dispose()
        return summary, None

    q_base_value = q_base.X
    d_base_value = d_base.X
    s_base_value = s_base.X
    q_tail_value = q_tail.X
    d_tail_value = d_tail.X
    s_tail_value = s_tail.X
    unserved_value = unserved.X
    tail_power_value = float(tail_power.X)
    residual_peak_value = float(residual_peak.X)

    base_cyclic, base_soc_error, base_soc_min = soc_diagnostics(
        s_base_value,
        q_base_value,
        d_base_value,
        base_retention,
        base_eta,
    )
    tail_cyclic, tail_soc_error, tail_soc_min = soc_diagnostics(
        s_tail_value,
        q_tail_value,
        d_tail_value,
        tail_retention,
        tail_eta,
    )
    balance_residual = (
        d_base_value + d_tail_value + unserved_value - base_task_mw
    )
    shared_recharge_violation = (
        q_base_value + q_tail_value - usable_recharge_mw
    )

    base_charge_mwh = float(q_base_value.sum())
    tail_charge_mwh = float(q_tail_value.sum())
    base_discharge_mwh = float(d_base_value.sum())
    tail_discharge_mwh = float(d_tail_value.sum())
    unserved_mwh = float(unserved_value.sum())
    task_mwh = float(base_task_mw.sum())
    efc_mw = equivalent_firm_capacity(base_task_mw, unserved_mwh)
    fixed_base_value = (
        base_power_mw * 1000.0 * base["fixed_cost_eur_kw_year"]
    )
    fixed_tail_value = (
        tail_power_value
        * 1000.0
        * tail["fixed_cost_eur_kw_year"]
    )
    base_vom_value = base["vom_eur_mwh"] * base_discharge_mwh
    tail_vom_value = tail["vom_eur_mwh"] * tail_discharge_mwh
    charge_cost_value = scenario["charge_price_eur_mwh"] * (
        base_charge_mwh + tail_charge_mwh
    )
    residual_energy_cost_value = (
        scenario["energy_value_eur_mwh"] * unserved_mwh
    )
    residual_firm_cost_value = (
        scenario["firm_value_eur_kw_year"]
        * 1000.0
        * residual_peak_value
    )
    system_cost_value = (
        fixed_base_value
        + fixed_tail_value
        + base_vom_value
        + tail_vom_value
        + charge_cost_value
        + residual_energy_cost_value
        + residual_firm_cost_value
    )
    gas_only_system_cost = (
        scenario["energy_value_eur_mwh"] * task_mwh
        + scenario["firm_value_eur_kw_year"]
        * 1000.0
        * float(base_task_mw.max())
    )
    tail_energy_capacity_mwh = tail["duration_h"] * tail_power_value
    tail_usable_capacity_mwh = (
        USABLE_SOC_FRACTION * tail_energy_capacity_mwh
    )
    shared_recharge_binding = (
        (usable_recharge_mw > NUMERICAL_ZERO)
        & (
            q_base_value + q_tail_value
            >= usable_recharge_mw - 1.0e-7
        )
    )
    summary = {
        "solver_status": status,
        "solver_status_name": "optimal",
        "solver_runtime_seconds": float(model.Runtime),
        "solver_iterations": float(model.IterCount),
        "solver_barrier_iterations": int(model.BarIterCount),
        "solver_node_count": float(model.NodeCount),
        "solver_variables": int(model.NumVars),
        "solver_constraints": int(model.NumConstrs),
        "solver_binary_variables": int(model.NumBinVars),
        "solver_integer_variables": int(model.NumIntVars),
        "base_power_mw": base_power_mw,
        "base_duration_h": base["duration_h"],
        "base_energy_capacity_mwh": base["duration_h"] * base_power_mw,
        "base_usable_energy_capacity_mwh": base_usable_capacity_mwh,
        "tail_power_upper_mw": tail_power_upper_mw,
        "tail_power_mode": tail_power_mode,
        "tail_power_mw": tail_power_value,
        "tail_energy_capacity_mwh": tail_energy_capacity_mwh,
        "tail_usable_energy_capacity_mwh": tail_usable_capacity_mwh,
        "base_task_mwh": task_mwh,
        "tail_task_mwh": float(tail_task_mw.sum()),
        "raw_recharge_mwh": float(recharge_mw.sum()),
        "usable_recharge_mwh": float(usable_recharge_mw.sum()),
        "task_recharge_overlap_mwh": float(
            recharge_mw[base_task_mw > NUMERICAL_ZERO].sum()
        ),
        "base_charge_mwh": base_charge_mwh,
        "tail_charge_mwh": tail_charge_mwh,
        "total_charge_mwh": base_charge_mwh + tail_charge_mwh,
        "base_discharge_mwh": base_discharge_mwh,
        "tail_discharge_mwh": tail_discharge_mwh,
        "total_discharge_mwh": base_discharge_mwh + tail_discharge_mwh,
        "unserved_eens_mwh": unserved_mwh,
        "served_share": (
            (base_discharge_mwh + tail_discharge_mwh) / task_mwh
            if task_mwh > 0.0
            else 0.0
        ),
        "portfolio_efc_mw": efc_mw,
        "portfolio_efc_ratio_to_base_power": efc_mw / base_power_mw,
        "firm_peak_reduction_mw": (
            float(base_task_mw.max()) - residual_peak_value
        ),
        "residual_peak_mw": residual_peak_value,
        "base_equivalent_full_cycles": (
            base_discharge_mwh / (base["duration_h"] * base_power_mw)
        ),
        "tail_equivalent_full_cycles": (
            tail_discharge_mwh / tail_energy_capacity_mwh
            if tail_energy_capacity_mwh > NUMERICAL_ZERO
            else 0.0
        ),
        "base_fixed_cost_eur": fixed_base_value,
        "tail_fixed_cost_eur": fixed_tail_value,
        "base_vom_cost_eur": base_vom_value,
        "tail_vom_cost_eur": tail_vom_value,
        "charge_opportunity_cost_eur": charge_cost_value,
        "residual_energy_cost_eur": residual_energy_cost_value,
        "residual_firm_cost_eur": residual_firm_cost_value,
        "storage_cost_eur": (
            fixed_base_value
            + fixed_tail_value
            + base_vom_value
            + tail_vom_value
            + charge_cost_value
        ),
        "system_cost_eur": system_cost_value,
        "gas_only_system_cost_eur": gas_only_system_cost,
        "net_benefit_vs_gas_only_eur": (
            gas_only_system_cost - system_cost_value
        ),
        "system_cost_eur_per_mwh_task": (
            system_cost_value / task_mwh if task_mwh > 0.0 else np.nan
        ),
        "base_soc_cyclic_residual_mwh": base_cyclic,
        "tail_soc_cyclic_residual_mwh": tail_cyclic,
        "base_max_soc_balance_residual_mwh": base_soc_error,
        "tail_max_soc_balance_residual_mwh": tail_soc_error,
        "base_soc_min_fraction": (
            SOC_MIN + base_soc_min / (base["duration_h"] * base_power_mw)
        ),
        "base_soc_max_fraction": (
            SOC_MIN
            + float(s_base_value.max())
            / (base["duration_h"] * base_power_mw)
        ),
        "tail_soc_min_fraction": (
            SOC_MIN + float(s_tail_value.min()) / tail_energy_capacity_mwh
            if tail_energy_capacity_mwh > NUMERICAL_ZERO
            else np.nan
        ),
        "tail_soc_max_fraction": (
            SOC_MIN + float(s_tail_value.max()) / tail_energy_capacity_mwh
            if tail_energy_capacity_mwh > NUMERICAL_ZERO
            else np.nan
        ),
        "maximum_task_balance_residual_mw": float(
            np.max(np.abs(balance_residual))
        ),
        "maximum_shared_recharge_violation_mw": float(
            max(0.0, np.max(shared_recharge_violation))
        ),
        "shared_recharge_binding_hours": int(shared_recharge_binding.sum()),
        "base_charge_power_binding_hours": int(
            np.sum(
                q_base_value
                >= base["charge_power_ratio"] * base_power_mw - 1.0e-7
            )
        ),
        "tail_charge_power_binding_hours": int(
            np.sum(
                (tail_power_value > NUMERICAL_ZERO)
                & (
                    q_tail_value
                    >= tail["charge_power_ratio"] * tail_power_value
                    - 1.0e-7
                )
            )
        ),
        "base_discharge_power_binding_hours": int(
            np.sum(d_base_value >= base_power_mw - 1.0e-7)
        ),
        "tail_discharge_power_binding_hours": int(
            np.sum(
                (tail_power_value > NUMERICAL_ZERO)
                & (d_tail_value >= tail_power_value - 1.0e-7)
            )
        ),
    }
    schedule = None
    if write_hourly:
        schedule = pd.DataFrame(
            {
                "model_hour": np.arange(hours, dtype=int),
                "base_task_c1_mw": base_task_mw,
                "tail_task_c3_q75_mw": tail_task_mw,
                "raw_recharge_r2_mw": recharge_mw,
                "usable_recharge_r2_mw": usable_recharge_mw,
                "base_charge_mw": q_base_value,
                "base_discharge_mw": d_base_value,
                "base_usable_soc_mwh": s_base_value,
                "tail_charge_mw": q_tail_value,
                "tail_discharge_mw": d_tail_value,
                "tail_usable_soc_mwh": s_tail_value,
                "unserved_task_mw": unserved_value,
            }
        )
    model.dispose()
    return summary, schedule


def add_reference_effects(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["country", "year", "chronology"]
    base_columns = keys + [
        "system_cost_eur",
        "unserved_eens_mwh",
        "portfolio_efc_mw",
        "firm_peak_reduction_mw",
        "total_discharge_mwh",
        "total_charge_mwh",
        "base_vom_cost_eur",
        "tail_vom_cost_eur",
        "tail_fixed_cost_eur",
    ]
    base = frame[frame["arm_type"].eq("base_only")][base_columns].copy()
    base = base.rename(
        columns={
            column: f"base_only_{column}"
            for column in base_columns
            if column not in keys
        }
    )
    result = frame.merge(base, on=keys, how="left", validate="many_to_one")
    result["incremental_system_cost_reduction_vs_base_only_eur"] = (
        result["base_only_system_cost_eur"] - result["system_cost_eur"]
    )
    result["incremental_eens_reduction_vs_base_only_mwh"] = (
        result["base_only_unserved_eens_mwh"]
        - result["unserved_eens_mwh"]
    )
    result["incremental_efc_vs_base_only_mw"] = (
        result["portfolio_efc_mw"]
        - result["base_only_portfolio_efc_mw"]
    )
    result["incremental_firm_peak_reduction_vs_base_only_mw"] = (
        result["firm_peak_reduction_mw"]
        - result["base_only_firm_peak_reduction_mw"]
    )
    result["incremental_discharge_vs_base_only_mwh"] = (
        result["total_discharge_mwh"]
        - result["base_only_total_discharge_mwh"]
    )
    result["incremental_charge_vs_base_only_mwh"] = (
        result["total_charge_mwh"] - result["base_only_total_charge_mwh"]
    )
    result["incremental_vom_vs_base_only_eur"] = (
        result["base_vom_cost_eur"]
        + result["tail_vom_cost_eur"]
        - result["base_only_base_vom_cost_eur"]
        - result["base_only_tail_vom_cost_eur"]
    )

    comparator_columns = keys + [
        "system_cost_eur",
        "unserved_eens_mwh",
        "portfolio_efc_mw",
        "firm_peak_reduction_mw",
        "total_discharge_mwh",
        "total_charge_mwh",
        "tail_power_mw",
        "tail_fixed_cost_eur",
    ]
    comparator = frame[
        frame["arm_type"].eq("extra_lfp4_comparator")
    ][comparator_columns].copy()
    comparator = comparator.rename(
        columns={
            column: f"extra_lfp4_{column}"
            for column in comparator_columns
            if column not in keys
        }
    )
    result = result.merge(
        comparator, on=keys, how="left", validate="many_to_one"
    )
    result["mix_net_benefit_vs_extra_lfp4_eur"] = (
        result["extra_lfp4_system_cost_eur"] - result["system_cost_eur"]
    )
    result["mix_eens_reduction_vs_extra_lfp4_mwh"] = (
        result["extra_lfp4_unserved_eens_mwh"]
        - result["unserved_eens_mwh"]
    )
    result["mix_efc_gain_vs_extra_lfp4_mw"] = (
        result["portfolio_efc_mw"]
        - result["extra_lfp4_portfolio_efc_mw"]
    )
    result["mix_firm_peak_gain_vs_extra_lfp4_mw"] = (
        result["firm_peak_reduction_mw"]
        - result["extra_lfp4_firm_peak_reduction_mw"]
    )
    result["mix_discharge_gain_vs_extra_lfp4_mwh"] = (
        result["total_discharge_mwh"]
        - result["extra_lfp4_total_discharge_mwh"]
    )
    result["mix_charge_change_vs_extra_lfp4_mwh"] = (
        result["total_charge_mwh"]
        - result["extra_lfp4_total_charge_mwh"]
    )
    candidate_comparison = result[
        result["arm_type"].eq("ldes_candidate")
    ].copy()
    return result, candidate_comparison


def build_chronology_pairing(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "system_cost_eur",
        "net_benefit_vs_gas_only_eur",
        "incremental_system_cost_reduction_vs_base_only_eur",
        "unserved_eens_mwh",
        "portfolio_efc_mw",
        "incremental_efc_vs_base_only_mw",
        "total_discharge_mwh",
        "tail_discharge_mwh",
        "tail_equivalent_full_cycles",
        "tail_power_mw",
        "mix_net_benefit_vs_extra_lfp4_eur",
    ]
    keys = [
        "country",
        "year",
        "arm_type",
        "config_id",
        "tech_id",
        "technology_en",
        "duration_h",
    ]
    values = frame[keys + ["chronology"] + metrics].copy()
    wide = values.pivot(
        index=keys, columns="chronology", values=metrics
    ).reset_index()
    wide.columns = [
        "_".join([str(part) for part in column if str(part)])
        if isinstance(column, tuple)
        else str(column)
        for column in wide.columns
    ]
    for metric in metrics:
        dispersed = f"{metric}_dispersed"
        clustered = f"{metric}_clustered"
        if dispersed in wide and clustered in wide:
            wide[f"clustered_minus_dispersed_{metric}"] = (
                wide[clustered] - wide[dispersed]
            )
    return wide


def make_qa(
    results: pd.DataFrame,
    chronology: pd.DataFrame,
    country_years: list[tuple[str, int]],
    chronologies: list[str],
    candidate_count: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    expected_rows = (
        len(country_years) * len(chronologies) * (candidate_count + 2)
    )
    failures: list[str] = []
    checks = {
        "expected_result_rows": expected_rows,
        "actual_result_rows": len(results),
        "all_solver_status_optimal": bool(
            results["solver_status_name"].eq("optimal").all()
        ),
        "all_models_continuous": bool(
            results["solver_binary_variables"].eq(0).all()
            and results["solver_integer_variables"].eq(0).all()
        ),
        "maximum_task_balance_residual_mw": float(
            results["maximum_task_balance_residual_mw"].max()
        ),
        "maximum_shared_recharge_violation_mw": float(
            results["maximum_shared_recharge_violation_mw"].max()
        ),
        "maximum_base_soc_balance_residual_mwh": float(
            results["base_max_soc_balance_residual_mwh"].max()
        ),
        "maximum_tail_soc_balance_residual_mwh": float(
            results["tail_max_soc_balance_residual_mwh"].max()
        ),
        "maximum_chronology_base_task_conservation_error_mwh": float(
            chronology["base_task_conservation_error_mwh"].abs().max()
        ),
        "maximum_chronology_tail_task_conservation_error_mwh": float(
            chronology["tail_task_conservation_error_mwh"].abs().max()
        ),
        "maximum_chronology_recharge_conservation_error_mwh": float(
            chronology["recharge_conservation_error_mwh"].abs().max()
        ),
        "maximum_base_only_tail_power_mw": float(
            results.loc[
                results["arm_type"].eq("base_only"), "tail_power_mw"
            ].abs().max()
        ),
        "maximum_system_cost_identity_error_eur": float(
            (
                results["system_cost_eur"]
                - (
                    results["base_fixed_cost_eur"]
                    + results["tail_fixed_cost_eur"]
                    + results["base_vom_cost_eur"]
                    + results["tail_vom_cost_eur"]
                    + results["charge_opportunity_cost_eur"]
                    + results["residual_energy_cost_eur"]
                    + results["residual_firm_cost_eur"]
                )
            )
            .abs()
            .max()
        ),
    }
    if checks["actual_result_rows"] != checks["expected_result_rows"]:
        failures.append("row_count")
    if not checks["all_solver_status_optimal"]:
        failures.append("solver_status")
    if not checks["all_models_continuous"]:
        failures.append("model_integrality")
    if checks["maximum_task_balance_residual_mw"] > BALANCE_TOLERANCE_MW:
        failures.append("task_balance")
    if checks["maximum_shared_recharge_violation_mw"] > BALANCE_TOLERANCE_MW:
        failures.append("shared_recharge")
    if checks["maximum_base_soc_balance_residual_mwh"] > SOC_TOLERANCE_MWH:
        failures.append("base_soc")
    if checks["maximum_tail_soc_balance_residual_mwh"] > SOC_TOLERANCE_MWH:
        failures.append("tail_soc")
    if checks["maximum_base_only_tail_power_mw"] > BALANCE_TOLERANCE_MW:
        failures.append("base_only_tail_power")
    return {
        "timestamp_utc": utc_now(),
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "checks": checks,
        "country_years": country_years,
        "chronologies": chronologies,
        "candidate_count": candidate_count,
        "elapsed_seconds": elapsed_seconds,
        "evidence_status": (
            "smoke_or_partial_run_not_manuscript_evidence"
            if len(country_years) < 132
            else "full_run_requires_independent_result_review"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE)
    parser.add_argument("--scenario-summary", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument(
        "--scenario-id",
        type=int,
        default=DEFAULT_SCENARIO_ID,
        help="Deterministic value scenario identifier in the scenario table",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--cluster-days",
        type=int,
        default=DEFAULT_CLUSTER_DAYS,
        help=(
            "Target number of adjacent q75 event days per controlled cluster. "
            "The dispersed chronology is unchanged."
        ),
    )
    parser.add_argument(
        "--country-years",
        type=str,
        default="DE:2023",
        help="Comma list such as DE:2023,ES:2022",
    )
    parser.add_argument("--max-country-years", type=int, default=None)
    parser.add_argument(
        "--chronologies", type=str, default="dispersed,clustered"
    )
    parser.add_argument(
        "--candidate-config-ids",
        type=str,
        default=None,
        help=(
            "Optional comma-separated non-LFP configurations.  The base-only "
            "and extra-4h-LFP comparator arms are always included."
        ),
    )
    parser.add_argument(
        "--tail-power-upper-multiplier",
        type=float,
        default=1.0,
        help="Tail discharge-power upper bound divided by common LFP power",
    )
    parser.add_argument(
        "--tail-power-mode",
        choices=["fixed", "optimized"],
        default="fixed",
        help=(
            "fixed keeps comparator and LDES tail power at the common "
            "prespecified scale; optimized permits 0--upper-bound selection"
        ),
    )
    parser.add_argument(
        "--recharge-multiplier",
        type=float,
        default=1.0,
        help=(
            "Multiplier applied to the common hourly renewable-surplus "
            "recharge series; used for recharge-availability counterfactuals"
        ),
    )
    parser.add_argument(
        "--recharge-profile-mode",
        choices=["observed", "uniform"],
        default="observed",
        help=(
            "observed retains the reordered hourly recharge profile; uniform "
            "redistributes exactly the same annual recharge evenly across all "
            "hours as a timing upper-bound counterfactual"
        ),
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--method",
        type=int,
        default=1,
        choices=[-1, 0, 1, 2, 3, 4, 5],
    )
    parser.add_argument("--write-hourly", action="store_true")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print one progress line per this many solved arms.",
    )
    args = parser.parse_args()
    if args.tail_power_upper_multiplier < 0.0:
        raise ValueError("--tail-power-upper-multiplier cannot be negative")
    if args.cluster_days <= 0:
        raise ValueError("--cluster-days must be a positive integer")
    if args.recharge_multiplier <= 0.0:
        raise ValueError("--recharge-multiplier must be positive")
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive")
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    requested_country_years = parse_country_years(args.country_years)
    chronologies = parse_csv_list(args.chronologies)
    if chronologies is None or any(
        item not in EXPECTED_CHRONOLOGIES for item in chronologies
    ):
        raise ValueError(
            f"Chronologies must be drawn from {EXPECTED_CHRONOLOGIES}"
        )
    requested_configs = parse_csv_list(args.candidate_config_ids)
    scenario = load_scenario(args.scenario_summary, args.scenario_id)
    usecols = [
        "country",
        "year",
        "source_hour_id",
        "source_day_id",
        "season_segment",
        "event_flag_q75",
        "raw_gas_mean_mw",
        "forced_discharge_power_mw",
        BASE_TASK_COLUMN,
        TAIL_TASK_COLUMN,
        RECHARGE_COLUMN,
    ]
    panel = pd.read_csv(args.panel, usecols=usecols)
    panel = panel.sort_values(
        ["country", "year", "source_hour_id"], kind="stable"
    ).reset_index(drop=True)
    panel, country_years = select_panel(
        panel, requested_country_years, args.max_country_years
    )
    catalogue = pd.read_csv(args.catalogue)
    base_config, candidates = select_catalogue(
        catalogue, requested_configs
    )
    comparator_config = base_config.copy()

    contract = {
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "panel": str(args.panel.resolve()),
        "panel_sha256": sha256(args.panel),
        "catalogue": str(args.catalogue.resolve()),
        "catalogue_sha256": sha256(args.catalogue),
        "scenario_summary": str(args.scenario_summary.resolve()),
        "scenario_summary_sha256": sha256(args.scenario_summary),
        "scenario_id": args.scenario_id,
        "scenario": scenario,
        "cluster_days": args.cluster_days,
        "country_years": country_years,
        "chronologies": chronologies,
        "base_config_id": BASE_CONFIG_ID,
        "candidate_config_ids": candidates["config_id"].tolist(),
        "tail_power_upper_multiplier": args.tail_power_upper_multiplier,
        "tail_power_mode": args.tail_power_mode,
        "recharge_multiplier": args.recharge_multiplier,
        "recharge_profile_mode": args.recharge_profile_mode,
        "base_task_column": BASE_TASK_COLUMN,
        "tail_task_column": TAIL_TASK_COLUMN,
        "recharge_column": RECHARGE_COLUMN,
        "soc_min": SOC_MIN,
        "soc_max": SOC_MAX,
        "threads": args.threads,
        "method": args.method,
        "write_hourly": args.write_hourly,
        "python": sys.version,
        "platform": platform.platform(),
        "gurobi_version": gp.gurobi.version(),
        "started_utc": utc_now(),
    }
    write_json(args.output / "run_contract.json", contract)

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    rows: list[dict[str, Any]] = []
    chronology_rows: list[dict[str, Any]] = []
    schedule_frames: list[pd.DataFrame] = []
    solved = 0
    total = len(country_years) * len(chronologies) * (len(candidates) + 2)
    for country, year in country_years:
        frame = panel[
            panel["country"].eq(country) & panel["year"].eq(year)
        ].copy()
        if len(frame) != 8760:
            raise RuntimeError(f"{country}-{year} has {len(frame)} hours")
        orders, order_stats = day_orders(frame, args.cluster_days)
        base_original = frame[BASE_TASK_COLUMN].to_numpy(float)
        tail_original = frame[TAIL_TASK_COLUMN].to_numpy(float)
        recharge_original = (
            frame[RECHARGE_COLUMN].to_numpy(float)
            * args.recharge_multiplier
        )
        forced_power = float(frame["forced_discharge_power_mw"].iloc[0])
        derived_power = (
            FORCED_POWER_SHARE * float(frame["raw_gas_mean_mw"].iloc[0])
        )
        if not np.isclose(forced_power, derived_power, rtol=0.0, atol=1e-8):
            raise RuntimeError(
                f"{country}-{year} forced-power definition is inconsistent"
            )
        tail_power_upper = (
            args.tail_power_upper_multiplier * forced_power
        )
        for chronology in chronologies:
            order = orders[chronology]
            base_task = reorder_days(base_original, order)
            tail_task = reorder_days(tail_original, order)
            recharge = reorder_days(recharge_original, order)
            if args.recharge_profile_mode == "uniform":
                recharge = np.full_like(
                    recharge,
                    fill_value=float(recharge.sum()) / len(recharge),
                )
            chronology_rows.append(
                {
                    "country": country,
                    "year": year,
                    "chronology": chronology,
                    **order_stats,
                    "base_task_mwh": float(base_task.sum()),
                    "tail_task_mwh": float(tail_task.sum()),
                    "recharge_mwh": float(recharge.sum()),
                    "base_task_conservation_error_mwh": float(
                        base_task.sum() - base_original.sum()
                    ),
                    "tail_task_conservation_error_mwh": float(
                        tail_task.sum() - tail_original.sum()
                    ),
                    "recharge_conservation_error_mwh": float(
                        recharge.sum() - recharge_original.sum()
                    ),
                }
            )
            arms: list[tuple[str, pd.Series, float]] = [
                ("base_only", comparator_config, 0.0),
                (
                    "extra_lfp4_comparator",
                    comparator_config,
                    tail_power_upper,
                ),
            ]
            arms.extend(
                ("ldes_candidate", config, tail_power_upper)
                for _, config in candidates.iterrows()
            )
            for arm_type, tail_config, upper in arms:
                summary, schedule = solve_portfolio_arm(
                    base_task,
                    tail_task,
                    recharge,
                    forced_power,
                    upper,
                    args.tail_power_mode,
                    base_config,
                    tail_config,
                    scenario,
                    env,
                    args.threads,
                    args.method,
                    args.write_hourly,
                )
                identity = config_identity(tail_config)
                if arm_type == "base_only":
                    identity = {
                        **identity,
                        "config_id": "base_only_lfp4",
                        "tech_id": "none",
                        "technology_en": "No incremental tail asset",
                        "duration_h": 0.0,
                    }
                row = {
                    "country": country,
                    "year": year,
                    "chronology": chronology,
                    "arm_type": arm_type,
                    **identity,
                    "scenario_id": args.scenario_id,
                    **scenario,
                    **summary,
                }
                rows.append(row)
                if schedule is not None:
                    schedule.insert(0, "config_id", identity["config_id"])
                    schedule.insert(0, "arm_type", arm_type)
                    schedule.insert(0, "chronology", chronology)
                    schedule.insert(0, "year", year)
                    schedule.insert(0, "country", country)
                    schedule_frames.append(schedule)
                solved += 1
                if solved % args.progress_every == 0 or solved == total:
                    print(
                        f"[solve {solved}/{total}] {country}-{year} "
                        f"{chronology} {identity['config_id']} "
                        f"{summary.get('solver_status_name')} "
                        f"{summary.get('solver_runtime_seconds', np.nan):.2f}s",
                        flush=True,
                    )
    env.dispose()

    results = pd.DataFrame(rows)
    chronology_frame = pd.DataFrame(chronology_rows)
    if not results["solver_status_name"].eq("optimal").all():
        output_csv_gz(
            results, args.output / "portfolio_arm_results_unfinished.csv.gz"
        )
        raise RuntimeError("At least one portfolio LP was not optimal")
    results, candidate_comparison = add_reference_effects(results)
    paired = build_chronology_pairing(results)
    output_csv_gz(results, args.output / "portfolio_arm_results.csv.gz")
    output_csv_gz(
        candidate_comparison,
        args.output / "ldes_vs_extra_lfp4_results.csv.gz",
    )
    paired.to_csv(
        args.output / "clustered_vs_dispersed_paired_effects.csv",
        index=False,
    )
    chronology_frame.to_csv(
        args.output / "chronology_conservation.csv", index=False
    )
    if schedule_frames:
        output_csv_gz(
            pd.concat(schedule_frames, ignore_index=True),
            args.output / "portfolio_hourly_schedules.csv.gz",
        )
    elapsed = time.perf_counter() - started
    qa = make_qa(
        results,
        chronology_frame,
        country_years,
        chronologies,
        len(candidates),
        elapsed,
    )
    write_json(args.output / "qa_summary.json", qa)
    if qa["status"] != "pass":
        raise RuntimeError(f"QA failed: {qa['failures']}")
    print(
        f"Completed {len(results)} LPs in {elapsed:.2f}s; "
        f"QA={qa['status']}; output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
