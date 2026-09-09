#!/usr/bin/env python3
"""HiGHS-equivalent natural-order focal portfolio experiment.

The Gurobi token service was unavailable when v44 was run.  The registered
portfolio formulation is a continuous LP, so this script reconstructs the
same fixed-power equations in scipy.optimize.linprog (HiGHS dual simplex).
It first supports an exact smoke comparison against the stored Gurobi S20
solution and then solves the unaltered natural chronology for all 132
country-years, for the additional-4 h-LFP and 24 h A-CAES tail arms.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, vstack


HERE = Path(__file__).resolve().parent
V44 = HERE.parent
V43 = V44.parent / "revision_nature_energy_v43"
RUNNER_PATH = V44 / "portfolio_model" / "run_v42_europe_portfolio_gurobi.py"
GUROBI_REFERENCE = V43 / "portfolio_model" / "full_132cy_fixed_s20_central" / "portfolio_arm_results.csv.gz"
OUTPUT = HERE / "natural_caes24_s20_fixed_highs"
HOURS = 8760
PRIMARY_ABS_TOL_EUR = 0.01


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_runner():
    spec = importlib.util.spec_from_file_location("v44_registered_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import registered model utilities")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_equalities(
    task: np.ndarray,
    base_eta: float,
    tail_eta: float,
    base_retention: float,
    tail_retention: float,
) -> tuple[csr_matrix, np.ndarray]:
    # Variable blocks qB,dB,sB,qT,dT,sT,u,r.
    q_b, d_b, s_b, q_t, d_t, s_t, u = [i * HOURS for i in range(7)]
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    hours = np.arange(HOURS)

    # Joint task balance.
    rows.extend([hours, hours, hours])
    cols.extend([d_b + hours, d_t + hours, u + hours])
    vals.extend([np.ones(HOURS), np.ones(HOURS), np.ones(HOURS)])

    # Cyclic base SOC.
    row_b = HOURS + hours
    previous = np.roll(hours, 1)
    rows.extend([row_b, row_b, row_b, row_b])
    cols.extend([s_b + hours, s_b + previous, q_b + hours, d_b + hours])
    vals.extend([
        np.ones(HOURS),
        -base_retention * np.ones(HOURS),
        -base_eta * np.ones(HOURS),
        np.full(HOURS, 1.0 / base_eta),
    ])

    # Cyclic tail SOC.
    row_t = 2 * HOURS + hours
    rows.extend([row_t, row_t, row_t, row_t])
    cols.extend([s_t + hours, s_t + previous, q_t + hours, d_t + hours])
    vals.extend([
        np.ones(HOURS),
        -tail_retention * np.ones(HOURS),
        -tail_eta * np.ones(HOURS),
        np.full(HOURS, 1.0 / tail_eta),
    ])
    matrix = coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(3 * HOURS, 7 * HOURS + 1),
    ).tocsr()
    return matrix, np.concatenate([task, np.zeros(2 * HOURS)])


def build_inequalities(recharge: np.ndarray) -> tuple[csr_matrix, np.ndarray]:
    q_b, q_t, u, residual_peak = 0, 3 * HOURS, 6 * HOURS, 7 * HOURS
    hours = np.arange(HOURS)
    shared_rows = hours
    peak_rows = HOURS + hours
    matrix = coo_matrix(
        (
            np.concatenate([
                np.ones(HOURS),
                np.ones(HOURS),
                np.ones(HOURS),
                -np.ones(HOURS),
            ]),
            (
                np.concatenate([shared_rows, shared_rows, peak_rows, peak_rows]),
                np.concatenate([
                    q_b + hours,
                    q_t + hours,
                    u + hours,
                    np.full(HOURS, residual_peak),
                ]),
            ),
        ),
        shape=(2 * HOURS, 7 * HOURS + 1),
    ).tocsr()
    return matrix, np.concatenate([recharge, np.zeros(HOURS)])


def solve_fixed_arm(
    task: np.ndarray,
    tail_task: np.ndarray,
    raw_recharge: np.ndarray,
    power: float,
    base: dict[str, float],
    tail: dict[str, float],
    scenario: dict[str, float],
) -> dict[str, float | int | str]:
    usable_recharge = np.where(task > 1.0e-10, 0.0, raw_recharge)
    base_eta = float(np.sqrt(base["rte"]))
    tail_eta = float(np.sqrt(tail["rte"]))
    base_retention = float((1.0 - base["self_discharge_pct_day"] / 100.0) ** (1.0 / 24.0))
    tail_retention = float((1.0 - tail["self_discharge_pct_day"] / 100.0) ** (1.0 / 24.0))
    a_eq, b_eq = build_equalities(task, base_eta, tail_eta, base_retention, tail_retention)
    a_ub, b_ub = build_inequalities(usable_recharge)

    q_b, d_b, s_b, q_t, d_t, s_t, u, r = [i * HOURS for i in range(8)]
    lower = np.zeros(7 * HOURS + 1)
    upper = np.full(7 * HOURS + 1, np.inf)
    upper[q_b:d_b] = np.minimum(usable_recharge, base["charge_power_ratio"] * power)
    upper[d_b:s_b] = np.minimum(task, power)
    upper[s_b:q_t] = 0.9 * base["duration_h"] * power
    upper[q_t:d_t] = tail["charge_power_ratio"] * power
    upper[d_t:s_t] = np.minimum(tail_task, power)
    upper[s_t:u] = 0.9 * tail["duration_h"] * power
    upper[u:r] = task
    bounds = np.column_stack([lower, upper])

    objective = np.zeros(7 * HOURS + 1)
    objective[q_b:d_b] = scenario["charge_price_eur_mwh"]
    objective[d_b:s_b] = base["vom_eur_mwh"]
    objective[q_t:d_t] = scenario["charge_price_eur_mwh"]
    objective[d_t:s_t] = tail["vom_eur_mwh"]
    objective[u:r] = scenario["energy_value_eur_mwh"]
    objective[r] = 1000.0 * scenario["firm_value_eur_kw_year"]
    started = time.perf_counter()
    primary = linprog(
        objective,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs-ds",
        options={"dual_feasibility_tolerance": 1.0e-8, "primal_feasibility_tolerance": 1.0e-8},
    )
    if not primary.success:
        raise RuntimeError(f"HiGHS primary solve failed: {primary.message}")
    charge_objective = np.zeros_like(objective)
    charge_objective[q_b:d_b] = 1.0
    charge_objective[q_t:d_t] = 1.0
    secondary = linprog(
        charge_objective,
        A_ub=vstack([a_ub, csr_matrix(objective.reshape(1, -1))], format="csr"),
        b_ub=np.append(b_ub, primary.fun + PRIMARY_ABS_TOL_EUR),
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs-ds",
        options={"dual_feasibility_tolerance": 1.0e-8, "primal_feasibility_tolerance": 1.0e-8},
    )
    if not secondary.success:
        raise RuntimeError(f"HiGHS secondary solve failed: {secondary.message}")
    x = secondary.x
    q_base, d_base, s_base = x[q_b:d_b], x[d_b:s_b], x[s_b:q_t]
    q_tail, d_tail, s_tail = x[q_t:d_t], x[d_t:s_t], x[s_t:u]
    unserved = x[u:r]
    residual_peak = float(x[r])
    base_charge, tail_charge = float(q_base.sum()), float(q_tail.sum())
    base_discharge, tail_discharge = float(d_base.sum()), float(d_tail.sum())
    eens = float(unserved.sum())
    runner = load_runner()
    efc = runner.equivalent_firm_capacity(task, eens)
    base_fixed = 1000.0 * power * base["fixed_cost_eur_kw_year"]
    tail_fixed = 1000.0 * power * tail["fixed_cost_eur_kw_year"]
    base_vom = base["vom_eur_mwh"] * base_discharge
    tail_vom = tail["vom_eur_mwh"] * tail_discharge
    charge_cost = scenario["charge_price_eur_mwh"] * (base_charge + tail_charge)
    system_cost = (
        base_fixed + tail_fixed + base_vom + tail_vom + charge_cost
        + scenario["energy_value_eur_mwh"] * eens
        + 1000.0 * scenario["firm_value_eur_kw_year"] * residual_peak
    )
    balance = d_base + d_tail + unserved - task
    base_soc = s_base - base_retention * np.roll(s_base, 1) - base_eta * q_base + d_base / base_eta
    tail_soc = s_tail - tail_retention * np.roll(s_tail, 1) - tail_eta * q_tail + d_tail / tail_eta
    return {
        "solver_status_name": "optimal",
        "solver_method": "scipy_highs_ds_two_stage",
        "solver_runtime_seconds": time.perf_counter() - started,
        "base_power_mw": power,
        "tail_power_mw": power,
        "base_task_mwh": float(task.sum()),
        "tail_task_mwh": float(tail_task.sum()),
        "raw_recharge_mwh": float(raw_recharge.sum()),
        "usable_recharge_mwh": float(usable_recharge.sum()),
        "base_charge_mwh": base_charge,
        "tail_charge_mwh": tail_charge,
        "total_charge_mwh": base_charge + tail_charge,
        "base_discharge_mwh": base_discharge,
        "tail_discharge_mwh": tail_discharge,
        "total_discharge_mwh": base_discharge + tail_discharge,
        "unserved_eens_mwh": eens,
        "portfolio_efc_mw": efc,
        "residual_peak_mw": residual_peak,
        "base_equivalent_full_cycles": base_discharge / (base["duration_h"] * power),
        "tail_equivalent_full_cycles": tail_discharge / (tail["duration_h"] * power),
        "base_fixed_cost_eur": base_fixed,
        "tail_fixed_cost_eur": tail_fixed,
        "base_vom_cost_eur": base_vom,
        "tail_vom_cost_eur": tail_vom,
        "charge_opportunity_cost_eur": charge_cost,
        "system_cost_eur": system_cost,
        "maximum_task_balance_residual_mw": float(np.abs(balance).max()),
        "maximum_shared_recharge_violation_mw": float(np.maximum(q_base + q_tail - usable_recharge, 0.0).max()),
        "maximum_base_soc_balance_residual_mwh": float(np.abs(base_soc).max()),
        "maximum_tail_soc_balance_residual_mwh": float(np.abs(tail_soc).max()),
    }


def inputs():
    runner = load_runner()
    panel = pd.read_csv(
        runner.DEFAULT_PANEL,
        usecols=[
            "country", "year", "source_hour_id", "source_day_id",
            "season_segment", "event_flag_q75", "raw_gas_mean_mw",
            "forced_discharge_power_mw", runner.BASE_TASK_COLUMN,
            runner.TAIL_TASK_COLUMN, runner.RECHARGE_COLUMN,
        ],
    ).sort_values(["country", "year", "source_hour_id"], kind="stable")
    catalogue = pd.read_csv(runner.DEFAULT_CATALOGUE)
    base_row, candidates = runner.select_catalogue(catalogue, ["caes_24h"])
    base = runner.model_parameters(base_row)
    caes = runner.model_parameters(candidates.iloc[0])
    scenario = runner.load_scenario(runner.DEFAULT_SCENARIO, 20)
    return runner, panel, base, caes, scenario


def smoke() -> None:
    runner, panel, base, caes, scenario = inputs()
    frame = panel[panel.country.eq("DE") & panel.year.eq(2023)]
    orders, _ = runner.day_orders(frame)
    order = orders["dispersed"]
    task = runner.reorder_days(frame[runner.BASE_TASK_COLUMN].to_numpy(float), order)
    tail_task = runner.reorder_days(frame[runner.TAIL_TASK_COLUMN].to_numpy(float), order)
    recharge = runner.reorder_days(frame[runner.RECHARGE_COLUMN].to_numpy(float), order)
    power = float(frame.forced_discharge_power_mw.iloc[0])
    reference = pd.read_csv(GUROBI_REFERENCE)
    checks = []
    for config_id, tail in [("li_lfp_4h", base), ("caes_24h", caes)]:
        result = solve_fixed_arm(task, tail_task, recharge, power, base, tail, scenario)
        ref = reference[
            reference.country.eq("DE")
            & reference.year.eq(2023)
            & reference.chronology.eq("dispersed")
            & reference.config_id.eq(config_id)
        ].iloc[0]
        for metric in [
            "total_charge_mwh", "total_discharge_mwh", "unserved_eens_mwh",
            "portfolio_efc_mw", "residual_peak_mw", "system_cost_eur",
        ]:
            absolute = abs(float(result[metric]) - float(ref[metric]))
            relative = absolute / max(abs(float(ref[metric])), 1.0)
            checks.append({"config_id": config_id, "metric": metric, "highs": result[metric], "gurobi": ref[metric], "absolute_difference": absolute, "relative_difference": relative})
    check = pd.DataFrame(checks)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    check.to_csv(OUTPUT / "highs_vs_gurobi_smoke.csv", index=False)
    qa = {
        "status": "pass" if check.relative_difference.max() < 1.0e-5 else "fail",
        "maximum_relative_difference": float(check.relative_difference.max()),
        "maximum_absolute_difference": float(check.absolute_difference.max()),
    }
    (OUTPUT / "smoke_qa.json").write_text(json.dumps(qa, indent=2) + "\n")
    print(check.to_string(index=False))
    print(json.dumps(qa, indent=2))
    if qa["status"] != "pass":
        raise RuntimeError("HiGHS-to-Gurobi equivalence smoke test failed")


def full() -> None:
    runner, panel, base, caes, scenario = inputs()
    smoke_qa = json.loads((OUTPUT / "smoke_qa.json").read_text())
    if smoke_qa.get("status") != "pass":
        raise RuntimeError("A passing smoke comparison is required before the full run")
    pairs = panel[["country", "year"]].drop_duplicates().sort_values(["country", "year"])
    rows = []
    for index, pair in enumerate(pairs.itertuples(index=False), start=1):
        frame = panel[panel.country.eq(pair.country) & panel.year.eq(pair.year)]
        task = frame[runner.BASE_TASK_COLUMN].to_numpy(float)
        tail_task = frame[runner.TAIL_TASK_COLUMN].to_numpy(float)
        recharge = frame[runner.RECHARGE_COLUMN].to_numpy(float)
        power = float(frame.forced_discharge_power_mw.iloc[0])
        for config_id, tech_id, duration, tail in [
            ("li_lfp_4h", "li_lfp", 4.0, base),
            ("caes_24h", "caes", 24.0, caes),
        ]:
            result = solve_fixed_arm(task, tail_task, recharge, power, base, tail, scenario)
            rows.append({"country": pair.country, "year": int(pair.year), "chronology": "natural", "config_id": config_id, "tech_id": tech_id, "duration_h": duration, **scenario, **result})
        if index % 10 == 0 or index == len(pairs):
            print(f"natural HiGHS solved {index}/{len(pairs)}", flush=True)
    output = pd.DataFrame(rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "natural_focal_arm_results.csv.gz"
    output.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
    qa = {
        "status": "pass",
        "country_year_count": int(len(pairs)),
        "row_count": len(output),
        "all_optimal": bool(output.solver_status_name.eq("optimal").all()),
        "maximum_task_balance_residual_mw": float(output.maximum_task_balance_residual_mw.max()),
        "maximum_shared_recharge_violation_mw": float(output.maximum_shared_recharge_violation_mw.max()),
        "maximum_soc_balance_residual_mwh": float(max(output.maximum_base_soc_balance_residual_mwh.max(), output.maximum_tail_soc_balance_residual_mwh.max())),
        "output_sha256": sha256(path),
    }
    (OUTPUT / "full_qa.json").write_text(json.dumps(qa, indent=2) + "\n")
    print(json.dumps(qa, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()
    smoke() if args.mode == "smoke" else full()


if __name__ == "__main__":
    main()
