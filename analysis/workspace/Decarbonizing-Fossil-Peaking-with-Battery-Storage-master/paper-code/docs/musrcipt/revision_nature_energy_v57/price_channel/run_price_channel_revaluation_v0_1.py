#!/usr/bin/env python3
"""Revalue the frozen European portfolio dispatch under hourly price channels.

S0 reproduces the central constant-value ledger.  S1 applies a scarcity
premium that rises with the ordinal day inside each q75 event run.  S2 maps
the usable renewable surplus linearly to an hourly charging opportunity cost.
S3a optionally reads country-hour price predictions estimated from licensed
day-ahead prices.  S4 assigns the prespecified scarcity cap to q75-event task
hours.  Dispatch is held fixed in every channel so that the calculation tests
the price ledger rather than a new operational policy.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, vstack


HERE = Path(__file__).resolve().parent
V46 = HERE.parent
sys.path.insert(0, str(V46 / "portfolio_model"))
import run_v42_europe_portfolio_gurobi as base  # noqa: E402
sys.path.insert(0, str(V46 / "joint_choice"))
import run_bounded_s20_highs as highs  # noqa: E402


BETAS = (0.25, 0.5, 1.0)
CHARGE_PRICE_MAX = 12.32
SCARCITY_CAP = 4000.0


def robust_linprog(
    objective: np.ndarray,
    *,
    a_ub,
    b_ub,
    a_eq,
    b_eq,
    bounds,
):
    """Solve one LP with deterministic HiGHS fallbacks.

    A small number of country-years (and several sub-MW community cases)
    trigger HiGHS status 15 under dual simplex although the same bounded LP
    is feasible.  Retrying with the interior-point engine and then the
    automatic HiGHS selector avoids silently dropping those experimental
    units.  Every accepted solution must satisfy SciPy's success flag.
    """
    attempts = (
        ("highs-ds", {"dual_feasibility_tolerance": 1.0e-8, "primal_feasibility_tolerance": 1.0e-8}),
        ("highs-ipm", {"dual_feasibility_tolerance": 1.0e-7, "primal_feasibility_tolerance": 1.0e-7,
                       "ipm_optimality_tolerance": 1.0e-9}),
        ("highs", {"presolve": True}),
    )
    messages: list[str] = []
    for method, options in attempts:
        result = linprog(
            objective,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=a_eq,
            b_eq=b_eq,
            bounds=bounds,
            method=method,
            options=options,
        )
        if result.success:
            return result
        messages.append(f"{method}: {result.message}")
    raise RuntimeError("; ".join(messages))


def solve_fixed_arm_highs(
    task: np.ndarray,
    tail_task: np.ndarray,
    recharge: np.ndarray,
    base_power: float,
    tail_power: float,
    base_config: pd.Series,
    tail_config: pd.Series,
    scenario: dict[str, float],
) -> tuple[dict[str, float], pd.DataFrame]:
    """Reproduce the registered fixed-power LP with SciPy HiGHS."""
    base_par = base.model_parameters(base_config)
    tail_par = base.model_parameters(tail_config)
    usable = np.where(task > base.NUMERICAL_ZERO, 0.0, recharge)
    eta_b, eta_t = np.sqrt(base_par["rte"]), np.sqrt(tail_par["rte"])
    ret_b = (1.0 - base_par["self_discharge_pct_day"] / 100.0) ** (1.0 / 24.0)
    ret_t = (1.0 - tail_par["self_discharge_pct_day"] / 100.0) ** (1.0 / 24.0)
    a_eq, b_eq = highs.build_equalities(task, eta_b, eta_t, ret_b, ret_t)
    a_ub, b_ub = highs.build_inequalities(
        usable, tail_par["charge_power_ratio"], tail_par["duration_h"]
    )
    h = highs.HOURS
    q_b, d_b, s_b, q_t, d_t, s_t, u = [i * h for i in range(7)]
    p, r = 7 * h, 7 * h + 1
    lower = np.zeros(7 * h + 2)
    upper = np.full(7 * h + 2, np.inf)
    upper[q_b:d_b] = np.minimum(usable, base_par["charge_power_ratio"] * base_power)
    upper[d_b:s_b] = np.minimum(task, base_power)
    upper[s_b:q_t] = base.USABLE_SOC_FRACTION * base_par["duration_h"] * base_power
    upper[q_t:d_t] = usable
    upper[d_t:s_t] = tail_task
    upper[u:p] = task
    lower[p] = tail_power
    upper[p] = tail_power
    bounds = np.column_stack([lower, upper])

    objective = np.zeros(7 * h + 2)
    objective[q_b:d_b] = scenario["charge_price_eur_mwh"]
    objective[d_b:s_b] = base_par["vom_eur_mwh"]
    objective[q_t:d_t] = scenario["charge_price_eur_mwh"]
    objective[d_t:s_t] = tail_par["vom_eur_mwh"]
    objective[u:p] = scenario["energy_value_eur_mwh"]
    objective[p] = 1000.0 * tail_par["fixed_cost_eur_kw_year"]
    objective[r] = 1000.0 * scenario["firm_value_eur_kw_year"]
    primary = robust_linprog(
        objective, a_ub=a_ub, b_ub=b_ub, a_eq=a_eq, b_eq=b_eq,
        bounds=bounds,
    )
    charge_objective = np.zeros_like(objective)
    charge_objective[q_b:d_b] = 1.0
    charge_objective[q_t:d_t] = 1.0
    scale = max(float(np.max(np.abs(objective))), 1.0)
    try:
        secondary = robust_linprog(
            charge_objective,
            a_ub=vstack([a_ub, csr_matrix((objective / scale).reshape(1, -1))], format="csr"),
            b_ub=np.append(b_ub, (primary.fun + highs.PRIMARY_ABS_TOL_EUR) / scale),
            a_eq=a_eq, b_eq=b_eq, bounds=bounds,
        )
        selected = secondary
        secondary_success = True
    except RuntimeError:
        # The secondary objective is only a deterministic tie breaker.  The
        # primary solution is already optimal and feasible; retaining it is
        # preferable to excluding an experimental unit when the extra
        # near-equality row is numerically degenerate.
        selected = primary
        secondary_success = False
    x = selected.x
    q_base, d_base = x[q_b:d_b], x[d_b:s_b]
    q_tail, d_tail = x[q_t:d_t], x[d_t:s_t]
    unserved = x[u:p]
    residual_peak = float(x[r])
    fixed_base = 1000.0 * base_power * base_par["fixed_cost_eur_kw_year"]
    fixed_tail = 1000.0 * tail_power * tail_par["fixed_cost_eur_kw_year"]
    summary = {
        "lexicographic_secondary_success": secondary_success,
        "tail_power_mw": tail_power,
        "tail_equivalent_full_cycles": (
            float(d_tail.sum()) / (tail_par["duration_h"] * tail_power)
            if tail_power > 0 else 0.0
        ),
        "unserved_eens_mwh": float(unserved.sum()),
        "portfolio_efc_mw": highs.equivalent_firm_capacity(task, float(unserved.sum())),
        "base_fixed_cost_eur": fixed_base,
        "tail_fixed_cost_eur": fixed_tail,
        "base_vom_cost_eur": base_par["vom_eur_mwh"] * float(d_base.sum()),
        "tail_vom_cost_eur": tail_par["vom_eur_mwh"] * float(d_tail.sum()),
        "residual_firm_cost_eur": 1000.0 * scenario["firm_value_eur_kw_year"] * residual_peak,
    }
    schedule = pd.DataFrame(
        {
            "base_charge_mw": q_base,
            "tail_charge_mw": q_tail,
            "unserved_task_mw": unserved,
        }
    )
    return summary, schedule


def event_run_index(frame: pd.DataFrame, order: list[int]) -> np.ndarray:
    daily = (
        frame.groupby("source_day_id", sort=True)
        .agg(event=("event_flag_q75", "max"), segment=("season_segment", "first"))
    )
    values: list[int] = []
    run = 0
    previous_segment = None
    for day in order:
        event = bool(daily.loc[day, "event"])
        segment = int(daily.loc[day, "segment"])
        if previous_segment is None or segment != previous_segment or not event:
            run = 0
        if event:
            run += 1
        values.extend([run if event else 0] * 24)
        previous_segment = segment
    return np.asarray(values, dtype=int)


def price_vectors(
    base_task: np.ndarray,
    recharge: np.ndarray,
    run_index: np.ndarray,
    scenario: dict[str, float],
    empirical_price: np.ndarray | None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    usable = np.where(base_task > base.NUMERICAL_ZERO, 0.0, recharge)
    maximum = float(usable.max())
    linked_charge = np.full(len(usable), CHARGE_PRICE_MAX, dtype=float)
    if maximum > 0.0:
        linked_charge = CHARGE_PRICE_MAX * (1.0 - np.clip(usable / maximum, 0.0, 1.0))
    linked_charge[usable <= base.NUMERICAL_ZERO] = CHARGE_PRICE_MAX
    energy0 = np.full(len(usable), scenario["energy_value_eur_mwh"], dtype=float)
    charge0 = np.full(len(usable), scenario["charge_price_eur_mwh"], dtype=float)
    vectors: dict[str, tuple[np.ndarray, np.ndarray]] = {"S0": (energy0, charge0)}
    for beta in BETAS:
        factor = 1.0 + beta * np.maximum(run_index - 1, 0)
        vectors[f"S1_beta_{beta:g}"] = (energy0 * factor, charge0)
    vectors["S2_surplus_linked_charge"] = (energy0, linked_charge)
    if empirical_price is not None:
        vectors["S3a_empirical_price"] = (empirical_price, charge0)
    scarcity = np.where(run_index > 0, SCARCITY_CAP, energy0)
    vectors["S4_scarcity_cap"] = (scarcity, charge0)
    return vectors


def revalue(
    summary: dict[str, object], schedule: pd.DataFrame, vectors: dict[str, tuple[np.ndarray, np.ndarray]]
) -> dict[str, float]:
    q = schedule["base_charge_mw"].to_numpy(float) + schedule["tail_charge_mw"].to_numpy(float)
    unserved = schedule["unserved_task_mw"].to_numpy(float)
    invariant = (
        float(summary["base_fixed_cost_eur"])
        + float(summary["tail_fixed_cost_eur"])
        + float(summary["base_vom_cost_eur"])
        + float(summary["tail_vom_cost_eur"])
        + float(summary["residual_firm_cost_eur"])
    )
    result: dict[str, float] = {}
    for label, (energy_price, charge_price) in vectors.items():
        energy_cost = float(np.dot(unserved, energy_price))
        charge_cost = float(np.dot(q, charge_price))
        result[f"{label}_residual_energy_cost_eur"] = energy_cost
        result[f"{label}_charge_cost_eur"] = charge_cost
        result[f"{label}_system_cost_eur"] = invariant + energy_cost + charge_cost
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--country-years", default=None)
    parser.add_argument("--chunk", type=int, choices=[1, 2, 3, 4], default=None)
    parser.add_argument("--empirical-price-predictions", type=Path, default=None)
    parser.add_argument("--prediction-shard-index", type=int, default=None)
    parser.add_argument("--prediction-shard-count", type=int, default=None)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    requested = base.parse_country_years(args.country_years)
    if args.chunk is not None:
        if requested is not None:
            raise ValueError("Use either --chunk or --country-years")
        contract = json.loads(
            (
                V46 / "portfolio_model" / "full_132cy_fixed_s20_central_chunks"
                / f"chunk{args.chunk}" / "run_contract.json"
            ).read_text(encoding="utf-8")
        )
        requested = [(str(country), int(year)) for country, year in contract["country_years"]]
    if args.prediction_shard_index is not None or args.prediction_shard_count is not None:
        if args.empirical_price_predictions is None or args.prediction_shard_index is None or args.prediction_shard_count is None:
            raise ValueError("Prediction sharding requires the prediction file, index and count")
        prediction_keys = pd.read_csv(
            args.empirical_price_predictions,
            usecols=["country", "year"],
        ).drop_duplicates().sort_values(["country", "year"])
        shards = np.array_split(prediction_keys.to_numpy(object), args.prediction_shard_count)
        requested = [(str(country), int(year)) for country, year in shards[args.prediction_shard_index - 1].tolist()]

    usecols = [
        "country", "year", "source_hour_id", "source_day_id", "season_segment",
        "event_flag_q75", "raw_gas_mean_mw", "forced_discharge_power_mw",
        base.BASE_TASK_COLUMN, base.TAIL_TASK_COLUMN, base.RECHARGE_COLUMN,
    ]
    panel = pd.read_csv(base.DEFAULT_PANEL, usecols=usecols).sort_values(
        ["country", "year", "source_hour_id"], kind="stable"
    )
    panel, country_years = base.select_panel(panel, requested, None)
    catalogue = pd.read_csv(base.DEFAULT_CATALOGUE)
    base_config, candidates = base.select_catalogue(catalogue, None)
    scenario = base.load_scenario(base.DEFAULT_SCENARIO, 20)

    empirical = None
    if args.empirical_price_predictions is not None:
        empirical = pd.read_csv(args.empirical_price_predictions)
        empirical = empirical.set_index(["country", "year", "source_hour_id"])["predicted_price_eur_mwh"]

    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for country, year in country_years:
        frame = panel[panel["country"].eq(country) & panel["year"].eq(year)].copy()
        orders, _ = base.day_orders(frame)
        original_base = frame[base.BASE_TASK_COLUMN].to_numpy(float)
        original_tail = frame[base.TAIL_TASK_COLUMN].to_numpy(float)
        original_recharge = frame[base.RECHARGE_COLUMN].to_numpy(float)
        forced_power = float(frame["forced_discharge_power_mw"].iloc[0])
        if empirical is not None and (country, year) in empirical.index.droplevel(2):
            original_empirical = empirical.loc[(country, year)].sort_index().to_numpy(float)
            if len(original_empirical) != 8760:
                original_empirical = None
        else:
            original_empirical = None

        for chronology, order in orders.items():
            base_task = base.reorder_days(original_base, order)
            tail_task = base.reorder_days(original_tail, order)
            recharge = base.reorder_days(original_recharge, order)
            run_index = event_run_index(frame, order)
            empirical_price = (
                base.reorder_days(original_empirical, order)
                if original_empirical is not None else None
            )
            vectors = price_vectors(base_task, recharge, run_index, scenario, empirical_price)
            arms: list[tuple[str, pd.Series, float]] = [
                ("base_only", base_config, 0.0),
                ("extra_lfp4_comparator", base_config, forced_power),
            ]
            arms.extend(("ldes_candidate", row, forced_power) for _, row in candidates.iterrows())
            for arm_type, config, upper in arms:
                summary, schedule = solve_fixed_arm_highs(
                    base_task, tail_task, recharge, forced_power, upper,
                    base_config, config, scenario,
                )
                identity = base.config_identity(config)
                if arm_type == "base_only":
                    identity.update(config_id="base_only_lfp4", tech_id="none", duration_h=0.0)
                rows.append(
                    {
                        "country": country, "year": year, "chronology": chronology,
                        "arm_type": arm_type, **identity,
                        "tail_power_mw": summary["tail_power_mw"],
                        "tail_equivalent_full_cycles": summary["tail_equivalent_full_cycles"],
                        "unserved_eens_mwh": summary["unserved_eens_mwh"],
                        "portfolio_efc_mw": summary["portfolio_efc_mw"],
                        "lexicographic_secondary_success": summary["lexicographic_secondary_success"],
                        "event_run_max_days": int(run_index.max()),
                        "empirical_price_available": empirical_price is not None,
                        **revalue(summary, schedule, vectors),
                    }
                )
        pd.DataFrame(rows).to_csv(
            args.output / "price_channel_arm_revaluation_checkpoint.csv.gz",
            index=False, compression={"method": "gzip", "mtime": 0},
        )
        print(f"completed {country}-{year}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(
        args.output / "price_channel_arm_revaluation.csv.gz",
        index=False, compression={"method": "gzip", "mtime": 0},
    )
    (args.output / "run_contract.json").write_text(
        json.dumps(
            {
                "country_years": country_years,
                "n_rows": len(out),
                "scenarios": [c.removesuffix("_system_cost_eur") for c in out if c.endswith("_system_cost_eur")],
                "betas": BETAS,
                "charge_price_max_eur_mwh": CHARGE_PRICE_MAX,
                "scarcity_cap_eur_mwh": SCARCITY_CAP,
                "dispatch_reoptimized": False,
                "elapsed_seconds": time.perf_counter() - started,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
