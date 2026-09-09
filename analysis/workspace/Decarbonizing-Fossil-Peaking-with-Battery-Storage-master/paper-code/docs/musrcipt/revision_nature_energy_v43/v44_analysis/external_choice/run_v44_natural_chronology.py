#!/usr/bin/env python3
"""Run the frozen v43 portfolio model on the observed (unreordered) chronology.

The v43 runner deliberately exposes only the dispersed and clustered day
orders.  This thin, auditable adapter imports that runner unchanged, adds the
identity day order as ``natural``, and then delegates to its command-line
entry point.  All optimization equations, costs and accounting therefore
remain byte-for-byte those of the v43 model.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
RUNNER = HERE.parents[1] / "portfolio_model" / "run_v42_europe_portfolio_gurobi.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("v43_portfolio_runner", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    runner = load_runner()
    original_day_orders = runner.day_orders

    def day_orders_with_natural(frame):
        orders, statistics = original_day_orders(frame)
        natural = list(range(365))
        orders = {**orders, "natural": natural}
        return orders, statistics

    runner.EXPECTED_CHRONOLOGIES = (
        "dispersed",
        "clustered",
        "natural",
    )
    runner.day_orders = day_orders_with_natural

    def load_scenario_for_natural(path, scenario_id):
        frame = pd.read_csv(path)
        subset = frame.loc[frame["scenario_id"].eq(scenario_id)]
        columns = [
            "energy_value_eur_mwh",
            "firm_value_eur_kw_year",
            "charge_price_eur_mwh",
        ]
        unique = subset[columns].drop_duplicates()
        if len(unique) != 1:
            raise RuntimeError(
                f"Scenario {scenario_id} parameters are not unique"
            )
        return {column: float(unique.iloc[0][column]) for column in columns}

    runner.load_scenario = load_scenario_for_natural
    runner.main()


if __name__ == "__main__":
    main()
