#!/usr/bin/env python3
"""Verify the reduced PECD LP against the registered v54 two-stage LP.

Run this script with the project Python 3.9 environment, which contains the
registered runner's gurobipy import.  No Gurobi solve is requested: the
registered SciPy/HiGHS reconstruction is used as the reference formulation.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
V54 = HERE.parent
REFERENCE_RUNNER = V54 / "joint_choice" / "run_natural_caes24_s20_highs.py"
PREPARED = HERE / "prepared" / "DE_pecd_1982_2016_standardized.npz"
REDUCED_RESULTS = HERE / "results" / "pecd_35y_arm_results_DE.csv.gz"
OUTPUT = HERE / "results" / "pecd_35y_reduced_model_equivalence_qa.json"

CAES24_FIXED_COST_EUR_KW_YEAR = 56.87352019939117
WEATHER_YEAR = 1996
ABSOLUTE_SYSTEM_COST_TOLERANCE_EUR = 0.011
PAIR_VALUE_TOLERANCE_EUR_KW_YEAR = 1.0e-7


def load_reference_module():
    spec = importlib.util.spec_from_file_location("registered_v54_highs", REFERENCE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import the registered v54 HiGHS runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = load_reference_module()
    _, _, base, caes, scenario = module.inputs()
    caes = dict(caes)
    caes["fixed_cost_eur_kw_year"] = CAES24_FIXED_COST_EUR_KW_YEAR
    prepared = np.load(PREPARED)
    index = int(np.flatnonzero(prepared["years"].astype(int) == WEATHER_YEAR)[0])
    power = float(prepared["fixed_common_power_mw"][0])
    task = prepared["base_task_mw"][index]
    tail_task = prepared["tail_task_mw"][index]
    recharge = prepared["recharge_mw"][index]
    reduced = pd.read_csv(REDUCED_RESULTS)
    reduced = reduced[reduced.weather_year.eq(WEATHER_YEAR)].set_index("config_id")

    rows = []
    reference_costs = {}
    for config_id, technology in [("li_lfp_4h", base), ("caes_24h", caes)]:
        answer = module.solve_fixed_arm(task, tail_task, recharge, power, base, technology, scenario)
        reference_cost = float(answer["system_cost_eur"])
        reduced_cost = float(reduced.loc[config_id, "system_cost_eur"])
        reference_costs[config_id] = reference_cost
        rows.append(
            {
                "config_id": config_id,
                "registered_two_stage_system_cost_eur": reference_cost,
                "reduced_primary_system_cost_eur": reduced_cost,
                "absolute_difference_eur": abs(reference_cost - reduced_cost),
                "relative_difference": abs(reference_cost - reduced_cost) / abs(reference_cost),
            }
        )

    reference_pair_value = (
        reference_costs["li_lfp_4h"] - reference_costs["caes_24h"]
    ) / (1000.0 * power)
    reduced_pair_value = (
        float(reduced.loc["li_lfp_4h", "system_cost_eur"])
        - float(reduced.loc["caes_24h", "system_cost_eur"])
    ) / (1000.0 * power)
    maximum_cost_difference = max(row["absolute_difference_eur"] for row in rows)
    pair_difference = abs(reference_pair_value - reduced_pair_value)
    qa = {
        "status": "pass",
        "country": "DE",
        "weather_year": WEATHER_YEAR,
        "registered_formulation": str(REFERENCE_RUNNER.resolve()),
        "comparison_rows": rows,
        "registered_pair_value_eur_kw_year": reference_pair_value,
        "reduced_pair_value_eur_kw_year": reduced_pair_value,
        "absolute_pair_value_difference_eur_kw_year": pair_difference,
        "maximum_absolute_system_cost_difference_eur": maximum_cost_difference,
        "absolute_system_cost_tolerance_eur": ABSOLUTE_SYSTEM_COST_TOLERANCE_EUR,
        "pair_value_tolerance_eur_kw_year": PAIR_VALUE_TOLERANCE_EUR_KW_YEAR,
    }
    if maximum_cost_difference > ABSOLUTE_SYSTEM_COST_TOLERANCE_EUR or pair_difference > PAIR_VALUE_TOLERANCE_EUR_KW_YEAR:
        qa["status"] = "fail"
    OUTPUT.write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(qa, indent=2))
    if qa["status"] != "pass":
        raise RuntimeError("Reduced-model equivalence QA failed")


if __name__ == "__main__":
    main()
