#!/usr/bin/env python3
"""Reoptimize fixed-power portfolios against empirical hourly price curves.

The experiment is deliberately limited to four contrasting national systems.
It tests whether the frozen-dispatch S3a result survives when the optimizer can
shift charge and discharge in response to the predicted day-ahead price path.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
V46 = HERE.parent
sys.path.insert(0, str(V46 / "portfolio_model"))
import run_v42_europe_portfolio_gurobi as base  # noqa: E402
import run_price_channel_revaluation_v0_1 as price  # noqa: E402


REPRESENTATIVE_COUNTRIES = ("AT", "DE", "DK", "FR")
PREDICTIONS = HERE / "empirical_price" / "empirical_price_predictions_main_panel.csv.gz"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "empirical_reoptimized")
    parser.add_argument("--countries", default=",".join(REPRESENTATIVE_COUNTRIES))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    countries = tuple(item.strip() for item in args.countries.split(",") if item.strip())
    predictions = pd.read_csv(PREDICTIONS)
    predictions = predictions[predictions.country.isin(countries)]
    keys = predictions[["country", "year"]].drop_duplicates().sort_values(["country", "year"])
    requested = [(str(c), int(y)) for c, y in keys.itertuples(index=False, name=None)]

    usecols = [
        "country", "year", "source_hour_id", "source_day_id", "season_segment",
        "event_flag_q75", "raw_gas_mean_mw", "forced_discharge_power_mw",
        base.BASE_TASK_COLUMN, base.TAIL_TASK_COLUMN, base.RECHARGE_COLUMN,
    ]
    panel = pd.read_csv(base.DEFAULT_PANEL, usecols=usecols).sort_values(
        ["country", "year", "source_hour_id"], kind="stable"
    )
    panel, country_years = base.select_panel(panel, requested, None)
    empirical = predictions.set_index(["country", "year", "source_hour_id"])["predicted_price_eur_mwh"]
    catalogue = pd.read_csv(base.DEFAULT_CATALOGUE)
    base_config, candidates = base.select_catalogue(catalogue, None)
    scenario = base.load_scenario(base.DEFAULT_SCENARIO, 20)
    rows = []
    started = time.perf_counter()
    for index, (country, year) in enumerate(country_years, start=1):
        frame = panel[panel.country.eq(country) & panel.year.eq(year)].copy()
        orders, _ = base.day_orders(frame)
        original_base = frame[base.BASE_TASK_COLUMN].to_numpy(float)
        original_tail = frame[base.TAIL_TASK_COLUMN].to_numpy(float)
        original_recharge = frame[base.RECHARGE_COLUMN].to_numpy(float)
        original_price = empirical.loc[(country, year)].sort_index().to_numpy(float)
        power_mw = float(frame.forced_discharge_power_mw.iloc[0])
        for chronology, order in orders.items():
            task = base.reorder_days(original_base, order)
            tail_task = base.reorder_days(original_tail, order)
            recharge = base.reorder_days(original_recharge, order)
            hourly_price = base.reorder_days(original_price, order)
            charge_price = np.full(8760, scenario["charge_price_eur_mwh"], dtype=float)
            vectors = {"S3b_empirical_reoptimized": (hourly_price, charge_price)}
            arms = [("base_only", base_config, 0.0), ("extra_lfp4_comparator", base_config, power_mw)]
            arms.extend(("ldes_candidate", row, power_mw) for _, row in candidates.iterrows())
            for arm_type, config, tail_power in arms:
                summary, schedule = price.solve_fixed_arm_highs(
                    task, tail_task, recharge, power_mw, tail_power,
                    base_config, config, scenario,
                    hourly_energy_value=hourly_price,
                    hourly_charge_price=charge_price,
                )
                identity = base.config_identity(config)
                if arm_type == "base_only":
                    identity.update(config_id="base_only_lfp4", tech_id="none", duration_h=0.0)
                rows.append({
                    "country": country, "year": year, "chronology": chronology,
                    "arm_type": arm_type, **identity, **summary,
                    **price.revalue(summary, schedule, vectors),
                })
        pd.DataFrame(rows).to_csv(
            args.output / "empirical_reoptimized_checkpoint.csv.gz", index=False,
            compression={"method": "gzip", "mtime": 0},
        )
        print(f"{index}/{len(country_years)} {country}-{year}", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(
        args.output / "empirical_reoptimized_arms.csv.gz", index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    (args.output / "run_contract.json").write_text(json.dumps({
        "countries": countries, "country_years": country_years,
        "n_country_years": len(country_years), "configuration_nodes": 49,
        "dispatch_reoptimized": True, "price_source": str(PREDICTIONS),
        "elapsed_seconds": time.perf_counter() - started,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
