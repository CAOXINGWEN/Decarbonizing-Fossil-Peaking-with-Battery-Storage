#!/usr/bin/env python3
"""Build comparable EFC-value ledgers for China and Japan applications.

The physical schedules are solved once.  Technology-year costs, hourly price
channels and firm-supply values are then revalued on those schedules.  This
keeps the mechanism experiment separate from the economic visibility test.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PRICE_POST = HERE.parent / "price_channel" / "postprocess_price_channels_v0_1.py"
COSTS = HERE.parent / "cost_and_decomposition" / "cost_catalogue_2025_2035_long.csv"
TIE_TOL = 0.01
CENTRAL_FIRM_VALUE = 103.79921785880974


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


post = load_module("cross_scale_price_post_v02", PRICE_POST)


def block_bootstrap(frame: pd.DataFrame, value: str, draws: int = 10000) -> tuple[float, float, float]:
    blocks = frame.groupby("block", as_index=False)[value].mean()[value].to_numpy(float)
    rng = np.random.default_rng(4702)
    estimates = blocks[rng.integers(0, len(blocks), size=(draws, len(blocks)))].mean(axis=1)
    return float(blocks.mean()), *np.quantile(estimates, [0.025, 0.975]).tolist()


def add_blocks(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["block"] = np.where(
        work.application_case.eq("china"),
        work.case_id,
        work.case_id.str.split("-").str[1],
    )
    return work


def ledger(
    arms: pd.DataFrame,
    costs: pd.DataFrame,
    *,
    scenario: str,
    cost_year: int,
    firm_value: float,
    ldes_cost_multiplier: float = 1.0,
) -> pd.DataFrame:
    work = arms.copy()
    cost = costs.loc[costs.learning_year.eq(cost_year)].set_index("config_id")[
        "annual_fixed_cost_eur_source_aware_per_kw_discharge_year"
    ]
    lfp_rate = float(cost.loc["li_lfp_4h"])
    group = ["application_case", "case_id", "case_year", "chronology"]
    work["common_power_mw"] = work.groupby(group, sort=False).tail_power_mw.transform("max")
    work["base_fixed_revalued_eur"] = work.common_power_mw * 1000.0 * lfp_rate
    tail_rate = work.config_id.map(cost).fillna(0.0)
    tail_rate = np.where(work.arm_type.eq("base_only"), 0.0, tail_rate)
    tail_rate = np.where(work.arm_type.eq("ldes_candidate"), tail_rate * ldes_cost_multiplier, tail_rate)
    work["tail_fixed_revalued_eur"] = work.tail_power_mw * 1000.0 * tail_rate
    work["efc_system_cost_eur"] = (
        work.base_fixed_revalued_eur
        + work.tail_fixed_revalued_eur
        + work.base_vom_cost_eur
        + work.tail_vom_cost_eur
        + work[f"{scenario}_charge_cost_eur"]
        + work[f"{scenario}_residual_energy_cost_eur"]
        - 1000.0 * firm_value * work.portfolio_efc_mw
    )
    base_cost = work.groupby(group, sort=False).efc_system_cost_eur.transform(
        lambda x: x.loc[work.loc[x.index, "arm_type"].eq("base_only")].iloc[0]
    )
    work["score_eur_kw_year"] = (base_cost - work.efc_system_cost_eur) / (work.common_power_mw * 1000.0)
    work["scenario"] = scenario
    work["cost_year"] = cost_year
    work["firm_value_eur_kw_year"] = firm_value
    work["ldes_cost_multiplier"] = ldes_cost_multiplier
    return work


def summarize_one(valued: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = ["application_case", "case_id", "case_year", "chronology"]
    cells, envelope, valleys = [], [], []
    for key, group in valued.groupby(keys, sort=False):
        lfp = group[group.arm_type.eq("extra_lfp4_comparator")].iloc[0]
        candidates = group[group.arm_type.eq("ldes_candidate")].copy()
        candidates["benefit_eur_kw_year"] = candidates.score_eur_kw_year - float(lfp.score_eur_kw_year)
        cells.append(candidates)
        duration_rows = []
        for duration, duration_group in candidates.groupby("duration_h"):
            best = duration_group.sort_values("benefit_eur_kw_year", ascending=False).iloc[0]
            duration_rows.append({
                "application_case": key[0], "case_id": key[1], "case_year": key[2],
                "chronology": key[3], "duration_h": duration,
                "benefit_eur_kw_year": best.benefit_eur_kw_year,
                "dominant_config_id": best.config_id, "dominant_tech_id": best.tech_id,
            })
        envelope.extend(duration_rows)
        valleys.append({
            "application_case": key[0], "case_id": key[1], "case_year": key[2],
            "chronology": key[3], **post.strict_valley(pd.DataFrame(duration_rows)),
        })
    return pd.concat(cells, ignore_index=True), pd.DataFrame(envelope), pd.DataFrame(valleys)


def main() -> None:
    paths = sorted(HERE.glob("full_*_shard*/cross_scale_results.csv.gz"))
    paths = [path for path in paths if "smoke" not in str(path)]
    if len(paths) != 4:
        raise RuntimeError(f"Expected four complete application shards, found {len(paths)}")
    arms = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    group = ["application_case", "case_id", "case_year", "chronology"]
    if not arms.groupby(group).size().eq(51).all():
        raise RuntimeError("Incomplete 51-arm application group")
    costs = pd.read_csv(COSTS)
    scenarios = [column.removesuffix("_system_cost_eur") for column in arms if column.endswith("_system_cost_eur")]

    all_cells, all_envelopes, all_valleys = [], [], []
    for scenario in scenarios:
        for cost_year in (2025, 2030, 2035):
            valued = ledger(
                arms, costs, scenario=scenario, cost_year=cost_year,
                firm_value=CENTRAL_FIRM_VALUE,
            )
            cells, envelope, valleys = summarize_one(valued)
            for frame in (cells, envelope, valleys):
                frame["scenario"] = scenario
                frame["cost_year"] = cost_year
                frame["firm_value_eur_kw_year"] = CENTRAL_FIRM_VALUE
                frame["ldes_cost_multiplier"] = 1.0
            all_cells.append(cells)
            all_envelopes.append(envelope)
            all_valleys.append(valleys)
    cells = pd.concat(all_cells, ignore_index=True)
    envelope = pd.concat(all_envelopes, ignore_index=True)
    valleys = pd.concat(all_valleys, ignore_index=True)

    paired = envelope.pivot(
        index=["scenario", "cost_year", "application_case", "case_id", "case_year", "duration_h"],
        columns="chronology", values="benefit_eur_kw_year",
    ).reset_index()
    paired["chronology_effect_eur_kw_year"] = paired.clustered - paired.dispersed
    paired = add_blocks(paired)
    duration_summary = []
    for key, subset in paired.groupby(["scenario", "cost_year", "application_case", "duration_h"]):
        estimate, low, high = block_bootstrap(subset, "chronology_effect_eur_kw_year")
        duration_summary.append({
            "scenario": key[0], "cost_year": key[1], "application_case": key[2],
            "duration_h": key[3], "mean_effect_eur_kw_year": estimate,
            "ci95_low": low, "ci95_high": high,
        })
    duration_summary = pd.DataFrame(duration_summary)
    valley_summary = valleys.groupby(
        ["scenario", "cost_year", "application_case", "chronology"], as_index=False
    ).agg(
        n_case_years=("case_id", "size"), strict_valleys=("strict_valley", "sum"),
        mean_width_when_present=("valley_width_h", lambda x: x[x > 0].mean()),
    )

    # Economic visibility surface under the S0 operational channel.  The
    # grid is a diagnostic of where the same physical duration response
    # becomes a strict positive-negative-positive investment envelope.
    phase_rows = []
    for cost_multiplier in (1.0, 0.85, 0.70):
        for firm_value in np.arange(0.0, 625.0, 25.0):
            valued = ledger(
                arms, costs, scenario="S0", cost_year=2035,
                firm_value=float(firm_value), ldes_cost_multiplier=cost_multiplier,
            )
            _, _, records = summarize_one(valued)
            summary = records.groupby(["application_case", "chronology"], as_index=False).agg(
                n_case_years=("case_id", "size"), strict_valleys=("strict_valley", "sum"),
                mean_width_when_present=("valley_width_h", lambda x: x[x > 0].mean()),
            )
            summary["firm_value_eur_kw_year"] = firm_value
            summary["ldes_cost_multiplier"] = cost_multiplier
            phase_rows.append(summary)
    phase = pd.concat(phase_rows, ignore_index=True)

    output = HERE / "summary_v0_2"
    output.mkdir(exist_ok=True)
    arms.to_csv(output / "cross_scale_arms.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    cells.to_csv(output / "cross_scale_efc_mix_cells.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    envelope.to_csv(output / "cross_scale_efc_duration_envelope.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    paired.to_csv(output / "cross_scale_efc_paired_duration.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    valleys.to_csv(output / "cross_scale_efc_valley_records.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    duration_summary.to_csv(output / "cross_scale_efc_duration_summary.csv", index=False)
    valley_summary.to_csv(output / "cross_scale_efc_valley_summary.csv", index=False)
    phase.to_csv(output / "cross_scale_efc_phase_scan.csv", index=False)
    qa = {
        "applications": sorted(arms.application_case.unique()),
        "case_years": arms[["application_case", "case_id", "case_year"]].drop_duplicates().groupby("application_case").size().to_dict(),
        "scenarios": scenarios, "cost_years": [2025, 2030, 2035],
        "complete_arm_groups": bool(arms.groupby(group).size().eq(51).all()),
        "phase_grid_firm_values": 25, "phase_grid_cost_multipliers": [1.0, 0.85, 0.70],
    }
    (output / "qa.json").write_text(json.dumps(qa, indent=2) + "\n")


if __name__ == "__main__":
    main()
