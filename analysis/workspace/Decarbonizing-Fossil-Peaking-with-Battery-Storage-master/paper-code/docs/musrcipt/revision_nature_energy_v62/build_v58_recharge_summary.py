#!/usr/bin/env python3
"""Legacy EFC revaluation retained for audit only.

Do not use this script for v62 headline results. The objective-aligned
replacement is ``build_v62_objective_aligned_recharge_summary.py``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
POSTPROCESS = ROOT / "cross_scale_cases" / "postprocess_cross_scale_v0_2.py"
COSTS = ROOT / "cost_and_decomposition" / "cost_catalogue_2025_2035_long.csv"
OUT = ROOT / "cross_scale_recharge_v58"
FIRM_VALUE = 103.79921785880974


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


post = load_module("v58_cross_scale_postprocess", POSTPROCESS)


def interval(frame: pd.DataFrame, value: str, draws: int = 10_000) -> tuple[float, float, float]:
    blocks = frame.groupby("block", as_index=False)[value].mean()[value].to_numpy(float)
    rng = np.random.default_rng(5803)
    sampled = blocks[rng.integers(0, len(blocks), size=(draws, len(blocks)))].mean(axis=1)
    return float(blocks.mean()), *np.quantile(sampled, [0.025, 0.975]).tolist()


def summarize_cross_scale(regime: str, multiplier: int, costs: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    path = OUT / f"{regime}_r{multiplier}" / "cross_scale_results.csv.gz"
    arms = pd.read_csv(path)
    valued = post.ledger(
        arms,
        costs,
        scenario="S0",
        cost_year=2035,
        firm_value=FIRM_VALUE,
    )
    keys = ["application_case", "case_id", "case_year", "chronology"]
    lfp = valued[valued.arm_type.eq("extra_lfp4_comparator")][keys + ["score_eur_kw_year"]].rename(
        columns={"score_eur_kw_year": "lfp_score"}
    )
    caes = valued[valued.config_id.eq("caes_24h")][
        keys + ["score_eur_kw_year", "tail_equivalent_full_cycles"]
    ].rename(columns={"score_eur_kw_year": "caes_score"})
    paired = caes.merge(lfp, on=keys, validate="one_to_one")
    paired["relative_value"] = paired.caes_score - paired.lfp_score
    wide = paired.pivot(
        index=["application_case", "case_id", "case_year"],
        columns="chronology",
        values=["relative_value", "tail_equivalent_full_cycles"],
    )
    wide.columns = [f"{metric}_{chronology}" for metric, chronology in wide.columns]
    wide = wide.reset_index()
    wide["chronology_effect"] = wide.relative_value_clustered - wide.relative_value_dispersed
    wide["cycle_effect"] = (
        wide.tail_equivalent_full_cycles_clustered
        - wide.tail_equivalent_full_cycles_dispersed
    )
    wide["block"] = np.where(
        wide.application_case.eq("china"),
        wide.case_id,
        wide.case_id.str.split("-").str[1],
    )
    estimate, low, high = interval(wide, "chronology_effect")
    cycle_estimate, cycle_low, cycle_high = interval(wide, "cycle_effect")
    summary = {
        "regime": regime.capitalize(),
        "recharge_multiplier": multiplier,
        "n_system_years": len(wide),
        "n_blocks": wide.block.nunique(),
        "country_block_mean_effect_eur_kw_year": estimate,
        "ci95_low": low,
        "ci95_high": high,
        "median_effect_eur_kw_year": float(wide.chronology_effect.median()),
        "country_block_mean_cycle_effect": cycle_estimate,
        "cycle_ci95_low": cycle_low,
        "cycle_ci95_high": cycle_high,
        "median_cycle_effect": float(wide.cycle_effect.median()),
    }
    wide["recharge_multiplier"] = multiplier
    return summary, wide


def main() -> None:
    costs = pd.read_csv(COSTS)
    summaries: list[dict] = []
    records: list[pd.DataFrame] = []
    for regime in ("china", "japan"):
        for multiplier in (1, 2, 4):
            summary, record = summarize_cross_scale(regime, multiplier, costs)
            summaries.append(summary)
            records.append(record)

    europe = pd.read_csv(
        ROOT / "figure_work" / "source_data" / "figure4_panel_c_recharge_multiplier_response.csv"
    )
    europe = europe[europe.config_id.eq("caes_24h")].copy()
    for row in europe.itertuples():
        summaries.append(
            {
                "regime": "Europe",
                "recharge_multiplier": int(row.recharge_quantity_multiplier),
                "n_system_years": int(row.n_country_years),
                "n_blocks": int(row.n_countries),
                "country_block_mean_effect_eur_kw_year": float(row.country_block_mean),
                "ci95_low": float(row.country_block_ci_low),
                "ci95_high": float(row.country_block_ci_high),
                "median_effect_eur_kw_year": float(row.median),
                "country_block_mean_cycle_effect": np.nan,
                "cycle_ci95_low": np.nan,
                "cycle_ci95_high": np.nan,
                "median_cycle_effect": np.nan,
            }
        )

    summary = pd.DataFrame(summaries).sort_values(["regime", "recharge_multiplier"])
    baseline = pd.read_csv(
        ROOT / "cross_scale_cases" / "summary_v0_2" / "cross_scale_efc_duration_summary.csv"
    )
    baseline = baseline[
        baseline.scenario.eq("S0")
        & baseline.cost_year.eq(2035)
        & baseline.duration_h.eq(24)
    ].set_index("application_case")
    qa_differences = {}
    for regime in ("china", "japan"):
        new = float(
            summary[
                summary.regime.eq(regime.capitalize())
                & summary.recharge_multiplier.eq(1)
            ].country_block_mean_effect_eur_kw_year.iloc[0]
        )
        old = float(baseline.loc[regime, "mean_effect_eur_kw_year"])
        qa_differences[regime] = new - old
        if abs(new - old) > 1e-9:
            raise RuntimeError(f"{regime} baseline mismatch: new={new}, old={old}")

    for regime, group in summary.groupby("regime"):
        baseline_effect = abs(
            float(group[group.recharge_multiplier.eq(1)].country_block_mean_effect_eur_kw_year.iloc[0])
        )
        index = summary.regime.eq(regime)
        summary.loc[index, "effect_retained_pct"] = (
            100
            * summary.loc[index, "country_block_mean_effect_eur_kw_year"].abs()
            / baseline_effect
        )

    OUT.mkdir(exist_ok=True)
    summary.to_csv(OUT / "three_regime_recharge_quantity_summary.csv", index=False)
    pd.concat(records, ignore_index=True).to_csv(
        OUT / "china_japan_recharge_quantity_records.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    qa = {
        "analysis": "v58_three_regime_recharge_quantity_intervention",
        "focal_comparison": "24 h A-CAES versus additional 4 h LFP",
        "recharge_multipliers": [1, 2, 4],
        "baseline_reproduction_difference_eur_kw_year": qa_differences,
        "baseline_reproduced_to_1e-9": True,
        "rows": len(summary),
    }
    (OUT / "qa.json").write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
