#!/usr/bin/env python3
"""Freeze objective-aligned recharge-quantity results for three regimes.

The reported value is the same quantity minimized by the solver:
``C_sys(additional 4 h LFP) - C_sys(24 h A-CAES)``.  EENS-equivalent firm
supply remains in the arm files as an operational diagnostic only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "cross_scale_recharge_v58"


def interval(
    frame: pd.DataFrame,
    value: str,
    draws: int = 10_000,
) -> tuple[float, float, float]:
    blocks = frame.groupby("block", as_index=False)[value].mean()[value].to_numpy(float)
    rng = np.random.default_rng(5803)
    sampled = blocks[rng.integers(0, len(blocks), size=(draws, len(blocks)))].mean(axis=1)
    return float(blocks.mean()), *np.quantile(sampled, [0.025, 0.975]).tolist()


def summarize_arms(
    arms: pd.DataFrame,
    *,
    regime: str,
    multiplier: int,
    keys: list[str],
    cost_column: str,
    id_column: str,
    year_column: str,
    block_values: pd.Series,
) -> tuple[dict[str, object], pd.DataFrame]:
    comparator = arms.loc[
        arms["arm_type"].eq("extra_lfp4_comparator"),
        keys + [cost_column],
    ].rename(columns={cost_column: "comparator_system_cost_eur"})
    caes = arms.loc[
        arms["config_id"].eq("caes_24h"),
        keys + [cost_column, "tail_power_mw", "tail_equivalent_full_cycles"],
    ].copy()
    paired = caes.merge(comparator, on=keys, validate="one_to_one")
    paired["relative_value"] = (
        paired["comparator_system_cost_eur"] - paired[cost_column]
    ) / (paired["tail_power_mw"] * 1000.0)
    wide = paired.pivot(
        index=keys[:-1],
        columns="chronology",
        values=["relative_value", "tail_equivalent_full_cycles"],
    )
    wide.columns = [f"{metric}_{chronology}" for metric, chronology in wide.columns]
    wide = wide.reset_index()
    wide["chronology_effect"] = (
        wide["relative_value_clustered"] - wide["relative_value_dispersed"]
    )
    wide["cycle_effect"] = (
        wide["tail_equivalent_full_cycles_clustered"]
        - wide["tail_equivalent_full_cycles_dispersed"]
    )
    lookup_index = pd.MultiIndex.from_frame(wide[[id_column, year_column]])
    wide["block"] = block_values.reindex(lookup_index).to_numpy()
    if wide["block"].isna().any():
        raise RuntimeError(f"Missing inference block in {regime} r{multiplier}")

    estimate, low, high = interval(wide, "chronology_effect")
    cycle_estimate, cycle_low, cycle_high = interval(wide, "cycle_effect")
    summary: dict[str, object] = {
        "regime": regime,
        "recharge_multiplier": multiplier,
        "n_system_years": len(wide),
        "n_blocks": wide["block"].nunique(),
        "country_block_mean_effect_eur_kw_year": estimate,
        "ci95_low": low,
        "ci95_high": high,
        "median_effect_eur_kw_year": float(wide["chronology_effect"].median()),
        "country_block_mean_cycle_effect": cycle_estimate,
        "cycle_ci95_low": cycle_low,
        "cycle_ci95_high": cycle_high,
        "median_cycle_effect": float(wide["cycle_effect"].median()),
    }
    wide = wide.rename(columns={id_column: "case_id", year_column: "case_year"})
    wide["application_case"] = regime.lower()
    wide["recharge_multiplier"] = multiplier
    return summary, wide


def summarize_cross_scale(regime: str, multiplier: int) -> tuple[dict, pd.DataFrame]:
    arms = pd.read_csv(OUT / f"{regime}_r{multiplier}" / "cross_scale_results.csv.gz")
    ids = arms[["case_id", "case_year"]].drop_duplicates().copy()
    ids["block"] = np.where(
        regime == "china",
        ids["case_id"],
        ids["case_id"].str.split("-").str[1],
    )
    blocks = ids.set_index(["case_id", "case_year"])["block"]
    return summarize_arms(
        arms,
        regime=regime.capitalize(),
        multiplier=multiplier,
        keys=["application_case", "case_id", "case_year", "chronology"],
        cost_column="S0_system_cost_eur",
        id_column="case_id",
        year_column="case_year",
        block_values=blocks,
    )


def summarize_europe(multiplier: int) -> tuple[dict, pd.DataFrame]:
    directory = {
        1: "full_132cy_fixed_s20_central",
        2: "recharge_intervention_fixed_s20_r2",
        4: "recharge_intervention_fixed_s20_r4",
    }[multiplier]
    arms = pd.read_csv(ROOT / "portfolio_model" / directory / "portfolio_arm_results.csv.gz")
    ids = arms[["country", "year"]].drop_duplicates().copy()
    ids["block"] = ids["country"]
    blocks = ids.set_index(["country", "year"])["block"]
    return summarize_arms(
        arms,
        regime="Europe",
        multiplier=multiplier,
        keys=["country", "year", "chronology"],
        cost_column="system_cost_eur",
        id_column="country",
        year_column="year",
        block_values=blocks,
    )


def main() -> None:
    summaries: list[dict[str, object]] = []
    records: list[pd.DataFrame] = []
    for multiplier in (1, 2, 4):
        summary, record = summarize_europe(multiplier)
        summaries.append(summary)
        records.append(record)
    for regime in ("china", "japan"):
        for multiplier in (1, 2, 4):
            summary, record = summarize_cross_scale(regime, multiplier)
            summaries.append(summary)
            records.append(record)

    summary = pd.DataFrame(summaries).sort_values(["regime", "recharge_multiplier"])
    baseline_records = pd.read_csv(
        ROOT
        / "figure_work/source_data_v58/harmonized_parity/three_regime_system_year_duration_records.csv.gz"
    )
    baseline_records = baseline_records.loc[
        np.isclose(baseline_records["duration_h"], 24.0)
    ].copy()
    qa_differences: dict[str, float] = {}
    for regime in ("Europe", "China", "Japan"):
        new = float(
            summary.loc[
                summary["regime"].eq(regime)
                & summary["recharge_multiplier"].eq(1),
                "country_block_mean_effect_eur_kw_year",
            ].iloc[0]
        )
        check = baseline_records.loc[baseline_records["regime"].eq(regime)].copy()
        check["block"] = np.where(
            check["regime"].eq("Japan"),
            check["system_id"].str.split("-").str[1],
            check["system_id"],
        )
        old, _, _ = interval(check, "signed_sequencing_effect_eur_kw_year")
        qa_differences[regime.lower()] = new - old
        if abs(new - old) > 1e-9:
            raise RuntimeError(f"{regime} baseline mismatch: new={new}, old={old}")

    for regime, group in summary.groupby("regime"):
        baseline_effect = abs(
            float(
                group.loc[
                    group["recharge_multiplier"].eq(1),
                    "country_block_mean_effect_eur_kw_year",
                ].iloc[0]
            )
        )
        index = summary["regime"].eq(regime)
        summary.loc[index, "effect_retained_pct"] = (
            100
            * summary.loc[index, "country_block_mean_effect_eur_kw_year"].abs()
            / baseline_effect
        )

    OUT.mkdir(exist_ok=True)
    summary.to_csv(OUT / "three_regime_recharge_quantity_summary.csv", index=False)
    pd.concat(records, ignore_index=True).to_csv(
        OUT / "three_regime_recharge_quantity_records.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    qa = {
        "analysis": "v62_three_regime_recharge_quantity_intervention",
        "focal_comparison": "24 h A-CAES versus additional 4 h LFP",
        "primary_value_metric": "C_sys(additional LFP)-C_sys(A-CAES)",
        "efc_role": "operational diagnostic only",
        "recharge_multipliers": [1, 2, 4],
        "baseline_reproduction_difference_eur_kw_year": qa_differences,
        "baseline_reproduced_to_1e-9": True,
        "rows": len(summary),
    }
    (OUT / "qa.json").write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
