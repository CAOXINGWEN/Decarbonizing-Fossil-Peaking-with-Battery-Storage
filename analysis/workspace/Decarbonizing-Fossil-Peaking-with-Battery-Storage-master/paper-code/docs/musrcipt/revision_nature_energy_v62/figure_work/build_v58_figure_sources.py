#!/usr/bin/env python3
"""Assemble clean, traceable source tables for the v58 main figures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = HERE / "source_data"
SRC49 = HERE / "source_data_v49"
OUT = HERE / "source_data_v58"
PARITY = OUT / "harmonized_parity"


def block_bootstrap(
    data: pd.DataFrame,
    value: str,
    block: str,
    seed: int,
    draws: int = 10_000,
) -> tuple[float, float, float]:
    block_values = data.groupby(block, sort=True)[value].mean().dropna().to_numpy(float)
    if len(block_values) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(block_values), size=(draws, len(block_values)))
    boot = block_values[indices].mean(axis=1)
    return (
        float(block_values.mean()),
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
    )


def event_records() -> tuple[pd.DataFrame, pd.DataFrame]:
    europe = pd.read_csv(SRC / "figure2_panel_b_country_year_geometry.csv")
    eu = pd.DataFrame(
        {
            "regime": "Europe",
            "system_id": europe["country"],
            "observation_year": europe["year"],
            "annual_task_native": europe["annual_task_twh"],
            "annual_task_unit": "TWh",
            "event_run_days": europe["natural_longest_event_run_days"],
            "event_task_share_pct": europe["event_task_share_pct"],
        }
    )

    china = pd.read_csv(SRC49 / "figure2b_china_map.csv")
    cn = pd.DataFrame(
        {
            "regime": "China",
            "system_id": "CN-" + china["province"],
            "observation_year": 1,
            "annual_task_native": china["annual_observed_peaking_task_gwh"],
            "annual_task_unit": "GWh",
            "event_run_days": china["longest_extreme_peaking_run_days"],
            "event_task_share_pct": china["extreme_peaking_share_pct"],
        }
    )

    japan = pd.read_csv(SRC49 / "figure2c_japan_map.csv")
    jp = pd.DataFrame(
        {
            "regime": "Japan",
            "system_id": (
                "JP-" + japan["prefecture"] + "-" + japan["mesh_id"].astype(str)
            ),
            "observation_year": japan["fiscal_year"],
            "annual_task_native": japan["annual_residual_import_mwh"],
            "annual_task_unit": "MWh",
            "event_run_days": japan["longest_extreme_import_run_days"],
            "event_task_share_pct": japan["extreme_import_share_pct"],
        }
    )

    records = pd.concat([eu, cn, jp], ignore_index=True)
    records["annual_task_percentile"] = records.groupby("regime")[
        "annual_task_native"
    ].rank(pct=True) * 100

    summary_rows: list[dict[str, object]] = []
    correlations = pd.read_csv(ROOT / "sequencing_parity/three_regime_task_event_correlations.csv")
    recovery = pd.read_csv(ROOT / "sequencing_parity/three_regime_recovery_window_summary.csv")
    for regime, frame in records.groupby("regime", sort=False):
        corr = correlations.loc[correlations["regime"].eq(regime)].iloc[0]
        rec_d = recovery.loc[
            recovery["regime"].eq(regime) & recovery["chronology"].eq("dispersed")
        ].iloc[0]
        rec_c = recovery.loc[
            recovery["regime"].eq(regime) & recovery["chronology"].eq("clustered")
        ].iloc[0]
        summary_rows.append(
            {
                "regime": regime,
                "n_system_years": len(frame),
                "n_systems": frame["system_id"].nunique(),
                "event_run_min_days": frame["event_run_days"].min(),
                "event_run_median_days": frame["event_run_days"].median(),
                "event_run_max_days": frame["event_run_days"].max(),
                "spearman_rho_task_vs_event": corr[
                    "spearman_rho_annual_task_vs_longest_event"
                ],
                "median_recovery_dispersed_h": rec_d[
                    "median_of_case_median_recovery_hours"
                ],
                "median_recovery_clustered_h": rec_c[
                    "median_of_case_median_recovery_hours"
                ],
                "zero_window_clustered_pct": 100
                * rec_c["median_zero_recovery_window_share"],
            }
        )
    summary = pd.DataFrame(summary_rows)
    return records, summary


def task_contract() -> pd.DataFrame:
    europe = pd.read_csv(SRC / "figure2_panel_b_country_year_geometry.csv")
    china = pd.read_csv(SRC49 / "figure2b_china_map.csv")
    japan = pd.read_csv(SRC49 / "figure2c_japan_map.csv")
    return pd.DataFrame(
        [
            {
                "regime": "Europe",
                "spatial_unit": "15 countries",
                "system_years": 132,
                "hourly_observations": 1_156_320,
                "task_definition": "fossil-gas generation above month-hour q10",
                "recharge_definition": "synchronous wind-solar surplus",
                "annual_task_value": europe["annual_task_twh"].sum(),
                "annual_task_unit": "TWh",
                "event_task_share_pct": 100
                * europe["annual_event_tail_twh"].sum()
                / europe["annual_task_twh"].sum(),
            },
            {
                "regime": "China",
                "spatial_unit": "26 provinces",
                "system_years": 26,
                "hourly_observations": 227_760,
                "task_definition": "observed gas-peaking output",
                "recharge_definition": "hourly wind-solar curtailment",
                "annual_task_value": china["annual_observed_peaking_task_gwh"].sum()
                / 1000,
                "annual_task_unit": "TWh",
                "event_task_share_pct": 100
                * (
                    china["annual_observed_peaking_task_gwh"]
                    * china["extreme_peaking_share_pct"]
                    / 100
                ).sum()
                / china["annual_observed_peaking_task_gwh"].sum(),
            },
            {
                "regime": "Japan",
                "spatial_unit": "19 community grids",
                "system_years": 27,
                "hourly_observations": 236_520,
                "task_definition": "community grid import above month-hour q10",
                "recharge_definition": "distributed-PV reverse flow",
                "annual_task_value": japan["annual_residual_import_mwh"].sum(),
                "annual_task_unit": "MWh",
                "event_task_share_pct": 100
                * (
                    japan["annual_residual_import_mwh"]
                    * japan["extreme_import_share_pct"]
                    / 100
                ).sum()
                / japan["annual_residual_import_mwh"].sum(),
            },
        ]
    )


def value_and_cycle_records() -> tuple[pd.DataFrame, pd.DataFrame]:
    value = pd.read_csv(PARITY / "three_regime_system_year_duration_records.csv.gz")
    value = value.loc[np.isclose(value["duration_h"], 24.0)].copy()

    eu_source = pd.read_csv(SRC / "figure3_panel_d_paired_mix_advantage.csv")
    eu_cycles = pd.DataFrame(
        {
            "regime": "Europe",
            "system_id": eu_source["country"],
            "observation_year": eu_source["year"],
            "inference_block": eu_source["country"],
            "cycles_dispersed": eu_source["tail_equivalent_full_cycles_dispersed"],
            "cycles_clustered": eu_source["tail_equivalent_full_cycles_clustered"],
            "cycle_effect": eu_source[
                "clustered_minus_dispersed_tail_equivalent_full_cycles"
            ],
        }
    )

    cross = pd.read_csv(
        ROOT / "cross_scale_recharge_v58/three_regime_recharge_quantity_records.csv.gz"
    )
    cross = cross.loc[
        cross["recharge_multiplier"].eq(1)
        & cross["application_case"].isin(["china", "japan"])
    ].copy()
    cross["regime"] = cross["application_case"].map(
        {"china": "China", "japan": "Japan"}
    )
    cross["inference_block"] = np.where(
        cross["regime"].eq("Japan"),
        cross["case_id"].str.split("-").str[1],
        cross["case_id"],
    )
    cross_cycles = cross.rename(
        columns={
            "case_id": "system_id",
            "case_year": "observation_year",
            "tail_equivalent_full_cycles_dispersed": "cycles_dispersed",
            "tail_equivalent_full_cycles_clustered": "cycles_clustered",
        }
    )[
        [
            "regime",
            "system_id",
            "observation_year",
            "inference_block",
            "cycles_dispersed",
            "cycles_clustered",
            "cycle_effect",
        ]
    ]
    cycles = pd.concat([eu_cycles, cross_cycles], ignore_index=True)
    merged = value.merge(
        cycles,
        on=["regime", "system_id", "observation_year"],
        how="left",
        validate="one_to_one",
    )
    if merged["cycle_effect"].isna().any():
        missing = merged.loc[
            merged["cycle_effect"].isna(),
            ["regime", "system_id", "observation_year"],
        ]
        raise ValueError(f"Missing cycle records: {missing.to_dict('records')}")
    return value, merged


def duration_summary(records: pd.DataFrame) -> pd.DataFrame:
    work = records.copy()
    work["inference_block"] = work["system_id"]
    japan = work["regime"].eq("Japan")
    work.loc[japan, "inference_block"] = work.loc[japan, "system_id"].str.split("-").str[1]

    rows: list[dict[str, object]] = []
    for r_index, (regime, regime_data) in enumerate(work.groupby("regime", sort=False)):
        for d_index, (duration, group) in enumerate(regime_data.groupby("duration_h", sort=True)):
            point, low, high = block_bootstrap(
                group,
                "signed_sequencing_effect_eur_kw_year",
                "inference_block",
                seed=58_000 + r_index * 1_000 + d_index,
            )
            rows.append(
                {
                    "regime": regime,
                    "duration_h": duration,
                    "block_mean_effect_eur_kw_year": point,
                    "ci95_low": low,
                    "ci95_high": high,
                    "system_year_median_effect_eur_kw_year": group[
                        "signed_sequencing_effect_eur_kw_year"
                    ].median(),
                    "n_system_years": len(group),
                    "n_inference_blocks": group["inference_block"].nunique(),
                }
            )
    return pd.DataFrame(rows)


def figure1_summary(task: pd.DataFrame, parity: pd.DataFrame) -> pd.DataFrame:
    recovery = pd.read_csv(ROOT / "sequencing_parity/three_regime_recovery_window_summary.csv")
    weather = json.loads(
        (
            ROOT
            / "weather_year_parity_v42/results/pecd_v42_75y_parity_summary.json"
        ).read_text(encoding="utf-8")
    )
    rows: list[dict[str, object]] = []
    for _, row in parity.iterrows():
        rows.append(
            {
                "evidence_layer": "controlled_regime",
                "regime": row["regime"],
                "metric": "24h_absolute_shift_median",
                "value": row["median_sequencing_shift_eur_kw_year"],
                "denominator": row["n_system_years"],
                "numerator": row["above_parity_count"],
                "unit": "EUR kW^-1 yr^-1",
            }
        )
    rows.extend(
        [
            {
                "evidence_layer": "natural_record",
                "regime": "Europe-33",
                "metric": "all_weather_pairs_reaching_cost_scale",
                "value": weather["all_pair_fraction_reaching_cost_scale"],
                "denominator": weather["all_unordered_weather_pairs"],
                "numerator": weather["all_pairs_reaching_cost_scale"],
                "unit": "fraction",
            },
            {
                "evidence_layer": "natural_record",
                "regime": "Europe-33",
                "metric": "adverse_decades_erasing_learning",
                "value": weather["matched_decade_fraction_erasing_learning"],
                "denominator": weather["matched_ten_year_transitions"],
                "numerator": weather["matched_decades_erasing_learning"],
                "unit": "fraction",
            },
            {
                "evidence_layer": "cost_benchmark",
                "regime": "common",
                "metric": "2025_2035_relative_cost_movement",
                "value": weather["cost_improvement_2025_2035_eur_kw_year"],
                "denominator": np.nan,
                "numerator": np.nan,
                "unit": "EUR kW^-1 yr^-1",
            },
        ]
    )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    events, event_summary = event_records()
    task = task_contract()
    value, value_cycles = value_and_cycle_records()
    durations = duration_summary(
        pd.read_csv(PARITY / "three_regime_system_year_duration_records.csv.gz")
    )
    parity_summary = pd.read_csv(PARITY / "three_regime_24h_summary.csv")

    events.to_csv(OUT / "figure2_task_event_records.csv", index=False)
    event_summary.to_csv(OUT / "figure2_regime_summary.csv", index=False)
    task.to_csv(OUT / "figure2_task_contract.csv", index=False)
    pd.read_csv(ROOT / "sequencing_parity/three_regime_recovery_window_case_year.csv").to_csv(
        OUT / "figure3_recovery_window_records.csv", index=False
    )
    value.to_csv(OUT / "figure3_24h_value_records.csv", index=False)
    value_cycles.to_csv(OUT / "figure3_24h_value_cycle_records.csv", index=False)
    pd.read_csv(ROOT / "cross_scale_recharge_v58/three_regime_recharge_quantity_summary.csv").to_csv(
        OUT / "figure4_recharge_intervention_summary.csv", index=False
    )
    durations.to_csv(OUT / "figure5_duration_response_summary.csv", index=False)
    pd.read_csv(PARITY / "three_regime_system_duration_exposure.csv").to_csv(
        OUT / "figure5_system_duration_exposure.csv", index=False
    )
    pd.read_csv(SRC / "figure5_panel_d_cluster_signature.csv").to_csv(
        OUT / "figure5_controlled_cluster_dose.csv", index=False
    )
    pd.read_csv(SRC / "figure5_panel_e_zero_crossing_summary.csv").to_csv(
        OUT / "figure5_controlled_crossing_summary.csv", index=False
    )
    figure1_summary(task, parity_summary).to_csv(
        OUT / "figure1_evidence_summary.csv", index=False
    )

    manifest = {
        "common_value_ledger": (
            "objective-aligned S0 annual system cost; residual peak valued at "
            "103.7992178588 EUR kW^-1 yr^-1; EFC is diagnostic only"
        ),
        "cost_comparison": "2025-2035 relative annual fixed-cost movement against 4 h LFP",
        "figure2_records": len(events),
        "figure3_records": len(value_cycles),
        "figure5_duration_rows": len(durations),
        "bootstrap_draws": 10_000,
        "inference_blocks": {
            "Europe": "country",
            "China": "province",
            "Japan": "prefecture",
        },
    }
    (OUT / "v58_figure_source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(task.to_string(index=False))
    print("\n", parity_summary.to_string(index=False))


if __name__ == "__main__":
    main()
