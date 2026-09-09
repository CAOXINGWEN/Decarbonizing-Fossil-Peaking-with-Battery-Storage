#!/usr/bin/env python3
"""Audit and summarize the v62 recharge-constraint factorial experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 5803


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def interval(frame: pd.DataFrame, value: str) -> tuple[float, float, float]:
    blocks = frame.groupby("block", as_index=False)[value].mean()[value].to_numpy(float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = blocks[
        rng.integers(0, len(blocks), size=(BOOTSTRAP_DRAWS, len(blocks)))
    ].mean(axis=1)
    return float(blocks.mean()), *np.quantile(sampled, [0.025, 0.975]).tolist()


def load_records() -> tuple[pd.DataFrame, list[dict[str, object]], float]:
    records: list[pd.DataFrame] = []
    contracts: list[dict[str, object]] = []
    maximum_conservation_error = 0.0
    paths = sorted(HERE.glob("*/arm_results.csv.gz"))
    if len(paths) != 24:
        raise RuntimeError(f"Expected 24 factorial cells, found {len(paths)}")
    for path in paths:
        contract_path = path.parent / "run_contract.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        contracts.append(contract)
        if int(contract["arm_rows"]) != int(contract["expected_arm_rows"]):
            raise RuntimeError(f"Row-count failure in {path.parent.name}")
        maximum_conservation_error = max(
            maximum_conservation_error,
            abs(float(contract["maximum_usable_recharge_conservation_error_mwh"])),
        )
        arms = pd.read_csv(path)
        keys = [
            "regime", "case_id", "case_year", "chronology",
            "energy_multiplier", "charge_power_multiplier", "window_mode",
        ]
        comparator = arms.loc[
            arms["arm_type"].eq("extra_lfp4_comparator"),
            keys + ["S0_system_cost_eur"],
        ].rename(columns={"S0_system_cost_eur": "lfp4_system_cost_eur"})
        caes = arms.loc[
            arms["config_id"].eq("caes_24h"),
            keys + [
                "S0_system_cost_eur", "tail_power_mw",
                "tail_equivalent_full_cycles", "unserved_eens_mwh",
            ],
        ].rename(columns={"S0_system_cost_eur": "caes24_system_cost_eur"})
        paired = caes.merge(comparator, on=keys, validate="one_to_one")
        paired["relative_value_eur_kw_year"] = (
            paired["lfp4_system_cost_eur"] - paired["caes24_system_cost_eur"]
        ) / (paired["tail_power_mw"] * 1000.0)
        wide = paired.pivot(
            index=[key for key in keys if key != "chronology"],
            columns="chronology",
            values=["relative_value_eur_kw_year", "tail_equivalent_full_cycles"],
        )
        wide.columns = [f"{metric}_{chronology}" for metric, chronology in wide.columns]
        wide = wide.reset_index()
        wide["sequencing_effect_eur_kw_year"] = (
            wide["relative_value_eur_kw_year_clustered"]
            - wide["relative_value_eur_kw_year_dispersed"]
        )
        wide["sequencing_cycle_effect"] = (
            wide["tail_equivalent_full_cycles_clustered"]
            - wide["tail_equivalent_full_cycles_dispersed"]
        )
        wide["block"] = np.where(
            wide["regime"].eq("japan"),
            wide["case_id"].str.split("-").str[1],
            wide["case_id"],
        )
        records.append(wide)
    return pd.concat(records, ignore_index=True), contracts, maximum_conservation_error


def cell_summary(records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    factors = ["regime", "energy_multiplier", "charge_power_multiplier", "window_mode"]
    for values, group in records.groupby(factors, sort=True):
        estimate, low, high = interval(group, "sequencing_effect_eur_kw_year")
        rows.append(
            {
                **dict(zip(factors, values)),
                "n_system_years": len(group),
                "n_blocks": group["block"].nunique(),
                "block_mean_effect_eur_kw_year": estimate,
                "ci95_low": low,
                "ci95_high": high,
                "system_year_median_effect_eur_kw_year": float(
                    group["sequencing_effect_eur_kw_year"].median()
                ),
                "negative_system_years": int(
                    (group["sequencing_effect_eur_kw_year"] < -1.0e-6).sum()
                ),
                "zero_system_years": int(
                    (group["sequencing_effect_eur_kw_year"].abs() <= 1.0e-6).sum()
                ),
                "positive_system_years": int(
                    (group["sequencing_effect_eur_kw_year"] > 1.0e-6).sum()
                ),
            }
        )
    result = pd.DataFrame(rows)
    baselines = result.loc[
        result["energy_multiplier"].eq(1)
        & result["charge_power_multiplier"].eq(1)
        & result["window_mode"].eq("observed"),
        ["regime", "block_mean_effect_eur_kw_year"],
    ].rename(columns={"block_mean_effect_eur_kw_year": "baseline_effect"})
    result = result.merge(baselines, on="regime", validate="many_to_one")
    result["effect_retained_pct"] = (
        100.0 * result["block_mean_effect_eur_kw_year"] / result["baseline_effect"]
    )
    return result.drop(columns="baseline_effect").sort_values(factors)


def contrast_summary(records: pd.DataFrame) -> pd.DataFrame:
    index = ["regime", "case_id", "case_year", "block"]
    wide = records.pivot(
        index=index,
        columns=["energy_multiplier", "charge_power_multiplier", "window_mode"],
        values="sequencing_effect_eur_kw_year",
    )
    if wide.isna().any().any() or len(wide.columns) != 8:
        raise RuntimeError("The factorial contrast table is not a complete 2 x 2 x 2 design")

    def column(e: int, p: int, a: str) -> pd.Series:
        return wide[(float(e), float(p), a)]

    contrasts: dict[str, pd.Series] = {}
    contrasts["energy_4x_minus_1x"] = sum(
        column(4, p, a) - column(1, p, a)
        for p in (1, 4) for a in ("observed", "expanded")
    ) / 4.0
    contrasts["charge_power_4x_minus_1x"] = sum(
        column(e, 4, a) - column(e, 1, a)
        for e in (1, 4) for a in ("observed", "expanded")
    ) / 4.0
    contrasts["expanded_minus_observed_windows"] = sum(
        column(e, p, "expanded") - column(e, p, "observed")
        for e in (1, 4) for p in (1, 4)
    ) / 4.0
    contrasts["energy_by_power_interaction"] = sum(
        (
            column(4, 4, a) - column(1, 4, a)
            - column(4, 1, a) + column(1, 1, a)
        )
        for a in ("observed", "expanded")
    ) / 2.0
    contrasts["energy_by_window_interaction"] = sum(
        (
            column(4, p, "expanded") - column(1, p, "expanded")
            - column(4, p, "observed") + column(1, p, "observed")
        )
        for p in (1, 4)
    ) / 2.0
    contrasts["power_by_window_interaction"] = sum(
        (
            column(e, 4, "expanded") - column(e, 1, "expanded")
            - column(e, 4, "observed") + column(e, 1, "observed")
        )
        for e in (1, 4)
    ) / 2.0
    contrasts["three_way_interaction"] = (
        column(4, 4, "expanded") - column(1, 4, "expanded")
        - column(4, 1, "expanded") + column(1, 1, "expanded")
        - column(4, 4, "observed") + column(1, 4, "observed")
        + column(4, 1, "observed") - column(1, 1, "observed")
    )

    rows: list[dict[str, object]] = []
    base = wide.reset_index()[index]
    for label, values in contrasts.items():
        frame = base.copy()
        frame["contrast"] = values.to_numpy(float)
        for regime, group in frame.groupby("regime", sort=True):
            estimate, low, high = interval(group, "contrast")
            rows.append(
                {
                    "regime": regime,
                    "contrast": label,
                    "n_system_years": len(group),
                    "n_blocks": group["block"].nunique(),
                    "block_mean_contrast_eur_kw_year": estimate,
                    "ci95_low": low,
                    "ci95_high": high,
                    "system_year_median_contrast_eur_kw_year": float(
                        group["contrast"].median()
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["regime", "contrast"])


def interpretive_summary(cells: pd.DataFrame) -> pd.DataFrame:
    selections = [
        ("baseline", 1.0, 1.0, "observed"),
        ("fourfold_energy_only", 4.0, 1.0, "observed"),
        ("fourfold_charge_power_only", 1.0, 4.0, "observed"),
        ("expanded_windows_only", 1.0, 1.0, "expanded"),
        ("joint_relaxation", 4.0, 4.0, "expanded"),
    ]
    rows: list[pd.DataFrame] = []
    for label, energy, power, window in selections:
        selected = cells.loc[
            cells["energy_multiplier"].eq(energy)
            & cells["charge_power_multiplier"].eq(power)
            & cells["window_mode"].eq(window)
        ].copy()
        selected.insert(1, "reported_cell", label)
        rows.append(selected)
    return pd.concat(rows, ignore_index=True).sort_values(["regime", "reported_cell"])


def baseline_reproduction(records: pd.DataFrame) -> tuple[dict[str, float], float]:
    baseline = records.loc[
        records["energy_multiplier"].eq(1)
        & records["charge_power_multiplier"].eq(1)
        & records["window_mode"].eq("observed"),
        ["regime", "case_id", "case_year", "sequencing_effect_eur_kw_year"],
    ].copy()
    frozen = pd.read_csv(
        ROOT / "cross_scale_recharge_v58" / "three_regime_recharge_quantity_records.csv.gz"
    )
    frozen = frozen.loc[
        frozen["recharge_multiplier"].eq(1),
        ["application_case", "case_id", "case_year", "chronology_effect"],
    ].rename(
        columns={
            "application_case": "regime",
            "chronology_effect": "frozen_effect",
        }
    )
    frozen["regime"] = frozen["regime"].str.lower()
    merged = baseline.merge(
        frozen,
        on=["regime", "case_id", "case_year"],
        validate="one_to_one",
    )
    merged["difference"] = (
        merged["sequencing_effect_eur_kw_year"] - merged["frozen_effect"]
    )
    by_regime = merged.groupby("regime")["difference"].apply(
        lambda values: float(values.abs().max())
    ).to_dict()
    return by_regime, float(merged["difference"].abs().max())


def main() -> None:
    records, contracts, maximum_conservation_error = load_records()
    cells = cell_summary(records)
    contrasts = contrast_summary(records)
    interpretive = interpretive_summary(cells)
    baseline_by_regime, maximum_baseline_difference = baseline_reproduction(records)

    records_path = HERE / "factorial_system_year_records.csv.gz"
    cells_path = HERE / "factorial_cell_summary.csv"
    contrasts_path = HERE / "factorial_contrast_summary.csv"
    interpretive_path = HERE / "factorial_interpretive_summary.csv"
    records.to_csv(
        records_path, index=False, compression={"method": "gzip", "mtime": 0}
    )
    cells.to_csv(cells_path, index=False)
    contrasts.to_csv(contrasts_path, index=False)
    interpretive.to_csv(interpretive_path, index=False)

    counts = records.groupby("regime")["case_id"].count().to_dict()
    expected_counts = {"europe": 132 * 8, "china": 26 * 8, "japan": 27 * 8}
    qa = {
        "analysis": "v62_recharge_constraint_factorial",
        "status": "pass",
        "design": "2 x 2 x 2 full factorial",
        "energy_multiplier_levels": [1, 4],
        "charge_power_multiplier_levels": [1, 4],
        "window_accessibility_levels": ["observed", "expanded"],
        "expanded_window_definition": (
            "chronology-specific usable annual recharge conserved and redistributed "
            "uniformly across every non-task hour"
        ),
        "network_modelled": False,
        "price_endogeneity_modelled_in_factorial": False,
        "price_correlation_evidence": "reported separately in Supplementary Table 9",
        "n_factorial_cells": len(cells),
        "n_system_year_cell_records": len(records),
        "record_counts_by_regime": counts,
        "expected_record_counts_by_regime": expected_counts,
        "record_counts_pass": counts == expected_counts,
        "contract_count": len(contracts),
        "maximum_usable_recharge_conservation_error_mwh": maximum_conservation_error,
        "maximum_baseline_reproduction_difference_eur_kw_year": maximum_baseline_difference,
        "baseline_reproduction_difference_by_regime_eur_kw_year": baseline_by_regime,
        "baseline_reproduced_to_1e-8": maximum_baseline_difference <= 1.0e-8,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_independent_block": {
            "europe": "country", "china": "province", "japan": "prefecture"
        },
        "outputs": {
            path.name: sha256(path)
            for path in (records_path, cells_path, contrasts_path, interpretive_path)
        },
    }
    if not qa["record_counts_pass"]:
        qa["status"] = "fail"
    if maximum_conservation_error > 1.0e-6:
        qa["status"] = "fail"
    if not qa["baseline_reproduced_to_1e-8"]:
        qa["status"] = "fail"
    (HERE / "qa.json").write_text(
        json.dumps(qa, indent=2) + "\n", encoding="utf-8"
    )
    if qa["status"] != "pass":
        raise RuntimeError(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
