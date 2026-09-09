#!/usr/bin/env python3
"""Merge the four non-overlapping fixed-scale S20 production chunks.

This utility is deliberately narrow: it refuses partial or duplicate
country-year coverage and reproduces the compressed CSV outputs expected by
the v42 postprocessor.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
CHUNK_ROOT = HERE / "full_132cy_fixed_s20_central_chunks"
OUTPUT = HERE / "full_132cy_fixed_s20_central"
CHUNKS = [CHUNK_ROOT / f"chunk{i}" for i in range(1, 5)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(
        path,
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )


def load_concat(filename: str) -> pd.DataFrame:
    paths = [chunk / filename for chunk in CHUNKS]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing chunk outputs: {missing}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    arm = load_concat("portfolio_arm_results.csv.gz")
    comparison = load_concat("ldes_vs_extra_lfp4_results.csv.gz")
    paired = load_concat("clustered_vs_dispersed_paired_effects.csv")
    chronology = load_concat("chronology_conservation.csv")

    country_years = (
        arm[["country", "year"]]
        .drop_duplicates()
        .sort_values(["country", "year"])
        .reset_index(drop=True)
    )
    if len(country_years) != 132:
        raise RuntimeError(
            f"Expected 132 country-years, obtained {len(country_years)}"
        )

    arm_keys = ["country", "year", "chronology", "config_id"]
    comparison_keys = ["country", "year", "chronology", "config_id"]
    paired_keys = ["country", "year", "config_id"]
    chronology_keys = ["country", "year", "chronology"]
    for label, frame, keys, expected in [
        ("arm", arm, arm_keys, 132 * 2 * 51),
        ("comparison", comparison, comparison_keys, 132 * 2 * 49),
        ("paired", paired, paired_keys, 132 * 51),
        ("chronology", chronology, chronology_keys, 132 * 2),
    ]:
        if len(frame) != expected:
            raise RuntimeError(
                f"{label}: expected {expected} rows, obtained {len(frame)}"
            )
        if frame.duplicated(keys).any():
            raise RuntimeError(f"{label}: duplicate keys detected")

    if set(arm["solver_status_name"].astype(str)) != {"optimal"}:
        raise RuntimeError("Not all merged arm solves are optimal")

    arm = arm.sort_values(arm_keys).reset_index(drop=True)
    comparison = comparison.sort_values(comparison_keys).reset_index(drop=True)
    paired = paired.sort_values(paired_keys).reset_index(drop=True)
    chronology = chronology.sort_values(chronology_keys).reset_index(drop=True)

    arm_path = OUTPUT / "portfolio_arm_results.csv.gz"
    comparison_path = OUTPUT / "ldes_vs_extra_lfp4_results.csv.gz"
    paired_path = OUTPUT / "clustered_vs_dispersed_paired_effects.csv"
    chronology_path = OUTPUT / "chronology_conservation.csv"
    write_csv_gz(arm, arm_path)
    write_csv_gz(comparison, comparison_path)
    paired.to_csv(paired_path, index=False)
    chronology.to_csv(chronology_path, index=False)

    contracts = [json.loads((chunk / "run_contract.json").read_text()) for chunk in CHUNKS]
    scenario_ids = {item["scenario_id"] for item in contracts}
    tail_modes = {item["tail_power_mode"] for item in contracts}
    recharge_multipliers = {
        item.get("recharge_multiplier", 1.0) for item in contracts
    }
    if scenario_ids != {20} or tail_modes != {"fixed"}:
        raise RuntimeError("Chunk contracts do not define fixed-scale S20")
    if recharge_multipliers != {1.0}:
        raise RuntimeError("Unexpected recharge multiplier in central chunks")

    merged_contract = dict(contracts[0])
    merged_contract["country_years"] = [
        [row.country, int(row.year)]
        for row in country_years.itertuples(index=False)
    ]
    merged_contract["merged_from_chunks"] = [str(path.resolve()) for path in CHUNKS]
    merged_contract["chunk_run_contract_sha256"] = {
        path.name: sha256(path / "run_contract.json") for path in CHUNKS
    }
    merged_contract["merge_script"] = str(Path(__file__).resolve())
    merged_contract["merge_script_sha256"] = sha256(Path(__file__).resolve())
    (OUTPUT / "run_contract.json").write_text(
        json.dumps(merged_contract, ensure_ascii=False, indent=2) + "\n"
    )

    maximum_base_soc_residual = float(
        arm["base_soc_cyclic_residual_mwh"].abs().max()
    )
    maximum_tail_soc_residual = float(
        arm["tail_soc_cyclic_residual_mwh"].abs().max()
    )
    maximum_accounting_residual = float(
        (
            arm["system_cost_eur"]
            - arm["residual_energy_cost_eur"]
            - arm["residual_firm_cost_eur"]
            - arm["storage_cost_eur"]
        )
        .abs()
        .max()
    )
    qa = {
        "status": "pass",
        "evidence_status": "full_132_country_year_manuscript_evidence",
        "country_year_count": 132,
        "arm_row_count": len(arm),
        "comparison_row_count": len(comparison),
        "paired_row_count": len(paired),
        "chronology_row_count": len(chronology),
        "all_arm_statuses_optimal": True,
        "maximum_task_balance_residual_mw": float(
            arm["maximum_task_balance_residual_mw"].abs().max()
        ),
        "maximum_shared_recharge_violation_mw": float(
            arm["maximum_shared_recharge_violation_mw"].clip(lower=0).max()
        ),
        "maximum_base_soc_balance_residual_mwh": maximum_base_soc_residual,
        "maximum_tail_soc_balance_residual_mwh": maximum_tail_soc_residual,
        "maximum_cyclic_soc_residual_mwh": max(
            maximum_base_soc_residual, maximum_tail_soc_residual
        ),
        "maximum_system_cost_identity_error_eur": (
            maximum_accounting_residual
        ),
        "maximum_accounting_identity_residual_eur": (
            maximum_accounting_residual
        ),
        "output_sha256": {
            path.name: sha256(path)
            for path in [arm_path, comparison_path, paired_path, chronology_path]
        },
    }
    (OUTPUT / "qa_summary.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
