#!/usr/bin/env python3
"""Build technology-specific 2025, 2030 and 2035 Gurobi catalogues.

The physical parameters and configuration identities are inherited from the
frozen 53-configuration catalogue.  Only annual fixed and variable operating
costs are replaced with the source-aware calendar-year values used in v43.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
V44 = HERE.parent
V40 = V44.parent / "revision_nature_energy_v40"
BASE = V40 / "currency_work" / "configuration_catalog_53_source_aware_eur.csv"
LONG = V44 / "cost_and_decomposition" / "cost_catalogue_2025_2035_long.csv"
OUT = HERE / "catalogues"
YEARS = (2025, 2030, 2035)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(BASE)
    long = pd.read_csv(LONG)
    if len(base) != 53 or base["config_id"].duplicated().any():
        raise RuntimeError("Frozen catalogue must contain 53 unique configurations")
    if set(long["learning_year"].unique()) != set(YEARS):
        raise RuntimeError("Calendar-year cost table must contain 2025, 2030 and 2035")

    outputs: dict[str, dict[str, object]] = {}
    for year in YEARS:
        costs = long[long["learning_year"].eq(year)].copy()
        if len(costs) != 53 or costs["config_id"].duplicated().any():
            raise RuntimeError(f"{year} costs must contain 53 unique configurations")
        frame = base.drop(
            columns=[
                "annual_fixed_cost_eur_per_kw_year_source_aware",
                "variable_om_eur_mwh_source_aware",
            ]
        ).merge(
            costs[
                [
                    "config_id",
                    "annual_fixed_cost_eur_source_aware_per_kw_discharge_year",
                    "variable_om_eur_source_aware_mwh",
                ]
            ],
            on="config_id",
            how="left",
            validate="one_to_one",
        )
        frame = frame.rename(
            columns={
                "annual_fixed_cost_eur_source_aware_per_kw_discharge_year":
                    "annual_fixed_cost_eur_per_kw_year_source_aware",
                "variable_om_eur_source_aware_mwh":
                    "variable_om_eur_mwh_source_aware",
            }
        )
        if frame[
            [
                "annual_fixed_cost_eur_per_kw_year_source_aware",
                "variable_om_eur_mwh_source_aware",
            ]
        ].isna().any().any():
            raise RuntimeError(f"Missing cost values in {year} catalogue")
        frame["model_cost_year"] = year
        path = OUT / f"configuration_catalog_53_source_aware_eur_{year}.csv"
        frame.to_csv(path, index=False)
        outputs[str(year)] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "rows": len(frame),
        }

    manifest = {
        "base_catalogue": str(BASE.resolve()),
        "base_sha256": sha256(BASE),
        "calendar_cost_table": str(LONG.resolve()),
        "calendar_cost_table_sha256": sha256(LONG),
        "outputs": outputs,
    }
    (OUT / "catalogue_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
