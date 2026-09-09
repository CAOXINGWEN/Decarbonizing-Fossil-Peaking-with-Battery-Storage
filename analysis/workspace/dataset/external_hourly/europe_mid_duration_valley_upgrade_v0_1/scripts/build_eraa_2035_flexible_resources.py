#!/usr/bin/env python3
"""Build audited ERAA 2035 battery, pumped-hydro and explicit-DSR inputs.

This adapter deliberately separates three resources that must not be hidden in
net demand or generic thermal capacity:

* utility-scale batteries that are explicitly ``Available on market``;
* closed- and open-loop pumped hydro with separate charge, discharge and
  energy limits; and
* explicit DSR activation bands with their released price and daily-hour cap.

The amended Annex 3 Table 2 capacity decisions are applied after the released
National Trends tables.  Table 2 reports an aggregate 3,400 MW Spanish battery
addition and aggregate DSR additions, but not all design dimensions.  The
adapter therefore writes the missing-dimension rules into an assumptions
ledger instead of silently inventing plant records:

* the Spanish battery is the released ``Battery Storage_6`` investment class,
  hence 6 h and 92% cycle efficiency;
* a DSR addition is distributed proportionally over the receiving zone's
  released explicit-DSR bands; where no band exists, it is split over the six
  released central industrial bands (250--4,400 EUR/MWh, 6 h/day).

The tables are model inputs, not evidence that those investments will occur.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "0.1.0"
TARGET_YEAR = 2035
ROOT = Path(__file__).resolve().parents[1]
EXTRACTED = ROOT / "raw" / "eraa_2025" / "extracted"
DEFAULT_DISPATCH_ROOT = ROOT / "processed" / "eraa_2035_dispatch_parameters_v0_1"
DEFAULT_PHASE1_ROOT = ROOT / "processed" / "eraa_2035_phase1_inputs_v0_1"
DEFAULT_OUTPUT = ROOT / "processed" / "eraa_2035_flexible_resources_v0_1"

DEFAULT_BATTERY = (
    EXTRACTED
    / "ERAA_2025_DashboardRawData1"
    / "Dashboard Raw Data"
    / "Batteries additional information.csv"
)
DEFAULT_HYDRO = (
    EXTRACTED
    / "ERAA_2025_DashboardRawData1"
    / "Dashboard Raw Data"
    / "Hydro additional information.csv"
)
DEFAULT_DSR = EXTRACTED / "Other Data" / "Explicit DSR detailed.csv"
DEFAULT_INVESTMENT = (
    EXTRACTED / "Economic and technical investment parameters_Default.xlsx"
)

FINAL_VERSION = "ERAA 2025 final"
IN_MARKET = "available on market"
BATTERY_CYCLE_EFFICIENCY = 0.92
BATTERY_INITIAL_SOC = 0.50
PUMPED_HYDRO_CYCLE_EFFICIENCY = 0.75
PUMPED_HYDRO_INITIAL_SOC = 0.50
CENTRAL_DSR_BANDS = (
    (250.0, 6.0),
    (500.0, 6.0),
    (900.0, 6.0),
    (1_500.0, 6.0),
    (3_500.0, 6.0),
    (4_400.0, 6.0),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize(value: object) -> str:
    return " ".join(
        str(value)
        .strip()
        .lower()
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .split()
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.10g")


def read_nonthermal_changes(dispatch_root: Path) -> pd.DataFrame:
    path = dispatch_root / "amended_annex3_nonthermal_capacity_changes_2035.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {
        "zone",
        "pemmbd_technology",
        "decision_variable",
        "target_year",
        "capacity_change_mw",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Annex 3 table missing columns {sorted(missing)}")
    frame = frame.loc[frame["target_year"].eq(TARGET_YEAR)].copy()
    return frame


def build_batteries(
    battery_path: Path,
    annex3: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(battery_path)
    required = {
        "MARKET_NODE",
        "OP_STAT",
        "MAX LOAD CAP (MW)",
        "STORAGE CAPACITY (MWh)",
        "TARGET_YEAR",
        "data_version",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"battery table missing columns {sorted(missing)}")
    raw["normalized_status"] = raw["OP_STAT"].map(normalize)
    selected = raw.loc[
        raw["TARGET_YEAR"].eq(TARGET_YEAR)
        & raw["data_version"].eq(FINAL_VERSION)
        & raw["normalized_status"].eq(IN_MARKET)
    ].copy()
    selected["discharge_power_mw"] = pd.to_numeric(
        selected["MAX LOAD CAP (MW)"], errors="raise"
    )
    selected["energy_capacity_mwh"] = pd.to_numeric(
        selected["STORAGE CAPACITY (MWh)"], errors="raise"
    )
    base = (
        selected.groupby("MARKET_NODE", as_index=False)
        .agg(
            base_discharge_power_mw=("discharge_power_mw", "sum"),
            base_energy_capacity_mwh=("energy_capacity_mwh", "sum"),
        )
        .rename(columns={"MARKET_NODE": "zone"})
    )

    additions = annex3.loc[
        annex3["pemmbd_technology"].eq("Battery")
        & annex3["decision_variable"].eq("Expansion")
    ].copy()
    addition = (
        additions.groupby("zone", as_index=False)["capacity_change_mw"]
        .sum()
        .rename(columns={"capacity_change_mw": "annex3_addition_power_mw"})
    )
    battery = base.merge(addition, on="zone", how="outer").fillna(0.0)
    battery["annex3_addition_duration_h"] = np.where(
        battery["annex3_addition_power_mw"].gt(0), 6.0, 0.0
    )
    battery["annex3_addition_energy_mwh"] = (
        battery["annex3_addition_power_mw"]
        * battery["annex3_addition_duration_h"]
    )
    battery["charge_power_mw"] = (
        battery["base_discharge_power_mw"]
        + battery["annex3_addition_power_mw"]
    )
    battery["discharge_power_mw"] = battery["charge_power_mw"]
    battery["energy_capacity_mwh"] = (
        battery["base_energy_capacity_mwh"]
        + battery["annex3_addition_energy_mwh"]
    )
    battery["charge_efficiency"] = BATTERY_CYCLE_EFFICIENCY
    battery["discharge_efficiency"] = 1.0
    battery["initial_soc_fraction"] = BATTERY_INITIAL_SOC
    battery["terminal_soc_fraction"] = BATTERY_INITIAL_SOC
    battery["source_status"] = (
        "ERAA_2025_final_available_on_market_plus_amended_Annex3"
    )
    battery = battery.sort_values("zone").reset_index(drop=True)

    excluded = raw.loc[
        raw["TARGET_YEAR"].eq(TARGET_YEAR)
        & raw["data_version"].eq(FINAL_VERSION)
        & ~raw["normalized_status"].eq(IN_MARKET)
    ].copy()
    excluded = (
        excluded.groupby(["OP_STAT", "normalized_status"], as_index=False)
        .agg(
            rows=("MARKET_NODE", "size"),
            power_mw=("MAX LOAD CAP (MW)", "sum"),
            energy_mwh=("STORAGE CAPACITY (MWh)", "sum"),
        )
        .sort_values("normalized_status")
    )
    excluded["reason"] = np.where(
        excluded["normalized_status"].str.contains("pv/battery", regex=False),
        "BTM battery is already represented in released net-demand profiles",
        "not released as an in-market utility battery",
    )
    return battery, excluded


def build_pumped_hydro(
    hydro_path: Path,
    phase1_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(hydro_path)
    required = {
        "MARKET_NODE",
        "PEMMDB_PLANT_TYPE",
        "MAX PUMPING CAP (MW)",
        "Storage Capacity [TWh]",
        "TARGET_YEAR",
        "data_version",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"hydro table missing columns {sorted(missing)}")
    raw = raw.loc[
        raw["TARGET_YEAR"].eq(TARGET_YEAR)
        & raw["data_version"].eq(FINAL_VERSION)
    ].copy()
    raw["plant_type"] = raw["PEMMDB_PLANT_TYPE"].map(normalize)
    raw = raw.loc[
        raw["plant_type"].isin(
            {"closed loop pumping", "open loop pumping"}
        )
    ].copy()

    capacity_path = phase1_root / "capacity_2035_final.csv"
    if not capacity_path.exists():
        raise FileNotFoundError(capacity_path)
    capacities = pd.read_csv(capacity_path)
    capacities = capacities.loc[
        capacities["data_version"].eq(FINAL_VERSION)
        & capacities["Target year"].eq(TARGET_YEAR)
        & capacities["Technology"].isin(
            ["Closed loop pumping", "Open loop pumping"]
        )
        & capacities["Operational_Status"].map(normalize).eq(IN_MARKET)
    ].copy()
    discharge = (
        capacities.groupby(["Market_Node", "Technology"], as_index=False)[
            "Value"
        ]
        .sum()
        .rename(
            columns={
                "Market_Node": "zone",
                "Technology": "plant_type_display",
                "Value": "discharge_power_mw",
            }
        )
    )
    discharge["plant_type"] = discharge["plant_type_display"].map(normalize)
    pump = raw.rename(
        columns={
            "MARKET_NODE": "zone",
            "MAX PUMPING CAP (MW)": "charge_power_mw",
            "Storage Capacity [TWh]": "storage_capacity_twh",
        }
    )[
        ["zone", "plant_type", "charge_power_mw", "storage_capacity_twh"]
    ].copy()
    pump["charge_power_mw"] = pd.to_numeric(
        pump["charge_power_mw"], errors="raise"
    )
    pump["storage_capacity_twh"] = pd.to_numeric(
        pump["storage_capacity_twh"], errors="coerce"
    ).fillna(0.0)
    pump = (
        pump.groupby(["zone", "plant_type"], as_index=False)
        .agg(
            charge_power_mw=("charge_power_mw", "sum"),
            storage_capacity_twh=("storage_capacity_twh", "sum"),
        )
    )
    psp = discharge.merge(
        pump, on=["zone", "plant_type"], how="outer", validate="one_to_one"
    ).fillna(0.0)
    psp["energy_capacity_mwh"] = psp["storage_capacity_twh"] * 1_000_000.0
    psp["charge_efficiency"] = PUMPED_HYDRO_CYCLE_EFFICIENCY
    psp["discharge_efficiency"] = 1.0
    psp["initial_soc_fraction"] = PUMPED_HYDRO_INITIAL_SOC
    psp["terminal_soc_fraction"] = PUMPED_HYDRO_INITIAL_SOC
    psp["natural_inflow_applicable"] = psp["plant_type"].eq(
        "open loop pumping"
    )
    psp["source_status"] = "ERAA_2025_final"
    psp = psp[
        [
            "zone",
            "plant_type",
            "charge_power_mw",
            "discharge_power_mw",
            "energy_capacity_mwh",
            "charge_efficiency",
            "discharge_efficiency",
            "initial_soc_fraction",
            "terminal_soc_fraction",
            "natural_inflow_applicable",
            "source_status",
        ]
    ].sort_values(["zone", "plant_type"]).reset_index(drop=True)

    audit = psp.copy()
    audit["usable_for_dispatch"] = (
        audit["charge_power_mw"].gt(0)
        & audit["discharge_power_mw"].gt(0)
        & audit["energy_capacity_mwh"].gt(0)
    )
    audit["issue"] = np.select(
        [
            audit["charge_power_mw"].le(0),
            audit["discharge_power_mw"].le(0),
            audit["energy_capacity_mwh"].le(0),
        ],
        [
            "missing_or_zero_released_pumping_power",
            "missing_or_zero_released_turbine_power",
            "missing_or_zero_released_storage_energy",
        ],
        default="none",
    )
    return psp, audit


def build_dsr(
    dsr_path: Path,
    annex3: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(dsr_path)
    required = {
        "MARKET_NODE",
        "DA activation price (EUR/MWh)",
        "Max hours dispatched per day [h]",
        "MAX CAP (MW)",
        "TARGET_YEAR",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"DSR table missing columns {sorted(missing)}")
    base = raw.loc[raw["TARGET_YEAR"].eq(TARGET_YEAR)].copy()
    base = base.rename(
        columns={
            "MARKET_NODE": "zone",
            "DA activation price (EUR/MWh)": "activation_price_eur_mwh",
            "Max hours dispatched per day [h]": "max_hours_per_day",
            "MAX CAP (MW)": "capacity_mw",
        }
    )
    base["capacity_component"] = "released_2035_explicit_dsr"
    base["allocation_rule"] = "released_band"

    additions = annex3.loc[
        annex3["pemmbd_technology"].eq("DSR")
        & annex3["decision_variable"].eq("Expansion")
    ]
    addition_by_zone = additions.groupby("zone")["capacity_change_mw"].sum()
    addition_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for zone, addition_mw in addition_by_zone.items():
        zone_base = base.loc[base["zone"].eq(zone)].copy()
        total = float(zone_base["capacity_mw"].sum())
        if total > 0:
            shares = zone_base["capacity_mw"].to_numpy(dtype=float) / total
            for (_, row), share in zip(zone_base.iterrows(), shares):
                addition_rows.append(
                    {
                        "zone": zone,
                        "activation_price_eur_mwh": float(
                            row["activation_price_eur_mwh"]
                        ),
                        "max_hours_per_day": float(row["max_hours_per_day"]),
                        "capacity_mw": float(addition_mw) * float(share),
                        "TARGET_YEAR": TARGET_YEAR,
                        "Country": row.get("Country", str(zone)[:2]),
                        "capacity_component": "amended_Annex3_expansion",
                        "allocation_rule": (
                            "proportional_to_zone_released_explicit_DSR_bands"
                        ),
                    }
                )
            rule = "proportional_to_zone_released_explicit_DSR_bands"
        else:
            for price, daily_hours in CENTRAL_DSR_BANDS:
                addition_rows.append(
                    {
                        "zone": zone,
                        "activation_price_eur_mwh": price,
                        "max_hours_per_day": daily_hours,
                        "capacity_mw": float(addition_mw)
                        / len(CENTRAL_DSR_BANDS),
                        "TARGET_YEAR": TARGET_YEAR,
                        "Country": str(zone)[:2],
                        "capacity_component": "amended_Annex3_expansion",
                        "allocation_rule": (
                            "equal_split_over_released_central_industrial_bands"
                        ),
                    }
                )
            rule = "equal_split_over_released_central_industrial_bands"
        audit_rows.append(
            {
                "zone": zone,
                "annex3_expansion_mw": float(addition_mw),
                "released_zone_base_mw": total,
                "allocation_rule": rule,
            }
        )
    additions_frame = pd.DataFrame(addition_rows)
    columns = [
        "zone",
        "activation_price_eur_mwh",
        "max_hours_per_day",
        "capacity_mw",
        "TARGET_YEAR",
        "Country",
        "capacity_component",
        "allocation_rule",
    ]
    dsr = pd.concat([base[columns], additions_frame[columns]], ignore_index=True)
    dsr["band_id"] = (
        dsr.groupby("zone").cumcount().add(1).map(lambda value: f"band_{value:02d}")
    )
    dsr["source_status"] = np.where(
        dsr["capacity_component"].eq("released_2035_explicit_dsr"),
        "released",
        "derived_from_official_aggregate_with_explicit_allocation_rule",
    )
    dsr = dsr.sort_values(
        ["zone", "activation_price_eur_mwh", "max_hours_per_day", "band_id"]
    ).reset_index(drop=True)
    return dsr, pd.DataFrame(audit_rows).sort_values("zone").reset_index(drop=True)


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists():
        if not args.force:
            raise FileExistsError(
                f"output exists; choose a new directory or pass --force: {output}"
            )
        shutil.rmtree(output)
    output.mkdir(parents=True)

    annex3 = read_nonthermal_changes(args.dispatch_root.resolve())
    batteries, battery_exclusions = build_batteries(
        args.battery.resolve(), annex3
    )
    psp, psp_audit = build_pumped_hydro(
        args.hydro.resolve(), args.phase1_root.resolve()
    )
    dsr, dsr_audit = build_dsr(args.dsr.resolve(), annex3)

    write_csv(batteries, output / "existing_utility_battery_2035.csv")
    write_csv(
        battery_exclusions,
        output / "battery_status_exclusions_2035.csv",
    )
    write_csv(psp, output / "existing_pumped_hydro_2035.csv")
    write_csv(psp_audit, output / "existing_pumped_hydro_2035_audit.csv")
    write_csv(dsr, output / "explicit_dsr_bands_2035.csv")
    write_csv(dsr_audit, output / "explicit_dsr_annex3_allocation_audit.csv")

    assumptions = pd.DataFrame(
        [
            {
                "parameter": "utility_battery_cycle_efficiency",
                "value": BATTERY_CYCLE_EFFICIENCY,
                "unit": "fraction",
                "basis": (
                    "ERAA Annex 2 default; represented on charging, with "
                    "discharge efficiency equal to one"
                ),
            },
            {
                "parameter": "utility_battery_initial_and_terminal_soc",
                "value": BATTERY_INITIAL_SOC,
                "unit": "fraction",
                "basis": "ERAA Annex 2 default",
            },
            {
                "parameter": "Spanish_Annex3_battery_duration",
                "value": 6.0,
                "unit": "h",
                "basis": (
                    "ERAA default investment class is Battery Storage_6; "
                    "Table 2 releases only power"
                ),
            },
            {
                "parameter": "pumped_hydro_cycle_efficiency",
                "value": PUMPED_HYDRO_CYCLE_EFFICIENCY,
                "unit": "fraction",
                "basis": "ERAA Annex 2 default",
            },
            {
                "parameter": "pumped_hydro_initial_and_terminal_soc",
                "value": PUMPED_HYDRO_INITIAL_SOC,
                "unit": "fraction",
                "basis": "ERAA Annex 2 default",
            },
            {
                "parameter": "Annex3_DSR_band_allocation",
                "value": np.nan,
                "unit": "rule",
                "basis": (
                    "proportional to zone released bands; where absent, equal "
                    "split over six released central industrial bands"
                ),
            },
        ]
    )
    write_csv(assumptions, output / "flexible_resource_assumptions.csv")

    input_paths = [
        args.battery.resolve(),
        args.hydro.resolve(),
        args.dsr.resolve(),
        args.investment.resolve(),
        args.dispatch_root.resolve()
        / "amended_annex3_nonthermal_capacity_changes_2035.csv",
        args.phase1_root.resolve() / "capacity_2035_final.csv",
    ]
    manifest = pd.DataFrame(
        [
            {
                "path": str(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in input_paths
        ]
    )
    write_csv(manifest, output / "source_manifest.csv")

    battery_addition = float(batteries["annex3_addition_power_mw"].sum())
    dsr_addition = float(
        dsr.loc[
            dsr["capacity_component"].eq("amended_Annex3_expansion"),
            "capacity_mw",
        ].sum()
    )
    official_battery = float(batteries["base_discharge_power_mw"].sum())
    psp_usable = psp_audit.loc[psp_audit["usable_for_dispatch"]]
    qa = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now(),
        "target_year": TARGET_YEAR,
        "evidence_status": "audited_model_input_adapter",
        "utility_battery_zones": int(len(batteries)),
        "official_in_market_utility_battery_power_mw": official_battery,
        "annex3_battery_addition_power_mw": battery_addition,
        "expected_annex3_battery_addition_power_mw": 3_400.0,
        "explicit_dsr_rows": int(len(dsr)),
        "explicit_dsr_zones": int(dsr["zone"].nunique()),
        "official_explicit_dsr_power_mw": float(
            dsr.loc[
                dsr["capacity_component"].eq("released_2035_explicit_dsr"),
                "capacity_mw",
            ].sum()
        ),
        "annex3_dsr_addition_power_mw": dsr_addition,
        "expected_annex3_dsr_addition_power_mw": 10_050.0,
        "pumped_hydro_rows": int(len(psp)),
        "pumped_hydro_usable_rows": int(len(psp_usable)),
        "pumped_hydro_power_mw": float(psp_usable["discharge_power_mw"].sum()),
        "pumped_hydro_pumping_power_mw": float(
            psp_usable["charge_power_mw"].sum()
        ),
        "pumped_hydro_energy_mwh": float(
            psp_usable["energy_capacity_mwh"].sum()
        ),
        "gates": {
            "only_in_market_utility_battery_admitted": bool(
                battery_exclusions["normalized_status"]
                .ne(IN_MARKET)
                .all()
            ),
            "btm_battery_excluded_from_dispatch_storage": bool(
                battery_exclusions["normalized_status"]
                .str.contains("pv/battery", regex=False)
                .any()
            ),
            "annex3_battery_reconciled": bool(
                np.isclose(battery_addition, 3_400.0, atol=1e-9)
            ),
            "annex3_dsr_reconciled": bool(
                np.isclose(dsr_addition, 10_050.0, atol=1e-8)
            ),
            "battery_energy_positive_where_power_positive": bool(
                batteries.loc[
                    batteries["discharge_power_mw"].gt(0),
                    "energy_capacity_mwh",
                ].gt(0).all()
            ),
            "pumped_hydro_missing_dimensions_explicitly_audited": bool(
                psp_audit["issue"].notna().all()
            ),
            "no_negative_capacity": bool(
                batteries[
                    ["charge_power_mw", "discharge_power_mw", "energy_capacity_mwh"]
                ]
                .ge(0)
                .all()
                .all()
                and psp[
                    ["charge_power_mw", "discharge_power_mw", "energy_capacity_mwh"]
                ]
                .ge(0)
                .all()
                .all()
                and dsr["capacity_mw"].ge(0).all()
            ),
        },
    }
    qa["pass"] = bool(all(qa["gates"].values()))
    write_json(output / "QA.json", qa)
    write_json(
        output / "run_manifest.json",
        {
            "script_version": SCRIPT_VERSION,
            "created_utc": qa["created_utc"],
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "output": str(output),
        },
    )
    return qa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--battery", type=Path, default=DEFAULT_BATTERY)
    parser.add_argument("--hydro", type=Path, default=DEFAULT_HYDRO)
    parser.add_argument("--dsr", type=Path, default=DEFAULT_DSR)
    parser.add_argument("--investment", type=Path, default=DEFAULT_INVESTMENT)
    parser.add_argument("--dispatch-root", type=Path, default=DEFAULT_DISPATCH_ROOT)
    parser.add_argument("--phase1-root", type=Path, default=DEFAULT_PHASE1_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
