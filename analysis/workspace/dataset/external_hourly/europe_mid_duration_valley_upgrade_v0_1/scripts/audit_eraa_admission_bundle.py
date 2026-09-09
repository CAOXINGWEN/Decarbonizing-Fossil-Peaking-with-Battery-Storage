#!/usr/bin/env python3
"""Independent admission audit for the ERAA 2025 confirmatory model.

This script does not build or run the model.  It checks whether the public
ERAA inputs and the current adapter are sufficiently reconciled for model
results to be admitted as manuscript evidence.  Outputs are deliberately
prefixed ``audit_eraa_admission_`` so that this independent audit cannot be
confused with production-model artefacts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "0.1.0"
TARGET_YEAR = 2035
BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "raw" / "eraa_2025"
EXTRACTED = RAW / "extracted"
OUT = BASE / "artifacts" / "research"

GENERATION_CAPACITY = (
    EXTRACTED
    / "ERAA_2025_DashboardRawData1"
    / "Dashboard Raw Data"
    / "GenerationCapacities.csv"
)
BATTERY_DETAILS = (
    EXTRACTED
    / "ERAA_2025_DashboardRawData1"
    / "Dashboard Raw Data"
    / "Batteries additional information.csv"
)
HYDRO_DETAILS = (
    EXTRACTED
    / "ERAA_2025_DashboardRawData1"
    / "Dashboard Raw Data"
    / "Hydro additional information.csv"
)
DSR_DETAILS = EXTRACTED / "Other Data" / "Explicit DSR detailed.csv"
ANNEX3 = RAW / "ERAA_2025_Annex_3_Detailed_Results_Amended.pdf"
RELEASE_NOTE = RAW / "ERAA_2025_ExplanatoryNoteOnDataRelease.pdf"
EVA_WORKBOOK = RAW / "ERAA_2025_EVA.xlsx"
CATALOG = (
    BASE
    / "model_adapter"
    / "energy_charts_2015_2024_representative_pilot"
    / "configuration_catalog_53.csv"
)
FULL_COST_TABLE = (
    BASE.parents[2]
    / "Decarbonizing-Fossil-Peaking-with-Battery-Storage-master"
    / "paper-code"
    / "docs"
    / "musrcipt"
    / "revision_nature_energy_v36"
    / "system_value_closure"
    / "inputs"
    / "technology_duration_configurations_2035.csv"
)
CURRENCY_LEDGER = (
    BASE
    / "processed"
    / "eraa_2035_dispatch_parameters_v0_1"
    / "currency_conversion_ledger.csv"
)

INPUT_BUILDER = BASE / "scripts" / "build_eraa_2035_phase1_inputs.py"
DISPATCH_BUILDER = BASE / "scripts" / "build_eraa_2035_dispatch_parameters.py"
ZONE_RUNNER = BASE / "scripts" / "run_eraa_2035_phase1_zone_task.py"
MODEL_CORE = BASE / "scripts" / "eraa_confirmatory_model_core.py"

INCLUDED_ENERGY_MARKET_STATUS = frozenset(
    {
        "available on market",
        "inelastic supply / fixed profile",
        "out of market - for pv/battery dispatch optimization",
    }
)
BTM_STATUS = "out of market - for pv/battery dispatch optimization"
AVAILABLE_STATUS = "available on market"

ANNEX_TECHNOLOGY_MAP = {
    "Battery": "Battery utility scale",
    "DSR": "DSR",
    "Gas CCGT": "Natural gas",
    "Gas OCGT": "Natural gas",
    "Hydrogen CCGT": "Hydrogen",
    "Hard coal": "Hard coal",
    "Lignite": "Lignite",
    "Light oil": "Light oil",
    "Heavy oil": "Heavy oil",
}
EXPECTED_ANNEX_2035 = {
    ("Battery", "Expansion"): 3_400,
    ("DSR", "Expansion"): 10_050,
    ("Gas CCGT", "Life Extension"): 6_010,
    ("Gas CCGT", "Retirement"): -30_660,
    ("Gas OCGT", "Expansion"): 26_980,
    ("Gas OCGT", "Life Extension"): 1_060,
    ("Gas OCGT", "Retirement"): -1_160,
    ("Hydrogen CCGT", "Expansion"): 3_850,
    ("Hard coal", "Retirement"): -170,
    ("Lignite", "Retirement"): -2_630,
    ("Light oil", "Retirement"): -2_020,
    ("Heavy oil", "Retirement"): -80,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_status(value: object) -> str:
    text = str(value).replace("–", "-").replace("—", "-").replace("−", "-")
    return re.sub(r"\s+", " ", text.strip().lower())


def pdf_text(path: Path) -> str:
    result = subprocess.run(
        ["pdftotext", "-layout", str(path), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def pdf_page_for(text: str, needle: str) -> int:
    position = text.index(needle)
    return text[:position].count("\f") + 1


def parse_numbers(value: str) -> list[int]:
    normalized = value.replace("−", "-").replace("–", "-")
    tokens = re.findall(r"[-+]?\s*\d[\d,]*", normalized)
    return [int(token.replace(" ", "").replace(",", "")) for token in tokens]


def parse_annex3_table2() -> pd.DataFrame:
    text = pdf_text(ANNEX3)
    start = text.index("Table 2: Capacity change proposed by EVA per study zone")
    stop = text.index("Grand Total", start)
    lines = text[start:stop].splitlines()
    technologies = "|".join(
        re.escape(item)
        for item in sorted(ANNEX_TECHNOLOGY_MAP, key=len, reverse=True)
    )
    descriptor = re.compile(
        rf"^\s*(?:(?P<zone>[A-Z]{{2,4}}[A-Z0-9]{{1,3}})\s+)?"
        rf"(?P<technology>{technologies})\s+"
        rf"(?P<decision>Life Extension|Retirement|Expansion)"
        rf"(?P<tail>.*)$"
    )

    rows: list[dict[str, Any]] = []
    current_zone: str | None = None
    index = 0
    while index < len(lines):
        line = lines[index].replace("\f", "")
        match = descriptor.match(line)
        if match is None:
            index += 1
            continue
        if match.group("zone"):
            current_zone = match.group("zone")
        if current_zone is None:
            raise ValueError("Annex 3 technology record appears before a zone")
        values = parse_numbers(match.group("tail"))
        consumed = index
        if len(values) != 4:
            values = []
            for probe in range(index + 1, min(index + 5, len(lines))):
                candidate = parse_numbers(lines[probe])
                if len(candidate) == 4:
                    values = candidate
                    consumed = probe
                    break
                if descriptor.match(lines[probe]):
                    break
        if len(values) != 4:
            raise ValueError(
                "Could not parse four target years for "
                f"{current_zone}/{match.group('technology')}/"
                f"{match.group('decision')}"
            )
        for year, change in zip((2028, 2030, 2033, 2035), values):
            rows.append(
                {
                    "zone": current_zone,
                    "pemmbd_technology": match.group("technology"),
                    "decision_variable": match.group("decision"),
                    "target_year": year,
                    "capacity_change_mw": float(change),
                    "mapped_capacity_technology": ANNEX_TECHNOLOGY_MAP[
                        match.group("technology")
                    ],
                    "source_file": ANNEX3.name,
                    "source_table": "Table 2",
                }
            )
        index = consumed + 1

    result = pd.DataFrame(rows)
    if len(result) != 121 * 4:
        raise ValueError(f"Expected 121 Annex rows x 4 years; got {len(result)}")
    selected = result.loc[result["target_year"].eq(TARGET_YEAR)].copy()
    observed = (
        selected.groupby(
            ["pemmbd_technology", "decision_variable"]
        )["capacity_change_mw"]
        .sum()
        .to_dict()
    )
    if observed != EXPECTED_ANNEX_2035:
        raise ValueError(
            "Amended Annex 3 aggregate validation failed: "
            f"observed={observed}"
        )
    if not np.isclose(selected["capacity_change_mw"].sum(), 14_630.0):
        raise ValueError("Amended Annex 3 2035 grand total is not 14,630 MW")
    return selected.reset_index(drop=True)


def read_generation_capacity() -> pd.DataFrame:
    frame = pd.read_csv(GENERATION_CAPACITY)
    frame = frame.loc[
        frame["data_version"].eq("ERAA 2025 final")
        & pd.to_numeric(frame["Target year"], errors="coerce").eq(TARGET_YEAR)
    ].copy()
    frame["Value"] = pd.to_numeric(frame["Value"], errors="raise")
    frame["status_normalized"] = frame["Operational_Status"].map(normalize_status)
    return frame


def build_post_eva_capacity_audit(
    annex: pd.DataFrame, generation: pd.DataFrame
) -> pd.DataFrame:
    changes = (
        annex.groupby(
            ["zone", "mapped_capacity_technology"], as_index=False
        )["capacity_change_mw"]
        .sum()
        .rename(columns={"mapped_capacity_technology": "technology"})
    )
    records: list[pd.DataFrame] = []
    scopes = {
        "all_operational_statuses": generation,
        "current_adapter_included_statuses": generation.loc[
            generation["status_normalized"].isin(INCLUDED_ENERGY_MARKET_STATUS)
        ],
    }
    for scope, subset in scopes.items():
        base = (
            subset.groupby(["Market_Node", "Technology"], as_index=False)["Value"]
            .sum()
            .rename(
                columns={
                    "Market_Node": "zone",
                    "Technology": "technology",
                    "Value": "national_trends_capacity_mw",
                }
            )
        )
        joined = changes.merge(
            base, on=["zone", "technology"], how="left", validate="one_to_one"
        )
        joined["source_key_present"] = joined[
            "national_trends_capacity_mw"
        ].notna()
        joined["national_trends_capacity_mw"] = joined[
            "national_trends_capacity_mw"
        ].fillna(0.0)
        joined["post_eva_capacity_mw"] = (
            joined["national_trends_capacity_mw"]
            + joined["capacity_change_mw"]
        )
        joined["post_eva_nonnegative"] = joined["post_eva_capacity_mw"].ge(
            -1e-9
        )
        joined["direct_broad_table_transfer_admissible"] = (
            joined["post_eva_nonnegative"]
            & (
                joined["source_key_present"]
                | joined["capacity_change_mw"].ge(0)
            )
        )
        joined["capacity_scope"] = scope
        joined["subtype_information_lost"] = joined["technology"].eq(
            "Natural gas"
        )
        joined["operation_parameter_transfer_required"] = joined[
            "technology"
        ].isin({"Battery utility scale", "DSR"})
        joined["mapping_disposition"] = np.select(
            [
                (
                    joined["capacity_change_mw"].lt(0)
                    & ~joined["source_key_present"]
                ),
                joined["technology"].eq("Natural gas"),
                joined["operation_parameter_transfer_required"],
            ],
            [
                "unresolved_cross_taxonomy_or_version_retirement",
                "aggregate_transfer_only_subtype_split_required",
                "capacity_transfer_possible_operation_parameters_required",
            ],
            default="direct_zone_class_transfer_arithmetically_admissible",
        )
        records.append(joined)
    return pd.concat(records, ignore_index=True)


def build_ie_unmapped_retirement_evidence(
    annex: pd.DataFrame, generation: pd.DataFrame
) -> pd.DataFrame:
    workbook = pd.read_excel(EVA_WORKBOOK, sheet_name="Cost Based")
    workbook_rows = workbook.loc[
        workbook["Node"].astype(str).eq("IE00")
        & pd.to_numeric(workbook["Year"], errors="coerce").eq(TARGET_YEAR)
        & workbook["Technology"].astype(str).str.startswith("Lignite")
        & workbook["EVA Type"].astype(str).eq("Retirement")
    ].copy()
    annex_rows = annex.loc[
        annex["zone"].eq("IE00")
        & annex["pemmbd_technology"].eq("Lignite")
        & annex["decision_variable"].eq("Retirement")
    ]
    broad_rows = generation.loc[
        generation["Market_Node"].eq("IE00")
        & generation["Technology"].eq("Lignite")
    ]
    if len(workbook_rows) != 1 or len(annex_rows) != 1 or len(broad_rows) != 1:
        raise ValueError("IE00 lignite mapping evidence is not unique")
    return pd.DataFrame(
        [
            {
                "zone": "IE00",
                "amended_annex3_technology": "Lignite",
                "amended_annex3_decision": "Retirement",
                "amended_annex3_2035_change_mw": float(
                    annex_rows.iloc[0]["capacity_change_mw"]
                ),
                "eva_workbook_cost_based_unit_class": str(
                    workbook_rows.iloc[0]["Technology"]
                ),
                "eva_workbook_2035_change_mw": float(
                    workbook_rows.iloc[0]["Capacity Change"]
                ),
                "public_broad_capacity_technology": "Lignite",
                "public_broad_capacity_2035_mw": float(
                    broad_rows.iloc[0]["Value"]
                ),
                "audit_interpretation": (
                    "the -110 MW retirement is present in both EVA sources, "
                    "but cannot be hosted by the public broad Lignite row; "
                    "the negative direct sum is a mapping diagnostic, not a "
                    "physical negative-capacity result"
                ),
                "admissible_main_model_rule": (
                    "require an official EVA-class/vintage-to-public-"
                    "technology crosswalk; "
                    "until then quarantine this transfer, do not clip it, and "
                    "either exclude IE00 from confirmatory attribution or "
                    "show result invariance across explicit bounded mappings"
                ),
            }
        ]
    )


def battery_profile_zones() -> set[str]:
    root = RAW / "selective_2035" / "ERAA_2025_DemandData"
    paths = root.rglob("*_Batteries_2035_National Trends.csv")
    return {path.name.split("_", 1)[0] for path in paths}


def build_resource_audit(generation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    batteries = pd.read_csv(BATTERY_DETAILS)
    batteries = batteries.loc[
        batteries["data_version"].eq("ERAA 2025 final")
        & pd.to_numeric(batteries["TARGET_YEAR"], errors="coerce").eq(TARGET_YEAR)
    ].copy()
    batteries["status_normalized"] = batteries["OP_STAT"].map(normalize_status)
    for column in ("MAX LOAD CAP (MW)", "STORAGE CAPACITY (MWh)"):
        batteries[column] = pd.to_numeric(batteries[column], errors="raise")

    hydro = pd.read_csv(HYDRO_DETAILS)
    hydro = hydro.loc[
        hydro["data_version"].eq("ERAA 2025 final")
        & pd.to_numeric(hydro["TARGET_YEAR"], errors="coerce").eq(TARGET_YEAR)
    ].copy()
    for column in ("MAX PUMPING CAP (MW)", "Storage Capacity [TWh]"):
        hydro[column] = pd.to_numeric(hydro[column], errors="coerce")

    dsr = pd.read_csv(DSR_DETAILS)
    dsr = dsr.loc[
        pd.to_numeric(dsr["TARGET_YEAR"], errors="coerce").eq(TARGET_YEAR)
    ].copy()
    dsr["MAX CAP (MW)"] = pd.to_numeric(dsr["MAX CAP (MW)"], errors="raise")

    gen_battery = generation.loc[
        generation["Technology"].astype(str).str.startswith("Battery")
    ]
    gen_hydro = generation.loc[
        generation["Technology"].isin(
            {
                "Closed loop pumping",
                "Open loop pumping",
                "Pondage",
                "Reservoir",
            }
        )
    ]

    def generation_sum(technology: str, status: str | None = None) -> float:
        selected = generation.loc[generation["Technology"].eq(technology)]
        if status is not None:
            selected = selected.loc[selected["status_normalized"].eq(status)]
        return float(selected["Value"].sum())

    def battery_detail_sum(status: str, column: str) -> float:
        return float(
            batteries.loc[batteries["status_normalized"].eq(status), column].sum()
        )

    rows = [
        {
            "resource": "behind_the_meter_battery_fixed_profile",
            "released_discharge_capacity_mw": float(
                gen_battery.loc[
                    gen_battery["status_normalized"].eq(BTM_STATUS), "Value"
                ].sum()
            ),
            "released_charge_capacity_mw": battery_detail_sum(
                BTM_STATUS, "MAX LOAD CAP (MW)"
            ),
            "released_energy_capacity_mwh": battery_detail_sum(
                BTM_STATUS, "STORAGE CAPACITY (MWh)"
            ),
            "current_model_representation": (
                "signed released BTM output is embedded in net load; "
                "negative net load is retained as potential surplus"
            ),
            "endogenous_in_current_core": False,
            "currently_double_counted": False,
            "admission_issue": (
                "fixed-profile identity is preserved and must not also enter "
                "as dispatchable storage; the officially price-responsive "
                "share is not re-optimized, and LUG1 has capacity but no "
                "released BTM profile"
            ),
        },
        {
            "resource": "implicit_dsr_ev_hp_and_btm_flexibility",
            "released_discharge_capacity_mw": float(
                generation.loc[
                    generation["Technology"].isin({"EV - iDSR", "HP - iDSR"})
                    & generation["status_normalized"].eq(AVAILABLE_STATUS),
                    "Value",
                ].sum()
            ),
            "released_charge_capacity_mw": np.nan,
            "released_energy_capacity_mwh": np.nan,
            "current_model_representation": (
                "EV and heat-pump demand remains embedded in load; signed "
                "BTM dispatch remains fixed"
            ),
            "endogenous_in_current_core": False,
            "currently_double_counted": False,
            "admission_issue": (
                "official price-responsive ratios and rescheduling "
                "constraints are not acquired or represented"
            ),
        },
        {
            "resource": "market_batteries_utility_plus_residential",
            "released_discharge_capacity_mw": float(
                gen_battery.loc[
                    gen_battery["status_normalized"].eq(AVAILABLE_STATUS), "Value"
                ].sum()
            ),
            "released_charge_capacity_mw": battery_detail_sum(
                AVAILABLE_STATUS, "MAX LOAD CAP (MW)"
            ),
            "released_energy_capacity_mwh": battery_detail_sum(
                AVAILABLE_STATUS, "STORAGE CAPACITY (MWh)"
            ),
            "current_model_representation": "omitted",
            "endogenous_in_current_core": False,
            "currently_double_counted": False,
            "admission_issue": (
                "existing storage can materially change scarcity, recharge "
                "windows and candidate marginal value"
            ),
        },
        {
            "resource": "closed_loop_pumped_hydro",
            "released_discharge_capacity_mw": generation_sum(
                "Closed loop pumping", AVAILABLE_STATUS
            ),
            "released_charge_capacity_mw": float(
                hydro.loc[
                    hydro["PEMMDB_PLANT_TYPE"].eq("Closed loop pumping"),
                    "MAX PUMPING CAP (MW)",
                ].sum()
            ),
            "released_energy_capacity_mwh": float(
                hydro.loc[
                    hydro["PEMMDB_PLANT_TYPE"].eq("Closed loop pumping"),
                    "Storage Capacity [TWh]",
                ].sum()
                * 1_000_000.0
            ),
            "current_model_representation": "omitted",
            "endogenous_in_current_core": False,
            "currently_double_counted": False,
            "admission_issue": (
                "34 GW-scale storage fleet absent from both charge and "
                "discharge decisions"
            ),
        },
        {
            "resource": "open_loop_pumped_hydro",
            "released_discharge_capacity_mw": generation_sum(
                "Open loop pumping", AVAILABLE_STATUS
            ),
            "released_charge_capacity_mw": float(
                hydro.loc[
                    hydro["PEMMDB_PLANT_TYPE"].eq("Open loop pumping"),
                    "MAX PUMPING CAP (MW)",
                ].sum()
            ),
            "released_energy_capacity_mwh": float(
                hydro.loc[
                    hydro["PEMMDB_PLANT_TYPE"].eq("Open loop pumping"),
                    "Storage Capacity [TWh]",
                ].sum()
                * 1_000_000.0
            ),
            "current_model_representation": (
                "discharge-only aggregate reservoir; pumping omitted"
            ),
            "endogenous_in_current_core": False,
            "currently_double_counted": False,
            "admission_issue": (
                "cannot absorb energy and therefore changes the mechanism "
                "being tested"
            ),
        },
        {
            "resource": "reservoir_and_pondage",
            "released_discharge_capacity_mw": float(
                gen_hydro.loc[
                    gen_hydro["Technology"].isin({"Reservoir", "Pondage"}),
                    "Value",
                ].sum()
            ),
            "released_charge_capacity_mw": 0.0,
            "released_energy_capacity_mwh": float(
                hydro.loc[
                    hydro["PEMMDB_PLANT_TYPE"].isin({"Reservoir", "Pondage"}),
                    "Storage Capacity [TWh]",
                ].sum()
                * 1_000_000.0
            ),
            "current_model_representation": (
                "aggregate reservoir with inflow, generation and spill"
            ),
            "endogenous_in_current_core": True,
            "currently_double_counted": False,
            "admission_issue": (
                "released minimum, period-energy and level constraints are "
                "not yet enforced"
            ),
        },
        {
            "resource": "explicit_demand_response",
            "released_discharge_capacity_mw": generation_sum(
                "DSR", AVAILABLE_STATUS
            ),
            "released_charge_capacity_mw": np.nan,
            "released_energy_capacity_mwh": np.nan,
            "current_model_representation": "omitted",
            "endogenous_in_current_core": False,
            "currently_double_counted": False,
            "admission_issue": (
                "activation price and maximum daily dispatch constraints "
                "are published but absent; Annex expansion cannot be assigned "
                "to activation tiers without an explicit rule"
            ),
        },
    ]
    resource = pd.DataFrame(rows)
    market_battery_generation_rows = gen_battery.loc[
        gen_battery["status_normalized"].eq(AVAILABLE_STATUS)
    ]
    market_battery_detail_rows = batteries.loc[
        batteries["status_normalized"].eq(AVAILABLE_STATUS)
    ]
    closed_loop_generation_rows = generation.loc[
        generation["Technology"].eq("Closed loop pumping")
        & generation["status_normalized"].eq(AVAILABLE_STATUS)
    ]
    closed_loop_detail_rows = hydro.loc[
        hydro["PEMMDB_PLANT_TYPE"].eq("Closed loop pumping")
    ]
    audit_metadata = {
        "behind_the_meter_battery_fixed_profile": {
            "discharge_capacity_source_and_field": (
                "GenerationCapacities.csv::Value"
            ),
            "charge_and_energy_source_fields": (
                "Batteries additional information.csv::MAX LOAD CAP (MW);"
                "STORAGE CAPACITY (MWh)"
            ),
            "sum_filter": (
                "2035; ERAA 2025 final; Battery residential; normalized "
                "status=out of market - for pv/battery dispatch optimization"
            ),
            "deduplication_key": (
                "Market_Node + Technology + normalized Operational_Status "
                "(capacity); MARKET_NODE + normalized OP_STAT (details)"
            ),
            "deduplication_rule": (
                "BTM capacity is reported separately and excluded from the "
                "market-battery aggregate; profile is applied once in net load"
            ),
        },
        "implicit_dsr_ev_hp_and_btm_flexibility": {
            "discharge_capacity_source_and_field": (
                "GenerationCapacities.csv::Value for EV - iDSR and HP - iDSR"
            ),
            "charge_and_energy_source_fields": "iDSR ratio package not acquired",
            "sum_filter": (
                "2035; ERAA 2025 final; normalized status=available on market"
            ),
            "deduplication_key": (
                "Market_Node + Technology + normalized Operational_Status"
            ),
            "deduplication_rule": (
                "EV/HP capacity is not pooled with explicit DSR; BTM flexible "
                "share is not added without its published ratio"
            ),
        },
        "market_batteries_utility_plus_residential": {
            "discharge_capacity_source_and_field": (
                "GenerationCapacities.csv::Value"
            ),
            "charge_and_energy_source_fields": (
                "Batteries additional information.csv::MAX LOAD CAP (MW);"
                "STORAGE CAPACITY (MWh)"
            ),
            "sum_filter": (
                "2035; ERAA 2025 final; Battery utility scale OR Battery "
                "residential; normalized status=available on market"
            ),
            "deduplication_key": (
                "Market_Node + Technology + normalized Operational_Status "
                "(capacity); MARKET_NODE + normalized OP_STAT (details)"
            ),
            "deduplication_rule": (
                "utility and market-responsive residential batteries are "
                "summed once; BTM and ancillary/other statuses are excluded; "
                "charge and discharge fields are complementary dimensions, "
                "not additive capacities"
            ),
            "source_component_breakdown": (
                "Battery utility scale="
                f"{generation_sum('Battery utility scale', AVAILABLE_STATUS):.0f}"
                " MW; Battery residential="
                f"{generation_sum('Battery residential', AVAILABLE_STATUS):.0f}"
                " MW"
            ),
            "discharge_source_row_count": len(market_battery_generation_rows),
            "discharge_source_zone_count": (
                market_battery_generation_rows["Market_Node"].nunique()
            ),
            "detail_source_row_count": len(market_battery_detail_rows),
            "detail_source_zone_count": (
                market_battery_detail_rows["MARKET_NODE"].nunique()
            ),
        },
        "closed_loop_pumped_hydro": {
            "discharge_capacity_source_and_field": (
                "GenerationCapacities.csv::Value"
            ),
            "charge_and_energy_source_fields": (
                "Hydro additional information.csv::MAX PUMPING CAP (MW);"
                "Storage Capacity [TWh] x 1,000,000"
            ),
            "sum_filter": (
                "2035; ERAA 2025 final; Closed loop pumping; normalized "
                "status=available on market"
            ),
            "deduplication_key": (
                "Market_Node + Technology + normalized Operational_Status "
                "(capacity); MARKET_NODE + PEMMDB_PLANT_TYPE (details)"
            ),
            "deduplication_rule": (
                "discharge, pump and energy fields describe the same fleet "
                "and are retained as separate constraint dimensions"
            ),
            "source_component_breakdown": (
                "Closed loop pumping="
                f"{generation_sum('Closed loop pumping', AVAILABLE_STATUS):.0f}"
                " MW discharge; "
                f"{closed_loop_detail_rows['MAX PUMPING CAP (MW)'].sum():.2f}"
                " MW pump; "
                f"{closed_loop_detail_rows['Storage Capacity [TWh]'].sum():.2f}"
                " TWh"
            ),
            "discharge_source_row_count": len(closed_loop_generation_rows),
            "discharge_source_zone_count": (
                closed_loop_generation_rows["Market_Node"].nunique()
            ),
            "detail_source_row_count": len(closed_loop_detail_rows),
            "detail_source_zone_count": (
                closed_loop_detail_rows["MARKET_NODE"].nunique()
            ),
        },
        "open_loop_pumped_hydro": {
            "discharge_capacity_source_and_field": (
                "GenerationCapacities.csv::Value"
            ),
            "charge_and_energy_source_fields": (
                "Hydro additional information.csv::MAX PUMPING CAP (MW);"
                "Storage Capacity [TWh] x 1,000,000"
            ),
            "sum_filter": (
                "2035; ERAA 2025 final; Open loop pumping; normalized "
                "status=available on market"
            ),
            "deduplication_key": (
                "Market_Node + Technology + normalized Operational_Status "
                "(capacity); MARKET_NODE + PEMMDB_PLANT_TYPE (details)"
            ),
            "deduplication_rule": (
                "discharge, pump and energy fields describe the same fleet "
                "and are retained as separate constraint dimensions"
            ),
        },
        "reservoir_and_pondage": {
            "discharge_capacity_source_and_field": (
                "GenerationCapacities.csv::Value"
            ),
            "charge_and_energy_source_fields": (
                "Hydro additional information.csv::Storage Capacity [TWh] "
                "x 1,000,000"
            ),
            "sum_filter": (
                "2035; ERAA 2025 final; Reservoir OR Pondage; normalized "
                "status=available on market"
            ),
            "deduplication_key": (
                "Market_Node + Technology + normalized Operational_Status "
                "(capacity); MARKET_NODE + PEMMDB_PLANT_TYPE (details)"
            ),
            "deduplication_rule": (
                "reservoir and pondage categories are summed once into the "
                "current aggregate reservoir representation"
            ),
        },
        "explicit_demand_response": {
            "discharge_capacity_source_and_field": (
                "GenerationCapacities.csv::Value"
            ),
            "charge_and_energy_source_fields": (
                "Explicit DSR detailed.csv::MAX CAP (MW), activation price, "
                "maximum hours dispatched per day"
            ),
            "sum_filter": (
                "2035; Technology=DSR; normalized status=available on market"
            ),
            "deduplication_key": (
                "Market_Node + normalized Operational_Status (capacity); "
                "MARKET_NODE + activation price + maximum daily hours "
                "(details)"
            ),
            "deduplication_rule": (
                "explicit DSR is kept separate from EV/HP iDSR and from "
                "ancillary-only DSR"
            ),
        },
    }
    metadata = (
        pd.DataFrame.from_dict(audit_metadata, orient="index")
        .rename_axis("resource")
        .reset_index()
    )
    resource = resource.merge(metadata, on="resource", validate="one_to_one")

    gen_dsr = (
        generation.loc[
            generation["Technology"].eq("DSR")
            & generation["status_normalized"].eq(AVAILABLE_STATUS)
        ]
        .groupby("Market_Node")["Value"]
        .sum()
        .rename("generation_capacity_available_mw")
    )
    detailed_dsr = (
        dsr.groupby("MARKET_NODE")["MAX CAP (MW)"]
        .sum()
        .rename("detailed_dsr_capacity_mw")
    )
    dsr_reconciliation = pd.concat([gen_dsr, detailed_dsr], axis=1).fillna(0.0)
    dsr_reconciliation["difference_detail_minus_generation_mw"] = (
        dsr_reconciliation["detailed_dsr_capacity_mw"]
        - dsr_reconciliation["generation_capacity_available_mw"]
    )
    dsr_reconciliation["within_rounding_tolerance_1mw"] = (
        dsr_reconciliation["difference_detail_minus_generation_mw"].abs().le(1.0)
    )
    dsr_reconciliation.index.name = "zone"
    dsr_reconciliation = dsr_reconciliation.reset_index()
    return resource, dsr_reconciliation


def build_btm_zone_audit(generation: pd.DataFrame) -> pd.DataFrame:
    profiles = battery_profile_zones()
    capacity = (
        generation.loc[
            generation["Technology"].eq("Battery residential")
            & generation["status_normalized"].eq(BTM_STATUS)
        ]
        .groupby("Market_Node")["Value"]
        .sum()
    )
    zones = sorted(set(capacity.index).union(profiles))
    rows = []
    for zone in zones:
        rows.append(
            {
                "zone": zone,
                "btm_battery_capacity_mw": float(capacity.get(zone, 0.0)),
                "released_btm_battery_profile_present": zone in profiles,
                "capacity_profile_consistent": (
                    (float(capacity.get(zone, 0.0)) > 0) == (zone in profiles)
                ),
                "current_handling": (
                    "fixed signed profile in net load"
                    if zone in profiles
                    else "silently zero unless explicitly gated"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_cost_conversion_audit() -> pd.DataFrame:
    catalog = pd.read_csv(CATALOG)
    full = pd.read_csv(FULL_COST_TABLE)
    currency = pd.read_csv(CURRENCY_LEDGER)
    selected = currency.loc[currency["scenario"].eq("central_cny_per_eur")]
    if len(selected) != 1:
        raise ValueError("Central CNY/EUR conversion rate is not unique")
    rate = float(selected.iloc[0]["cny_per_eur"])

    source_columns = [
        "tech_id",
        "duration_h",
        "capex_power_basis",
        "parameter_basis",
        "annual_fixed_cost_cny_per_kwh_capacity_year",
        "annual_fixed_cost_cny_per_kw_discharge_year",
        "annualized_power_capex_cny_per_kw_year",
        "fixed_om_cny_per_kwh_capacity_year",
        "annual_replacement_reserve_cny_per_kw_year",
    ]
    merged = catalog.merge(
        full[source_columns],
        on=["tech_id", "duration_h"],
        how="left",
        validate="one_to_one",
    )
    if merged[source_columns[2:]].isna().any().any():
        raise ValueError("At least one of the 53 configurations lacks source cost fields")

    merged["central_cny_per_eur"] = rate
    merged["source_all_in_annual_cost_cny_per_kw_discharge_year"] = merged[
        "annual_fixed_cost_cny_per_kw_discharge_year"
    ]
    merged["model_all_in_energy_coefficient_eur_per_kwh_year"] = (
        merged["fixed_cost_cny_per_kwh_year"] / rate
    )
    merged["model_reference_annual_cost_eur_per_kw_discharge_year"] = (
        merged["model_all_in_energy_coefficient_eur_per_kwh_year"]
        * merged["duration_h"]
    )
    merged["source_reference_annual_cost_eur_per_kw_discharge_year"] = (
        merged["source_all_in_annual_cost_cny_per_kw_discharge_year"] / rate
    )
    merged["reference_cost_reconstruction_error_eur_per_kw_year"] = (
        merged["model_reference_annual_cost_eur_per_kw_discharge_year"]
        - merged["source_reference_annual_cost_eur_per_kw_discharge_year"]
    )
    merged["model_discharge_power_cost_eur_per_kw_year"] = 0.0
    merged["bundled_annual_power_cost_eur_per_kw_year"] = (
        merged["annual_power_cost_cny_per_kw_year"] / rate
    )
    merged["model_incremental_charge_power_cost_eur_per_kw_year"] = merged[
        "bundled_annual_power_cost_eur_per_kw_year"
    ]
    merged["reference_charge_power_ratio"] = merged["charge_power_ratio"]
    merged["reference_configuration_double_counted"] = (
        merged["reference_cost_reconstruction_error_eur_per_kw_year"].abs()
        > 1e-9
    )
    merged["incremental_charge_component_separately_sourced"] = False
    merged["incremental_charge_cost_interpretation"] = np.where(
        merged["tech_id"].eq("caes"),
        (
            "bundled CAES turbine+compressor+BOP annual power block used as "
            "a sensitivity price for each extra charging kW; no "
            "compressor-only marginal-cost evidence"
        ),
        (
            "bundled reference power block used as a sensitivity price for "
            "each extra charging kW; no charge-interface-only source field"
        ),
    )
    keep = [
        "config_id",
        "tech_id",
        "technology_en",
        "duration_h",
        "central_cny_per_eur",
        "fixed_cost_cny_per_kwh_year",
        "source_all_in_annual_cost_cny_per_kw_discharge_year",
        "model_all_in_energy_coefficient_eur_per_kwh_year",
        "model_reference_annual_cost_eur_per_kw_discharge_year",
        "source_reference_annual_cost_eur_per_kw_discharge_year",
        "reference_cost_reconstruction_error_eur_per_kw_year",
        "model_discharge_power_cost_eur_per_kw_year",
        "annual_power_cost_cny_per_kw_year",
        "bundled_annual_power_cost_eur_per_kw_year",
        "model_incremental_charge_power_cost_eur_per_kw_year",
        "reference_charge_power_ratio",
        "capex_power_basis",
        "parameter_basis",
        "reference_configuration_double_counted",
        "incremental_charge_component_separately_sourced",
        "incremental_charge_cost_interpretation",
    ]
    return merged[keep].sort_values(["duration_h", "tech_id"]).reset_index(drop=True)


def build_hard_gates(
    post_eva: pd.DataFrame,
    resource: pd.DataFrame,
    dsr: pd.DataFrame,
    btm: pd.DataFrame,
    cost: pd.DataFrame,
) -> pd.DataFrame:
    current_scope = post_eva.loc[
        post_eva["capacity_scope"].eq("current_adapter_included_statuses")
    ]
    input_bundle = BASE / "processed" / "eraa_2035_phase1_inputs_v0_1"
    required_bundle_files = ("QA.json", "run_manifest.json", "source_manifest.csv")
    missing_bundle_files = [
        name for name in required_bundle_files if not (input_bundle / name).exists()
    ]
    bundle_qa_pass = False
    if not missing_bundle_files:
        try:
            bundle_qa = json.loads((input_bundle / "QA.json").read_text())
            bundle_qa_pass = bool(bundle_qa.get("pass", False))
        except (OSError, json.JSONDecodeError):
            bundle_qa_pass = False
    checks = [
        {
            "gate_id": "G01_post_eva_capacity_transfer",
            "domain": "capacity",
            "pass": bool(current_scope["post_eva_nonnegative"].all()),
            "blocking": True,
            "evidence": (
                f"{(~current_scope['post_eva_nonnegative']).sum()} mapped "
                "zone-technology record(s) become negative; IE00 lignite is "
                "the observed inconsistency"
            ),
            "required_resolution": (
                "obtain an EVA-class/vintage-to-public-technology crosswalk; "
                "quarantine the IE00 transfer and do not clip it"
            ),
        },
        {
            "gate_id": "G02_existing_market_batteries",
            "domain": "existing flexibility",
            "pass": bool(
                resource.loc[
                    resource["resource"].eq(
                        "market_batteries_utility_plus_residential"
                    ),
                    "endogenous_in_current_core",
                ].all()
            ),
            "blocking": True,
            "evidence": (
                "151,882 MW released market battery discharge capacity and "
                "588,150 MWh energy are omitted"
            ),
            "required_resolution": (
                "add existing battery SOC, charge/discharge limits and cyclic "
                "boundary; exclude BTM fixed-profile capacity"
            ),
        },
        {
            "gate_id": "G03_pumped_hydro",
            "domain": "existing flexibility",
            "pass": False,
            "blocking": True,
            "evidence": (
                "closed-loop pumping is omitted and open-loop pumping is "
                "represented without pumping"
            ),
            "required_resolution": (
                "model released pumping power, storage states and applicable "
                "hydro constraints"
            ),
        },
        {
            "gate_id": "G04_explicit_dsr",
            "domain": "existing flexibility",
            "pass": False,
            "blocking": True,
            "evidence": (
                f"explicit DSR is absent; "
                f"{(~dsr['within_rounding_tolerance_1mw']).sum()} zone(s) "
                "also fail the capacity/detail reconciliation"
            ),
            "required_resolution": (
                "add tiered activation prices and daily duration constraints; "
                "reconcile detailed capacities and assign Annex expansion"
            ),
        },
        {
            "gate_id": "G05_btm_and_implicit_dsr",
            "domain": "demand and implicit flexibility",
            "pass": False,
            "blocking": True,
            "evidence": (
                "the official price-responsive BTM/EV/heat-pump shares are "
                "not re-optimized; additionally "
                f"{(~btm['capacity_profile_consistent']).sum()} zone(s) have "
                "a BTM capacity/profile mismatch and LUG1 is missing a profile"
            ),
            "required_resolution": (
                "acquire the iDSR ratios and implement rescheduling; obtain "
                "the LUG1 BTM profile or explicitly exclude the zone"
            ),
        },
        {
            "gate_id": "G06_full_network_model",
            "domain": "network",
            "pass": False,
            "blocking": True,
            "evidence": (
                "current phase-1 runner solves one atomic zone with zero "
                "interconnection; NTC tables are built but not used there"
            ),
            "required_resolution": (
                "solve the predeclared 52-zone connected model, with NTC "
                "central case and flow-based sensitivity where applicable"
            ),
        },
        {
            "gate_id": "G07_incremental_charge_power_cost",
            "domain": "candidate cost",
            "pass": bool(
                cost["incremental_charge_component_separately_sourced"].all()
            ),
            "blocking": True,
            "evidence": (
                "reference configurations are not double counted, but none "
                "of 53 configurations has a separately sourced marginal "
                "charging-interface cost; CAES source power cost is bundled"
            ),
            "required_resolution": (
                "fix charge/discharge ratio at its source value in the central "
                "case, or source technology-specific marginal charging-power "
                "costs; retain bundled multipliers only as sensitivity"
            ),
        },
        {
            "gate_id": "G08_sequential_outages",
            "domain": "adequacy",
            "pass": False,
            "blocking": True,
            "evidence": (
                "current dispatch adapter uses expected availability rather "
                "than sequential forced-outage paths"
            ),
            "required_resolution": (
                "run predeclared sequential outage paths for the confirmatory "
                "adequacy/ELCC evidence"
            ),
        },
        {
            "gate_id": "G09_gas_subtype_mapping",
            "domain": "thermal fleet",
            "pass": False,
            "blocking": True,
            "evidence": (
                "Annex reports CCGT and OCGT decisions but the public "
                "capacity table provides only aggregate Natural gas"
            ),
            "required_resolution": (
                "apply exact net gas transfer before transparent split "
                "sensitivities; never label inferred shares as observed"
            ),
        },
        {
            "gate_id": "G10_processed_input_bundle_integrity",
            "domain": "reproducibility",
            "pass": bool(not missing_bundle_files and bundle_qa_pass),
            "blocking": True,
            "evidence": (
                "phase-1 input bundle is missing "
                + (
                    ", ".join(missing_bundle_files)
                    if missing_bundle_files
                    else "a passing QA status"
                )
            ),
            "required_resolution": (
                "complete the atomic input build and require QA, run manifest "
                "and source manifest before any solver result is admitted"
            ),
        },
    ]
    return pd.DataFrame(checks)


def write_decision_memo(
    annex: pd.DataFrame,
    post_eva: pd.DataFrame,
    resource: pd.DataFrame,
    dsr: pd.DataFrame,
    btm: pd.DataFrame,
    cost: pd.DataFrame,
    gates: pd.DataFrame,
    release_page: int,
    annex_page: int,
) -> None:
    current = post_eva.loc[
        post_eva["capacity_scope"].eq("current_adapter_included_statuses")
    ]
    negative = current.loc[current["post_eva_capacity_mw"].lt(-1e-9)]
    caes24 = cost.loc[cost["config_id"].eq("caes_24h")].iloc[0]
    market_battery = resource.loc[
        resource["resource"].eq("market_batteries_utility_plus_residential")
    ].iloc[0]
    hcl = resource.loc[
        resource["resource"].eq("closed_loop_pumped_hydro")
    ].iloc[0]
    memo = f"""# ERAA 2025 confirmatory-evidence admission audit

## Decision

**Not admitted for confirmatory manuscript evidence.** The current model can
remain a development calculation, but {int(gates['blocking'].sum())} blocking
gates are evaluated and {int((gates['blocking'] & ~gates['pass']).sum())}
remain open.

## Capacity provenance and amended EVA transfer

The official explanatory note describes `GenerationCapacities.csv` as the
known-development capacity set used as the *starting point* of the Economic
Viability Assessment (PDF page {release_page}). Therefore the string
`ERAA 2025 final` identifies the final released National Trends input, not a
post-EVA fleet. The amended Annex 3 Table 2 (starting on PDF page {annex_page})
contains 121 zone–technology decisions. Its 2035 changes reproduce the
published +14,630 MW grand total exactly.

The transfer is nevertheless not lossless. Directly adding the amended
decisions to the public broad-technology table produces {len(negative)}
negative arithmetic record:
`IE00 / Lignite = {negative['post_eva_capacity_mw'].iloc[0]:.0f} MW`. The
−110 MW retirement is present both in amended Annex 3 and in the EVA workbook,
where its finer class is `Lignite old 2`; the public 2035 broad table contains
0 MW of IE00 lignite. The negative number is therefore a mapping diagnostic,
not a physical post-EVA capacity. It shows that this finer EVA class/vintage
retirement cannot be losslessly assigned to the public broad class. The main
model must obtain an official EVA-class/vintage-to-public-technology crosswalk.
Until then it must quarantine this transfer and either exclude IE00 from
confirmatory attribution or demonstrate result invariance across explicit
bounded mappings. Clipping the result to zero is not acceptable.

Natural-gas changes are arithmetically mappable only to the aggregate
`Natural gas` class. The public capacity table does not identify CCGT and OCGT
shares, so any subsequent split remains a declared sensitivity rather than an
observed fleet composition.

## Existing flexibility and double counting

The public inputs contain
{market_battery['released_discharge_capacity_mw']:,.0f} MW of market battery
discharge capacity, {market_battery['released_charge_capacity_mw']:,.0f} MW of
charge capacity and {market_battery['released_energy_capacity_mwh']:,.0f} MWh
of energy capacity. These assets are absent from the current optimization.
The model also omits {hcl['released_discharge_capacity_mw']:,.0f} MW of
closed-loop pumped-hydro discharge capacity and represents open-loop pumping
as discharge-only hydro. Explicit DSR is omitted despite released activation
prices and maximum daily dispatch durations. Moreover,
{int((~dsr['within_rounding_tolerance_1mw']).sum())} zones do not reconcile
within 1 MW between the high-level DSR capacity table and the detailed file.

The BTM accounting identity is correct but incomplete. Released signed BTM
battery output is embedded in net load, so the corresponding out-of-market
residential battery capacity must not also enter wholesale storage as a second
copy. The current core does not add it, hence there is no present double count.
However, the official release note states that the price-responsive share of
household PV–battery systems, EVs and heat pumps is re-optimized through iDSR.
Those ratios and rescheduling constraints are absent from the current model.
LUG1 also reports 38 MW of BTM battery capacity but has no released BTM battery
profile in the acquired input package; treating it as zero is not admissible
without an explicit sample exclusion.

## Candidate-cost ledger

For all {len(cost)} candidate configurations, the current objective reconstructs
the source all-in annual reference cost to within
{cost['reference_cost_reconstruction_error_eur_per_kw_year'].abs().max():.3e}
EUR kW-1 yr-1. The reference power block is therefore paid once, not twice.

For 24 h CAES, the all-in reference cost is
{caes24['model_reference_annual_cost_eur_per_kw_discharge_year']:.2f}
EUR kW-1 yr-1 at the central conversion rate. The separate coefficient applied
to charging capacity above the 1.5:1 reference ratio is
{caes24['model_incremental_charge_power_cost_eur_per_kw_year']:.2f}
EUR kW-1 yr-1. That coefficient is the full bundled
turbine–compressor–balance-of-plant annual power block, not a sourced
compressor-only marginal expansion cost. It is defensible as a sensitivity or
upper-bound price, but not as the confirmatory central estimate. The central
case should either hold the source charge/discharge ratio fixed or use a
separately sourced marginal charging-interface cost.

## Admission rule

No ERAA result should enter a main-text claim until every blocking row in
`audit_eraa_admission_hard_gates.csv` passes. In particular, a successful LP
solve is not equivalent to evidence admission while existing storage, pumped
hydro, explicit DSR, the connected network and sequential outages remain
outside the model.
"""
    (OUT / "AUDIT_ERAA_ADMISSION_DECISION.md").write_text(memo, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    required = [
        GENERATION_CAPACITY,
        BATTERY_DETAILS,
        HYDRO_DETAILS,
        DSR_DETAILS,
        ANNEX3,
        RELEASE_NOTE,
        EVA_WORKBOOK,
        CATALOG,
        FULL_COST_TABLE,
        CURRENCY_LEDGER,
        INPUT_BUILDER,
        DISPATCH_BUILDER,
        ZONE_RUNNER,
        MODEL_CORE,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required audit inputs missing: {missing}")

    release_text = pdf_text(RELEASE_NOTE)
    release_needle = "GenerationCapacities.csv"
    start_needle = "starting point in Economic Viability Assessment"
    if release_needle not in release_text or start_needle not in release_text:
        raise ValueError("Capacity provenance statement not found in release note")
    release_page = pdf_page_for(release_text, release_needle)
    annex_text = pdf_text(ANNEX3)
    annex_page = pdf_page_for(
        annex_text, "Table 2: Capacity change proposed by EVA per study zone"
    )

    annex = parse_annex3_table2()
    generation = read_generation_capacity()
    post_eva = build_post_eva_capacity_audit(annex, generation)
    ie_unmapped = build_ie_unmapped_retirement_evidence(annex, generation)
    resource, dsr = build_resource_audit(generation)
    btm = build_btm_zone_audit(generation)
    cost = build_cost_conversion_audit()
    gates = build_hard_gates(post_eva, resource, dsr, btm, cost)

    outputs = {
        "audit_eraa_admission_annex3_2035_changes.csv": annex,
        "audit_eraa_admission_post_eva_capacity_check.csv": post_eva,
        "audit_eraa_admission_ie00_unmapped_retirement.csv": ie_unmapped,
        "audit_eraa_admission_existing_flex_resources.csv": resource,
        "audit_eraa_admission_dsr_reconciliation.csv": dsr,
        "audit_eraa_admission_btm_zone_check.csv": btm,
        "audit_eraa_admission_cost_conversion_53.csv": cost,
        "audit_eraa_admission_hard_gates.csv": gates,
    }
    for name, frame in outputs.items():
        frame.to_csv(OUT / name, index=False)

    write_decision_memo(
        annex,
        post_eva,
        resource,
        dsr,
        btm,
        cost,
        gates,
        release_page,
        annex_page,
    )

    current = post_eva.loc[
        post_eva["capacity_scope"].eq("current_adapter_included_statuses")
    ]
    qa = {
        "script": Path(__file__).name,
        "script_version": SCRIPT_VERSION,
        "audit_script_sha256": sha256(Path(__file__)),
        "target_year": TARGET_YEAR,
        "evidence_status": "independent_admission_audit_not_model_result",
        "source_files": {
            str(path.relative_to(BASE.parents[2])): sha256(path)
            for path in required
        },
        "official_source_evidence": {
            "generation_capacity_is_eva_starting_point": True,
            "release_note_pdf_page": release_page,
            "amended_annex3_table2_pdf_start_page": annex_page,
        },
        "annex3": {
            "zone_technology_decision_rows_2035": len(annex),
            "grand_total_capacity_change_mw": float(
                annex["capacity_change_mw"].sum()
            ),
            "mapped_zone_technology_rows_2035": int(
                annex.groupby(["zone", "mapped_capacity_technology"]).ngroups
            ),
        },
        "post_eva": {
            "negative_zone_technology_rows": int(
                current["post_eva_capacity_mw"].lt(-1e-9).sum()
            ),
            "all_mapped_classes": bool(
                annex["mapped_capacity_technology"].notna().all()
            ),
            "nonnegative_gate_pass": bool(
                current["post_eva_capacity_mw"].ge(-1e-9).all()
            ),
            "ie00_lignite_interpretation": (
                "unresolved cross-taxonomy EVA-class/vintage mapping; "
                "negative direct sum is a diagnostic, not a physical capacity"
            ),
        },
        "cost": {
            "configuration_count": len(cost),
            "reference_cost_max_abs_reconstruction_error_eur_per_kw_year": float(
                cost[
                    "reference_cost_reconstruction_error_eur_per_kw_year"
                ].abs().max()
            ),
            "reference_configuration_double_count_count": int(
                cost["reference_configuration_double_counted"].sum()
            ),
            "separately_sourced_incremental_charge_cost_count": int(
                cost["incremental_charge_component_separately_sourced"].sum()
            ),
        },
        "gates": {
            "blocking_gate_count": int(gates["blocking"].sum()),
            "blocking_gate_fail_count": int(
                (gates["blocking"] & ~gates["pass"]).sum()
            ),
            "confirmatory_evidence_admitted": bool(
                gates.loc[gates["blocking"], "pass"].all()
            ),
        },
    }
    qa_path = OUT / "audit_eraa_admission_qa.json"
    qa_path.write_text(
        json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
