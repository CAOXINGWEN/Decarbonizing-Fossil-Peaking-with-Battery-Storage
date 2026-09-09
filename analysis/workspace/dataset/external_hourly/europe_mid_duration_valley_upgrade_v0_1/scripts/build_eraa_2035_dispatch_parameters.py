#!/usr/bin/env python3
"""Build auditable ERAA 2025-final dispatch and network parameters for 2035.

The output is an adapter layer, not a model result.  It converts ERAA-final
installed capacities, official fuel/carbon assumptions, maintenance profiles,
forced-outage rates, derating factors and released NTC series into deterministic
tables consumed by the phase-1 solver.

ERAA publishes natural-gas and hydrogen capacity only as aggregate fuel groups.
The builder therefore writes three explicit CCGT/OCGT splits instead of
pretending that a plant-level split was observed.  The expected-availability
table is suitable for the first full weather screen.  Sequential outage paths
remain a preregistered sensitivity and must not be confused with this
expectation screen.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCRIPT_VERSION = "0.1.0"
TARGET_YEAR = 2035
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTRACTED = PROJECT_ROOT / "raw" / "eraa_2025" / "extracted"
DEFAULT_OUTPUT = PROJECT_ROOT / "processed" / "eraa_2035_dispatch_parameters_v0_1"

DEFAULT_CAPACITY = (
    EXTRACTED
    / "ERAA_2025_DashboardRawData1"
    / "Dashboard Raw Data"
    / "GenerationCapacities.csv"
)
DEFAULT_MAINTENANCE = (
    EXTRACTED
    / "ERAA_2025_DashboardRawData1"
    / "Dashboard Raw Data"
    / "Maintenance_profile.csv"
)
DEFAULT_COMMODITY = (
    EXTRACTED
    / "ERAA_2025_DashboardRawData1"
    / "Dashboard Raw Data"
    / "Commodity Prices.xlsx"
)
DEFAULT_FORCED_OUTAGE = EXTRACTED / "Other Data" / "Forced_outage_rate_extraction.csv"
DEFAULT_DERATING = EXTRACTED / "Other Data" / "Derating_data.csv"
DEFAULT_MUST_RUN = EXTRACTED / "Other Data" / "Must-run_data.csv"
DEFAULT_INVESTMENT = EXTRACTED / "Economic and technical investment parameters_Default.xlsx"
DEFAULT_NTC = EXTRACTED / "NTCs" / "NTCs Consolidated TY2035.xlsx"
DEFAULT_ANNEX_1 = (
    PROJECT_ROOT
    / "raw"
    / "eraa_2025"
    / "ERAA_2025_Annex_1_Input_Data_Assumptions_Amended.pdf"
)
DEFAULT_ANNEX_3 = (
    PROJECT_ROOT
    / "raw"
    / "eraa_2025"
    / "ERAA_2025_Annex_3_Detailed_Results_Amended.pdf"
)
DEFAULT_EVA = PROJECT_ROOT / "raw" / "eraa_2025" / "ERAA_2025_EVA.xlsx"

INCLUDED_STATUS = frozenset(
    {
        "available on market",
        "inelastic supply / fixed profile",
        "out of market - for pv/battery dispatch optimization",
    }
)
EXCLUDED_AGGREGATE_ZONES = frozenset({"DK00", "ISEM", "IT00", "LU00", "NO00", "SE00"})

THERMAL_TECHNOLOGIES = (
    "gas_ccgt",
    "gas_ocgt",
    "hydrogen_ccgt",
    "hydrogen_ocgt",
    "nuclear",
    "hard_coal",
    "lignite",
    "light_oil",
    "heavy_oil",
)
CAPACITY_TECHNOLOGY_MAP = {
    "Nuclear": "nuclear",
    "Hard coal": "hard_coal",
    "Lignite": "lignite",
    "Light oil": "light_oil",
    "Heavy oil": "heavy_oil",
}
GROUP_BY_TECHNOLOGY = {
    "gas_ccgt": "Gas",
    "gas_ocgt": "Gas",
    "hydrogen_ccgt": "Hydrogen",
    "hydrogen_ocgt": "Hydrogen",
    "nuclear": "Nuclear",
    "hard_coal": "Coal",
    "lignite": "Lignite",
    "light_oil": "Oil",
    "heavy_oil": "Oil",
}
MAINTENANCE_LABEL = {
    "gas_ccgt": "Gas",
    "gas_ocgt": "Gas",
    "hydrogen_ccgt": "Hydrogen",
    "hydrogen_ocgt": "Hydrogen",
    "nuclear": "Nuclear",
    "hard_coal": "Hard coal",
    "lignite": "Lignite",
    "light_oil": "Light oil",
    "heavy_oil": "Heavy oil",
}
DEFAULT_FOR = {
    "gas_ccgt": 0.05,
    "gas_ocgt": 0.08,
    "hydrogen_ccgt": 0.05,
    "hydrogen_ocgt": 0.08,
    "nuclear": 0.05,
    "hard_coal": 0.10,
    "lignite": 0.10,
    "light_oil": 0.08,
    "heavy_oil": 0.08,
}
SPLIT_CASES = {
    "all_ccgt": (1.0, 0.0),
    "equal_ccgt_ocgt": (0.5, 0.5),
    "all_ocgt": (0.0, 1.0),
}
EVA_AUDIT_CASES = ("cost_based", "revenue_based_A", "revenue_based_B")
MODEL_CAPACITY_CASES = ("pre_eva_pending_unit_transfer",)
ANNEX3_CAPACITY_CASE = "amended_annex3_risk_averse_enhanced_hurdle"
ANNEX3_SPLIT_CASES = (
    "retirement_feasible_max_ccgt",
    "retirement_feasible_equal_residual",
    "retirement_feasible_max_ocgt",
)
ANNEX3_TECHNOLOGY_MAP = {
    "Gas CCGT": "gas_ccgt",
    "Gas OCGT": "gas_ocgt",
    "Hydrogen CCGT": "hydrogen_ccgt",
    "Hard coal": "hard_coal",
    "Lignite": "lignite",
    "Light oil": "light_oil",
    "Heavy oil": "heavy_oil",
}

# ERAA 2025 amended Annex 1 / released Commodity Prices workbook, 2035,
# expressed in constant 2024 euros.
OFFICIAL_2035_PRICES = {
    "gas_blend_eur2024_per_gj": 10.602895,
    "natural_gas_eur2024_per_gj": 9.798326,
    "hydrogen_blue_eur2024_per_gj": 17.837831,
    "hard_coal_eur2024_per_gj": 4.001410,
    "light_oil_eur2024_per_gj": 19.502259,
    "heavy_oil_eur2024_per_gj": 15.997947,
    "nuclear_eur2024_per_gj": 0.607745,
    "carbon_eur2024_per_t": 197.505516,
}
LIGNITE_PRICE_BY_GROUP = {
    "G1": 1.859005,
    "G2": 2.390149,
    "G3": 3.147029,
    "G4": 4.116368,
}
LIGNITE_GROUP_ZONES = {
    "G1": {"BG00", "MK00", "CZ00"},
    "G2": {"SK00", "DE00", "RS00", "PL00", "ME00", "UKNI", "BA00", "IE00"},
    "G3": {"SI00", "RO00", "HU00"},
    "G4": {"GR00", "GR03", "TR00"},
}

# Efficiency and emissions are the released Common Data central categories.
# VOM uses the 2035 default investment workbook where an exact corresponding
# row exists; otherwise the Common Data value is retained.
TECHNICAL_ASSUMPTIONS = {
    "gas_ccgt": (0.60, 57.0, 2.323896, "Gas blend"),
    "gas_ocgt": (0.42, 57.0, 2.462406, "Gas blend"),
    "hydrogen_ccgt": (0.60, 0.0, 2.323896, "Hydrogen"),
    "hydrogen_ocgt": (0.42, 0.0, 2.462406, "Hydrogen"),
    "nuclear": (0.33, 0.0, 8.208021, "Nuclear"),
    "hard_coal": (0.40, 94.0, 2.859675, "Hard coal"),
    "lignite": (0.40, 101.0, 3.574593, "Lignite"),
    "light_oil": (0.35, 78.0, 3.283209, "Light oil"),
    "heavy_oil": (0.40, 78.0, 3.288646, "Heavy oil"),
}

# ECB series EXR.A.CNY.EUR.SP00.A, annual average, 2024.
ECB_2024_CNY_PER_EUR = 7.787469921875
ECB_SERIES_URL = (
    "https://data-api.ecb.europa.eu/service/data/EXR/"
    "A.CNY.EUR.SP00.A?startPeriod=2024&endPeriod=2024&format=csvdata"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_status(value: object) -> str:
    text = str(value).strip().lower().replace("\u2013", "-").replace("\u2014", "-")
    return " ".join(text.split())


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(json_ready(payload), indent=2, sort_keys=True, ensure_ascii=False)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.write("\n")
    os.replace(temporary, path)


def write_csv(frame: pd.DataFrame, path: Path, *, compressed: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as raw:
        temporary = Path(raw.name)
        if compressed:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as gz:
                with io.TextIOWrapper(gz, encoding="utf-8", newline="") as text:
                    frame.to_csv(text, index=False, lineterminator="\n", float_format="%.10g")
        else:
            with io.TextIOWrapper(raw, encoding="utf-8", newline="") as text:
                frame.to_csv(text, index=False, lineterminator="\n", float_format="%.10g")
    os.replace(temporary, path)


def prepare_output(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"output exists: {path}; pass --force")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def read_capacity(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "data_version",
        "Target year",
        "Market_Node",
        "Technology",
        "Operational_Status",
        "Value",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing {missing}")
    selected = frame.loc[
        frame["data_version"].eq("ERAA 2025 final")
        & pd.to_numeric(frame["Target year"], errors="coerce").eq(TARGET_YEAR)
    ].copy()
    selected["Value"] = pd.to_numeric(selected["Value"], errors="raise")
    selected["status_normalized"] = selected["Operational_Status"].map(normalize_status)
    selected["included_energy_market"] = selected["status_normalized"].isin(INCLUDED_STATUS)
    if selected.empty or selected["Value"].lt(0).any():
        raise ValueError(f"{path}: invalid ERAA-final 2035 capacity")
    forbidden = EXCLUDED_AGGREGATE_ZONES.intersection(selected["Market_Node"].astype(str))
    if forbidden:
        raise ValueError(f"aggregate capacity zones present: {sorted(forbidden)}")
    return selected


def normalize_eva_technology(value: object) -> str:
    technology = str(value).strip()
    lower = technology.lower()
    if lower.startswith("hydrogen "):
        return "Hydrogen"
    if lower.startswith("gas "):
        return "Natural gas"
    if lower.startswith("hard coal"):
        return "Hard coal"
    if lower.startswith("lignite"):
        return "Lignite"
    if lower.startswith("light oil"):
        return "Light oil"
    if lower.startswith("heavy oil"):
        return "Heavy oil"
    if lower.startswith("battery"):
        return "Battery utility scale"
    if lower.startswith("industry_dsr") or technology == "DSR":
        return "DSR"
    return technology


def read_eva_changes(path: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    cost = pd.read_excel(path, sheet_name="Cost Based")
    cost = cost.loc[pd.to_numeric(cost["Year"], errors="coerce").eq(TARGET_YEAR)].copy()
    cost["eva_case"] = "cost_based"
    rows.append(cost)
    revenue = pd.read_excel(path, sheet_name="Revenue Based")
    revenue = revenue.loc[
        pd.to_numeric(revenue["Year"], errors="coerce").eq(TARGET_YEAR)
    ].copy()
    revenue["eva_case"] = revenue["Model version"].map(
        {
            "Implementation A": "revenue_based_A",
            "Implementation B": "revenue_based_B",
        }
    )
    if revenue["eva_case"].isna().any():
        raise ValueError(f"{path}: unknown revenue-based EVA implementation")
    rows.append(revenue)
    combined = pd.concat(rows, ignore_index=True)
    combined["capacity_technology"] = combined["Technology"].map(
        normalize_eva_technology
    )
    combined["Capacity Change"] = pd.to_numeric(
        combined["Capacity Change"], errors="raise"
    )
    return (
        combined.groupby(
            ["eva_case", "Node", "capacity_technology"], as_index=False
        )["Capacity Change"]
        .sum()
        .rename(
            columns={
                "Node": "zone",
                "Capacity Change": "eva_capacity_change_mw",
            }
        )
    )


def read_amended_annex3_changes(path: Path) -> pd.DataFrame:
    """Parse amended Annex 3 Table 2 without relying on the superseded workbook."""

    completed = subprocess.run(
        ["pdftotext", "-layout", str(path), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = completed.stdout.splitlines()
    try:
        start = next(
            index
            for index, line in enumerate(lines)
            if "Table 2: Capacity change proposed by EVA per study zone" in line
        )
        stop = next(
            index
            for index, line in enumerate(lines[start:], start)
            if "Grand Total" in line
        )
    except StopIteration as error:
        raise ValueError(f"{path}: amended Annex 3 Table 2 not found") from error

    descriptor = re.compile(
        r"^(?:(?P<zone>[A-Z][A-Z0-9]{3,4})\s+)?"
        r"(?P<technology>.*?)\s+"
        r"(?P<decision>Life Extension|Retirement|Expansion)"
        r"(?P<tail>.*)$"
    )
    number = re.compile(r"[-+]?\s*\d[\d,]*")

    def values_from(text: str) -> list[int]:
        normalized = text.replace("- ", "-")
        tokens = number.findall(normalized)
        return [
            int(token.replace(" ", "").replace(",", ""))
            for token in tokens
        ]

    rows: list[dict[str, Any]] = []
    zone: str | None = None
    table = lines[start:stop]
    index = 0
    while index < len(table):
        line = table[index].replace("\f", "").strip()
        match = descriptor.match(line)
        if (
            match is None
            or not match.group("technology").strip()
            or "Decision" in line
        ):
            index += 1
            continue
        if match.group("zone"):
            zone = match.group("zone")
        if zone is None:
            raise ValueError(f"{path}: technology row precedes a study zone")
        values = values_from(match.group("tail"))
        consumed = index
        if len(values) != 4:
            values = []
            for probe in range(index + 1, min(index + 5, len(table))):
                candidate = values_from(table[probe].strip())
                if len(candidate) == 4:
                    values = candidate
                    consumed = probe
                    break
                if descriptor.match(table[probe].strip()):
                    break
        if len(values) != 4:
            raise ValueError(
                f"{path}: four target-year values not found for "
                f"{zone}/{match.group('technology').strip()}/"
                f"{match.group('decision')}"
            )
        for year, value in zip((2028, 2030, 2033, 2035), values):
            rows.append(
                {
                    "source_case": ANNEX3_CAPACITY_CASE,
                    "zone": zone,
                    "pemmbd_technology": match.group("technology").strip(),
                    "decision_variable": match.group("decision"),
                    "target_year": year,
                    "capacity_change_mw": float(value),
                    "source": (
                        "ERAA_2025_Annex_3_Detailed_Results_Amended_"
                        "Table_2"
                    ),
                }
            )
        index = consumed + 1

    result = pd.DataFrame(rows)
    selected = result.loc[result["target_year"].eq(TARGET_YEAR)]
    if len(result) != 121 * 4:
        raise ValueError(
            f"{path}: expected 121 Table-2 decision rows x four years, "
            f"observed {len(result)} long rows"
        )
    if abs(float(selected["capacity_change_mw"].sum()) - 14_630.0) > 1e-8:
        raise ValueError(
            f"{path}: 2035 grand-total reconciliation failed; observed "
            f"{selected['capacity_change_mw'].sum():g} MW"
        )
    de = selected.loc[selected["zone"].eq("DE00")].set_index(
        ["pemmbd_technology", "decision_variable"]
    )["capacity_change_mw"]
    required_de = {
        ("DSR", "Expansion"): 820.0,
        ("Gas CCGT", "Life Extension"): 180.0,
        ("Gas OCGT", "Expansion"): 23_140.0,
        ("Gas OCGT", "Life Extension"): 1_060.0,
    }
    for key, expected in required_de.items():
        if key not in de.index or float(de.loc[key]) != expected:
            raise ValueError(f"{path}: DE00 cross-check failed for {key}")
    return result


def build_annex3_capacity_cases(
    capacity: pd.DataFrame,
    annex3_long: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply official 2035 changes using retirement-feasible fuel splits.

    ERAA releases the National Trends starting fleet only as aggregate natural
    gas and hydrogen capacity.  Annex 3 reports CCGT/OCGT decisions
    separately.  The primary split therefore assigns the minimum pre-EVA
    OCGT/CCGT capacity required to make every reported retirement feasible and
    allocates the residual gas capacity to CCGT.  Equal-residual and
    maximum-OCGT allocations are written as sensitivities.
    """

    included = capacity.loc[capacity["included_energy_market"]].copy()
    zones = sorted(included["Market_Node"].astype(str).unique())
    base = (
        included.groupby(["Market_Node", "Technology"], as_index=False)["Value"]
        .sum()
        .rename(
            columns={
                "Market_Node": "zone",
                "Technology": "capacity_technology",
                "Value": "source_capacity_mw",
            }
        )
    )
    changes = annex3_long.loc[annex3_long["target_year"].eq(TARGET_YEAR)].copy()
    changes["model_technology"] = changes["pemmbd_technology"].map(
        ANNEX3_TECHNOLOGY_MAP
    )
    thermal_changes = (
        changes.loc[changes["model_technology"].notna()]
        .groupby(
            ["zone", "model_technology", "decision_variable"], as_index=False
        )["capacity_change_mw"]
        .sum()
    )
    nonthermal = changes.loc[
        ~changes["pemmbd_technology"].isin(ANNEX3_TECHNOLOGY_MAP)
    ].copy()

    output_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    residual_share = {
        "retirement_feasible_max_ccgt": 1.0,
        "retirement_feasible_equal_residual": 0.5,
        "retirement_feasible_max_ocgt": 0.0,
    }

    for split_case in ANNEX3_SPLIT_CASES:
        ccgt_residual_share = residual_share[split_case]
        split_rows: list[dict[str, Any]] = []
        for zone in zones:
            one_base = base.loc[base["zone"].eq(zone)].set_index(
                "capacity_technology"
            )["source_capacity_mw"]
            one_change = thermal_changes.loc[thermal_changes["zone"].eq(zone)]
            change_by_tech = (
                one_change.groupby("model_technology")["capacity_change_mw"]
                .sum()
                .to_dict()
            )
            retirement = (
                one_change.loc[
                    one_change["decision_variable"].eq("Retirement")
                ]
                .groupby("model_technology")["capacity_change_mw"]
                .sum()
                .to_dict()
            )
            gas_total = float(one_base.get("Natural gas", 0.0))
            minimum_ccgt = max(0.0, -float(retirement.get("gas_ccgt", 0.0)))
            minimum_ocgt = max(0.0, -float(retirement.get("gas_ocgt", 0.0)))
            residual_gas = gas_total - minimum_ccgt - minimum_ocgt
            if residual_gas < -1e-8:
                conflict_rows.append(
                    {
                        "zone": zone,
                        "split_case": split_case,
                        "capacity_group": "Natural gas",
                        "source_capacity_mw": gas_total,
                        "minimum_retirement_feasible_capacity_mw": (
                            minimum_ccgt + minimum_ocgt
                        ),
                        "shortfall_mw": -residual_gas,
                        "blocking_conflict": True,
                        "resolution": "unresolved; split case excluded",
                        "conflict": (
                            "reported CCGT+OCGT retirements exceed released "
                            "aggregate natural-gas starting capacity"
                        ),
                    }
                )
                residual_gas = 0.0
            pre_values = {
                "gas_ccgt": (
                    minimum_ccgt + max(residual_gas, 0.0) * ccgt_residual_share
                ),
                "gas_ocgt": (
                    minimum_ocgt
                    + max(residual_gas, 0.0) * (1.0 - ccgt_residual_share)
                ),
            }
            hydrogen_total = float(one_base.get("Hydrogen", 0.0))
            pre_values["hydrogen_ccgt"] = (
                hydrogen_total * ccgt_residual_share
            )
            pre_values["hydrogen_ocgt"] = (
                hydrogen_total * (1.0 - ccgt_residual_share)
            )
            direct_uplift: dict[str, float] = {}
            for source, target in CAPACITY_TECHNOLOGY_MAP.items():
                released = float(one_base.get(source, 0.0))
                retirement_minimum = max(
                    0.0, -float(retirement.get(target, 0.0))
                )
                inferred = max(released, retirement_minimum)
                pre_values[target] = inferred
                direct_uplift[target] = inferred - released
                if inferred > released + 1e-8:
                    conflict_rows.append(
                        {
                            "zone": zone,
                            "split_case": split_case,
                            "capacity_group": target,
                            "source_capacity_mw": released,
                            "minimum_retirement_feasible_capacity_mw": (
                                retirement_minimum
                            ),
                            "shortfall_mw": inferred - released,
                            "blocking_conflict": False,
                            "resolution": (
                                "infer the official retirement amount as a "
                                "unit-level pre-EVA lower bound; retain the "
                                "published aggregate mismatch in this audit"
                            ),
                            "conflict": (
                                "Annex 3 retirement exceeds the released "
                                "broad starting-capacity row"
                            ),
                        }
                    )

            for technology in THERMAL_TECHNOLOGIES:
                starting = float(pre_values.get(technology, 0.0))
                change = float(change_by_tech.get(technology, 0.0))
                post = starting + change
                negative = post < -1e-8
                audit = {
                    "zone": zone,
                    "eva_case": ANNEX3_CAPACITY_CASE,
                    "split_case": split_case,
                    "technology": technology,
                    "pre_eva_inferred_capacity_mw": starting,
                    "annex3_net_capacity_change_mw": change,
                    "post_eva_capacity_mw": post,
                    "negative_post_capacity_flag": negative,
                    "retirement_feasibility_uplift_mw": float(
                        direct_uplift.get(technology, 0.0)
                    ),
                    "mapping_basis": (
                        "minimum technology-specific retirement capacity plus "
                        f"{ccgt_residual_share:g} CCGT share of residual fuel "
                        "aggregate"
                    ),
                }
                audit_rows.append(audit)
                if negative:
                    conflict_rows.append(
                        {
                            "zone": zone,
                            "split_case": split_case,
                            "capacity_group": technology,
                            "source_capacity_mw": starting,
                            "minimum_retirement_feasible_capacity_mw": -change,
                            "shortfall_mw": -post,
                            "blocking_conflict": True,
                            "resolution": "unresolved; split case excluded",
                            "conflict": "post-EVA technology capacity is negative",
                        }
                    )
                split_rows.append(
                    {
                        "target_year": TARGET_YEAR,
                        "zone": zone,
                        "eva_case": ANNEX3_CAPACITY_CASE,
                        "split_case": split_case,
                        "technology": technology,
                        "capacity_mw": post,
                        "capacity_basis": (
                            "amended_Annex3_Table2_plus_retirement_feasible_"
                            "split_of_released_fuel_aggregate"
                        ),
                    }
                )
        split_conflict = any(
            row["split_case"] == split_case
            and bool(row["blocking_conflict"])
            for row in conflict_rows
        )
        if not split_conflict:
            output_rows.extend(split_rows)

    return (
        pd.DataFrame(output_rows),
        pd.DataFrame(audit_rows),
        pd.DataFrame(conflict_rows),
        nonthermal,
    )


def build_capacity_splits(
    capacity: pd.DataFrame,
    eva_changes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    included = capacity.loc[capacity["included_energy_market"]].copy()
    zones = sorted(included["Market_Node"].astype(str).unique())
    raw = (
        included.groupby(["Market_Node", "Technology"], as_index=False)["Value"]
        .sum()
        .rename(columns={"Market_Node": "zone", "Value": "source_capacity_mw"})
    )
    reconciliation_rows: list[dict[str, Any]] = []
    pre_keys = raw.rename(columns={"Technology": "capacity_technology"})
    all_keys = pd.MultiIndex.from_product(
        [EVA_AUDIT_CASES, zones], names=["eva_case", "zone"]
    ).to_frame(index=False)
    technologies = sorted(
        set(pre_keys["capacity_technology"].astype(str))
        | set(eva_changes["capacity_technology"].astype(str))
    )
    all_keys = all_keys.merge(
        pd.DataFrame({"capacity_technology": technologies}), how="cross"
    )
    all_keys = all_keys.merge(
        pre_keys[["zone", "capacity_technology", "source_capacity_mw"]],
        on=["zone", "capacity_technology"],
        how="left",
    ).merge(
        eva_changes,
        on=["eva_case", "zone", "capacity_technology"],
        how="left",
    )
    all_keys[["source_capacity_mw", "eva_capacity_change_mw"]] = all_keys[
        ["source_capacity_mw", "eva_capacity_change_mw"]
    ].fillna(0.0)
    all_keys["post_eva_capacity_mw"] = (
        all_keys["source_capacity_mw"] + all_keys["eva_capacity_change_mw"]
    )
    all_keys["broad_addition_negative_flag"] = all_keys[
        "post_eva_capacity_mw"
    ].lt(-1e-5)
    all_keys["broad_addition_valid_for_model"] = False
    all_keys["audit_interpretation"] = (
        "EVA decisions act on the unit-level eligible fleet and cannot be "
        "added directly to Dashboard broad capacity totals"
    )
    reconciliation_rows.extend(all_keys.to_dict("records"))

    rows: list[dict[str, Any]] = []
    for eva_case in MODEL_CAPACITY_CASES:
        post = pre_keys.rename(
            columns={"source_capacity_mw": "post_eva_capacity_mw"}
        )
        for split_case, (ccgt_share, ocgt_share) in SPLIT_CASES.items():
            for zone in zones:
                one = post.loc[post["zone"].eq(zone)].set_index(
                    "capacity_technology"
                )["post_eva_capacity_mw"]
                gas = float(one.get("Natural gas", 0.0))
                hydrogen = float(one.get("Hydrogen", 0.0))
                values = {
                    "gas_ccgt": gas * ccgt_share,
                    "gas_ocgt": gas * ocgt_share,
                    # Hydrogen split is held equal to the natural-gas
                    # sensitivity because ERAA releases only a fuel aggregate.
                    "hydrogen_ccgt": hydrogen * ccgt_share,
                    "hydrogen_ocgt": hydrogen * ocgt_share,
                }
                for source, target in CAPACITY_TECHNOLOGY_MAP.items():
                    values[target] = float(one.get(source, 0.0))
                for technology in THERMAL_TECHNOLOGIES:
                    rows.append(
                        {
                            "target_year": TARGET_YEAR,
                            "zone": zone,
                            "eva_case": eva_case,
                            "split_case": split_case,
                            "technology": technology,
                            "capacity_mw": values.get(technology, 0.0),
                            "capacity_basis": (
                                "pre_EVA_ERAA_aggregate_pending_unit_transfer"
                                if technology not in {
                                    "gas_ccgt",
                                    "gas_ocgt",
                                    "hydrogen_ccgt",
                                    "hydrogen_ocgt",
                                }
                                else "sensitivity_split_of_ERAA_fuel_aggregate"
                            ),
                        }
                    )
    exclusions = included.loc[
        ~included["Technology"].isin(
            {
                "Natural gas",
                "Hydrogen",
                *CAPACITY_TECHNOLOGY_MAP.keys(),
            }
        ),
        ["Market_Node", "Technology", "Value", "Operational_Status"],
    ].copy()
    exclusions = exclusions.rename(columns={"Market_Node": "zone", "Value": "capacity_mw"})
    exclusions["phase1_treatment"] = "excluded_from_thermal_dispatch_adapter"
    exclusions["reason"] = (
        "hydro_storage_DSR_electrolyser_or_profiled_resource_requires_"
        "separate_model_component"
    )
    return pd.DataFrame(rows), exclusions, pd.DataFrame(reconciliation_rows)


def lignite_price(zone: str) -> tuple[float, str]:
    for group, zones in LIGNITE_GROUP_ZONES.items():
        if zone in zones:
            return LIGNITE_PRICE_BY_GROUP[group], group
    # Lignite capacity is absent outside the released groups; this fallback is
    # written to the ledger and cannot silently affect a nonzero capacity.
    return LIGNITE_PRICE_BY_GROUP["G2"], "G2_fallback_zero_capacity_only"


def build_costs(zones: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    carbon = OFFICIAL_2035_PRICES["carbon_eur2024_per_t"]
    for zone in sorted(set(zones)):
        for technology in THERMAL_TECHNOLOGIES:
            efficiency, co2_kg_gj, vom, fuel_label = TECHNICAL_ASSUMPTIONS[technology]
            if technology.startswith("gas_"):
                fuel = OFFICIAL_2035_PRICES["gas_blend_eur2024_per_gj"]
                fuel_key = "Gas blend"
            elif technology.startswith("hydrogen_"):
                fuel = OFFICIAL_2035_PRICES["hydrogen_blue_eur2024_per_gj"]
                fuel_key = "Hydrogen (Blue)"
            elif technology == "hard_coal":
                fuel = OFFICIAL_2035_PRICES["hard_coal_eur2024_per_gj"]
                fuel_key = "Hard coal"
            elif technology == "light_oil":
                fuel = OFFICIAL_2035_PRICES["light_oil_eur2024_per_gj"]
                fuel_key = "Light oil"
            elif technology == "heavy_oil":
                fuel = OFFICIAL_2035_PRICES["heavy_oil_eur2024_per_gj"]
                fuel_key = "Heavy oil"
            elif technology == "nuclear":
                fuel = OFFICIAL_2035_PRICES["nuclear_eur2024_per_gj"]
                fuel_key = "Nuclear"
            else:
                fuel, group = lignite_price(zone)
                fuel_key = f"Lignite {group}"
            fuel_cost = fuel * 3.6 / efficiency
            carbon_cost = co2_kg_gj * 3.6 / efficiency / 1_000.0 * carbon
            rows.append(
                {
                    "target_year": TARGET_YEAR,
                    "currency_basis": "EUR2024",
                    "zone": zone,
                    "technology": technology,
                    "fuel_label": fuel_key,
                    "fuel_price_eur2024_per_gj": fuel,
                    "efficiency_ncv": efficiency,
                    "co2_factor_kg_per_net_gj": co2_kg_gj,
                    "carbon_price_eur2024_per_t": carbon,
                    "fuel_cost_eur2024_per_mwh": fuel_cost,
                    "carbon_cost_eur2024_per_mwh": carbon_cost,
                    "variable_om_eur2024_per_mwh": vom,
                    "marginal_cost_eur2024_per_mwh": fuel_cost + carbon_cost + vom,
                    "source": "ERAA_2025_amended_Annex1_and_released_workbooks",
                }
            )
    return pd.DataFrame(rows)


def read_group_value(
    frame: pd.DataFrame,
    zone: str,
    group: str,
    *,
    zone_column: str,
    group_column: str,
    value_column: str,
    default: float,
) -> tuple[float, str]:
    match = frame.loc[
        frame[zone_column].astype(str).eq(zone)
        & frame[group_column].astype(str).eq(group)
    ]
    if match.empty:
        return float(default), "Common_Data_default"
    values = pd.to_numeric(match[value_column], errors="raise")
    if len(values) != 1:
        raise ValueError(f"duplicate {group} parameter for {zone}")
    return float(values.iloc[0]), "ERAA_zone_weighted"


def build_expected_availability(
    capacity_splits: pd.DataFrame,
    maintenance_path: Path,
    forced_path: Path,
    derating_path: Path,
) -> pd.DataFrame:
    maintenance = pd.read_csv(maintenance_path)
    maintenance["timestamp"] = pd.to_datetime(
        maintenance["Date"], format="%d/%m/%Y", errors="raise"
    )
    maintenance = maintenance.loc[maintenance["timestamp"].dt.year.eq(TARGET_YEAR)].copy()
    maintenance["day_of_year"] = maintenance["timestamp"].dt.dayofyear.astype(int) - 1
    maintenance["maintenance_mw"] = (
        pd.to_numeric(maintenance["Maintenance (GW)"], errors="raise") * 1_000.0
    )
    forced = pd.read_csv(forced_path)
    forced = forced.loc[
        pd.to_numeric(forced["TARGET_YEAR"], errors="coerce").eq(TARGET_YEAR)
    ].copy()
    derating = pd.read_csv(derating_path)
    derating = derating.loc[
        pd.to_numeric(derating["TARGET_YEAR"], errors="coerce").eq(TARGET_YEAR)
    ].copy()

    rows: list[dict[str, Any]] = []
    for record in capacity_splits.itertuples(index=False):
        group = GROUP_BY_TECHNOLOGY[record.technology]
        label = MAINTENANCE_LABEL[record.technology]
        forced_rate, forced_source = read_group_value(
            forced,
            record.zone,
            group,
            zone_column="MARKET_NODE",
            group_column="Technology",
            value_column="Weighted Average Forced Outage Rate",
            default=DEFAULT_FOR[record.technology],
        )
        derating_ratio, derating_source = read_group_value(
            derating,
            record.zone,
            group,
            zone_column="MARKET_NODE",
            group_column="PEMMDB_FUEL_TYPE",
            value_column="Weighted Average Derating Ratio",
            default=1.0,
        )
        if not (0 <= forced_rate < 1 and 0 <= derating_ratio <= 1):
            raise ValueError(f"invalid availability inputs for {record.zone}/{record.technology}")

        group_rows = capacity_splits.loc[
            capacity_splits["eva_case"].eq(record.eva_case)
            &
            capacity_splits["split_case"].eq(record.split_case)
            & capacity_splits["zone"].eq(record.zone)
            & capacity_splits["technology"].map(GROUP_BY_TECHNOLOGY).eq(group)
        ]
        group_capacity = float(group_rows["capacity_mw"].sum())
        share = float(record.capacity_mw / group_capacity) if group_capacity > 0 else 0.0
        one_maintenance = (
            maintenance.loc[
                maintenance["Bidding Zone"].astype(str).eq(record.zone)
                & maintenance["Technology"].astype(str).eq(label),
                ["day_of_year", "maintenance_mw"],
            ]
            .groupby("day_of_year", as_index=False)["maintenance_mw"]
            .sum()
            .set_index("day_of_year")["maintenance_mw"]
        )
        for day in range(365):
            allocated = float(one_maintenance.get(day, 0.0)) * share
            remaining = max(float(record.capacity_mw) - allocated, 0.0)
            fraction = (
                remaining / float(record.capacity_mw) * derating_ratio * (1.0 - forced_rate)
                if record.capacity_mw > 0
                else 0.0
            )
            rows.append(
                {
                    "target_year": TARGET_YEAR,
                    "zone": record.zone,
                    "eva_case": record.eva_case,
                    "split_case": record.split_case,
                    "technology": record.technology,
                    "day_of_year": day,
                    "capacity_mw": record.capacity_mw,
                    "maintenance_mw_allocated": allocated,
                    "derating_ratio": derating_ratio,
                    "derating_source": derating_source,
                    "forced_outage_rate": forced_rate,
                    "forced_outage_source": forced_source,
                    "expected_available_fraction": fraction,
                    "availability_mode": "maintenance_x_derating_x_expected_FOR",
                }
            )
    result = pd.DataFrame(rows)
    if not result["expected_available_fraction"].between(0, 1).all():
        raise ValueError("expected availability outside [0,1]")
    return result


def parse_investment_row(path: Path, sheet: str, technology: str) -> float:
    frame = pd.read_excel(path, sheet_name=sheet)
    reference_column = "Reference Technology"
    match = frame.loc[
        frame["PEMMDB Technology"].astype(str).eq(technology)
        & frame[reference_column].astype(str).eq("Expansion")
    ]
    if len(match) != 1:
        raise ValueError(f"{path}/{sheet}: expected one {technology} expansion row")
    year_column = TARGET_YEAR if TARGET_YEAR in match.columns else str(TARGET_YEAR)
    return float(pd.to_numeric(match.iloc[0][year_column], errors="raise"))


def build_firm_costs(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, technology in (("gas_ccgt_new", "Gas CCGT new"), ("gas_ocgt_new", "Gas OCGT new")):
        capex = parse_investment_row(path, "CAPEX", technology)
        fom = parse_investment_row(path, "FOM", technology)
        lifetime = parse_investment_row(path, "Economic Lifetime", technology)
        wacc = parse_investment_row(path, "WACC", technology)
        crf = wacc * (1 + wacc) ** lifetime / ((1 + wacc) ** lifetime - 1)
        rows.append(
            {
                "target_year": TARGET_YEAR,
                "currency_basis": "EUR2024",
                "firm_addition_technology": label,
                "capex_eur2024_per_kw": capex,
                "fixed_om_eur2024_per_kw_year": fom,
                "economic_lifetime_years": lifetime,
                "wacc_real": wacc,
                "capital_recovery_factor": crf,
                "annual_cost_eur2024_per_kw_year": capex * crf + fom,
                "source": "ERAA_2025_default_economic_and_technical_investment_parameters",
            }
        )
    return pd.DataFrame(rows)


def read_ntc_sheet(path: Path, sheet: str) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name=sheet, header=None)
    if sheet not in {"HVAC", "HVDC"}:
        raise ValueError("read_ntc_sheet accepts HVAC or HVDC")
    if len(frame) < 16 + 8_760:
        raise ValueError(f"{path}/{sheet}: truncated NTC sheet")
    metadata = frame.iloc[:16]
    data = frame.iloc[16 : 16 + 8_760]
    dates = data.iloc[:, 0].astype(str)
    hours = pd.to_numeric(data.iloc[:, 1], errors="raise").astype(int)
    if not hours.between(1, 24).all():
        raise ValueError(f"{path}/{sheet}: hour outside 1..24")
    rows: list[pd.DataFrame] = []
    for column in range(2, frame.shape[1]):
        edge_id = str(metadata.iat[7, column]).strip()
        from_zone = str(metadata.iat[8, column]).strip()
        to_zone = str(metadata.iat[9, column]).strip()
        if (
            not edge_id
            or edge_id.lower() == "nan"
            or not from_zone
            or from_zone.lower() == "nan"
            or not to_zone
            or to_zone.lower() == "nan"
        ):
            continue
        capacity = pd.to_numeric(data.iloc[:, column], errors="coerce")
        if capacity.isna().all():
            continue
        if capacity.isna().any() or capacity.lt(0).any():
            raise ValueError(f"{path}/{sheet}/{edge_id}: invalid hourly NTC")
        forced_rate = pd.to_numeric(
            pd.Series([metadata.iat[12, column]]), errors="coerce"
        ).iloc[0]
        forced_rate = float(forced_rate) if pd.notna(forced_rate) else 0.0
        one = pd.DataFrame(
            {
                "target_year": TARGET_YEAR,
                "network_technology": sheet,
                "edge_id": edge_id,
                "from_zone": from_zone,
                "to_zone": to_zone,
                "date_ddmm": dates.to_numpy(),
                "hour": hours.to_numpy() - 1,
                "hour_of_year": np.arange(8_760, dtype=np.int16),
                "forced_outage_rate_pct": forced_rate,
                "for_direction": str(metadata.iat[14, column]).strip(),
                "capacity_mw_raw": capacity.to_numpy(dtype=float),
                "capacity_mw_expected_after_for": (
                    capacity.to_numpy(dtype=float) * (1.0 - forced_rate / 100.0)
                ),
            }
        )
        rows.append(one)
    if not rows:
        raise ValueError(f"{path}/{sheet}: no NTC series")
    return pd.concat(rows, ignore_index=True)


def read_zone_limits(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name="Limits", header=None)
    if len(frame) < 10 + 8_760:
        raise ValueError(f"{path}/Limits: truncated")
    metadata = frame.iloc[:10]
    data = frame.iloc[10 : 10 + 8_760]
    rows: list[pd.DataFrame] = []
    for column in range(2, frame.shape[1]):
        curve_id = str(metadata.iat[7, column]).strip()
        curve_type = str(metadata.iat[8, column]).strip()
        if not curve_id or curve_id.lower() == "nan":
            continue
        value = pd.to_numeric(data.iloc[:, column], errors="coerce")
        if value.isna().all():
            continue
        if value.isna().any() or value.lt(0).any():
            raise ValueError(f"{path}/Limits/{curve_id}: invalid values")
        zone = curve_id.split(" - ", 1)[0].strip()
        rows.append(
            pd.DataFrame(
                {
                    "target_year": TARGET_YEAR,
                    "zone": zone,
                    "curve_id": curve_id,
                    "curve_type": curve_type,
                    "hour_of_year": np.arange(8_760, dtype=np.int16),
                    "limit_mw": value.to_numpy(dtype=float),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def make_currency_ledger() -> pd.DataFrame:
    rates = (
        ("low_cny_per_eur", ECB_2024_CNY_PER_EUR * 0.9),
        ("central_cny_per_eur", ECB_2024_CNY_PER_EUR),
        ("high_cny_per_eur", ECB_2024_CNY_PER_EUR * 1.1),
    )
    return pd.DataFrame(
        [
            {
                "scenario": name,
                "source_currency": "CNY",
                "target_currency": "EUR",
                "price_year": 2024,
                "cny_per_eur": rate,
                "conversion_rule": "EUR2024_value=CNY2024_value/cny_per_eur",
                "source": "ECB_EXR.A.CNY.EUR.SP00.A",
                "source_url": ECB_SERIES_URL,
                "interpretation": (
                    "pure unit conversion preserves within-library technology ranking; "
                    "it changes storage cost relative to ERAA EUR system values"
                ),
            }
            for name, rate in rates
        ]
    )


def source_manifest(paths: Iterable[Path]) -> pd.DataFrame:
    rows = []
    for path in sorted(set(paths)):
        rows.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(rows)


def execute(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "capacity": args.capacity_file.resolve(),
        "maintenance": args.maintenance_file.resolve(),
        "commodity": args.commodity_file.resolve(),
        "forced": args.forced_outage_file.resolve(),
        "derating": args.derating_file.resolve(),
        "must_run": args.must_run_file.resolve(),
        "investment": args.investment_file.resolve(),
        "ntc": args.ntc_file.resolve(),
        "annex_1": args.annex_1.resolve(),
        "annex_3": args.annex_3.resolve(),
        "eva": args.eva_file.resolve(),
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")
    output = args.output.resolve()
    prepare_output(output, args.force)

    capacity = read_capacity(paths["capacity"])
    eva_changes = read_eva_changes(paths["eva"])
    capacity_splits, exclusions, eva_reconciliation = build_capacity_splits(
        capacity, eva_changes
    )
    annex3_long = read_amended_annex3_changes(paths["annex_3"])
    (
        annex3_capacity,
        annex3_capacity_audit,
        annex3_capacity_conflicts,
        annex3_nonthermal,
    ) = build_annex3_capacity_cases(capacity, annex3_long)
    if not annex3_capacity.empty:
        capacity_splits = pd.concat(
            [capacity_splits, annex3_capacity], ignore_index=True
        )
    costs = build_costs(capacity_splits["zone"])
    availability = build_expected_availability(
        capacity_splits,
        paths["maintenance"],
        paths["forced"],
        paths["derating"],
    )
    firm = build_firm_costs(paths["investment"])
    ntc = pd.concat(
        [read_ntc_sheet(paths["ntc"], "HVAC"), read_ntc_sheet(paths["ntc"], "HVDC")],
        ignore_index=True,
    )
    limits = read_zone_limits(paths["ntc"])
    currency = make_currency_ledger()

    write_csv(capacity_splits, output / "thermal_capacity_split_cases.csv")
    write_csv(
        eva_reconciliation,
        output / "capacity_pre_post_eva_reconciliation.csv",
    )
    write_csv(
        annex3_long,
        output / "amended_annex3_table2_capacity_changes_long.csv",
    )
    write_csv(
        annex3_capacity_audit,
        output / "amended_annex3_post_eva_capacity_audit.csv",
    )
    write_csv(
        annex3_capacity_conflicts,
        output / "amended_annex3_post_eva_capacity_conflicts.csv",
    )
    write_csv(
        annex3_nonthermal,
        output / "amended_annex3_nonthermal_capacity_changes_2035.csv",
    )
    write_csv(exclusions, output / "thermal_capacity_exclusions.csv")
    write_csv(costs, output / "thermal_variable_costs_2035_eur2024.csv")
    write_csv(
        availability,
        output / "thermal_expected_availability_daily.csv.gz",
        compressed=True,
    )
    write_csv(firm, output / "firm_addition_costs_2035_eur2024.csv")
    write_csv(ntc, output / "ntc_hourly.csv.gz", compressed=True)
    write_csv(limits, output / "network_zone_limits_hourly.csv.gz", compressed=True)
    write_csv(currency, output / "currency_conversion_ledger.csv")
    manifest = source_manifest(paths.values())
    write_csv(manifest, output / "source_manifest.csv")

    nonzero = capacity_splits.loc[capacity_splits["capacity_mw"].gt(0)]
    availability_key_count = availability[
        ["zone", "eva_case", "split_case", "technology", "day_of_year"]
    ].drop_duplicates().shape[0]
    qa = {
        "script_version": SCRIPT_VERSION,
        "target_year": TARGET_YEAR,
        "evidence_status": (
            "dispatch_parameter_adapter_with_amended_annex3_"
            "retirement_feasible_aggregate_mapping"
        ),
        "capacity_zones": int(capacity_splits["zone"].nunique()),
        "split_cases": sorted(capacity_splits["split_case"].unique()),
        "eva_cases": sorted(capacity_splits["eva_case"].unique()),
        "eva_audit_cases": sorted(eva_reconciliation["eva_case"].unique()),
        "broad_eva_addition_negative_rows": int(
            eva_reconciliation["broad_addition_negative_flag"].sum()
        ),
        "annex3_table2_2035_grand_total_mw": float(
            annex3_long.loc[
                annex3_long["target_year"].eq(TARGET_YEAR),
                "capacity_change_mw",
            ].sum()
        ),
        "annex3_retirement_feasible_split_cases_ready": sorted(
            annex3_capacity["split_case"].unique()
        ),
        "annex3_capacity_conflict_rows": len(annex3_capacity_conflicts),
        "annex3_blocking_capacity_conflict_rows": int(
            annex3_capacity_conflicts["blocking_conflict"].sum()
        ),
        "post_eva_capacity_ready": bool(
            "retirement_feasible_max_ccgt"
            in set(annex3_capacity["split_case"].astype(str))
        ),
        "thermal_technologies": list(THERMAL_TECHNOLOGIES),
        "nonzero_capacity_rows": len(nonzero),
        "expected_availability_rows": len(availability),
        "expected_availability_key_count": availability_key_count,
        "ntc_edges": int(ntc["edge_id"].nunique()),
        "ntc_hourly_rows": len(ntc),
        "zone_limit_curves": int(limits["curve_id"].nunique()),
        "currency": {
            "system": "EUR2024",
            "storage_source": "CNY2024",
            "ecb_2024_cny_per_eur": ECB_2024_CNY_PER_EUR,
            "sensitivity": [
                ECB_2024_CNY_PER_EUR * 0.9,
                ECB_2024_CNY_PER_EUR,
                ECB_2024_CNY_PER_EUR * 1.1,
            ],
        },
        "official_2035_prices": OFFICIAL_2035_PRICES,
        "gates": {
            "availability_unique": availability_key_count == len(availability),
            "availability_bounded": bool(
                availability["expected_available_fraction"].between(0, 1).all()
            ),
            "ntc_8760_per_edge": bool(
                ntc.groupby(["network_technology", "edge_id"])["hour_of_year"]
                .nunique()
                .eq(8_760)
                .all()
            ),
            "no_currency_mixing": True,
            "annex3_grand_total_reconciled": bool(
                abs(
                    float(
                        annex3_long.loc[
                            annex3_long["target_year"].eq(TARGET_YEAR),
                            "capacity_change_mw",
                        ].sum()
                    )
                    - 14_630.0
                )
                <= 1e-8
            ),
            "annex3_primary_capacity_nonnegative": bool(
                "retirement_feasible_max_ccgt"
                in set(annex3_capacity["split_case"].astype(str))
            ),
        },
        "interpretation": [
            (
                "Natural-gas and hydrogen CCGT/OCGT shares are sensitivity "
                "splits of official aggregate capacity, not observed fleet shares."
            ),
            (
                "GenerationCapacities.csv is the National Trends pre-EVA "
                "starting point. The model post-EVA case uses the amended "
                "Annex 3 Table 2 central risk-averse/enhanced-hurdle changes. "
                "Because starting gas capacity is released only as a fuel "
                "aggregate, its CCGT/OCGT allocation is the minimum split "
                "that makes every reported retirement feasible; the residual "
                "fuel aggregate is varied transparently across three cases."
            ),
            (
                "Expected FOR is the first full-screen availability mode. "
                "Sequential 15-path outages are a separate sensitivity."
            ),
            (
                "Must-run records are preserved in the source manifest but "
                "not injected exogenously because the current core cannot spill "
                "thermal must-run output independently of VRE."
            ),
            (
                "HVAC/HVDC NTC series are released deterministic transport "
                "limits. CORE flow-based domains remain a separate sensitivity."
            ),
        ],
    }
    qa["pass"] = all(qa["gates"].values())
    write_json(output / "QA.json", qa)
    write_json(
        output / "run_manifest.json",
        {
            "script": Path(__file__).name,
            "script_version": SCRIPT_VERSION,
            "generated_at_utc": utc_now(),
            "output": output,
            "source_manifest_sha256": sha256_file(output / "source_manifest.csv"),
            "qa_sha256": sha256_file(output / "QA.json"),
            "evidence_status": qa["evidence_status"],
        },
    )
    return qa


def self_test() -> dict[str, Any]:
    costs = build_costs(["DE00", "BG00", "RO00"])
    gas = costs.loc[
        costs["zone"].eq("DE00") & costs["technology"].eq("gas_ccgt")
    ].iloc[0]
    expected = (
        OFFICIAL_2035_PRICES["gas_blend_eur2024_per_gj"] * 3.6 / 0.60
        + 57.0 * 3.6 / 0.60 / 1_000.0 * OFFICIAL_2035_PRICES["carbon_eur2024_per_t"]
        + 2.323896
    )
    currency = make_currency_ledger()
    rank_source = np.argsort(np.array([100.0, 200.0, 300.0]))
    rank_target = np.argsort(
        np.array([100.0, 200.0, 300.0])
        / float(currency.loc[currency["scenario"].eq("central_cny_per_eur"), "cny_per_eur"].iloc[0])
    )
    return {
        "script_version": SCRIPT_VERSION,
        "pass": bool(
            abs(float(gas["marginal_cost_eur2024_per_mwh"]) - expected) < 1e-10
            and np.array_equal(rank_source, rank_target)
        ),
        "tests": {
            "gas_ccgt_cost_formula": {
                "observed": float(gas["marginal_cost_eur2024_per_mwh"]),
                "expected": expected,
            },
            "pure_currency_conversion_preserves_ranking": bool(
                np.array_equal(rank_source, rank_target)
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--capacity-file", type=Path, default=DEFAULT_CAPACITY)
    parser.add_argument("--maintenance-file", type=Path, default=DEFAULT_MAINTENANCE)
    parser.add_argument("--commodity-file", type=Path, default=DEFAULT_COMMODITY)
    parser.add_argument("--forced-outage-file", type=Path, default=DEFAULT_FORCED_OUTAGE)
    parser.add_argument("--derating-file", type=Path, default=DEFAULT_DERATING)
    parser.add_argument("--must-run-file", type=Path, default=DEFAULT_MUST_RUN)
    parser.add_argument("--investment-file", type=Path, default=DEFAULT_INVESTMENT)
    parser.add_argument("--ntc-file", type=Path, default=DEFAULT_NTC)
    parser.add_argument("--annex-1", type=Path, default=DEFAULT_ANNEX_1)
    parser.add_argument("--annex-3", type=Path, default=DEFAULT_ANNEX_3)
    parser.add_argument("--eva-file", type=Path, default=DEFAULT_EVA)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = self_test() if args.self_test else execute(args)
    print(json.dumps(json_ready(payload), indent=2, sort_keys=True))
    return 0 if payload.get("pass", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
