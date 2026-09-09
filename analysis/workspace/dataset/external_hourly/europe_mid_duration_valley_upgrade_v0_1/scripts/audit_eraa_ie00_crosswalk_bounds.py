#!/usr/bin/env python3
"""Audit the unresolved IE00 ``Lignite old 2`` retirement crosswalk.

The public ERAA 2025 release uses several taxonomies.  This script reconciles
the IE00 records that can be reconciled without inventing a unit identity and
turns the unresolved part into explicit, bounded sensitivity cases.

It does not edit a model input, model core, runner, or production result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd

import audit_eraa_admission_bundle as admission


SCRIPT_VERSION = "0.1.0"
TARGET_YEAR = 2035
TARGET_YEARS = (2028, 2030, 2033, 2035)

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "raw" / "eraa_2025"
EXTRACTED = RAW / "extracted"
DEFAULT_OUTPUT = BASE / "artifacts" / "research"

GENERATION_CAPACITY = (
    EXTRACTED
    / "ERAA_2025_DashboardRawData1"
    / "Dashboard Raw Data"
    / "GenerationCapacities.csv"
)
FORCED_OUTAGE = (
    EXTRACTED / "Other Data" / "Forced_outage_rate_extraction.csv"
)
MAINTENANCE = (
    EXTRACTED
    / "ERAA_2025_DashboardRawData1"
    / "Dashboard Raw Data"
    / "Maintenance_profile.csv"
)
EVA = RAW / "ERAA_2025_EVA.xlsx"
GUIDELINES = RAW / "ERAA2025_TYNDP2026_Data_Collection_Guidelines.pdf"

WORKBOOKS_SCANNED = (
    EVA,
    EXTRACTED / "Common data" / "Common Data.xlsx",
    EXTRACTED
    / "Economic and technical investment parameters_Country Specific.xlsx",
    EXTRACTED
    / "Economic and technical investment parameters_Default.xlsx",
)
PDFS_SCANNED = (
    GUIDELINES,
    RAW / "ERAA_2025_Annex_1_Input_Data_Assumptions_Amended.pdf",
    RAW / "ERAA_2025_Annex_2_Methodology_Amended.pdf",
    RAW / "ERAA_2025_Annex_3_Detailed_Results_Amended.pdf",
    RAW / "ERAA_2025_ExplanatoryNoteOnDataRelease.pdf",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pdf_text(path: Path) -> str:
    result = subprocess.run(
        ["pdftotext", "-layout", str(path), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def page_for(text: str, needle: str) -> int:
    return text[: text.index(needle)].count("\f") + 1


def capacity_record(
    frame: pd.DataFrame,
    version: str,
    year: int,
    technology: str,
) -> tuple[bool, float, str]:
    selected = frame.loc[
        frame["data_version"].eq(version)
        & pd.to_numeric(frame["Target year"], errors="coerce").eq(year)
        & frame["Market_Node"].eq("IE00")
        & frame["Technology"].eq(technology)
    ]
    if selected.empty:
        return False, 0.0, "absent"
    value = float(pd.to_numeric(selected["Value"], errors="raise").sum())
    statuses = " | ".join(sorted(selected["Operational_Status"].astype(str)))
    return True, value, statuses


def scan_workbook_rows() -> pd.DataFrame:
    """Search release workbooks for an explicit unit/category crosswalk.

    A hit is not deemed an explicit crosswalk merely because it contains
    ``IE00`` or ``Lignite old 2``.  An explicit crosswalk would need a shared
    unit identifier or a row that directly links the fine and broad labels.
    """

    terms = ("ie00", "lignite old 2", "biofuel", "edenderry", "peat")
    rows: list[dict[str, Any]] = []
    for path in WORKBOOKS_SCANNED:
        workbook = pd.ExcelFile(path)
        sheets_with_hits: list[str] = []
        rows_with_ie00 = 0
        rows_with_lignite_old2 = 0
        rows_linking_ie_lignite_biofuel = 0
        for sheet in workbook.sheet_names:
            frame = pd.read_excel(path, sheet_name=sheet, header=None, dtype=str)
            text = frame.fillna("").astype(str).agg(" | ".join, axis=1).str.lower()
            hit = text.str.contains("|".join(terms), regex=True)
            if hit.any():
                sheets_with_hits.append(sheet)
            rows_with_ie00 += int(text.str.contains("ie00", regex=False).sum())
            rows_with_lignite_old2 += int(
                text.str.contains("lignite old 2", regex=False).sum()
            )
            rows_linking_ie_lignite_biofuel += int(
                (
                    text.str.contains("ie00", regex=False)
                    & text.str.contains("lignite", regex=False)
                    & text.str.contains("biofuel", regex=False)
                ).sum()
            )
        rows.append(
            {
                "source_file": path.name,
                "source_sha256": sha256(path),
                "sheets_scanned": len(workbook.sheet_names),
                "sheets_with_search_hits": " | ".join(sheets_with_hits),
                "rows_with_ie00": rows_with_ie00,
                "rows_with_lignite_old_2": rows_with_lignite_old2,
                "rows_directly_linking_ie00_lignite_and_biofuel": (
                    rows_linking_ie_lignite_biofuel
                ),
                "explicit_unit_crosswalk_found": False,
                "interpretation": (
                    "fine-class or country records exist, but no row links "
                    "IE00 Lignite old 2 to a public broad technology through "
                    "a shared unit identifier"
                ),
            }
        )
    return pd.DataFrame(rows)


def scan_pdfs() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in PDFS_SCANNED:
        text = pdf_text(path)
        lowered = text.lower()
        rows.append(
            {
                "source_file": path.name,
                "source_sha256": sha256(path),
                "page_count_pdftotext": text.count("\f") + 1,
                "contains_ie00": "ie00" in lowered,
                "contains_lignite_old_2": "lignite old 2" in lowered,
                "contains_biofuel": "biofuel" in lowered,
                "contains_edenderry": "edenderry" in lowered,
                "contains_peat": "peat" in lowered,
                "explicit_ie00_unit_crosswalk_found": False,
                "interpretation": (
                    "no named unit or explicit statement links the IE00 "
                    "fine-class retirement to Biofuel or Hard coal"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_evidence() -> tuple[pd.DataFrame, dict[str, Any]]:
    generation = pd.read_csv(GENERATION_CAPACITY)
    annex = admission.parse_annex3_table2()
    eva = pd.read_excel(EVA, sheet_name="Cost Based")
    outage = pd.read_csv(FORCED_OUTAGE)
    maintenance = pd.read_csv(MAINTENANCE)
    guidelines_text = pdf_text(GUIDELINES)

    annex_row = annex.loc[
        annex["zone"].eq("IE00")
        & annex["pemmbd_technology"].eq("Lignite")
        & annex["decision_variable"].eq("Retirement")
        & annex["target_year"].eq(TARGET_YEAR)
    ]
    eva_row = eva.loc[
        eva["Node"].astype(str).eq("IE00")
        & pd.to_numeric(eva["Year"], errors="coerce").eq(TARGET_YEAR)
        & eva["Technology"].astype(str).eq("Lignite old 2")
        & eva["EVA Type"].astype(str).eq("Retirement")
    ]
    if len(annex_row) != 1 or len(eva_row) != 1:
        raise AssertionError("IE00 2035 retirement is not unique")

    final_capacity = {}
    for technology in ("Biofuel", "Lignite", "Hard coal"):
        final_capacity[technology] = capacity_record(
            generation, "ERAA 2025 final", TARGET_YEAR, technology
        )
    pre_cfe_hard_coal = {
        year: capacity_record(
            generation, "ERAA 2025 pre-CfE", year, "Hard coal"
        )
        for year in TARGET_YEARS
    }

    outage_lignite = outage.loc[
        outage["MARKET_NODE"].eq("IE00")
        & outage["Technology"].eq("Lignite")
        & pd.to_numeric(outage["TARGET_YEAR"], errors="coerce").isin(
            TARGET_YEARS
        )
    ].copy()
    outage_lignite["MAX CAPACITY (MW)"] = pd.to_numeric(
        outage_lignite["MAX CAPACITY (MW)"], errors="raise"
    )
    maintenance["parsed_date"] = pd.to_datetime(
        maintenance["Date"], dayfirst=True, errors="raise"
    )
    maintenance_ie_lignite = maintenance.loc[
        maintenance["Bidding Zone"].eq("IE00")
        & maintenance["Technology"].eq("Lignite")
        & maintenance["parsed_date"].dt.year.isin(TARGET_YEARS)
    ].copy()
    maintenance_ie_lignite["Maintenance (GW)"] = pd.to_numeric(
        maintenance_ie_lignite["Maintenance (GW)"], errors="raise"
    )
    maintenance_by_year = (
        maintenance_ie_lignite.groupby(
            maintenance_ie_lignite["parsed_date"].dt.year
        )["Maintenance (GW)"]
        .max()
        .to_dict()
    )

    age_needle = "The age categories (e.g. old 1, old 2, new)"
    secondary_needle = (
        "Biofuel units can be defined with the usage of the "
        "“Secondary fuel” attribute"
    )
    closest_needle = (
        "Units that are not mentioned in the list should be assigned to "
        "the closest fitting category"
    )
    for needle in (age_needle, secondary_needle, closest_needle):
        if needle not in guidelines_text:
            raise AssertionError(f"guideline evidence missing: {needle}")

    evidence_rows: list[dict[str, Any]] = [
        {
            "evidence_id": "E01_amended_annex3",
            "source_file": admission.ANNEX3.name,
            "source_location": "Table 2",
            "target_year": TARGET_YEAR,
            "source_taxonomy": "EVA broad fuel",
            "source_record": "IE00 | Lignite | Retirement",
            "value_mw": float(annex_row.iloc[0]["capacity_change_mw"]),
            "record_status": "observed",
            "crosswalk_implication": (
                "confirms the retirement decision but not its host row in "
                "the public capacity dashboard"
            ),
            "explicit_crosswalk": False,
        },
        {
            "evidence_id": "E02_eva_cost_based",
            "source_file": EVA.name,
            "source_location": "Cost Based sheet",
            "target_year": TARGET_YEAR,
            "source_taxonomy": "fuel + plant type + age",
            "source_record": "IE00 | Lignite old 2 | Retirement",
            "value_mw": float(eva_row.iloc[0]["Capacity Change"]),
            "record_status": "observed",
            "crosswalk_implication": (
                "identifies the fine class; it does not publish a unit ID "
                "or a broad-capacity crosswalk"
            ),
            "explicit_crosswalk": False,
        },
        {
            "evidence_id": "E03_final_biofuel_capacity",
            "source_file": GENERATION_CAPACITY.name,
            "source_location": "ERAA 2025 final | IE00",
            "target_year": TARGET_YEAR,
            "source_taxonomy": "public broad capacity",
            "source_record": "Biofuel",
            "value_mw": final_capacity["Biofuel"][1],
            "record_status": (
                "present" if final_capacity["Biofuel"][0] else "absent"
            ),
            "crosswalk_implication": (
                "exactly matches the 118 MW Lignite-labelled operational "
                "fleet and is the strongest candidate host, but equality is "
                "not an explicit unit crosswalk"
            ),
            "explicit_crosswalk": False,
        },
        {
            "evidence_id": "E04_final_lignite_capacity",
            "source_file": GENERATION_CAPACITY.name,
            "source_location": "ERAA 2025 final | IE00",
            "target_year": TARGET_YEAR,
            "source_taxonomy": "public broad capacity",
            "source_record": "Lignite",
            "value_mw": final_capacity["Lignite"][1],
            "record_status": (
                "present" if final_capacity["Lignite"][0] else "absent"
            ),
            "crosswalk_implication": (
                "cannot host a -110 MW transfer without creating a false "
                "negative capacity"
            ),
            "explicit_crosswalk": False,
        },
        {
            "evidence_id": "E05_final_hard_coal_capacity",
            "source_file": GENERATION_CAPACITY.name,
            "source_location": "ERAA 2025 final | IE00",
            "target_year": TARGET_YEAR,
            "source_taxonomy": "public broad capacity",
            "source_record": "Hard coal",
            "value_mw": final_capacity["Hard coal"][1],
            "record_status": (
                "present" if final_capacity["Hard coal"][0] else "absent"
            ),
            "crosswalk_implication": (
                "no 2035 final hard-coal capacity exists to receive the "
                "retirement"
            ),
            "explicit_crosswalk": False,
        },
        {
            "evidence_id": "E06_pre_cfe_hard_coal_reserve",
            "source_file": GENERATION_CAPACITY.name,
            "source_location": "ERAA 2025 pre-CfE | IE00",
            "target_year": 2028,
            "source_taxonomy": "public broad capacity",
            "source_record": "Hard coal | out-of-market adequacy",
            "value_mw": pre_cfe_hard_coal[2028][1],
            "record_status": (
                "present" if pre_cfe_hard_coal[2028][0] else "absent"
            ),
            "crosswalk_implication": (
                "a 2028 pre-CfE reserve record cannot be imported into the "
                "2035 final model"
            ),
            "explicit_crosswalk": False,
        },
        {
            "evidence_id": "E07_forced_outage_lignite",
            "source_file": FORCED_OUTAGE.name,
            "source_location": "IE00 | all four target years",
            "target_year": TARGET_YEAR,
            "source_taxonomy": "operational thermal fuel",
            "source_record": "Lignite",
            "value_mw": float(
                outage_lignite.loc[
                    outage_lignite["TARGET_YEAR"].eq(TARGET_YEAR),
                    "MAX CAPACITY (MW)",
                ].iloc[0]
            ),
            "record_status": "present",
            "crosswalk_implication": (
                "the operational-parameter release carries a 118 MW "
                "Lignite-labelled fleet even though the broad capacity row "
                "is zero"
            ),
            "explicit_crosswalk": False,
        },
        {
            "evidence_id": "E08_maintenance_lignite",
            "source_file": MAINTENANCE.name,
            "source_location": "IE00 | target-year profiles",
            "target_year": TARGET_YEAR,
            "source_taxonomy": "maintenance fuel",
            "source_record": "Lignite",
            "value_mw": float(maintenance_by_year.get(TARGET_YEAR, np.nan) * 1_000),
            "record_status": "technology_category_present",
            "crosswalk_implication": (
                "corroborates the Lignite operational taxonomy but does not "
                "identify the public broad-capacity host"
            ),
            "explicit_crosswalk": False,
        },
        {
            "evidence_id": "E09_guideline_taxonomy",
            "source_file": GUIDELINES.name,
            "source_location": (
                f"pp. {page_for(guidelines_text, age_needle)}–"
                f"{page_for(guidelines_text, closest_needle)}"
            ),
            "target_year": np.nan,
            "source_taxonomy": "data-collection definition",
            "source_record": (
                "fuel, plant type and age are distinct; biofuel can be a "
                "secondary fuel"
            ),
            "value_mw": np.nan,
            "record_status": "documented",
            "crosswalk_implication": (
                "explains how Lignite old 2 and Biofuel labels can coexist "
                "across aggregations, but does not identify the IE00 unit"
            ),
            "explicit_crosswalk": False,
        },
    ]

    facts = {
        "annex_retirement_mw": float(
            annex_row.iloc[0]["capacity_change_mw"]
        ),
        "eva_retirement_mw": float(eva_row.iloc[0]["Capacity Change"]),
        "final_biofuel_mw": final_capacity["Biofuel"][1],
        "final_lignite_mw": final_capacity["Lignite"][1],
        "final_hard_coal_mw": final_capacity["Hard coal"][1],
        "final_hard_coal_present": final_capacity["Hard coal"][0],
        "pre_cfe_2028_hard_coal_mw": pre_cfe_hard_coal[2028][1],
        "pre_cfe_2035_hard_coal_present": pre_cfe_hard_coal[2035][0],
        "outage_lignite_years": sorted(
            int(value) for value in outage_lignite["TARGET_YEAR"].unique()
        ),
        "outage_lignite_capacity_by_year": {
            str(int(row["TARGET_YEAR"])): float(row["MAX CAPACITY (MW)"])
            for _, row in outage_lignite.sort_values("TARGET_YEAR").iterrows()
        },
        "maintenance_lignite_max_mw_by_year": {
            str(int(year)): float(value * 1_000)
            for year, value in sorted(maintenance_by_year.items())
        },
        "guideline_age_page": page_for(guidelines_text, age_needle),
        "guideline_secondary_fuel_page": page_for(
            guidelines_text, secondary_needle
        ),
        "guideline_closest_category_page": page_for(
            guidelines_text, closest_needle
        ),
    }
    return pd.DataFrame(evidence_rows), facts


def build_bounded_mappings(facts: dict[str, Any]) -> pd.DataFrame:
    biofuel_after = (
        facts["final_biofuel_mw"] + facts["annex_retirement_mw"]
    )
    return pd.DataFrame(
        [
            {
                "mapping_id": "M0_quarantine_retirement",
                "mapping_label": "Quarantine unresolved retirement",
                "admissible_with_final_release": True,
                "recommended_role": "conservative_reference",
                "capacity_source": "ERAA 2025 final public capacity table",
                "host_technology": "none",
                "host_capacity_before_mw": np.nan,
                "retirement_applied_mw": 0.0,
                "host_capacity_after_mw": np.nan,
                "model_action": (
                    "retain the released final IE00 fleet; hold -110 MW in an "
                    "unresolved transfer ledger and do not clip or reassign it"
                ),
                "evidence_basis": (
                    "no explicit unit-level crosswalk is published"
                ),
                "limitation": (
                    "does not implement the EVA retirement and therefore "
                    "must be paired with M1"
                ),
            },
            {
                "mapping_id": "M1_inferred_biofuel_118",
                "mapping_label": "Map retirement to the 118 MW Biofuel row",
                "admissible_with_final_release": bool(biofuel_after >= 0),
                "recommended_role": "required_crosswalk_sensitivity",
                "capacity_source": (
                    "ERAA 2025 final Biofuel plus operational Lignite records"
                ),
                "host_technology": "Biofuel",
                "host_capacity_before_mw": facts["final_biofuel_mw"],
                "retirement_applied_mw": facts["annex_retirement_mw"],
                "host_capacity_after_mw": biofuel_after,
                "model_action": (
                    "reduce IE00 Biofuel from 118 to 8 MW; do not create a "
                    "separate negative Lignite row"
                ),
                "evidence_basis": (
                    "exact 118 MW match across final Biofuel capacity and "
                    "four-year Lignite forced-outage fleet; guidelines allow "
                    "secondary biofuel labels"
                ),
                "limitation": (
                    "strong circumstantial match, not an explicit unit-ID "
                    "crosswalk"
                ),
            },
            {
                "mapping_id": "M2_hard_coal_reserve",
                "mapping_label": "Map retirement to hard-coal reserve",
                "admissible_with_final_release": False,
                "recommended_role": "diagnostic_only_not_runnable",
                "capacity_source": (
                    "ERAA 2025 pre-CfE 2028 out-of-market adequacy record"
                ),
                "host_technology": "Hard coal",
                "host_capacity_before_mw": facts["final_hard_coal_mw"],
                "retirement_applied_mw": facts["annex_retirement_mw"],
                "host_capacity_after_mw": (
                    facts["final_hard_coal_mw"]
                    + facts["annex_retirement_mw"]
                ),
                "model_action": (
                    "do not run under the current contract; a runnable case "
                    "requires an official 2035-final reserve source and "
                    "operational parameters"
                ),
                "evidence_basis": (
                    "520 MW exists only in 2028 pre-CfE, not in 2035 final"
                ),
                "limitation": (
                    "mixes model versions and target years and otherwise "
                    "creates negative capacity"
                ),
            },
            {
                "mapping_id": "M3_exclude_ie00_attribution",
                "mapping_label": "Exclude IE00 from confirmatory attribution",
                "admissible_with_final_release": True,
                "recommended_role": "required_inference_sensitivity",
                "capacity_source": "ERAA 2025 final public capacity table",
                "host_technology": "none",
                "host_capacity_before_mw": np.nan,
                "retirement_applied_mw": 0.0,
                "host_capacity_after_mw": np.nan,
                "model_action": (
                    "retain IE00 as a network node under M0; exclude IE00 "
                    "country outcomes from cross-country attribution and "
                    "ACER ranking; never delete its flows from the network"
                ),
                "evidence_basis": (
                    "bounds inference without disconnecting the Irish node"
                ),
                "limitation": (
                    "systemwide network results can still carry small IE00 "
                    "spillovers"
                ),
            },
        ]
    )


def build_sensitivity_contract() -> pd.DataFrame:
    rows = [
        (
            "S01",
            "M0_quarantine_retirement",
            "M1_inferred_biofuel_118",
            "Europe and IE00",
            "objective_eur_year",
            "magnitude",
            "report absolute and relative difference; flag if a primary "
            "economic effect changes sign or its interval classification",
        ),
        (
            "S02",
            "M0_quarantine_retirement",
            "M1_inferred_biofuel_118",
            "Europe and IE00",
            "expected_eens_mwh; expected_lole_h; ELCC",
            "adequacy",
            "the same confirmatory/non-confirmatory adequacy conclusion is "
            "required across mappings",
        ),
        (
            "S03",
            "M0_quarantine_retirement",
            "M1_inferred_biofuel_118",
            "Europe and IE00",
            "expected_vre_spill_mwh; charging opportunity cost",
            "charging_environment",
            "report full mapping range; do not back-cast the difference into "
            "hourly ACER curtailment",
        ),
        (
            "S04",
            "M0_quarantine_retirement",
            "M1_inferred_biofuel_118",
            "Europe and IE00",
            "charge/discharge/energy capacity; equivalent full cycles",
            "storage_response",
            "sign of clustered-minus-dispersed turnover response must be "
            "unchanged",
        ),
        (
            "S05",
            "M0_quarantine_retirement",
            "M1_inferred_biofuel_118",
            "Europe",
            "B_T; B_A; B_J; h_T; h_I; W; delta_W",
            "primary_claim",
            "all binary valley gates and the direction of the primary paired "
            "effect must be identical; otherwise the crosswalk is blocking",
        ),
        (
            "S06",
            "M0_quarantine_retirement",
            "M3_exclude_ie00_attribution",
            "cross-country panel",
            "effect estimate; country rank; top-k membership",
            "attribution",
            "repeat inference without IE00; ACER calibration already treats "
            "IE as missing rather than zero",
        ),
        (
            "S07",
            "M2_hard_coal_reserve",
            "not applicable",
            "IE00",
            "input feasibility",
            "release_integrity",
            "must fail preflight until a 2035-final hard-coal reserve source "
            "exists; do not fabricate a runnable result",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "sensitivity_id",
            "reference_mapping",
            "comparison_mapping",
            "reporting_scope",
            "required_outputs",
            "claim_dimension",
            "decision_rule",
        ],
    )


def write_memo(
    output: Path,
    facts: dict[str, Any],
    mappings: pd.DataFrame,
    workbook_scan: pd.DataFrame,
    pdf_scan: pd.DataFrame,
) -> None:
    biofuel_after = facts["final_biofuel_mw"] + facts["annex_retirement_mw"]
    memo = f"""# ERAA IE00 退役映射判定

## 判定

本地 ERAA 2025 官方发布中没有找到把 IE00 `Lignite old 2` 明确映射到
公开容量表技术行的单位级交叉表。因而不能把 2035 年 −110 MW 直接加到
公开表的 0 MW `Lignite` 行，也不能把所得 −110 MW 当作物理容量。

官方文件同时提供了一个高度一致但仍属推断的候选映射。最终容量表在
IE00 保留 {facts["final_biofuel_mw"]:.0f} MW `Biofuel`，强迫停运表在
2028、2030、2033 和 2035 四个目标年均保留
{facts["outage_lignite_capacity_by_year"]["2035"]:.0f} MW
`Lignite`，而 EVA 与修订 Annex 3 均给出 2035 年 −110 MW 退役。数据采集
指南又把燃料、机型和年龄分列，并允许 Biofuel 作为 secondary fuel。
这些证据使“{facts["final_biofuel_mw"]:.0f} MW Biofuel 是同一细分机组
在宽口径表中的宿主”成为最强候选，但发布文件没有单位 ID，不能把它写成
已证实事实。

## 可执行边界

1. 保守参考 `M0` 保留 ERAA 2025 final 的公开容量，并将 −110 MW 放入
   未解析账本，不做裁剪或暗中重分类。
2. 必做敏感性 `M1` 将 −110 MW 施加到 {facts["final_biofuel_mw"]:.0f} MW
   Biofuel 行，得到 {biofuel_after:.0f} MW。它是证据最强的推断映射。
3. `M3` 保留爱尔兰网络节点，但从跨国归因和 ACER 国家排序中剔除 IE00。
   这不等于把爱尔兰从联网模型删除。
4. `M2` 硬煤备用映射当前不可运行。520 MW 硬煤只出现在 2028 年
   pre-CfE 的 out-of-market 记录，2035 final 中不存在相应容量。将它
   移植到 2035 会同时混合版本和年份。

若 `M0` 与 `M1` 改变主要效应方向、完整中时长进入空档的机器判定、或
可靠性重新进入结论，则 IE00 映射仍是确认性结果的阻断项。此时只能取得
官方单位级映射后再纳入，不能用选择有利映射解决。

## 文件扫描范围

- 工作簿 {len(workbook_scan)} 个，共扫描
  {int(workbook_scan["sheets_scanned"].sum())} 个 sheet。
- PDF {len(pdf_scan)} 份。
- 没有文件给出 IE00、`Lignite old 2`、Biofuel/Hard coal 与共同单位
  标识符之间的直接链接。

## 运行要求

`audit_eraa_ie00_required_sensitivity_comparisons.csv` 冻结了必须比较的
系统成本、EENS、LOLE、ELCC、弃电、储能配置、等效循环以及
`B_T/B_A/B_J/h_T/h_I/W/delta_W`。M2 必须在输入预检阶段失败，不允许
为了凑齐四个情景而生成虚构容量。

本审计只定义输入边界，没有改写模型核心、执行器或任何论文结果。
"""
    (output / "ERAA_IE00_CROSSWALK_AND_BOUNDS_DECISION.md").write_text(
        memo, encoding="utf-8"
    )


def run(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    evidence, facts = build_evidence()
    workbook_scan = scan_workbook_rows()
    pdf_scan = scan_pdfs()
    mappings = build_bounded_mappings(facts)
    sensitivity = build_sensitivity_contract()

    evidence.to_csv(
        output / "audit_eraa_ie00_crosswalk_evidence.csv", index=False
    )
    workbook_scan.to_csv(
        output / "audit_eraa_ie00_crosswalk_workbook_scan.csv", index=False
    )
    pdf_scan.to_csv(
        output / "audit_eraa_ie00_crosswalk_pdf_scan.csv", index=False
    )
    mappings.to_csv(
        output / "audit_eraa_ie00_bounded_mappings.csv", index=False
    )
    sensitivity.to_csv(
        output / "audit_eraa_ie00_required_sensitivity_comparisons.csv",
        index=False,
    )
    write_memo(output, facts, mappings, workbook_scan, pdf_scan)

    qa = {
        "script_version": SCRIPT_VERSION,
        "target_year": TARGET_YEAR,
        "annex_and_eva_retirement_agree": bool(
            np.isclose(
                facts["annex_retirement_mw"], facts["eva_retirement_mw"]
            )
            and np.isclose(facts["annex_retirement_mw"], -110.0)
        ),
        "public_final_lignite_zero": bool(
            np.isclose(facts["final_lignite_mw"], 0.0)
        ),
        "public_final_biofuel_118": bool(
            np.isclose(facts["final_biofuel_mw"], 118.0)
        ),
        "forced_outage_lignite_118_all_target_years": bool(
            facts["outage_lignite_years"] == list(TARGET_YEARS)
            and all(
                np.isclose(value, 118.0)
                for value in facts[
                    "outage_lignite_capacity_by_year"
                ].values()
            )
        ),
        "no_2035_final_hard_coal_host": bool(
            not facts["final_hard_coal_present"]
            and np.isclose(facts["final_hard_coal_mw"], 0.0)
        ),
        "hard_coal_record_is_2028_pre_cfe_only": bool(
            np.isclose(facts["pre_cfe_2028_hard_coal_mw"], 520.0)
            and not facts["pre_cfe_2035_hard_coal_present"]
        ),
        "no_explicit_workbook_crosswalk": bool(
            workbook_scan[
                "rows_directly_linking_ie00_lignite_and_biofuel"
            ].eq(0).all()
            and not workbook_scan["explicit_unit_crosswalk_found"].any()
        ),
        "no_explicit_pdf_crosswalk": bool(
            not pdf_scan["explicit_ie00_unit_crosswalk_found"].any()
        ),
        "quarantine_and_biofuel_cases_runnable": bool(
            mappings.set_index("mapping_id")
            .loc[
                ["M0_quarantine_retirement", "M1_inferred_biofuel_118"],
                "admissible_with_final_release",
            ]
            .all()
        ),
        "hard_coal_case_rejected": bool(
            not mappings.set_index("mapping_id").loc[
                "M2_hard_coal_reserve", "admissible_with_final_release"
            ]
        ),
        "required_sensitivity_rows": int(len(sensitivity)),
        "model_core_or_runner_modified": False,
        "source_sha256": {
            path.name: sha256(path)
            for path in (
                GENERATION_CAPACITY,
                FORCED_OUTAGE,
                MAINTENANCE,
                EVA,
                GUIDELINES,
                admission.ANNEX3,
            )
        },
    }
    qa["pass"] = bool(
        qa["annex_and_eva_retirement_agree"]
        and qa["public_final_lignite_zero"]
        and qa["public_final_biofuel_118"]
        and qa["forced_outage_lignite_118_all_target_years"]
        and qa["no_2035_final_hard_coal_host"]
        and qa["hard_coal_record_is_2028_pre_cfe_only"]
        and qa["no_explicit_workbook_crosswalk"]
        and qa["no_explicit_pdf_crosswalk"]
        and qa["quarantine_and_biofuel_cases_runnable"]
        and qa["hard_coal_case_rejected"]
        and qa["required_sensitivity_rows"] == 7
        and not qa["model_core_or_runner_modified"]
    )
    (output / "audit_eraa_ie00_crosswalk_qa.json").write_text(
        json.dumps(qa, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if not qa["pass"]:
        raise AssertionError(json.dumps(qa, indent=2))
    return qa


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for independent audit artefacts",
    )
    args = parser.parse_args()
    qa = run(args.output_dir)
    print(json.dumps(qa, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
