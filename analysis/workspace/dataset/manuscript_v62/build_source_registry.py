#!/usr/bin/env python3
"""Build the manuscript-v62 data and cost provenance registry.

The project contains very large raw archives and existing code refers to their
current paths.  Organization is therefore catalogue-first: no source file is
moved or renamed.  This script freezes paths, byte sizes, checksums, roles,
derivation code and unresolved provenance fields in one machine-readable index.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
DATASET = HERE.parent
WORKSPACE = DATASET.parent
PROJECT = WORKSPACE / "Decarbonizing-Fossil-Peaking-with-Battery-Storage-master"
MANUSCRIPT = (
    PROJECT
    / "paper-code/docs/musrcipt/revision_nature_energy_v62"
)
EUROPE = DATASET / "external_hourly/europe_mid_duration_valley_upgrade_v0_1"
PECD = MANUSCRIPT / "weather_year_parity_v42"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(WORKSPACE.resolve()))
    except ValueError:
        return str(path.resolve())


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    def add(
        asset_id: str,
        family: str,
        layer: str,
        path: Path,
        source: str,
        scope: str,
        role: str,
        status: str,
        *,
        licence_or_access: str = "not_recorded",
        derivation_script: str = "",
        checksum: str | None = None,
        size_bytes: int | None = None,
        action_required: str = "none",
        notes: str = "",
    ) -> None:
        exists = path.exists()
        if exists and path.is_file():
            observed_size = path.stat().st_size
        else:
            observed_size = size_bytes
        if checksum is None and exists and path.is_file():
            checksum = sha256(path)
        rows.append(
            {
                "asset_id": asset_id,
                "family": family,
                "layer": layer,
                "path_relative_to_workspace": display_path(path),
                "exists": exists,
                "bytes": observed_size,
                "sha256": checksum or "",
                "source_or_provider": source,
                "spatial_temporal_scope": scope,
                "analysis_role": role,
                "derivation_script": derivation_script,
                "licence_or_access": licence_or_access,
                "status": status,
                "action_required": action_required,
                "notes": notes,
            }
        )

    # China: retain the distinction between a candidate upstream workbook and
    # the actual model-facing daily peaking input.  No unsupported lineage is
    # inferred between them.
    add(
        "CN_GAS_PIPELINE_CANDIDATE_RAW",
        "China",
        "candidate_upstream_raw",
        DATASET / "总数据.xlsx",
        "State Grid Corporation of China (internal contextual workbook; not used by v62)",
        "2019 daily pipeline/supply-point/user gas throughput; 1,337 records",
        "candidate contextual source; not read by the v62 model",
        "available_not_linked_not_used",
        licence_or_access="non-public internal data; not used by v62",
        action_required="none",
        notes="Workbook content is gas-pipeline/user throughput, not a self-describing provincial power-generation table; no transformation script was found.",
    )
    china_daily = PROJECT / "data/Daily Natural Gas Peaking Power Generation by Province in China.xlsx"
    add(
        "CN_GAS_PEAKING_DAILY_ANALYSIS",
        "China",
        "analysis_input",
        china_daily,
        "State Grid Corporation of China (SGCC); non-public internal official data supplied to the authors",
        "26 provinces x 365 mapped days",
        "daily provincial gas-peaking task used by v62",
        "complete_restricted_source",
        licence_or_access="non-public SGCC internal data; complete raw source will be provided in the submission-stage confidential review package",
        derivation_script=display_path(
            MANUSCRIPT / "cross_scale_cases/build_china_province_hourly_panel_v0_1.py"
        ),
        action_required="none",
        notes="Author confirms that the complete raw source was obtained from SGCC and is available for submission-stage review. SHA-256 agrees with the existing China panel QA. Daily energy is distributed uniformly across 24 model hours.",
    )
    for month in range(1, 13):
        add(
            f"CN_CURTAILMENT_2020_{month:02d}",
            "China",
            "raw",
            DATASET / f"30日弃电/TotalCurt_Hourly_2020{month:02d}.csv",
            "State Grid Corporation of China (SGCC); non-public internal official data supplied to the authors",
            f"station-level hourly wind/solar curtailment, 2020-{month:02d}",
            "hourly recharge-availability input after provincial aggregation",
            "complete_restricted_source",
            licence_or_access="non-public SGCC internal data; complete raw source will be provided in the submission-stage confidential review package",
            derivation_script=display_path(PROJECT / "paper-code/prepare_curtailment.py"),
            action_required="none",
            notes="Author confirms that the complete raw source was obtained from SGCC and is available for submission-stage review.",
        )
    add(
        "CN_PROVINCE_CROSSWALK",
        "China",
        "raw_support",
        DATASET / "省份修正合并.geojson",
        "author-compiled crosswalk for the SGCC internal curtailment records",
        "Chinese generating-station/province mapping",
        "maps station records to province",
        "complete_restricted_support",
        licence_or_access="derived support for non-public SGCC internal data; included in the submission-stage confidential review package",
        derivation_script=display_path(PROJECT / "paper-code/prepare_curtailment.py"),
        action_required="none",
    )
    add(
        "CN_MODEL_PANEL",
        "China",
        "processed",
        MANUSCRIPT / "cross_scale_cases/china_province_panel/china_province_event_task_panel.csv.gz",
        "derived by authors",
        "26 province-years; 227,760 hourly rows",
        "model-facing event-task and recharge panel",
        "complete_qa_pass",
        derivation_script=display_path(
            MANUSCRIPT / "cross_scale_cases/build_china_province_hourly_panel_v0_1.py"
        ),
        notes="Panel QA verifies non-negativity, 8,760 hours per province and annual energy conservation.",
    )

    # Japan.
    japan_qa = json.loads(
        (MANUSCRIPT / "cross_scale_cases/japan_community_panel/qa.json").read_text()
    )
    add(
        "JP_SELF_CONSUMPTION_ARCHIVE",
        "Japan",
        "raw",
        DATASET / "一桥数据/2025-09-30_Hitotsubashi University_Self-Consumption (Low-Voltage).zip",
        "Hitotsubashi University",
        "low-voltage self-consumption grid records; Kanto selection used",
        "source archive for community import/reverse-flow analysis",
        "integrity_verified_access_terms_pending",
        licence_or_access="research data supplied to authors; redistribution terms not recorded",
        checksum=japan_qa["archive_sha256"],
        action_required="confirm redistribution/controlled-access statement before deposit",
    )
    add(
        "JP_STANDARD_STATISTICS_ARCHIVE",
        "Japan",
        "raw_not_used",
        DATASET / "一桥数据/2025-09-30_Hitotsubashi University_Standard Statistics.zip",
        "Hitotsubashi University",
        "standard-statistics companion archive",
        "retained source; not used in v62 model",
        "available_not_used",
        licence_or_access="research data supplied to authors; redistribution terms not recorded",
        checksum="not_computed_large_unused_archive",
        action_required="none for analysis; document only if deposited",
    )
    add(
        "JP_MODEL_PANEL",
        "Japan",
        "processed",
        MANUSCRIPT / "cross_scale_cases/japan_community_panel/japan_community_event_task_panel.csv.gz",
        "derived by authors from Hitotsubashi archive",
        "19 grids; 27 complete fiscal years; 236,520 hourly rows",
        "model-facing community event-task and recharge panel",
        "complete_qa_pass",
        derivation_script=display_path(
            MANUSCRIPT / "cross_scale_cases/build_japan_community_hourly_panel_v0_1.py"
        ),
    )

    # Europe: import the existing canonical inventory rather than duplicating
    # 170 raw-file records.  The inventory points to the per-file raw manifest.
    europe_inventory_path = EUROPE / "metadata/EUROPE_CANONICAL_DATA_INVENTORY_v0_1.csv"
    inventory = pd.read_csv(europe_inventory_path)
    add(
        "EU_CANONICAL_INVENTORY",
        "Europe",
        "manifest",
        europe_inventory_path,
        "authors' frozen inventory of Fraunhofer ISE Energy-Charts assets",
        "17 countries, 2015-2024",
        "canonical entry point for raw and processed European data",
        "complete_verified",
        licence_or_access="Energy-Charts CC BY 4.0",
    )
    for item in inventory.itertuples(index=False):
        asset_path = EUROPE / str(item.relative_path)
        checksum = str(item.sha256)
        if checksum.startswith("see_"):
            checksum = checksum
        try:
            inventory_size = int(item.bytes)
        except (TypeError, ValueError):
            inventory_size = None
        add(
            str(item.asset_id),
            "Europe",
            "raw" if "RAW" in str(item.asset_id) else "processed",
            asset_path,
            str(item.source),
            str(item.spatial_temporal_scope),
            str(item.role),
            str(item.status),
            licence_or_access=str(item.licence_or_access),
            checksum=checksum,
            size_bytes=inventory_size,
            notes=str(item.notes),
        )

    # 75-year PECD weather chain.  The download manifest already contains an
    # exact request and checksum for every raw archive.
    pecd_manifest_path = PECD / "pecd_v42_download_manifest.json"
    pecd_manifest = json.loads(pecd_manifest_path.read_text())
    add(
        "PECD42_DOWNLOAD_MANIFEST",
        "Europe-75y",
        "manifest",
        pecd_manifest_path,
        "Copernicus Climate Data Store SIS Energy / ENTSO-E PECD v4.2",
        "1950-2025 calendar years; 75 complete July-June windows",
        "request, technology and checksum contract for all PECD raw archives",
        "complete_verified",
        licence_or_access="Copernicus Licence; PECD data access conditions apply",
    )
    for archive in pecd_manifest["archives"]:
        add(
            f"PECD42_{archive['product_id']}_{min(archive['years'])}_{max(archive['years'])}",
            "Europe-75y",
            "raw",
            PECD / archive["archive"],
            "Copernicus Climate Data Store SIS Energy / ENTSO-E PECD v4.2",
            f"{min(archive['years'])}-{max(archive['years'])}; {archive['product_id']}",
            "raw weather/reanalysis archive",
            "complete_verified",
            licence_or_access="Copernicus Licence; PECD data access conditions apply",
            checksum=archive["sha256"],
            size_bytes=int(archive["size_bytes"]),
            derivation_script=display_path(PECD / "build_pecd_v42_hourly_panel.py"),
        )
    add(
        "PECD42_75_WINDOW_PANEL",
        "Europe-75y",
        "processed",
        PECD / "prepared/Europe33_pecd_v42_1950_2025_75_windows.npz",
        "derived by authors from PECD v4.2",
        "Europe33, 75 July-June operating windows",
        "model-ready demand/wind/solar weather sequences",
        "complete_qa_pass",
        derivation_script=display_path(PECD / "build_pecd_v42_hourly_panel.py"),
    )
    add(
        "PECD42_75_WINDOW_RESULTS",
        "Europe-75y",
        "processed_result",
        PECD / "results/pecd_v42_75_window_relative_values_2025.csv",
        "derived by authors",
        "75 matched weather windows",
        "objective-aligned natural chronology value estimates",
        "complete_qa_pass",
        derivation_script=display_path(PECD / "compute_pecd_v42_75y_parity.py"),
    )

    # Cost library and supporting evidence.
    cost_files = [
        (
            "COST_PNNL_ESGC_WORKBOOK",
            "raw",
            "ESGC_Cost_Performance_Database_v2024.xlsx",
            "Pacific Northwest National Laboratory ESGC",
            "upstream component cost/performance database",
        ),
        (
            "COST_53_CONFIG_SNAPSHOTS",
            "analysis_input",
            "M2_LDES_53_configuration_cost_snapshots_v0_1_source_aware_eur.csv",
            "author-compiled from cited PNNL, DOE, NREL, LDES Council/EPRI and market anchors",
            "53 duration-technology configurations x 2025/2030/2035",
        ),
        (
            "COST_DURATION_YEAR_LIBRARY",
            "analysis_input",
            "M2_LDES_duration_year_parameter_library_v0_2_currency_provenance.csv",
            "author-compiled source-aware parameter library",
            "duration-year cost, performance, FX and replacement parameters",
        ),
        (
            "COST_LEARNING_EVIDENCE",
            "evidence",
            "M2_LDES_learning_curve_evidence_v0_7.csv",
            "author evidence map to PNNL, DOE and LDES Council/EPRI",
            "technology-level learning anchors",
        ),
        (
            "COST_SOURCE_NOTES",
            "documentation",
            "M2_LDES_来源清单与证据等级.md",
            "authors",
            "source list and evidence grading",
        ),
        (
            "COST_BUILD_NOTES",
            "documentation",
            "M2_LDES技术参数库构建说明.md",
            "authors",
            "parameter-library construction notes",
        ),
    ]
    for asset_id, layer, filename, source, role in cost_files:
        add(
            asset_id,
            "Technology-cost",
            layer,
            DATASET / filename,
            source,
            "16 technologies; 53 released duration configurations; 2025/2030/2035",
            role,
            "complete_source_aware",
            derivation_script=(
                display_path(DATASET / "build_M2_LDES_currency_provenance_v0_2.py")
                if layer == "analysis_input"
                else ""
            ),
            licence_or_access="mixed public reports/databases; see per-row URLs and source notes",
            notes="Author-selected interpolation/extrapolation and market calibration remain labelled per row.",
        )

    registry = pd.DataFrame(rows).sort_values(["family", "layer", "asset_id"])
    registry.to_csv(HERE / "source_registry.csv", index=False)
    (HERE / "source_registry.json").write_text(
        json.dumps(registry.to_dict("records"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    cost = pd.read_csv(DATASET / "M2_LDES_53_configuration_cost_snapshots_v0_1_source_aware_eur.csv")
    cost_map = cost.loc[cost["learning_year"].eq(2025)].copy()
    cost_map = cost_map[
        [
            "tech_id",
            "technology_en",
            "duration_h",
            "configuration_status",
            "direct_duration_anchor",
            "structural_extrapolation",
            "primary_source",
            "source_url_primary",
            "source_url_secondary",
            "technology_currency_route",
            "technology_currency_basis",
            "evidence_class",
            "confidence",
            "parameter_basis",
        ]
    ].sort_values(["duration_h", "tech_id"])
    if len(cost_map) != 53:
        raise RuntimeError(f"Expected 53 released 2025 configurations, found {len(cost_map)}")
    cost_map.to_csv(HERE / "technology_cost_source_map_53.csv", index=False)

    gaps = registry.loc[
        registry["action_required"].ne("none"),
        ["asset_id", "family", "status", "action_required", "notes"],
    ]
    gaps.to_csv(HERE / "provenance_gaps_requiring_author_input.csv", index=False)

    checksum_rows = registry.loc[
        registry["sha256"].str.fullmatch(r"[0-9a-f]{64}"),
        ["sha256", "path_relative_to_workspace"],
    ]
    (HERE / "checksums.sha256").write_text(
        "".join(f"{row.sha256}  {row.path_relative_to_workspace}\n" for row in checksum_rows.itertuples()),
        encoding="utf-8",
    )
    qa = {
        "status": "pass",
        "assets": int(len(registry)),
        "families": registry.groupby("family")["asset_id"].count().to_dict(),
        "all_registered_paths_exist": bool(registry["exists"].all()),
        "technology_configurations": int(len(cost_map)),
        "unresolved_assets": int(len(gaps)),
        "raw_files_moved_or_renamed": False,
        "organization_method": "catalogue-first non-destructive registry",
    }
    if not qa["all_registered_paths_exist"] or qa["technology_configurations"] != 53:
        qa["status"] = "fail"
    (HERE / "qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(qa, ensure_ascii=False, indent=2))
    if qa["status"] != "pass":
        raise RuntimeError(qa)


if __name__ == "__main__":
    main()
