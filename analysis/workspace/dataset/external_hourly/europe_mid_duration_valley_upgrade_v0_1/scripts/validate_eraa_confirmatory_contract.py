#!/usr/bin/env python3
"""Validate the frozen ERAA 2025 confirmatory-model input contract.

This validator is intentionally read-only.  It distinguishes a download that
is still incomplete (exit code 2) from an integrity or schema failure in an
already present file (exit code 3).  A successful return means that the raw
files are eligible for the next build stage; it does not mean that a
confirmatory model has been run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = ROOT / "raw" / "eraa_2025"
EXTRACTED = RAW / "extracted"
ARTIFACTS = ROOT / "artifacts" / "research"
TARGET_YEARS = (2028, 2030, 2033, 2035)
EXPECTED_WEATHER_SCENARIOS = set(range(1, 37))
EXPECTED_OUTAGE_SCENARIOS = set(range(1, 16))
EXCLUDED_AGGREGATES = {"DK00", "ISEM", "IT00", "LU00", "NO00", "SE00"}
REQUIRED_ATOMIC_COMPONENTS = {
    "DKE1",
    "DKW1",
    "IE00",
    "UKNI",
    "ITCN",
    "ITCS",
    "ITN1",
    "ITS1",
    "ITSA",
    "ITSI",
    "LUG1",
    "LUV1",
    "NOM1",
    "NON1",
    "NOS1",
    "NOS2",
    "NOS3",
    "SE01",
    "SE02",
    "SE03",
    "SE04",
}

SMALL_PACKAGES = (
    "ERAA_2025_CommonData.zip",
    "ERAA_2025_Dashboard_RawData.zip",
    "ERAA_2025_EconomicTechnicalInvestmentParameters.zip",
    "ERAA_2025_OtherData.zip",
    "ERAA_2025_FBDomains.zip",
    "ERAA_2025_HourlyENSData.zip",
    "ERAA_2025_NTCs.zip",
)
LARGE_PACKAGES = (
    "ERAA_2025_DemandData.zip",
    "ERAA_2025_PECDRES.zip",
)

V36_CATALOGUE = (
    ROOT.parents[2]
    / "Decarbonizing-Fossil-Peaking-with-Battery-Storage-master"
    / "paper-code"
    / "docs"
    / "musrcipt"
    / "revision_nature_energy_v36"
    / "system_value_closure"
    / "inputs"
    / "technology_duration_configurations_2035.csv"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_check(
    checks: list[dict[str, Any]],
    name: str,
    status: str,
    detail: str,
    path: Path | None = None,
) -> None:
    row: dict[str, Any] = {"check": name, "status": status, "detail": detail}
    if path is not None:
        row["path"] = str(path)
    checks.append(row)


def check_zip(path: Path, checks: list[dict[str, Any]], required_now: bool) -> None:
    if not path.exists():
        add_check(
            checks,
            f"zip:{path.name}",
            "fail" if required_now else "pending",
            "file is absent",
            path,
        )
        return
    try:
        with zipfile.ZipFile(path) as archive:
            bad_member = archive.testzip()
            member_count = len(archive.infolist())
        if bad_member:
            add_check(
                checks,
                f"zip:{path.name}",
                "fail",
                f"CRC failure in member {bad_member}",
                path,
            )
        else:
            add_check(
                checks,
                f"zip:{path.name}",
                "pass",
                (
                    f"{path.stat().st_size} bytes; {member_count} members; "
                    f"sha256={sha256(path)}"
                ),
                path,
            )
    except zipfile.BadZipFile as exc:
        add_check(
            checks,
            f"zip:{path.name}",
            "fail",
            f"bad zip: {exc}",
            path,
        )


def check_dashboard_versions(checks: list[dict[str, Any]]) -> None:
    dashboard = (
        EXTRACTED
        / "ERAA_2025_DashboardRawData1"
        / "Dashboard Raw Data"
    )
    specifications = (
        ("GenerationCapacities.csv", "data_version", "Target year"),
        ("Aggregated_Demand.csv", "data version", "Target_Year"),
        ("Reserve requirements.csv", "data_version", "YEAR"),
        (
            "Batteries additional information.csv",
            "data_version",
            "TARGET_YEAR",
        ),
        ("Hydro additional information.csv", "data_version", "TARGET_YEAR"),
    )
    for filename, version_column, year_column in specifications:
        path = dashboard / filename
        if not path.exists():
            add_check(
                checks,
                f"dashboard:{filename}",
                "fail",
                "extracted file is absent",
                path,
            )
            continue
        try:
            frame = pd.read_csv(path, usecols=[version_column, year_column])
            final = frame.loc[frame[version_column].eq("ERAA 2025 final")]
            years = set(pd.to_numeric(final[year_column], errors="coerce").dropna())
            missing = set(TARGET_YEARS).difference(int(value) for value in years)
            versions = sorted(frame[version_column].dropna().astype(str).unique())
            if missing:
                add_check(
                    checks,
                    f"dashboard:{filename}",
                    "fail",
                    f"ERAA 2025 final missing target years {sorted(missing)}",
                    path,
                )
            else:
                add_check(
                    checks,
                    f"dashboard:{filename}",
                    "pass",
                    (
                        "all four target years have ERAA 2025 final rows; "
                        f"raw file also contains versions={versions}"
                    ),
                    path,
                )
        except Exception as exc:  # schema failure should be explicit
            add_check(
                checks,
                f"dashboard:{filename}",
                "fail",
                f"schema/read failure: {type(exc).__name__}: {exc}",
                path,
            )


def check_ens(checks: list[dict[str, Any]]) -> None:
    required = {"WS", "FOS", "Date", "Bidding Zone", "ENS (MWh)"}
    for year in TARGET_YEARS:
        path = EXTRACTED / f"ENS_hourly_data_{year}.csv"
        if not path.exists():
            add_check(
                checks,
                f"hourly_ens:{year}",
                "fail",
                "extracted ENS file is absent",
                path,
            )
            continue
        try:
            frame = pd.read_csv(path)
            missing_columns = required.difference(frame.columns)
            if missing_columns:
                raise ValueError(f"missing columns {sorted(missing_columns)}")
            if frame.duplicated(["WS", "FOS", "Date", "Bidding Zone"]).any():
                raise ValueError("duplicate sparse ENS primary keys")
            if frame["ENS (MWh)"].lt(0).any():
                raise ValueError("negative ENS values")
            weather = set(pd.to_numeric(frame["WS"], errors="raise").astype(int))
            outages = set(pd.to_numeric(frame["FOS"], errors="raise").astype(int))
            if not weather.issubset(EXPECTED_WEATHER_SCENARIOS):
                raise ValueError(f"unexpected weather labels {sorted(weather)}")
            if not outages.issubset(EXPECTED_OUTAGE_SCENARIOS):
                raise ValueError(f"unexpected outage labels {sorted(outages)}")
            # The released files are sparse positive-ENS output.  Entirely
            # zero WS/FOS combinations can therefore be absent.
            add_check(
                checks,
                f"hourly_ens:{year}",
                "pass",
                (
                    f"{len(frame)} sparse rows; WS range={min(weather)}-"
                    f"{max(weather)} ({len(weather)} represented); FOS "
                    f"range={min(outages)}-{max(outages)} "
                    f"({len(outages)} represented)"
                ),
                path,
            )
        except Exception as exc:
            add_check(
                checks,
                f"hourly_ens:{year}",
                "fail",
                f"schema/read failure: {type(exc).__name__}: {exc}",
                path,
            )


def check_zone_hierarchy(checks: list[dict[str, Any]]) -> None:
    path = ARTIFACTS / "ERAA_2025_ENS_ZONE_HIERARCHY_AUDIT.csv"
    if not path.exists():
        add_check(
            checks,
            "atomic_zone_hierarchy",
            "fail",
            "zone hierarchy audit is absent",
            path,
        )
        return
    try:
        frame = pd.read_csv(path)
        required = {
            "target_year",
            "candidate_type",
            "parent_zone",
            "exact_reconciliation",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"missing columns {sorted(missing)}")
        prespecified = frame.loc[
            frame["candidate_type"].eq("prespecified_aggregate")
        ]
        reconciled = set(
            prespecified.loc[
                prespecified["exact_reconciliation"].astype(bool),
                "parent_zone",
            ]
        )
        expected_rows = len(TARGET_YEARS) * len(EXCLUDED_AGGREGATES)
        if len(prespecified) != expected_rows:
            raise ValueError(
                f"expected {expected_rows} aggregate-year tests, "
                f"observed {len(prespecified)}"
            )
        missing_rules = EXCLUDED_AGGREGATES.difference(reconciled)
        if missing_rules:
            raise ValueError(
                f"aggregate codes not exactly reconciled: {sorted(missing_rules)}"
            )
        add_check(
            checks,
            "atomic_zone_hierarchy",
            "pass",
            (
                "all aggregate/alias rules reconcile in four target years; "
                f"exclude={sorted(EXCLUDED_AGGREGATES)}"
            ),
            path,
        )
    except Exception as exc:
        add_check(
            checks,
            "atomic_zone_hierarchy",
            "fail",
            f"hierarchy failure: {type(exc).__name__}: {exc}",
            path,
        )


def check_catalogue(checks: list[dict[str, Any]]) -> None:
    path = V36_CATALOGUE
    if not path.exists():
        add_check(
            checks,
            "technology_duration_catalogue",
            "fail",
            "v36 audited 53-configuration catalogue is absent",
            path,
        )
        return
    try:
        frame = pd.read_csv(path)
        required = {
            "tech_id",
            "duration_h",
            "rte",
            "charge_power_ratio",
            "cycle_life_cycles",
            "structural_extrapolation",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"missing columns {sorted(missing)}")
        main = frame.loc[
            frame["duration_h"].le(168)
            & ~frame["structural_extrapolation"].astype(bool)
        ]
        structural = frame.loc[frame["structural_extrapolation"].astype(bool)]
        structural_durations = set(structural["duration_h"].astype(float))
        if len(frame) != 53 or len(main) != 50 or len(structural) != 3:
            raise ValueError(
                f"expected 53 total/50 main/3 structural, observed "
                f"{len(frame)}/{len(main)}/{len(structural)}"
            )
        if structural_durations != {240.0, 336.0, 720.0}:
            raise ValueError(
                f"unexpected structural durations {sorted(structural_durations)}"
            )
        lfp = frame.loc[frame["tech_id"].eq("li_lfp")]
        if lfp.empty or not lfp["cycle_life_cycles"].gt(0).all():
            raise ValueError("LFP cycle-life evidence is absent or non-positive")
        add_check(
            checks,
            "technology_duration_catalogue",
            "pass",
            (
                "53 configurations: 50 in 4-168 h main domain and "
                "3 H2 structural nodes at 240/336/720 h; "
                f"sha256={sha256(path)}"
            ),
            path,
        )
    except Exception as exc:
        add_check(
            checks,
            "technology_duration_catalogue",
            "fail",
            f"catalogue failure: {type(exc).__name__}: {exc}",
            path,
        )


def check_network_contract(checks: list[dict[str, Any]]) -> None:
    """Require the main flow-based domains and the NTC boundary package."""

    required = (
        (
            EXTRACTED / "FB-Domain-CORE_full.xlsx",
            "primary_core_flow_based_domain",
        ),
        (
            EXTRACTED / "FB-Domain-NORDIC.xlsx",
            "primary_nordic_flow_based_domain",
        ),
        (
            EXTRACTED / "List of CNECs" / "CORE" / "cnec_info_2035.csv",
            "primary_2035_core_cnec_list",
        ),
        (RAW / "ERAA_2025_NTCs.zip", "ntc_boundary_and_fallback"),
    )
    for path, role in required:
        if not path.exists() or path.stat().st_size == 0:
            add_check(
                checks,
                f"network:{role}",
                "fail",
                "required network input is absent or empty",
                path,
            )
            continue
        add_check(
            checks,
            f"network:{role}",
            "pass",
            f"{path.stat().st_size} bytes; sha256={sha256(path)}",
            path,
        )


def build_report() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for filename in SMALL_PACKAGES:
        check_zip(RAW / filename, checks, required_now=True)
    for filename in LARGE_PACKAGES:
        check_zip(RAW / filename, checks, required_now=False)
        partial_dir = RAW / f".{filename}.ranges"
        if not (RAW / filename).exists() and partial_dir.exists():
            parts = list(partial_dir.glob("*.part"))
            add_check(
                checks,
                f"download_progress:{filename}",
                "pending",
                f"{len(parts)} persistent range parts are present",
                partial_dir,
            )
    check_dashboard_versions(checks)
    check_ens(checks)
    check_zone_hierarchy(checks)
    check_catalogue(checks)
    check_network_contract(checks)

    fail_count = sum(row["status"] == "fail" for row in checks)
    pending_count = sum(row["status"] == "pending" for row in checks)
    status = "fail" if fail_count else ("pending" if pending_count else "pass")
    return {
        "contract": "eraa_2025_confirmatory_v0_1",
        "status": status,
        "confirmatory_primary_target_year": 2035,
        "temporal_sensitivity_target_years": [2028, 2030, 2033],
        "primary_network": "CORE_and_Nordic_flow_based_domains",
        "boundary_and_fallback_network": "directional_NTC",
        "target_years": list(TARGET_YEARS),
        "expected_weather_scenarios_per_target_year": 36,
        "expected_outage_scenarios_per_weather_scenario": 15,
        "excluded_aggregate_zone_codes": sorted(EXCLUDED_AGGREGATES),
        "required_atomic_components": sorted(REQUIRED_ATOMIC_COMPONENTS),
        "fail_count": fail_count,
        "pending_count": pending_count,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional new path for the machine-readable audit report.",
    )
    parser.add_argument(
        "--allow-pending",
        action="store_true",
        help="Return zero when only large downloads are pending.",
    )
    args = parser.parse_args()

    report = build_report()
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")

    if report["status"] == "fail":
        sys.exit(3)
    if report["status"] == "pending" and not args.allow_pending:
        sys.exit(2)


if __name__ == "__main__":
    main()
