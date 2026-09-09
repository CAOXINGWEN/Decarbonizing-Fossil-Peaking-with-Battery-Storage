#!/usr/bin/env python3
"""Build the v0.2 Energy-Charts fossil-gas provenance release.

This is an annotation-only successor to the frozen v0.1 gas-task panel.  It:

1. derives hourly provider-value and short-gap-fill flags from the missing
   mask *before* interpolation;
2. verifies the reconstructed gas series against the frozen v0.1 values;
3. emits a v0.2 panel without writing any v0.1 path;
4. emits a complete reproducibility manifest for the 170 frozen
   Energy-Charts ``public_power`` JSON files; and
5. replaces the superseded adapter wording with the current actual aggregate
   fossil-gas event-task data contract.

The aggregate ``Fossil gas`` category is not an OCGT/CCGT or unit-level
peaker series.  The archived files were retrieved from the Fraunhofer ISE
Energy-Charts API, not directly from the ENTSO-E Transparency API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import build_energy_charts_actual_supply_response as legacy


SCHEMA_VERSION = "2.0.0"
DATASET_VERSION = "v0.2"
RELEASE_DATE = "2026-07-27"
API_BASE = "https://api.energy-charts.info"
API_ENDPOINT = "/public_power"
API_DOCUMENTATION_VERSION = "1.6"
API_DOCUMENTATION_URL = "https://api.energy-charts.info/"
API_SOURCES_URL = (
    "https://www.energy-charts.info/sources.html?c=DE&l=en"
)
LICENSE_NAME = "Creative Commons Attribution 4.0 International"
LICENSE_SHORT = "CC BY 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
LICENSE_SCOPE = (
    "Energy-Charts API data unless otherwise stated; no endpoint-specific "
    "exception is stated for public_power in API documentation v1.6"
)
ATTRIBUTION = (
    "Fraunhofer Institute for Solar Energy Systems ISE, Energy-Charts.info"
)
ENTSOE_SEMANTIC_MAPPING = "A75 actual generation per type | A16 realised | B04 fossil gas"

CURRENT_CONTRACT = (
    "actual aggregate public-net fossil-gas generation is used as an "
    "event-task proxy; it is not unit-level OCGT/CCGT peaker output"
)
RECHARGE_CONTRACT = (
    "recharge is a separate R2 potential renewable-surplus input; "
    "gas-to-storage charging is prohibited in the confirmatory model"
)
LEGACY_ADAPTER_CONTRACT = (
    "observed Fossil gas is adapter load; run with vre_energy_share=0; "
    "charging energy is endogenously generated and explicitly costed"
)
LEGACY_ADAPTER_STATUS = "superseded_non_evidential"

EXPECTED_INPUT_SHA256 = {
    "processed/energy_charts_europe_hourly_eligible_2015_2024.csv.gz": (
        "54fa19f4f51c432ae417d39844e78072ef49297f87661946081f4392030c21d1"
    ),
    "processed/energy_charts_observed_gas_task_panel_2015_2024.csv.gz": (
        "5f2e8ab49120050fac2ef8bd1b11e20a58863442631a8d9baccf5b924f0bf3ab"
    ),
    "artifacts/research/energy_charts_actual_supply_response_qa.csv": (
        "adf64abbd7a886d9616931098f68c2760bfcf15e71886aaf8aabdea1b73af747"
    ),
    "artifacts/research/energy_charts_observed_gas_task_prespecified_sample.csv": (
        "cf1bbfae8cf1212ab3aeab2b70a5fc7cb9148c6f308310e385a7d7f043ff9ef8"
    ),
    "metadata/energy_charts_download_receipt.json": (
        "8172423556904ee0c2cb75bb8b6b0f0fe719f0fbce0ae7acfc6cede2b1c0c0b3"
    ),
    "metadata/energy_charts_panel_source_files.csv": (
        "112f0efa8bf8777522a7b16ad8fe340c2059e3d9b37538b1d73217fdfc2dd760"
    ),
    "metadata/energy_charts_public_power_local_provenance.csv": (
        "fb2da7d89adcdf975d81b60d739dd560fe1d6d841980640223a6e934756fa36d"
    ),
    "metadata/energy_charts_public_power_local_provenance.json": (
        "923aa01600bd7d29f460f14bf3f9222413876095ea2fbc486e42dcfa064d543b"
    ),
    "artifacts/research/energy_charts_public_power_local_provenance_qa.json": (
        "0d048719024878b56ca8f3e1be6f37a2c15ff32a17e6a8c7f7261ab8ec526aeb"
    ),
}

V1_PANEL = Path(
    "processed/energy_charts_observed_gas_task_panel_2015_2024.csv.gz"
)
V2_PANEL = Path(
    "processed/energy_charts_observed_gas_task_panel_2015_2024_v0_2.csv.gz"
)
V1_COUNTRY_YEAR_QA = Path(
    "artifacts/research/energy_charts_actual_supply_response_qa.csv"
)
PRESPECIFIED_SAMPLE = Path(
    "artifacts/research/energy_charts_observed_gas_task_prespecified_sample.csv"
)
V2_COUNTRY_YEAR_QA = Path(
    "artifacts/research/"
    "energy_charts_observed_gas_task_imputation_by_country_year_v0_2.csv"
)
V2_QA = Path(
    "artifacts/research/"
    "energy_charts_observed_gas_task_panel_v0_2_qa.json"
)
RAW_MANIFEST_CSV = Path(
    "metadata/energy_charts_public_power_frozen_raw_manifest_v0_2.csv"
)
RAW_MANIFEST_JSON = Path(
    "metadata/energy_charts_public_power_frozen_raw_manifest_v0_2.json"
)
DATA_CONTRACT_JSON = Path(
    "metadata/energy_charts_observed_gas_task_data_contract_v0_2.json"
)
V2_MANIFEST_JSON = Path(
    "metadata/energy_charts_observed_gas_task_v0_2_manifest.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def atomic_write_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(
        temporary,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        float_format="%.6f",
        lineterminator="\n",
    )
    temporary.replace(path)


def require_frozen_inputs(root: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    failures: list[str] = []
    for relative, expected in EXPECTED_INPUT_SHA256.items():
        path = root / relative
        if not path.is_file():
            failures.append(f"missing frozen input: {relative}")
            continue
        actual = sha256(path)
        observed[relative] = actual
        if actual != expected:
            failures.append(
                f"frozen input hash mismatch: {relative}: "
                f"expected {expected}, found {actual}"
            )
    if failures:
        raise RuntimeError("\n".join(failures))
    return observed


def trace_fill_from_prefill_mask(
    values: pd.Series,
    minimum_coverage: float,
    maximum_gap_h: int,
) -> tuple[pd.Series, pd.DataFrame, dict[str, Any]]:
    """Fill one series and retain origin flags from the pre-fill mask."""

    source = values.astype(float).copy()
    missing_before_fill = source.isna()
    provider_available = ~missing_before_fill
    coverage = float(provider_available.mean())
    maximum_gap = legacy.longest_missing_run(source)
    eligible = bool(
        coverage >= minimum_coverage
        and maximum_gap <= maximum_gap_h
        and provider_available.any()
    )

    origin = pd.Series(
        "excluded_ineligible_country_year",
        index=source.index,
        dtype="object",
    )
    fill_method = pd.Series(
        "not_filled_ineligible_country_year",
        index=source.index,
        dtype="object",
    )
    short_gap_filled = pd.Series(False, index=source.index, dtype=bool)
    output = pd.Series(np.nan, index=source.index, dtype=float)
    linear_filled = pd.Series(False, index=source.index, dtype=bool)
    boundary_filled = pd.Series(False, index=source.index, dtype=bool)

    if eligible:
        linear = source.interpolate(
            method="time",
            limit=maximum_gap_h,
            limit_direction="both",
            limit_area="inside",
        )
        linear_filled = missing_before_fill & linear.notna()
        after_linear_missing = linear.isna()
        filled = linear.ffill(limit=maximum_gap_h).bfill(
            limit=maximum_gap_h
        )
        boundary_filled = (
            missing_before_fill & after_linear_missing & filled.notna()
        )
        if filled.isna().any():
            raise RuntimeError(
                "Eligible gas series retains missing values after filling"
            )
        if not (
            missing_before_fill
            == (linear_filled | boundary_filled)
        ).all():
            raise RuntimeError(
                "Pre-fill missing mask is not exhausted by tracked methods"
            )
        output = filled
        origin.loc[provider_available] = "energy_charts_hourly"
        origin.loc[linear_filled] = "filled_time_linear_interior"
        origin.loc[boundary_filled] = (
            "filled_nearest_value_year_boundary"
        )
        fill_method.loc[provider_available] = "none"
        fill_method.loc[linear_filled] = "time_linear_interior"
        fill_method.loc[boundary_filled] = (
            "nearest_value_year_boundary"
        )
        short_gap_filled = missing_before_fill.copy()

    flags = pd.DataFrame(
        {
            "gas_task_provider_hourly_available_before_fill": (
                provider_available.to_numpy(bool)
            ),
            "gas_task_short_gap_filled": (
                short_gap_filled.to_numpy(bool)
            ),
            "gas_task_fill_method": fill_method.to_numpy(object),
            "gas_task_value_origin": origin.to_numpy(object),
        },
        index=source.index,
    )
    record = {
        "coverage": coverage,
        "max_missing_gap_h": int(maximum_gap),
        "eligible": eligible,
        "provider_hourly_values": int(provider_available.sum()),
        "missing_before_fill_hours": int(missing_before_fill.sum()),
        "short_gap_filled_hours": int(short_gap_filled.sum()),
        "linear_filled_hours": int(linear_filled.sum()),
        "boundary_filled_hours": int(boundary_filled.sum()),
    }
    return output, flags, record


def build_response_cache(root: Path) -> dict[str, pd.DataFrame]:
    raw_dir = root / "raw" / "energy_charts" / "public_power"
    by_country: dict[str, list[Path]] = {}
    for path in sorted(raw_dir.glob("public_power_*_*.json")):
        match = legacy.RAW_PATTERN.search(path.name)
        if not match:
            continue
        api_code = match.group(1)
        iso2 = legacy.API_TO_ISO.get(api_code, api_code.upper())
        by_country.setdefault(iso2, []).append(path)
    if not by_country:
        raise RuntimeError("No frozen public_power JSON files found")
    return {
        country: legacy.build_country_response(paths)
        for country, paths in sorted(by_country.items())
    }


def build_hourly_flags(
    root: Path,
    v1: pd.DataFrame,
    minimum_coverage: float,
    maximum_gap_h: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    response_cache = build_response_cache(root)
    flag_parts: list[pd.DataFrame] = []
    qa_rows: list[dict[str, Any]] = []

    for (country, year), group in v1.groupby(
        ["country", "year"], sort=False
    ):
        timestamps = pd.to_datetime(group["timestamp_utc"], utc=True)
        raw_gas = response_cache[str(country)].reindex(
            pd.DatetimeIndex(timestamps)
        )["fossil_gas_mw"]
        rebuilt, flags, record = trace_fill_from_prefill_mask(
            raw_gas,
            minimum_coverage,
            maximum_gap_h,
        )

        legacy_filled, legacy_record = legacy.fill_component(
            raw_gas,
            minimum_coverage,
            maximum_gap_h,
        )
        legacy_used = (
            legacy_filled
            if bool(legacy_record["eligible"])
            else pd.Series(np.nan, index=raw_gas.index, dtype=float)
        )
        if not np.allclose(
            rebuilt.to_numpy(float),
            legacy_used.to_numpy(float),
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        ):
            raise RuntimeError(
                f"Tracked fill differs from v0.1 fill for {country}-{year}"
            )
        for key in ("coverage", "max_missing_gap_h", "eligible"):
            if legacy_record[key] != record[key]:
                raise RuntimeError(
                    f"Tracked {key} differs from v0.1 for {country}-{year}"
                )
        if (
            int(legacy_record["imputed_hours"])
            != record["short_gap_filled_hours"]
        ):
            raise RuntimeError(
                f"Tracked fill count differs from v0.1 for {country}-{year}"
            )

        rebuilt_task = np.maximum(rebuilt.to_numpy(float), 0.0)
        frozen_task = pd.to_numeric(
            group["observed_gas_peaking_task_mw"], errors="coerce"
        ).to_numpy(float)
        rounded_rebuilt = np.round(rebuilt_task, 6)
        if not np.allclose(
            rounded_rebuilt,
            frozen_task,
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        ):
            difference = np.abs(rounded_rebuilt - frozen_task)
            maximum = (
                float(np.nanmax(difference))
                if np.isfinite(difference).any()
                else np.nan
            )
            raise RuntimeError(
                f"Rebuilt gas task differs from frozen v0.1 for "
                f"{country}-{year}; max difference={maximum}"
            )

        part = flags.reset_index(drop=True)
        part.index = group.index
        part["gas_task_rebuilt_from_prefill_mask_mw"] = rebuilt_task
        flag_parts.append(part)
        qa_rows.append(
            {
                "country": str(country),
                "year": int(year),
                **record,
                "legacy_imputed_hours": int(
                    legacy_record["imputed_hours"]
                ),
                "legacy_fill_count_exact_match": bool(
                    int(legacy_record["imputed_hours"])
                    == record["short_gap_filled_hours"]
                ),
                "rebuilt_gas_task_exact_after_6dp_serialization": True,
            }
        )

    flags = pd.concat(flag_parts).sort_index()
    if not flags.index.equals(v1.index):
        raise RuntimeError("Hourly flag rows do not align with frozen v0.1")
    return flags, pd.DataFrame(qa_rows).sort_values(["country", "year"])


def build_raw_manifest(
    root: Path,
) -> tuple[pd.DataFrame, dict[str, Any], str]:
    raw_dir = root / "raw" / "energy_charts" / "public_power"
    source = pd.read_csv(
        root / "metadata" / "energy_charts_panel_source_files.csv"
    )
    source_lookup = {
        (str(row.api_code), int(row.source_local_calendar_year)): row
        for row in source.itertuples(index=False)
    }
    receipt_payload = json.loads(
        (
            root / "metadata" / "energy_charts_download_receipt.json"
        ).read_text(encoding="utf-8")
    )
    receipt_lookup = {
        (str(row["country_api_code"]), int(row["year"])): row
        for row in receipt_payload["records"]
        if row.get("endpoint") == "public_power"
    }

    rows: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("public_power_*_*.json")):
        match = legacy.RAW_PATTERN.search(path.name)
        if not match:
            raise RuntimeError(f"Unexpected raw filename: {path.name}")
        api_code, year_text = match.groups()
        year = int(year_text)
        country = legacy.API_TO_ISO.get(api_code, api_code.upper())
        key = (api_code, year)
        if key not in source_lookup:
            raise RuntimeError(f"Raw file absent from source manifest: {key}")
        source_row = source_lookup[key]
        payload = json.loads(path.read_text(encoding="utf-8"))
        required_keys = {
            "unix_seconds",
            "production_types",
            "deprecated",
        }
        required_keys_present = required_keys.issubset(payload)
        if not required_keys_present:
            raise RuntimeError(f"Required API keys missing in {path}")

        timestamps = np.asarray(payload["unix_seconds"], dtype=np.int64)
        step_minutes = (
            float(np.median(np.diff(timestamps))) / 60.0
            if len(timestamps) > 1
            else np.nan
        )
        gas_entries = [
            item
            for item in payload["production_types"]
            if str(item.get("name")) == "Fossil gas"
        ]
        gas_values = (
            pd.to_numeric(
                pd.Series(gas_entries[0]["data"]), errors="coerce"
            )
            if len(gas_entries) == 1
            else pd.Series(dtype=float)
        )
        if len(gas_entries) > 1:
            raise RuntimeError(f"Duplicate Fossil gas series in {path}")

        relative = path.relative_to(root).as_posix()
        canonical_url = (
            f"{API_BASE}{API_ENDPOINT}?country={api_code}"
            f"&start={year}-01-01&end={year}-12-31"
        )
        bytes_count = int(path.stat().st_size)
        digest = sha256(path)
        manifest_url = str(source_row.source_url)
        manifest_path = Path(str(source_row.source_file))
        if manifest_path.is_absolute():
            manifest_path = manifest_path.relative_to(root)
        elif "raw" in manifest_path.parts:
            raw_index = manifest_path.parts.index("raw")
            manifest_path = Path(*manifest_path.parts[raw_index:])
        source_match = bool(
            manifest_url == canonical_url
            and manifest_path.as_posix() == relative
            and int(source_row.source_bytes) == bytes_count
            and str(source_row.source_sha256) == digest
            and int(source_row.raw_timestamps) == len(timestamps)
        )
        if not source_match:
            raise RuntimeError(f"Source-manifest mismatch for {path}")

        receipt = receipt_lookup.get(key)
        receipt_present = receipt is not None
        receipt_status = (
            str(receipt.get("status")) if receipt_present else None
        )
        receipt_match = None
        if receipt_present:
            receipt_match = bool(
                receipt.get("url") == canonical_url
                and receipt.get("local_path") == relative
                and int(receipt.get("bytes", -1)) == bytes_count
                and receipt.get("sha256") == digest
                and int(receipt.get("n_timestamps", -1))
                == len(timestamps)
            )
            if not receipt_match:
                raise RuntimeError(f"Historic receipt mismatch for {path}")

        receipt_time = (
            str(receipt.get("retrieved_utc"))
            if receipt_present and receipt.get("retrieved_utc")
            else None
        )
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset_version": DATASET_VERSION,
                "provider": (
                    "Fraunhofer Institute for Solar Energy Systems ISE"
                ),
                "provider_product": "Energy-Charts",
                "provider_compilation": True,
                "upstream_source_description": (
                    "Energy-Charts processed compilation of ENTSO-E and "
                    "other official providers"
                ),
                "direct_entsoe_api": False,
                "entsoe_semantic_mapping_candidate": (
                    ENTSOE_SEMANTIC_MAPPING
                ),
                "entsoe_value_level_lineage_verified": False,
                "http_method": "GET",
                "api_endpoint": f"{API_BASE}{API_ENDPOINT}",
                "api_parameter_country": api_code,
                "api_parameter_start": f"{year}-01-01",
                "api_parameter_end": f"{year}-12-31",
                "canonical_url": canonical_url,
                "country_api_code": api_code,
                "country_iso2": country,
                "source_local_calendar_year": year,
                "relative_path": relative,
                "bytes": bytes_count,
                "sha256": digest,
                "native_timestamp_count": int(len(timestamps)),
                "native_step_minutes": step_minutes,
                "deprecated": bool(payload["deprecated"]),
                "required_keys_present": required_keys_present,
                "fossil_gas_exact_field_present": bool(
                    len(gas_entries) == 1
                ),
                "fossil_gas_native_nonmissing_values": int(
                    gas_values.notna().sum()
                ),
                "source_manifest_exact_match": source_match,
                "historic_receipt_present": receipt_present,
                "historic_receipt_status": receipt_status,
                "historic_request_started_utc": (
                    receipt_time
                    if receipt_status == "downloaded"
                    else None
                ),
                "historic_local_reuse_checked_utc": (
                    receipt_time if receipt_status == "reused" else None
                ),
                "historic_original_download_time_known": False,
                "historic_receipt_exact_match": receipt_match,
                "http_response_headers_archived": False,
                "api_documentation_version": (
                    API_DOCUMENTATION_VERSION
                ),
                "api_documentation_url": API_DOCUMENTATION_URL,
                "provider_sources_url": API_SOURCES_URL,
                "license_name": LICENSE_NAME,
                "license_short": LICENSE_SHORT,
                "license_url": LICENSE_URL,
                "license_scope": LICENSE_SCOPE,
                "attribution_required": True,
                "attribution_text": ATTRIBUTION,
            }
        )

    frame = pd.DataFrame(rows).sort_values(
        ["country_api_code", "source_local_calendar_year"]
    )
    if len(frame) != 170:
        raise RuntimeError(
            f"Expected 170 frozen public_power files, found {len(frame)}"
        )
    if frame[["country_api_code", "source_local_calendar_year"]].duplicated().any():
        raise RuntimeError("Duplicate country-year key in raw manifest")
    if int(frame["historic_receipt_present"].sum()) != 97:
        raise RuntimeError("Expected 97 historic public_power receipts")
    status_counts = (
        frame["historic_receipt_status"]
        .fillna("missing")
        .value_counts()
        .to_dict()
    )
    expected_status = {"downloaded": 86, "reused": 11, "missing": 73}
    if status_counts != expected_status:
        raise RuntimeError(
            f"Unexpected receipt statuses: {status_counts}"
        )
    if not frame["source_manifest_exact_match"].all():
        raise RuntimeError("Not all raw files match the frozen source manifest")
    if not frame.loc[
        frame["historic_receipt_present"],
        "historic_receipt_exact_match",
    ].all():
        raise RuntimeError("Not all available receipts match current files")

    inventory_digest = hashlib.sha256()
    for row in frame.itertuples(index=False):
        inventory_digest.update(
            (
                f"{row.relative_path}\t{row.sha256}\t{row.bytes}\n"
            ).encode("utf-8")
        )
    inventory_sha256 = inventory_digest.hexdigest()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": DATASET_VERSION,
        "release_date": RELEASE_DATE,
        "scope": (
            "all 170 frozen Energy-Charts public_power raw JSON files "
            "supporting the European hourly panel"
        ),
        "provider": {
            "name": (
                "Fraunhofer Institute for Solar Energy Systems ISE"
            ),
            "product": "Energy-Charts",
            "api_endpoint": f"{API_BASE}{API_ENDPOINT}",
            "api_documentation_version": API_DOCUMENTATION_VERSION,
            "api_documentation_url": API_DOCUMENTATION_URL,
            "sources_url": API_SOURCES_URL,
            "provider_compilation": True,
            "direct_entsoe_api": False,
            "entsoe_semantic_mapping_candidate": (
                ENTSOE_SEMANTIC_MAPPING
            ),
            "entsoe_value_level_lineage_verified": False,
        },
        "license": {
            "name": LICENSE_NAME,
            "short": LICENSE_SHORT,
            "url": LICENSE_URL,
            "scope": LICENSE_SCOPE,
            "attribution_required": True,
            "attribution_text": ATTRIBUTION,
        },
        "summary": {
            "raw_files": int(len(frame)),
            "countries": int(frame["country_iso2"].nunique()),
            "years": sorted(
                int(value)
                for value in frame[
                    "source_local_calendar_year"
                ].unique()
            ),
            "historic_receipts": int(
                frame["historic_receipt_present"].sum()
            ),
            "historic_downloaded_receipts": int(
                (frame["historic_receipt_status"] == "downloaded").sum()
            ),
            "historic_reused_receipts": int(
                (frame["historic_receipt_status"] == "reused").sum()
            ),
            "missing_historic_receipts": int(
                (~frame["historic_receipt_present"]).sum()
            ),
            "all_required_keys_present": bool(
                frame["required_keys_present"].all()
            ),
            "all_deprecated_false": bool(
                (~frame["deprecated"]).all()
            ),
            "all_source_manifest_exact_match": bool(
                frame["source_manifest_exact_match"].all()
            ),
            "raw_content_inventory_sha256": inventory_sha256,
        },
        "receipt_time_semantics": {
            "downloaded": (
                "historic_request_started_utc was recorded immediately "
                "before the download function; response completion time and "
                "headers were not archived"
            ),
            "reused": (
                "historic_local_reuse_checked_utc is a local verification "
                "time, not the original download time"
            ),
            "missing": (
                "no historic HTTP transaction was reconstructed; canonical "
                "URL and current content hash remain available"
            ),
        },
        "records": rows,
    }
    return frame, manifest, inventory_sha256


def compare_v1_v2(
    v1: pd.DataFrame,
    v2: pd.DataFrame,
) -> dict[str, Any]:
    shared = list(v1.columns)
    excluded_metadata = {
        "gas_task_source",
        "gas_task_model_contract",
    }
    compared = [
        column for column in shared if column not in excluded_metadata
    ]
    failures: list[str] = []
    maximum_numeric_difference = 0.0
    numeric_columns: list[str] = []
    exact_non_numeric_columns: list[str] = []
    for column in compared:
        left = v1[column]
        right = v2[column]
        if pd.api.types.is_numeric_dtype(left.dtype):
            numeric_columns.append(column)
            left_values = pd.to_numeric(left, errors="coerce").to_numpy(
                float
            )
            right_values = pd.to_numeric(right, errors="coerce").to_numpy(
                float
            )
            if not np.array_equal(
                np.isnan(left_values), np.isnan(right_values)
            ):
                failures.append(f"missingness differs: {column}")
                continue
            difference = np.abs(left_values - right_values)
            finite = np.isfinite(difference)
            column_maximum = (
                float(difference[finite].max())
                if finite.any()
                else 0.0
            )
            maximum_numeric_difference = max(
                maximum_numeric_difference, column_maximum
            )
            if column_maximum != 0.0:
                failures.append(
                    f"numeric values differ: {column}: {column_maximum}"
                )
        else:
            exact_non_numeric_columns.append(column)
            if not left.fillna("<NA>").equals(right.fillna("<NA>")):
                failures.append(f"values differ: {column}")
    if failures:
        raise RuntimeError("\n".join(failures))
    return {
        "row_count_equal": bool(len(v1) == len(v2)),
        "shared_columns_checked": compared,
        "excluded_metadata_columns": sorted(excluded_metadata),
        "numeric_columns_exact": numeric_columns,
        "non_numeric_columns_exact": exact_non_numeric_columns,
        "maximum_absolute_numeric_difference": (
            maximum_numeric_difference
        ),
        "all_shared_nonmetadata_values_exact": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=legacy.MIN_COVERAGE,
    )
    parser.add_argument(
        "--max-gap-hours",
        type=int,
        default=legacy.MAX_GAP_H,
    )
    args = parser.parse_args()
    root = args.root.resolve()

    frozen_hashes_before = require_frozen_inputs(root)
    v1_path = root / V1_PANEL
    v1 = pd.read_csv(v1_path)
    if len(v1) != 1_357_800:
        raise RuntimeError(
            f"Unexpected frozen v0.1 row count: {len(v1)}"
        )
    if v1[["timestamp_utc", "country", "year"]].duplicated().any():
        raise RuntimeError("Frozen v0.1 panel has duplicate hourly keys")

    flags, country_year_qa = build_hourly_flags(
        root,
        v1,
        args.coverage_threshold,
        args.max_gap_hours,
    )
    legacy_qa = pd.read_csv(root / V1_COUNTRY_YEAR_QA)[
        [
            "country",
            "year",
            "fossil_gas_mw_imputed_hours",
            "fossil_gas_mw_eligible",
        ]
    ]
    country_year_qa = country_year_qa.merge(
        legacy_qa,
        on=["country", "year"],
        how="left",
        validate="one_to_one",
    )
    country_year_qa["fill_count_exact_match_to_v0_1_qa"] = (
        country_year_qa["short_gap_filled_hours"].astype(int)
        == country_year_qa["fossil_gas_mw_imputed_hours"].astype(int)
    )
    country_year_qa["eligibility_exact_match_to_v0_1_qa"] = (
        country_year_qa["eligible"].astype(bool)
        == country_year_qa["fossil_gas_mw_eligible"].astype(bool)
    )
    if not country_year_qa[
        "fill_count_exact_match_to_v0_1_qa"
    ].all():
        raise RuntimeError("Country-year fill counts differ from v0.1 QA")
    if not country_year_qa[
        "eligibility_exact_match_to_v0_1_qa"
    ].all():
        raise RuntimeError("Country-year eligibility differs from v0.1 QA")

    sample = pd.read_csv(root / PRESPECIFIED_SAMPLE)[
        ["country", "year", "main_prespecified_sample"]
    ]
    sample_lookup = {
        (str(row.country), int(row.year)): bool(
            row.main_prespecified_sample
        )
        for row in sample.itertuples(index=False)
    }
    main_flag = np.fromiter(
        (
            sample_lookup[(str(country), int(year))]
            for country, year in zip(v1["country"], v1["year"])
        ),
        dtype=bool,
        count=len(v1),
    )

    v2 = v1.copy()
    v2["gas_task_source"] = (
        "Fraunhofer ISE Energy-Charts public_power: aggregate public-net "
        "Fossil gas"
    )
    v2["gas_task_provider_hourly_available_before_fill"] = flags[
        "gas_task_provider_hourly_available_before_fill"
    ].to_numpy(bool)
    v2["gas_task_short_gap_filled"] = flags[
        "gas_task_short_gap_filled"
    ].to_numpy(bool)
    v2["gas_task_fill_method"] = flags[
        "gas_task_fill_method"
    ].to_numpy(object)
    v2["gas_task_value_origin"] = flags[
        "gas_task_value_origin"
    ].to_numpy(object)
    v2["gas_task_main_prespecified_sample"] = main_flag
    v2["gas_task_data_contract_version"] = DATASET_VERSION
    v2["gas_task_model_contract"] = CURRENT_CONTRACT
    v2["gas_task_recharge_contract"] = RECHARGE_CONTRACT
    v2["gas_task_legacy_adapter_contract_status"] = (
        LEGACY_ADAPTER_STATUS
    )

    all_fill_count = int(v2["gas_task_short_gap_filled"].sum())
    main_fill_count = int(
        v2.loc[
            v2["gas_task_main_prespecified_sample"],
            "gas_task_short_gap_filled",
        ].sum()
    )
    if all_fill_count != 524:
        raise RuntimeError(
            f"Expected 524 eligible-panel fills, found {all_fill_count}"
        )
    if main_fill_count != 443:
        raise RuntimeError(
            f"Expected 443 main-sample fills, found {main_fill_count}"
        )
    method_counts = (
        v2.loc[
            v2["gas_task_short_gap_filled"],
            "gas_task_fill_method",
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    raw_frame, raw_manifest, raw_inventory_sha = build_raw_manifest(
        root
    )
    atomic_write_csv(raw_frame, root / RAW_MANIFEST_CSV)
    raw_manifest["csv"] = {
        "path": RAW_MANIFEST_CSV.as_posix(),
        "bytes": int((root / RAW_MANIFEST_CSV).stat().st_size),
        "sha256": sha256(root / RAW_MANIFEST_CSV),
    }
    atomic_write_text(
        root / RAW_MANIFEST_JSON,
        json.dumps(
            raw_manifest, ensure_ascii=False, indent=2
        )
        + "\n",
    )

    data_contract = {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": DATASET_VERSION,
        "release_date": RELEASE_DATE,
        "status": "current",
        "source": {
            "provider": (
                "Fraunhofer Institute for Solar Energy Systems ISE"
            ),
            "product": "Energy-Charts",
            "endpoint": f"{API_BASE}{API_ENDPOINT}",
            "exact_field": "Fossil gas",
            "scope": "aggregate public-net electricity generation",
            "direct_entsoe_api": False,
            "entsoe_semantic_mapping_candidate": (
                ENTSOE_SEMANTIC_MAPPING
            ),
        },
        "task_contract": CURRENT_CONTRACT,
        "recharge_contract": RECHARGE_CONTRACT,
        "short_gap_provenance": {
            "source_of_flags": (
                "missing mask before any interpolation or boundary filling"
            ),
            "interior_method": "time-linear within country UTC model-year",
            "boundary_method": (
                "nearest reported value within country UTC model-year"
            ),
            "minimum_coverage": float(args.coverage_threshold),
            "maximum_consecutive_gap_hours": int(
                args.max_gap_hours
            ),
            "all_eligible_panel_filled_hours": all_fill_count,
            "main_132_country_year_sample_filled_hours": main_fill_count,
        },
        "legacy_adapter_contract": {
            "status": LEGACY_ADAPTER_STATUS,
            "text": LEGACY_ADAPTER_CONTRACT,
            "numeric_aliases_retained_in_v0_2": [
                "load_mw",
                "wind_mw",
                "solar_mw",
            ],
            "retention_reason": (
                "numeric compatibility audit only; aliases do not define "
                "the current confirmatory data contract"
            ),
        },
    }
    atomic_write_text(
        root / DATA_CONTRACT_JSON,
        json.dumps(
            data_contract, ensure_ascii=False, indent=2
        )
        + "\n",
    )

    atomic_write_csv(country_year_qa, root / V2_COUNTRY_YEAR_QA)
    atomic_write_gzip_csv(v2, root / V2_PANEL)
    reread_v2 = pd.read_csv(root / V2_PANEL)
    comparison = compare_v1_v2(v1, reread_v2)

    script_relative = Path(__file__).resolve().relative_to(root)
    v2_manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": DATASET_VERSION,
        "release_date": RELEASE_DATE,
        "release_type": (
            "provenance annotation release with no numerical change from "
            "frozen v0.1"
        ),
        "inputs": {
            relative: {
                "sha256": digest,
                "frozen_expected_sha256": EXPECTED_INPUT_SHA256[relative],
                "hash_match": True,
            }
            for relative, digest in frozen_hashes_before.items()
        },
        "data_contract": {
            "path": DATA_CONTRACT_JSON.as_posix(),
            "sha256": sha256(root / DATA_CONTRACT_JSON),
            "current_task_contract": CURRENT_CONTRACT,
            "recharge_contract": RECHARGE_CONTRACT,
            "legacy_adapter_contract_status": LEGACY_ADAPTER_STATUS,
        },
        "hourly_panel": {
            "path": V2_PANEL.as_posix(),
            "bytes": int((root / V2_PANEL).stat().st_size),
            "sha256": sha256(root / V2_PANEL),
            "rows": int(len(v2)),
            "country_years": int(
                v2[["country", "year"]].drop_duplicates().shape[0]
            ),
            "countries": int(v2["country"].nunique()),
            "all_eligible_panel_short_gap_filled_hours": (
                all_fill_count
            ),
            "main_sample_short_gap_filled_hours": main_fill_count,
            "fill_method_counts": {
                str(key): int(value)
                for key, value in method_counts.items()
            },
        },
        "country_year_imputation_qa": {
            "path": V2_COUNTRY_YEAR_QA.as_posix(),
            "sha256": sha256(root / V2_COUNTRY_YEAR_QA),
            "all_fill_counts_match_v0_1_qa": bool(
                country_year_qa[
                    "fill_count_exact_match_to_v0_1_qa"
                ].all()
            ),
        },
        "frozen_raw_manifest": {
            "csv_path": RAW_MANIFEST_CSV.as_posix(),
            "csv_sha256": sha256(root / RAW_MANIFEST_CSV),
            "json_path": RAW_MANIFEST_JSON.as_posix(),
            "json_sha256": sha256(root / RAW_MANIFEST_JSON),
            "records": int(len(raw_frame)),
            "raw_content_inventory_sha256": raw_inventory_sha,
        },
        "v0_1_v0_2_comparison": comparison,
        "builder": {
            "path": script_relative.as_posix(),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "qa_path": V2_QA.as_posix(),
    }
    atomic_write_text(
        root / V2_MANIFEST_JSON,
        json.dumps(
            v2_manifest, ensure_ascii=False, indent=2
        )
        + "\n",
    )

    frozen_hashes_after = require_frozen_inputs(root)
    frozen_unchanged = frozen_hashes_before == frozen_hashes_after
    qa = {
        "schema_version": SCHEMA_VERSION,
        "qa_type": "energy_charts_fossil_gas_provenance_v0_2",
        "dataset_version": DATASET_VERSION,
        "v0_1_frozen_inputs_unchanged": frozen_unchanged,
        "v0_1_frozen_input_hashes": frozen_hashes_after,
        "outputs": {
            "panel": {
                "path": V2_PANEL.as_posix(),
                "bytes": int((root / V2_PANEL).stat().st_size),
                "sha256": sha256(root / V2_PANEL),
            },
            "country_year_imputation_qa": {
                "path": V2_COUNTRY_YEAR_QA.as_posix(),
                "bytes": int(
                    (root / V2_COUNTRY_YEAR_QA).stat().st_size
                ),
                "sha256": sha256(root / V2_COUNTRY_YEAR_QA),
            },
            "raw_manifest_csv": {
                "path": RAW_MANIFEST_CSV.as_posix(),
                "bytes": int(
                    (root / RAW_MANIFEST_CSV).stat().st_size
                ),
                "sha256": sha256(root / RAW_MANIFEST_CSV),
            },
            "raw_manifest_json": {
                "path": RAW_MANIFEST_JSON.as_posix(),
                "bytes": int(
                    (root / RAW_MANIFEST_JSON).stat().st_size
                ),
                "sha256": sha256(root / RAW_MANIFEST_JSON),
            },
            "data_contract": {
                "path": DATA_CONTRACT_JSON.as_posix(),
                "bytes": int(
                    (root / DATA_CONTRACT_JSON).stat().st_size
                ),
                "sha256": sha256(root / DATA_CONTRACT_JSON),
            },
            "dataset_manifest": {
                "path": V2_MANIFEST_JSON.as_posix(),
                "bytes": int(
                    (root / V2_MANIFEST_JSON).stat().st_size
                ),
                "sha256": sha256(root / V2_MANIFEST_JSON),
            },
        },
        "counts": {
            "hourly_rows": int(len(v2)),
            "country_years": int(len(country_year_qa)),
            "raw_public_power_files": int(len(raw_frame)),
            "raw_files_with_historic_receipts": int(
                raw_frame["historic_receipt_present"].sum()
            ),
            "raw_files_without_historic_receipts": int(
                (~raw_frame["historic_receipt_present"]).sum()
            ),
            "all_eligible_panel_short_gap_filled_hours": (
                all_fill_count
            ),
            "main_132_country_year_short_gap_filled_hours": (
                main_fill_count
            ),
            "fill_method_counts": {
                str(key): int(value)
                for key, value in method_counts.items()
            },
        },
        "comparison": comparison,
        "gates": {
            "v0_1_frozen_inputs_unchanged": frozen_unchanged,
            "all_155_country_year_fill_counts_match_v0_1_qa": bool(
                country_year_qa[
                    "fill_count_exact_match_to_v0_1_qa"
                ].all()
            ),
            "all_155_country_year_eligibility_matches_v0_1_qa": bool(
                country_year_qa[
                    "eligibility_exact_match_to_v0_1_qa"
                ].all()
            ),
            "main_sample_fill_flags_equal_443": (
                main_fill_count == 443
            ),
            "eligible_panel_fill_flags_equal_524": (
                all_fill_count == 524
            ),
            "v0_1_v0_2_shared_nonmetadata_values_exact": bool(
                comparison[
                    "all_shared_nonmetadata_values_exact"
                ]
            ),
            "v0_1_v0_2_max_numeric_difference_is_zero": bool(
                comparison[
                    "maximum_absolute_numeric_difference"
                ]
                == 0.0
            ),
            "raw_manifest_has_170_records": bool(
                len(raw_frame) == 170
            ),
            "raw_manifest_receipt_counts_match_86_11_73": bool(
                int(
                    (
                        raw_frame["historic_receipt_status"]
                        == "downloaded"
                    ).sum()
                )
                == 86
                and int(
                    (
                        raw_frame["historic_receipt_status"]
                        == "reused"
                    ).sum()
                )
                == 11
                and int(
                    (~raw_frame["historic_receipt_present"]).sum()
                )
                == 73
            ),
            "all_raw_source_manifest_hashes_match": bool(
                raw_frame["source_manifest_exact_match"].all()
            ),
            "current_contract_replaces_legacy_adapter": bool(
                set(v2["gas_task_data_contract_version"])
                == {DATASET_VERSION}
                and set(
                    v2[
                        "gas_task_legacy_adapter_contract_status"
                    ]
                )
                == {LEGACY_ADAPTER_STATUS}
            ),
        },
    }
    qa["pass"] = bool(all(qa["gates"].values()))
    if not qa["pass"]:
        raise RuntimeError(
            "v0.2 QA failed: "
            + json.dumps(qa["gates"], ensure_ascii=False)
        )
    atomic_write_text(
        root / V2_QA,
        json.dumps(qa, ensure_ascii=False, indent=2) + "\n",
    )

    print(
        json.dumps(
            {
                "pass": qa["pass"],
                "v0_2_panel": qa["outputs"]["panel"],
                "v0_2_manifest": qa["outputs"]["dataset_manifest"],
                "raw_manifest_csv": qa["outputs"][
                    "raw_manifest_csv"
                ],
                "raw_manifest_json": qa["outputs"][
                    "raw_manifest_json"
                ],
                "data_contract": qa["outputs"]["data_contract"],
                "qa": {
                    "path": V2_QA.as_posix(),
                    "main_sample_fill_flags": main_fill_count,
                    "eligible_panel_fill_flags": all_fill_count,
                    "maximum_numeric_difference": comparison[
                        "maximum_absolute_numeric_difference"
                    ],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
