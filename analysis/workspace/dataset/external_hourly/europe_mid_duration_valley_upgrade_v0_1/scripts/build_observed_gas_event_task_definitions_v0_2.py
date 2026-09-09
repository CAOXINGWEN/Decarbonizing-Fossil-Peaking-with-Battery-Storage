#!/usr/bin/env python3
"""Build the non-overwriting Stage-0 v0.2 provenance release.

Stage-0 v0.2 is an annotation-only successor to the frozen v0.1 table.  It
adopts a canonical aggregate-fossil-gas field, carries the provider/fill
lineage from the audited source panel, and fixes two downstream sensitivity
samples.  Every pre-existing numerical task, event, recharge and EFC value is
required to remain exactly equal to Stage-0 v0.1 after serialization.
"""

from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

RELEASE_DATE = "2026-07-27"
SCHEMA_VERSION = "2.0.0"
DATASET_VERSION = "v0.2"
TASK_CONTRACT = (
    "actual aggregate public-net fossil-gas generation is used as an "
    "event-task proxy"
)
CANONICAL_TASK_FIELD = "observed_aggregate_fossil_gas_generation_mw"
LEGACY_TASK_ALIAS = "observed_gas_peaking_task_mw"
LEGACY_ALIAS_STATUS = "deprecated_compatibility_only"

V1_STAGE0 = Path(
    "processed/observed_gas_event_task_definitions_v0_1.csv.gz"
)
V1_COUNTRY_YEAR_QA = Path(
    "artifacts/research/"
    "observed_gas_event_task_definitions_v0_1_country_year_qa.csv"
)
V1_MACHINE_QA = Path(
    "artifacts/research/OBSERVED_GAS_EVENT_TASK_STAGE0_QA.json"
)
V2_SOURCE_PANEL = Path(
    "processed/"
    "energy_charts_observed_gas_task_panel_2015_2024_v0_2.csv.gz"
)
V2_SOURCE_PANEL_QA = Path(
    "artifacts/research/"
    "energy_charts_observed_gas_task_panel_v0_2_qa.json"
)
V2_SOURCE_CONTRACT = Path(
    "metadata/energy_charts_observed_gas_task_data_contract_v0_2.json"
)
V2_SOURCE_MANIFEST = Path(
    "metadata/energy_charts_observed_gas_task_v0_2_manifest.json"
)
PRESPECIFIED_SAMPLE = Path(
    "artifacts/research/"
    "energy_charts_observed_gas_task_prespecified_sample.csv"
)
CONFIRMATORY_DESIGN = Path(
    "artifacts/research/OBSERVED_GAS_EVENT_TASK_CONFIRMATORY_DESIGN.md"
)

V2_STAGE0 = Path(
    "processed/observed_gas_event_task_definitions_v0_2.csv.gz"
)
V2_COUNTRY_YEAR_QA = Path(
    "artifacts/research/"
    "observed_gas_event_task_definitions_v0_2_country_year_qa.csv"
)
V2_MACHINE_QA = Path(
    "artifacts/research/OBSERVED_GAS_EVENT_TASK_STAGE0_V0_2_QA.json"
)
V2_CONFIG = Path(
    "metadata/observed_gas_event_task_stage0_v0_2_config.json"
)
V2_MANIFEST = Path(
    "metadata/observed_gas_event_task_stage0_v0_2_manifest.json"
)
V2_MEMO = Path(
    "artifacts/research/OBSERVED_GAS_EVENT_TASK_STAGE0_V0_2_MEMO.md"
)
V2_RELEASE_AUDIT = Path(
    "artifacts/research/"
    "OBSERVED_GAS_EVENT_TASK_STAGE0_V0_2_RELEASE_AUDIT.md"
)
V2_RELEASE_STATE = Path(
    "metadata/observed_gas_event_task_stage0_v0_2_release_state.json"
)
V2_BUILD_LOCK = Path(
    "metadata/.observed_gas_event_task_stage0_v0_2_build.lock"
)

V2_SUITE_COMMIT_ORDER = [
    V2_STAGE0,
    V2_COUNTRY_YEAR_QA,
    V2_CONFIG,
    V2_MACHINE_QA,
    V2_MEMO,
    V2_MANIFEST,
    V2_RELEASE_AUDIT,
]

EXPECTED_INPUT_SHA256 = {
    str(V1_STAGE0): (
        "2a901503a5a517723493b5bdfb3c26ee4ee8dc4b784d922bd26d419e8fd47d50"
    ),
    str(V1_COUNTRY_YEAR_QA): (
        "c3843a31f1945e029798eb218a12fe282e8044a4a01b0d21a7c03862791a8b4c"
    ),
    str(V1_MACHINE_QA): (
        "60ac201c64a2979bc640351159d012d53f3a7f4464e227bc1307de42614a5162"
    ),
    str(V2_SOURCE_PANEL): (
        "31cd2b0f4b8a2f38e37addf77296affe9629155bf7a97870e7e650aebaac26ce"
    ),
    str(V2_SOURCE_PANEL_QA): (
        "ec3bd11794c77b228dcbc203139e6395b2aca97daebc37ad3b642ba8c41b9ae7"
    ),
    str(V2_SOURCE_CONTRACT): (
        "34a52ba7dfedf487c2cb033ce549f23563ea5238e895d05fd22930d746c1678a"
    ),
    str(V2_SOURCE_MANIFEST): (
        "7e37f44cb0f1a0a08d6336595a179d627d5142fa03bfa4610201786888db1c7c"
    ),
    str(PRESPECIFIED_SAMPLE): (
        "cf1bbfae8cf1212ab3aeab2b70a5fc7cb9148c6f308310e385a7d7f043ff9ef8"
    ),
    str(CONFIRMATORY_DESIGN): (
        "e7ff3fbc18f94afce719cb43c75195f9effde194a452a4571572d390d1300d5f"
    ),
}

EXPECTED_ROWS = 1_156_320
EXPECTED_COUNTRY_YEARS = 132
EXPECTED_COUNTRIES = 15
EXPECTED_FILLED_HOURS = 443
EXPECTED_ZERO_FILL_COUNTRY_YEARS = 76

SOURCE_PROVENANCE_COLUMNS = [
    "timestamp_utc",
    "country",
    "year",
    LEGACY_TASK_ALIAS,
    "gas_task_provider_hourly_available_before_fill",
    "gas_task_short_gap_filled",
    "gas_task_fill_method",
    "gas_task_value_origin",
    "gas_task_main_prespecified_sample",
]

DERIVED_NUMERIC_COLUMNS = [
    "source_hour_id",
    "source_day_id",
    "source_hour_of_day",
    "calendar_month_utc",
    "season_segment",
    "season_days",
    "season_day_rank_by_above_floor_desc",
    "month_hour_q10_floor_mw",
    "above_floor_gas_mw",
    "daily_above_floor_gas_mwh",
    "season_daily_above_floor_q70_mwh",
    "season_daily_above_floor_q75_mwh",
    "season_daily_above_floor_q80_mwh",
    "event_target_days_q70",
    "event_target_days_q75",
    "event_target_days_q80",
    "event_flag_q70",
    "event_flag_q75",
    "event_flag_q80",
    "task_c0_mw",
    "task_c1_mw",
    "task_c2_q75_mw",
    "task_c3_q70_mw",
    "task_c3_q75_mw",
    "task_c3_q80_mw",
    "task_c4_q75_mw",
    "raw_gas_mean_mw",
    "forced_discharge_power_mw",
    "lfp4_nominal_energy_mwh",
    "caes24_nominal_energy_mwh",
    "r2_scaling_alpha",
    "r2_potential_recharge_mw",
    "task_efc_upper_c0_lfp4",
    "task_efc_upper_c0_caes24",
    "task_efc_upper_c1_lfp4",
    "task_efc_upper_c1_caes24",
    "task_efc_upper_c2_q75_lfp4",
    "task_efc_upper_c2_q75_caes24",
    "task_efc_upper_c3_q70_lfp4",
    "task_efc_upper_c3_q70_caes24",
    "task_efc_upper_c3_q75_lfp4",
    "task_efc_upper_c3_q75_caes24",
    "task_efc_upper_c3_q80_lfp4",
    "task_efc_upper_c3_q80_caes24",
    "task_efc_upper_c4_q75_lfp4",
    "task_efc_upper_c4_q75_caes24",
]

OLD_HASH_COLUMNS = [
    "source_panel_sha256",
    "prespecified_sample_sha256",
    "confirmatory_design_sha256",
    "derivation_script_sha256",
    "definition_config_sha256",
]

NEW_PROVENANCE_COLUMNS = [
    CANONICAL_TASK_FIELD,
    "observed_aggregate_fossil_gas_provider_value_mw",
    "observed_aggregate_fossil_gas_provider_value_available_before_fill",
    "observed_aggregate_fossil_gas_short_gap_filled",
    "observed_aggregate_fossil_gas_fill_method",
    "observed_aggregate_fossil_gas_value_origin",
    "observed_gas_peaking_task_mw_alias_status",
    "country_year_short_gap_filled_hours",
    "preregistered_sensitivity_s1_include_excluding_filled_hours",
    "preregistered_sensitivity_s2_zero_fill_country_year",
]

NEW_LINEAGE_COLUMNS = [
    "stage0_v0_1_output_sha256",
    "source_panel_data_contract_sha256",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def gzip_payload_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(
        temporary,
        index=False,
        float_format="%.15g",
        lineterminator="\n",
    )
    temporary.replace(path)


def atomic_write_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_handle,
            compresslevel=6,
            mtime=0,
        ) as gzip_handle:
            with io.TextIOWrapper(
                gzip_handle,
                encoding="utf-8",
                newline="",
            ) as text_handle:
                frame.to_csv(
                    text_handle,
                    index=False,
                    float_format="%.15g",
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


def write_release_state(
    root: Path,
    status: str,
    details: dict[str, Any],
) -> None:
    state = {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": DATASET_VERSION,
        "release_date": RELEASE_DATE,
        "status": status,
        **details,
    }
    atomic_write_text(
        root / V2_RELEASE_STATE,
        json.dumps(state, indent=2, ensure_ascii=False) + "\n",
    )


def commit_staged_suite(stage_root: Path, root: Path) -> None:
    missing = [
        str(relative)
        for relative in V2_SUITE_COMMIT_ORDER
        if not (stage_root / relative).is_file()
    ]
    if missing:
        raise RuntimeError(
            "Staged release suite is incomplete: " + ", ".join(missing)
        )
    for relative in V2_SUITE_COMMIT_ORDER:
        final = root / relative
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage_root / relative, final)


def build_config(v1_machine_qa: dict[str, Any]) -> dict[str, Any]:
    v1_definitions = v1_machine_qa["definitions"]
    frozen_numerical_definitions = {
        key: value
        for key, value in v1_definitions.items()
        if key not in {"stage", "version", "task_source_field"}
    }
    frozen_numerical_definitions["task_definitions"] = {
        "notation": {
            "G_t": CANONICAL_TASK_FIELD,
            "F_month_hour": "month-hour q10 floor",
            "X_t": "max(G_t-F_month_hour,0)",
            "A_d": "sum of X_t over the intact UTC day",
            "I_q_d": (
                "season-specific ranked event-day indicator at quantile q"
            ),
            "Q_q_s": (
                "linear quantile q of A_d within the fixed season segment"
            ),
        },
        "C0": "G_t",
        "C1": "X_t",
        "C2_q75": "G_t * I_q75_d",
        "C3_q70": "X_t * I_q70_d",
        "C3_q75": "X_t * I_q75_d",
        "C3_q80": "X_t * I_q80_d",
        "C4_q75": "X_t * max(A_d-Q_q75_s,0)/A_d; zero when A_d=0",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": DATASET_VERSION,
        "release_date": RELEASE_DATE,
        "stage": 0,
        "release_type": (
            "canonical-field and hourly-provenance annotation release; "
            "no numerical-definition change"
        ),
        "task_contract": TASK_CONTRACT,
        "canonical_task_source_field": CANONICAL_TASK_FIELD,
        "canonical_source_scope": "aggregate public-net electricity generation",
        "legacy_alias": {
            "field": LEGACY_TASK_ALIAS,
            "status": LEGACY_ALIAS_STATUS,
            "required_identity_max_abs_difference_mw": 0.0,
        },
        "frozen_numerical_definitions": frozen_numerical_definitions,
        "inherited_stage0_v0_1_definition_config_sha256": (
            v1_machine_qa["hashes"]["definition_config_sha256"]
        ),
        "numerical_invariance_gate": {
            "reference": str(V1_STAGE0),
            "columns": DERIVED_NUMERIC_COLUMNS,
            "required_maximum_absolute_difference": 0.0,
        },
        "hourly_provenance": {
            "provider_value": (
                "provider-reported value before filling; missing on a "
                "tracked filled hour"
            ),
            "short_gap_filled": (
                "true only when the pre-fill provider value was missing "
                "and the audited short-gap rule supplied the final value"
            ),
            "fill_method": (
                "none, time_linear_interior, or "
                "nearest_value_year_boundary in the main sample"
            ),
        },
        "preregistered_sensitivity_sets": {
            "s1_exclude_filled_hours": {
                "unit": "hour",
                "include_when": (
                    "observed_aggregate_fossil_gas_short_gap_filled is false"
                ),
                "expected_excluded_hours": EXPECTED_FILLED_HOURS,
            },
            "s2_zero_fill_country_years": {
                "unit": "complete country-year",
                "include_when": (
                    "country_year_short_gap_filled_hours equals zero"
                ),
                "expected_country_years": EXPECTED_ZERO_FILL_COUNTRY_YEARS,
            },
            "registration_scope": (
                "fixed in Stage-0 v0.2 before any downstream v0.2 "
                "effect estimation; no external registry timestamp claimed"
            ),
        },
        "out_of_scope": [
            "chronology intervention",
            "optimization",
            "effect estimation",
            "Stage-1 solver modification",
            "Stage-2 contract modification",
            "manuscript modification",
        ],
    }


def load_and_align(
    root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    v1 = pd.read_csv(root / V1_STAGE0)
    source = pd.read_csv(
        root / V2_SOURCE_PANEL,
        usecols=SOURCE_PROVENANCE_COLUMNS,
    )
    source["gas_task_main_prespecified_sample"] = (
        source["gas_task_main_prespecified_sample"]
        .astype(str)
        .str.lower()
        .isin({"true", "1"})
    )
    source = source[
        source["gas_task_main_prespecified_sample"]
    ].copy()
    source = source.sort_values(
        ["country", "year", "timestamp_utc"],
        kind="mergesort",
    ).reset_index(drop=True)
    v1 = v1.sort_values(
        ["country", "year", "source_hour_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    v1_time = pd.to_datetime(v1["timestamp_utc"], utc=True, errors="raise")
    source_time = pd.to_datetime(
        source["timestamp_utc"], utc=True, errors="raise"
    )
    keys_exact = bool(
        len(v1) == len(source)
        and v1["country"].equals(source["country"])
        and v1["year"].equals(source["year"])
        and np.array_equal(
            v1_time.astype("int64").to_numpy(),
            source_time.astype("int64").to_numpy(),
        )
    )
    if not keys_exact:
        raise RuntimeError("Stage-0 v0.1 and source-panel v0.2 keys differ")

    alias = v1[LEGACY_TASK_ALIAS].to_numpy(dtype=float)
    source_values = source[LEGACY_TASK_ALIAS].to_numpy(dtype=float)
    source_max_diff = float(np.max(np.abs(alias - source_values)))
    if source_max_diff != 0.0:
        raise RuntimeError(
            "Canonical source value differs from Stage-0 v0.1 alias: "
            f"{source_max_diff}"
        )
    return v1, source, {
        "hourly_keys_exact": keys_exact,
        "source_value_maximum_absolute_difference_mw": source_max_diff,
    }


def annotate(
    v1: pd.DataFrame,
    source: pd.DataFrame,
    hashes: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    filled = source["gas_task_short_gap_filled"].astype(bool)
    provider_available = source[
        "gas_task_provider_hourly_available_before_fill"
    ].astype(bool)
    canonical = source[LEGACY_TASK_ALIAS].astype(float)

    fill_counts = (
        pd.DataFrame(
            {
                "country": source["country"],
                "year": source["year"],
                "short_gap_filled": filled.astype(int),
            }
        )
        .groupby(["country", "year"], sort=True)["short_gap_filled"]
        .sum()
        .rename("country_year_short_gap_filled_hours")
    )
    row_fill_counts = pd.MultiIndex.from_frame(
        source[["country", "year"]]
    ).map(fill_counts)
    row_fill_counts = pd.Series(
        row_fill_counts,
        index=v1.index,
        dtype="int64",
    )
    zero_fill_cy = row_fill_counts.eq(0)

    output = v1.drop(columns=OLD_HASH_COLUMNS).copy()
    alias_position = output.columns.get_loc(LEGACY_TASK_ALIAS)
    output.insert(alias_position, CANONICAL_TASK_FIELD, canonical.to_numpy())
    insert_position = output.columns.get_loc(LEGACY_TASK_ALIAS) + 1
    provider_value = canonical.where(provider_available, np.nan)
    added = {
        "observed_aggregate_fossil_gas_provider_value_mw": (
            provider_value.to_numpy()
        ),
        "observed_aggregate_fossil_gas_provider_value_available_before_fill": (
            provider_available.to_numpy()
        ),
        "observed_aggregate_fossil_gas_short_gap_filled": filled.to_numpy(),
        "observed_aggregate_fossil_gas_fill_method": source[
            "gas_task_fill_method"
        ].to_numpy(),
        "observed_aggregate_fossil_gas_value_origin": source[
            "gas_task_value_origin"
        ].to_numpy(),
        "observed_gas_peaking_task_mw_alias_status": LEGACY_ALIAS_STATUS,
        "country_year_short_gap_filled_hours": row_fill_counts.to_numpy(),
        "preregistered_sensitivity_s1_include_excluding_filled_hours": (
            (~filled).to_numpy()
        ),
        "preregistered_sensitivity_s2_zero_fill_country_year": (
            zero_fill_cy.to_numpy()
        ),
    }
    for offset, (column, values) in enumerate(added.items()):
        output.insert(insert_position + offset, column, values)

    output["source_panel_sha256"] = hashes["source_panel_sha256"]
    output["prespecified_sample_sha256"] = hashes[
        "prespecified_sample_sha256"
    ]
    output["confirmatory_design_sha256"] = hashes[
        "confirmatory_design_sha256"
    ]
    output["derivation_script_sha256"] = hashes[
        "derivation_script_sha256"
    ]
    output["definition_config_sha256"] = hashes[
        "definition_config_sha256"
    ]
    output["stage0_v0_1_output_sha256"] = hashes[
        "stage0_v0_1_output_sha256"
    ]
    output["source_panel_data_contract_sha256"] = hashes[
        "source_panel_data_contract_sha256"
    ]

    provenance = (
        pd.DataFrame(
            {
                "country": source["country"],
                "year": source["year"],
                "provider_value_hours": provider_available.astype(int),
                "short_gap_filled_hours": filled.astype(int),
                "time_linear_interior_hours": source[
                    "gas_task_fill_method"
                ].eq("time_linear_interior").astype(int),
                "nearest_value_year_boundary_hours": source[
                    "gas_task_fill_method"
                ].eq("nearest_value_year_boundary").astype(int),
            }
        )
        .groupby(["country", "year"], as_index=False, sort=True)
        .sum()
    )
    provenance["zero_fill_country_year"] = provenance[
        "short_gap_filled_hours"
    ].eq(0)
    provenance[
        "s1_retained_hours_excluding_filled"
    ] = provenance["provider_value_hours"]
    provenance[
        "s2_retained_complete_country_year_hours"
    ] = np.where(
        provenance["zero_fill_country_year"],
        8760,
        0,
    )

    checks = {
        "rows_equal_1156320": len(output) == EXPECTED_ROWS,
        "country_years_equal_132": len(provenance)
        == EXPECTED_COUNTRY_YEARS,
        "countries_equal_15": provenance["country"].nunique()
        == EXPECTED_COUNTRIES,
        "filled_hours_equal_443": int(filled.sum())
        == EXPECTED_FILLED_HOURS,
        "provider_and_fill_flags_partition_rows": int(
            provider_available.sum() + filled.sum()
        )
        == EXPECTED_ROWS,
        "provider_and_fill_flags_are_disjoint": not bool(
            (provider_available & filled).any()
        ),
        "provider_value_missing_exactly_on_filled_hours": bool(
            provider_value.isna().equals(filled)
        ),
        "zero_fill_country_years_equal_76": int(
            provenance["zero_fill_country_year"].sum()
        )
        == EXPECTED_ZERO_FILL_COUNTRY_YEARS,
        "s1_retains_1155877_hours": int((~filled).sum())
        == EXPECTED_ROWS - EXPECTED_FILLED_HOURS,
        "s2_retains_665760_hours": int(zero_fill_cy.sum())
        == EXPECTED_ZERO_FILL_COUNTRY_YEARS * 8760,
        "canonical_equals_legacy_alias": bool(
            np.array_equal(
                output[CANONICAL_TASK_FIELD].to_numpy(float),
                output[LEGACY_TASK_ALIAS].to_numpy(float),
            )
        ),
        "legacy_alias_marked_deprecated_on_every_row": bool(
            output[
                "observed_gas_peaking_task_mw_alias_status"
            ].eq(LEGACY_ALIAS_STATUS).all()
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "Stage-0 v0.2 annotation checks failed:\n"
            + json.dumps(checks, indent=2)
        )
    return output, provenance, checks


def numerical_comparison(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
) -> dict[str, Any]:
    column_differences: dict[str, float] = {}
    for column in DERIVED_NUMERIC_COLUMNS:
        left = reference[column].to_numpy(dtype=float)
        right = candidate[column].to_numpy(dtype=float)
        column_differences[column] = float(np.max(np.abs(left - right)))
    maximum = max(column_differences.values(), default=0.0)
    return {
        "reference": str(V1_STAGE0),
        "columns_checked": DERIVED_NUMERIC_COLUMNS,
        "column_maximum_absolute_differences": column_differences,
        "maximum_absolute_derived_numeric_difference": maximum,
        "all_derived_numeric_columns_exact": maximum == 0.0,
    }


def build_country_year_qa(
    root: Path,
    provenance: pd.DataFrame,
    hashes: dict[str, str],
) -> pd.DataFrame:
    qa = pd.read_csv(root / V1_COUNTRY_YEAR_QA)
    qa = qa.merge(
        provenance,
        on=["country", "year"],
        how="left",
        validate="one_to_one",
    )
    qa["v0_2_annotation_qa_pass"] = (
        qa["qa_pass"].astype(bool)
        & qa["provider_value_hours"]
        .add(qa["short_gap_filled_hours"])
        .eq(qa["hours"])
    )
    qa["source_panel_sha256"] = hashes["source_panel_sha256"]
    qa["prespecified_sample_sha256"] = hashes[
        "prespecified_sample_sha256"
    ]
    qa["confirmatory_design_sha256"] = hashes[
        "confirmatory_design_sha256"
    ]
    qa["derivation_script_sha256"] = hashes[
        "derivation_script_sha256"
    ]
    qa["definition_config_sha256"] = hashes[
        "definition_config_sha256"
    ]
    qa["stage0_v0_1_output_sha256"] = hashes[
        "stage0_v0_1_output_sha256"
    ]
    qa["source_panel_data_contract_sha256"] = hashes[
        "source_panel_data_contract_sha256"
    ]
    if not bool(qa["v0_2_annotation_qa_pass"].all()):
        raise RuntimeError("A country-year failed Stage-0 v0.2 annotation QA")
    return qa


def build_memo(qa: dict[str, Any]) -> str:
    comparison = qa["v0_1_v0_2_numerical_comparison"]
    sensitivity = qa["sensitivity_sets"]
    return f"""# Observed fossil-gas event task Stage 0 v0.2 memo

Release date: {RELEASE_DATE}

## Outcome

Stage 0 v0.2 adopts `{CANONICAL_TASK_FIELD}` as the canonical task-source
field. Its contract is an **actual aggregate public-net fossil-gas event-task
proxy**. The numerical alias
`observed_gas_peaking_task_mw` is retained only for compatibility and is
marked `{LEGACY_ALIAS_STATUS}` in every row.

This is an annotation-only release. It does not alter chronology, task
construction, recharge construction, asset sizing, EFC calculation or any
optimization input.

## Numerical invariance

The serialized v0.2 table was compared row by row with frozen Stage 0 v0.1.
All {len(comparison['columns_checked'])} derived numerical and Boolean columns
covering C0--C4, C3-q75, event definitions, R2, fixed sizing and task-side EFC
are exact. The maximum absolute difference is
`{comparison['maximum_absolute_derived_numeric_difference']:.1f}`.

The canonical source and deprecated alias are also exactly equal, with a
maximum absolute difference of `0.0 MW`.

## Hourly provenance

Each of the {qa['coverage']['rows']:,} main-sample hours carries:

- the provider value before filling, blank only on a tracked filled hour;
- the provider-value availability flag before filling;
- the short-gap-filled flag;
- the exact fill method and value origin.

The main sample contains {qa['provenance']['short_gap_filled_hours']} tracked
filled hours, of which
{qa['provenance']['time_linear_interior_hours']} are interior linear fills and
{qa['provenance']['nearest_value_year_boundary_hours']} are boundary
nearest-value fills.

## Fixed sensitivity samples

Two sensitivity samples are fixed before any downstream v0.2 effect
estimation:

1. S1 excludes the {sensitivity['s1_exclude_filled_hours']['excluded_hours']}
   tracked filled hours and retains
   {sensitivity['s1_exclude_filled_hours']['retained_hours']:,} provider-value
   hours.
2. S2 retains the
   {sensitivity['s2_zero_fill_country_years']['country_years']} complete
   country-years with no filled hour, totalling
   {sensitivity['s2_zero_fill_country_years']['retained_hours']:,} hours.

These are sample-membership flags only. Stage 0 v0.2 does not estimate either
sensitivity result.

## Scope

No Stage 1 solver, Stage 2 contract or manuscript file is read for mutation or
written by this builder.
"""


def build_release_audit(
    qa: dict[str, Any],
    output_hashes: dict[str, str],
) -> str:
    comparison = qa["v0_1_v0_2_numerical_comparison"]
    return f"""# Observed fossil-gas event task Stage 0 v0.2 release audit

Release date: {RELEASE_DATE}

## Release decision

**PASS.** Stage 0 v0.2 is a non-overwriting provenance release for the
**actual aggregate public-net fossil-gas event-task proxy**.

The canonical task field is `{CANONICAL_TASK_FIELD}`. The retained
`{LEGACY_TASK_ALIAS}` field is marked `{LEGACY_ALIAS_STATUS}` and has zero
numerical difference from the canonical field.

## Frozen numerical contract

- Rows: {qa['coverage']['rows']:,}
- Countries: {qa['coverage']['countries']}
- Country-years: {qa['coverage']['country_years']}
- Frozen derived columns compared: {len(comparison['columns_checked'])}
- Maximum absolute v0.1/v0.2 derived numerical difference:
  `{comparison['maximum_absolute_derived_numeric_difference']:.1f}`
- C0--C4, C3-q75 and R2 numerical definitions changed: **no**

## Provenance and sensitivity gates

- Provider-value hours: {qa['provenance']['provider_value_hours']:,}
- Tracked filled hours: {qa['provenance']['short_gap_filled_hours']}
- Interior linear fills:
  {qa['provenance']['time_linear_interior_hours']}
- Boundary nearest-value fills:
  {qa['provenance']['nearest_value_year_boundary_hours']}
- Zero-fill country-years:
  {qa['sensitivity_sets']['s2_zero_fill_country_years']['country_years']}
- S1 retained hours:
  {qa['sensitivity_sets']['s1_exclude_filled_hours']['retained_hours']:,}
- S2 retained hours:
  {qa['sensitivity_sets']['s2_zero_fill_country_years']['retained_hours']:,}

Both sensitivity fields are membership flags fixed before downstream v0.2
analysis. No effect estimate is produced here.

## Frozen-input protection

All nine frozen inputs were checked before writing and rechecked afterwards.
Every observed SHA-256 matched its embedded expected value. No v0.1 path was
written.

## Output inventory

```text
scripts/build_observed_gas_event_task_definitions_v0_2.py
{output_hashes['builder']}

processed/observed_gas_event_task_definitions_v0_2.csv.gz
{output_hashes['hourly_table']}

decompressed CSV payload SHA-256
{output_hashes['hourly_table_decompressed_csv']}

artifacts/research/observed_gas_event_task_definitions_v0_2_country_year_qa.csv
{output_hashes['country_year_qa']}

artifacts/research/OBSERVED_GAS_EVENT_TASK_STAGE0_V0_2_QA.json
{output_hashes['machine_qa']}

metadata/observed_gas_event_task_stage0_v0_2_config.json
{output_hashes['config']}

metadata/observed_gas_event_task_stage0_v0_2_manifest.json
{output_hashes['manifest']}

artifacts/research/OBSERVED_GAS_EVENT_TASK_STAGE0_V0_2_MEMO.md
{output_hashes['memo']}

metadata/observed_gas_event_task_stage0_v0_2_release_state.json
written last as the downstream release gate
```

## Reproduction

Run from the repository root:

```bash
/home/cxw/anaconda3/bin/python \\
  dataset/external_hourly/europe_mid_duration_valley_upgrade_v0_1/scripts/\\
build_observed_gas_event_task_definitions_v0_2.py
```

The builder performs no network access. It holds an exclusive build lock,
marks the release `invalidated_in_progress`, generates the complete suite in a
temporary directory, replaces each final file atomically, verifies every
manifest reference, and writes the final `pass` release state last.
"""


def build_staged_release(
    root: Path,
    stage_root: Path,
    frozen_before: dict[str, str],
) -> dict[str, Any]:
    v1_machine_qa = json.loads(
        (root / V1_MACHINE_QA).read_text(encoding="utf-8")
    )
    config = build_config(v1_machine_qa)
    config_text = json.dumps(config, indent=2, ensure_ascii=False) + "\n"
    atomic_write_text(stage_root / V2_CONFIG, config_text)

    hashes = {
        "source_panel_sha256": frozen_before[str(V2_SOURCE_PANEL)],
        "prespecified_sample_sha256": frozen_before[
            str(PRESPECIFIED_SAMPLE)
        ],
        "confirmatory_design_sha256": frozen_before[
            str(CONFIRMATORY_DESIGN)
        ],
        "derivation_script_sha256": sha256(Path(__file__).resolve()),
        "definition_config_sha256": sha256(stage_root / V2_CONFIG),
        "stage0_v0_1_output_sha256": frozen_before[str(V1_STAGE0)],
        "source_panel_data_contract_sha256": frozen_before[
            str(V2_SOURCE_CONTRACT)
        ],
    }

    v1, source, alignment_checks = load_and_align(root)
    output, provenance, annotation_checks = annotate(
        v1,
        source,
        hashes,
    )
    atomic_write_csv_gz(output, stage_root / V2_STAGE0)

    serialized = pd.read_csv(stage_root / V2_STAGE0)
    comparison = numerical_comparison(v1, serialized)
    if not comparison["all_derived_numeric_columns_exact"]:
        raise RuntimeError(
            "Serialized v0.2 numerical values differ from Stage-0 v0.1"
        )

    canonical_alias_max_diff = float(
        np.max(
            np.abs(
                serialized[CANONICAL_TASK_FIELD].to_numpy(float)
                - serialized[LEGACY_TASK_ALIAS].to_numpy(float)
            )
        )
    )
    if canonical_alias_max_diff != 0.0:
        raise RuntimeError("Serialized canonical field and alias differ")

    country_year_qa = build_country_year_qa(root, provenance, hashes)
    atomic_write_csv(country_year_qa, stage_root / V2_COUNTRY_YEAR_QA)

    filled_rows = source["gas_task_short_gap_filled"].astype(bool)
    fill_methods = source.loc[
        filled_rows,
        "gas_task_fill_method",
    ].value_counts()
    provider_hours = int(
        source[
            "gas_task_provider_hourly_available_before_fill"
        ].astype(bool).sum()
    )
    zero_fill_country_years = int(
        provenance["zero_fill_country_year"].sum()
    )
    s2_hours = int(
        country_year_qa[
            "s2_retained_complete_country_year_hours"
        ].sum()
    )
    checks = {
        "hourly_keys_exact": alignment_checks["hourly_keys_exact"],
        "source_value_maximum_absolute_difference_zero": (
            alignment_checks[
                "source_value_maximum_absolute_difference_mw"
            ]
            == 0.0
        ),
        **annotation_checks,
        "serialized_derived_numeric_max_difference_zero": (
            comparison[
                "maximum_absolute_derived_numeric_difference"
            ]
            == 0.0
        ),
        "serialized_canonical_alias_max_difference_zero": (
            canonical_alias_max_diff == 0.0
        ),
        "all_country_year_annotation_rows_pass": bool(
            country_year_qa["v0_2_annotation_qa_pass"].all()
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "Stage-0 v0.2 QA failed:\n"
            + json.dumps(checks, indent=2)
        )

    qa: dict[str, Any] = {
        "status": "pass",
        "schema_version": SCHEMA_VERSION,
        "dataset_version": DATASET_VERSION,
        "release_date": RELEASE_DATE,
        "stage": 0,
        "release_type": (
            "canonical-field and hourly-provenance annotation release; "
            "no numerical-definition change"
        ),
        "task_contract": TASK_CONTRACT,
        "scope": (
            "task construction and data-contract audit only; no chronology "
            "intervention, optimization or effect estimation"
        ),
        "checks": checks,
        "coverage": {
            "rows": int(len(serialized)),
            "countries": int(serialized["country"].nunique()),
            "country_years": int(
                serialized[["country", "year"]]
                .drop_duplicates()
                .shape[0]
            ),
        },
        "canonical_field": {
            "name": CANONICAL_TASK_FIELD,
            "source_scope": "aggregate public-net electricity generation",
            "legacy_alias": LEGACY_TASK_ALIAS,
            "legacy_alias_status": LEGACY_ALIAS_STATUS,
            "canonical_alias_maximum_absolute_difference_mw": (
                canonical_alias_max_diff
            ),
        },
        "provenance": {
            "provider_value_hours": provider_hours,
            "short_gap_filled_hours": int(filled_rows.sum()),
            "time_linear_interior_hours": int(
                fill_methods.get("time_linear_interior", 0)
            ),
            "nearest_value_year_boundary_hours": int(
                fill_methods.get("nearest_value_year_boundary", 0)
            ),
            "source_panel": str(V2_SOURCE_PANEL),
            "source_panel_data_contract": str(V2_SOURCE_CONTRACT),
        },
        "sensitivity_sets": {
            "s1_exclude_filled_hours": {
                "marker": (
                    "preregistered_sensitivity_s1_include_"
                    "excluding_filled_hours"
                ),
                "excluded_hours": int(filled_rows.sum()),
                "retained_hours": int((~filled_rows).sum()),
            },
            "s2_zero_fill_country_years": {
                "marker": (
                    "preregistered_sensitivity_s2_zero_fill_country_year"
                ),
                "country_years": zero_fill_country_years,
                "retained_hours": s2_hours,
            },
            "registration_scope": config[
                "preregistered_sensitivity_sets"
            ]["registration_scope"],
        },
        "v0_1_v0_2_numerical_comparison": comparison,
        "frozen_inputs": {
            relative: {
                "sha256": observed,
                "frozen_expected_sha256": EXPECTED_INPUT_SHA256[relative],
                "hash_match": (
                    observed == EXPECTED_INPUT_SHA256[relative]
                ),
            }
            for relative, observed in frozen_before.items()
        },
        "hashes": {
            **hashes,
            "hourly_output_sha256": sha256(stage_root / V2_STAGE0),
            "hourly_output_decompressed_csv_sha256": gzip_payload_sha256(
                stage_root / V2_STAGE0
            ),
            "country_year_qa_sha256": sha256(
                stage_root / V2_COUNTRY_YEAR_QA
            ),
        },
        "outputs": {
            "hourly_table": str(V2_STAGE0),
            "country_year_qa": str(V2_COUNTRY_YEAR_QA),
            "machine_qa": str(V2_MACHINE_QA),
            "config": str(V2_CONFIG),
            "manifest": str(V2_MANIFEST),
            "memo": str(V2_MEMO),
            "release_audit": str(V2_RELEASE_AUDIT),
        },
        "output_columns": list(serialized.columns),
    }
    atomic_write_text(
        stage_root / V2_MACHINE_QA,
        json.dumps(qa, indent=2, ensure_ascii=False) + "\n",
    )
    atomic_write_text(stage_root / V2_MEMO, build_memo(qa))

    frozen_after = require_frozen_inputs(root)
    if frozen_before != frozen_after:
        raise RuntimeError("A frozen input changed during the build")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": DATASET_VERSION,
        "release_date": RELEASE_DATE,
        "release_type": qa["release_type"],
        "task_contract": TASK_CONTRACT,
        "inputs": qa["frozen_inputs"],
        "canonical_field": qa["canonical_field"],
        "sensitivity_sets": qa["sensitivity_sets"],
        "numerical_comparison": {
            "reference": str(V1_STAGE0),
            "columns_checked": len(DERIVED_NUMERIC_COLUMNS),
            "maximum_absolute_derived_numeric_difference": comparison[
                "maximum_absolute_derived_numeric_difference"
            ],
            "all_derived_numeric_columns_exact": comparison[
                "all_derived_numeric_columns_exact"
            ],
        },
        "outputs": {
            str(V2_STAGE0): {
                "sha256": sha256(stage_root / V2_STAGE0),
                "decompressed_csv_sha256": gzip_payload_sha256(
                    stage_root / V2_STAGE0
                ),
                "bytes": (stage_root / V2_STAGE0).stat().st_size,
                "rows": int(len(serialized)),
            },
            str(V2_COUNTRY_YEAR_QA): {
                "sha256": sha256(stage_root / V2_COUNTRY_YEAR_QA),
                "bytes": (
                    stage_root / V2_COUNTRY_YEAR_QA
                ).stat().st_size,
                "rows": int(len(country_year_qa)),
            },
            str(V2_MACHINE_QA): {
                "sha256": sha256(stage_root / V2_MACHINE_QA),
                "bytes": (stage_root / V2_MACHINE_QA).stat().st_size,
            },
            str(V2_CONFIG): {
                "sha256": sha256(stage_root / V2_CONFIG),
                "bytes": (stage_root / V2_CONFIG).stat().st_size,
            },
            str(V2_MEMO): {
                "sha256": sha256(stage_root / V2_MEMO),
                "bytes": (stage_root / V2_MEMO).stat().st_size,
            },
        },
        "builder": {
            "path": str(
                Path(__file__).resolve().relative_to(root)
            ),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "atomic_release_protocol": {
            "exclusive_build_lock": str(V2_BUILD_LOCK),
            "in_progress_status": "invalidated_in_progress",
            "staged_before_commit": True,
            "individual_file_replacement": "atomic os.replace",
            "release_gate_written_last": str(V2_RELEASE_STATE),
        },
        "frozen_inputs_unchanged_after_build": True,
    }
    atomic_write_text(
        stage_root / V2_MANIFEST,
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    )

    output_hashes = {
        "builder": sha256(Path(__file__).resolve()),
        "hourly_table": sha256(stage_root / V2_STAGE0),
        "hourly_table_decompressed_csv": gzip_payload_sha256(
            stage_root / V2_STAGE0
        ),
        "country_year_qa": sha256(stage_root / V2_COUNTRY_YEAR_QA),
        "machine_qa": sha256(stage_root / V2_MACHINE_QA),
        "config": sha256(stage_root / V2_CONFIG),
        "manifest": sha256(stage_root / V2_MANIFEST),
        "memo": sha256(stage_root / V2_MEMO),
    }
    atomic_write_text(
        stage_root / V2_RELEASE_AUDIT,
        build_release_audit(qa, output_hashes),
    )
    output_hashes["release_audit"] = sha256(
        stage_root / V2_RELEASE_AUDIT
    )

    return {
        "qa": qa,
        "manifest": manifest,
        "output_hashes": output_hashes,
        "rows": int(len(serialized)),
        "filled_hours": int(filled_rows.sum()),
        "zero_fill_country_years": zero_fill_country_years,
        "maximum_absolute_derived_numeric_difference": comparison[
            "maximum_absolute_derived_numeric_difference"
        ],
    }


def verify_committed_release(
    root: Path,
    staged_result: dict[str, Any],
) -> dict[str, Any]:
    qa = json.loads(
        (root / V2_MACHINE_QA).read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (root / V2_MANIFEST).read_text(encoding="utf-8")
    )
    actual_hashes = {
        "builder": sha256(Path(__file__).resolve()),
        "hourly_table": sha256(root / V2_STAGE0),
        "hourly_table_decompressed_csv": gzip_payload_sha256(
            root / V2_STAGE0
        ),
        "country_year_qa": sha256(root / V2_COUNTRY_YEAR_QA),
        "machine_qa": sha256(root / V2_MACHINE_QA),
        "config": sha256(root / V2_CONFIG),
        "manifest": sha256(root / V2_MANIFEST),
        "memo": sha256(root / V2_MEMO),
        "release_audit": sha256(root / V2_RELEASE_AUDIT),
    }
    staged_hashes = staged_result["output_hashes"]
    suite_hashes_match_staging = all(
        actual_hashes[key] == value
        for key, value in staged_hashes.items()
    )

    manifest_reference_checks: dict[str, bool] = {}
    manifest_key_to_actual_key = {
        str(V2_STAGE0): "hourly_table",
        str(V2_COUNTRY_YEAR_QA): "country_year_qa",
        str(V2_MACHINE_QA): "machine_qa",
        str(V2_CONFIG): "config",
        str(V2_MEMO): "memo",
    }
    for relative, actual_key in manifest_key_to_actual_key.items():
        manifest_reference_checks[
            f"{relative}:sha256"
        ] = (
            manifest["outputs"][relative]["sha256"]
            == actual_hashes[actual_key]
        )
    manifest_reference_checks[
        f"{V2_STAGE0}:decompressed_csv_sha256"
    ] = (
        manifest["outputs"][str(V2_STAGE0)][
            "decompressed_csv_sha256"
        ]
        == actual_hashes["hourly_table_decompressed_csv"]
    )
    manifest_reference_checks["builder:sha256"] = (
        manifest["builder"]["sha256"] == actual_hashes["builder"]
    )

    qa_reference_checks = {
        "qa_hourly_output_sha256": (
            qa["hashes"]["hourly_output_sha256"]
            == actual_hashes["hourly_table"]
        ),
        "qa_hourly_output_decompressed_csv_sha256": (
            qa["hashes"][
                "hourly_output_decompressed_csv_sha256"
            ]
            == actual_hashes["hourly_table_decompressed_csv"]
        ),
        "qa_country_year_sha256": (
            qa["hashes"]["country_year_qa_sha256"]
            == actual_hashes["country_year_qa"]
        ),
        "qa_config_sha256": (
            qa["hashes"]["definition_config_sha256"]
            == actual_hashes["config"]
        ),
        "qa_builder_sha256": (
            qa["hashes"]["derivation_script_sha256"]
            == actual_hashes["builder"]
        ),
    }
    checks = {
        "suite_hashes_match_staging": suite_hashes_match_staging,
        "all_manifest_references_match_committed_files": all(
            manifest_reference_checks.values()
        ),
        "all_qa_references_match_committed_files": all(
            qa_reference_checks.values()
        ),
        "qa_status_pass": qa["status"] == "pass",
        "manifest_numerical_gate_pass": bool(
            manifest["numerical_comparison"][
                "all_derived_numeric_columns_exact"
            ]
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "Committed Stage-0 v0.2 suite is internally inconsistent:\n"
            + json.dumps(
                {
                    "checks": checks,
                    "manifest_reference_checks": (
                        manifest_reference_checks
                    ),
                    "qa_reference_checks": qa_reference_checks,
                },
                indent=2,
            )
        )
    return {
        "checks": checks,
        "manifest_reference_checks": manifest_reference_checks,
        "qa_reference_checks": qa_reference_checks,
        "hashes": actual_hashes,
    }


def read_previous_release_state(root: Path) -> dict[str, Any] | None:
    path = root / V2_RELEASE_STATE
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    lock_path = root / V2_BUILD_LOCK
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("a+b") as lock_handle:
        try:
            fcntl.flock(
                lock_handle.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as error:
            raise RuntimeError(
                "Another Stage-0 v0.2 build holds the exclusive lock"
            ) from error

        previous_state = read_previous_release_state(root)
        in_progress = {
            "task_contract": TASK_CONTRACT,
            "canonical_task_source_field": CANONICAL_TASK_FIELD,
            "reason": (
                "exclusive staged build in progress; downstream use "
                "prohibited until status returns to pass"
            ),
            "builder_sha256": sha256(Path(__file__).resolve()),
        }
        write_release_state(
            root,
            "invalidated_in_progress",
            in_progress,
        )

        try:
            frozen_before = require_frozen_inputs(root)
            staging_parent = root / "metadata"
            with tempfile.TemporaryDirectory(
                prefix=".stage0_v0_2_staging_",
                dir=staging_parent,
            ) as temporary:
                stage_root = Path(temporary)
                staged_result = build_staged_release(
                    root,
                    stage_root,
                    frozen_before,
                )
                frozen_after_staging = require_frozen_inputs(root)
                if frozen_before != frozen_after_staging:
                    raise RuntimeError(
                        "A frozen input changed while staging the release"
                    )
                commit_staged_suite(stage_root, root)

            frozen_after_commit = require_frozen_inputs(root)
            if frozen_before != frozen_after_commit:
                raise RuntimeError(
                    "A frozen input changed while committing the release"
                )
            committed = verify_committed_release(root, staged_result)
            current_hashes = committed["hashes"]
            previous_hashes = (
                previous_state.get("suite_hashes")
                if previous_state
                and previous_state.get("status") == "pass"
                else None
            )
            consecutive_identical = (
                isinstance(previous_hashes, dict)
                and previous_hashes == current_hashes
            )
            final_state = {
                "task_contract": TASK_CONTRACT,
                "canonical_task_source_field": CANONICAL_TASK_FIELD,
                "legacy_alias": {
                    "field": LEGACY_TASK_ALIAS,
                    "status": LEGACY_ALIAS_STATUS,
                },
                "release_gate": (
                    "written last after committed-file, QA and manifest "
                    "cross-checks"
                ),
                "consecutive_complete_build_hashes_identical": (
                    consecutive_identical
                ),
                "suite_hashes": current_hashes,
                "committed_consistency_checks": committed["checks"],
                "manifest_reference_checks": committed[
                    "manifest_reference_checks"
                ],
                "qa_reference_checks": committed[
                    "qa_reference_checks"
                ],
                "frozen_inputs_unchanged": True,
            }
            write_release_state(root, "pass", final_state)
        except Exception as error:
            write_release_state(
                root,
                "invalidated_in_progress",
                {
                    **in_progress,
                    "failure_type": type(error).__name__,
                    "failure_message": str(error),
                },
            )
            raise
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    print(
        json.dumps(
            {
                "status": "pass",
                "hourly_output": str(root / V2_STAGE0),
                "rows": staged_result["rows"],
                "filled_hours": staged_result["filled_hours"],
                "zero_fill_country_years": staged_result[
                    "zero_fill_country_years"
                ],
                "maximum_absolute_derived_numeric_difference": (
                    staged_result[
                        "maximum_absolute_derived_numeric_difference"
                    ]
                ),
                "hourly_output_sha256": committed["hashes"][
                    "hourly_table"
                ],
                "hourly_output_decompressed_csv_sha256": committed[
                    "hashes"
                ]["hourly_table_decompressed_csv"],
                "consecutive_complete_build_hashes_identical": (
                    consecutive_identical
                ),
                "release_state": str(root / V2_RELEASE_STATE),
                "release_audit": str(root / V2_RELEASE_AUDIT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
