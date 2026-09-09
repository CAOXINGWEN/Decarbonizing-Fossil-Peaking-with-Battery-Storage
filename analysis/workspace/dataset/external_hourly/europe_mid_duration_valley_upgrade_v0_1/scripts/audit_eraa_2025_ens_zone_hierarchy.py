#!/usr/bin/env python3
"""Audit aggregate and atomic bidding-zone rows in ERAA 2025 hourly ENS.

The published hourly ENS tables are sparse: absent zone-hour rows imply zero
ENS.  Several national/market aggregates are reported alongside their
component bidding zones.  This script compares each reported aggregate with
the sum of its components on the full union of positive timestamps so that a
pan-European total can be formed without double counting.

This script is deliberately separate from ``audit_eraa_2025_ens_events.py``.
"""

from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_RAW = ROOT / "raw" / "eraa_2025" / "extracted"
DEFAULT_ARTIFACTS = ROOT / "artifacts" / "research"
YEARS = (2028, 2030, 2033, 2035)
KEYS = ["WS", "FOS", "Date"]

# Candidate duplicate/aggregate definitions implied by ERAA output
# nomenclature.  SE02 has no positive ENS row in the released files and is
# therefore restored as zero when the component sum is formed.  LU00 is an
# exact alias of LUG1 in the hourly ENS release; LUV1 is a separate atomic
# model zone and must not be included in the LU00 reconciliation.
HIERARCHIES: dict[str, tuple[str, ...]] = {
    "DK00": ("DKE1", "DKW1"),
    "IT00": ("ITCN", "ITCS", "ITN1", "ITS1", "ITSA", "ITSI"),
    "NO00": ("NOM1", "NON1", "NOS1", "NOS2", "NOS3"),
    "SE00": ("SE01", "SE02", "SE03", "SE04"),
    "ISEM": ("IE00", "UKNI"),
    "LU00": ("LUG1",),
}

# Plausible-looking codes that must not be collapsed without evidence.
NEGATIVE_CONTROL_HIERARCHIES: dict[str, tuple[str, ...]] = {
    "GR00_as_parent_of_GR03": ("GR03",),
    "UK00_as_parent_of_UKNI": ("UKNI",),
}

ABS_TOL_MWH = 1e-3
REL_TOL = 1e-8


def sequence_fingerprint(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Hash each sparse positive ENS vector and retain canonical records."""

    positive = frame.loc[
        frame["ENS (MWh)"].gt(0),
        [*KEYS, "Bidding Zone", "ENS (MWh)"],
    ].copy()
    positive["Date"] = pd.to_datetime(
        positive["Date"], dayfirst=True, errors="raise"
    )
    records: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []
    for zone, group in positive.groupby("Bidding Zone", observed=True):
        canonical = (
            group[[*KEYS, "ENS (MWh)"]]
            .sort_values(KEYS)
            .reset_index(drop=True)
        )
        key_array = np.column_stack(
            [
                canonical["WS"].to_numpy(dtype="<i8"),
                canonical["FOS"].to_numpy(dtype="<i8"),
                canonical["Date"].astype("int64").to_numpy(dtype="<i8"),
            ]
        )
        value_array = canonical["ENS (MWh)"].to_numpy(dtype="<f8")
        support_hash = hashlib.sha256(key_array.tobytes()).hexdigest()
        exact_hash = hashlib.sha256(
            key_array.tobytes() + value_array.tobytes()
        ).hexdigest()
        records[str(zone)] = canonical
        rows.append(
            {
                "bidding_zone": str(zone),
                "positive_key_count": int(len(canonical)),
                "total_ens_mwh": float(value_array.sum()),
                "support_sha256": support_hash,
                "exact_sequence_sha256": exact_hash,
            }
        )
    return pd.DataFrame(rows), records


def classify_sequence_pair(
    zone_a: str,
    zone_b: str,
    exact_values: bool,
    released_model_nodes: set[str],
) -> tuple[str, str]:
    """Return semantic interpretation and aggregation action for a pair."""

    pair = frozenset((zone_a, zone_b))
    if pair == frozenset(("LU00", "LUG1")):
        return (
            "exact_output_alias_confirmed_by_model_node_ontology",
            "exclude_LU00_keep_LUG1",
        )
    for parent, components in HIERARCHIES.items():
        for component in components:
            if pair == frozenset((parent, component)) and exact_values:
                return (
                    "year_specific_parent_component_identity",
                    f"exclude_{parent}_by_hierarchy",
                )
    if pair in (
        frozenset(("DE00", "LUG1")),
        frozenset(("DE00", "LU00")),
    ):
        return (
            "same_support_and_year_specific_scale_but_distinct_DE_LU_ENS_volumes",
            "retain_DE00_and_LUG1_after_excluding_LU00_alias",
        )
    if zone_a in released_model_nodes and zone_b in released_model_nodes:
        if exact_values:
            return (
                "identical_output_but_separate_released_study_nodes",
                "retain_both_unless_external_hierarchy_proves_alias",
            )
        return (
            "synchronised_scarcity_with_distinct_ENS_volumes",
            "retain_both",
        )
    if exact_values:
        return (
            "identical_output_requires_hierarchy_interpretation",
            "no_automatic_exclusion",
        )
    return ("shared_positive_support_only", "retain_both")


def scan_duplicate_sequences(
    frame: pd.DataFrame,
    target_year: int,
    demand_nodes: set[str],
    capacity_nodes: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find exact vectors and equal-support vectors across every zone pair."""

    fingerprints, records = sequence_fingerprint(frame)
    released_model_nodes = demand_nodes | capacity_nodes
    fingerprints.insert(0, "target_year", target_year)
    fingerprints["in_demand_nodes"] = fingerprints["bidding_zone"].isin(
        demand_nodes
    )
    fingerprints["in_capacity_nodes"] = fingerprints["bidding_zone"].isin(
        capacity_nodes
    )
    fingerprints["excluded_by_confirmed_hierarchy"] = fingerprints[
        "bidding_zone"
    ].isin(HIERARCHIES)

    pair_rows: list[dict[str, object]] = []
    fingerprint_lookup = fingerprints.set_index("bidding_zone")
    for support_hash, group in fingerprints.groupby(
        "support_sha256", observed=True
    ):
        zones = sorted(group["bidding_zone"].tolist())
        if len(zones) < 2:
            continue
        for zone_a, zone_b in combinations(zones, 2):
            a = records[zone_a]
            b = records[zone_b]
            # Equal support hashes are verified by an explicit key comparison
            # before the values are compared.
            keys_equal = a[KEYS].equals(b[KEYS])
            if not keys_equal:
                raise AssertionError(
                    f"SHA-256 support collision for {target_year} "
                    f"{zone_a} and {zone_b}"
                )
            values_a = a["ENS (MWh)"].to_numpy(dtype=float)
            values_b = b["ENS (MWh)"].to_numpy(dtype=float)
            exact_values = np.array_equal(values_a, values_b)
            within_tolerance = bool(
                np.allclose(
                    values_a,
                    values_b,
                    rtol=REL_TOL,
                    atol=ABS_TOL_MWH,
                )
            )
            difference = values_a - values_b
            correlation = (
                float(np.corrcoef(values_a, values_b)[0, 1])
                if len(values_a) >= 2
                and np.std(values_a) > 0
                and np.std(values_b) > 0
                else np.nan
            )
            ratios = values_a / values_b
            median_ratio = float(np.median(ratios))
            constant_scale = bool(
                np.allclose(
                    values_a,
                    median_ratio * values_b,
                    rtol=REL_TOL,
                    atol=ABS_TOL_MWH,
                )
            )
            interpretation, action = classify_sequence_pair(
                zone_a,
                zone_b,
                exact_values,
                released_model_nodes,
            )
            pair_rows.append(
                {
                    "target_year": target_year,
                    "zone_a": zone_a,
                    "zone_b": zone_b,
                    "positive_key_count": int(len(a)),
                    "support_sha256": support_hash,
                    "exact_sequence_sha256_a": fingerprint_lookup.loc[
                        zone_a, "exact_sequence_sha256"
                    ],
                    "exact_sequence_sha256_b": fingerprint_lookup.loc[
                        zone_b, "exact_sequence_sha256"
                    ],
                    "same_sparse_support": True,
                    "exact_value_sequence": bool(exact_values),
                    "within_numeric_tolerance": within_tolerance,
                    "total_ens_a_mwh": float(values_a.sum()),
                    "total_ens_b_mwh": float(values_b.sum()),
                    "total_a_to_b_ratio": float(
                        values_a.sum() / values_b.sum()
                    ),
                    "maximum_absolute_difference_mwh": float(
                        np.max(np.abs(difference))
                    ),
                    "value_correlation": correlation,
                    "constant_scalar_multiple": constant_scale,
                    "both_released_model_nodes": bool(
                        zone_a in released_model_nodes
                        and zone_b in released_model_nodes
                    ),
                    "interpretation": interpretation,
                    "europe_sum_action": action,
                }
            )
    return fingerprints, pd.DataFrame(pair_rows)


def read_year(path: Path, year: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {*KEYS, "Bidding Zone", "ENS (MWh)"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns {sorted(missing)}")
    frame = frame[[*KEYS, "Bidding Zone", "ENS (MWh)"]].copy()
    frame["target_year"] = year
    frame["ENS (MWh)"] = pd.to_numeric(frame["ENS (MWh)"], errors="raise")
    if frame["ENS (MWh)"].lt(0).any():
        raise ValueError(f"{path} contains negative ENS")
    duplicate = frame.duplicated([*KEYS, "Bidding Zone"])
    if duplicate.any():
        raise ValueError(
            f"{path} contains {int(duplicate.sum())} duplicate zone-hour keys"
        )
    return frame


def compare_hierarchy(
    frame: pd.DataFrame,
    target_year: int,
    hierarchy_name: str,
    parent: str,
    children: tuple[str, ...],
    candidate_type: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    zones_present = set(frame["Bidding Zone"].unique())
    parent_present = parent in zones_present
    present_children = tuple(child for child in children if child in zones_present)
    missing_children = tuple(child for child in children if child not in zones_present)

    selected = frame.loc[
        frame["Bidding Zone"].isin((parent, *children)),
        [*KEYS, "Bidding Zone", "ENS (MWh)"],
    ]
    wide = selected.pivot(index=KEYS, columns="Bidding Zone", values="ENS (MWh)")
    wide = wide.reindex(columns=[parent, *children], fill_value=0.0).fillna(0.0)
    parent_series = wide[parent]
    child_sum = wide[list(children)].sum(axis=1)
    difference = parent_series - child_sum
    scale = np.maximum(np.maximum(parent_series.abs(), child_sum.abs()), 1.0)
    equal = difference.abs().le(ABS_TOL_MWH + REL_TOL * scale)

    detail = wide.reset_index()
    detail.insert(0, "target_year", target_year)
    detail.insert(1, "hierarchy_name", hierarchy_name)
    detail["parent_ens_mwh"] = parent_series.to_numpy()
    detail["component_sum_ens_mwh"] = child_sum.to_numpy()
    detail["difference_mwh"] = difference.to_numpy()
    detail["within_tolerance"] = equal.to_numpy()
    detail = detail.loc[
        detail["parent_ens_mwh"].gt(0)
        | detail["component_sum_ens_mwh"].gt(0)
    ].copy()

    parent_positive = parent_series.gt(0)
    child_positive = child_sum.gt(0)
    parent_total = float(parent_series.sum())
    child_total = float(child_sum.sum())
    max_abs_difference = float(difference.abs().max()) if len(difference) else 0.0
    mismatch_count = int((~equal).sum())
    union_positive_count = int((parent_positive | child_positive).sum())
    equality_share = (
        float(equal.loc[parent_positive | child_positive].mean())
        if union_positive_count
        else np.nan
    )
    total_ratio = (
        parent_total / child_total if not np.isclose(child_total, 0.0) else np.nan
    )

    row: dict[str, object] = {
        "target_year": target_year,
        "candidate_type": candidate_type,
        "hierarchy_name": hierarchy_name,
        "parent_zone": parent,
        "component_zones": "|".join(children),
        "parent_present": parent_present,
        "present_components": "|".join(present_children),
        "components_without_positive_rows": "|".join(missing_children),
        "union_positive_zone_hours": union_positive_count,
        "parent_positive_zone_hours": int(parent_positive.sum()),
        "component_positive_zone_hours": int(child_positive.sum()),
        "parent_only_positive_zone_hours": int((parent_positive & ~child_positive).sum()),
        "component_only_positive_zone_hours": int((~parent_positive & child_positive).sum()),
        "parent_total_ens_mwh": parent_total,
        "component_total_ens_mwh": child_total,
        "parent_to_component_total_ratio": total_ratio,
        "signed_difference_mwh": float(difference.sum()),
        "absolute_difference_mwh": float(difference.abs().sum()),
        "maximum_absolute_hourly_difference_mwh": max_abs_difference,
        "mismatched_union_zone_hours": mismatch_count,
        "within_tolerance_share": equality_share,
        "exact_reconciliation": bool(
            parent_present
            and len(present_children) >= 1
            and union_positive_count >= 1
            and mismatch_count == 0
        ),
    }
    return row, detail


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include="float").columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.6g}"
        )
    headers = [str(column) for column in display.columns]
    rows = [[str(value) for value in row] for row in display.to_numpy()]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    header = "| " + " | ".join(
        value.ljust(widths[index]) for index, value in enumerate(headers)
    ) + " |"
    rule = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [
        "| "
        + " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(row)
        )
        + " |"
        for row in rows
    ]
    return "\n".join([header, rule, *body])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)

    frames: dict[int, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []
    mismatches: list[pd.DataFrame] = []

    for year in YEARS:
        path = args.raw / f"ENS_hourly_data_{year}.csv"
        frames[year] = read_year(path, year)
        for parent, children in HIERARCHIES.items():
            row, detail = compare_hierarchy(
                frames[year],
                year,
                parent,
                parent,
                children,
                "prespecified_aggregate",
            )
            rows.append(row)
            if not row["exact_reconciliation"]:
                mismatches.append(detail.loc[~detail["within_tolerance"]])
        for name, children in NEGATIVE_CONTROL_HIERARCHIES.items():
            parent = name.split("_as_parent_of_")[0]
            row, detail = compare_hierarchy(
                frames[year],
                year,
                name,
                parent,
                children,
                "negative_control",
            )
            rows.append(row)
            mismatches.append(detail.loc[~detail["within_tolerance"]])

    audit = pd.DataFrame(rows)
    output_csv = args.artifacts / "ERAA_2025_ENS_ZONE_HIERARCHY_AUDIT.csv"
    audit.to_csv(output_csv, index=False)

    mismatch_frame = (
        pd.concat(mismatches, ignore_index=True)
        if mismatches
        else pd.DataFrame()
    )
    mismatch_frame.to_csv(
        args.artifacts / "ERAA_2025_ENS_ZONE_HIERARCHY_MISMATCHES.csv",
        index=False,
    )

    confirmed = audit.loc[
        audit["candidate_type"].eq("prespecified_aggregate")
        & audit["exact_reconciliation"]
    ]
    confirmed_all_years = (
        confirmed.groupby("parent_zone", observed=True)["target_year"].nunique()
        .eq(len(YEARS))
    )
    aggregate_zones = sorted(
        confirmed_all_years.loc[confirmed_all_years].index.tolist()
    )
    all_reported_zones = sorted(
        set().union(
            *[
                set(frame["Bidding Zone"].dropna().unique())
                for frame in frames.values()
            ]
        )
    )
    declared_component_zones = set().union(*map(set, HIERARCHIES.values()))
    atomic_zones = sorted(
        (set(all_reported_zones) | declared_component_zones).difference(
            aggregate_zones
        )
    )
    aggregation_rows: list[dict[str, object]] = []
    for year, frame in frames.items():
        naive_total = float(frame["ENS (MWh)"].sum())
        atomic_total = float(
            frame.loc[
                ~frame["Bidding Zone"].isin(aggregate_zones), "ENS (MWh)"
            ].sum()
        )
        excluded_total = float(
            frame.loc[
                frame["Bidding Zone"].isin(aggregate_zones), "ENS (MWh)"
            ].sum()
        )
        aggregation_rows.append(
            {
                "target_year": year,
                "naive_all_reported_codes_ens_mwh": naive_total,
                "atomic_zone_ens_mwh": atomic_total,
                "excluded_duplicate_ens_mwh": excluded_total,
                "naive_minus_atomic_ens_mwh": naive_total - atomic_total,
                "naive_to_atomic_ratio": naive_total / atomic_total,
                "overcount_share_of_atomic": (
                    (naive_total - atomic_total) / atomic_total
                ),
            }
        )
    aggregation_summary = pd.DataFrame(aggregation_rows)
    aggregation_summary.to_csv(
        args.artifacts / "ERAA_2025_ENS_ATOMIC_AGGREGATION_SUMMARY.csv",
        index=False,
    )
    dashboard_root = (
        args.raw
        / "ERAA_2025_DashboardRawData1"
        / "Dashboard Raw Data"
    )
    demand_nodes = set(
        pd.read_csv(
            dashboard_root / "Aggregated_Demand.csv",
            usecols=["Market_Node"],
        )["Market_Node"].dropna()
    )
    capacity_nodes = set(
        pd.read_csv(
            dashboard_root / "GenerationCapacities.csv",
            usecols=["Market_Node"],
        )["Market_Node"].dropna()
    )
    released_model_nodes = demand_nodes | capacity_nodes
    aggregate_codes_absent_from_model_nodes = all(
        zone not in released_model_nodes for zone in aggregate_zones
    )
    separate_control_nodes_present = all(
        zone in released_model_nodes
        for zone in ("GR00", "GR03", "UK00", "UKNI")
    )
    audit["semantic_role"] = np.where(
        audit["candidate_type"].eq("prespecified_aggregate"),
        np.where(
            audit["parent_zone"].eq("LU00"),
            "exact_published_alias",
            "published_aggregate",
        ),
        "negative_control_separate_study_zones",
    )
    audit["europe_sum_action"] = np.where(
        audit["candidate_type"].eq("prespecified_aggregate")
        & audit["exact_reconciliation"],
        "exclude_parent",
        "retain_separate_zones",
    )
    # Rewrite the hierarchy CSV with the semantic decision columns added.
    audit.to_csv(output_csv, index=False)

    fingerprint_parts: list[pd.DataFrame] = []
    sequence_pair_parts: list[pd.DataFrame] = []
    for year, frame in frames.items():
        fingerprints, sequence_pairs = scan_duplicate_sequences(
            frame,
            year,
            demand_nodes,
            capacity_nodes,
        )
        fingerprint_parts.append(fingerprints)
        if not sequence_pairs.empty:
            sequence_pair_parts.append(sequence_pairs)
    zone_fingerprints = pd.concat(fingerprint_parts, ignore_index=True)
    sequence_pairs = (
        pd.concat(sequence_pair_parts, ignore_index=True)
        if sequence_pair_parts
        else pd.DataFrame()
    )
    zone_fingerprints.to_csv(
        args.artifacts / "ERAA_2025_ENS_ZONE_SEQUENCE_FINGERPRINTS.csv",
        index=False,
    )
    sequence_pairs.to_csv(
        args.artifacts / "ERAA_2025_ENS_SEQUENCE_DUPLICATE_AUDIT.csv",
        index=False,
    )

    exact_sequence_pairs = sequence_pairs.loc[
        sequence_pairs["exact_value_sequence"]
    ].copy()
    stable_exact_pairs = (
        exact_sequence_pairs.groupby(
            ["zone_a", "zone_b"], observed=True
        )["target_year"]
        .nunique()
        .reset_index(name="years_exact")
    )
    stable_exact_pairs = stable_exact_pairs.loc[
        stable_exact_pairs["years_exact"].eq(len(YEARS))
    ]
    de_lu_pairs = sequence_pairs.loc[
        sequence_pairs.apply(
            lambda row: frozenset((row["zone_a"], row["zone_b"]))
            in (
                frozenset(("DE00", "LUG1")),
                frozenset(("DE00", "LU00")),
            ),
            axis=1,
        )
    ].copy()
    automatic_exclusion_actions = set(
        sequence_pairs.loc[
            sequence_pairs["europe_sum_action"].str.startswith(
                "exclude_", na=False
            ),
            "europe_sum_action",
        ]
    )
    new_sequence_exclusion_required = any(
        not any(zone in action for zone in aggregate_zones)
        for action in automatic_exclusion_actions
    )

    compact = audit[
        [
            "target_year",
            "hierarchy_name",
            "candidate_type",
            "union_positive_zone_hours",
            "parent_total_ens_mwh",
            "component_total_ens_mwh",
            "maximum_absolute_hourly_difference_mwh",
            "mismatched_union_zone_hours",
            "exact_reconciliation",
        ]
    ]
    sequence_compact = sequence_pairs[
        [
            "target_year",
            "zone_a",
            "zone_b",
            "positive_key_count",
            "exact_value_sequence",
            "total_a_to_b_ratio",
            "maximum_absolute_difference_mwh",
            "value_correlation",
            "interpretation",
            "europe_sum_action",
        ]
    ]
    de_lug1_pairs = de_lu_pairs.loc[
        de_lu_pairs.apply(
            lambda row: frozenset((row["zone_a"], row["zone_b"]))
            == frozenset(("DE00", "LUG1")),
            axis=1,
        )
    ]
    de_lu_compact = de_lug1_pairs[
        [
            "target_year",
            "zone_a",
            "zone_b",
            "positive_key_count",
            "exact_value_sequence",
            "total_ens_a_mwh",
            "total_ens_b_mwh",
            "total_a_to_b_ratio",
            "maximum_absolute_difference_mwh",
            "value_correlation",
            "constant_scalar_multiple",
        ]
    ]
    exact_sequence_compact = exact_sequence_pairs[
        [
            "target_year",
            "zone_a",
            "zone_b",
            "positive_key_count",
            "total_ens_a_mwh",
            "total_ens_b_mwh",
            "maximum_absolute_difference_mwh",
            "europe_sum_action",
        ]
    ]
    report = f"""# ERAA 2025 hourly ENS study-zone/output hierarchy audit

## Decision

Five reported aggregates (`DK00`, `IT00`, `NO00`, `SE00`, and `ISEM`) and one
exact alias (`LU00`) reconcile to their component bidding zones at every
positive union timestamp in all four target years, within an absolute
tolerance of {ABS_TOL_MWH:g} MWh plus a relative tolerance of {REL_TOL:g}.
They must therefore be excluded whenever ENS is summed across Europe.

The exact atomic-zone rule is:

1. start from all reported bidding-zone codes;
2. remove `DK00`, `IT00`, `NO00`, `SE00`, `ISEM`, and `LU00`;
3. retain `DKE1`, `DKW1`, `ITCN`, `ITCS`, `ITN1`, `ITS1`, `ITSA`, `ITSI`,
   `NOM1`, `NON1`, `NOS1`, `NOS2`, `NOS3`, `SE01`, `SE02`, `SE03`, `SE04`,
   `IE00`, `UKNI`, and `LUG1`, treating an absent sparse row as zero;
4. retain all other reported codes, including `GR00`, `GR03`, `UK00`, and
   `UKNI`. Retain `LUV1` independently from `LUG1`, because `LU00` is an exact
   alias of `LUG1`, not the sum of `LUG1` and `LUV1`. The tests do not support
   collapsing `GR03` into `GR00` or `UKNI` into `UK00`.

`SE02` has no positive ENS row in these releases but remains an atomic model
zone with zero observed ENS in the sparse files. Its absence is not a reason
to substitute `SE00`.

## Automated sparse-sequence duplicate scan

Every positive zone × (`WS`, `FOS`, timestamp) vector was canonically sorted
and assigned both a support SHA-256 and a value-inclusive SHA-256. Candidate
hash matches were then compared key by key and value by value, so the decision
does not rely on a hash match alone. The scan covers {len(zone_fingerprints)}
zone-year vectors. It finds {len(sequence_pairs)} same-support year-pairs and
{len(exact_sequence_pairs)} exact-value year-pairs. The only pair that remains
exact in all four target years is `LU00`–`LUG1`
({len(stable_exact_pairs)} stable exact pair).

{markdown_table(exact_sequence_compact)}

No unanticipated exact sequence duplicate is found. The full set of
same-support candidates consists only of `DE00`, `LU00`, and `LUG1`.
After removing the `LU00` alias, `DE00` and `LUG1` still have exactly the same
positive ENS timestamps in every year, but they do not contain the same MWh
observations:

{markdown_table(de_lu_compact)}

The DE00/LUG1 relationship is a synchronized and year-specifically scaled
scarcity output, not duplicate energy. ENTSO-E's ERAA 2025 explanatory note
defines `DE00` and `LUG1` as separate study zones in Table 1, and both occur
as separate nodes in the released demand and capacity inputs. Germany and
Luxembourg nevertheless form one market bidding zone with a common market
price, which is consistent with shared shortage timestamps. The common market
area does not make their zonal ENS volumes interchangeable or redundant.
Accordingly, both `DE00` and `LUG1` must be retained and summed.

The automated scan requires no additional exclusion beyond the six codes
already listed (`new_sequence_exclusion_required =
{new_sequence_exclusion_required}`). Fingerprints and all candidate pairs are
reported in `ERAA_2025_ENS_ZONE_SEQUENCE_FINGERPRINTS.csv` and
`ERAA_2025_ENS_SEQUENCE_DUPLICATE_AUDIT.csv`.

## Quantitative reconciliation

{markdown_table(compact)}

## Consequence for European totals

{markdown_table(aggregation_summary)}

Summing every reported code would count the aggregate/alias rows a second
time. The `atomic_zone_ens_mwh` column is the appropriate arithmetic total.
The released demand and capacity inputs independently support the hierarchy:
all six excluded codes are absent from the input model-node set
(`{aggregate_codes_absent_from_model_nodes}`), whereas `GR00`, `GR03`,
`UK00`, and `UKNI` occur as separate input nodes
(`{separate_control_nodes_present}`). `SE02` occurs in both inputs despite
having no positive hourly ENS record.

## Machine-readable rule

- Aggregate codes to exclude: `{",".join(aggregate_zones)}`
- Atomic model codes after restoring declared zero-only components:
  `{",".join(atomic_zones)}`

## Unresolved cases and scope

- The files report only positive ENS observations. The audit therefore treats
  absent zone-hour rows as zero, consistent with the sparse-release format.
- This establishes arithmetic non-double-counting, not a permanent ENTSO-E
  geographic ontology. The atomic list should be regenerated if a later ERAA
  release changes its study-zone set.
- `GR00` and `GR03` are retained as separate model zones. Their series do not
  reconcile as parent and child.
- `UK00` is retained as Great Britain. `UKNI` is retained as Northern Ireland;
  only the all-island `ISEM` row is excluded.
- `LU00` exactly duplicates `LUG1` at every tested key. `LUV1` is separately
  retained; adding it to `LU00` would omit genuine `LUV1` ENS, whereas retaining
  all three codes would double count `LUG1`.
- Identical all-zero vectors are not used to merge study zones. Model nodes
  with no positive ENS remain independent spatial units.

## Sources for the semantic decision

- ENTSO-E, *ERAA 2025 Call-for-Evidence dataset explanatory note*, pp. 6–8
  and Table 1, local source
  `raw/eraa_2025/ERAA_2025_ExplanatoryNoteOnDataRelease.pdf`.
- Bundesnetzagentur, *Monitoring Report 2023*, which states that Germany and
  Luxembourg constitute a common bidding zone with uniform prices:
  https://data.bundesnetzagentur.de/Bundesnetzagentur/SharedDocs/Downloads/EN/Areas/ElectricityGas/CollectionCompanySpecificData/Monitoring/MonitoringReport2023.pdf
"""
    (args.artifacts / "ERAA_2025_ENS_ZONE_HIERARCHY_AUDIT.md").write_text(
        report,
        encoding="utf-8",
    )

    print(output_csv)
    print(args.artifacts / "ERAA_2025_ENS_ZONE_HIERARCHY_AUDIT.md")


if __name__ == "__main__":
    main()
