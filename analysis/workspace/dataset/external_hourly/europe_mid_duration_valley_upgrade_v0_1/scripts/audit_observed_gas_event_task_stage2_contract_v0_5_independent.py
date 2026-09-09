#!/usr/bin/env python3
"""Independent, non-executing audit of the Stage 2 v0.5 five-file contract.

This program reads the frozen contract artifacts and their declared provenance.
It does not generate a chronology inventory, instantiate an optimization model,
or modify any v0.5 contract file.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any


EXPECTED_CHRONOLOGY_SHA = (
    "5c07339211a4a49f54747f8dff090d54147947465a27743f98ef237351437016"
)
EXPECTED_INFERENCE_SHA = (
    "0cad5ff636b774354356f29ecc71ac834212b115ea95e8edb0cd6a651908ed8b"
)
STALE_INFERENCE_SHA = (
    "8edb8a683861e29c263a3af32ba44dd3b9392a6af202152d5f4c0a92c1a62862"
)
EXPECTED_V04_AUDIT_SHA = (
    "aa9c961b4595eec11c38bf37b4970bbe61936ea294bfc08d66850b7b732c16ef"
)
EXPECTED_LFP_IDS = {
    "clean_fom_frep_0_468106",
    "clean_fom_frep_0_565037",
    "bundled_fom_no_replacement",
    "zero_fom_frep_0_468106",
    "zero_fom_frep_0_565037",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_gzip_payload(path: Path) -> str:
    h = hashlib.sha256()
    with gzip.open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_subobject_sha(obj: Any) -> str:
    payload = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_csv_rectangular(path: Path) -> tuple[list[str], list[dict[str, str]], bool]:
    raw_rows: list[list[str]]
    with path.open("r", encoding="utf-8", newline="") as fh:
        raw_rows = list(csv.reader(fh))
    widths = {len(row) for row in raw_rows}
    rectangular = len(widths) == 1
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        header = list(reader.fieldnames or [])
        rows = list(reader)
    return header, rows, rectangular


def type7(values: list[Fraction], p: Fraction) -> Fraction:
    x = sorted(values)
    h = Fraction(len(x) - 1) * p
    lo = h.numerator // h.denominator
    hi = lo if h.denominator == 1 else lo + 1
    gamma = h - lo
    return (1 - gamma) * x[lo] + gamma * x[hi]


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def check(
        self,
        check_id: str,
        condition: bool,
        observed: Any,
        expected: Any,
        note: str = "",
    ) -> None:
        self.checks.append(
            {
                "check_id": check_id,
                "status": "PASS" if condition else "FAIL",
                "observed": observed,
                "expected": expected,
                "note": note,
            }
        )

    @property
    def failures(self) -> list[dict[str, Any]]:
        return [row for row in self.checks if row["status"] == "FAIL"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    base = root / (
        "dataset/external_hourly/europe_mid_duration_valley_upgrade_v0_1/"
        "artifacts/research"
    )

    paths = {
        "markdown": base / "OBSERVED_GAS_EVENT_TASK_STAGE2_FORMAL_CONTRACT_v0_5.md",
        "json": base / "OBSERVED_GAS_EVENT_TASK_STAGE2_CONTRACT_v0_5.json",
        "flat_csv": base / "observed_gas_event_task_stage2_contract_v0_5.csv",
        "audit_response_csv": base
        / "observed_gas_event_task_stage2_v0_5_audit_response.csv",
        "static_qa_json": base
        / "OBSERVED_GAS_EVENT_TASK_STAGE2_CONTRACT_v0_5_STATIC_QA.json",
    }
    v04_audit_path = (
        base / "OBSERVED_GAS_EVENT_TASK_STAGE2_CONTRACT_V0_4_FINAL_INDEPENDENT_AUDIT.md"
    )
    a = Audit()

    actual_package_hashes = {key: sha256_file(path) for key, path in paths.items()}
    contract = json.loads(paths["json"].read_text(encoding="utf-8"))
    static_qa = json.loads(paths["static_qa_json"].read_text(encoding="utf-8"))
    md_text = paths["markdown"].read_text(encoding="utf-8")
    all_five_text = "\n".join(
        p.read_text(encoding="utf-8", errors="strict") for p in paths.values()
    )
    flat_header, flat_rows, flat_rectangular = read_csv_rectangular(paths["flat_csv"])
    audit_header, audit_rows, audit_rectangular = read_csv_rectangular(
        paths["audit_response_csv"]
    )
    flat_index = {(r["section"], r["item_id"]): r for r in flat_rows}

    # File/package identity and syntax.
    a.check("P00_json_contract_parse", isinstance(contract, dict), True, True)
    a.check("P01_static_qa_parse", isinstance(static_qa, dict), True, True)
    a.check(
        "P02_flat_csv_rectangular",
        flat_rectangular and len(flat_header) == 11,
        {"rectangular": flat_rectangular, "columns": len(flat_header), "data_rows": len(flat_rows)},
        {"rectangular": True, "columns": 11, "data_rows": 124},
    )
    a.check(
        "P03_audit_csv_rectangular",
        audit_rectangular and len(audit_header) == 9,
        {
            "rectangular": audit_rectangular,
            "columns": len(audit_header),
            "data_rows": len(audit_rows),
        },
        {"rectangular": True, "columns": 9, "data_rows": 20},
    )
    duplicate_pairs = [
        pair
        for pair, count in Counter(
            (r["section"], r["item_id"]) for r in flat_rows
        ).items()
        if count != 1
    ]
    a.check("P04_flat_unique_section_item", not duplicate_pairs, duplicate_pairs, [])
    qa_package_hashes = {
        row["role"]: row["sha256"] for row in static_qa["package_files"][:4]
    }
    role_by_key = {
        "markdown": "human-readable contract",
        "json": "machine-readable JSON contract",
        "flat_csv": "machine-readable flat contract",
        "audit_response_csv": "audit response ledger",
    }
    for key, role in role_by_key.items():
        a.check(
            f"P05_package_hash_{key}",
            qa_package_hashes.get(role) == actual_package_hashes[key],
            actual_package_hashes[key],
            qa_package_hashes.get(role),
        )

    # A01: independently recompute both canonical subobject hashes.
    chronology_sha = canonical_subobject_sha(contract["chronology_randomization"])
    inference_sha = canonical_subobject_sha(contract["inference"])
    stored_hashes = contract["canonical_subobject_hashes"]
    flat_chronology_sha = flat_index[("randomization", "chronology_spec_hash")]["value"]
    flat_inference_sha = flat_index[("inference", "inference_spec_hash")]["value"]
    audit_a01 = next(r for r in audit_rows if r["item_id"] == "A01")
    a.check(
        "A01_chronology_independent_sha",
        chronology_sha
        == stored_hashes["chronology_randomization_sha256"]
        == flat_chronology_sha
        == EXPECTED_CHRONOLOGY_SHA,
        {
            "recomputed": chronology_sha,
            "json": stored_hashes["chronology_randomization_sha256"],
            "csv": flat_chronology_sha,
        },
        EXPECTED_CHRONOLOGY_SHA,
    )
    a.check(
        "A01_inference_independent_sha",
        inference_sha
        == stored_hashes["inference_sha256"]
        == flat_inference_sha
        == EXPECTED_INFERENCE_SHA,
        {
            "recomputed": inference_sha,
            "json": stored_hashes["inference_sha256"],
            "csv": flat_inference_sha,
        },
        EXPECTED_INFERENCE_SHA,
    )
    qa_sub = static_qa["canonical_subobject_automated_recomputation"]
    a.check(
        "A01_static_qa_literals",
        qa_sub["chronology_randomization"]["recomputed_sha256"] == chronology_sha
        and qa_sub["chronology_randomization"]["json_stored_sha256"] == chronology_sha
        and qa_sub["chronology_randomization"]["flat_csv_sha256"] == chronology_sha
        and qa_sub["inference"]["recomputed_sha256"] == inference_sha
        and qa_sub["inference"]["json_stored_sha256"] == inference_sha
        and qa_sub["inference"]["flat_csv_sha256"] == inference_sha
        and qa_sub["inference"]["audit_response_sha256_literal"] == inference_sha,
        qa_sub,
        "all stored chronology/inference values equal independent recomputation",
    )
    a.check(
        "A01_audit_ledger_literal",
        EXPECTED_INFERENCE_SHA in audit_a01["static_verification"]
        and audit_a01["status"] == "resolved_contract_level",
        {
            "verification": audit_a01["static_verification"],
            "status": audit_a01["status"],
        },
        EXPECTED_INFERENCE_SHA,
    )
    a.check(
        "A01_stale_hash_absent",
        STALE_INFERENCE_SHA not in all_five_text,
        STALE_INFERENCE_SHA if STALE_INFERENCE_SHA in all_five_text else "absent",
        "absent from all v0.5 files",
    )

    # A02: continuous weighted first-hit, 1e-4 hour solver interval,
    # continuous/sign-aware propagation, and a narrowly scoped integer gate.
    endpoints = contract["mechanism_endpoints"]
    cert = endpoints["endpoint_certification"]
    aggregation = endpoints["event_aggregation_contract"]
    layering = aggregation["first_hit_estimand_layering"]
    propagation = aggregation[
        "weighted_first_hit_solver_bound_and_bootstrap_propagation"
    ]
    a.check(
        "A02_continuous_estimand_json",
        "continuous" in layering["country_year_energy_weighted_tau_bar"]
        and "not an integer" in layering["country_year_energy_weighted_tau_bar"]
        and "continuous rational" in aggregation[
            "country_year_cell_endpoint_formulas"
        ]["weighted_first_hit_hours_by_band"],
        layering,
        "tau_bar=sum_j w_j*tau_j is continuous, not integer",
    )
    gate_values = {
        "endpoint_certification": cert[
            "weighted_first_hit_hours_absolute_bound_width_max_hours"
        ],
        "first_hit_layering": layering["continuous_gate_hours"],
        "solver_and_qa": contract["solver_and_qa"][
            "weighted_first_hit_hours_absolute_bound_width_max_hours"
        ],
        "static_assertion": contract["static_assertions"][
            "weighted_first_hit_hours_bound_width_max_must_equal"
        ],
        "flat_csv": Decimal(
            flat_index[("endpoint", "weighted_first_hit_certification")]["value"]
        ),
        "static_qa": Decimal(
            str(
                static_qa["v0_4_blocker_repairs"]["A02"][
                    "absolute_solver_bound_width_max_hours"
                ]
            )
        ),
    }
    a.check(
        "A02_bound_width_gate_all_representations",
        all(Decimal(str(v)) == Decimal("0.0001") for v in gate_values.values()),
        {k: str(v) for k, v in gate_values.items()},
        "0.0001 h",
    )
    a.check(
        "A02_global_bound_and_incumbent",
        cert["continuous_solver_bound_status_required"] is True
        and "global lower bound" in cert[
            "weighted_first_hit_continuous_endpoint_rule"
        ]
        and "feasible upper bound" in cert[
            "weighted_first_hit_continuous_endpoint_rule"
        ]
        and "0<=U-L<=0.0001 h" in cert[
            "weighted_first_hit_continuous_endpoint_rule"
        ]
        and "valid global lower"
        in flat_index[("endpoint", "weighted_first_hit_certification")][
            "formula_or_rule"
        ]
        and "feasible upper"
        in flat_index[("endpoint", "weighted_first_hit_certification")][
            "formula_or_rule"
        ],
        cert,
        "valid global solver bound and feasible incumbent; width <= 0.0001 h",
    )
    a.check(
        "A02_integer_scope_excludes_weighted_tau",
        "never the energy-weighted" in cert["integer_endpoint_scope"]
        and "raw censored event count" in cert["integer_endpoint_scope"]
        and "optional separately certified event-level tau_j" in layering[
            "integer_gate_applies_to"
        ]
        and "country-year energy-weighted tau_bar" in layering[
            "integer_gate_forbidden_for"
        ]
        and "Never applies to sum_j w_j*tau_j" in flat_index[
            ("endpoint", "integer_certification")
        ]["admission_rule"],
        {
            "json_scope": cert["integer_endpoint_scope"],
            "json_forbidden": layering["integer_gate_forbidden_for"],
            "csv_scope": flat_index[("endpoint", "integer_certification")][
                "admission_rule"
            ],
        },
        "integer gate only for raw censored count and optional event-level tau_j",
    )
    a.check(
        "A02_continuous_bootstrap_propagation",
        propagation["integer_rounding_at_any_aggregate_or_bootstrap_layer_forbidden"]
        is True
        and "aggregate lower and upper endpoints separately"
        in propagation["phase_seed_country_year"]
        and "coefficient-sign" in propagation["linear_contrast"]
        and "same nonnegative pigeonhole weights" in propagation["bootstrap_draw"]
        and "type7_q0.025" in propagation["formal_interval"]
        and "without rounding"
        in flat_index[
            ("endpoint_aggregation", "weighted_first_hit_bootstrap")
        ]["value"],
        propagation,
        "continuous bounds propagated separately through phase/seed, contrasts, draws and type-7 limits",
    )
    a.check(
        "A02_markdown_parity",
        all(
            literal in md_text
            for literal in (
                r"\bar\tau=\sum_jw_j\tau_j",
                r"10^{-4}\ {\rm h}",
                "任何层级均不得整数取整",
                "原始删失事件数",
            )
        ),
        "required literals present",
        "continuous tau_bar, 1e-4 h, no aggregate rounding, raw censored count",
    )
    audit_a02 = next(r for r in audit_rows if r["item_id"] == "A02")
    a.check(
        "A02_audit_ledger_and_static_qa",
        "0.0001 h" in audit_a02["static_verification"]
        and audit_a02["status"] == "resolved_contract_level"
        and static_qa["v0_4_blocker_repairs"]["A02"][
            "integer_rounding_for_weighted_tau_bar_forbidden"
        ]
        is True
        and static_qa["v0_4_blocker_repairs"]["A02"][
            "continuous_phase_seed_contrast_bootstrap_bound_propagation_frozen"
        ]
        is True,
        {
            "ledger": audit_a02,
            "qa": static_qa["v0_4_blocker_repairs"]["A02"],
        },
        "A02 resolved consistently in ledger and QA",
    )

    # Independent exact-rational toy confirms that weighted tau is noninteger.
    weights = [Fraction(1, 5), Fraction(3, 10), Fraction(1, 2)]
    tau_optimistic = [1, 1, 2]
    tau_conservative = [2, 1, 4]
    tau_opt = sum((w * t for w, t in zip(weights, tau_optimistic)), Fraction())
    tau_cons = sum(
        (w * t for w, t in zip(weights, tau_conservative)), Fraction()
    )
    a.check(
        "A02_exact_rational_noninteger_toy",
        tau_opt == Fraction(3, 2)
        and tau_cons == Fraction(27, 10)
        and tau_opt.denominator != 1
        and tau_cons.denominator != 1,
        {
            "optimistic": f"{tau_opt.numerator}/{tau_opt.denominator}",
            "conservative": f"{tau_cons.numerator}/{tau_cons.denominator}",
        },
        {"optimistic": "3/2", "conservative": "27/10"},
    )
    # Sign-aware contrast and nonnegative draw aggregation remain continuous.
    cell_a = (Fraction(149996, 100000), Fraction(150004, 100000))
    cell_b = (Fraction(269996, 100000), Fraction(270004, 100000))
    contrast = (cell_a[0] - cell_b[1], cell_a[1] - cell_b[0])
    second_effect = (Fraction(9999, 25000), Fraction(10001, 25000))
    draw = (
        Fraction(1, 3) * contrast[0] + Fraction(2, 3) * second_effect[0],
        Fraction(1, 3) * contrast[1] + Fraction(2, 3) * second_effect[1],
    )
    lower_draws = [draw[0], draw[0] + Fraction(1, 10000), draw[0] + Fraction(2, 10000)]
    upper_draws = [draw[1], draw[1] + Fraction(1, 10000), draw[1] + Fraction(2, 10000)]
    q_lower = type7(lower_draws, Fraction(1, 40))
    q_upper = type7(upper_draws, Fraction(39, 40))
    a.check(
        "A02_continuous_interval_toy",
        contrast[0] <= contrast[1]
        and draw[0] <= draw[1]
        and q_lower.denominator != 1
        and q_upper.denominator != 1,
        {
            "contrast": [str(contrast[0]), str(contrast[1])],
            "draw": [str(draw[0]), str(draw[1])],
            "type7_formal": [str(q_lower), str(q_upper)],
        },
        "ordered continuous interval at every layer without integer rounding",
    )

    # B01: exact Spearman operands, invariant N_E, and closed 21-field bytes.
    chrono = contract["chronology_randomization"]
    orth = chrono["orthogonal_phase_selection"]
    inventory = chrono["inventory_contract"]
    schema = inventory["record_schema"]
    top_fields = set(inventory["exact_top_level_fields"])
    required_fields = set(schema["required"])
    property_fields = set(schema["properties"])
    a.check(
        "B01_spearman_operands",
        "exact 24-hour decimal-text sum of task_c3_q75_mw"
        in orth["spearman"]["left_operand"]
        and "exact 24-hour decimal-text sum of r2_potential_recharge_mw"
        in orth["spearman"]["right_operand"]
        and orth["spearman"]["indicator_vectors_forbidden"] is True
        and chrono["daily_energy_arithmetic"][
            "binary_float_preaggregation_forbidden"
        ]
        is True,
        orth["spearman"],
        "exact mapped daily task and R2 energy; no indicators/floating preaggregation",
    )
    a.check(
        "B01_NE_definition",
        "sum_{d=0}^{364} H_task[d]" in orth["N_E"]["definition"]
        and orth["N_E"]["zero_rule"].startswith("N_E=0 is a hard")
        and "sole denominator" in orth["N_E"]["use"],
        orth["N_E"],
        "positive invariant annual source event-day count and sole overlap denominator",
    )
    a.check(
        "B01_closed_21_fields",
        len(top_fields) == 21
        and top_fields == required_fields == property_fields
        and schema["additionalProperties"] is False,
        {
            "exact": len(top_fields),
            "required": len(required_fields),
            "properties": len(property_fields),
            "sets_equal": top_fields == required_fields == property_fields,
        },
        "21 identical closed fields",
    )
    nested_object_errors: list[str] = []

    def inspect_schema(node: Any, path: str) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" and node.get("additionalProperties") is not False:
                nested_object_errors.append(path)
            if "properties" in node and "required_keys" in node:
                if set(node["properties"]) != set(node["required_keys"]):
                    nested_object_errors.append(path + ":key-set")
            for key, value in node.items():
                inspect_schema(value, f"{path}.{key}")
        elif isinstance(node, list):
            for idx, value in enumerate(node):
                inspect_schema(value, f"{path}[{idx}]")

    inspect_schema(schema, "record_schema")
    a.check(
        "B01_nested_schema_closed",
        not nested_object_errors,
        nested_object_errors,
        [],
    )
    a.check(
        "B01_canonical_bytes",
        inventory["number_contract"]["binary_float"].startswith("forbidden")
        and inventory["number_contract"]["null"] == "forbidden"
        and "fixed-point" in inventory["number_contract"]["canonical_decimal_string"]
        and "including the final record"
        in inventory["canonical_serialization"]["line_ending"]
        and "exact concatenation"
        in inventory["canonical_serialization"]["inventory_hash"],
        {
            "number_contract": inventory["number_contract"],
            "canonical_serialization": inventory["canonical_serialization"],
        },
        "no floats/null/NaN; canonical Decimal; terminal LF; exact file hash",
    )

    # B02 exact-rational event aggregation and zero rules.
    soc = [Fraction(-1, 10), Fraction(1, 5), Fraction(2, 5)]
    weighted_soc = sum((w * x for w, x in zip(weights, soc)), Fraction())
    hit_opt = [1, 0, 1]
    hit_cons = [1, 0, 0]
    rec_opt = sum((w * x for w, x in zip(weights, hit_opt)), Fraction())
    rec_cons = sum((w * x for w, x in zip(weights, hit_cons)), Fraction())
    cens_opt = 1 - rec_opt
    cens_cons = 1 - rec_cons
    event_charge = [10, 0, 2]
    gap_charge = [4, 0, 6]
    weighted_event_charge = sum(
        (w * x for w, x in zip(weights, event_charge)), Fraction()
    )
    weighted_gap_charge = sum(
        (w * x for w, x in zip(weights, gap_charge)), Fraction()
    )
    eligible = [2, 0, 4]
    definite = [1, 0, 1]
    possible = [2, 0, 3]
    denom = sum((w * x for w, x in zip(weights, eligible)), Fraction())
    sat_lower = sum(
        (w * x for w, x in zip(weights, definite)), Fraction()
    ) / denom
    sat_upper = sum(
        (w * x for w, x in zip(weights, possible)), Fraction()
    ) / denom
    a.check(
        "B02_event_aggregation_toy",
        weighted_soc == Fraction(6, 25)
        and rec_opt == Fraction(7, 10)
        and rec_cons == Fraction(1, 5)
        and cens_opt == Fraction(3, 10)
        and cens_cons == Fraction(4, 5)
        and weighted_event_charge == Fraction(3, 1)
        and weighted_gap_charge == Fraction(19, 5)
        and sat_lower == Fraction(7, 24)
        and sat_upper == Fraction(19, 24),
        {
            "weighted_soc": str(weighted_soc),
            "recovery": [str(rec_opt), str(rec_cons)],
            "censored": [str(cens_opt), str(cens_cons)],
            "event_charge": str(weighted_event_charge),
            "gap_charge": str(weighted_gap_charge),
            "saturation": [str(sat_lower), str(sat_upper)],
        },
        {
            "weighted_soc": "6/25",
            "recovery": ["7/10", "1/5"],
            "censored": ["3/10", "4/5"],
            "event_charge": "3",
            "gap_charge": "19/5",
            "saturation": ["7/24", "19/24"],
        },
    )
    a.check(
        "B02_zero_rules",
        "hard country-year failure"
        in aggregation["zero_rules"]["no_events_or_zero_annual_event_energy"]
        and "first_hit_hours=1"
        in aggregation["zero_rules"]["zero_gap"]
        and "not_identified"
        in aggregation["zero_rules"][
            "zero_weighted_eligible_saturation_denominator"
        ],
        aggregation["zero_rules"],
        "zero-event hard fail; zero-gap censored tau=1; zero saturation exposure unidentified",
    )
    a.check(
        "B02_annual_totals_not_weighted",
        "never multiplied by event weights"
        in aggregation["annual_non_event_weighted_endpoints"]["rule"],
        aggregation["annual_non_event_weighted_endpoints"]["rule"],
        "direct annual totals are not event-weighted",
    )

    # B03 exact-rational nonlinear retention.
    retention = endpoints["retention"]
    interval = (Fraction(4, 3), Fraction(3, 1))
    ratio_mean = Fraction(1 + 3, 2)
    mean_ratio = Fraction(1 + 9, 1) / Fraction(1 + 3, 1)
    draw_interval = (
        Fraction(2, 3) * interval[0] + Fraction(1, 3) * Fraction(0, 1),
        Fraction(2, 3) * interval[1] + Fraction(1, 3) * Fraction(1, 2),
    )
    a.check(
        "B03_interval_and_order_toy",
        interval == (Fraction(4, 3), Fraction(3, 1))
        and ratio_mean == Fraction(2, 1)
        and mean_ratio == Fraction(5, 2)
        and draw_interval == (Fraction(8, 9), Fraction(13, 6)),
        {
            "pair_interval": [str(x) for x in interval],
            "mean_pairwise_ratios": str(ratio_mean),
            "ratio_of_means": str(mean_ratio),
            "draw_interval": [str(x) for x in draw_interval],
        },
        {
            "pair_interval": ["4/3", "3"],
            "mean_pairwise_ratios": "2",
            "ratio_of_means": "5/2",
            "draw_interval": ["8/9", "13/6"],
        },
    )
    a.check(
        "B03_contract_rules",
        retention["ratio_of_means_forbidden"] is True
        and retention["certified_interval_rule"]["lower"] == "L_R=L_CC/U_DD"
        and retention["certified_interval_rule"]["upper"] == "U_R=U_CC/L_DD"
        and "not_identified"
        in retention["certified_interval_rule"][
            "denominator_interval_touches_zero"
        ]
        and retention["bootstrap_recomputation"]["ratio_of_draw_means_forbidden"]
        is True
        and retention["bootstrap_recomputation"][
            "ratio_of_country_means_forbidden"
        ]
        is True
        and "entire draw is not_identified"
        in retention["bootstrap_recomputation"]["not_identified_pair_rule"],
        retention,
        "pairwise monotone interval division before all aggregation, fail-closed zero denominator",
    )

    # B04 five LFP accounting cases and common-direction gate.
    matrix = contract["lfp_cycle_degradation"]["formal_accounting_cross_matrix"]
    matrix_by_id = {row["scenario_id"]: row for row in matrix}
    flat_lfp_ids = {
        r["item_id"] for r in flat_rows if r["section"] == "lfp_accounting"
    } - {"double_count_gate", "five_node_cross_gate"}
    a.check(
        "B04_five_scenario_identity",
        set(matrix_by_id) == EXPECTED_LFP_IDS == flat_lfp_ids
        and len(matrix) == 5,
        {
            "json": sorted(matrix_by_id),
            "csv": sorted(flat_lfp_ids),
        },
        sorted(EXPECTED_LFP_IDS),
    )
    a.check(
        "B04_scenario_parameters",
        matrix_by_id["clean_fom_frep_0_468106"]["replaceable_fraction"]
        == 0.468106
        and matrix_by_id["clean_fom_frep_0_565037"]["replaceable_fraction"]
        == 0.565037
        and matrix_by_id["bundled_fom_no_replacement"][
            "explicit_component_replacement"
        ]
        is False
        and matrix_by_id["bundled_fom_no_replacement"]["replaceable_fraction"]
        == 0.0
        and matrix_by_id["zero_fom_frep_0_468106"][
            "fom_eur2024_per_kw_year"
        ]
        == 0.0
        and matrix_by_id["zero_fom_frep_0_565037"][
            "fom_eur2024_per_kw_year"
        ]
        == 0.0,
        matrix,
        "two explicit fractions crossed with clean/zero FOM plus one bundled/no-replacement case",
    )
    common_gate = contract["lfp_cycle_degradation"]["formal_cross_matrix_execution"]
    a.check(
        "B04_common_direction_gate",
        contract["lfp_cycle_degradation"][
            "economic_direction_claim_requires_five_scenario_interval_gate"
        ]
        is True
        and "every one of the five certified"
        in common_gate["common_direction_admission_gate"]
        and "touching zero" in common_gate["common_direction_admission_gate"]
        and "mixed signs" in common_gate["common_direction_admission_gate"]
        and ("lfp_accounting", "five_node_cross_gate") in flat_index,
        common_gate,
        "all five certified CAES-minus-LFP intervals exclude zero on the same side",
    )

    # R01-R03.
    lower = contract["lfp_cycle_degradation"][
        "certified_global_lower_approximation"
    ]
    global_cert = lower["global_relative_error_certificate"]
    a.check(
        "R01_global_relative_error_certificate",
        lower["direction"] == "F_lower(N)<=F(N) for every N in [0,Nmax]"
        and lower["max_uniform_relative_error"] == 0.0001
        and "epsilon_abs_UB/Fmin_LB"
        in global_cert["relative_error_upper_bound"]
        and "RNDD" in global_cert["transcendental_arithmetic"]
        and "RNDU" in global_cert["transcendental_arithmetic"]
        and global_cert["finite_dense_grid_is_not_a_certificate"] is True
        and lower["gap_clamping_to_zero_forbidden"] is True,
        lower,
        "analytic all-domain supporting-tangent certificate, relative error <=1e-4",
    )
    ratio_map = contract["charge_interface_economics"][
        "break_even_interface_cost"
    ]["reference_ratio_value"]
    a.check(
        "R02_reference_ratio_map",
        ratio_map == {"li_lfp_4h": 1.0, "caes_24h": 1.5}
        and flat_index[
            ("charge_economic", "break_even_reference_ratio_map")
        ]["value"]
        == "li_lfp_4h=1.0|caes_24h=1.5",
        {
            "json": ratio_map,
            "csv": flat_index[
                ("charge_economic", "break_even_reference_ratio_map")
            ]["value"],
        },
        {"li_lfp_4h": 1.0, "caes_24h": 1.5},
    )
    a.check(
        "R03_flat_sensitivity",
        flat_lfp_ids == EXPECTED_LFP_IDS
        and "all five"
        in flat_index[("lfp_accounting", "five_node_cross_gate")]["value"],
        sorted(flat_lfp_ids),
        sorted(EXPECTED_LFP_IDS),
    )

    # 20-object provenance, including decompressed canonical payload.
    source_objects = contract["source_hashes"]
    source_by_id = {row["object_id"]: row for row in source_objects}
    csv_source_by_id = {
        row["item_id"]: row for row in flat_rows if row["section"] == "provenance"
    }
    provenance_mismatches: list[dict[str, str]] = []
    for object_id, item in source_by_id.items():
        path = root / item["path"]
        if object_id == "stage0_canonical_decompressed_payload":
            actual = sha256_gzip_payload(path)
        else:
            actual = sha256_file(path) if path.is_file() else "MISSING"
        csv_item = csv_source_by_id.get(object_id, {})
        if (
            actual != item["sha256"]
            or csv_item.get("source_path") != item["path"]
            or csv_item.get("source_sha256") != item["sha256"]
        ):
            provenance_mismatches.append(
                {
                    "object_id": object_id,
                    "actual": actual,
                    "json": item["sha256"],
                    "csv": csv_item.get("source_sha256", "MISSING"),
                }
            )
    a.check(
        "PROV_20_objects",
        len(source_objects) == 20
        and len(source_by_id) == 20
        and set(source_by_id) == set(csv_source_by_id)
        and not provenance_mismatches,
        {
            "json_count": len(source_objects),
            "json_unique": len(source_by_id),
            "csv_count": len(csv_source_by_id),
            "mismatches": provenance_mismatches,
        },
        {"count": 20, "mismatches": []},
    )

    # Independently rebuild the prespecified 118-country-year sensitivity.
    sample_path = root / contract["samples"]["floor_material_15c_118cy"][
        "sample_list_path"
    ]
    _, sample_rows, sample_rectangular = read_csv_rectangular(sample_path)
    reconstructed = sorted(
        f"{row['country']}-{int(row['year'])}"
        for row in sample_rows
        if row["main_prespecified_sample"].strip().lower() == "true"
        and Decimal(row["above_floor_gas_share_of_system_load"])
        >= Decimal("0.01")
    )
    json_keys = sorted(
        contract["samples"]["floor_material_15c_118cy"]["country_year_keys"]
    )
    reconstructed_counts = Counter(key[:2] for key in reconstructed)
    a.check(
        "SAMPLE_118_country_years",
        sample_rectangular
        and len(reconstructed) == 118
        and len(set(reconstructed)) == 118
        and reconstructed == json_keys,
        {
            "count": len(reconstructed),
            "unique": len(set(reconstructed)),
            "country_counts": dict(sorted(reconstructed_counts.items())),
            "keys_match_json": reconstructed == json_keys,
        },
        {"count": 118, "keys_match_json": True},
    )

    # Kgas and strict separation of mechanism and economic evidence layers.
    econ = contract["economic_accounting"]
    layers = econ["evidence_layers"]
    a.check(
        "LEDGER_Kgas",
        set(econ["residual_gas_capacity_constraints"])
        == {
            "Kgas_MW>=gas_t_MW for every hour",
            "Kgas_MW>=0",
        }
        and "max_t(gas_t_MW)"
        in econ["residual_gas_capacity_at_economic_optimum"]
        and flat_index[("economic", "residual_gas_capacity")]["value"]
        == "Kgas>=gas_t and Kgas>=0",
        {
            "json": econ["residual_gas_capacity_constraints"],
            "csv": flat_index[("economic", "residual_gas_capacity")]["value"],
        },
        "Kgas>=g_t for all t and Kgas>=0",
    )
    a.check(
        "LEDGER_mechanism_economic_separation",
        "only over Fstar" in layers["physical_mechanism_layer"]
        and "complete storage physical feasible set"
        in layers["economic_layer_objective"]
        and layers["economic_layer_must_not_be_restricted_to_Fstar"] is True
        and layers["physical_lexicographic_incumbent_direct_pricing_forbidden"]
        is True
        and "certified economic-cost optimum interval" in layers["benefit_source"],
        layers,
        "mechanism endpoints over Fstar; economic optimum over full feasible set",
    )

    # Immutable v0.4, v0.3 and v0.2 predecessors.
    v04_files = {
        "markdown": base / "OBSERVED_GAS_EVENT_TASK_STAGE2_FORMAL_CONTRACT_v0_4.md",
        "json": base / "OBSERVED_GAS_EVENT_TASK_STAGE2_CONTRACT_v0_4.json",
        "flat_csv": base / "observed_gas_event_task_stage2_contract_v0_4.csv",
        "audit_response_csv": base
        / "observed_gas_event_task_stage2_v0_4_audit_response.csv",
        "static_qa_json": base
        / "OBSERVED_GAS_EVENT_TASK_STAGE2_CONTRACT_v0_4_STATIC_QA.json",
    }
    v04_actual = {key: sha256_file(path) for key, path in v04_files.items()}
    a.check(
        "PREDECESSOR_v04_five_files",
        v04_actual == contract["predecessor_v0_4_hashes"],
        v04_actual,
        contract["predecessor_v0_4_hashes"],
    )
    v04_audit_actual = sha256_file(v04_audit_path)
    a.check(
        "PREDECESSOR_v04_final_audit",
        v04_audit_actual
        == contract["v0_4_final_independent_audit_sha256"]
        == EXPECTED_V04_AUDIT_SHA,
        v04_audit_actual,
        EXPECTED_V04_AUDIT_SHA,
    )
    v03_files = {
        "markdown": base / "OBSERVED_GAS_EVENT_TASK_STAGE2_FORMAL_CONTRACT_v0_3.md",
        "json": base / "OBSERVED_GAS_EVENT_TASK_STAGE2_CONTRACT_v0_3.json",
        "flat_csv": base / "observed_gas_event_task_stage2_contract_v0_3.csv",
        "audit_response_csv": base
        / "observed_gas_event_task_stage2_v0_3_audit_response.csv",
        "static_qa_json": base
        / "OBSERVED_GAS_EVENT_TASK_STAGE2_CONTRACT_v0_3_STATIC_QA.json",
    }
    v03_actual = {key: sha256_file(path) for key, path in v03_files.items()}
    a.check(
        "PREDECESSOR_v03_five_files",
        v03_actual == contract["predecessor_v0_3_hashes"],
        v03_actual,
        contract["predecessor_v0_3_hashes"],
    )
    v03_audit_actual = sha256_file(
        base / "OBSERVED_GAS_EVENT_TASK_STAGE2_CONTRACT_V0_3_FINAL_INDEPENDENT_AUDIT.md"
    )
    a.check(
        "PREDECESSOR_v03_final_audit",
        v03_audit_actual == contract["v0_3_final_independent_audit_sha256"],
        v03_audit_actual,
        contract["v0_3_final_independent_audit_sha256"],
    )
    v02_mismatches: list[dict[str, str]] = []
    for row in contract["predecessor_v0_2_hashes"]:
        actual = sha256_file(root / row["path"])
        if actual != row["sha256"]:
            v02_mismatches.append(
                {"path": row["path"], "actual": actual, "expected": row["sha256"]}
            )
    a.check(
        "PREDECESSOR_v02_five_files",
        len(contract["predecessor_v0_2_hashes"]) == 5 and not v02_mismatches,
        {"count": len(contract["predecessor_v0_2_hashes"]), "mismatches": v02_mismatches},
        {"count": 5, "mismatches": []},
    )

    # State and authorization boundary.
    state_observed = {
        "status": contract["status"],
        "chronology_inventory_materialized": contract[
            "chronology_inventory_materialized"
        ],
        "formal_runner_frozen": contract["formal_runner_frozen"],
        "optimization_started_by_contract_task": contract[
            "optimization_started_by_contract_task"
        ],
        "optimization_authorized_by_this_contract_task": contract[
            "optimization_authorized_by_this_contract_task"
        ],
        "manuscript_evidence_admitted": contract["manuscript_evidence_admitted"],
        "dual_generator_dry_run_authorized": static_qa["overall_decision"][
            "dual_generator_dry_run_authorized"
        ],
        "formal_run_authorization": static_qa["overall_decision"][
            "formal_run_authorization"
        ],
    }
    state_expected = {
        "status": "preregistered_not_executed",
        "chronology_inventory_materialized": False,
        "formal_runner_frozen": False,
        "optimization_started_by_contract_task": False,
        "optimization_authorized_by_this_contract_task": False,
        "manuscript_evidence_admitted": False,
        "dual_generator_dry_run_authorized": False,
        "formal_run_authorization": "FAIL_PENDING",
    }
    inventory_candidates = sorted(
        str(path.relative_to(root))
        for path in (
            root
            / "dataset/external_hourly/europe_mid_duration_valley_upgrade_v0_1"
        ).rglob("*")
        if path.is_file()
        and (
            "chronology_inventory" in path.name.lower()
            or "inventory_chronology" in path.name.lower()
            or "stage2_inventory" in path.name.lower()
        )
    )
    a.check(
        "STATE_no_execution_no_authorization",
        state_observed == state_expected and not inventory_candidates,
        {"state": state_observed, "inventory_candidates": inventory_candidates},
        {"state": state_expected, "inventory_candidates": []},
    )

    failed = a.failures
    result = {
        "audit_id": "observed_gas_event_task_stage2_contract_v0_5_final_independent_audit",
        "audit_date": "2026-07-27",
        "independence": {
            "trusted_static_qa": False,
            "canonical_hashes_recomputed": True,
            "all_provenance_hashes_recomputed": True,
            "sample_118_reconstructed_from_source_rows": True,
            "rational_toy_cases_recomputed": True,
            "v0_5_contract_files_modified": False,
            "chronology_inventory_materialized": False,
            "optimization_started": False,
        },
        "package_hashes": actual_package_hashes,
        "canonical_subobject_hashes": {
            "chronology_randomization": chronology_sha,
            "inference": inference_sha,
        },
        "check_count": len(a.checks),
        "pass_count": len(a.checks) - len(failed),
        "fail_count": len(failed),
        "checks": a.checks,
        "overall_decision": "PASS" if not failed else "FAIL",
        "authorization_boundary": {
            "independent_v0_5_contract_audit": "PASS" if not failed else "FAIL",
            "dual_independent_inventory_generator_dry_run": (
                "MAY_PROCEED_IN_SEPARATE_NON_MATERIALIZING_TASK"
                if not failed
                else "NOT_AUTHORIZED"
            ),
            "chronology_inventory_materialization": "NOT_AUTHORIZED",
            "formal_runner_freeze": "NOT_AUTHORIZED",
            "formal_optimization": "NOT_AUTHORIZED",
            "manuscript_evidence": "NOT_ADMITTED",
        },
    }

    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
