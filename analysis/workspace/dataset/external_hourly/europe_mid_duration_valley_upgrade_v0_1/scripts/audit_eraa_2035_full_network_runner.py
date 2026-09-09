#!/usr/bin/env python3
"""Independent, fail-closed audit of the ERAA 2035 full-network runner.

The audit deliberately does not edit the model runner.  It records the exact
SHA-256 of the file under review, evaluates static and source-data contracts,
and (when Gurobi is available) solves a small 168-hour two-node system through
the runner's public ``solve_model`` function.

Outputs are written to ``artifacts/research``:

* ``audit_eraa_full_network_runner_gates.csv``
* ``audit_eraa_full_network_runner_findings.csv``
* ``audit_eraa_full_network_runner_qa.json``
* ``audit_eraa_full_network_synthetic_results.csv``
* ``ERAA_FULL_NETWORK_RUNNER_INDEPENDENT_AUDIT.md``

This script distinguishes three scopes.  A ``task`` gate can be closed by one
runner execution, a ``suite`` gate requires aggregation or a prescribed
sensitivity suite, and a ``cross_implementation`` gate requires agreement
between this runner and the historical/core implementations.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import re
import sys
import tempfile
import traceback
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_RUNNER = HERE / "run_eraa_2035_full_network_gurobi.py"
DEFAULT_CORE = HERE / "eraa_confirmatory_model_core.py"
DEFAULT_ARTIFACTS = ROOT / "artifacts" / "research"
STATUS_VALUES = {"pass", "fail", "warn", "pending", "not_run"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


class Audit:
    def __init__(self) -> None:
        self.gates: list[dict[str, Any]] = []
        self.findings: list[dict[str, Any]] = []

    def gate(
        self,
        gate_id: str,
        scope: str,
        status: str,
        critical: bool,
        evidence: str,
        required_for: str,
    ) -> None:
        if status not in STATUS_VALUES:
            raise ValueError(status)
        self.gates.append(
            {
                "gate_id": gate_id,
                "scope": scope,
                "status": status,
                "critical": bool(critical),
                "required_for": required_for,
                "evidence": evidence,
            }
        )

    def finding(
        self,
        finding_id: str,
        severity: str,
        classification: str,
        title: str,
        evidence: str,
        implication: str,
    ) -> None:
        self.findings.append(
            {
                "finding_id": finding_id,
                "severity": severity,
                "classification": classification,
                "title": title,
                "evidence": evidence,
                "implication": implication,
            }
        )


def parse_default_args(module) -> argparse.Namespace:
    original = sys.argv[:]
    try:
        sys.argv = [str(DEFAULT_RUNNER), "--config-id", "caes_24h"]
        return module.parse_args()
    finally:
        sys.argv = original


def contains_all(text: str, needles: tuple[str, ...]) -> bool:
    return all(needle in text for needle in needles)


def static_and_source_audit(
    audit: Audit,
    module,
    source: str,
    runner: Path,
) -> dict[str, Any]:
    details: dict[str, Any] = {}

    try:
        compile(source, str(runner), "exec")
        audit.gate(
            "runner_compiles",
            "task",
            "pass",
            True,
            "Python source compiled without syntax error.",
            "NTC screen",
        )
    except Exception as exc:
        audit.gate(
            "runner_compiles",
            "task",
            "fail",
            True,
            repr(exc),
            "NTC screen",
        )

    fixed_test = bool(module.fixed_cost_analytic_self_test())
    audit.gate(
        "candidate_fixed_cost_unit_self_test",
        "task",
        "pass" if fixed_test else "fail",
        True,
        "2 MW × 5 EUR kW-yr = 10,000 EUR/yr and "
        "2 MW × 10 h × 3 EUR kWh-yr = 60,000 EUR/yr.",
        "NTC screen",
    )
    idsr_test = bool(
        getattr(module, "idsr_window_analytic_self_test", lambda: False)()
    )
    audit.gate(
        "idsr_cyclic_window_self_test",
        "task",
        "pass" if idsr_test else "fail",
        True,
        "The analytical two-day EV/HP test joins the overnight window across "
        "the cyclic year boundary.",
        "NTC screen",
    )

    currency_ok = contains_all(
        source,
        (
            '"low": "low_cny_per_eur"',
            '"central": "central_cny_per_eur"',
            '"high": "high_cny_per_eur"',
        ),
    )
    audit.gate(
        "currency_scenario_mapping",
        "task",
        "pass" if currency_ok else "fail",
        True,
        "Runner maps command-line low/central/high to ledger scenario names.",
        "NTC screen",
    )

    storage_container_ok = (
        "indexed = {item.config_id: item for item in configurations}" in source
        and "return indexed, rate" in source
    )
    audit.gate(
        "storage_configuration_container",
        "task",
        "pass" if storage_container_ok else "fail",
        True,
        "Configurations are indexed by config_id before .values()/.items() use.",
        "NTC screen",
    )

    cyclic_soc_ok = contains_all(
        source,
        (
            "candidate_cycle_initial",
            "storage.hourly_retention * candidate_soc[:, -1]",
            "candidate_stock",
            "storage.soc_min_fraction",
            "storage.soc_max_fraction",
        ),
    )
    audit.gate(
        "candidate_cyclic_soc",
        "task",
        "pass" if cyclic_soc_ok else "fail",
        True,
        "Candidate SOC is bounded and the first hour is linked to the last "
        "hour with hourly retention.",
        "NTC screen",
    )

    rated_efc_ok = contains_all(
        source,
        (
            "rated_energy = total_pdis * storage.duration_h",
            '"equivalent_full_cycles_output_basis"',
            "candidate_discharge_mwh / rated_energy",
            "delivered_discharge_mwh_divided_by_rated_energy_mwh",
        ),
    )
    audit.gate(
        "canonical_rated_energy_efc_in_full_runner",
        "task",
        "pass" if rated_efc_ok else "fail",
        True,
        "Canonical EFC is delivered discharge divided by rated energy capacity; "
        "usable-window and internal-withdrawal diagnostics are separately named.",
        "NTC screen",
    )

    core_text = DEFAULT_CORE.read_text(encoding="utf-8")
    core_rated = (
        re.search(
            r'"equivalent_full_cycles_output_basis"\s*:\s*\(\s*'
            r"expected_discharge\s*/\s*rated_energy",
            core_text,
            flags=re.S,
        )
        is not None
    )
    audit.gate(
        "canonical_efc_cross_implementation",
        "cross_implementation",
        "pass" if core_rated else "fail",
        True,
        "Historical/core implementation "
        + (
            "uses the same rated-energy denominator."
            if core_rated
            else "still uses a usable-SOC-window denominator under the "
            "canonical EFC name."
        ),
        "manuscript aggregation",
    )
    if not core_rated:
        audit.finding(
            "F_EFC_CORE",
            "major",
            "仍阻断",
            "Core and full runner use different EFC denominators",
            "The full runner uses rated energy, whereas "
            "eraa_confirmatory_model_core.py still divides by the usable SOC "
            "window under the canonical field name.",
            "Historical/synthetic results must be regenerated or their field "
            "must be renamed before pooling with the ERAA suite.",
        )
    smoke_manifest_path = (
        ROOT
        / "model_adapter"
        / "eraa_confirmatory_smoke_v0_1"
        / "run_manifest.json"
    )
    current_core_sha = sha256(DEFAULT_CORE)
    if smoke_manifest_path.exists():
        smoke_manifest = json.loads(
            smoke_manifest_path.read_text(encoding="utf-8")
        )
        smoke_core_sha = str(
            smoke_manifest.get("core_script_sha256", "")
        )
        smoke_fresh = smoke_core_sha == current_core_sha
        evidence = (
            f"current core={current_core_sha}; synthetic manifest "
            f"core={smoke_core_sha or 'missing'}."
        )
    else:
        smoke_fresh = False
        evidence = f"Missing {smoke_manifest_path}."
    audit.gate(
        "derived_core_results_regenerated_after_efc_change",
        "cross_implementation",
        "pass" if smoke_fresh else "fail",
        True,
        evidence,
        "manuscript aggregation",
    )
    if not smoke_fresh:
        audit.finding(
            "F_STALE_CORE_OUTPUT",
            "major",
            "仍阻断",
            "Stored synthetic/core results predate the canonical EFC change",
            evidence,
            "Regenerate derived summaries or perform a traceable algebraic "
            "migration before using their EFC values.",
        )
    legacy_core = (
        ROOT.parents[2]
        / "Decarbonizing-Fossil-Peaking-with-Battery-Storage-master"
        / "paper-code"
        / "docs"
        / "musrcipt"
        / "revision_nature_energy_v36"
        / "system_value_closure"
        / "run_v36_system_value_closure.py"
    )
    if legacy_core.exists():
        legacy_text = legacy_core.read_text(encoding="utf-8")
        legacy_uses_rated = bool(
            re.search(
                r"expected_equivalent_full_cycles_output_basis"
                r'"\s*:\s*\(\s*expected_discharge\s*/\s*rated_energy',
                legacy_text,
                flags=re.S,
            )
        )
        legacy_evidence = (
            f"{legacy_core}: "
            + (
                "rated-energy definition detected."
                if legacy_uses_rated
                else "still divides by the 5–95% usable SOC window; existing "
                "155-country-year EFC values require a 0.9 multiplier or rerun."
            )
        )
    else:
        legacy_uses_rated = False
        legacy_evidence = f"Missing legacy European screen core {legacy_core}."
    audit.gate(
        "historical_europe_screen_efc_migrated",
        "cross_implementation",
        "pass" if legacy_uses_rated else "fail",
        True,
        legacy_evidence,
        "manuscript aggregation",
    )
    if not legacy_uses_rated:
        audit.finding(
            "F_LEGACY_EFC",
            "major",
            "仍阻断",
            "The existing 155-country-year European screen uses the old EFC denominator",
            legacy_evidence,
            "Apply the traceable 0.9 denominator migration and regenerate every "
            "absolute EFC table, figure and manuscript statement before pooling "
            "those results with the ERAA screen.",
        )

    cost_units_ok = contains_all(
        source,
        (
            "annual_discharge_power_cost_eur_kw",
            "* 1_000.0",
            "annual_energy_cost_eur_kwh",
            "* duration_h",
        ),
    )
    audit.gate(
        "candidate_fixed_cost_units",
        "task",
        "pass" if cost_units_ok else "fail",
        True,
        "Power cost converts MW to kW; energy cost converts MW×h to kWh.",
        "NTC screen",
    )

    objective_ok = contains_all(
        source,
        (
            "objective_components",
            "component_error",
            "objective component audit failed",
            "candidate_incremental_charge_interface_cost_eur_year",
            "additional_dispatchable_consumption_net_value_eur_year",
        ),
    )
    audit.gate(
        "objective_reconstruction_contract",
        "task",
        "pass" if objective_ok else "fail",
        True,
        "All material objective terms, including additional flexible demand "
        "value, are explicitly reconstructed and compared with Gurobi ObjVal.",
        "NTC screen",
    )

    vre_spill_ok = contains_all(
        source,
        (
            'name="vre_spill"',
            "system.wind",
            "system.solar",
            "system.btm_pv",
            "system.csp_storage_reference",
            "expression = expression - vre_spill",
        ),
    )
    audit.gate(
        "vre_potential_endogenous_use_and_spill",
        "task",
        "pass" if vre_spill_ok else "fail",
        True,
        "Wind, solar, BTM PV and fixed CSP potential enter nodal balance with "
        "an explicit bounded spill decision; charging energy is therefore "
        "endogenous rather than an exogenous curtailment allocation.",
        "NTC screen",
    )

    residual_families = (
        ("balance", re.compile(r"maximum_.*balance.*residual", re.I)),
        ("candidate_soc", re.compile(r"maximum_.*candidate.*soc.*residual", re.I)),
        (
            "existing_battery_soc",
            re.compile(r"maximum_.*(battery|existing_storage).*soc.*residual", re.I),
        ),
        (
            "pumped_hydro_soc",
            re.compile(r"maximum_.*(pumped|psp).*soc.*residual", re.I),
        ),
        (
            "reservoir_soc",
            re.compile(r"maximum_.*reservoir.*(soc|stock).*residual", re.I),
        ),
        ("network", re.compile(r"maximum_.*network.*(residual|violation)", re.I)),
        ("dsr", re.compile(r"maximum_.*dsr.*(residual|violation)", re.I)),
        (
            "must_run",
            re.compile(r"maximum_.*must_run.*(residual|violation)", re.I),
        ),
    )
    present_residuals = [
        name for name, pattern in residual_families if pattern.search(source)
    ]
    residual_ok = len(present_residuals) == len(residual_families)
    audit.gate(
        "physical_residual_reconstruction",
        "task",
        "pass" if residual_ok else "fail",
        True,
        f"Detected {len(present_residuals)}/{len(residual_families)} required "
        f"residual families: {present_residuals}.",
        "NTC screen",
    )

    simultaneous_absolute = (
        "maximum_simultaneous_charge_discharge_mwh" in source
    )
    simultaneous_relative = any(
        token in source
        for token in (
            "maximum_simultaneous_charge_discharge_fraction",
            "maximum_simultaneous_charge_discharge_ratio",
            "simultaneous_charge_discharge_to_throughput_ratio",
        )
    )
    # Accept either an inline clipped-overlap expression or the clearer helper
    # form used by the frozen runner.  In both cases negative numerical noise
    # must be clipped before the minimum is taken.
    clipped = bool(
        re.search(
            r"simultaneous_[a-z_]+\s*=\s*float\(\s*np\.minimum\(\s*"
            r"np\.maximum\(",
            source,
            flags=re.S,
        )
        or contains_all(
            source,
            (
                "nonnegative_charge = np.maximum",
                "nonnegative_discharge = np.maximum",
                "simultaneous_overlap = float(",
                "np.minimum(",
            ),
        )
    )
    simultaneous_ok = (
        simultaneous_absolute and simultaneous_relative and clipped
    )
    audit.gate(
        "same_hour_charge_discharge_absolute_and_relative",
        "task",
        "pass" if simultaneous_ok else "fail",
        True,
        "Requires clipped non-negative overlap plus both absolute MWh and "
        "overlap/throughput diagnostics.",
        "NTC screen",
    )

    idsr_fields = {
        "idsr_ev_baseline",
        "idsr_hp_baseline",
        "idsr_schedules",
        "idsr_shifted_load",
        "idsr_window_energy",
        "idsr_energy_residual",
    }
    missing_idsr = sorted(item for item in idsr_fields if item not in source)
    audit.gate(
        "official_ev_hp_idsr_modelled",
        "task",
        "pass" if not missing_idsr else "fail",
        True,
        f"Missing iDSR implementation tokens: {missing_idsr}.",
        "NTC screen",
    )

    additional_fields = {
        "additional_consumers",
        "additional_dispatchable_consumption",
        "power_to_heat_availability_case",
    }
    missing_additional = sorted(
        item for item in additional_fields if item not in source
    )
    audit.gate(
        "additional_flexible_consumption_modelled",
        "task",
        "pass" if not missing_additional else "fail",
        True,
        f"Missing additional-consumer tokens: {missing_additional}.",
        "NTC screen",
    )

    must_run_formula = bool(
        re.search(
            r"hourly_floor\s*=\s*(?:float\()?ratio\)?\s*\*\s*available",
            source,
        )
        or re.search(
            r"hourly_floor\s*=\s*float\(row\.ratio\)\s*\*\s*available",
            source,
        )
    )
    must_run_ok = (
        contains_all(
            source,
            (
                "load_must_run_contracts",
                "must_run_floor",
                "Weighted Average Must-run Ratio",
            ),
        )
        and must_run_formula
    )
    audit.gate(
        "must_run_contracts_modelled",
        "task",
        "pass" if must_run_ok else "fail",
        True,
        "Required formula is weighted must-run ratio × current hourly "
        f"post-EVA available host; detected={must_run_formula}.",
        "NTC screen",
    )

    hydro_ok = contains_all(
        source,
        (
            "add_hydro_contract_constraints",
            "minimum_generation_energy",
            "maximum_generation_energy",
            "minimum_reservoir_level",
            "maximum_reservoir_level",
            "hydro_contract_violation_mwh",
        ),
    )
    audit.gate(
        "released_hydro_constraints_modelled",
        "task",
        "pass" if hydro_ok else "fail",
        True,
        "Power, period-energy and reservoir-level constraints are instantiated "
        "with audited penalised slack.",
        "NTC screen",
    )

    hydro_calendar_fixed = (
        '"chronology_treatment": "calendar_fixed_common_control"' in source
    )
    audit.gate(
        "weather_related_hydrology_synchronized",
        "suite",
        "fail" if hydro_calendar_fixed else "pass",
        True,
        "Frozen confirmatory specification moves weather-related hydrology "
        "with each synchronized day block; the runner currently "
        + (
            "keeps all hydrology on the target calendar."
            if hydro_calendar_fixed
            else "does not declare the all-hydro calendar-fixed treatment."
        ),
        "confirmatory paired suite",
    )
    if hydro_calendar_fixed:
        audit.finding(
            "F_HYDRO_CHRONOLOGY",
            "major",
            "仅 suite 层阻断",
            "Hydrology chronology differs from the frozen paired-intervention contract",
            "The runner keeps inflow and hydro constraints calendar-fixed, while "
            "the specification requires premarked weather-related hydrology to "
            "move with load/wind/solar day blocks.",
            "Either move weather-related inflow/potential in the main treatment "
            "or label this as a fixed-hydrology controlled estimand and complete "
            "a synchronized-hydrology sensitivity before confirmation.",
        )

    dsr_path = module.DEFAULT_FLEXIBLE / "explicit_dsr_bands_2035.csv"
    dsr = pd.read_csv(dsr_path)
    dsr_required = {
        "zone",
        "capacity_mw",
        "activation_price_eur_mwh",
        "max_hours_per_day",
    }
    dsr_complete = (
        dsr_required.issubset(dsr.columns)
        and not dsr[list(dsr_required)].isna().any().any()
        and dsr["capacity_mw"].gt(0).all()
        and dsr["activation_price_eur_mwh"].ge(0).all()
        and dsr["max_hours_per_day"].between(0, 24).all()
    )
    details["dsr_rows"] = int(len(dsr))
    details["dsr_capacity_mw"] = float(dsr["capacity_mw"].sum())
    audit.gate(
        "detailed_dsr_row_completeness",
        "task",
        "pass" if dsr_complete else "fail",
        True,
        f"{len(dsr)} rows; {dsr['capacity_mw'].sum():.6f} MW; zone, "
        "capacity, price band and daily-hour fields checked.",
        "NTC screen",
    )

    chronology_qa_path = module.DEFAULT_CHRONOLOGY / "QA.json"
    chronology_qa = json.loads(chronology_qa_path.read_text(encoding="utf-8"))
    order_path = module.DEFAULT_CHRONOLOGY / "synchronized_day_orders.csv"
    orders = pd.read_csv(order_path)
    chronology_ok = (
        bool(chronology_qa.get("pass", False))
        and orders["weather_scenario"].nunique() == 36
        and set(orders["chronology"].astype(str)) == {"dispersed", "clustered"}
        and len(orders) == 36 * 2 * 365
    )
    details["chronology_rows"] = int(len(orders))
    audit.gate(
        "synchronized_chronology_source_qa",
        "task",
        "pass" if chronology_ok else "fail",
        True,
        f"QA pass={chronology_qa.get('pass')}; "
        f"{orders['weather_scenario'].nunique()} weather scenarios × "
        f"{orders['chronology'].nunique()} paired chronologies × 365 days.",
        "NTC screen",
    )

    edges, zones = module.graph_nodes(module.DEFAULT_DISPATCH)
    network_ok = len(zones) == 75 and len(edges) == 269
    details["ntc_nodes"] = len(zones)
    details["ntc_edges"] = len(edges)
    audit.gate(
        "ntc_graph_coverage",
        "task",
        "pass" if network_ok else "fail",
        True,
        f"{len(zones)} nodes and {len(edges)} directed NTC edges.",
        "NTC screen",
    )
    zone_limit_source = pd.read_csv(
        module.DEFAULT_DISPATCH / "network_zone_limits_hourly.csv.gz",
        usecols=["curve_type"],
    )
    released_curve_types = set(
        zone_limit_source["curve_type"].astype(str).unique()
    )
    supported_curve_tokens = {
        "Net export limit": "net export",
        "Net import limit": "net import",
        "Gross export limit": "gross export",
        "Gross import limit": "gross import",
    }
    unsupported_curves = sorted(
        curve
        for curve in released_curve_types
        if curve not in supported_curve_tokens
        or supported_curve_tokens[curve] not in source.lower()
    )
    details["released_network_zone_limit_curve_types"] = sorted(
        released_curve_types
    )
    audit.gate(
        "released_net_and_gross_zone_limits_modelled",
        "task",
        "pass" if not unsupported_curves else "fail",
        True,
        f"Released curve types={sorted(released_curve_types)}; "
        f"unsupported={unsupported_curves}.",
        "NTC screen",
    )

    catalog = pd.read_csv(module.DEFAULT_CATALOG)
    config_args = SimpleNamespace(
        currency_scenario="central",
        dispatch_root=module.DEFAULT_DISPATCH,
        catalog=module.DEFAULT_CATALOG,
        technology_library=module.DEFAULT_TECH_LIBRARY,
        config_id=catalog["config_id"].astype(str).tolist(),
        incremental_charge_cost_multiplier=1.0,
        lfp_degradation_multiplier=1.0,
        maximum_charge_to_discharge_ratio=10.0,
    )
    configurations, currency_rate = module.storage_configurations(config_args)
    main_count = sum(item.main_evidence_domain for item in configurations.values())
    structural_count = sum(
        item.structural_extrapolation for item in configurations.values()
    )
    details.update(
        {
            "configuration_count": len(configurations),
            "main_configuration_count": int(main_count),
            "structural_configuration_count": int(structural_count),
            "currency_cny_per_eur": float(currency_rate),
        }
    )
    configs_ok = (
        len(configurations) == len(catalog) == 53
        and main_count == 50
        and structural_count == 3
    )
    audit.gate(
        "technology_duration_catalog_complete",
        "task",
        "pass" if configs_ok else "fail",
        True,
        f"{len(configurations)} total, {main_count} main-domain and "
        f"{structural_count} structural configurations.",
        "NTC screen",
    )
    parameter_rows = list(configurations.values())
    parameter_contract_ok = bool(
        parameter_rows
        and all(
            math.isfinite(float(item.duration_h))
            and float(item.duration_h) > 0
            and 0 < float(item.round_trip_efficiency) <= 1
            and np.isclose(float(item.soc_min_fraction), 0.05)
            and np.isclose(float(item.soc_max_fraction), 0.95)
            and float(item.annual_discharge_power_cost_eur_kw) >= 0
            and float(item.annual_charge_power_cost_eur_kw) >= 0
            and float(item.annual_energy_cost_eur_kwh) >= 0
            and float(item.variable_om_eur_mwh) >= 0
            and float(item.degradation_cost_eur_mwh_discharged) >= 0
            for item in parameter_rows
        )
    )
    audit.gate(
        "technology_parameter_units_and_soc_contract",
        "task",
        "pass" if parameter_contract_ok else "fail",
        True,
        "All 53 configurations have finite positive duration/efficiency, "
        "non-negative annualized EUR kW⁻¹-yr⁻¹, EUR kWh⁻¹-yr⁻¹ and EUR "
        "MWh⁻¹ terms, and the common 5–95% cyclic SOC window.",
        "NTC screen",
    )
    lfp_rows = [
        item
        for item in parameter_rows
        if str(item.tech_id).lower() in {"lfp", "li_lfp"}
    ]
    lfp_degradation_ok = bool(
        lfp_rows
        and all(
            float(item.degradation_cost_eur_mwh_discharged) > 0
            for item in lfp_rows
        )
    )
    details["candidate_lfp_degradation_costs_eur_mwh"] = {
        str(item.config_id): float(
            item.degradation_cost_eur_mwh_discharged
        )
        for item in lfp_rows
    }
    audit.gate(
        "candidate_lfp_throughput_degradation_internalized",
        "task",
        "pass" if lfp_degradation_ok else "fail",
        True,
        f"Candidate LFP marginal degradation terms: "
        f"{details['candidate_lfp_degradation_costs_eur_mwh']}.",
        "NTC screen",
    )

    capacity = pd.read_csv(module.DEFAULT_INPUT / "capacity_2035_final.csv")
    excluded_labels = {"Small biomass", "Waste", "Geothermal"}
    excluded = capacity.loc[
        capacity["Target year"].eq(2035)
        & capacity["data_version"].eq("ERAA 2025 final")
        & capacity["Technology"].isin(excluded_labels)
        & capacity["Operational_Status"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("available on market")
    ]
    excluded_mw = (
        excluded.groupby("Technology")["Value"].sum().sort_index().to_dict()
    )
    details["unmodelled_dispatchable_capacity_mw"] = excluded_mw
    source_includes_other = all(
        token in source
        for token in (
            "small_biomass",
            "waste",
            "geothermal",
        )
    )
    audit.gate(
        "other_dispatchable_supply_coverage",
        "task",
        "pass" if source_includes_other or sum(excluded_mw.values()) == 0 else "fail",
        True,
        f"Released available-on-market capacity outside the thermal adapter: "
        f"{excluded_mw}; explicit runner support={source_includes_other}.",
        "NTC screen",
    )
    if not source_includes_other and sum(excluded_mw.values()) > 0:
        audit.finding(
            "F_OTHER_SUPPLY",
            "major",
            "仍阻断",
            "About 27 GW of released small biomass, waste and geothermal capacity is absent",
            str(excluded_mw),
            "Include the resources with documented costs/availability or quantify "
            "and pre-specify their exclusion before interpreting adequacy or ELCC.",
        )
    audit.gate(
        "other_res_parameter_bounds_compared",
        "suite",
        "pending",
        True,
        "Small biomass, waste and geothermal require a pre-specified "
        "quarantine-versus-dispatchable bound (including cost and availability "
        "sensitivity) at every admitted boundary configuration.",
        "confirmatory paired suite",
    )

    cache_ok = contains_all(
        source,
        (
            "Path(__file__).resolve()",
            "CORE_PATH.resolve()",
            "ZONE_RUNNER_PATH.resolve()",
            "actual_input_sha256",
        ),
    )
    audit.gate(
        "baseline_cache_source_hashes",
        "task",
        "pass" if cache_ok else "fail",
        True,
        "Baseline cache key includes runner, core, helper and actual input hashes.",
        "NTC screen",
    )

    current_runner_sha = sha256(runner)
    released_smokes: list[dict[str, Any]] = []
    model_results_root = ROOT / "model_results"
    if model_results_root.exists():
        for manifest_path in model_results_root.rglob("run_manifest.json"):
            qa_path = manifest_path.parent / "QA.json"
            if not qa_path.exists():
                continue
            try:
                manifest = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
                smoke_qa = json.loads(qa_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            source_files = manifest.get("source_files", {})
            recorded_sha = next(
                (
                    str(value)
                    for key, value in source_files.items()
                    if Path(str(key)).name == runner.name
                ),
                "",
            )
            if not recorded_sha:
                continue
            metrics_path = manifest_path.parent / "run_metrics.csv"
            hydro_violation_mwh = math.nan
            hydro_violation_share = math.nan
            if metrics_path.exists():
                smoke_metrics = pd.read_csv(metrics_path)
                if "hydro_contract_violation_mwh" in smoke_metrics:
                    hydro_violation_mwh = float(
                        smoke_metrics[
                            "hydro_contract_violation_mwh"
                        ].max()
                    )
                if (
                    "hydro_contract_violation_share_of_hydro_output"
                    in smoke_metrics
                ):
                    hydro_violation_share = float(
                        smoke_metrics[
                            "hydro_contract_violation_share_of_hydro_output"
                        ].max()
                    )
            hydro_audit = (
                smoke_qa.get("source_audit", {})
                .get("system", {})
                .get("hydro", {})
            )
            released_smokes.append(
                {
                    "path": str(manifest_path.parent),
                    "hours": int(smoke_qa.get("hours", 0)),
                    "qa_pass": bool(smoke_qa.get("pass", False)),
                    "runner_sha256": recorded_sha,
                    "maximum_reconstructed_physical_residual": (
                        smoke_qa.get(
                            "maximum_reconstructed_physical_residual"
                        )
                    ),
                    "hydro_contract_violation_mwh": hydro_violation_mwh,
                    "hydro_contract_violation_share_of_hydro_output": (
                        hydro_violation_share
                    ),
                    "released_reservoir_bound_conflicts_quarantined": int(
                        hydro_audit.get(
                            "released_reservoir_bound_conflicts_quarantined",
                            0,
                        )
                    ),
                }
            )
    matching_released_smokes = [
        row
        for row in released_smokes
        if row["runner_sha256"] == current_runner_sha
        and row["qa_pass"]
        and row["hours"] >= 168
    ]
    details["released_input_network_smokes"] = released_smokes
    audit.gate(
        "released_input_168h_network_integration_smoke",
        "task",
        "pass" if matching_released_smokes else "fail",
        True,
        (
            f"Matching frozen-SHA released-input smokes: "
            f"{matching_released_smokes}."
            if matching_released_smokes
            else f"No persisted ≥168 h, QA-pass real-input smoke matches "
            f"runner SHA-256 {current_runner_sha}; discovered={released_smokes}."
        ),
        "NTC screen",
    )
    matching_hydro_slack = max(
        (
            float(
                row[
                    "hydro_contract_violation_share_of_hydro_output"
                ]
            )
            for row in matching_released_smokes
            if math.isfinite(
                float(
                    row[
                        "hydro_contract_violation_share_of_hydro_output"
                    ]
                )
            )
        ),
        default=math.nan,
    )
    matching_hydro_conflicts = max(
        (
            int(row["released_reservoir_bound_conflicts_quarantined"])
            for row in matching_released_smokes
        ),
        default=0,
    )
    hydro_release_exact = bool(
        matching_released_smokes
        and math.isfinite(matching_hydro_slack)
        and matching_hydro_slack <= 1e-6
        and matching_hydro_conflicts == 0
    )
    audit.gate(
        "released_hydro_contract_exact_or_sensitivity_closed",
        "suite",
        "pass" if hydro_release_exact else "pending",
        True,
        f"Matching real-input smoke maximum hydro soft-slack share="
        f"{matching_hydro_slack}; quarantined released reservoir-bound "
        f"conflicts={matching_hydro_conflicts}. Full-year penalty, "
        "zero-slack/relaxed-bound and conflict-treatment sensitivities are "
        "required if either is non-zero.",
        "confirmatory paired suite",
    )
    if matching_released_smokes and not hydro_release_exact:
        audit.finding(
            "F_HYDRO_RELEASE_CONTRACT",
            "major",
            "仅 suite 层阻断",
            "Released hydro constraints are not yet an exact confirmatory contract",
            f"The final 168 h real-input smoke uses a soft-slack share of "
            f"{matching_hydro_slack:.4%} of hydro output and quarantines "
            f"{matching_hydro_conflicts} released reservoir-bound conflicts.",
            "Before manuscript admission, complete the full-year penalty and "
            "constraint-treatment sensitivity and show that paired storage "
            "effects are stable; do not label the current screen hydro-exact.",
        )

    storage_cache_ok = (
        "storage_physics" in source
        or "storage_contract" in source
        or "storage_parameter_sha256" in source
    )
    audit.gate(
        "elcc_cache_storage_physics_hash",
        "task",
        "pass" if storage_cache_ok else "fail",
        True,
        "ELCC cache must change when duration, efficiency, SOC, self-discharge "
        "or variable/degradation costs change under the same config_id.",
        "NTC screen",
    )

    lole_ok = (
        "np.count_nonzero(system_unserved > REPORT_TOL)" in source
        and "expected_lole" in source
    )
    eens_ok = "expected_eens = float(unserved.X.sum())" in source
    audit.gate(
        "eens_lole_task_definitions",
        "task",
        "pass" if lole_ok and eens_ok else "fail",
        True,
        "EENS sums hourly unserved MWh and LOLE counts a system hour once if "
        "any node is short.",
        "NTC screen",
    )

    admission_false = (
        'qa["manuscript_evidence_admitted"] = False' in source
        and '"model_scope": "networked_NTC_screen"' in source
    )
    audit.gate(
        "single_task_release_fails_closed",
        "task",
        "pass" if admission_false else "fail",
        True,
        "A single NTC task cannot self-promote to manuscript evidence.",
        "NTC screen",
    )

    suite_tokens = {
        "flow_based": "official_CORE_and_Nordic_flow_based_domains_modelled",
        "sequential_outages": "fifteen_sequential_outage_traces_completed",
        "ie_suite": "IE_M0_M1_M3_suite_compared",
        "paired_aggregation": "paired_chronology_weather_configuration_suite_aggregated",
        "dsr_reconciliation": "detailed_DSR_high_level_crosswalk_suite_reconciled",
    }
    for short, token in suite_tokens.items():
        explicit_pending = (
            f'"{token}": False' in source
            or f"'{token}': False" in source
        )
        audit.gate(
            f"suite_{short}",
            "suite",
            "pending" if explicit_pending else "fail",
            True,
            f"Per-task runner leaves {token} "
            + ("explicitly false." if explicit_pending else "unresolved."),
            "manuscript confirmation",
        )

    site_mask = bool(
        re.search(
            r"[\"']technology_specific_site_masks_applied_to_key_boundaries"
            r"[\"']\s*:\s*True",
            source,
        )
        and re.search(
            r"(site_mask_file|technology_site_mask|deployability_mask)",
            source,
            re.I,
        )
    )
    audit.gate(
        "technology_specific_site_masks",
        "suite",
        "pass" if site_mask else "pending",
        True,
        "CAES and cavern-hydrogen deployment must be restricted by "
        "technology-specific eligible-zone masks in key boundary validations.",
        "deployability claim",
    )

    inner_location = (
        "candidate_location_fixed_ex_ante" not in source
        and "candidate_discharge_power" in source
    )
    audit.gate(
        "candidate_location_endogenous_in_confirmatory_validation",
        "suite",
        "pass" if inner_location else "pending",
        True,
        "The NTC screen fixes load-proportional candidate location ex ante; "
        "key flow-based boundary configurations require endogenous location "
        "inside site masks.",
        "confirmatory boundary",
    )
    single_weather_task = (
        "--weather-scenario" in source
        and "args.weather_scenario" in source
        and "scenario_probabilities" not in inspect.getsource(module.solve_model)
    )
    audit.gate(
        "shared_first_stage_across_weather_outage_and_chronology",
        "suite",
        "pending" if single_weather_task else "pass",
        True,
        "The NTC runner solves one weather-scenario/chronology task at a time. "
        "Confirmatory validation must share firm capacity and candidate "
        "Pdis/Pch/E across weather scenarios, outage traces and paired "
        "chronologies, or explicitly freeze a pre-registered design before "
        "transferring it to the paired endpoint.",
        "two-stage confirmatory model",
    )
    if single_weather_task:
        audit.finding(
            "F_SHARED_FIRST_STAGE",
            "major",
            "仅 suite 层阻断",
            "Independent weather/chronology optima are not a two-stage shared design",
            "Firm additions and candidate charging-interface power are optimized "
            "inside each single-WS/single-chronology solve.",
            "A key-boundary multi-scenario model must share first-stage variables; "
            "otherwise aggregation embeds perfect-weather information and paired "
            "differences mix chronology with design re-optimization.",
        )

    sequential_claim_safe = (
        "expected_available_fraction" in source
        and "fifteen_sequential_outage_traces_completed" in source
        and '"fifteen_sequential_outage_traces_completed": False' in source
    )
    audit.gate(
        "expected_availability_not_mislabelled_sequential",
        "task",
        "pass" if sequential_claim_safe else "fail",
        True,
        "The task uses calendar-fixed expected daily availability and leaves "
        "the 15 sequential-outage gate false.",
        "NTC screen",
    )
    calendar_controls_ok = contains_all(
        source,
        (
            "Maintenance/outage expectation is fixed on the target calendar",
            "network_ub, zone_limits = load_network",
            "ntc_hourly.csv.gz",
        ),
    )
    audit.gate(
        "maintenance_and_ntc_calendar_fixed_across_pair",
        "task",
        "pass" if calendar_controls_ok else "fail",
        True,
        "Expected maintenance/FOR availability and released NTC/network "
        "limits remain on target calendar hours rather than moving with "
        "weather-day blocks.",
        "paired NTC screen",
    )
    audit.gate(
        "full_8760_hour_ntc_task_executed",
        "task",
        "pending",
        False,
        "This independent runner audit uses a 168-hour synthetic system; it "
        "does not claim that any 8,760-hour weather/configuration task has run.",
        "manuscript suite",
    )
    audit.gate(
        "full_year_elcc_completed",
        "task",
        "pending",
        False,
        "ELCC is smoke-tested on the synthetic system only. Full-year EENS/LOLE "
        "brackets must be complete for every admitted configuration.",
        "manuscript suite",
    )

    return details


def make_synthetic_system(module, hours: int):
    if hours <= 0 or hours % 24:
        raise ValueError("synthetic horizon must be a positive whole-day")
    zones = ("AA00", "BB00")
    z_count = len(zones)
    hour = np.arange(hours)
    daily_hour = hour % 24
    gross = np.vstack(
        [
            13.0
            + 8.0 * ((daily_hour >= 18) & (daily_hour <= 21))
            + 0.6 * np.sin(2 * np.pi * hour / 168),
            11.0
            + 6.0 * ((daily_hour >= 17) & (daily_hour <= 22))
            + 0.4 * np.cos(2 * np.pi * hour / 168),
        ]
    )
    wind = np.vstack(
        [
            2.0 + 0.8 * np.cos(2 * np.pi * hour / 48),
            2.5 + 0.6 * np.sin(2 * np.pi * hour / 72),
        ]
    )
    solar_one = np.maximum(0.0, 8.0 * np.sin(np.pi * (daily_hour - 6) / 12))
    solar = np.vstack([solar_one, 0.75 * solar_one])
    zeros = np.zeros_like(gross)
    ev = np.zeros_like(gross)
    ev[0, :] = 0.4
    hp = np.zeros_like(gross)
    load = np.maximum(gross - wind - solar, 0.0)

    thermal = pd.DataFrame(
        {
            "zone": zones,
            "technology": ("gas_ocgt", "gas_ccgt"),
            "capacity_mw": (12.0, 12.0),
            "variable_cost_eur_mwh": (120.0, 90.0),
        }
    )
    thermal_ub = np.full((2, hours), 12.0)
    edges = pd.DataFrame(
        {
            "edge_id": ("AA00_BB00", "BB00_AA00"),
            "from_zone": ("AA00", "BB00"),
            "to_zone": ("BB00", "AA00"),
        }
    )
    batteries = pd.DataFrame(
        [
            {
                "zone": "BB00",
                "charge_power_mw": 0.5,
                "discharge_power_mw": 0.5,
                "energy_capacity_mwh": 2.0,
                "charge_efficiency": 0.95,
                "discharge_efficiency": 0.95,
                "initial_soc_fraction": 0.5,
                "terminal_soc_fraction": 0.5,
            }
        ]
    )
    empty_contract: list[dict[str, Any]] = []
    psp = pd.DataFrame(
        [
            {
                "zone": "AA00",
                "hydro_code": "HCL",
                "energy_capacity_mwh": 4.0,
                "charge_efficiency": 0.9,
                "discharge_efficiency": 0.9,
                "initial_soc_fraction": 0.5,
                "terminal_soc_fraction": 0.5,
                "inflow_mwh": np.zeros(hours),
                "charge_ub_mw": np.full(hours, 0.5),
                "charge_lb_mw": np.zeros(hours),
                "discharge_ub_mw": np.full(hours, 0.5),
                "discharge_lb_mw": np.zeros(hours),
                "period_constraints": empty_contract,
            }
        ]
    )
    ror = pd.DataFrame(
        [
            {
                "zone": "AA00",
                "hydro_code": "HRR",
                "inflow_mwh": np.full(hours, 0.15),
                "discharge_ub_mw": np.full(hours, 0.15),
                "discharge_lb_mw": np.zeros(hours),
                "charge_ub_mw": np.full(hours, np.nan),
                "charge_lb_mw": np.zeros(hours),
                "period_constraints": empty_contract,
            }
        ]
    )
    reservoir = pd.DataFrame(
        [
            {
                "zone": "BB00",
                "hydro_code": "HRI",
                "energy_capacity_mwh": 12.0,
                "initial_soc_fraction": 0.5,
                "terminal_soc_fraction": 0.5,
                "inflow_mwh": np.full(hours, 0.05),
                "discharge_ub_mw": np.full(hours, 0.4),
                "discharge_lb_mw": np.zeros(hours),
                "charge_ub_mw": np.full(hours, np.nan),
                "charge_lb_mw": np.zeros(hours),
                "period_constraints": empty_contract,
            }
        ]
    )
    dsr = pd.DataFrame(
        [
            {
                "zone": "AA00",
                "capacity_mw": 0.25,
                "activation_price_eur_mwh": 500.0,
                "max_hours_per_day": 1.0,
            }
        ]
    )
    must_run = pd.DataFrame(
        [
            {
                "zone": "AA00",
                "broad_fuel": "Gas",
                "ratio": 0.05,
                "source_derated_capacity_mw": 12.0,
                "source_must_run_floor_mw": 0.6,
                "thermal_indices": np.array([0], dtype=int),
                "post_eva_nameplate_capacity_mw": 12.0,
                "annual_available_capacity_mwh": float(12.0 * hours),
                "annual_must_run_floor_mwh": float(0.05 * 12.0 * hours),
                "standalone_fixed_injection": False,
            }
        ]
    )
    schedules = pd.DataFrame(
        [
            {
                "zone": "AA00",
                "resource_type": "EV",
                "utc_offset_h": 0,
                "start_hours_utc": (7, 13, 19),
                "durations_h": (6, 6, 12),
            }
        ]
    )
    additional = pd.DataFrame(
        [
            {
                "zone": "AA00",
                "resource_type": "electrolyser_optional_buyer",
                "availability_lower_bound_mw": 0.0,
                "availability_upper_bound_mw": 0.1,
                "activation_price_eur_mwh": 30.0,
            }
        ]
    )
    payload = {
        "zones": zones,
        "demand_zones": frozenset(zones),
        "hours": hours,
        "load": load,
        "gross_load": gross,
        "wind": wind,
        "solar": solar,
        "btm_pv": zeros.copy(),
        "btm_battery_reference": zeros.copy(),
        "csp_storage_reference": zeros.copy(),
        "idsr_ev_baseline": ev,
        "idsr_hp_baseline": hp,
        "idsr_btm_reference": zeros.copy(),
        "idsr_schedules": schedules,
        "idsr_btm_batteries": pd.DataFrame(
            columns=(
                "zone",
                "price_sensitive_charge_power_mw",
                "price_sensitive_discharge_power_mw",
                "price_sensitive_energy_capacity_mwh",
                "round_trip_efficiency",
                "initial_soc_fraction",
                "terminal_soc_fraction",
            )
        ),
        "additional_consumers": additional,
        "ror": ror,
        "thermal": thermal,
        "thermal_ub": thermal_ub,
        "firm_availability": np.ones((z_count, hours)),
        "firm_variable_cost": np.full(z_count, 150.0),
        "firm_annual_cost_eur_kw": 120.0,
        "edges": edges,
        "network_ub": np.full((2, hours), 4.0),
        "zone_limits": pd.DataFrame(
            columns=("zone", "curve_type", "hour_of_year", "limit_mw")
        ),
        "batteries": batteries,
        "pumped_hydro": psp,
        "dsr": dsr,
        "must_run": must_run,
        "reservoir": reservoir,
        "resource_audit": {
            "must_run": {"case": "on"},
            "hydro": {"chronology_treatment": "calendar_fixed_common_control"},
            "idsr": {},
        },
    }
    fields = set(module.SystemData.__annotations__)
    missing = sorted(fields.difference(payload))
    extra = sorted(set(payload).difference(fields))
    if missing:
        raise ValueError(f"synthetic factory missing SystemData fields: {missing}")
    for key in extra:
        payload.pop(key)
    return module.SystemData(**payload)


def synthetic_audit(
    audit: Audit,
    module,
    hours: int,
) -> list[dict[str, Any]]:
    core = load_module("eraa_core_for_independent_audit", module.CORE_PATH)
    storage = core.StorageTechnology(
        config_id="audit_caes_8h",
        tech_id="caes",
        duration_h=8.0,
        round_trip_efficiency=0.64,
        self_discharge_pct_day=0.01,
        soc_min_fraction=0.05,
        soc_max_fraction=0.95,
        reference_charge_to_discharge_power_ratio=1.5,
        maximum_charge_to_discharge_power_ratio=4.0,
        annual_discharge_power_cost_eur_kw=2.0,
        annual_charge_power_cost_eur_kw=1.5,
        annual_energy_cost_eur_kwh=1.0,
        variable_om_eur_mwh=1.0,
        degradation_cost_eur_mwh_discharged=0.0,
        main_evidence_domain=True,
        structural_extrapolation=False,
    )
    storage.validate()
    system = make_synthetic_system(module, hours)
    args = parse_default_args(module)
    overrides = {
        "quiet": True,
        "threads": 1,
        "method": 1,
        "time_limit": 0.0,
        "log_file": None,
        "feasibility_tolerance": 1e-8,
        "optimality_tolerance": 1e-8,
        "maximum_firm_addition_mw": 100.0,
        "value_of_lost_load": 15_000.0,
        "hydro_variable_cost": 5.0,
        "hydro_constraint_violation_penalty": 10_000.0,
        "hydro_spill_penalty": 1e-7,
        "vre_spill_penalty": 1e-6,
        "flow_regularization": 1e-8,
        "existing_storage_throughput_regularization": 1e-4,
        "existing_battery_degradation_cost_eur_mwh_discharged": 1.0,
        "keep_candidate_dispatch": True,
        "power_to_heat_availability_case": "zero",
        "simultaneous_dispatch_tolerance_mwh": 1e-3,
    }
    for key, value in overrides.items():
        setattr(args, key, value)
    candidate = np.array([2.0, 0.0])
    baseline = module.solve_model(
        system,
        storage,
        np.zeros(2),
        "baseline",
        args,
        None,
        "independent-audit-baseline",
    )
    baseline_firm = (
        baseline.first_stage.set_index("zone")
        .reindex(system.zones)["firm_addition_mw"]
        .to_numpy(dtype=float)
    )
    joint = module.solve_model(
        system,
        storage,
        candidate,
        "joint",
        args,
        None,
        "independent-audit-baseline",
    )
    repeat = module.solve_model(
        system,
        storage,
        candidate,
        "repeat_use_fixed_firm",
        args,
        baseline_firm,
        "independent-audit-baseline",
    )
    solved = (baseline, joint, repeat)
    rows: list[dict[str, Any]] = []
    for result in solved:
        row = dict(result.metrics)
        row["synthetic_hours"] = hours
        rows.append(row)

    objective_ok = all(
        float(item.metrics["objective_component_error_eur"])
        <= max(
            1e-3,
            1e-8 * abs(float(item.metrics["objective_eur_year"])),
        )
        for item in solved
    )
    audit.gate(
        "synthetic_objective_reconstruction",
        "task",
        "pass" if objective_ok else "fail",
        True,
        "Baseline, joint and fixed-firm repeat-use solves reconstruct ObjVal.",
        "NTC screen",
    )

    fixed_cost_expected = (
        candidate.sum()
        * storage.annual_discharge_power_cost_eur_kw
        * 1_000.0
        + candidate.sum()
        * storage.duration_h
        * storage.annual_energy_cost_eur_kwh
        * 1_000.0
    )
    fixed_cost_observed = float(
        joint.metrics["candidate_discharge_power_cost_eur_year"]
        + joint.metrics["candidate_energy_capacity_cost_eur_year"]
    )
    fixed_ok = np.isclose(fixed_cost_observed, fixed_cost_expected, atol=1e-6)
    audit.gate(
        "synthetic_candidate_fixed_cost",
        "task",
        "pass" if fixed_ok else "fail",
        True,
        f"Expected={fixed_cost_expected:.9f}, "
        f"observed={fixed_cost_observed:.9f} EUR/yr.",
        "NTC screen",
    )

    discharge = float(joint.metrics["candidate_discharge_mwh"])
    rated = float(candidate.sum() * storage.duration_h)
    efc_expected = discharge / rated
    efc_observed = float(joint.metrics["equivalent_full_cycles_output_basis"])
    efc_ok = np.isclose(efc_observed, efc_expected, atol=1e-10)
    audit.gate(
        "synthetic_canonical_efc",
        "task",
        "pass" if efc_ok else "fail",
        True,
        f"Delivered={discharge:.9f} MWh, rated={rated:.9f} MWh, "
        f"EFC={efc_observed:.12g}.",
        "NTC screen",
    )

    dispatch = joint.candidate_dispatch
    if dispatch.empty:
        reliability_ok = False
        reliability_evidence = "Candidate dispatch was not retained."
    else:
        reconstructed_eens = float(dispatch["system_unserved_mw"].sum())
        reconstructed_lole = float(
            np.count_nonzero(
                dispatch["system_unserved_mw"].to_numpy()
                > module.REPORT_TOL
            )
        )
        reliability_ok = bool(
            np.isclose(
                reconstructed_eens,
                joint.metrics["expected_eens_mwh"],
                atol=1e-8,
            )
            and np.isclose(
                reconstructed_lole,
                joint.metrics["expected_lole_h"],
                atol=1e-8,
            )
        )
        reliability_evidence = (
            f"EENS={reconstructed_eens:.12g} MWh; "
            f"LOLE={reconstructed_lole:.12g} h."
        )
    audit.gate(
        "synthetic_eens_lole_reconstruction",
        "task",
        "pass" if reliability_ok else "fail",
        True,
        reliability_evidence,
        "NTC screen",
    )

    residual_columns = [
        key
        for key in joint.metrics
        if (
            key.startswith("maximum_")
            and ("residual" in key or "violation" in key)
            or key
            in {
                "idsr_energy_conservation_residual_mwh",
                "hydro_contract_violation_mwh",
                "solver_maximum_constraint_violation",
                "solver_maximum_bound_violation",
            }
        )
        and isinstance(joint.metrics[key], (int, float, np.number))
    ]
    residual_values = {
        key: float(joint.metrics[key]) for key in residual_columns
    }
    residual_ok = bool(
        residual_values
        and all(
            math.isfinite(value)
            and abs(value) <= 1e-4
            for key, value in residual_values.items()
            if "objective" not in key
            and "hydro_contract_violation" not in key
        )
        and float(
            joint.metrics.get("hydro_contract_violation_mwh", 0.0)
        )
        <= 1e-4
    )
    audit.gate(
        "synthetic_physical_residuals",
        "task",
        "pass" if residual_ok else "fail",
        True,
        json.dumps(json_ready(residual_values), ensure_ascii=False),
        "NTC screen",
    )

    overlap_checks = {
        str(item.metrics["variant"]): {
            "absolute_mwh": float(
                item.metrics["maximum_simultaneous_charge_discharge_mwh"]
            ),
            "throughput_ratio": float(
                item.metrics["maximum_simultaneous_charge_discharge_ratio"]
            ),
        }
        for item in solved
    }
    simultaneous_ok = all(
        row["absolute_mwh"] <= args.simultaneous_dispatch_tolerance_mwh
        or row["throughput_ratio"]
        <= args.simultaneous_dispatch_relative_tolerance
        for row in overlap_checks.values()
    )
    audit.gate(
        "synthetic_same_hour_charge_discharge",
        "task",
        "pass" if simultaneous_ok else "fail",
        True,
        json.dumps(json_ready(overlap_checks), ensure_ascii=False),
        "NTC screen",
    )

    repeat_first = (
        repeat.first_stage.set_index("zone")
        .reindex(system.zones)["candidate_charge_power_mw"]
        .to_numpy(dtype=float)
    )
    with tempfile.TemporaryDirectory(prefix="eraa_elcc_audit_") as temporary:
        args.cache_root = Path(temporary)
        args.elcc_tolerance_mw = 0.25
        args.elcc_max_iterations = 12
        args.elcc_max_upper_multiplier = 4.0
        args.elcc_reliability_relative_tolerance = 1e-8
        args.elcc_eens_absolute_tolerance_mwh = 1e-6
        args.elcc_lole_tolerance_h = 1e-8
        elcc = module.calculate_elcc(
            system=system,
            storage=storage,
            candidate_pdis=candidate,
            candidate_pcharge=repeat_first,
            baseline_firm=baseline_firm,
            baseline_metrics=baseline.metrics,
            baseline_cache_key="independent-audit-baseline",
            args=args,
        )
    elcc_complete = bool(elcc.get("elcc_complete", False))
    elcc_bounds_ok = bool(
        elcc_complete
        and math.isfinite(float(elcc["elcc_lower_mw"]))
        and math.isfinite(float(elcc["elcc_upper_mw"]))
        and 0
        <= float(elcc["elcc_lower_mw"])
        <= float(elcc["elcc_upper_mw"])
    )
    audit.gate(
        "synthetic_elcc_bisection",
        "task",
        "pass" if elcc_bounds_ok else "fail",
        True,
        json.dumps(json_ready(elcc), ensure_ascii=False),
        "NTC screen",
    )
    for row in rows:
        row.update({f"synthetic_{key}": value for key, value in elcc.items()})
    return rows


def write_memo(
    path: Path,
    runner_sha: str,
    audit: Audit,
    qa: dict[str, Any],
) -> None:
    gates = pd.DataFrame(audit.gates)
    task_failed = gates.loc[
        gates["scope"].isin(["task", "cross_implementation"])
        & gates["critical"]
        & gates["status"].isin(["fail", "not_run"])
    ]
    suite_pending = gates.loc[
        gates["scope"].eq("suite")
        & gates["status"].isin(["fail", "pending", "not_run"])
    ]
    passed = gates.loc[gates["status"].eq("pass")]
    lines = [
        "# ERAA 2035 full-network runner 独立审计",
        "",
        f"- 审计时间（UTC）: `{qa['created_utc']}`",
        f"- runner SHA-256: `{runner_sha}`",
        f"- runner task ready: `{qa['runner_task_ready']}`",
        f"- cross-implementation ready: `{qa['cross_implementation_ready']}`",
        f"- manuscript evidence ready: `{qa['manuscript_evidence_ready']}`",
        "",
        "## 已修复",
        "",
    ]
    if passed.empty:
        lines.append("- 无。")
    else:
        for row in passed.itertuples(index=False):
            lines.append(f"- `{row.gate_id}`：{row.evidence}")
    lines.extend(["", "## 仍阻断", ""])
    if task_failed.empty:
        lines.append("- 当前独立审计未发现 task/cross-implementation 硬失败。")
    else:
        for row in task_failed.itertuples(index=False):
            lines.append(f"- `{row.gate_id}`：{row.evidence}")
    lines.extend(["", "## 仅 suite 层阻断", ""])
    if suite_pending.empty:
        lines.append("- 无。")
    else:
        for row in suite_pending.itertuples(index=False):
            lines.append(f"- `{row.gate_id}`：{row.evidence}")
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "该 runner 只能生成 `networked_NTC_screen`。即使单个 8,760 h "
            "任务全部通过，也必须由独立 suite aggregator 完成 36 个天气情景的"
            "成对汇总、关键边界配置的 CORE/Nordic flow-based 验证、15 条顺序"
            "停运路径、IE M0/M1/M3、DSR 口径敏感性和技术场址掩膜后，才可进入"
            "稿件证据。任何一个单任务 QA 都不得自行把 "
            "`manuscript_evidence_admitted` 置为 true。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--synthetic-hours", type=int, default=168)
    parser.add_argument("--skip-synthetic", action="store_true")
    args = parser.parse_args()

    runner = args.runner.resolve()
    artifacts = args.artifacts.resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    start_sha = sha256(runner)
    source = runner.read_text(encoding="utf-8")
    audit = Audit()
    module = load_module("eraa_full_runner_independent_audit", runner)
    details = static_and_source_audit(audit, module, source, runner)

    synthetic_rows: list[dict[str, Any]] = []
    synthetic_error = ""
    if args.skip_synthetic:
        audit.gate(
            "synthetic_168h_runner_smoke",
            "task",
            "not_run",
            True,
            "Skipped by command line.",
            "NTC screen",
        )
    else:
        try:
            synthetic_rows = synthetic_audit(
                audit, module, args.synthetic_hours
            )
            audit.gate(
                "synthetic_168h_runner_smoke",
                "task",
                "pass",
                True,
                f"Baseline, joint and repeat-use variants solved for "
                f"{args.synthetic_hours} hours.",
                "NTC screen",
            )
        except Exception:
            synthetic_error = traceback.format_exc()
            audit.gate(
                "synthetic_168h_runner_smoke",
                "task",
                "fail",
                True,
                synthetic_error.splitlines()[-1],
                "NTC screen",
            )
            audit.finding(
                "F_SYNTHETIC",
                "major",
                "仍阻断",
                "The runner did not pass the independent 168-hour smoke test",
                synthetic_error,
                "Repair or explain the failure and rerun this audit against a "
                "frozen SHA before launching the full suite.",
            )

    end_sha = sha256(runner)
    stable = start_sha == end_sha
    audit.gate(
        "runner_sha_stable_during_audit",
        "task",
        "pass" if stable else "fail",
        True,
        f"start={start_sha}; end={end_sha}",
        "NTC screen",
    )

    gates = pd.DataFrame(audit.gates)
    findings = pd.DataFrame(audit.findings)
    if findings.empty:
        findings = pd.DataFrame(
            columns=(
                "finding_id",
                "severity",
                "classification",
                "title",
                "evidence",
                "implication",
            )
        )
    task_ready = bool(
        not gates.loc[
            gates["scope"].eq("task")
            & gates["critical"]
            & ~gates["status"].eq("pass")
        ].shape[0]
    )
    cross_ready = bool(
        not gates.loc[
            gates["scope"].eq("cross_implementation")
            & gates["critical"]
            & ~gates["status"].eq("pass")
        ].shape[0]
    )
    suite_ready = bool(
        not gates.loc[
            gates["scope"].eq("suite")
            & gates["critical"]
            & ~gates["status"].eq("pass")
        ].shape[0]
    )
    qa = {
        "created_utc": utc_now(),
        "runner_path": str(runner),
        "runner_sha256_start": start_sha,
        "runner_sha256_end": end_sha,
        "runner_sha_stable": stable,
        "runner_task_ready": task_ready,
        "cross_implementation_ready": cross_ready,
        "suite_ready": suite_ready,
        "manuscript_evidence_ready": bool(
            task_ready and cross_ready and suite_ready
        ),
        "manuscript_evidence_admitted": False,
        "synthetic_error": synthetic_error or None,
        "details": details,
        "gate_counts": gates.groupby(["scope", "status"]).size().to_dict(),
    }
    gates.to_csv(
        artifacts / "audit_eraa_full_network_runner_gates.csv", index=False
    )
    findings.to_csv(
        artifacts / "audit_eraa_full_network_runner_findings.csv", index=False
    )
    pd.DataFrame(synthetic_rows).to_csv(
        artifacts / "audit_eraa_full_network_synthetic_results.csv",
        index=False,
    )
    atomic_json(
        artifacts / "audit_eraa_full_network_runner_qa.json",
        qa,
    )
    write_memo(
        artifacts / "ERAA_FULL_NETWORK_RUNNER_INDEPENDENT_AUDIT.md",
        end_sha,
        audit,
        qa,
    )
    print(json.dumps(json_ready(qa), ensure_ascii=False, indent=2))
    if not stable:
        raise SystemExit("runner changed during audit")


if __name__ == "__main__":
    main()
