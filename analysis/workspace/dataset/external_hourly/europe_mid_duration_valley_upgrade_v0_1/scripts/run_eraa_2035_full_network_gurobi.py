#!/usr/bin/env python3
"""Run the ERAA 2035 paired-chronology model on the complete NTC graph.

This is the confirmatory transport-model backend.  It retains all 75 nodes in
the released NTC graph, including zero-load offshore/resource nodes, while the
52 demand nodes carry the synchronized intervention.  It also represents the
released post-EVA thermal fleet, in-market utility batteries, usable pumped
hydro, explicit DSR activation bands and physical interconnection limits.

Candidate storage is a fixed signed tranche.  Its discharge power is allocated
ex ante in proportion to annual mean load over EU-27 nodes (primary) or all
European demand nodes (sensitivity); the optimizer cannot relocate it between
weather scenarios or paired chronologies.  Charging-interface power above the
source-catalogue reference ratio is endogenous and explicitly repriced.

The Gurobi backend is required for full 8,760-hour runs.  A short, whole-day
development run is allowed for integrity testing but is labelled non-evidential.
The no-candidate baseline is cached once for each physical
weather-scenario/chronology/fleet/network task and reused across configurations.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCRIPT_VERSION = "0.2.5"
RESERVOIR_ENDPOINT_CONTRACT_VERSION = "ERAA_2025_Annex2_6.1_v0.2.5"
TARGET_YEAR = 2035
HOURS_PER_YEAR = 8_760
ERAA_RUNNER_DATA_ROLE = (
    "forward_looking_ERAA_2035_system_mechanism_and_reliability_screen"
)
ERAA_FOSSIL_GAS_DATA_ROLE = (
    "endogenous_simulated_ERAA_2035_fossil_gas_generation"
)
HISTORICAL_FOSSIL_GAS_TASK_ROLE = (
    "historical_observed_hourly_aggregate_Fossil_gas_empirical_peaking_task"
)
ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = ROOT / "processed" / "eraa_2035_phase1_inputs_v0_1"
DEFAULT_CHRONOLOGY = (
    ROOT / "processed" / "eraa_2035_synchronized_chronologies_v0_1"
)
DEFAULT_DISPATCH = ROOT / "processed" / "eraa_2035_dispatch_parameters_v0_1"
DEFAULT_FLEXIBLE = ROOT / "processed" / "eraa_2035_flexible_resources_v0_1"
DEFAULT_IDSR = (
    ROOT / "processed" / "eraa_2035_idsr_and_dispatchable_load_v0_1"
)
DEFAULT_MUST_RUN = (
    ROOT / "raw" / "eraa_2025" / "extracted" / "Other Data" / "Must-run_data.csv"
)
DEFAULT_RESERVOIR_LEVELS = (
    ROOT
    / "raw"
    / "eraa_2025"
    / "extracted"
    / "Other Data"
    / "Reservoir Levels.csv"
)
DEFAULT_CATALOG = (
    ROOT
    / "model_adapter"
    / "energy_charts_2015_2024_representative_pilot"
    / "configuration_catalog_53.csv"
)
DEFAULT_TECH_LIBRARY = (
    REPOSITORY_ROOT / "dataset" / "M2_LDES_technology_parameter_library_v0_2.csv"
)
DEFAULT_CACHE = ROOT / "processed" / "eraa_2035_full_network_baseline_cache_v0_1"
DEFAULT_OUTPUT = ROOT / "model_results" / "eraa_2035_full_network_v0_1"
CORE_PATH = Path(__file__).with_name("eraa_confirmatory_model_core.py")
ZONE_RUNNER_PATH = Path(__file__).with_name("run_eraa_2035_phase1_zone_task.py")

EXCLUDED_AGGREGATE_ZONES = {"DK00", "ISEM", "IT00", "LU00", "NO00", "SE00"}
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
OTHER_RES_TECHNOLOGIES = (
    "Small biomass",
    "Waste",
    "Geothermal",
    "Marine",
)
# Stable machine-readable identifiers used in the source-closure audit.
OTHER_RES_MODEL_KEYS = (
    "small_biomass",
    "waste",
    "geothermal",
    "marine",
)
HYDRO_LABEL_TO_CODE = {
    "Closed loop pumping": "HCL",
    "Open loop pumping": "HOL",
    "Pondage": "HPI",
    "Reservoir": "HRI",
    "Run of river": "HRR",
}
EU27 = {
    "AT",
    "BE",
    "BG",
    "HR",
    "CY",
    "CZ",
    "DK",
    "EE",
    "FI",
    "FR",
    "DE",
    "GR",
    "HU",
    "IE",
    "IT",
    "LV",
    "LT",
    "LU",
    "MT",
    "NL",
    "PL",
    "PT",
    "RO",
    "SK",
    "SI",
    "ES",
    "SE",
}
REPORT_TOL = 1e-6
EXPECTED_NETWORK_ONLY_ZERO_NODES = {
    "DKBH_OFF",
    "DKN2_OFF",
    "ITCO",
    "ITVI",
    "LTH1_OFF",
    "NL0F_OFF",
    "PLE0",
    "PLI0",
    "TR00",
}
EXPECTED_PSP_EXCLUDED_MISSING_ENERGY = {
    "AT00:HCL",
    "BA00:HOL",
    "CY00:HCL",
    "CZ00:HCL",
    "CZ00:HOL",
    "GR00:HOL",
    "IE00:HCL",
    "ITSI:HCL",
    "ITSI:HOL",
    "PL00:HOL",
    "SI00:HOL",
    "SK00:HCL",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_payload(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def json_ready(value: Any) -> Any:
    """Convert NumPy scalars and non-finite floats to strict JSON values."""

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
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return value


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(
            json_ready(payload),
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")
    os.replace(temporary, path)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as raw:
        temporary = Path(raw.name)
    if path.suffix == ".gz":
        with gzip.open(
            temporary,
            mode="wt",
            encoding="utf-8",
            newline="",
            compresslevel=6,
        ) as handle:
            frame.to_csv(
                handle,
                index=False,
                lineterminator="\n",
                float_format="%.10g",
            )
    else:
        with temporary.open(
            mode="w", encoding="utf-8", newline=""
        ) as handle:
            frame.to_csv(
                handle,
                index=False,
                lineterminator="\n",
                float_format="%.10g",
            )
    os.replace(temporary, path)


def prepare_output(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"output exists: {path}; pass --force")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def read_qa(
    path: Path,
    label: str,
    eligible_keys: tuple[str, ...] = ("pass",),
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{label} QA absent: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    admitted = [
        key for key in eligible_keys if bool(payload.get(key, False))
    ]
    if not admitted:
        raise ValueError(
            f"{label} QA has none of the eligible true gates "
            f"{eligible_keys}: {path}"
        )
    payload["_accepted_gate"] = admitted[0]
    return payload


def reorder_days(values: np.ndarray, order: list[int]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.shape != (HOURS_PER_YEAR,):
        raise ValueError(f"expected 8760 values, observed {values.shape}")
    if sorted(order) != list(range(365)):
        raise ValueError("day order is not a permutation of 0..364")
    return values.reshape(365, 24)[np.asarray(order)].reshape(-1)


def cyclic_window_ids(
    hours: int,
    start_hours_utc: Iterable[int],
    durations_h: Iterable[int],
) -> np.ndarray:
    """Map hours to cyclic daily flexibility windows.

    Windows that cross the annual (or development-slice) boundary receive one
    common identifier.  This is the direct compact equivalent of the ERAA
    Annex 2 stock boundary condition: flexible energy cannot leave its
    prescribed window, including the overnight EV window.
    """

    starts = tuple(int(value) for value in start_hours_utc)
    durations = tuple(int(value) for value in durations_h)
    if hours <= 0 or hours % 24:
        raise ValueError("flexibility windows require a positive whole-day horizon")
    if (
        not starts
        or len(starts) != len(durations)
        or any(value < 0 or value >= 24 for value in starts)
        or any(value <= 0 for value in durations)
        or sum(durations) != 24
    ):
        raise ValueError(
            "daily flexibility-window starts/durations are incomplete"
        )
    if len(set(starts)) != len(starts):
        raise ValueError("flexibility-window start hours are duplicated")
    ordered = sorted(zip(starts, durations))
    ordered_starts = tuple(item[0] for item in ordered)
    for index, (start, duration) in enumerate(ordered):
        next_start = ordered[(index + 1) % len(ordered)][0]
        observed = (next_start - start) % 24
        if observed == 0:
            observed = 24
        if observed != duration:
            raise ValueError(
                "flexibility-window durations do not match successive starts"
            )

    # Include the preceding day's starts so the first hours can be assigned to
    # an overnight window.  Modulo-normalising its start joins that prefix to
    # the matching suffix at the end of the cyclic horizon.
    days = hours // 24
    boundaries = np.array(
        [
            day * 24 + start
            for day in range(-1, days)
            for start in ordered_starts
        ],
        dtype=int,
    )
    timeline = np.arange(hours, dtype=int)
    locations = np.searchsorted(boundaries, timeline, side="right") - 1
    if (locations < 0).any():
        raise RuntimeError("failed to assign the first flexibility window")
    starts_for_hour = np.mod(boundaries[locations], hours)
    unique = {
        value: index
        for index, value in enumerate(sorted(set(starts_for_hour.tolist())))
    }
    result = np.array(
        [unique[value] for value in starts_for_hour], dtype=np.int32
    )
    expected_windows = days * len(starts)
    if len(np.unique(result)) != expected_windows:
        raise RuntimeError("cyclic flexibility-window count is inconsistent")
    return result


def idsr_window_analytic_self_test() -> bool:
    ev = cyclic_window_ids(48, (7, 13, 19), (6, 6, 12))
    hp = cyclic_window_ids(48, (5, 11, 17, 23), (6, 6, 6, 6))
    # EV hours 0..6 and 43..47 form the one cyclic overnight window.
    return bool(
        len(np.unique(ev)) == 6
        and np.unique(np.concatenate((ev[:7], ev[43:]))).size == 1
        and len(np.unique(hp)) == 8
    )


def read_order(
    chronology_root: Path, weather_scenario: str, chronology: str
) -> tuple[list[int], str]:
    path = chronology_root / "synchronized_day_orders.csv"
    frame = pd.read_csv(path)
    selected = frame.loc[
        frame["weather_scenario"].astype(str).eq(weather_scenario)
        & frame["chronology"].astype(str).eq(chronology)
    ].sort_values("new_day")
    if len(selected) != 365:
        raise ValueError(
            f"{weather_scenario}/{chronology}: expected 365 day-order rows"
        )
    hashes = selected["order_sha256"].astype(str).unique()
    if len(hashes) != 1:
        raise ValueError("chronology has multiple order hashes")
    order = selected["source_day"].astype(int).tolist()
    return order, str(hashes[0])


def country_from_zone(zone: str) -> str:
    # Every ERAA atomic bidding/resource node starts with its ISO alpha-2
    # country code; UK/UKNI are outside EU-27.
    return str(zone)[:2]


@dataclass
class SystemData:
    zones: tuple[str, ...]
    demand_zones: frozenset[str]
    hours: int
    load: np.ndarray  # released clipped net-load diagnostic [Z,T]
    gross_load: np.ndarray  # gross demand before BTM resources [Z,T]
    wind: np.ndarray  # [Z,T]
    solar: np.ndarray  # [Z,T]
    btm_pv: np.ndarray  # released fixed BTM PV output [Z,T]
    btm_battery_reference: np.ndarray  # signed net BTM battery output [Z,T]
    csp_storage_reference: np.ndarray  # released fixed CSP-storage output [Z,T]
    idsr_ev_baseline: np.ndarray  # price-sensitive EV demand before shifting
    idsr_hp_baseline: np.ndarray  # price-sensitive HP demand before shifting
    idsr_btm_reference: np.ndarray  # signed flexible BTM reference [Z,T]
    idsr_schedules: pd.DataFrame
    idsr_btm_batteries: pd.DataFrame
    additional_consumers: pd.DataFrame
    ror: pd.DataFrame
    thermal: pd.DataFrame
    thermal_ub: np.ndarray  # [R,T]
    firm_availability: np.ndarray  # [Z,T]
    firm_variable_cost: np.ndarray  # [Z]
    firm_annual_cost_eur_kw: float
    edges: pd.DataFrame
    network_ub: np.ndarray  # [E,T]
    zone_limits: pd.DataFrame
    batteries: pd.DataFrame
    pumped_hydro: pd.DataFrame
    dsr: pd.DataFrame
    must_run: pd.DataFrame
    reservoir: pd.DataFrame
    resource_audit: dict[str, Any]


def graph_nodes(dispatch_root: Path) -> tuple[pd.DataFrame, tuple[str, ...]]:
    ntc_path = dispatch_root / "ntc_hourly.csv.gz"
    header = pd.read_csv(
        ntc_path, usecols=["edge_id", "from_zone", "to_zone"], nrows=500_000
    ).drop_duplicates()
    # nrows covers many hours but not every edge.  Read one calendar hour to
    # recover all 269 released edges without loading the full table twice.
    all_edges = pd.read_csv(
        ntc_path,
        usecols=["edge_id", "from_zone", "to_zone", "hour_of_year"],
    )
    edges = (
        all_edges.loc[all_edges["hour_of_year"].eq(0)]
        .drop(columns="hour_of_year")
        .drop_duplicates("edge_id")
        .sort_values("edge_id")
        .reset_index(drop=True)
    )
    if len(edges) != 269:
        raise ValueError(f"expected 269 NTC edges, observed {len(edges)}")
    zones = tuple(
        sorted(set(edges["from_zone"]).union(edges["to_zone"]))
    )
    if len(zones) != 75:
        raise ValueError(f"expected 75 NTC graph nodes, observed {len(zones)}")
    excluded = EXCLUDED_AGGREGATE_ZONES.intersection(zones)
    if excluded:
        raise ValueError(f"aggregate nodes entered NTC graph: {sorted(excluded)}")
    return edges, zones


def load_hourly_profiles(
    input_root: Path,
    zones: tuple[str, ...],
    weather_scenario: str,
    order: list[int],
    flexible_resource_zones: set[str],
    nl0a_treatment: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    frozenset[str],
    dict[str, Any],
]:
    z_count = len(zones)
    load = np.zeros((z_count, HOURS_PER_YEAR), dtype=float)
    gross_load = np.zeros_like(load)
    wind = np.zeros_like(load)
    solar = np.zeros_like(load)
    btm_pv = np.zeros_like(load)
    btm_battery_reference = np.zeros_like(load)
    csp_storage_reference = np.zeros_like(load)
    coverage = pd.read_csv(input_root / "zone_coverage.csv")
    coverage_by_zone = coverage.set_index("zone")
    expected_demand_zones = set(
        coverage.loc[
            coverage["expected_demand_zone"].astype(bool), "zone"
        ].astype(str)
    )
    capacity = pd.read_csv(input_root / "capacity_2035_final.csv")
    capacity_zones = set(capacity["Market_Node"].astype(str))
    hydro_capacity = pd.read_csv(input_root / "hydro_capacity_2035_final.csv")
    hydro_zones = set(hydro_capacity["MARKET_NODE"].astype(str))
    demand_zones: set[str] = set()
    direct_profile_nodes: list[str] = []
    audited_zero_nodes: list[str] = []
    source_sums: list[dict[str, Any]] = []
    required = {
        "weather_scenario",
        "hour_of_year",
        "load_mw",
        "gross_demand_mw",
        "btm_pv_mw",
        "btm_battery_net_output_mw",
        "wind_potential_mw",
        "solar_potential_mw",
        "demand_role",
    }
    optional = {
        "csp_with_storage_dispatched_reference_mw",
    }
    for z_index, zone in enumerate(zones):
        path = (
            input_root
            / "hourly_profiles"
            / f"zone={zone}"
            / "hourly_profiles.csv.gz"
        )
        if not path.exists():
            # Only a graph endpoint proven to have no demand, generation,
            # hydro or flexible resource may receive an audited zero vector.
            conflicts = {
                "expected_demand": zone in expected_demand_zones,
                "capacity": zone in capacity_zones,
                "hydro": zone in hydro_zones,
                "flexible_resource": zone in flexible_resource_zones,
                "coverage_record": zone in coverage_by_zone.index,
            }
            if any(conflicts.values()):
                raise FileNotFoundError(
                    f"missing profile for non-empty graph node {zone}: "
                    f"{conflicts}; path={path}"
                )
            audited_zero_nodes.append(zone)
            continue
        direct_profile_nodes.append(zone)
        columns = set(pd.read_csv(path, nrows=0).columns)
        missing = required.difference(columns)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        frame = pd.read_csv(
            path, usecols=sorted(required | (optional & columns))
        )
        frame = frame.loc[
            frame["weather_scenario"].astype(str).eq(weather_scenario)
        ].sort_values("hour_of_year")
        if len(frame) != HOURS_PER_YEAR or not np.array_equal(
            frame["hour_of_year"].to_numpy(dtype=int),
            np.arange(HOURS_PER_YEAR),
        ):
            raise ValueError(f"{path}/{weather_scenario}: invalid hour axis")
        role = set(frame["demand_role"].astype(str))
        if role == {"modelled_demand"}:
            demand_zones.add(zone)
        elif role != {"zero_load_resource_or_network_node"}:
            raise ValueError(f"{path}: inconsistent demand role {role}")
        load_source = frame["load_mw"].to_numpy(dtype=float)
        gross_source = frame["gross_demand_mw"].to_numpy(dtype=float)
        btm_pv_source = frame["btm_pv_mw"].to_numpy(dtype=float)
        btm_battery_source = frame[
            "btm_battery_net_output_mw"
        ].to_numpy(dtype=float)
        wind_source = frame["wind_potential_mw"].to_numpy(dtype=float)
        solar_source = frame["solar_potential_mw"].to_numpy(dtype=float)
        csp_source = np.zeros(HOURS_PER_YEAR, dtype=float)
        if "csp_with_storage_dispatched_reference_mw" in frame:
            csp_source = frame[
                "csp_with_storage_dispatched_reference_mw"
            ].to_numpy(dtype=float)
        identity = (
            gross_source - btm_pv_source - btm_battery_source
        )
        clipped = np.maximum(identity, 0.0)
        if not np.allclose(clipped, load_source, rtol=0, atol=1e-5):
            raise ValueError(
                f"{path}: gross/BTM/clipped-load identity does not reconcile"
            )
        for name, values in (
            ("load", load_source),
            ("gross_load", gross_source),
            ("btm_pv", btm_pv_source),
            ("btm_battery_reference", btm_battery_source),
            ("wind", wind_source),
            ("solar", solar_source),
            ("csp_storage_reference", csp_source),
        ):
            signed = name == "btm_battery_reference"
            if not np.isfinite(values).all() or (
                not signed and (values < 0).any()
            ):
                raise ValueError(f"{path}/{name}: invalid values")
            transformed = reorder_days(values, order)
            if not np.isclose(
                values.sum(), transformed.sum(), rtol=1e-11, atol=1e-5
            ):
                raise ValueError(f"{path}/{name}: chronology does not conserve")
            {
                "load": load,
                "gross_load": gross_load,
                "btm_pv": btm_pv,
                "btm_battery_reference": btm_battery_reference,
                "wind": wind,
                "solar": solar,
                "csp_storage_reference": csp_storage_reference,
            }[name][z_index] = transformed
            source_sums.append(
                {
                    "zone": zone,
                    "variable": name,
                    "annual_sum": float(values.sum()),
                }
            )
    # NL0A_OFF has a released 50 MW offshore-wind shard but no NTC endpoint.
    # Preserve its hourly injection exactly through one explicit, lossless
    # Dutch-node mapping.  This is not presented as an official topology
    # crosswalk and is exposed as a two-node sensitivity in the CLI.
    nl0a_path = (
        input_root
        / "hourly_profiles"
        / "zone=NL0A_OFF"
        / "hourly_profiles.csv.gz"
    )
    if not nl0a_path.exists():
        raise FileNotFoundError(nl0a_path)
    source = pd.read_csv(nl0a_path)
    source = source.loc[
        source["weather_scenario"].astype(str).eq(weather_scenario)
    ].sort_values("hour_of_year")
    if len(source) != HOURS_PER_YEAR:
        raise ValueError("NL0A_OFF source shard is incomplete")
    if source["load_mw"].abs().max() > 1e-12:
        raise ValueError("NL0A_OFF unexpectedly carries demand")
    source_wind = reorder_days(
        source["wind_potential_mw"].to_numpy(dtype=float), order
    )
    source_solar = reorder_days(
        source["solar_potential_mw"].to_numpy(dtype=float), order
    )
    zero_nodes_before_mapping = len(audited_zero_nodes)
    mapping: dict[str, str] = {}
    if nl0a_treatment == "paired_exclusion":
        # Central treatment.  The isolated 50 MW wind and 50 MW electrolyser
        # are removed together because the released NTC has no NL0A endpoint.
        profile_informed_graph_nodes = len(direct_profile_nodes)
    elif nl0a_treatment == "map_wind_to_NL00":
        # Topology sensitivity only.  The wind injection is conserved at the
        # Dutch demand node; the electrolyser remains quarantined because no
        # official operating profile/value is released.
        target = zones.index("NL00")
        wind[target] += source_wind
        solar[target] += source_solar
        mapping = {"NL0A_OFF": "NL00"}
        profile_informed_graph_nodes = len(direct_profile_nodes)
    else:
        raise ValueError(f"unknown NL0A treatment {nl0a_treatment}")

    if len(demand_zones) != 52:
        raise ValueError(f"expected 52 demand nodes, observed {len(demand_zones)}")
    audit = {
        "ntc_graph_nodes": len(zones),
        "demand_nodes": len(demand_zones),
        "zero_load_resource_or_network_nodes": len(zones) - len(demand_zones),
        "offshore_nodes_retained": sorted(
            zone for zone in zones if "_OFF" in zone
        ),
        "direct_profile_graph_nodes": len(direct_profile_nodes),
        "audited_zero_network_nodes_before_NL0A_mapping": zero_nodes_before_mapping,
        "audited_zero_network_nodes_after_NL0A_mapping": len(
            audited_zero_nodes
        ),
        "audited_zero_network_node_ids": sorted(audited_zero_nodes),
        "out_of_graph_profile_sources": 1,
        "NL0A_OFF_treatment": nl0a_treatment,
        "out_of_graph_profile_mapping": mapping,
        "NL0A_OFF_electrolyser_capacity_mw_quarantined": 50.0,
        "profile_informed_plus_zero_node_identity": (
            f"75={profile_informed_graph_nodes}+{len(audited_zero_nodes)}"
        ),
        "NL0A_OFF_wind_energy_mwh_source": float(source_wind.sum()),
        "NL0A_OFF_wind_energy_mwh_injected": (
            float(source_wind.sum())
            if nl0a_treatment == "map_wind_to_NL00"
            else 0.0
        ),
        "annual_conservation_rows": len(source_sums),
    }
    return (
        load,
        gross_load,
        wind,
        solar,
        btm_pv,
        btm_battery_reference,
        csp_storage_reference,
        frozenset(demand_zones),
        audit,
    )


UTC_COUNTRIES = {"IE", "PT", "UK"}
UTC_PLUS_2_COUNTRIES = {
    "BG",
    "CY",
    "EE",
    "FI",
    "GR",
    "LT",
    "LV",
    "MD",
    "RO",
}


def idsr_schedule(zone: str, resource_type: str) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    country = country_from_zone(zone)
    offset = 0 if country in UTC_COUNTRIES else 2 if country in UTC_PLUS_2_COUNTRIES else 1
    if resource_type == "EV":
        starts = {
            0: (7, 13, 19),
            1: (8, 14, 20),
            2: (9, 15, 21),
            -1: (6, 12, 18),
        }[offset]
        durations = (6, 6, 12)
    elif resource_type == "HP":
        starts = {
            0: (5, 11, 17, 23),
            1: (0, 6, 12, 18),
            2: (1, 7, 13, 19),
            -1: (4, 10, 16, 22),
        }[offset]
        durations = (6, 6, 6, 6)
    else:
        raise ValueError(resource_type)
    return offset, starts, durations


def load_idsr_inputs(
    root: Path,
    zones: tuple[str, ...],
    demand_zones: frozenset[str],
    weather_scenario: str,
    order: list[int],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    qa = read_qa(root / "QA.json", "iDSR adapter")
    required_gates = {
        "36_weather_scenarios",
        "52_demand_nodes",
        "8760_hours_per_weather_scenario",
        "annual_ratio_multiplication_exact",
        "annex2_windows_complete",
        "btm_capacity_multiplication_exact",
        "hourly_row_identity",
        "signed_balance_reference_reconstruction_exact",
        "serialized_output_reread",
    }
    gates = qa.get("gates", {})
    failed_gates = sorted(
        gate for gate in required_gates if not bool(gates.get(gate, False))
    )
    if failed_gates:
        raise ValueError(
            f"{root / 'QA.json'}: required iDSR gates failed {failed_gates}"
        )

    ev = np.zeros((len(zones), HOURS_PER_YEAR), dtype=float)
    hp = np.zeros_like(ev)
    btm_reference = np.zeros_like(ev)
    z_lookup = {zone: index for index, zone in enumerate(zones)}

    hourly_path = root / "idsr_hourly_2035.csv.gz"
    required_hourly = {
        "weather_scenario",
        "zone",
        "hour_of_year",
        "ev_price_sensitive_baseline_mw",
        "hp_price_sensitive_baseline_mw",
        "btm_price_sensitive_reference_net_output_mw",
    }
    observed_columns = set(pd.read_csv(hourly_path, nrows=0).columns)
    missing = sorted(required_hourly.difference(observed_columns))
    if missing:
        raise ValueError(f"{hourly_path}: missing columns {missing}")
    selected_chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        hourly_path,
        usecols=sorted(required_hourly),
        chunksize=500_000,
    ):
        selected = chunk.loc[
            chunk["weather_scenario"].astype(str).eq(weather_scenario)
        ]
        if not selected.empty:
            selected_chunks.append(selected)
    if not selected_chunks:
        raise ValueError(f"{hourly_path}: weather scenario absent {weather_scenario}")
    hourly = pd.concat(selected_chunks, ignore_index=True)
    expected_rows = len(demand_zones) * HOURS_PER_YEAR
    if len(hourly) != expected_rows:
        raise ValueError(
            f"{hourly_path}/{weather_scenario}: expected {expected_rows} rows, "
            f"observed {len(hourly)}"
        )
    hourly["zone"] = hourly["zone"].astype(str)
    if set(hourly["zone"]) != set(demand_zones):
        raise ValueError(
            f"{hourly_path}/{weather_scenario}: demand-zone set is inconsistent"
        )
    if hourly.duplicated(["zone", "hour_of_year"]).any():
        raise ValueError(
            f"{hourly_path}/{weather_scenario}: duplicate zone-hour key"
        )
    for zone, frame in hourly.groupby("zone", sort=False):
        frame = frame.sort_values("hour_of_year")
        if len(frame) != HOURS_PER_YEAR or not np.array_equal(
            pd.to_numeric(
                frame["hour_of_year"], errors="raise"
            ).to_numpy(dtype=int),
            np.arange(HOURS_PER_YEAR),
        ):
            raise ValueError(
                f"{hourly_path}/{weather_scenario}/{zone}: invalid hour axis"
            )
        zone_index = z_lookup[zone]
        for column, target, signed in (
            ("ev_price_sensitive_baseline_mw", ev, False),
            ("hp_price_sensitive_baseline_mw", hp, False),
            (
                "btm_price_sensitive_reference_net_output_mw",
                btm_reference,
                True,
            ),
        ):
            values = pd.to_numeric(
                frame[column], errors="raise"
            ).to_numpy(dtype=float)
            if not np.isfinite(values).all() or (
                not signed and (values < -REPORT_TOL).any()
            ):
                raise ValueError(f"{hourly_path}/{zone}/{column}: invalid values")
            target[zone_index] = reorder_days(values, order)

    def pipe_integers(value: Any) -> tuple[int, ...]:
        parsed = tuple(int(item) for item in str(value).split("|"))
        if not parsed:
            raise ValueError("empty iDSR window specification")
        return parsed

    window_path = root / "idsr_windows_2035.csv"
    schedules = pd.read_csv(window_path)
    required_windows = {
        "zone",
        "resource_type",
        "time_zone_utc_offset",
        "window_start_hours_utc",
        "window_durations_h",
    }
    missing = sorted(required_windows.difference(schedules.columns))
    if missing:
        raise ValueError(f"{window_path}: missing columns {missing}")
    schedules["zone"] = schedules["zone"].astype(str)
    schedules["resource_type"] = (
        schedules["resource_type"].astype(str).str.upper()
    )
    if (
        len(schedules) != 2 * len(demand_zones)
        or set(schedules["zone"]) != set(demand_zones)
        or set(schedules["resource_type"]) != {"EV", "HP"}
        or schedules.duplicated(["zone", "resource_type"]).any()
    ):
        raise ValueError(f"{window_path}: incomplete EV/HP window table")
    schedules["utc_offset_h"] = pd.to_numeric(
        schedules["time_zone_utc_offset"], errors="raise"
    ).astype(int)
    schedules["start_hours_utc"] = schedules[
        "window_start_hours_utc"
    ].map(pipe_integers)
    schedules["durations_h"] = schedules["window_durations_h"].map(
        pipe_integers
    )
    for row in schedules.itertuples(index=False):
        cyclic_window_ids(
            48,
            row.start_hours_utc,
            row.durations_h,
        )
    schedules = schedules[
        [
            "zone",
            "resource_type",
            "utc_offset_h",
            "start_hours_utc",
            "durations_h",
        ]
    ].sort_values(["zone", "resource_type"]).reset_index(drop=True)

    btm_path = root / "idsr_btm_battery_2035.csv"
    btm = pd.read_csv(btm_path)
    required_btm = {
        "zone",
        "price_sensitive_ratio",
        "price_sensitive_charge_power_mw",
        "price_sensitive_discharge_power_mw",
        "price_sensitive_energy_capacity_mwh",
        "round_trip_efficiency",
        "initial_soc_fraction",
        "terminal_soc_fraction",
        "capacity_record_available",
        "released_reference_profile_available",
    }
    missing = sorted(required_btm.difference(btm.columns))
    if missing:
        raise ValueError(f"{btm_path}: missing columns {missing}")
    btm["zone"] = btm["zone"].astype(str)
    if (
        len(btm) != len(demand_zones)
        or set(btm["zone"]) != set(demand_zones)
        or btm["zone"].duplicated().any()
    ):
        raise ValueError(f"{btm_path}: BTM zone contract is incomplete")
    numeric_btm = [
        "price_sensitive_ratio",
        "price_sensitive_charge_power_mw",
        "price_sensitive_discharge_power_mw",
        "price_sensitive_energy_capacity_mwh",
        "round_trip_efficiency",
        "initial_soc_fraction",
        "terminal_soc_fraction",
    ]
    btm[numeric_btm] = btm[numeric_btm].apply(
        pd.to_numeric, errors="raise"
    )
    if (
        (btm[numeric_btm] < 0).any().any()
        or not btm["round_trip_efficiency"].between(0, 1).all()
        or not btm["initial_soc_fraction"].between(0, 1).all()
        or not btm["terminal_soc_fraction"].between(0, 1).all()
    ):
        raise ValueError(f"{btm_path}: invalid BTM parameters")
    btm = btm.loc[
        btm["price_sensitive_energy_capacity_mwh"].gt(REPORT_TOL)
    ].copy()
    if (
        set(btm["zone"]) != {"AT00", "CH00"}
        or not btm["capacity_record_available"].astype(bool).all()
        or not btm["released_reference_profile_available"].astype(bool).all()
    ):
        raise ValueError(
            f"{btm_path}: active flexible BTM contract differs from AT00/CH00"
        )
    btm = btm.sort_values("zone").reset_index(drop=True)

    additional_path = root / "additional_dispatchable_consumption_2035.csv"
    additional_source = pd.read_csv(additional_path)
    required_additional = {
        "zone",
        "resource_type",
        "nameplate_capacity_mw",
        "activation_price_eur_mwh",
        "availability_lower_bound_mw",
        "availability_upper_bound_mw",
        "central_case_admitted",
        "nl0a_paired_isolation",
    }
    missing = sorted(required_additional.difference(additional_source.columns))
    if missing:
        raise ValueError(f"{additional_path}: missing columns {missing}")
    additional_source["zone"] = additional_source["zone"].astype(str)
    outside = sorted(set(additional_source["zone"]).difference(zones))
    if outside != ["NL0A_OFF"]:
        raise ValueError(f"additional-consumer zones outside graph: {outside}")
    nl0a = additional_source.loc[
        additional_source["zone"].eq("NL0A_OFF")
    ]
    if (
        len(nl0a) != 1
        or not nl0a["nl0a_paired_isolation"].astype(bool).all()
        or not np.isclose(
            float(nl0a["nameplate_capacity_mw"].iloc[0]), 50.0
        )
    ):
        raise ValueError("NL0A_OFF paired 50 MW electrolyser audit failed")
    resource_type = additional_source["resource_type"].astype(str)
    additional = additional_source.loc[
        additional_source["zone"].isin(zones)
        & (
            (
                resource_type.eq("electrolyser_optional_buyer")
                & additional_source["central_case_admitted"].astype(bool)
            )
            | resource_type.eq("power_to_heat")
        )
    ].copy()
    for column in (
        "nameplate_capacity_mw",
        "activation_price_eur_mwh",
        "availability_lower_bound_mw",
        "availability_upper_bound_mw",
    ):
        additional[column] = pd.to_numeric(
            additional[column], errors="raise"
        )
    if (
        additional["nameplate_capacity_mw"].lt(0).any()
        or additional["activation_price_eur_mwh"].lt(0).any()
        or additional["availability_lower_bound_mw"].lt(0).any()
        or (
            additional["availability_upper_bound_mw"]
            < additional["availability_lower_bound_mw"]
        ).any()
    ):
        raise ValueError("additional dispatchable consumption has invalid values")
    audit = {
        "adapter_qa": qa,
        "zones_loaded": len(demand_zones),
        "schedule_rows": len(schedules),
        "ev_flexible_mwh": float(ev.sum()),
        "hp_flexible_mwh": float(hp.sum()),
        "btm_price_sensitive_reference_mwh": float(btm_reference.sum()),
        "btm_price_sensitive_battery_rows": len(btm),
        "btm_price_sensitive_charge_power_mw": float(
            btm["price_sensitive_charge_power_mw"].sum()
        ),
        "btm_price_sensitive_energy_capacity_mwh": float(
            btm["price_sensitive_energy_capacity_mwh"].sum()
        ),
        "additional_consumers": len(additional),
        "additional_capacity_mw": float(
            additional["nameplate_capacity_mw"].sum()
        ),
        "isolated_NL0A_electrolyser_quarantined": outside == ["NL0A_OFF"],
        "timezone_contract": (
            "ERAA Annex 2 Table 8 fixed UTC windows; EU zone groups encoded "
            "explicitly without daylight-saving shifts"
        ),
    }
    return (
        ev,
        hp,
        btm_reference,
        schedules,
        btm,
        additional,
        audit,
    )


def expand_hydro_periods(
    frame: pd.DataFrame,
    code: str,
    default_power_mw: float,
) -> dict[str, Any]:
    inflow = np.zeros(HOURS_PER_YEAR, dtype=float)
    max_turbine = np.full(HOURS_PER_YEAR, default_power_mw, dtype=float)
    min_turbine = np.zeros(HOURS_PER_YEAR, dtype=float)
    max_pump = np.full(HOURS_PER_YEAR, np.nan, dtype=float)
    min_pump = np.zeros(HOURS_PER_YEAR, dtype=float)
    period_constraints: list[dict[str, Any]] = []
    selected = frame.loc[frame["hydro_code"].astype(str).eq(code)]
    for row in selected.itertuples(index=False):
        resolution = str(row.time_resolution).strip().lower()
        span = 24 if resolution == "day" else 168 if resolution == "week" else 0
        if span == 0:
            raise ValueError(f"unknown hydro resolution {resolution}")
        start = (int(row.period) - 1) * span
        nominal_stop = start + span
        stop = min(nominal_stop, HOURS_PER_YEAR)
        if start < 0 or start >= stop:
            raise ValueError(f"invalid hydro period {row.period}/{resolution}")
        coverage_fraction = (stop - start) / span
        raw_inflow = pd.to_numeric(
            pd.Series([row.inflow_raw]), errors="coerce"
        ).iloc[0]
        if pd.notna(raw_inflow):
            # ERAA releases daily/weekly inflow as a 24/168-hour energy lot.
            # The Gregorian 8,760-hour horizon contains only the first 24 h
            # of nominal week 53.  Keep the nominal hourly rate instead of
            # compressing the complete weekly energy lot into that last day.
            inflow[start:stop] = float(raw_inflow) / span
        raw_turb = pd.to_numeric(
            pd.Series([row.max_turb_raw]), errors="coerce"
        ).iloc[0]
        if pd.notna(raw_turb):
            max_turbine[start:stop] = float(raw_turb) * 1_000.0
        raw_pump = pd.to_numeric(
            pd.Series([row.max_pump_raw]), errors="coerce"
        ).iloc[0]
        if pd.notna(raw_pump):
            max_pump[start:stop] = float(raw_pump) * 1_000.0
        raw_min_turb = pd.to_numeric(
            pd.Series([row.min_turb_raw]), errors="coerce"
        ).iloc[0]
        if pd.notna(raw_min_turb):
            min_turbine[start:stop] = float(raw_min_turb) * 1_000.0
        raw_min_pump = pd.to_numeric(
            pd.Series([row.min_pump_raw]), errors="coerce"
        ).iloc[0]
        if pd.notna(raw_min_pump):
            min_pump[start:stop] = float(raw_min_pump) * 1_000.0
        values: dict[str, float | None] = {}
        for source, target, multiplier in (
            ("max_turb_en_raw", "max_generation_mwh", 1_000.0),
            ("min_turb_en_raw", "min_generation_mwh", 1_000.0),
            ("max_res_raw", "max_reservoir_fraction", 1.0),
            ("min_res_raw", "min_reservoir_fraction", 1.0),
        ):
            raw = pd.to_numeric(
                pd.Series([getattr(row, source)]), errors="coerce"
            ).iloc[0]
            if pd.isna(raw):
                values[target] = None
            else:
                value = float(raw) * multiplier
                if target in {
                    "max_generation_mwh",
                    "min_generation_mwh",
                }:
                    value *= coverage_fraction
                values[target] = value
        period_constraints.append(
            {
                "resolution": resolution,
                "period": int(row.period),
                "start_hour": start,
                "stop_hour": stop,
                "nominal_stop_hour": nominal_stop,
                "calendar_coverage_fraction": coverage_fraction,
                "calendar_period_prorated": bool(
                    coverage_fraction < 1.0 - 1e-12
                ),
                "reservoir_bound_conflict_quarantined": False,
                **values,
            }
        )
    if np.any(min_turbine > max_turbine + 1e-8):
        raise ValueError(f"{code}: MIN_TURB exceeds MAX_TURB")
    finite_pump = np.isfinite(max_pump)
    if np.any(min_pump[finite_pump] > max_pump[finite_pump] + 1e-8):
        raise ValueError(f"{code}: MIN_PUMP exceeds MAX_PUMP")
    for constraint in period_constraints:
        low = constraint["min_reservoir_fraction"]
        high = constraint["max_reservoir_fraction"]
        if low is not None and not 0 <= float(low) <= 1:
            raise ValueError(f"{code}: invalid MIN_RES {low}")
        if high is not None and not 0 <= float(high) <= 1:
            raise ValueError(f"{code}: invalid MAX_RES {high}")
        if low is not None and high is not None and float(low) > float(high):
            # The released CH00 HCL/HOL files contain MAX_RES=0.45 and
            # MIN_RES=0.55.  Do not silently swap official fields.  Quarantine
            # only this contradictory checkpoint and retain all other hydro
            # power, inflow and energy constraints.
            constraint["reservoir_bound_conflict_quarantined"] = True
            constraint["min_reservoir_fraction"] = None
            constraint["max_reservoir_fraction"] = None
    return {
        "inflow_mwh": inflow,
        "discharge_ub_mw": max_turbine,
        "discharge_lb_mw": min_turbine,
        "charge_ub_mw": max_pump,
        "charge_lb_mw": min_pump,
        "period_constraints": period_constraints,
    }


def hydro_partial_period_mapping_self_test() -> bool:
    """Verify that nominal week 53 is prorated rather than compressed."""

    fixture = pd.DataFrame(
        {
            "hydro_code": ["HRI"],
            "time_resolution": ["week"],
            "period": [53],
            "inflow_raw": [168.0],
            "max_turb_raw": [2.0],
            "min_turb_raw": [1.0],
            "max_pump_raw": [3.0],
            "min_pump_raw": [0.5],
            "max_turb_en_raw": [0.168],
            "min_turb_en_raw": [0.084],
            "max_res_raw": [0.8],
            "min_res_raw": [0.2],
        }
    )
    expanded = expand_hydro_periods(fixture, "HRI", 9_999.0)
    period = expanded["period_constraints"][0]
    fraction = 24.0 / 168.0
    return bool(
        np.isclose(expanded["inflow_mwh"].sum(), 168.0 * fraction)
        and np.allclose(expanded["discharge_ub_mw"][-24:], 2_000.0)
        and np.allclose(expanded["discharge_lb_mw"][-24:], 1_000.0)
        and np.isclose(period["max_generation_mwh"], 168.0 * fraction)
        and np.isclose(period["min_generation_mwh"], 84.0 * fraction)
        and np.isclose(period["calendar_coverage_fraction"], fraction)
        and period["nominal_stop_hour"] == 8_904
        and period["stop_hour"] == 8_760
    )


def following_weather_scenario(weather_scenario: str) -> str | None:
    """Return the next numbered ERAA WS without inventing a wraparound."""

    label = str(weather_scenario).strip()
    if not label.startswith("WS") or not label[2:].isdigit():
        return None
    index = int(label[2:])
    if index < 1 or index >= 36:
        return None
    return f"WS{index + 1:02d}"


def reservoir_trajectory_value(
    constraints: list[dict[str, Any]],
    period: int,
    *,
    fixed_only: bool,
) -> float | None:
    """Return a weekly reservoir value without using row-order fallbacks.

    A fixed trajectory value exists only when both released bounds are
    present and numerically identical.  A midpoint likewise requires both
    bounds.  This prevents a non-fixed nominal week-53 interval from being
    mistaken for an annual terminal stock.
    """

    values: list[float] = []
    for row in constraints:
        if (
            str(row.get("resolution", "")).strip().lower() != "week"
            or int(row.get("period", -1)) != int(period)
        ):
            continue
        low = row.get("min_reservoir_fraction")
        high = row.get("max_reservoir_fraction")
        if low is None or high is None:
            continue
        low_value = float(low)
        high_value = float(high)
        if fixed_only and not np.isclose(
            low_value, high_value, rtol=0, atol=1e-12
        ):
            continue
        values.append(0.5 * (low_value + high_value))
    if not values:
        return None
    if not np.allclose(values, values[0], rtol=0, atol=1e-12):
        raise ValueError(
            f"conflicting reservoir trajectory values for week {period}: "
            f"{values}"
        )
    return float(values[0])


def resolve_annex2_reservoir_endpoints(
    constraints: list[dict[str, Any]],
    current_official: tuple[float, float] | None,
    next_weather_official_initial: float | None,
) -> tuple[float, float, str, str]:
    """Apply the ERAA 2025 Annex 2 section 6.1 endpoint hierarchy.

    Released annual start/end levels take precedence.  Where they are absent,
    the initial stock uses the current-WS week-1 trajectory midpoint and then
    50%.  The terminal stock uses a fixed current-WS week-53/52 trajectory,
    then a genuinely WS-specific released initial stock for the following WS,
    then the current-WS week-52 trajectory midpoint, and finally 50%.
    """

    if current_official is not None:
        return (
            float(current_official[0]),
            float(current_official[1]),
            "official_annual_levels",
            "official_annual_levels",
        )

    initial = reservoir_trajectory_value(
        constraints, 1, fixed_only=False
    )
    if initial is None:
        initial = 0.5
        initial_source = "standard_50pct"
    else:
        initial_source = "current_ws_week1_trajectory_midpoint"

    terminal: float | None = None
    terminal_source = ""
    for period in (53, 52):
        terminal = reservoir_trajectory_value(
            constraints, period, fixed_only=True
        )
        if terminal is not None:
            terminal_source = (
                f"current_ws_week{period}_fixed_trajectory"
            )
            break
    if terminal is None and next_weather_official_initial is not None:
        terminal = float(next_weather_official_initial)
        terminal_source = "next_ws_official_initial_level"
    if terminal is None:
        terminal = reservoir_trajectory_value(
            constraints, 52, fixed_only=False
        )
        if terminal is not None:
            terminal_source = "current_ws_week52_trajectory_midpoint"
    if terminal is None:
        terminal = 0.5
        terminal_source = "standard_50pct"

    return (
        float(initial),
        float(terminal),
        initial_source,
        terminal_source,
    )


def hydro_endpoint_fallback_self_test() -> bool:
    """Exercise every Annex 2 section 6.1 fallback branch."""

    def weekly(
        period: int, low: float | None, high: float | None
    ) -> dict[str, Any]:
        return {
            "resolution": "week",
            "period": period,
            "min_reservoir_fraction": low,
            "max_reservoir_fraction": high,
        }

    official = resolve_annex2_reservoir_endpoints(
        [weekly(1, 0.2, 0.8), weekly(52, 0.3, 0.7)],
        (0.41, 0.39),
        0.61,
    )
    fixed_53 = resolve_annex2_reservoir_endpoints(
        [
            weekly(1, 0.2, 0.8),
            weekly(52, 0.3, 0.7),
            weekly(53, 0.62, 0.62),
        ],
        None,
        0.61,
    )
    fixed_52 = resolve_annex2_reservoir_endpoints(
        [weekly(1, 0.2, 0.8), weekly(52, 0.64, 0.64)],
        None,
        0.61,
    )
    next_ws = resolve_annex2_reservoir_endpoints(
        [
            weekly(1, 0.2, 0.8),
            weekly(52, 0.3, 0.7),
            weekly(53, 0.1, 0.9),
        ],
        None,
        0.61,
    )
    week_52 = resolve_annex2_reservoir_endpoints(
        [
            weekly(1, 0.2, 0.8),
            weekly(52, 0.3, 0.7),
            weekly(53, 0.1, 0.9),
        ],
        None,
        None,
    )
    standard = resolve_annex2_reservoir_endpoints([], None, None)
    return bool(
        official
        == (
            0.41,
            0.39,
            "official_annual_levels",
            "official_annual_levels",
        )
        and fixed_53
        == (
            0.5,
            0.62,
            "current_ws_week1_trajectory_midpoint",
            "current_ws_week53_fixed_trajectory",
        )
        and fixed_52
        == (
            0.5,
            0.64,
            "current_ws_week1_trajectory_midpoint",
            "current_ws_week52_fixed_trajectory",
        )
        and next_ws
        == (
            0.5,
            0.61,
            "current_ws_week1_trajectory_midpoint",
            "next_ws_official_initial_level",
        )
        and week_52
        == (
            0.5,
            0.5,
            "current_ws_week1_trajectory_midpoint",
            "current_ws_week52_trajectory_midpoint",
        )
        and standard
        == (0.5, 0.5, "standard_50pct", "standard_50pct")
        and following_weather_scenario("WS01") == "WS02"
        and following_weather_scenario("WS36") is None
    )


def load_official_reservoir_endpoints(
    path: Path,
    weather_scenario: str,
) -> tuple[
    dict[tuple[str, str], tuple[float, float]],
    dict[tuple[str, str, str], float],
    dict[str, Any],
]:
    """Load the released annual hydro start/end stocks with ERAA precedence.

    ERAA Annex 2 section 6.1 gives TSO-provided annual levels precedence over
    the weekly-trajectory and 50% fallbacks.  Exact target-year and
    weather-scenario rows therefore rank ahead of the released all-year and
    all-other-weather rows.
    """

    frame = pd.read_csv(path)
    required = {
        "MARKET_NODE",
        "TARGET_YEAR",
        "WEATHER_SCENARIO",
        "RESERVOIR_LEVEL_START",
        "RESERVOIR_LEVEL_END",
        "TECHNOLOGY",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing reservoir-level columns {missing}")
    technology_to_code = {
        label: code
        for label, code in HYDRO_LABEL_TO_CODE.items()
        if code in {"HRI", "HPI", "HOL", "HCL"}
    }
    frame = frame.loc[frame["TECHNOLOGY"].isin(technology_to_code)].copy()
    frame["hydro_code"] = frame["TECHNOLOGY"].map(technology_to_code)
    frame["target_rank"] = np.select(
        [
            frame["TARGET_YEAR"].astype(str).eq(f"ERAA_{TARGET_YEAR}"),
            frame["TARGET_YEAR"].astype(str).eq("All_years_ERAA"),
        ],
        [0, 1],
        default=99,
    )
    frame["weather_rank"] = np.select(
        [
            frame["WEATHER_SCENARIO"]
            .astype(str)
            .eq(f"ERAA-{weather_scenario}"),
            frame["WEATHER_SCENARIO"]
            .astype(str)
            .eq("All (other) weather scenarios"),
        ],
        [0, 1],
        default=99,
    )
    eligible = frame.loc[
        frame["target_rank"].lt(99) & frame["weather_rank"].lt(99)
    ].copy()
    eligible = eligible.sort_values(
        ["MARKET_NODE", "hydro_code", "target_rank", "weather_rank"]
    )
    best_rank = eligible.groupby(
        ["MARKET_NODE", "hydro_code"], as_index=False
    )[["target_rank", "weather_rank"]].first()
    selected = eligible.merge(
        best_rank,
        on=[
            "MARKET_NODE",
            "hydro_code",
            "target_rank",
            "weather_rank",
        ],
        how="inner",
        validate="many_to_one",
    )
    duplicates = selected.duplicated(
        ["MARKET_NODE", "hydro_code"], keep=False
    )
    if duplicates.any():
        keys = (
            selected.loc[duplicates, ["MARKET_NODE", "hydro_code"]]
            .drop_duplicates()
            .astype(str)
            .agg(":".join, axis=1)
            .tolist()
        )
        raise ValueError(f"{path}: duplicate best reservoir endpoints {keys}")
    for column in ("RESERVOIR_LEVEL_START", "RESERVOIR_LEVEL_END"):
        selected[column] = pd.to_numeric(selected[column], errors="raise")
        if not selected[column].between(0, 1).all():
            raise ValueError(f"{path}: {column} outside [0,1]")
    lookup = {
        (str(row.MARKET_NODE), str(row.hydro_code)): (
            float(row.RESERVOIR_LEVEL_START),
            float(row.RESERVOIR_LEVEL_END),
        )
        for row in selected.itertuples(index=False)
    }
    exact_weather = frame.loc[
        frame["target_rank"].lt(99)
        & frame["WEATHER_SCENARIO"]
        .astype(str)
        .str.fullmatch(r"ERAA-WS\d{2}")
    ].copy()
    if not exact_weather.empty:
        exact_weather["weather_scenario"] = (
            exact_weather["WEATHER_SCENARIO"]
            .astype(str)
            .str.removeprefix("ERAA-")
        )
        exact_weather = exact_weather.sort_values(
            [
                "MARKET_NODE",
                "hydro_code",
                "weather_scenario",
                "target_rank",
            ]
        )
        best_exact_rank = exact_weather.groupby(
            ["MARKET_NODE", "hydro_code", "weather_scenario"],
            as_index=False,
        )["target_rank"].first()
        exact_weather = exact_weather.merge(
            best_exact_rank,
            on=[
                "MARKET_NODE",
                "hydro_code",
                "weather_scenario",
                "target_rank",
            ],
            how="inner",
            validate="many_to_one",
        )
        exact_duplicates = exact_weather.duplicated(
            ["MARKET_NODE", "hydro_code", "weather_scenario"],
            keep=False,
        )
        if exact_duplicates.any():
            keys = (
                exact_weather.loc[
                    exact_duplicates,
                    ["MARKET_NODE", "hydro_code", "weather_scenario"],
                ]
                .drop_duplicates()
                .astype(str)
                .agg(":".join, axis=1)
                .tolist()
            )
            raise ValueError(
                f"{path}: duplicate exact-WS reservoir initials {keys}"
            )
        exact_weather["RESERVOIR_LEVEL_START"] = pd.to_numeric(
            exact_weather["RESERVOIR_LEVEL_START"], errors="raise"
        )
        if not exact_weather["RESERVOIR_LEVEL_START"].between(0, 1).all():
            raise ValueError(
                f"{path}: exact-WS RESERVOIR_LEVEL_START outside [0,1]"
            )
    exact_initial_lookup = {
        (
            str(row.MARKET_NODE),
            str(row.hydro_code),
            str(row.weather_scenario),
        ): float(row.RESERVOIR_LEVEL_START)
        for row in exact_weather.itertuples(index=False)
    }
    audit = {
        "source_file": str(path),
        "source_rows": int(len(frame)),
        "eligible_rows": int(len(eligible)),
        "selected_unique_endpoints": int(len(lookup)),
        "target_precedence": [
            f"ERAA_{TARGET_YEAR}",
            "All_years_ERAA",
        ],
        "weather_precedence": [
            f"ERAA-{weather_scenario}",
            "All (other) weather scenarios",
        ],
        "weather_specific_initial_records": int(
            len(exact_initial_lookup)
        ),
    }
    return lookup, exact_initial_lookup, audit


def load_hydro(
    input_root: Path,
    zones: tuple[str, ...],
    weather_scenario: str,
    order: list[int],
    psp_adapter: pd.DataFrame,
    reservoir_level_file: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    capacity = pd.read_csv(input_root / "capacity_2035_final.csv")
    capacity = capacity.loc[
        capacity["Target year"].eq(TARGET_YEAR)
        & capacity["data_version"].eq("ERAA 2025 final")
        & capacity["Technology"].isin(HYDRO_LABEL_TO_CODE)
    ].copy()
    capacity_by = (
        capacity.groupby(["Market_Node", "Technology"])["Value"].sum().to_dict()
    )
    hydro_energy = pd.read_csv(input_root / "hydro_capacity_2035_final.csv")
    hydro_energy = hydro_energy.loc[
        hydro_energy["TARGET_YEAR"].eq(TARGET_YEAR)
        & hydro_energy["data_version"].eq("ERAA 2025 final")
    ].copy()
    energy_by = (
        hydro_energy.groupby(["MARKET_NODE", "PEMMDB_PLANT_TYPE"])[
            "Storage Capacity [TWh]"
        ]
        .sum(min_count=1)
        .fillna(0.0)
        .to_dict()
    )
    z_lookup = {zone: index for index, zone in enumerate(zones)}
    (
        official_endpoints,
        exact_weather_initials,
        endpoint_audit,
    ) = load_official_reservoir_endpoints(
        reservoir_level_file, weather_scenario
    )
    next_weather_scenario = following_weather_scenario(weather_scenario)
    ror_rows: list[dict[str, Any]] = []
    reservoir_rows: list[dict[str, Any]] = []
    psp_rows: list[dict[str, Any]] = []
    missing_energy: list[dict[str, Any]] = []

    def endpoint_pair(
        zone: str,
        code: str,
        constraints: list[dict[str, Any]],
    ) -> tuple[float, float, str, str, str]:
        official = official_endpoints.get((zone, code))
        next_initial = (
            exact_weather_initials.get(
                (zone, code, next_weather_scenario)
            )
            if next_weather_scenario is not None
            else None
        )
        (
            initial_fraction,
            terminal_fraction,
            initial_source,
            terminal_source,
        ) = resolve_annex2_reservoir_endpoints(
            constraints,
            official,
            next_initial,
        )
        pair_source = (
            "official_annual_levels"
            if initial_source == terminal_source == "official_annual_levels"
            else "annex2_section_6_1_fallback"
        )
        return (
            initial_fraction,
            terminal_fraction,
            pair_source,
            initial_source,
            terminal_source,
        )

    for zone in zones:
        periodic_path = (
            input_root
            / "hydro_periodic"
            / f"zone={zone}"
            / "hydro_periodic.csv.gz"
        )
        if periodic_path.exists():
            periodic = pd.read_csv(periodic_path)
            periodic = periodic.loc[
                periodic["weather_scenario"].astype(str).eq(weather_scenario)
            ].copy()
        else:
            periodic = pd.DataFrame(
                columns=[
                    "hydro_code",
                    "time_resolution",
                    "period",
                    "inflow_raw",
                    "max_turb_raw",
                    "max_pump_raw",
                ]
            )
        for label, code in HYDRO_LABEL_TO_CODE.items():
            power = float(capacity_by.get((zone, label), 0.0))
            contract = expand_hydro_periods(
                periodic, code, power
            )
            # Central identification treatment: hydro inflow and all released
            # hydro constraints remain on the target calendar in both paired
            # chronologies.  Only load, wind, solar and iDSR day blocks move.
            if code == "HRR" and power > 0:
                ror_rows.append(
                    {
                        "zone": zone,
                        "hydro_code": code,
                        **contract,
                    }
                )
            elif code in {"HRI", "HPI"} and power > 0:
                energy = float(energy_by.get((zone, label), 0.0)) * 1_000_000
                if energy <= 0:
                    missing_energy.append(
                        {
                            "zone": zone,
                            "hydro_code": code,
                            "discharge_power_mw": power,
                            "reason": "no_positive_released_storage_energy",
                        }
                    )
                    # No storage dimension is fabricated.  The released
                    # inflow/power/energy constraints are retained through a
                    # zero-stock hydro resource.
                    ror_rows.append(
                        {
                            "zone": zone,
                            "hydro_code": code,
                            **contract,
                        }
                    )
                else:
                    (
                        initial_fraction,
                        terminal_fraction,
                        endpoint_source,
                        initial_endpoint_source,
                        terminal_endpoint_source,
                    ) = endpoint_pair(
                        zone, code, contract["period_constraints"]
                    )
                    reservoir_rows.append(
                        {
                            "zone": zone,
                            "hydro_code": code,
                            "energy_capacity_mwh": energy,
                            "initial_soc_fraction": initial_fraction,
                            "terminal_soc_fraction": terminal_fraction,
                            "annual_endpoint_source": endpoint_source,
                            "annual_endpoint_initial_source": (
                                initial_endpoint_source
                            ),
                            "annual_endpoint_terminal_source": (
                                terminal_endpoint_source
                            ),
                            **contract,
                        }
                    )

    usable_psp = psp_adapter.loc[
        psp_adapter["charge_power_mw"].gt(0)
        & psp_adapter["discharge_power_mw"].gt(0)
        & psp_adapter["energy_capacity_mwh"].gt(0)
    ].copy()
    excluded_psp = psp_adapter.loc[
        ~(
            psp_adapter["charge_power_mw"].gt(0)
            & psp_adapter["discharge_power_mw"].gt(0)
            & psp_adapter["energy_capacity_mwh"].gt(0)
        )
    ].copy()
    for row in usable_psp.itertuples(index=False):
        zone = str(row.zone)
        if zone not in z_lookup:
            raise ValueError(f"pumped hydro zone absent from NTC graph: {zone}")
        code = "HCL" if str(row.plant_type) == "closed loop pumping" else "HOL"
        periodic_path = (
            input_root
            / "hydro_periodic"
            / f"zone={zone}"
            / "hydro_periodic.csv.gz"
        )
        if periodic_path.exists():
            periodic = pd.read_csv(periodic_path)
            periodic = periodic.loc[
                periodic["weather_scenario"].astype(str).eq(weather_scenario)
            ]
        else:
            periodic = pd.DataFrame(
                columns=[
                    "hydro_code",
                    "time_resolution",
                    "period",
                    "inflow_raw",
                    "max_turb_raw",
                    "max_pump_raw",
                ]
            )
        contract = expand_hydro_periods(
            periodic, code, float(row.discharge_power_mw)
        )
        if code == "HCL":
            contract["inflow_mwh"][:] = 0.0
        elif not bool(row.natural_inflow_applicable):
            raise ValueError(f"{zone}/{code}: open-loop inflow flag is false")
        pump = np.full(HOURS_PER_YEAR, float(row.charge_power_mw))
        periodic_pump = contract["charge_ub_mw"]
        mask = np.isfinite(periodic_pump)
        pump[mask] = periodic_pump[mask]
        contract["charge_ub_mw"] = pump
        (
            initial_fraction,
            terminal_fraction,
            endpoint_source,
            initial_endpoint_source,
            terminal_endpoint_source,
        ) = endpoint_pair(zone, code, contract["period_constraints"])
        psp_rows.append(
            {
                "zone": zone,
                "hydro_code": code,
                "energy_capacity_mwh": float(row.energy_capacity_mwh),
                "charge_efficiency": float(row.charge_efficiency),
                "discharge_efficiency": float(row.discharge_efficiency),
                "initial_soc_fraction": initial_fraction,
                "terminal_soc_fraction": terminal_fraction,
                "annual_endpoint_source": endpoint_source,
                "annual_endpoint_initial_source": initial_endpoint_source,
                "annual_endpoint_terminal_source": terminal_endpoint_source,
                **contract,
            }
        )
    ror = pd.DataFrame(ror_rows)
    reservoir = pd.DataFrame(reservoir_rows)
    psp = pd.DataFrame(psp_rows)
    endpoint_rows = pd.concat(
        [
            reservoir[
                [
                    "zone",
                    "hydro_code",
                    "initial_soc_fraction",
                    "terminal_soc_fraction",
                    "annual_endpoint_source",
                    "annual_endpoint_initial_source",
                    "annual_endpoint_terminal_source",
                ]
            ].assign(resource_type="reservoir"),
            psp[
                [
                    "zone",
                    "hydro_code",
                    "initial_soc_fraction",
                    "terminal_soc_fraction",
                    "annual_endpoint_source",
                    "annual_endpoint_initial_source",
                    "annual_endpoint_terminal_source",
                ]
            ].assign(resource_type="pumped_hydro"),
        ],
        ignore_index=True,
    )
    endpoint_audit.update(
        {
            "admitted_stock_resource_count": int(len(endpoint_rows)),
            "official_endpoint_match_count": int(
                endpoint_rows["annual_endpoint_source"]
                .eq("official_annual_levels")
                .sum()
            ),
            "fallback_endpoint_count": int(
                endpoint_rows["annual_endpoint_source"]
                .eq("annex2_section_6_1_fallback")
                .sum()
            ),
            "initial_endpoint_source_counts": {
                str(key): int(value)
                for key, value in endpoint_rows[
                    "annual_endpoint_initial_source"
                ]
                .value_counts()
                .sort_index()
                .items()
            },
            "terminal_endpoint_source_counts": {
                str(key): int(value)
                for key, value in endpoint_rows[
                    "annual_endpoint_terminal_source"
                ]
                .value_counts()
                .sort_index()
                .items()
            },
            "next_weather_scenario_considered": (
                next_weather_scenario
            ),
            "endpoint_contract_version": (
                RESERVOIR_ENDPOINT_CONTRACT_VERSION
            ),
            "initial_fallback_precedence": [
                "official_annual_levels",
                "current_ws_week1_trajectory_midpoint",
                "standard_50pct",
            ],
            "terminal_fallback_precedence": [
                "official_annual_levels",
                "current_ws_week53_or_week52_fixed_trajectory",
                "next_ws_official_initial_level_if_weather_specific",
                "current_ws_week52_trajectory_midpoint",
                "standard_50pct",
            ],
            "admitted_endpoint_records": endpoint_rows.to_dict("records"),
        }
    )
    constraint_counts = {
        key: int(
            sum(
                row[key] is not None
                for frame in (ror, reservoir, psp)
                for constraints in (
                    frame["period_constraints"].tolist()
                    if not frame.empty
                    else []
                )
                for row in constraints
            )
        )
        for key in (
            "max_generation_mwh",
            "min_generation_mwh",
            "max_reservoir_fraction",
            "min_reservoir_fraction",
        )
    }
    quarantined_level_conflicts = [
        f"{row.zone}:{row.hydro_code}:{constraint['resolution']}:"
        f"{constraint['period']}"
        for frame in (ror, reservoir, psp)
        for row in frame.itertuples(index=False)
        for constraint in row.period_constraints
        if constraint.get("reservoir_bound_conflict_quarantined", False)
    ]
    prorated_calendar_periods = [
        {
            "zone": str(row.zone),
            "hydro_code": str(row.hydro_code),
            "resolution": str(constraint["resolution"]),
            "period": int(constraint["period"]),
            "start_hour": int(constraint["start_hour"]),
            "stop_hour": int(constraint["stop_hour"]),
            "nominal_stop_hour": int(constraint["nominal_stop_hour"]),
            "coverage_fraction": float(
                constraint["calendar_coverage_fraction"]
            ),
        }
        for frame in (ror, reservoir, psp)
        for row in frame.itertuples(index=False)
        for constraint in row.period_constraints
        if constraint.get("calendar_period_prorated", False)
    ]
    audit = {
        "run_of_river_or_zero_stock_records": len(ror),
        "reservoir_records": len(reservoir),
        "pumped_hydro_records_admitted": len(psp),
        "pumped_hydro_records_excluded_missing_dimension": int(
            len(psp_adapter) - len(usable_psp)
        ),
        "pumped_hydro_excluded_missing_dimension_ids": sorted(
            f"{row.zone}:{'HCL' if row.plant_type == 'closed loop pumping' else 'HOL'}"
            for row in excluded_psp.itertuples(index=False)
        ),
        "ordinary_hydro_missing_energy_records": missing_energy,
        "annual_reservoir_endpoints": endpoint_audit,
        "open_loop_inflow_only_where_flagged": True,
        "chronology_treatment": "calendar_fixed_common_control",
        "released_period_constraint_counts": constraint_counts,
        "released_reservoir_bound_conflicts_quarantined": len(
            quarantined_level_conflicts
        ),
        "released_reservoir_bound_conflict_ids": sorted(
            quarantined_level_conflicts
        ),
        "calendar_partial_period_energy_prorated": True,
        "calendar_partial_period_definition": (
            "daily/weekly inflow and period-energy lots retain their nominal "
            "24/168-hour rate; only the 8,760-hour covered fraction is used"
        ),
        "calendar_prorated_period_count": len(prorated_calendar_periods),
        "calendar_prorated_periods": prorated_calendar_periods,
    }
    return ror, reservoir, psp, audit


def load_thermal(
    input_root: Path,
    dispatch_root: Path,
    zones: tuple[str, ...],
    eva_case: str,
    split_case: str,
    hours: int,
    ie_crosswalk_case: str,
    biofuel_variable_cost_eur_mwh: float,
    biofuel_availability_fraction: float,
    other_res_treatment: str,
    other_res_variable_cost_eur_mwh: float,
    other_res_availability_fraction: float,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    dict[str, Any],
]:
    capacity = pd.read_csv(
        dispatch_root / "thermal_capacity_split_cases.csv"
    )
    selected = capacity.loc[
        capacity["eva_case"].astype(str).eq(eva_case)
        & capacity["split_case"].astype(str).eq(split_case)
        & capacity["zone"].astype(str).isin(zones)
        & capacity["capacity_mw"].gt(0)
    ][["zone", "technology", "capacity_mw"]].copy()
    selected = selected.sort_values(["zone", "technology"]).reset_index(drop=True)
    if selected.empty:
        raise ValueError("no thermal fleet selected")
    cost = pd.read_csv(
        dispatch_root / "thermal_variable_costs_2035_eur2024.csv"
    )
    cost_map = cost.set_index(["zone", "technology"])[
        "marginal_cost_eur2024_per_mwh"
    ]
    selected["variable_cost_eur_mwh"] = [
        float(cost_map.loc[(row.zone, row.technology)])
        for row in selected.itertuples(index=False)
    ]
    # Biofuel is a released thermal category but is outside the nine-category
    # post-EVA adapter.  It is added here so the mandatory IE M0/M1 bound
    # changes an actual dispatchable record rather than a bookkeeping row.
    released_capacity = pd.read_csv(input_root / "capacity_2035_final.csv")
    biofuel = released_capacity.loc[
        released_capacity["data_version"].eq("ERAA 2025 final")
        & released_capacity["Target year"].eq(TARGET_YEAR)
        & released_capacity["Market_Node"].astype(str).isin(zones)
        & released_capacity["Technology"].eq("Biofuel")
        & released_capacity["Operational_Status"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("available on market")
        & released_capacity["Value"].gt(0)
    ][["Market_Node", "Value"]].rename(
        columns={"Market_Node": "zone", "Value": "capacity_mw"}
    )
    biofuel = biofuel.groupby("zone", as_index=False)["capacity_mw"].sum()
    if ie_crosswalk_case == "M1":
        if "IE00" not in set(biofuel["zone"]):
            raise ValueError("IE00 released Biofuel host is absent")
        biofuel.loc[biofuel["zone"].eq("IE00"), "capacity_mw"] = 8.0
    elif ie_crosswalk_case not in {"M0", "M3"}:
        raise ValueError(f"unsupported IE crosswalk case {ie_crosswalk_case}")
    biofuel["technology"] = "biofuel_dispatchable"
    biofuel["variable_cost_eur_mwh"] = biofuel_variable_cost_eur_mwh
    other_res_source = released_capacity.loc[
        released_capacity["data_version"].eq("ERAA 2025 final")
        & released_capacity["Target year"].eq(TARGET_YEAR)
        & released_capacity["Market_Node"].astype(str).isin(zones)
        & released_capacity["Technology"].isin(OTHER_RES_TECHNOLOGIES)
        & released_capacity["Operational_Status"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("available on market")
        & released_capacity["Value"].gt(0)
    ][["Market_Node", "Technology", "Value"]].rename(
        columns={
            "Market_Node": "zone",
            "Technology": "source_technology",
            "Value": "capacity_mw",
        }
    )
    other_res_source = (
        other_res_source.groupby(
            ["zone", "source_technology"], as_index=False
        )["capacity_mw"].sum()
    )
    if other_res_treatment == "dispatchable_sensitivity":
        other_res = (
            other_res_source.groupby("zone", as_index=False)["capacity_mw"]
            .sum()
        )
        other_res["technology"] = "other_res_dispatchable"
        other_res["variable_cost_eur_mwh"] = (
            other_res_variable_cost_eur_mwh
        )
    elif other_res_treatment == "quarantine":
        other_res = pd.DataFrame(
            columns=[
                "zone",
                "capacity_mw",
                "technology",
                "variable_cost_eur_mwh",
            ]
        )
    else:
        raise ValueError(f"unknown Other RES treatment {other_res_treatment}")
    thermal_parts = [selected, biofuel]
    if not other_res.empty:
        thermal_parts.append(other_res)
    selected = pd.concat(thermal_parts, ignore_index=True)
    selected = selected.sort_values(["zone", "technology"]).reset_index(drop=True)
    availability = pd.read_csv(
        dispatch_root / "thermal_expected_availability_daily.csv.gz"
    )
    availability = availability.loc[
        availability["eva_case"].astype(str).eq(eva_case)
        & availability["split_case"].astype(str).eq(split_case)
    ].copy()
    availability_map = {
        (str(zone), str(technology)): group.sort_values("day_of_year")[
            "expected_available_fraction"
        ].to_numpy(dtype=float)
        for (zone, technology), group in availability.groupby(
            ["zone", "technology"]
        )
    }
    ub = np.zeros((len(selected), hours), dtype=float)
    for index, row in enumerate(selected.itertuples(index=False)):
        if str(row.technology) == "biofuel_dispatchable":
            daily = np.full(365, biofuel_availability_fraction)
        elif str(row.technology) == "other_res_dispatchable":
            daily = np.full(365, other_res_availability_fraction)
        else:
            daily = availability_map.get((str(row.zone), str(row.technology)))
        if daily is None or len(daily) != 365:
            raise ValueError(f"missing daily availability for {row.zone}/{row.technology}")
        # Maintenance/outage expectation is fixed on the target calendar and
        # is intentionally not permuted with the weather chronology.
        ub[index] = float(row.capacity_mw) * np.repeat(daily, 24)[:hours]
    z_lookup = {zone: index for index, zone in enumerate(zones)}
    firm_availability = np.full((len(zones), hours), 0.92, dtype=float)
    firm_variable_cost = np.full(len(zones), np.nan, dtype=float)
    for zone in zones:
        daily = availability_map.get((zone, "gas_ocgt"))
        if daily is not None:
            firm_availability[z_lookup[zone]] = np.repeat(daily, 24)[:hours]
        key = (zone, "gas_ocgt")
        if key in cost_map.index:
            firm_variable_cost[z_lookup[zone]] = float(cost_map.loc[key])
    fallback = float(cost.loc[cost["technology"].eq("gas_ocgt"), "marginal_cost_eur2024_per_mwh"].median())
    firm_variable_cost = np.nan_to_num(firm_variable_cost, nan=fallback)
    firm_costs = pd.read_csv(
        dispatch_root / "firm_addition_costs_2035_eur2024.csv"
    )
    firm_annual = float(
        firm_costs.set_index("firm_addition_technology").loc[
            "gas_ocgt_new", "annual_cost_eur2024_per_kw_year"
        ]
    )
    thermal_audit = {
        "other_res_treatment": other_res_treatment,
        "other_res_source_technologies": list(OTHER_RES_TECHNOLOGIES),
        "other_res_model_keys": list(OTHER_RES_MODEL_KEYS),
        "other_res_available_on_market_source_rows": len(other_res_source),
        "other_res_available_on_market_source_capacity_mw": float(
            other_res_source["capacity_mw"].sum()
        ),
        "other_res_dispatch_rows": len(other_res),
        "other_res_dispatch_capacity_mw": float(
            other_res["capacity_mw"].sum()
            if not other_res.empty
            else 0.0
        ),
        "other_res_variable_cost_eur_mwh": (
            other_res_variable_cost_eur_mwh
            if other_res_treatment == "dispatchable_sensitivity"
            else None
        ),
        "other_res_availability_fraction": (
            other_res_availability_fraction
            if other_res_treatment == "dispatchable_sensitivity"
            else None
        ),
        "other_res_parameter_evidence_class": (
            "author_defined_dispatch_sensitivity_pending_sourced_cost_suite"
            if other_res_treatment == "dispatchable_sensitivity"
            else "released_capacity_quarantined"
        ),
        "inelastic_other_res_without_released_hourly_profile_not_dispatched": (
            True
        ),
    }
    return (
        selected,
        ub,
        firm_availability,
        firm_variable_cost,
        firm_annual,
        thermal_audit,
    )


def load_must_run_contracts(
    path: Path,
    thermal: pd.DataFrame,
    thermal_ub: np.ndarray,
    case: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Map released broad-fuel ratios onto the amended post-EVA fleet."""

    if case not in {"on", "off"}:
        raise ValueError(f"unknown must-run case {case}")
    source = pd.read_csv(path)
    source = source.loc[source["TARGET_YEAR"].eq(TARGET_YEAR)].copy()
    technology_map = {
        "Coal": {"hard_coal"},
        "Lignite": {"lignite"},
        "Nuclear": {"nuclear"},
        "Oil": {"light_oil", "heavy_oil"},
        "Gas": {"gas_ccgt", "gas_ocgt"},
        "other": {"other_res_dispatchable", "biofuel_dispatchable"},
    }
    rows: list[dict[str, Any]] = []
    absent_hosts: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    for _, selected_source in source.iterrows():
        zone = str(selected_source["MARKET_NODE"])
        broad = str(selected_source["Technology"])
        ratio = float(selected_source["Weighted Average Must-run Ratio"])
        derated_source_mw = float(
            selected_source["Sum of Derated Capacity (MW)"]
        )
        if not 0 <= ratio <= 1:
            raise ValueError(f"invalid must-run ratio {zone}/{broad}: {ratio}")
        mapped = technology_map.get(broad)
        if mapped is None:
            quarantined.append(
                {
                    "zone": zone,
                    "broad_fuel": broad,
                    "ratio": ratio,
                    "source_derated_capacity_mw": derated_source_mw,
                    "reason": "no_defensible_post_eva_technology_crosswalk",
                }
            )
            continue
        indices = thermal.index[
            thermal["zone"].astype(str).eq(zone)
            & thermal["technology"].astype(str).isin(mapped)
        ].to_numpy(dtype=int)
        if len(indices) == 0:
            if broad == "other":
                # The final German small-biomass/waste records are explicitly
                # labelled inelastic/fixed profile, but their hourly profile
                # is not in the release. Preserve only the independently
                # released derated must-run floor as a constant common-control
                # injection; do not fabricate dispatch flexibility above it.
                source_floor_mw = derated_source_mw * ratio
                rows.append(
                    {
                        "zone": zone,
                        "broad_fuel": broad,
                        "ratio": ratio,
                        "source_derated_capacity_mw": derated_source_mw,
                        "source_must_run_floor_mw": source_floor_mw,
                        "thermal_indices": indices,
                        "standalone_fixed_injection": True,
                        "post_eva_nameplate_capacity_mw": 0.0,
                        "annual_available_capacity_mwh": (
                            derated_source_mw * thermal_ub.shape[1]
                        ),
                        "annual_must_run_floor_mwh": (
                            source_floor_mw * thermal_ub.shape[1]
                        ),
                    }
                )
                continue
            absent_hosts.append(
                {
                    "zone": zone,
                    "broad_fuel": broad,
                    "ratio": ratio,
                    "source_derated_capacity_mw": derated_source_mw,
                    "reason": "no_positive_amended_post_eva_host",
                }
            )
            continue
        available = thermal_ub[indices].sum(axis=0)
        rows.append(
            {
                "zone": zone,
                "broad_fuel": broad,
                "ratio": ratio,
                "source_derated_capacity_mw": derated_source_mw,
                "thermal_indices": indices,
                "standalone_fixed_injection": False,
                "post_eva_nameplate_capacity_mw": float(
                    thermal.loc[indices, "capacity_mw"].sum()
                ),
                "annual_available_capacity_mwh": float(available.sum()),
                "annual_must_run_floor_mwh": float(
                    ratio * available.sum()
                ),
            }
        )
    contracts = pd.DataFrame(rows)
    audit = {
        "case": case,
        "source_rows_2035": len(source),
        "mapped_group_rows": len(contracts),
        "mapped_annual_available_capacity_mwh": float(
            contracts["annual_available_capacity_mwh"].sum()
            if not contracts.empty
            else 0.0
        ),
        "mapped_annual_must_run_floor_mwh": float(
            contracts["annual_must_run_floor_mwh"].sum()
            if not contracts.empty
            else 0.0
        ),
        "absent_post_eva_hosts": absent_hosts,
        "quarantined_rows": quarantined,
        "mapping_rule": (
            "released weighted ratio times calendar-fixed hourly available "
            "capacity of amended post-EVA broad-fuel host"
        ),
    }
    return contracts, audit


def load_network(
    dispatch_root: Path,
    edges: pd.DataFrame,
    hours: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    ntc = pd.read_csv(dispatch_root / "ntc_hourly.csv.gz")
    ntc = ntc.loc[ntc["hour_of_year"].lt(hours)].copy()
    # Eight directed pairs have parallel HVAC and HVDC components under one
    # released edge_id.  Their transfer capabilities add; pivoting raw rows
    # would be both invalid and physically understate the interface.
    ntc = (
        ntc.groupby(["edge_id", "hour_of_year"], as_index=False)[
            "capacity_mw_expected_after_for"
        ]
        .sum()
    )
    pivot = ntc.pivot(
        index="edge_id",
        columns="hour_of_year",
        values="capacity_mw_expected_after_for",
    ).reindex(edges["edge_id"])
    if pivot.shape != (len(edges), hours) or pivot.isna().any().any():
        raise ValueError("NTC matrix is incomplete")
    zone_limits = pd.read_csv(
        dispatch_root / "network_zone_limits_hourly.csv.gz"
    )
    zone_limits = zone_limits.loc[
        zone_limits["hour_of_year"].lt(hours)
    ].copy()
    return pivot.to_numpy(dtype=float), zone_limits


def truncate_resource_arrays(frame: pd.DataFrame, hours: int) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    for column in result.columns:
        if column.endswith("_mw") or column.endswith("_mwh"):
            if result[column].map(lambda value: isinstance(value, np.ndarray)).any():
                result[column] = result[column].map(lambda value: value[:hours])
    if "period_constraints" in result:
        def truncate_contract(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            retained: list[dict[str, Any]] = []
            for row in rows:
                start = int(row["start_hour"])
                if start >= hours:
                    continue
                one = dict(row)
                one["period_fully_observed"] = bool(
                    int(one["stop_hour"]) <= hours
                )
                one["stop_hour"] = min(int(one["stop_hour"]), hours)
                retained.append(one)
            return retained

        result["period_constraints"] = result["period_constraints"].map(
            truncate_contract
        )
    if hours < HOURS_PER_YEAR and {
        "initial_soc_fraction",
        "terminal_soc_fraction",
    }.issubset(result.columns):
        result["terminal_soc_fraction"] = result["initial_soc_fraction"]
    return result


def run_of_river_contract_structural_lower_bound(
    frame: pd.DataFrame,
) -> tuple[float, list[dict[str, Any]]]:
    """Return the exact minimum released-contract slack for zero-stock hydro.

    Run-of-river and zero-stock resources must satisfy generation plus spill
    equal to inflow.  Some released minimum-power and period-energy fields are
    therefore mutually incompatible with inflow or turbine capacity.  This
    helper solves that small separable problem analytically, without dispatch
    or economic assumptions, so the unavoidable part is not mistaken for
    solver or model slack.
    """

    total = 0.0
    details: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        inflow = np.asarray(row.inflow_mwh, dtype=float)
        upper = np.asarray(row.discharge_ub_mw, dtype=float)
        lower = np.asarray(row.discharge_lb_mw, dtype=float)
        available = np.minimum(inflow, upper)
        if (
            np.any(~np.isfinite(available))
            or np.any(~np.isfinite(lower))
            or np.any(available < -REPORT_TOL)
            or np.any(lower < -REPORT_TOL)
        ):
            raise ValueError(
                f"{row.zone}/{row.hydro_code}: invalid zero-stock hydro bounds"
            )
        available = np.maximum(available, 0.0)
        unavoidable_power = np.maximum(lower - available, 0.0)
        resource_total = float(unavoidable_power.sum())
        intervals: list[tuple[int, int]] = []
        period_details: list[dict[str, Any]] = []
        for item in row.period_constraints:
            start = int(item["start_hour"])
            stop = int(item["stop_hour"])
            fully_observed = bool(
                item.get("period_fully_observed", stop <= len(available))
            )
            minimum_energy = item["min_generation_mwh"]
            maximum_energy = item["max_generation_mwh"]
            if (
                not fully_observed
                or (minimum_energy is None and maximum_energy is None)
            ):
                continue
            if start < 0 or stop <= start or stop > len(available):
                raise ValueError(
                    f"{row.zone}/{row.hydro_code}: invalid hydro contract span"
                )
            if any(start < previous_stop and previous_start < stop
                   for previous_start, previous_stop in intervals):
                raise ValueError(
                    f"{row.zone}/{row.hydro_code}: overlapping energy contracts"
                )
            intervals.append((start, stop))
            interval_available = available[start:stop]
            interval_lower = lower[start:stop]
            generation_ceiling = float(interval_available.sum())
            satisfiable_lower_power = float(
                np.minimum(interval_lower, interval_available).sum()
            )
            energy_minimum = (
                0.0 if minimum_energy is None else float(minimum_energy)
            )
            energy_maximum = (
                math.inf if maximum_energy is None else float(maximum_energy)
            )
            if (
                energy_minimum < -REPORT_TOL
                or energy_maximum < -REPORT_TOL
            ):
                raise ValueError(
                    f"{row.zone}/{row.hydro_code}: negative energy contract"
                )
            candidates = [
                0.0,
                generation_ceiling,
                satisfiable_lower_power,
                energy_minimum,
            ]
            if math.isfinite(energy_maximum):
                candidates.append(energy_maximum)
            candidates = [
                min(max(value, 0.0), generation_ceiling)
                for value in candidates
                if math.isfinite(value)
            ]

            def additional_slack(generation_mwh: float) -> float:
                return (
                    max(satisfiable_lower_power - generation_mwh, 0.0)
                    + max(energy_minimum - generation_mwh, 0.0)
                    + (
                        max(generation_mwh - energy_maximum, 0.0)
                        if math.isfinite(energy_maximum)
                        else 0.0
                    )
                )

            extra = min(additional_slack(value) for value in candidates)
            resource_total += extra
            if extra > REPORT_TOL:
                period_details.append(
                    {
                        "start_hour": start,
                        "stop_hour": stop,
                        "additional_structural_lower_bound_mwh": float(extra),
                    }
                )
        total += resource_total
        if resource_total > REPORT_TOL:
            details.append(
                {
                    "zone": str(row.zone),
                    "hydro_code": str(row.hydro_code),
                    "minimum_power_structural_lower_bound_mwh": float(
                        unavoidable_power.sum()
                    ),
                    "period_energy_additional_lower_bound_mwh": float(
                        resource_total - unavoidable_power.sum()
                    ),
                    "total_structural_lower_bound_mwh": float(resource_total),
                    "periods": period_details,
                }
            )
    return float(total), details


def development_reservoir_terminal_boundary_lower_bound(
    frame: pd.DataFrame,
    hours: int,
) -> tuple[float, list[dict[str, Any]]]:
    """Return slack forced only by a truncated equal-endpoint reservoir run."""

    if hours == HOURS_PER_YEAR or frame.empty:
        return 0.0, []
    total = 0.0
    details: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        intervals: list[tuple[int, int]] = []
        minimum_generation = 0.0
        for item in row.period_constraints:
            start = int(item["start_hour"])
            stop = int(item["stop_hour"])
            lower = item["min_generation_mwh"]
            fully_observed = bool(
                item.get("period_fully_observed", stop <= hours)
            )
            if lower is None or not fully_observed:
                continue
            if start < 0 or stop <= start or stop > hours:
                raise ValueError(
                    f"{row.zone}/{row.hydro_code}: invalid reservoir contract span"
                )
            if any(start < previous_stop and previous_start < stop
                   for previous_start, previous_stop in intervals):
                raise ValueError(
                    f"{row.zone}/{row.hydro_code}: overlapping energy contracts"
                )
            intervals.append((start, stop))
            minimum_generation += float(lower)
        inflow = float(np.asarray(row.inflow_mwh, dtype=float).sum())
        lower_bound = max(minimum_generation - inflow, 0.0)
        total += lower_bound
        if lower_bound > REPORT_TOL:
            details.append(
                {
                    "zone": str(row.zone),
                    "hydro_code": str(row.hydro_code),
                    "minimum_generation_mwh": float(minimum_generation),
                    "inflow_mwh": inflow,
                    "development_terminal_boundary_lower_bound_mwh": float(
                        lower_bound
                    ),
                }
            )
    return float(total), details


def load_system(args: argparse.Namespace) -> tuple[SystemData, dict[str, Any]]:
    input_root = args.input.resolve()
    chronology_root = args.chronology_root.resolve()
    dispatch_root = args.dispatch_root.resolve()
    flexible_root = args.flexible_root.resolve()
    idsr_root = args.idsr_root.resolve()
    input_qa = read_qa(
        input_root / "QA.json",
        "phase-1 standardization",
        eligible_keys=("pass", "release_eligible"),
    )
    chronology_qa = read_qa(
        chronology_root / "QA.json", "synchronized chronology"
    )
    dispatch_qa = read_qa(dispatch_root / "QA.json", "dispatch adapter")
    flexible_qa = read_qa(flexible_root / "QA.json", "flexible-resource adapter")
    idsr_qa = read_qa(idsr_root / "QA.json", "iDSR adapter")
    if not dispatch_qa.get("post_eva_capacity_ready", False):
        raise ValueError("dispatch adapter has no admitted post-EVA fleet")

    order, order_hash = read_order(
        chronology_root, args.weather_scenario, args.chronology
    )
    edges, zones = graph_nodes(dispatch_root)
    batteries = pd.read_csv(
        flexible_root / "existing_utility_battery_2035.csv"
    )
    psp_adapter = pd.read_csv(
        flexible_root / "existing_pumped_hydro_2035.csv"
    )
    dsr = pd.read_csv(flexible_root / "explicit_dsr_bands_2035.csv")
    flexible_zones = set(batteries["zone"].astype(str))
    flexible_zones.update(psp_adapter["zone"].astype(str))
    flexible_zones.update(dsr["zone"].astype(str))
    (
        load,
        gross_load,
        wind,
        solar,
        btm_pv,
        btm_battery_reference,
        csp_storage_reference,
        demand_zones,
        profile_audit,
    ) = load_hourly_profiles(
        input_root,
        zones,
        args.weather_scenario,
        order,
        flexible_zones,
        args.nl0a_treatment,
    )
    (
        idsr_ev,
        idsr_hp,
        idsr_btm_reference,
        idsr_schedules,
        idsr_btm_batteries,
        additional_consumers,
        idsr_audit,
    ) = load_idsr_inputs(
        idsr_root,
        zones,
        demand_zones,
        args.weather_scenario,
        order,
    )
    if np.any(idsr_ev + idsr_hp > gross_load + 1e-5):
        raise ValueError("price-sensitive EV/HP demand exceeds gross demand")
    for label, frame in (
        ("battery", batteries),
        ("pumped hydro", psp_adapter),
        ("DSR", dsr),
    ):
        outside = set(frame["zone"].astype(str)).difference(zones)
        if outside:
            raise ValueError(f"{label} zones outside NTC graph: {sorted(outside)}")

    ror, reservoir, psp, hydro_audit = load_hydro(
        input_root,
        zones,
        args.weather_scenario,
        order,
        psp_adapter,
        args.reservoir_level_file.resolve(),
    )
    hours = args.development_hour_limit or HOURS_PER_YEAR
    (
        thermal,
        thermal_ub,
        firm_availability,
        firm_variable_cost,
        firm_cost,
        thermal_audit,
    ) = (
        load_thermal(
            input_root,
            dispatch_root,
            zones,
            args.capacity_case,
            args.split_case,
            hours,
            args.ie_crosswalk_case,
            args.biofuel_variable_cost_eur_mwh,
            args.biofuel_availability_fraction,
            args.other_res_treatment,
            args.other_res_variable_cost_eur_mwh,
            args.other_res_availability_fraction,
        )
    )
    must_run, must_run_audit = load_must_run_contracts(
        args.must_run_file.resolve(),
        thermal,
        thermal_ub,
        args.must_run_case,
    )
    network_ub, zone_limits = load_network(dispatch_root, edges, hours)
    ror = truncate_resource_arrays(ror, hours)
    reservoir = truncate_resource_arrays(reservoir, hours)
    psp = truncate_resource_arrays(psp, hours)
    (
        official_ror_structural_lower_bound_mwh,
        official_ror_structural_lower_bound_details,
    ) = run_of_river_contract_structural_lower_bound(ror)
    (
        development_terminal_boundary_lower_bound_mwh,
        development_terminal_boundary_lower_bound_details,
    ) = development_reservoir_terminal_boundary_lower_bound(
        reservoir, hours
    )
    hydro_audit.update(
        {
            "official_ror_structural_lower_bound_mwh": (
                official_ror_structural_lower_bound_mwh
            ),
            "official_ror_structural_lower_bound_details": (
                official_ror_structural_lower_bound_details
            ),
            "development_terminal_boundary_condition_active": (
                hours < HOURS_PER_YEAR
            ),
            "development_terminal_boundary_lower_bound_mwh": (
                development_terminal_boundary_lower_bound_mwh
            ),
            "development_terminal_boundary_lower_bound_details": (
                development_terminal_boundary_lower_bound_details
            ),
            "admissible_hydro_contract_slack_lower_bound_mwh": (
                official_ror_structural_lower_bound_mwh
                + development_terminal_boundary_lower_bound_mwh
            ),
            "hydro_contract_slack_audit_definition": (
                "released zero-stock hydro constraints that are analytically "
                "incompatible with inflow/turbine bounds, plus the separate "
                "equal-initial-and-terminal-stock boundary imposed only on "
                "truncated development runs"
            ),
        }
    )

    system = SystemData(
        zones=zones,
        demand_zones=demand_zones,
        hours=hours,
        load=load[:, :hours],
        gross_load=gross_load[:, :hours],
        wind=wind[:, :hours],
        solar=solar[:, :hours],
        btm_pv=btm_pv[:, :hours],
        btm_battery_reference=btm_battery_reference[:, :hours],
        csp_storage_reference=csp_storage_reference[:, :hours],
        idsr_ev_baseline=idsr_ev[:, :hours],
        idsr_hp_baseline=idsr_hp[:, :hours],
        idsr_btm_reference=idsr_btm_reference[:, :hours],
        idsr_schedules=idsr_schedules,
        idsr_btm_batteries=idsr_btm_batteries,
        additional_consumers=additional_consumers,
        ror=ror,
        thermal=thermal,
        thermal_ub=thermal_ub,
        firm_availability=firm_availability,
        firm_variable_cost=firm_variable_cost,
        firm_annual_cost_eur_kw=firm_cost,
        edges=edges,
        network_ub=network_ub,
        zone_limits=zone_limits,
        batteries=batteries,
        pumped_hydro=psp,
        dsr=dsr,
        must_run=must_run,
        reservoir=reservoir,
        resource_audit={
            "profile": profile_audit,
            "hydro": hydro_audit,
            "thermal": thermal_audit,
            "must_run": must_run_audit,
            "idsr": idsr_audit,
            "input_QA_hash": sha256(input_root / "QA.json"),
            "chronology_QA_hash": sha256(chronology_root / "QA.json"),
            "dispatch_QA_hash": sha256(dispatch_root / "QA.json"),
            "flexible_QA_hash": sha256(flexible_root / "QA.json"),
            "must_run_source_hash": sha256(args.must_run_file.resolve()),
            "idsr_QA_hash": sha256(idsr_root / "QA.json"),
        },
    )
    audit = {
        "order_sha256": order_hash,
        "system": system.resource_audit,
        "input_qa": input_qa,
        "chronology_qa": chronology_qa,
        "dispatch_qa": dispatch_qa,
        "flexible_qa": flexible_qa,
        "idsr_qa": idsr_qa,
    }
    return system, audit


def candidate_allocation(
    system: SystemData,
    scope: str,
    tranche_share: float,
) -> np.ndarray:
    if not 0 < tranche_share < 1:
        raise ValueError("tranche share must lie between zero and one")
    mean_load = system.gross_load.mean(axis=1)
    eligible = np.array(
        [
            zone in system.demand_zones
            and (scope == "extended_europe" or country_from_zone(zone) in EU27)
            for zone in system.zones
        ],
        dtype=bool,
    )
    if not eligible.any() or mean_load[eligible].sum() <= 0:
        raise ValueError(f"no positive-load deployment nodes for {scope}")
    shares = np.zeros(len(system.zones))
    shares[eligible] = mean_load[eligible] / mean_load[eligible].sum()
    total = tranche_share * float(mean_load[eligible].sum())
    allocation = total * shares
    if not np.isclose(allocation.sum(), total, atol=1e-8):
        raise RuntimeError("candidate allocation does not reconcile")
    return allocation


def storage_configurations(args: argparse.Namespace):
    core = load_module("eraa_core_for_full_network", CORE_PATH)
    helper = load_module("eraa_zone_helper_for_full_network", ZONE_RUNNER_PATH)
    ledger_scenario = {
        "low": "low_cny_per_eur",
        "central": "central_cny_per_eur",
        "high": "high_cny_per_eur",
    }[args.currency_scenario]
    rate = helper.currency_rate(
        args.dispatch_root.resolve(), ledger_scenario
    )
    configurations = helper.build_storage_configurations(
        core,
        args.catalog.resolve(),
        args.technology_library.resolve(),
        args.config_id,
        rate,
        args.incremental_charge_cost_multiplier,
        args.lfp_degradation_multiplier,
        args.maximum_charge_to_discharge_ratio,
    )
    indexed = {item.config_id: item for item in configurations}
    if len(indexed) != len(configurations):
        raise ValueError("duplicate storage configuration identifiers")
    return indexed, rate


@dataclass
class SolveResult:
    variant: str
    metrics: dict[str, Any]
    first_stage: pd.DataFrame
    candidate_dispatch: pd.DataFrame
    fossil_gas_dispatch: pd.DataFrame
    hydro_slack_ledger: pd.DataFrame


def candidate_fixed_cost_components(
    discharge_power_mw: float,
    duration_h: float,
    annual_discharge_power_cost_eur_kw: float,
    annual_energy_cost_eur_kwh: float,
) -> tuple[float, float]:
    """Return annual discharge-interface and energy-capacity cost separately."""

    if min(
        discharge_power_mw,
        duration_h,
        annual_discharge_power_cost_eur_kw,
        annual_energy_cost_eur_kwh,
    ) < 0:
        raise ValueError("candidate fixed-cost inputs must be non-negative")
    discharge = (
        discharge_power_mw
        * annual_discharge_power_cost_eur_kw
        * 1_000.0
    )
    energy = (
        discharge_power_mw
        * duration_h
        * annual_energy_cost_eur_kwh
        * 1_000.0
    )
    return float(discharge), float(energy)


def fixed_cost_analytic_self_test() -> bool:
    discharge, energy = candidate_fixed_cost_components(
        discharge_power_mw=2.0,
        duration_h=10.0,
        annual_discharge_power_cost_eur_kw=5.0,
        annual_energy_cost_eur_kwh=3.0,
    )
    return bool(
        np.isclose(discharge, 10_000.0)
        and np.isclose(energy, 60_000.0)
        and np.isclose(discharge + energy, 70_000.0)
    )


def configure_gurobi(model, args: argparse.Namespace) -> None:
    model.Params.OutputFlag = int(not args.quiet)
    model.Params.Threads = args.threads
    model.Params.Method = args.method
    model.Params.Presolve = 2
    model.Params.NumericFocus = 1
    model.Params.FeasibilityTol = args.feasibility_tolerance
    model.Params.OptimalityTol = args.optimality_tolerance
    if args.time_limit > 0:
        model.Params.TimeLimit = args.time_limit
    if args.log_file:
        model.Params.LogFile = str(args.log_file.resolve())


def add_hydro_contract_constraints(
    *,
    model,
    generation,
    discharge_lb_mw: np.ndarray,
    period_constraints: list[dict[str, Any]],
    penalty_eur_mwh: float,
    name: str,
    soc=None,
    energy_capacity_mwh: float | None = None,
    charge=None,
    charge_lb_mw: np.ndarray | None = None,
    soft_nonlevel_source_discrepancy: bool = False,
) -> tuple[dict[str, list[Any]], dict[str, int]]:
    """Apply every released hydro lower/energy/level constraint.

    ERAA permits reservoir-level trajectory relaxation below VoLL when
    otherwise infeasible.  Power, pumping and period-energy constraints remain
    hard for stocked reservoirs and PSP.  Only independently verified
    zero-stock source-contract inconsistencies may receive an explicit,
    separately reported non-level slack.
    """

    slacks: dict[str, list[Any]] = {
        "source_contract_discrepancy": [],
        "level_trajectory": [],
    }
    counts = {
        "minimum_power": 0,
        "minimum_pumping_power": 0,
        "maximum_generation_energy": 0,
        "minimum_generation_energy": 0,
        "maximum_reservoir_level": 0,
        "minimum_reservoir_level": 0,
    }
    lower = np.asarray(discharge_lb_mw, dtype=float)
    positive = np.flatnonzero(lower > REPORT_TOL)
    if len(positive):
        if soft_nonlevel_source_discrepancy:
            slack = model.addMVar(
                len(positive),
                lb=0.0,
                obj=penalty_eur_mwh,
                name=f"{name}_minimum_power_source_discrepancy",
            )
            model.addConstr(
                generation[positive] + slack >= lower[positive],
                name=f"{name}_minimum_power",
            )
            slacks["source_contract_discrepancy"].append(slack)
        else:
            model.addConstr(
                generation[positive] >= lower[positive],
                name=f"{name}_minimum_power",
            )
        counts["minimum_power"] += len(positive)
    if charge is not None and charge_lb_mw is not None:
        lower_charge = np.asarray(charge_lb_mw, dtype=float)
        positive_charge = np.flatnonzero(lower_charge > REPORT_TOL)
        if len(positive_charge):
            if soft_nonlevel_source_discrepancy:
                slack = model.addMVar(
                    len(positive_charge),
                    lb=0.0,
                    obj=penalty_eur_mwh,
                    name=(
                        f"{name}_minimum_pumping_power_source_discrepancy"
                    ),
                )
                model.addConstr(
                    charge[positive_charge] + slack
                    >= lower_charge[positive_charge],
                    name=f"{name}_minimum_pumping_power",
                )
                slacks["source_contract_discrepancy"].append(slack)
            else:
                model.addConstr(
                    charge[positive_charge] >= lower_charge[positive_charge],
                    name=f"{name}_minimum_pumping_power",
                )
            counts["minimum_pumping_power"] += len(positive_charge)
    for item in period_constraints:
        start = int(item["start_hour"])
        stop = int(item["stop_hour"])
        if start >= generation.shape[0] or stop <= start:
            continue
        period_fully_observed = bool(
            item.get("period_fully_observed", stop <= generation.shape[0])
        )
        upper = item["max_generation_mwh"]
        if upper is not None and period_fully_observed:
            if soft_nonlevel_source_discrepancy:
                slack = model.addVar(
                    lb=0.0,
                    obj=penalty_eur_mwh,
                    name=f"{name}_max_energy_source_discrepancy[{start}]",
                )
                model.addConstr(
                    generation[start:stop].sum()
                    <= float(upper) + slack,
                    name=f"{name}_max_energy[{start}]",
                )
                slacks["source_contract_discrepancy"].append(slack)
            else:
                model.addConstr(
                    generation[start:stop].sum() <= float(upper),
                    name=f"{name}_max_energy[{start}]",
                )
            counts["maximum_generation_energy"] += 1
        lower_energy = item["min_generation_mwh"]
        if lower_energy is not None and period_fully_observed:
            if soft_nonlevel_source_discrepancy:
                slack = model.addVar(
                    lb=0.0,
                    obj=penalty_eur_mwh,
                    name=f"{name}_min_energy_source_discrepancy[{start}]",
                )
                model.addConstr(
                    generation[start:stop].sum() + slack
                    >= float(lower_energy),
                    name=f"{name}_min_energy[{start}]",
                )
                slacks["source_contract_discrepancy"].append(slack)
            else:
                model.addConstr(
                    generation[start:stop].sum() >= float(lower_energy),
                    name=f"{name}_min_energy[{start}]",
                )
            counts["minimum_generation_energy"] += 1
        if soc is None:
            continue
        if energy_capacity_mwh is None or energy_capacity_mwh <= 0:
            raise ValueError(f"{name}: reservoir level without storage size")
        # The state variable is end-of-hour.  A period starting at hour t>0
        # therefore observes state t-1.  Period 1 is represented by the
        # explicit initial stock, chosen from its released level interval.
        if start == 0:
            continue
        state_index = min(start - 1, soc.shape[0] - 1)
        high = item["max_reservoir_fraction"]
        if high is not None:
            slack = model.addVar(
                lb=0.0,
                obj=penalty_eur_mwh,
                name=f"{name}_max_level_slack[{start}]",
            )
            model.addConstr(
                soc[state_index]
                <= float(high) * energy_capacity_mwh + slack,
                name=f"{name}_max_level[{start}]",
            )
            slacks["level_trajectory"].append(slack)
            counts["maximum_reservoir_level"] += 1
        low = item["min_reservoir_fraction"]
        if low is not None:
            slack = model.addVar(
                lb=0.0,
                obj=penalty_eur_mwh,
                name=f"{name}_min_level_slack[{start}]",
            )
            model.addConstr(
                soc[state_index] + slack
                >= float(low) * energy_capacity_mwh,
                name=f"{name}_min_level[{start}]",
            )
            slacks["level_trajectory"].append(slack)
            counts["minimum_reservoir_level"] += 1
    return slacks, counts


def solve_model(
    system: SystemData,
    storage,
    candidate_pdis: np.ndarray,
    variant: str,
    args: argparse.Namespace,
    baseline_firm: np.ndarray | None,
    baseline_cache_key: str,
    load_increment_mw_by_zone: np.ndarray | None = None,
    fixed_candidate_charge_power_mw_by_zone: np.ndarray | None = None,
) -> SolveResult:
    import gurobipy as gp
    from gurobipy import GRB

    if variant not in {
        "baseline",
        "joint",
        "repeat_use_fixed_firm",
        "adequacy_baseline_fixed_firm",
    }:
        raise ValueError(variant)
    if (
        variant in {"repeat_use_fixed_firm", "adequacy_baseline_fixed_firm"}
        and baseline_firm is None
    ):
        raise ValueError("fixed-firm solve requires cached baseline firm capacity")
    z_count = len(system.zones)
    hours = system.hours
    z_lookup = {zone: index for index, zone in enumerate(system.zones)}
    if load_increment_mw_by_zone is None:
        load_increment = np.zeros(z_count, dtype=float)
    else:
        load_increment = np.asarray(load_increment_mw_by_zone, dtype=float)
        if load_increment.shape != (z_count,):
            raise ValueError("load-increment vector shape mismatch")
        if not np.isfinite(load_increment).all() or (load_increment < 0).any():
            raise ValueError("load-increment vector contains invalid values")
    model = gp.Model(f"ERAA2035_{variant}_{storage.config_id}")
    configure_gurobi(model, args)

    thermal_cost = system.thermal["variable_cost_eur_mwh"].to_numpy()[:, None]
    thermal = model.addMVar(
        (len(system.thermal), hours),
        lb=0.0,
        ub=system.thermal_ub,
        obj=np.broadcast_to(thermal_cost, system.thermal_ub.shape),
        name="thermal",
    )
    must_run = system.must_run.reset_index(drop=True)
    must_run_enabled = (
        system.resource_audit["must_run"]["case"] == "on"
        and not must_run.empty
    )
    standalone_must_run_by_zone_hour = np.zeros((z_count, hours))
    if must_run_enabled:
        for row in must_run.itertuples(index=False):
            indices = np.asarray(row.thermal_indices, dtype=int)
            if bool(row.standalone_fixed_injection):
                standalone_must_run_by_zone_hour[
                    z_lookup[str(row.zone)], :
                ] += float(row.source_must_run_floor_mw)
                continue
            available = system.thermal_ub[indices].sum(axis=0)
            hourly_floor = float(row.ratio) * available
            model.addConstr(
                thermal[indices, :].sum(axis=0)
                >= hourly_floor,
                name=f"must_run_floor[{row.zone},{row.broad_fuel}]",
            )
    demand_mask = np.array(
        [zone in system.demand_zones for zone in system.zones], dtype=float
    )
    if variant in {"repeat_use_fixed_firm", "adequacy_baseline_fixed_firm"}:
        firm_lb = np.asarray(baseline_firm, dtype=float)
        firm_ub = firm_lb.copy()
    else:
        firm_lb = np.zeros(z_count)
        firm_ub = demand_mask * args.maximum_firm_addition_mw
    firm = model.addMVar(
        z_count,
        lb=firm_lb,
        ub=firm_ub,
        obj=np.full(
            z_count, system.firm_annual_cost_eur_kw * 1_000.0
        ),
        name="firm_addition",
    )
    firm_generation = model.addMVar(
        (z_count, hours),
        lb=0.0,
        obj=np.broadcast_to(
            system.firm_variable_cost[:, None], (z_count, hours)
        ),
        name="firm_generation",
    )
    for zone in range(z_count):
        model.addConstr(
            firm_generation[zone, :]
            <= system.firm_availability[zone, :] * firm[zone],
            name=f"firm_availability[{system.zones[zone]}]",
        )

    flow = model.addMVar(
        (len(system.edges), hours),
        lb=0.0,
        ub=system.network_ub,
        obj=args.flow_regularization,
        name="flow",
    )
    vre_spill = model.addMVar(
        (z_count, hours),
        lb=0.0,
        ub=(
            system.wind
            + system.solar
            + system.btm_pv
            + system.csp_storage_reference
        ),
        obj=args.vre_spill_penalty,
        name="vre_spill",
    )
    unserved = model.addMVar(
        (z_count, hours),
        lb=0.0,
        ub=np.broadcast_to(
            system.gross_load
            + load_increment[:, None]
            + args.maximum_firm_addition_mw,
            (z_count, hours),
        ),
        obj=args.value_of_lost_load,
        name="unserved",
    )
    hydro_source_contract_discrepancy_slacks: list[Any] = []
    hydro_level_trajectory_slacks: list[Any] = []
    hydro_contract_counts = {
        "minimum_power": 0,
        "minimum_pumping_power": 0,
        "maximum_generation_energy": 0,
        "minimum_generation_energy": 0,
        "maximum_reservoir_level": 0,
        "minimum_reservoir_level": 0,
    }

    # Run-of-river and zero-stock hydro obey inflow balance and every released
    # turbine-power/energy constraint; excess inflow can be spilled.
    ror = system.ror.reset_index(drop=True)
    rr_count = len(ror)
    rr_generation = model.addMVar(
        (rr_count, hours),
        lb=0.0,
        ub=(
            np.stack(ror["discharge_ub_mw"].to_list())
            if rr_count
            else np.zeros((0, hours))
        ),
        obj=args.hydro_variable_cost,
        name="run_of_river_generation",
    )
    rr_spill = model.addMVar(
        (rr_count, hours),
        lb=0.0,
        obj=args.hydro_spill_penalty,
        name="run_of_river_spill",
    )
    for index, row in enumerate(ror.itertuples(index=False)):
        inflow = np.asarray(row.inflow_mwh, dtype=float)
        model.addConstr(
            rr_generation[index, :] + rr_spill[index, :] == inflow,
            name=f"run_of_river_inflow[{index}]",
        )
        slacks, counts = add_hydro_contract_constraints(
            model=model,
            generation=rr_generation[index, :],
            discharge_lb_mw=np.asarray(row.discharge_lb_mw, dtype=float),
            period_constraints=row.period_constraints,
            penalty_eur_mwh=args.hydro_constraint_violation_penalty,
            name=f"run_of_river[{row.zone}_{row.hydro_code}]",
            soft_nonlevel_source_discrepancy=True,
        )
        hydro_source_contract_discrepancy_slacks.extend(
            slacks["source_contract_discrepancy"]
        )
        hydro_level_trajectory_slacks.extend(slacks["level_trajectory"])
        for key, value in counts.items():
            hydro_contract_counts[key] += value

    # Existing in-market utility batteries.
    battery = system.batteries.reset_index(drop=True)
    b_count = len(battery)
    b_charge = model.addMVar(
        (b_count, hours),
        lb=0.0,
        ub=np.broadcast_to(
            battery["charge_power_mw"].to_numpy()[:, None],
            (b_count, hours),
        ),
        obj=args.existing_storage_throughput_regularization,
        name="existing_battery_charge",
    )
    b_discharge = model.addMVar(
        (b_count, hours),
        lb=0.0,
        ub=np.broadcast_to(
            battery["discharge_power_mw"].to_numpy()[:, None],
            (b_count, hours),
        ),
        obj=(
            args.existing_storage_throughput_regularization
            + args.existing_battery_degradation_cost_eur_mwh_discharged
        ),
        name="existing_battery_discharge",
    )
    b_soc = model.addMVar(
        (b_count, hours),
        lb=0.0,
        ub=np.broadcast_to(
            battery["energy_capacity_mwh"].to_numpy()[:, None],
            (b_count, hours),
        ),
        name="existing_battery_soc",
    )
    for index, row in enumerate(battery.itertuples(index=False)):
        energy = float(row.energy_capacity_mwh)
        model.addConstr(
            b_soc[index, 0]
            - float(row.charge_efficiency) * b_charge[index, 0]
            + b_discharge[index, 0] / float(row.discharge_efficiency)
            == float(row.initial_soc_fraction) * energy,
            name=f"battery_initial[{index}]",
        )
        model.addConstr(
            b_soc[index, 1:]
            - b_soc[index, :-1]
            - float(row.charge_efficiency) * b_charge[index, 1:]
            + b_discharge[index, 1:] / float(row.discharge_efficiency)
            == 0,
            name=f"battery_stock[{index}]",
        )
        model.addConstr(
            b_soc[index, -1] == float(row.terminal_soc_fraction) * energy,
            name=f"battery_terminal[{index}]",
        )

    # The released phase-1 net-load profile embeds the complete out-of-market
    # BTM battery reference trajectory.  The iDSR adapter identifies the
    # price-sensitive share, which is added back to the fixed balance and
    # redispatched here with the released power/energy dimensions.
    idsr_btm = system.idsr_btm_batteries.reset_index(drop=True)
    ib_count = len(idsr_btm)
    ib_charge = model.addMVar(
        (ib_count, hours),
        lb=0.0,
        ub=np.broadcast_to(
            idsr_btm["price_sensitive_charge_power_mw"].to_numpy()[:, None],
            (ib_count, hours),
        ),
        obj=args.existing_storage_throughput_regularization,
        name="idsr_btm_battery_charge",
    )
    ib_discharge = model.addMVar(
        (ib_count, hours),
        lb=0.0,
        ub=np.broadcast_to(
            idsr_btm[
                "price_sensitive_discharge_power_mw"
            ].to_numpy()[:, None],
            (ib_count, hours),
        ),
        obj=(
            args.existing_storage_throughput_regularization
            + args.existing_battery_degradation_cost_eur_mwh_discharged
        ),
        name="idsr_btm_battery_discharge",
    )
    ib_soc = model.addMVar(
        (ib_count, hours),
        lb=0.0,
        ub=np.broadcast_to(
            idsr_btm[
                "price_sensitive_energy_capacity_mwh"
            ].to_numpy()[:, None],
            (ib_count, hours),
        ),
        name="idsr_btm_battery_soc",
    )
    for index, row in enumerate(idsr_btm.itertuples(index=False)):
        energy = float(row.price_sensitive_energy_capacity_mwh)
        eta = math.sqrt(float(row.round_trip_efficiency))
        model.addConstr(
            ib_soc[index, 0]
            - eta * ib_charge[index, 0]
            + ib_discharge[index, 0] / eta
            == float(row.initial_soc_fraction) * energy,
            name=f"idsr_btm_initial[{index}]",
        )
        model.addConstr(
            ib_soc[index, 1:]
            - ib_soc[index, :-1]
            - eta * ib_charge[index, 1:]
            + ib_discharge[index, 1:] / eta
            == 0,
            name=f"idsr_btm_stock[{index}]",
        )
        model.addConstr(
            ib_soc[index, -1]
            == float(row.terminal_soc_fraction) * energy,
            name=f"idsr_btm_terminal[{index}]",
        )

    # Existing pumped hydro with a positive released energy bound.  Records
    # without one remain excluded and are reported in QA, never dispatched as
    # zero-energy resources.
    psp = system.pumped_hydro.reset_index(drop=True)
    p_count = len(psp)
    p_charge_ub = (
        np.stack(psp["charge_ub_mw"].to_list())
        if p_count
        else np.zeros((0, hours))
    )
    p_discharge_ub = (
        np.stack(psp["discharge_ub_mw"].to_list())
        if p_count
        else np.zeros((0, hours))
    )
    p_charge = model.addMVar(
        (p_count, hours),
        lb=0.0,
        ub=p_charge_ub,
        obj=args.existing_storage_throughput_regularization,
        name="psp_charge",
    )
    p_discharge = model.addMVar(
        (p_count, hours),
        lb=0.0,
        ub=p_discharge_ub,
        obj=args.existing_storage_throughput_regularization,
        name="psp_discharge",
    )
    p_soc = model.addMVar(
        (p_count, hours),
        lb=0.0,
        ub=(
            np.broadcast_to(
                psp["energy_capacity_mwh"].to_numpy()[:, None],
                (p_count, hours),
            )
            if p_count
            else np.zeros((0, hours))
        ),
        name="psp_soc",
    )
    p_spill = model.addMVar(
        (p_count, hours),
        lb=0.0,
        obj=args.hydro_spill_penalty,
        name="psp_spill",
    )
    for index, row in enumerate(psp.itertuples(index=False)):
        energy = float(row.energy_capacity_mwh)
        inflow = np.asarray(row.inflow_mwh, dtype=float)
        model.addConstr(
            p_soc[index, 0]
            - float(row.charge_efficiency) * p_charge[index, 0]
            + p_discharge[index, 0] / float(row.discharge_efficiency)
            + p_spill[index, 0]
            == float(row.initial_soc_fraction) * energy + inflow[0],
            name=f"psp_initial[{index}]",
        )
        model.addConstr(
            p_soc[index, 1:]
            - p_soc[index, :-1]
            - float(row.charge_efficiency) * p_charge[index, 1:]
            + p_discharge[index, 1:] / float(row.discharge_efficiency)
            + p_spill[index, 1:]
            == inflow[1:],
            name=f"psp_stock[{index}]",
        )
        model.addConstr(
            p_soc[index, -1] == float(row.terminal_soc_fraction) * energy,
            name=f"psp_terminal[{index}]",
        )
        slacks, counts = add_hydro_contract_constraints(
            model=model,
            generation=p_discharge[index, :],
            discharge_lb_mw=np.asarray(row.discharge_lb_mw, dtype=float),
            period_constraints=row.period_constraints,
            penalty_eur_mwh=args.hydro_constraint_violation_penalty,
            name=f"psp[{row.zone}_{row.hydro_code}]",
            soc=p_soc[index, :],
            energy_capacity_mwh=energy,
            charge=p_charge[index, :],
            charge_lb_mw=np.asarray(row.charge_lb_mw, dtype=float),
        )
        hydro_source_contract_discrepancy_slacks.extend(
            slacks["source_contract_discrepancy"]
        )
        hydro_level_trajectory_slacks.extend(slacks["level_trajectory"])
        for key, value in counts.items():
            hydro_contract_counts[key] += value

    reservoir = system.reservoir.reset_index(drop=True)
    r_count = len(reservoir)
    r_generation = model.addMVar(
        (r_count, hours),
        lb=0.0,
        ub=(
            np.stack(reservoir["discharge_ub_mw"].to_list())
            if r_count
            else np.zeros((0, hours))
        ),
        obj=args.hydro_variable_cost,
        name="reservoir_generation",
    )
    r_soc = model.addMVar(
        (r_count, hours),
        lb=0.0,
        ub=(
            np.broadcast_to(
                reservoir["energy_capacity_mwh"].to_numpy()[:, None],
                (r_count, hours),
            )
            if r_count
            else np.zeros((0, hours))
        ),
        name="reservoir_soc",
    )
    r_spill = model.addMVar(
        (r_count, hours),
        lb=0.0,
        obj=args.hydro_spill_penalty,
        name="reservoir_spill",
    )
    for index, row in enumerate(reservoir.itertuples(index=False)):
        inflow = np.asarray(row.inflow_mwh, dtype=float)
        energy = float(row.energy_capacity_mwh)
        model.addConstr(
            r_soc[index, 0]
            + r_generation[index, 0]
            + r_spill[index, 0]
            == float(row.initial_soc_fraction) * energy + inflow[0],
            name=f"reservoir_initial[{index}]",
        )
        model.addConstr(
            r_soc[index, 1:]
            - r_soc[index, :-1]
            + r_generation[index, 1:]
            + r_spill[index, 1:]
            == inflow[1:],
            name=f"reservoir_stock[{index}]",
        )
        model.addConstr(
            r_soc[index, -1] == float(row.terminal_soc_fraction) * energy,
            name=f"reservoir_terminal[{index}]",
        )
        slacks, counts = add_hydro_contract_constraints(
            model=model,
            generation=r_generation[index, :],
            discharge_lb_mw=np.asarray(row.discharge_lb_mw, dtype=float),
            period_constraints=row.period_constraints,
            penalty_eur_mwh=args.hydro_constraint_violation_penalty,
            name=f"reservoir[{row.zone}_{row.hydro_code}]",
            soc=r_soc[index, :],
            energy_capacity_mwh=energy,
            soft_nonlevel_source_discrepancy=(
                system.hours < HOURS_PER_YEAR
            ),
        )
        hydro_source_contract_discrepancy_slacks.extend(
            slacks["source_contract_discrepancy"]
        )
        hydro_level_trajectory_slacks.extend(slacks["level_trajectory"])
        for key, value in counts.items():
            hydro_contract_counts[key] += value

    dsr = system.dsr.reset_index(drop=True)
    d_count = len(dsr)
    dsr_dispatch = model.addMVar(
        (d_count, hours),
        lb=0.0,
        ub=np.broadcast_to(
            dsr["capacity_mw"].to_numpy()[:, None], (d_count, hours)
        ),
        obj=np.broadcast_to(
            dsr["activation_price_eur_mwh"].to_numpy()[:, None],
            (d_count, hours),
        ),
        name="explicit_dsr",
    )
    days = hours // 24
    for index, row in enumerate(dsr.itertuples(index=False)):
        model.addConstr(
            dsr_dispatch[index, :].reshape((days, 24)).sum(axis=1)
            <= float(row.capacity_mw) * float(row.max_hours_per_day),
            name=f"dsr_daily_hours[{index}]",
        )

    # Official implicit DSR shifts only the price-sensitive EV/HP component.
    # Each ERAA Annex 2 time window conserves energy exactly.
    schedule_rows: list[dict[str, Any]] = []
    for row in system.idsr_schedules.itertuples(index=False):
        zone_index = z_lookup[str(row.zone)]
        baseline = (
            system.idsr_ev_baseline[zone_index]
            if str(row.resource_type) == "EV"
            else system.idsr_hp_baseline[zone_index]
        )
        if baseline.sum() <= REPORT_TOL:
            continue
        window_ids = cyclic_window_ids(
            hours, row.start_hours_utc, row.durations_h
        )
        upper = np.zeros(hours, dtype=float)
        for window in np.unique(window_ids):
            mask = window_ids == window
            upper[mask] = float(baseline[mask].max())
        schedule_rows.append(
            {
                "zone": str(row.zone),
                "resource_type": str(row.resource_type),
                "zone_index": zone_index,
                "baseline": baseline,
                "window_ids": window_ids,
                "upper": upper,
            }
        )
    if schedule_rows:
        idsr_shifted = model.addMVar(
            (len(schedule_rows), hours),
            lb=0.0,
            ub=np.stack([row["upper"] for row in schedule_rows]),
            name="idsr_shifted_load",
        )
        for index, row in enumerate(schedule_rows):
            for window in np.unique(row["window_ids"]):
                positions = np.flatnonzero(row["window_ids"] == window)
                model.addConstr(
                    idsr_shifted[index, positions].sum()
                    == float(np.asarray(row["baseline"])[positions].sum()),
                    name=f"idsr_window_energy[{index},{int(window)}]",
                )
    else:
        idsr_shifted = None

    # Released electrolysers are price-responsive demand.  P2H lacks an
    # hourly heat-availability profile, so zero/nameplate form explicit bounds.
    additional = system.additional_consumers.reset_index(drop=True)
    additional_capacity = additional[
        "availability_upper_bound_mw"
    ].to_numpy(dtype=float)
    additional_lower = additional[
        "availability_lower_bound_mw"
    ].to_numpy(dtype=float)
    p2h = (
        additional["resource_type"]
        .astype(str)
        .eq("power_to_heat")
        .to_numpy()
    )
    if args.power_to_heat_availability_case == "zero":
        additional_capacity[p2h] = 0.0
    elif args.power_to_heat_availability_case != "nameplate":
        raise ValueError(args.power_to_heat_availability_case)
    additional_consumption = model.addMVar(
        (len(additional), hours),
        lb=np.broadcast_to(
            additional_lower[:, None], (len(additional), hours)
        ),
        ub=np.broadcast_to(
            additional_capacity[:, None], (len(additional), hours)
        ),
        obj=-np.broadcast_to(
            additional["activation_price_eur_mwh"].to_numpy(dtype=float)[:, None],
            (len(additional), hours),
        ),
        name="additional_dispatchable_consumption",
    )

    candidate_enabled = variant in {"joint", "repeat_use_fixed_firm"}
    candidate_nodes = np.flatnonzero(candidate_pdis > REPORT_TOL)
    if candidate_enabled:
        pdis = candidate_pdis[candidate_nodes]
        c_count = len(candidate_nodes)
        reference_pcharge = (
            storage.reference_charge_to_discharge_power_ratio * pdis
        )
        maximum_pcharge = (
            storage.maximum_charge_to_discharge_power_ratio * pdis
        )
        if fixed_candidate_charge_power_mw_by_zone is not None:
            fixed_pcharge_all = np.asarray(
                fixed_candidate_charge_power_mw_by_zone, dtype=float
            )
            if fixed_pcharge_all.shape != (z_count,):
                raise ValueError("fixed candidate charge-power shape mismatch")
            fixed_pcharge = fixed_pcharge_all[candidate_nodes]
            if (
                (fixed_pcharge < reference_pcharge - 1e-7).any()
                or (fixed_pcharge > maximum_pcharge + 1e-7).any()
            ):
                raise ValueError(
                    "fixed candidate charge power lies outside design bounds"
                )
            pcharge_lb = fixed_pcharge
            pcharge_ub = fixed_pcharge
        else:
            pcharge_lb = reference_pcharge
            pcharge_ub = maximum_pcharge
        candidate_pcharge = model.addMVar(
            c_count,
            lb=pcharge_lb,
            ub=pcharge_ub,
            name="candidate_charge_power",
        )
        candidate_extra = model.addMVar(
            c_count,
            lb=0.0,
            obj=storage.annual_charge_power_cost_eur_kw * 1_000.0,
            name="candidate_incremental_charge_power",
        )
        model.addConstr(
            candidate_extra
            >= candidate_pcharge - reference_pcharge,
            name="candidate_incremental_interface",
        )
        candidate_charge = model.addMVar(
            (c_count, hours), lb=0.0, name="candidate_charge"
        )
        candidate_discharge = model.addMVar(
            (c_count, hours),
            lb=0.0,
            ub=np.broadcast_to(pdis[:, None], (c_count, hours)),
            obj=(
                storage.variable_om_eur_mwh
                + storage.degradation_cost_eur_mwh_discharged
            ),
            name="candidate_discharge",
        )
        model.addConstr(
            candidate_charge
            <= candidate_pcharge[:, None],
            name="candidate_charge_interface",
        )
        energy = pdis * storage.duration_h
        candidate_soc = model.addMVar(
            (c_count, hours),
            lb=np.broadcast_to(
                (energy * storage.soc_min_fraction)[:, None],
                (c_count, hours),
            ),
            ub=np.broadcast_to(
                (energy * storage.soc_max_fraction)[:, None],
                (c_count, hours),
            ),
            name="candidate_soc",
        )
        model.addConstr(
            candidate_soc[:, 0]
            - storage.hourly_retention * candidate_soc[:, -1]
            - storage.eta_charge * candidate_charge[:, 0]
            + candidate_discharge[:, 0] / storage.eta_discharge
            == 0,
            name="candidate_cycle_initial",
        )
        model.addConstr(
            candidate_soc[:, 1:]
            - storage.hourly_retention * candidate_soc[:, :-1]
            - storage.eta_charge * candidate_charge[:, 1:]
            + candidate_discharge[:, 1:] / storage.eta_discharge
            == 0,
            name="candidate_stock",
        )
        (
            candidate_discharge_power_cost,
            candidate_energy_capacity_cost,
        ) = candidate_fixed_cost_components(
            discharge_power_mw=float(candidate_pdis.sum()),
            duration_h=storage.duration_h,
            annual_discharge_power_cost_eur_kw=(
                storage.annual_discharge_power_cost_eur_kw
            ),
            annual_energy_cost_eur_kwh=(
                storage.annual_energy_cost_eur_kwh
            ),
        )
        candidate_fixed_cost = (
            candidate_discharge_power_cost
            + candidate_energy_capacity_cost
        )
        model.ObjCon = candidate_fixed_cost
    else:
        candidate_pcharge = None
        candidate_extra = None
        candidate_charge = None
        candidate_discharge = None
        candidate_soc = None
        candidate_discharge_power_cost = 0.0
        candidate_energy_capacity_cost = 0.0
        candidate_fixed_cost = 0.0

    thermal_by_zone = {
        zone: system.thermal.index[
            system.thermal["zone"].astype(str).eq(zone)
        ].to_numpy(dtype=int)
        for zone in system.zones
    }
    edges_out = {
        zone: system.edges.index[
            system.edges["from_zone"].astype(str).eq(zone)
        ].to_numpy(dtype=int)
        for zone in system.zones
    }
    edges_in = {
        zone: system.edges.index[
            system.edges["to_zone"].astype(str).eq(zone)
        ].to_numpy(dtype=int)
        for zone in system.zones
    }
    battery_by_zone = {
        zone: battery.index[battery["zone"].astype(str).eq(zone)].to_numpy()
        for zone in system.zones
    }
    idsr_btm_by_zone = {
        zone: idsr_btm.index[
            idsr_btm["zone"].astype(str).eq(zone)
        ].to_numpy()
        for zone in system.zones
    }
    psp_by_zone = {
        zone: psp.index[psp["zone"].astype(str).eq(zone)].to_numpy()
        for zone in system.zones
    }
    ror_by_zone = {
        zone: ror.index[ror["zone"].astype(str).eq(zone)].to_numpy()
        for zone in system.zones
    }
    reservoir_by_zone = {
        zone: reservoir.index[
            reservoir["zone"].astype(str).eq(zone)
        ].to_numpy()
        for zone in system.zones
    }
    dsr_by_zone = {
        zone: dsr.index[dsr["zone"].astype(str).eq(zone)].to_numpy()
        for zone in system.zones
    }
    idsr_by_zone = {
        zone: np.array(
            [
                index
                for index, row in enumerate(schedule_rows)
                if row["zone"] == zone
            ],
            dtype=int,
        )
        for zone in system.zones
    }
    additional_by_zone = {
        zone: additional.index[
            additional["zone"].astype(str).eq(zone)
        ].to_numpy(dtype=int)
        for zone in system.zones
    }
    candidate_position = {
        int(zone_index): position
        for position, zone_index in enumerate(candidate_nodes)
    }
    balance_constraints: list[Any] = []
    for zone_index, zone in enumerate(system.zones):
        expression = (
            firm_generation[zone_index, :]
            + standalone_must_run_by_zone_hour[zone_index, :]
        )
        rows = thermal_by_zone[zone]
        if len(rows):
            expression = expression + thermal[rows, :].sum(axis=0)
        rows = battery_by_zone[zone]
        if len(rows):
            expression = (
                expression
                + b_discharge[rows, :].sum(axis=0)
                - b_charge[rows, :].sum(axis=0)
            )
        rows = idsr_btm_by_zone[zone]
        if len(rows):
            expression = (
                expression
                + ib_discharge[rows, :].sum(axis=0)
                - ib_charge[rows, :].sum(axis=0)
            )
        rows = psp_by_zone[zone]
        if len(rows):
            expression = (
                expression
                + p_discharge[rows, :].sum(axis=0)
                - p_charge[rows, :].sum(axis=0)
            )
        rows = ror_by_zone[zone]
        if len(rows):
            expression = expression + rr_generation[rows, :].sum(axis=0)
        rows = reservoir_by_zone[zone]
        if len(rows):
            expression = expression + r_generation[rows, :].sum(axis=0)
        rows = dsr_by_zone[zone]
        if len(rows):
            expression = expression + dsr_dispatch[rows, :].sum(axis=0)
        if zone_index in candidate_position:
            position = candidate_position[zone_index]
            expression = (
                expression
                + candidate_discharge[position, :]
                - candidate_charge[position, :]
            )
        outgoing = edges_out[zone]
        incoming = edges_in[zone]
        if len(outgoing):
            expression = expression - flow[outgoing, :].sum(axis=0)
        if len(incoming):
            expression = expression + flow[incoming, :].sum(axis=0)
        expression = expression - vre_spill[zone_index, :] + unserved[zone_index, :]
        rhs = (
            system.gross_load[zone_index]
            - system.idsr_ev_baseline[zone_index]
            - system.idsr_hp_baseline[zone_index]
            + load_increment[zone_index]
            - system.wind[zone_index]
            - system.solar[zone_index]
            - system.btm_pv[zone_index]
            - (
                system.btm_battery_reference[zone_index]
                - system.idsr_btm_reference[zone_index]
            )
            - system.csp_storage_reference[zone_index]
        )
        rows = idsr_by_zone[zone]
        if len(rows):
            rhs = rhs + idsr_shifted[rows, :].sum(axis=0)
        rows = additional_by_zone[zone]
        if len(rows):
            rhs = rhs + additional_consumption[rows, :].sum(axis=0)
        balance_constraints.append(
            model.addConstr(expression == rhs, name=f"balance[{zone}]")
        )

    # Released zone-level net import/export limits supplement individual NTCs.
    for (zone, curve_type), group in system.zone_limits.groupby(
        ["zone", "curve_type"]
    ):
        if zone not in z_lookup:
            raise ValueError(f"zone-limit node absent from graph: {zone}")
        values = (
            group.sort_values("hour_of_year")
            .set_index("hour_of_year")["limit_mw"]
            .reindex(range(hours))
        )
        if values.isna().any():
            raise ValueError(f"incomplete zone limit {zone}/{curve_type}")
        outgoing = edges_out[zone]
        incoming = edges_in[zone]
        net_export = (
            flow[outgoing, :].sum(axis=0)
            if len(outgoing)
            else np.zeros(hours)
        )
        if len(incoming):
            net_export = net_export - flow[incoming, :].sum(axis=0)
        if str(curve_type).lower().startswith("net export"):
            model.addConstr(
                net_export <= values.to_numpy(dtype=float),
                name=f"net_export_limit[{zone}]",
            )
        elif str(curve_type).lower().startswith("net import"):
            model.addConstr(
                -net_export <= values.to_numpy(dtype=float),
                name=f"net_import_limit[{zone}]",
            )
        elif str(curve_type).lower().startswith("gross export"):
            gross_export = (
                flow[outgoing, :].sum(axis=0)
                if len(outgoing)
                else np.zeros(hours)
            )
            model.addConstr(
                gross_export <= values.to_numpy(dtype=float),
                name=f"gross_export_limit[{zone}]",
            )
        elif str(curve_type).lower().startswith("gross import"):
            gross_import = (
                flow[incoming, :].sum(axis=0)
                if len(incoming)
                else np.zeros(hours)
            )
            model.addConstr(
                gross_import <= values.to_numpy(dtype=float),
                name=f"gross_import_limit[{zone}]",
            )
        else:
            raise ValueError(f"unknown zone-limit type {curve_type}")

    model.update()
    build_seconds = time.monotonic() - model._creationtime if hasattr(model, "_creationtime") else math.nan
    solve_start = time.monotonic()
    model.optimize()
    solve_seconds = time.monotonic() - solve_start
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(
            f"{variant}/{storage.config_id}: Gurobi status {model.Status}; "
            f"SolCount={model.SolCount}"
        )

    objective = float(model.ObjVal)
    firm_values = firm.X
    expected_eens = float(unserved.X.sum())
    system_unserved = unserved.X.sum(axis=0)
    expected_lole = float(np.count_nonzero(system_unserved > REPORT_TOL))
    thermal_generation = thermal.X.sum(axis=1)
    fossil_gas_by_zone_hour = np.asarray(
        firm_generation.X, dtype=float
    ).copy()
    for thermal_index, thermal_row in enumerate(
        system.thermal.itertuples(index=False)
    ):
        if str(thermal_row.technology) in {"gas_ccgt", "gas_ocgt"}:
            fossil_gas_by_zone_hour[
                z_lookup[str(thermal_row.zone)]
            ] += thermal.X[thermal_index]
    fossil_gas_system_hour = fossil_gas_by_zone_hour.sum(axis=0)
    common_residual_by_zone_hour = (
        system.gross_load
        - system.wind
        - system.solar
        - system.btm_pv
        - system.btm_battery_reference
        - system.csp_storage_reference
    )
    common_residual_system_hour = common_residual_by_zone_hour.sum(axis=0)
    q75_common_residual_mw = float(
        np.quantile(common_residual_system_hour, 0.75)
    )
    q75_common_residual_mask = (
        common_residual_system_hour >= q75_common_residual_mw
    )
    thermal_by_technology = (
        pd.DataFrame(
            {
                "technology": system.thermal["technology"].to_numpy(),
                "generation_mwh": thermal_generation,
            }
        )
        .groupby("technology")["generation_mwh"]
        .sum()
        .to_dict()
    )
    firm_generation_mwh = float(firm_generation.X.sum())
    thermal_variable_cost = float(
        np.sum(thermal.X * thermal_cost)
        + np.sum(firm_generation.X * system.firm_variable_cost[:, None])
    )
    firm_capacity_cost = float(
        firm_values.sum()
        * system.firm_annual_cost_eur_kw
        * 1_000.0
    )
    dsr_cost = float(
        np.sum(
            dsr_dispatch.X
            * dsr["activation_price_eur_mwh"].to_numpy()[:, None]
        )
    )
    idsr_baseline_mwh = float(
        sum(np.asarray(row["baseline"], dtype=float).sum() for row in schedule_rows)
    )
    idsr_shifted_mwh = (
        float(idsr_shifted.X.sum()) if idsr_shifted is not None else 0.0
    )
    idsr_shift_volume_mwh = (
        0.5
        * float(
            sum(
                np.abs(
                    idsr_shifted.X[index]
                    - np.asarray(row["baseline"], dtype=float)
                ).sum()
                for index, row in enumerate(schedule_rows)
            )
        )
        if idsr_shifted is not None
        else 0.0
    )
    idsr_energy_residual_mwh = abs(idsr_shifted_mwh - idsr_baseline_mwh)
    additional_value_cost = -float(
        np.sum(
            additional_consumption.X
            * additional["activation_price_eur_mwh"].to_numpy(dtype=float)[:, None]
        )
    )
    additional_by_type = (
        pd.DataFrame(
            {
                "resource_type": additional["resource_type"].astype(str),
                "consumption_mwh": additional_consumption.X.sum(axis=1),
            }
        )
        .groupby("resource_type")["consumption_mwh"]
        .sum()
        .to_dict()
    )
    candidate_charge_mwh = (
        float(candidate_charge.X.sum()) if candidate_enabled else 0.0
    )
    candidate_discharge_mwh = (
        float(candidate_discharge.X.sum()) if candidate_enabled else 0.0
    )
    candidate_variable_cost = (
        candidate_discharge_mwh
        * (
            storage.variable_om_eur_mwh
            + storage.degradation_cost_eur_mwh_discharged
        )
        if candidate_enabled
        else 0.0
    )
    candidate_charge_cost = (
        float(candidate_extra.X.sum())
        * storage.annual_charge_power_cost_eur_kw
        * 1_000.0
        if candidate_enabled
        else 0.0
    )
    hydro_source_contract_discrepancy_mwh = float(
        sum(
            np.asarray(item.X, dtype=float).sum()
            for item in hydro_source_contract_discrepancy_slacks
        )
    )
    hydro_level_trajectory_soft_slack_mwh = float(
        sum(
            np.asarray(item.X, dtype=float).sum()
            for item in hydro_level_trajectory_slacks
        )
    )
    hydro_contract_violation_mwh = (
        hydro_source_contract_discrepancy_mwh
        + hydro_level_trajectory_soft_slack_mwh
    )
    hydro_contract_official_ror_structural_lower_bound_mwh = float(
        system.resource_audit["hydro"][
            "official_ror_structural_lower_bound_mwh"
        ]
    )
    hydro_contract_development_terminal_boundary_lower_bound_mwh = float(
        system.resource_audit["hydro"][
            "development_terminal_boundary_lower_bound_mwh"
        ]
    )
    hydro_contract_admissible_lower_bound_mwh = float(
        system.resource_audit["hydro"][
            "admissible_hydro_contract_slack_lower_bound_mwh"
        ]
    )
    hydro_contract_excess_slack_mwh = max(
        hydro_source_contract_discrepancy_mwh
        - hydro_contract_admissible_lower_bound_mwh,
        0.0,
    )
    hydro_output_mwh = float(
        rr_generation.X.sum()
        + r_generation.X.sum()
        + p_discharge.X.sum()
    )
    hydro_contract_violation_cost = (
        hydro_contract_violation_mwh
        * args.hydro_constraint_violation_penalty
    )
    hydro_source_contract_discrepancy_cost = (
        hydro_source_contract_discrepancy_mwh
        * args.hydro_constraint_violation_penalty
    )
    hydro_level_trajectory_soft_slack_cost = (
        hydro_level_trajectory_soft_slack_mwh
        * args.hydro_constraint_violation_penalty
    )

    def hydro_slack_records(
        items: list[Any], constraint_class: str
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for item in items:
            values = np.asarray(item.X, dtype=float)
            if values.ndim == 0:
                variables = [item]
                values = values.reshape(1)
            else:
                variables = np.asarray(item.tolist(), dtype=object).reshape(-1)
                values = values.reshape(-1)
            for variable, value in zip(variables, values):
                if float(value) <= REPORT_TOL:
                    continue
                variable_name = str(variable.VarName)
                if variable_name.startswith("run_of_river["):
                    resource_type = "run_of_river_or_zero_stock"
                elif variable_name.startswith("reservoir["):
                    resource_type = "reservoir"
                elif variable_name.startswith("psp["):
                    resource_type = "pumped_hydro"
                else:
                    resource_type = "unparsed"
                resource_id = ""
                if "[" in variable_name and "]" in variable_name:
                    resource_id = variable_name.split("[", 1)[1].split("]", 1)[0]
                if "_" in resource_id:
                    zone, hydro_code = resource_id.rsplit("_", 1)
                else:
                    zone, hydro_code = "", ""
                records.append(
                    {
                        "variant": variant,
                        "constraint_class": constraint_class,
                        "resource_type": resource_type,
                        "zone": zone,
                        "hydro_code": hydro_code,
                        "variable_name": variable_name,
                        "slack_mwh": float(value),
                    }
                )
        return records

    hydro_slack_ledger = pd.DataFrame(
        hydro_slack_records(
            hydro_source_contract_discrepancy_slacks,
            "source_contract_discrepancy",
        )
        + hydro_slack_records(
            hydro_level_trajectory_slacks,
            "soft_level_trajectory",
        ),
        columns=[
            "variant",
            "constraint_class",
            "resource_type",
            "zone",
            "hydro_code",
            "variable_name",
            "slack_mwh",
        ],
    )
    hydro_slack_ledger_residual_mwh = abs(
        float(hydro_slack_ledger["slack_mwh"].sum())
        - hydro_contract_violation_mwh
    )
    existing_storage_throughput_cost = float(
        (
            b_charge.X.sum()
            + b_discharge.X.sum()
            + ib_charge.X.sum()
            + ib_discharge.X.sum()
            + p_charge.X.sum()
            + p_discharge.X.sum()
        )
        * args.existing_storage_throughput_regularization
    )
    existing_battery_degradation_cost = float(
        (b_discharge.X.sum() + ib_discharge.X.sum())
        * args.existing_battery_degradation_cost_eur_mwh_discharged
    )

    def maximum_absolute(chunks: Iterable[np.ndarray]) -> float:
        maxima = [
            float(np.max(np.abs(np.asarray(chunk, dtype=float))))
            for chunk in chunks
            if np.asarray(chunk).size
        ]
        return max(maxima, default=0.0)

    # Reconstruct the principal physical equalities and inequalities from the
    # primal solution.  These diagnostics are independent of Gurobi's global
    # quality attributes and therefore identify which physical family, if
    # any, is responsible for a numerical violation.
    thermal_values = np.maximum(np.asarray(thermal.X, dtype=float), 0.0)
    firm_generation_values = np.maximum(
        np.asarray(firm_generation.X, dtype=float), 0.0
    )
    flow_values = np.maximum(np.asarray(flow.X, dtype=float), 0.0)
    balance_residual_chunks: list[np.ndarray] = []
    for zone_index, zone in enumerate(system.zones):
        reconstructed = (
            firm_generation_values[zone_index]
            + standalone_must_run_by_zone_hour[zone_index]
        )
        rows = thermal_by_zone[zone]
        if len(rows):
            reconstructed += thermal_values[rows].sum(axis=0)
        rows = battery_by_zone[zone]
        if len(rows):
            reconstructed += (
                np.asarray(b_discharge.X[rows], dtype=float).sum(axis=0)
                - np.asarray(b_charge.X[rows], dtype=float).sum(axis=0)
            )
        rows = idsr_btm_by_zone[zone]
        if len(rows):
            reconstructed += (
                np.asarray(ib_discharge.X[rows], dtype=float).sum(axis=0)
                - np.asarray(ib_charge.X[rows], dtype=float).sum(axis=0)
            )
        rows = psp_by_zone[zone]
        if len(rows):
            reconstructed += (
                np.asarray(p_discharge.X[rows], dtype=float).sum(axis=0)
                - np.asarray(p_charge.X[rows], dtype=float).sum(axis=0)
            )
        rows = ror_by_zone[zone]
        if len(rows):
            reconstructed += np.asarray(
                rr_generation.X[rows], dtype=float
            ).sum(axis=0)
        rows = reservoir_by_zone[zone]
        if len(rows):
            reconstructed += np.asarray(
                r_generation.X[rows], dtype=float
            ).sum(axis=0)
        rows = dsr_by_zone[zone]
        if len(rows):
            reconstructed += np.asarray(
                dsr_dispatch.X[rows], dtype=float
            ).sum(axis=0)
        if candidate_enabled and zone_index in candidate_position:
            position = candidate_position[zone_index]
            reconstructed += (
                np.asarray(candidate_discharge.X[position], dtype=float)
                - np.asarray(candidate_charge.X[position], dtype=float)
            )
        outgoing = edges_out[zone]
        incoming = edges_in[zone]
        if len(outgoing):
            reconstructed -= flow_values[outgoing].sum(axis=0)
        if len(incoming):
            reconstructed += flow_values[incoming].sum(axis=0)
        reconstructed -= np.asarray(vre_spill.X[zone_index], dtype=float)
        reconstructed += np.asarray(unserved.X[zone_index], dtype=float)
        rhs = (
            system.gross_load[zone_index]
            - system.idsr_ev_baseline[zone_index]
            - system.idsr_hp_baseline[zone_index]
            + load_increment[zone_index]
            - system.wind[zone_index]
            - system.solar[zone_index]
            - system.btm_pv[zone_index]
            - (
                system.btm_battery_reference[zone_index]
                - system.idsr_btm_reference[zone_index]
            )
            - system.csp_storage_reference[zone_index]
        )
        rows = idsr_by_zone[zone]
        if len(rows):
            rhs = rhs + np.asarray(
                idsr_shifted.X[rows], dtype=float
            ).sum(axis=0)
        rows = additional_by_zone[zone]
        if len(rows):
            rhs = rhs + np.asarray(
                additional_consumption.X[rows], dtype=float
            ).sum(axis=0)
        balance_residual_chunks.append(reconstructed - rhs)
    maximum_nodal_balance_residual_mw = maximum_absolute(
        balance_residual_chunks
    )

    battery_soc_residuals: list[np.ndarray] = []
    for index, row in enumerate(battery.itertuples(index=False)):
        energy = float(row.energy_capacity_mwh)
        stock = np.asarray(b_soc.X[index], dtype=float)
        charge = np.asarray(b_charge.X[index], dtype=float)
        discharge = np.asarray(b_discharge.X[index], dtype=float)
        battery_soc_residuals.extend(
            [
                np.array(
                    [
                        stock[0]
                        - float(row.charge_efficiency) * charge[0]
                        + discharge[0] / float(row.discharge_efficiency)
                        - float(row.initial_soc_fraction) * energy
                    ]
                ),
                (
                    stock[1:]
                    - stock[:-1]
                    - float(row.charge_efficiency) * charge[1:]
                    + discharge[1:] / float(row.discharge_efficiency)
                ),
                np.array(
                    [
                        stock[-1]
                        - float(row.terminal_soc_fraction) * energy
                    ]
                ),
            ]
        )
    maximum_existing_battery_soc_residual_mwh = maximum_absolute(
        battery_soc_residuals
    )

    idsr_btm_soc_residuals: list[np.ndarray] = []
    for index, row in enumerate(idsr_btm.itertuples(index=False)):
        energy = float(row.price_sensitive_energy_capacity_mwh)
        eta = math.sqrt(float(row.round_trip_efficiency))
        stock = np.asarray(ib_soc.X[index], dtype=float)
        charge = np.asarray(ib_charge.X[index], dtype=float)
        discharge = np.asarray(ib_discharge.X[index], dtype=float)
        idsr_btm_soc_residuals.extend(
            [
                np.array(
                    [
                        stock[0]
                        - eta * charge[0]
                        + discharge[0] / eta
                        - float(row.initial_soc_fraction) * energy
                    ]
                ),
                stock[1:]
                - stock[:-1]
                - eta * charge[1:]
                + discharge[1:] / eta,
                np.array(
                    [
                        stock[-1]
                        - float(row.terminal_soc_fraction) * energy
                    ]
                ),
            ]
        )
    maximum_idsr_btm_battery_soc_residual_mwh = maximum_absolute(
        idsr_btm_soc_residuals
    )

    psp_soc_residuals: list[np.ndarray] = []
    for index, row in enumerate(psp.itertuples(index=False)):
        energy = float(row.energy_capacity_mwh)
        inflow = np.asarray(row.inflow_mwh, dtype=float)
        stock = np.asarray(p_soc.X[index], dtype=float)
        charge = np.asarray(p_charge.X[index], dtype=float)
        discharge = np.asarray(p_discharge.X[index], dtype=float)
        spill = np.asarray(p_spill.X[index], dtype=float)
        psp_soc_residuals.extend(
            [
                np.array(
                    [
                        stock[0]
                        - float(row.charge_efficiency) * charge[0]
                        + discharge[0] / float(row.discharge_efficiency)
                        + spill[0]
                        - float(row.initial_soc_fraction) * energy
                        - inflow[0]
                    ]
                ),
                (
                    stock[1:]
                    - stock[:-1]
                    - float(row.charge_efficiency) * charge[1:]
                    + discharge[1:] / float(row.discharge_efficiency)
                    + spill[1:]
                    - inflow[1:]
                ),
                np.array(
                    [
                        stock[-1]
                        - float(row.terminal_soc_fraction) * energy
                    ]
                ),
            ]
        )
    maximum_pumped_hydro_soc_residual_mwh = maximum_absolute(
        psp_soc_residuals
    )

    reservoir_soc_residuals: list[np.ndarray] = []
    for index, row in enumerate(reservoir.itertuples(index=False)):
        energy = float(row.energy_capacity_mwh)
        inflow = np.asarray(row.inflow_mwh, dtype=float)
        stock = np.asarray(r_soc.X[index], dtype=float)
        generation = np.asarray(r_generation.X[index], dtype=float)
        spill = np.asarray(r_spill.X[index], dtype=float)
        reservoir_soc_residuals.extend(
            [
                np.array(
                    [
                        stock[0]
                        + generation[0]
                        + spill[0]
                        - float(row.initial_soc_fraction) * energy
                        - inflow[0]
                    ]
                ),
                (
                    stock[1:]
                    - stock[:-1]
                    + generation[1:]
                    + spill[1:]
                    - inflow[1:]
                ),
                np.array(
                    [
                        stock[-1]
                        - float(row.terminal_soc_fraction) * energy
                    ]
                ),
            ]
        )
    maximum_reservoir_soc_residual_mwh = maximum_absolute(
        reservoir_soc_residuals
    )

    run_of_river_residuals = [
        np.asarray(rr_generation.X[index], dtype=float)
        + np.asarray(rr_spill.X[index], dtype=float)
        - np.asarray(row.inflow_mwh, dtype=float)
        for index, row in enumerate(ror.itertuples(index=False))
    ]
    maximum_run_of_river_balance_residual_mwh = maximum_absolute(
        run_of_river_residuals
    )

    candidate_soc_residuals: list[np.ndarray] = []
    if candidate_enabled:
        stock = np.asarray(candidate_soc.X, dtype=float)
        charge = np.asarray(candidate_charge.X, dtype=float)
        discharge = np.asarray(candidate_discharge.X, dtype=float)
        candidate_soc_residuals.extend(
            [
                (
                    stock[:, 0]
                    - storage.hourly_retention * stock[:, -1]
                    - storage.eta_charge * charge[:, 0]
                    + discharge[:, 0] / storage.eta_discharge
                ),
                (
                    stock[:, 1:]
                    - storage.hourly_retention * stock[:, :-1]
                    - storage.eta_charge * charge[:, 1:]
                    + discharge[:, 1:] / storage.eta_discharge
                ),
            ]
        )
    maximum_candidate_soc_residual_mwh = maximum_absolute(
        candidate_soc_residuals
    )

    edge_violation = np.maximum(flow_values - system.network_ub, 0.0)
    zone_limit_violations: list[np.ndarray] = []
    for (zone, curve_type), group in system.zone_limits.groupby(
        ["zone", "curve_type"]
    ):
        values = (
            group.sort_values("hour_of_year")
            .set_index("hour_of_year")["limit_mw"]
            .reindex(range(hours))
            .to_numpy(dtype=float)
        )
        outgoing = edges_out[str(zone)]
        incoming = edges_in[str(zone)]
        net_export = np.zeros(hours)
        if len(outgoing):
            net_export += flow_values[outgoing].sum(axis=0)
        if len(incoming):
            net_export -= flow_values[incoming].sum(axis=0)
        if str(curve_type).lower().startswith("net export"):
            zone_limit_violations.append(np.maximum(net_export - values, 0.0))
        elif str(curve_type).lower().startswith("net import"):
            zone_limit_violations.append(np.maximum(-net_export - values, 0.0))
        elif str(curve_type).lower().startswith("gross export"):
            gross_export = (
                flow_values[outgoing].sum(axis=0)
                if len(outgoing)
                else np.zeros(hours)
            )
            zone_limit_violations.append(
                np.maximum(gross_export - values, 0.0)
            )
        elif str(curve_type).lower().startswith("gross import"):
            gross_import = (
                flow_values[incoming].sum(axis=0)
                if len(incoming)
                else np.zeros(hours)
            )
            zone_limit_violations.append(
                np.maximum(gross_import - values, 0.0)
            )
        else:
            raise ValueError(f"unknown zone-limit type {curve_type}")
    maximum_network_constraint_violation_mw = max(
        maximum_absolute([edge_violation]),
        maximum_absolute(zone_limit_violations),
    )

    dsr_violations: list[np.ndarray] = []
    for index, row in enumerate(dsr.itertuples(index=False)):
        dispatch = np.maximum(
            np.asarray(dsr_dispatch.X[index], dtype=float), 0.0
        )
        dsr_violations.append(
            np.maximum(
                dispatch.reshape((days, 24)).sum(axis=1)
                - float(row.capacity_mw) * float(row.max_hours_per_day),
                0.0,
            )
        )
    maximum_dsr_daily_limit_violation_mwh = maximum_absolute(dsr_violations)

    must_run_violations: list[np.ndarray] = []
    if must_run_enabled:
        for row in must_run.itertuples(index=False):
            if bool(row.standalone_fixed_injection):
                continue
            indices = np.asarray(row.thermal_indices, dtype=int)
            available = system.thermal_ub[indices].sum(axis=0)
            generation = thermal_values[indices].sum(axis=0)
            must_run_violations.append(
                np.maximum(float(row.ratio) * available - generation, 0.0)
            )
    maximum_must_run_floor_violation_mw = maximum_absolute(
        must_run_violations
    )

    idsr_window_residuals: list[np.ndarray] = []
    if idsr_shifted is not None:
        for index, row in enumerate(schedule_rows):
            for window in np.unique(row["window_ids"]):
                positions = np.flatnonzero(row["window_ids"] == window)
                idsr_window_residuals.append(
                    np.array(
                        [
                            np.asarray(
                                idsr_shifted.X[index, positions],
                                dtype=float,
                            ).sum()
                            - np.asarray(
                                row["baseline"], dtype=float
                            )[positions].sum()
                        ]
                    )
                )
    maximum_idsr_window_energy_residual_mwh = maximum_absolute(
        idsr_window_residuals
    )

    firm_availability_violation = np.maximum(
        firm_generation_values
        - system.firm_availability * np.asarray(firm.X)[:, None],
        0.0,
    )
    maximum_firm_availability_violation_mw = maximum_absolute(
        [firm_availability_violation]
    )
    solver_maximum_constraint_violation = float(
        getattr(model, "ConstrVio", math.nan)
    )
    solver_maximum_bound_violation = float(
        getattr(model, "BoundVio", math.nan)
    )

    def overlap_diagnostic(
        charge_values: np.ndarray, discharge_values: np.ndarray
    ) -> tuple[float, float, float]:
        nonnegative_charge = np.maximum(
            np.asarray(charge_values, dtype=float), 0.0
        )
        nonnegative_discharge = np.maximum(
            np.asarray(discharge_values, dtype=float), 0.0
        )
        simultaneous_overlap = float(
            np.minimum(
                np.maximum(np.asarray(charge_values, dtype=float), 0.0),
                np.maximum(np.asarray(discharge_values, dtype=float), 0.0),
            ).sum()
        )
        throughput = float(
            (nonnegative_charge + nonnegative_discharge).sum()
        )
        ratio = (
            simultaneous_overlap / throughput
            if throughput > REPORT_TOL
            else 0.0
        )
        return simultaneous_overlap, throughput, ratio

    (
        simultaneous_battery,
        throughput_battery,
        simultaneous_ratio_battery,
    ) = overlap_diagnostic(b_charge.X, b_discharge.X)
    (
        simultaneous_idsr_btm,
        throughput_idsr_btm,
        simultaneous_ratio_idsr_btm,
    ) = overlap_diagnostic(ib_charge.X, ib_discharge.X)
    (
        simultaneous_psp,
        throughput_psp,
        simultaneous_ratio_psp,
    ) = overlap_diagnostic(p_charge.X, p_discharge.X)
    if candidate_enabled:
        (
            simultaneous_candidate,
            throughput_candidate,
            simultaneous_ratio_candidate,
        ) = overlap_diagnostic(candidate_charge.X, candidate_discharge.X)
    else:
        simultaneous_candidate = 0.0
        throughput_candidate = 0.0
        simultaneous_ratio_candidate = 0.0
    objective_components = {
        "firm_capacity_cost_eur_year": firm_capacity_cost,
        "thermal_variable_cost_eur_year": thermal_variable_cost,
        "explicit_dsr_activation_cost_eur_year": dsr_cost,
        "additional_dispatchable_consumption_net_value_eur_year": (
            additional_value_cost
        ),
        "existing_storage_throughput_regularization_cost_eur_year": (
            existing_storage_throughput_cost
        ),
        "existing_battery_degradation_cost_eur_year": (
            existing_battery_degradation_cost
        ),
        "candidate_discharge_power_cost_eur_year": (
            candidate_discharge_power_cost
        ),
        "candidate_energy_capacity_cost_eur_year": (
            candidate_energy_capacity_cost
        ),
        "candidate_incremental_charge_interface_cost_eur_year": candidate_charge_cost,
        "candidate_variable_and_degradation_cost_eur_year": candidate_variable_cost,
        "unserved_energy_cost_eur_year": expected_eens
        * args.value_of_lost_load,
        "vre_spill_cost_eur_year": float(vre_spill.X.sum())
        * args.vre_spill_penalty,
        "hydro_spill_cost_eur_year": float(
            rr_spill.X.sum() + p_spill.X.sum() + r_spill.X.sum()
        )
        * args.hydro_spill_penalty,
        "hydro_variable_cost_eur_year": float(
            rr_generation.X.sum() + r_generation.X.sum()
        )
        * args.hydro_variable_cost,
        "hydro_source_contract_discrepancy_penalty_eur_year": (
            hydro_source_contract_discrepancy_cost
        ),
        "hydro_level_trajectory_soft_penalty_eur_year": (
            hydro_level_trajectory_soft_slack_cost
        ),
        "flow_regularization_cost_eur_year": float(flow.X.sum())
        * args.flow_regularization,
    }
    component_error = abs(sum(objective_components.values()) - objective)
    if component_error > max(1e-3, 1e-8 * abs(objective)):
        raise RuntimeError(
            f"objective component audit failed: {component_error} EUR"
        )
    economic_objective_excluding_hydro_feasibility_penalties = (
        objective - hydro_contract_violation_cost
    )

    balance_duals = np.vstack(
        [np.asarray(constraint.Pi, dtype=float) for constraint in balance_constraints]
    )
    charging_opportunity_cost = 0.0
    if candidate_enabled:
        for position, zone_index in enumerate(candidate_nodes):
            charging_opportunity_cost += float(
                np.sum(
                    candidate_charge.X[position]
                    * balance_duals[zone_index]
                )
            )
    total_pdis = float(candidate_pdis.sum()) if candidate_enabled else 0.0
    usable_energy = (
        total_pdis
        * storage.duration_h
        * (storage.soc_max_fraction - storage.soc_min_fraction)
    )
    rated_energy = total_pdis * storage.duration_h
    metrics = {
        "variant": variant,
        "config_id": storage.config_id if candidate_enabled else "__baseline__",
        "technology": storage.tech_id if candidate_enabled else "__none__",
        "duration_h": storage.duration_h if candidate_enabled else 0.0,
        "objective_eur_year": objective,
        "economic_objective_excluding_hydro_feasibility_penalties_eur_year": (
            economic_objective_excluding_hydro_feasibility_penalties
        ),
        "reliability_evidence_class": (
            "deterministic_expected_availability_screen"
        ),
        "deterministic_expected_availability_unserved_energy_mwh": (
            expected_eens
        ),
        "deterministic_expected_availability_loss_of_load_hours": (
            expected_lole
        ),
        "expected_eens_mwh": expected_eens,
        "expected_lole_h": expected_lole,
        "firm_addition_mw": float(firm_values.sum()),
        "firm_generation_mwh": firm_generation_mwh,
        "gas_generation_mwh": float(
            thermal_by_technology.get("gas_ccgt", 0.0)
            + thermal_by_technology.get("gas_ocgt", 0.0)
            + firm_generation_mwh
        ),
        "simulated_fossil_gas_generation_mwh": float(
            fossil_gas_system_hour.sum()
        ),
        "simulated_fossil_gas_generation_peak_mw": float(
            fossil_gas_system_hour.max(initial=0.0)
        ),
        "fossil_gas_output_data_role": ERAA_FOSSIL_GAS_DATA_ROLE,
        "fossil_gas_is_observed_actual": False,
        "q75_common_residual_demand_threshold_mw": (
            q75_common_residual_mw
        ),
        "simulated_fossil_gas_generation_in_q75_common_residual_hours_mwh": (
            float(fossil_gas_system_hour[q75_common_residual_mask].sum())
        ),
        "simulated_fossil_gas_q75_hour_count": int(
            q75_common_residual_mask.sum()
        ),
        "fossil_gas_start_count_available": False,
        "fossil_gas_start_count": math.nan,
        "hydrogen_generation_mwh": float(
            thermal_by_technology.get("hydrogen_ccgt", 0.0)
            + thermal_by_technology.get("hydrogen_ocgt", 0.0)
        ),
        "must_run_case": system.resource_audit["must_run"]["case"],
        "must_run_mapped_group_count": len(must_run),
        "must_run_floor_mwh": float(
            must_run["annual_must_run_floor_mwh"].sum()
            if must_run_enabled
            else 0.0
        ),
        "standalone_inelastic_must_run_generation_mwh": float(
            standalone_must_run_by_zone_hour.sum()
        ),
        "existing_battery_charge_mwh": float(b_charge.X.sum()),
        "existing_battery_discharge_mwh": float(b_discharge.X.sum()),
        "idsr_btm_battery_charge_mwh": float(ib_charge.X.sum()),
        "idsr_btm_battery_discharge_mwh": float(ib_discharge.X.sum()),
        "pumped_hydro_charge_mwh": float(p_charge.X.sum()),
        "pumped_hydro_discharge_mwh": float(p_discharge.X.sum()),
        "run_of_river_generation_mwh": float(rr_generation.X.sum()),
        "run_of_river_spill_mwh": float(rr_spill.X.sum()),
        "reservoir_generation_mwh": float(r_generation.X.sum()),
        "reservoir_spill_mwh": float(r_spill.X.sum()),
        "hydro_contract_violation_mwh": hydro_contract_violation_mwh,
        "hydro_source_contract_discrepancy_mwh": (
            hydro_source_contract_discrepancy_mwh
        ),
        "hydro_level_trajectory_soft_slack_mwh": (
            hydro_level_trajectory_soft_slack_mwh
        ),
        "hydro_constraint_violation_cost_eur_year": (
            hydro_contract_violation_cost
        ),
        "hydro_positive_slack_ledger_rows": int(len(hydro_slack_ledger)),
        "hydro_slack_ledger_residual_mwh": (
            hydro_slack_ledger_residual_mwh
        ),
        "hydro_contract_official_ror_structural_lower_bound_mwh": (
            hydro_contract_official_ror_structural_lower_bound_mwh
        ),
        "hydro_contract_development_terminal_boundary_lower_bound_mwh": (
            hydro_contract_development_terminal_boundary_lower_bound_mwh
        ),
        "hydro_contract_admissible_lower_bound_mwh": (
            hydro_contract_admissible_lower_bound_mwh
        ),
        "hydro_contract_excess_slack_mwh": (
            hydro_contract_excess_slack_mwh
        ),
        "hydro_output_mwh": hydro_output_mwh,
        "hydro_contract_violation_share_of_hydro_output": (
            hydro_contract_violation_mwh / hydro_output_mwh
            if hydro_output_mwh > REPORT_TOL
            else math.nan
        ),
        "hydro_contract_constraint_count": int(
            sum(hydro_contract_counts.values())
        ),
        **{
            f"hydro_contract_{key}_count": int(value)
            for key, value in hydro_contract_counts.items()
        },
        "explicit_dsr_dispatch_mwh": float(dsr_dispatch.X.sum()),
        "idsr_baseline_flexible_mwh": idsr_baseline_mwh,
        "idsr_rescheduled_mwh": idsr_shifted_mwh,
        "idsr_shift_volume_mwh": idsr_shift_volume_mwh,
        "idsr_energy_conservation_residual_mwh": (
            idsr_energy_residual_mwh
        ),
        "maximum_idsr_window_energy_residual_mwh": (
            maximum_idsr_window_energy_residual_mwh
        ),
        "electrolyser_consumption_mwh": float(
            additional_by_type.get("electrolyser_optional_buyer", 0.0)
        ),
        "power_to_heat_consumption_mwh": float(
            additional_by_type.get("power_to_heat", 0.0)
        ),
        "power_to_heat_availability_case": (
            args.power_to_heat_availability_case
        ),
        "existing_battery_simultaneous_charge_discharge_mwh": (
            simultaneous_battery
        ),
        "idsr_btm_battery_simultaneous_charge_discharge_mwh": (
            simultaneous_idsr_btm
        ),
        "pumped_hydro_simultaneous_charge_discharge_mwh": simultaneous_psp,
        "candidate_simultaneous_charge_discharge_mwh": (
            simultaneous_candidate
        ),
        "maximum_simultaneous_charge_discharge_mwh": max(
            simultaneous_battery,
            simultaneous_idsr_btm,
            simultaneous_psp,
            simultaneous_candidate,
        ),
        "existing_battery_charge_discharge_throughput_mwh": (
            throughput_battery
        ),
        "idsr_btm_battery_charge_discharge_throughput_mwh": (
            throughput_idsr_btm
        ),
        "pumped_hydro_charge_discharge_throughput_mwh": throughput_psp,
        "candidate_charge_discharge_throughput_mwh": throughput_candidate,
        "existing_battery_simultaneous_charge_discharge_to_throughput_ratio": (
            simultaneous_ratio_battery
        ),
        "idsr_btm_battery_simultaneous_charge_discharge_to_throughput_ratio": (
            simultaneous_ratio_idsr_btm
        ),
        "pumped_hydro_simultaneous_charge_discharge_to_throughput_ratio": (
            simultaneous_ratio_psp
        ),
        "candidate_simultaneous_charge_discharge_to_throughput_ratio": (
            simultaneous_ratio_candidate
        ),
        "maximum_simultaneous_charge_discharge_ratio": max(
            simultaneous_ratio_battery,
            simultaneous_ratio_idsr_btm,
            simultaneous_ratio_psp,
            simultaneous_ratio_candidate,
        ),
        "maximum_nodal_balance_residual_mw": (
            maximum_nodal_balance_residual_mw
        ),
        "maximum_candidate_soc_residual_mwh": (
            maximum_candidate_soc_residual_mwh
        ),
        "maximum_existing_battery_soc_residual_mwh": (
            maximum_existing_battery_soc_residual_mwh
        ),
        "maximum_idsr_btm_battery_soc_residual_mwh": (
            maximum_idsr_btm_battery_soc_residual_mwh
        ),
        "maximum_pumped_hydro_soc_residual_mwh": (
            maximum_pumped_hydro_soc_residual_mwh
        ),
        "maximum_reservoir_soc_residual_mwh": (
            maximum_reservoir_soc_residual_mwh
        ),
        "maximum_run_of_river_balance_residual_mwh": (
            maximum_run_of_river_balance_residual_mwh
        ),
        "maximum_network_constraint_violation_mw": (
            maximum_network_constraint_violation_mw
        ),
        "maximum_dsr_daily_limit_violation_mwh": (
            maximum_dsr_daily_limit_violation_mwh
        ),
        "maximum_must_run_floor_violation_mw": (
            maximum_must_run_floor_violation_mw
        ),
        "maximum_firm_availability_violation_mw": (
            maximum_firm_availability_violation_mw
        ),
        "solver_maximum_constraint_violation": (
            solver_maximum_constraint_violation
        ),
        "solver_maximum_bound_violation": solver_maximum_bound_violation,
        "vre_spill_mwh": float(vre_spill.X.sum()),
        "candidate_discharge_power_mw": total_pdis,
        "candidate_charge_power_mw": (
            float(candidate_pcharge.X.sum()) if candidate_enabled else 0.0
        ),
        "candidate_incremental_charge_power_mw": (
            float(candidate_extra.X.sum()) if candidate_enabled else 0.0
        ),
        "candidate_charge_mwh": candidate_charge_mwh,
        "candidate_discharge_mwh": candidate_discharge_mwh,
        "candidate_efc_usable_window_output_basis": (
            candidate_discharge_mwh / usable_energy
            if usable_energy > REPORT_TOL
            else 0.0
        ),
        "equivalent_full_cycles_output_basis": (
            candidate_discharge_mwh / rated_energy
            if rated_energy > REPORT_TOL
            else 0.0
        ),
        "candidate_efc_rated_energy": (
            candidate_discharge_mwh / rated_energy
            if rated_energy > REPORT_TOL
            else 0.0
        ),
        "candidate_efc_usable_soc_window": (
            candidate_discharge_mwh / usable_energy
            if usable_energy > REPORT_TOL
            else 0.0
        ),
        "candidate_efc_internal_withdrawal_usable_window": (
            candidate_discharge_mwh
            / storage.eta_discharge
            / usable_energy
            if usable_energy > REPORT_TOL
            else 0.0
        ),
        "equivalent_full_cycles_definition": (
            "delivered_discharge_mwh_divided_by_rated_energy_mwh"
        ),
        "flat_load_increment_mw": float(load_increment.sum()),
        "candidate_charging_opportunity_cost_eur": charging_opportunity_cost,
        "candidate_charge_ceiling_binding": bool(
            candidate_enabled
            and np.any(
                candidate_pcharge.X
                >= (
                    storage.maximum_charge_to_discharge_power_ratio
                    * candidate_pdis[candidate_nodes]
                    - 1e-5
                )
            )
        ),
        "objective_component_error_eur": component_error,
        "gurobi_runtime_s": float(model.Runtime),
        "wall_solve_seconds": solve_seconds,
        "gurobi_iterations": float(model.IterCount),
        "gurobi_barrier_iterations": float(model.BarIterCount),
        "gurobi_status": int(model.Status),
        "baseline_cache_key": baseline_cache_key,
        **objective_components,
    }
    if metrics["candidate_charge_ceiling_binding"]:
        raise RuntimeError(
            "candidate charging-power ceiling binds; enlarge numerical ceiling"
        )
    first_stage = pd.DataFrame(
        {
            "zone": system.zones,
            "firm_addition_mw": firm_values,
            "candidate_discharge_power_mw": (
                candidate_pdis if candidate_enabled else np.zeros(z_count)
            ),
        }
    )
    first_stage["candidate_charge_power_mw"] = 0.0
    first_stage["candidate_incremental_charge_power_mw"] = 0.0
    if candidate_enabled:
        first_stage.loc[
            candidate_nodes, "candidate_charge_power_mw"
        ] = candidate_pcharge.X
        first_stage.loc[
            candidate_nodes, "candidate_incremental_charge_power_mw"
        ] = candidate_extra.X
    dispatch = pd.DataFrame()
    if args.keep_candidate_dispatch and candidate_enabled:
        dispatch = pd.DataFrame(
            {
                "hour_of_year": np.arange(hours),
                "candidate_charge_mw": candidate_charge.X.sum(axis=0),
                "candidate_discharge_mw": candidate_discharge.X.sum(axis=0),
                "candidate_soc_mwh": candidate_soc.X.sum(axis=0),
                "system_unserved_mw": system_unserved,
                "system_balance_dual_eur_mwh_load_weighted": np.average(
                    balance_duals,
                    axis=0,
                    weights=np.maximum(
                        system.gross_load.mean(axis=1), 1e-12
                    ),
                ),
            }
        )
    fossil_gas_dispatch = pd.DataFrame(
        {
            "zone": np.repeat(np.asarray(system.zones), hours),
            "hour_of_year": np.tile(np.arange(hours), z_count),
            "simulated_fossil_gas_generation_mw": (
                fossil_gas_by_zone_hour.reshape(-1)
            ),
            "common_residual_demand_mw": (
                common_residual_by_zone_hour.reshape(-1)
            ),
            "unserved_mw": np.asarray(unserved.X, dtype=float).reshape(-1),
        }
    )
    model.dispose()
    return SolveResult(
        variant,
        metrics,
        first_stage,
        dispatch,
        fossil_gas_dispatch,
        hydro_slack_ledger,
    )


def baseline_cache_payload(
    system: SystemData,
    args: argparse.Namespace,
    order_hash: str,
) -> dict[str, Any]:
    input_root = args.input.resolve()
    chronology_root = args.chronology_root.resolve()
    dispatch_root = args.dispatch_root.resolve()
    flexible_root = args.flexible_root.resolve()
    idsr_root = args.idsr_root.resolve()
    sources = [
        Path(__file__).resolve(),
        CORE_PATH.resolve(),
        ZONE_RUNNER_PATH.resolve(),
        input_root / "QA.json",
        input_root / "zone_coverage.csv",
        input_root / "capacity_2035_final.csv",
        input_root / "hydro_capacity_2035_final.csv",
        chronology_root / "QA.json",
        chronology_root / "synchronized_day_orders.csv",
        dispatch_root / "QA.json",
        dispatch_root / "thermal_capacity_split_cases.csv",
        dispatch_root / "thermal_variable_costs_2035_eur2024.csv",
        dispatch_root / "thermal_expected_availability_daily.csv.gz",
        dispatch_root / "firm_addition_costs_2035_eur2024.csv",
        dispatch_root / "ntc_hourly.csv.gz",
        dispatch_root / "network_zone_limits_hourly.csv.gz",
        flexible_root / "QA.json",
        flexible_root / "existing_utility_battery_2035.csv",
        flexible_root / "existing_pumped_hydro_2035.csv",
        flexible_root / "explicit_dsr_bands_2035.csv",
        idsr_root / "QA.json",
        idsr_root / "idsr_hourly_2035.csv.gz",
        idsr_root / "idsr_windows_2035.csv",
        idsr_root / "idsr_btm_battery_2035.csv",
        idsr_root / "additional_dispatchable_consumption_2035.csv",
        args.must_run_file.resolve(),
        args.reservoir_level_file.resolve(),
    ]
    sources.extend(
        sorted(
            input_root.glob(
                "hourly_profiles/zone=*/hourly_profiles.csv.gz"
            )
        )
    )
    sources.extend(
        sorted(
            input_root.glob(
                "hydro_periodic/zone=*/hydro_periodic.csv.gz"
            )
        )
    )
    missing = [str(path) for path in sources if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"baseline cache source files absent: {missing[:10]}"
        )
    return {
        "script_version": SCRIPT_VERSION,
        "reservoir_endpoint_contract_version": (
            RESERVOIR_ENDPOINT_CONTRACT_VERSION
        ),
        "weather_scenario": args.weather_scenario,
        "chronology": args.chronology,
        "order_sha256": order_hash,
        "capacity_case": args.capacity_case,
        "split_case": args.split_case,
        "hours": system.hours,
        "value_of_lost_load": args.value_of_lost_load,
        "maximum_firm_addition_mw": args.maximum_firm_addition_mw,
        "hydro_variable_cost": args.hydro_variable_cost,
        "hydro_constraint_violation_penalty": (
            args.hydro_constraint_violation_penalty
        ),
        "vre_spill_penalty": args.vre_spill_penalty,
        "hydro_spill_penalty": args.hydro_spill_penalty,
        "flow_regularization": args.flow_regularization,
        "existing_storage_throughput_regularization": (
            args.existing_storage_throughput_regularization
        ),
        "existing_battery_degradation_cost_eur_mwh_discharged": (
            args.existing_battery_degradation_cost_eur_mwh_discharged
        ),
        "feasibility_tolerance": args.feasibility_tolerance,
        "optimality_tolerance": args.optimality_tolerance,
        "nl0a_treatment": args.nl0a_treatment,
        "ie_crosswalk_case": args.ie_crosswalk_case,
        "biofuel_variable_cost_eur_mwh": (
            args.biofuel_variable_cost_eur_mwh
        ),
        "biofuel_availability_fraction": (
            args.biofuel_availability_fraction
        ),
        "other_res_treatment": args.other_res_treatment,
        "other_res_variable_cost_eur_mwh": (
            args.other_res_variable_cost_eur_mwh
        ),
        "other_res_availability_fraction": (
            args.other_res_availability_fraction
        ),
        "must_run_case": args.must_run_case,
        "power_to_heat_availability_case": (
            args.power_to_heat_availability_case
        ),
        "actual_input_sha256": {
            str(path): sha256(path) for path in sorted(set(sources))
        },
    }


def save_baseline_cache(
    cache_dir: Path,
    key: str,
    result: SolveResult,
    payload: dict[str, Any],
) -> None:
    target = cache_dir / key
    target.mkdir(parents=True, exist_ok=True)
    atomic_write_json(target / "cache_contract.json", payload)
    atomic_write_json(target / "baseline_metrics.json", result.metrics)
    atomic_write_csv(result.first_stage, target / "baseline_first_stage.csv")
    atomic_write_csv(
        result.fossil_gas_dispatch,
        target / "baseline_fossil_gas_dispatch.csv.gz",
    )
    atomic_write_csv(
        result.hydro_slack_ledger,
        target / "baseline_hydro_slack_ledger.csv",
    )


def load_baseline_cache(
    cache_dir: Path, key: str, payload: dict[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    target = cache_dir / key
    contract = target / "cache_contract.json"
    metrics = target / "baseline_metrics.json"
    first = target / "baseline_first_stage.csv"
    gas = target / "baseline_fossil_gas_dispatch.csv.gz"
    hydro_slack = target / "baseline_hydro_slack_ledger.csv"
    if not (
        contract.exists()
        and metrics.exists()
        and first.exists()
        and gas.exists()
        and hydro_slack.exists()
    ):
        return None
    observed = json.loads(contract.read_text(encoding="utf-8"))
    if observed != payload:
        raise ValueError(f"baseline cache contract mismatch: {target}")
    return (
        json.loads(metrics.read_text(encoding="utf-8")),
        pd.read_csv(first),
        pd.read_csv(gas),
        pd.read_csv(hydro_slack),
    )


def reliability_eligible(
    metrics: dict[str, Any],
    reference_eens_mwh: float,
    reference_lole_h: float,
    args: argparse.Namespace,
) -> bool:
    eens_tolerance = max(
        args.elcc_eens_absolute_tolerance_mwh,
        abs(reference_eens_mwh) * args.elcc_reliability_relative_tolerance,
    )
    return bool(
        float(metrics["expected_eens_mwh"])
        <= reference_eens_mwh + eens_tolerance
        and float(metrics["expected_lole_h"])
        <= reference_lole_h + args.elcc_lole_tolerance_h
    )


def reliability_margins(
    metrics: dict[str, Any],
    reference_eens_mwh: float,
    reference_lole_h: float,
    args: argparse.Namespace,
) -> dict[str, float | bool]:
    """Return signed reliability margins for an LCC endpoint.

    A non-negative margin passes the corresponding constraint.  Binding flags
    identify which metric rejects the upper bisection endpoint instead of
    hiding that distinction inside a single eligibility Boolean.
    """

    eens_tolerance = max(
        args.elcc_eens_absolute_tolerance_mwh,
        abs(reference_eens_mwh) * args.elcc_reliability_relative_tolerance,
    )
    eens_margin = (
        reference_eens_mwh
        + eens_tolerance
        - float(metrics["expected_eens_mwh"])
    )
    lole_margin = (
        reference_lole_h
        + args.elcc_lole_tolerance_h
        - float(metrics["expected_lole_h"])
    )
    return {
        "upper_probe_eens_margin_mwh": float(eens_margin),
        "upper_probe_lole_margin_h": float(lole_margin),
        "upper_probe_eens_binding": bool(eens_margin < 0),
        "upper_probe_lole_binding": bool(lole_margin < 0),
    }


def load_carrying_capability(
    *,
    system: SystemData,
    storage,
    candidate_pdis: np.ndarray,
    candidate_pcharge: np.ndarray | None,
    baseline_firm: np.ndarray,
    baseline_metrics: dict[str, Any],
    baseline_cache_key: str,
    resource_case: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Bisection of flat-load carrying capability in a deterministic screen.

    ``resource_case`` is ``no_candidate`` or ``candidate``.  Candidate
    discharge, energy and charging-interface capacities are fixed throughout
    the search.  Every evaluated endpoint is written to an interruption-safe
    cache. Expected availability is not a sequential outage ensemble, so this
    diagnostic is not standard probabilistic EENS, LOLE or ELCC.
    """

    if resource_case not in {"no_candidate", "candidate"}:
        raise ValueError(resource_case)
    total_candidate = float(candidate_pdis.sum())
    if total_candidate <= 0:
        raise ValueError("ELCC requires a positive candidate tranche")
    shares = candidate_pdis / total_candidate
    reference_eens = float(baseline_metrics["expected_eens_mwh"])
    reference_lole = float(baseline_metrics["expected_lole_h"])
    storage_contract = {
        "config_id": str(storage.config_id),
        "tech_id": str(storage.tech_id),
        "duration_h": float(storage.duration_h),
        "round_trip_efficiency": float(storage.round_trip_efficiency),
        "eta_charge": float(storage.eta_charge),
        "eta_discharge": float(storage.eta_discharge),
        "self_discharge_pct_day": float(storage.self_discharge_pct_day),
        "hourly_retention": float(storage.hourly_retention),
        "soc_min_fraction": float(storage.soc_min_fraction),
        "soc_max_fraction": float(storage.soc_max_fraction),
        "reference_charge_to_discharge_power_ratio": float(
            storage.reference_charge_to_discharge_power_ratio
        ),
        "maximum_charge_to_discharge_power_ratio": float(
            storage.maximum_charge_to_discharge_power_ratio
        ),
        "annual_discharge_power_cost_eur_kw": float(
            storage.annual_discharge_power_cost_eur_kw
        ),
        "annual_charge_power_cost_eur_kw": float(
            storage.annual_charge_power_cost_eur_kw
        ),
        "annual_energy_cost_eur_kwh": float(
            storage.annual_energy_cost_eur_kwh
        ),
        "variable_om_eur_mwh": float(storage.variable_om_eur_mwh),
        "degradation_cost_eur_mwh_discharged": float(
            storage.degradation_cost_eur_mwh_discharged
        ),
    }
    contract = {
        "script_version": SCRIPT_VERSION,
        "baseline_cache_key": baseline_cache_key,
        "resource_case": resource_case,
        "config_id": (
            storage.config_id if resource_case == "candidate" else "__baseline__"
        ),
        "candidate_pdis_sha256": hashlib.sha256(
            np.asarray(candidate_pdis, dtype=np.float64).tobytes()
        ).hexdigest(),
        "candidate_pcharge_sha256": (
            hashlib.sha256(
                np.asarray(candidate_pcharge, dtype=np.float64).tobytes()
            ).hexdigest()
            if candidate_pcharge is not None
            else None
        ),
        "reference_eens_mwh": reference_eens,
        "reference_lole_h": reference_lole,
        "tolerance_mw": args.elcc_tolerance_mw,
        "maximum_upper_multiplier": args.elcc_max_upper_multiplier,
        "reliability_relative_tolerance": (
            args.elcc_reliability_relative_tolerance
        ),
        "eens_absolute_tolerance_mwh": (
            args.elcc_eens_absolute_tolerance_mwh
        ),
        "lole_tolerance_h": args.elcc_lole_tolerance_h,
        "storage_contract": storage_contract,
        "storage_parameter_sources_sha256": {
            "catalog": sha256(args.catalog.resolve()),
            "technology_library": sha256(
                args.technology_library.resolve()
            ),
            "core": sha256(CORE_PATH.resolve()),
            "zone_helper": sha256(ZONE_RUNNER_PATH.resolve()),
        },
    }
    key = hash_payload(contract)
    path = args.cache_root.resolve() / "elcc" / f"{key}.json"
    probes: dict[str, dict[str, Any]] = {}
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("contract") != contract:
            raise ValueError(f"ELCC cache contract mismatch: {path}")
        if saved.get("complete", False):
            return saved["result"]
        probes = saved.get("probes", {})

    def persist(result: dict[str, Any] | None = None) -> None:
        atomic_write_json(
            path,
            {
                "contract": contract,
                "complete": result is not None,
                "probes": probes,
                "result": result,
            },
        )

    def evaluate(increment_mw: float) -> tuple[bool, dict[str, Any]]:
        probe_key = f"{increment_mw:.9f}"
        if probe_key in probes:
            one = probes[probe_key]
            return bool(one["eligible"]), one["metrics"]
        increment = shares * float(increment_mw)
        if resource_case == "candidate":
            variant = "repeat_use_fixed_firm"
            allocation = candidate_pdis
            fixed_charge = candidate_pcharge
        else:
            variant = "adequacy_baseline_fixed_firm"
            allocation = np.zeros_like(candidate_pdis)
            fixed_charge = None
        solved = solve_model(
            system,
            storage,
            allocation,
            variant,
            args,
            baseline_firm,
            baseline_cache_key,
            load_increment_mw_by_zone=increment,
            fixed_candidate_charge_power_mw_by_zone=fixed_charge,
        )
        eligible = reliability_eligible(
            solved.metrics, reference_eens, reference_lole, args
        )
        compact_metrics = {
            "expected_eens_mwh": solved.metrics["expected_eens_mwh"],
            "expected_lole_h": solved.metrics["expected_lole_h"],
            "objective_eur_year": solved.metrics["objective_eur_year"],
            "gurobi_runtime_s": solved.metrics["gurobi_runtime_s"],
            "candidate_charge_power_mw": solved.metrics[
                "candidate_charge_power_mw"
            ],
        }
        probes[probe_key] = {
            "increment_mw": float(increment_mw),
            "eligible": eligible,
            "metrics": compact_metrics,
        }
        persist()
        return eligible, compact_metrics

    low = 0.0
    # At zero increment, no-candidate fixed-firm reliability is identical to
    # the cached baseline; the candidate zero endpoint was already solved in
    # the repeat-use variant.  Record the analytical endpoint explicitly.
    probes.setdefault(
        f"{0.0:.9f}",
        {
            "increment_mw": 0.0,
            "eligible": True,
            "metrics": {
                "expected_eens_mwh": reference_eens,
                "expected_lole_h": reference_lole,
                "objective_eur_year": math.nan,
                "gurobi_runtime_s": 0.0,
                "candidate_charge_power_mw": (
                    float(np.asarray(candidate_pcharge).sum())
                    if resource_case == "candidate"
                    and candidate_pcharge is not None
                    else 0.0
                ),
            },
        },
    )
    high = total_candidate
    maximum_high = total_candidate * args.elcc_max_upper_multiplier
    high_ok, high_metrics = evaluate(high)
    while high_ok and high < maximum_high - 1e-9:
        low = high
        high = min(high * 2.0, maximum_high)
        high_ok, high_metrics = evaluate(high)
    if high_ok:
        result = {
            "resource_case": resource_case,
            "lower_mw": high,
            "upper_mw": math.nan,
            "right_censored": True,
            "iterations": len(probes) - 1,
            "cache_key": key,
            "complete_bracket": False,
            **reliability_margins(
                high_metrics, reference_eens, reference_lole, args
            ),
        }
        persist(result)
        return result
    iterations = 0
    while (
        high - low > args.elcc_tolerance_mw
        and iterations < args.elcc_max_iterations
    ):
        middle = 0.5 * (low + high)
        ok, middle_metrics = evaluate(middle)
        if ok:
            low = middle
        else:
            high = middle
            high_metrics = middle_metrics
        iterations += 1
    result = {
        "resource_case": resource_case,
        "lower_mw": low,
        "upper_mw": high,
        "right_censored": False,
        "iterations": iterations,
        "cache_key": key,
        "complete_bracket": True,
        **reliability_margins(
            high_metrics, reference_eens, reference_lole, args
        ),
    }
    persist(result)
    return result


def calculate_elcc(
    *,
    system: SystemData,
    storage,
    candidate_pdis: np.ndarray,
    candidate_pcharge: np.ndarray,
    baseline_firm: np.ndarray,
    baseline_metrics: dict[str, Any],
    baseline_cache_key: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Return an ELCC-style expected-availability screening diagnostic."""

    baseline_lcc = load_carrying_capability(
        system=system,
        storage=storage,
        candidate_pdis=candidate_pdis,
        candidate_pcharge=None,
        baseline_firm=baseline_firm,
        baseline_metrics=baseline_metrics,
        baseline_cache_key=baseline_cache_key,
        resource_case="no_candidate",
        args=args,
    )
    candidate_lcc = load_carrying_capability(
        system=system,
        storage=storage,
        candidate_pdis=candidate_pdis,
        candidate_pcharge=candidate_pcharge,
        baseline_firm=baseline_firm,
        baseline_metrics=baseline_metrics,
        baseline_cache_key=baseline_cache_key,
        resource_case="candidate",
        args=args,
    )
    complete = bool(
        baseline_lcc["complete_bracket"]
        and candidate_lcc["complete_bracket"]
    )
    if complete:
        lower = max(
            0.0,
            float(candidate_lcc["lower_mw"])
            - float(baseline_lcc["upper_mw"]),
        )
        upper = max(
            0.0,
            float(candidate_lcc["upper_mw"])
            - float(baseline_lcc["lower_mw"]),
        )
        point = lower
    else:
        lower = math.nan
        upper = math.nan
        point = math.nan
    return {
        "elcc_evidence_class": (
            "deterministic_expected_availability_screen_not_standard_ELCC"
        ),
        "standard_probabilistic_elcc_evidence_admitted": False,
        "elcc_mw": point,
        "elcc_lower_mw": lower,
        "elcc_upper_mw": upper,
        "elcc_complete": complete,
        "baseline_load_headroom_lower_mw": baseline_lcc["lower_mw"],
        "baseline_load_headroom_upper_mw": baseline_lcc["upper_mw"],
        "candidate_gross_lcc_lower_mw": candidate_lcc["lower_mw"],
        "candidate_gross_lcc_upper_mw": candidate_lcc["upper_mw"],
        "baseline_lcc_cache_key": baseline_lcc["cache_key"],
        "candidate_lcc_cache_key": candidate_lcc["cache_key"],
        "baseline_upper_probe_eens_binding": baseline_lcc[
            "upper_probe_eens_binding"
        ],
        "baseline_upper_probe_lole_binding": baseline_lcc[
            "upper_probe_lole_binding"
        ],
        "baseline_upper_probe_eens_margin_mwh": baseline_lcc[
            "upper_probe_eens_margin_mwh"
        ],
        "baseline_upper_probe_lole_margin_h": baseline_lcc[
            "upper_probe_lole_margin_h"
        ],
        "candidate_upper_probe_eens_binding": candidate_lcc[
            "upper_probe_eens_binding"
        ],
        "candidate_upper_probe_lole_binding": candidate_lcc[
            "upper_probe_lole_binding"
        ],
        "candidate_upper_probe_eens_margin_mwh": candidate_lcc[
            "upper_probe_eens_margin_mwh"
        ],
        "candidate_upper_probe_lole_margin_h": candidate_lcc[
            "upper_probe_lole_margin_h"
        ],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.backend != "gurobi":
        raise ValueError("full-network confirmation requires --backend gurobi")
    if (
        args.physical_residual_tolerance <= 0
        or args.simultaneous_dispatch_relative_tolerance <= 0
        or args.simultaneous_dispatch_tolerance_mwh < 0
    ):
        raise ValueError("numerical audit tolerances must be positive")
    if args.existing_storage_throughput_regularization <= max(
        args.vre_spill_penalty, args.hydro_spill_penalty
    ):
        raise ValueError(
            "existing-storage throughput regularization must be strictly "
            "greater than every spill penalty"
        )
    if args.existing_battery_degradation_cost_eur_mwh_discharged < 0:
        raise ValueError(
            "existing-battery degradation cost must be non-negative"
        )
    if args.other_res_variable_cost_eur_mwh < 0 or not (
        0 < args.other_res_availability_fraction <= 1
    ):
        raise ValueError(
            "Other RES cost must be non-negative and availability must lie "
            "in (0, 1]"
        )
    if not (
        0 < args.hydro_constraint_violation_penalty < args.value_of_lost_load
    ):
        raise ValueError(
            "hydro-constraint penalty must be positive and below VoLL"
        )
    if args.development_hour_limit:
        if (
            args.development_hour_limit > HOURS_PER_YEAR
            or args.development_hour_limit % 24
        ):
            raise ValueError("development hour limit must be a whole-day <=8760")
    output = args.output.resolve()
    prepare_output(output, args.force)
    system, audit = load_system(args)
    storages, currency_rate = storage_configurations(args)
    candidate_pdis = candidate_allocation(
        system, args.deployment_scope, args.tranche_share
    )
    cache_payload = baseline_cache_payload(
        system, args, audit["order_sha256"]
    )
    cache_key = hash_payload(cache_payload)
    cache = load_baseline_cache(args.cache_root.resolve(), cache_key, cache_payload)
    cache_hit = cache is not None
    dummy_storage = next(iter(storages.values()))
    if cache is None:
        baseline = solve_model(
            system,
            dummy_storage,
            np.zeros(len(system.zones)),
            "baseline",
            args,
            None,
            cache_key,
        )
        save_baseline_cache(
            args.cache_root.resolve(), cache_key, baseline, cache_payload
        )
        baseline_metrics = baseline.metrics
        baseline_first = baseline.first_stage
        baseline_gas_dispatch = baseline.fossil_gas_dispatch
        baseline_hydro_slack_ledger = baseline.hydro_slack_ledger
    else:
        (
            baseline_metrics,
            baseline_first,
            baseline_gas_dispatch,
            baseline_hydro_slack_ledger,
        ) = cache
    atomic_write_csv(
        baseline_gas_dispatch,
        output / "fossil_gas_dispatch" / "baseline.csv.gz",
    )
    atomic_write_csv(
        baseline_hydro_slack_ledger,
        output / "hydro_slack_ledger" / "baseline.csv",
    )
    baseline_firm = (
        baseline_first.set_index("zone")
        .reindex(system.zones)["firm_addition_mw"]
        .to_numpy(dtype=float)
    )

    rows = [
        {
            "weather_scenario": args.weather_scenario,
            "chronology": args.chronology,
            "deployment_scope": args.deployment_scope,
            "capacity_case": args.capacity_case,
            "split_case": args.split_case,
            "ie_crosswalk_case": args.ie_crosswalk_case,
            "nl0a_treatment": args.nl0a_treatment,
            "must_run_case_argument": args.must_run_case,
            "power_to_heat_availability_case_argument": (
                args.power_to_heat_availability_case
            ),
            "existing_battery_degradation_cost_eur_mwh_discharged_argument": (
                args.existing_battery_degradation_cost_eur_mwh_discharged
            ),
            "other_res_treatment": args.other_res_treatment,
            "other_res_variable_cost_eur_mwh_argument": (
                args.other_res_variable_cost_eur_mwh
            ),
            "other_res_availability_fraction_argument": (
                args.other_res_availability_fraction
            ),
            **baseline_metrics,
        }
    ]
    first_rows = []
    dispatch_rows = []
    for storage in storages.values():
        solved: dict[str, SolveResult] = {}
        for variant in ("joint", "repeat_use_fixed_firm"):
            result = solve_model(
                system,
                storage,
                candidate_pdis,
                variant,
                args,
                baseline_firm if variant == "repeat_use_fixed_firm" else None,
                cache_key,
            )
            solved[variant] = result
            safe_config = str(storage.config_id).replace("/", "_")
            atomic_write_csv(
                result.fossil_gas_dispatch,
                output
                / "fossil_gas_dispatch"
                / f"{safe_config}__{variant}.csv.gz",
            )
            atomic_write_csv(
                result.hydro_slack_ledger,
                output
                / "hydro_slack_ledger"
                / f"{safe_config}__{variant}.csv",
            )
            first = result.first_stage.copy()
            first.insert(0, "config_id", storage.config_id)
            first.insert(0, "variant", variant)
            first_rows.append(first)
            if not result.candidate_dispatch.empty:
                dispatch = result.candidate_dispatch.copy()
                dispatch.insert(0, "config_id", storage.config_id)
                dispatch.insert(0, "variant", variant)
                dispatch_rows.append(dispatch)
        tranche_kw = float(candidate_pdis.sum()) * 1_000.0
        joint_benefit = (
            float(
                baseline_metrics[
                    "economic_objective_excluding_hydro_feasibility_penalties_eur_year"
                ]
            )
            - solved["joint"].metrics[
                "economic_objective_excluding_hydro_feasibility_penalties_eur_year"
            ]
        ) / tranche_kw
        repeat_benefit = (
            float(
                baseline_metrics[
                    "economic_objective_excluding_hydro_feasibility_penalties_eur_year"
                ]
            )
            - solved["repeat_use_fixed_firm"].metrics[
                "economic_objective_excluding_hydro_feasibility_penalties_eur_year"
            ]
        ) / tranche_kw
        joint_penalized_objective_benefit = (
            float(baseline_metrics["objective_eur_year"])
            - solved["joint"].metrics["objective_eur_year"]
        ) / tranche_kw
        repeat_penalized_objective_benefit = (
            float(baseline_metrics["objective_eur_year"])
            - solved["repeat_use_fixed_firm"].metrics["objective_eur_year"]
        ) / tranche_kw
        adequacy = joint_benefit - repeat_benefit
        if args.skip_elcc:
            elcc = {
                "elcc_evidence_class": (
                    "deterministic_expected_availability_screen_not_run"
                ),
                "standard_probabilistic_elcc_evidence_admitted": False,
                "elcc_mw": math.nan,
                "elcc_lower_mw": math.nan,
                "elcc_upper_mw": math.nan,
                "elcc_complete": False,
                "baseline_load_headroom_lower_mw": math.nan,
                "baseline_load_headroom_upper_mw": math.nan,
                "candidate_gross_lcc_lower_mw": math.nan,
                "candidate_gross_lcc_upper_mw": math.nan,
                "baseline_lcc_cache_key": "",
                "candidate_lcc_cache_key": "",
            }
        else:
            repeat_first = solved[
                "repeat_use_fixed_firm"
            ].first_stage.set_index("zone").reindex(system.zones)
            elcc = calculate_elcc(
                system=system,
                storage=storage,
                candidate_pdis=candidate_pdis,
                candidate_pcharge=repeat_first[
                    "candidate_charge_power_mw"
                ].to_numpy(dtype=float),
                baseline_firm=baseline_firm,
                baseline_metrics=baseline_metrics,
                baseline_cache_key=cache_key,
                args=args,
            )
        for result in solved.values():
            rows.append(
                {
                    "weather_scenario": args.weather_scenario,
                    "chronology": args.chronology,
                    "deployment_scope": args.deployment_scope,
                    "capacity_case": args.capacity_case,
                    "split_case": args.split_case,
                    "ie_crosswalk_case": args.ie_crosswalk_case,
                    "nl0a_treatment": args.nl0a_treatment,
                    "must_run_case_argument": args.must_run_case,
                    "power_to_heat_availability_case_argument": (
                        args.power_to_heat_availability_case
                    ),
                    "existing_battery_degradation_cost_eur_mwh_discharged_argument": (
                        args.existing_battery_degradation_cost_eur_mwh_discharged
                    ),
                    "other_res_treatment": args.other_res_treatment,
                    "other_res_variable_cost_eur_mwh_argument": (
                        args.other_res_variable_cost_eur_mwh
                    ),
                    "other_res_availability_fraction_argument": (
                        args.other_res_availability_fraction
                    ),
                    **result.metrics,
                    "signed_joint_benefit_eur_kw_year": joint_benefit,
                    "signed_repeat_use_benefit_eur_kw_year": repeat_benefit,
                    "residual_adequacy_interaction_benefit_eur_kw_year": adequacy,
                    "signed_joint_penalized_objective_benefit_eur_kw_year": (
                        joint_penalized_objective_benefit
                    ),
                    "signed_repeat_use_penalized_objective_benefit_eur_kw_year": (
                        repeat_penalized_objective_benefit
                    ),
                    **elcc,
                }
            )
    metrics = pd.DataFrame(rows)
    atomic_write_csv(metrics, output / "run_metrics.csv")
    first_stage_frame = (
        pd.concat(first_rows, ignore_index=True)
        if first_rows
        else pd.DataFrame()
    )
    if first_rows:
        atomic_write_csv(
            first_stage_frame,
            output / "first_stage.csv",
        )
    if dispatch_rows:
        atomic_write_csv(
            pd.concat(dispatch_rows, ignore_index=True),
            output / "candidate_dispatch.csv",
        )

    flexible_qa = audit["flexible_qa"]
    profile_audit = audit["system"]["profile"]
    hydro_audit = audit["system"]["hydro"]
    idsr_audit = audit["system"]["idsr"]
    idsr_adapter_gates = idsr_audit["adapter_qa"].get("gates", {})
    excluded_psp_ids = set(
        hydro_audit["pumped_hydro_excluded_missing_dimension_ids"]
    )
    if first_stage_frame.empty:
        candidate_location_fixed = False
    else:
        candidate_location_fixed = True
        for (_, _), group in first_stage_frame.groupby(
            ["variant", "config_id"]
        ):
            observed = (
                group.set_index("zone")
                .reindex(system.zones)["candidate_discharge_power_mw"]
                .to_numpy(dtype=float)
            )
            candidate_location_fixed &= bool(
                np.allclose(observed, candidate_pdis, rtol=0, atol=1e-8)
            )
    physical_residual_columns = [
        "maximum_nodal_balance_residual_mw",
        "maximum_candidate_soc_residual_mwh",
        "maximum_existing_battery_soc_residual_mwh",
        "maximum_idsr_btm_battery_soc_residual_mwh",
        "maximum_pumped_hydro_soc_residual_mwh",
        "maximum_reservoir_soc_residual_mwh",
        "maximum_run_of_river_balance_residual_mwh",
        "maximum_network_constraint_violation_mw",
        "maximum_dsr_daily_limit_violation_mwh",
        "maximum_must_run_floor_violation_mw",
        "maximum_firm_availability_violation_mw",
        "maximum_idsr_window_energy_residual_mwh",
        "solver_maximum_constraint_violation",
        "solver_maximum_bound_violation",
    ]
    maximum_reconstructed_physical_residual = float(
        np.max(
            metrics[physical_residual_columns]
            .fillna(np.inf)
            .to_numpy(dtype=float),
            initial=0.0,
        )
    )
    full_year = system.hours == HOURS_PER_YEAR
    hydro_contract_components_audited = bool(
        metrics[
            [
                "hydro_contract_violation_mwh",
                "hydro_source_contract_discrepancy_mwh",
                "hydro_level_trajectory_soft_slack_mwh",
                "hydro_contract_official_ror_structural_lower_bound_mwh",
                "hydro_contract_development_terminal_boundary_lower_bound_mwh",
                "hydro_contract_admissible_lower_bound_mwh",
                "hydro_contract_excess_slack_mwh",
            ]
        ]
        .apply(pd.to_numeric, errors="coerce")
        .notna()
        .all()
        .all()
        and (
            (
                metrics["hydro_source_contract_discrepancy_mwh"]
                - metrics["hydro_contract_admissible_lower_bound_mwh"]
            )
            .abs()
            .le(args.hydro_constraint_slack_tolerance_mwh)
            .all()
        )
        and bool(
            audit["system"]["hydro"][
                "development_terminal_boundary_condition_active"
            ]
            == (not full_year)
        )
    )
    endpoint_records = audit["system"]["hydro"][
        "annual_reservoir_endpoints"
    ]["admitted_endpoint_records"]
    endpoint_lookup = {
        (str(row["zone"]), str(row["hydro_code"])): (
            float(row["initial_soc_fraction"]),
            float(row["terminal_soc_fraction"]),
            str(row["annual_endpoint_source"]),
            str(row["annual_endpoint_initial_source"]),
            str(row["annual_endpoint_terminal_source"]),
        )
        for row in endpoint_records
    }
    official_endpoint_gold = {
        ("LT00", "HCL"): (1.0, 1.0),
        ("SK00", "HOL"): (0.8568, 0.816266667),
        ("SK00", "HRI"): (0.7616, 0.6742),
        ("MK00", "HRI"): (0.34, 0.26),
        ("BG00", "HCL"): (0.42, 0.41),
    }
    official_endpoint_gold_pass = bool(
        all(
            key in endpoint_lookup
            and endpoint_lookup[key][2] == "official_annual_levels"
            and np.allclose(
                endpoint_lookup[key][:2],
                expected,
                rtol=0,
                atol=1e-9,
            )
            for key, expected in official_endpoint_gold.items()
        )
    )
    annex2_fallback_gold = {
        ("NOM1", "HRI"): (0.6410, 0.6460),
        ("NON1", "HRI"): (0.6740, 0.6865),
        ("NOS1", "HRI"): (0.6775, 0.7100),
        ("NOS2", "HRI"): (0.6715, 0.6870),
        ("NOS3", "HRI"): (0.6335, 0.6630),
        ("PT00", "HRI"): (0.4955, 0.4915),
    }
    annex2_fallback_gold_pass = bool(
        all(
            key in endpoint_lookup
            and endpoint_lookup[key][2]
            == "annex2_section_6_1_fallback"
            and endpoint_lookup[key][3]
            == "current_ws_week1_trajectory_midpoint"
            and endpoint_lookup[key][4]
            == "current_ws_week52_trajectory_midpoint"
            and np.allclose(
                endpoint_lookup[key][:2],
                expected,
                rtol=0,
                atol=1e-9,
            )
            for key, expected in annex2_fallback_gold.items()
        )
    )
    execution_gates = {
        "post_eva_fleet_applied": args.capacity_case
        == "amended_annex3_risk_averse_enhanced_hurdle",
        "official_in_market_utility_battery_modelled": bool(
            np.isclose(
                system.batteries["discharge_power_mw"].sum(),
                flexible_qa[
                    "official_in_market_utility_battery_power_mw"
                ]
                + flexible_qa["annex3_battery_addition_power_mw"],
                atol=1e-6,
            )
        ),
        "price_sensitive_BTM_reference_undone_and_redispatched": bool(
            flexible_qa["gates"][
                "btm_battery_excluded_from_dispatch_storage"
            ]
            and set(system.idsr_btm_batteries["zone"]) == {"AT00", "CH00"}
            and np.isfinite(system.idsr_btm_reference).all()
        ),
        "official_pumped_hydro_with_positive_energy_bound_modelled": bool(
            len(system.pumped_hydro)
            == flexible_qa["pumped_hydro_usable_rows"]
            == 31
        ),
        "pumped_hydro_missing_energy_excluded_not_dispatched": audit["system"][
            "hydro"
        ]["pumped_hydro_records_excluded_missing_dimension"]
        == 12
        and excluded_psp_ids == EXPECTED_PSP_EXCLUDED_MISSING_ENERGY,
        "open_loop_natural_inflow_rule_applied": audit["system"]["hydro"][
            "open_loop_inflow_only_where_flagged"
        ],
        "official_annual_hydro_endpoints_applied_before_fallback": bool(
            audit["system"]["hydro"]["annual_reservoir_endpoints"][
                "official_endpoint_match_count"
            ]
            == 30
            and official_endpoint_gold_pass
        ),
        "annex2_section_6_1_hydro_endpoint_fallback_applied": bool(
            hydro_endpoint_fallback_self_test()
            and annex2_fallback_gold_pass
            and audit["system"]["hydro"][
                "annual_reservoir_endpoints"
            ]["endpoint_contract_version"]
            == RESERVOIR_ENDPOINT_CONTRACT_VERSION
        ),
        "final_partial_week_energy_and_inflow_prorated": bool(
            hydro_partial_period_mapping_self_test()
            and
            audit["system"]["hydro"][
                "calendar_partial_period_energy_prorated"
            ]
            and audit["system"]["hydro"][
                "calendar_prorated_period_count"
            ]
            > 0
            and all(
                np.isclose(
                    float(row["coverage_fraction"]),
                    24.0 / 168.0,
                    rtol=0,
                    atol=1e-12,
                )
                for row in audit["system"]["hydro"][
                    "calendar_prorated_periods"
                ]
            )
        ),
        "hydro_calendar_fixed_common_control": audit["system"]["hydro"][
            "chronology_treatment"
        ]
        == "calendar_fixed_common_control",
        "released_hydro_period_constraints_instantiated": bool(
            metrics["hydro_contract_constraint_count"].fillna(0).gt(0).all()
        ),
        "hydro_source_discrepancy_and_soft_levels_separately_audited": bool(
            metrics["hydro_contract_violation_mwh"].notna().all()
            and metrics["hydro_source_contract_discrepancy_mwh"].notna().all()
            and metrics["hydro_level_trajectory_soft_slack_mwh"].notna().all()
            and metrics[
                "hydro_constraint_violation_cost_eur_year"
            ].notna().all()
        ),
        "hydro_positive_slack_ledger_reconciles": bool(
            metrics["hydro_slack_ledger_residual_mwh"]
            .fillna(np.inf)
            .le(args.hydro_constraint_slack_tolerance_mwh)
            .all()
            and (
                output / "hydro_slack_ledger" / "baseline.csv"
            ).exists()
        ),
        "hydro_structural_and_development_slack_components_audited": (
            hydro_contract_components_audited
        ),
        "full_year_hydro_source_discrepancy_excess_below_tolerance": bool(
            not full_year
            or metrics["hydro_contract_excess_slack_mwh"]
            .fillna(np.inf)
            .le(args.hydro_constraint_slack_tolerance_mwh)
            .all()
        ),
        "official_explicit_dsr_with_daily_hour_limits_modelled": bool(
            len(system.dsr) == flexible_qa["explicit_dsr_rows"]
            and np.isclose(
                system.dsr["capacity_mw"].sum(),
                flexible_qa["official_explicit_dsr_power_mw"]
                + flexible_qa["annex3_dsr_addition_power_mw"],
                atol=1e-6,
            )
            and system.dsr["max_hours_per_day"].between(0, 24).all()
        ),
        "full_75_node_269_edge_network_modelled": len(system.zones) == 75
        and len(system.edges) == 269,
        "all_52_demand_nodes_modelled": len(system.demand_zones) == 52,
        "profile_and_network_node_identity_exact": bool(
            profile_audit["direct_profile_graph_nodes"] == 66
            and set(profile_audit["audited_zero_network_node_ids"])
            == EXPECTED_NETWORK_ONLY_ZERO_NODES
            and profile_audit["out_of_graph_profile_sources"] == 1
        ),
        "isolated_NL0A_treatment_explicit": bool(
            profile_audit["NL0A_OFF_treatment"] == args.nl0a_treatment
            and profile_audit[
                "NL0A_OFF_electrolyser_capacity_mw_quarantined"
            ]
            == 50.0
        ),
        "official_EV_HP_window_energy_conserved": bool(
            metrics["maximum_idsr_window_energy_residual_mwh"]
            .fillna(np.inf)
            .le(args.physical_residual_tolerance)
            .all()
            and metrics["idsr_energy_conservation_residual_mwh"]
            .fillna(np.inf)
            .le(args.physical_residual_tolerance)
            .all()
            and len(system.idsr_schedules) == 104
            and set(system.idsr_schedules["resource_type"]) == {"EV", "HP"}
            and all(
                bool(idsr_adapter_gates.get(gate, False))
                for gate in (
                    "36_weather_scenarios",
                    "52_demand_nodes",
                    "8760_hours_per_weather_scenario",
                    "annual_ratio_multiplication_exact",
                    "annex2_windows_complete",
                    "ev_hp_never_exceeds_phase1_gross_demand",
                    "hourly_row_identity",
                    "signed_balance_reference_reconstruction_exact",
                    "serialized_output_reread",
                )
            )
            and idsr_window_analytic_self_test()
        ),
        "official_price_sensitive_BTM_battery_share_modelled": bool(
            len(system.idsr_btm_batteries) == 2
            and set(system.idsr_btm_batteries["zone"]) == {"AT00", "CH00"}
            and bool(
                idsr_adapter_gates.get(
                    "btm_capacity_multiplication_exact", False
                )
            )
            and metrics["maximum_idsr_btm_battery_soc_residual_mwh"]
            .fillna(np.inf)
            .le(args.physical_residual_tolerance)
            .all()
        ),
        "additional_dispatchable_consumption_treatment_explicit": bool(
            idsr_audit["additional_consumers"] == 44
            and idsr_audit["isolated_NL0A_electrolyser_quarantined"]
            and bool(
                idsr_adapter_gates.get(
                    "additional_consumption_bounds_are_zero_nameplate",
                    False,
                )
            )
            and bool(
                idsr_adapter_gates.get(
                    "p2h_not_admitted_without_hourly_heat_availability",
                    False,
                )
            )
            and args.power_to_heat_availability_case in {"zero", "nameplate"}
        ),
        "available_on_market_other_RES_capacity_accounted_by_explicit_bound": bool(
            (
                args.other_res_treatment == "dispatchable_sensitivity"
                and np.isclose(
                    audit["system"]["thermal"][
                        "other_res_dispatch_capacity_mw"
                    ],
                    audit["system"]["thermal"][
                        "other_res_available_on_market_source_capacity_mw"
                    ],
                    atol=1e-6,
                )
            )
            or (
                args.other_res_treatment == "quarantine"
                and np.isclose(
                    audit["system"]["thermal"][
                        "other_res_dispatch_capacity_mw"
                    ],
                    0.0,
                    atol=1e-9,
                )
                and audit["system"]["thermal"][
                    "other_res_available_on_market_source_capacity_mw"
                ]
                > 0
            )
        ),
        "must_run_other_inelastic_floor_preserved_without_fabricated_profile": (
            bool(
                not audit["system"]["must_run"]["quarantined_rows"]
                and (
                    args.must_run_case == "off"
                    or (
                        args.other_res_treatment == "quarantine"
                        and any(
                            bool(value)
                            for value in system.must_run.get(
                                "standalone_fixed_injection",
                                pd.Series(dtype=bool),
                            )
                        )
                    )
                    or (
                        args.other_res_treatment
                        == "dispatchable_sensitivity"
                        and not any(
                            bool(value)
                            for value in system.must_run.get(
                                "standalone_fixed_injection",
                                pd.Series(dtype=bool),
                            )
                        )
                    )
                )
            )
        ),
        "must_run_absent_post_eva_hosts_explicitly_audited_and_excluded": bool(
            audit["system"]["must_run"]["source_rows_2035"]
            == (
                audit["system"]["must_run"]["mapped_group_rows"]
                + len(
                    audit["system"]["must_run"]["absent_post_eva_hosts"]
                )
                + len(audit["system"]["must_run"]["quarantined_rows"])
            )
            and all(
                row.get("reason")
                == "no_positive_amended_post_eva_host"
                for row in audit["system"]["must_run"][
                    "absent_post_eva_hosts"
                ]
            )
            and not {
                (str(row["zone"]), str(row["broad_fuel"]))
                for row in audit["system"]["must_run"][
                    "absent_post_eva_hosts"
                ]
            }
            & {
                (str(row.zone), str(row.broad_fuel))
                for row in system.must_run.itertuples(index=False)
            }
        ),
        "simulated_fossil_gas_node_hour_output_written": bool(
            (output / "fossil_gas_dispatch" / "baseline.csv.gz").exists()
            and len(baseline_gas_dispatch) == len(system.zones) * system.hours
        ),
        "ERAA_fossil_gas_output_labelled_forward_simulation": bool(
            metrics["fossil_gas_output_data_role"]
            .eq(ERAA_FOSSIL_GAS_DATA_ROLE)
            .all()
            and metrics["fossil_gas_is_observed_actual"].eq(False).all()
        ),
        "candidate_location_fixed_ex_ante": candidate_location_fixed,
        "baseline_solved_once_or_reused_from_exact_cache": bool(
            cache_hit or baseline_metrics["variant"] == "baseline"
        ),
        "candidate_charge_ceiling_not_binding": bool(
            not metrics["candidate_charge_ceiling_binding"].fillna(False).any()
        ),
        "all_models_optimal": bool(
            metrics["gurobi_status"].fillna(2).eq(2).all()
        ),
        "objective_components_reconcile": bool(
            metrics["objective_component_error_eur"].fillna(0).le(
                np.maximum(
                    1e-3,
                    metrics["objective_eur_year"].abs().fillna(0) * 1e-8,
                )
            ).all()
        ),
        "reconstructed_physical_residuals_below_tolerance": bool(
            maximum_reconstructed_physical_residual
            <= args.physical_residual_tolerance
        ),
        "same_hour_charge_discharge_below_relative_or_numerical_tolerance": bool(
            (
                (
                    metrics["maximum_simultaneous_charge_discharge_ratio"]
                    .fillna(np.inf)
                    .le(args.simultaneous_dispatch_relative_tolerance)
                )
                | (
                    metrics["maximum_simultaneous_charge_discharge_mwh"]
                    .fillna(np.inf)
                    .le(args.simultaneous_dispatch_tolerance_mwh)
                )
            ).all()
        ),
        "candidate_fixed_cost_analytic_self_test": (
            fixed_cost_analytic_self_test()
        ),
    }
    candidate_metrics = metrics.loc[
        metrics["variant"].isin(["joint", "repeat_use_fixed_firm"])
    ]
    confirmation_gates = {
        "full_8760_hour_task": full_year,
        "full_year_hydro_source_discrepancy_excess_below_tolerance": bool(
            full_year
            and metrics["hydro_contract_excess_slack_mwh"]
            .fillna(np.inf)
            .le(args.hydro_constraint_slack_tolerance_mwh)
            .all()
        ),
        "deterministic_expected_availability_LCC_screen_completed": bool(
            not candidate_metrics.empty
            and candidate_metrics["elcc_complete"].eq(True).all()
        ),
        "ELCC_completed_for_every_configuration": False,
        "standard_EENS_LOLE_ELCC_from_sequential_ensemble": False,
        "official_EV_and_HP_iDSR_profiles_modelled": bool(
            execution_gates["official_EV_HP_window_energy_conserved"]
        ),
        "sourced_other_RES_cost_sensitivity_suite_completed": False,
        "shared_candidate_charge_interface_across_paired_chronologies": False,
        "shared_first_stage_across_weather_and_outage_scenarios": False,
        "IE_M0_M1_M3_suite_compared": False,
        "post_eva_fleet_exact_after_IE_suite": False,
        "paired_chronology_weather_configuration_suite_aggregated": False,
        "official_CORE_and_Nordic_flow_based_domains_modelled": False,
        "fifteen_sequential_outage_traces_completed": False,
        "detailed_DSR_high_level_crosswalk_suite_reconciled": False,
        "co_reordered_hydro_sensitivity_completed": False,
        "other_RES_quarantine_vs_dispatchable_parameter_grid_completed": False,
        "must_run_source_post_eva_reconciliation_suite_completed": False,
        "technology_specific_site_masks_applied_to_key_boundaries": False,
        "candidate_location_endogenous_within_site_masks": False,
    }
    qa = {
        "script_version": SCRIPT_VERSION,
        "reservoir_endpoint_contract_version": (
            RESERVOIR_ENDPOINT_CONTRACT_VERSION
        ),
        "created_utc": utc_now(),
        "backend": args.backend,
        "model_scope": "networked_NTC_screen",
        "single_weather_screen": True,
        "weather_scenario": args.weather_scenario,
        "chronology": args.chronology,
        "deployment_scope": args.deployment_scope,
        "hours": system.hours,
        "data_roles": {
            "this_runner": ERAA_RUNNER_DATA_ROLE,
            "fossil_gas_output": ERAA_FOSSIL_GAS_DATA_ROLE,
            "historical_empirical_main_task": (
                HISTORICAL_FOSSIL_GAS_TASK_ROLE
            ),
            "historical_actual_fossil_gas_loaded_by_this_runner": False,
            "residual_load_is_historical_gas_proxy": False,
        },
        "baseline_cache_key": cache_key,
        "baseline_cache_hit": cache_hit,
        "candidate_total_discharge_power_mw": float(candidate_pdis.sum()),
        "candidate_deployment_nodes": int(
            np.count_nonzero(candidate_pdis > REPORT_TOL)
        ),
        "currency_cny_per_eur": currency_rate,
        "maximum_reconstructed_physical_residual": (
            maximum_reconstructed_physical_residual
        ),
        "physical_residual_columns": physical_residual_columns,
        "execution_gates": execution_gates,
        "confirmation_gates": confirmation_gates,
        "diagnostic_gates": {
            "hydro_contract_realised_zero_slack_informational_only": bool(
                metrics["hydro_contract_violation_mwh"]
                .fillna(np.inf)
                .le(args.hydro_constraint_slack_tolerance_mwh)
                .all()
            ),
            "hydro_structural_slack_lower_bound_reconciled": (
                hydro_contract_components_audited
            ),
            "development_terminal_boundary_condition_active": (
                not full_year
            ),
            "annual_hydro_terminal_endpoint_enforced_in_this_run": (
                full_year
            ),
            "annual_hydro_terminal_endpoint_loaded_but_replaced_by_initial_for_development": (
                not full_year
            ),
            "deterministic_reliability_screen_only": True,
            "official_hourly_ENS_external_calibration_pending": True,
        },
        "hydro_contract_slack_audit": {
            "reservoir_endpoint_contract_version": (
                RESERVOIR_ENDPOINT_CONTRACT_VERSION
            ),
            "official_ror_structural_lower_bound_mwh": float(
                audit["system"]["hydro"][
                    "official_ror_structural_lower_bound_mwh"
                ]
            ),
            "development_terminal_boundary_lower_bound_mwh": float(
                audit["system"]["hydro"][
                    "development_terminal_boundary_lower_bound_mwh"
                ]
            ),
            "development_terminal_boundary_condition_active": (
                not full_year
            ),
            "maximum_realised_excess_slack_mwh": float(
                metrics["hydro_contract_excess_slack_mwh"]
                .fillna(np.inf)
                .max()
            ),
            "maximum_level_trajectory_soft_slack_mwh": float(
                metrics["hydro_level_trajectory_soft_slack_mwh"]
                .fillna(np.inf)
                .max()
            ),
        },
        "full_year": full_year,
        "evidence_status": (
            "full_year_system_cost_only_no_reliability_admission"
            if full_year and args.skip_elcc and all(execution_gates.values())
            else (
                "full_network_task_complete_pending_confirmatory_suite"
                if full_year and all(execution_gates.values())
                else "development_or_execution_gate_incomplete"
            )
        ),
        "source_audit": audit,
    }
    qa["pass"] = bool(all(execution_gates.values()))
    # A single WS/chronology/configuration task can never promote itself to
    # manuscript evidence.  Promotion belongs to the independent suite
    # aggregator after ELCC, M0/M1/M3, iDSR and paired-weather gates pass.
    qa["manuscript_evidence_admitted"] = False
    atomic_write_json(output / "QA.json", qa)
    atomic_write_json(
        output / "run_manifest.json",
        {
            "script_version": SCRIPT_VERSION,
            "reservoir_endpoint_contract_version": (
                RESERVOIR_ENDPOINT_CONTRACT_VERSION
            ),
            "created_utc": qa["created_utc"],
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "source_files": {
                str(path): sha256(path)
                for path in (
                    Path(__file__).resolve(),
                    CORE_PATH.resolve(),
                    ZONE_RUNNER_PATH.resolve(),
                    args.input.resolve() / "QA.json",
                    args.chronology_root.resolve() / "QA.json",
                    args.dispatch_root.resolve() / "QA.json",
                    args.flexible_root.resolve() / "QA.json",
                    args.idsr_root.resolve() / "QA.json",
                    args.idsr_root.resolve() / "idsr_hourly_2035.csv.gz",
                    args.idsr_root.resolve() / "idsr_windows_2035.csv",
                    args.idsr_root.resolve() / "idsr_btm_battery_2035.csv",
                    args.idsr_root.resolve()
                    / "additional_dispatchable_consumption_2035.csv",
                    args.must_run_file.resolve(),
                    args.reservoir_level_file.resolve(),
                    args.catalog.resolve(),
                    args.technology_library.resolve(),
                )
            },
        },
    )
    return qa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--chronology-root", type=Path, default=DEFAULT_CHRONOLOGY)
    parser.add_argument("--dispatch-root", type=Path, default=DEFAULT_DISPATCH)
    parser.add_argument("--flexible-root", type=Path, default=DEFAULT_FLEXIBLE)
    parser.add_argument("--idsr-root", type=Path, default=DEFAULT_IDSR)
    parser.add_argument("--must-run-file", type=Path, default=DEFAULT_MUST_RUN)
    parser.add_argument(
        "--reservoir-level-file",
        type=Path,
        default=DEFAULT_RESERVOIR_LEVELS,
        help=(
            "Released annual hydro start/end levels. TSO-provided values "
            "precede the explicit ERAA Annex 2 section 6.1 hierarchy: fixed "
            "week-52/53 terminal trajectory, genuinely weather-specific "
            "next-WS initial level, current-WS week-52 midpoint, then 50%."
        ),
    )
    parser.add_argument(
        "--must-run-case",
        choices=("on", "off"),
        default="on",
        help=(
            "Central screen applies released broad-fuel must-run ratios to "
            "calendar-fixed amended post-EVA available capacity; off is a "
            "required sensitivity."
        ),
    )
    parser.add_argument(
        "--power-to-heat-availability-case",
        choices=("zero", "nameplate"),
        default="zero",
        help=(
            "The released P2H file has no hourly heat-availability profile. "
            "Zero is the conservative central bound; nameplate is an explicit "
            "upper-bound sensitivity. Electrolysers use released nameplate."
        ),
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--technology-library", type=Path, default=DEFAULT_TECH_LIBRARY)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--weather-scenario", default="WS01")
    parser.add_argument(
        "--chronology", choices=("dispersed", "clustered"), default="dispersed"
    )
    parser.add_argument(
        "--deployment-scope",
        choices=("eu27", "extended_europe"),
        default="eu27",
    )
    parser.add_argument(
        "--nl0a-treatment",
        choices=("paired_exclusion", "map_wind_to_NL00"),
        default="paired_exclusion",
        help=(
            "Central treatment removes the isolated 50 MW wind and 50 MW "
            "electrolyser together; the mapping option is a topology "
            "sensitivity and does not admit the quarantined electrolyser."
        ),
    )
    parser.add_argument(
        "--capacity-case",
        default="amended_annex3_risk_averse_enhanced_hurdle",
    )
    parser.add_argument(
        "--split-case", default="retirement_feasible_max_ccgt"
    )
    parser.add_argument(
        "--ie-crosswalk-case",
        choices=("M0", "M1", "M3"),
        default="M0",
        help=(
            "M0 quarantines the unresolved -110 MW retirement; M1 applies it "
            "to the 118 MW released Biofuel host; M3 retains IE physically "
            "but flags it for exclusion from country attribution."
        ),
    )
    parser.add_argument(
        "--biofuel-variable-cost-eur-mwh",
        type=float,
        default=129.23943,
        help=(
            "Transparent dispatch sensitivity: 2035 released biomethane "
            "13.99327 EUR/GJ at 40%% efficiency plus 3.3 EUR/MWh VOM."
        ),
    )
    parser.add_argument(
        "--biofuel-availability-fraction", type=float, default=0.90
    )
    parser.add_argument(
        "--other-res-treatment",
        choices=("dispatchable_sensitivity", "quarantine"),
        default="quarantine",
        help=(
            "Quarantine is the lower-bound central screen because the release "
            "does not provide a common hourly profile or marginal cost. The "
            "dispatchable sensitivity admits all available-on-market Small "
            "biomass, Waste, Geothermal and Marine capacity using explicit "
            "cost and availability arguments. Confirmation requires both "
            "bounds and a sourced parameter grid."
        ),
    )
    parser.add_argument(
        "--other-res-variable-cost-eur-mwh",
        type=float,
        default=50.0,
    )
    parser.add_argument(
        "--other-res-availability-fraction",
        type=float,
        default=0.92,
    )
    parser.add_argument(
        "--config-id", action="append", required=True, help="repeatable"
    )
    parser.add_argument(
        "--currency-scenario",
        choices=("low", "central", "high"),
        default="central",
    )
    parser.add_argument("--tranche-share", type=float, default=0.01)
    parser.add_argument(
        "--incremental-charge-cost-multiplier", type=float, default=1.0
    )
    parser.add_argument("--lfp-degradation-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--maximum-charge-to-discharge-ratio", type=float, default=10.0
    )
    parser.add_argument("--value-of-lost-load", type=float, default=15_000.0)
    parser.add_argument(
        "--maximum-firm-addition-mw", type=float, default=100_000.0
    )
    parser.add_argument("--hydro-variable-cost", type=float, default=5.0)
    parser.add_argument(
        "--hydro-constraint-violation-penalty",
        type=float,
        default=10_000.0,
        help=(
            "EUR/MWh penalty for ERAA-permitted soft reservoir-level "
            "trajectories and separately quarantined zero-stock source "
            "contract discrepancies; must be below VoLL. Stocked-resource "
            "power and period-energy constraints remain hard."
        ),
    )
    parser.add_argument(
        "--hydro-constraint-slack-tolerance-mwh",
        type=float,
        default=1e-4,
    )
    parser.add_argument("--vre-spill-penalty", type=float, default=1e-6)
    parser.add_argument("--hydro-spill-penalty", type=float, default=1e-7)
    parser.add_argument("--flow-regularization", type=float, default=1e-8)
    parser.add_argument(
        "--existing-storage-throughput-regularization",
        type=float,
        default=1e-4,
        help=(
            "EUR/MWh on both charge and discharge; must exceed every spill "
            "penalty to rule out loss-making same-hour cycling"
        ),
    )
    parser.add_argument(
        "--existing-battery-degradation-cost-eur-mwh-discharged",
        type=float,
        default=0.0,
        help=(
            "Explicit marginal degradation sensitivity for both in-market "
            "utility batteries and the redispatched price-sensitive BTM "
            "battery share. Zero represents sunk existing assets and is not "
            "promoted as a central evidence value; the confirmatory suite "
            "must report independently sourced non-zero sensitivities."
        ),
    )
    parser.add_argument(
        "--simultaneous-dispatch-tolerance-mwh",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--simultaneous-dispatch-relative-tolerance",
        type=float,
        default=1e-6,
        help=(
            "Maximum same-hour charge/discharge overlap divided by total "
            "charge-plus-discharge throughput for each storage family."
        ),
    )
    parser.add_argument(
        "--physical-residual-tolerance",
        type=float,
        default=1e-4,
        help=(
            "Maximum reconstructed nodal, stock, network, DSR, must-run or "
            "iDSR-window residual admitted by a task execution gate."
        ),
    )
    parser.add_argument("--backend", choices=("gurobi",), default="gurobi")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument(
        "--method",
        type=int,
        default=2,
        help="Gurobi LP method; 2 is barrier",
    )
    parser.add_argument("--feasibility-tolerance", type=float, default=1e-7)
    parser.add_argument("--optimality-tolerance", type=float, default=1e-7)
    parser.add_argument("--time-limit", type=float, default=0.0)
    parser.add_argument("--elcc-tolerance-mw", type=float, default=10.0)
    parser.add_argument("--elcc-max-iterations", type=int, default=30)
    parser.add_argument(
        "--elcc-max-upper-multiplier", type=float, default=8.0
    )
    parser.add_argument(
        "--elcc-reliability-relative-tolerance", type=float, default=1e-8
    )
    parser.add_argument(
        "--elcc-eens-absolute-tolerance-mwh", type=float, default=1e-5
    )
    parser.add_argument("--elcc-lole-tolerance-h", type=float, default=1e-6)
    parser.add_argument(
        "--skip-elcc",
        action="store_true",
        help=(
            "Skip the deterministic expected-availability LCC search. A full-year "
            "run remains useful as a system-cost and execution pilot but is "
            "explicitly non-evidential until the reliability search is completed."
        ),
    )
    parser.add_argument(
        "--development-hour-limit",
        type=int,
        default=0,
        help="whole-day truncation; automatically non-evidential",
    )
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--keep-candidate-dispatch", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    qa = run(args)
    print(json.dumps(qa, ensure_ascii=False, indent=2))
    return 0 if bool(qa.get("pass", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
