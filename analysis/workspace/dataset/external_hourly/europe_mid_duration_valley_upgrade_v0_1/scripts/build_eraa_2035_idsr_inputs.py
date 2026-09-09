#!/usr/bin/env python3
"""Standardize ERAA 2035 implicit-DSR and dispatchable-consumption inputs.

The official DemandData archive contains, for every demand node, hourly EV and
heat-pump load components for 36 weather scenarios.  ERAA also publishes the
TSO-estimated price-sensitive share of each component.  This script multiplies
the two without changing chronology and writes model-facing zone shards.

It does not optimize demand shifting.  The downstream network model must apply
the ERAA 2025 time-window constraints: four six-hour windows per day for heat
pumps, and two six-hour plus one twelve-hour nighttime window for EVs, with
energy conserved inside every window.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import zipfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / "raw" / "eraa_2025" / "ERAA_2025_DemandData.zip"
DEFAULT_ARCHIVE_QA = (
    ROOT
    / "raw"
    / "eraa_2025"
    / "ERAA_2025_DemandData.full_archive_qa.json"
)
DEFAULT_PHASE1 = ROOT / "processed" / "eraa_2035_phase1_inputs_v0_1"
DEFAULT_OUTPUT = ROOT / "processed" / "eraa_2035_idsr_inputs_v0_1"
WEATHER_SCENARIOS = tuple(f"WS{number:02d}" for number in range(1, 37))
HOURS_PER_YEAR = 8_760


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path, compression: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".csv.gz" if compression else ".csv"
    if not path.name.endswith(suffix):
        raise ValueError(path)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
        newline="",
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(
            temporary,
            index=False,
            compression=(
                {"method": "gzip", "compresslevel": 6, "mtime": 0}
                if compression
                else None
            ),
            lineterminator="\n",
            float_format="%.10g",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def prepare_output(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"output exists: {path}; pass --force")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def read_component(
    archive: zipfile.ZipFile, zone: str, component: str
) -> pd.DataFrame:
    member = (
        "Demand data/Detailed demand/"
        f"{zone}_Demand_{component}_2035_National Trends.csv"
    )
    if member not in archive.namelist():
        raise FileNotFoundError(member)
    frame = pd.read_csv(archive.open(member))
    required = {"Date", "Month", "Day", "Hour", *WEATHER_SCENARIOS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{member}: missing {sorted(missing)}")
    if len(frame) != HOURS_PER_YEAR:
        raise ValueError(
            f"{member}: expected {HOURS_PER_YEAR} rows, observed {len(frame)}"
        )
    values = frame.loc[:, WEATHER_SCENARIOS].apply(
        pd.to_numeric, errors="raise"
    )
    array = values.to_numpy(dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{member}: invalid hourly values")
    if component == "EVpart" and (array < 0).any():
        raise ValueError(f"{member}: negative EV load")
    if not np.array_equal(
        pd.to_numeric(frame["Hour"], errors="raise").to_numpy(dtype=int),
        np.arange(HOURS_PER_YEAR) % 24,
    ):
        raise ValueError(f"{member}: noncanonical hour-of-day axis")
    return values


def execute(args: argparse.Namespace) -> dict[str, Any]:
    archive_path = args.archive.resolve()
    archive_qa_path = args.archive_qa.resolve()
    phase1 = args.phase1.resolve()
    output = args.output.resolve()
    archive_qa = json.loads(archive_qa_path.read_text(encoding="utf-8"))
    if not archive_qa.get("pass", False):
        raise ValueError("full DemandData archive QA does not pass")
    observed_sha = sha256(archive_path)
    if observed_sha != archive_qa["archive_sha256"]:
        raise ValueError("DemandData archive SHA256 differs from verified QA")

    coverage = pd.read_csv(phase1 / "zone_coverage.csv")
    demand_zones = sorted(
        coverage.loc[
            coverage["expected_demand_zone"].astype(bool), "zone"
        ].astype(str)
    )
    if len(demand_zones) != 52:
        raise ValueError(
            f"expected 52 official demand nodes, observed {len(demand_zones)}"
        )
    prepare_output(output, args.force)

    with zipfile.ZipFile(archive_path) as archive:
        ratio_member = "Demand data/iDSR_ratio.csv"
        ratios = pd.read_csv(archive.open(ratio_member))
        ratios = ratios.loc[
            pd.to_numeric(ratios["TARGET_YEAR"], errors="raise").eq(2035)
        ].copy()
        ratios = ratios.rename(
            columns={
                "MARKET_NODE": "zone",
                "Ratio of price-sensitive consumer - Evs": "ev_ratio",
                "Ratio of price-sensitive consumer - Heat Pumps": "hp_ratio",
                (
                    "Ratio of price-sensitive consumer - "
                    "Non-Market Participating Batteries"
                ): "oom_battery_ratio",
            }
        )
        ratio_columns = ["ev_ratio", "hp_ratio", "oom_battery_ratio"]
        for column in ratio_columns:
            ratios[column] = pd.to_numeric(ratios[column], errors="raise")
            if not ratios[column].between(0, 1).all():
                raise ValueError(f"{column}: ratios outside [0,1]")
        ratios = ratios[
            ["zone", "TARGET_YEAR", *ratio_columns]
        ].drop_duplicates("zone")
        missing_ratio = sorted(set(demand_zones).difference(ratios["zone"]))
        if missing_ratio:
            raise ValueError(f"missing 2035 iDSR ratios: {missing_ratio}")

        audit_rows: list[dict[str, Any]] = []
        for zone in demand_zones:
            one_ratio = ratios.set_index("zone").loc[zone]
            ev = read_component(archive, zone, "EVpart")
            hp = read_component(archive, zone, "HPpart")
            ev_flexible = ev.to_numpy(dtype=float) * float(one_ratio.ev_ratio)
            hp_raw = hp.to_numpy(dtype=float)
            # The DFT HP correction can be negative when efficient heat pumps
            # replace electric-resistance heating (Annex 2, section on HP
            # load correction).  A negative correction is not consumptive
            # demand that can be time-shifted.  Retain it in gross demand as a
            # fixed adjustment and expose only the positive HP component to
            # the non-negative iDSR formulation in Annex 2, Appendix 2.
            hp_positive = np.maximum(hp_raw, 0.0)
            hp_negative_fixed = np.minimum(hp_raw, 0.0)
            hp_flexible = hp_positive * float(one_ratio.hp_ratio)

            phase1_path = (
                phase1
                / "hourly_profiles"
                / f"zone={zone}"
                / "hourly_profiles.csv.gz"
            )
            phase1_load = pd.read_csv(
                phase1_path,
                usecols=[
                    "weather_scenario",
                    "hour_of_year",
                    "gross_demand_mw",
                    "btm_net_output_mw",
                    "net_load_after_btm_mw",
                ],
            ).sort_values(["weather_scenario", "hour_of_year"])
            expected_weather = np.repeat(WEATHER_SCENARIOS, HOURS_PER_YEAR)
            expected_hour = np.tile(np.arange(HOURS_PER_YEAR), 36)
            if (
                len(phase1_load) != 36 * HOURS_PER_YEAR
                or not np.array_equal(
                    phase1_load["weather_scenario"].astype(str).to_numpy(),
                    expected_weather,
                )
                or not np.array_equal(
                    phase1_load["hour_of_year"].to_numpy(dtype=int),
                    expected_hour,
                )
            ):
                raise ValueError(f"{phase1_path}: incompatible hour axis")
            flexible_total = (ev_flexible + hp_flexible).flatten(order="F")
            gross_demand_values = phase1_load["gross_demand_mw"].to_numpy(
                dtype=float
            )
            # EVpart and HPpart are components of the released gross-demand
            # series.  They must therefore be checked against gross demand,
            # not against the clipped load after behind-the-meter output.
            # At hours with high BTM PV the latter can be zero even though
            # price-sensitive EV/HP consumption remains strictly positive.
            excess = flexible_total - gross_demand_values
            maximum_excess = float(np.max(excess))
            if maximum_excess > max(
                1e-5, 1e-8 * float(gross_demand_values.max())
            ):
                raise ValueError(
                    f"{zone}: flexible EV+HP exceeds released gross demand by "
                    f"{maximum_excess:g} MW"
                )

            shard = pd.DataFrame(
                {
                    "weather_scenario": expected_weather,
                    "hour_of_year": expected_hour,
                    "ev_component_mw": ev.to_numpy(dtype=float).flatten(
                        order="F"
                    ),
                    "hp_component_mw": hp.to_numpy(dtype=float).flatten(
                        order="F"
                    ),
                    "hp_positive_component_mw": hp_positive.flatten(
                        order="F"
                    ),
                    "hp_negative_fixed_adjustment_mw": (
                        hp_negative_fixed.flatten(order="F")
                    ),
                    "ev_flexible_mw": ev_flexible.flatten(order="F"),
                    "hp_flexible_mw": hp_flexible.flatten(order="F"),
                }
            )
            shard["total_flexible_ev_hp_mw"] = (
                shard["ev_flexible_mw"] + shard["hp_flexible_mw"]
            )
            write_csv(
                shard,
                output
                / "hourly_profiles"
                / f"zone={zone}"
                / "idsr_flexible_load.csv.gz",
                compression=True,
            )
            audit_rows.append(
                {
                    "zone": zone,
                    "ev_ratio": float(one_ratio.ev_ratio),
                    "hp_ratio": float(one_ratio.hp_ratio),
                    "oom_battery_ratio": float(
                        one_ratio.oom_battery_ratio
                    ),
                    "hourly_rows": int(len(shard)),
                    "weather_scenarios": int(
                        shard["weather_scenario"].nunique()
                    ),
                    "annual_ev_flexible_mwh_mean_ws": float(
                        ev_flexible.sum(axis=0).mean()
                    ),
                    "annual_hp_flexible_mwh_mean_ws": float(
                        hp_flexible.sum(axis=0).mean()
                    ),
                    "negative_hp_component_hours": int(
                        np.count_nonzero(hp_negative_fixed)
                    ),
                    "annual_negative_hp_fixed_adjustment_mwh_mean_ws": float(
                        hp_negative_fixed.sum(axis=0).mean()
                    ),
                    "maximum_flexible_share_of_total_load": float(
                        np.divide(
                            flexible_total,
                            gross_demand_values,
                            out=np.zeros_like(flexible_total),
                            where=gross_demand_values > 0,
                        ).max()
                    ),
                    "flexible_load_not_above_gross_demand": True,
                }
            )

        additional = pd.read_csv(
            archive.open("Demand data/Additional dispatchable consumption.csv")
        )
        additional = additional.loc[
            pd.to_numeric(
                additional["TARGET_YEAR"], errors="raise"
            ).eq(2035)
        ].copy()
        additional = additional.rename(
            columns={
                "MARKET_NODE": "zone",
                "PEMMDB_PLANT_TYPE": "consumer_type",
                "MAX CAPACITY (MW)": "maximum_consumption_mw",
                "Weighted Average Electrolyser Efficiency": (
                    "weighted_average_efficiency_pct"
                ),
                "Weighted Average Activation Price": (
                    "activation_price_eur_mwh"
                ),
                "TARGET_YEAR": "target_year",
            }
        )
        numeric = [
            "maximum_consumption_mw",
            "weighted_average_efficiency_pct",
            "activation_price_eur_mwh",
        ]
        for column in numeric:
            additional[column] = pd.to_numeric(
                additional[column], errors="raise"
            )
        if (
            (additional["maximum_consumption_mw"] < 0).any()
            or not np.isfinite(additional[numeric].to_numpy(dtype=float)).all()
        ):
            raise ValueError("invalid dispatchable-consumption parameters")
        additional["has_phase1_profile_node"] = additional["zone"].isin(
            coverage["zone"].astype(str)
        )
        additional = additional.sort_values(
            ["zone", "consumer_type"]
        ).reset_index(drop=True)

    write_csv(ratios.sort_values("zone"), output / "idsr_ratios_2035.csv")
    write_csv(
        pd.DataFrame(audit_rows).sort_values("zone"),
        output / "idsr_profile_audit.csv",
    )
    write_csv(
        additional,
        output / "additional_dispatchable_consumption_2035.csv",
    )
    qa = {
        "source_archive": str(archive_path),
        "source_archive_sha256": observed_sha,
        "source_archive_zip_integrity_pass": bool(
            archive_qa["zip_integrity_pass"]
        ),
        "target_year": 2035,
        "demand_nodes": len(demand_zones),
        "weather_scenarios": 36,
        "hourly_rows_per_zone": 36 * HOURS_PER_YEAR,
        "idsr_ratio_rows": int(len(ratios)),
        "additional_dispatchable_consumption_rows": int(len(additional)),
        "additional_dispatchable_consumption_capacity_gw": float(
            additional["maximum_consumption_mw"].sum() / 1_000.0
        ),
        "negative_hp_component_hours": int(
            sum(row["negative_hp_component_hours"] for row in audit_rows)
        ),
        "negative_hp_component_treatment": (
            "retained as a fixed gross-demand adjustment; only the positive "
            "HP component is eligible for non-negative time shifting"
        ),
        "all_flexible_ev_hp_not_above_gross_demand": bool(
            all(
                row["flexible_load_not_above_gross_demand"]
                for row in audit_rows
            )
        ),
        "downstream_window_contract": {
            "heat_pumps": "four 6-hour windows per day",
            "electric_vehicles": (
                "two 6-hour windows and one 12-hour nighttime window per day"
            ),
            "energy_conservation": "exactly within each time window",
            "source": (
                "ERAA 2025 Annex 2, section 2.3.2 and Appendix 2"
            ),
        },
    }
    qa["pass"] = bool(
        qa["source_archive_zip_integrity_pass"]
        and qa["demand_nodes"] == 52
        and qa["weather_scenarios"] == 36
        and qa["idsr_ratio_rows"] >= 52
        and qa["all_flexible_ev_hp_not_above_gross_demand"]
    )
    write_json(output / "QA.json", qa)
    return qa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--archive-qa", type=Path, default=DEFAULT_ARCHIVE_QA)
    parser.add_argument("--phase1", type=Path, default=DEFAULT_PHASE1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    qa = execute(parse_args())
    print(json.dumps(qa, ensure_ascii=False, indent=2))
    return 0 if qa.get("pass", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
