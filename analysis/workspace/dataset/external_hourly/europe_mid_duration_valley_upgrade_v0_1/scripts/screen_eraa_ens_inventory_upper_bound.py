#!/usr/bin/env python3
"""Screen the duration structure of ERAA ENS with a transparent upper bound.

This is not a storage-dispatch or ELCC model.  It asks a narrower question:
given each reported local ENS episode, how much energy could a storage tranche
possibly cover if it were fully restored before every episode?  The resulting
curve is deliberately optimistic because it ignores inter-event recharge,
network redispatch, efficiency losses and endogenous dispatch.  It is useful
only for checking whether the released adequacy events contain a long-duration
inventory tail worth testing in the confirmatory model.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


EXCLUDED_AGGREGATE_ZONE_CODES = {
    "DK00",
    "ISEM",
    "IT00",
    "LU00",
    "NO00",
    "SE00",
}
DEFAULT_DURATIONS = (4, 8, 12, 24, 48, 72, 96, 168, 240, 336, 720)
N_WEATHER = 36
N_OUTAGE = 15


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_mean_demand_mw(dashboard_zip: Path) -> pd.DataFrame:
    with zipfile.ZipFile(dashboard_zip) as archive:
        matches = [
            name
            for name in archive.namelist()
            if name.endswith("/Aggregated_Demand.csv")
            or name == "Aggregated_Demand.csv"
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected one Aggregated_Demand.csv, found {len(matches)}"
            )
        raw = archive.read(matches[0])
    demand = pd.read_csv(io.BytesIO(raw))
    selected = demand.loc[
        demand["data version"].eq("ERAA 2025 final")
        & demand["Measure"].eq("Hourly Demand (GW)")
        & demand["Measure_Type"].eq("AVG"),
        ["Target_Year", "Market_Node", "Value"],
    ].copy()
    selected = selected.rename(
        columns={
            "Target_Year": "target_year",
            "Market_Node": "bidding_zone",
            "Value": "mean_load_gw",
        }
    )
    selected = selected.loc[
        ~selected["bidding_zone"].isin(EXCLUDED_AGGREGATE_ZONE_CODES)
    ].copy()
    selected["mean_load_mw"] = selected.pop("mean_load_gw") * 1_000.0
    if selected.duplicated(["target_year", "bidding_zone"]).any():
        raise ValueError("duplicate final mean-demand rows")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--events",
        type=Path,
        default=root / "processed" / "eraa_2025_atomic_zone_ens_events.csv.gz",
    )
    parser.add_argument(
        "--dashboard-zip",
        type=Path,
        default=root / "raw" / "eraa_2025" / "ERAA_2025_Dashboard_RawData.zip",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            root
            / "artifacts"
            / "research"
            / "ERAA_2025_ENS_INVENTORY_UPPER_BOUND.csv"
        ),
    )
    parser.add_argument("--tranche-share", type=float, default=0.01)
    parser.add_argument(
        "--durations",
        nargs="+",
        type=float,
        default=list(DEFAULT_DURATIONS),
    )
    args = parser.parse_args()
    if not 0 < args.tranche_share < 1:
        raise ValueError("tranche share must lie between zero and one")
    durations = np.asarray(sorted(set(args.durations)), dtype=float)
    if (durations <= 0).any():
        raise ValueError("durations must be positive")

    events = pd.read_csv(args.events)
    required = {
        "target_year",
        "weather_scenario",
        "forced_outage_scenario",
        "bidding_zone",
        "duration_h",
        "ens_mwh",
    }
    missing = required.difference(events)
    if missing:
        raise ValueError(f"event table lacks {sorted(missing)}")
    if events["bidding_zone"].isin(EXCLUDED_AGGREGATE_ZONE_CODES).any():
        raise ValueError("aggregate/alias zones entered the atomic event table")
    demand = load_mean_demand_mw(args.dashboard_zip)
    events = events.merge(
        demand,
        how="left",
        on=["target_year", "bidding_zone"],
        validate="many_to_one",
    )
    events["tranche_power_mw"] = (
        events["mean_load_mw"] * args.tranche_share
    )
    matched = events["tranche_power_mw"].notna()

    rows: list[dict[str, float | int | str]] = []
    scenario_count = N_WEATHER * N_OUTAGE
    for target_year, frame in events.groupby("target_year", sort=True):
        baseline_all = float(frame["ens_mwh"].sum() / scenario_count)
        matched_frame = frame.loc[frame["tranche_power_mw"].notna()].copy()
        baseline_matched = float(
            matched_frame["ens_mwh"].sum() / scenario_count
        )
        previous_avoided = 0.0
        for duration in durations:
            possible = np.minimum.reduce(
                [
                    matched_frame["ens_mwh"].to_numpy(float),
                    (
                        matched_frame["tranche_power_mw"]
                        * matched_frame["duration_h"]
                    ).to_numpy(float),
                    (
                        matched_frame["tranche_power_mw"] * duration
                    ).to_numpy(float),
                ]
            )
            avoided = float(possible.sum() / scenario_count)
            rows.append(
                {
                    "target_year": int(target_year),
                    "duration_h": float(duration),
                    "tranche_share_of_zone_mean_load": args.tranche_share,
                    "expected_baseline_ens_all_zones_mwh": baseline_all,
                    "expected_baseline_ens_matched_zones_mwh": baseline_matched,
                    "expected_ens_avoidable_upper_bound_mwh": avoided,
                    "incremental_avoidable_mwh_from_previous_node": (
                        avoided - previous_avoided
                    ),
                    "avoidable_share_of_all_ens": (
                        avoided / baseline_all if baseline_all else np.nan
                    ),
                    "avoidable_share_of_matched_ens": (
                        avoided / baseline_matched
                        if baseline_matched
                        else np.nan
                    ),
                    "evidence_status": (
                        "fully_recharged_each_event_inventory_upper_bound"
                    ),
                }
            )
            previous_avoided = avoided

    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    qa = {
        "pass": True,
        "evidence_status": "screen_only_not_dispatch_not_elcc",
        "event_input": str(args.events),
        "event_input_sha256": sha256(args.events),
        "dashboard_input": str(args.dashboard_zip),
        "dashboard_input_sha256": sha256(args.dashboard_zip),
        "target_years": sorted(int(value) for value in events.target_year.unique()),
        "durations_h": durations.tolist(),
        "tranche_share": args.tranche_share,
        "event_rows": int(len(events)),
        "matched_event_rows": int(matched.sum()),
        "unmatched_event_rows": int((~matched).sum()),
        "unmatched_zones": sorted(
            events.loc[~matched, "bidding_zone"].unique().tolist()
        ),
        "assumptions": [
            "storage is fully recharged before every local ENS event",
            "round-trip losses are ignored",
            "network and dispatch feedbacks are ignored",
            "event-hour profile is replaced by an energy-and-duration upper bound",
        ],
    }
    qa_path = args.output.with_suffix(".qa.json")
    qa_path.write_text(
        json.dumps(qa, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# ERAA 2025 ENS 库存时长上界筛查",
        "",
        (
            "该筛查假定每个本地 ENS 事件开始前储能均已充满，因此是长时库存"
            "价值的乐观上界，不是调度结果、ELCC 或中时长谷证据。"
        ),
        "",
    ]
    for target_year, frame in result.groupby("target_year", sort=True):
        last_material = frame.loc[
            frame["incremental_avoidable_mwh_from_previous_node"]
            > 0.001
            * frame["expected_baseline_ens_all_zones_mwh"]
        ]
        saturation = (
            float(last_material["duration_h"].max())
            if not last_material.empty
            else np.nan
        )
        lines.extend(
            [
                f"## {int(target_year)}",
                "",
                (
                    f"1% 分区平均负荷功率的乐观可避免 ENS 上界在 "
                    f"{saturation:g} h 之后，每增加一个测试时长节点带来的收益"
                    "已低于该年基准 ENS 的 0.1%。"
                ),
                "",
            ]
        )
    args.output.with_suffix(".md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
