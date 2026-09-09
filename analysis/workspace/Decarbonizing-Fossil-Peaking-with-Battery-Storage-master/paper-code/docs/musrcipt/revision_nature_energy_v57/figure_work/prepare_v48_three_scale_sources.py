#!/usr/bin/env python3
"""Prepare frozen source data for the v48 three-scale empirical atlas.

This script does not solve storage dispatch.  It freezes the directly observed
task/recharge definitions, derives map-level event metrics, and gathers the
already solved duration and strict-valley results used by Figures 2 and 5.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORKSPACE = ROOT.parents[5]
OUT = HERE / "source_data_v48"
OUT.mkdir(exist_ok=True)

CHINA_PANEL = ROOT / "cross_scale_cases/china_province_panel/china_province_event_task_panel.csv.gz"
JAPAN_PANEL = ROOT / "cross_scale_cases/japan_community_panel/japan_community_event_task_panel.csv.gz"
CROSS = ROOT / "cross_scale_cases/summary_v0_2"
POSTPROCESS = ROOT / "cross_scale_cases/postprocess_cross_scale_v0_2.py"
EUROPE_MAP = HERE / "source_data/figure2_panel_a_country_map_summary.csv"
EUROPE_DURATION = ROOT / "source_data/v43_figure5b_duration_envelope_bootstrap_10000.csv"
EUROPE_VALLEYS = HERE / "source_data/figure5_panel_e_conditional_valley_records.csv"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def longest_true_run(values: pd.Series) -> int:
    array = values.astype(bool).to_numpy()
    if not array.any():
        return 0
    changes = np.r_[True, array[1:] != array[:-1]]
    groups = np.cumsum(changes)
    return int(pd.Series(array).groupby(groups).sum().max())


def mesh_center(mesh_id: int | str) -> tuple[float, float]:
    """Return the centre of a Japanese 500 m standard grid cell.

    The first eight digits identify a third-level mesh.  The ninth digit uses
    the official 1=SW, 2=SE, 3=NW, 4=NE half-mesh convention.
    """
    code = str(int(mesh_id)).zfill(9)
    p, q = int(code[:2]), int(code[2:4])
    r, s, t, u = map(int, code[4:8])
    quadrant = int(code[8])
    lat = p / 1.5 + r * (5 / 60) + t * (30 / 3600)
    lon = q + 100 + s * (7.5 / 60) + u * (45 / 3600)
    lat_half, lon_half = 15 / 3600, 22.5 / 3600
    north = quadrant in (3, 4)
    east = quadrant in (2, 4)
    lat += (lat_half if north else 0) + lat_half / 2
    lon += (lon_half if east else 0) + lon_half / 2
    return lat, lon


def china_map_source() -> pd.DataFrame:
    panel = pd.read_csv(CHINA_PANEL)
    daily = (
        panel.drop_duplicates(["province", "source_day_id"])
        .sort_values(["province", "source_day_id"])
    )
    rows = []
    for province, group in daily.groupby("province", sort=True):
        hourly = panel[panel.province.eq(province)]
        rows.append(
            {
                "province": province,
                "longest_extreme_peaking_run_days": longest_true_run(group.event_flag_q75),
                "annual_observed_peaking_task_gwh": hourly.task_c1_mw.sum() / 1e3,
                "extreme_peaking_share_pct": 100 * hourly.task_c3_q75_mw.sum() / hourly.task_c1_mw.sum(),
                "recharge_to_peaking_ratio": hourly.r2_potential_recharge_mw.sum() / hourly.task_c1_mw.sum(),
                "q10_baseline_applied": False,
                "q75_role": "extreme_peaking_day_label_only",
            }
        )
    return pd.DataFrame(rows)


def japan_map_source() -> pd.DataFrame:
    panel = pd.read_csv(JAPAN_PANEL)
    case_cols = ["prefecture", "mesh_id", "fiscal_year"]
    daily = panel.drop_duplicates(case_cols + ["calendar_day"]).sort_values(case_cols + ["calendar_day"])
    rows = []
    for key, group in daily.groupby(case_cols, sort=True):
        hourly = panel[
            panel.prefecture.eq(key[0]) & panel.mesh_id.eq(key[1]) & panel.fiscal_year.eq(key[2])
        ]
        lat, lon = mesh_center(key[1])
        rows.append(
            {
                "prefecture": key[0],
                "mesh_id": int(key[1]),
                "fiscal_year": int(key[2]),
                "latitude": lat,
                "longitude": lon,
                "longest_extreme_import_run_days": longest_true_run(group.event_flag_q75),
                "annual_residual_import_mwh": hourly.task_c1_mw.sum(),
                "extreme_import_share_pct": 100 * hourly.task_c3_q75_mw.sum() / hourly.task_c1_mw.sum(),
                "pv_reverse_flow_to_task_ratio": hourly.r2_potential_recharge_mw.sum() / hourly.task_c1_mw.sum(),
            }
        )
    return pd.DataFrame(rows)


def selected_cross_scale_valleys() -> tuple[pd.DataFrame, pd.DataFrame]:
    post = load_module("v48_cross_scale_postprocess", POSTPROCESS)
    arms = pd.concat(
        [pd.read_csv(path) for path in sorted((ROOT / "cross_scale_cases").glob("full_*_shard*/cross_scale_results.csv.gz"))],
        ignore_index=True,
    )
    costs = pd.read_csv(ROOT / "cost_and_decomposition/cost_catalogue_2025_2035_long.csv")
    selections = [
        ("china", 2030, 103.79921785880974, 1.0, "China central 2030"),
        ("china", 2035, 300.0, 1.0, "China clustered-valley phase"),
        ("japan", 2035, 600.0, 1.0, "Japan clustered-valley phase"),
    ]
    records, envelopes = [], []
    for application, year, firm, multiplier, label in selections:
        valued = post.ledger(
            arms[arms.application_case.eq(application)], costs,
            scenario="S0", cost_year=year, firm_value=firm,
            ldes_cost_multiplier=multiplier,
        )
        _, envelope, valley = post.summarize_one(valued)
        for frame in (envelope, valley):
            frame["selection_label"] = label
            frame["cost_year"] = year
            frame["firm_value_eur_kw_year"] = firm
            frame["ldes_cost_multiplier"] = multiplier
        records.append(valley)
        envelopes.append(envelope)
    return pd.concat(records, ignore_index=True), pd.concat(envelopes, ignore_index=True)


def main() -> None:
    europe = pd.read_csv(EUROPE_MAP)
    china = china_map_source()
    japan = japan_map_source()
    europe.to_csv(OUT / "figure2a_europe_map.csv", index=False)
    china.to_csv(OUT / "figure2b_china_map.csv", index=False)
    japan.to_csv(OUT / "figure2c_japan_map.csv", index=False)

    distribution = pd.concat(
        [
            pd.DataFrame({"scale": "Europe", "event_run_days": europe.median_longest_event_days}),
            pd.DataFrame({"scale": "China", "event_run_days": china.longest_extreme_peaking_run_days}),
            pd.DataFrame({"scale": "Japan", "event_run_days": japan.longest_extreme_import_run_days}),
        ],
        ignore_index=True,
    )
    distribution.to_csv(OUT / "figure2d_three_scale_event_runs.csv", index=False)

    data_contract = pd.DataFrame(
        [
            ("Europe", "actual hourly fossil-gas generation above a month-hour q10 floor", "synchronous wind-solar energy balance", "q75 event tail", 15, 132, 1156320),
            ("China", "observed provincial gas-peaking task; no q10 subtraction", "hourly wind-solar curtailment", "q75 extreme peaking days", 26, 26, 227760),
            ("Japan", "community grid import above a month-hour q10 floor", "distributed-PV reverse flow", "q75 extreme import days", 5, 27, 236520),
        ],
        columns=["scale", "task_definition", "recharge_definition", "q75_role", "independent_blocks", "case_years", "hours"],
    )
    data_contract.to_csv(OUT / "figure2e_three_scale_data_contract.csv", index=False)

    duration = pd.read_csv(CROSS / "cross_scale_efc_duration_summary.csv")
    duration = duration[duration.scenario.eq("S0") & duration.cost_year.eq(2035)].copy()
    duration.to_csv(OUT / "figure5bc_china_japan_duration_response.csv", index=False)
    pd.read_csv(EUROPE_DURATION).to_csv(OUT / "figure5a_europe_duration_response.csv", index=False)

    valley_records, selected_envelopes = selected_cross_scale_valleys()
    valley_records.to_csv(OUT / "figure5e_selected_cross_scale_valley_records.csv", index=False)
    selected_envelopes.to_csv(OUT / "figure5f_selected_cross_scale_envelopes.csv", index=False)

    european_records = pd.read_csv(EUROPE_VALLEYS)
    europe_counts = european_records.groupby("chronology", as_index=False).agg(
        n_cases=("country_year_id", "size"), strict_valleys=("strict_valley", "sum"),
        mean_width_h=("valley_width_h", lambda x: x[x > 0].mean()),
    )
    europe_counts["scale"] = "Europe"
    europe_counts["selection_label"] = "E=98.54, Q=0, F=600, 2035 costs"

    other_counts = valley_records.groupby(
        ["application_case", "selection_label", "chronology"], as_index=False
    ).agg(
        n_cases=("case_id", "size"), strict_valleys=("strict_valley", "sum"),
        mean_width_h=("valley_width_h", lambda x: x[x > 0].mean()),
    )
    other_counts["scale"] = other_counts.application_case.map({"china": "China", "japan": "Japan"})
    visibility = pd.concat(
        [europe_counts, other_counts.drop(columns="application_case")], ignore_index=True, sort=False
    )
    visibility.to_csv(OUT / "figure5d_three_scale_strict_valley_counts.csv", index=False)

    china_qa = json.loads((ROOT / "cross_scale_cases/china_province_panel/qa.json").read_text())
    qa = {
        "china_task_definition": china_qa["task_definition"],
        "china_q10_baseline_applied": china_qa["q10_baseline_applied"],
        "china_q75_role": china_qa["q75_role"],
        "china_task_energy_matches_source": china_qa["annual_task_energy_matches_source_after_kwh_to_mwh"],
        "europe_cases": int(europe.country_years.sum()),
        "china_cases": int(len(china)),
        "japan_case_years": int(len(japan)),
        "japan_unique_communities": int(japan.mesh_id.nunique()),
    }
    (OUT / "v48_three_scale_source_qa.json").write_text(json.dumps(qa, indent=2) + "\n")


if __name__ == "__main__":
    main()
