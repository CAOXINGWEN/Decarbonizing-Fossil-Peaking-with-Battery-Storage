"""
M4 true-dispatch LFP + CAES hybrid sweep for seasonal-MC Fig 5 panels.

For every province/panel/dispatch-window in the seasonal-MC no-LFP4h OSR
figure, this script:

  1. reads that panel's OSR-selected LFP and CAES durations;
  2. builds sparse capacity grids from the existing single-tech capacity sweeps
     at satisfy-rate anchors;
  3. runs the existing daily dispatch-window hybrid simulator on the original
     M3 demand/curtailment/charge-window inputs;
  4. writes real hybrid dispatch results for two dispatch policies:
       - lfp_first:  LFP discharges/charges first, CAES serves residual;
       - caes_first: CAES discharges/charges first, LFP serves residual.

The model is the project's existing day-step dispatch with hourly charge and
discharge windows, not a curve-level algebraic proxy.

Output:
  output/m4_economics/m4_hybrid_lfp_caes_seasonal_mc_no_lfp4h.csv
"""

from __future__ import annotations

from pathlib import Path
import argparse
import time
import warnings

import numpy as np
import pandas as pd

import m4_hybrid_search as hybrid
from hybrid_simulator import simulate_hybrid_steady
from hybrid_simulator_planA import simulate_hybrid_steady_planA
from event_clustering import EN_TO_CN
from ldes_params import load_tech_library
from ldes_economics import load_gas_prices, load_electricity_prices
import m4_visualize_freqlen_seasonal_mc_per_province_osr_no_lfp4h as seasonal


warnings.filterwarnings("ignore")

seasonal_plotter = seasonal.plotter

PAPER_CODE_DIR = Path(__file__).resolve().parent
M3_DIR = PAPER_CODE_DIR / "output" / "m3_dispatch"
M4_DIR = PAPER_CODE_DIR / "output" / "m4_economics"
SOURCE_DIR = (
    M4_DIR
    / "nature_figure"
    / "freqlen_per_province_osr_seasonal_mc_no_lfp4h"
    / "source_data"
)
CHAMPIONS_CSV = SOURCE_DIR / seasonal_plotter.CHAMPION_FILENAME
OUT_CSV = M4_DIR / "m4_hybrid_lfp_caes_seasonal_mc_no_lfp4h.csv"

SCENARIO = "baseline"
CAPACITY_ANCHOR_PCTS = sorted({*range(0, 101, 5), 99})
POLICY_SIMULATORS = {
    "lfp_first": simulate_hybrid_steady,
    "caes_first": simulate_hybrid_steady_planA,
}
POLICY_LABELS = {
    "lfp_first": "LFP-first dispatch",
    "caes_first": "CAES-first dispatch",
}

NEEDED_SINGLE_COLS = [
    "scenario",
    "dispatch_window_h",
    "duration_h",
    "province_en",
    "capacity_mwh",
    "satisfy_rate_pct",
]


def load_prices():
    gas = load_gas_prices()
    elec = load_electricity_prices()
    return (
        gas.set_index("province_cn")["gas_price_cny_m3"].to_dict(),
        elec.set_index("province_cn")["electricity_price_cny_kwh"].to_dict(),
    )


def load_inputs(stem: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    demand = pd.read_csv(M3_DIR / f"demand_{stem}.csv", index_col=0)
    curt = pd.read_csv(M3_DIR / f"curtailment_{stem}.csv", index_col=0)
    cw = pd.read_csv(M3_DIR / f"charge_window_{stem}.csv", index_col=0)
    return demand, curt, cw


def load_single_tech(stem: str, tech_id: str) -> pd.DataFrame:
    fp = M4_DIR / f"m4_single_tech_{stem}_{tech_id}.csv"
    if not fp.exists():
        raise FileNotFoundError(fp)
    df = pd.read_csv(fp, usecols=NEEDED_SINGLE_COLS, low_memory=False)
    df = df[df["scenario"].eq(SCENARIO)].copy()
    df["duration_h"] = df["duration_h"].astype(int)
    df["dispatch_window_h"] = df["dispatch_window_h"].astype(float)
    return df


def load_champion_map(province_filter: str | None = None) -> dict:
    if not CHAMPIONS_CSV.exists():
        raise FileNotFoundError(CHAMPIONS_CSV)
    champions = pd.read_csv(CHAMPIONS_CSV)
    champions = champions[champions["tech_id"].isin(["li_lfp", "caes"])].copy()
    if province_filter:
        needle = province_filter.lower()
        mask = (
            champions["province_en"].astype(str).str.lower().str.contains(needle)
            | champions["province_cn"].astype(str).str.lower().str.contains(needle)
        )
        champions = champions[mask].copy()

    out = {}
    for (stem, dw, prov_en), cell in champions.groupby(
        ["stem", "dispatch_window_h", "province_en"], sort=False
    ):
        techs = {r.tech_id: r for r in cell.itertuples(index=False)}
        if "li_lfp" not in techs or "caes" not in techs:
            continue
        out[(str(stem), float(dw), str(prov_en))] = {
            "province_cn": str(techs["li_lfp"].province_cn),
            "lfp_duration_h": int(techs["li_lfp"].duration_h),
            "caes_duration_h": int(techs["caes"].duration_h),
            "lfp_single_osr_pct": float(techs["li_lfp"].osr_pct),
            "caes_single_osr_pct": float(techs["caes"].osr_pct),
        }
    return out


def capacity_anchors(
    single_df: pd.DataFrame,
    province_en: str,
    dw_h: float,
    duration_h: int,
) -> list[float]:
    sub = single_df[
        single_df["province_en"].eq(province_en)
        & single_df["dispatch_window_h"].eq(float(dw_h))
        & single_df["duration_h"].eq(int(duration_h))
    ].copy()
    if sub.empty:
        return [0.0]
    sub = sub.sort_values(["capacity_mwh", "satisfy_rate_pct"]).dropna(
        subset=["capacity_mwh", "satisfy_rate_pct"]
    )
    caps = [0.0]
    max_cap = float(sub["capacity_mwh"].max())
    sat = sub["satisfy_rate_pct"].to_numpy(dtype=float)
    cap = sub["capacity_mwh"].to_numpy(dtype=float)
    for target in CAPACITY_ANCHOR_PCTS:
        if target <= 0:
            caps.append(0.0)
            continue
        ok = sat >= float(target)
        caps.append(float(cap[ok].min()) if ok.any() else max_cap)
    caps.append(max_cap)
    return sorted({round(x, 6) for x in caps if np.isfinite(x)})


def evaluate_cell(
    stem: str,
    row_idx: int,
    row_label: str,
    dw_h: float,
    province_en: str,
    champion: dict,
    demand_df: pd.DataFrame,
    curt_df: pd.DataFrame,
    cw_df: pd.DataFrame,
    lfp_single: pd.DataFrame,
    caes_single: pd.DataFrame,
    gas_map: dict,
    elec_map: dict,
    tech_lib: pd.DataFrame,
    policies: list[str],
) -> list[dict]:
    province_cn = champion["province_cn"]
    if (
        province_en not in demand_df.columns
        or province_en not in curt_df.columns
        or province_en not in cw_df.columns
        or province_cn not in gas_map
        or province_cn not in elec_map
    ):
        return []

    lfp_dur = int(champion["lfp_duration_h"])
    caes_dur = int(champion["caes_duration_h"])
    lfp_tp = hybrid.get_tech_params(tech_lib, "li_lfp", SCENARIO, lfp_dur)
    caes_tp = hybrid.get_tech_params(tech_lib, "caes", SCENARIO, caes_dur)
    if lfp_tp is None or caes_tp is None:
        return []

    lfp_caps = capacity_anchors(lfp_single, province_en, dw_h, lfp_dur)
    caes_caps = capacity_anchors(caes_single, province_en, dw_h, caes_dur)
    if len(lfp_caps) <= 1 and len(caes_caps) <= 1:
        return []

    d_arr = demand_df[province_en].values.astype(np.float64)
    c_arr = curt_df[province_en].values.astype(np.float64)
    w_arr = cw_df[province_en].values.astype(np.float64)
    dw_arr = np.full(d_arr.shape[0], float(dw_h), dtype=np.float64)
    gas_p = float(gas_map[province_cn])
    elec_p = float(elec_map[province_cn])

    rows = []
    for policy in policies:
        hybrid.simulate_hybrid_steady = POLICY_SIMULATORS[policy]
        for lfp_cap in lfp_caps:
            for caes_cap in caes_caps:
                if lfp_cap == 0.0 and caes_cap == 0.0:
                    continue
                r = hybrid.evaluate_combo(
                    d_arr,
                    c_arr,
                    w_arr,
                    dw_arr,
                    lfp_tp,
                    float(lfp_cap),
                    caes_tp,
                    float(caes_cap),
                    gas_p,
                    elec_p,
                    float(dw_h),
                )
                r.update(
                    {
                        "scenario": SCENARIO,
                        "experiment": "seasonal_mc",
                        "stem": stem,
                        "row_idx": int(row_idx),
                        "row_label": row_label,
                        "dispatch_window_h": float(dw_h),
                        "province_en": province_en,
                        "province_cn": province_cn,
                        "policy": policy,
                        "policy_label": POLICY_LABELS[policy],
                        "ldes_tech": "caes",
                        "lfp_duration_h": lfp_dur,
                        "ldes_duration_h": caes_dur,
                        "caes_duration_h": caes_dur,
                        "lfp_capacity_mwh": float(lfp_cap),
                        "ldes_capacity_mwh": float(caes_cap),
                        "caes_capacity_mwh": float(caes_cap),
                        "lfp_single_osr_pct": champion["lfp_single_osr_pct"],
                        "caes_single_osr_pct": champion["caes_single_osr_pct"],
                        "lfp_grid_n": len(lfp_caps),
                        "caes_grid_n": len(caes_caps),
                    }
                )
                rows.append(r)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--province", default=None, help="Optional province filter.")
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["lfp_first", "caes_first"],
        choices=sorted(POLICY_SIMULATORS),
    )
    parser.add_argument("--output", default=str(OUT_CSV))
    args = parser.parse_args()

    out_csv = Path(args.output)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists():
        out_csv.unlink()

    champion_map = load_champion_map(args.province)
    if not champion_map:
        raise SystemExit("No LFP+CAES champion duration cells matched.")

    gas_map, elec_map = load_prices()
    tech_lib = load_tech_library()

    written_header = False
    total_rows = 0
    t_start = time.time()

    for _, stem, row_idx, row_label in seasonal_plotter.SCAN_ROWS:
        t_stem = time.time()
        print(f"=== load inputs: {stem} ({row_label}) ===", flush=True)
        demand_df, curt_df, cw_df = load_inputs(stem)
        lfp_single = load_single_tech(stem, "li_lfp")
        caes_single = load_single_tech(stem, "caes")

        for dw_h in seasonal_plotter.DW_VALUES:
            t_dw = time.time()
            cell_rows = []
            provinces = sorted(
                {
                    prov
                    for (s, dw, prov) in champion_map
                    if s == stem and float(dw) == float(dw_h)
                }
            )
            for province_en in provinces:
                champion = champion_map[(stem, float(dw_h), province_en)]
                rows = evaluate_cell(
                    stem,
                    int(row_idx),
                    row_label,
                    float(dw_h),
                    province_en,
                    champion,
                    demand_df,
                    curt_df,
                    cw_df,
                    lfp_single,
                    caes_single,
                    gas_map,
                    elec_map,
                    tech_lib,
                    args.policies,
                )
                cell_rows.extend(rows)

            if cell_rows:
                chunk = pd.DataFrame(cell_rows)
                chunk.to_csv(
                    out_csv,
                    mode="a",
                    header=not written_header,
                    index=False,
                )
                written_header = True
                total_rows += len(chunk)
            print(
                f"  [{stem}|dw={dw_h:.0f}h] provinces={len(provinces)} "
                f"rows={len(cell_rows):,} in {time.time() - t_dw:.1f}s "
                f"(total {total_rows:,})",
                flush=True,
            )

        print(f"=== {stem} done in {(time.time() - t_stem) / 60:.1f} min ===", flush=True)

    print(
        f"=== hybrid LFP+CAES seasonal-MC sweep done: {total_rows:,} rows, "
        f"{(time.time() - t_start) / 60:.1f} min ===",
        flush=True,
    )
    print(f"output: {out_csv}", flush=True)


if __name__ == "__main__":
    main()
