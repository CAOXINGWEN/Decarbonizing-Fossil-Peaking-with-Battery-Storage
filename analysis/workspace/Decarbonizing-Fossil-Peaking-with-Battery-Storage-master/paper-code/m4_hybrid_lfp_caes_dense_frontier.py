"""
Dense full-range LFP + CAES hybrid frontier for seasonal-MC marginal OSR curves.

The earlier hybrid overlay used sparse capacity anchors. This script builds a
much denser capacity grid from the complete single-technology capacity scans:
for each province/panel/dispatch window, it maps a fine supply-ratio grid onto
the minimum capacity found in the 25 MWh single-tech scan. The resulting LFP
and CAES capacity grids cover the full searched capacity range from zero to the
single-tech maximum, rather than only OSR anchor points.

The output retains the lower convex envelope of strict hybrid supply-cost
points. Except for the zero anchor, a point must have positive LFP capacity,
positive CAES capacity, positive LFP supply, and positive CAES supply before it
can enter the hybrid frontier. This keeps the hybrid curves from collapsing to
pure single-technology frontiers.

Output:
  output/m4_economics/m4_hybrid_lfp_caes_dense_frontier_no_lfp4h.csv
"""

from __future__ import annotations

from pathlib import Path
import argparse
import time
import warnings

import numpy as np
import pandas as pd

import m4_hybrid_search as hybrid
import m4_hybrid_lfp_caes_seasonal_mc_sweep as sparse
from hybrid_simulator import simulate_hybrid_steady
from hybrid_simulator_planA import simulate_hybrid_steady_planA
import m4_visualize_freqlen_seasonal_mc_per_province_osr_no_lfp4h as seasonal


warnings.filterwarnings("ignore")

seasonal_plotter = seasonal.plotter

PAPER_CODE_DIR = Path(__file__).resolve().parent
M3_DIR = PAPER_CODE_DIR / "output" / "m3_dispatch"
M4_DIR = PAPER_CODE_DIR / "output" / "m4_economics"
OUT_CSV = M4_DIR / "m4_hybrid_lfp_caes_dense_frontier_no_lfp4h.csv"
FRONTIER_MODE = "dense_full_range_strict_combo_lower_convex_envelope"

SCENARIO = "baseline"
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


def dense_capacity_grid(
    single_df: pd.DataFrame,
    province_en: str,
    dw_h: float,
    duration_h: int,
    supply_step_pct: float,
) -> list[float]:
    """Map a fine supply-ratio grid to capacities from the 25 MWh scan."""
    sub = single_df[
        single_df["province_en"].eq(province_en)
        & single_df["dispatch_window_h"].eq(float(dw_h))
        & single_df["duration_h"].eq(int(duration_h))
    ].copy()
    if sub.empty:
        return [0.0]
    sub = sub.dropna(subset=["capacity_mwh", "satisfy_rate_pct"]).sort_values(
        ["capacity_mwh", "satisfy_rate_pct"]
    )
    if sub.empty:
        return [0.0]

    caps = [0.0]
    max_cap = float(sub["capacity_mwh"].max())
    sat = sub["satisfy_rate_pct"].to_numpy(dtype=float)
    cap = sub["capacity_mwh"].to_numpy(dtype=float)
    targets = np.arange(supply_step_pct, 100.0 + 1e-9, supply_step_pct)
    for target in targets:
        ok = sat >= target
        caps.append(float(cap[ok].min()) if ok.any() else max_cap)
    caps.append(max_cap)
    return sorted({round(x, 6) for x in caps if np.isfinite(x)})


def zero_frontier_row(
    stem: str,
    row_idx: int,
    row_label: str,
    dw_h: float,
    province_en: str,
    province_cn: str,
    policy: str,
    champion: dict,
    gas_unit_cost: float,
    total_demand_annual_kwh: float,
    lfp_grid_n: int,
    caes_grid_n: int,
    evaluated_pairs_n: int,
) -> dict:
    return {
        "target_supply_pct": 0.0,
        "target_annual_supply_kwh": 0.0,
        "satisfy_rate_pct": 0.0,
        "annual_supply_kwh": 0.0,
        "annual_supply_lfp_kwh": 0.0,
        "annual_supply_ldes_kwh": 0.0,
        "lfp_supply_share": np.nan,
        "annual_cost_cny": 0.0,
        "unit_cost_cny_kwh": np.nan,
        "gas_unit_cost_cny_kwh": gas_unit_cost,
        "cost_ratio_to_gas": np.nan,
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
        "lfp_duration_h": int(champion["lfp_duration_h"]),
        "ldes_duration_h": int(champion["caes_duration_h"]),
        "caes_duration_h": int(champion["caes_duration_h"]),
        "lfp_capacity_mwh": 0.0,
        "ldes_capacity_mwh": 0.0,
        "caes_capacity_mwh": 0.0,
        "lfp_single_osr_pct": float(champion["lfp_single_osr_pct"]),
        "caes_single_osr_pct": float(champion["caes_single_osr_pct"]),
        "lfp_grid_n": int(lfp_grid_n),
        "caes_grid_n": int(caes_grid_n),
        "evaluated_pairs_n": int(evaluated_pairs_n),
        "total_demand_annual_kwh": float(total_demand_annual_kwh),
        "frontier_mode": FRONTIER_MODE,
    }


def lower_convex_envelope_indices(x: np.ndarray, y: np.ndarray) -> list[int]:
    order = np.lexsort((y, x))
    points: list[tuple[float, float, int]] = []
    for idx in order:
        xi = float(x[idx])
        yi = float(y[idx])
        if not (np.isfinite(xi) and np.isfinite(yi)):
            continue
        if xi < 0 or yi < 0:
            continue
        if points and np.isclose(xi, points[-1][0], rtol=0.0, atol=1e-6):
            if yi < points[-1][1]:
                points[-1] = (xi, yi, int(idx))
            continue
        points.append((xi, yi, int(idx)))
        while len(points) >= 3:
            x1, y1, _ = points[-3]
            x2, y2, _ = points[-2]
            x3, y3, _ = points[-1]
            if x2 <= x1 or x3 <= x2:
                points.pop(-2)
                continue
            s12 = (y2 - y1) / (x2 - x1)
            s23 = (y3 - y2) / (x3 - x2)
            if s23 < s12 - 1e-12:
                points.pop(-2)
            else:
                break
    return [idx for _, _, idx in points]


def convex_envelope_frontier(
    rows: list[dict],
    zero_row: dict,
    total_demand_annual_kwh: float,
) -> list[dict]:
    if total_demand_annual_kwh <= 0:
        return [zero_row]

    if not rows:
        return [zero_row]

    candidates = pd.DataFrame(rows)
    candidates = candidates[
        np.isfinite(candidates["annual_supply_kwh"])
        & np.isfinite(candidates["annual_supply_lfp_kwh"])
        & np.isfinite(candidates["annual_supply_ldes_kwh"])
        & np.isfinite(candidates["annual_cost_cny"])
        & np.isfinite(candidates["lfp_capacity_mwh"])
        & np.isfinite(candidates["caes_capacity_mwh"])
        & (candidates["annual_supply_kwh"] > 0)
        & (candidates["annual_supply_lfp_kwh"] > 0)
        & (candidates["annual_supply_ldes_kwh"] > 0)
        & (candidates["annual_cost_cny"] >= 0)
        & (candidates["lfp_capacity_mwh"] > 0)
        & (candidates["caes_capacity_mwh"] > 0)
    ].copy()
    if candidates.empty:
        return [zero_row]

    df = pd.concat([pd.DataFrame([zero_row]), candidates], ignore_index=True)
    df = df[
        np.isfinite(df["annual_supply_kwh"])
        & np.isfinite(df["annual_cost_cny"])
        & (df["annual_supply_kwh"] >= 0)
        & (df["annual_cost_cny"] >= 0)
    ].copy()
    if df.empty:
        return [zero_row]

    df = df.sort_values(["annual_supply_kwh", "annual_cost_cny"]).drop_duplicates(
        "annual_supply_kwh", keep="first"
    )
    hull_pos = lower_convex_envelope_indices(
        df["annual_supply_kwh"].to_numpy(dtype=float),
        df["annual_cost_cny"].to_numpy(dtype=float),
    )

    frontier = []
    for pos in hull_pos:
        row = df.iloc[int(pos)].to_dict()
        annual_supply = float(row["annual_supply_kwh"])
        row["target_supply_pct"] = float(annual_supply / total_demand_annual_kwh * 100.0)
        row["target_annual_supply_kwh"] = annual_supply
        if annual_supply <= 0:
            row["satisfy_rate_pct"] = 0.0
            row["unit_cost_cny_kwh"] = np.nan
            row["cost_ratio_to_gas"] = np.nan
        else:
            row["satisfy_rate_pct"] = row["target_supply_pct"]
            row["unit_cost_cny_kwh"] = float(row["annual_cost_cny"]) / annual_supply
            gas_unit = float(row["gas_unit_cost_cny_kwh"])
            row["cost_ratio_to_gas"] = (
                row["unit_cost_cny_kwh"] / gas_unit if gas_unit > 0 else np.nan
            )
        row["frontier_mode"] = FRONTIER_MODE
        frontier.append(row)
    return frontier


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
    capacity_supply_step_pct: float,
    target_step_pct: float,
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

    lfp_caps = dense_capacity_grid(
        lfp_single, province_en, dw_h, lfp_dur, capacity_supply_step_pct
    )
    caes_caps = dense_capacity_grid(
        caes_single, province_en, dw_h, caes_dur, capacity_supply_step_pct
    )
    if len(lfp_caps) <= 1 and len(caes_caps) <= 1:
        return []

    d_arr = demand_df[province_en].values.astype(np.float64)
    c_arr = curt_df[province_en].values.astype(np.float64)
    w_arr = cw_df[province_en].values.astype(np.float64)
    dw_arr = np.full(d_arr.shape[0], float(dw_h), dtype=np.float64)
    gas_p = float(gas_map[province_cn])
    elec_p = float(elec_map[province_cn])
    gas_unit_cost = gas_p / hybrid.GAS_POWER_PER_M3 + float(hybrid.CONFIG["gas"]["generator_cost"])
    total_demand_annual_kwh = float(np.sum(d_arr)) * 365.0 / len(d_arr)
    evaluated_pairs_n = (len(lfp_caps) * len(caes_caps)) - 1

    out_rows = []
    for policy in policies:
        hybrid.simulate_hybrid_steady = POLICY_SIMULATORS[policy]
        eval_rows = []
        for lfp_cap in lfp_caps:
            for caes_cap in caes_caps:
                if lfp_cap == 0.0 and caes_cap == 0.0:
                    continue
                result = hybrid.evaluate_combo(
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
                result.update(
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
                        "evaluated_pairs_n": evaluated_pairs_n,
                        "total_demand_annual_kwh": total_demand_annual_kwh,
                        "frontier_mode": FRONTIER_MODE,
                    }
                )
                eval_rows.append(result)

        zero_row = zero_frontier_row(
            stem,
            row_idx,
            row_label,
            dw_h,
            province_en,
            province_cn,
            policy,
            champion,
            gas_unit_cost,
            total_demand_annual_kwh,
            len(lfp_caps),
            len(caes_caps),
            evaluated_pairs_n,
        )
        out_rows.extend(convex_envelope_frontier(eval_rows, zero_row, total_demand_annual_kwh))
    return out_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--province", default=None)
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["lfp_first", "caes_first"],
        choices=sorted(POLICY_SIMULATORS),
    )
    parser.add_argument("--capacity-supply-step-pct", type=float, default=0.5)
    parser.add_argument(
        "--target-step-pct",
        type=float,
        default=0.5,
        help="Kept for CLI compatibility; convex-envelope output does not use target bins.",
    )
    parser.add_argument("--output", default=str(OUT_CSV))
    args = parser.parse_args()

    out_csv = Path(args.output)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists():
        out_csv.unlink()

    champion_map = sparse.load_champion_map(args.province)
    if not champion_map:
        raise SystemExit("No LFP+CAES champion duration cells matched.")

    gas_map, elec_map = sparse.load_prices()
    tech_lib = sparse.load_tech_library()

    written_header = False
    total_rows = 0
    total_eval_pairs = 0
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
                    args.capacity_supply_step_pct,
                    args.target_step_pct,
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
                total_eval_pairs += int(chunk["evaluated_pairs_n"].max()) * max(len(args.policies), 1)
            print(
                f"  [{stem}|dw={dw_h:.0f}h] provinces={len(provinces)} "
                f"frontier_rows={len(cell_rows):,} in {time.time() - t_dw:.1f}s "
                f"(total rows {total_rows:,})",
                flush=True,
            )

        print(f"=== {stem} done in {(time.time() - t_stem) / 60:.1f} min ===", flush=True)

    print(
        f"=== dense hybrid LFP+CAES frontier done: {total_rows:,} rows, "
        f"{(time.time() - t_start) / 60:.1f} min ===",
        flush=True,
    )
    print(f"output: {out_csv}", flush=True)
    print(f"approx evaluated grid pairs: {total_eval_pairs:,}", flush=True)


if __name__ == "__main__":
    main()
