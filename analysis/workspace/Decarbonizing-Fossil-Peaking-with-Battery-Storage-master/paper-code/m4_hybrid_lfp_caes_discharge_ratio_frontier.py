"""
Fixed-discharge-share LFP + CAES hybrid frontiers for seasonal-MC marginal OSR figures.

This is different from fixed capacity shares. For every target discharge share
LFP:CAES = 90:10, ..., 10:90, both LFP and CAES capacities are scanned on a
two-dimensional grid. The dispatch then enforces either:

  - hard_quota: each resource can only serve its target daily demand share;
  - soft_quota: target shares are served first, then unmet demand can be
    supplied by residual deliverable energy from either resource.

Outputs:
  output/m4_economics/m4_hybrid_lfp_caes_discharge_ratio_hard_quota_frontier_no_lfp4h.csv
  output/m4_economics/m4_hybrid_lfp_caes_discharge_ratio_soft_quota_frontier_no_lfp4h.csv
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
import m4_visualize_freqlen_seasonal_mc_per_province_osr_no_lfp4h as seasonal
from hybrid_simulator_discharge_ratio import (
    HARD_QUOTA,
    SOFT_QUOTA,
    simulate_hybrid_steady_discharge_ratio,
)


warnings.filterwarnings("ignore")

seasonal_plotter = seasonal.plotter

PAPER_CODE_DIR = Path(__file__).resolve().parent
M3_DIR = PAPER_CODE_DIR / "output" / "m3_dispatch"
M4_DIR = PAPER_CODE_DIR / "output" / "m4_economics"

SCENARIO = "baseline"
DISCHARGE_LFP_SHARES_PCT = list(range(90, 0, -10))
QUOTA_MODE_CODES = {
    "hard_quota": HARD_QUOTA,
    "soft_quota": SOFT_QUOTA,
}
QUOTA_MODE_LABELS = {
    "hard_quota": "Hard discharge quota",
    "soft_quota": "Soft discharge quota with fallback",
}

NEEDED_SINGLE_COLS = [
    "scenario",
    "dispatch_window_h",
    "duration_h",
    "province_en",
    "capacity_mwh",
    "satisfy_rate_pct",
]


def output_csv_for_mode(mode: str) -> Path:
    return M4_DIR / f"m4_hybrid_lfp_caes_discharge_ratio_{mode}_frontier_no_lfp4h.csv"


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


def max_single_capacity(
    single_df: pd.DataFrame,
    province_en: str,
    dw_h: float,
    duration_h: int,
) -> float:
    sub = single_df[
        single_df["province_en"].eq(province_en)
        & single_df["dispatch_window_h"].eq(float(dw_h))
        & single_df["duration_h"].eq(int(duration_h))
    ]
    if sub.empty:
        return 0.0
    cap = sub["capacity_mwh"].replace([np.inf, -np.inf], np.nan).dropna()
    return float(cap.max()) if not cap.empty else 0.0


def two_dim_capacity_grid(
    lfp_single: pd.DataFrame,
    caes_single: pd.DataFrame,
    province_en: str,
    dw_h: float,
    lfp_duration_h: int,
    caes_duration_h: int,
    grid_n: int,
) -> pd.DataFrame:
    """Return positive LFP x CAES capacity pairs over each single-tech scan range."""
    lfp_max = max_single_capacity(lfp_single, province_en, dw_h, lfp_duration_h)
    caes_max = max_single_capacity(caes_single, province_en, dw_h, caes_duration_h)
    if lfp_max <= 0 or caes_max <= 0:
        return pd.DataFrame(columns=["lfp_capacity_mwh", "caes_capacity_mwh"])

    lfp_caps = np.linspace(0.0, lfp_max, int(grid_n))[1:]
    caes_caps = np.linspace(0.0, caes_max, int(grid_n))[1:]
    pairs = []
    for lfp_cap in lfp_caps:
        for caes_cap in caes_caps:
            pairs.append((lfp_cap, caes_cap))
    if not pairs:
        return pd.DataFrame(columns=["lfp_capacity_mwh", "caes_capacity_mwh"])
    out = pd.DataFrame(pairs, columns=["lfp_capacity_mwh", "caes_capacity_mwh"])
    return out.round(6).drop_duplicates().reset_index(drop=True)


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


def lower_convex_envelope_frame(df: pd.DataFrame) -> pd.DataFrame:
    clean = df[
        np.isfinite(df["annual_supply_kwh"])
        & np.isfinite(df["annual_cost_cny"])
        & (df["annual_supply_kwh"] >= 0)
        & (df["annual_cost_cny"] >= 0)
    ].copy()
    if clean.empty:
        return clean
    clean = clean.sort_values(["annual_supply_kwh", "annual_cost_cny"]).drop_duplicates(
        "annual_supply_kwh", keep="first"
    ).reset_index(drop=True)
    idx = lower_convex_envelope_indices(
        clean["annual_supply_kwh"].to_numpy(dtype=float),
        clean["annual_cost_cny"].to_numpy(dtype=float),
    )
    if not idx:
        return clean.iloc[0:0].copy()
    return clean.iloc[idx].sort_values("annual_supply_kwh").reset_index(drop=True)


def evaluate_combo_discharge_ratio(
    demand_arr: np.ndarray,
    curt_arr: np.ndarray,
    cw_arr: np.ndarray,
    dw_arr: np.ndarray,
    lfp_tp,
    lfp_cap_mwh: float,
    caes_tp,
    caes_cap_mwh: float,
    gas_price: float,
    elec_price: float,
    lfp_discharge_share: float,
    quota_mode_code: int,
) -> dict:
    soc_min = float(hybrid.CONFIG["storage"]["soc_min"])
    soc_max = float(hybrid.CONFIG["storage"]["soc_max"])
    discount = float(hybrid.CONFIG["cost"]["discount_rate"])
    gen_cost = float(hybrid.CONFIG["gas"]["generator_cost"])

    lfp_cap_kwh = lfp_cap_mwh * 1000.0
    caes_cap_kwh = caes_cap_mwh * 1000.0
    lfp_disch_pwr_kw = lfp_cap_kwh / lfp_tp.duration_h if lfp_cap_mwh > 0 else 0.0
    lfp_ch_pwr_kw = lfp_disch_pwr_kw * lfp_tp.charge_power_ratio
    caes_disch_pwr_kw = caes_cap_kwh / caes_tp.duration_h if caes_cap_mwh > 0 else 0.0
    caes_ch_pwr_kw = caes_disch_pwr_kw * caes_tp.charge_power_ratio

    eps = 1e-6
    res = simulate_hybrid_steady_discharge_ratio(
        demand_arr,
        curt_arr,
        cw_arr,
        dw_arr,
        lfp_cap_kwh if lfp_cap_kwh > 0 else eps,
        lfp_ch_pwr_kw,
        lfp_disch_pwr_kw,
        lfp_tp.charge_eff,
        lfp_tp.discharge_eff,
        lfp_tp.self_discharge_rate_day,
        caes_cap_kwh if caes_cap_kwh > 0 else eps,
        caes_ch_pwr_kw,
        caes_disch_pwr_kw,
        caes_tp.charge_eff,
        caes_tp.discharge_eff,
        caes_tp.self_discharge_rate_day,
        soc_min,
        soc_max,
        float(lfp_discharge_share),
        int(quota_mode_code),
    )
    sup_lfp, sup_caes, ch_lfp, ch_caes, spill, *_ = res

    n_days = demand_arr.shape[0]
    year_factor = 365.0 / n_days
    annual_supply_lfp = sup_lfp * year_factor
    annual_supply_caes = sup_caes * year_factor
    annual_supply_total = annual_supply_lfp + annual_supply_caes
    total_demand_annual = float(np.sum(demand_arr)) * year_factor
    satisfy_pct = (
        annual_supply_total / total_demand_annual * 100.0
        if total_demand_annual > 0
        else 0.0
    )

    capex_lfp, ann_lfp_fixed = hybrid.annualized_cost(lfp_tp, lfp_cap_mwh, discount)
    capex_caes, ann_caes_fixed = hybrid.annualized_cost(caes_tp, caes_cap_mwh, discount)
    annual_var_om = (
        lfp_tp.variable_om_cny_mwh * annual_supply_lfp / 1000.0
        + caes_tp.variable_om_cny_mwh * annual_supply_caes / 1000.0
    )
    annual_cost = ann_lfp_fixed + ann_caes_fixed + annual_var_om
    unit_cost = annual_cost / annual_supply_total if annual_supply_total > 0 else np.nan
    annual_revenue = annual_supply_total * elec_price
    gas_unit_cost = gas_price / hybrid.GAS_POWER_PER_M3 + gen_cost

    return {
        "satisfy_rate_pct": satisfy_pct,
        "annual_supply_kwh": annual_supply_total,
        "annual_supply_lfp_kwh": annual_supply_lfp,
        "annual_supply_ldes_kwh": annual_supply_caes,
        "lfp_supply_share": (
            annual_supply_lfp / annual_supply_total
            if annual_supply_total > 0
            else np.nan
        ),
        "annual_charge_kwh": (ch_lfp + ch_caes) * year_factor,
        "annual_spillage_kwh": spill * year_factor,
        "capex_lfp_cny": capex_lfp,
        "capex_ldes_cny": capex_caes,
        "capex_total_cny": capex_lfp + capex_caes,
        "annual_cost_cny": annual_cost,
        "annual_revenue_cny": annual_revenue,
        "annual_net_profit_cny": annual_revenue - annual_cost,
        "unit_cost_cny_kwh": unit_cost,
        "gas_unit_cost_cny_kwh": gas_unit_cost,
        "cost_ratio_to_gas": (
            unit_cost / gas_unit_cost if pd.notna(unit_cost) else np.nan
        ),
    }


def zero_row(
    stem: str,
    row_idx: int,
    row_label: str,
    dw_h: float,
    province_en: str,
    province_cn: str,
    quota_mode: str,
    champion: dict,
    lfp_discharge_share_pct: int,
    gas_unit_cost: float,
    total_demand_annual_kwh: float,
    grid_n: int,
    evaluated_pairs_n: int,
) -> dict:
    caes_share_pct = 100 - int(lfp_discharge_share_pct)
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
        "quota_mode": quota_mode,
        "quota_mode_label": QUOTA_MODE_LABELS[quota_mode],
        "ldes_tech": "caes",
        "lfp_duration_h": int(champion["lfp_duration_h"]),
        "ldes_duration_h": int(champion["caes_duration_h"]),
        "caes_duration_h": int(champion["caes_duration_h"]),
        "lfp_capacity_mwh": 0.0,
        "ldes_capacity_mwh": 0.0,
        "caes_capacity_mwh": 0.0,
        "lfp_discharge_share_pct": int(lfp_discharge_share_pct),
        "caes_discharge_share_pct": int(caes_share_pct),
        "target_discharge_ratio_label": f"LFP{int(lfp_discharge_share_pct)}-CAES{int(caes_share_pct)}",
        "lfp_single_osr_pct": float(champion["lfp_single_osr_pct"]),
        "caes_single_osr_pct": float(champion["caes_single_osr_pct"]),
        "capacity_grid_n": int(grid_n),
        "evaluated_pairs_n": int(evaluated_pairs_n),
        "total_demand_annual_kwh": float(total_demand_annual_kwh),
        "frontier_mode": f"{quota_mode}_fixed_discharge_ratio_2d_capacity_lower_convex_envelope",
    }


def evaluate_ratio_cell(
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
    quota_mode: str,
    lfp_discharge_share_pct: int,
    capacity_grid_n: int,
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

    cap_pairs = two_dim_capacity_grid(
        lfp_single,
        caes_single,
        province_en,
        dw_h,
        lfp_dur,
        caes_dur,
        capacity_grid_n,
    )
    if cap_pairs.empty:
        return []

    d_arr = demand_df[province_en].values.astype(np.float64)
    c_arr = curt_df[province_en].values.astype(np.float64)
    w_arr = cw_df[province_en].values.astype(np.float64)
    dw_arr = np.full(d_arr.shape[0], float(dw_h), dtype=np.float64)
    gas_p = float(gas_map[province_cn])
    elec_p = float(elec_map[province_cn])
    gen_cost = float(hybrid.CONFIG["gas"]["generator_cost"])
    gas_unit_cost = gas_p / hybrid.GAS_POWER_PER_M3 + gen_cost
    total_demand_annual_kwh = float(np.sum(d_arr)) * 365.0 / len(d_arr)
    quota_code = QUOTA_MODE_CODES[quota_mode]
    lfp_share = float(lfp_discharge_share_pct) / 100.0
    caes_share_pct = 100 - int(lfp_discharge_share_pct)

    rows = []
    for pair in cap_pairs.itertuples(index=False):
        result = evaluate_combo_discharge_ratio(
            d_arr,
            c_arr,
            w_arr,
            dw_arr,
            lfp_tp,
            float(pair.lfp_capacity_mwh),
            caes_tp,
            float(pair.caes_capacity_mwh),
            gas_p,
            elec_p,
            lfp_share,
            quota_code,
        )
        lfp_cap = float(pair.lfp_capacity_mwh)
        caes_cap = float(pair.caes_capacity_mwh)
        result.update(
            {
                "target_supply_pct": result["satisfy_rate_pct"],
                "target_annual_supply_kwh": result["annual_supply_kwh"],
                "scenario": SCENARIO,
                "experiment": "seasonal_mc",
                "stem": stem,
                "row_idx": int(row_idx),
                "row_label": row_label,
                "dispatch_window_h": float(dw_h),
                "province_en": province_en,
                "province_cn": province_cn,
                "quota_mode": quota_mode,
                "quota_mode_label": QUOTA_MODE_LABELS[quota_mode],
                "ldes_tech": "caes",
                "lfp_duration_h": lfp_dur,
                "ldes_duration_h": caes_dur,
                "caes_duration_h": caes_dur,
                "lfp_capacity_mwh": lfp_cap,
                "ldes_capacity_mwh": caes_cap,
                "caes_capacity_mwh": caes_cap,
                "lfp_discharge_share_pct": int(lfp_discharge_share_pct),
                "caes_discharge_share_pct": int(caes_share_pct),
                "target_discharge_ratio_label": f"LFP{int(lfp_discharge_share_pct)}-CAES{int(caes_share_pct)}",
                "lfp_single_osr_pct": champion["lfp_single_osr_pct"],
                "caes_single_osr_pct": champion["caes_single_osr_pct"],
                "capacity_grid_n": int(capacity_grid_n),
                "evaluated_pairs_n": int(len(cap_pairs)),
                "total_demand_annual_kwh": total_demand_annual_kwh,
                "frontier_mode": f"{quota_mode}_fixed_discharge_ratio_2d_capacity_lower_convex_envelope",
            }
        )
        rows.append(result)

    zero = zero_row(
        stem,
        row_idx,
        row_label,
        dw_h,
        province_en,
        province_cn,
        quota_mode,
        champion,
        lfp_discharge_share_pct,
        gas_unit_cost,
        total_demand_annual_kwh,
        capacity_grid_n,
        len(cap_pairs),
    )
    front = lower_convex_envelope_frame(
        pd.concat([pd.DataFrame([zero]), pd.DataFrame(rows)], ignore_index=True)
    )
    if front.empty:
        return []
    front["target_supply_pct"] = (
        front["annual_supply_kwh"].astype(float) / total_demand_annual_kwh * 100.0
    )
    front["target_annual_supply_kwh"] = front["annual_supply_kwh"]
    front["satisfy_rate_pct"] = front["target_supply_pct"]
    front["unit_cost_cny_kwh"] = np.where(
        front["annual_supply_kwh"].astype(float) > 0,
        front["annual_cost_cny"].astype(float) / front["annual_supply_kwh"].astype(float),
        np.nan,
    )
    front["cost_ratio_to_gas"] = (
        front["unit_cost_cny_kwh"] / front["gas_unit_cost_cny_kwh"]
    )
    return front.to_dict("records")


def run_mode(args, quota_mode: str) -> Path:
    out_csv = Path(args.output) if args.output else output_csv_for_mode(quota_mode)
    if args.output and len(args.modes) > 1:
        raise SystemExit("--output can only be used with a single --modes value")
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
    t_start = time.time()

    for _, stem, row_idx, row_label in seasonal_plotter.SCAN_ROWS:
        t_stem = time.time()
        print(f"=== [{quota_mode}] load inputs: {stem} ({row_label}) ===", flush=True)
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
                for lfp_share_pct in DISCHARGE_LFP_SHARES_PCT:
                    rows = evaluate_ratio_cell(
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
                        quota_mode,
                        int(lfp_share_pct),
                        int(args.capacity_grid_n),
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
                f"  [{quota_mode}|{stem}|dw={dw_h:.0f}h] provinces={len(provinces)} "
                f"frontier_rows={len(cell_rows):,} in {time.time() - t_dw:.1f}s "
                f"(total rows {total_rows:,})",
                flush=True,
            )

        print(
            f"=== [{quota_mode}] {stem} done in {(time.time() - t_stem) / 60:.1f} min ===",
            flush=True,
        )

    print(
        f"=== [{quota_mode}] discharge-ratio frontier done: {total_rows:,} rows, "
        f"{(time.time() - t_start) / 60:.1f} min ===",
        flush=True,
    )
    print(f"output: {out_csv}", flush=True)
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--province", default=None)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["hard_quota", "soft_quota"],
        choices=sorted(QUOTA_MODE_CODES),
    )
    parser.add_argument("--capacity-grid-n", type=int, default=21)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    for quota_mode in args.modes:
        run_mode(args, quota_mode)


if __name__ == "__main__":
    main()
