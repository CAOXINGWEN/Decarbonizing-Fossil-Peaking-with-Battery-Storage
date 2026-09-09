"""
Fixed-discharge-share LFP 4h + CAES hybrid frontiers for learning-curve figures.

This is a conservative LDES-tail sensitivity:

* LFP duration is fixed at 4h, matching the first-paper BESS learning-curve
  anchor.
* CAES duration is inherited from the existing seasonal-MC CAES OSR champion
  for each province / panel / dispatch-window cell.
* Dispatch is true daily hybrid dispatch under hard or soft discharge quota.

Output defaults to:
  output/m4_economics_lfp4h_learning_curve/frontier/
    m4_hybrid_lfp4h_caes_discharge_ratio_{mode}_frontier_dw24h.csv
"""

from __future__ import annotations

from pathlib import Path
import argparse
import time
import warnings

import numpy as np
import pandas as pd

import m4_hybrid_lfp_caes_discharge_ratio_frontier as ratio_frontier
import m4_hybrid_lfp_caes_seasonal_mc_sweep as sparse
import m4_visualize_freqlen_seasonal_mc_per_province_osr_no_lfp4h as seasonal


warnings.filterwarnings("ignore")

PAPER_CODE_DIR = Path(__file__).resolve().parent
OUT_ROOT = PAPER_CODE_DIR / "output" / "m4_economics_lfp4h_learning_curve"
FRONTIER_DIR = OUT_ROOT / "frontier"

seasonal_plotter = seasonal.plotter


def output_csv_for_mode(mode: str, dw_values: list[float]) -> Path:
    dw_token = "_".join(f"{int(x)}h" if float(x).is_integer() else str(x) for x in dw_values)
    return FRONTIER_DIR / f"m4_hybrid_lfp4h_caes_discharge_ratio_{mode}_frontier_dw{dw_token}.csv"


def load_lfp4h_caes_champion_map(
    province_filter: str | None,
    dw_values: list[float],
) -> dict[tuple[str, float, str], dict]:
    """Use fixed LFP 4h and CAES OSR champion duration for each cell."""
    if not sparse.CHAMPIONS_CSV.exists():
        raise FileNotFoundError(sparse.CHAMPIONS_CSV)

    champions = pd.read_csv(sparse.CHAMPIONS_CSV)
    champions = champions[champions["tech_id"].astype(str).eq("caes")].copy()
    champions = champions[champions["dispatch_window_h"].astype(float).isin([float(x) for x in dw_values])].copy()
    if province_filter:
        needle = province_filter.lower()
        champions = champions[
            champions["province_en"].astype(str).str.lower().str.contains(needle)
            | champions["province_cn"].astype(str).str.lower().str.contains(needle)
        ].copy()

    out: dict[tuple[str, float, str], dict] = {}
    for row in champions.itertuples(index=False):
        out[(str(row.stem), float(row.dispatch_window_h), str(row.province_en))] = {
            "province_cn": str(row.province_cn),
            "lfp_duration_h": 4,
            "caes_duration_h": int(row.duration_h),
            "lfp_single_osr_pct": np.nan,
            "caes_single_osr_pct": float(row.osr_pct),
        }
    return out


def run_mode(args: argparse.Namespace, quota_mode: str) -> Path:
    dw_values = [float(x) for x in args.dw_values]
    out_csv = Path(args.output) if args.output else output_csv_for_mode(quota_mode, dw_values)
    if args.output and len(args.modes) > 1:
        raise SystemExit("--output can only be used with a single --modes value")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists():
        out_csv.unlink()

    champion_map = load_lfp4h_caes_champion_map(args.province, dw_values)
    if not champion_map:
        raise SystemExit("No LFP4h+CAES cells matched.")

    gas_map, elec_map = sparse.load_prices()
    tech_lib = sparse.load_tech_library()

    written_header = False
    total_rows = 0
    t_start = time.time()

    scan_rows = [
        (experiment, stem, int(row_idx), row_label)
        for experiment, stem, row_idx, row_label in seasonal_plotter.SCAN_ROWS
    ]
    for _, stem, row_idx, row_label in scan_rows:
        t_stem = time.time()
        print(f"=== [{quota_mode}] load inputs: {stem} ({row_label}) ===", flush=True)
        demand_df, curt_df, cw_df = ratio_frontier.load_inputs(stem)
        lfp_single = ratio_frontier.load_single_tech(stem, "li_lfp")
        caes_single = ratio_frontier.load_single_tech(stem, "caes")

        for dw_h in dw_values:
            t_dw = time.time()
            cell_rows: list[dict] = []
            provinces = sorted(
                {
                    prov
                    for (s, dw, prov) in champion_map
                    if s == stem and float(dw) == float(dw_h)
                }
            )
            for province_en in provinces:
                champion = champion_map[(stem, float(dw_h), province_en)]
                for lfp_share_pct in ratio_frontier.DISCHARGE_LFP_SHARES_PCT:
                    rows = ratio_frontier.evaluate_ratio_cell(
                        stem=stem,
                        row_idx=int(row_idx),
                        row_label=row_label,
                        dw_h=float(dw_h),
                        province_en=province_en,
                        champion=champion,
                        demand_df=demand_df,
                        curt_df=curt_df,
                        cw_df=cw_df,
                        lfp_single=lfp_single,
                        caes_single=caes_single,
                        gas_map=gas_map,
                        elec_map=elec_map,
                        tech_lib=tech_lib,
                        quota_mode=quota_mode,
                        lfp_discharge_share_pct=int(lfp_share_pct),
                        capacity_grid_n=int(args.capacity_grid_n),
                    )
                    cell_rows.extend(rows)

            if cell_rows:
                chunk = pd.DataFrame(cell_rows)
                chunk.to_csv(out_csv, mode="a", header=not written_header, index=False)
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
        f"=== [{quota_mode}] LFP4h+CAES frontier done: {total_rows:,} rows, "
        f"{(time.time() - t_start) / 60:.1f} min ===",
        flush=True,
    )
    print(f"output: {out_csv}", flush=True)
    return out_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--province", default=None)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["hard_quota", "soft_quota"],
        choices=sorted(ratio_frontier.QUOTA_MODE_CODES),
    )
    parser.add_argument("--dw-values", nargs="+", type=float, default=[24.0])
    parser.add_argument("--capacity-grid-n", type=int, default=21)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for quota_mode in args.modes:
        run_mode(args, quota_mode)


if __name__ == "__main__":
    main()
