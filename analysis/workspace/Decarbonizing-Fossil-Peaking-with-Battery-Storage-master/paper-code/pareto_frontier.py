"""
M5 Pareto frontier module for the second paper.

The module converts M4 economic curves into fixed substitution-depth slices,
then identifies province-specific efficiency-cost Pareto frontiers. Efficiency
is maximized and unit peaking cost is minimized.

Default outputs:
  - province_depth_points_<scenario>.csv
  - province_depth_frontier_<scenario>.csv
  - province_frontier_metrics_<scenario>.csv
  - province_frontier_hypervolume_<scenario>.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PAPER_CODE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PAPER_CODE_DIR / "output" / "m5_pareto"
M4_OUTPUT_DIR = PAPER_CODE_DIR / "output" / "m4_economics"

DEFAULT_DEPTHS = [75.0, 85.0, 90.0, 95.0, 99.0]
DEFAULT_COST_SCENARIOS = ["baseline", "low_cost", "high_cost"]

CURVE_USECOLS = {
    "tech_id",
    "technology_cn",
    "technology_en",
    "family",
    "cost_scenario",
    "dispatch_scenario",
    "duration_h",
    "rte",
    "dispatch_rte",
    "province_en",
    "province_cn",
    "power_mw",
    "capacity_mwh",
    "total_demand_kwh",
    "total_supply_kwh",
    "satisfy_rate_pct",
    "efc",
    "gas_unit_cost_cny_kwh",
    "unit_cost_cny_kwh",
    "annual_cost_cny",
    "annual_revenue_cny",
    "annual_net_profit_cny",
    "unit_profit_cny_per_kwh_capacity",
    "irr_pct",
}

INTERP_COLS = [
    "power_mw",
    "capacity_mwh",
    "total_supply_kwh",
    "efc",
    "unit_cost_cny_kwh",
    "annual_cost_cny",
    "annual_revenue_cny",
    "annual_net_profit_cny",
    "unit_profit_cny_per_kwh_capacity",
    "irr_pct",
]

GROUP_COLS = [
    "cost_scenario",
    "province_en",
    "province_cn",
    "tech_id",
    "technology_cn",
    "duration_h",
]


def _first_present(row: pd.Series, names: Iterable[str], default=np.nan):
    for name in names:
        if name in row.index:
            value = row[name]
            if pd.notna(value):
                return value
    return default


def default_curve_path(cost_scenario: str) -> Path:
    return M4_OUTPUT_DIR / f"m4_curve_points_{cost_scenario}.csv"


def read_curve_points(path: Path, cost_scenario: str) -> pd.DataFrame:
    """Read only the M4 point columns needed by M5."""
    if not path.exists():
        raise FileNotFoundError(f"M4 curve point table not found: {path}")

    df = pd.read_csv(path, usecols=lambda col: col in CURVE_USECOLS)
    if "cost_scenario" not in df.columns:
        df["cost_scenario"] = cost_scenario
    if "dispatch_rte" not in df.columns:
        df["dispatch_rte"] = df["rte"]
    if "technology_en" not in df.columns:
        df["technology_en"] = ""
    if "family" not in df.columns:
        df["family"] = ""

    numeric_cols = [
        "duration_h",
        "rte",
        "dispatch_rte",
        "power_mw",
        "capacity_mwh",
        "total_demand_kwh",
        "total_supply_kwh",
        "satisfy_rate_pct",
        "efc",
        "gas_unit_cost_cny_kwh",
        "unit_cost_cny_kwh",
        "annual_cost_cny",
        "annual_revenue_cny",
        "annual_net_profit_cny",
        "unit_profit_cny_per_kwh_capacity",
        "irr_pct",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required = [
        "cost_scenario",
        "province_en",
        "province_cn",
        "tech_id",
        "technology_cn",
        "duration_h",
        "dispatch_rte",
        "satisfy_rate_pct",
        "unit_cost_cny_kwh",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"M4 curve point table is missing required columns: {missing}")

    df = df.dropna(subset=["satisfy_rate_pct"])
    return df


def _prepare_curve(group: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in INTERP_COLS if col in group.columns]
    keep_cols = ["satisfy_rate_pct"] + cols
    out = group[keep_cols].copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["satisfy_rate_pct", "unit_cost_cny_kwh"])
    out = out.sort_values(["satisfy_rate_pct", "capacity_mwh"], kind="mergesort")

    # Duplicate satisfy rates usually occur near saturation. The lowest-cost row
    # is the relevant candidate for an efficiency-cost frontier.
    if not out.empty:
        out = (
            out.sort_values(["satisfy_rate_pct", "unit_cost_cny_kwh", "capacity_mwh"])
            .drop_duplicates(subset=["satisfy_rate_pct"], keep="first")
            .sort_values("satisfy_rate_pct")
            .reset_index(drop=True)
        )
    return out


def build_depth_points(curve_points: pd.DataFrame, depths: list[float]) -> pd.DataFrame:
    """Interpolate each province-tech-duration curve at fixed depth slices."""
    rows: list[dict] = []

    for keys, group in curve_points.groupby(GROUP_COLS, sort=False):
        base = dict(zip(GROUP_COLS, keys))
        first = group.iloc[0]
        base.update(
            {
                "technology_en": _first_present(first, ["technology_en"], ""),
                "family": _first_present(first, ["family"], ""),
                "dispatch_scenario": _first_present(first, ["dispatch_scenario"], ""),
                "rte": float(_first_present(first, ["rte", "dispatch_rte"])),
                "dispatch_rte": float(_first_present(first, ["dispatch_rte", "rte"])),
                "total_demand_kwh": float(_first_present(first, ["total_demand_kwh"])),
                "gas_unit_cost_cny_kwh": float(
                    _first_present(first, ["gas_unit_cost_cny_kwh"])
                ),
            }
        )

        curve = _prepare_curve(group)
        max_satisfy = float(curve["satisfy_rate_pct"].max()) if not curve.empty else np.nan
        min_satisfy = float(curve["satisfy_rate_pct"].min()) if not curve.empty else np.nan

        for depth in depths:
            row = base.copy()
            row["target_depth_pct"] = float(depth)
            row["min_satisfy_rate_pct"] = min_satisfy
            row["max_satisfy_rate_pct"] = max_satisfy
            row["is_reachable"] = bool(pd.notna(max_satisfy) and max_satisfy >= depth)
            row["interpolated"] = False

            if row["is_reachable"]:
                x = curve["satisfy_rate_pct"].to_numpy(dtype=float)
                row["interpolated"] = bool(not np.any(np.isclose(x, depth)))
                row["satisfy_rate_pct"] = float(depth)
                for col in INTERP_COLS:
                    if col in curve.columns:
                        y = curve[col].to_numpy(dtype=float)
                        finite = np.isfinite(x) & np.isfinite(y)
                        if finite.sum() == 0:
                            row[col] = np.nan
                        elif finite.sum() == 1:
                            row[col] = float(y[finite][0])
                        else:
                            row[col] = float(np.interp(depth, x[finite], y[finite]))
                if pd.notna(row.get("unit_cost_cny_kwh")) and pd.notna(
                    row.get("gas_unit_cost_cny_kwh")
                ):
                    row["unit_cost_to_gas_ratio"] = (
                        row["unit_cost_cny_kwh"] / row["gas_unit_cost_cny_kwh"]
                    )
                else:
                    row["unit_cost_to_gas_ratio"] = np.nan
            else:
                row["satisfy_rate_pct"] = np.nan
                for col in INTERP_COLS:
                    row[col] = np.nan
                row["unit_cost_to_gas_ratio"] = np.nan

            rows.append(row)

    return pd.DataFrame(rows)


def mark_pareto_frontier(depth_points: pd.DataFrame) -> pd.DataFrame:
    """Mark non-dominated points for each province-depth group."""
    out = depth_points.copy()
    out["is_pareto_frontier"] = False
    out["dominated_by"] = ""
    out["frontier_order"] = np.nan
    out["frontier_slope_to_next"] = np.nan

    group_cols = ["cost_scenario", "province_en", "province_cn", "target_depth_pct"]
    for _, idx in out.groupby(group_cols, sort=False).groups.items():
        group = out.loc[idx]
        candidates = group[
            group["is_reachable"]
            & np.isfinite(group["dispatch_rte"])
            & np.isfinite(group["unit_cost_cny_kwh"])
        ].copy()
        if candidates.empty:
            continue

        eff = candidates["dispatch_rte"].to_numpy(dtype=float)
        cost = candidates["unit_cost_cny_kwh"].to_numpy(dtype=float)
        labels = (
            candidates["tech_id"].astype(str)
            + "_"
            + candidates["duration_h"].astype(str)
            + "h"
        ).to_numpy()

        dominated = np.zeros(len(candidates), dtype=bool)
        dominator = np.full(len(candidates), "", dtype=object)
        for i in range(len(candidates)):
            for j in range(len(candidates)):
                if i == j:
                    continue
                weakly_better = eff[j] >= eff[i] and cost[j] <= cost[i]
                strictly_better = eff[j] > eff[i] or cost[j] < cost[i]
                if weakly_better and strictly_better:
                    dominated[i] = True
                    dominator[i] = labels[j]
                    break

        candidate_idx = candidates.index.to_numpy()
        frontier_idx = candidate_idx[~dominated]
        out.loc[candidate_idx, "dominated_by"] = dominator
        out.loc[frontier_idx, "is_pareto_frontier"] = True

        frontier = out.loc[frontier_idx].sort_values(
            ["dispatch_rte", "unit_cost_cny_kwh", "tech_id", "duration_h"],
            kind="mergesort",
        )
        if not frontier.empty:
            orders = np.arange(1, len(frontier) + 1, dtype=int)
            out.loc[frontier.index, "frontier_order"] = orders

            eff_sorted = frontier["dispatch_rte"].to_numpy(dtype=float)
            cost_sorted = frontier["unit_cost_cny_kwh"].to_numpy(dtype=float)
            slopes = np.full(len(frontier), np.nan)
            for k in range(len(frontier) - 1):
                d_eff = eff_sorted[k + 1] - eff_sorted[k]
                if abs(d_eff) > 1e-12:
                    slopes[k] = (cost_sorted[k + 1] - cost_sorted[k]) / d_eff
            out.loc[frontier.index, "frontier_slope_to_next"] = slopes

    return out


def compute_hypervolume(
    frontier: pd.DataFrame,
    cost_col: str,
    reference_efficiency: float = 0.0,
    reference_cost: float | None = None,
) -> float:
    """2D hypervolume for maximizing efficiency and minimizing cost."""
    if frontier.empty or cost_col not in frontier.columns:
        return float("nan")

    pts = frontier[["dispatch_rte", cost_col]].replace([np.inf, -np.inf], np.nan)
    pts = pts.dropna().sort_values(["dispatch_rte", cost_col])
    if pts.empty:
        return float("nan")

    if reference_cost is None:
        reference_cost = float(pts[cost_col].max() * 1.05)
    if not np.isfinite(reference_cost):
        return float("nan")

    hv = 0.0
    prev_eff = reference_efficiency
    for _, row in pts.iterrows():
        eff = float(row["dispatch_rte"])
        cost = float(row[cost_col])
        width = max(0.0, eff - prev_eff)
        height = max(0.0, reference_cost - cost)
        hv += width * height
        prev_eff = max(prev_eff, eff)
    return float(hv)


def _join_combo_labels(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    labels = [
        f"{row.tech_id}:{int(row.duration_h)}h"
        for row in df.sort_values(["frontier_order", "tech_id"]).itertuples()
    ]
    return ";".join(labels)


def build_frontier_metrics(points: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build per province-depth metrics and a compact hypervolume table."""
    rows: list[dict] = []
    group_cols = ["cost_scenario", "province_en", "province_cn", "target_depth_pct"]

    for keys, group in points.groupby(group_cols, sort=False):
        cost_scenario, province_en, province_cn, target_depth = keys
        reachable = group[
            group["is_reachable"]
            & np.isfinite(group["dispatch_rte"])
            & np.isfinite(group["unit_cost_cny_kwh"])
        ]
        frontier = reachable[reachable["is_pareto_frontier"]].copy()

        row = {
            "cost_scenario": cost_scenario,
            "province_en": province_en,
            "province_cn": province_cn,
            "target_depth_pct": float(target_depth),
            "n_tech_duration_total": int(group.shape[0]),
            "n_reachable": int(reachable.shape[0]),
            "n_frontier": int(frontier.shape[0]),
            "frontier_share": (
                float(frontier.shape[0] / reachable.shape[0])
                if reachable.shape[0] > 0
                else np.nan
            ),
            "frontier_tech_duration": _join_combo_labels(frontier),
        }

        if reachable.empty:
            rows.append(row)
            continue

        best_cost = reachable.sort_values(
            ["unit_cost_cny_kwh", "dispatch_rte"], ascending=[True, False]
        ).iloc[0]
        best_eff = reachable.sort_values(
            ["dispatch_rte", "unit_cost_cny_kwh"], ascending=[False, True]
        ).iloc[0]

        row.update(
            {
                "min_unit_cost_cny_kwh": float(reachable["unit_cost_cny_kwh"].min()),
                "median_unit_cost_cny_kwh": float(
                    reachable["unit_cost_cny_kwh"].median()
                ),
                "max_unit_cost_cny_kwh": float(reachable["unit_cost_cny_kwh"].max()),
                "min_cost_to_gas_ratio": float(
                    reachable["unit_cost_to_gas_ratio"].min()
                ),
                "median_cost_to_gas_ratio": float(
                    reachable["unit_cost_to_gas_ratio"].median()
                ),
                "max_cost_to_gas_ratio": float(
                    reachable["unit_cost_to_gas_ratio"].max()
                ),
                "min_efficiency": float(reachable["dispatch_rte"].min()),
                "max_efficiency": float(reachable["dispatch_rte"].max()),
                "best_cost_tech_id": best_cost["tech_id"],
                "best_cost_duration_h": int(best_cost["duration_h"]),
                "best_cost_unit_cost_cny_kwh": float(best_cost["unit_cost_cny_kwh"]),
                "best_efficiency_tech_id": best_eff["tech_id"],
                "best_efficiency_duration_h": int(best_eff["duration_h"]),
                "best_efficiency_rte": float(best_eff["dispatch_rte"]),
            }
        )

        if not frontier.empty:
            frontier = frontier.sort_values(["dispatch_rte", "unit_cost_cny_kwh"])
            slopes = frontier["frontier_slope_to_next"].replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            row.update(
                {
                    "frontier_min_unit_cost_cny_kwh": float(
                        frontier["unit_cost_cny_kwh"].min()
                    ),
                    "frontier_max_unit_cost_cny_kwh": float(
                        frontier["unit_cost_cny_kwh"].max()
                    ),
                    "frontier_efficiency_span": float(
                        frontier["dispatch_rte"].max() - frontier["dispatch_rte"].min()
                    ),
                    "frontier_cost_span_cny_kwh": float(
                        frontier["unit_cost_cny_kwh"].max()
                        - frontier["unit_cost_cny_kwh"].min()
                    ),
                    "frontier_slope_median": float(slopes.median())
                    if not slopes.empty
                    else np.nan,
                    "frontier_slope_min": float(slopes.min())
                    if not slopes.empty
                    else np.nan,
                    "frontier_slope_max": float(slopes.max())
                    if not slopes.empty
                    else np.nan,
                    "frontier_slope_std": float(slopes.std(ddof=0))
                    if len(slopes) > 1
                    else 0.0 if len(slopes) == 1 else np.nan,
                }
            )

            cost_ref = float(reachable["unit_cost_cny_kwh"].max() * 1.05)
            ratio_ref = float(reachable["unit_cost_to_gas_ratio"].max() * 1.05)
            hv_cost = compute_hypervolume(
                frontier,
                cost_col="unit_cost_cny_kwh",
                reference_efficiency=0.0,
                reference_cost=cost_ref,
            )
            hv_ratio = compute_hypervolume(
                frontier,
                cost_col="unit_cost_to_gas_ratio",
                reference_efficiency=0.0,
                reference_cost=ratio_ref,
            )
            denom_cost = (
                float(frontier["dispatch_rte"].max())
                * max(0.0, cost_ref - float(frontier["unit_cost_cny_kwh"].min()))
            )
            denom_ratio = (
                float(frontier["dispatch_rte"].max())
                * max(0.0, ratio_ref - float(frontier["unit_cost_to_gas_ratio"].min()))
            )
            row.update(
                {
                    "hypervolume_cny_kwh_eff": hv_cost,
                    "hypervolume_cost_ratio_eff": hv_ratio,
                    "hypervolume_cost_reference_cny_kwh": cost_ref,
                    "hypervolume_cost_ratio_reference": ratio_ref,
                    "hypervolume_cny_normalized": hv_cost / denom_cost
                    if denom_cost > 0
                    else np.nan,
                    "hypervolume_ratio_normalized": hv_ratio / denom_ratio
                    if denom_ratio > 0
                    else np.nan,
                }
            )

        rows.append(row)

    metrics = pd.DataFrame(rows)
    hv_cols = [
        "cost_scenario",
        "province_en",
        "province_cn",
        "target_depth_pct",
        "n_reachable",
        "n_frontier",
        "hypervolume_cny_kwh_eff",
        "hypervolume_cost_ratio_eff",
        "hypervolume_cny_normalized",
        "hypervolume_ratio_normalized",
    ]
    hypervolume = metrics[[col for col in hv_cols if col in metrics.columns]].copy()
    return metrics, hypervolume


def run_pareto_frontier(
    curve_points_csv: Path | None = None,
    cost_scenario: str = "baseline",
    depths: list[float] | None = None,
    output_dir: Path = OUTPUT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    depths = depths or DEFAULT_DEPTHS
    curve_path = curve_points_csv or default_curve_path(cost_scenario)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[M5] reading M4 curve points: {curve_path}")
    curves = read_curve_points(curve_path, cost_scenario=cost_scenario)
    print(f"[M5] rows={len(curves):,}, groups={curves.groupby(GROUP_COLS).ngroups:,}")

    depth_points = build_depth_points(curves, depths=depths)
    depth_points = mark_pareto_frontier(depth_points)
    frontier = depth_points[
        depth_points["is_reachable"] & depth_points["is_pareto_frontier"]
    ].copy()
    metrics, hypervolume = build_frontier_metrics(depth_points)

    points_path = output_dir / f"province_depth_points_{cost_scenario}.csv"
    frontier_path = output_dir / f"province_depth_frontier_{cost_scenario}.csv"
    metrics_path = output_dir / f"province_frontier_metrics_{cost_scenario}.csv"
    hv_path = output_dir / f"province_frontier_hypervolume_{cost_scenario}.csv"

    depth_points.to_csv(points_path, index=False)
    frontier.to_csv(frontier_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    hypervolume.to_csv(hv_path, index=False)

    print(f"[M5] depth points={len(depth_points):,}")
    print(f"[M5] frontier points={len(frontier):,}")
    print(f"[M5] metric rows={len(metrics):,}")
    print(f"[M5] outputs written to {output_dir}")
    return depth_points, frontier, metrics


def run_all_cost_scenarios(
    cost_scenarios: list[str] | None = None,
    depths: list[float] | None = None,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    cost_scenarios = cost_scenarios or DEFAULT_COST_SCENARIOS
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_all = []
    hypervolume_all = []

    for cost_scenario in cost_scenarios:
        run_pareto_frontier(
            cost_scenario=cost_scenario,
            depths=depths,
            output_dir=output_dir,
        )
        metrics_path = output_dir / f"province_frontier_metrics_{cost_scenario}.csv"
        hv_path = output_dir / f"province_frontier_hypervolume_{cost_scenario}.csv"
        if metrics_path.exists():
            metrics_all.append(pd.read_csv(metrics_path))
        if hv_path.exists():
            hypervolume_all.append(pd.read_csv(hv_path))

    if metrics_all:
        pd.concat(metrics_all, ignore_index=True).to_csv(
            output_dir / "province_frontier_metrics_all_scenarios.csv",
            index=False,
        )
    if hypervolume_all:
        pd.concat(hypervolume_all, ignore_index=True).to_csv(
            output_dir / "province_frontier_hypervolume_all_scenarios.csv",
            index=False,
        )

    manifest = pd.DataFrame(
        {
            "cost_scenario": cost_scenarios,
            "scenario_type": ["cost_sensitivity"] * len(cost_scenarios),
            "dispatch_assumption": ["shared_m3_baseline"] * len(cost_scenarios),
            "description": [
                {
                    "baseline": "base power cost + base energy cost + base M3 dispatch",
                    "low_cost": "low power cost + low energy cost; physical dispatch still from baseline M3",
                    "high_cost": "high power cost + high energy cost; physical dispatch still from baseline M3",
                }.get(s, s)
                for s in cost_scenarios
            ],
        }
    )
    manifest.to_csv(output_dir / "m5_run_manifest.csv", index=False)
    print(f"[M5] combined scenario outputs written to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the M5 Pareto frontier module.")
    parser.add_argument(
        "--cost-scenario",
        default="baseline",
        help="M4 cost scenario to consume, e.g. baseline, low_cost, high_cost.",
    )
    parser.add_argument(
        "--all-scenarios",
        action="store_true",
        help="Run baseline, low_cost, and high_cost in sequence and write combined outputs.",
    )
    parser.add_argument(
        "--curve-points-csv",
        default=None,
        help="Optional explicit M4 curve point table path.",
    )
    parser.add_argument(
        "--depths",
        nargs="+",
        type=float,
        default=DEFAULT_DEPTHS,
        help="Target substitution depths in percent.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory for M5 CSV outputs.",
    )
    args = parser.parse_args()

    if args.all_scenarios:
        run_all_cost_scenarios(
            cost_scenarios=DEFAULT_COST_SCENARIOS,
            depths=args.depths,
            output_dir=Path(args.output_dir),
        )
        return

    run_pareto_frontier(
        curve_points_csv=Path(args.curve_points_csv)
        if args.curve_points_csv
        else None,
        cost_scenario=args.cost_scenario,
        depths=args.depths,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
