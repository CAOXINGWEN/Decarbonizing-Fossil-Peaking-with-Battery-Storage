"""
脱碳进程 demand 多节点 / 多年份生成器。

复用 build_2yr_demand.py 中的 Markov 链 + 极端事件注入工具，
按照 4 个脱碳进程节点（Y1 / Y3 / Y5 / Y10）生成 5 年长度的 demand 序列：

  Y1  (现状, 2026)        : demand×1.0,  extreme 3 ev × 5d × 1.3,  1 真实年 + 4 仿真年
  Y3  (中度脱碳, ~2028)   : demand×1.4,  extreme 5 ev × 7d × 1.5,  5 纯仿真年
  Y5  (深度脱碳, ~2030)   : demand×1.8,  extreme 7 ev × 10d × 1.7, 5 纯仿真年
  Y10 (高度脱碳, ~2035)   : demand×2.5,  extreme 10 ev × 14d × 2.0, 5 纯仿真年

每个节点输出三件套（行=日, 列=省 EN）：
  output/m3_dispatch/demand_5yr_{node}.csv          (1825 行)
  output/m3_dispatch/curtailment_5yr_{node}.csv     (1825 行)
  output/m3_dispatch/charge_window_5yr_{node}.csv   (1825 行)

配套 source data:
  output/m3_dispatch/progression_demand_summary.csv

用法（默认全部生成）:
  python build_progression_demand.py
  python build_progression_demand.py --node Y3
  python build_progression_demand.py --seed 2026
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import CONFIG
from event_clustering import EN_TO_CN
from monte_carlo_simulation import fit_season_distributions
from build_2yr_demand import (
    load_real_demand_long_format,
    generate_one_virtual_year,
    augment_virtual_year,
)


PAPER_CODE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PAPER_CODE_DIR / "output" / "m3_dispatch"
SUMMARY_OUT = OUTPUT_DIR / "progression_demand_summary.csv"

# ========= 节点参数表 =========
PROGRESSION_NODES = {
    "Y1": {
        "label": "Y1 (现状, baseline)",
        "demand_multiplier": 1.0,
        "extreme_boost": 1.3,
        "extreme_events": 3,
        "event_length_days": 5,
        "event_intensity": 1.5,
        "use_real_year": True,        # 第 0 年用真实数据
        "intensity_multiplier": 1.05, # 仿真年整体强度倍数
        "n_years": 5,
    },
    "Y3": {
        "label": "Y3 (中度脱碳)",
        "demand_multiplier": 1.4,
        "extreme_boost": 1.5,
        "extreme_events": 5,
        "event_length_days": 7,
        "event_intensity": 1.6,
        "use_real_year": False,
        "intensity_multiplier": 1.10,
        "n_years": 5,
    },
    "Y5": {
        "label": "Y5 (深度脱碳)",
        "demand_multiplier": 1.8,
        "extreme_boost": 1.7,
        "extreme_events": 7,
        "event_length_days": 10,
        "event_intensity": 1.8,
        "use_real_year": False,
        "intensity_multiplier": 1.15,
        "n_years": 5,
    },
    "Y10": {
        "label": "Y10 (高度脱碳, 接近 100% RE)",
        "demand_multiplier": 2.5,
        "extreme_boost": 2.0,
        "extreme_events": 10,
        "event_length_days": 14,
        "event_intensity": 2.0,
        "use_real_year": False,
        "intensity_multiplier": 1.20,
        "n_years": 5,
    },
}


def replicate_curtailment_n_years(n_years: int, scale_perturb: float,
                                    rng: np.random.Generator):
    csv_path = OUTPUT_DIR / "daily_provincial_curtailment.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, index_col=0)
    rows = []
    for _ in range(n_years):
        scale = 1.0 + rng.uniform(-scale_perturb, scale_perturb,
                                    size=df.shape[1])
        rows.append(df.values * scale)
    full = np.concatenate(rows, axis=0)
    return pd.DataFrame(full, columns=df.columns)


def replicate_charge_window_n_years(n_years: int):
    csv_path = OUTPUT_DIR / "daily_provincial_charge_window_h.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, index_col=0)
    return pd.concat([df] * n_years, axis=0, ignore_index=True)


def build_one_node_demand(node_key: str, params: dict,
                            real_df: pd.DataFrame,
                            season_params: dict,
                            rng: np.random.Generator) -> pd.DataFrame:
    """生成一个节点的 5 年 demand DataFrame（行=日, 列=中文省名）。

    步骤：
      1) 若 use_real_year：第 0 年用真实数据（不做极端注入）
      2) 其余年份：每年生成虚拟年 → 极端注入
      3) 全部年份 demand × demand_multiplier
    """
    n_years = params["n_years"]
    yr_blocks = []

    if params["use_real_year"]:
        yr_blocks.append(real_df.reset_index(drop=True).copy())
        n_virtual = n_years - 1
        start_year = 1
    else:
        n_virtual = n_years
        start_year = 0

    for y in range(n_virtual):
        year_idx = start_year + y
        dates = pd.date_range(f"203{year_idx}-01-01", periods=365, freq="D")
        virtual = generate_one_virtual_year(
            season_params, dates,
            intensity_factor=params["intensity_multiplier"],
            rng=rng,
        )
        enhanced = augment_virtual_year(
            virtual,
            extreme_boost=params["extreme_boost"],
            n_events=params["extreme_events"],
            length_days=params["event_length_days"],
            event_intensity=params["event_intensity"],
            rng=rng,
        )
        yr_blocks.append(enhanced.reset_index(drop=True))

    col_order = real_df.columns.tolist()
    yr_blocks = [b.reindex(columns=col_order).fillna(0.0) for b in yr_blocks]
    full = pd.concat(yr_blocks, axis=0, ignore_index=True)
    full.index.name = "day_idx"

    full = full * params["demand_multiplier"]
    return full


def summarize_node(node_key: str, params: dict,
                    demand_df: pd.DataFrame) -> dict:
    arr = demand_df.values
    nonzero = arr[arr > 0]
    q75 = np.percentile(nonzero, 75) if len(nonzero) > 0 else 0.0
    extreme_days = int((arr > q75).sum())
    return {
        "node": node_key,
        "label": params["label"],
        "demand_multiplier": params["demand_multiplier"],
        "extreme_events_per_yr": params["extreme_events"],
        "event_length_days": params["event_length_days"],
        "extreme_boost": params["extreme_boost"],
        "n_years": params["n_years"],
        "use_real_year": params["use_real_year"],
        "total_demand_TWh": arr.sum() / 1e9,
        "max_single_day_GWh": arr.max() / 1e6,
        "extreme_days_total": extreme_days,
        "q75_threshold_kwh": q75,
        "n_days": demand_df.shape[0],
        "n_provinces": demand_df.shape[1],
    }


def write_node_outputs(node_key: str, demand_df: pd.DataFrame,
                        curt_df, cw_df):
    cn_to_en = {v: k for k, v in EN_TO_CN.items()}
    out_demand = demand_df.copy()
    out_demand.columns = [cn_to_en.get(c, c) for c in out_demand.columns]
    p1 = OUTPUT_DIR / f"demand_5yr_{node_key}.csv"
    out_demand.to_csv(p1)
    print(f"  -> {p1.name}  shape={out_demand.shape}")

    if curt_df is not None:
        p2 = OUTPUT_DIR / f"curtailment_5yr_{node_key}.csv"
        curt_df.to_csv(p2)
        print(f"  -> {p2.name}  shape={curt_df.shape}")
    if cw_df is not None:
        p3 = OUTPUT_DIR / f"charge_window_5yr_{node_key}.csv"
        cw_df.to_csv(p3)
        print(f"  -> {p3.name}  shape={cw_df.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成 4 个脱碳节点 × 5 年 demand 序列")
    parser.add_argument("--node", choices=list(PROGRESSION_NODES.keys()) + ["all"],
                        default="all", help="选择要生成的节点（默认全部）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--curt-perturb", type=float, default=0.05,
                        help="curtailment 年间随机扰动比例")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[Step 1] 加载真实 demand 并拟合季节 Markov 参数")
    real_df = load_real_demand_long_format()
    season_params = fit_season_distributions(real_df, CONFIG)
    print(f"  完成 {len(season_params)} 省, real_df shape={real_df.shape}")

    nodes_to_run = (list(PROGRESSION_NODES.keys()) if args.node == "all"
                      else [args.node])
    summaries = []

    for node_key in nodes_to_run:
        params = PROGRESSION_NODES[node_key]
        # 每个 node 独立 RNG，方便对比
        node_rng = np.random.default_rng(args.seed + hash(node_key) % 100000)
        print(f"\n[Step 2] === {node_key}: {params['label']} ===")
        print(f"  demand×{params['demand_multiplier']}, "
              f"extreme {params['extreme_events']} ev × "
              f"{params['event_length_days']}d × "
              f"{params['extreme_boost']} boost, "
              f"use_real_year={params['use_real_year']}")

        demand_df = build_one_node_demand(
            node_key, params, real_df, season_params, node_rng)
        curt_df = replicate_curtailment_n_years(
            params["n_years"], args.curt_perturb, node_rng)
        cw_df = replicate_charge_window_n_years(params["n_years"])

        write_node_outputs(node_key, demand_df, curt_df, cw_df)
        summaries.append(summarize_node(node_key, params, demand_df))

    print("\n[Step 3] 写入 summary")
    sum_df = pd.DataFrame(summaries)
    sum_df.to_csv(SUMMARY_OUT, index=False)
    print(f"  -> {SUMMARY_OUT.name}")
    print("\n=== 节点对比 ===")
    print(sum_df[["node", "demand_multiplier", "extreme_events_per_yr",
                    "event_length_days", "extreme_boost",
                    "total_demand_TWh", "max_single_day_GWh",
                    "extreme_days_total"]].to_string(index=False))


if __name__ == "__main__":
    main()
