"""
M4 实验 B + C — 极端事件「频率扫描」与「长度扫描」demand 生成器。

实验设计：
  - **共同**：5 年 markov 仿真（1 真实 + 4 markov）+ Y1 物理参数
    （demand_multiplier=1.0、extreme_boost=1.3、event_intensity=1.5、
    intensity_multiplier=1.05）
  - **B 频率扫描**：events ∈ {3, 5, 7, 10}/年，length=5 天固定
  - **C 长度扫描**：events=5/年固定，length ∈ {5, 7, 10, 14} 天

7 个新节点（freq5_len5 兼任 B-2 与 C-1）：
  freq3_len5, freq5_len5, freq7_len5, freq10_len5,
  freq5_len7, freq5_len10, freq5_len14

输出：
  output/m3_dispatch/demand_<stem>.csv
  output/m3_dispatch/curtailment_<stem>.csv (复制 5 次)
  output/m3_dispatch/charge_window_<stem>.csv (复制 5 次)

用法：python build_freqlen_sweep_demand.py
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from config import CONFIG
from event_clustering import EN_TO_CN
from monte_carlo_simulation import fit_season_distributions
from build_2yr_demand import load_real_demand_long_format
import build_progression_demand as bpd


PAPER_CODE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PAPER_CODE_DIR / "output" / "m3_dispatch"

# Y1 共享物理参数（demand_multiplier=1.0）
BASE_PHY = {
    "demand_multiplier": 1.0,
    "extreme_boost": 1.3,
    "event_intensity": 1.5,
    "intensity_multiplier": 1.05,
    "use_real_year": True,
    "n_years": 5,
}


def build_demand(events: int, length: int,
                  real_df: pd.DataFrame,
                  season_params: dict,
                  rng: np.random.Generator) -> pd.DataFrame:
    """以 Y1 物理参数 + 自定义 (events, length) 生成 5 年 demand。"""
    params = {**BASE_PHY,
                "extreme_events": events,
                "event_length_days": length,
                "label": f"freq{events}_len{length}"}
    return bpd.build_one_node_demand("freqlen_sweep", params,
                                       real_df, season_params, rng)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    real_df = load_real_demand_long_format()
    print(f"真实 demand: {real_df.shape}")
    season_params = fit_season_distributions(real_df, CONFIG)
    print(f"  季节模型: {len(season_params)} 省")

    cn_to_en = {v: k for k, v in EN_TO_CN.items()}

    # 节点定义：(events, length, stem)
    # 注意 freq5_len5 兼任 B-2 与 C-1
    nodes = [
        (3,  5,  "freq3_len5"),
        (5,  5,  "freq5_len5"),
        (7,  5,  "freq7_len5"),
        (10, 5,  "freq10_len5"),
        (5,  7,  "freq5_len7"),
        (5,  10, "freq5_len10"),
        (5,  14, "freq5_len14"),
    ]

    # 真实 curtailment / charge_window（已英文列），复制 5 次
    curt_real = pd.read_csv(
        OUTPUT_DIR / "daily_provincial_curtailment.csv", index_col=0)
    cw_real = pd.read_csv(
        OUTPUT_DIR / "daily_provincial_charge_window_h.csv", index_col=0)

    rng = np.random.default_rng(2026)
    summary = []
    for events, length, stem in nodes:
        node_rng = np.random.default_rng(
            rng.integers(0, 2**32 - 1))  # 每节点独立 seed

        print(f"\n=== 生成 {stem} (events={events}/yr, length={length}d) ===")
        demand_df = build_demand(events, length, real_df,
                                   season_params, node_rng)
        # 列名 中文 -> 英文
        demand_df.columns = [cn_to_en.get(c, c) for c in demand_df.columns]
        out_demand = OUTPUT_DIR / f"demand_{stem}.csv"
        demand_df.to_csv(out_demand)

        # curtailment / charge_window 复制 5 次（5 年）
        curt = pd.concat([curt_real] * 5, axis=0, ignore_index=True)
        cw = pd.concat([cw_real] * 5, axis=0, ignore_index=True)
        curt.index.name = "day_idx"
        cw.index.name = "day_idx"
        curt.to_csv(OUTPUT_DIR / f"curtailment_{stem}.csv")
        cw.to_csv(OUTPUT_DIR / f"charge_window_{stem}.csv")

        total = demand_df.values.sum() / 1e9   # TWh
        peak = demand_df.values.max() / 1e6    # GWh
        # 极端日数 = q90 阈值之上的天数总数
        nonzero = demand_df.values[demand_df.values > 0]
        if len(nonzero) > 0:
            q90 = np.percentile(nonzero, 90)
            extreme_days = (demand_df.values > q90).sum()
        else:
            extreme_days = 0
        print(f"  saved: {out_demand.name}, shape={demand_df.shape}")
        print(f"  total={total:.2f} TWh, peak={peak:.1f} GWh, "
              f"extreme_days(q90)={extreme_days}")
        summary.append({
            "stem": stem, "events": events, "length": length,
            "n_days": demand_df.shape[0],
            "total_demand_TWh": total,
            "max_single_day_GWh": peak,
            "extreme_days_q90": extreme_days,
        })

    sdf = pd.DataFrame(summary)
    sdf.to_csv(OUTPUT_DIR / "freqlen_sweep_demand_summary.csv", index=False)
    print(f"\n=== 摘要 ===")
    print(sdf.to_string(index=False))


if __name__ == "__main__":
    main()
