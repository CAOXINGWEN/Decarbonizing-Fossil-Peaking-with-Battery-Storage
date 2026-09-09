"""
M4 实验 B/C/D — q75 阈值版（与 M1 模块官方阈值一致）

重新生成 16 组合 demand：freq ∈ {4, 8, 12, 16} × len ∈ {5, 7, 10, 14}
基线 freq=4/yr ≈ q75 真实 ≥3 天事件均值 6.0/省/年 的 0.7×

原 freq3/5/7/10 系列保留作为旧版（基于 q90），但新分析以 q75 为准。

输出：
  output/m3_dispatch/demand_freq<F>_len<L>.csv
  output/m3_dispatch/curtailment_freq<F>_len<L>.csv
  output/m3_dispatch/charge_window_freq<F>_len<L>.csv
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

BASE_PHY = {
    "demand_multiplier": 1.0,
    "extreme_boost": 1.3,
    "event_intensity": 1.5,
    "intensity_multiplier": 1.05,
    "use_real_year": True,
    "n_years": 5,
}

EVENTS = [4, 8, 12, 16]
LENGTHS = [5, 7, 10, 14]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    real_df = load_real_demand_long_format()
    season_params = fit_season_distributions(real_df, CONFIG)
    cn_to_en = {v: k for k, v in EN_TO_CN.items()}

    curt_real = pd.read_csv(
        OUTPUT_DIR / "daily_provincial_curtailment.csv", index_col=0)
    cw_real = pd.read_csv(
        OUTPUT_DIR / "daily_provincial_charge_window_h.csv", index_col=0)

    rng = np.random.default_rng(20260601)

    summary = []
    for events in EVENTS:
        for length in LENGTHS:
            stem = f"freq{events}_len{length}"
            node_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))

            print(f"\n=== 生成 {stem} (events={events}/yr, length={length}d) ===")
            params = {**BASE_PHY,
                        "extreme_events": events,
                        "event_length_days": length,
                        "label": stem}
            demand = bpd.build_one_node_demand(
                "freqlen_grid_q75", params, real_df, season_params, node_rng)
            demand.columns = [cn_to_en.get(c, c) for c in demand.columns]
            out_demand = OUTPUT_DIR / f"demand_{stem}.csv"
            demand.to_csv(out_demand)

            curt = pd.concat([curt_real] * 5, axis=0, ignore_index=True)
            cw = pd.concat([cw_real] * 5, axis=0, ignore_index=True)
            curt.index.name = "day_idx"
            cw.index.name = "day_idx"
            curt.to_csv(OUTPUT_DIR / f"curtailment_{stem}.csv")
            cw.to_csv(OUTPUT_DIR / f"charge_window_{stem}.csv")

            total = demand.values.sum() / 1e9
            peak = demand.values.max() / 1e6
            nonzero = demand.values[demand.values > 0]
            q75 = np.percentile(nonzero, 75) if len(nonzero) > 0 else 0
            ext_q75 = (demand.values > q75).sum() if q75 > 0 else 0
            print(f"  shape={demand.shape}, total={total:.2f} TWh, "
                  f"peak={peak:.1f} GWh, q75 ext_days={ext_q75}")
            summary.append({
                "stem": stem, "events": events, "length": length,
                "n_days": demand.shape[0],
                "total_demand_TWh": total,
                "max_single_day_GWh": peak,
                "extreme_days_q75": ext_q75,
            })

    sdf = pd.DataFrame(summary)
    sdf.to_csv(OUTPUT_DIR / "freqlen_grid_q75_demand_summary.csv", index=False)
    print(f"\n=== 摘要 ===")
    print(sdf.to_string(index=False))


if __name__ == "__main__":
    main()
