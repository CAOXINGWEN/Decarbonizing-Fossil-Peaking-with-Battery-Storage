"""
M4 实验 D — 频率×长度热力图扫描，补齐 9 个 demand 节点。

完成 4×4 网格中 B+C 已有的 7 个之外的 9 个节点：
  freq3_len7, freq3_len10, freq3_len14,
  freq7_len7, freq7_len10, freq7_len14,
  freq10_len7, freq10_len10, freq10_len14

物理参数与 B/C 一致：5 年 markov + Y1 物理（demand_multiplier=1.0、
extreme_boost=1.3、event_intensity=1.5、intensity_multiplier=1.05）。

输出：output/m3_dispatch/demand_<stem>.csv 等
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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    real_df = load_real_demand_long_format()
    season_params = fit_season_distributions(real_df, CONFIG)
    cn_to_en = {v: k for k, v in EN_TO_CN.items()}

    curt_real = pd.read_csv(
        OUTPUT_DIR / "daily_provincial_curtailment.csv", index_col=0)
    cw_real = pd.read_csv(
        OUTPUT_DIR / "daily_provincial_charge_window_h.csv", index_col=0)

    rng = np.random.default_rng(2027)
    nodes = []
    for events in [3, 7, 10]:
        for length in [7, 10, 14]:
            nodes.append((events, length, f"freq{events}_len{length}"))

    summary = []
    for events, length, stem in nodes:
        node_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        print(f"\n=== 生成 {stem} (events={events}, length={length}) ===")
        params = {**BASE_PHY,
                    "extreme_events": events,
                    "event_length_days": length,
                    "label": stem}
        demand = bpd.build_one_node_demand("freqlen_grid", params,
                                              real_df, season_params, node_rng)
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
        q90 = np.percentile(nonzero, 90) if len(nonzero) > 0 else 0
        ext = (demand.values > q90).sum() if q90 > 0 else 0
        print(f"  saved: {out_demand.name}, shape={demand.shape}, "
              f"total={total:.2f} TWh, peak={peak:.1f} GWh, q90 ext={ext}")
        summary.append({"stem": stem, "events": events, "length": length,
                          "n_days": demand.shape[0],
                          "total_demand_TWh": total,
                          "max_single_day_GWh": peak,
                          "extreme_days_q90": ext})

    sdf = pd.DataFrame(summary)
    sdf.to_csv(OUTPUT_DIR / "freqlen_grid_demand_summary.csv", index=False)
    print(f"\n=== 摘要 ===")
    print(sdf.to_string(index=False))


if __name__ == "__main__":
    main()
