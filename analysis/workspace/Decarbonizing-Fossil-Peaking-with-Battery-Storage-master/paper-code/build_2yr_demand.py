"""
基于 monte_carlo_simulation.py 的方法生成"虚拟第二年 + 极端事件加强"demand 序列。

核心逻辑：
  1) 对真实 365 天 demand 拟合 3 状态 Markov + 季节强度分布（复用现有方法）
  2) 用同一套季节参数生成虚拟第二年（与原文方法完全一致）
  3) 在虚拟第二年里**加强极端事件**：
       - state=2 日强度乘以 extreme_boost
       - 注入 extreme_events 次连续连峰（长度 event_length_days，强度 event_intensity）
  4) 拼接成 730 天双年序列
  5) 同步生成 730 天的 curtailment / charge_window（直接复制单年 + 小扰动）

输出:
  output/m3_dispatch/demand_2yr_augmented.csv
  output/m3_dispatch/curtailment_2yr_augmented.csv
  output/m3_dispatch/charge_window_2yr_augmented.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import CONFIG, MC_CONFIG
from event_clustering import EN_TO_CN
from monte_carlo_simulation import (
    fit_season_distributions, classify_states,
    estimate_transition_matrix, compute_steady_state,
    sample_intensity, get_season_for_month,
)


PAPER_CODE_DIR = Path(__file__).resolve().parent
DATA_PATH = PAPER_CODE_DIR.parent / "data" / \
    "Daily Natural Gas Peaking Power Generation by Province in China.xlsx"
OUTPUT_DIR = PAPER_CODE_DIR / "output" / "m3_dispatch"
DEMAND_OUT = OUTPUT_DIR / "demand_2yr_augmented.csv"
CURT_OUT = OUTPUT_DIR / "curtailment_2yr_augmented.csv"
CW_OUT = OUTPUT_DIR / "charge_window_2yr_augmented.csv"


def load_real_demand_long_format() -> pd.DataFrame:
    """加载真实 demand 数据（行=日期，列=中文省名以匹配 CONFIG）。"""
    raw = pd.read_excel(DATA_PATH, sheet_name=0)
    raw = raw.set_index(raw.columns[0])
    n_days = raw.shape[1]
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    df = pd.DataFrame(raw.values.T, index=dates, columns=raw.index)
    # 英文 → 中文（CONFIG provinces 用中文）
    df.columns = [EN_TO_CN.get(c, c) for c in df.columns]
    return df


def generate_one_virtual_year(season_params, dates, intensity_factor,
                               rng) -> pd.DataFrame:
    """根据已拟合的季节参数生成一个虚拟年的 demand。

    Returns DataFrame, 行 = 日期，列 = 省。
    """
    seasons = MC_CONFIG["seasons"]
    provinces = list(season_params.keys())
    n_days = len(dates)
    out = pd.DataFrame(0.0, index=dates, columns=provinces)

    for province in provinces:
        for season_name, month_list in seasons.items():
            params = season_params[province].get(season_name)
            if params is None:
                continue
            day_mask = dates.month.isin(month_list)
            day_indices = np.where(day_mask)[0]
            if len(day_indices) == 0:
                continue

            T = params["transition_matrix"]
            cum_T = np.cumsum(T, axis=1)
            pi = compute_steady_state(T)
            cum_pi = np.cumsum(pi)

            # 单序列生成（不像 MC 那样并行多次仿真）
            n_season_days = len(day_indices)
            states = np.zeros(n_season_days, dtype=int)
            r0 = rng.random()
            states[0] = 0 if r0 < cum_pi[0] else (1 if r0 < cum_pi[1] else 2)
            r = rng.random(n_season_days - 1)
            for t in range(n_season_days - 1):
                cum0 = cum_T[states[t], 0]
                cum1 = cum_T[states[t], 1]
                states[t + 1] = (0 if r[t] < cum0 else
                                 (1 if r[t] < cum1 else 2))

            # 强度采样
            intensity = np.zeros(n_season_days, dtype=float)
            mask_low = states == 1
            mask_high = states == 2
            n_low = mask_low.sum()
            n_high = mask_high.sum()
            if n_low > 0:
                intensity[mask_low] = sample_intensity(
                    params["small_dist_name"], params["small_dist_params"],
                    params["small_values"], intensity_factor, n_low, rng,
                )
            if n_high > 0:
                intensity[mask_high] = sample_intensity(
                    params["large_dist_name"], params["large_dist_params"],
                    params["large_values"], intensity_factor, n_high, rng,
                )

            out.iloc[day_indices, out.columns.get_loc(province)] = intensity

    return out


def amplify_extreme_days(demand_arr: np.ndarray, q75: float,
                          extreme_boost: float) -> np.ndarray:
    """对 state=2 日（demand > q75）整体乘以 extreme_boost。"""
    out = demand_arr.copy()
    mask = out > q75
    out[mask] *= extreme_boost
    return out


def inject_consecutive_peaks(demand_arr: np.ndarray, q75: float,
                              n_events: int, length_days: int,
                              intensity: float, rng: np.random.Generator
                              ) -> np.ndarray:
    """在 demand 序列里注入 n_events 次长度 length_days 的连峰事件。

    每次连峰：在该窗口内每天 demand = max(原值, q75) × intensity。
    选择起点保证两次事件间隔 >= 2*length_days（避免重叠）。
    """
    out = demand_arr.copy()
    n_days = len(out)
    if n_events == 0 or length_days == 0:
        return out
    candidates = list(range(n_days - length_days))
    rng.shuffle(candidates)
    chosen = []
    for s in candidates:
        if all(abs(s - c) > length_days * 2 for c in chosen):
            chosen.append(s)
        if len(chosen) == n_events:
            break
    for s in chosen:
        for i in range(s, s + length_days):
            base = max(out[i], q75)
            out[i] = base * intensity
    return out


def augment_virtual_year(virtual_df: pd.DataFrame,
                          extreme_boost: float,
                          n_events: int,
                          length_days: int,
                          event_intensity: float,
                          rng) -> pd.DataFrame:
    """对虚拟年逐省做极端事件加强。"""
    out = virtual_df.copy()
    for prov in out.columns:
        arr = out[prov].values.astype(float)
        nonzero = arr[arr > 0]
        if len(nonzero) == 0:
            continue
        q75 = np.percentile(nonzero, 75)
        arr = amplify_extreme_days(arr, q75, extreme_boost)
        arr = inject_consecutive_peaks(arr, q75, n_events, length_days,
                                         event_intensity, rng)
        out[prov] = arr
    return out


def replicate_curtailment(n_years: int, scale_perturb: float, rng):
    """读取 daily_provincial_curtailment.csv 并复制 n 年（年间 ±perturb 扰动）。"""
    csv_path = OUTPUT_DIR / "daily_provincial_curtailment.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, index_col=0)
    n_days = len(df)
    rows = []
    for y in range(n_years):
        scale = 1.0 + rng.uniform(-scale_perturb, scale_perturb,
                                   size=df.shape[1])
        rows.append(df.values * scale)
    full = np.concatenate(rows, axis=0)
    out = pd.DataFrame(full, columns=df.columns)
    return out


def replicate_charge_window(n_years: int):
    """充电窗口直接复制（小时级，不做扰动避免不真实）。"""
    csv_path = OUTPUT_DIR / "daily_provincial_charge_window_h.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, index_col=0)
    full = pd.concat([df] * n_years, axis=0, ignore_index=True)
    return full


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成虚拟第二年 + 极端事件加强 demand 序列",
    )
    parser.add_argument("--intensity-multiplier", type=float, default=1.05,
                        help="虚拟年整体强度倍数")
    parser.add_argument("--extreme-boost", type=float, default=1.3,
                        help="state=2 日的额外强度乘数")
    parser.add_argument("--extreme-events", type=int, default=3,
                        help="注入连峰事件次数")
    parser.add_argument("--event-length-days", type=int, default=5,
                        help="每次连峰长度（天）")
    parser.add_argument("--event-intensity", type=float, default=1.5,
                        help="连峰日强度乘数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--curt-perturb", type=float, default=0.05,
                        help="弃电年间扰动比例")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("Step 1) 加载真实 demand → 拟合季节 Markov 参数")
    real_df = load_real_demand_long_format()
    season_params = fit_season_distributions(real_df, CONFIG)
    print(f"  完成 {len(season_params)} 省的季节参数拟合")

    print("Step 2) 生成虚拟第二年（基于 Markov 链）")
    dates_yr2 = pd.date_range("2024-01-01", periods=365, freq="D")
    virtual_df = generate_one_virtual_year(
        season_params, dates_yr2,
        intensity_factor=args.intensity_multiplier, rng=rng,
    )
    print(f"  虚拟年 shape: {virtual_df.shape}")
    print(f"  各省总量对比（real vs virtual）:")
    for prov in virtual_df.columns[:5]:
        real_total = real_df[prov].sum() / 1e9
        virt_total = virtual_df[prov].sum() / 1e9
        print(f"    {prov:<20s}  real={real_total:6.2f} GWh  "
              f"virtual={virt_total:6.2f} GWh")

    print("Step 3) 极端事件加强")
    print(f"  extreme_boost={args.extreme_boost}, "
          f"events={args.extreme_events}×{args.event_length_days}d "
          f"@ {args.event_intensity}x")
    enhanced_df = augment_virtual_year(
        virtual_df,
        extreme_boost=args.extreme_boost,
        n_events=args.extreme_events,
        length_days=args.event_length_days,
        event_intensity=args.event_intensity,
        rng=rng,
    )

    print("Step 4) 拼接 730 天序列")
    # 列顺序必须与真实 demand 一致
    col_order = real_df.columns.tolist()
    real_aligned = real_df[col_order].reset_index(drop=True)
    enhanced_aligned = enhanced_df[col_order].reset_index(drop=True)
    full_demand = pd.concat([real_aligned, enhanced_aligned],
                             axis=0, ignore_index=True)
    full_demand.index.name = "day_idx"
    # 输出时把中文列名转回英文（M3 ldes_simulator 用英文）
    cn_to_en = {v: k for k, v in EN_TO_CN.items()}
    full_demand.columns = [cn_to_en.get(c, c) for c in full_demand.columns]
    print(f"  730 天 demand shape: {full_demand.shape}")
    full_demand.to_csv(DEMAND_OUT)
    print(f"  ✓ {DEMAND_OUT}")

    print("Step 5) 复制 curtailment / charge_window 到 730 天")
    curt = replicate_curtailment(n_years=2, scale_perturb=args.curt_perturb,
                                  rng=rng)
    if curt is not None:
        curt.to_csv(CURT_OUT)
        print(f"  ✓ {CURT_OUT}  shape: {curt.shape}")
    else:
        print(f"  ! curtailment 单年 CSV 不存在，跳过")
    cw = replicate_charge_window(n_years=2)
    if cw is not None:
        cw.to_csv(CW_OUT)
        print(f"  ✓ {CW_OUT}  shape: {cw.shape}")

    # 关键统计对比
    print("\n=== 真实年 vs 加强虚拟年 关键指标对比 ===")
    print(f"{'指标':<28s}  {'真实':>12s}  {'加强虚拟':>14s}  {'变化':>10s}")
    for label, fn in [
        ("总量 (GWh)",
         lambda d: d.sum().sum() / 1e9),
        ("平均极端日数/省",
         lambda d: np.mean([
             ((d[c].values > np.percentile(d[c].values[d[c].values > 0], 75)
               if (d[c].values > 0).any() else False)).sum()
             for c in d.columns
         ])),
        ("最大单日 (GWh)", lambda d: d.values.max() / 1e6),
    ]:
        try:
            v_real = fn(real_aligned)
            v_virt = fn(enhanced_aligned)
            diff = (v_virt - v_real) / v_real * 100
            print(f"{label:<28s}  {v_real:>12.3f}  {v_virt:>14.3f}  "
                  f"{diff:>+9.1f}%")
        except Exception as e:
            print(f"{label} 计算失败: {e}")


if __name__ == "__main__":
    main()
