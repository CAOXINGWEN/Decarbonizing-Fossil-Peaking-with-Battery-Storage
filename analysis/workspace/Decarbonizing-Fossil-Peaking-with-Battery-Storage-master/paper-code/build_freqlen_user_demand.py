"""
M4 实验 F + G — 用户定义的频率/长度扫描 demand 生成器

用户定义：
  - 频率 = 365 天里非零调峰需求日的占比（真实 26 省均值 49%）
  - 长度 = 连续非零调峰段的平均长度（真实 26 省均值 2.4 天）

实验 F（频率扫描）：保持每省真实 demand 数值分布，只新增非零日
  目标 freq ∈ {60%, 70%, 80%, 90%}
  新增的零→非零日，从该省真实非零分布中重采样填入
  长度自然形成（不强制重组）

实验 G（长度扫描）：保持每省真实非零天数，重组使连续段平均长度达到目标
  目标 len ∈ {5, 7, 10, 14} 天
  保持总非零天数 = 真实，但把它们重新排成更长的连续段

每节点复制 5 年（与 progression 5 年仿真长度一致）。

输出：
  output/m3_dispatch/demand_F<freq>_real.csv         e.g. F60_real, F70_real
  output/m3_dispatch/demand_G<len>_real.csv          e.g. G5_real, G14_real
  output/m3_dispatch/curtailment_F60_real.csv 等

用法：python build_freqlen_user_demand.py
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from event_clustering import EN_TO_CN
from build_2yr_demand import load_real_demand_long_format

PAPER_CODE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PAPER_CODE_DIR / "output" / "m3_dispatch"

FREQS = [60, 70, 80, 90]   # 百分比
LENS = [5, 7, 10, 14]      # 天


def build_freq_demand_one_year(real_arr_prov: np.ndarray,
                                 target_freq_pct: float,
                                 rng: np.random.Generator) -> np.ndarray:
    """对单省 365 天数组，提高非零日比例到目标值。

    步骤：
      1. 找出真实非零日的位置和数值分布
      2. 计算需要新增的非零日数量
      3. 从真实非零数值分布中随机抽样 N 个值
      4. 随机选 N 个零日位置，填入抽样值

    保持原非零日的数值不变（保留真实分布）。
    """
    out = real_arr_prov.copy()
    n_days = len(out)
    nonzero_mask = out > 0
    n_nonzero_real = nonzero_mask.sum()
    target_n = int(round(target_freq_pct / 100.0 * n_days))
    n_to_add = max(0, target_n - n_nonzero_real)
    if n_to_add == 0:
        return out
    # 从真实非零分布中重采样
    nonzero_values = out[nonzero_mask]
    sampled = rng.choice(nonzero_values, size=n_to_add, replace=True)
    # 随机选零日位置（不重复）
    zero_positions = np.where(~nonzero_mask)[0]
    if len(zero_positions) >= n_to_add:
        chosen = rng.choice(zero_positions, size=n_to_add, replace=False)
    else:
        chosen = zero_positions
        sampled = sampled[:len(chosen)]
    out[chosen] = sampled
    return out


def build_len_demand_one_year(real_arr_prov: np.ndarray,
                                target_avg_len: float,
                                rng: np.random.Generator) -> np.ndarray:
    """对单省 365 天数组，重组使连续非零段平均长度达到目标值。

    步骤：
      1. 提取真实非零日数值（不变）
      2. 总非零天数 N = 真实非零天数
      3. 段数 m = N / target_avg_len（几乎确定）
      4. 段长一致：每段 = target_avg_len 天（向下取整）
      5. 段间均匀分布在 365 天内
      6. 段内填入从真实非零分布抽样
    """
    n_days = len(real_arr_prov)
    nonzero_values = real_arr_prov[real_arr_prov > 0]
    n_nonzero = len(nonzero_values)
    if n_nonzero == 0:
        return np.zeros_like(real_arr_prov)
    L = int(round(target_avg_len))
    n_segments = max(1, n_nonzero // L)
    out = np.zeros_like(real_arr_prov)
    # 段起点均匀分布
    if n_segments == 1:
        starts = [n_days // 2 - L // 2]
    else:
        spacing = n_days / n_segments
        starts = [int(i * spacing) + rng.integers(0, max(1, int(spacing - L)))
                    for i in range(n_segments)]
    # 段内填入随机抽样
    used = 0
    for s in starts:
        s = min(s, n_days - L)
        if used + L > n_nonzero:
            actual_L = n_nonzero - used
        else:
            actual_L = L
        if actual_L <= 0:
            break
        # 从真实非零分布中抽样 actual_L 个
        seg_values = rng.choice(nonzero_values, size=actual_L, replace=True)
        out[s:s + actual_L] = seg_values
        used += actual_L
    # 如还有剩余非零日没用完（n_nonzero % L 余数），随机分散
    remaining = n_nonzero - used
    if remaining > 0:
        zeros_pos = np.where(out == 0)[0]
        if len(zeros_pos) >= remaining:
            extra = rng.choice(zeros_pos, size=remaining, replace=False)
            extra_vals = rng.choice(nonzero_values, size=remaining, replace=True)
            out[extra] = extra_vals
    return out


def replicate_5_years(demand_yr1: pd.DataFrame) -> pd.DataFrame:
    """365 天 × 5 复制为 1825 天。"""
    return pd.concat([demand_yr1] * 5, axis=0, ignore_index=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    real_df = load_real_demand_long_format()
    print(f"真实 demand: {real_df.shape}")
    cn_to_en = {v: k for k, v in EN_TO_CN.items()}
    real_en = real_df.copy()
    real_en.columns = [cn_to_en.get(c, c) for c in real_en.columns]
    real_arr = real_en.values.astype(np.float64)

    # 真实 curtailment / charge_window（已英文列）
    curt_real = pd.read_csv(
        OUTPUT_DIR / "daily_provincial_curtailment.csv", index_col=0)
    cw_real = pd.read_csv(
        OUTPUT_DIR / "daily_provincial_charge_window_h.csv", index_col=0)

    rng = np.random.default_rng(20260602)
    summary = []

    # 实验 F：频率扫描
    for freq_pct in FREQS:
        stem = f"F{freq_pct}_real"
        print(f"\n=== F: 频率 {freq_pct}% ({stem}) ===")
        new_yr = real_arr.copy()
        for j in range(real_arr.shape[1]):
            new_yr[:, j] = build_freq_demand_one_year(
                real_arr[:, j], freq_pct, rng)
        # 计算实际频率
        actual_freq = (new_yr > 0).mean(axis=0).mean() * 100
        # 复制 5 年
        df_yr1 = pd.DataFrame(new_yr, columns=real_en.columns)
        df_5yr = replicate_5_years(df_yr1)
        df_5yr.index.name = "day_idx"
        df_5yr.to_csv(OUTPUT_DIR / f"demand_{stem}.csv")

        curt = pd.concat([curt_real] * 5, axis=0, ignore_index=True)
        cw = pd.concat([cw_real] * 5, axis=0, ignore_index=True)
        curt.index.name = "day_idx"; cw.index.name = "day_idx"
        curt.to_csv(OUTPUT_DIR / f"curtailment_{stem}.csv")
        cw.to_csv(OUTPUT_DIR / f"charge_window_{stem}.csv")

        total = df_5yr.values.sum() / 1e9
        peak = df_5yr.values.max() / 1e6
        print(f"  目标 {freq_pct}%, 实际 {actual_freq:.1f}%, "
              f"shape={df_5yr.shape}, total={total:.2f} TWh, peak={peak:.1f} GWh")
        summary.append({"experiment": "F", "stem": stem,
                          "target_freq_pct": freq_pct,
                          "actual_freq_pct": actual_freq,
                          "total_TWh_5yr": total, "peak_GWh": peak})

    # 实验 G：长度扫描
    for length in LENS:
        stem = f"G{length}_real"
        print(f"\n=== G: 长度 {length}d ({stem}) ===")
        new_yr = real_arr.copy()
        for j in range(real_arr.shape[1]):
            new_yr[:, j] = build_len_demand_one_year(
                real_arr[:, j], length, rng)
        # 计算实际平均段长
        def find_runs(b):
            runs, cur = [], 0
            for v in b:
                if v: cur += 1
                else:
                    if cur > 0: runs.append(cur); cur = 0
            if cur > 0: runs.append(cur)
            return runs

        all_runs = []
        for j in range(new_yr.shape[1]):
            all_runs.extend(find_runs(new_yr[:, j] > 0))
        all_runs = np.array(all_runs) if all_runs else np.array([0])
        actual_avg_len = all_runs.mean() if len(all_runs) > 0 else 0

        df_yr1 = pd.DataFrame(new_yr, columns=real_en.columns)
        df_5yr = replicate_5_years(df_yr1)
        df_5yr.index.name = "day_idx"
        df_5yr.to_csv(OUTPUT_DIR / f"demand_{stem}.csv")

        curt = pd.concat([curt_real] * 5, axis=0, ignore_index=True)
        cw = pd.concat([cw_real] * 5, axis=0, ignore_index=True)
        curt.index.name = "day_idx"; cw.index.name = "day_idx"
        curt.to_csv(OUTPUT_DIR / f"curtailment_{stem}.csv")
        cw.to_csv(OUTPUT_DIR / f"charge_window_{stem}.csv")

        total = df_5yr.values.sum() / 1e9
        peak = df_5yr.values.max() / 1e6
        print(f"  目标 {length}d, 实际 avg_len={actual_avg_len:.2f}d, "
              f"shape={df_5yr.shape}, total={total:.2f} TWh, peak={peak:.1f} GWh")
        summary.append({"experiment": "G", "stem": stem,
                          "target_avg_len_d": length,
                          "actual_avg_len_d": actual_avg_len,
                          "total_TWh_5yr": total, "peak_GWh": peak})

    sdf = pd.DataFrame(summary)
    sdf.to_csv(OUTPUT_DIR / "freqlen_user_demand_summary.csv", index=False)
    print(f"\n=== 摘要 ===")
    print(sdf.to_string(index=False))


if __name__ == "__main__":
    main()
