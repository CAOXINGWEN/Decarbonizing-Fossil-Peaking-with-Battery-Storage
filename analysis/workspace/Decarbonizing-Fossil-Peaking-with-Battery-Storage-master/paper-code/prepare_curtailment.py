"""
弃电数据预处理：将站级小时弃电聚合为省级日弃电。

输入: dataset/30日弃电/TotalCurt_Hourly_*.csv + 省份修正合并.geojson
输出: paper-code/output/m3_dispatch/daily_provincial_curtailment.csv
      (365 天 × 26 省, kWh)
      paper-code/output/m3_dispatch/daily_provincial_charge_window_h.csv
      (365 天 × 26 省, h)

注意：弃电数据有 33 个区域（如 Hebei north/south），需合并映射到
      需求数据的 26 省。
"""

import os
import json

import numpy as np
import pandas as pd

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "dataset")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "m3_dispatch")

# 弃电区域 → 需求数据 26 省的映射（合并子区域）
PROVINCE_MERGE_MAP = {
    "Hebei north": "Hebei",
    "Hebei south": "Hebei",
    "Inner Mongolia east": "Inner Mongolia",
    "Inner Mongolia west": "Inner Mongolia",
}

# 需求数据中的 26 省
TARGET_PROVINCES = {
    "Anhui", "Beijing", "Fujian", "Gansu", "Guangdong", "Guangxi",
    "Guizhou", "Hebei", "Henan", "Hubei", "Hunan", "Inner Mongolia",
    "Jiangsu", "Jiangxi", "Jilin", "Liaoning", "Ningxia", "Qinghai",
    "Shaanxi", "Shandong", "Shanxi", "Sichuan", "Tianjin", "Xinjiang",
    "Yunnan", "Zhejiang",
}


def build_station_province_map():
    """从 GeoJSON 构建 ObjectId → Province(English) 映射。"""
    geojson_path = os.path.join(DATASET_DIR, "省份修正合并.geojson")
    with open(geojson_path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    mapping = {}
    for feat in gj["features"]:
        props = feat["properties"]
        obj_id = str(props.get("ObjectId", ""))
        province = props.get("State_Province", "")
        if obj_id and province:
            mapping[obj_id] = province
    return mapping


def process_monthly_csv(filepath, station_map):
    """处理单月 CSV，返回省级逐日总弃电与逐小时聚合曲线。

    省份名会通过 PROVINCE_MERGE_MAP 合并，并只保留 TARGET_PROVINCES。
    """
    df = pd.read_csv(filepath, encoding="utf-8-sig", low_memory=False)
    id_col = df.columns[0]
    hour_cols = df.columns[1:]

    province_daily = {}
    province_hourly = {}

    for _, row in df.iterrows():
        obj_id = str(row[id_col]).strip()
        if obj_id.endswith(".0"):
            obj_id = obj_id[:-2]
        province = station_map.get(obj_id)
        if not province:
            continue

        province = PROVINCE_MERGE_MAP.get(province, province)
        if province not in TARGET_PROVINCES:
            continue

        if province not in province_daily:
            province_daily[province] = {}
            province_hourly[province] = {}

        for col in hour_cols:
            col_str = str(col)
            date_str = col_str[:8]
            mm_dd = f"{date_str[4:6]}-{date_str[6:8]}"
            if mm_dd == "02-29":
                continue
            hour_str = col_str[8:10]
            if not hour_str.isdigit():
                continue
            hour = int(hour_str)
            val = row[col]
            if pd.notna(val):
                value = float(val)
                province_daily[province][mm_dd] = (
                    province_daily[province].get(mm_dd, 0.0) + value
                )
                hour_key = (mm_dd, hour)
                province_hourly[province][hour_key] = (
                    province_hourly[province].get(hour_key, 0.0) + value
                )

    return province_daily, province_hourly


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("构建站点-省份映射...")
    station_map = build_station_province_map()
    print(f"  {len(station_map)} 个站点有省份标签")

    curt_dir = os.path.join(DATASET_DIR, "30日弃电")
    monthly_files = sorted([f for f in os.listdir(curt_dir) if f.endswith(".csv")])
    print(f"  找到 {len(monthly_files)} 个月度文件")

    all_province_daily = {}
    all_province_hourly = {}

    for i, fname in enumerate(monthly_files):
        print(f"  处理 [{i+1}/{len(monthly_files)}] {fname}...")
        filepath = os.path.join(curt_dir, fname)
        monthly, monthly_hourly = process_monthly_csv(filepath, station_map)

        for province, days in monthly.items():
            if province not in all_province_daily:
                all_province_daily[province] = {}
                all_province_hourly[province] = {}
            for mm_dd, val in days.items():
                all_province_daily[province][mm_dd] = (
                    all_province_daily[province].get(mm_dd, 0.0) + val
                )
        for province, hour_map in monthly_hourly.items():
            if province not in all_province_hourly:
                all_province_hourly[province] = {}
            for hour_key, value in hour_map.items():
                all_province_hourly[province][hour_key] = (
                    all_province_hourly[province].get(hour_key, 0.0) + value
                )

    # 组装为 DataFrame
    all_dates = sorted(set(
        d for days in all_province_daily.values() for d in days.keys()
    ))
    provinces = sorted(all_province_daily.keys())

    result = pd.DataFrame(index=all_dates, columns=provinces, dtype=float)
    result.index.name = "mm_dd"
    for prov in provinces:
        for dd in all_dates:
            result.loc[dd, prov] = all_province_daily.get(prov, {}).get(dd, 0.0)

    result = result.fillna(0.0)

    # Effective charge window: equivalent number of uniformly available charging
    # hours implied by the hourly curtailment profile on each day.
    charge_window = result.copy()
    for prov in provinces:
        effective_h = []
        hourly = all_province_hourly.get(prov, {})
        for dd in all_dates:
            total = float(result.loc[dd, prov])
            if total <= 0.0:
                effective_h.append(0.0)
                continue
            hourly_values = np.array(
                [hourly.get((dd, hour), 0.0) for hour in range(24)],
                dtype=float,
            )
            denom = float(np.sum(hourly_values * hourly_values))
            if denom <= 0.0:
                effective_h.append(0.0)
            else:
                effective_h.append((total * total) / denom)
        charge_window[prov] = np.asarray(effective_h, dtype=float)

    out_path = os.path.join(OUTPUT_DIR, "daily_provincial_curtailment.csv")
    window_path = os.path.join(OUTPUT_DIR, "daily_provincial_charge_window_h.csv")
    result.to_csv(out_path)
    charge_window.to_csv(window_path)
    print(f"\n=== 弃电数据预处理完成 ===")
    print(f"输出: {out_path}")
    print(f"输出: {window_path}")
    print(f"维度: {result.shape[0]} 天 × {result.shape[1]} 省")
    print(f"省份: {list(result.columns)}")
    print(f"\n各省年总弃电 (GWh):")
    totals = result.sum() / 1e6
    print(totals.sort_values(ascending=False).to_string())

    return result


if __name__ == "__main__":
    main()
