"""
M2 LDES 技术参数库读取与情景生成。

读取 M2_LDES_technology_parameter_library_v0_1.csv，生成 M3 调度仿真
所需的技术-情景参数组合。

核心公式:
    CAPEX = power_cost_cny_kw × P_kw + energy_cost_cny_kwh × E_kwh
    E_kwh = P_kw × duration_h
    annualized_capex = CAPEX × CRF(lifetime, discount_rate)
"""

import os

import numpy as np
import pandas as pd

from config import CONFIG

# ───────────────────────────────────────────────────────────────────
# 参数库路径
# ───────────────────────────────────────────────────────────────────

DEFAULT_CSV_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "dataset",
    "M2_LDES_technology_parameter_library_v0_2.csv",
)

LFP_DURATION_OVERRIDES_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "dataset",
    "M2_LFP_duration_cost_overrides_v0_1.csv",
)

LDES_LEARNING_CURVES_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "dataset",
    "M2_LDES_learning_cost_curves_v0_7.csv",
)

# ───────────────────────────────────────────────────────────────────
# 主结果技术组 / 敏感性技术组
# ───────────────────────────────────────────────────────────────────

PRIMARY_TECHS = ["li_lfp", "vrfb", "fecr_flow", "caes",
                 "laes", "h2_cavern", "gravity", "molten_salt_tes"]


# ───────────────────────────────────────────────────────────────────
# CRF 计算
# ───────────────────────────────────────────────────────────────────

def calculate_crf(discount_rate, years):
    """资本回收因子。"""
    return (discount_rate * (1 + discount_rate) ** years) / \
           ((1 + discount_rate) ** years - 1)


# ───────────────────────────────────────────────────────────────────
# 参数库加载
# ───────────────────────────────────────────────────────────────────

def load_tech_library(csv_path=None):
    """读取 LDES 技术参数 CSV，返回 DataFrame（tech_id 为索引）。"""
    if csv_path is None:
        csv_path = DEFAULT_CSV_PATH
    df = pd.read_csv(csv_path)
    df = df.set_index("tech_id")
    return df


def load_lfp_duration_overrides(csv_path=None):
    """读取 LFP 长时成本覆盖表，按 duration_h 建索引。"""
    if csv_path is None:
        csv_path = LFP_DURATION_OVERRIDES_PATH
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    if "duration_h" not in df.columns:
        return pd.DataFrame()
    df["duration_h"] = pd.to_numeric(df["duration_h"], errors="coerce")
    df = df.dropna(subset=["duration_h"]).copy()
    df["duration_h"] = df["duration_h"].astype(int)
    return df.set_index("duration_h")


def load_ldes_learning_curves(csv_path=None, curve_case="reference"):
    """Read component-level LDES learning factors from the M2 curve table."""
    if csv_path is None:
        csv_path = LDES_LEARNING_CURVES_PATH
    if not os.path.exists(csv_path):
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    required = {
        "tech_id",
        "curve_case",
        "learning_year",
        "power_cost_factor_vs_2025",
        "energy_cost_factor_vs_2025",
        "fixed_om_factor_vs_2025",
        "variable_om_factor_vs_2025",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

    if curve_case is not None:
        df = df[df["curve_case"].astype(str).eq(str(curve_case))].copy()

    df["learning_year"] = pd.to_numeric(df["learning_year"], errors="coerce")
    factor_cols = [
        "power_cost_factor_vs_2025",
        "energy_cost_factor_vs_2025",
        "fixed_om_factor_vs_2025",
        "variable_om_factor_vs_2025",
    ]
    for col in factor_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["tech_id", "learning_year"]).copy()
    df["learning_year"] = df["learning_year"].astype(int)
    return df.sort_values(["tech_id", "curve_case", "learning_year"]).reset_index(drop=True)


def component_learning_factors_for_tech(
    tech_id,
    learning_year,
    learning_curves=None,
    fallback_capex_factor=1.0,
):
    """Return component learning factors for one technology and year.

    Missing technologies fall back to the caller-provided generic CAPEX factor
    for power and energy, preserving legacy behavior until a technology-specific
    curve is added.
    """
    fallback = {
        "power_cost_factor_vs_2025": float(fallback_capex_factor),
        "energy_cost_factor_vs_2025": float(fallback_capex_factor),
        "fixed_om_factor_vs_2025": 1.0,
        "variable_om_factor_vs_2025": 1.0,
        "learning_curve_case": "fallback_config",
        "learning_curve_source_key": "CONFIG_year_costs",
        "learning_curve_confidence": "fallback",
        "learning_curve_notes": "No M2 technology-specific learning curve row; using legacy generic CAPEX factor.",
    }
    if learning_curves is None:
        learning_curves = load_ldes_learning_curves()
    if learning_curves is None or learning_curves.empty:
        return fallback

    sub = learning_curves[learning_curves["tech_id"].astype(str).eq(str(tech_id))].copy()
    if sub.empty:
        return fallback

    year = int(learning_year)
    exact = sub[sub["learning_year"].eq(year)]
    if not exact.empty:
        row = exact.iloc[0]
        out = {
            "power_cost_factor_vs_2025": float(row["power_cost_factor_vs_2025"]),
            "energy_cost_factor_vs_2025": float(row["energy_cost_factor_vs_2025"]),
            "fixed_om_factor_vs_2025": float(row["fixed_om_factor_vs_2025"]),
            "variable_om_factor_vs_2025": float(row["variable_om_factor_vs_2025"]),
            "learning_curve_case": str(row.get("curve_case", "reference")),
            "learning_curve_source_key": str(row.get("source_key", "")),
            "learning_curve_confidence": str(row.get("confidence", "")),
            "learning_curve_notes": str(row.get("notes", "")),
        }
        return out

    sub = sub.sort_values("learning_year")
    years = sub["learning_year"].to_numpy(dtype=float)
    out = {}
    for col, default in [
        ("power_cost_factor_vs_2025", fallback["power_cost_factor_vs_2025"]),
        ("energy_cost_factor_vs_2025", fallback["energy_cost_factor_vs_2025"]),
        ("fixed_om_factor_vs_2025", 1.0),
        ("variable_om_factor_vs_2025", 1.0),
    ]:
        values = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(years) & np.isfinite(values)
        if ok.any():
            out[col] = float(np.interp(float(year), years[ok], values[ok]))
        else:
            out[col] = float(default)

    nearest_idx = (sub["learning_year"] - year).abs().idxmin()
    nearest = sub.loc[nearest_idx]
    out.update(
        {
            "learning_curve_case": str(nearest.get("curve_case", "reference")),
            "learning_curve_source_key": str(nearest.get("source_key", "")),
            "learning_curve_confidence": str(nearest.get("confidence", "")),
            "learning_curve_notes": f"Interpolated/extrapolated from M2 curve anchors. {nearest.get('notes', '')}",
        }
    )
    return out


def apply_lfp_duration_cost_override(
    tech_id,
    scenario_name,
    duration_h,
    power_cost_cny_kw,
    energy_cost_cny_kwh,
    fixed_om_cny_kw_yr,
    overrides_csv_path=None,
):
    """Apply duration-specific LFP cost evidence for 6/8/10h sensitivity runs.

    The 4h LFP benchmark remains controlled by the main M2 row so it stays
    comparable with the first paper. Longer LFP durations use all-in
    duration-specific CNY/kWh costs from the override table.
    """
    if tech_id != "li_lfp":
        return power_cost_cny_kw, energy_cost_cny_kwh, fixed_om_cny_kw_yr

    duration_key = int(round(float(duration_h)))
    overrides = load_lfp_duration_overrides(overrides_csv_path)
    if overrides.empty or duration_key not in overrides.index:
        return power_cost_cny_kw, energy_cost_cny_kwh, fixed_om_cny_kw_yr

    row = overrides.loc[duration_key]
    if int(row.get("apply_override", 0)) != 1:
        return power_cost_cny_kw, energy_cost_cny_kwh, fixed_om_cny_kw_yr

    scenario_suffix = {
        "baseline": "base",
        "low_cost": "low",
        "high_cost": "high",
    }.get(scenario_name)
    if scenario_suffix is None:
        return power_cost_cny_kw, energy_cost_cny_kwh, fixed_om_cny_kw_yr

    capex_col = f"model_capex_cny_kwh_{scenario_suffix}"
    fixed_om_col = f"model_fixed_om_cny_kw_yr_{scenario_suffix}"
    if capex_col not in row.index:
        return power_cost_cny_kw, energy_cost_cny_kwh, fixed_om_cny_kw_yr

    capex_cny_kwh = float(row[capex_col])
    if not np.isfinite(capex_cny_kwh) or capex_cny_kwh <= 0:
        return power_cost_cny_kw, energy_cost_cny_kwh, fixed_om_cny_kw_yr

    fixed_om_override = fixed_om_cny_kw_yr
    if fixed_om_col in row.index:
        value = float(row[fixed_om_col])
        if np.isfinite(value) and value >= 0:
            fixed_om_override = value

    return 0.0, capex_cny_kwh, fixed_om_override


# ───────────────────────────────────────────────────────────────────
# 单技术参数提取
# ───────────────────────────────────────────────────────────────────

class TechParams:
    """一个技术在一个情景下的完整参数集。"""

    def __init__(self, tech_id, scenario, duration_h,
                 power_cost_cny_kw, energy_cost_cny_kwh,
                 rte, self_discharge_pct_day,
                 calendar_life_years, cycle_life_cycles,
                 response_time_min,
                 fixed_om_cny_kw_yr, variable_om_cny_mwh,
                 replacement_year, technology_cn, technology_en,
                 charge_power_ratio=1.0):
        self.tech_id = tech_id
        self.scenario = scenario
        self.duration_h = duration_h
        self.power_cost_cny_kw = power_cost_cny_kw
        self.energy_cost_cny_kwh = energy_cost_cny_kwh
        self.rte = rte
        self.self_discharge_pct_day = self_discharge_pct_day
        self.calendar_life_years = calendar_life_years
        self.cycle_life_cycles = cycle_life_cycles
        self.response_time_min = response_time_min
        self.fixed_om_cny_kw_yr = fixed_om_cny_kw_yr
        self.variable_om_cny_mwh = variable_om_cny_mwh
        self.replacement_year = replacement_year
        self.technology_cn = technology_cn
        self.technology_en = technology_en
        self.charge_power_ratio = float(charge_power_ratio)

        # 派生参数
        # CSV 中该字段按”百分数”存储，例如 0.03 表示 0.03%/day。
        # 调度核心需要的是 0-1 之间的日损耗比例。
        self.self_discharge_rate_day = self_discharge_pct_day / 100.0
        self.charge_eff = np.sqrt(rte)
        self.discharge_eff = np.sqrt(rte)

    def capex_total(self, power_mw):
        """计算总 CAPEX (CNY)。power_mw 为放电功率。"""
        discharge_power_kw = power_mw * 1000
        charge_power_kw = discharge_power_kw * self.charge_power_ratio
        rated_power_kw = max(charge_power_kw, discharge_power_kw)
        energy_kwh = discharge_power_kw * self.duration_h
        return (self.power_cost_cny_kw * rated_power_kw +
                self.energy_cost_cny_kwh * energy_kwh)

    def capex_per_kwh(self):
        """折算的单位能量 CAPEX (CNY/kWh)。"""
        rated_power_factor = max(self.charge_power_ratio, 1.0)
        return (self.power_cost_cny_kw * rated_power_factor / self.duration_h
                + self.energy_cost_cny_kwh)

    def annualized_cost(self, power_mw, discount_rate=0.036):
        """年化总成本 (CNY/yr)，含 CAPEX 年化 + 固定运维。"""
        discharge_power_kw = power_mw * 1000
        rated_power_kw = discharge_power_kw * max(self.charge_power_ratio, 1.0)
        crf = calculate_crf(discount_rate, self.calendar_life_years)
        capex = self.capex_total(power_mw)
        annual_capex = capex * crf
        annual_fixed_om = self.fixed_om_cny_kw_yr * rated_power_kw
        return annual_capex + annual_fixed_om

    def __repr__(self):
        ratio_str = f", CPR={self.charge_power_ratio}" if self.charge_power_ratio != 1.0 else ""
        return (f"TechParams({self.tech_id}, {self.scenario}, "
                f"{self.duration_h}h, RTE={self.rte:.2f}{ratio_str})")


# ───────────────────────────────────────────────────────────────────
# 情景生成
# ───────────────────────────────────────────────────────────────────

SCENARIOS = {
    "baseline": {"power_suffix": "base", "energy_suffix": "base", "rte_suffix": "base"},
    "low_cost": {"power_suffix": "low", "energy_suffix": "low", "rte_suffix": "high"},
    "high_cost": {"power_suffix": "high", "energy_suffix": "high", "rte_suffix": "low"},
}


def extract_tech_params(row, tech_id, scenario_name, duration_h):
    """从 DataFrame 的一行中提取指定情景的 TechParams。"""
    scen = SCENARIOS[scenario_name]
    power_cost = row[f"power_cost_cny_kw_{scen['power_suffix']}"]
    energy_cost = row[f"energy_cost_cny_kwh_{scen['energy_suffix']}"]
    rte = row[f"rte_{scen['rte_suffix']}"]
    fixed_om = row["fixed_om_cny_kw_yr"]

    power_cost, energy_cost, fixed_om = apply_lfp_duration_cost_override(
        tech_id=tech_id,
        scenario_name=scenario_name,
        duration_h=duration_h,
        power_cost_cny_kw=power_cost,
        energy_cost_cny_kwh=energy_cost,
        fixed_om_cny_kw_yr=fixed_om,
    )

    return TechParams(
        tech_id=tech_id,
        scenario=scenario_name,
        duration_h=duration_h,
        power_cost_cny_kw=power_cost,
        energy_cost_cny_kwh=energy_cost,
        rte=rte,
        self_discharge_pct_day=row["self_discharge_pct_day"],
        calendar_life_years=int(row["calendar_life_years"]),
        cycle_life_cycles=int(row["cycle_life_cycles"]),
        response_time_min=row["response_time_min"],
        fixed_om_cny_kw_yr=fixed_om,
        variable_om_cny_mwh=row["variable_om_cny_mwh"],
        replacement_year=int(row["replacement_year"]),
        technology_cn=row["technology_cn"],
        technology_en=row["technology_en"],
        charge_power_ratio=row.get("charge_power_ratio", 1.0),
    )


def generate_scenarios(csv_path=None, tech_ids=None, durations=None,
                       scenario_names=None):
    """生成技术 × 时长 × 情景的完整参数组合。

    Parameters
    ----------
    csv_path : str, optional
        参数库 CSV 路径
    tech_ids : list[str], optional
        技术 ID 列表，默认为主结果技术组
    durations : dict[str, list[float]], optional
        每个技术的时长扫描列表 {tech_id: [h1, h2, ...]}
        默认使用技术的 default_duration_h
    scenario_names : list[str], optional
        情景列表，默认 ["baseline", "low_cost", "high_cost"]

    Returns
    -------
    list[TechParams]
    """
    df = load_tech_library(csv_path)

    if tech_ids is None:
        tech_ids = PRIMARY_TECHS
    if scenario_names is None:
        scenario_names = list(SCENARIOS.keys())

    results = []
    for tech_id in tech_ids:
        if tech_id not in df.index:
            print(f"WARNING: {tech_id} not found in parameter library, skipped")
            continue
        row = df.loc[tech_id]

        if durations is not None and tech_id in durations:
            dur_list = durations[tech_id]
        else:
            dur_list = [row["default_duration_h"]]

        for dur in dur_list:
            if dur < row["duration_min_h"] or dur > row["duration_max_h"]:
                continue
            for scen_name in scenario_names:
                params = extract_tech_params(row, tech_id, scen_name, dur)
                results.append(params)

    return results


def generate_duration_sweep(csv_path=None, tech_ids=None, scenario_name="baseline"):
    """为每个技术生成从 duration_min 到 duration_max 的时长扫描。

    返回 list[TechParams]，按 [4h, 6h, 8h, 10h, 12h, 16h, 24h, 48h, 72h, 168h]
    中属于技术有效范围的子集。
    """
    sweep_durations = [4, 6, 8, 10, 12, 16, 24, 48, 72, 168]
    df = load_tech_library(csv_path)

    if tech_ids is None:
        tech_ids = PRIMARY_TECHS

    results = []
    for tech_id in tech_ids:
        if tech_id not in df.index:
            continue
        row = df.loc[tech_id]
        valid_durs = [d for d in sweep_durations
                      if row["duration_min_h"] <= d <= row["duration_max_h"]]
        for dur in valid_durs:
            params = extract_tech_params(row, tech_id, scenario_name, dur)
            results.append(params)

    return results


# ───────────────────────────────────────────────────────────────────
# 便捷函数：打印参数摘要
# ───────────────────────────────────────────────────────────────────

def print_scenario_summary(params_list):
    """打印情景参数摘要表。"""
    rows = []
    for p in params_list:
        rows.append({
            "tech_id": p.tech_id,
            "technology_cn": p.technology_cn,
            "scenario": p.scenario,
            "duration_h": p.duration_h,
            "power_cost": p.power_cost_cny_kw,
            "energy_cost": p.energy_cost_cny_kwh,
            "capex_per_kwh": p.capex_per_kwh(),
            "rte": p.rte,
            "self_discharge_%/day": p.self_discharge_pct_day,
            "life_years": p.calendar_life_years,
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


# ───────────────────────────────────────────────────────────────────
# 入口
# ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== M2 LDES 参数库加载测试 ===\n")

    print("--- 主结果技术组 × 默认时长 × 3 情景 ---")
    params = generate_scenarios()
    print_scenario_summary(params)

    print(f"\n共 {len(params)} 个参数组合\n")

    print("--- 时长扫描 (baseline) ---")
    sweep = generate_duration_sweep()
    print_scenario_summary(sweep)
    print(f"\n共 {len(sweep)} 个时长扫描点")
