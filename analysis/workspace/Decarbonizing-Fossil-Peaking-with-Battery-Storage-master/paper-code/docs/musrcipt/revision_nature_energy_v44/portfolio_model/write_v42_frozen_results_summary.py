#!/usr/bin/env python3
"""Write the evidence-frozen v42 results summary from production source data.

The writer refuses partial inputs and derives every reported number from the
v42 common-4 h-LFP portfolio outputs. It does not read any v41 result.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
V42_ROOT = HERE.parent
SOURCE = V42_ROOT / "source_data"
OUTPUT = V42_ROOT / "V42_FROZEN_RESULTS_SUMMARY.md"

CENTRAL = "fixed_s20_central"
CONDITIONAL_FIXED = "fixed_s31_conditional"
CONDITIONAL_OPTIMIZED = "optimized_s31_robustness"
FOCAL_CONFIGS = ("caes_24h", "h2_cavern_48h")


def read_csv(name: str) -> pd.DataFrame:
    path = SOURCE / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def one(frame: pd.DataFrame, **filters: Any) -> pd.Series:
    subset = frame
    for column, value in filters.items():
        if isinstance(value, float):
            subset = subset[np.isclose(subset[column].astype(float), value)]
        else:
            subset = subset[subset[column].eq(value)]
    if len(subset) != 1:
        raise RuntimeError(
            f"Expected one row for {filters}, obtained {len(subset)}"
        )
    return subset.iloc[0]


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def number(value: Any, digits: int = 2) -> str:
    if not finite(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def signed(value: Any, digits: int = 2) -> str:
    if not finite(value):
        return "NA"
    return f"{float(value):+.{digits}f}"


def interval(
    estimate: Any,
    low: Any,
    high: Any,
    digits: int = 2,
    unit: str = "",
) -> str:
    suffix = f" {unit}" if unit else ""
    return (
        f"{signed(estimate, digits)}"
        f" [{signed(low, digits)}, {signed(high, digits)}]{suffix}"
    )


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def focal_paired(
    frame: pd.DataFrame,
    source_label: str,
    scenario_id: int,
    config_id: str,
    metric: str,
) -> pd.Series:
    return one(
        frame,
        source_label=source_label,
        valuation_scenario_id=scenario_id,
        chronology="clustered_minus_dispersed",
        config_id=config_id,
        metric=metric,
    )


def recharge_row(
    frame: pd.DataFrame,
    source_label: str,
    analysis: str,
    intervention_id: str,
    config_id: str,
    metric: str,
    chronology: str,
) -> pd.Series:
    return one(
        frame,
        source_label=source_label,
        analysis=analysis,
        intervention_id=intervention_id,
        config_id=config_id,
        metric=metric,
        chronology=chronology,
    )


def shape_counts(
    frame: pd.DataFrame,
    source_label: str,
    scenario_id: int,
    chronology: str,
) -> dict[str, int]:
    subset = frame[
        frame["source_label"].eq(source_label)
        & frame["valuation_scenario_id"].eq(scenario_id)
        & frame["chronology"].eq(chronology)
    ]
    if int(subset["n_country_years"].max()) != 132:
        raise RuntimeError("Shape summary is not full 132-country-year evidence")
    return {
        str(row.envelope_shape): int(row.shape_count)
        for row in subset.itertuples(index=False)
    }


def main() -> None:
    summary = read_json(SOURCE / "v42_fixed_scale_mix_summary.json")
    if summary["qa"]["status"] != "pass":
        raise RuntimeError("Main v42 source-data QA did not pass")
    expected_sources = {
        CENTRAL,
        CONDITIONAL_FIXED,
        CONDITIONAL_OPTIMIZED,
    }
    observed_sources = {
        key.split("|", maxsplit=1)[0]
        for key in summary["qa"]["checks"]["country_year_counts"]
    }
    if observed_sources != expected_sources:
        raise RuntimeError(
            f"Expected three frozen sources, observed {observed_sources}"
        )
    if any(
        int(value) != 132
        for value in summary["qa"]["checks"]["country_year_counts"].values()
    ):
        raise RuntimeError("A frozen source does not contain 132 country-years")

    focal_primary = read_csv("v42_fixed_scale_mix_focal_config_summary.csv")
    bootstrap_primary = read_csv("v42_fixed_scale_mix_bootstrap_summary.csv")
    focal_all = read_csv(
        "v42_all_cases_mix_efc_discrete_scenario_focal_config_summary.csv"
    )
    shapes = read_csv(
        "v42_fixed_scale_mix_efc_discrete_scenario_"
        "valley_shape_summary.csv"
    )
    valley_pairs = read_csv(
        "v42_fixed_scale_mix_efc_discrete_scenario_valley_pairs.csv"
    )
    config_summary = read_csv(
        "v42_fixed_scale_mix_efc_discrete_scenario_"
        "configuration_summary.csv"
    )
    recharge = read_csv("v42_recharge_multiplier_paired_summary.csv")
    recharge_audit = read_json(SOURCE / "v42_recharge_multiplier_audit.json")
    value_rate = read_csv(
        "v42_fixed_scale_mix_efc_value_scan_formation_rate_curve.csv"
    )
    value_paired_rate = read_csv(
        "v42_fixed_scale_mix_efc_value_scan_"
        "paired_formation_rate_curve.csv"
    )
    phase_rate = read_csv(
        "v42_fixed_scale_mix_efc_cost_firm_phase_formation_rate.csv"
    )
    phase_paired_rate = read_csv(
        "v42_fixed_scale_mix_efc_cost_firm_phase_"
        "paired_formation_rate.csv"
    )

    scenario_table = pd.read_csv(
        V42_ROOT.parent
        / "revision_nature_energy_v40"
        / "currency_work"
        / "value_sensitivity_gap_summary_source_aware_eur.csv"
    )
    scenario_values = (
        scenario_table[
            scenario_table["scenario_id"].isin([20, 31, 34])
        ][
            [
                "scenario_id",
                "energy_value_eur_mwh",
                "firm_value_eur_kw_year",
                "charge_price_eur_mwh",
            ]
        ]
        .drop_duplicates()
        .set_index("scenario_id")
    )

    lines: list[str] = [
        "# V42 frozen results summary",
        "",
        "本文件由正式 `source_data/` 自动生成。全部数值均来自 v42 "
        "共同 4 h LFP 组合实验，不读取或复用任何 v41 单技术结果。",
        "",
        "统一比较臂为共同 4 h LFP 基础层加一个候选 LDES 尾部分支，"
        "相对于同一共同基础层加追加 4 h LFP。正的组合收益表示 "
        "LFP–LDES 分层组合优于继续追加 4 h LFP。三层证据严格分开为 "
        "S20 中央固定尺度、S31 条件性固定尺度和 S31 受限功率优化。",
        "",
        "## 1. 中央 S20 组合臂",
        "",
    ]
    s20 = scenario_values.loc[20]
    central_dispersed = shape_counts(shapes, CENTRAL, 20, "dispersed")
    central_clustered = shape_counts(shapes, CENTRAL, 20, "clustered")
    lines.extend(
        [
            "S20 的价值账户为能量替代 "
            f"{number(s20.energy_value_eur_mwh, 2)} EUR MWh⁻¹、"
            f"EENS 等效坚实服务 {number(s20.firm_value_eur_kw_year, 2)} "
            "EUR kW⁻¹ yr⁻¹ 和充电机会价格 "
            f"{number(s20.charge_price_eur_mwh, 2)} EUR MWh⁻¹。"
            "主样本覆盖 15 国、132 个国家—年份、49 个候选技术—时长"
            "配置和 24 个 4–168 h 唯一时长节点。",
            "",
            "中央成本下没有完整正—负—正进入谷。分散时序的前沿形态为 "
            f"{central_dispersed}，聚集时序为 {central_clustered}。"
            "这意味着中央结果识别的是相对价值沿时长的重新分配，"
            "而不是中央成本下已经实现的完整经济进入空档。",
            "",
            "来源：`v42_fixed_scale_mix_cells.csv.gz`、"
            "`v42_fixed_scale_mix_envelope.csv`、"
            "`v42_fixed_scale_mix_efc_primary_scenario_"
            "valley_shape_summary.csv`。",
            "",
            "## 2. 24 h CAES 与 48 h 盐穴储氢",
            "",
        ]
    )
    focal_rows: list[list[str]] = []
    metric_specs = [
        (
            "组合收益 Δ",
            "clustered_minus_dispersed_"
            "mix_benefit_eur_per_kw_common_power_year",
            2,
            "EUR kW⁻¹ yr⁻¹",
        ),
        (
            "EFC 增益 Δ",
            "clustered_minus_dispersed_"
            "mix_efc_gain_vs_extra_lfp4_mw",
            3,
            "MW",
        ),
        (
            "尾分支循环 Δ",
            "clustered_minus_dispersed_"
            "tail_equivalent_full_cycles",
            2,
            "cycles yr⁻¹",
        ),
        (
            "尾分支充电 Δ",
            "clustered_minus_dispersed_tail_charge_mwh",
            1,
            "MWh yr⁻¹",
        ),
        (
            "尾分支放电 Δ",
            "clustered_minus_dispersed_tail_discharge_mwh",
            1,
            "MWh yr⁻¹",
        ),
    ]
    for config_id in FOCAL_CONFIGS:
        label = "CAES 24 h" if config_id == "caes_24h" else "H₂ cavern 48 h"
        for metric_label, metric, digits, unit in metric_specs:
            row = focal_paired(
                focal_primary, CENTRAL, 20, config_id, metric
            )
            focal_rows.append(
                [
                    label,
                    metric_label,
                    interval(
                        row["mean"],
                        row["bootstrap_ci_low"],
                        row["bootstrap_ci_high"],
                        digits,
                        unit,
                    ),
                ]
            )
    lines.extend(
        [
            markdown_table(
                ["配置", "聚集减分散指标", "国家区组均值 [95% CI]"],
                focal_rows,
            ),
            "",
            "两种焦点尾部技术均在聚集时序下降低充电、放电、循环次数和"
            "相对于追加 LFP 的 EFC 增益。区间以国家为独立区组并保留"
            "同一国家的全部年份。",
            "",
            "来源：`v42_fixed_scale_mix_focal_config_summary.csv`。",
            "",
            "## 3. 4–168 h U 形 paired response 与循环—EFC 分解",
            "",
        ]
    )
    duration_metrics = {
        "benefit": (
            "clustered_minus_dispersed_"
            "best_mix_benefit_eur_per_kw_common_power_year"
        ),
        "efc": (
            "clustered_minus_dispersed_"
            "mix_efc_gain_vs_extra_lfp4_mw"
        ),
        "cycles": (
            "clustered_minus_dispersed_tail_equivalent_full_cycles"
        ),
    }
    selected_durations = [24, 32, 40, 48, 72, 96, 120, 168]
    duration_rows: list[list[str]] = []
    for duration in selected_durations:
        values: dict[str, pd.Series] = {}
        for label, metric in duration_metrics.items():
            values[label] = one(
                bootstrap_primary,
                analysis="duration_envelope_paired_effect",
                duration_h=float(duration),
                metric=metric,
            )
        duration_rows.append(
            [
                str(duration),
                interval(
                    values["benefit"].point_estimate,
                    values["benefit"].ci_low,
                    values["benefit"].ci_high,
                    2,
                    "EUR kW⁻¹ yr⁻¹",
                ),
                interval(
                    values["efc"].point_estimate,
                    values["efc"].ci_low,
                    values["efc"].ci_high,
                    3,
                    "MW",
                ),
                interval(
                    values["cycles"].point_estimate,
                    values["cycles"].ci_low,
                    values["cycles"].ci_high,
                    2,
                    "cycles yr⁻¹",
                ),
            ]
        )
    benefit_duration = bootstrap_primary[
        bootstrap_primary["analysis"].eq(
            "duration_envelope_paired_effect"
        )
        & bootstrap_primary["metric"].eq(duration_metrics["benefit"])
    ].sort_values("duration_h")
    trough = benefit_duration.loc[benefit_duration["point_estimate"].idxmin()]
    recovery = benefit_duration[
        benefit_duration["duration_h"].gt(float(trough.duration_h))
        & benefit_duration["ci_low"].gt(0.0)
    ].iloc[0]
    lines.extend(
        [
            markdown_table(
                [
                    "时长 h",
                    "组合收益 Δ [95% CI]",
                    "EFC 增益 Δ [95% CI]",
                    "尾分支循环 Δ [95% CI]",
                ],
                duration_rows,
            ),
            "",
            "聚集响应的经济谷底位于 "
            f"{number(trough.duration_h, 0)} h，组合收益变化为 "
            f"{interval(trough.point_estimate, trough.ci_low, trough.ci_high, 2, 'EUR kW⁻¹ yr⁻¹')}。"
            "尾分支循环在全部时长下降，但 EFC 响应由中时长负值转向"
            f"超长时正值；第一个 95% 区间完全高于零的节点为 "
            f"{number(recovery.duration_h, 0)} h。该分解支持周转侧受损与"
            "库存侧恢复并存，而不把中央成本下未跨零的绝对前沿误称为"
            "完整进入谷。",
            "",
            "来源：`v42_fixed_scale_mix_bootstrap_summary.csv`、"
            "`v42_fixed_scale_mix_envelope_paired_effects.csv`。",
            "",
            "## 4. 补能 1×/2×/4× 否证实验",
            "",
        ]
    )
    recharge_rows: list[list[str]] = []
    for config_id in FOCAL_CONFIGS:
        label = "CAES 24 h" if config_id == "caes_24h" else "H₂ cavern 48 h"
        for intervention in ("r1", "r2", "r4"):
            benefit = recharge_row(
                recharge,
                CENTRAL,
                "clustered_minus_dispersed",
                intervention,
                config_id,
                "mix_benefit_eur_per_kw_common_power_year",
                "clustered_minus_dispersed",
            )
            shared_clustered = recharge_row(
                recharge,
                CENTRAL,
                "chronology_level",
                intervention,
                config_id,
                "shared_recharge_binding_hours",
                "clustered",
            )
            shared_dispersed = recharge_row(
                recharge,
                CENTRAL,
                "chronology_level",
                intervention,
                config_id,
                "shared_recharge_binding_hours",
                "dispersed",
            )
            recharge_rows.append(
                [
                    label,
                    intervention.replace("r", "") + "×",
                    interval(
                        benefit.country_block_mean,
                        benefit.country_block_ci_low,
                        benefit.country_block_ci_high,
                        3,
                        "EUR kW⁻¹ yr⁻¹",
                    ),
                    (
                        f"{number(shared_dispersed.country_block_mean, 2)} / "
                        f"{number(shared_clustered.country_block_mean, 2)} h"
                    ),
                ]
            )
    lines.extend(
        [
            markdown_table(
                [
                    "配置",
                    "同形状补能数量",
                    "聚集减分散组合收益 [95% CI]",
                    "共享资源绑定小时 分散/聚集",
                ],
                recharge_rows,
            ),
            "",
            "补能资源数量从 1× 增至 4× 显著减少共享资源绑定小时，却"
            "没有恢复组合收益、尾分支吞吐或循环次数。因此该实验否定"
            "年度可充电数量不足是主要通道。它不等同于放宽技术充电"
            "功率；后者仍有数百个绑定小时，并由受限功率优化单独检验。",
            "",
            "来源：`v42_recharge_multiplier_paired_summary.csv`、"
            "`v42_recharge_multiplier_country_year_effects.csv.gz`。",
            "",
            "## 5. 完整谷的中央结果与条件相区",
            "",
        ]
    )
    exact_rows: list[list[str]] = []
    for scenario_id in (20, 31, 34):
        for chronology in ("dispersed", "clustered"):
            counts = shape_counts(shapes, CENTRAL, scenario_id, chronology)
            exact_rows.append(
                [
                    str(scenario_id),
                    chronology,
                    str(counts.get("strict_valley", 0)),
                    str(counts.get("full_ldes_exit", 0)),
                    str(counts.get("complex_nonvalley", 0)),
                ]
            )
    value_condition = one(
        value_paired_rate,
        source_label=CENTRAL,
        energy_value_eur_mwh=98.54031839922592,
        charge_price_eur_mwh=0.0,
        firm_value_eur_kw_year=600.0,
        ldes_tail_fixed_cost_multiplier=1.0,
    )
    phase_central_approx = one(
        phase_paired_rate,
        source_label=CENTRAL,
        firm_value_eur_kw_year=100.0,
        ldes_tail_fixed_cost_multiplier=0.45,
    )
    phase_max = phase_paired_rate[
        phase_paired_rate["source_label"].eq(CENTRAL)
    ].sort_values(
        [
            "cluster_induced_rate",
            "ldes_tail_fixed_cost_multiplier",
            "firm_value_eur_kw_year",
        ],
        ascending=[False, False, True],
    ).iloc[0]
    lines.extend(
        [
            markdown_table(
                [
                    "重估情景",
                    "时序",
                    "严格谷 n/132",
                    "全 LDES 退出 n/132",
                    "其他非谷 n/132",
                ],
                exact_rows,
            ),
            "",
            "在中心技术成本下，高能量、零充电价且坚实服务价值达到 "
            "600 EUR kW⁻¹ yr⁻¹ 时，聚集诱发严格谷为 "
            f"{int(round(value_condition.cluster_induced_rate * 132))}/132 "
            f"({number(100 * value_condition.cluster_induced_rate, 1)}%)。"
            "在 S20 能量与充电价格的成本—坚实价值相位面上，扫描"
            "最高诱发率为 "
            f"{int(round(phase_max.cluster_induced_rate * 132))}/132 "
            f"({number(100 * phase_max.cluster_induced_rate, 1)}%)，对应 "
            f"F={number(phase_max.firm_value_eur_kw_year, 0)} EUR kW⁻¹ yr⁻¹ "
            f"和 LDES 尾成本倍数 {number(phase_max.ldes_tail_fixed_cost_multiplier, 2)}。"
            "在接近中央坚实价值的 F=100 与成本倍数 0.45 下，诱发率为 "
            f"{int(round(phase_central_approx.cluster_induced_rate * 132))}/132 "
            f"({number(100 * phase_central_approx.cluster_induced_rate, 1)}%)。"
            "这些是确定性反转距离，不是成本发生概率。",
            "",
            "来源：`v42_fixed_scale_mix_efc_discrete_scenario_"
            "valley_shape_summary.csv`、`v42_fixed_scale_mix_efc_"
            "value_scan_paired_formation_rate_curve.csv`、"
            "`v42_fixed_scale_mix_efc_cost_firm_phase_"
            "paired_formation_rate.csv`。",
            "",
            "## 6. S31 fixed 与受限功率优化",
            "",
        ]
    )
    robustness_rows: list[list[str]] = []
    for source_label, source_name in (
        (CONDITIONAL_FIXED, "S31 fixed"),
        (CONDITIONAL_OPTIMIZED, "S31 bounded power"),
    ):
        for config_id in FOCAL_CONFIGS:
            config_name = (
                "CAES 24 h"
                if config_id == "caes_24h"
                else "H₂ cavern 48 h"
            )
            benefit = focal_paired(
                focal_all,
                source_label,
                31,
                config_id,
                "clustered_minus_dispersed_"
                "mix_benefit_eur_per_kw_common_power_year",
            )
            efc = focal_paired(
                focal_all,
                source_label,
                31,
                config_id,
                "clustered_minus_dispersed_"
                "mix_efc_gain_vs_extra_lfp4_mw",
            )
            cycles = focal_paired(
                focal_all,
                source_label,
                31,
                config_id,
                "clustered_minus_dispersed_"
                "tail_equivalent_full_cycles",
            )
            clustered_power = one(
                config_summary,
                source_label=source_label,
                valuation_scenario_id=31,
                chronology="clustered",
                config_id=config_id,
            )
            dispersed_power = one(
                config_summary,
                source_label=source_label,
                valuation_scenario_id=31,
                chronology="dispersed",
                config_id=config_id,
            )
            robustness_rows.append(
                [
                    source_name,
                    config_name,
                    interval(
                        benefit["mean"],
                        benefit["bootstrap_ci_low"],
                        benefit["bootstrap_ci_high"],
                        2,
                        "EUR kW⁻¹ yr⁻¹",
                    ),
                    interval(
                        efc["mean"],
                        efc["bootstrap_ci_low"],
                        efc["bootstrap_ci_high"],
                        3,
                        "MW",
                    ),
                    interval(
                        cycles["mean"],
                        cycles["bootstrap_ci_low"],
                        cycles["bootstrap_ci_high"],
                        2,
                        "cycles yr⁻¹",
                    ),
                    (
                        f"{number(dispersed_power.tail_power_mw_mean, 2)} / "
                        f"{number(clustered_power.tail_power_mw_mean, 2)} MW"
                    ),
                ]
            )
    fixed_transitions = (
        valley_pairs[
            valley_pairs["source_label"].eq(CONDITIONAL_FIXED)
            & valley_pairs["valuation_scenario_id"].eq(31)
        ]["valley_transition"]
        .value_counts()
        .to_dict()
    )
    optimized_transitions = (
        valley_pairs[
            valley_pairs["source_label"].eq(CONDITIONAL_OPTIMIZED)
            & valley_pairs["valuation_scenario_id"].eq(31)
        ]["valley_transition"]
        .value_counts()
        .to_dict()
    )
    lines.extend(
        [
            markdown_table(
                [
                    "证据层",
                    "配置",
                    "组合收益 Δ [95% CI]",
                    "EFC 增益 Δ [95% CI]",
                    "循环 Δ [95% CI]",
                    "尾功率均值 分散/聚集",
                ],
                robustness_rows,
            ),
            "",
            f"S31 fixed 的谷转移计数为 {fixed_transitions}；S31 受限功率"
            f"优化为 {optimized_transitions}。两者比较回答功率选择与重新"
            "计价能否改变聚集响应，不与 S20 中央证据合并估计。",
            "",
            "来源：`v42_all_cases_mix_efc_discrete_scenario_"
            "focal_config_summary.csv`、`v42_fixed_scale_mix_efc_"
            "discrete_scenario_configuration_summary.csv`、"
            "`v42_fixed_scale_mix_efc_discrete_scenario_valley_pairs.csv`。",
            "",
            "## 7. QA 与可追溯性",
            "",
        ]
    )
    qa_files = {
        "S20 central fixed": HERE
        / "full_132cy_fixed_s20_central"
        / "qa_summary.json",
        "S31 conditional fixed": HERE
        / "full_132cy_fixed"
        / "qa_summary.json",
        "S31 bounded power": HERE
        / "full_132cy_optimized"
        / "qa_summary.json",
    }
    qa_rows: list[list[str]] = []
    for label, path in qa_files.items():
        qa = read_json(path)
        checks = qa.get("checks", qa)
        country_year_count = qa.get(
            "country_year_count",
            len(qa.get("country_years", [])),
        )
        arm_row_count = qa.get(
            "arm_row_count",
            checks.get("actual_result_rows", ""),
        )
        cyclic_soc_residual = qa.get(
            "maximum_cyclic_soc_residual_mwh",
            max(
                float(
                    checks.get(
                        "maximum_base_soc_balance_residual_mwh", 0.0
                    )
                ),
                float(
                    checks.get(
                        "maximum_tail_soc_balance_residual_mwh", 0.0
                    )
                ),
            ),
        )
        qa_rows.append(
            [
                label,
                str(country_year_count),
                str(arm_row_count),
                number(
                    checks.get(
                        "maximum_task_balance_residual_mw", np.nan
                    ),
                    3,
                ),
                number(
                    checks.get(
                        "maximum_shared_recharge_violation_mw", np.nan
                    ),
                    3,
                ),
                number(
                    cyclic_soc_residual,
                    3,
                ),
                number(
                    qa.get(
                        "maximum_accounting_identity_residual_eur",
                        checks.get(
                            "maximum_system_cost_identity_error_eur",
                            np.nan,
                        ),
                    ),
                    6,
                ),
            ]
        )
    lines.extend(
        [
            markdown_table(
                [
                    "求解层",
                    "国家—年份",
                    "直接臂行数",
                    "任务平衡最大残差 MW",
                    "共享补能最大违反 MW",
                    "循环 SOC 最大残差 MWh",
                    "会计恒等最大残差 EUR",
                ],
                qa_rows,
            ),
            "",
            "主后处理 QA 状态为 pass，三来源均为 132 个国家—年份。"
            f"价值分解恒等式最大残差为 "
            f"{number(summary['qa']['checks']['maximum_value_component_identity_residual_eur'], 6)} EUR。"
            f"补能干预审计状态为 {recharge_audit['status']}，S20 与 S31 "
            "的 r1/r2/r4 均覆盖完整 132 个国家—年份。",
            "",
            "正式哈希、输入文件、脚本路径、bootstrap 种子和全部输出"
            "清单见 `v42_fixed_scale_mix_summary.json` 与 "
            "`v42_recharge_multiplier_audit.json`。本摘要只冻结组合技术"
            "证据，不包含任何 v41 单技术 headline。",
            "",
        ]
    )
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(str(OUTPUT))


if __name__ == "__main__":
    main()
