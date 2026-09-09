#!/usr/bin/env python3
"""Build a source-aware USD/CNY/EUR extension of the M2 storage cost library.

This script does not replace the frozen v0.1 library. It adds:

1. a model-USD-equivalent value reconstructed from the historical model
   conversion of 7.2 CNY per USD;
2. EUR values calculated through both USD and CNY routes;
3. a technology-level source-aware EUR reporting alias applied consistently
   to CAPEX, fixed O&M, variable O&M and annual fixed cost;
4. a complete route-comparison table and a 2025/2030/2035 summary.

The reconstructed USD values are explicitly labelled ``model_equivalent``.
They are not represented as raw source observations for CNY-anchored or
author-calibrated technologies.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
INPUT = HERE / "M2_LDES_duration_year_parameter_library_v0_1.csv"
OUTPUT_LIBRARY = (
    HERE / "M2_LDES_duration_year_parameter_library_v0_2_currency_provenance.csv"
)
OUTPUT_COST = HERE / "M2_LDES_duration_cost_paths_v0_2_currency_provenance.csv"
OUTPUT_COMPARISON = HERE / "M2_LDES_duration_currency_route_comparison_v0_2.csv"
OUTPUT_SUMMARY = (
    HERE / "M2_LDES_duration_currency_route_comparison_summary_v0_2.csv"
)
OUTPUT_XLSX = (
    HERE / "M2_LDES_duration_year_parameter_library_v0_2_currency_provenance.xlsx"
)
OUTPUT_NOTE = HERE / "M2_LDES美元直换欧元与人民币换算欧元对比_v0_2.md"

MODEL_CNY_PER_USD = 7.2
ECB_2025_USD_PER_EUR = 1.1300
ECB_2025_CNY_PER_EUR = 8.1185043137255
EUR_PER_USD = 1.0 / ECB_2025_USD_PER_EUR
EUR_PER_CNY = 1.0 / ECB_2025_CNY_PER_EUR
ECB_USD_SERIES = "EXR.A.USD.EUR.SP00.A"
ECB_CNY_SERIES = "EXR.A.CNY.EUR.SP00.A"
ECB_USD_URL = (
    "https://data.ecb.europa.eu/data/datasets/EXR/"
    "EXR.A.USD.EUR.SP00.A"
)
ECB_CNY_URL = (
    "https://data.ecb.europa.eu/data/datasets/EXR/"
    "EXR.A.CNY.EUR.SP00.A"
)

CAPEX_BASIS = {
    "li_lfp": "CNY_direct_or_hybrid_China_anchor",
    "vrfb": "CNY_direct_calibrated_to_China_EPC",
    "caes": "USD_components_with_author_learning_path",
    "fecr_flow": "USD_model_proxy_not_raw_technology_quote",
    "laes": "USD_model_proxy_not_raw_technology_quote",
    "gravity": "USD_model_proxy_from_US_evidence",
    "molten_salt_tes": "USD_model_proxy_from_US_thermal_evidence",
    "h2_cavern": "USD_model_proxy_from_US_hydrogen_evidence",
}

TECHNOLOGY_ROUTE = {
    tech_id: ("CNY_to_EUR" if tech_id in {"li_lfp", "vrfb"} else "USD_to_EUR")
    for tech_id in CAPEX_BASIS
}

USD_STATUS = {
    "li_lfp": "counterfactual_model_equivalent_only",
    "vrfb": "counterfactual_model_equivalent_only",
    "caes": "reconstructed_model_USD_path_from_ESGC_components_and_author_anchors",
    "fecr_flow": "reconstructed_USD_equivalent_proxy",
    "laes": "reconstructed_USD_equivalent_proxy",
    "gravity": "reconstructed_USD_equivalent_proxy",
    "molten_salt_tes": "reconstructed_USD_equivalent_proxy",
    "h2_cavern": "reconstructed_USD_equivalent_proxy",
}

MONEY_COLUMNS = [
    "power_cost_cny_kw",
    "energy_cost_cny_kwh",
    "total_capex_cny_kwh",
    "annualized_capex_cny_per_kwh_capacity_year",
    "fixed_om_cny_kw_yr",
    "variable_om_cny_mwh",
    "fixed_cost_cny_per_kwh_capacity_year",
]


def usd_column(cny_column: str) -> str:
    return cny_column.replace("_cny_", "_usd_model_equivalent_")


def eur_via_usd_column(cny_column: str) -> str:
    return cny_column.replace("_cny_", "_eur_via_usd_")


def eur_via_cny_column(cny_column: str) -> str:
    return cny_column.replace("_cny_", "_eur_via_cny_")


def reporting_eur_column(cny_column: str) -> str:
    return cny_column.replace("_cny_", "_eur_")


def add_currency_routes(source: pd.DataFrame) -> pd.DataFrame:
    frame = source.copy()
    frame["library_version"] = "v0.2-currency-provenance"
    frame["capex_currency_basis"] = frame["tech_id"].map(CAPEX_BASIS)
    frame["technology_currency_route"] = frame["tech_id"].map(
        TECHNOLOGY_ROUTE
    )
    # Retained as a backwards-compatible alias for earlier v0.2 consumers.
    frame["capex_reporting_route"] = frame["technology_currency_route"]
    frame["usd_value_status"] = frame["tech_id"].map(USD_STATUS)
    frame["model_fx_cny_per_usd"] = MODEL_CNY_PER_USD
    frame["reporting_fx_usd_per_eur"] = ECB_2025_USD_PER_EUR
    frame["reporting_fx_cny_per_eur"] = ECB_2025_CNY_PER_EUR
    frame["reporting_fx_reference_year"] = 2025
    frame["reporting_fx_usd_series_key"] = ECB_USD_SERIES
    frame["reporting_fx_cny_series_key"] = ECB_CNY_SERIES
    frame["reporting_fx_usd_source_url"] = ECB_USD_URL
    frame["reporting_fx_cny_source_url"] = ECB_CNY_URL

    for cny_name in MONEY_COLUMNS:
        values = pd.to_numeric(frame[cny_name], errors="raise")
        frame[usd_column(cny_name)] = values / MODEL_CNY_PER_USD
        frame[eur_via_usd_column(cny_name)] = (
            frame[usd_column(cny_name)] / ECB_2025_USD_PER_EUR
        )
        frame[eur_via_cny_column(cny_name)] = values / ECB_2025_CNY_PER_EUR

    usd_route_mask = frame["technology_currency_route"].eq("USD_to_EUR")
    for cny_name in [
        "power_cost_cny_kw",
        "energy_cost_cny_kwh",
        "total_capex_cny_kwh",
        "annualized_capex_cny_per_kwh_capacity_year",
    ]:
        report_name = reporting_eur_column(cny_name)
        frame[report_name] = frame[eur_via_cny_column(cny_name)]
        frame.loc[usd_route_mask, report_name] = frame.loc[
            usd_route_mask, eur_via_usd_column(cny_name)
        ]

    # Apply one technology-level route to every technology-cost component.
    # This deliberately keeps all LFP and VRFB components on CNY -> EUR,
    # including the LFP duration-specific fixed-O&M values.
    frame["fixed_om_reporting_route"] = frame["technology_currency_route"]
    frame["fixed_om_eur_kw_yr"] = frame["fixed_om_eur_via_cny_kw_yr"]
    frame.loc[usd_route_mask, "fixed_om_eur_kw_yr"] = frame.loc[
        usd_route_mask, "fixed_om_eur_via_usd_kw_yr"
    ]

    frame["variable_om_reporting_route"] = frame[
        "technology_currency_route"
    ]
    frame["variable_om_eur_mwh"] = frame["variable_om_eur_via_cny_mwh"]
    frame.loc[usd_route_mask, "variable_om_eur_mwh"] = frame.loc[
        usd_route_mask, "variable_om_eur_via_usd_mwh"
    ]

    # Fixed cost is recomputed from its source-aware CAPEX and fixed-O&M
    # components; both now use the same technology-level route.
    frame["fixed_cost_reporting_route"] = frame[
        "technology_currency_route"
    ]
    frame["fixed_cost_eur_per_kwh_capacity_year"] = (
        frame["annualized_capex_eur_per_kwh_capacity_year"]
        + frame["fixed_om_eur_kw_yr"] / frame["duration_h"]
    )
    frame["reporting_currency"] = "EUR"
    return frame


def build_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    comparison = frame[
        [
            "tech_id",
            "technology_cn",
            "technology_en",
            "duration_h",
            "learning_year",
            "capex_currency_basis",
            "technology_currency_route",
            "capex_reporting_route",
            "usd_value_status",
            "total_capex_cny_kwh",
            "total_capex_usd_model_equivalent_kwh",
            "total_capex_eur_via_usd_kwh",
            "total_capex_eur_via_cny_kwh",
            "total_capex_eur_kwh",
        ]
    ].copy()
    comparison["usd_minus_cny_route_eur_kwh"] = (
        comparison["total_capex_eur_via_usd_kwh"]
        - comparison["total_capex_eur_via_cny_kwh"]
    )
    comparison["usd_minus_cny_route_pct"] = (
        comparison["usd_minus_cny_route_eur_kwh"]
        / comparison["total_capex_eur_via_cny_kwh"]
        * 100.0
    )
    comparison["mechanically_lower_route"] = np.where(
        comparison["usd_minus_cny_route_eur_kwh"].lt(0),
        "USD_to_EUR",
        np.where(
            comparison["usd_minus_cny_route_eur_kwh"].gt(0),
            "CNY_to_EUR",
            "same",
        ),
    )
    comparison["model_fx_cny_per_usd"] = MODEL_CNY_PER_USD
    comparison["ecb_2025_usd_per_eur"] = ECB_2025_USD_PER_EUR
    comparison["ecb_2025_cny_per_eur"] = ECB_2025_CNY_PER_EUR
    return comparison


def build_anchor_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    anchor = comparison[
        comparison["learning_year"].isin([2025, 2030, 2035])
    ].copy()
    id_columns = [
        "tech_id",
        "technology_cn",
        "technology_en",
        "duration_h",
        "capex_currency_basis",
        "technology_currency_route",
        "capex_reporting_route",
        "usd_value_status",
    ]
    value_columns = [
        "total_capex_eur_via_usd_kwh",
        "total_capex_eur_via_cny_kwh",
        "total_capex_eur_kwh",
        "usd_minus_cny_route_eur_kwh",
    ]
    parts: list[pd.DataFrame] = []
    for year in [2025, 2030, 2035]:
        part = anchor[anchor["learning_year"].eq(year)][id_columns + value_columns]
        part = part.rename(
            columns={
                column: f"{year}_{column}"
                for column in value_columns
            }
        )
        parts.append(part)
    summary = parts[0]
    for part in parts[1:]:
        summary = summary.merge(part, on=id_columns, how="inner", validate="one_to_one")
    return summary.sort_values(["tech_id", "duration_h"]).reset_index(drop=True)


def build_metadata() -> pd.DataFrame:
    implied_cny_per_usd = ECB_2025_CNY_PER_EUR / ECB_2025_USD_PER_EUR
    route_ratio = (
        ECB_2025_CNY_PER_EUR
        / (MODEL_CNY_PER_USD * ECB_2025_USD_PER_EUR)
    )
    return pd.DataFrame(
        [
            {"field": "model_cny_per_usd", "value": MODEL_CNY_PER_USD},
            {"field": "ecb_2025_usd_per_eur", "value": ECB_2025_USD_PER_EUR},
            {"field": "ecb_2025_cny_per_eur", "value": ECB_2025_CNY_PER_EUR},
            {
                "field": "ecb_2025_implied_cny_per_usd",
                "value": implied_cny_per_usd,
            },
            {
                "field": "usd_route_vs_cny_route_ratio",
                "value": route_ratio,
            },
            {
                "field": "usd_route_vs_cny_route_pct",
                "value": (route_ratio - 1.0) * 100.0,
            },
            {"field": "ecb_usd_series_key", "value": ECB_USD_SERIES},
            {"field": "ecb_cny_series_key", "value": ECB_CNY_SERIES},
            {"field": "ecb_usd_source_url", "value": ECB_USD_URL},
            {"field": "ecb_cny_source_url", "value": ECB_CNY_URL},
            {
                "field": "interpretation",
                "value": (
                    "USD model-equivalent columns reconstruct the historical "
                    "7.2 CNY/USD model path; they are not raw USD observations "
                    "for CNY-anchored technologies."
                ),
            },
        ]
    )


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without the optional tabulate package."""
    headers = [str(column) for column in frame.columns]
    rows = [
        [str(value).replace("|", r"\|").replace("\n", " ") for value in row]
        for row in frame.itertuples(index=False, name=None)
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_note(summary: pd.DataFrame) -> None:
    display = summary.copy()
    display["技术"] = (
        display["technology_cn"].astype(str)
        + "（"
        + display["technology_en"].astype(str)
        + "）"
    )
    display["时长 h"] = display["duration_h"].astype(int)
    basis_labels = {
        "CNY_direct_or_hybrid_China_anchor": "中国价格锚点，正式采用 CNY→EUR",
        "CNY_direct_calibrated_to_China_EPC": "中国 EPC 锚点，正式采用 CNY→EUR",
        "USD_components_with_author_learning_path": "美元组件/学习路径，正式采用 USD→EUR",
        "USD_model_proxy_not_raw_technology_quote": "美元等值技术代理，正式采用 USD→EUR",
        "USD_model_proxy_from_US_evidence": "美国证据校准的美元等值代理，正式采用 USD→EUR",
        "USD_model_proxy_from_US_thermal_evidence": "美国储热证据校准的美元等值代理，正式采用 USD→EUR",
        "USD_model_proxy_from_US_hydrogen_evidence": "美国储氢证据校准的美元等值代理，正式采用 USD→EUR",
    }
    display["来源口径"] = display["capex_currency_basis"].map(basis_labels)
    display["较低换算路径"] = "USD→EUR"
    display["相对低幅"] = "0.215%"
    for year in [2025, 2030, 2035]:
        display[f"{year} USD→EUR"] = display[
            f"{year}_total_capex_eur_via_usd_kwh"
        ].map(lambda value: f"{value:.2f}")
        display[f"{year} CNY→EUR"] = display[
            f"{year}_total_capex_eur_via_cny_kwh"
        ].map(lambda value: f"{value:.2f}")
    columns = [
        "技术",
        "时长 h",
        "来源口径",
        "2025 USD→EUR",
        "2025 CNY→EUR",
        "2030 USD→EUR",
        "2030 CNY→EUR",
        "2035 USD→EUR",
        "2035 CNY→EUR",
        "较低换算路径",
        "相对低幅",
    ]
    table = dataframe_to_markdown(display[columns])
    route_pct = (
        ECB_2025_CNY_PER_EUR
        / (MODEL_CNY_PER_USD * ECB_2025_USD_PER_EUR)
        - 1.0
    ) * 100.0
    text = f"""# M2 LDES 美元直换欧元与人民币换算欧元对比 v0.2

日期：2026-07-27

## 汇率与结论

- 旧模型换算：1 USD = {MODEL_CNY_PER_USD:.4f} CNY。
- ECB 2025 年平均：1 EUR = {ECB_2025_USD_PER_EUR:.4f} USD。
- ECB 2025 年平均：1 EUR = {ECB_2025_CNY_PER_EUR:.10f} CNY。
- 对由 USD×7.2 形成的同一模型价格，USD 直换 EUR 比 CNY 转 EUR
  低 {abs(route_pct):.4f}%。
- 两条路径只有在 CNY/EUR 等于 7.2×USD/EUR 时才完全相同。

## 来源口径

LFP 和 VRFB 的绝对成本已锚定中国价格，因此其 USD 数值仅为换算
路径比较所用的模型美元等值，不是原始美国报价。两类技术的 CAPEX、
固定运维、可变运维和年固定成本统一采用 CNY→EUR。CAES 使用 ESGC
美元组件与作者学习路径；Fe–Cr、LAES、重力、熔盐和盐穴储氢保留
美元等值代理结构，这六类技术的全部技术成本组件统一采用 USD→EUR，
但不应表述为每个模型时长都有一条美国数据库直接报价。

## 2025、2030 与 2035 年逐技术—时长比较

单位均为 EUR kWh⁻¹。USD→EUR 列使用模型美元等值除以 1.1300；
CNY→EUR 列使用当前 CNY 值除以 8.1185043137255。

{table}
"""
    OUTPUT_NOTE.write_text(text, encoding="utf-8")


def validate(
    source: pd.DataFrame,
    frame: pd.DataFrame,
    comparison: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    if len(frame) != len(source) or len(comparison) != len(source):
        raise AssertionError("Currency extension changed the row count")
    if frame.duplicated(["tech_id", "duration_h", "learning_year"]).any():
        raise AssertionError("Duplicate technology-duration-year key")
    if len(summary) != 31:
        raise AssertionError(f"Expected 31 anchor-summary rows; found {len(summary)}")
    for cny_name in MONEY_COLUMNS:
        usd_name = usd_column(cny_name)
        via_usd_name = eur_via_usd_column(cny_name)
        via_cny_name = eur_via_cny_column(cny_name)
        if not np.allclose(
            frame[usd_name],
            frame[cny_name] / MODEL_CNY_PER_USD,
            rtol=0.0,
            atol=1e-10,
        ):
            raise AssertionError(f"USD model-equivalent check failed for {cny_name}")
        if not np.allclose(
            frame[via_usd_name],
            frame[usd_name] / ECB_2025_USD_PER_EUR,
            rtol=0.0,
            atol=1e-10,
        ):
            raise AssertionError(f"USD→EUR check failed for {cny_name}")
        if not np.allclose(
            frame[via_cny_name],
            frame[cny_name] / ECB_2025_CNY_PER_EUR,
            rtol=0.0,
            atol=1e-10,
        ):
            raise AssertionError(f"CNY→EUR check failed for {cny_name}")
        report_name = reporting_eur_column(cny_name)
        expected = np.where(
            frame["technology_currency_route"].eq("USD_to_EUR"),
            frame[via_usd_name],
            frame[via_cny_name],
        )
        if not np.allclose(
            frame[report_name],
            expected,
            rtol=0.0,
            atol=1e-10,
        ):
            raise AssertionError(
                f"Source-aware reporting route failed for {cny_name}"
            )
    for route_column in [
        "capex_reporting_route",
        "fixed_om_reporting_route",
        "variable_om_reporting_route",
        "fixed_cost_reporting_route",
    ]:
        if not frame[route_column].equals(
            frame["technology_currency_route"]
        ):
            raise AssertionError(
                f"Inconsistent technology route in {route_column}"
            )
    if not comparison["mechanically_lower_route"].eq("USD_to_EUR").all():
        raise AssertionError("Unexpected route ordering")


def main() -> None:
    source = pd.read_csv(INPUT)
    frame = add_currency_routes(source)
    comparison = build_comparison(frame)
    summary = build_anchor_summary(comparison)
    metadata = build_metadata()
    validate(source, frame, comparison, summary)

    cost_prefixes = (
        "tech_id",
        "technology_en",
        "duration_h",
        "learning_year",
        "capex_currency_basis",
        "technology_currency_route",
        "capex_reporting_route",
        "usd_value_status",
        "reporting_currency",
    )
    cost_columns = list(cost_prefixes) + [
        column
        for column in frame.columns
        if any(
            token in column
            for token in (
                "power_cost_",
                "energy_cost_",
                "total_capex_",
                "annualized_capex_",
                "fixed_om_",
                "variable_om_",
                "fixed_cost_",
                "reporting_fx_",
                "model_fx_",
            )
        )
        and column not in cost_prefixes
    ]
    cost = frame[cost_columns].copy()

    frame.to_csv(OUTPUT_LIBRARY, index=False, encoding="utf-8-sig")
    cost.to_csv(OUTPUT_COST, index=False, encoding="utf-8-sig")
    comparison.to_csv(OUTPUT_COMPARISON, index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="parameter_library", index=False)
        cost.to_excel(writer, sheet_name="cost_paths", index=False)
        comparison.to_excel(writer, sheet_name="currency_route_comparison", index=False)
        summary.to_excel(writer, sheet_name="anchor_year_summary", index=False)
        metadata.to_excel(writer, sheet_name="fx_metadata", index=False)
    write_note(summary)
    print(f"[M2-currency-v0.2] rows={len(frame)} -> {OUTPUT_LIBRARY}")
    print(f"[M2-currency-v0.2] comparison={len(comparison)} -> {OUTPUT_COMPARISON}")
    print(f"[M2-currency-v0.2] anchor rows={len(summary)} -> {OUTPUT_SUMMARY}")
    print(f"[M2-currency-v0.2] workbook={OUTPUT_XLSX}")
    print(f"[M2-currency-v0.2] note={OUTPUT_NOTE}")


if __name__ == "__main__":
    main()
