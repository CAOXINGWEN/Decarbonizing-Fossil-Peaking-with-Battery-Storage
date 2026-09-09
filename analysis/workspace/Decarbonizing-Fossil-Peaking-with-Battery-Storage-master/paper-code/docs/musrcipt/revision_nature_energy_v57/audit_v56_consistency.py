#!/usr/bin/env python3
"""Fail-fast consistency audit for the PECD v4.2 v56 submission package."""

from __future__ import annotations

import argparse
import gzip
import json
import re
import subprocess
from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parent
WEATHER = ROOT / "weather_year_parity_v42"
PROCESSED = WEATHER / "processed"
RESULTS = WEATHER / "results"
FIGURES = ROOT / "figure_work"

TEX_FILES = [ROOT / "musript_new_v56.tex", ROOT / "musript_new_v56_supplementary.tex"]
STALE_TEXT = (
    "PECD 2021.3",
    "1982--2016 median",
    "9.59 & 8.07",
    "73/561",
    " & 1/24\\\\",
    "8,925 within-country",
    "653 pairs",
    "15/15 国",
    "11/15 国",
    "9/375",
    "fig_v55_figure1_natural",
    "fig_v55_figure2_natural",
    "fig_v55_ed12_country_natural",
)

QA_FILES = (
    PROCESSED / "demand_model_qa.json",
    PROCESSED / "pecd_v42_europe33_hourly_qa.json",
    RESULTS / "pecd_v42_75_window_preparation_qa.json",
    RESULTS / "pecd_v42_75_window_optimization_qa.json",
    RESULTS / "pecd_v42_75y_parity_summary.json",
)

TABLE_FILES = {
    RESULTS / "pecd_v42_75_window_relative_values_2025.csv": 75,
    RESULTS / "pecd_v42_65_matched_decade_transitions.csv": 65,
    RESULTS / "pecd_v42_75_window_all_pairs.csv.gz": 2775,
}

FIGURE_STEMS = (
    FIGURES / "main_figures" / "fig_v56_figure1_natural_sequencing_framework",
    FIGURES / "main_figures" / "fig_v56_figure2_natural_weather_parity",
    FIGURES / "extended_data_figures" / "ed_fig_v56_demand_model_validation",
)

SOURCE_DATA = (
    "figure1_framework_metrics.csv",
    "figure2a_75_window_chronology.csv",
    "figure2b_cost_year_distributions.csv",
    "figure2c_worst_decade_daily_sequences.csv",
    "figure2d_65_matched_decades.csv",
    "figure2e_all_pair_matrix.csv",
    "figure2f_risk_accounting.csv",
    "figure2g_recovery_value_relation.csv",
    "figure2_source_data_manifest.txt",
    "extended_data_figure12a_country_holdout_fit.csv",
    "extended_data_figure12b_holdout_sample.csv.gz",
    "extended_data_figure12c_month_hour_surface.csv",
    "extended_data_figure12d_hourly_error.csv",
    "extended_data_figure12e_reconstructed_load_chronology.csv",
    "extended_data_figure12f_provenance.csv",
)


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def row_count(path: Path) -> int:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def audit_text(failures: list[str]) -> None:
    for path in TEX_FILES:
        require(path.exists(), f"missing manuscript file: {path.name}", failures)
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for stale in STALE_TEXT:
            require(stale not in text, f"{path.name} retains stale v55 text: {stale}", failures)
        require("weather_year_parity_v42/results/pecd_v42_v56_results_macros.tex" in text,
                f"{path.name} does not import generated v56 result macros", failures)
    main_text = TEX_FILES[0].read_text(encoding="utf-8") if TEX_FILES[0].exists() else ""
    supp_text = TEX_FILES[1].read_text(encoding="utf-8") if TEX_FILES[1].exists() else ""
    require("fig_v56_figure1_natural_sequencing_framework.pdf" in main_text,
            "main manuscript does not reference v56 Figure 1", failures)
    require("fig_v56_figure2_natural_weather_parity.pdf" in main_text,
            "main manuscript does not reference v56 Figure 2", failures)
    require("ed_fig_v56_demand_model_validation.pdf" in supp_text,
            "supplement does not reference v56 demand validation figure", failures)


def audit_results(failures: list[str]) -> None:
    for path in QA_FILES:
        require(path.exists(), f"missing QA/result file: {path.relative_to(ROOT)}", failures)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            require(payload.get("status") == "pass", f"QA did not pass: {path.name}", failures)

    summary_path = RESULTS / "pecd_v42_75y_parity_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        require(summary.get("complete_july_june_operating_years") == 75,
                "summary does not contain 75 complete operating years", failures)
        require(summary.get("all_unordered_weather_pairs") == 2775,
                "summary does not contain 2,775 unordered pairs", failures)
        require(summary.get("matched_ten_year_transitions") == 65,
                "summary does not contain 65 matched decade transitions", failures)
        require(abs(float(summary.get("cost_improvement_2025_2035_eur_kw_year", -1)) - 5.296) < 0.002,
                "summary cost scale is not the 60-year-life 24 h A-CAES value", failures)

    for path, expected in TABLE_FILES.items():
        require(path.exists(), f"missing result table: {path.name}", failures)
        if path.exists():
            require(row_count(path) == expected,
                    f"{path.name} has {row_count(path)} rows; expected {expected}", failures)

    coverage_path = PROCESSED / "pecd_v42_zone_weight_matching_qa.csv"
    require(coverage_path.exists(), "missing PECD-zone structural-weight audit", failures)
    if coverage_path.exists():
        coverage = pd.read_csv(coverage_path)
        require(len(coverage) == 5, "zone audit does not contain all five PECD products", failures)
        require(coverage.matched_weight_share.ge(0.95).all(),
                "one or more PECD products retain less than 95% of structural weight after allocation", failures)
        require(coverage.minimum_positive_country_weight_share.ge(0.95).all(),
                "one or more countries retain less than 95% of structural weight after allocation", failures)

    failure_path = PROCESSED / "pecd_v42_zone_weight_matching_failures.csv"
    require(failure_path.exists(), "missing PECD-zone allocation failure ledger", failures)
    if failure_path.exists():
        require(row_count(failure_path) == 0,
                "PECD-zone allocation failure ledger is not empty", failures)

    inventory_path = PROCESSED / "pecd_v42_zone_inventory.csv"
    require(inventory_path.exists(), "missing PECD-zone allocation inventory", failures)
    if inventory_path.exists():
        inventory = pd.read_csv(
            inventory_path,
            usecols=["aggregation_method", "country_total_weight_mw"],
        )
        require(not inventory.empty, "PECD-zone allocation inventory is empty", failures)
        unweighted = inventory.aggregation_method.str.startswith("unweighted")
        require(inventory.loc[unweighted, "country_total_weight_mw"].eq(0.0).all(),
                "PECD-zone allocation used an unweighted fallback for positive structural weight", failures)

    macro_path = RESULTS / "pecd_v42_v56_results_macros.tex"
    require(macro_path.exists(), "missing generated LaTeX result macros", failures)
    if macro_path.exists():
        macro_text = macro_path.read_text(encoding="utf-8")
        require("PENDING" not in macro_text, "generated result macros contain PENDING", failures)
        for name, value in (("NatWindows", "75"), ("NatPairs", "2,775"), ("NatDecades", "65")):
            pattern = rf"\\newcommand\{{\\{name}\}}\{{{re.escape(value)}\}}"
            require(re.search(pattern, macro_text) is not None,
                    f"generated macro {name} does not equal {value}", failures)


def audit_figures(failures: list[str]) -> None:
    for stem in FIGURE_STEMS:
        for suffix in (".pdf", ".svg", ".tiff", ".png"):
            path = stem.with_suffix(suffix)
            require(path.exists() and path.stat().st_size > 0,
                    f"missing or empty figure export: {path.relative_to(ROOT)}", failures)
        grey = stem.with_name(stem.name + "_grayscale").with_suffix(".png")
        require(grey.exists() and grey.stat().st_size > 0,
                f"missing grayscale proof: {grey.relative_to(ROOT)}", failures)
        png = stem.with_suffix(".png")
        if png.exists():
            width, height = Image.open(png).size
            require(width >= 2000 and height >= 1000,
                    f"{png.name} raster proof is only {width} x {height} px", failures)
        pdf = stem.with_suffix(".pdf")
        if pdf.exists():
            completed = subprocess.run(
                ["pdfinfo", str(pdf)], check=False, capture_output=True, text=True
            )
            require(completed.returncode == 0 and re.search(r"^Pages:\s+1\s*$", completed.stdout, re.M),
                    f"{pdf.name} is not a readable one-page PDF", failures)

    source = FIGURES / "source_data_v56"
    for name in SOURCE_DATA:
        path = source / name
        require(path.exists() and path.stat().st_size > 0,
                f"missing figure source data: {name}", failures)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Audit manuscript wiring and stale text without requiring computed outputs.",
    )
    args = parser.parse_args()
    failures: list[str] = []
    audit_text(failures)
    if not args.preflight:
        audit_results(failures)
        audit_figures(failures)
    if failures:
        print("V56 AUDIT: FAIL")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("V56 AUDIT: PASS" + (" (preflight)" if args.preflight else ""))


if __name__ == "__main__":
    main()
