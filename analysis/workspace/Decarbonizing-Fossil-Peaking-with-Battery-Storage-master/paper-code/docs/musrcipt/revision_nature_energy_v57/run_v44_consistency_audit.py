#!/usr/bin/env python3
"""Final manuscript, figure and numerical consistency audit for v44."""

from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
MAIN = HERE / "musript_new_v44.tex"
SI = HERE / "musript_new_v44_supplementary.tex"
REPORT = HERE / "V44_FINAL_CONSISTENCY_AUDIT.md"
MACHINE = HERE / "V44_FINAL_CONSISTENCY_AUDIT.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_csv(relative: str) -> pd.DataFrame:
    path = HERE / relative
    require(path.exists(), f"Missing source table: {relative}")
    return pd.read_csv(path)


def main() -> None:
    main_tex = MAIN.read_text(encoding="utf-8")
    si_tex = SI.read_text(encoding="utf-8")
    both = main_tex + "\n" + si_tex
    checks: list[str] = []

    for token in [
        "fig_v42",
        "references_v43",
        "Reproducibility assets for v43",
        "fig_v44_figure6_procurement_boundaries",
        "pending final 132-country-year quality assurance",
        "26.499--109.68",
        "15.6946--109.68",
        "reserved for the final comparison",
        "pending final qa",
        "xxx",
        "author_input_needed",
    ]:
        require(token.lower() not in both.lower(), f"Stale manuscript token: {token}")
    require("：" not in main_tex, "Chinese colon remains in the main manuscript")
    require("“" not in main_tex and "”" not in main_tex, "Chinese quotation marks remain")
    require("\\section*{结论}" not in main_tex, "Separate Conclusions section remains")
    require(main_tex.count("575") == 0 and main_tex.count("m_C=0.7") == 0,
            "Range-out structural point leaked into the main manuscript")
    checks.append("No stale-version, punctuation, separate-conclusion or range-out-main-text token")

    result_match = re.search(r"\\section\*\{结果\}(.*?)\\section\*\{讨论\}", main_tex, re.S)
    require(result_match is not None, "Cannot locate Results section")
    result_blocks = re.split(r"\\subsection\*\{[^}]+\}", result_match.group(1))[1:]
    require(len(result_blocks) == 4, f"Expected four Results subsections, found {len(result_blocks)}")
    paragraph_counts = []
    for block in result_blocks:
        prose = re.split(r"\\begin\{figure\}", block, maxsplit=1)[0]
        paragraphs = [p for p in re.split(r"\n\s*\n", prose.strip())
                      if p.strip() and not p.lstrip().startswith("\\")]
        paragraph_counts.append(len(paragraphs))
    require(paragraph_counts == [3, 3, 3, 3], f"Results paragraph contract: {paragraph_counts}")
    discussion = re.search(r"\\section\*\{讨论\}(.*?)\\section\*\{方法\}", main_tex, re.S)
    require(discussion is not None, "Cannot locate Discussion section")
    discussion_paragraphs = [p for p in re.split(r"\n\s*\n", discussion.group(1).strip()) if p.strip()]
    require(len(discussion_paragraphs) == 4, "Discussion must contain three interpretation paragraphs and one limitation paragraph")
    require(discussion_paragraphs[-1].startswith("本文的国家级单节点筛查"), "The sole limitations paragraph is not last")
    checks.append("Results contain four three-paragraph arguments; limitations occupy one final Discussion paragraph")

    graphics = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{\\detokenize\{([^}]+)\}\}", both)
    require(len(graphics) == 16, f"Expected 16 included figures, found {len(graphics)}")
    for relative in graphics:
        graphic = HERE / relative
        require(graphic.exists(), f"Missing included figure: {relative}")
        require(graphic.stat().st_size > 10_000, f"Empty or truncated included figure: {relative}")
    for suffix in ["pdf", "svg", "png", "tiff"]:
        require((HERE / "figure_work/main_figures" / f"fig_v44_figure6_decision_consequences.{suffix}").exists(),
                f"Missing Figure 6 {suffix}")
    ed_numbers = sorted(map(int, re.findall(r"Extended Data Figure (\d+)", si_tex)))
    table_numbers = sorted(map(int, re.findall(r"Supplementary Table (\d+)", si_tex)))
    require(ed_numbers == list(range(1, 11)), f"Extended Data numbering: {ed_numbers}")
    require(table_numbers == list(range(1, 12)), f"Supplementary Table numbering: {table_numbers}")
    checks.append("All 6 main and 10 Extended Data figures exist; Extended Data 1–10 and Tables 1–11 are consecutive")

    bounded_root = HERE / "joint_choice" / "bounded_s20_highs"
    bounded_qa_path = bounded_root / "BOUNDED_ALL_YEAR_FINAL_QA.json"
    require(bounded_qa_path.exists(), "Missing bounded joint-choice final QA")
    bounded_qa = json.loads(bounded_qa_path.read_text(encoding="utf-8"))
    require(bounded_qa.get("status") == "pass"
            and bounded_qa.get("all_year_QA_passed") is True
            and bounded_qa.get("all_contract_checks_pass") is True,
            "Bounded joint-choice final QA did not pass")
    require(bounded_qa.get("total_country_year_pairs") == 396,
            "Bounded joint-choice total country-year pairs changed")
    primary = bounded_qa["residual_peak_system_cost_flip_counts"]
    secondary = bounded_qa["efc_equivalent_trajectory_screen_flip_counts"]
    require([primary[str(y)]["identity_flip"] for y in (2025, 2030, 2035)] == [2, 2, 3],
            "Primary system-cost joint-choice identity flips changed")
    require([primary[str(y)]["participation_flip"] for y in (2025, 2030, 2035)] == [0, 0, 0],
            "Primary system-cost participation flips changed")
    require([secondary[str(y)]["identity_flip"] for y in (2025, 2030, 2035)] == [0, 3, 22],
            "Secondary EFC trajectory-screen identity flips changed")
    require(bounded_qa.get("secondary_account_role") ==
            "revaluation_of_residual_peak_system_cost_optimized_trajectories__not_efc_objective_optimization",
            "Secondary account role changed")
    require(bounded_qa["maximum_solver_or_conservation_residual"] < 1e-6,
            "Bounded physical/conservation residual exceeds tolerance")
    require(bounded_qa["maximum_accounting_residual_eur"] < 1e-5,
            "Bounded accounting residual exceeds tolerance")
    ed6_paths = {}
    for suffix in ["pdf", "svg", "png", "tiff"]:
        ed6 = HERE / "figure_work" / "extended_data" / f"fig_v44_ed6_bounded_joint_procurement.{suffix}"
        require(ed6.exists() and ed6.stat().st_size > 10_000,
                f"Missing or truncated Extended Data Figure 6 {suffix}")
        ed6_paths[suffix] = ed6
    ed6_manifest_path = HERE / "figure_work" / "manifests" / "fig_v44_ed6_bounded_joint_procurement_manifest.json"
    require(ed6_manifest_path.exists(), "Missing Extended Data Figure 6 manifest")
    ed6_manifest = json.loads(ed6_manifest_path.read_text(encoding="utf-8"))
    for suffix, path in ed6_paths.items():
        observed_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_sha = ed6_manifest["outputs"][path.name]["sha256"]
        require(observed_sha == manifest_sha, f"Extended Data Figure 6 {suffix} hash differs from manifest")
    require("轨迹重估" in both, "The EFC trajectory screen is not identified as a revaluation")
    require("EFC 目标" in both or "EFC目标" in both,
            "The manuscript does not distinguish the EFC trajectory screen from EFC-objective optimization")
    checks.append("Bounded power-duration joint procurement passes all-year QA; primary 2/2/3 flips and secondary 0/3/22 trajectory-screen flips retain their distinct estimands")

    refs = (HERE / "references_v44.tex").read_text(encoding="utf-8")
    ref_count = len(re.findall(r"\\item\s", refs))
    require(ref_count == 62, f"Expected 62 references, found {ref_count}")
    require("\\input{references_v44.tex}" in main_tex and "\\input{references_v44.tex}" in si_tex,
            "Main and SI do not share references_v44")
    checks.append("Main and Supplementary share a 62-item v44 reference list")

    flips = read_csv("figure_work/source_data/figure6_panel_a_robust_flip_summary.csv")
    require(flips["disjoint_tie_set_flips"].tolist() == [0, 3, 22], "Figure 6a flip counts changed")
    row35 = flips.loc[flips["cost_year"].eq(2035)].iloc[0]
    require(np.isclose(row35["disjoint_tie_set_flip_rate"], 22 / 132), "2035 flip rate changed")
    require(np.isclose(row35["country_block_bootstrap_ci95_low"], 0.0601503759398496)
            and np.isclose(row35["country_block_bootstrap_ci95_high"], 0.2857142857142857),
            "2035 flip interval changed")
    movement = read_csv("figure_work/source_data/figure6_panel_b_value_shift_cost_path_summary.csv").iloc[0]
    require(int(movement["clustering_effect_at_least_cost_path_span_count"]) == 60,
            "Cost-path comparison count changed")
    capacity = read_csv("figure_work/source_data/figure6_panel_c_capacity_market_anchors.csv")
    priority = capacity.loc[capacity["anchor_scope"].eq("four_priority_markets")]
    contextual = capacity.loc[capacity["anchor_scope"].eq("contextual_SEM_high")]
    require(np.isclose(priority["eur_per_kw_year"].min(), 15.6946)
            and np.isclose(priority["eur_per_kw_year"].max(), 90.4316),
            "Four-priority-market capacity-price envelope changed")
    require(len(contextual) == 1
            and np.isclose(contextual["eur_per_kw_year"].iloc[0], 160.577),
            "Contextual all-island SEM high anchor changed")
    capacity_curve = read_csv("figure_work/source_data/figure6_panel_c_capacity_anchor_flip_curve.csv")
    band_2035 = capacity_curve.loc[
        capacity_curve["cost_year"].eq(2035)
        & capacity_curve["within_official_capacity_price_band"].astype(bool)
    ]
    require(int(band_2035["selected_class_flip_count"].min()) == 0
            and int(band_2035["selected_class_flip_count"].max()) == 15,
            "2035 decision-flip envelope within observed capacity prices changed")
    natural = read_csv("figure_work/source_data/figure6_panel_d_natural_fixed_effect_coefficient.csv").iloc[0]
    require(np.isclose(natural["coefficient_per_10pp_eur_kw_year"], -1.1685890504445635),
            "Natural fixed-effect coefficient changed")
    boundary = read_csv("figure_work/source_data/figure6_panel_e_2035_tail_decision_boundary.csv")
    require(int(boundary["robust_tail_exit"].sum()) == 22, "Figure 6e exit count changed")
    checks.append("Figure 6 source data reproduce 0/3/22 flips, the 15.6946–90.4316 priority-market band, its 0–15 flip envelope, the 160.577 contextual SEM high, 60/132 cost-scale cases and the natural-year slope")

    catalogue = read_csv("source_data/v43_cost_catalogue_2025_2035_source_aware_eur.csv")
    require(catalogue.groupby("learning_year")["config_id"].nunique().to_dict() == {2025: 53, 2030: 53, 2035: 53},
            "Cost catalogue does not contain 53 configurations in every year")
    valleys = read_csv("source_data/v43_year_specific_valley_summary.csv")
    central = valleys[valleys["revaluation_case_id"].eq("central_actual_year_cost")]
    require(len(central) == 6 and central["strict_valley_count"].eq(0).all(),
            "Central calendar-year strict-valley result changed")
    structural = read_csv("figure_work/source_data/ed5_conditional_peak_boundaries.csv")
    require(len(structural) == 92
            and np.isclose(structural["left_boundary_h"].median(), 24)
            and np.isclose(structural["right_boundary_h"].median(), 44)
            and np.isclose(structural["valley_width_h"].median(), 20),
            "Range-out structural-valley geometry changed")
    checks.append("The three cost years retain 53 configurations and no strict central valley; the SI-only structural point retains the audited 92 cases")

    for pdf in [HERE / "musript_new_v44.pdf", HERE / "musript_new_v44_supplementary.pdf",
                HERE / "musript_new_v44_main_pdfunite.pdf"]:
        require(pdf.exists() and pdf.stat().st_size > 100_000, f"Missing or empty compiled PDF: {pdf.name}")
    for log in [HERE / "musript_new_v44.log", HERE / "musript_new_v44_supplementary.log"]:
        text = log.read_text(encoding="utf-8", errors="ignore")
        require("! LaTeX Error" not in text
                and "Overfull \\hbox" not in text
                and "Overfull \\vbox" not in text,
                f"Compilation/layout error in {log.name}")
    checks.append("Both PDFs compile without LaTeX errors or overfull boxes")

    payload = {"status": "PASS", "checks": checks, "results_paragraph_counts": paragraph_counts,
               "references": ref_count, "main_figures": 6, "extended_data_figures": 10,
               "supplementary_tables": 11}
    MACHINE.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    REPORT.write_text("# V44 final consistency audit\n\nStatus: **PASS**\n\n## Passed gates\n\n" +
                      "\n".join(f"- {item}." for item in checks) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
