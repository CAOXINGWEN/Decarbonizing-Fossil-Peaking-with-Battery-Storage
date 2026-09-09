#!/usr/bin/env python3
"""Deterministic text–figure–source consistency gate for v51."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
MAIN = ROOT / "musript_new_v51.tex"
SUPP = ROOT / "musript_new_v51_supplementary.tex"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    main_text = MAIN.read_text(encoding="utf-8")
    supp_text = SUPP.read_text(encoding="utf-8")
    both = main_text + "\n" + supp_text
    checks: list[dict[str, object]] = []

    def check(name: str, condition: bool, evidence: object = "") -> None:
        checks.append({"check": name, "pass": bool(condition), "evidence": str(evidence)})

    title = "稀缺调峰事件聚集重塑储能的跨时长价值与配置选择"
    check("title_main", title in main_text)
    check("title_supplementary", title in supp_text)
    check("reference_file_main", "references_v51.tex" in main_text and "references_v50.tex" not in main_text)
    check("supplementary_has_no_redundant_reference_list",
          "references_v51.tex" not in supp_text and "references_v50.tex" not in supp_text)

    intro = main_text.split("\\section*{引言}", 1)[1].split("\\begin{figure}", 1)[0]
    definition = "本文将其中的负收益区定义为中时长谷"
    check("mid_duration_valley_prepared_before_naming", definition in intro)
    check("sequencing_parity_follows_valley_mechanism", intro.find(definition) < intro.find("时序平价假说"))

    results = main_text.split("\\section*{结果}", 1)[1].split("\\section*{讨论}", 1)[0]
    headings = re.findall(r"\\subsection\*\{([^}]+)\}", results)
    expected_headings = [
        "三种平衡体制分离年度调峰规模与事件时序",
        "恢复窗口将事件时序转化为跨时长储能价值",
        "恢复时间而非补能总量控制事件间库存重建",
        "事件时序进入技术学习与配置选择的同一价值账本",
        "稳健性与系统尺度筛查限定结果外推",
    ]
    check("five_part_result_chain", headings == expected_headings, headings)
    result_blocks = re.split(r"\\subsection\*\{[^}]+\}", results)[1:]
    paragraph_counts = []
    for block in result_blocks:
        prose = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", "", block, flags=re.S)
        prose = re.sub(r"\\begin\{table\}.*?\\end\{table\}", "", prose, flags=re.S)
        paragraphs = [p for p in re.split(r"\n\s*\n", prose.strip()) if p and not p.lstrip().startswith("\\")]
        paragraph_counts.append(len(paragraphs))
    check("three_paragraph_results", paragraph_counts == [3, 3, 3, 3, 3], paragraph_counts)
    for token in ["条件性", "不构成第二套", "尚未内生", "不用于直接比较"]:
        check(f"no_defensive_results_{token}", token not in results)

    graphics = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{\\detokenize\{([^}]+)\}\}", both)
    missing = [path for path in graphics if not (ROOT / path).exists()]
    check("all_included_graphics_exist", not missing, missing)
    main_stems = [
        "fig_v51_figure1_sequencing_parity_framework",
        "fig_v51_figure2_three_regime_event_atlas",
        "fig_v51_figure3_portfolio_cycling_ledger",
        "fig_v51_figure4_recovery_interface_tests",
        "fig_v51_figure5_cross_duration_reversal",
        "fig_v51_figure6_sequencing_parity_choices",
    ]
    for stem in main_stems:
        check(f"main_uses_{stem}", f"{stem}.pdf" in main_text)
        for suffix in ["pdf", "svg", "png", "tiff", "_grayscale.png"]:
            name = f"{stem}{suffix}" if suffix.startswith("_") else f"{stem}.{suffix}"
            path = ROOT / "figure_work/main_figures" / name
            check(f"asset_{name}", path.exists() and path.stat().st_size > 10_000,
                  path.stat().st_size if path.exists() else 0)
    check("no_stale_main_figures", "fig_v50_figure" not in main_text)
    ed_numbers = sorted(set(map(int, re.findall(r"Extended Data Figure (\d+)", supp_text))))
    table_numbers = sorted(set(map(int, re.findall(r"Supplementary Table (\d+)", supp_text))))
    check("extended_data_1_to_10", ed_numbers == list(range(1, 11)), ed_numbers)
    check("supplementary_tables_1_to_5", table_numbers == list(range(1, 6)), table_numbers)
    ed_expected = [
        "fig_v51_ed1_cost_paths_53_eur",
        "fig_v51_ed2_event_definition_sensitivity",
        "fig_v51_ed3_numerical_qa",
        "fig_v51_ed4_full_technology_duration_response",
        "fig_v51_ed5_conditional_phase_details",
        "fig_v51_ed6_bounded_joint_procurement",
        "fig_v51_ed7_fixed_scale_mix_details",
        "fig_v51_ed8_mix_sensitivity",
        "fig_v51_ed9_mix_heterogeneity",
        "fig_v51_ed10_eraa_system_screen",
    ]
    ed_graphics = [path for path in graphics if "figure_work/extended_data/" in path]
    check("supplement_uses_10_v51_extended_data_pdfs",
          sorted(Path(path).stem for path in ed_graphics) == sorted(ed_expected),
          ed_graphics)
    stale_ed_tokens = ["fig_v42_ed", "fig_v43_ed", "fig_v44_ed", "fig_v50_ed", "fixed_bounded_tail_power"]
    check("no_stale_extended_data_references",
          not any(token in supp_text for token in stale_ed_tokens),
          [token for token in stale_ed_tokens if token in supp_text])
    for stem in ed_expected:
        for suffix, minimum_size in [("pdf", 10_000), ("png", 50_000)]:
            path = ROOT / "figure_work/extended_data" / f"{stem}.{suffix}"
            check(f"supp_ed_asset_{stem}.{suffix}", path.exists() and path.stat().st_size > minimum_size,
                  path.stat().st_size if path.exists() else 0)
    contact_sheet = ROOT / "figure_work/extended_data/V51_REFERENCE_STYLE_CONTACT_SHEET.png"
    check("supp_extended_data_contact_sheet", contact_sheet.exists() and contact_sheet.stat().st_size > 100_000,
          contact_sheet.stat().st_size if contact_sheet.exists() else 0)
    supp_greyscale = ROOT / "figure_work/extended_data/V51_REFERENCE_STYLE_GRAYSCALE_CONTACT_SHEET.png"
    check("supp_extended_data_grayscale_contact_sheet", supp_greyscale.exists() and supp_greyscale.stat().st_size > 100_000,
          supp_greyscale.stat().st_size if supp_greyscale.exists() else 0)
    supp_page_qa = ROOT / "qa_render_v51_reference/supp_contact.png"
    check("supp_extended_data_pdf_page_contact_sheet", supp_page_qa.exists() and supp_page_qa.stat().st_size > 100_000,
          supp_page_qa.stat().st_size if supp_page_qa.exists() else 0)
    main_page_qa = ROOT / "qa_render_v51_reference/main_contact.png"
    check("main_pdf_figure2_to_6_page_contact_sheet", main_page_qa.exists() and main_page_qa.stat().st_size > 100_000,
          main_page_qa.stat().st_size if main_page_qa.exists() else 0)
    check("updated_caption_figure2", "Aligned maps" in main_text and "Two-level distribution summaries" in main_text)
    check("updated_caption_figure3", "Violin densities and raw country--year values" in main_text and "Fixed tail shares" in main_text)
    check("updated_caption_figure4", "Aligned dispersed and clustered daily task" in main_text and "Country classification" in main_text)
    check("updated_caption_figure5", "Three-regime matrix" in main_text and "Raw observed crossing-duration distributions" in main_text)
    check("updated_caption_figure6", "Purchase composition" in main_text and "Tail-entry or exit flip rate" in main_text)
    check("main_panel_limit", all(token in main_text for token in ["\\textbf{g}, Conserved C1-task",
                                                                   "\\textbf{h}, Median retention",
                                                                   "\\textbf{h}, Country fixed-effect",
                                                                   "\\textbf{h}, Natural longest-event",
                                                                   "\\textbf{h}, Tail-entry or exit"]), "Fig. 2: 7 panels; Figs. 3-6: 8 panels")
    check("supplementary_is_evidence_only", "Supplementary Note" not in supp_text and
          "Supplementary Tables and Extended Data Figures" in supp_text)
    check("reference_style_generator_registered",
          "plot_v51_reference_style_figures" in (ROOT / "figure_work/plot_v51_main_figures.py").read_text(encoding="utf-8"))
    check("updated_supplementary_captions",
          all(token in supp_text for token in ["Empirical cumulative distributions", "Technology-by-duration heat map",
                                               "Dispersed-to-clustered selected-duration transitions",
                                               "Cluster-induced strict-valley incidence"]))
    check("supp_table1_sample_arithmetic",
          "132 & 1,156,320 country-hours" in supp_text and 132 * 8760 == 1_156_320)
    contract = pd.read_csv(ROOT / "figure_work/source_data_v49/figure2e_three_scale_data_contract.csv").set_index("scale")
    check("supp_table2_cross_scale_units_match_source",
          contract.loc["Europe", ["independent_blocks", "case_years", "hours"]].astype(int).tolist() == [15, 132, 1_156_320] and
          contract.loc["China", ["independent_blocks", "case_years", "hours"]].astype(int).tolist() == [26, 26, 227_760] and
          contract.loc["Japan", ["independent_blocks", "case_years", "hours"]].astype(int).tolist() == [5, 27, 236_520] and
          all(token in supp_text for token in ["132 国家--年份", "26 省级年份", "27 社区--年份"]))
    catalogue = pd.read_csv(ROOT / "cost_and_decomposition/cost_catalogue_2025_2035_long.csv")
    catalogue_2025 = catalogue[catalogue.learning_year == 2025]
    config_counts = catalogue_2025.groupby("tech_id").duration_h.nunique().to_dict()
    expected_config_counts = {"li_lfp": 4, "vrfb": 3, "fecr_flow": 3, "gravity": 3,
                              "molten_salt_tes": 5, "caes": 9, "laes": 9, "h2_cavern": 17}
    check("supp_table3_configuration_counts_match_catalogue",
          config_counts == expected_config_counts and sum(config_counts.values()) == 53, config_counts)
    physical = catalogue_2025.groupby("tech_id").agg(
        duration_min=("duration_h", "min"), duration_max=("duration_h", "max"),
        rte=("rte", "first"), charge_ratio=("charge_power_ratio", "first"),
        self_discharge=("self_discharge_pct_day", "first"), life=("calendar_life_years", "first"),
        cycles=("cycle_life_cycles", "first"))
    check("supp_table3_physical_parameters_match_catalogue",
          np.allclose(physical.loc["caes", ["duration_min", "duration_max", "rte", "charge_ratio", "self_discharge", "life", "cycles"]].astype(float),
                      [8, 24, .55, 1.5, .01, 50, 30000]) and
          np.allclose(physical.loc["h2_cavern", ["duration_min", "duration_max", "rte", "charge_ratio", "self_discharge", "life", "cycles"]].astype(float),
                      [24, 720, .37, 3.0, .001, 25, 15000]))

    europe = pd.read_csv(ROOT / "figure_work/source_data_v49/figure5a_europe_duration_response.csv")
    e = europe.set_index("duration_h").point_estimate
    check("europe_duration_response",
          np.allclose([e[24], e[32], e[40], e[96], e[168]],
                      [-5.2060209053, -7.6264979616, -7.5479750615, 1.2687884937, 6.2019003466]))
    cross = pd.read_csv(ROOT / "figure_work/source_data_v49/figure5bc_china_japan_duration_response.csv")
    def effect(case: str, duration: int) -> float:
        return float(cross[(cross.application_case == case) & (cross.duration_h == duration)].mean_effect_eur_kw_year.iloc[0])
    check("china_sign_reversal", effect("china", 24) < 0 and effect("china", 40) < 0 and effect("china", 240) > 0)
    check("japan_sign_reversal", effect("japan", 32) < 0 and effect("japan", 72) > 0)

    parity = pd.read_csv(ROOT / "figure_work/source_data/figure6_panel_b_value_shift_cost_path_ecdf.csv")
    check("parity_median_5_48", np.isclose(parity.absolute_cluster_effect_eur_kw_year.median(), 5.48, atol=0.02),
          parity.absolute_cluster_effect_eur_kw_year.median())
    check("parity_60_of_132", int(parity.cluster_effect_at_least_cost_path_span.sum()) == 60,
          int(parity.cluster_effect_at_least_cost_path_span.sum()))
    flips = pd.read_csv(ROOT / "figure_work/source_data/figure6_panel_a_robust_flip_summary.csv")
    check("tail_flip_counts", flips.disjoint_tie_set_flips.astype(int).tolist() == [0, 3, 22],
          flips.disjoint_tie_set_flips.tolist())
    joint_summary = pd.read_csv(ROOT / "joint_choice/bounded_s20_highs/bounded_joint_choice_summary_all_years.csv")
    def summary_counts(account: str, metric: str) -> list[int]:
        q = joint_summary[(joint_summary.account == account) & (joint_summary.metric == metric)].sort_values("learning_year")
        return q["count"].astype(int).tolist()
    check("main_table1_joint_choice_rows_match_source",
          summary_counts("residual_peak_system_cost", "identity_flip") == [2, 2, 3] and
          summary_counts("residual_peak_system_cost", "any_material_procurement_change") == [4, 3, 4] and
          summary_counts("efc_equivalent_trajectory_screen", "identity_flip") == [0, 3, 22])
    fixed_flips = pd.read_csv(ROOT / "joint_choice/joint_choice_flip_summary.csv")
    forced = fixed_flips[(fixed_flips.model_mode == "fixed") & (fixed_flips.account == "efc_value") &
                         (fixed_flips.decision_scope == "mandatory_tail")].sort_values("cost_year")
    catalogue_flips = pd.read_csv(ROOT / "decision_flip/decision_flip_year_summary.csv").sort_values("cost_year")
    check("main_table1_forced_tail_and_catalogue_rows_match_source",
          forced.disjoint_tie_set_flips.astype(int).tolist() == [0, 1, 1] and
          catalogue_flips.catalogue_duration_identity_flips.astype(int).tolist() == [13, 12, 8])
    market_anchors = pd.read_csv(ROOT / "figure_work/source_data/figure6_panel_c_capacity_market_anchors.csv")
    priority = market_anchors[market_anchors.within_priority_market_range].eur_per_kw_year
    check("main_table1_priority_market_range_matches_source",
          np.isclose(priority.min(), 15.6946) and np.isclose(priority.max(), 90.4316) and
          "15.7--90.4 EUR" in main_text)
    sensitivity = pd.read_csv(ROOT / "sequencing_parity/cost_duration_discount_sensitivity.csv")
    table4_expected = {
        (0.020, 8): 2.51, (0.020, 16): 2.95, (0.020, 24): 3.29,
        (0.036, 8): 4.96, (0.036, 16): 5.55, (0.036, 24): 6.03,
        (0.060, 8): 9.29, (0.060, 16): 10.15, (0.060, 24): 10.84,
        (0.080, 8): 13.26, (0.080, 16): 14.37, (0.080, 24): 15.27,
    }
    rounded = {
        (round(float(r.discount_rate), 3), int(r.caes_duration_h)):
        round(float(r.absolute_relative_cost_movement_eur_kw_year), 2)
        for r in sensitivity.itertuples()
    }
    check("supp_table4_matches_cost_source",
          all(np.isclose(rounded[k], v, atol=0.005) for k, v in table4_expected.items()),
          {str(k): rounded.get(k) for k in table4_expected})
    europe_valleys = pd.read_csv(ROOT / "figure_work/source_data/figure5_panel_e_conditional_valley_records.csv")
    clustered_valleys = europe_valleys[(europe_valleys.chronology == "clustered") & europe_valleys.strict_valley]
    check("figure5e_clustered_valley_records_85", len(clustered_valleys) == 85, len(clustered_valleys))
    price_table_terms = ["固定价格", "事件稀缺加价", "补能价格联动", "实际日前价格", "稀缺封顶"]
    check("supp_table5_price_scenarios_present",
          all(term in supp_text for term in price_table_terms),
          [term for term in price_table_terms if term not in supp_text])
    counts = pd.read_csv(ROOT / "figure_work/source_data_v49/figure5d_three_scale_strict_valley_counts.csv")
    selected = counts[((counts.scale == "Europe") & counts.selection_label.str.contains("E=98")) |
                      ((counts.scale == "China") & (counts.selection_label == "China clustered-valley phase")) |
                      ((counts.scale == "Japan") & (counts.selection_label == "Japan clustered-valley phase"))]
    actual = {(r.scale, r.chronology): int(r.strict_valleys) for r in selected.itertuples()}
    check("three_regime_valley_counts", actual == {("Europe", "clustered"): 85, ("Europe", "dispersed"): 0,
                                                    ("China", "clustered"): 26, ("China", "dispersed"): 0,
                                                    ("Japan", "clustered"): 8, ("Japan", "dispersed"): 0}, actual)

    references = (ROOT / "references_v51.tex").read_text(encoding="utf-8")
    check("reference_count_63", references.count("\\item ") == 63, references.count("\\item "))
    for name in ["musript_new_v51.pdf", "musript_new_v51_supplementary.pdf"]:
        path = ROOT / name
        check(f"compiled_{name}", path.exists() and path.stat().st_size > 100_000,
              path.stat().st_size if path.exists() else 0)
    for name in ["musript_new_v51.log", "musript_new_v51_supplementary.log"]:
        log = (ROOT / name).read_text(encoding="utf-8", errors="ignore")
        forbidden = ["! LaTeX Error", "Undefined control sequence", "Float too large", "Overfull \\hbox", "Overfull \\vbox"]
        check(f"clean_log_{name}", not any(token in log for token in forbidden))

    outputs = [ROOT / "musript_new_v51.pdf", ROOT / "musript_new_v51_supplementary.pdf", MAIN, SUPP,
               ROOT / "references_v51.tex", ROOT / "figure_work/plot_v51_main_figures.py",
               ROOT / "figure_work/plot_v51_reference_style_figures.py",
               ROOT / "figure_work/plot_v51_extended_data.py", ROOT / "V51_FIGURE_REDESIGN_CONTRACT.md",
               ROOT / "V51_FIGURE_REDESIGN_QA.md"]
    outputs += sorted((ROOT / "figure_work/main_figures").glob("fig_v51_*"))
    outputs += sorted((ROOT / "figure_work/extended_data").glob("fig_v51_*.pdf"))
    outputs += sorted((ROOT / "figure_work/extended_data").glob("fig_v51_*.png"))
    outputs += [contact_sheet, supp_greyscale, supp_page_qa,
                ROOT / "figure_work/main_figures/V51_REFERENCE_STYLE_MAIN_CONTACT_SHEET.png",
                ROOT / "figure_work/main_figures/V51_REFERENCE_STYLE_MAIN_GRAYSCALE_CONTACT_SHEET.png"]
    outputs.append(main_page_qa)
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in outputs if p.exists()}
    status = "PASS" if all(row["pass"] for row in checks) else "FAIL"
    payload = {"status": status, "checks": checks, "sha256": hashes}
    (ROOT / "V51_FINAL_CONSISTENCY_AUDIT.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n")
    lines = ["# V51 final consistency audit", "", f"Status: **{status}**", ""]
    for row in checks:
        lines.append(f"- {'PASS' if row['pass'] else 'FAIL'} — {row['check']}" +
                     (f": {row['evidence']}" if row['evidence'] else ""))
    (ROOT / "V51_FINAL_CONSISTENCY_AUDIT.md").write_text("\n".join(lines)+"\n")
    (ROOT / "V51_OUTPUT_SHA256.txt").write_text("\n".join(f"{v}  {k}" for k,v in hashes.items())+"\n")
    print(json.dumps({"status": status, "passed": sum(r["pass"] for r in checks), "total": len(checks)}, ensure_ascii=False))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
