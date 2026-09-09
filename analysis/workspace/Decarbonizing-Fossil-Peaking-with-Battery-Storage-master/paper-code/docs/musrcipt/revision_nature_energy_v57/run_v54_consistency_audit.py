#!/usr/bin/env python3
"""Deterministic text, figure, and source-data consistency gate for v54."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
MAIN = ROOT / "musript_new_v54.tex"
SUPP = ROOT / "musript_new_v54_supplementary.tex"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    main_text = MAIN.read_text(encoding="utf-8")
    supp_text = SUPP.read_text(encoding="utf-8")
    before_discussion = main_text.split(r"\section*{讨论}", 1)[0]
    results = main_text.split(r"\section*{结果}", 1)[1].split(r"\section*{讨论}", 1)[0]
    checks: list[dict[str, object]] = []

    def check(name: str, condition: bool, evidence: object = "") -> None:
        checks.append({"check": name, "pass": bool(condition), "evidence": str(evidence)})

    title = "时序平价揭示稀缺事件聚集下的储能投资敞口"
    check("v54_title_main", title in main_text)
    check("v54_title_supplementary", title in supp_text)
    check("v54_reference_file", r"\input{references_v54.tex}" in main_text)
    stale_tokens = [token for token in ["v52", "v53", "fig_v52", "fig_v53"] if token in main_text or token in supp_text]
    check("no_stale_version_tokens_in_active_tex", not stale_tokens, stale_tokens)

    intro = main_text.split(r"\section*{引言}", 1)[1].split(r"\begin{figure}", 1)[0]
    intro_paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", intro.strip())
        if paragraph.strip() and not paragraph.lstrip().startswith("\\")
    ]
    intro_jobs = ["两种作用方向相反", "储能投资账本", "恢复窗口", "两个可检验命题", "我们以三个并列应用尺度"]
    check("introduction_exactly_five_paragraphs", len(intro_paragraphs) == 5, len(intro_paragraphs))
    check("introduction_five_part_argument", all(token in paragraph for token, paragraph in zip(intro_jobs, intro_paragraphs)))
    check("sequencing_parity_defined_in_introduction", "本文将这一比较称为时序平价" in intro)

    expected_headings = [
        "三种平衡体制分离年度调峰规模与事件时序",
        "恢复窗口将事件时序转化为跨时长储能价值",
        "恢复时间而非补能总量控制事件间库存重建",
        "事件簇长度沿时长轴移动价值反转",
        "时序平价标出三体制的中时长投资敞口",
    ]
    headings = re.findall(r"\\subsection\*\{([^}]+)\}", results)
    check("five_part_results_chain", headings == expected_headings, headings)
    blocks = re.split(r"\\subsection\*\{[^}]+\}", results)[1:]
    paragraph_counts = []
    for block in blocks:
        prose = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", "", block, flags=re.S)
        prose = re.sub(r"\\begin\{table\}.*?\\end\{table\}", "", prose, flags=re.S)
        paragraphs = [
            paragraph for paragraph in re.split(r"\n\s*\n", prose.strip())
            if paragraph and not paragraph.lstrip().startswith("\\")
        ]
        paragraph_counts.append(len(paragraphs))
    check("result_paragraph_structure", paragraph_counts == [3, 3, 4, 4, 4], paragraph_counts)

    defensive_terms = [
        "不意味着", "不用于直接比较", "尚未内生", "限定结果外推", "稳健性与系统尺度",
        "不能证明", "无法外推", "谨慎解释", "保守估计", "仅为", "只用于", "可能导致", "可能改变",
    ]
    defensive_hits = [term for term in defensive_terms if term in before_discussion]
    check("no_defensive_language_before_discussion", not defensive_hits, defensive_hits)
    check("limitations_concentrated_in_discussion", "当前单节点组合调度未共同内生" in main_text)

    graphics = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{\\detokenize\{([^}]+)\}\}", main_text + "\n" + supp_text)
    missing_graphics = [path for path in graphics if not (ROOT / path).exists()]
    check("all_included_graphics_exist", not missing_graphics, missing_graphics)

    main_stems = [
        "fig_v54_figure1_sequencing_parity_framework",
        "fig_v54_figure2_three_regime_event_atlas",
        "fig_v54_figure3_portfolio_cycling_ledger",
        "fig_v54_figure4_recovery_interface_tests",
        "fig_v54_figure5_cross_duration_reversal",
        "fig_v54_figure6_sequencing_parity_exposure",
    ]
    for stem in main_stems:
        check(f"main_uses_{stem}", f"{stem}.pdf" in main_text)
        for suffix in ["pdf", "svg", "png", "tiff", "_grayscale.png"]:
            filename = f"{stem}{suffix}" if suffix.startswith("_") else f"{stem}.{suffix}"
            path = ROOT / "figure_work/main_figures" / filename
            check(f"main_asset_{filename}", path.exists() and path.stat().st_size > 10_000,
                  path.stat().st_size if path.exists() else 0)

    ed_stems = [
        "fig_v54_ed1_cost_paths_53_eur",
        "fig_v54_ed2_event_definition_sensitivity",
        "fig_v54_ed3_numerical_qa",
        "fig_v54_ed4_full_technology_duration_response",
        "fig_v54_ed5_conditional_phase_details",
        "fig_v54_ed6_bounded_joint_procurement",
        "fig_v54_ed7_fixed_scale_mix_details",
        "fig_v54_ed8_mix_sensitivity",
        "fig_v54_ed9_mix_heterogeneity",
        "fig_v54_ed10_eraa_system_screen",
    ]
    for stem in ed_stems:
        check(f"supp_uses_{stem}", f"{stem}.pdf" in supp_text)
        for suffix in ["pdf", "png"]:
            path = ROOT / "figure_work/extended_data" / f"{stem}.{suffix}"
            check(f"extended_asset_{stem}.{suffix}", path.exists() and path.stat().st_size > 10_000,
                  path.stat().st_size if path.exists() else 0)
    check("supplement_is_evidence_only", "Supplementary Note" not in supp_text)

    parity_dir = ROOT / "figure_work/source_data/v54_parity_exposure"
    summary = pd.read_csv(parity_dir / "figure6b_24h_cross_regime_parity_summary.csv").set_index("regime")
    expected_counts = {"Europe": (132, 68), "China": (26, 26), "Japan": (27, 22)}
    actual_counts = {
        regime: (int(summary.loc[regime, "n_system_years"]), int(summary.loc[regime, "above_parity_count"]))
        for regime in expected_counts
    }
    check("three_regime_24h_parity_counts", actual_counts == expected_counts, actual_counts)
    expected_medians = {"Europe": 5.47739487, "China": 59.50398690, "Japan": 9.25236489}
    actual_medians = {regime: float(summary.loc[regime, "median_sequencing_shift_eur_kw_year"]) for regime in expected_medians}
    check("three_regime_24h_parity_medians",
          all(np.isclose(actual_medians[key], value, atol=1e-6) for key, value in expected_medians.items()),
          actual_medians)
    sensitivity = pd.read_csv(ROOT / "sequencing_parity/cost_duration_discount_sensitivity.csv")
    central_cost_row = sensitivity[
        np.isclose(sensitivity.discount_rate, 0.036)
        & np.isclose(sensitivity.caes_duration_h, 24.0)
    ].iloc[0]
    central_cost_movement = float(central_cost_row.absolute_relative_cost_movement_eur_kw_year)
    check("matched_cost_movement_5_30",
          np.allclose(summary.median_matched_cost_movement_eur_kw_year, central_cost_movement),
          central_cost_movement)
    caes_2025_cost = float(central_cost_row.caes_2025_annual_fixed_cost_eur_kw_year)
    parameter_library = pd.read_csv(
        ROOT.parents[4] / "dataset/M2_LDES_technology_parameter_library_v0_2.csv"
    ).set_index("tech_id")
    check("caes_60_year_2025_annualized_cost",
          int(parameter_library.loc["caes", "calendar_life_years"]) == 60
          and np.isclose(caes_2025_cost, 56.87352019939118, atol=1e-9)
          and "56.87" in main_text and "56.87" in supp_text,
          {"life_years": int(parameter_library.loc["caes", "calendar_life_years"]),
           "caes_2025_eur_kw_year": caes_2025_cost})

    records = pd.read_csv(parity_dir / "figure6cde_system_year_parity_records.csv")
    exposure = pd.read_csv(parity_dir / "figure6cde_system_duration_exposure.csv")
    check("parity_source_row_counts", len(records) == 4599 and len(exposure) == 1575,
          {"records": len(records), "exposure": len(exposure)})
    at_24 = records[np.isclose(records.duration_h, 24)].groupby("regime").above_parity.agg(["size", "sum"])
    counts_24 = {regime: (int(at_24.loc[regime, "size"]), int(at_24.loc[regime, "sum"])) for regime in expected_counts}
    check("heatmap_24h_counts_match_summary", counts_24 == expected_counts, counts_24)
    china_band = records[(records.regime == "China") & records.duration_h.between(8, 24)]
    check("china_8_to_24h_all_above_parity", bool(china_band.above_parity.all()), china_band.shape[0])
    check("heatmap_source_has_three_regimes", set(exposure.regime) == {"Europe", "China", "Japan"})

    manifest_path = parity_dir / "v54_parity_exposure_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_hashes_ok = all(sha256(ROOT / relative) == digest for relative, digest in manifest["output_sha256"].items())
    check("parity_manifest_output_hashes", manifest_hashes_ok)

    check("parity_values_present_in_abstract",
          all(token in main_text.split(r"\section*{引言}", 1)[0] for token in ["5.48", "59.50", "9.25", "68/132", "26/26", "22/27"]))
    abstract = main_text.split(r"\section*{摘要}", 1)[1].split(r"\section*{引言}", 1)[0]
    check("abstract_is_claim_led_not_heatmap_led", "热力图" not in abstract and "时序平价假说" in abstract)
    parity_sections = [block for heading, block in zip(headings, blocks) if heading in expected_headings[1:4]]
    check("three_mechanism_results_connect_to_parity", all("时序平价" in block for block in parity_sections))
    check("three_heatmaps_described_in_results", "三张敞口热力图" in results and "图 6c--e" in results)
    check("figure6_caption_has_three_heatmaps", "Configuration-resolved parity-exposure heat maps" in main_text)
    check("figure4_panel_order_matches_layout", all(token in main_text for token in [
        r"\textbf{b}, Country medians", r"\textbf{c}, Country classification",
        r"\textbf{d}, Clustered-to-dispersed retention", r"\textbf{g}, Country--year relations",
    ]))
    check("figure5_panel_order_matches_layout", all(token in main_text for token in [
        r"\textbf{b}, Country median", r"\textbf{c}, Country share",
        r"\textbf{g}, Separate European", r"\textbf{h}, Natural longest-event",
    ]))
    figure_script = (ROOT / "figure_work/plot_v54_reference_style_figures.py").read_text(encoding="utf-8")
    figure6_source = figure_script.split("def figure6()", 1)[1].split("def main()", 1)[0]
    check("figure6_heatmaps_are_vector_meshes", "image=heat_ax.pcolormesh" in figure6_source and "heat_ax.imshow" not in figure6_source)
    check("main_panel_limit_eight", all(token in main_text for token in [r"\textbf{h}, Violin distributions", r"\textbf{h}, Country fixed-effect", r"\textbf{h}, Natural longest-event", r"\textbf{h}, Tail-entry or exit"]))

    references = (ROOT / "references_v54.tex").read_text(encoding="utf-8")
    check("reference_count_63", references.count(r"\item ") == 63, references.count(r"\item "))
    for name in ["musript_new_v54.pdf", "musript_new_v54_supplementary.pdf"]:
        path = ROOT / name
        check(f"compiled_{name}", path.exists() and path.stat().st_size > 100_000,
              path.stat().st_size if path.exists() else 0)
    for name in ["musript_new_v54.log", "musript_new_v54_supplementary.log"]:
        log = (ROOT / name).read_text(encoding="utf-8", errors="ignore")
        forbidden = ["! LaTeX Error", "Undefined control sequence", "Float too large", "Overfull \\hbox", "Overfull \\vbox"]
        check(f"clean_{name}", not any(token in log for token in forbidden))

    qa_files = [ROOT / "qa_render_v54/main_contact.png", ROOT / "qa_render_v54/supp_contact.png"]
    check("pdf_contact_sheets", all(path.exists() and path.stat().st_size > 100_000 for path in qa_files))

    outputs = [
        MAIN, SUPP, ROOT / "references_v54.tex", ROOT / "musript_new_v54.pdf",
        ROOT / "musript_new_v54_supplementary.pdf", ROOT / "figure_work/build_v54_parity_exposure.py",
        ROOT / "figure_work/plot_v54_reference_style_figures.py", ROOT / "figure_work/plot_v54_extended_data.py",
        manifest_path, ROOT / "V54_REVISION_QA.md", *qa_files,
        ROOT / "figure_work/qa_v54/V54_MAIN_COLOR_CONTACT_SHEET.png",
        ROOT / "figure_work/qa_v54/V54_MAIN_GRAYSCALE_CONTACT_SHEET.png",
    ]
    outputs += sorted((ROOT / "figure_work/main_figures").glob("fig_v54_*"))
    outputs += sorted((ROOT / "figure_work/extended_data").glob("fig_v54_*.pdf"))
    outputs += sorted((ROOT / "figure_work/extended_data").glob("fig_v54_*.png"))
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in outputs if path.exists()}
    status = "PASS" if all(row["pass"] for row in checks) else "FAIL"
    payload = {"status": status, "checks": checks, "sha256": hashes}
    (ROOT / "V54_FINAL_CONSISTENCY_AUDIT.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = ["# V54 final consistency audit", "", f"Status: **{status}**", ""]
    for row in checks:
        evidence = f": {row['evidence']}" if row["evidence"] else ""
        lines.append(f"- {'PASS' if row['pass'] else 'FAIL'} - {row['check']}{evidence}")
    (ROOT / "V54_FINAL_CONSISTENCY_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "V54_OUTPUT_SHA256.txt").write_text(
        "\n".join(f"{digest}  {name}" for name, digest in hashes.items()) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": status, "passed": sum(row["pass"] for row in checks), "total": len(checks)}, ensure_ascii=False))
    if status != "PASS":
        failed = [row for row in checks if not row["pass"]]
        print(json.dumps(failed, ensure_ascii=False, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
