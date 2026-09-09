#!/usr/bin/env python3
"""Deterministic manuscript, figure and source-data gate for the v49 package."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
MAIN = ROOT / "musript_new_v49.tex"
SUPP = ROOT / "musript_new_v49_supplementary.tex"
SRC = ROOT / "figure_work/source_data_v49"
WORKSPACE = ROOT.parents[4]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    main_text = MAIN.read_text(encoding="utf-8")
    supp_text = SUPP.read_text(encoding="utf-8")
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, evidence: str = "") -> None:
        checks.append({"check": name, "pass": bool(passed), "evidence": evidence})

    title = "稀缺调峰事件聚集重塑储能的跨时长价值与配置选择"
    check("title_exact", title in main_text)
    check("supplement_title_exact", title in supp_text)
    check("v49_reference_file_main", "references_v49.tex" in main_text and "references_v48.tex" not in main_text)
    check("v49_reference_file_supp", "references_v49.tex" in supp_text and "references_v48.tex" not in supp_text)
    abstract = main_text.split("\\section*{摘要}", 1)[1].split("\\section*{引言}", 1)[0]
    removed_clause = "也可使中国 2030 年分散序列中的 24/26 个谷消失"
    check("requested_abstract_clause_removed", removed_clause not in abstract)

    graphics = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{\\detokenize\{([^}]+)\}\}", main_text + "\n" + supp_text)
    missing = [path for path in graphics if not (ROOT / path).exists()]
    check("all_graphics_exist", not missing, "; ".join(missing))
    for number, stem in {
        1: "fig_v49_figure1_framework.pdf",
        2: "fig_v49_figure2_world_three_scale_empirical_settings.pdf",
        5: "fig_v49_figure5_three_scale_mid_duration_valley.pdf",
    }.items():
        check(f"v49_figure{number}_used", stem in main_text)
    check("figure5_caption_has_panels_a_to_h", all(f"\\textbf{{{letter}}}," in main_text for letter in "abcdefgh"))

    china = pd.read_csv(SRC / "figure2b_china_map.csv")
    check("china_26_provinces", len(china) == 26, str(len(china)))
    check("china_no_q10", not china.q10_baseline_applied.astype(bool).any(), str(china.q10_baseline_applied.unique()))
    check("china_q75_label_only", set(china.q75_role) == {"extreme_peaking_day_label_only"})
    check("china_task_10_71_twh", abs(china.annual_observed_peaking_task_gwh.sum() / 1000 - 10.712871770959199) < 1e-9)
    check("china_wording_main", "不扣除 q10 或任何低位基线" in main_text and "q75 只用于标记" in main_text)
    check("china_wording_supp", "中国 C1 不扣除 q10" in supp_text)

    japan = pd.read_csv(SRC / "figure2c_japan_map.csv")
    check("japan_27_case_years", len(japan) == 27, str(len(japan)))
    check("japan_19_unique_meshes", japan.mesh_id.nunique() == 19, str(japan.mesh_id.nunique()))
    check("japan_5_prefectures", japan.prefecture.nunique() == 5, str(japan.prefecture.nunique()))

    natural_earth = WORKSPACE / "dataset/external_geospatial/world_natural_earth_110m_v5_1_1/ne_110m_admin_0_countries.shp"
    japan_root = WORKSPACE / "dataset/external_geospatial/japan_mlit_n03_2024"
    check("natural_earth_world_basemap_present", natural_earth.exists(), str(natural_earth))
    check("japan_official_boundaries_present", len(list(japan_root.glob("N03-*/*.geojson"))) == 5)

    europe = pd.read_csv(SRC / "figure5a_europe_duration_response.csv")
    expected_europe = {24: -5.2060209053, 32: -7.6264979616, 40: -7.5479750615, 96: 1.2687884937, 168: 6.2019003466}
    actual_europe = {int(row.duration_h): row.point_estimate for row in europe.itertuples()}
    check("europe_duration_response", all(abs(actual_europe[h] - value) < 1e-8 for h, value in expected_europe.items()), str({h: actual_europe[h] for h in expected_europe}))

    cross = pd.read_csv(SRC / "figure5bc_china_japan_duration_response.csv")
    def effect(application: str, duration: int) -> float:
        return float(cross[(cross.application_case == application) & (cross.duration_h == duration)].mean_effect_eur_kw_year.iloc[0])
    check("china_24_40_negative_240_positive", effect("china", 24) < 0 and effect("china", 40) < 0 and effect("china", 240) > 0)
    check("japan_32_negative_72_positive", effect("japan", 32) < 0 and effect("japan", 72) > 0)

    counts = pd.read_csv(SRC / "figure5d_three_scale_strict_valley_counts.csv")
    expected_counts = {
        ("Europe", "dispersed", "E=98.54, Q=0, F=600, 2035 costs"): 0,
        ("Europe", "clustered", "E=98.54, Q=0, F=600, 2035 costs"): 85,
        ("China", "dispersed", "China central 2030"): 24,
        ("China", "clustered", "China central 2030"): 0,
        ("China", "dispersed", "China clustered-valley phase"): 0,
        ("China", "clustered", "China clustered-valley phase"): 26,
        ("Japan", "dispersed", "Japan clustered-valley phase"): 0,
        ("Japan", "clustered", "Japan clustered-valley phase"): 8,
    }
    actual_counts = {(row.scale, row.chronology, row.selection_label): int(row.strict_valleys) for row in counts.itertuples()}
    check("strict_valley_counts_source", all(actual_counts.get(key) == value for key, value in expected_counts.items()), str(actual_counts))
    for token in ("85/132", "24/26", "0/26", "26/26", "0/27", "8/27"):
        check(f"main_reports_{token.replace('/', '_')}", token in main_text)

    records = pd.read_csv(SRC / "figure5e_selected_cross_scale_valley_records.csv")
    china_phase = records[(records.selection_label == "China clustered-valley phase") & (records.chronology == "clustered") & records.strict_valley]
    japan_phase = records[(records.selection_label == "Japan clustered-valley phase") & (records.chronology == "clustered") & records.strict_valley]
    check("china_width_mean_40_median_36", abs(china_phase.valley_width_h.mean() - 40) < 1e-9 and abs(china_phase.valley_width_h.median() - 36) < 1e-9)
    check("japan_width_mean_39_median_36", abs(japan_phase.valley_width_h.mean() - 39) < 1e-9 and abs(japan_phase.valley_width_h.median() - 36) < 1e-9)
    check("japan_left_boundary_24", set(japan_phase.left_boundary_h) == {24.0})
    check("japan_right_boundary_48_72", japan_phase.right_boundary_h.min() == 48 and japan_phase.right_boundary_h.max() == 72)

    references = (ROOT / "references_v49.tex").read_text(encoding="utf-8")
    check("reference_count_65", references.count("\\item ") == 65, str(references.count("\\item ")))
    check("natural_earth_reference_present", "Natural Earth" in references and "5.1.1" in references)

    outputs = [
        ROOT / "musript_new_v49.pdf",
        ROOT / "musript_new_v49_supplementary.pdf",
        ROOT / "musript_new_v49_main_pdfunite.pdf",
    ]
    for path in outputs:
        check(f"compiled_{path.name}", path.exists() and path.stat().st_size > 100_000, str(path.stat().st_size if path.exists() else 0))
    for name in ("musript_new_v49.log", "musript_new_v49_supplementary.log"):
        log = (ROOT / name).read_text(encoding="utf-8", errors="ignore")
        forbidden = ("! LaTeX Error", "Undefined control sequence", "Float too large", "Overfull \\hbox", "Overfull \\vbox")
        check(f"clean_layout_log_{name}", not any(token in log for token in forbidden))

    figure_files = sorted((ROOT / "figure_work/main_figures").glob("fig_v49_*"))
    checksum_targets = outputs + [MAIN, SUPP, ROOT / "references_v49.tex"] + figure_files
    checksums = {str(path.relative_to(ROOT)): sha256(path) for path in checksum_targets if path.exists()}
    result = {"status": "pass" if all(item["pass"] for item in checks) else "fail", "checks": checks, "sha256": checksums}
    (ROOT / "V49_FINAL_CONSISTENCY_AUDIT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# v49 final consistency audit", "", f"Overall status: **{result['status'].upper()}**", ""]
    for row in checks:
        line = f"- {'PASS' if row['pass'] else 'FAIL'} — {row['check']}"
        if row["evidence"]:
            line += f": {row['evidence']}"
        lines.append(line)
    (ROOT / "V49_FINAL_CONSISTENCY_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "V49_OUTPUT_SHA256.txt").write_text("\n".join(f"{digest}  {name}" for name, digest in checksums.items()) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
