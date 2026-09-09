#!/usr/bin/env python3
"""Deterministic consistency gate for the v48 manuscript package."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
MAIN = ROOT / "musript_new_v48.tex"
SUPP = ROOT / "musript_new_v48_supplementary.tex"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    main_text = MAIN.read_text(encoding="utf-8")
    supp_text = SUPP.read_text(encoding="utf-8")
    checks = []

    def check(name: str, passed: bool, evidence: str = "") -> None:
        checks.append({"check": name, "pass": bool(passed), "evidence": evidence})

    check("title_exact", "稀缺调峰事件聚集重塑储能的跨时长价值与配置选择" in main_text)
    check("supplement_title_exact", "稀缺调峰事件聚集重塑储能的跨时长价值与配置选择" in supp_text)
    check("v48_reference_file_main", "references_v48.tex" in main_text and "references_v47.tex" not in main_text)
    check("v48_reference_file_supp", "references_v48.tex" in supp_text and "references_v47.tex" not in supp_text)

    graphics = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{\\detokenize\{([^}]+)\}\}", main_text + "\n" + supp_text)
    missing = [path for path in graphics if not (ROOT / path).exists()]
    check("all_graphics_exist", not missing, "; ".join(missing))
    check("v48_figure1_used", "fig_v48_figure1_framework.pdf" in main_text)
    check("v48_figure2_used", "fig_v48_figure2_three_scale_empirical_settings.pdf" in main_text)
    check("v48_figure5_used", "fig_v48_figure5_three_scale_valley.pdf" in main_text)

    china_qa = json.loads((ROOT / "cross_scale_cases/china_province_panel/qa.json").read_text())
    check("china_no_q10", china_qa.get("q10_baseline_applied") is False, str(china_qa.get("q10_baseline_applied")))
    check("china_direct_task_definition", china_qa.get("task_definition") == "observed_provincial_gas_peaking_direct_no_baseline_subtraction")
    check("china_source_energy_identity", china_qa.get("annual_task_energy_matches_source_after_kwh_to_mwh") is True)
    check("china_panel_rebuild_unchanged", sha256(ROOT / "cross_scale_cases/china_province_panel/china_province_event_task_panel.csv.gz") == "83829e7f86ff83710c73a6f06919c2be54a1c3ebdf127a9f415cf92689f32551")
    check("china_wording_main", "不扣除 q10 或任何低位基线" in main_text and "q75 只用于标记" in main_text)
    check("china_wording_supp", "中国 C1 不扣除 q10" in supp_text)

    counts = pd.read_csv(ROOT / "figure_work/source_data_v48/figure5d_three_scale_strict_valley_counts.csv")
    expected = {
        ("Europe", "dispersed", "E=98.54, Q=0, F=600, 2035 costs"): 0,
        ("Europe", "clustered", "E=98.54, Q=0, F=600, 2035 costs"): 85,
        ("China", "dispersed", "China central 2030"): 24,
        ("China", "clustered", "China central 2030"): 0,
        ("China", "dispersed", "China clustered-valley phase"): 0,
        ("China", "clustered", "China clustered-valley phase"): 26,
        ("Japan", "dispersed", "Japan clustered-valley phase"): 0,
        ("Japan", "clustered", "Japan clustered-valley phase"): 8,
    }
    actual = {(r.scale, r.chronology, r.selection_label): int(r.strict_valleys) for r in counts.itertuples()}
    check("strict_valley_counts_source", all(actual.get(k) == v for k, v in expected.items()), str(actual))
    for token in ("85/132", "24/26", "0/26", "26/26", "0/27", "8/27"):
        check(f"main_reports_{token.replace('/', '_')}", token in main_text)

    japan = pd.read_csv(ROOT / "figure_work/source_data_v48/figure2c_japan_map.csv")
    check("japan_case_years_27", len(japan) == 27, str(len(japan)))
    check("japan_unique_meshes_19", japan.mesh_id.nunique() == 19, str(japan.mesh_id.nunique()))
    check("japan_official_boundaries_present", len(list((ROOT.parents[4] / "dataset/external_geospatial/japan_mlit_n03_2024").glob("N03-*/*.geojson"))) == 5)

    references = (ROOT / "references_v48.tex").read_text(encoding="utf-8")
    check("reference_count_64", references.count("\\item ") == 64, str(references.count("\\item ")))
    check("mlit_reference_present", "National Land Numerical Information" in references)

    for name in ("musript_new_v48.pdf", "musript_new_v48_supplementary.pdf", "musript_new_v48_main_pdfunite.pdf"):
        path = ROOT / name
        check(f"compiled_{name}", path.exists() and path.stat().st_size > 100_000, str(path.stat().st_size if path.exists() else 0))
    for name in ("musript_new_v48.log", "musript_new_v48_supplementary.log"):
        log = (ROOT / name).read_text(encoding="utf-8", errors="ignore")
        check(f"no_latex_error_{name}", "! LaTeX Error" not in log and "Undefined control sequence" not in log)

    result = {"status": "pass" if all(x["pass"] for x in checks) else "fail", "checks": checks}
    (ROOT / "V48_FINAL_CONSISTENCY_AUDIT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = ["# v48 final consistency audit", "", f"Overall status: **{result['status'].upper()}**", ""]
    for row in checks:
        lines.append(f"- {'PASS' if row['pass'] else 'FAIL'} — {row['check']}" + (f": {row['evidence']}" if row['evidence'] else ""))
    (ROOT / "V48_FINAL_CONSISTENCY_AUDIT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
