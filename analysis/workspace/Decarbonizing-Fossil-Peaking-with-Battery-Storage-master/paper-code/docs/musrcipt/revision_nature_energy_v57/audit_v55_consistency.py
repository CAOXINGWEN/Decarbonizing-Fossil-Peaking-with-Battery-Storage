#!/usr/bin/env python3
"""Cross-check the frozen v55 manuscript, figures and natural-weather ledger."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MAIN = ROOT / "musript_new_v55.tex"
SUPP = ROOT / "musript_new_v55_supplementary.tex"
SUMMARY = ROOT / "weather_year_parity/results_europe/pecd_europe_weather_parity_summary.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def pdf_pages(path: Path) -> int:
    output = subprocess.check_output(["pdfinfo", str(path)], text=True)
    match = re.search(r"^Pages:\s+(\d+)$", output, flags=re.MULTILINE)
    require(match is not None, f"cannot read page count: {path.name}")
    return int(match.group(1))


def main() -> None:
    text = MAIN.read_text(encoding="utf-8")
    supp = SUPP.read_text(encoding="utf-8")
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))

    require(summary["status"] == "pass", "natural-weather summary did not pass")
    europe = summary["europe33_2025_2035"]
    require(abs(europe["relative_cost_improvement_eur_kw_year"] - 5.2960380996609935) < 1e-10, "cost threshold changed")
    require(abs(europe["full_natural_span_eur_kw_year"] - 9.585429031930671) < 1e-10, "Europe-33 span changed")
    require(europe["all_pairs_with_absolute_shift_reaching_cost_improvement"] == 73, "all-pair count changed")
    require(europe["horizon_matched_adverse_transitions_erasing_learning"] == 1, "matched-decade count changed")
    require(summary["all_aggregate_weather_cost_cells_caes24_outperforms_lfp4"] == 0, "economic-crossover result changed")

    for stale in ("6.03", "59.52", "36.33", "35.8", "40.8", "references_v54"):
        require(stale not in text, f"stale manuscript value or path: {stale}")

    required_main = (
        "9.59 EUR",
        "8.07 EUR",
        "73 个组合",
        "1 个抵消",
        "0/34 个经济交叉",
        "56.87",
        "44.42",
        "references_v55.tex",
    )
    for phrase in required_main:
        require(phrase in text, f"missing main-text anchor: {phrase}")

    figure_paths = re.findall(r"\\detokenize\{([^}]+\.pdf)\}", text)
    require(len(figure_paths) == 6, f"expected 6 main figures, found {len(figure_paths)}")
    for relative in figure_paths:
        require((ROOT / relative).is_file(), f"missing main figure: {relative}")

    require("Supplementary Table 6" in supp, "natural-weather ledger absent from supplement")
    require("Extended Data Figure 11" in supp, "three-regime atlas absent from supplement")
    require("Extended Data Figure 12" in supp, "country natural-weather figure absent from supplement")

    for log_name in ("musript_new_v55.log", "musript_new_v55_supplementary.log"):
        log = (ROOT / log_name).read_text(encoding="utf-8", errors="replace")
        for error in ("Overfull", "Undefined control sequence", "Float too large", "Missing $ inserted"):
            require(error not in log, f"{error} in {log_name}")

    require(pdf_pages(ROOT / "musript_new_v55.pdf") == 19, "unexpected main PDF length")
    require(pdf_pages(ROOT / "musript_new_v55_supplementary.pdf") == 15, "unexpected supplement PDF length")
    print("v55 consistency audit: PASS")


if __name__ == "__main__":
    main()
