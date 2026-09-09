#!/usr/bin/env python3
"""Audit the frozen numerical, figure and manuscript contracts for v43."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
MAIN = HERE / "musript_new_v43.tex"
SI = HERE / "musript_new_v43_supplementary.tex"
REPORT = HERE / "V43_FINAL_CONSISTENCY_AUDIT.md"


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

    stale_tokens = [
        "ClusterSignaturePending",
        "V43-SIGNATURE",
        "fig_v42",
        "references_v42",
        "122、127 和 123",
        "96--132 h",
        "72--96 h",
    ]
    for token in stale_tokens:
        require(token not in both, f"Stale manuscript token: {token}")
    require("：" not in main_tex, "Chinese colon remains in main manuscript")
    require("“" not in main_tex and "”" not in main_tex, "Chinese quote remains")
    require(
        "\\section*{结论}" not in main_tex,
        "A separate conclusion section remains",
    )
    checks.append("No stale signature, v42 figure, placeholder or style token")

    main_figures = [
        "fig_v43_figure1_framework",
        "fig_v43_figure2_europe_event_design",
        "fig_v43_figure3_joint_portfolio_value",
        "fig_v43_figure4_shared_recharge_mechanism",
        "fig_v43_figure5_conditional_valley",
        "fig_v43_figure6_value_and_power_sensitivity",
    ]
    extended = [
        "fig_v43_ed1_cost_paths_53_eur",
        "fig_v43_ed2_event_definition_sensitivity",
        "fig_v43_ed3_numerical_qa",
        "fig_v43_ed4_full_technology_duration_response",
        "fig_v43_ed5_conditional_phase_details",
        "fig_v43_ed6_fixed_bounded_tail_power",
        "fig_v43_ed7_fixed_scale_mix_details",
        "fig_v43_ed8_mix_sensitivity",
        "fig_v43_ed9_mix_heterogeneity",
        "fig_v43_ed10_eraa_system_screen",
    ]
    for stem in main_figures:
        for suffix in [".pdf", ".svg", ".png", ".tiff"]:
            require(
                (HERE / "figure_work/main_figures" / f"{stem}{suffix}").exists(),
                f"Missing main-figure output: {stem}{suffix}",
            )
    for stem in extended:
        for suffix in [".pdf", ".svg", ".png", ".tiff"]:
            require(
                (HERE / "figure_work/extended_data" / f"{stem}{suffix}").exists(),
                f"Missing Extended Data output: {stem}{suffix}",
            )
    status = read_csv("figure_work/manifests/v43_figure_status.csv")
    require(len(status) == 16, "Figure status manifest must contain 16 figures")
    require(
        status["status"].isin(["built", "released_frozen"]).all(),
        "Figure status manifest contains a non-final state",
    )
    checks.append("Figure 1–6 and Extended Data 1–10 all have final outputs")

    crossing = read_csv("source_data/v43_figure5e_zero_crossing_summary.csv")
    crossing = crossing.set_index("target_cluster_days")
    expected = {
        2: (128, 4, 0, 60.0, 48.0, 72.0, 60.0, 72.0),
        4: (131, 1, 0, 96.0, 72.0, 100.0, 72.0, 98.0),
        6: (126, 0, 6, 120.0, 96.0, 144.0, 96.0, 120.0),
    }
    columns = [
        "observed_crossings",
        "left_censored",
        "right_censored",
        "median_crossing_h",
        "q25_crossing_h",
        "q75_crossing_h",
        "median_country_block_ci95_low",
        "median_country_block_ci95_high",
    ]
    for cluster_days, values in expected.items():
        observed = crossing.loc[cluster_days, columns].to_numpy(float)
        require(
            np.allclose(observed, np.asarray(values, float)),
            f"Unexpected {cluster_days} d crossing summary: {observed}",
        )
    checks.append("2/4/6 d crossing summaries match the final EFC ledger")

    signature = read_csv(
        "source_data/v43_figure5d_cluster_signature_summary.csv"
    )
    panel_b = read_csv(
        "figure_work/source_data/figure5_panel_b_central_clustering_effect.csv"
    )
    four = signature[signature["target_cluster_days"].eq(4)][
        ["duration_h", "country_block_mean"]
    ]
    comparison = four.merge(
        panel_b[["duration_h", "point_estimate"]],
        on="duration_h",
        how="inner",
        validate="one_to_one",
    )
    require(
        len(comparison) == len(four),
        "Figure 5b and the 4 d signature do not cover the same duration domain",
    )
    ledger_error = float(
        (
            comparison["country_block_mean"] - comparison["point_estimate"]
        )
        .abs()
        .max()
    )
    require(
        ledger_error < 1.0e-7,
        f"Figure 5 signature-ledger mismatch: {ledger_error}",
    )
    checks.append(
        f"Figure 5 4 d mechanism signature closes to panel b ({ledger_error:.2e})"
    )

    catalogue = read_csv(
        "source_data/v43_cost_catalogue_2025_2035_source_aware_eur.csv"
    )
    require(
        catalogue.groupby("learning_year")["config_id"].nunique().to_dict()
        == {2025: 53, 2030: 53, 2035: 53},
        "Cost catalogue is not 53 configurations in each formal cost year",
    )
    require(
        catalogue[catalogue["learning_year"].eq(2035)]
        .loc[lambda x: ~x["tech_id"].eq("li_lfp"), "config_id"]
        .nunique()
        == 49,
        "The portfolio catalogue does not contain 49 non-LFP candidates",
    )
    years = read_csv("source_data/v43_year_specific_valley_summary.csv")
    central = years[
        years["revaluation_case_id"].eq("central_actual_year_cost")
    ]
    require(
        len(central) == 6
        and central["strict_valley_count"].eq(0).all(),
        "A calendar-year central case has a strict valley",
    )
    checks.append("All three formal cost years contain 53 configurations and 0/132 valleys")

    boundaries = read_csv(
        "figure_work/source_data/ed5_conditional_peak_boundaries.csv"
    )
    require(len(boundaries) == 92, "Conditional peak must contain 92 valleys")
    require(
        np.isclose(boundaries["left_boundary_h"].median(), 24.0)
        and np.isclose(boundaries["right_boundary_h"].median(), 44.0)
        and np.isclose(boundaries["valley_width_h"].median(), 20.0),
        "Conditional peak boundary medians are inconsistent",
    )
    checks.append("Conditional peak is 92/132 with 24/44 h boundaries and 20 h width")

    qa_paths = [
        "portfolio_model/full_132cy_fixed_s20_central/qa_summary.json",
        "portfolio_model/full_132cy_optimized/qa_summary.json",
        "experiments_signature/full_s20_fixed_cluster2/qa_summary.json",
        "experiments_signature/full_s20_fixed_cluster4/qa_summary.json",
        "experiments_signature/full_s20_fixed_cluster6/qa_summary.json",
    ]
    for relative in qa_paths:
        qa = json.loads((HERE / relative).read_text(encoding="utf-8"))
        require(
            qa.get("status", "pass") == "pass"
            and not qa.get("failures", []),
            f"Failed solver QA: {relative}",
        )
    checks.append("Central, bounded-power and 2/4/6 d Gurobi runs pass QA")

    section = re.search(
        r"\\section\*\{结果\}(.*?)\\section\*\{讨论\}",
        main_tex,
        flags=re.S,
    )
    require(section is not None, "Cannot locate Results section")
    subsections = re.split(r"\\subsection\*\{[^}]+\}", section.group(1))[1:]
    require(len(subsections) == 4, "Results must contain four subsections")
    paragraph_counts = []
    for block in subsections:
        block = re.split(r"\\begin\{figure\}", block, maxsplit=1)[0]
        paragraphs = [
            paragraph
            for paragraph in re.split(r"\n\s*\n", block.strip())
            if paragraph.strip() and not paragraph.lstrip().startswith("\\")
        ]
        paragraph_counts.append(len(paragraphs))
    require(
        paragraph_counts == [3, 3, 3, 3],
        f"Results paragraph contract failed: {paragraph_counts}",
    )
    checks.append("R1–R4 each contain three evidence-led paragraphs")

    report = [
        "# V43 final consistency audit",
        "",
        "Status: **PASS**",
        "",
        "## Passed gates",
        "",
    ]
    report.extend(f"- {item}." for item in checks)
    report.extend(
        [
            "",
            "## Interpretation guardrails retained",
            "",
            "- Natural task scale and event clustering are associated "
            "(\u03c1 = -0.62), so the manuscript claims experimental separation "
            "rather than observational orthogonality.",
            "- The 96 h sign reversal is attributed to relative cycling return; "
            "coverage return reinforces the response from 120 h onward.",
            "- The strict mid-duration valley is absent from the three central "
            "cost years and is reported as a conditional economic phase.",
            "- The conditional cost multiplier applies only to the 2035 LDES "
            "tail; the additional 4 h LFP comparator remains at 2035 cost.",
            "",
        ]
    )
    REPORT.write_text("\n".join(report), encoding="utf-8")
    print(f"PASS: wrote {REPORT}")


if __name__ == "__main__":
    main()
