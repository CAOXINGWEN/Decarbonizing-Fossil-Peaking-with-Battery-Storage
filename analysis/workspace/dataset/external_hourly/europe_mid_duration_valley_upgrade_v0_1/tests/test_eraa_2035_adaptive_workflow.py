#!/usr/bin/env python3
"""Self-tests for the non-solver ERAA adaptive workflow."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eraa_2035_adaptive_workflow import (  # noqa: E402
    CHRONOLOGIES,
    DEFAULT_CATALOG,
    KEY_CONFIG_IDS,
    WEATHER_SCENARIOS,
    audit_baseline_cache,
    boundary_analysis,
    build_adaptive_manifest,
    compute_scale_estimate,
    evaluate_dense_trigger,
    load_catalog,
    load_weather_mapping,
    resolve_manifest,
    result_schema_contract,
    validate_results,
)


def synthetic_trigger_results(
    manifest: pd.DataFrame,
    *,
    sign_change: bool,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    states = manifest[
        [
            "weather_scenario",
            "chronology",
            "deployment_scope",
            "capacity_case",
            "split_case",
            "baseline_cache_group_id",
        ]
    ].drop_duplicates()
    cache_by_state: dict[tuple[str, ...], str] = {}
    for task in states.itertuples(index=False):
        state = (
            task.weather_scenario,
            task.chronology,
            task.deployment_scope,
            task.capacity_case,
            task.split_case,
        )
        cache_key = f"runner_{task.baseline_cache_group_id}"
        cache_by_state[state] = cache_key
        rows.append(
            {
                "weather_scenario": task.weather_scenario,
                "chronology": task.chronology,
                "deployment_scope": task.deployment_scope,
                "capacity_case": task.capacity_case,
                "split_case": task.split_case,
                "variant": "baseline",
                "config_id": "__baseline__",
                "technology": "none",
                "duration_h": 0.0,
                "objective_eur_year": 1_000.0,
                "signed_joint_benefit_eur_kw_year": np.nan,
                "baseline_cache_key": cache_key,
            }
        )
    for task in manifest.loc[manifest["stage"].eq("key_screen")].itertuples(
        index=False
    ):
        state = (
            task.weather_scenario,
            task.chronology,
            task.deployment_scope,
            task.capacity_case,
            task.split_case,
        )
        value = 5.0
        if (
            sign_change
            and task.weather_scenario == "WS01"
            and float(task.duration_h) >= 24.0
        ):
            value = -5.0
        rows.append(
            {
                "weather_scenario": task.weather_scenario,
                "chronology": task.chronology,
                "deployment_scope": task.deployment_scope,
                "capacity_case": task.capacity_case,
                "split_case": task.split_case,
                "variant": "joint",
                "config_id": task.config_id,
                "technology": task.technology,
                "duration_h": float(task.duration_h),
                "objective_eur_year": 995.0,
                "signed_joint_benefit_eur_kw_year": value,
                "baseline_cache_key": cache_by_state[state],
            }
        )
    return pd.DataFrame(rows)


def synthetic_complete_unit_values(catalog: pd.DataFrame) -> pd.DataFrame:
    main = catalog.loc[
        ~catalog["structural_extrapolation"]
        & catalog["duration_h"].between(4.0, 168.0, inclusive="both")
    ]
    rows: list[dict[str, object]] = []
    for weather in WEATHER_SCENARIOS:
        for chronology in CHRONOLOGIES:
            for config in main.itertuples(index=False):
                duration = float(config.duration_h)
                turnover_end = 24.0 if chronology == "dispersed" else 8.0
                bt = 4.0 if duration <= turnover_end else -2.0
                if duration <= turnover_end:
                    bj = 4.0
                elif duration < 72.0:
                    bj = -3.0
                else:
                    bj = 5.0
                ba = 4.0 if duration >= 72.0 else -1.0
                elcc = 10.0 if duration >= 72.0 else -1.0
                rows.append(
                    {
                        "weather_scenario": weather,
                        "chronology": chronology,
                        "config_id": config.config_id,
                        "tech_id": config.tech_id,
                        "duration_h": duration,
                        "main_structural_extrapolation": False,
                        "bt": bt,
                        "bj": bj,
                        "ba": ba,
                        "elcc": elcc,
                    }
                )
    return pd.DataFrame(rows)


class AdaptiveManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_catalog(DEFAULT_CATALOG)
        cls.manifest = build_adaptive_manifest(
            cls.catalog,
            output_root=Path("/tmp/not_written"),
            model_input_fingerprint="selftest-input",
        )

    def test_frozen_catalog_split_and_compute_scale(self) -> None:
        self.assertEqual(len(KEY_CONFIG_IDS), 12)
        self.assertEqual(
            int(self.manifest["baseline_cache_group_id"].nunique()), 72
        )
        self.assertEqual(
            int(
                self.manifest["task_type"].eq(
                    "candidate_cache_warm"
                ).sum()
            ),
            72,
        )
        self.assertEqual(
            int(self.manifest["stage"].eq("key_screen").sum()), 864
        )
        self.assertEqual(
            int(self.manifest["stage"].eq("dense_followup").sum()), 2_736
        )
        self.assertEqual(len(self.manifest), 3_600)
        self.assertFalse(self.manifest["adaptive_task_id"].duplicated().any())
        scale = compute_scale_estimate(self.manifest)
        self.assertEqual(scale["initial_key_stage_optimization_solves"], 1_800)
        self.assertEqual(
            scale["maximum_full_spectrum_optimization_solves"], 7_272
        )
        self.assertEqual(
            scale["baseline_solves_avoided_at_full_spectrum"], 3_528
        )

    def test_manifest_cli_and_direct_elcc_contract_match_runner(self) -> None:
        runner = ROOT / "scripts" / "run_eraa_2035_full_network_gurobi.py"
        tree = ast.parse(runner.read_text(encoding="utf-8"))
        runner_options = {
            node.args[0].value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and node.args[0].value.startswith("--")
        }
        manifest_options = set().union(
            *(
                set(json.loads(value))
                for value in self.manifest["runner_cli_arguments_json"]
            )
        )
        self.assertTrue(manifest_options.issubset(runner_options))
        runner_source = runner.read_text(encoding="utf-8")
        for field in (
            '"elcc_mw"',
            '"elcc_lower_mw"',
            '"elcc_upper_mw"',
            '"equivalent_full_cycles_output_basis"',
        ):
            self.assertIn(field, runner_source)
        schema = result_schema_contract()
        self.assertEqual(schema["remaining_external_field"], [])
        self.assertIn(
            "writes elcc_mw directly",
            schema["elcc_postprocessing_table"]["status"],
        )

    def test_final_schema_accepts_direct_runner_elcc(self) -> None:
        common = {
            "weather_scenario": "WS01",
            "chronology": "dispersed",
            "deployment_scope": "eu27",
            "capacity_case": "amended_annex3_risk_averse_enhanced_hurdle",
            "split_case": "retirement_feasible_max_ccgt",
            "technology": "adiabatic CAES",
            "duration_h": 24.0,
            "baseline_cache_key": "runner-cache-key",
            "expected_eens_mwh": 1.0,
            "expected_lole_h": 1.0,
            "candidate_charge_power_mw": 1.0,
            "candidate_discharge_power_mw": 1.0,
            "equivalent_full_cycles_output_basis": 1.0,
            "gas_generation_mwh": 1.0,
            "vre_spill_mwh": 1.0,
        }
        rows = [
            {
                **common,
                "variant": "baseline",
                "config_id": "__baseline__",
                "technology": "none",
                "duration_h": 0.0,
                "objective_eur_year": 100.0,
                "signed_joint_benefit_eur_kw_year": np.nan,
                "signed_repeat_use_benefit_eur_kw_year": np.nan,
                "residual_adequacy_interaction_benefit_eur_kw_year": np.nan,
                "elcc_mw": np.nan,
            },
            {
                **common,
                "variant": "joint",
                "config_id": "caes_24h",
                "objective_eur_year": 90.0,
                "signed_joint_benefit_eur_kw_year": 5.0,
                "signed_repeat_use_benefit_eur_kw_year": 3.0,
                "residual_adequacy_interaction_benefit_eur_kw_year": 2.0,
                "elcc_mw": 10.0,
            },
            {
                **common,
                "variant": "repeat_use_fixed_firm",
                "config_id": "caes_24h",
                "objective_eur_year": 94.0,
                "signed_joint_benefit_eur_kw_year": 5.0,
                "signed_repeat_use_benefit_eur_kw_year": 3.0,
                "residual_adequacy_interaction_benefit_eur_kw_year": 2.0,
                "elcc_mw": 10.0,
            },
        ]
        validated = validate_results(pd.DataFrame(rows), mode="final")
        self.assertEqual(len(validated), 3)
        direct = validated.loc[validated["variant"].eq("joint"), "elcc_mw"]
        self.assertEqual(direct.tolist(), [10.0])

    def test_baseline_cache_and_no_trigger(self) -> None:
        results = validate_results(
            synthetic_trigger_results(self.manifest, sign_change=False),
            mode="trigger",
        )
        cache = audit_baseline_cache(results, self.manifest)
        self.assertTrue(cache["pass"])
        decision, diagnostics = evaluate_dense_trigger(
            results, self.manifest
        )
        self.assertFalse(decision["activate_dense_followup"])
        self.assertEqual(len(diagnostics), 0)
        resolved = resolve_manifest(self.manifest, decision)
        dense = resolved.loc[resolved["stage"].eq("dense_followup")]
        self.assertTrue(dense["status"].eq("skipped_no_key_sign_change").all())

    def test_sign_change_activates_all_dense_tasks(self) -> None:
        results = validate_results(
            synthetic_trigger_results(self.manifest, sign_change=True),
            mode="trigger",
        )
        decision, diagnostics = evaluate_dense_trigger(
            results, self.manifest
        )
        self.assertTrue(decision["activate_dense_followup"])
        self.assertGreater(len(diagnostics), 0)
        resolved = resolve_manifest(self.manifest, decision)
        dense = resolved.loc[resolved["stage"].eq("dense_followup")]
        self.assertEqual(len(dense), 2_736)
        self.assertTrue(dense["activation_state"].eq("activated").all())

    def test_missing_key_result_blocks_decision(self) -> None:
        results = validate_results(
            synthetic_trigger_results(self.manifest, sign_change=True),
            mode="trigger",
        )
        joint_index = results.index[results["variant"].eq("joint")][0]
        results = results.drop(index=joint_index)
        decision, diagnostics = evaluate_dense_trigger(
            results, self.manifest
        )
        self.assertTrue(decision["status"].startswith("blocked"))
        self.assertFalse(decision["activate_dense_followup"])
        self.assertEqual(len(diagnostics), 0)

    def test_identical_baseline_references_are_collapsed(self) -> None:
        results = synthetic_trigger_results(self.manifest, sign_change=False)
        duplicate = results.loc[results["variant"].eq("baseline")].iloc[[0]]
        validated = validate_results(
            pd.concat([results, duplicate], ignore_index=True),
            mode="trigger",
        )
        self.assertEqual(
            int(validated["variant"].eq("baseline").sum()), 72
        )

    def test_conflicting_baseline_references_are_rejected(self) -> None:
        results = synthetic_trigger_results(self.manifest, sign_change=False)
        duplicate = results.loc[results["variant"].eq("baseline")].iloc[[0]].copy()
        duplicate["objective_eur_year"] = 1_001.0
        with self.assertRaisesRegex(ValueError, "baseline references disagree"):
            validate_results(
                pd.concat([results, duplicate], ignore_index=True),
                mode="trigger",
            )


class BoundaryAggregationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_catalog(DEFAULT_CATALOG)
        cls.mapping = load_weather_mapping()

    def test_complete_synthetic_valley_and_positive_delta_w(self) -> None:
        values = synthetic_complete_unit_values(self.catalog)
        result = boundary_analysis(
            values,
            self.mapping,
            bootstrap_draws=40,
            block_length=3,
            seed=2026072603,
            minimum_valid_delta_draw_fraction=0.95,
        )
        boundaries = result["population_boundaries"].set_index("chronology")
        self.assertTrue(bool(boundaries.loc["dispersed", "confirmed"]))
        self.assertTrue(bool(boundaries.loc["clustered", "confirmed"]))
        self.assertEqual(float(boundaries.loc["dispersed", "h_T_h"]), 24.0)
        self.assertEqual(float(boundaries.loc["clustered", "h_T_h"]), 8.0)
        self.assertEqual(float(boundaries.loc["dispersed", "h_I_h"]), 72.0)
        self.assertEqual(float(boundaries.loc["clustered", "h_I_h"]), 72.0)
        delta = result["delta_w_summary"]
        self.assertAlmostEqual(
            float(delta["delta_W_point"]),
            math.log(72.0 / 8.0) - math.log(72.0 / 24.0),
        )
        self.assertGreater(float(delta["delta_W_ci_lower"]), 0.0)

    def test_right_censoring_is_not_zero_width(self) -> None:
        values = synthetic_complete_unit_values(self.catalog)
        values["bj"] = np.where(values["duration_h"] <= 24.0, 4.0, -3.0)
        values["ba"] = -1.0
        values["elcc"] = -1.0
        result = boundary_analysis(
            values,
            self.mapping,
            bootstrap_draws=20,
            seed=9,
        )
        boundaries = result["population_boundaries"]
        self.assertFalse(boundaries["confirmed"].any())
        self.assertTrue(boundaries["W_log_ratio"].isna().all())
        delta = result["delta_w_summary"]
        self.assertTrue(math.isnan(float(delta["delta_W_point"])))
        self.assertEqual(
            delta["primary_case_status"],
            "population_boundary_unidentified_or_censored",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
