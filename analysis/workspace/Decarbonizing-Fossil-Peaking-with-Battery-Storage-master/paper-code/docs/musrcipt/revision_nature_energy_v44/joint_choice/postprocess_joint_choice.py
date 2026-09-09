#!/usr/bin/env python3
"""Audit fixed and bounded-power procurement identities under central values.

The script treats the common 4 h LFP layer as sunk across the discrete choice
set and lets the incremental choice be one of: no tail layer, an additional
4 h LFP tail, or one of the 49 released non-LFP configurations.  It reports
both the manuscript EFC-value account and the Gurobi residual-peak system-cost
account.  All value scores are normalized by common base power, with larger
scores indicating a preferred option.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
V44 = HERE.parent
V43 = V44.parent / "revision_nature_energy_v43"
FIXED_INPUT = V43 / "portfolio_model" / "full_132cy_fixed_s20_central" / "portfolio_arm_results.csv.gz"
COST_INPUT = V43 / "cost_and_decomposition" / "cost_catalogue_2025_2035_long.csv"
CHUNK_ROOT = HERE / "optimized_s20_chunks"
OPTIMIZED_FULL = HERE / "optimized_s20_full"
FALLBACK_OPTIMIZED = V43 / "portfolio_model" / "full_132cy_optimized" / "portfolio_arm_results.csv.gz"
TIE_TOL = 0.01  # EUR kW^-1 yr^-1
POWER_ZERO_TOL_MW = 1.0e-7


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})


def load_optimized_chunks() -> tuple[pd.DataFrame, str]:
    required = [CHUNK_ROOT / f"chunk{chunk}" / "qa_summary.json" for chunk in range(1, 5)]
    if not all(path.exists() for path in required):
        frame = pd.read_csv(FALLBACK_OPTIMIZED)
        if set(frame.scenario_id) != {31} or set(frame.tail_power_mode) != {"optimized"}:
            raise RuntimeError("Fallback bounded-power file is not the audited scenario-31 run")
        if len(frame) != 132 * 2 * 51 or set(frame.solver_status_name) != {"optimal"}:
            raise RuntimeError("Fallback bounded-power file does not pass coverage/status checks")
        return frame, "bounded_s31_physics_revalued_at_s20"
    frames: list[pd.DataFrame] = []
    country_year_sets: list[set[tuple[str, int]]] = []
    for chunk in range(1, 5):
        root = CHUNK_ROOT / f"chunk{chunk}"
        qa = json.loads((root / "qa_summary.json").read_text())
        contract = json.loads((root / "run_contract.json").read_text())
        if qa.get("status") != "pass" or not qa.get("checks", {}).get("all_solver_status_optimal", qa.get("all_arm_statuses_optimal", False)):
            raise RuntimeError(f"chunk{chunk} failed its solver QA")
        if contract.get("scenario_id") != 20 or contract.get("tail_power_mode") != "optimized":
            raise RuntimeError(f"chunk{chunk} does not implement central bounded power")
        frame = pd.read_csv(root / "portfolio_arm_results.csv.gz")
        frames.append(frame)
        country_year_sets.append(set(zip(frame.country.astype(str), frame.year.astype(int))))
    for left in range(4):
        for right in range(left + 1, 4):
            if country_year_sets[left] & country_year_sets[right]:
                raise RuntimeError("Optimized chunk country-years overlap")
    result = pd.concat(frames, ignore_index=True)
    if len(set().union(*country_year_sets)) != 132 or len(result) != 132 * 2 * 51:
        raise RuntimeError("Optimized chunks do not close to 132 country-years and 13,464 arms")
    keys = ["country", "year", "chronology", "config_id"]
    if result.duplicated(keys).any() or set(result.solver_status_name) != {"optimal"}:
        raise RuntimeError("Optimized merged output has duplicates or non-optimal rows")
    result = result.sort_values(keys).reset_index(drop=True)
    OPTIMIZED_FULL.mkdir(parents=True, exist_ok=True)
    out = OPTIMIZED_FULL / "portfolio_arm_results.csv.gz"
    write_csv_gz(result, out)
    qa = {
        "status": "pass",
        "scenario_id": 20,
        "tail_power_mode": "optimized_0_to_1x_common_power",
        "country_year_count": 132,
        "arm_row_count": len(result),
        "all_solver_statuses_optimal": True,
        "maximum_task_balance_residual_mw": float(result.maximum_task_balance_residual_mw.abs().max()),
        "maximum_shared_recharge_violation_mw": float(result.maximum_shared_recharge_violation_mw.clip(lower=0).max()),
        "maximum_cyclic_soc_residual_mwh": float(max(result.base_soc_cyclic_residual_mwh.abs().max(), result.tail_soc_cyclic_residual_mwh.abs().max())),
        "output_sha256": sha256(out),
    }
    (OPTIMIZED_FULL / "qa_summary.json").write_text(json.dumps(qa, indent=2) + "\n")
    return result, "bounded_s20_exact"


def cost_lookup() -> tuple[pd.DataFrame, dict[tuple[int, str], float]]:
    cost = pd.read_csv(COST_INPUT)
    value_col = "annual_fixed_cost_eur_source_aware_per_kw_discharge_year"
    if len(cost) != 159 or cost.groupby("learning_year").size().to_dict() != {2025: 53, 2030: 53, 2035: 53}:
        raise RuntimeError("The cost catalogue is not 53 configurations x 3 years")
    lookup = {
        (int(row.learning_year), str(row.config_id)): float(getattr(row, value_col))
        for row in cost.itertuples(index=False)
    }
    return cost, lookup


def revalue_arms(
    frame: pd.DataFrame,
    model_mode: str,
    valuation: dict[str, float],
) -> pd.DataFrame:
    _, lookup = cost_lookup()
    outputs: list[pd.DataFrame] = []
    for cost_year in (2025, 2030, 2035):
        x = frame.copy()
        lfp_rate = lookup[(cost_year, "li_lfp_4h")]
        candidate_rate = x.config_id.map(
            lambda config: lfp_rate if config in {"base_only_lfp4", "li_lfp_4h"} else lookup[(cost_year, str(config))]
        ).astype(float)
        x["cost_year"] = cost_year
        x["model_mode"] = model_mode
        x["base_fixed_cost_revalued_eur"] = lfp_rate * x.base_power_mw * 1000.0
        x["tail_fixed_cost_revalued_eur"] = np.where(
            x.arm_type.eq("base_only"),
            0.0,
            candidate_rate * x.tail_power_mw * 1000.0,
        )
        x["valuation_energy_value_eur_mwh"] = valuation["energy_value_eur_mwh"]
        x["valuation_firm_value_eur_kw_year"] = valuation["firm_value_eur_kw_year"]
        x["valuation_charge_price_eur_mwh"] = valuation["charge_price_eur_mwh"]
        x["efc_net_value_revalued_eur"] = (
            valuation["energy_value_eur_mwh"] * x.total_discharge_mwh
            + 1000.0 * valuation["firm_value_eur_kw_year"] * x.portfolio_efc_mw
            - x.base_fixed_cost_revalued_eur
            - x.tail_fixed_cost_revalued_eur
            - x.base_vom_cost_eur
            - x.tail_vom_cost_eur
            - valuation["charge_price_eur_mwh"] * x.total_charge_mwh
        )
        x["system_cost_revalued_eur"] = (
            x.base_fixed_cost_revalued_eur
            + x.tail_fixed_cost_revalued_eur
            + x.base_vom_cost_eur
            + x.tail_vom_cost_eur
            + valuation["charge_price_eur_mwh"] * x.total_charge_mwh
            + valuation["energy_value_eur_mwh"] * x.unserved_eens_mwh
            + 1000.0 * valuation["firm_value_eur_kw_year"] * x.residual_peak_mw
        )
        base = x[x.arm_type.eq("base_only")][
            ["country", "year", "chronology", "efc_net_value_revalued_eur", "system_cost_revalued_eur"]
        ].rename(
            columns={
                "efc_net_value_revalued_eur": "base_only_efc_net_value_eur",
                "system_cost_revalued_eur": "reference_base_only_system_cost_revalued_eur",
            }
        )
        x = x.merge(base, on=["country", "year", "chronology"], validate="many_to_one")
        denominator = x.base_power_mw * 1000.0
        x["efc_value_score_eur_kw_year"] = (
            x.efc_net_value_revalued_eur - x.base_only_efc_net_value_eur
        ) / denominator
        x["system_cost_score_eur_kw_year"] = (
            x.reference_base_only_system_cost_revalued_eur
            - x.system_cost_revalued_eur
        ) / denominator
        x["selected_tail_power_share"] = x.tail_power_mw / x.base_power_mw
        x["decision_id"] = x.config_id.astype(str)
        x["decision_family"] = np.where(
            x.arm_type.eq("base_only"),
            "no_tail",
            np.where(x.arm_type.eq("extra_lfp4_comparator"), "LFP", "LDES"),
        )
        x["decision_tech_id"] = x.tech_id.astype(str)
        x["decision_duration_h"] = x.duration_h.astype(float)
        zero = x.tail_power_mw.le(POWER_ZERO_TOL_MW) | x.arm_type.eq("base_only")
        x.loc[zero, ["decision_id", "decision_family", "decision_tech_id"]] = ["no_tail", "no_tail", "none"]
        x.loc[zero, "decision_duration_h"] = 0.0
        x["tail_cost_rate_eur_per_common_kw_year"] = np.where(
            zero, 0.0, candidate_rate * x.selected_tail_power_share
        )
        outputs.append(x)
    return pd.concat(outputs, ignore_index=True)


def select_choices(
    values: pd.DataFrame,
    score_column: str,
    account: str,
    decision_scope: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["model_mode", "cost_year", "country", "year", "chronology"]
    choice_rows: list[dict[str, object]] = []
    long_rows: list[pd.DataFrame] = []
    for group_key, group in values.groupby(keys, sort=True):
        # Collapse the many optimized P=0 candidate labels to one no-tail option.
        collapsed = (
            group.sort_values([score_column, "decision_id"], ascending=[False, True])
            .drop_duplicates("decision_id", keep="first")
            .sort_values([score_column, "decision_id"], ascending=[False, True])
            .reset_index(drop=True)
        )
        top = float(collapsed.iloc[0][score_column])
        tied = collapsed[collapsed[score_column].ge(top - TIE_TOL)].copy()
        if "no_tail" in set(tied.decision_id):
            canonical = tied[tied.decision_id.eq("no_tail")].iloc[0]
        else:
            canonical = tied.sort_values("decision_id").iloc[0]
        distinct_alternatives = collapsed[~collapsed.decision_id.eq(canonical.decision_id)]
        runner_up = distinct_alternatives.iloc[0] if len(distinct_alternatives) else canonical
        model_mode, cost_year, country, year, chronology = group_key
        choice_rows.append(
            {
                "model_mode": model_mode,
                "cost_year": cost_year,
                "country": country,
                "year": year,
                "chronology": chronology,
                "account": account,
                "decision_scope": decision_scope,
                "choice_id": canonical.decision_id,
                "choice_family": canonical.decision_family,
                "choice_tech_id": canonical.decision_tech_id,
                "choice_duration_h": canonical.decision_duration_h,
                "choice_tail_power_mw": canonical.tail_power_mw,
                "choice_tail_power_share": canonical.selected_tail_power_share,
                "choice_score_eur_kw_year": canonical[score_column],
                "tie_set": "|".join(sorted(set(tied.decision_id.astype(str)))),
                "tie_set_size": len(set(tied.decision_id.astype(str))),
                "runner_up_id": runner_up.decision_id,
                "runner_up_family": runner_up.decision_family,
                "runner_up_tail_power_share": runner_up.selected_tail_power_share,
                "runner_up_score_eur_kw_year": runner_up[score_column],
                "decision_margin_eur_kw_year": canonical[score_column] - runner_up[score_column],
                "choice_tail_cost_rate_eur_per_common_kw_year": canonical.tail_cost_rate_eur_per_common_kw_year,
                "runner_up_tail_cost_rate_eur_per_common_kw_year": runner_up.tail_cost_rate_eur_per_common_kw_year,
            }
        )
        collapsed = collapsed.assign(
            account=account,
            decision_scope=decision_scope,
            decision_score_eur_kw_year=collapsed[score_column],
        )
        long_rows.append(collapsed)
    return pd.DataFrame(choice_rows), pd.concat(long_rows, ignore_index=True)


def cost_path_band(choice_id: str, runner_up_id: str, model_mode: str, power_share_choice: float, power_share_runner: float) -> float:
    _, lookup = cost_lookup()
    def rate(year: int, option: str, share: float) -> float:
        if option == "no_tail":
            return 0.0
        config = "li_lfp_4h" if option == "li_lfp_4h" else option
        return lookup[(year, config)] * share
    differentials = [
        rate(year, choice_id, power_share_choice) - rate(year, runner_up_id, power_share_runner)
        for year in (2025, 2030, 2035)
    ]
    return float(max(differentials) - min(differentials))


def pair_choices(choices: pd.DataFrame) -> pd.DataFrame:
    index = [
        "model_mode",
        "cost_year",
        "country",
        "year",
        "account",
        "decision_scope",
    ]
    wide = choices.pivot(index=index, columns="chronology").reset_index()
    wide.columns = [
        "_".join(str(part) for part in col if str(part)) if isinstance(col, tuple) else str(col)
        for col in wide.columns
    ]
    d_id = wide.choice_id_dispersed.astype(str)
    c_id = wide.choice_id_clustered.astype(str)
    d_family = wide.choice_family_dispersed.astype(str)
    c_family = wide.choice_family_clustered.astype(str)
    wide["canonical_identity_flip"] = d_id.ne(c_id)
    wide["disjoint_tie_set_flip"] = [
        set(str(d).split("|")).isdisjoint(set(str(c).split("|")))
        for d, c in zip(wide.tie_set_dispersed, wide.tie_set_clustered)
    ]
    wide["family_flip"] = d_family.ne(c_family)
    wide["tail_participation_flip"] = d_family.eq("no_tail").ne(c_family.eq("no_tail"))
    wide["technology_flip"] = wide.choice_tech_id_dispersed.astype(str).ne(wide.choice_tech_id_clustered.astype(str))
    wide["duration_flip"] = wide.choice_duration_h_dispersed.ne(wide.choice_duration_h_clustered)
    direction = np.full(len(wide), "no_flip", dtype=object)
    flip = wide.canonical_identity_flip
    direction[flip & d_family.ne("no_tail") & c_family.eq("no_tail")] = "tail_to_no_tail"
    direction[flip & d_family.eq("no_tail") & c_family.ne("no_tail")] = "no_tail_to_tail"
    direction[flip & d_family.eq("LDES") & c_family.eq("LFP")] = "LDES_to_LFP"
    direction[flip & d_family.eq("LFP") & c_family.eq("LDES")] = "LFP_to_LDES"
    direction[flip & d_family.eq("LDES") & c_family.eq("LDES")] = "within_LDES"
    wide["flip_direction"] = direction
    wide["clustered_minus_dispersed_choice_score_eur_kw_year"] = (
        wide.choice_score_eur_kw_year_clustered - wide.choice_score_eur_kw_year_dispersed
    )
    # The fixed-power cost-path comparison is exact.  For bounded power it is
    # a frozen-power distance and is labeled as such in the report.
    wide["cost_path_span_eur_kw_year"] = [
        cost_path_band(cid, rid, mode, cps, rps)
        for cid, rid, mode, cps, rps in zip(
            wide.choice_id_dispersed,
            wide.runner_up_id_dispersed,
            wide.model_mode,
            wide.choice_tail_power_share_dispersed,
            wide.runner_up_tail_power_share_dispersed,
        )
    ]
    wide["margin_within_cost_path_span"] = (
        wide.decision_margin_eur_kw_year_dispersed <= wide.cost_path_span_eur_kw_year + TIE_TOL
    )
    return wide


def summarize(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in paired.groupby(
        ["model_mode", "cost_year", "account", "decision_scope"], sort=True
    ):
        model_mode, cost_year, account, decision_scope = keys
        country = group.groupby("country").disjoint_tie_set_flip.agg(["sum", "count"])
        seed = 440_804 + int(cost_year) + sum(ord(char) for char in f"{model_mode}{account}{decision_scope}")
        rng = np.random.default_rng(seed)
        weights = rng.multinomial(
            len(country),
            np.full(len(country), 1.0 / len(country)),
            size=10_000,
        )
        block_rates = (weights @ country["sum"].to_numpy(float)) / (
            weights @ country["count"].to_numpy(float)
        )
        rows.append(
            {
                "model_mode": model_mode,
                "cost_year": cost_year,
                "account": account,
                "decision_scope": decision_scope,
                "n_country_years": len(group),
                "canonical_identity_flips": int(group.canonical_identity_flip.sum()),
                "canonical_identity_flip_rate": float(group.canonical_identity_flip.mean()),
                "disjoint_tie_set_flips": int(group.disjoint_tie_set_flip.sum()),
                "disjoint_tie_set_flip_rate": float(group.disjoint_tie_set_flip.mean()),
                "country_block_bootstrap_ci95_low": float(np.quantile(block_rates, 0.025)),
                "country_block_bootstrap_ci95_high": float(np.quantile(block_rates, 0.975)),
                "country_block_bootstrap_draws": 10_000,
                "family_flips": int(group.family_flip.sum()),
                "tail_participation_flips": int(group.tail_participation_flip.sum()),
                "LDES_to_LFP_flips": int(group.flip_direction.eq("LDES_to_LFP").sum()),
                "LFP_to_LDES_flips": int(group.flip_direction.eq("LFP_to_LDES").sum()),
                "tail_to_no_tail_flips": int(group.flip_direction.eq("tail_to_no_tail").sum()),
                "no_tail_to_tail_flips": int(group.flip_direction.eq("no_tail_to_tail").sum()),
                "within_LDES_flips": int(group.flip_direction.eq("within_LDES").sum()),
                "dispersed_margin_median_eur_kw_year": float(group.decision_margin_eur_kw_year_dispersed.median()),
                "dispersed_margin_q25_eur_kw_year": float(group.decision_margin_eur_kw_year_dispersed.quantile(0.25)),
                "dispersed_margin_q75_eur_kw_year": float(group.decision_margin_eur_kw_year_dispersed.quantile(0.75)),
                "margin_within_cost_path_span_count": int(group.margin_within_cost_path_span.sum()),
                "margin_within_cost_path_span_rate": float(group.margin_within_cost_path_span.mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    fixed = pd.read_csv(FIXED_INPUT)
    optimized, optimized_mode = load_optimized_chunks()
    if set(fixed.scenario_id) != {20} or set(fixed.tail_power_mode) != {"fixed"}:
        raise RuntimeError("Fixed input is not the central S20 controlled-power solve")

    valuation = {
        column: float(fixed[column].iloc[0])
        for column in [
            "energy_value_eur_mwh",
            "firm_value_eur_kw_year",
            "charge_price_eur_mwh",
        ]
    }
    all_choice_frames: list[pd.DataFrame] = []
    all_paired_frames: list[pd.DataFrame] = []
    output_manifest: dict[str, str] = {}
    for mode, frame in [("fixed", fixed), (optimized_mode, optimized)]:
        values = revalue_arms(frame, mode, valuation)
        # The bounded 2025/2030 rows are frozen-2035-power revaluations.  The
        # bounded 2035 system-cost rows are the exact joint optimization.
        if mode == "bounded_s31_physics_revalued_at_s20":
            values["evidence_role"] = (
                "scenario31_bounded_power_physics_revalued_at_central_s20__not_reoptimized"
            )
        elif mode == "bounded_s20_exact":
            values["evidence_role"] = np.where(
                values.cost_year.eq(2035),
                "exact_bounded_power_for_system_cost__trajectory_screen_for_efc_value",
                "frozen_2035_bounded_power_revaluation",
            )
        else:
            values["evidence_role"] = "exact_fixed_power_revaluation"
        safe_mode = mode.replace("_physics_revalued_at_s20", "_frozen")
        value_path = HERE / f"{safe_mode}_power_option_values.csv.gz"
        write_csv_gz(values, value_path)
        output_manifest[value_path.name] = sha256(value_path)
        choice_parts: list[pd.DataFrame] = []
        collapsed_parts: list[pd.DataFrame] = []
        for score, account in [
            ("efc_value_score_eur_kw_year", "efc_value"),
            ("system_cost_score_eur_kw_year", "residual_peak_system_cost"),
        ]:
            for decision_scope, subset in [
                ("voluntary_tail", values),
                ("mandatory_tail", values[~values.arm_type.eq("base_only")]),
            ]:
                choice, collapsed = select_choices(
                    subset, score, account, decision_scope
                )
                choice_parts.append(choice)
                collapsed_parts.append(collapsed)
        choices = pd.concat(choice_parts, ignore_index=True)
        paired = pair_choices(choices)
        choice_path = HERE / f"{safe_mode}_power_joint_choices.csv"
        pair_path = HERE / f"{safe_mode}_power_paired_decision_flips.csv"
        choices.to_csv(choice_path, index=False)
        paired.to_csv(pair_path, index=False)
        output_manifest[choice_path.name] = sha256(choice_path)
        output_manifest[pair_path.name] = sha256(pair_path)
        all_choice_frames.append(choices)
        all_paired_frames.append(paired)

    paired_all = pd.concat(all_paired_frames, ignore_index=True)
    summary = summarize(paired_all)
    summary_path = HERE / "joint_choice_flip_summary.csv"
    summary.to_csv(summary_path, index=False)
    output_manifest[summary_path.name] = sha256(summary_path)

    # Per-configuration LDES-versus-LFP sign flips under the controlled scale.
    fixed_values = pd.read_csv(HERE / "fixed_power_option_values.csv.gz")
    candidates = fixed_values[fixed_values.arm_type.eq("ldes_candidate")].copy()
    comparator = fixed_values[fixed_values.arm_type.eq("extra_lfp4_comparator")][
        [
            "cost_year",
            "country",
            "year",
            "chronology",
            "efc_value_score_eur_kw_year",
        ]
    ].rename(columns={"efc_value_score_eur_kw_year": "lfp_score_eur_kw_year"})
    candidates = candidates.merge(
        comparator,
        on=["cost_year", "country", "year", "chronology"],
        validate="many_to_one",
    )
    candidates["mix_advantage_vs_extra_lfp_eur_kw_year"] = (
        candidates.efc_value_score_eur_kw_year - candidates.lfp_score_eur_kw_year
    )
    cell = candidates.pivot(
        index=["cost_year", "country", "year", "config_id", "tech_id", "duration_h"],
        columns="chronology",
        values="mix_advantage_vs_extra_lfp_eur_kw_year",
    ).reset_index()
    cell["strict_sign_flip"] = (
        (cell.dispersed > TIE_TOL) & (cell.clustered < -TIE_TOL)
    ) | ((cell.dispersed < -TIE_TOL) & (cell.clustered > TIE_TOL))
    cell["flip_direction"] = np.where(
        (cell.dispersed > TIE_TOL) & (cell.clustered < -TIE_TOL),
        "LDES_to_LFP",
        np.where(
            (cell.dispersed < -TIE_TOL) & (cell.clustered > TIE_TOL),
            "LFP_to_LDES",
            "no_flip",
        ),
    )
    cell_path = HERE / "fixed_power_configuration_sign_flips.csv.gz"
    write_csv_gz(cell, cell_path)
    output_manifest[cell_path.name] = sha256(cell_path)

    qa = {
        "status": "pass",
        "tie_tolerance_eur_kw_year": TIE_TOL,
        "fixed_input_sha256": sha256(FIXED_INPUT),
        "cost_input_sha256": sha256(COST_INPUT),
        "fixed_country_years": int(fixed[["country", "year"]].drop_duplicates().shape[0]),
        "bounded_country_years": int(optimized[["country", "year"]].drop_duplicates().shape[0]),
        "non_lfp_configuration_count": int(fixed[fixed.arm_type.eq("ldes_candidate")].config_id.nunique()),
        "cost_rows": int(pd.read_csv(COST_INPUT).shape[0]),
        "output_sha256": output_manifest,
    }
    (HERE / "joint_choice_qa.json").write_text(json.dumps(qa, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
