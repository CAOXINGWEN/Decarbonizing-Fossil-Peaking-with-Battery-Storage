#!/usr/bin/env python3
"""Reality-anchored decision-flip scan for the v44 fixed-power catalogue.

The analysis keeps the hourly physical trajectories fixed and rebuilds the
absolute EFC-based value ledger for every arm.  Three mutually exclusive
valuation branches are reported:

1. annual firm-capacity product value (vF),
2. direct reliability welfare (country VOLL times avoided EENS, with vE=vF=0),
3. high energy-replacement value and zero charge-opportunity cost.

Branch 3 is evaluated twice: first on the S20 trajectories (a frozen-path
revaluation), then on the trajectories solved under the corresponding S31
dispatch objective.  This makes the distinction between revaluation and
scenario-consistent dispatch explicit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
V44 = HERE.parent
PORTFOLIO = V44 / "portfolio_model"
S20_ARMS = PORTFOLIO / "full_132cy_fixed_s20_central" / "portfolio_arm_results.csv.gz"
S31_ARMS = PORTFOLIO / "full_132cy_fixed" / "portfolio_arm_results.csv.gz"
COSTS = V44 / "cost_and_decomposition" / "cost_catalogue_2025_2035_long.csv"
VOLL = V44 / "value_anchors" / "official_voll_anchors.csv"
CAPACITY_ANCHORS = (
    V44 / "v44_analysis" / "value_anchors" / "capacity_market_anchors.csv"
)
JOINT_SUMMARY = V44 / "joint_choice" / "joint_choice_flip_summary.csv"

YEARS = (2025, 2030, 2035)
CENTRAL_VE = 65.23332265557728
CENTRAL_CQ = 6.15876989995162
CENTRAL_VF = 103.79921785880974
HIGH_VE = 98.54031839922592
HIGH_CQ = 0.0
CAPACITY_EMPIRICAL_LOW = 15.6946
CAPACITY_EMPIRICAL_HIGH = 90.4316
CAPACITY_SEM_HIGH = 160.577
ACER_FALLBACK_VOLL = 10_000.0
NUMERICAL_EPS_PER_KW = 1.0e-6
TIE_TOL_PER_KW = 0.01

# ACER entries available for countries in the 15-country panel.  Countries
# absent from the ACER table use ACER's declared European planning fallback.
COUNTRY_TO_VOLL_JURISDICTION = {
    "BE": "Belgium",
    "CZ": "Czechia",
    "DE": "Germany-Luxembourg",
    "ES": "Spain",
    "FR": "France",
    "GR": "Greece",
    "IT": "Italy",
    "SI": "Slovenia",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    s20 = pd.read_csv(S20_ARMS)
    s31 = pd.read_csv(S31_ARMS)
    costs = pd.read_csv(COSTS)
    voll = pd.read_csv(VOLL)
    expected_rows = 132 * 2 * 51
    for label, frame in (("S20", s20), ("S31", s31)):
        if len(frame) != expected_rows:
            raise RuntimeError(f"{label}: expected {expected_rows} arm rows, found {len(frame)}")
        if not frame["solver_status_name"].eq("optimal").all():
            raise RuntimeError(f"{label}: non-optimal trajectories are present")
        if frame.duplicated(["country", "year", "chronology", "arm_type", "config_id"]).any():
            raise RuntimeError(f"{label}: arm key is not unique")
        if frame.loc[frame["arm_type"].eq("ldes_candidate"), "config_id"].nunique() != 49:
            raise RuntimeError(f"{label}: expected 49 LDES configurations")
    if costs.groupby("learning_year").size().reindex(YEARS).to_dict() != {
        2025: 53,
        2030: 53,
        2035: 53,
    }:
        raise RuntimeError("Cost catalogue does not contain 53 configurations per year")
    return s20, s31, costs, voll


def build_country_voll(arms: pd.DataFrame, voll: pd.DataFrame) -> pd.DataFrame:
    # Prefer the single harmonised ACER 2024 cross-country table.  The source
    # file also retains a 2021 Belgian CREG survey value as a methodological
    # lower sensitivity; it must not silently overwrite Belgium's ACER value.
    acer = voll.loc[voll["source_type"].str.startswith("ACER", na=False)].copy()
    if acer["jurisdiction"].duplicated().any():
        raise RuntimeError("ACER VOLL jurisdiction is not unique")
    lookup = acer.set_index("jurisdiction")["reported_voll_eur_per_mwh"].to_dict()
    rows = []
    for country in sorted(arms["country"].unique()):
        jurisdiction = COUNTRY_TO_VOLL_JURISDICTION.get(country)
        if jurisdiction is None:
            value = ACER_FALLBACK_VOLL
            status = "ACER European fallback"
            source_jurisdiction = "European fallback"
        else:
            value = float(lookup[jurisdiction])
            status = "ACER country/bidding-zone value"
            source_jurisdiction = jurisdiction
        rows.append(
            {
                "country": country,
                "voll_eur_mwh": value,
                "voll_assignment_status": status,
                "voll_source_jurisdiction": source_jurisdiction,
            }
        )
    return pd.DataFrame(rows)


def cost_map(costs: pd.DataFrame, cost_year: int) -> pd.Series:
    selected = costs.loc[
        costs["learning_year"].eq(cost_year),
        ["config_id", "annual_fixed_cost_eur_source_aware_per_kw_discharge_year"],
    ]
    if selected["config_id"].duplicated().any():
        raise RuntimeError(f"Duplicate cost configuration in {cost_year}")
    return selected.set_index("config_id")[
        "annual_fixed_cost_eur_source_aware_per_kw_discharge_year"
    ]


def revalue_arms(
    arms: pd.DataFrame,
    costs: pd.DataFrame,
    country_voll: pd.DataFrame,
    *,
    branch_id: str,
    dispatch_basis: str,
    cost_year: int,
    energy_value: float,
    charge_price: float,
    firm_value: float = 0.0,
    reliability_mode: bool = False,
) -> pd.DataFrame:
    """Rebuild absolute annual arm value on stored physical trajectories."""

    work = arms.copy()
    cmap = cost_map(costs, cost_year)
    lfp_fixed = float(cmap.loc["li_lfp_4h"])
    candidate_fixed = work["config_id"].map(cmap)
    candidate_missing = (
        work["arm_type"].eq("ldes_candidate") & candidate_fixed.isna()
    )
    if candidate_missing.any():
        missing = sorted(work.loc[candidate_missing, "config_id"].unique())
        raise RuntimeError(f"Missing candidate costs in {cost_year}: {missing}")

    work["base_fixed_cost_revalued_eur"] = (
        lfp_fixed * work["base_power_mw"] * 1000.0
    )
    tail_fixed_rate = np.select(
        [
            work["arm_type"].eq("base_only"),
            work["arm_type"].eq("extra_lfp4_comparator"),
        ],
        [0.0, lfp_fixed],
        default=candidate_fixed,
    )
    work["tail_fixed_cost_revalued_eur"] = (
        tail_fixed_rate.astype(float) * work["tail_power_mw"] * 1000.0
    )
    work = work.merge(country_voll, on="country", how="left", validate="many_to_one")
    if work["voll_eur_mwh"].isna().any():
        raise RuntimeError("Missing VOLL assignment")

    work["energy_return_eur"] = (
        0.0 if reliability_mode else energy_value * work["total_discharge_mwh"]
    )
    work["firm_return_eur"] = (
        0.0
        if reliability_mode
        else firm_value * 1000.0 * work["portfolio_efc_mw"]
    )
    # Adding VOLL times served task energy is algebraically equivalent to
    # VOLL times avoided EENS in every pairwise comparison because annual task
    # energy is common to all arms within a country-year chronology.
    work["reliability_return_eur"] = (
        work["voll_eur_mwh"]
        * (work["base_task_mwh"] - work["unserved_eens_mwh"])
        if reliability_mode
        else 0.0
    )
    work["charge_cost_revalued_eur"] = charge_price * work["total_charge_mwh"]
    work["variable_om_cost_eur"] = work["base_vom_cost_eur"] + work["tail_vom_cost_eur"]
    work["net_value_revalued_eur"] = (
        work["energy_return_eur"]
        + work["firm_return_eur"]
        + work["reliability_return_eur"]
        - work["base_fixed_cost_revalued_eur"]
        - work["tail_fixed_cost_revalued_eur"]
        - work["variable_om_cost_eur"]
        - work["charge_cost_revalued_eur"]
    )
    work["common_base_power_kw"] = work["base_power_mw"] * 1000.0
    work["net_value_revalued_eur_kw_year"] = (
        work["net_value_revalued_eur"] / work["common_base_power_kw"]
    )
    work["branch_id"] = branch_id
    work["dispatch_basis"] = dispatch_basis
    work["valuation_mode"] = (
        "scenario_consistent_fixed_power_dispatch"
        if dispatch_basis == "S31_high_value_solved"
        else "frozen_trajectory_revaluation"
    )
    work["cost_year"] = int(cost_year)
    work["valuation_energy_value_eur_mwh"] = 0.0 if reliability_mode else float(energy_value)
    work["valuation_charge_price_eur_mwh"] = float(charge_price)
    work["valuation_firm_value_eur_kw_year"] = 0.0 if reliability_mode else float(firm_value)
    work["reliability_mode"] = bool(reliability_mode)
    return work


def rank_first(frame: pd.DataFrame, subset: pd.Series | None = None) -> pd.DataFrame:
    work = frame if subset is None else frame.loc[subset]
    keys = ["branch_id", "dispatch_basis", "cost_year", "country", "year", "chronology"]
    return (
        work.sort_values(
            keys + ["net_value_revalued_eur", "config_id"],
            ascending=[True] * len(keys) + [False, True],
            kind="stable",
        )
        .drop_duplicates(keys, keep="first")
        .copy()
    )


def select_with_ties(
    frame: pd.DataFrame,
    subset: pd.Series | None = None,
    *,
    prefer_no_tail: bool = False,
) -> pd.DataFrame:
    """Select a canonical option and retain the full 0.01-EUR tie set.

    This follows the registered joint-choice convention: all options within
    0.01 EUR kW-1 yr-1 of the top score form a set-valued optimum.  If the
    no-tail option belongs to that set it is the conservative canonical
    choice; otherwise the lexicographically first configuration is used.
    """

    work = (frame if subset is None else frame.loc[subset]).copy()
    keys = ["branch_id", "dispatch_basis", "cost_year", "country", "year", "chronology"]
    score = "net_value_revalued_eur_kw_year"
    work["_top_score"] = work.groupby(keys, sort=False)[score].transform("max")
    tied = work.loc[work[score].ge(work["_top_score"] - TIE_TOL_PER_KW)].copy()
    tied["_canonical_priority"] = np.where(
        prefer_no_tail & tied["arm_type"].eq("base_only"), 0, 1
    )
    canonical = (
        tied.sort_values(
            keys + ["_canonical_priority", "config_id"],
            ascending=[True] * len(keys) + [True, True],
            kind="stable",
        )
        .drop_duplicates(keys, keep="first")
        .copy()
    )
    class_map = {
        "base_only": "no_tail",
        "extra_lfp4_comparator": "additional_lfp4",
        "ldes_candidate": "ldes",
    }
    tied["_choice_class"] = tied["arm_type"].map(class_map)
    tied["_duration_token"] = tied["duration_h"].map(lambda value: f"{float(value):g}")

    def joined(values: pd.Series) -> str:
        return "|".join(sorted(set(values.astype(str))))

    tie_sets = (
        tied.groupby(keys, sort=True)
        .agg(
            tie_set=("config_id", joined),
            tie_class_set=("_choice_class", joined),
            tie_tech_set=("tech_id", joined),
            tie_duration_set=("_duration_token", joined),
            tie_set_size=("config_id", lambda values: int(values.nunique())),
        )
        .reset_index()
    )
    canonical = canonical.merge(tie_sets, on=keys, how="left", validate="one_to_one")
    return canonical.drop(columns=["_top_score", "_canonical_priority"], errors="ignore")


def build_choices(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["branch_id", "dispatch_basis", "cost_year", "country", "year", "chronology"]
    all_best = select_with_ties(frame, prefer_no_tail=True)
    ldes = select_with_ties(frame, frame["arm_type"].eq("ldes_candidate"))
    lfp = frame.loc[frame["arm_type"].eq("extra_lfp4_comparator")].copy()
    base = frame.loc[frame["arm_type"].eq("base_only")].copy()
    tail = select_with_ties(frame, ~frame["arm_type"].eq("base_only"))

    def extract(source: pd.DataFrame, prefix: str) -> pd.DataFrame:
        cols = keys + [
            "arm_type",
            "config_id",
            "tech_id",
            "duration_h",
            "tail_power_mw",
            "unserved_eens_mwh",
            "portfolio_efc_mw",
            "net_value_revalued_eur",
            "net_value_revalued_eur_kw_year",
            "voll_eur_mwh",
            "voll_assignment_status",
            "valuation_energy_value_eur_mwh",
            "valuation_charge_price_eur_mwh",
            "valuation_firm_value_eur_kw_year",
            "valuation_mode",
        ]
        for optional in (
            "tie_set",
            "tie_class_set",
            "tie_tech_set",
            "tie_duration_set",
            "tie_set_size",
        ):
            if optional in source.columns:
                cols.append(optional)
        return source[cols].rename(
            columns={column: f"{prefix}{column}" for column in cols if column not in keys}
        )

    out = extract(all_best, "selected_")
    for source, prefix in ((ldes, "best_ldes_"), (lfp, "lfp_"), (base, "no_tail_"), (tail, "best_tail_")):
        out = out.merge(extract(source, prefix), on=keys, how="left", validate="one_to_one")

    out["selected_class"] = out["selected_arm_type"].map(
        {
            "base_only": "no_tail",
            "extra_lfp4_comparator": "additional_lfp4",
            "ldes_candidate": "ldes",
        }
    )
    # Untoleranced identities are retained only as an audit trail.  The
    # manuscript-facing counts use the set-valued 0.01-EUR rule above.
    out["selected_class_untoleranced"] = np.where(
        out["best_tail_net_value_revalued_eur_kw_year"]
        > out["no_tail_net_value_revalued_eur_kw_year"] + NUMERICAL_EPS_PER_KW,
        out["best_tail_arm_type"].map(
            {
                "extra_lfp4_comparator": "additional_lfp4",
                "ldes_candidate": "ldes",
            }
        ),
        "no_tail",
    )
    out["selected_config_id_untoleranced"] = np.where(
        out["best_tail_net_value_revalued_eur_kw_year"]
        > out["no_tail_net_value_revalued_eur_kw_year"] + NUMERICAL_EPS_PER_KW,
        out["best_tail_config_id"],
        out["no_tail_config_id"],
    )
    out["ldes_minus_lfp_eur_kw_year"] = (
        out["best_ldes_net_value_revalued_eur_kw_year"]
        - out["lfp_net_value_revalued_eur_kw_year"]
    )
    out["best_tail_minus_no_tail_eur_kw_year"] = (
        out["best_tail_net_value_revalued_eur_kw_year"]
        - out["no_tail_net_value_revalued_eur_kw_year"]
    )
    out["ldes_vs_lfp_choice"] = np.select(
        [
            out["ldes_minus_lfp_eur_kw_year"] > TIE_TOL_PER_KW,
            out["ldes_minus_lfp_eur_kw_year"] < -TIE_TOL_PER_KW,
        ],
        ["ldes", "additional_lfp4"],
        default="tie",
    )
    out["tail_entry_choice"] = np.select(
        [
            out["best_tail_minus_no_tail_eur_kw_year"] > TIE_TOL_PER_KW,
            out["best_tail_minus_no_tail_eur_kw_year"] < -TIE_TOL_PER_KW,
        ],
        ["tail_enters", "no_tail"],
        default="tie",
    )
    return out.sort_values(keys).reset_index(drop=True)


def pair_choices(choices: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["branch_id", "dispatch_basis", "cost_year", "country", "year"]
    value_cols = [column for column in choices.columns if column not in id_cols + ["chronology"]]
    wide = choices.pivot(index=id_cols, columns="chronology", values=value_cols)
    wide.columns = [f"{metric}_{chronology}" for metric, chronology in wide.columns]
    wide = wide.reset_index()

    def disjoint(left: pd.Series, right: pd.Series) -> pd.Series:
        return pd.Series(
            [
                set(str(a).split("|")).isdisjoint(set(str(b).split("|")))
                for a, b in zip(left, right)
            ],
            index=left.index,
        )

    # Set-valued robust flips.  Canonical identities are still exported, but
    # a flip is counted only if the dispersed and clustered 0.01-EUR tie sets
    # do not overlap.
    set_comparisons = {
        "selected_class_flip": "selected_tie_class_set",
        "selected_config_flip": "selected_tie_set",
        "selected_tech_flip": "selected_tie_tech_set",
        "selected_duration_flip": "selected_tie_duration_set",
        "best_ldes_config_flip": "best_ldes_tie_set",
        "best_ldes_tech_flip": "best_ldes_tie_tech_set",
        "best_ldes_duration_flip": "best_ldes_tie_duration_set",
    }
    direction_metrics = {
        "selected_class_flip": "selected_class",
        "selected_config_flip": "selected_config_id",
        "selected_tech_flip": "selected_tech_id",
        "selected_duration_flip": "selected_duration_h",
        "best_ldes_config_flip": "best_ldes_config_id",
        "best_ldes_tech_flip": "best_ldes_tech_id",
        "best_ldes_duration_flip": "best_ldes_duration_h",
    }
    for output, tie_metric in set_comparisons.items():
        wide[output] = disjoint(
            wide[f"{tie_metric}_dispersed"], wide[f"{tie_metric}_clustered"]
        )
        metric = direction_metrics[output]
        wide[f"{output}_direction"] = (
            wide[f"{metric}_dispersed"].astype(str)
            + "_to_"
            + wide[f"{metric}_clustered"].astype(str)
        )

    # LDES-versus-LFP and tail-entry changes use strict sign crossings beyond
    # the same tolerance.  A tie-to-choice transition is not a robust flip.
    wide["ldes_vs_lfp_flip"] = (
        (
            wide["ldes_minus_lfp_eur_kw_year_dispersed"] > TIE_TOL_PER_KW
        )
        & (
            wide["ldes_minus_lfp_eur_kw_year_clustered"] < -TIE_TOL_PER_KW
        )
    ) | (
        (
            wide["ldes_minus_lfp_eur_kw_year_dispersed"] < -TIE_TOL_PER_KW
        )
        & (
            wide["ldes_minus_lfp_eur_kw_year_clustered"] > TIE_TOL_PER_KW
        )
    )
    wide["tail_entry_exit_flip"] = (
        (
            wide["best_tail_minus_no_tail_eur_kw_year_dispersed"] > TIE_TOL_PER_KW
        )
        & (
            wide["best_tail_minus_no_tail_eur_kw_year_clustered"] < -TIE_TOL_PER_KW
        )
    ) | (
        (
            wide["best_tail_minus_no_tail_eur_kw_year_dispersed"] < -TIE_TOL_PER_KW
        )
        & (
            wide["best_tail_minus_no_tail_eur_kw_year_clustered"] > TIE_TOL_PER_KW
        )
    )
    for output, metric in (
        ("ldes_vs_lfp_flip", "ldes_vs_lfp_choice"),
        ("tail_entry_exit_flip", "tail_entry_choice"),
    ):
        wide[f"{output}_direction"] = (
            wide[f"{metric}_dispersed"].astype(str)
            + "_to_"
            + wide[f"{metric}_clustered"].astype(str)
        )

    wide["selected_class_flip_untoleranced"] = wide[
        "selected_class_untoleranced_dispersed"
    ].ne(wide["selected_class_untoleranced_clustered"])
    wide["selected_config_flip_untoleranced"] = wide[
        "selected_config_id_untoleranced_dispersed"
    ].ne(wide["selected_config_id_untoleranced_clustered"])
    wide["delta_ldes_minus_lfp_clustered_minus_dispersed_eur_kw_year"] = (
        wide["ldes_minus_lfp_eur_kw_year_clustered"]
        - wide["ldes_minus_lfp_eur_kw_year_dispersed"]
    )
    wide["delta_tail_entry_margin_clustered_minus_dispersed_eur_kw_year"] = (
        wide["best_tail_minus_no_tail_eur_kw_year_clustered"]
        - wide["best_tail_minus_no_tail_eur_kw_year_dispersed"]
    )
    return wide.sort_values(id_cols).reset_index(drop=True)


def summarize_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    group = ["branch_id", "dispatch_basis", "cost_year"]
    rows = []
    for key, frame in pairs.groupby(group, sort=True):
        row = dict(zip(group, key))
        row.update(
            {
                "n_country_years": len(frame),
                "ldes_vs_lfp_flip_count": int(frame["ldes_vs_lfp_flip"].sum()),
                "ldes_vs_lfp_flip_rate": float(frame["ldes_vs_lfp_flip"].mean()),
                "selected_class_flip_count": int(frame["selected_class_flip"].sum()),
                "selected_class_flip_rate": float(frame["selected_class_flip"].mean()),
                "selected_class_flip_untoleranced_count": int(
                    frame["selected_class_flip_untoleranced"].sum()
                ),
                "selected_config_flip_count": int(frame["selected_config_flip"].sum()),
                "selected_config_flip_rate": float(frame["selected_config_flip"].mean()),
                "selected_config_flip_untoleranced_count": int(
                    frame["selected_config_flip_untoleranced"].sum()
                ),
                "selected_tech_flip_count": int(frame["selected_tech_flip"].sum()),
                "selected_duration_flip_count": int(frame["selected_duration_flip"].sum()),
                "best_ldes_config_flip_count": int(frame["best_ldes_config_flip"].sum()),
                "best_ldes_tech_flip_count": int(frame["best_ldes_tech_flip"].sum()),
                "best_ldes_duration_flip_count": int(frame["best_ldes_duration_flip"].sum()),
                "tail_entry_exit_flip_count": int(frame["tail_entry_exit_flip"].sum()),
                "tail_entry_exit_flip_rate": float(frame["tail_entry_exit_flip"].mean()),
                "median_ldes_lfp_effect_eur_kw_year": float(
                    frame["delta_ldes_minus_lfp_clustered_minus_dispersed_eur_kw_year"].median()
                ),
                "median_tail_entry_effect_eur_kw_year": float(
                    frame["delta_tail_entry_margin_clustered_minus_dispersed_eur_kw_year"].median()
                ),
            }
        )
        for prefix, column in (
            ("ldes_lfp", "ldes_vs_lfp_flip_direction"),
            ("selected_class", "selected_class_flip_direction"),
            ("entry", "tail_entry_exit_flip_direction"),
        ):
            counts = frame.loc[frame[column.replace("_direction", "")], column].value_counts()
            for direction, count in counts.items():
                row[f"{prefix}_{direction}_count"] = int(count)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group).reset_index(drop=True)


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    s20, s31, costs, voll = load_inputs()
    country_voll = build_country_voll(s20, voll)
    country_voll.to_csv(HERE / "country_voll_assignments.csv", index=False)

    arms_valued = []
    vf_grid = sorted(
        set(
            [float(value) for value in range(0, 126, 5)]
            + [CAPACITY_EMPIRICAL_LOW, CENTRAL_VF, CAPACITY_EMPIRICAL_HIGH]
        )
    )
    for year in YEARS:
        for vf in vf_grid:
            arms_valued.append(
                revalue_arms(
                    s20,
                    costs,
                    country_voll,
                    branch_id=f"A_capacity_vf_{vf:g}",
                    dispatch_basis="S20_central_solved",
                    cost_year=year,
                    energy_value=CENTRAL_VE,
                    charge_price=CENTRAL_CQ,
                    firm_value=vf,
                )
            )

        arms_valued.append(
            revalue_arms(
                s20,
                costs,
                country_voll,
                branch_id="B_country_voll_direct_reliability",
                dispatch_basis="S20_central_solved",
                cost_year=year,
                # Direct-reliability ledger: avoided EENS is valued once via
                # country VoLL.  Setting vE explicitly to zero prevents the
                # same served energy from also receiving replacement value.
                energy_value=0.0,
                charge_price=CENTRAL_CQ,
                firm_value=0.0,
                reliability_mode=True,
            )
        )
        arms_valued.append(
            revalue_arms(
                s20,
                costs,
                country_voll,
                branch_id="C_high_energy_zero_charge_frozen_S20",
                dispatch_basis="S20_central_solved",
                cost_year=year,
                energy_value=HIGH_VE,
                charge_price=HIGH_CQ,
                firm_value=CENTRAL_VF,
            )
        )
        arms_valued.append(
            revalue_arms(
                s31,
                costs,
                country_voll,
                branch_id="C_high_energy_zero_charge_reoptimized_S31",
                dispatch_basis="S31_high_value_solved",
                cost_year=year,
                energy_value=HIGH_VE,
                charge_price=HIGH_CQ,
                firm_value=CENTRAL_VF,
            )
        )

    choices = pd.concat([build_choices(frame) for frame in arms_valued], ignore_index=True)
    pairs = pair_choices(choices)
    summary = summarize_pairs(pairs)

    choices.to_csv(HERE / "anchor_scan_chronology_choices.csv.gz", index=False, compression="gzip")
    pairs.to_csv(HERE / "anchor_scan_paired_decisions.csv.gz", index=False, compression="gzip")
    summary.to_csv(HERE / "anchor_scan_summary.csv", index=False)

    capacity = summary.loc[summary["branch_id"].str.startswith("A_capacity")].copy()
    capacity["firm_value_eur_kw_year"] = capacity["branch_id"].str.replace(
        "A_capacity_vf_", "", regex=False
    ).astype(float)
    capacity["within_official_capacity_price_band"] = capacity[
        "firm_value_eur_kw_year"
    ].between(CAPACITY_EMPIRICAL_LOW, CAPACITY_EMPIRICAL_HIGH)
    capacity = capacity.sort_values(
        ["cost_year", "firm_value_eur_kw_year"], kind="stable"
    ).reset_index(drop=True)
    capacity.to_csv(HERE / "capacity_product_flip_curve.csv", index=False)

    observed = pairs.loc[
        pairs[
            [
                "ldes_vs_lfp_flip",
                "selected_class_flip",
                "selected_config_flip",
                "tail_entry_exit_flip",
            ]
        ].any(axis=1)
    ].copy()
    observed.to_csv(HERE / "anchor_scan_observed_flip_cases.csv", index=False)

    # QA: the EFC-ledger revaluation must recover the existing central 2035
    # candidate-minus-LFP cells.
    existing = pd.read_csv(
        V44 / "cost_and_decomposition" / "year_specific_revaluation_cells.csv.gz"
    )
    existing = existing.loc[
        existing["revaluation_case_id"].eq("central_actual_year_cost")
        & existing["cost_year"].eq(2035),
        ["country", "year", "chronology", "config_id", "mix_benefit_revalued_eur_kw_year"],
    ]
    reconstructed = choices.loc[
        choices["branch_id"].eq(f"A_capacity_vf_{CENTRAL_VF:g}")
        & choices["cost_year"].eq(2035),
        [
            "country",
            "year",
            "chronology",
            "best_ldes_config_id",
            "best_ldes_net_value_revalued_eur_kw_year",
            "lfp_net_value_revalued_eur_kw_year",
        ],
    ].copy()
    reconstructed["benefit"] = (
        reconstructed["best_ldes_net_value_revalued_eur_kw_year"]
        - reconstructed["lfp_net_value_revalued_eur_kw_year"]
    )
    check = reconstructed.merge(
        existing,
        left_on=["country", "year", "chronology", "best_ldes_config_id"],
        right_on=["country", "year", "chronology", "config_id"],
        validate="one_to_one",
    )
    central_reconstruction_error = float(
        (check["benefit"] - check["mix_benefit_revalued_eur_kw_year"]).abs().max()
    )

    high_frozen = pairs.loc[
        pairs["branch_id"].eq("C_high_energy_zero_charge_frozen_S20")
        & pairs["cost_year"].eq(2035)
    ]
    high_reoptimized = pairs.loc[
        pairs["branch_id"].eq("C_high_energy_zero_charge_reoptimized_S31")
        & pairs["cost_year"].eq(2035)
    ]
    physical_keys = ["country", "year", "chronology", "arm_type", "config_id"]
    physical = s20.merge(s31, on=physical_keys, suffixes=("_s20", "_s31"), validate="one_to_one")
    physical_differences = {
        metric: float((physical[f"{metric}_s20"] - physical[f"{metric}_s31"]).abs().max())
        for metric in (
            "total_discharge_mwh",
            "total_charge_mwh",
            "unserved_eens_mwh",
            "portfolio_efc_mw",
        )
    }
    central_2035 = pairs.loc[
        pairs["branch_id"].eq(f"A_capacity_vf_{CENTRAL_VF:g}")
        & pairs["cost_year"].eq(2035)
    ]
    empirical_capacity = capacity.loc[
        capacity["within_official_capacity_price_band"]
    ]
    empirical_ranges = {
        str(year): {
            "minimum_robust_selected_config_flips": int(
                empirical_capacity.loc[
                    empirical_capacity["cost_year"].eq(year),
                    "selected_config_flip_count",
                ].min()
            ),
            "maximum_robust_selected_config_flips": int(
                empirical_capacity.loc[
                    empirical_capacity["cost_year"].eq(year),
                    "selected_config_flip_count",
                ].max()
            ),
        }
        for year in YEARS
    }
    joint = pd.read_csv(JOINT_SUMMARY)
    joint = joint.loc[
        joint["model_mode"].eq("fixed")
        & joint["account"].eq("efc_value")
        & joint["decision_scope"].eq("voluntary_tail")
    ]
    joint_counts = {
        str(int(row.cost_year)): int(row.disjoint_tie_set_flips)
        for row in joint.itertuples(index=False)
    }
    anchor_central_counts = {
        str(year): int(
            pairs.loc[
                pairs["branch_id"].eq(f"A_capacity_vf_{CENTRAL_VF:g}")
                & pairs["cost_year"].eq(year),
                "selected_config_flip",
            ].sum()
        )
        for year in YEARS
    }
    checks = {
        "status": "pass",
        "tie_tolerance_eur_kw_year": TIE_TOL_PER_KW,
        "country_year_count": int(s20[["country", "year"]].drop_duplicates().shape[0]),
        "country_count": int(s20["country"].nunique()),
        "configuration_count": int(s20.loc[s20["arm_type"].eq("ldes_candidate"), "config_id"].nunique()),
        "country_specific_acer_voll_count": int(
            country_voll["voll_assignment_status"].eq("ACER country/bidding-zone value").sum()
        ),
        "acer_fallback_voll_count": int(
            country_voll["voll_assignment_status"].eq("ACER European fallback").sum()
        ),
        "belgium_harmonised_acer_voll_eur_mwh": float(
            country_voll.loc[country_voll["country"].eq("BE"), "voll_eur_mwh"].iloc[0]
        ),
        "capacity_vf_grid": vf_grid,
        "capacity_official_band_eur_kw_year": [CAPACITY_EMPIRICAL_LOW, CAPACITY_EMPIRICAL_HIGH],
        "capacity_sem_high_reference_eur_kw_year": CAPACITY_SEM_HIGH,
        "central_vf_percent_above_priority_market_upper_bound": float(
            100.0 * (CENTRAL_VF / CAPACITY_EMPIRICAL_HIGH - 1.0)
        ),
        "central_vf_below_sem_high_reference": bool(CENTRAL_VF < CAPACITY_SEM_HIGH),
        "central_2035_efc_benefit_reconstruction_max_abs_error_eur_kw_year": central_reconstruction_error,
        "reliability_branch_all_vf_zero": bool(
            choices.loc[
                choices["branch_id"].eq("B_country_voll_direct_reliability"),
                "selected_valuation_firm_value_eur_kw_year",
            ].eq(0).all()
        ),
        "reliability_branch_all_ve_zero": bool(
            choices.loc[
                choices["branch_id"].eq("B_country_voll_direct_reliability"),
                "selected_valuation_energy_value_eur_mwh",
            ].eq(0).all()
        ),
        "no_tail_all_tail_power_zero": bool(
            choices["no_tail_tail_power_mw"].abs().le(1.0e-12).all()
        ),
        "all_choice_rows_have_132_country_years_per_chronology": bool(
            choices.groupby(["branch_id", "dispatch_basis", "cost_year", "chronology"]).size().eq(132).all()
        ),
        "all_pair_rows_have_132_country_years": bool(
            pairs.groupby(["branch_id", "dispatch_basis", "cost_year"]).size().eq(132).all()
        ),
        "high_branch_frozen_2035_ldes_lfp_flips": int(high_frozen["ldes_vs_lfp_flip"].sum()),
        "high_branch_reoptimized_2035_ldes_lfp_flips": int(high_reoptimized["ldes_vs_lfp_flip"].sum()),
        "high_branch_75_of_132_verified": bool(
            high_frozen["ldes_vs_lfp_flip"].sum() == 75
            and high_reoptimized["ldes_vs_lfp_flip"].sum() == 75
        ),
        "central_103p799_2035_robust_overall_flips": int(
            central_2035["selected_config_flip"].sum()
        ),
        "central_103p799_2035_untoleranced_overall_flips": int(
            central_2035["selected_config_flip_untoleranced"].sum()
        ),
        "empirical_capacity_band_robust_overall_flip_ranges": empirical_ranges,
        "joint_choice_central_robust_flip_counts": joint_counts,
        "anchor_scan_central_robust_flip_counts": anchor_central_counts,
        "central_counts_match_joint_choice": anchor_central_counts == joint_counts,
        "S20_vs_S31_physical_max_abs_differences": physical_differences,
    }
    failures = []
    # CSV round-tripping of the frozen source ledger introduces errors at the
    # 1e-8 EUR kW-1 yr-1 level; 1e-7 remains far below reporting precision.
    if central_reconstruction_error > 1.0e-7:
        failures.append("central_ledger_reconstruction")
    for boolean_check in (
        "reliability_branch_all_vf_zero",
        "reliability_branch_all_ve_zero",
        "no_tail_all_tail_power_zero",
        "all_choice_rows_have_132_country_years_per_chronology",
        "all_pair_rows_have_132_country_years",
        "high_branch_75_of_132_verified",
        "central_counts_match_joint_choice",
    ):
        if not checks[boolean_check]:
            failures.append(boolean_check)
    if checks["belgium_harmonised_acer_voll_eur_mwh"] != 13_000.0:
        failures.append("belgium_acer_voll_selection")
    if checks["central_103p799_2035_robust_overall_flips"] != 22:
        failures.append("central_2035_robust_flip_count")
    if checks["central_103p799_2035_untoleranced_overall_flips"] != 23:
        failures.append("central_2035_untoleranced_flip_count")
    checks["failures"] = failures
    checks["status"] = "pass" if not failures else "fail"
    with (HERE / "qa_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(checks, handle, indent=2, ensure_ascii=False)

    manifest = {
        "script": Path(__file__).name,
        "analysis_contract": {
            "decision_tie_tolerance_eur_kw_year": TIE_TOL_PER_KW,
            "A_capacity_product": {
                "dispatch": "S20 fixed-power trajectories",
                "vE_eur_mwh": CENTRAL_VE,
                "cQ_eur_mwh": CENTRAL_CQ,
                "vF_grid_eur_kw_year": vf_grid,
            },
            "B_direct_reliability": {
                "dispatch": "S20 fixed-power trajectories",
                "vE_eur_mwh": 0.0,
                "cQ_eur_mwh": CENTRAL_CQ,
                "vF_eur_kw_year": 0.0,
                "coverage_formula": "country VOLL * (EENS_comparator - EENS_candidate)",
                "missing_country_rule": f"ACER European fallback {ACER_FALLBACK_VOLL:g} EUR/MWh",
            },
            "C_high_energy_zero_charge": {
                "vE_eur_mwh": HIGH_VE,
                "cQ_eur_mwh": HIGH_CQ,
                "vF_eur_kw_year": CENTRAL_VF,
                "dispatch_comparisons": ["frozen S20", "scenario-consistent S31"],
            },
        },
        "inputs": {
            str(path.relative_to(V44)): {"sha256": sha256(path)}
            for path in (
                S20_ARMS,
                S31_ARMS,
                COSTS,
                VOLL,
                CAPACITY_ANCHORS,
                JOINT_SUMMARY,
            )
        },
        "outputs": {
            name: {"sha256": sha256(HERE / name)}
            for name in (
                "country_voll_assignments.csv",
                "anchor_scan_chronology_choices.csv.gz",
                "anchor_scan_paired_decisions.csv.gz",
                "anchor_scan_summary.csv",
                "capacity_product_flip_curve.csv",
                "anchor_scan_observed_flip_cases.csv",
                "qa_summary.json",
            )
        },
    }
    with (HERE / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    if failures:
        raise RuntimeError(f"QA failed: {failures}")


if __name__ == "__main__":
    main()
