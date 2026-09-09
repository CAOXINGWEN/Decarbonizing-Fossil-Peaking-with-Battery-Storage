#!/usr/bin/env python3
"""Build representative full-year dispatch profiles for the Figure 3 preview."""

from __future__ import annotations

import os
import sys
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "portfolio_model"))
sys.path.insert(0, str(ROOT / "joint_choice"))
sys.path.insert(0, str(ROOT / "price_channel"))

import run_v42_europe_portfolio_gurobi as base  # noqa: E402
import run_bounded_s20_highs as highs  # noqa: E402
import run_price_channel_revaluation_v0_1 as price  # noqa: E402


SOURCE = HERE.parent / "source_data_v58"
OUT = HERE / "source_data"
PROFILE_PATH = OUT / "figure3_representative_dispatch_profiles.csv.gz"
SELECTION_PATH = OUT / "figure3_representative_dispatch_selection.csv"
DECOMPOSITION_PATH = OUT / "figure3_value_component_effects.csv.gz"
FIRM_VALUE_EUR_KW_YEAR = 103.79921785880974


def load_panel(regime: str) -> pd.DataFrame:
    common = [
        "source_hour_id",
        "source_day_id",
        "season_segment",
        "event_flag_q75",
        base.BASE_TASK_COLUMN,
        base.TAIL_TASK_COLUMN,
        base.RECHARGE_COLUMN,
        "forced_discharge_power_mw",
    ]
    if regime == "Europe":
        frame = pd.read_csv(base.DEFAULT_PANEL, usecols=["country", "year", *common])
        frame["system_id"] = frame["country"].astype(str)
        frame["observation_year"] = frame["year"].astype(int)
    elif regime == "China":
        path = ROOT / "cross_scale_cases" / "china_province_panel" / "china_province_event_task_panel.csv.gz"
        frame = pd.read_csv(path, usecols=["province", "year", *common])
        frame["system_id"] = "CN-" + frame["province"].astype(str)
        frame["observation_year"] = frame["year"].astype(int)
    elif regime == "Japan":
        path = ROOT / "cross_scale_cases" / "japan_community_panel" / "japan_community_event_task_panel.csv.gz"
        frame = pd.read_csv(path, usecols=["prefecture", "mesh_id", "fiscal_year", *common])
        frame["system_id"] = (
            "JP-" + frame["prefecture"].astype(str) + "-" + frame["mesh_id"].astype(str)
        )
        frame["observation_year"] = frame["fiscal_year"].astype(int)
    else:
        raise ValueError(regime)
    return frame.sort_values(
        ["system_id", "observation_year", "source_hour_id"], kind="stable"
    ).reset_index(drop=True)


def day_orders(frame: pd.DataFrame, cluster_days: int = 4) -> dict[str, list[int]]:
    daily = (
        frame.groupby("source_day_id", sort=True)
        .agg(segment=("season_segment", "first"), event=("event_flag_q75", "max"))
        .reset_index()
    )
    if len(daily) != 365:
        raise RuntimeError("Representative system-year must contain 365 complete days")
    dispersed: list[int] = []
    clustered: list[int] = []
    for segment in sorted(daily["segment"].unique()):
        subset = daily.loc[daily["segment"].eq(segment)]
        event_days = subset.loc[subset["event"].astype(bool), "source_day_id"].astype(int).tolist()
        normal_days = subset.loc[~subset["event"].astype(bool), "source_day_id"].astype(int).tolist()
        dispersed.extend(base.interleave(event_days, normal_days))
        clustered.extend(base.interleave_clusters(event_days, normal_days, cluster_days))
    expected = list(range(365))
    for label, order in (("dispersed", dispersed), ("clustered", clustered)):
        if len(order) != 365 or sorted(order) != expected:
            raise RuntimeError(f"Invalid {label} intact-day order")
    return {"dispersed": dispersed, "clustered": clustered}


def solve_fixed_arm_full(
    task: np.ndarray,
    tail_task: np.ndarray,
    recharge: np.ndarray,
    power: float,
    base_config: pd.Series,
    tail_config: pd.Series,
    scenario: dict[str, float],
) -> tuple[dict[str, float], pd.DataFrame]:
    """Registered HiGHS LP with all hourly dispatch and SOC variables returned."""
    base_par = base.model_parameters(base_config)
    tail_par = base.model_parameters(tail_config)
    usable = np.where(task > base.NUMERICAL_ZERO, 0.0, recharge)
    eta_b, eta_t = np.sqrt(base_par["rte"]), np.sqrt(tail_par["rte"])
    ret_b = (1.0 - base_par["self_discharge_pct_day"] / 100.0) ** (1.0 / 24.0)
    ret_t = (1.0 - tail_par["self_discharge_pct_day"] / 100.0) ** (1.0 / 24.0)
    a_eq, b_eq = highs.build_equalities(task, eta_b, eta_t, ret_b, ret_t)
    a_ub, b_ub = highs.build_inequalities(
        usable, tail_par["charge_power_ratio"], tail_par["duration_h"]
    )
    hours = highs.HOURS
    q_b, d_b, s_b, q_t, d_t, s_t, u = [i * hours for i in range(7)]
    p, residual = 7 * hours, 7 * hours + 1
    lower = np.zeros(7 * hours + 2)
    upper = np.full(7 * hours + 2, np.inf)
    upper[q_b:d_b] = np.minimum(usable, base_par["charge_power_ratio"] * power)
    upper[d_b:s_b] = np.minimum(task, power)
    upper[s_b:q_t] = base.USABLE_SOC_FRACTION * base_par["duration_h"] * power
    upper[q_t:d_t] = usable
    upper[d_t:s_t] = tail_task
    upper[u:p] = task
    lower[p] = power
    upper[p] = power
    bounds = np.column_stack([lower, upper])

    objective = np.zeros(7 * hours + 2)
    objective[q_b:d_b] = scenario["charge_price_eur_mwh"]
    objective[d_b:s_b] = base_par["vom_eur_mwh"]
    objective[q_t:d_t] = scenario["charge_price_eur_mwh"]
    objective[d_t:s_t] = tail_par["vom_eur_mwh"]
    objective[u:p] = scenario["energy_value_eur_mwh"]
    objective[p] = 1000.0 * tail_par["fixed_cost_eur_kw_year"]
    objective[residual] = 1000.0 * scenario["firm_value_eur_kw_year"]
    primary = price.robust_linprog(
        objective,
        a_ub=a_ub,
        b_ub=b_ub,
        a_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
    )
    charge_objective = np.zeros_like(objective)
    charge_objective[q_b:d_b] = 1.0
    charge_objective[q_t:d_t] = 1.0
    scale = max(float(np.max(np.abs(objective))), 1.0)
    try:
        selected = price.robust_linprog(
            charge_objective,
            a_ub=vstack(
                [a_ub, csr_matrix((objective / scale).reshape(1, -1))], format="csr"
            ),
            b_ub=np.append(
                b_ub, (primary.fun + highs.PRIMARY_ABS_TOL_EUR) / scale
            ),
            a_eq=a_eq,
            b_eq=b_eq,
            bounds=bounds,
        )
        secondary_success = True
    except RuntimeError:
        selected = primary
        secondary_success = False

    x = selected.x
    schedule = pd.DataFrame(
        {
            "model_hour": np.arange(hours, dtype=int),
            "base_task_mw": task,
            "tail_task_mw": tail_task,
            "available_recharge_mw": recharge,
            "usable_recharge_mw": usable,
            "base_charge_mw": x[q_b:d_b],
            "base_discharge_mw": x[d_b:s_b],
            "base_usable_soc_mwh": x[s_b:q_t],
            "tail_charge_mw": x[q_t:d_t],
            "tail_discharge_mw": x[d_t:s_t],
            "tail_usable_soc_mwh": x[s_t:u],
            "unserved_task_mw": x[u:p],
        }
    )
    summary = {
        "secondary_success": secondary_success,
        "tail_power_mw": power,
        "tail_capacity_mwh": tail_par["duration_h"] * power,
        "tail_usable_capacity_mwh": base.USABLE_SOC_FRACTION * tail_par["duration_h"] * power,
        "tail_cycles": float(schedule["tail_discharge_mw"].sum())
        / (tail_par["duration_h"] * power),
        "maximum_task_balance_error_mw": float(
            np.max(
                np.abs(
                    schedule["base_discharge_mw"]
                    + schedule["tail_discharge_mw"]
                    + schedule["unserved_task_mw"]
                    - task
                )
            )
        ),
    }
    return summary, schedule


def representative_units(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime, frame in records.groupby("regime", sort=False):
        median = float(frame["signed_sequencing_effect_eur_kw_year"].median())
        row = frame.loc[
            (frame["signed_sequencing_effect_eur_kw_year"] - median).abs().idxmin()
        ]
        rows.append(
            {
                "regime": regime,
                "system_id": str(row["system_id"]),
                "observation_year": int(row["observation_year"]),
                "regime_median_effect_eur_kw_year": median,
                "selected_effect_eur_kw_year": float(
                    row["signed_sequencing_effect_eur_kw_year"]
                ),
            }
        )
    return pd.DataFrame(rows)


def decompose_system_cost_arms(
    arms: pd.DataFrame,
    *,
    regime: str,
    keys: list[str],
    id_column: str,
    year_column: str,
    charge_column: str,
    residual_energy_column: str,
    system_cost_column: str,
) -> pd.DataFrame:
    """Decompose the exact candidate-versus-comparator system-cost benefit."""
    components = [
        "fixed_cost",
        "variable_om",
        "charging",
        "energy_service",
        "firm_service",
    ]
    candidate = arms.loc[
        arms["arm_type"].eq("ldes_candidate") & arms["config_id"].eq("caes_24h")
    ].set_index(keys)
    comparator = arms.loc[
        arms["arm_type"].eq("extra_lfp4_comparator")
    ].set_index(keys)
    joined = candidate.join(comparator, lsuffix="_candidate", rsuffix="_comparator")
    denominator = joined["tail_power_mw_candidate"] * 1000.0
    joined["fixed_cost"] = (
        joined["base_fixed_cost_eur_comparator"]
        + joined["tail_fixed_cost_eur_comparator"]
        - joined["base_fixed_cost_eur_candidate"]
        - joined["tail_fixed_cost_eur_candidate"]
    ) / denominator
    joined["variable_om"] = (
        joined["base_vom_cost_eur_comparator"]
        + joined["tail_vom_cost_eur_comparator"]
        - joined["base_vom_cost_eur_candidate"]
        - joined["tail_vom_cost_eur_candidate"]
    ) / denominator
    joined["charging"] = (
        joined[f"{charge_column}_comparator"]
        - joined[f"{charge_column}_candidate"]
    ) / denominator
    joined["energy_service"] = (
        joined[f"{residual_energy_column}_comparator"]
        - joined[f"{residual_energy_column}_candidate"]
    ) / denominator
    joined["firm_service"] = (
        joined["residual_firm_cost_eur_comparator"]
        - joined["residual_firm_cost_eur_candidate"]
    ) / denominator
    joined["total"] = (
        joined[f"{system_cost_column}_comparator"]
        - joined[f"{system_cost_column}_candidate"]
    ) / denominator
    wide = joined.reset_index().pivot(
        index=keys[:-1], columns="chronology", values=[*components, "total"]
    )
    effect = pd.DataFrame(index=wide.index)
    for column in [*components, "total"]:
        effect[column] = wide[(column, "clustered")] - wide[(column, "dispersed")]
    effect = effect.reset_index().rename(
        columns={id_column: "system_id", year_column: "observation_year"}
    )
    drop = [column for column in keys[:-1] if column not in {id_column, year_column}]
    effect = effect.drop(columns=drop)
    effect.insert(0, "regime", regime)
    return effect


def build_value_decomposition() -> pd.DataFrame:
    components = [
        "fixed_cost",
        "variable_om",
        "charging",
        "energy_service",
        "firm_service",
    ]
    output = [
        decompose_system_cost_arms(
            pd.read_csv(
                ROOT / "portfolio_model/full_132cy_fixed_s20_central/portfolio_arm_results.csv.gz"
            ),
            regime="Europe",
            keys=["country", "year", "chronology"],
            id_column="country",
            year_column="year",
            charge_column="charge_opportunity_cost_eur",
            residual_energy_column="residual_energy_cost_eur",
            system_cost_column="system_cost_eur",
        )
    ]
    for regime, relative_path in (
        ("China", "cross_scale_recharge_v58/china_r1/cross_scale_results.csv.gz"),
        ("Japan", "cross_scale_recharge_v58/japan_r1/cross_scale_results.csv.gz"),
    ):
        output.append(
            decompose_system_cost_arms(
                pd.read_csv(ROOT / relative_path),
                regime=regime,
                keys=["application_case", "case_id", "case_year", "chronology"],
                id_column="case_id",
                year_column="case_year",
                charge_column="S0_charge_cost_eur",
                residual_energy_column="S0_residual_energy_cost_eur",
                system_cost_column="S0_system_cost_eur",
            )
        )

    result = pd.concat(output, ignore_index=True)
    result["component_sum"] = result[components].sum(axis=1)
    if float((result["component_sum"] - result["total"]).abs().max()) > 1.0e-6:
        raise RuntimeError("Value-component identity failed")
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = pd.read_csv(SOURCE / "figure3_24h_value_cycle_records.csv")
    selected = representative_units(records)
    catalogue = pd.read_csv(base.DEFAULT_CATALOGUE)
    base_config, candidates = base.select_catalogue(catalogue, ["caes_24h"])
    tail_config = candidates.iloc[0]
    scenario = base.load_scenario(base.DEFAULT_SCENARIO, 20)

    profile_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    for unit in selected.itertuples(index=False):
        panel = load_panel(unit.regime)
        frame = panel.loc[
            panel["system_id"].eq(unit.system_id)
            & panel["observation_year"].eq(unit.observation_year)
        ].copy()
        if len(frame) != 8760:
            raise RuntimeError(
                f"{unit.regime} {unit.system_id} {unit.observation_year}: {len(frame)} rows"
            )
        orders = day_orders(frame, cluster_days=4)
        original_task = frame[base.BASE_TASK_COLUMN].to_numpy(float)
        original_tail = frame[base.TAIL_TASK_COLUMN].to_numpy(float)
        original_recharge = frame[base.RECHARGE_COLUMN].to_numpy(float)
        power = float(frame["forced_discharge_power_mw"].iloc[0])
        for chronology, order in orders.items():
            task = base.reorder_days(original_task, order)
            tail_task = base.reorder_days(original_tail, order)
            recharge = base.reorder_days(original_recharge, order)
            summary, schedule = solve_fixed_arm_full(
                task,
                tail_task,
                recharge,
                power,
                base_config,
                tail_config,
                scenario,
            )
            schedule.insert(0, "chronology", chronology)
            schedule.insert(0, "observation_year", unit.observation_year)
            schedule.insert(0, "system_id", unit.system_id)
            schedule.insert(0, "regime", unit.regime)
            schedule["tail_soc_pct"] = (
                100.0
                * schedule["tail_usable_soc_mwh"]
                / float(summary["tail_usable_capacity_mwh"])
            )
            profile_frames.append(schedule)
            audit_rows.append(
                {
                    **unit._asdict(),
                    "chronology": chronology,
                    **summary,
                }
            )

    profiles = pd.concat(profile_frames, ignore_index=True)
    profiles.to_csv(
        PROFILE_PATH,
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    pd.DataFrame(audit_rows).to_csv(SELECTION_PATH, index=False)
    decomposition = build_value_decomposition()
    decomposition.to_csv(
        DECOMPOSITION_PATH,
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    if profiles.groupby(["regime", "chronology"]).size().ne(8760).any():
        raise RuntimeError("Incomplete representative dispatch profile")
    if pd.DataFrame(audit_rows)["maximum_task_balance_error_mw"].max() > 1.0e-6:
        raise RuntimeError("Task-balance identity failed")
    print(pd.DataFrame(audit_rows).to_string(index=False))
    print("\nValue decomposition medians")
    print(
        decomposition.groupby("regime")[[*[
            "fixed_cost", "variable_om", "charging", "energy_service", "firm_service"
        ], "total"]].median().to_string()
    )


if __name__ == "__main__":
    main()
