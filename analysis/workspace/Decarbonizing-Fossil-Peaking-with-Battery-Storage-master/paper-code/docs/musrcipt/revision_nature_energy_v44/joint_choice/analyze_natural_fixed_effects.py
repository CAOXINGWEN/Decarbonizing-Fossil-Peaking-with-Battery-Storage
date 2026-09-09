#!/usr/bin/env python3
"""Country fixed-effect test linking natural clustering to CAES-LFP value.

The independent unit is a country-year (n=132) and the resampling block is
country (15 blocks).  Country fixed effects are absorbed by within-country
demeaning.  The analysis is observational and is kept separate from the
paired chronology experiment.
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
RESULTS = HERE / "natural_caes24_s20_fixed_highs" / "natural_focal_arm_results.csv.gz"
PANEL = Path(
    str(__import__("pathlib").Path(__file__).resolve().parents[6] / 'dataset/external_hourly/europe_mid_duration_valley_upgrade_v0_1/processed/observed_gas_event_task_definitions_v0_2.csv.gz')
)
FIXED_VALUES = HERE / "fixed_power_option_values.csv.gz"
BOOTSTRAP_DRAWS = 10_000
SEED = 44_0804


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def longest_run(flags: np.ndarray, segments: np.ndarray) -> int:
    best = run = 0
    previous = None
    for flag, segment in zip(flags, segments):
        if segment != previous:
            run = 0
        run = run + 1 if flag else 0
        best = max(best, run)
        previous = segment
    return int(best)


def geometry() -> pd.DataFrame:
    panel = pd.read_csv(
        PANEL,
        usecols=[
            "country", "year", "source_day_id", "season_segment",
            "event_flag_q75", "task_c1_mw", "task_c3_q75_mw",
            "r2_potential_recharge_mw",
        ],
    )
    rows = []
    for (country, year), group in panel.groupby(["country", "year"], sort=True):
        daily = group.groupby("source_day_id", sort=True).agg(
            segment=("season_segment", "first"),
            event=("event_flag_q75", "max"),
            task=("task_c1_mw", "sum"),
            tail=("task_c3_q75_mw", "sum"),
            recharge=("r2_potential_recharge_mw", "sum"),
        )
        recovery = []
        for _, season in daily.groupby("segment", sort=True):
            event_positions = np.flatnonzero(season.event.to_numpy(bool))
            if len(event_positions) > 1:
                recovery.extend((np.diff(event_positions) - 1) * 24.0)
        recovery_array = np.asarray(recovery, dtype=float)
        rows.append(
            {
                "country": country,
                "year": int(year),
                "natural_longest_event_run_days": longest_run(
                    daily.event.to_numpy(bool), daily.segment.to_numpy()
                ),
                "natural_zero_recovery_window_share": float(
                    np.mean(recovery_array == 0.0)
                ),
                "natural_median_recovery_window_h": float(np.median(recovery_array)),
                "annual_task_mwh": float(daily.task.sum()),
                "annual_tail_task_mwh": float(daily["tail"].sum()),
                "annual_recharge_mwh": float(daily.recharge.sum()),
                "recharge_to_task_ratio": float(daily.recharge.sum() / daily.task.sum()),
            }
        )
    return pd.DataFrame(rows)


def value_outcome() -> pd.DataFrame:
    result = pd.read_csv(RESULTS)
    result["net_value_eur"] = (
        result.energy_value_eur_mwh * result.total_discharge_mwh
        + 1000.0 * result.firm_value_eur_kw_year * result.portfolio_efc_mw
        - result.base_fixed_cost_eur
        - result.tail_fixed_cost_eur
        - result.base_vom_cost_eur
        - result.tail_vom_cost_eur
        - result.charge_price_eur_mwh * result.total_charge_mwh
    )
    wide = result.pivot(
        index=["country", "year", "base_power_mw"],
        columns="config_id",
        values="net_value_eur",
    ).reset_index()
    wide["natural_caes24_minus_lfp4_eur_kw_year"] = (
        wide.caes_24h - wide.li_lfp_4h
    ) / (wide.base_power_mw * 1000.0)
    return wide[
        ["country", "year", "base_power_mw", "natural_caes24_minus_lfp4_eur_kw_year"]
    ]


def within(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return np.column_stack(
        [
            frame[column].to_numpy(float)
            - frame.groupby("country")[column].transform("mean").to_numpy(float)
            for column in columns
        ]
    )


def regress(
    frame: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    model_id: str,
) -> tuple[list[dict[str, object]], np.ndarray]:
    y = within(frame, [outcome]).ravel()
    x = within(frame, predictors)
    coefficient = np.linalg.lstsq(x, y, rcond=None)[0]
    fitted = x @ coefficient
    within_r2 = 1.0 - float(np.sum((y - fitted) ** 2) / np.sum(y**2))
    countries = sorted(frame.country.unique())
    country_index = frame.country.map({country: i for i, country in enumerate(countries)}).to_numpy(int)
    rng = np.random.default_rng(SEED + sum(ord(char) for char in model_id))
    draws = np.full((BOOTSTRAP_DRAWS, len(predictors)), np.nan)
    for draw in range(BOOTSTRAP_DRAWS):
        sampled = rng.integers(0, len(countries), len(countries))
        counts = np.bincount(sampled, minlength=len(countries))
        weights = counts[country_index].astype(float)
        keep = weights > 0
        root_weight = np.sqrt(weights[keep])
        xw = x[keep] * root_weight[:, None]
        yw = y[keep] * root_weight
        if np.linalg.matrix_rank(xw) == len(predictors):
            draws[draw] = np.linalg.lstsq(xw, yw, rcond=None)[0]
    rows = []
    for index, predictor in enumerate(predictors):
        valid = draws[:, index][np.isfinite(draws[:, index])]
        rows.append(
            {
                "model_id": model_id,
                "outcome": outcome,
                "predictor": predictor,
                "coefficient": coefficient[index],
                "bootstrap_ci95_low": np.quantile(valid, 0.025),
                "bootstrap_ci95_high": np.quantile(valid, 0.975),
                "bootstrap_draws_requested": BOOTSTRAP_DRAWS,
                "bootstrap_draws_valid": len(valid),
                "country_blocks": len(countries),
                "country_years": len(frame),
                "within_r2": within_r2,
            }
        )
    return rows, draws


def main() -> None:
    frame = geometry().merge(value_outcome(), on=["country", "year"], validate="one_to_one")
    frame["log_annual_task_mwh"] = np.log(frame.annual_task_mwh)
    frame["log_recharge_to_task_ratio"] = np.log(frame.recharge_to_task_ratio.clip(lower=1.0e-9))
    outcome = "natural_caes24_minus_lfp4_eur_kw_year"
    specifications = [
        ("longest_only", ["natural_longest_event_run_days"]),
        ("zero_window_only", ["natural_zero_recovery_window_share"]),
        (
            "joint_geometry",
            ["natural_longest_event_run_days", "natural_zero_recovery_window_share"],
        ),
        (
            "joint_geometry_plus_scale",
            [
                "natural_longest_event_run_days",
                "natural_zero_recovery_window_share",
                "log_annual_task_mwh",
                "log_recharge_to_task_ratio",
            ],
        ),
    ]
    coefficient_rows = []
    for model_id, predictors in specifications:
        rows, _ = regress(frame, outcome, predictors, model_id)
        coefficient_rows.extend(rows)
    coefficients = pd.DataFrame(coefficient_rows)

    # Direct natural decision-boundary diagnostic among the two focal tails.
    switches = []
    for country, group in frame.groupby("country", sort=True):
        values = group[outcome]
        switches.append(
            {
                "country": country,
                "years": len(group),
                "caes_winning_years": int((values > 0.01).sum()),
                "lfp_winning_years": int((values < -0.01).sum()),
                "near_tie_years": int((values.abs() <= 0.01).sum()),
                "natural_sign_switch_within_country": bool(
                    (values > 0.01).any() and (values < -0.01).any()
                ),
                "natural_value_range_eur_kw_year": float(values.max() - values.min()),
            }
        )
    switch_frame = pd.DataFrame(switches)

    # Synthetic benchmark: paired 4-day chronology effect divided by the
    # controlled three-day increase in longest event run (1 d to 4 d).
    fixed = pd.read_csv(FIXED_VALUES)
    focal = fixed[(fixed.cost_year.eq(2035)) & fixed.config_id.isin(["caes_24h", "li_lfp_4h"])]
    pivot = focal.pivot(
        index=["country", "year", "chronology"],
        columns="config_id",
        values="efc_value_score_eur_kw_year",
    ).reset_index()
    pivot["mix"] = pivot.caes_24h - pivot.li_lfp_4h
    paired = pivot.pivot(index=["country", "year"], columns="chronology", values="mix").reset_index()
    paired["clustered_minus_dispersed_mix_eur_kw_year"] = paired.clustered - paired.dispersed
    synthetic_effect = float(
        paired.groupby("country").clustered_minus_dispersed_mix_eur_kw_year.mean().mean()
    )
    synthetic = {
        "country_block_mean_clustered_minus_dispersed_mix_eur_kw_year": synthetic_effect,
        "controlled_longest_run_change_days": 3.0,
        "implied_slope_eur_kw_year_per_longest_run_day": synthetic_effect / 3.0,
    }

    frame.to_csv(HERE / "natural_fixed_effect_panel.csv", index=False)
    coefficients.to_csv(HERE / "natural_fixed_effect_coefficients.csv", index=False)
    switch_frame.to_csv(HERE / "natural_focal_decision_switches_by_country.csv", index=False)
    qa = {
        "status": "pass",
        "country_years": len(frame),
        "countries": frame.country.nunique(),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "input_sha256": {
            "natural_results": sha256(RESULTS),
            "panel": sha256(PANEL),
            "fixed_values": sha256(FIXED_VALUES),
        },
        "countries_with_natural_caes_lfp_sign_switch": int(
            switch_frame.natural_sign_switch_within_country.sum()
        ),
        "synthetic_benchmark": synthetic,
    }
    (HERE / "natural_fixed_effect_qa.json").write_text(json.dumps(qa, indent=2) + "\n")
    print(coefficients.to_string(index=False))
    print(switch_frame.to_string(index=False))
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
