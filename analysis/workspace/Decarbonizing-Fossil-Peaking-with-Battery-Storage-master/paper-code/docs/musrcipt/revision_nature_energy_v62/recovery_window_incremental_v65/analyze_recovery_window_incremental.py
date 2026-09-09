#!/usr/bin/env python3
"""Compare event-duration and recovery-window models for NVSB natural years.

The observational unit is a country-year (n=132). Country fixed effects are
absorbed by within-country demeaning. Generalization is evaluated by leaving
one country out, fitting slopes on the other 14 countries, and predicting
annual deviations from the held-out country's mean. Country is also the block
used for the 10,000-draw bootstrap.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
INPUT = ROOT / "joint_choice" / "natural_fixed_effect_panel.csv"
RESULTS = HERE / "results"
SOURCE = HERE / "source_data"
BOOTSTRAP_DRAWS = 10_000
SEED = 44_0804
OUTCOME = "natural_caes24_minus_lfp4_eur_kw_year"
COST_COMPARATOR = 5.296

MODELS: dict[str, list[str]] = {
    "longest_event": ["natural_longest_event_run_days"],
    "recovery_window": ["natural_zero_recovery_window_share"],
    "joint_geometry": [
        "natural_longest_event_run_days",
        "natural_zero_recovery_window_share",
    ],
    "joint_controlled": [
        "natural_longest_event_run_days",
        "natural_zero_recovery_window_share",
        "log_annual_task_mwh",
        "log_recharge_to_task_ratio",
    ],
}

BOOTSTRAP_SEED_KEYS = {
    "longest_event": "longest_only",
    "recovery_window": "zero_window_only",
    "joint_geometry": "joint_geometry",
    "joint_controlled": "joint_geometry_plus_scale",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def within(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = frame[columns].astype(float)
    means = frame.groupby("country")[columns].transform("mean").astype(float)
    return (values - means).to_numpy(float)


def fit_within(frame: pd.DataFrame, predictors: list[str]) -> tuple[np.ndarray, float]:
    x = within(frame, predictors)
    y = within(frame, [OUTCOME]).ravel()
    if np.linalg.matrix_rank(x) != len(predictors):
        raise RuntimeError(f"rank-deficient design for {predictors}")
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta
    r2 = 1.0 - float(np.sum(residual**2) / np.sum(y**2))
    return beta, r2


def coefficient_bootstrap(
    frame: pd.DataFrame, predictors: list[str], model_id: str
) -> tuple[np.ndarray, np.ndarray]:
    beta, _ = fit_within(frame, predictors)
    x = within(frame, predictors)
    y = within(frame, [OUTCOME]).ravel()
    countries = sorted(frame.country.unique())
    mapping = {country: index for index, country in enumerate(countries)}
    country_index = frame.country.map(mapping).to_numpy(int)
    seed_key = BOOTSTRAP_SEED_KEYS[model_id]
    rng = np.random.default_rng(SEED + sum(ord(char) for char in seed_key))
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
    return beta, draws


def leave_one_country_out(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_id, predictors in MODELS.items():
        for country, held_out in frame.groupby("country", sort=True):
            training = frame.loc[frame.country.ne(country)].copy()
            beta, _ = fit_within(training, predictors)
            x_test = held_out[predictors].astype(float) - held_out[predictors].astype(float).mean()
            y_test = held_out[OUTCOME].astype(float) - held_out[OUTCOME].astype(float).mean()
            predicted = x_test.to_numpy(float) @ beta
            for (_, record), observed, fitted in zip(
                held_out.iterrows(), y_test.to_numpy(float), predicted
            ):
                rows.append(
                    {
                        "model_id": model_id,
                        "country": country,
                        "year": int(record.year),
                        "observed_country_centred_value_eur_kw_year": observed,
                        "predicted_country_centred_value_eur_kw_year": fitted,
                        "residual_eur_kw_year": observed - fitted,
                    }
                )
    return pd.DataFrame(rows)


def summarize_prediction(group: pd.DataFrame) -> dict[str, float]:
    observed = group.observed_country_centred_value_eur_kw_year.to_numpy(float)
    predicted = group.predicted_country_centred_value_eur_kw_year.to_numpy(float)
    residual = observed - predicted
    sst = float(np.sum(observed**2))
    observed_sd = float(np.std(observed))
    predicted_sd = float(np.std(predicted))
    pearson = (
        float(np.corrcoef(observed, predicted)[0, 1])
        if observed_sd > 0.0 and predicted_sd > 0.0
        else np.nan
    )
    return {
        "loco_r2": 1.0 - float(np.sum(residual**2)) / sst,
        "loco_rmse_eur_kw_year": float(np.sqrt(np.mean(residual**2))),
        "loco_mae_eur_kw_year": float(np.mean(np.abs(residual))),
        "loco_pearson_r": pearson,
    }


def prediction_bootstrap(predictions: pd.DataFrame) -> pd.DataFrame:
    countries = sorted(predictions.country.unique())
    rng = np.random.default_rng(SEED + 991)
    rows: list[dict[str, object]] = []
    by_model = {model: group.copy() for model, group in predictions.groupby("model_id")}
    for draw in range(BOOTSTRAP_DRAWS):
        sampled = rng.choice(countries, size=len(countries), replace=True)
        counts = pd.Series(sampled).value_counts()
        for model_id, group in by_model.items():
            weights = group.country.map(counts).fillna(0).to_numpy(float)
            observed = group.observed_country_centred_value_eur_kw_year.to_numpy(float)
            predicted = group.predicted_country_centred_value_eur_kw_year.to_numpy(float)
            sse = float(np.sum(weights * (observed - predicted) ** 2))
            sst = float(np.sum(weights * observed**2))
            rows.append(
                {
                    "draw": draw,
                    "model_id": model_id,
                    "loco_r2": 1.0 - sse / sst,
                    "loco_rmse_eur_kw_year": float(np.sqrt(sse / weights.sum())),
                }
            )
    return pd.DataFrame(rows)


def latex_number(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def write_macros(comparison: pd.DataFrame, coefficients: pd.DataFrame) -> None:
    indexed = comparison.set_index("model_id")
    controlled = coefficients.loc[
        coefficients.model_id.eq("joint_controlled")
        & coefficients.predictor.eq("natural_zero_recovery_window_share")
    ].iloc[0]
    recovery_per_ten = float(controlled.coefficient) / 10.0
    recovery_low_per_ten = float(controlled.bootstrap_ci95_low) / 10.0
    recovery_high_per_ten = float(controlled.bootstrap_ci95_high) / 10.0
    scale_share = 100.0 * abs(recovery_per_ten) / COST_COMPARATOR
    lines = [
        rf"\newcommand{{\RecoveryCountryYears}}{{{len(pd.read_csv(INPUT))}}}",
        rf"\newcommand{{\RecoveryCountryBlocks}}{{{pd.read_csv(INPUT).country.nunique()}}}",
        rf"\newcommand{{\RecoveryBootstrapDraws}}{{{BOOTSTRAP_DRAWS:,}}}",
        rf"\newcommand{{\LongestWithinRtwo}}{{{latex_number(indexed.loc['longest_event', 'within_r2'])}}}",
        rf"\newcommand{{\RecoveryWithinRtwo}}{{{latex_number(indexed.loc['recovery_window', 'within_r2'])}}}",
        rf"\newcommand{{\JointWithinRtwo}}{{{latex_number(indexed.loc['joint_geometry', 'within_r2'])}}}",
        rf"\newcommand{{\ControlledWithinRtwo}}{{{latex_number(indexed.loc['joint_controlled', 'within_r2'])}}}",
        rf"\newcommand{{\LongestLocoRtwo}}{{{latex_number(indexed.loc['longest_event', 'loco_r2'])}}}",
        rf"\newcommand{{\RecoveryLocoRtwo}}{{{latex_number(indexed.loc['recovery_window', 'loco_r2'])}}}",
        rf"\newcommand{{\JointLocoRtwo}}{{{latex_number(indexed.loc['joint_geometry', 'loco_r2'])}}}",
        rf"\newcommand{{\ControlledLocoRtwo}}{{{latex_number(indexed.loc['joint_controlled', 'loco_r2'])}}}",
        rf"\newcommand{{\RecoveryIncrementalWithinRtwo}}{{{latex_number(indexed.loc['joint_geometry', 'within_r2'] - indexed.loc['longest_event', 'within_r2'])}}}",
        rf"\newcommand{{\RecoveryIncrementalLocoRtwo}}{{{latex_number(indexed.loc['joint_geometry', 'loco_r2'] - indexed.loc['longest_event', 'loco_r2'])}}}",
        rf"\newcommand{{\ControlledRecoveryCoefTenPp}}{{{latex_number(recovery_per_ten, 2)}}}",
        rf"\newcommand{{\ControlledRecoveryCoefTenPpLow}}{{{latex_number(recovery_low_per_ten, 2)}}}",
        rf"\newcommand{{\ControlledRecoveryCoefTenPpHigh}}{{{latex_number(recovery_high_per_ten, 2)}}}",
        rf"\newcommand{{\RecoveryTenPpCostScaleShare}}{{{latex_number(scale_share, 1)}}}",
    ]
    (RESULTS / "recovery_window_incremental_macros.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(INPUT)
    required = {OUTCOME, "country", "year", *sum(MODELS.values(), [])}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"missing columns: {missing}")
    if len(frame) != 132 or frame.country.nunique() != 15:
        raise RuntimeError("expected 132 country-years in 15 countries")
    if frame.duplicated(["country", "year"]).any():
        raise RuntimeError("country-year keys are not unique")
    if not np.isfinite(frame[list(required.difference({"country"}))].select_dtypes("number")).all().all():
        raise RuntimeError("non-finite analysis input")

    coefficient_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    for model_id, predictors in MODELS.items():
        beta, draws = coefficient_bootstrap(frame, predictors, model_id)
        _, within_r2 = fit_within(frame, predictors)
        for index, predictor in enumerate(predictors):
            valid = draws[:, index][np.isfinite(draws[:, index])]
            coefficient_rows.append(
                {
                    "model_id": model_id,
                    "predictor": predictor,
                    "coefficient": beta[index],
                    "bootstrap_ci95_low": np.quantile(valid, 0.025),
                    "bootstrap_ci95_high": np.quantile(valid, 0.975),
                    "bootstrap_draws_valid": len(valid),
                    "country_years": len(frame),
                    "country_blocks": frame.country.nunique(),
                }
            )
        model_rows.append(
            {
                "model_id": model_id,
                "predictors": "|".join(predictors),
                "predictor_count": len(predictors),
                "within_r2": within_r2,
            }
        )

    coefficients = pd.DataFrame(coefficient_rows)
    predictions = leave_one_country_out(frame)
    bootstrap_metrics = prediction_bootstrap(predictions)
    comparison = pd.DataFrame(model_rows)
    for model_id, group in predictions.groupby("model_id"):
        metrics = summarize_prediction(group)
        for name, value in metrics.items():
            comparison.loc[comparison.model_id.eq(model_id), name] = value
        boot = bootstrap_metrics.loc[bootstrap_metrics.model_id.eq(model_id)]
        comparison.loc[comparison.model_id.eq(model_id), "loco_r2_ci95_low"] = boot.loco_r2.quantile(0.025)
        comparison.loc[comparison.model_id.eq(model_id), "loco_r2_ci95_high"] = boot.loco_r2.quantile(0.975)

    country_metrics = []
    for (model_id, country), group in predictions.groupby(["model_id", "country"]):
        metrics = summarize_prediction(group)
        country_metrics.append({"model_id": model_id, "country": country, **metrics})
    country_metrics_frame = pd.DataFrame(country_metrics)

    baseline_within = float(comparison.loc[comparison.model_id.eq("longest_event"), "within_r2"].iloc[0])
    baseline_loco = float(comparison.loc[comparison.model_id.eq("longest_event"), "loco_r2"].iloc[0])
    comparison["incremental_within_r2_vs_longest_event"] = comparison.within_r2 - baseline_within
    comparison["incremental_loco_r2_vs_longest_event"] = comparison.loco_r2 - baseline_loco

    coefficients.to_csv(RESULTS / "coefficient_summary.csv", index=False)
    comparison.to_csv(RESULTS / "model_comparison.csv", index=False)
    predictions.to_csv(SOURCE / "loco_predictions.csv", index=False)
    country_metrics_frame.to_csv(SOURCE / "country_loco_metrics.csv", index=False)
    frame.to_csv(SOURCE / "analysis_panel_132_country_years.csv", index=False)
    bootstrap_metrics.to_csv(RESULTS / "loco_country_bootstrap_metrics.csv", index=False)
    write_macros(comparison, coefficients)

    outputs = [
        RESULTS / "coefficient_summary.csv",
        RESULTS / "model_comparison.csv",
        SOURCE / "loco_predictions.csv",
        SOURCE / "country_loco_metrics.csv",
        SOURCE / "analysis_panel_132_country_years.csv",
        RESULTS / "loco_country_bootstrap_metrics.csv",
        RESULTS / "recovery_window_incremental_macros.tex",
    ]
    qa = {
        "status": "pass",
        "input": str(INPUT.relative_to(ROOT)),
        "input_sha256": sha256(INPUT),
        "country_years": len(frame),
        "countries": frame.country.nunique(),
        "minimum_years_per_country": int(frame.groupby("country").size().min()),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "loco_folds": frame.country.nunique(),
        "excluded_observations": 0,
        "outcome": OUTCOME,
        "cost_comparator_eur_kw_year": COST_COMPARATOR,
        "outputs_sha256": {path.name: sha256(path) for path in outputs},
    }
    (RESULTS / "qa.json").write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    print(comparison.to_string(index=False))
    print(coefficients.to_string(index=False))
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
