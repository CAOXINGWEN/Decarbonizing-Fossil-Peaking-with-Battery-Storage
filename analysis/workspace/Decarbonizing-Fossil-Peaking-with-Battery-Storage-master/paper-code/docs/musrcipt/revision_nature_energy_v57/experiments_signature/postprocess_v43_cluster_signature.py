#!/usr/bin/env python3
"""Freeze the 2/4/6-day cluster-length signature for v43 figures.

The independent bootstrap block is country.  A country-year zero crossing is
the first positive duration after the within-path trough for which every
subsequent tested duration remains positive.  Paths that never return above
zero by 168 h are explicitly right-censored; paths that never fall to or below
zero are left-censored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
V43 = HERE.parent
SOURCE = V43 / "source_data"
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260729
MIN_DURATION_H = 24.0
MAX_DURATION_H = 168.0
NATURAL_GEOMETRY = (
    HERE / "results_recovery_windows" / "natural_event_geometry.csv"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_run(path: Path) -> tuple[pd.DataFrame, dict[str, object], Path]:
    contract_path = path / "run_contract.json"
    arms_path = path / "portfolio_arm_results.csv.gz"
    if not contract_path.exists() or not arms_path.exists():
        raise FileNotFoundError(f"Incomplete signature run: {path}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if int(contract["scenario_id"]) != 20:
        raise RuntimeError(f"{path} is scenario {contract['scenario_id']}, not S20")
    if str(contract["tail_power_mode"]).lower() != "fixed":
        raise RuntimeError(f"{path} is not the registered fixed-power design")
    if not np.isclose(float(contract["recharge_multiplier"]), 1.0):
        raise RuntimeError(f"{path} does not retain the 1x recharge quantity")
    if str(contract["recharge_profile_mode"]).lower() != "observed":
        raise RuntimeError(f"{path} does not retain observed recharge timing")

    arms = pd.read_csv(arms_path)
    candidates = arms[
        arms["arm_type"].eq("ldes_candidate")
        & arms["duration_h"].between(
            MIN_DURATION_H, MAX_DURATION_H, inclusive="both"
        )
    ].copy()
    comparator = arms[
        arms["arm_type"].eq("extra_lfp4_comparator")
    ].copy()
    if candidates.empty or comparator.empty:
        raise RuntimeError(f"{path} has no complete candidate/comparator arms")

    # Reconstruct the exact Figure 5 EFC-valued ledger from solved arms.
    # The runner's residual-peak helper column and tail-power normalization are
    # deliberately not used because neither defines the manuscript estimand.
    def add_efc_value(frame: pd.DataFrame, output: str) -> pd.DataFrame:
        result = frame.copy()
        result[output] = (
            result["energy_value_eur_mwh"]
            * result["total_discharge_mwh"]
            + result["firm_value_eur_kw_year"]
            * 1000.0
            * result["portfolio_efc_mw"]
            - result["base_fixed_cost_eur"]
            - result["tail_fixed_cost_eur"]
            - result["base_vom_cost_eur"]
            - result["tail_vom_cost_eur"]
            - result["charge_price_eur_mwh"]
            * result["total_charge_mwh"]
        )
        return result

    candidates = add_efc_value(candidates, "candidate_efc_net_value_eur")
    comparator = add_efc_value(comparator, "comparator_efc_net_value_eur")
    comparison_keys = ["country", "year", "chronology"]
    if comparator.duplicated(comparison_keys).any():
        raise RuntimeError(f"{path} has non-unique extra-LFP comparators")
    candidates = candidates.merge(
        comparator[
            comparison_keys + ["comparator_efc_net_value_eur"]
        ],
        on=comparison_keys,
        how="left",
        validate="many_to_one",
    )
    denominator_kw = candidates["base_power_mw"] * 1000.0
    if (denominator_kw <= 0).any():
        raise RuntimeError(f"{path} has non-positive common base power")
    candidates["benefit_eur_kw_common_power_year"] = (
        candidates["candidate_efc_net_value_eur"]
        - candidates["comparator_efc_net_value_eur"]
    ) / denominator_kw

    # Match Figure 5 exactly: select the best technology separately within
    # each chronology and duration, then pair clustered minus dispersed.
    envelope_keys = [
        "country",
        "year",
        "chronology",
        "duration_h",
    ]
    candidates = candidates.sort_values(
        envelope_keys
        + [
            "benefit_eur_kw_common_power_year",
            "fixed_cost_eur_kw_year",
            "config_id",
        ],
        ascending=[True] * len(envelope_keys) + [False, True, True],
        kind="stable",
    )
    envelope = candidates.drop_duplicates(envelope_keys, keep="first")
    index = ["country", "year", "duration_h"]
    benefit = envelope.pivot(
        index=index,
        columns="chronology",
        values="benefit_eur_kw_common_power_year",
    ).reset_index()
    selected_config = envelope.pivot(
        index=index,
        columns="chronology",
        values="config_id",
    ).reset_index()
    selected_config = selected_config.rename(
        columns={
            "clustered": "config_id_clustered",
            "dispersed": "config_id_dispersed",
        }
    )
    frame = benefit.merge(
        selected_config,
        on=index,
        how="left",
        validate="one_to_one",
    )
    if {"clustered", "dispersed"} - set(frame.columns):
        raise RuntimeError(f"{path} does not contain both chronologies")
    frame["paired_effect_eur_kw_year"] = (
        frame["clustered"] - frame["dispersed"]
    )
    frame["target_cluster_days"] = int(contract["cluster_days"])
    frame["country_year_id"] = (
        frame["country"].astype(str)
        + "-"
        + frame["year"].astype(int).astype(str)
    )
    return frame, contract, arms_path


def country_block_summary(records: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, object]] = []
    for (cluster_days, duration_h), group in records.groupby(
        ["target_cluster_days", "duration_h"], sort=True
    ):
        country = (
            group.groupby("country")["paired_effect_eur_kw_year"]
            .mean()
            .sort_index()
        )
        values = country.to_numpy(float)
        indices = rng.integers(
            0, len(values), size=(BOOTSTRAP_DRAWS, len(values))
        )
        draws = values[indices].mean(axis=1)
        rows.append(
            {
                "target_cluster_days": int(cluster_days),
                "duration_h": float(duration_h),
                "country_block_mean": float(values.mean()),
                "ci95_low": float(np.quantile(draws, 0.025)),
                "ci95_high": float(np.quantile(draws, 0.975)),
                "country_year_median": float(
                    group["paired_effect_eur_kw_year"].median()
                ),
                "n_countries": int(country.size),
                "n_country_years": int(
                    group[["country", "year"]].drop_duplicates().shape[0]
                ),
            }
        )
    return pd.DataFrame(rows)


def select_duration_envelope(records: pd.DataFrame) -> pd.DataFrame:
    """Validate the chronology-specific duration envelope."""
    keys = [
        "target_cluster_days",
        "country",
        "year",
        "country_year_id",
        "duration_h",
    ]
    if records.duplicated(keys).any():
        raise RuntimeError("Chronology-specific duration envelope is not unique")
    return records.copy()


def crossing_records(records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["target_cluster_days", "country", "year", "country_year_id"]
    for key, group in records.groupby(keys, sort=True):
        curve = (
            group.groupby("duration_h")["paired_effect_eur_kw_year"]
            .max()
            .sort_index()
        )
        durations = curve.index.to_numpy(float)
        effects = curve.to_numpy(float)
        if np.all(effects > 0):
            status = "left_censored"
            crossing = float(durations.min())
        elif np.all(effects <= 0):
            status = "right_censored"
            crossing = np.nan
        else:
            trough = int(np.argmin(effects))
            crossing_index = next(
                (
                    index
                    for index in range(trough + 1, len(effects))
                    if effects[index] > 0 and np.all(effects[index:] > 0)
                ),
                None,
            )
            if crossing_index is None:
                status = "right_censored"
                crossing = np.nan
            else:
                status = "observed"
                crossing = float(durations[crossing_index])
        rows.append(
            {
                "target_cluster_days": int(key[0]),
                "country": key[1],
                "year": int(key[2]),
                "country_year_id": key[3],
                "crossing_status": status,
                "zero_crossing_duration_h": crossing,
                "right_censor_duration_h": float(durations.max()),
                "minimum_effect_eur_kw_year": float(effects.min()),
                "minimum_effect_duration_h": float(
                    durations[int(np.argmin(effects))]
                ),
                "maximum_tested_duration_h": float(durations.max()),
                "tested_duration_count": int(len(durations)),
            }
        )
    return pd.DataFrame(rows)


def _country_block_median_interval(group: pd.DataFrame) -> tuple[float, float]:
    """Bootstrap the observed crossing median with country as the block."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    countries = np.array(sorted(group["country"].unique()))
    observed = group[group["crossing_status"].eq("observed")]
    if observed.empty:
        return np.nan, np.nan
    durations = np.sort(observed["zero_crossing_duration_h"].unique())
    counts = (
        observed.groupby(["country", "zero_crossing_duration_h"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=countries, columns=durations, fill_value=0)
        .to_numpy(dtype=np.int16)
    )
    sample_indices = rng.integers(
        0,
        len(countries),
        size=(BOOTSTRAP_DRAWS, len(countries)),
    )
    sampled_counts = counts[sample_indices].sum(axis=1)
    totals = sampled_counts.sum(axis=1)
    valid = totals > 0
    if not np.any(valid):
        return np.nan, np.nan
    cumulative = np.cumsum(sampled_counts[valid], axis=1)
    lower_rank = (totals[valid] - 1) // 2
    upper_rank = totals[valid] // 2
    lower_index = np.argmax(
        cumulative > lower_rank[:, np.newaxis],
        axis=1,
    )
    upper_index = np.argmax(
        cumulative > upper_rank[:, np.newaxis],
        axis=1,
    )
    draws = 0.5 * (
        durations[lower_index] + durations[upper_index]
    )
    return (
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def crossing_summary(records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cluster_days, group in records.groupby(
        "target_cluster_days", sort=True
    ):
        observed = group[group["crossing_status"].eq("observed")][
            "zero_crossing_duration_h"
        ]
        median_ci_low, median_ci_high = _country_block_median_interval(group)
        rows.append(
            {
                "target_cluster_days": int(cluster_days),
                "n_country_years": int(len(group)),
                "observed_crossings": int(len(observed)),
                "left_censored": int(
                    group["crossing_status"].eq("left_censored").sum()
                ),
                "right_censored": int(
                    group["crossing_status"].eq("right_censored").sum()
                ),
                "median_crossing_h": (
                    float(observed.median()) if len(observed) else np.nan
                ),
                "q25_crossing_h": (
                    float(observed.quantile(0.25)) if len(observed) else np.nan
                ),
                "q75_crossing_h": (
                    float(observed.quantile(0.75)) if len(observed) else np.nan
                ),
                "median_country_block_ci95_low": median_ci_low,
                "median_country_block_ci95_high": median_ci_high,
                "right_censored_share": float(
                    group["crossing_status"].eq("right_censored").mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _spearman(
    x: pd.Series | np.ndarray, y: pd.Series | np.ndarray
) -> float:
    x_rank = pd.Series(np.asarray(x, dtype=float)).rank(
        method="average"
    ).to_numpy()
    y_rank = pd.Series(np.asarray(y, dtype=float)).rank(
        method="average"
    ).to_numpy()
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def natural_association_summary(
    crossings: pd.DataFrame, natural: pd.DataFrame
) -> pd.DataFrame:
    """Relate observed controlled crossings to natural event geometry.

    Right-censored and left-censored cases remain in the reported sample counts
    but are not assigned fabricated crossing durations.
    """
    joined = crossings.merge(
        natural[
            [
                "country",
                "year",
                "natural_longest_event_run_days",
                "natural_median_recovery_hours",
            ]
        ],
        on=["country", "year"],
        how="left",
        validate="many_to_one",
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, object]] = []
    for cluster_days, group in joined.groupby(
        "target_cluster_days", sort=True
    ):
        observed = group[group["crossing_status"].eq("observed")].dropna(
            subset=[
                "zero_crossing_duration_h",
                "natural_longest_event_run_days",
            ]
        )
        point = _spearman(
            observed["natural_longest_event_run_days"],
            observed["zero_crossing_duration_h"],
        )
        countries = sorted(group["country"].unique())
        observed_by_country = []
        for country in countries:
            block = group[
                group["country"].eq(country)
                & group["crossing_status"].eq("observed")
            ].dropna(
                subset=[
                    "zero_crossing_duration_h",
                    "natural_longest_event_run_days",
                ]
            )
            observed_by_country.append(
                block[
                    [
                        "natural_longest_event_run_days",
                        "zero_crossing_duration_h",
                    ]
                ].to_numpy(float)
            )
        sample_indices = rng.integers(
            0,
            len(countries),
            size=(BOOTSTRAP_DRAWS, len(countries)),
        )
        draws: list[float] = []
        for sampled in sample_indices:
            blocks = [
                observed_by_country[index]
                for index in sampled
                if observed_by_country[index].size
            ]
            if not blocks:
                continue
            draw = np.concatenate(blocks, axis=0)
            if (
                len(draw) > 2
                and np.unique(draw[:, 0]).size > 1
                and np.unique(draw[:, 1]).size > 1
            ):
                draws.append(
                    _spearman(draw[:, 0], draw[:, 1])
                )
        rows.append(
            {
                "target_cluster_days": int(cluster_days),
                "observed_crossings": int(len(observed)),
                "right_censored": int(
                    group["crossing_status"].eq("right_censored").sum()
                ),
                "left_censored": int(
                    group["crossing_status"].eq("left_censored").sum()
                ),
                "spearman_rho_natural_longest_event_vs_crossing": point,
                "country_block_ci95_low": (
                    float(np.quantile(draws, 0.025)) if draws else np.nan
                ),
                "country_block_ci95_high": (
                    float(np.quantile(draws, 0.975)) if draws else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    args = parser.parse_args()

    frames = []
    contracts = []
    inputs: dict[str, str] = {}
    for run in args.runs:
        frame, contract, analysis_path = load_run(run)
        frames.append(frame)
        contracts.append(contract)
        inputs[str((run / "run_contract.json").resolve())] = sha256(
            run / "run_contract.json"
        )
        inputs[str(analysis_path.resolve())] = sha256(analysis_path)
    raw_records = pd.concat(frames, ignore_index=True)
    records = select_duration_envelope(raw_records)
    expected_clusters = {2, 4, 6}
    actual_clusters = set(records["target_cluster_days"].astype(int))
    if actual_clusters != expected_clusters:
        raise RuntimeError(
            f"Expected cluster sizes {expected_clusters}; got {actual_clusters}"
        )
    if records[["country", "year"]].drop_duplicates().shape[0] != 132:
        raise RuntimeError("Signature runs do not cover all 132 country-years")

    signature = country_block_summary(records)
    crossings = crossing_records(records)
    crossing_stats = crossing_summary(crossings)
    if not NATURAL_GEOMETRY.exists():
        raise FileNotFoundError(
            f"Natural-event geometry not found: {NATURAL_GEOMETRY}"
        )
    natural = pd.read_csv(NATURAL_GEOMETRY)
    association = natural_association_summary(crossings, natural)
    inputs[str(NATURAL_GEOMETRY.resolve())] = sha256(NATURAL_GEOMETRY)

    SOURCE.mkdir(parents=True, exist_ok=True)
    outputs = {
        SOURCE / "v43_cluster_signature_country_year.csv.gz": records,
        SOURCE / "v43_figure5d_cluster_signature_summary.csv": signature,
        SOURCE / "v43_figure5e_zero_crossing_country_year.csv": crossings,
        SOURCE / "v43_figure5e_zero_crossing_summary.csv": crossing_stats,
        SOURCE / "v43_ed9_natural_crossing_association_summary.csv": association,
    }
    for path, frame in outputs.items():
        frame.to_csv(
            path,
            index=False,
            compression="gzip" if path.suffix == ".gz" else None,
        )

    manifest = {
        "analysis_id": "v43_cluster_length_signature_postprocess",
        "inputs": inputs,
        "outputs": {str(path): sha256(path) for path in outputs},
        "country_years": 132,
        "countries": 15,
        "cluster_days": [2, 4, 6],
        "scenario_id": 20,
        "tail_power_mode": "fixed",
        "duration_domain_h": [MIN_DURATION_H, MAX_DURATION_H],
        "bootstrap": {
            "independent_block": "country",
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
        },
        "crossing_definition": (
            "first positive duration after the within-path trough for which "
            "all subsequent tested durations remain positive"
        ),
        "duration_selection": (
            "maximum EFC-valued benefit across candidate technologies within "
            "each chronology, country-year, cluster length and tested "
            "duration; clustered-minus-dispersed pairing is applied after "
            "chronology-specific selection"
        ),
        "valuation_estimator": (
            "(energy value × discharge + task-firm-supply unit value × EFC "
            "− base and tail fixed cost − base and tail VOM − charge price × "
            "charge) candidate-minus-extra-4h-LFP, normalized by common base "
            "power"
        ),
    }
    manifest_path = SOURCE / "v43_cluster_signature_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
