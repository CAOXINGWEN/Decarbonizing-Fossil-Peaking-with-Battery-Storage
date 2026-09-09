"""
M4 seasonal MC frequency/length demand generator.

This is the season-constrained replacement for the earlier MC-q75 freqlen grid.
It keeps the seasonal Markov demand generator, injects extreme events using
province-specific seasonal long-tail weights from M1, and builds a matching
5-year seasonal bootstrap Monte Carlo series for curtailment and charge windows.

Outputs:
  output/m3_dispatch/demand_seasmc_freq<F>_len<L>.csv
  output/m3_dispatch/curtailment_seasmc_freq<F>_len<L>.csv
  output/m3_dispatch/charge_window_seasmc_freq<F>_len<L>.csv
  output/m3_dispatch/freqlen_seasonal_mc_demand_summary.csv
  output/m3_dispatch/freqlen_seasonal_mc_event_log.csv
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from config import CONFIG, MC_CONFIG
from event_clustering import EN_TO_CN
from monte_carlo_simulation import fit_season_distributions
from build_2yr_demand import (
    load_real_demand_long_format,
    generate_one_virtual_year,
)


PAPER_CODE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PAPER_CODE_DIR / "output" / "m3_dispatch"
M1_DIR = PAPER_CODE_DIR / "output" / "m1_clustering"

EVENTS = [4, 8, 12, 16]
LENGTHS = [5, 7, 10, 14]
BASE_PARAMS = {
    "demand_multiplier": 1.0,
    "extreme_boost": 1.3,
    "event_intensity": 1.5,
    "intensity_multiplier": 1.05,
    "use_real_year": True,
    "n_years": 5,
}

SEASON_ORDER = ["spring", "summer", "autumn", "winter"]
LOW_CHARGE_WEIGHT = 0.60
LOW_WINDOW_WEIGHT = 0.40
STRESS_POWER = 1.5
CHARGE_BOOTSTRAP_BLOCK_DAYS = 3


def largest_remainder_counts(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Integer allocation that preserves the requested total count."""
    clean = np.array([max(0.0, float(weights.get(s, 0.0))) for s in SEASON_ORDER])
    if clean.sum() <= 0:
        clean = np.ones(len(SEASON_ORDER), dtype=float)
    raw = clean / clean.sum() * int(total)
    counts = np.floor(raw).astype(int)
    remainder = int(total) - int(counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw - counts))
        for idx in order[:remainder]:
            counts[idx] += 1
    return {season: int(counts[i]) for i, season in enumerate(SEASON_ORDER)}


def month_day_dates() -> pd.DatetimeIndex:
    return pd.date_range("2023-01-01", periods=365, freq="D")


def season_for_days(dates: pd.DatetimeIndex) -> np.ndarray:
    month_to_season = {
        month: season
        for season, months in MC_CONFIG["seasons"].items()
        for month in months
    }
    return np.array([month_to_season[d.month] for d in dates])


def contiguous_season_starts(seasons_by_day: np.ndarray,
                             season: str,
                             length_days: int) -> np.ndarray:
    starts = []
    n_days = len(seasons_by_day)
    for s in range(0, n_days - length_days + 1):
        if np.all(seasons_by_day[s:s + length_days] == season):
            starts.append(s)
    return np.array(starts, dtype=int)


def rank_low(values: np.ndarray) -> np.ndarray:
    """Return high scores for low values, with a small positive floor."""
    if len(values) == 0:
        return values.astype(float)
    ranks = pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=float)
    return np.clip(1.0 - ranks + 1.0 / len(values), 0.05, 1.0)


def start_weights_from_charge(curt_arr: np.ndarray,
                              cw_arr: np.ndarray,
                              starts: np.ndarray,
                              length_days: int) -> np.ndarray:
    if len(starts) == 0:
        return np.array([], dtype=float)
    curt_mean = np.array([
        float(np.mean(curt_arr[s:s + length_days])) for s in starts
    ])
    cw_mean = np.array([
        float(np.mean(cw_arr[s:s + length_days])) for s in starts
    ])
    stress = (
        LOW_CHARGE_WEIGHT * rank_low(curt_mean)
        + LOW_WINDOW_WEIGHT * rank_low(cw_mean)
    )
    weights = np.power(np.clip(stress, 0.01, None), STRESS_POWER)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.ones(len(starts), dtype=float)
    return weights / weights.sum()


def choose_starts(starts: np.ndarray,
                  weights: np.ndarray,
                  n_events: int,
                  length_days: int,
                  chosen_global: list[int],
                  rng: np.random.Generator) -> list[int]:
    """Weighted non-overlap sampling within a season."""
    if n_events <= 0 or len(starts) == 0:
        return []
    candidates = starts.astype(int).copy()
    probs = weights.astype(float).copy()
    selected = []
    min_gap = max(1, int(length_days))

    for _ in range(n_events):
        valid = np.ones(len(candidates), dtype=bool)
        for c in chosen_global + selected:
            valid &= np.abs(candidates - int(c)) >= min_gap
        if not valid.any():
            break
        cand = candidates[valid]
        cand_probs = probs[valid]
        cand_probs = cand_probs / cand_probs.sum()
        picked = int(rng.choice(cand, p=cand_probs))
        selected.append(picked)

    return selected


def sample_season_blocks(source_indices: np.ndarray,
                         n_days: int,
                         block_days: int,
                         rng: np.random.Generator) -> np.ndarray:
    """Circular seasonal block bootstrap day indices."""
    if len(source_indices) == 0:
        return np.array([], dtype=int)
    picked = []
    while len(picked) < n_days:
        start = int(rng.integers(0, len(source_indices)))
        for k in range(max(1, int(block_days))):
            picked.append(int(source_indices[(start + k) % len(source_indices)]))
            if len(picked) == n_days:
                break
    return np.array(picked, dtype=int)


def build_charge_side_mc_5yr(curt_real: pd.DataFrame,
                             cw_real: pd.DataFrame,
                             rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build 1 real year + 4 seasonal bootstrap MC years for charge-side data.

    Curtailment and charge-window values are sampled as paired daily observations
    within each province-season so their observed day-level correlation is kept.
    The same 5-year charge-side series is reused for all freqlen nodes.
    """
    n_years = BASE_PARAMS["n_years"]
    dates = month_day_dates()
    seasons_by_day = season_for_days(dates)
    provinces = list(curt_real.columns)

    curt_blocks = [curt_real.reset_index(drop=True).copy()]
    cw_blocks = [cw_real.reset_index(drop=True).copy()]
    summary_rows = []

    for year_idx in range(1, n_years):
        curt_year = pd.DataFrame(0.0, index=np.arange(365), columns=provinces)
        cw_year = pd.DataFrame(0.0, index=np.arange(365), columns=provinces)

        for prov in provinces:
            if prov not in cw_real.columns:
                continue
            for season in SEASON_ORDER:
                target_idx = np.where(seasons_by_day == season)[0]
                sampled_idx = sample_season_blocks(
                    target_idx, len(target_idx),
                    CHARGE_BOOTSTRAP_BLOCK_DAYS, rng,
                )
                curt_vals = curt_real.iloc[sampled_idx][prov].to_numpy(dtype=float)
                cw_vals = cw_real.iloc[sampled_idx][prov].to_numpy(dtype=float)
                curt_year.loc[target_idx, prov] = curt_vals
                cw_year.loc[target_idx, prov] = np.clip(cw_vals, 0.0, 24.0)

                summary_rows.append({
                    "year_idx": year_idx,
                    "province_en": prov,
                    "season": season,
                    "source": "seasonal_block_bootstrap",
                    "n_days": int(len(target_idx)),
                    "mean_curtailment_kwh": float(np.mean(curt_vals)),
                    "mean_charge_window_h": float(np.mean(cw_vals)),
                })

        curt_blocks.append(curt_year)
        cw_blocks.append(cw_year)

    curt_full = pd.concat(curt_blocks, axis=0, ignore_index=True)
    cw_full = pd.concat(cw_blocks, axis=0, ignore_index=True)
    curt_full.index.name = "day_idx"
    cw_full.index.name = "day_idx"

    return curt_full, cw_full, pd.DataFrame(summary_rows)


def load_season_metrics() -> dict[str, dict[str, dict[str, float]]]:
    fp = M1_DIR / "province_season_clustering.csv"
    df = pd.read_csv(fp)
    metrics: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for row in df.itertuples(index=False):
        metrics[str(row.province_cn)][str(row.season)] = {
            "n_extreme_days": float(row.n_extreme_days),
            "n_runs": float(row.n_runs),
            "mean_run_length": float(row.mean_run_length),
            "max_run_length": float(row.max_run_length),
            "p22": float(row.p22),
        }
    return metrics


def seasonal_weights(metrics: dict[str, dict[str, float]],
                     mode: str) -> dict[str, float]:
    weights = {}
    for season in SEASON_ORDER:
        m = metrics.get(season, {})
        if mode == "frequency":
            score = m.get("n_runs", 0.0)
        else:
            score = (
                m.get("n_runs", 0.0)
                * max(1.0, m.get("max_run_length", 0.0))
                * (1.0 + m.get("p22", 0.0))
            )
        weights[season] = float(score)
    return weights


def amplify_extreme_days_by_season(arr: np.ndarray,
                                   province_cn: str,
                                   season_params: dict,
                                   seasons_by_day: np.ndarray,
                                   extreme_boost: float) -> np.ndarray:
    out = arr.copy()
    for season in SEASON_ORDER:
        params = season_params.get(province_cn, {}).get(season)
        if not params:
            continue
        q75 = float(params.get("q75", 0.0))
        idx = np.where(seasons_by_day == season)[0]
        mask = out[idx] > q75
        out[idx[mask]] *= float(extreme_boost)
    return out


def inject_seasonal_events(arr: np.ndarray,
                           province_cn: str,
                           province_en: str,
                           target_events: int,
                           length_days: int,
                           mode: str,
                           year_idx: int,
                           stem: str,
                           season_metrics: dict,
                           season_params: dict,
                           seasons_by_day: np.ndarray,
                           curt_year: pd.DataFrame,
                           cw_year: pd.DataFrame,
                           rng: np.random.Generator) -> tuple[np.ndarray, list[dict]]:
    out = arr.copy()
    metrics = season_metrics.get(province_cn, {})
    weights = seasonal_weights(metrics, mode)
    counts = largest_remainder_counts(target_events, weights)
    chosen: list[int] = []
    event_rows: list[dict] = []

    curt_arr = (
        curt_year[province_en].to_numpy(dtype=float)
        if province_en in curt_year.columns else np.ones(365, dtype=float)
    )
    cw_arr = (
        cw_year[province_en].to_numpy(dtype=float)
        if province_en in cw_year.columns else np.ones(365, dtype=float)
    )

    for season in SEASON_ORDER:
        n_season_events = counts.get(season, 0)
        starts = contiguous_season_starts(seasons_by_day, season, length_days)
        start_weights = start_weights_from_charge(
            curt_arr, cw_arr, starts, length_days)
        picked = choose_starts(
            starts, start_weights, n_season_events, length_days, chosen, rng)
        chosen.extend(picked)

        params = season_params.get(province_cn, {}).get(season, {})
        season_q75 = float(params.get("q75", 0.0))
        for s in picked:
            for i in range(s, s + length_days):
                out[i] = max(out[i], season_q75) * BASE_PARAMS["event_intensity"]
            event_rows.append({
                "stem": stem,
                "year_idx": year_idx,
                "province_cn": province_cn,
                "province_en": province_en,
                "season": season,
                "start_day_idx": int(s),
                "end_day_idx": int(s + length_days - 1),
                "length_days": int(length_days),
                "mean_curtailment_kwh": float(np.mean(curt_arr[s:s + length_days])),
                "mean_charge_window_h": float(np.mean(cw_arr[s:s + length_days])),
                "target_events_for_season": int(n_season_events),
            })

    return out, event_rows


def build_one_node_demand(stem: str,
                          target_events: int,
                          length_days: int,
                          mode: str,
                          real_df: pd.DataFrame,
                          season_params: dict,
                          season_metrics: dict,
                          curt_mc: pd.DataFrame,
                          cw_mc: pd.DataFrame,
                          rng: np.random.Generator) -> tuple[pd.DataFrame, list[dict]]:
    n_years = BASE_PARAMS["n_years"]
    dates = month_day_dates()
    seasons_by_day = season_for_days(dates)
    yr_blocks = []
    event_rows: list[dict] = []

    if BASE_PARAMS["use_real_year"]:
        yr_blocks.append(real_df.reset_index(drop=True).copy())
        start_year = 1
        n_virtual = n_years - 1
    else:
        start_year = 0
        n_virtual = n_years

    cn_to_en = {v: k for k, v in EN_TO_CN.items()}
    for offset in range(n_virtual):
        year_idx = start_year + offset
        virtual_dates = pd.date_range(f"203{year_idx}-01-01", periods=365, freq="D")
        start = year_idx * 365
        stop = start + 365
        curt_year = curt_mc.iloc[start:stop].reset_index(drop=True)
        cw_year = cw_mc.iloc[start:stop].reset_index(drop=True)
        virtual = generate_one_virtual_year(
            season_params, virtual_dates,
            intensity_factor=BASE_PARAMS["intensity_multiplier"],
            rng=rng,
        )
        enhanced = virtual.copy()
        for province_cn in enhanced.columns:
            province_en = cn_to_en.get(province_cn, province_cn)
            arr = enhanced[province_cn].to_numpy(dtype=float)
            arr = amplify_extreme_days_by_season(
                arr, province_cn, season_params, seasons_by_day,
                BASE_PARAMS["extreme_boost"],
            )
            arr, rows = inject_seasonal_events(
                arr, province_cn, province_en,
                target_events, length_days, mode, year_idx, stem,
                season_metrics, season_params, seasons_by_day,
                curt_year, cw_year, rng,
            )
            enhanced[province_cn] = arr
            event_rows.extend(rows)
        yr_blocks.append(enhanced.reset_index(drop=True))

    col_order = real_df.columns.tolist()
    full = pd.concat(
        [b.reindex(columns=col_order).fillna(0.0) for b in yr_blocks],
        axis=0,
        ignore_index=True,
    )
    full.index.name = "day_idx"
    full = full * BASE_PARAMS["demand_multiplier"]
    return full, event_rows


def mean_run_length(arr: np.ndarray) -> float:
    runs = []
    for j in range(arr.shape[1]):
        cur = 0
        for v in arr[:, j] > 0:
            if v:
                cur += 1
            elif cur > 0:
                runs.append(cur)
                cur = 0
        if cur > 0:
            runs.append(cur)
    return float(np.mean(runs)) if runs else 0.0


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    real_df = load_real_demand_long_format()
    season_params = fit_season_distributions(real_df, CONFIG)
    season_metrics = load_season_metrics()
    cn_to_en = {v: k for k, v in EN_TO_CN.items()}

    curt_real = pd.read_csv(
        OUTPUT_DIR / "daily_provincial_curtailment.csv", index_col=0)
    cw_real = pd.read_csv(
        OUTPUT_DIR / "daily_provincial_charge_window_h.csv", index_col=0)

    rng = np.random.default_rng(20260603)
    charge_rng = np.random.default_rng(20260603 + 101)
    curt_mc, cw_mc, charge_mc_summary = build_charge_side_mc_5yr(
        curt_real, cw_real, charge_rng)
    charge_mc_summary.to_csv(
        OUTPUT_DIR / "freqlen_seasonal_mc_charge_side_summary.csv",
        index=False,
    )
    print(
        "=== charge-side seasonal MC ===\n"
        f"  curtailment shape={curt_mc.shape}, "
        f"charge_window shape={cw_mc.shape}, "
        f"summary rows={len(charge_mc_summary)}",
        flush=True,
    )
    summary_rows = []
    all_event_rows = []

    for events in EVENTS:
        for length in LENGTHS:
            stem = f"seasmc_freq{events}_len{length}"
            mode = "frequency" if length == 5 else "length"
            node_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
            print(
                f"\n=== {stem}: events={events}/yr, length={length}d, "
                f"mode={mode} ===",
                flush=True,
            )
            demand_cn, event_rows = build_one_node_demand(
                stem, events, length, mode,
                real_df, season_params, season_metrics,
                curt_mc, cw_mc, node_rng,
            )
            demand = demand_cn.copy()
            demand.columns = [cn_to_en.get(c, c) for c in demand.columns]
            demand.to_csv(OUTPUT_DIR / f"demand_{stem}.csv")

            curt_mc.to_csv(OUTPUT_DIR / f"curtailment_{stem}.csv")
            cw_mc.to_csv(OUTPUT_DIR / f"charge_window_{stem}.csv")

            arr = demand.to_numpy(dtype=float)
            total = arr.sum() / 1e9
            peak = arr.max() / 1e6
            freq_pct = float((arr > 0).mean(axis=0).mean() * 100.0)
            avg_len = mean_run_length(arr)
            curt_total = float(curt_mc.to_numpy(dtype=float).sum() / 1e9)
            cw_mean = float(cw_mc.to_numpy(dtype=float).mean())
            event_df = pd.DataFrame(event_rows)
            if not event_df.empty:
                season_counts = (
                    event_df.groupby("season").size()
                    .reindex(SEASON_ORDER, fill_value=0).to_dict()
                )
            else:
                season_counts = {s: 0 for s in SEASON_ORDER}
            print(
                f"  shape={demand.shape}, total={total:.2f} TWh, "
                f"peak={peak:.1f} GWh, freq={freq_pct:.1f}%, "
                f"avg_run={avg_len:.2f}d, injected={len(event_rows)}",
                flush=True,
            )
            print(f"  seasonal injected counts: {season_counts}", flush=True)

            summary_rows.append({
                "stem": stem,
                "events_per_virtual_year": events,
                "length_days": length,
                "mode": mode,
                "n_days": demand.shape[0],
                "total_demand_TWh": total,
                "max_single_day_GWh": peak,
                "actual_nonzero_freq_pct": freq_pct,
                "actual_mean_run_length_days": avg_len,
                "charge_side": "1 real year + 4 seasonal bootstrap MC years",
                "total_curtailment_TWh_5yr": curt_total,
                "mean_charge_window_h_5yr": cw_mean,
                "injected_event_count": len(event_rows),
                "spring_events": season_counts["spring"],
                "summer_events": season_counts["summer"],
                "autumn_events": season_counts["autumn"],
                "winter_events": season_counts["winter"],
            })
            all_event_rows.extend(event_rows)

    pd.DataFrame(summary_rows).to_csv(
        OUTPUT_DIR / "freqlen_seasonal_mc_demand_summary.csv", index=False)
    pd.DataFrame(all_event_rows).to_csv(
        OUTPUT_DIR / "freqlen_seasonal_mc_event_log.csv", index=False)
    print("\n=== seasonal MC freqlen demand complete ===", flush=True)


if __name__ == "__main__":
    main()
