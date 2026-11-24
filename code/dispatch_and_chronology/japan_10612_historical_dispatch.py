"""Re-optimize real dates for the complete paired Japanese fiscal years.

Reuses the registered cyclic dispatch model, price scenario, technology
catalogue and immutable hourly inputs. Each completed shard is written once
by atomic rename. q50 serves as a record-by-record replication control.
"""
from pathlib import Path
import argparse
import json
import multiprocessing as mp
import sys
import time

WORK = Path(__file__).resolve().parents[1]
ORIGINAL = WORK.parent / "japan_5309_audit_20261117"
sys.path.insert(0, str(ORIGINAL / "code"))
import run_full_10612_suite as model
import numpy as np
import pandas as pd

QUANTILES = (5, 10, 20, 25, 30, 35, 50)
N_RECORDS = 10612
SHARD_SIZE = 12
SHARDS = WORK / "results/real_shards"
MONTH_MASKS = tuple(model._CAL_M == month for month in range(1, 13))
SEASONS = tuple(model._SEG == season for season in range(4))


def task_profiles(imports):
    """Compute all month-by-clock-hour quantiles within the given fiscal year."""
    levels = np.array(QUANTILES) / 100
    residual = np.empty((len(levels), model.N_HOURS), dtype=np.float64)
    for mask in MONTH_MASKS:
        block = imports[mask].reshape(-1, 24)
        baselines = np.quantile(block, levels, axis=0)
        excess = np.maximum(block[None, :, :] - baselines[:, None, :], 0)
        residual[:, mask] = excess.reshape(len(levels), -1)
    daily = residual.reshape(len(levels), model.N_DAYS, 24).sum(axis=2)
    flags = np.empty(daily.shape, dtype=bool)
    for mask in SEASONS:
        seasonal_load = daily[:, mask]
        flags[:, mask] = seasonal_load >= np.quantile(seasonal_load, 0.75, axis=1)[:, None]
    return zip(QUANTILES, residual, flags)


def solve_branch(indexer, tail):
    base = model._CAT58["li_lfp_4h"]
    lp = model.PersistentBranchLP(indexer, base, tail, model.BASE_SCENARIO_EUR)
    try:
        return lp.full_solution()
    finally:
        lp.dispose()


def solve_record(row_index):
    meta = model._META.iloc[row_index]
    uid, _ = model._uid_of(meta.system_id, int(meta.observation_year))
    imports = model._HOURLY[row_index, :, 0]
    exports = model._HOURLY[row_index, :, 1]
    total_import = float(imports.sum())
    power_mw = 0.01 * float(imports.mean())
    if not np.isclose(power_mw, meta.P_ref_mw, rtol=1e-12, atol=1e-15):
        raise ValueError(f"Power mismatch in {uid}")

    result = []
    for q, task, flags in task_profiles(imports):
        tail = task * np.repeat(flags, 24)
        inputs = model._transform_rule(np.column_stack((task, tail, exports)) / power_mw)
        indexer = model.CollapsedIndexer(inputs)
        battery = solve_branch(indexer, model._CAT58["li_lfp_4h"])
        hybrid = solve_branch(indexer, model._CAT58["caes_24h"])
        budget = battery["cost_usd_kw_year"] - hybrid["cost_usd_kw_year"]
        runs = model._runs(flags)
        result.append({
            "row_idx": row_index, "uid": uid, "system_id": meta.system_id,
            "prefecture": meta.prefecture, "fiscal_year": int(meta.observation_year), "q": q,
            "ref_power_mw": power_mw, "import_mwh": total_import, "task_mwh": float(task.sum()),
            "task_share_of_import": float(task.sum() / total_import),
            "tail_task_mwh": float(tail.sum()), "task_hours": int((inputs[:, 0] > 1e-10).sum()),
            "charge_hours": int((inputs[:, 2] > 1e-9).sum()), "event_days": int(flags.sum()),
            "events": len(runs), "mean_event_days": float(flags.sum() / len(runs)),
            "long_windows": model._long_windows_fast(inputs),
            "lfp_cost_usd_kw_yr": battery["cost_usd_kw_year"],
            "hybrid_cost_usd_kw_yr": hybrid["cost_usd_kw_year"], "B_usd_kw_yr": budget,
            "lfp_tail_discharge_mwh_per_mw": battery["tail_discharge_mwh_per_mw"],
            "hybrid_tail_discharge_mwh_per_mw": hybrid["tail_discharge_mwh_per_mw"],
            "lfp_efc": battery["tail_discharge_mwh_per_mw"] / 4,
            "acaes_efc": hybrid["tail_discharge_mwh_per_mw"] / 24,
            "lfp_status": battery["solver_status"], "hybrid_status": hybrid["solver_status"],
        })
    return result


def shard_path(start):
    return SHARDS / f"real_{start:05d}.csv.gz"


def solve_shard(start):
    output = shard_path(start)
    records = []
    stop = min(start + SHARD_SIZE, N_RECORDS)
    for index in range(start, stop):
        records.extend(solve_record(index))
    frame = pd.DataFrame.from_records(records)
    temporary = output.with_suffix(".part")
    frame.to_csv(temporary, index=False, compression={"method": "gzip", "compresslevel": 3})
    temporary.replace(output)
    model._trim_heap()
    return stop - start


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--workers", type=int, default=96)
    args = parser.parse_args()
    if len(model._META) != N_RECORDS or not np.array_equal(model._META.row_idx, np.arange(N_RECORDS)):
        raise ValueError("The Japanese cohort or the mapping to hourly rows has changed")
    SHARDS.mkdir(exist_ok=True, parents=True)
    starts = range(0, N_RECORDS, SHARD_SIZE)
    remaining = [start for start in starts if not shard_path(start).is_file()]
    finished = N_RECORDS - sum(min(SHARD_SIZE, N_RECORDS - i) for i in remaining)
    workers = model.limit_workers(args.workers)
    start_time = time.monotonic()
    print(json.dumps(dict(records=N_RECORDS, q=QUANTILES, workers=workers,
                          completed_before_start=finished)), flush=True)
    if remaining:
        with mp.get_context("fork").Pool(workers, initializer=model._init_worker,
                                         maxtasksperchild=14) as pool:
            for counter, completed in enumerate(pool.imap_unordered(solve_shard, remaining), 1):
                finished += completed
                if counter % 40 == 0 or finished == N_RECORDS:
                    print(json.dumps(dict(records_finished=finished, records_total=N_RECORDS,
                                          elapsed_s=round(time.monotonic() - start_time, 1))), flush=True)
    if finished != N_RECORDS:
        raise RuntimeError(f"Unsolved records: {N_RECORDS-finished}")


if __name__ == "__main__":
    main()
