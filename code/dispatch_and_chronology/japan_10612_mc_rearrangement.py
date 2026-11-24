"""Randomize event placement for the Japanese real-date baseline cohort.

Each checkpoint contains the two-portfolio optimal cost difference and number
of long windows for seeds 0..999, indexed by seed, for one fiscal year and q.
Both portfolios share the identical permutation. Complete q50 matrices were
computed by the registered suite and are checked against fresh optimizations.
"""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time

HERE = Path(__file__).resolve().parents[1]
REAL = HERE.parent / "japan_baseline_10612_20261118"
sys.path.insert(0, str(REAL / "code"))
from run_real import model, task_profiles, N_RECORDS, QUANTILES
import numpy as np
import pandas as pd
from compact_dispatch import CompactLP, DayGraph

RESULTS = HERE / "results"
STAGING = Path("/tmp/japan_multibaseline_mc_10612_20261118")
SEEDS = 1000
NEW_Q = (10, 35, 5, 20, 25, 30)
Q50 = model.OUT_DIR
REAL_TABLE = REAL / "results/japan_real_baseline_records.csv.gz"
BASELINE = pd.read_csv(REAL_TABLE, usecols=["row_idx", "uid", "q", "B_usd_kw_yr", "long_windows"])
REF_B = BASELINE.pivot(index="row_idx", columns="q", values="B_usd_kw_yr")
REF_WINDOWS = BASELINE.pivot(index="row_idx", columns="q", values="long_windows")
UIDS = BASELINE.query("q==50").sort_values("row_idx").uid.to_numpy()


def checkpoint(q, row):
    return STAGING / f"q{q:02d}" / f"{row:05d}.npz"


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inputs(row, q):
    meta = model._META.iloc[row]
    uid, key = model._uid_of(meta.system_id, int(meta.observation_year))
    if uid != UIDS[row]:
        raise RuntimeError(f"Cohort row mismatch at {row}")
    imports, exports = model._HOURLY[row].T
    _, task, flags = next(entry for entry in task_profiles(imports) if entry[0] == q)
    power = .01 * imports.mean()
    arr = model._transform_rule(np.c_[task, task*np.repeat(flags, 24), exports]/power)
    return key, flags, arr


def orders_and_windows(key, flags, arr, seeds):
    segments = model._precompute_obs_segments(model._SEG, flags)
    hourly_days = arr.reshape(365, 24, 3)
    prev = np.empty((len(seeds), 365), dtype=np.int32)
    windows = np.empty(len(seeds), dtype=np.int16)
    for pos, seed in enumerate(seeds):
        order = model._fast_obs_random_order(key, segments, int(seed))
        if pos in (0, len(seeds)-1) and not np.array_equal(np.sort(order), np.arange(365)):
            raise RuntimeError(f"Invalid day permutation: {key}, seed {seed}")
        prev[pos] = model._prev_from_order(order)
        windows[pos] = model._long_windows_fast(hourly_days[order].reshape(8760, 3))
    return prev, windows


def solve_order(previous):
    """Visit nearby calendar permutations for the simplex warm start only."""
    n = len(previous)
    remaining = np.ones(n, dtype=bool)
    current = model.INIT_PREV
    for _ in range(n):
        distances = np.count_nonzero(previous != current, axis=1)
        distances[~remaining] = 366
        index = int(np.argmin(distances))
        remaining[index] = False
        current = previous[index]
        yield index


def simulate(row, q, seeds, compact=True):
    key, flags, arr = inputs(row, q)
    actual_windows = model._long_windows_fast(arr)
    if actual_windows != REF_WINDOWS.at[row, q]:
        raise RuntimeError(f"Real-date window mismatch: {row}, {q}")
    previous, windows = orders_and_windows(key, flags, arr, seeds)
    battery = model._CAT58["li_lfp_4h"]
    caes = model._CAT58["caes_24h"]
    if compact:
        graph = DayGraph(arr)
        short = CompactLP(graph, battery, battery, model)
        long = CompactLP(graph, battery, caes, model)
    else:
        graph = model.CollapsedIndexer(arr)
        short = model.PersistentBranchLP(graph, battery, battery, model.BASE_SCENARIO_EUR)
        long = model.PersistentBranchLP(graph, battery, caes, model.BASE_SCENARIO_EUR)
    try:
        real = short.solve_cost_usd() - long.solve_cost_usd()
        if abs(real - REF_B.at[row, q]) > 1e-6:
            raise RuntimeError(f"Real-date optimum mismatch: {row}, q{q}, {real}")
        costs = np.empty(len(seeds), dtype=np.float64)
        for i in solve_order(previous):
            short.rewire(previous[i])
            long.rewire(previous[i])
            costs[i] = short.solve_cost_usd() - long.solve_cost_usd()
    finally:
        short.dispose()
        long.dispose()
    if not np.isfinite(costs).all():
        raise RuntimeError(f"Non-finite optimized cost: {key}, q{q}")
    return costs, windows, real


def calculate_case(payload):
    row, q = payload
    output = checkpoint(q, row)
    if output.is_file():
        return q, row, 0.0
    start = time.monotonic()
    costs, windows, actual = simulate(row, q, range(SEEDS))
    temporary = output.with_suffix(".tmp.npz")
    np.savez(temporary, B=costs, lw=windows, q=q, row=row, uid=UIDS[row], real_B=actual, engine="compact")
    temporary.replace(output)
    model._trim_heap()
    return q, row, round(time.monotonic()-start, 2)


def validate_archive():
    model._init_worker()
    costs = np.load(Q50/"japan_mc_B.npy", mmap_mode="r")
    windows = np.load(Q50/"japan_mc_lw.npy", mmap_mode="r")
    if costs.shape != (N_RECORDS, SEEDS) or windows.shape != costs.shape:
        raise RuntimeError("Incomplete q50 matrix")
    max_error = 0.0
    seeds = [0, 17, 511, 999]
    for row in [0, 1643, 3968, 5801, 8810, 10611]:
        recomputed, counts, _ = simulate(row, 50, seeds, compact=False)
        max_error = max(max_error, float(np.max(abs(recomputed-costs[row, seeds]))))
        if not np.array_equal(counts, windows[row, seeds]):
            raise RuntimeError(f"q50 seed/window mismatch at row {row}")
    if max_error > 1e-5:
        raise RuntimeError(f"q50 seed/cost mismatch: {max_error}")
    real = pd.read_csv(Q50/"japan_records.csv.gz").sort_values("row_idx")
    assert np.array_equal(real.uid, UIDS)
    assert np.isfinite(costs).all() and (windows >= 0).all()
    per_record = pd.read_csv(Q50/"japan_placements_records.csv.gz").set_index("uid").loc[UIDS]
    assert np.allclose((costs>0).mean(axis=1), per_record.mc_win36, atol=1e-12, rtol=0)
    result = dict(records=N_RECORDS, seeds=SEEDS, pairs=N_RECORDS*SEEDS,
                  all_values_finite=True, checked_optimizations=len(seeds)*6*2,
                  max_B_difference=max_error, windows_identical_in_checks=True,
                  winning_placements=int((costs>0).sum()), within_row_probabilities_verified=N_RECORDS,
                  sha256_B=file_digest(Q50/"japan_mc_B.npy"), sha256_windows=file_digest(Q50/"japan_mc_lw.npy"))
    write_json(RESULTS/"q50_verification.json", result)
    print(json.dumps(result), flush=True)


def run(workers):
    contract = dict(source_hourly_sha256=file_digest(model.DATA_DIR/"japan_dual_hourly.npy"),
                    source_calendar_sha256=file_digest(model.DATA_DIR/"japan_calendar.npz"),
                    source_metadata_sha256=file_digest(model.DATA_DIR/"japan_dual_metadata.csv"),
                    real_results_sha256=file_digest(REAL_TABLE),
                    model_sha256=file_digest(Path(model.__file__)), seeds=SEEDS, q=list(QUANTILES),
                    target="observed_lengths_random_gaps_registered_cyclic_LP")
    prior = RESULTS/"input_contract.json"
    if prior.exists() and json.loads(prior.read_text()) != contract:
        raise RuntimeError("The source inputs changed since the previous run")
    expected = json.loads((REAL/"results/verification.json").read_text())
    if contract["source_hourly_sha256"] != expected["annual_import_array_sha256"]:
        raise RuntimeError("The source hourly array differs from the verified baseline cohort")
    write_json(prior, contract)
    STAGING.mkdir(mode=0o700, exist_ok=True, parents=True)
    stamp = STAGING/"input_contract.json"
    if stamp.exists() and json.loads(stamp.read_text()) != contract:
        raise RuntimeError("Staging checkpoints belong to different inputs")
    write_json(stamp, contract)
    for q in NEW_Q:
        (STAGING/f"q{q:02d}").mkdir(exist_ok=True)
    # The shuffled processing order mixes prefectures/years; seeds remain fixed by unit and season.
    rows = np.random.default_rng(20261118).permutation(N_RECORDS).tolist()
    todo = [(row, q) for row in rows for q in NEW_Q if not checkpoint(q, row).is_file()]
    completed = {q: N_RECORDS - sum(qq==q for _,qq in todo) for q in NEW_Q}
    completed[50] = N_RECORDS
    workers = model.limit_workers(workers)
    start = time.monotonic()

    def progress(status, new):
        elapsed = time.monotonic()-start
        counts = {str(q): completed[q] for q in sorted(completed)}
        result = dict(status=status, complete_records_by_q=counts, placements_done=sum(completed.values())*SEEDS,
                      target_placements=N_RECORDS*len(QUANTILES)*SEEDS, worker_processes=workers,
                      new_cases_this_run=new, wall_seconds=round(elapsed, 1), unix_time=time.time())
        if new:
            result["estimated_hours_remaining"] = round((len(todo)-new)*elapsed/new/3600, 2)
        write_json(RESULTS/"progress.json", result)
        print(json.dumps(result), flush=True)

    progress("running", 0)
    previous_update = time.monotonic()
    if todo:
        with mp.get_context("fork").Pool(workers, initializer=model._init_worker, maxtasksperchild=25) as pool:
            for new, (q, row, seconds) in enumerate(pool.imap_unordered(calculate_case, todo, chunksize=1), 1):
                completed[q] += 1
                if time.monotonic()-previous_update >= 30 or new == len(todo):
                    progress("running", new)
                    previous_update = time.monotonic()
                    if (RESULTS/"stop_after_checkpoint").is_file():
                        progress("stopped_at_checkpoint", new)
                        raise SystemExit(75)
    progress("all_seeds_solved", len(todo))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--check-q50", action="store_true")
    parser.add_argument("--workers", type=int, default=128)
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.check_q50:
        validate_archive()
    else:
        with (RESULTS/"run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            run(args.workers)
