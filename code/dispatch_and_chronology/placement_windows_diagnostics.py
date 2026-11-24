"""Charging windows on real dates and under the 1,000 random start/end placements (R3 mechanism; no optimisation).

For every main-text record (185) and placement (seed -1 = real dates, seeds 0-999 = obs_random_order of the report), counts
  gap1      gaps of exactly one ordinary day between consecutive high-demand-day runs (within the operating year)
  long_win  charging windows (runs of hours without high-demand-day task) whose usable surplus can bring the A-CAES from
            empty to usable capacity: sum_t min(R_t, 1.5) * 0.55 >= 21.6 MWh/MW
Writes analysis/placement_windows.csv.gz and prints group medians.
"""
from pathlib import Path
import sys, multiprocessing as mp
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'figures/code'))
import data as D
sys.path.insert(0, str(D.REV / 'placement_mc_20261027/code'))
from run_obs_placement import obs_random_order, ce   # noqa: E402

SEEDS = [-1] + list(range(1000))


def stats(arr, flags):
    call = arr[:, 1] > 1e-10
    r = np.where(call, 0.0, np.minimum(arr[:, 2], 1.5))
    run = np.cumsum(np.r_[0, np.diff(call.astype(np.int8)) != 0])
    e = np.bincount(run, weights=r)
    first = np.r_[0, np.flatnonzero(np.diff(run)) + 1]
    is_call = call[first]
    long_win = int(((e * 0.55 >= 21.6 - 1e-9) & ~is_call).sum())
    # ordinary-day gaps between event runs
    f = flags.astype(int); edges = np.diff(np.r_[0, f, 0]); starts = np.flatnonzero(edges == 1); ends = np.flatnonzero(edges == -1)
    gaps = starts[1:] - ends[:-1]
    return long_win, int((gaps == 1).sum()), len(starts)


def job(uid):
    arr, ev = D.hourly(uid)
    z = np.load(D.INP / f'{uid}.npz')
    ens = ce.ChronologyEnsemble(pd.DataFrame({'segment': z['segment'], 'event': z['event']}), str(z['unit_key']))
    rows = []
    for s in SEEDS:
        o = np.arange(365) if s < 0 else np.asarray(obs_random_order(ens, s))
        a = arr if s < 0 else ce.reorder_days(arr, o)
        lw, g1, ne = stats(a, ev[o])
        rows.append((uid, s, lw, g1, ne))
    return rows


if __name__ == '__main__':
    sys.path.insert(0, str(D.REV / 'real_vs_placement_20261029/code'))
    from cpu_budget import limit_workers
    rec = D.records(); rec = rec[rec.group.isin(D.MAIN)]
    with mp.get_context('spawn').Pool(limit_workers(32)) as pool:
        out = [r for rows in pool.imap_unordered(job, rec.uid.tolist(), chunksize=2) for r in rows]
    d = pd.DataFrame(out, columns=['uid', 'seed', 'long_windows', 'one_day_gaps', 'events']).merge(rec[['uid', 'group']], on='uid')
    d['group'] = d.group.astype(str)
    d.to_csv(HERE / 'placement_windows.csv.gz', index=False)
    real = d[d.seed < 0].set_index('uid'); rnd = d[d.seed >= 0].groupby('uid')[['long_windows', 'one_day_gaps', 'events']].median()
    t = real[['group', 'long_windows', 'one_day_gaps', 'events']].join(rnd, rsuffix='_random')
    print(t.groupby('group')[['long_windows', 'long_windows_random', 'one_day_gaps', 'one_day_gaps_random', 'events', 'events_random']].median())
    print('records with fewer long windows on real dates than the random median:',
          t.assign(f=t.long_windows < t.long_windows_random).groupby('group').f.agg(['sum', 'size']).to_dict())
