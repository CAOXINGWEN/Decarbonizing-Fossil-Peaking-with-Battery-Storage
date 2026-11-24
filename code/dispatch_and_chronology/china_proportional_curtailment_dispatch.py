"""Chinese provinces with the hourly peaking task allocated in proportion to the official 2020 typical workday load curve
above its daily minimum (main provincial setting from 2026-11-06; shapes from ../provincial_intraday).

Everything else is identical to remaining_experiments_20261103 (common.solve, COPT, source_output_rated, 2035 costs, 3.6 %,
central rule; random placements obs_random_order seeds 0-999; central event extension ens.orders(n)['clustered']).
Jobs (one CSV per job in results/chunks, written atomically, so the run is resumable):
  real          real dates, both portfolios, full decomposition and delivery
  n1..n12       central event extension
  place_<k>     random placements, seeds 50k..50k+49 (B, delivery and cost components per placement)
  catalogue     the 53 added-storage configurations
  price_x       residual-task-energy price x 0.5, 0.75, 1, 1.25
  rule_r        operating rules registered, all_task, netted_tail, netted_all_task
Memory guard (cpu_budget.json): starts only when MemAvailable has stayed above 45 GB for 5 min; the pool is stopped when
MemAvailable falls below 25 GB (wait, then resume) or when the workers' summed RSS exceeds 28 GB (resume with 75 % of the
workers). Launch inside a systemd user scope with MemoryMax=32G.
  nohup systemd-run --user --scope -p MemoryMax=32G -p MemoryHigh=30G python run_prop.py --workers 64 > logs/run.log 2>&1 &
"""
from pathlib import Path
import sys, os, json, time, argparse, datetime, multiprocessing as mp
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
V86 = HERE.parents[1]
REV = V86.parent / 'Nature_Energy_v85/revision'
sys.path.insert(0, str(REV / 'remaining_experiments_20261103/code'))
sys.path.insert(0, str(REV / 'real_vs_placement_20261029/code'))
sys.path.insert(0, str(HERE.parent / 'provincial_intraday'))
CH = HERE / 'results/chunks'; LOG = HERE / 'logs'
CH.mkdir(parents=True, exist_ok=True); LOG.mkdir(exist_ok=True)
BUDGET = json.loads((REV / 'charging_window_restructure_20261102/cpu_budget.json').read_text())
MIN_AVAIL, RESUME_AVAIL, SUSTAIN, GUARD_RSS = (BUDGET['min_system_available_gb'], BUDGET['resume_system_available_gb'],
                                               BUDGET['resume_sustain_minutes'] * 60, BUDGET['guard_memory_gb'])
RULES = ['registered', 'all_task', 'netted_tail', 'netted_all_task']
PRICES = [.5, .75, 1., 1.25]


def emit(**d):
    print(json.dumps({'time': datetime.datetime.now().isoformat(timespec='seconds'), **d}), flush=True)


def available_gb():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 2**20


def provinces():
    rec = pd.read_csv(REV / 'remaining_experiments_20261103/inputs/records.csv', dtype={'uid': str})
    rec = rec[rec.group == 'provincial'].copy()
    rec['province'] = rec.system_id.str.split('-', n=1).str[1]
    assert len(rec) == 26
    return dict(zip(rec.uid, rec.province))


def jobs():
    out = []
    for uid in provinces():
        out.append(dict(uid=uid, kind='real', key=f'{uid}_real'))
        out += [dict(uid=uid, kind='central', n=n, key=f'{uid}_n{n}') for n in range(1, 13)]
        out += [dict(uid=uid, kind='place', s0=k * 50, key=f'{uid}_place_{k:02d}') for k in range(20)]
        out.append(dict(uid=uid, kind='catalogue', key=f'{uid}_catalogue'))
        out += [dict(uid=uid, kind='price', mult=m, key=f'{uid}_price_{m:g}') for m in PRICES]
        out += [dict(uid=uid, kind='rule', rule=r, key=f'{uid}_rule_{r}') for r in RULES]
    return out


_CTX = {}


def ctx():
    if not _CTX:
        import common as C
        sys.path.insert(0, str(REV / 'placement_mc_20261027/code'))
        import run_obs_placement as O
        sys.path.insert(0, str(C.V82 / 'issue6_window_saturation_20261019/scripts'))
        import window_diagnostic as WD
        sys.path.insert(0, str(C.V82 / 'advisor_key_tests_20261022/t6_window'))
        from cyclic_diagnostic import cyclic_recursion
        WD.recursion = cyclic_recursion
        import run_intraday as RI
        _CTX.update(C=C, O=O, WD=WD, shapes=RI.shapes(), prov=provinces())
    return _CTX


def prop_inputs(uid):
    """Daily peaking energy allocated by the official workday curve above its daily minimum; C3 on the same high-demand days."""
    c = ctx(); C = c['C']
    raw, ens = C.inputs(uid)
    daily = raw[:, 0].reshape(365, 24).sum(1)
    shape = c['shapes'][(c['prov'][uid], 'workday')]
    w = shape / shape.sum()
    task = daily[:, None] * w[None, :]
    ev = ens.daily.event.astype(bool).to_numpy()
    assert np.allclose(task.sum(1), daily)
    return np.c_[task.ravel(), (task * ev[:, None]).ravel(), raw[:, 2]], ens


def solve_pair(arr, mult=1.):
    C = ctx()['C']
    lf, ca = C.solve(arr, 'li_lfp_4h', mult), C.solve(arr, 'caes_24h', mult)
    row = dict(B=lf['cost_usd_kw_year'] - ca['cost_usd_kw_year'],
               lfp_tail_discharge_mwh_per_mw=lf['tail_discharge_mwh_per_mw'], hybrid_tail_discharge_mwh_per_mw=ca['tail_discharge_mwh_per_mw'],
               lfp_total_discharge_mwh_per_mw=lf['total_discharge_mwh_per_mw'], hybrid_total_discharge_mwh_per_mw=ca['total_discharge_mwh_per_mw'],
               lfp_residual_peak_per_mw=lf['residual_peak_per_mw'], hybrid_residual_peak_per_mw=ca['residual_peak_per_mw'],
               solver_status=min(lf['solver_status'], ca['solver_status']))
    row['additional_delivery_mwh_per_mw'] = row['hybrid_total_discharge_mwh_per_mw'] - row['lfp_total_discharge_mwh_per_mw']
    for k in C.COMP:
        row[f'B_{k}'] = lf[f'{k}_usd_kw_year'] - ca[f'{k}_usd_kw_year']
    return row


def job(j):
    path = CH / f"{j['key']}.csv"
    if path.exists():
        return j['key'], 'cached'
    c = ctx(); C, O, WD = c['C'], c['O'], c['WD']
    raw, ens = prop_inputs(j['uid'])
    rows = []
    if j['kind'] in ('real', 'central', 'price', 'rule'):
        order = np.asarray(ens.orders(j['n'])['clustered']) if j['kind'] == 'central' else np.arange(365)
        r2 = C.ce.reorder_days(raw, order)
        arr = C.transform(r2, j.get('rule', 'registered'))
        row = {**j, **solve_pair(arr, j.get('mult', 1.))}
        wd = WD.branch(arr, 24.0, .55, 1.5)
        row.update(window_full=wd['saturated'], window_count=wd['windows'], chargeable_hours=int((arr[:, 2] > 1e-9).sum()),
                   task_hours=int((arr[:, 0] > 1e-10).sum()))
        rows.append(row)
    elif j['kind'] == 'place':
        for s in range(j['s0'], j['s0'] + 50):
            order = np.asarray(O.obs_random_order(ens, s))
            arr = C.transform(C.ce.reorder_days(raw, order))
            row = {'uid': j['uid'], 'seed': s, **solve_pair(arr)}
            row['caes_full'] = WD.branch(arr, 24.0, .55, 1.5)['saturated']
            rows.append(row)
    elif j['kind'] == 'catalogue':
        arr = C.transform(raw)
        for cid in C.CAT:
            ans = C.solve(arr, cid)
            rows.append(dict(uid=j['uid'], variant=cid, tech=C.CAT[cid]['tech_id'], duration_h=C.CAT[cid]['duration_h'],
                             cost_usd_kw_year=ans['cost_usd_kw_year'], tail_discharge_mwh_per_mw=ans['tail_discharge_mwh_per_mw'],
                             solver_status=ans['solver_status']))
        C.model.matrices.cache_clear()
    tmp = path.with_suffix('.tmp'); pd.DataFrame(rows).to_csv(tmp, index=False); tmp.replace(path)
    return j['key'], 'ok'


def wait_for_memory():
    t0 = None
    while True:
        a = available_gb()
        if a >= RESUME_AVAIL:
            t0 = t0 or time.monotonic()
            if time.monotonic() - t0 >= SUSTAIN:
                emit(event='memory_ok', available_gb=round(a, 1)); return
        else:
            if t0 is not None:
                emit(event='memory_dropped', available_gb=round(a, 1))
            t0 = None
        time.sleep(30)


def pool_rss_gb(pool):
    tot = 0
    for p in getattr(pool, '_pool', []):
        try:
            for line in open(f'/proc/{p.pid}/status'):
                if line.startswith('VmRSS:'):
                    tot += int(line.split()[1]); break
        except (FileNotFoundError, ProcessLookupError):
            pass
    return tot / 2**20


def run(workers):
    while True:
        todo = [j for j in jobs() if not (CH / f"{j['key']}.csv").exists()]
        emit(event='pending', jobs=len(todo), workers=workers)
        if not todo:
            return
        wait_for_memory()
        pool = mp.get_context('spawn').Pool(workers, maxtasksperchild=20)
        # the guard runs in the main thread: terminating a pool from another thread leaves imap_unordered blocked
        it = pool.imap_unordered(job, todo, chunksize=1)
        done, why, t, last = 0, None, time.monotonic(), 0.
        while done < len(todo):
            if time.monotonic() - last >= 10:
                a, rss = available_gb(), pool_rss_gb(pool); last = time.monotonic()
                if a < MIN_AVAIL:
                    why = f'system available {a:.1f} GB'
                elif rss > GUARD_RSS:
                    why = f'workers rss {rss:.1f} GB'
                if why:
                    emit(event='guard_stop', reason=why); break
            try:
                it.next(timeout=5)
            except mp.TimeoutError:
                continue
            except Exception as e:                  # worker failure: stop this round, retry the remaining jobs
                why = f'error {repr(e)[:200]}'; emit(event='pool_error', error=why); break
            done += 1
            if done % 25 == 0 or done == len(todo):
                emit(event='progress', done=done, of=len(todo), minutes=round((time.monotonic() - t) / 60, 1),
                     available_gb=round(available_gb(), 1), workers_rss_gb=round(pool_rss_gb(pool), 1))
        pool.terminate(); pool.join()
        why = why or 'finished'
        if why.startswith('workers rss'):
            workers = max(8, int(workers * .75))


if __name__ == '__main__':
    from cpu_budget import apply_cpu_budget, limit_workers
    apply_cpu_budget()
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=64); ap.add_argument('--test', action='store_true')
    a = ap.parse_args()
    if a.test:
        t = time.monotonic(); j = [x for x in jobs() if x['kind'] == 'real'][0]
        (CH / f"{j['key']}.csv").unlink(missing_ok=True)
        print(job(j), round(time.monotonic() - t, 1), 's'); sys.exit()
    emit(event='start', jobs=len(jobs()), workers=limit_workers(a.workers))
    run(limit_workers(a.workers))
    emit(event='all_done')
