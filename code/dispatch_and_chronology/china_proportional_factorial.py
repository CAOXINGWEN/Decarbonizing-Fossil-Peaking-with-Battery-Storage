"""Surplus-energy x charge-power x charging-access factorial for the 26 provinces with the typical-curve allocation.

Same 2 x 2 x 2 design, solver call and summary as Nature_Energy_v85/revision/surplus_factorial_q50_20261103
(factorial_q50.py); only the provincial inputs change (run_prop.prop_inputs, central rule applied after the allocation, so
the daily-minimum hour keeps its surplus). National and community records are taken from the delivered q50 run.
Check: the (1, 1, observed) cell reproduces results/provincial_real.csv (real dates) and provincial_central.csv.gz (n = 12).
Memory guard and resumable chunks as run_prop.py. Launch as a systemd user service:
  systemd-run --user --unit=prop-factorial -p MemoryMax=32G -p MemoryHigh=30G --working-directory=$PWD \
      /bin/bash -c "exec python factorial_prop.py --workers 26 >> logs/factorial.log 2>&1"
  python factorial_prop.py --summary
"""
from pathlib import Path
import sys, json, time, argparse, multiprocessing as mp
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_prop as P                                                     # noqa: E402
FQ = P.REV / 'surplus_factorial_q50_20261103'
CH = HERE / 'results/factorial_chunks'; OUT = HERE / 'results'
CH.mkdir(parents=True, exist_ok=True)
CELLS = [(e, p, a) for e in (1, 4) for p in (1, 4) for a in ('observed', 'expanded')]


def F():
    sys.path.insert(0, str(FQ / 'code'))
    import factorial_q50 as f
    return f


def path(uid, e, p, a):
    return CH / f'{uid}_e{e}_p{p}_{a}.csv'


def jobs():
    return [dict(uid=u, e=e, p=p, a=a, key=path(u, e, p, a).stem) for u in P.provinces() for e, p, a in CELLS]


def job(j):
    out = path(j['uid'], j['e'], j['p'], j['a'])
    if out.exists():
        return j['key'], 'cached'
    f = F(); C = P.ctx()['C']
    u = {x['uid']: x for x in f.units()}[j['uid']]
    raw, ens = P.prop_inputs(j['uid'])
    arr = C.transform(raw)
    sc = {**f.R.P.run.M['scenario'], 'firm_value_eur_kw_year': f.R.P.run.vf(f.R.P.RATE) / f.R.P.FX}
    rows = []
    for order, o in (('real', None), ('n12', np.asarray(ens.orders(12)['clustered']))):
        a0 = arr if o is None else f.ce.reorder_days(arr, o)
        a, usable, err = f.access(a0, j['e'], j['a'])
        rows.append(dict(uid=j['uid'], regime='China', block=f.block(u), order=order, energy_multiplier=j['e'],
                         charge_power_multiplier=j['p'], access=j['a'], B=f.solve_B(a, sc, j['p']), usable_surplus=usable,
                         conservation_error=err))
    tmp = out.with_suffix('.tmp'); pd.DataFrame(rows).to_csv(tmp, index=False); tmp.replace(out)
    return j['key'], 'ok'


def run(workers):
    while True:
        todo = [j for j in jobs() if not path(j['uid'], j['e'], j['p'], j['a']).exists()]
        P.emit(event='pending', jobs=len(todo), workers=workers)
        if not todo:
            return
        P.wait_for_memory()
        pool = mp.get_context('spawn').Pool(workers, maxtasksperchild=20)
        it = pool.imap_unordered(job, todo, chunksize=1)
        done, why, t, last = 0, None, time.monotonic(), 0.
        while done < len(todo):
            if time.monotonic() - last >= 10:
                a, rss = P.available_gb(), P.pool_rss_gb(pool); last = time.monotonic()
                if a < P.MIN_AVAIL:
                    why = f'system available {a:.1f} GB'
                elif rss > P.GUARD_RSS:
                    why = f'workers rss {rss:.1f} GB'
                if why:
                    P.emit(event='guard_stop', reason=why); break
            try:
                it.next(timeout=5)
            except mp.TimeoutError:
                continue
            except Exception as e:
                why = f'error {repr(e)[:200]}'; P.emit(event='pool_error', error=why); break
            done += 1
            if done % 25 == 0 or done == len(todo):
                P.emit(event='progress', done=done, of=len(todo), minutes=round((time.monotonic() - t) / 60, 1),
                       available_gb=round(P.available_gb(), 1), workers_rss_gb=round(P.pool_rss_gb(pool), 1))
        pool.terminate(); pool.join()
        if (why or '').startswith('workers rss'):
            workers = max(8, int(workers * .75))


def summary():
    f = F()
    prov = pd.concat([pd.read_csv(x, dtype={'uid': str}) for x in sorted(CH.glob('*.csv'))], ignore_index=True)
    assert len(prov) == 26 * 8 * 2, len(prov)
    # reproduction of the main provincial runs
    base = prov[(prov.energy_multiplier == 1) & (prov.charge_power_multiplier == 1) & (prov.access == 'observed')]
    real = pd.read_csv(OUT / 'provincial_real.csv', dtype={'uid': str}).set_index('uid').B
    c12 = pd.read_csv(OUT / 'provincial_central.csv.gz', dtype={'uid': str}); c12 = c12[c12.target == 12].set_index('uid').B
    b = base.set_index(['order', 'uid']).B
    e_real = float((b.loc['real'] - real.reindex(b.loc['real'].index)).abs().max())
    e12 = float((b.loc['n12'] - c12.reindex(b.loc['n12'].index)).abs().max())
    assert e_real < 1e-6 and e12 < 1e-6, (e_real, e12)
    other = pd.read_csv(FQ / 'results/factorial_q50_records.csv.gz', dtype={'uid': str})
    other = other[other.regime != 'China']
    d = pd.concat([other, prov[other.columns]], ignore_index=True)
    assert len(d) == 185 * 16 and d.uid.nunique() == 185
    d.to_csv(OUT / 'factorial_records.csv.gz', index=False)
    # same cells and contrasts as factorial_q50.summary
    f.CH = CH; f.OUT = OUT
    w = d.pivot_table(index=['uid', 'regime', 'block', 'energy_multiplier', 'charge_power_multiplier', 'access'], columns='order', values='B').reset_index()
    w['dB12'] = w.n12 - w.real
    w['context'] = w.regime.map({'Europe': 'national', 'China': 'provincial', 'Japan': 'community'})
    cells, contrasts = [], []
    for outcome in ('real', 'dB12'):
        for (g, e, p, a), x in w.groupby(['context', 'energy_multiplier', 'charge_power_multiplier', 'access']):
            est, lo, hi = f.interval(x, outcome)
            cells.append(dict(outcome=outcome, context=g, energy_multiplier=e, charge_power_multiplier=p, access=a, n=len(x), blocks=x.block.nunique(),
                              block_mean=est, ci95_low=lo, ci95_high=hi, median=float(x[outcome].median()),
                              hybrid_cheaper=int((x[outcome] >= 0).sum()) if outcome == 'real' else None))
        wide = w.pivot_table(index=['uid', 'context', 'block'], columns=['energy_multiplier', 'charge_power_multiplier', 'access'], values=outcome)
        col = lambda e, p, a: wide[(e, p, a)]
        C = {'surplus_energy_4x': sum(col(4, p, a) - col(1, p, a) for p in (1, 4) for a in ('observed', 'expanded')) / 4,
             'charge_power_4x': sum(col(e, 4, a) - col(e, 1, a) for e in (1, 4) for a in ('observed', 'expanded')) / 4,
             'expanded_access': sum(col(e, p, 'expanded') - col(e, p, 'observed') for e in (1, 4) for p in (1, 4)) / 4}
        idx = wide.reset_index()[['uid', 'context', 'block']]
        for name, vals in C.items():
            z = idx.copy(); z['v'] = vals.to_numpy(float)
            for g, x in z.groupby('context'):
                est, lo, hi = f.interval(x, 'v')
                contrasts.append(dict(outcome=outcome, context=g, contrast=name, n=len(x), blocks=x.block.nunique(), block_mean=est, ci95_low=lo,
                                      ci95_high=hi, median=float(x.v.median()), positive=int((x.v > 1e-6).sum()), negative=int((x.v < -1e-6).sum())))
    pd.DataFrame(cells).to_csv(OUT / 'factorial_cells.csv', index=False)
    pd.DataFrame(contrasts).to_csv(OUT / 'factorial_contrasts.csv', index=False)
    qa = dict(max_abs_diff_real=e_real, max_abs_diff_n12=e12, max_conservation_error=float(prov.conservation_error.abs().max()))
    (OUT / 'factorial_qa.json').write_text(json.dumps(qa, indent=1))
    pd.set_option('display.width', 250)
    print(json.dumps(qa)); print(pd.DataFrame(contrasts).round(2).to_string(index=False))
    print(pd.DataFrame(cells).query("outcome == 'real'").round(2).to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=26); ap.add_argument('--summary', action='store_true')
    ap.add_argument('--test', action='store_true')
    # author-approved overrides of the cpu_budget.json memory thresholds (GB); defaults keep the budget file
    ap.add_argument('--start-avail', type=float, default=None); ap.add_argument('--min-avail', type=float, default=None)
    a = ap.parse_args()
    if a.start_avail is not None:
        P.RESUME_AVAIL, P.SUSTAIN = a.start_avail, 60
    if a.min_avail is not None:
        P.MIN_AVAIL = a.min_avail
    if a.summary:
        summary(); sys.exit()
    if a.test:
        # flat allocation through this path reproduces the delivered factorial input; then one provincial job
        f = F(); C = P.ctx()['C']
        u = next(x for x in f.units() if x['regime'] == 'China')
        raw, _ = C.inputs(u['uid']); arr_f, _ = f.inputs(u)
        assert np.allclose(C.transform(raw), arr_f), 'input paths differ'
        t = time.monotonic(); j = jobs()[0]; path(j['uid'], j['e'], j['p'], j['a']).unlink(missing_ok=True)
        print(job(j), round(time.monotonic() - t, 1), 's')
        r = pd.read_csv(path(j['uid'], j['e'], j['p'], j['a'])); print(r[['order', 'B']])
        print('main real B', pd.read_csv(OUT / 'provincial_real.csv', dtype={'uid': str}).set_index('uid').B[j['uid']]); sys.exit()
    from cpu_budget import apply_cpu_budget, limit_workers
    apply_cpu_budget()
    P.emit(event='start', jobs=len(jobs()), workers=limit_workers(a.workers), start_avail_gb=P.RESUME_AVAIL, min_avail_gb=P.MIN_AVAIL)
    run(limit_workers(a.workers))
    P.emit(event='all_done')
