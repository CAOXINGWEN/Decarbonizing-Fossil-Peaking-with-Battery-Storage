"""Recompute all 540 manuscript procurement designs with resumable per-design output."""
from pathlib import Path
import argparse
import gc
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
import traceback
import warnings
from functools import lru_cache

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning

warnings.filterwarnings('ignore', message='Unrecognized options detected.*', category=OptimizeWarning)
HERE = Path(__file__).resolve().parent
WORK = HERE.parent
REV = WORK.parent
CAP = REV / 'capacity_expansion_validation'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def prepare():
    base = module('capacity_v68_prepare', HERE / 'capacity_model_v68.py')
    chron = module('chronology_v68_prepare', CAP / 'run_v62_chronology_procurement_counterfactual.py')
    provenance = {
        'model': sha(HERE / 'capacity_model_v68.py'), 'chronology': sha(Path(chron.__file__)),
        'panel': sha(base.PANEL_PATH), 'costs': sha(base.COST_CATALOGUE_PATH),
        'values': sha(base.VALUE_SCENARIO_PATH), 'registered_runner': sha(base.REGISTERED_RUNNER_PATH),
        'cost_year': 2035, 'value_scenario': 20,
    }
    fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    cache = WORK / 'scenario_cache'
    cache.mkdir(exist_ok=True)
    completed = cache / 'manifest.json'
    if completed.exists():
        saved = json.loads(completed.read_text())
        assert saved['fingerprint'] == fingerprint
        return saved
    usecols = ['country','year','timestamp_utc','source_day_id','season_segment',
               'event_flag_q75','task_c3_q75_mw','raw_gas_mean_mw',base.TASK_COLUMN,base.RECHARGE_COLUMN]
    panel = pd.read_csv(base.PANEL_PATH, usecols=usecols)
    runner = base.load_registered_runner()
    designs = []
    conservation = []
    countries = sorted(panel['country'].unique())
    for country in countries:
        frame = panel.loc[panel['country'].eq(country)].copy()
        for step in (2,4,6):
            ensembles, checks = chron.build_ensemble_scenarios(base, frame, runner, step)
            for row in checks:
                conservation.append({'country': country, 'planning_step_hours': step, **row})
            with (cache / f'{country}_{step}h.pkl').open('wb') as stream:
                pickle.dump(ensembles, stream, protocol=4)
            for ensemble in ('observed','dispersed','clustered','combined'):
                modes = ['diversified']
                if ensemble == 'combined' and step in (4,6):
                    modes.extend(base.CONFIG_IDS)
                for risk in (0.,0.5):
                    for mode in modes:
                        designs.append({'country':country,'step':step,'ensemble':ensemble,'risk':risk,'mode':mode})
        print(f'prepared {country}', flush=True)
    assert len(designs) == 540 and len(countries) == 15
    pd.DataFrame(conservation).to_csv(cache / 'chronology_conservation.csv', index=False)
    audit = REV.parents[4] / 'artifacts/nature_energy_v67_issue2_capacity_bound_20260906/binding_designs.csv'
    binding = pd.read_csv(audit)
    urgent = {(r.country,int(r.planning_step_hours),r.ensemble,float(r.risk_weight),r.mode)
              for r in binding.itertuples()}
    for d in designs:
        d['priority'] = (d['country'],d['step'],d['ensemble'],d['risk'],d['mode']) in urgent
        d['key'] = f"{d['country']}_{d['step']}h_{d['ensemble']}_{d['risk']:g}_{d['mode']}"
    designs.sort(key=lambda d: (not d['priority'], d['step'], d['mode'] != 'diversified', d['key']))
    manifest = {'fingerprint':fingerprint,'provenance':provenance,'designs':designs,
                'countries':countries,'priority_count':sum(d['priority'] for d in designs)}
    assert manifest['priority_count'] == 12
    write_json(completed, manifest)
    return manifest


BASE = None


def initialize(fingerprint):
    global BASE, TECH, VALUE, FINGERPRINT
    BASE = module('capacity_v68_worker', HERE / 'capacity_model_v68.py')
    TECH = BASE.load_technologies(2035)
    VALUE = BASE.load_value_scenario(20)
    FINGERPRINT = fingerprint


@lru_cache(maxsize=2)
def scenarios(country, step):
    with (WORK / 'scenario_cache' / f'{country}_{step}h.pkl').open('rb') as stream:
        return pickle.load(stream)


def solve(design):
    start = time.time()
    target = WORK / 'designs' / (design['key'] + '.json')
    try:
        summary, capacities, metrics = BASE.solve_country_design_highs(
            design['country'], scenarios(design['country'],design['step'])[design['ensemble']],
            TECH,VALUE,design['risk'],design['mode'],design['step'],1)
        data = {'fingerprint':FINGERPRINT,'design':design,'summary':summary,
                'capacities':capacities,'scenario_metrics':metrics,'elapsed_seconds':time.time()-start}
        write_json(target,data)
        result = {'key':design['key'],'status':'ok','seconds':round(time.time()-start,1),
                  'objective':summary['solver_objective_eur_per_normalized_mw'],
                  'power':summary['total_power_capacity_per_normalized_mw'],
                  'violation':summary['maximum_inequality_violation']}
    except Exception:
        result = {'key':design['key'],'status':'error','traceback':traceback.format_exc()}
        write_json(WORK / 'errors' / (design['key']+'.json'),result)
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--priority-only',action='store_true')
    parser.add_argument('--exclude-priority',action='store_true')
    args = parser.parse_args()
    for name in ('designs','errors','qa'):
        (WORK/name).mkdir(exist_ok=True)
    manifest = prepare()
    pending = []
    for d in manifest['designs']:
        if args.priority_only and not d['priority']:
            continue
        if args.exclude_priority and d['priority']:
            continue
        path = WORK/'designs'/(d['key']+'.json')
        if path.exists():
            assert json.loads(path.read_text())['fingerprint'] == manifest['fingerprint']
        else:
            pending.append(d)
    print(json.dumps({'pending':len(pending),'workers':args.workers,'priority_only':args.priority_only}),flush=True)
    # Spawn isolates the solver runtime; limit each process lifetime for bounded memory.
    with mp.get_context('spawn').Pool(args.workers, initializer=initialize,
                                     initargs=(manifest['fingerprint'],), maxtasksperchild=12) as pool:
        for index, result in enumerate(pool.imap_unordered(solve,pending,chunksize=1),1):
            print(json.dumps({'completed_this_run':index,**result}),flush=True)
    count = len(list((WORK/'designs').glob('*.json')))
    write_json(WORK/'qa/run_progress.json',{'completed':count,'expected':540,
               'fingerprint':manifest['fingerprint'],'error_files':len(list((WORK/'errors').glob('*.json')))})
    print(f'completed {count}/540',flush=True)


if __name__ == '__main__':
    main()
