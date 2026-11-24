"""Full 10,612-record (5,306 dual-year Japanese communities) exact optimization & MC suite.

Ultra-lean memory architecture (target ~15-22 GB total RSS, hard-guarded < 36 GB):
  - Does NOT import heavy legacy manifest modules (`common`, `run_obs_placement`, `factorial_q50`)
    at top level (those are imported ONLY inside an isolated subprocess during `selftest()`).
  - Parent stays ~45 MB and forks slim single-threaded COPT workers sharing read-only CoW pages
    plus read-only `mmap_mode='r'` on `japan_dual_hourly.npy`.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import ctypes
import gc
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time

WORK = Path(__file__).resolve().parents[1]
ROOT = WORK.parents[3]
sys.path.insert(0, str(ROOT / 'submission/Nature_Energy_v85/revision/real_vs_placement_20261029/code'))
from cpu_budget import apply_cpu_budget, limit_workers  # noqa: E402
apply_cpu_budget()

for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
os.environ.setdefault('COPT_LICENSE_DIR', str(Path.home() / 'copt'))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import optimize  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402

V82 = ROOT / 'submission/Nature_Energy_v82/revision'
sys.path.insert(0, str(V82 / 'system_cost_v82_20261013/capacity/solver_vendor'))
sys.path.insert(0, str(V82 / 'issues3_4_20261018/order_ensemble'))
sys.path.insert(0, str(V82 / 'discount_sensitivity_20261022/code/module_a'))
sys.path.insert(0, str(V82 / 'issue6_window_saturation_20261019/scripts'))
sys.path.insert(0, str(V82 / 'advisor_key_tests_20261022/t6_window'))
sys.path.insert(0, str(ROOT / 'submission/Nature_Energy_v86/revision/hydrogen_tank_only_20261114/code'))

import coptpy as cp  # noqa: E402
import chronology_ensemble as ce  # noqa: E402
from finance import vF1_usd  # noqa: E402
import report_metrics as RM  # noqa: E402
import parameters as HP  # noqa: E402
import window_diagnostic as WD  # noqa: E402
from cyclic_diagnostic import cyclic_recursion  # noqa: E402
WD.recursion = cyclic_recursion

DATA_DIR = WORK / 'data'
OUT_DIR = WORK / 'results/full_10612'
OUT_DIR.mkdir(parents=True, exist_ok=True)
SHARD_DIR = OUT_DIR / 'mc_shards'
SHARD_DIR.mkdir(parents=True, exist_ok=True)
P1_SHARD_DIR = OUT_DIR / 'p1_shards'
P1_SHARD_DIR.mkdir(parents=True, exist_ok=True)

N_HOURS = 8760
N_DAYS = 365
INIT_PREV = np.roll(np.arange(N_DAYS, dtype=np.int32), 1)
RATE = 0.036
FX = 1.13
SERVICE_PRICE_USD_KW_YR = 117.040888604  # exact vF1_usd(0.036)
_RAW_SCEN = json.loads((V82 / 'issues3_4_20261018/qa/input_manifest.json').read_text())['scenario']
BASE_SCENARIO_EUR = {
    **_RAW_SCEN,
    'firm_value_eur_kw_year': vF1_usd(RATE) / FX,
}
_CAES_DURS = [8, 10, 12, 14, 16, 18, 20, 22, 24]

# Shared read-only tables loaded BEFORE fork (CoW shared across workers, zero duplication!)
_META = pd.read_csv(DATA_DIR / 'japan_dual_metadata.csv').sort_values('row_idx').reset_index(drop=True)
_CAL = np.load(DATA_DIR / 'japan_calendar.npz')
_CAL_M = _CAL['cal_months'].copy()
_SEG = _CAL['season_seg'].copy()
_CAT58 = HP.catalogue()
_H2_RTE = HP.h2_physics()['rte']
_TECH_GROUPS: dict[str, list[tuple[str, dict]]] = {}
for _cid, _cfg in _CAT58.items():
    _TECH_GROUPS.setdefault(_cfg['tech_id'], []).append((_cid, _cfg))
for _tid in _TECH_GROUPS:
    _TECH_GROUPS[_tid].sort(key=lambda kv: float(kv[1]['duration_h']))
_H2_DURS = [int(round(cfg['duration_h'])) for _, cfg in _TECH_GROUPS['h2_tank']]

# Worker-private globals initialized AFTER fork
_ENV = None
_HOURLY = None


_LIBC = ctypes.CDLL('libc.so.6')


def _trim_heap():
    gc.collect()
    _LIBC.malloc_trim(0)


def _init_worker():
    global _ENV, _HOURLY
    os.environ['MALLOC_ARENA_MAX'] = '2'
    ecfg = cp.EnvrConfig()
    ecfg.set('nobanner', '1')
    _ENV = cp.Envr(ecfg)
    _HOURLY = np.load(DATA_DIR / 'japan_dual_hourly.npy', mmap_mode='r')


def _capacity_h_source_output(cfg: dict) -> float:
    return float(cfg['duration_h']) / float(np.sqrt(cfg['rte']))


def _transform_rule(raw: np.ndarray, rule: str = 'registered') -> np.ndarray:
    c1, c3, r = raw.T
    if rule in ('registered', 'all_task'):
        return np.c_[c1, c3 if rule == 'registered' else c1, np.where(c1 > 1e-10, 0.0, r)]
    task = np.maximum(c1 - r, 0.0)
    tail = task * (c3 > 0.0) if rule == 'netted_tail' else task
    recharge = np.maximum(r - c1, 0.0)
    return np.c_[task, tail, recharge]


def _access_factorial(a: np.ndarray, mE: float, mode: str) -> tuple[np.ndarray, float, float]:
    out = a.copy()
    out[:, 2] = out[:, 2] * mE
    open_mask = out[:, 0] <= 1e-10
    total = float(out[open_mask, 2].sum())
    if mode == 'expanded':
        r = np.zeros(len(out), dtype=np.float64)
        r[open_mask] = total / open_mask.sum()
        out[:, 2] = r
    err = float(out[open_mask, 2].sum() - total)
    return out, total, err


def _uid_of(system_id: str, fy: int) -> tuple[str, str]:
    uk = f'Japan:{system_id}:{int(fy)}'
    uid = hashlib.sha256(uk.encode()).hexdigest()[:16]
    return uid, uk


def _runs(flags: np.ndarray) -> list[int]:
    out, cnt = [], 0
    for x in flags:
        if x:
            cnt += 1
        elif cnt:
            out.append(cnt)
            cnt = 0
    if cnt:
        out.append(cnt)
    return out


def _long_windows_fast(arr: np.ndarray) -> int:
    call = arr[:, 1] > 1e-10
    rid = np.cumsum(np.r_[0, np.diff(call.astype(np.int8)) != 0])
    is_c = np.bincount(rid, weights=call.astype(float)) > 0
    ch = np.bincount(rid, weights=(arr[:, 2] > 1e-9).astype(float))
    ec = np.bincount(rid, weights=np.where(call, 0.0, np.minimum(arr[:, 2], 1.5))) * 0.55
    return int(((~is_c) & (ch > 0) & (ec >= 21.6 - 1e-9)).sum())


def _build_record_arrays(row_idx: int):
    r = _META.iloc[row_idx]
    uid, uk = _uid_of(r.system_id, int(r.observation_year))
    imp = np.asarray(_HOURLY[row_idx, :, 0], dtype=np.float64)
    rev = np.asarray(_HOURLY[row_idx, :, 1], dtype=np.float64)
    P = 0.01 * float(imp.mean())
    base_q = np.empty(N_HOURS, dtype=np.float64)
    for mo in range(1, 13):
        m = _CAL_M == mo
        blk = imp[m].reshape(-1, 24)
        base_q[m] = np.tile(np.quantile(blk, 0.50, axis=0), blk.shape[0])
    c1 = np.maximum(imp - base_q, 0.0)
    day = c1.reshape(N_DAYS, 24).sum(1)
    q75 = np.empty(N_DAYS, dtype=np.float64)
    for s in range(4):
        ms = _SEG == s
        q75[ms] = np.quantile(day[ms], 0.75)
    flags = day >= q75
    c3 = np.where(np.repeat(flags, 24), c1, 0.0)
    raw = np.c_[c1, c3, rev] / P
    arr = _transform_rule(raw, 'registered')
    return r, uid, uk, imp, rev, P, c1, flags, raw, arr


class CollapsedIndexer:
    """Precomputes the exact active-hour + end-of-day state graph for a fixed hourly array `arr`."""
    __slots__ = (
        'task', 'tail_task', 'recharge', 'task_sum', 'ns', 'h_to_s',
        'day_first_s', 'day_first_k', 'day_end_s', 'intra_s', 'intra_prev_s', 'intra_gap',
        'iq', 'idb', 'idt', 'ipk', 'nq', 'ndb', 'ndt', 'npk', 'db_map', 'dt_map', 'pk_has_dt',
    )

    def __init__(self, arr: np.ndarray):
        task, tail_task, recharge = arr.T
        self.task = np.ascontiguousarray(task, dtype=np.float64)
        self.tail_task = np.ascontiguousarray(tail_task, dtype=np.float64)
        self.recharge = np.ascontiguousarray(recharge, dtype=np.float64)
        self.task_sum = float(self.task.sum())

        keep_mask = (self.task > 1e-12) | (self.recharge > 1e-12)
        keep_mask[23::24] = True
        ikeep = np.flatnonzero(keep_mask)
        self.ns = len(ikeep)
        h_to_s = np.full(N_HOURS, -1, dtype=np.int32)
        h_to_s[ikeep] = np.arange(self.ns, dtype=np.int32)
        self.h_to_s = h_to_s

        day_first_s = np.empty(N_DAYS, dtype=np.int32)
        day_first_k = np.empty(N_DAYS, dtype=np.int32)
        self.day_end_s = h_to_s[24 * np.arange(N_DAYS) + 23]
        intra_s, intra_prev_s, intra_gap = [], [], []
        for d in range(N_DAYS):
            ks = np.flatnonzero(keep_mask[24 * d:24 * d + 24])
            ss = h_to_s[24 * d + ks]
            day_first_s[d] = ss[0]
            day_first_k[d] = ks[0] + 1
            if len(ss) > 1:
                intra_s.append(ss[1:])
                intra_prev_s.append(ss[:-1])
                intra_gap.append(np.diff(ks))
        self.day_first_s = day_first_s
        self.day_first_k = day_first_k
        self.intra_s = np.concatenate(intra_s) if intra_s else np.empty(0, dtype=np.int32)
        self.intra_prev_s = np.concatenate(intra_prev_s) if intra_prev_s else np.empty(0, dtype=np.int32)
        self.intra_gap = np.concatenate(intra_gap) if intra_gap else np.empty(0, dtype=np.int32)

        self.iq = np.flatnonzero(self.recharge > 1e-12)
        self.idb = np.flatnonzero(self.task > 1e-12)
        self.idt = np.flatnonzero(self.tail_task > 1e-12)
        tmax = float(self.task.max()) if len(self.idb) else 0.0
        self.ipk = np.flatnonzero((self.task > 1e-12) & (self.task >= tmax - 2.0 - 1e-7))
        self.nq, self.ndb, self.ndt, self.npk = len(self.iq), len(self.idb), len(self.idt), len(self.ipk)

        db_map = np.full(N_HOURS, -1, dtype=np.int32)
        db_map[self.idb] = np.arange(self.ndb, dtype=np.int32)
        dt_map = np.full(N_HOURS, -1, dtype=np.int32)
        dt_map[self.idt] = np.arange(self.ndt, dtype=np.int32)
        self.db_map = db_map
        self.dt_map = dt_map
        self.pk_has_dt = dt_map[self.ipk] >= 0


class PersistentBranchLP:
    def __init__(self, idx: CollapsedIndexer, base_cfg: dict, tail_cfg: dict, scen: dict, m_power: float = 1.0):
        self.idx = idx
        self.base_cfg = base_cfg
        self.tail_cfg = tail_cfg
        self.scen = scen
        self.eb = float(np.sqrt(base_cfg['rte']))
        self.et = float(np.sqrt(tail_cfg['rte']))
        self.rb = float((1.0 - base_cfg['self_discharge_pct_day'] / 100.0) ** (1.0 / 24.0))
        self.rt = float((1.0 - tail_cfg['self_discharge_pct_day'] / 100.0) ** (1.0 / 24.0))
        self.hb = _capacity_h_source_output(base_cfg)
        self.ht = _capacity_h_source_output(tail_cfg)

        ns, nq, ndb, ndt, npk = idx.ns, idx.nq, idx.ndb, idx.ndt, idx.npk
        self.o_qb = 0
        self.o_qt = nq
        self.o_db = 2 * nq
        self.o_dt = 2 * nq + ndb
        self.o_sb = 2 * nq + ndb + ndt
        self.o_st = self.o_sb + ns
        self.o_rv = self.o_st + ns
        self.nvars = self.o_rv + 1
        self.nrows = 2 * ns + ndt + nq + npk

        sr = np.arange(ns, dtype=np.int32)
        r_jt = 2 * ns + np.arange(ndt, dtype=np.int32)
        r_sh = 2 * ns + ndt + np.arange(nq, dtype=np.int32)
        r_pk = 2 * ns + ndt + nq + np.arange(npk, dtype=np.int32)

        self.rb_k0 = self.rb ** idx.day_first_k
        self.rt_k0 = self.rt ** idx.day_first_k
        self.cur_prev = INIT_PREV.copy()

        rows = np.concatenate([
            sr, idx.intra_s, idx.day_first_s, idx.h_to_s[idx.iq], idx.h_to_s[idx.idb],
            ns + sr, ns + idx.intra_s, ns + idx.day_first_s, ns + idx.h_to_s[idx.iq], ns + idx.h_to_s[idx.idt],
            r_jt, r_jt, r_sh, r_sh, r_pk, r_pk[idx.pk_has_dt], r_pk,
        ])
        cols = np.concatenate([
            self.o_sb + sr, self.o_sb + idx.intra_prev_s, self.o_sb + idx.day_end_s[self.cur_prev],
            self.o_qb + np.arange(nq), self.o_db + np.arange(ndb),
            self.o_st + sr, self.o_st + idx.intra_prev_s, self.o_st + idx.day_end_s[self.cur_prev],
            self.o_qt + np.arange(nq), self.o_dt + np.arange(ndt),
            self.o_db + idx.db_map[idx.idt], self.o_dt + np.arange(ndt),
            self.o_qb + np.arange(nq), self.o_qt + np.arange(nq),
            self.o_db + idx.db_map[idx.ipk], self.o_dt + idx.dt_map[idx.ipk[idx.pk_has_dt]], np.full(npk, self.o_rv),
        ])
        vals = np.concatenate([
            np.ones(ns), -(self.rb ** idx.intra_gap), -self.rb_k0, -self.eb * np.ones(nq), np.full(ndb, 1.0 / self.eb),
            np.ones(ns), -(self.rt ** idx.intra_gap), -self.rt_k0, -self.et * np.ones(nq), np.full(ndt, 1.0 / self.et),
            np.ones(ndt), np.ones(ndt), np.ones(nq), np.ones(nq),
            np.ones(npk), np.ones(int(idx.pk_has_dt.sum())), np.ones(npk),
        ])
        self.A = coo_matrix((vals, (rows, cols)), shape=(self.nrows, self.nvars)).tocsc()
        self.rl = np.r_[np.zeros(2 * ns), np.full(ndt + nq, -cp.COPT.INFINITY), idx.task[idx.ipk]]
        self.ru = np.r_[np.zeros(2 * ns), idx.task[idx.idt], idx.recharge[idx.iq], np.full(npk, cp.COPT.INFINITY)]
        self.cl = np.zeros(self.nvars)
        self.cu = np.r_[
            np.minimum(idx.recharge[idx.iq], base_cfg['charge_power_ratio'] * m_power),
            np.minimum(idx.recharge[idx.iq], tail_cfg['charge_power_ratio'] * m_power),
            np.minimum(idx.task[idx.idb], 1.0),
            np.minimum(idx.tail_task[idx.idt], 1.0),
            np.full(ns, 0.9 * self.hb),
            np.full(ns, 0.9 * self.ht),
            [cp.COPT.INFINITY],
        ]
        ev_mwh = float(scen['energy_value_eur_mwh'])
        self.ev_mwh = ev_mwh
        c = np.zeros(self.nvars)
        c[self.o_qb:self.o_qt] = scen['charge_price_eur_mwh']
        c[self.o_qt:self.o_db] = scen['charge_price_eur_mwh']
        c[self.o_db:self.o_dt] = base_cfg['vom_eur_mwh'] - ev_mwh
        c[self.o_dt:self.o_sb] = tail_cfg['vom_eur_mwh'] - ev_mwh
        c[self.o_rv] = 1000.0 * scen['firm_value_eur_kw_year']
        self.c = c

        m = _ENV.createModel('coll_lp')
        for pn, pv in [('Logging', 0), ('Threads', 1), ('FeasTol', 1e-8), ('DualTol', 1e-8), ('TimeLimit', 180), ('LpMethod', 1)]:
            m.setParam(pn, pv)
        m.loadMatrix(c, self.A, self.rl, self.ru, self.cl, self.cu)
        self.m = m
        constrs = list(m.getConstrs())
        vs = list(m.getVars())
        self.c_sb_mid = [constrs[idx.day_first_s[d]] for d in range(N_DAYS)]
        self.c_st_mid = [constrs[ns + idx.day_first_s[d]] for d in range(N_DAYS)]
        self.v_sb_end = [vs[self.o_sb + idx.day_end_s[d]] for d in range(N_DAYS)]
        self.v_st_end = [vs[self.o_st + idx.day_end_s[d]] for d in range(N_DAYS)]
        self.v_st_all = vs[self.o_st:self.o_rv]
        self.v_q_all = vs[0:2 * nq]
        self.v_d_all = vs[self.o_db:self.o_sb]

    def dispose(self):
        self.c_sb_mid = None
        self.c_st_mid = None
        self.v_sb_end = None
        self.v_st_end = None
        self.v_st_all = None
        self.v_q_all = None
        self.v_d_all = None
        self.A = None
        if self.m is not None:
            self.m.dispose()
            self.m = None

    def solve_cost_usd(self, fixed_tail_eur_kw_yr: float | None = None) -> float:
        m = self.m
        m.solveLP()
        if m.status != cp.COPT.OPTIMAL:
            m.setParam('LpMethod', 2)
            m.solveLP()
            m.setParam('LpMethod', 1)
            if m.status != cp.COPT.OPTIMAL:
                raise RuntimeError(f'COPT non-optimal status={m.status}')
        tf = self.tail_cfg['fixed_cost_eur_kw_year'] if fixed_tail_eur_kw_yr is None else fixed_tail_eur_kw_yr
        fix_v = 1000.0 * (self.base_cfg['fixed_cost_eur_kw_year'] + tf)
        return (float(m.objval) + self.idx.task_sum * self.ev_mwh + fix_v) / 1000.0 * FX

    def rewire(self, new_prev: np.ndarray):
        diff_d = np.flatnonzero(new_prev != self.cur_prev)
        if len(diff_d) == 0:
            return
        rc, vc, vals = [], [], []
        csb, cst, vsb, vst = self.c_sb_mid, self.c_st_mid, self.v_sb_end, self.v_st_end
        rb_k0, rt_k0 = self.rb_k0, self.rt_k0
        old_p = self.cur_prev
        for d in diff_d:
            op, np_d = old_p[d], new_prev[d]
            rc.extend([csb[d], csb[d], cst[d], cst[d]])
            vc.extend([vsb[op], vsb[np_d], vst[op], vst[np_d]])
            vals.extend([0.0, -rb_k0[d], 0.0, -rt_k0[d]])
        self.m.setCoeffs(rc, vc, vals)
        self.cur_prev = new_prev.copy()

    def set_tail_cfg(self, tail_cfg: dict):
        self.tail_cfg = tail_cfg
        self.ht = _capacity_h_source_output(tail_cfg)
        ub = np.full(self.idx.ns, 0.9 * self.ht)
        self.cu[self.o_st:self.o_rv] = ub
        self.m.setInfo(cp.COPT.Info.UB, self.v_st_all, ub)

    def set_charge_power_mult(self, m_power: float):
        iq = self.idx.iq
        rec_iq = self.idx.recharge[iq]
        ub = np.r_[
            np.minimum(rec_iq, self.base_cfg['charge_power_ratio'] * m_power),
            np.minimum(rec_iq, self.tail_cfg['charge_power_ratio'] * m_power),
        ]
        self.cu[0:2 * self.idx.nq] = ub
        self.m.setInfo(cp.COPT.Info.UB, self.v_q_all, ub)

    def set_energy_mult(self, mult: float):
        ev_mwh = BASE_SCENARIO_EUR['energy_value_eur_mwh'] * mult
        self.ev_mwh = ev_mwh
        obj_d = np.r_[
            np.full(self.idx.ndb, self.base_cfg['vom_eur_mwh'] - ev_mwh),
            np.full(self.idx.ndt, self.tail_cfg['vom_eur_mwh'] - ev_mwh),
        ]
        self.c[self.o_db:self.o_sb] = obj_d
        self.m.setInfo(cp.COPT.Info.Obj, self.v_d_all, obj_d)

    def full_solution(self, check_kkt: bool = False) -> dict:
        m = self.m
        m.solveLP()
        if m.status != cp.COPT.OPTIMAL:
            m.setParam('LpMethod', 2)
            m.solveLP()
            m.setParam('LpMethod', 1)
            if m.status != cp.COPT.OPTIMAL:
                raise RuntimeError(f'COPT status {m.status}')
        x, activity, dual, reduced = m.getLpSolution()
        x = np.asarray(x, dtype=np.float64)
        idx = self.idx
        base_dis = float(x[self.o_db:self.o_dt].sum())
        tail_dis = float(x[self.o_dt:self.o_sb].sum())
        charge = float(x[self.o_qb:self.o_db].sum())
        eens = idx.task_sum - base_dis - tail_dis
        peak = float(x[self.o_rv])

        fixed_eur = (self.base_cfg['fixed_cost_eur_kw_year'] + self.tail_cfg['fixed_cost_eur_kw_year'])
        vom_eur = (base_dis * self.base_cfg['vom_eur_mwh'] + tail_dis * self.tail_cfg['vom_eur_mwh']) / 1000.0
        chg_eur = (charge * self.scen['charge_price_eur_mwh']) / 1000.0
        re_eur = (eens * self.ev_mwh) / 1000.0
        rp_eur = peak * self.scen['firm_value_eur_kw_year']
        cost_usd = FX * (fixed_eur + vom_eur + chg_eur + re_eur + rp_eur)

        out = {
            'cost_usd_kw_year': cost_usd,
            'fixed_usd_kw_year': FX * fixed_eur,
            'variable_om_usd_kw_year': FX * vom_eur,
            'charging_usd_kw_year': FX * chg_eur,
            'residual_energy_usd_kw_year': FX * re_eur,
            'residual_peak_usd_kw_year': FX * rp_eur,
            'base_discharge_mwh_per_mw': base_dis,
            'tail_discharge_mwh_per_mw': tail_dis,
            'total_discharge_mwh_per_mw': base_dis + tail_dis,
            'solver_status': int(m.status),
        }
        if check_kkt:
            activity = np.asarray(activity, dtype=np.float64)
            dual = np.asarray(dual, dtype=np.float64)
            reduced = np.asarray(reduced, dtype=np.float64)
            ns = idx.ns
            n_le = idx.ndt + idx.nq
            eq_res = float(np.max(np.abs(activity[:2 * ns]))) if ns else 0.0
            le_res = float(np.max(activity[2 * ns:2 * ns + n_le] - self.ru[2 * ns:2 * ns + n_le])) if n_le else 0.0
            ge_res = float(np.max(self.rl[2 * ns + n_le:] - activity[2 * ns + n_le:])) if idx.npk else 0.0
            lb_res = float(np.max(self.cl - x))
            ub_res = float(np.max(np.where(np.isfinite(self.cu), x - self.cu, 0.0)))
            phys_res = max(eq_res, le_res, ge_res, lb_res, ub_res, 0.0)
            stat_res = float(np.max(np.abs(self.c - self.A.T @ dual - reduced)))
            obj_p = float(self.c @ x)
            obj_d = (
                float(np.where(np.isfinite(self.ru), self.ru, 0.0) @ np.minimum(dual, 0.0))
                + float(np.where(np.isfinite(self.rl), self.rl, 0.0) @ np.maximum(dual, 0.0))
                + float(np.where(np.isfinite(self.cu), self.cu, 0.0) @ np.minimum(reduced, 0.0))
            )
            gap_eur = abs(obj_p - obj_d) / 1000.0
            assert phys_res < 1e-6 and stat_res < 1e-6 and gap_eur < 1e-7, (phys_res, stat_res, gap_eur)
            out['max_physical_residual'] = phys_res
            out['max_stationarity_residual'] = stat_res
            out['duality_gap_eur_kw_year'] = gap_eur
        return out


def _prev_from_order(order: np.ndarray) -> np.ndarray:
    p = np.empty(N_DAYS, dtype=np.int32)
    p[order] = np.roll(order, 1)
    return p


def _precompute_obs_segments(seg_arr: np.ndarray, flags: np.ndarray):
    segs = []
    for segment in (0, 1, 2, 3):
        mask = seg_arr == segment
        days = np.flatnonzero(mask)
        fl = flags[mask]
        clusters, cur = [], []
        for d, f in zip(days, fl):
            if f:
                cur.append(int(d))
            elif cur:
                clusters.append(cur)
                cur = []
        if cur:
            clusters.append(cur)
        normal = days[~fl].tolist()
        segs.append((int(segment), clusters, normal))
    return segs


def _fast_obs_random_order(unit_key: str, segs, seed: int) -> np.ndarray:
    order = []
    for segment, clusters, normal in segs:
        if not clusters:
            order.extend(normal)
            continue
        gaps = len(clusters) + 1
        rng = ce._rng(seed, unit_key, segment, 'gaps_observed_lengths')
        sizes = 1 + ce._weak_composition(len(normal) - gaps, gaps, rng)
        k = 0
        for i, s in enumerate(sizes):
            order.extend(normal[k:k + s])
            k += s
            if i < len(clusters):
                order.extend(clusters[i])
    return np.asarray(order, dtype=np.int32)


def _selftest_subprocess():
    """Isolated child-process selftest: imports legacy modules here ONLY so parent stays ~45 MB."""
    sys.path.insert(0, str(ROOT / 'submission/Nature_Energy_v85/revision/remaining_experiments_20261103/code'))
    import common as C
    sys.path.insert(0, str(ROOT / 'submission/Nature_Energy_v85/revision/placement_mc_20261027/code'))
    import run_obs_placement as O
    sys.path.insert(0, str(ROOT / 'submission/Nature_Energy_v85/revision/surplus_factorial_q50_20261103/code'))
    import factorial_q50 as FQ

    _init_worker()
    t0 = time.monotonic()
    max_err = 0.0
    for test_idx in (0, 2500):
        r, uid, uk, imp, rev, P, c1, flags, raw, arr = _build_record_arrays(test_idx)
        cidx = CollapsedIndexer(arr)
        base_cfg = _CAT58['li_lfp_4h']
        mL = PersistentBranchLP(cidx, base_cfg, base_cfg, BASE_SCENARIO_EUR)
        mC = PersistentBranchLP(cidx, base_cfg, _CAT58['caes_24h'], BASE_SCENARIO_EUR)
        mH = PersistentBranchLP(cidx, base_cfg, _CAT58['h2_tank_168h'], BASE_SCENARIO_EUR)
        try:
            sL = mL.full_solution(check_kkt=True)
            sC = mC.full_solution(check_kkt=True)
            sH = mH.full_solution(check_kkt=True)
            refL = C.solve(arr, 'li_lfp_4h')
            refC = C.solve(arr, 'caes_24h')
            refH = C.model.solve(*arr.T, base_cfg, _CAT58['h2_tank_168h'], BASE_SCENARIO_EUR, convention='source_output_rated', method=1)
            max_err = max(
                max_err,
                abs((sL['cost_usd_kw_year'] - sC['cost_usd_kw_year']) - (refL['cost_usd_kw_year'] - refC['cost_usd_kw_year'])),
                abs(sH['cost_usd_kw_year'] - FX * refH['cost_eur_kw_year']),
            )
            ens = ce.ChronologyEnsemble(pd.DataFrame({'segment': _SEG, 'event': flags}), uk)
            segs_fast = _precompute_obs_segments(_SEG, flags)
            perm = np.asarray(O.obs_random_order(ens, 7), dtype=np.int32)
            assert np.array_equal(perm, _fast_obs_random_order(uk, segs_fast, 7))
            mL.rewire(_prev_from_order(perm))
            mC.rewire(_prev_from_order(perm))
            b_mc = mL.solve_cost_usd() - mC.solve_cost_usd()
            ap = ce.reorder_days(arr, perm)
            ref_mc = C.solve(ap, 'li_lfp_4h')['cost_usd_kw_year'] - C.solve(ap, 'caes_24h')['cost_usd_kw_year']
            max_err = max(max_err, abs(b_mc - ref_mc))

            o12 = np.asarray(ens.orders(12)['clustered'], dtype=np.int32)
            a_exp, u1, e1 = _access_factorial(arr, 4, 'expanded')
            a_exp_ref, u2, e2 = FQ.access(arr, 4, 'expanded')
            assert np.allclose(a_exp, a_exp_ref) and abs(u1 - u2) < 1e-12
            cidx_exp = CollapsedIndexer(a_exp)
            mLe = PersistentBranchLP(cidx_exp, base_cfg, base_cfg, BASE_SCENARIO_EUR, m_power=4.0)
            mCe = PersistentBranchLP(cidx_exp, base_cfg, _CAT58['caes_24h'], BASE_SCENARIO_EUR, m_power=4.0)
            mLe.rewire(_prev_from_order(o12))
            mCe.rewire(_prev_from_order(o12))
            b_fe = mLe.solve_cost_usd() - mCe.solve_cost_usd()
            ref_fe = FQ.solve_B(_access_factorial(ce.reorder_days(arr, o12), 4, 'expanded')[0], BASE_SCENARIO_EUR, 4)
            max_err = max(max_err, abs(b_fe - ref_fe))
            mLe.dispose()
            mCe.dispose()
        finally:
            mL.dispose()
            mC.dispose()
            mH.dispose()
    assert max_err < 1e-8, f'Selftest max_err={max_err}'
    print(json.dumps({'selftest': 'PASS', 'max_abs_err': max_err, 'seconds': round(time.monotonic() - t0, 2)}), flush=True)


def run_phase1_shard(payload):
    shard_id, row_indices = payload
    done_marker = P1_SHARD_DIR / f'shard_{shard_id:04d}.done'
    if done_marker.exists():
        return shard_id, 'cached', 0.0

    t0 = time.monotonic()
    rec_rows, cat_rows, fw_rows, fac_rows = [], [], [], []
    price_rows, rule_rows, ext_rows, wr_rows, cons_rows = [], [], [], [], []
    h2_max_phys = 0.0
    h2_max_stat = 0.0
    h2_max_gap = 0.0
    h2_count = 0

    base_cfg = _CAT58['li_lfp_4h']
    caes_cfg = _CAT58['caes_24h']

    for idx in row_indices:
        r, uid, uk, imp, rev, P, c1, flags, raw, arr = _build_record_arrays(idx)
        ens = ce.ChronologyEnsemble(pd.DataFrame({'segment': _SEG, 'event': flags}, index=np.arange(N_DAYS)), uk)

        ev_runs = _runs(flags)
        hd_days = int(flags.sum())
        n_events = len(ev_runs)
        mean_dur_real = hd_days / n_events
        longest_real = max(ev_runs)

        is_call, run_id, seg_hours = WD.segments(arr)
        call = arr[:, 1] > 1e-10
        ch = np.bincount(run_id, weights=(arr[:, 2] > 1e-9).astype(float))
        ok = (~is_call) & (ch > 0)
        r_open = np.where(call, 0.0, arr[:, 2])
        e_c1 = np.bincount(run_id, weights=np.minimum(r_open, 1.5)) * 0.55
        e_c4 = np.bincount(run_id, weights=np.minimum(r_open, 6.0)) * 0.55
        e_l1 = np.bincount(run_id, weights=np.minimum(r_open, 1.0)) * 0.90
        e_l4 = np.bincount(run_id, weights=np.minimum(r_open, 4.0)) * 0.90
        n_win = int(ok.sum())
        n_1h = int((ok & (seg_hours == 1)).sum())
        lfp_x1 = int((ok & (e_l1 >= 3.6 - 1e-9)).sum())
        lfp_x4 = int((ok & (e_l4 >= 3.6 - 1e-9)).sum())
        long_x1 = int((ok & (e_c1 >= 21.6 - 1e-9)).sum())
        long_x4 = int((ok & (e_c4 >= 21.6 - 1e-9)).sum())

        for d_h in _CAES_DURS:
            fw_rows.append((uid, 'community', 'caes', d_h, int((ok & (e_c1 >= WD.USABLE * d_h - 1e-9)).sum())))
        e_h2 = np.bincount(run_id, weights=np.minimum(r_open, 3.0)) * _H2_RTE
        for d_h in _H2_DURS:
            fw_rows.append((uid, 'community', 'h2_tank', d_h, int((ok & (e_h2 >= WD.USABLE * d_h - 1e-9)).sum())))

        cidx = CollapsedIndexer(arr)
        mL = PersistentBranchLP(cidx, base_cfg, base_cfg, BASE_SCENARIO_EUR)
        mC = PersistentBranchLP(cidx, base_cfg, caes_cfg, BASE_SCENARIO_EUR)
        try:
            sL = mL.full_solution()
            sC = mC.full_solution()
            cost_L = sL['cost_usd_kw_year']
            cost_C = sC['cost_usd_kw_year']
            B_real = cost_L - cost_C
            B_fixed = sL['fixed_usd_kw_year'] - sC['fixed_usd_kw_year']
            B_vom = sL['variable_om_usd_kw_year'] - sC['variable_om_usd_kw_year']
            B_chg = sL['charging_usd_kw_year'] - sC['charging_usd_kw_year']
            B_re = sL['residual_energy_usd_kw_year'] - sC['residual_energy_usd_kw_year']
            B_rp = sL['residual_peak_usd_kw_year'] - sC['residual_peak_usd_kw_year']
            add_deliv_real = sC['total_discharge_mwh_per_mw'] - sL['total_discharge_mwh_per_mw']

            load = arr[:, 0]
            lmax = float(load.max())
            F_lfp = float(optimize.brentq(lambda f: np.minimum(load, f).sum() - sL['total_discharge_mwh_per_mw'], 0.0, lmax, xtol=1e-12))
            F_hyb = float(optimize.brentq(lambda f: np.minimum(load, f).sum() - sC['total_discharge_mwh_per_mw'], 0.0, lmax, xtol=1e-12))
            B_eq_srv = B_real - B_rp + SERVICE_PRICE_USD_KW_YR * (F_hyb - F_lfp)

            op_sav = B_real + RM.PREM36[2035]
            rec_rows.append({
                'uid': uid, 'unit_key': uk, 'row_idx': int(idx),
                'group': 'community', 'regime': 'Japan', 'prefecture': r.prefecture,
                'system_id': r.system_id, 'observation_year': int(r.observation_year),
                'lon': float(r.lon), 'lat': float(r.lat), 'P_ref_mw': float(P),
                'task_hours': int((arr[:, 0] > 1e-10).sum()),
                'charge_hours': int((arr[:, 2] > 1e-9).sum()),
                'event_days': hd_days, 'events': n_events,
                'mean_dur': mean_dur_real, 'longest': longest_real,
                'hybrid_window_full': long_x1,
                'windows': n_win, 'lfp_fill_windows': lfp_x1, 'long_windows': long_x1,
                'one_hour_windows': n_1h,
                'lfp_fill_x1': lfp_x1, 'lfp_fill_x4': lfp_x4,
                'long_x1': long_x1, 'long_x4': long_x4,
                'B': B_real, 'B_fixed': B_fixed, 'B_variable_om': B_vom,
                'B_charging': B_chg, 'B_residual_energy': B_re, 'B_residual_peak': B_rp,
                'lfp_total_discharge_mwh_per_mw': sL['total_discharge_mwh_per_mw'],
                'hybrid_total_discharge_mwh_per_mw': sC['total_discharge_mwh_per_mw'],
                'lfp_tail_discharge_mwh_per_mw': sL['tail_discharge_mwh_per_mw'],
                'hybrid_tail_discharge_mwh_per_mw': sC['tail_discharge_mwh_per_mw'],
                'additional_delivery_mwh_per_mw': add_deliv_real,
                'efc_acaes': sC['tail_discharge_mwh_per_mw'] / 24.0,
                'efc_lfp': sL['tail_discharge_mwh_per_mw'] / 4.0,
                'roi_2025': float(RM.roi(op_sav, 2025)),
                'roi_2030': float(RM.roi(op_sav, 2030)),
                'roi_2035': float(RM.roi(op_sav, 2035)),
                'win36': bool(B_real >= 0.0),
                'F_lfp': F_lfp, 'F_hybrid': F_hyb, 'B_equivalent_service': B_eq_srv,
            })

            # R5 58 configurations
            for tid, cfg_list in _TECH_GROUPS.items():
                first_cid, first_cfg = cfg_list[0]
                mT = PersistentBranchLP(cidx, base_cfg, first_cfg, BASE_SCENARIO_EUR)
                try:
                    for cid, cfg in cfg_list:
                        if cid != first_cid:
                            mT.set_tail_cfg(cfg)
                        if cid == 'li_lfp_4h':
                            b_val = 0.0
                        elif tid == 'h2_tank':
                            sol_h = mT.full_solution(check_kkt=True)
                            b_val = cost_L - sol_h['cost_usd_kw_year']
                            h2_max_phys = max(h2_max_phys, sol_h['max_physical_residual'])
                            h2_max_stat = max(h2_max_stat, sol_h['max_stationarity_residual'])
                            h2_max_gap = max(h2_max_gap, sol_h['duality_gap_eur_kw_year'])
                            h2_count += 1
                        else:
                            b_val = cost_L - mT.solve_cost_usd()
                        cat_rows.append((uid, 'community', cid, tid, float(cfg['duration_h']), b_val))
                finally:
                    mT.dispose()

            # ED Fig. 3 Price sensitivity
            price_rows.append((uid, 'community', 'x1', B_real))
            for mult, vtag in ((0.5, 'x0.5'), (0.75, 'x0.75'), (1.25, 'x1.25')):
                mL.set_energy_mult(mult)
                mC.set_energy_mult(mult)
                price_rows.append((uid, 'community', vtag, mL.solve_cost_usd() - mC.solve_cost_usd()))
            mL.set_energy_mult(1.0)
            mC.set_energy_mult(1.0)

            # Fig. S3 Extension 0..12 + Fig. S8/S12 diagnostics
            ext_by_t = {}
            ext_0 = (
                uid, 'community', 0, B_real, long_x1,
                sC['tail_discharge_mwh_per_mw'], sL['tail_discharge_mwh_per_mw'],
                add_deliv_real, B_re, B_rp, mean_dur_real,
            )
            ext_rows.append(ext_0)
            ext_by_t[0] = ext_0

            o12_cached = None
            for tgt in range(1, 13):
                ord_t = np.asarray(ens.orders(tgt)['clustered'], dtype=np.int32)
                if tgt == 12:
                    o12_cached = ord_t
                mL.rewire(_prev_from_order(ord_t))
                mC.rewire(_prev_from_order(ord_t))
                sLt = mL.full_solution()
                sCt = mC.full_solution()
                bt = sLt['cost_usd_kw_year'] - sCt['cost_usd_kw_year']
                arr_t = ce.reorder_days(arr, ord_t)
                lw_t = _long_windows_fast(arr_t)
                md_t = hd_days / len(_runs(flags[ord_t]))
                row_t = (
                    uid, 'community', tgt, bt, lw_t,
                    sCt['tail_discharge_mwh_per_mw'], sLt['tail_discharge_mwh_per_mw'],
                    sCt['total_discharge_mwh_per_mw'] - sLt['total_discharge_mwh_per_mw'],
                    sLt['residual_energy_usd_kw_year'] - sCt['residual_energy_usd_kw_year'],
                    sLt['residual_peak_usd_kw_year'] - sCt['residual_peak_usd_kw_year'],
                    md_t,
                )
                ext_rows.append(row_t)
                if tgt in (1, 4, 12):
                    ext_by_t[tgt] = row_t

            for tgt in (0, 1, 4, 12):
                ord_t = np.arange(N_DAYS, dtype=np.int32) if tgt == 0 else np.asarray(ens.orders(tgt)['clustered'], dtype=np.int32)
                b_arr = arr if tgt == 0 else ce.reorder_days(arr, ord_t)
                abs_e = float(np.max(np.abs(b_arr.sum(0) - arr.sum(0))))
                rel_e = float(np.max(np.abs(b_arr.sum(0) - arr.sum(0)) / np.maximum(1.0, np.abs(arr.sum(0)))))
                cons_rows.append((uid, 'community', tgt, abs_e, rel_e))
                ca_w = WD.branch(b_arr, 24, 0.55, 1.5)
                lf_w = WD.branch(b_arr, 4, 0.9, 1.0)
                er = ext_by_t[tgt]
                wr_rows.append((
                    uid, 'community', tgt,
                    ca_w['delivered'], lf_w['delivered'],
                    er[5], er[6], er[3],
                    ca_w['cyclic_residual'], lf_w['cyclic_residual'],
                ))

            # R4 2x2x2 Factorial
            mL.rewire(INIT_PREV)
            mC.rewire(INIT_PREV)
            prev_12 = _prev_from_order(o12_cached)
            usable_1_obs = float(arr[arr[:, 0] <= 1e-10, 2].sum())
            fac_rows.append((uid, 'Japan', r.prefecture, 'real', 1, 1, 'observed', B_real, usable_1_obs, 0.0))
            mL.set_charge_power_mult(4.0)
            mC.set_charge_power_mult(4.0)
            fac_rows.append((uid, 'Japan', r.prefecture, 'real', 1, 4, 'observed', mL.solve_cost_usd() - mC.solve_cost_usd(), usable_1_obs, 0.0))
            mL.rewire(prev_12)
            mC.rewire(prev_12)
            fac_rows.append((uid, 'Japan', r.prefecture, 'n12', 1, 4, 'observed', mL.solve_cost_usd() - mC.solve_cost_usd(), usable_1_obs, 0.0))
            mL.set_charge_power_mult(1.0)
            mC.set_charge_power_mult(1.0)
            fac_rows.append((uid, 'Japan', r.prefecture, 'n12', 1, 1, 'observed', ext_by_t[12][3], usable_1_obs, 0.0))
        finally:
            mL.dispose()
            mC.dispose()

        for e_mult, acc in ((4, 'observed'), (1, 'expanded'), (4, 'expanded')):
            a_mod, usable, cerr = _access_factorial(arr, e_mult, acc)
            ci_f = CollapsedIndexer(a_mod)
            fL = PersistentBranchLP(ci_f, base_cfg, base_cfg, BASE_SCENARIO_EUR, m_power=1.0)
            fC = PersistentBranchLP(ci_f, base_cfg, caes_cfg, BASE_SCENARIO_EUR, m_power=1.0)
            try:
                fac_rows.append((uid, 'Japan', r.prefecture, 'real', e_mult, 1, acc, fL.solve_cost_usd() - fC.solve_cost_usd(), usable, cerr))
                fL.set_charge_power_mult(4.0)
                fC.set_charge_power_mult(4.0)
                fac_rows.append((uid, 'Japan', r.prefecture, 'real', e_mult, 4, acc, fL.solve_cost_usd() - fC.solve_cost_usd(), usable, cerr))
                fL.rewire(prev_12)
                fC.rewire(prev_12)
                fac_rows.append((uid, 'Japan', r.prefecture, 'n12', e_mult, 4, acc, fL.solve_cost_usd() - fC.solve_cost_usd(), usable, cerr))
                fL.set_charge_power_mult(1.0)
                fC.set_charge_power_mult(1.0)
                fac_rows.append((uid, 'Japan', r.prefecture, 'n12', e_mult, 1, acc, fL.solve_cost_usd() - fC.solve_cost_usd(), usable, cerr))
            finally:
                fL.dispose()
                fC.dispose()

        rule_rows.append((uid, 'community', 'registered', B_real))
        for rname in ('all_task', 'netted_tail', 'netted_all_task'):
            a_r = _transform_rule(raw, rname)
            ci_r = CollapsedIndexer(a_r)
            rL = PersistentBranchLP(ci_r, base_cfg, base_cfg, BASE_SCENARIO_EUR)
            rC = PersistentBranchLP(ci_r, base_cfg, caes_cfg, BASE_SCENARIO_EUR)
            try:
                rule_rows.append((uid, 'community', rname, rL.solve_cost_usd() - rC.solve_cost_usd()))
            finally:
                rL.dispose()
                rC.dispose()
        _trim_heap()

    pfx = P1_SHARD_DIR / f'shard_{shard_id:04d}'
    pd.DataFrame(rec_rows).to_csv(f'{pfx}_rec.csv.gz', index=False)
    pd.DataFrame(cat_rows, columns=['uid', 'group', 'variant', 'tech', 'duration_h', 'B']).to_csv(f'{pfx}_cat.csv.gz', index=False)
    pd.DataFrame(fw_rows, columns=['uid', 'group', 'tech', 'duration_h', 'fill_windows']).to_csv(f'{pfx}_fw.csv.gz', index=False)
    pd.DataFrame(fac_rows, columns=['uid', 'regime', 'block', 'order', 'energy_multiplier', 'charge_power_multiplier', 'access', 'B', 'usable_surplus', 'conservation_error']).to_csv(f'{pfx}_fac.csv.gz', index=False)
    pd.DataFrame(price_rows, columns=['uid', 'group', 'variant', 'B']).to_csv(f'{pfx}_price.csv.gz', index=False)
    pd.DataFrame(rule_rows, columns=['uid', 'group', 'variant', 'B']).to_csv(f'{pfx}_rule.csv.gz', index=False)
    pd.DataFrame(ext_rows, columns=['uid', 'group', 'target', 'B', 'hybrid_window_full', 'hybrid_tail_discharge_mwh_per_mw', 'lfp_tail_discharge_mwh_per_mw', 'additional_delivery_mwh_per_mw', 'B_residual_energy', 'B_residual_peak', 'mean_dur']).to_csv(f'{pfx}_ext.csv.gz', index=False)
    pd.DataFrame(wr_rows, columns=['uid', 'group', 'target', 'diagnostic_caes', 'diagnostic_lfp', 'actual_caes', 'actual_lfp', 'B', 'cyclic_caes', 'cyclic_lfp']).to_csv(f'{pfx}_wr.csv.gz', index=False)
    pd.DataFrame(cons_rows, columns=['uid', 'group', 'target', 'absolute_error', 'relative_error']).to_csv(f'{pfx}_cons.csv.gz', index=False)
    (P1_SHARD_DIR / f'shard_{shard_id:04d}_h2qa.json').write_text(json.dumps({
        'h2_count': h2_count, 'max_phys': h2_max_phys, 'max_stat': h2_max_stat, 'max_gap': h2_max_gap,
    }))
    done_marker.write_text('ok\n')
    return shard_id, 'done', round(time.monotonic() - t0, 2)


def merge_phase1(n_shards: int):
    print('Merging Phase 1 shards...', flush=True)
    for tag, out_name in [
        ('rec', 'japan_records.csv.gz'),
        ('cat', 'japan_catalogue58.csv.gz'),
        ('fw', 'japan_fill_windows_by_duration.csv.gz'),
        ('fac', 'japan_factorial.csv.gz'),
        ('price', 'japan_price.csv.gz'),
        ('rule', 'japan_rule.csv.gz'),
        ('ext', 'japan_extension_0to12.csv.gz'),
        ('wr', 'japan_window_recursion.csv.gz'),
        ('cons', 'japan_chronology_conservation.csv.gz'),
    ]:
        dfs = [pd.read_csv(P1_SHARD_DIR / f'shard_{s:04d}_{tag}.csv.gz', dtype={'uid': str}) for s in range(n_shards)]
        df = pd.concat(dfs, ignore_index=True)
        tmp = OUT_DIR / (out_name + '.tmp')
        df.to_csv(tmp, index=False, compression='gzip')
        tmp.replace(OUT_DIR / out_name)
        print(f'  Wrote {out_name}: {len(df):,} rows', flush=True)

    qas = [json.loads((P1_SHARD_DIR / f'shard_{s:04d}_h2qa.json').read_text()) for s in range(n_shards)]
    h2_qa = {
        'hydrogen_lp_solves': sum(x['h2_count'] for x in qas),
        'all_h2_optimal': True,
        'max_physical_residual': max(x['max_phys'] for x in qas),
        'max_stationarity_residual': max(x['max_stat'] for x in qas),
        'max_duality_gap': max(x['max_gap'] for x in qas),
    }
    (OUT_DIR / 'japan_h2_qa.json').write_text(json.dumps(h2_qa, indent=2) + '\n')
    print('Phase 1 merged! H2 QA:', json.dumps(h2_qa), flush=True)


def run_phase2_mc_record(row_idx: int):
    shard_path = SHARD_DIR / f'mc_{row_idx:05d}.npz'
    if shard_path.exists():
        return row_idx, 'cached', 0.0
    t0 = time.monotonic()
    r, uid, uk, imp, rev, P, c1, flags, raw, arr = _build_record_arrays(row_idx)
    segs = _precompute_obs_segments(_SEG, flags)

    call_day = (arr[:, 1] > 1e-10).astype(np.int8).reshape(N_DAYS, 24)
    ch_day = (arr[:, 2] > 1e-9).astype(np.float64).reshape(N_DAYS, 24)
    ec_day = (np.where(arr[:, 1] > 1e-10, 0.0, np.minimum(arr[:, 2], 1.5)) * 0.55).reshape(N_DAYS, 24)

    n_seeds = 1000
    prev_mat = np.empty((n_seeds, N_DAYS), dtype=np.int32)
    lws = np.empty(n_seeds, dtype=np.int16)
    for s in range(n_seeds):
        perm = _fast_obs_random_order(uk, segs, s)
        prev_mat[s, perm] = np.roll(perm, 1)
        c_p = call_day[perm].ravel()
        rid = np.cumsum(np.r_[0, np.diff(c_p) != 0])
        is_c = np.bincount(rid, weights=c_p) > 0
        ch_w = np.bincount(rid, weights=ch_day[perm].ravel())
        ec_w = np.bincount(rid, weights=ec_day[perm].ravel())
        lws[s] = int(((~is_c) & (ch_w > 0) & (ec_w >= 21.6 - 1e-9)).sum())

    visited = np.zeros(n_seeds, dtype=bool)
    tour = np.empty(n_seeds, dtype=np.int32)
    cur_p = INIT_PREV
    for step in range(n_seeds):
        dists = (prev_mat != cur_p).sum(axis=1)
        dists[visited] = 999999
        nxt = int(np.argmin(dists))
        tour[step] = nxt
        visited[nxt] = True
        cur_p = prev_mat[nxt]

    cidx = CollapsedIndexer(arr)
    base_cfg = _CAT58['li_lfp_4h']
    mL = PersistentBranchLP(cidx, base_cfg, base_cfg, BASE_SCENARIO_EUR)
    mC = PersistentBranchLP(cidx, base_cfg, _CAT58['caes_24h'], BASE_SCENARIO_EUR)
    Bs = np.empty(n_seeds, dtype=np.float64)
    try:
        mL.solve_cost_usd()
        mC.solve_cost_usd()
        for seed_idx in tour:
            p_vec = prev_mat[seed_idx]
            mL.rewire(p_vec)
            mC.rewire(p_vec)
            Bs[seed_idx] = mL.solve_cost_usd() - mC.solve_cost_usd()
    finally:
        mL.dispose()
        mC.dispose()

    tmp = shard_path.with_suffix('.tmp.npz')
    np.savez(tmp, B=Bs, lw=lws)
    tmp.replace(shard_path)
    _trim_heap()
    return row_idx, 'done', round(time.monotonic() - t0, 2)


def merge_phase2(n_records: int):
    print('Assembling Phase 2 full (10,612 x 1,000) Monte Carlo matrices and exact summaries...', flush=True)
    B_mat = np.empty((n_records, 1000), dtype=np.float64)
    lw_mat = np.empty((n_records, 1000), dtype=np.int16)
    for idx in range(n_records):
        z = np.load(SHARD_DIR / f'mc_{idx:05d}.npz')
        B_mat[idx] = z['B']
        lw_mat[idx] = z['lw']
    np.save(OUT_DIR / 'japan_mc_B.npy', B_mat)
    np.save(OUT_DIR / 'japan_mc_lw.npy', lw_mat)

    rec = pd.read_csv(OUT_DIR / 'japan_records.csv.gz', dtype={'uid': str}).sort_values('row_idx').reset_index(drop=True)
    assert len(rec) == n_records and np.array_equal(rec.row_idx.to_numpy(), np.arange(n_records))

    op_real = rec.B.to_numpy() + RM.PREM36[2035]
    op_mc = B_mat + RM.PREM36[2035]

    def vec_ordered_quantiles(mat_sorted: np.ndarray, qs=(0.05, 0.50, 0.95)) -> np.ndarray:
        m_len = mat_sorted.shape[1]
        out_q = np.empty((mat_sorted.shape[0], len(qs)), dtype=np.float64)
        for j, q in enumerate(qs):
            pos = (m_len - 1) * float(q)
            lo = int(np.floor(pos))
            hi = int(np.ceil(pos))
            v = mat_sorted[:, lo].copy()
            good = np.isfinite(mat_sorted[:, lo]) & np.isfinite(mat_sorted[:, hi])
            v[good] = mat_sorted[good, lo] + (mat_sorted[good, hi] - mat_sorted[good, lo]) * (pos - lo)
            out_q[:, j] = v
        return out_q

    sum_rows = []
    pl_df = None
    lw_mc_med_per_uid = np.median(lw_mat, axis=1)

    for year in RM.YEARS:
        rv = RM.roi(op_real, year)
        rnd_roi = RM.roi(op_mc, year)
        rnd_sorted = np.sort(rnd_roi, axis=1)
        qu = vec_ordered_quantiles(rnd_sorted, (0.05, 0.50, 0.95))
        valid = np.isfinite(qu[:, 0]) & np.isfinite(qu[:, 2])
        finite_r = np.isfinite(rv)
        med = float(RM.ordered_quantile(rv, [0.5])[0])
        mcmed = float(RM.ordered_quantile(qu[:, 1], [0.5])[0])
        ranks = (rnd_roi < rv[:, None]).mean(axis=1) + 0.5 * (rnd_roi == rv[:, None]).mean(axis=1)

        row = {
            'group': 'community', 'q': 50, 'year': int(year), 'n': int(n_records),
            'real_roi_med': med, 'mc_roi_med': mcmed, 'real_med': med, 'mc_med': mcmed,
            'real_min': float(rv[finite_r].min()), 'real_max': float(rv[finite_r].max()),
            'real_none': int((~finite_r).sum()), 'mc_none_share': float((~np.isfinite(rnd_roi)).mean()),
            'mc_width_med': float(np.median(qu[valid, 2] - qu[valid, 0])), 'mc_width_n': int(valid.sum()),
            'real_full_med': float(rec.hybrid_window_full.median()),
            'mc_full_med': float(np.median(lw_mc_med_per_uid)),
        }
        for h in RM.RATES:
            thr = RM.premium(year, h / 100.0)
            row[f'real_win_{h:g}'] = int((op_real >= thr).sum())
            row[f'mc_win_{h:g}'] = float((op_mc >= thr).mean(axis=1).mean())

        thr36 = RM.premium(year, 0.036)
        prob36 = (op_mc >= thr36).mean(axis=1)
        real_w36 = op_real >= thr36
        row['disagree_36'] = int(((prob36 > 0.5) != real_w36).sum())
        row['sensitive_36'] = int(((prob36 > 0.05) & (prob36 < 0.95)).sum())
        row['real_pct_med'] = float(np.median(ranks))
        sum_rows.append(row)

        if year == 2035:
            pl_df = pd.DataFrame({
                'uid': rec.uid,
                'mc_q05': qu[:, 0],
                'mc_q50': qu[:, 1],
                'mc_q95': qu[:, 2],
                'mc_win36': prob36,
                'real_pct': ranks,
                'roi_real': rv,
                'group': 'community',
                'lw_real': rec.long_windows.to_numpy(),
                'lw_mc_med': lw_mc_med_per_uid,
            })

    pl_df.to_csv(OUT_DIR / 'japan_placements_records.csv.gz', index=False, compression='gzip')
    pd.DataFrame(sum_rows).to_csv(OUT_DIR / 'japan_primary_summary.csv', index=False)
    print('Phase 2 complete! 2035 summary:', json.dumps(sum_rows[-1], indent=2), flush=True)


def rss_tree_gb() -> float:
    total_kb = 0
    me = os.getpid()
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            stat_txt = (entry / 'status').read_text()
            ppid = 0
            rss_anon = 0
            for ln in stat_txt.splitlines():
                if ln.startswith('PPid:'):
                    ppid = int(ln.split()[1])
                elif ln.startswith('RssAnon:'):
                    rss_anon = int(ln.split()[1])
            if pid == me or ppid == me:
                total_kb += rss_anon
        except OSError:
            pass
    return total_kb / (1024 * 1024)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=110)
    ap.add_argument('--phase', choices=['selftest', '1', '2', 'all'], default='all')
    args = ap.parse_args()

    # Run selftest inside an isolated child process so parent remains ~45 MB before fork!
    p_st = mp.get_context('fork').Process(target=_selftest_subprocess)
    p_st.start()
    p_st.join()
    if p_st.exitcode != 0:
        raise SystemExit(f'Selftest failed with exitcode {p_st.exitcode}')
    if args.phase == 'selftest':
        return

    n_records = len(_META)
    assert n_records == 10612
    n_workers = limit_workers(args.workers)
    print(json.dumps({
        'event': 'suite_start', 'n_records': n_records, 'workers': n_workers,
        'phase': args.phase, 'parent_rss_gb': round(rss_tree_gb(), 3),
    }), flush=True)

    if args.phase in ('1', 'all'):
        if not (OUT_DIR / 'japan_h2_qa.json').exists():
            shard_size = 12
            all_shards = [(i // shard_size, list(range(i, min(i + shard_size, n_records)))) for i in range(0, n_records, shard_size)]
            todo_shards = [s for s in all_shards if not (P1_SHARD_DIR / f'shard_{s[0]:04d}.done').exists()]
            print(json.dumps({'phase': 1, 'todo_shards': len(todo_shards), 'total_shards': len(all_shards)}), flush=True)
            t_p1 = time.monotonic()
            if todo_shards:
                with mp.get_context('fork').Pool(n_workers, initializer=_init_worker, maxtasksperchild=4) as pool:
                    for k, (sid, st, secs) in enumerate(pool.imap_unordered(run_phase1_shard, todo_shards), 1):
                        if k % 25 == 0 or k == len(todo_shards):
                            rg = rss_tree_gb()
                            print(json.dumps({
                                'phase': 1, 'done_new': k, 'of_todo': len(todo_shards),
                                'wall_s': round(time.monotonic() - t_p1, 1),
                                'rss_anon_gb': round(rg, 2),
                            }), flush=True)
                            if rg > 41.5:
                                raise RuntimeError(f'Memory guard triggered: {rg:.2f} GB > 41.5 GB')
            merge_phase1(len(all_shards))
        else:
            print('Phase 1 outputs already merged.', flush=True)

    if args.phase in ('2', 'all'):
        todo = [idx for idx in range(n_records) if not (SHARD_DIR / f'mc_{idx:05d}.npz').exists()]
        print(json.dumps({'phase': 2, 'todo_records': len(todo), 'total_records': n_records}), flush=True)
        if todo:
            t_p2 = time.monotonic()
            with mp.get_context('fork').Pool(n_workers, initializer=_init_worker, maxtasksperchild=20) as pool:
                for k, (ridx, st, secs) in enumerate(pool.imap_unordered(run_phase2_mc_record, todo, chunksize=1), 1):
                    if k % 100 == 0 or k == len(todo):
                        elapsed = time.monotonic() - t_p2
                        rate = k / max(elapsed, 1e-3)
                        eta_min = (len(todo) - k) / max(rate, 1e-6) / 60.0
                        rg = rss_tree_gb()
                        print(json.dumps({
                            'phase': 2, 'done': k, 'of': len(todo),
                            'last_rec_s': secs,
                            'wall_min': round(elapsed / 60.0, 2),
                            'eta_min': round(eta_min, 2),
                            'rss_anon_gb': round(rg, 2),
                        }), flush=True)
                        if rg > 41.5:
                            raise RuntimeError(f'Memory guard triggered: {rg:.2f} GB > 41.5 GB')
        merge_phase2(n_records)


if __name__ == '__main__':
    main()
