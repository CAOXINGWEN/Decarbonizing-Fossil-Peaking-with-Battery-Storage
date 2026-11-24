"""Exact elimination of within-day states during intervals without charging.

Store output-side energy x (MWh), so x[t] = retention*x[t-1] + rte*q[t] - d[t].
In an interval without charging the state is non-increasing. Bounds at the
endpoints enforce every intermediate bound. Keep both ends of charging hours
and every day boundary. Intermediate discharge coefficients retain the exact
hour-specific decay. Daily boundaries can then be rewired exactly as before.
"""
from dataclasses import dataclass
import numpy as np
from scipy.sparse import coo_matrix


@dataclass
class DayGraph:
    arr: np.ndarray

    def __post_init__(self):
        load, high, surplus = self.arr.T
        self.task_sum = float(load.sum())
        hours = np.arange(len(load), dtype=np.int32)
        charging = surplus > 1e-12
        day_end = hours % 24 == 23
        keep = day_end | charging
        keep[:-1] |= charging[1:]
        # Within a pure charging run, the next hour with at least 2.5 MW per
        # MW can supply BOTH branches' maximum charging powers. Intervening
        # upper bounds at such joins are redundant at optimum. If an inventory
        # exceeds K, clip it to K. At an ensuing hour this can require at most
        # (1-retention)*K/rte extra grid charge, feasible under that guarantee.
        # Clipping lowers total charging cost and preserves both endpoint states.
        pure = charging & (load<=1e-12)
        removable = pure[:-1] & pure[1:] & (surplus[1:]>=2.5) & ~day_end[:-1]
        keep[:-1] &= ~removable
        ends = hours[keep]
        state_of_hour = np.full(len(load), -1, dtype=np.int32)
        state_of_hour[keep] = np.arange(len(ends), dtype=np.int32)
        next_end = np.minimum.accumulate(np.where(keep, hours, len(hours))[::-1])[::-1]
        self.ns = len(ends)
        self.state_at_next_end = state_of_hour[next_end]
        self.decay_steps = next_end-hours
        first = np.r_[True, ends[1:]//24 != ends[:-1]//24]
        self.first_state = np.flatnonzero(first)
        self.first_steps = ends[first]%24 + 1
        self.interior = np.flatnonzero(~first)
        self.interior_steps = ends[self.interior] - ends[self.interior-1]
        self.last_state = state_of_hour[day_end]
        self.charge = np.flatnonzero(charging)
        self.base_ub = np.where(load>1e-12, np.minimum(load, 1.0), 0)
        self.tail_ub = np.where(high>1e-12, np.minimum(high, 1.0), 0)
        self.peak = np.flatnonzero((load>1e-12) & (load>=load.max()-2.0-1e-7))
        self.is_peak = np.zeros(len(load), dtype=bool)
        self.is_peak[self.peak] = True

        # Guaranteed room for one branch after allowing the other its entire
        # discharge-power bound. Cumulate only within a discharge-only interval.
        starts = np.maximum.accumulate(np.where(charging | (hours%24==0), hours, 0))
        def earlier_capacity(own, other):
            guaranteed = np.minimum(own, np.maximum(0.0, load-other))
            prefix = np.r_[0.0, guaranteed.cumsum()]
            return prefix[:-1] - prefix[starts]
        self.base_earlier = earlier_capacity(self.base_ub, self.tail_ub)
        self.tail_earlier = earlier_capacity(self.tail_ub, self.base_ub)


class CompactLP:
    """Registered two-storage cost LP on the reduced physical state graph."""
    def __init__(self, graph, base, tail, model):
        self.model = model
        self.idx = graph
        self.base_cfg, self.tail_cfg = base, tail
        scen = model.BASE_SCENARIO_EUR
        self.ev_mwh = scen["energy_value_eur_mwh"]
        rb = (1.0-base["self_discharge_pct_day"]/100)**(1/24)
        rt = (1.0-tail["self_discharge_pct_day"]/100)**(1/24)
        kb, kt = .9*base["duration_h"], .9*tail["duration_h"]

        # A non-peak later discharge is dominated when earlier guaranteed
        # discharge capacity since the last recharge already exceeds the
        # maximum initial energy. Shifting that discharge earlier saves losses,
        # leaves later inventories feasible, and cannot increase residual peak.
        db = np.flatnonzero((graph.base_ub>0) & ((graph.base_earlier<kb+1e-7) | graph.is_peak))
        dt = np.flatnonzero((graph.tail_ub>0) & ((graph.tail_earlier<kt+1e-7) | graph.is_peak))
        iq = graph.charge
        iq_b = np.minimum(graph.arr[iq,2], base["charge_power_ratio"])
        iq_t = np.minimum(graph.arr[iq,2], tail["charge_power_ratio"])
        load = graph.arr[:,0]
        ns, nq = graph.ns, len(iq)
        o_qt, o_db = nq, 2*nq
        o_dt = o_db+len(db)
        o_sb = o_dt+len(dt)
        o_st, o_rv = o_sb+ns, o_sb+2*ns
        dbase = np.full(len(load), -1, dtype=np.int32)
        dtail = dbase.copy()
        dbase[db] = o_db+np.arange(len(db))
        dtail[dt] = o_dt+np.arange(len(dt))
        share_d = np.flatnonzero((dbase>=0) & (dtail>=0) & (graph.base_ub+graph.tail_ub>load))
        share_q = np.flatnonzero(iq_b+iq_t>graph.arr[iq,2])
        pk = graph.peak
        rp = 2*ns+len(share_d)+len(share_q)+np.arange(len(pk))
        nrows = 2*ns+len(share_d)+len(share_q)+len(pk)
        rows, cols, vals = [], [], []

        def add(rr, cc, value):
            rows.append(rr)
            cols.append(cc)
            vals.append(np.full(len(rr), value) if np.isscalar(value) else value)

        state = np.arange(ns, dtype=np.int32)
        self.cur_prev = model.INIT_PREV.copy()
        self.rb_k0 = rb**graph.first_steps
        self.rt_k0 = rt**graph.first_steps
        for offset, store_col, charge_col, discharge_col, discharge_hours, cfg, retention in [
            (0, o_sb, 0, o_db, db, base, rb), (ns, o_st, o_qt, o_dt, dt, tail, rt)
        ]:
            add(offset+state, store_col+state, 1.)
            add(offset+graph.interior, store_col+graph.interior-1, -(retention**graph.interior_steps))
            add(offset+graph.first_state, store_col+graph.last_state[self.cur_prev], -(retention**graph.first_steps))
            add(offset+graph.state_at_next_end[iq], charge_col+np.arange(nq), -cfg["rte"]*retention**graph.decay_steps[iq])
            add(offset+graph.state_at_next_end[discharge_hours], discharge_col+np.arange(len(discharge_hours)),
                retention**graph.decay_steps[discharge_hours])
        rd = 2*ns+np.arange(len(share_d))
        rq = 2*ns+len(share_d)+np.arange(len(share_q))
        add(rd, dbase[share_d], 1.)
        add(rd, dtail[share_d], 1.)
        add(rq, share_q, 1.)
        add(rq, o_qt+share_q, 1.)
        for dmap in [dbase, dtail]:
            mask = dmap[pk]>=0
            add(rp[mask], dmap[pk[mask]], 1.)
        add(rp, np.full(len(pk), o_rv, dtype=np.int32), 1.)
        matrix = coo_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),
                            shape=(nrows,o_rv+1)).tocsc()
        inf = model.cp.COPT.INFINITY
        rl = np.r_[np.zeros(2*ns), np.full(len(share_d)+len(share_q),-inf), load[pk]]
        ru = np.r_[np.zeros(2*ns), load[share_d], graph.arr[iq[share_q],2], np.full(len(pk),inf)]
        ub = np.r_[iq_b, iq_t, graph.base_ub[db], graph.tail_ub[dt], np.full(ns,kb), np.full(ns,kt),inf]
        obj = np.r_[np.full(2*nq,scen["charge_price_eur_mwh"]),
                    np.full(len(db),base["vom_eur_mwh"]-self.ev_mwh),
                    np.full(len(dt),tail["vom_eur_mwh"]-self.ev_mwh),
                    np.zeros(2*ns), 1000*scen["firm_value_eur_kw_year"]]
        self.m = model._ENV.createModel("compact_daily")
        for key, value in [("Logging",0),("Threads",1),("FeasTol",1e-8),("DualTol",1e-8),
                           ("TimeLimit",180),("LpMethod",1)]:
            self.m.setParam(key,value)
        self.m.loadMatrix(obj,matrix,rl,ru,np.zeros(o_rv+1),ub)
        constraints = self.m.getConstrs()
        variables = self.m.getVars()
        self.c_sb_mid = [constraints[i] for i in graph.first_state]
        self.c_st_mid = [constraints[ns+i] for i in graph.first_state]
        self.v_sb_end = [variables[o_sb+i] for i in graph.last_state]
        self.v_st_end = [variables[o_st+i] for i in graph.last_state]
        self.nrows, self.ncols = matrix.shape

    def rewire(self, previous):
        return self.model.PersistentBranchLP.rewire(self, previous)

    def solve_cost_usd(self):
        return self.model.PersistentBranchLP.solve_cost_usd(self)

    def dispose(self):
        self.m.dispose()
        self.m = None
