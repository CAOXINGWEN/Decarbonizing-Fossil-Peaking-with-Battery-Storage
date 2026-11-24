"""Exact paired q5 portfolio optimization with all five European season segments."""
from pathlib import Path
import argparse
import fcntl
import json
import sys
import time
from europe_inputs import OUT, INPUT, ROOT, BASE, V85, legacy, C, M, np, pd, VERSION
from suite_common import signatures, contract, parallel, sha, write_json, RM
import compact_dispatch
from compact_dispatch import DayGraph, CompactLP
import run_mc

META=ARR=FLAGS=SEG=None
CHECK=OUT/"checkpoints"


def setup():
    global META,ARR,FLAGS,SEG
    CHECK.mkdir(exist_ok=True)
    META=pd.read_csv(INPUT/"records.csv",dtype={"uid":str},float_precision="round_trip")
    ARR,FLAGS,SEG=[np.load(INPUT/(n+".npy"),mmap_mode="r") for n in ("hourly","flags","segments")]
    assert ARR.shape==(132,8760,3) and len(META)==132
    receipt=json.loads((INPUT/"inputs.json").read_text())
    assert all(sha(INPUT/k)==v for k,v in receipt["outputs"].items())
    sc=signatures([Path(__file__),Path(compact_dispatch.__file__),Path(run_mc.__file__),Path(M.__file__),
                   Path(sys.modules["suite_common"].__file__),INPUT/"inputs.json",Path(legacy.O.__file__)])
    contract(OUT,dict(solver=VERSION,seeds=[0,999],q=5,source_sha256=sc))


def segments(flags, labels):
    result=[]
    for sg in sorted(np.unique(labels)):
        dates=np.flatnonzero(labels==sg)
        ev=dates[flags[dates]]
        clusters=[g.tolist() for g in np.split(ev,np.flatnonzero(np.diff(ev)!=1)+1) if len(g)]
        normal=dates[~flags[dates]].tolist()
        assert not clusters or len(normal)>=len(clusters)+1
        result.append((int(sg),clusters,normal))
    return result


class CertifiedCompact(CompactLP):
    def __init__(self,graph,tail):
        super().__init__(graph,M._CAT58["li_lfp_4h"],tail,M)
        self.matrix=self.m.getA().tocsc()
        v=self.m.getVars()
        cons=self.m.getConstrs()
        self.coeff=np.asarray(self.m.getInfo("Obj",v))
        self.lo=np.asarray(self.m.getInfo("LB",cons))
        self.hi=np.asarray(self.m.getInfo("UB",cons))
        self.ub=np.asarray(self.m.getInfo("UB",v))
        self.flo=np.where(self.lo <= -M.cp.COPT.INFINITY,0.,self.lo)
        self.fhi=np.where(self.hi >= M.cp.COPT.INFINITY,0.,self.hi)
        self.fub=np.where(self.ub >= M.cp.COPT.INFINITY,0.,self.ub)
        self.ends=self.ncols-1-2*graph.ns+graph.last_state
        self.old=self.ends[M.INIT_PREV]
        self.audit=np.zeros(3)
        self.cold=0
        self.nsolves=0

    def certificate(self,m):
        x,act,dual,red=map(np.asarray,m.getLpSolution())
        physical=max(0.,(self.lo-act).max(),(act-self.hi).max(),(-x).max(),(x-self.ub).max())
        stat=self.coeff-self.matrix.T@dual-red
        graph=self.idx
        first=graph.first_state
        new=self.ends[self.cur_prev]
        for off,factor in ((0,self.rb_k0),(graph.ns,self.rt_k0)):
            update=factor*dual[first+off]
            stat[self.old+off]-=update
            stat[new+off]+=update
        dual_obj=self.flo@np.maximum(dual,0.)+self.fhi@np.minimum(dual,0.)+self.fub@np.minimum(red,0.)
        gap=abs(self.coeff@x-dual_obj)/1000
        return np.array([physical,np.max(np.abs(stat)),gap])

    def certified(self):
        value=super().solve_cost_usd()
        error=self.certificate(self.m)
        if error.max()>=1e-8:
            actual=self.m.getA()
            for method in (2,1,3):
                fresh=M._ENV.createModel("europe_certificate")
                try:
                    for k,v in (("Logging",0),("Threads",1),("Presolve",0),("DualPerturb",0),("Scaling",0),
                                ("FeasTol",1e-10),("DualTol",1e-10),("MatrixTol",1e-12),("LpMethod",method)):
                        fresh.setParam(k,v)
                    fresh.loadMatrix(self.coeff,actual,self.lo,self.hi,np.zeros_like(self.ub),self.ub)
                    fresh.solveLP()
                    if fresh.status!=M.cp.COPT.OPTIMAL: continue
                    error=self.certificate(fresh)
                    if error.max()<1e-8:
                        fixed=self.base_cfg["fixed_cost_eur_kw_year"]+self.tail_cfg["fixed_cost_eur_kw_year"]
                        value=(fresh.objval+self.idx.task_sum*self.ev_mwh+1000*fixed)*M.FX/1000.
                        self.cold+=1
                        break
                finally:
                    fresh.dispose()
        assert np.isfinite(error).all() and error.max()<1e-8, error
        self.nsolves+=1
        self.audit=np.maximum(self.audit,error)
        return value


def one(idx):
    rec=META.iloc[idx]
    arr,flags,sg=ARR[idx],FLAGS[idx],SEG[idx]
    segments5=segments(flags,sg)
    assert [v[0] for v in segments5]==[1,2,3,4,5]
    ens=M.ce.ChronologyEnsemble(pd.DataFrame(dict(segment=sg,event=flags)),rec.unit_key)
    byday=arr.reshape(365,24,3)
    prev=np.empty((1000,365),dtype=np.int32)
    windows=np.empty(1000,dtype=np.int16)
    saturated=np.empty_like(windows)
    for seed in range(1000):
        order=M._fast_obs_random_order(rec.unit_key,segments5,seed)
        assert np.array_equal(np.sort(order),np.arange(365)) and np.array_equal(sg[order],sg)
        if seed in (0,999):
            assert np.array_equal(order,legacy.O.obs_random_order(ens,seed))
        prev[seed]=M._prev_from_order(order)
        a=byday[order].reshape(8760,3)
        windows[seed]=M._long_windows_fast(a)
        saturated[seed]=M.WD.branch(a,24,.55,1.5)["saturated"]
    graph=DayGraph(arr)
    models=[]
    try:
        for key in ("li_lfp_4h","caes_24h"):
            models.append(CertifiedCompact(graph,M._CAT58[key]))
        l,h=models
        real=l.certified()-h.certified()
        assert np.isfinite(real) and (real>0)==(rec.reference_B>0),(rec.uid,real,rec.reference_B)
        b=np.empty(1000)
        for i in run_mc.solve_order(prev):
            l.rewire(prev[i])
            h.rewire(prev[i])
            b[i]=l.certified()-h.certified()
        counts=sum(x.nsolves for x in models)
        audit=np.maximum(l.audit,h.audit)
        cold=l.cold+h.cold
    finally:
        for model in models: model.dispose()
    assert np.isfinite(b).all() and counts==2002
    sat_real=M.WD.branch(arr,24,.55,1.5)["saturated"]
    assert sat_real==rec.archive_saturated
    file=CHECK/f"{idx:03d}.npz"
    tmp=file.with_suffix(".partial.npz")
    np.savez_compressed(tmp,uid=rec.uid,real_B=real,B=b,long_windows=windows,saturated=saturated,
                        lw_real=M._long_windows_fast(arr),sat_real=sat_real,max_residuals=audit,
                        nsolves=counts,cold=cold)
    tmp.replace(file)
    M._trim_heap()
    return idx


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--workers",type=int,default=16)
    cli.add_argument("--indices",type=int,nargs="+")
    args=cli.parse_args()
    setup()
    with (OUT/"run.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        todo=[i for i in (args.indices or range(132)) if not (CHECK/f"{i:03d}.npz").exists()]
        np.random.default_rng(20261120).shuffle(todo)
        if todo:
            stats=parallel(OUT,todo,one,args.workers)
            write_json(CHECK/f"execution_{time.time_ns()}.json",stats)
