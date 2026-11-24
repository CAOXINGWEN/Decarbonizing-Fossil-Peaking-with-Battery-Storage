"""Extend the tank-storage candidate grid to the common 4h study boundary."""
from recompute import *

def extend(uid):
    p=OUT/f'{uid}.json';d=json.loads(p.read_text());done={x['variant'] for x in d['hydrogen']}
    arr,_=D.hourly(uid)
    for cid,cfg in P.catalogue().items():
        if cfg['tech_id']!=P.H2_ID or cid in done:continue
        a=solve(arr,cid)
        d['hydrogen'].append(dict(uid=uid,variant=cid,tech=P.H2_ID,duration_h=cfg['duration_h'],
                                 B=d['lfp']['cost_usd_kw_year']-a['cost_usd_kw_year'],**a))
    assert len(d['hydrogen'])==22
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d));tmp.replace(p)
    C.model.matrices.cache_clear()
    return uid

if __name__=='__main__':
    P.export();uids=[p.stem for p in sorted(OUT.glob('*.json'))]
    for i in range(0,len(uids),8):
        while available()<25:time.sleep(10)
        with mp.get_context('spawn').Pool(8) as pool:list(pool.imap_unordered(extend,uids[i:i+8]))
        print(json.dumps({'extended':min(i+8,len(uids)),'of':len(uids)}),flush=True)
