"""Hydrogen storage-medium replacement; every non-hydrogen parameter is frozen.

The original discharge-rated hydrogen power bundle already includes compression.
Its 2025/2030/2035 learning curve and replacement schedule are retained exactly.
DEA Type-I 200bar tank + installation replaces cavern energy capacity only.
"""
from pathlib import Path
from functools import lru_cache
import json,sys,csv
import numpy as np
HERE=Path(__file__).resolve().parent;WORK=HERE.parent;V86=WORK.parents[1];ROOT=V86.parents[1]
V82=ROOT/'submission/Nature_Energy_v82/revision'
SOURCES=WORK/'sources'
sys.path.insert(0,str(V82/'discount_sensitivity_20261022/code/module_a'))
import finance as OLD
FX=OLD.FX;RATE=.036;H2_ID='h2_tank';H2_LHV=33.33;H2_KWH_AC_PER_KG=19.3
H2_ELECTROLYSIS_KWH_AC_KG=39.4/.725/(.98**2)

@lru_cache(None)
def dea(year=2035):
    import openpyxl
    w=openpyxl.load_workbook(SOURCES/'DEA_storage.xlsx',data_only=True,read_only=True)
    s=w['151a Hydrogen Storage - Tanks'];rows={str(r[1]):list(r) for r in s.iter_rows(values_only=True) if len(r)>1 and r[1] is not None};w.close()
    years=[2020,2030,2040,2050]
    pick=lambda name:float(np.interp(year,years,[float(rows[name][j]) for j in (3,4,5,6)]))
    # Recover compression energy from DEA's storage-only efficiency definition,
    # removing its one-percentage-point gas-loss allowance (steel tanks: negligible permeation).
    comp=[H2_LHV*(1/(float(rows['Round trip efficiency [%]'][j])+.01)-1) for j in (3,4,5,6)]
    return dict(tank_eur_kwh_h2=1000*pick('Type I tanks component [M€/MWh]'),
                installation_eur_kwh_h2=1000*pick('Installation, equipment, manhours [M€/MWh]'),
                compression_kwh_ac_kg=float(np.interp(year,years,comp)),source_price_year=2020,
                interpolation_year=year)

def h2_physics():
    compression=dea(2035)['compression_kwh_ac_kg']
    return dict(electrolysis_kwh_ac_kg=H2_ELECTROLYSIS_KWH_AC_KG,compression_kwh_ac_kg=compression,
                generation_kwh_ac_kg=H2_KWH_AC_PER_KG,rte=H2_KWH_AC_PER_KG/(H2_ELECTROLYSIS_KWH_AC_KG+compression),
                charge_power_ratio=3.,self_discharge_pct_day=0.)

@lru_cache(None)
def cost_components(cid,year=2035,rate=RATE,tank_factor=1.):
    original=cid.replace('h2_tank_','h2_cavern_')
    try:old=OLD.cost_components(original,year,rate)
    except KeyError:
        assert cid.startswith('h2_tank_')
        old=dict(OLD.cost_components('h2_cavern_24h',year,rate),duration_h=float(cid.rsplit('_',1)[1][:-1]))
    if not cid.startswith('h2_tank_'):return dict(old)
    h=old['duration_h'];src=dea(year);conversion=H2_LHV/H2_KWH_AC_PER_KG
    tank=src['tank_eur_kwh_h2']*FX*conversion*tank_factor
    install=src['installation_eur_kwh_h2']*FX*conversion
    ce=tank+install;cp=old['power_capex_usd_kw']
    # Preserve the original unit-power FOM path; add explicitly capacity-dependent
    # ground-tank and installation maintenance (DEA Table6: 1250/300000,1000/150000).
    storage_fom=h*((1250/300000)*tank+(1000/150000)*install)
    fom=old['fixed_om_usd_kw_year']+storage_fom
    annual_capex=OLD.crf(rate,old['calendar_life_years'])*(cp+h*ce)
    total=annual_capex+old['annualized_replacement_usd_kw_year']+fom
    return dict(old,config_id=cid,tech_id=H2_ID,energy_capex_usd_kwh=ce,
                fixed_om_usd_kw_year=fom,annualized_capex_usd_kw_year=annual_capex,
                annual_fixed_cost_usd_kw_year=total,annual_fixed_cost_eur_kw_year=total/FX,
                tank_capex_usd_kwh_output=tank,installation_capex_usd_kwh_output=install,
                retained_unit_power_fom_usd_kw_year=old['fixed_om_usd_kw_year'],storage_fom_usd_kw_year=storage_fom,
                power_cost_policy='unchanged v85 discharge-rated bundle: electrolyzer/fuel cell/compressor/inverter/BOP',
                compressor_capital_policy='already included in retained power bundle; no duplicate capital addition',
                energy_cost_policy='DEA Type-I tank + installation; source-year interpolation, not cavern learning')

def annual_fixed_cost_usd(cid,year=2035,rate=RATE):return cost_components(cid,year,rate)['annual_fixed_cost_usd_kw_year']

@lru_cache(None)
def catalogue():
    old=json.loads((V82/'issues3_4_20261018/qa/input_manifest.json').read_text())['catalogue'];out={}
    for cid,cfg in old.items():
        key=cid.replace('h2_cavern_','h2_tank_');c=dict(cfg)
        if c['tech_id']=='h2_cavern':c.update(tech_id=H2_ID,rte=h2_physics()['rte'],self_discharge_pct_day=0.)
        c['fixed_cost_eur_kw_year']=annual_fixed_cost_usd(key)/FX;out[key]=c
    for h in (4,8,12,16,20):
        key=f'h2_tank_{h}h';out[key]=dict(out['h2_tank_24h'],duration_h=float(h),fixed_cost_eur_kw_year=annual_fixed_cost_usd(key)/FX)
    return out

def export():
    obj=dict(configuration='hydrogen_tank_only',physical_parameters=h2_physics(),
             dea={y:dea(y) for y in (2025,2030,2035)},catalogue=catalogue(),
             retained_technologies=[c for c in catalogue() if not c.startswith('h2_tank_')],
             currency_policy='retain v85 source-price convention and FX 1.13 USD/EUR',
             learning_curve='dataset/M2_LDES_learning_cost_curves_v0_7.csv',
             scope='Hydrogen storage medium only; A-CAES and all other technologies retain v85 parameters')
    (WORK/'parameters/parameters.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    rows=[cost_components(cid,y) for y in (2025,2030,2035) for cid in catalogue()]
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with (WORK/'parameters/cost_catalogue_2025_2035.csv').open('w') as f:
        wr=csv.DictWriter(f,fieldnames=keys);wr.writeheader();wr.writerows(rows)
    # Equality checks cover the complete annualization, not just headline CAPEX.
    for cid in catalogue():
        if cid.startswith('h2_tank_'):continue
        for y in (2025,2030,2035):
            assert cost_components(cid,y)==OLD.cost_components(cid,y,RATE),(cid,y)
    return obj

if __name__=='__main__':
    export();print(json.dumps({'physics':h2_physics(),'h2_24h':cost_components('h2_tank_24h'),'caes_24h':cost_components('caes_24h')},indent=2))
