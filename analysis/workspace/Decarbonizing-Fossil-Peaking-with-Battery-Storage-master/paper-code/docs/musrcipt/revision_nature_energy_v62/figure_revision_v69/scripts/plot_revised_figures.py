"""Render the four revised v69 figures from the archived numerical records."""
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import importlib.util
import inspect
import json
import sys

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parents[1]
REV = WORK.parent
SRC = WORK/'source_data'
OUT = WORK/'figures'
QA = WORK/'qa'
for p in [OUT, QA]:
    p.mkdir(parents=True, exist_ok=True)

MM = 1/25.4
WIDTH_MM = 183.0
INK = '#25343D'
MUTED = '#6D787E'
LIGHT = '#DCE3E5'
BLUE = '#4C91B5'
TEAL = '#398F82'
CORAL = '#D96F55'
GOLD = '#D6A43A'
PURPLE = '#7568AA'
SAGE = '#73A492'
TAUPE = '#A4876B'


def configure():
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 6.5, 'axes.labelsize': 6.5, 'axes.titlesize': 6.8,
        'xtick.labelsize': 6.0, 'ytick.labelsize': 6.0, 'legend.fontsize': 6.0,
        'axes.linewidth': .65, 'xtick.major.width': .6, 'ytick.major.width': .6,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.unicode_minus': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
        'savefig.facecolor': 'white', 'text.color': INK, 'axes.labelcolor': INK,
        'xtick.color': INK, 'ytick.color': INK,
    })


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def label(ax, letter, x=-.12, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8,
            fontweight='bold', ha='left', va='bottom', color=INK)


def export(fig, name, record):
    mpl.rcParams.update({'pdf.fonttype':42, 'svg.fonttype':'none'})
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Legends for the revised panels occupy dedicated space outside data axes.
    legend_checks=[]
    for ax in fig.axes:
        leg=ax.get_legend()
        if leg is not None and getattr(leg, '_require_outside', False):
            box=leg.get_window_extent(renderer)
            clear=not box.overlaps(ax.get_window_extent(renderer))
            legend_checks.append({'axis':ax.get_label(), 'outside_data':clear})
            assert clear, (name,ax.get_label(),'legend intersects data axes')
    fig.savefig(OUT/f'{name}.pdf', bbox_inches='tight', pad_inches=2/72)
    fig.savefig(OUT/f'{name}.svg', bbox_inches='tight', pad_inches=2/72)
    fig.savefig(OUT/f'{name}.png',dpi=300,bbox_inches='tight',pad_inches=2/72)
    fig.savefig(OUT/f'{name}.tiff',dpi=600,bbox_inches='tight',pad_inches=2/72,
                pil_kwargs={'compression':'tiff_lzw'})
    rgb=plt.imread(OUT/f'{name}.png')[...,:3]
    gray=rgb@np.array([.2126,.7152,.0722])
    plt.imsave(OUT/f'{name}_grayscale.png',gray,cmap='gray',vmin=0,vmax=1)
    visible=[a for a in fig.findobj(Text) if a.get_visible() and a.get_text().strip()]
    record.update({'backend':'python-matplotlib','size_inches':list(fig.get_size_inches()),
                   'minimum_text_size_pt':min(a.get_fontsize() for a in visible),
                   'legend_checks':legend_checks,'pdf_text_type':42,'svg_text':'editable',
                   'tiff_dpi':600})
    (QA/f'{name}.json').write_text(json.dumps(record,indent=2))
    (QA/f'{name}_labels.json').write_text(json.dumps([a.get_text() for a in visible],indent=2))
    plt.close(fig)
    print('Rendered',name,flush=True)


def demand_validation():
    old=module('v69_demand_template',REV/'figure_work/plot_v57_demand_model_validation.py')
    old.SOURCE=SRC
    configure()
    fit=pd.read_csv(SRC/'demand_country_year_fit.csv')
    sample=pd.read_csv(SRC/'demand_holdout_sample.csv.gz')
    surface=pd.read_csv(SRC/'demand_month_hour_surface.csv')
    annual=pd.read_csv(SRC/'demand_annual_shape.csv')
    qa=json.loads((SRC/'demand_model_qa.json').read_text())
    assert len(fit)==31 and fit.country.nunique()==16
    assert len(sample)==120000 and len(surface)==288 and len(annual)==76
    fig=plt.figure(figsize=(WIDTH_MM*MM,165*MM))
    a=fig.add_axes([.085,.715,.395,.235],label='a')
    b=fig.add_axes([.650,.715,.315,.235],label='b')
    c1=fig.add_axes([.085,.390,.245,.205],label='c_observed')
    c2=fig.add_axes([.365,.390,.245,.205],label='c_predicted')
    d=fig.add_axes([.765,.390,.210,.205],label='d')
    e=fig.add_axes([.085,.095,.530,.180],label='e')
    f=fig.add_axes([.765,.095,.210,.180],label='f')
    old.plot_a(a,fit)
    a.get_legend().remove()
    leg=a.legend(loc='lower right',bbox_to_anchor=(1,1.015),ncol=2,frameon=False,
                 borderaxespad=0,handlelength=1.2,columnspacing=1.1)
    leg._require_outside=True
    a.set_xlabel('Country temporal-validation fit')
    old.plot_b(b,sample,qa)
    old.plot_d(d,sample)
    for text in d.texts:
        if text.get_text()=='d':text.set_position((-.18,1.14))
    # Paired surfaces retain a common scale and a dedicated colour-bar axis.
    matrices=[surface.pivot(index='month',columns='hour',values=key).sort_index()
              for key in ['observed_load_pu','predicted_load_pu']]
    vmin=min(m.to_numpy().min() for m in matrices)
    vmax=max(m.to_numpy().max() for m in matrices)
    for ax,matrix,title in zip([c1,c2],matrices,['Observed','Predicted']):
        im=ax.imshow(matrix,origin='upper',aspect='auto',cmap='YlGnBu',
                     vmin=vmin,vmax=vmax,interpolation='nearest')
        ax.set_title(title,pad=5)
        ax.set_xticks([0,6,12,18,23],['00','06','12','18','23'])
        ax.set_yticks([0,2,5,8,11],['Jan','Mar','Jun','Sep','Dec'])
        ax.set_xlabel('Hour (UTC)')
        ax.spines[:].set_visible(False)
    c1.set_ylabel('Month');c2.tick_params(labelleft=False)
    label(c1,'c',x=-.20,y=1.14)
    cbax=fig.add_axes([.628,.390,.011,.205])
    cb=fig.colorbar(im,cax=cbax)
    cb.set_label('Normalized load',fontsize=6,labelpad=3)
    cb.ax.tick_params(labelsize=5.8,length=2)
    e.plot(annual.calendar_year,annual.p95_p05_width_pu,color=TEAL,lw=1.1,
           marker='o',ms=1.8,label='95th–5th percentile width')
    e.plot(annual.calendar_year,annual.maximum_daily_mean_pu,color=GOLD,lw=1.1,
           marker='s',ms=1.8,label='Maximum daily mean')
    e.set(xlim=(1949,2026),xticks=[1950,1975,2000,2025],
          xlabel='Calendar year',ylabel='Normalized load-shape metric')
    leg=e.legend(loc='lower left',bbox_to_anchor=(0,1.08),ncol=2,frameon=False,
                 borderaxespad=0,columnspacing=1.1,handlelength=1.6,fontsize=6)
    leg._require_outside=True
    label(e,'e',x=-.12,y=1.28)
    offsets=dict(zip(sorted(fit.country.unique()),np.linspace(-.105,.105,16)))
    paired=0
    for country,g in fit.groupby('country'):
        g=g.sort_values('year');xx=g.year.to_numpy()-2023+offsets[country]
        if len(g)==2:
            f.plot(xx,g.weighted_rmse_pu,color='#B5C0C5',lw=.65,zorder=1)
            paired+=1
        for x,row in zip(xx,g.itertuples()):
            f.scatter(x,row.weighted_rmse_pu,s=17,
                      color=BLUE if row.year==2023 else TEAL,
                      marker='o' if row.year==2023 else 's',
                      edgecolor='white',linewidth=.35,zorder=3)
    assert paired==15
    f.set(xlim=(-.3,1.3),ylim=(.045,.145),xticks=[0,1],xticklabels=['2023','2024'],
          xlabel='Validation year',ylabel='Country-level RMSE (p.u.)')
    f.set_yticks([.05,.08,.11,.14])
    label(f,'f',x=-.20,y=1.28)
    fit.to_csv(SRC/'extended_data_figure1f_yearly_rmse.csv',index=False)
    for ax in [a,b,c1,c2,d,e,f]:
        ax.grid(False)
    export(fig,'extended_data_figure1',{
        'country_year_rows':31,'countries':16,'complete_pairs':15,'unpaired_points_retained':1,
        'hourly_sample_rows':120000,'surface_cells':288,'annual_shape_rows':76,
        'exclusions':[], 'reused_panels':'a–d retain their original statistics; e uses all original annual records',
        'f_definition':'Country-year RMSE from the complete saved fit table; lines connect countries with both years.'})


def input_coverage():
    configure()
    inventory=pd.read_csv(SRC/'zone_inventory.csv')
    weights=pd.read_csv(SRC/'capacity_weights.csv')
    matching=pd.read_csv(SRC/'zone_matching.csv')
    windows=pd.read_csv(SRC/'operating_year_inventory.csv')
    event_days=pd.read_csv(SRC/'operating_year_event_days.csv.gz',index_col=0)
    assert len(inventory)==9348 and len(weights)==33 and len(windows)==75
    assert windows.hours.eq(8760).all() and windows.event_days.eq(92).all()
    assert event_days.shape==(75,365) and event_days.sum(axis=1).eq(92).all()
    assert np.array_equal(event_days.index,windows.window_start_year)
    fig=plt.figure(figsize=(WIDTH_MM*MM,167*MM))
    a=fig.add_axes([.09,.780,.405,.170],label='a')
    b=fig.add_axes([.680,.780,.295,.170],label='b')
    c=fig.add_axes([.09,.365,.530,.285],label='c')
    d=fig.add_axes([.770,.365,.205,.285],label='d')
    e=fig.add_axes([.09,.095,.405,.185],label='e')
    f=fig.add_axes([.675,.095,.300,.210],label='f')
    products=['onshore','offshore','solar','temperature']
    names=['Onshore','Offshore','Solar PV','Temperature']
    colors=[BLUE,PURPLE,GOLD,TEAL]
    for i,color in enumerate(colors):
        a.plot([1950,2025],[i,i],color=color,lw=6,solid_capstyle='butt')
    a.plot([2021.5,2025],[3,3],color=CORAL,lw=6,solid_capstyle='butt')
    a.axvline(2021.5,color=INK,lw=.65,ls='--')
    a.set(xlim=(1948,2027),ylim=(3.65,-.6),yticks=range(4),yticklabels=names,
          xticks=[1950,1975,2000,2025],xlabel='Calendar-year coverage')
    label(a,'a',x=-.14)
    zone_rows=[]
    for i,(product,name,color) in enumerate(zip(products,names,colors)):
        q=inventory.loc[inventory['product'].str.startswith(product)]
        regions=set('|'.join(q.regions).split('|'))
        countries=q.country.nunique()
        zone_rows.append({'product':product,'regions':len(regions),'countries':countries})
        b.hlines(i,0,len(regions),color=color,lw=2.0)
        b.scatter(len(regions),i,color=color,s=25,zorder=3,edgecolor='white',linewidth=.45)
        b.text(len(regions)+4,i,str(len(regions)),ha='left',va='center',fontsize=6.2)
    b.set(xlim=(0,130),ylim=(3.65,-.6),xticks=[0,50,100],yticks=range(4),
          yticklabels=[f'{n} ({r["countries"]} countries)' for n,r in zip(names,zone_rows)],
          xlabel='Represented PECD zones')
    label(b,'b',x=-.31)
    pd.DataFrame(zone_rows).to_csv(SRC/'extended_data_figure2b_spatial_coverage.csv',index=False)
    columns=['onshore_capacity_mw','offshore_capacity_mw','solar_capacity_mw']
    weights['total']=weights[columns].sum(axis=1)
    top=weights.nlargest(11,'total').copy()
    other=weights.loc[~weights.country.isin(top.country),columns].sum()
    data=pd.concat([top,pd.DataFrame([{'country':'Other 22',**other.to_dict()}])],ignore_index=True)
    yy=np.arange(12);left=np.zeros(12)
    for col,color,name in zip(columns,colors[:3],names[:3]):
        values=data[col].to_numpy()/1000
        c.barh(yy,values,left=left,color=color,height=.68,label=name,edgecolor='white',lw=.25)
        left+=values
    c.set(yticks=yy,yticklabels=data.country,xlabel='Fixed ERAA 2030 VRE capacity weight (GW)')
    c.invert_yaxis()
    leg=c.legend(loc='lower left',bbox_to_anchor=(0,1.045),ncol=3,frameon=False,borderaxespad=0)
    leg._require_outside=True
    label(c,'c',x=-.11,y=1.15)
    q=inventory[['product','country','aggregation_method']].drop_duplicates()
    counts=q.groupby(['product','aggregation_method']).size().unstack(fill_value=0)
    left=np.zeros(3)
    for method,color,name in [('matched_zone_weighted',TEAL,'Direct match'),
                              ('matched_zone_weighted_plus_country_residual',BLUE,'Match + balance'),
                              ('country_total_distributed_to_pecd_regions',CORAL,'Country allocation'),
                              ('unweighted_zero_structural_weight','#B9C2C6','Zero capacity')]:
        vals=counts.reindex(products[:3])[method].to_numpy()
        d.barh(np.arange(3),vals,left=left,color=color,height=.56,label=name)
        for i,x,v in zip(range(3),left,vals):
            if v>=3:
                d.text(x+v/2,i,str(v),ha='center',va='center',color='white',fontsize=6,fontweight='bold')
        left+=vals
    assert np.array_equal(left,np.array([33,24,33]))
    counts.reindex(products[:3]).to_csv(SRC/'extended_data_figure2d_matching_counts.csv')
    d.set(yticks=range(3),yticklabels=names[:3],xlabel='Countries',xlim=(0,33),xticks=[0,10,20,30])
    d.invert_yaxis()
    leg=d.legend(loc='lower right',bbox_to_anchor=(1,1.045),ncol=1,frameon=False,
                 borderaxespad=0,fontsize=5.8,handlelength=1.2,labelspacing=.35)
    leg._require_outside=True
    label(d,'d',x=-.25,y=1.15)
    overall=100*matching.matched_weight_share.to_numpy()
    minimum=100*matching.minimum_positive_country_weight_share.to_numpy()
    yy=np.arange(len(matching))
    e.hlines(yy,95,overall,color=mpl.colors.to_rgba(TEAL,.35),lw=4)
    e.scatter(overall,yy,s=23,color=TEAL,edgecolor='white',linewidth=.5,label='All weights')
    e.scatter(minimum,yy,s=18,marker='D',color=CORAL,edgecolor='white',linewidth=.4,label='Minimum country')
    for y,v in zip(yy,overall):
        e.text(v-.08,y,f'{v:.1f}%',ha='right',va='center',fontsize=5.8)
    e.set(yticks=yy,yticklabels=matching['product'].str.replace('temperature_','Temp ',regex=False),
          xlim=(95,100.35),xlabel='ERAA weight represented in PECD zones (%)')
    e.invert_yaxis()
    leg=e.legend(loc='lower left',bbox_to_anchor=(0,1.045),ncol=2,frameon=False,
                 borderaxespad=0,fontsize=5.8,columnspacing=1)
    leg._require_outside=True
    label(e,'e',x=-.14,y=1.19)
    # Every cell is a saved daily event flag in a complete July–June window.
    f.imshow(event_days.to_numpy(),origin='upper',interpolation='nearest',aspect='auto',
             cmap=ListedColormap(['#F1F3F3',BLUE]),vmin=0,vmax=1,
             extent=[0,365,2024.5,1949.5])
    f.set(xticks=[15,107,199,289,350],xticklabels=['Jul','Oct','Jan','Apr','Jun'],
          yticks=[1950,1975,2000,2024],
          xlabel='Operating month (July–June)',ylabel='Operating-year start')
    label(f,'f',x=-.24,y=1.06)
    legend=fig.legend(handles=[Patch(facecolor=BLUE,label='Event day'),
                               Patch(facecolor='#F1F3F3',edgecolor=LIGHT,label='Ordinary day')],
                      loc='lower right',bbox_to_anchor=(.985,.002),ncol=2,frameon=False,
                      fontsize=5.8,columnspacing=.9,handlelength=1.0)
    export(fig,'extended_data_figure2',{
        'inventory_rows_used':len(inventory),'spatial_coverage':zone_rows,'countries_in_capacity_weights':33,
        'matching_country_totals':left.tolist(),
        'capacity_aggregation':'Eleven largest country totals and sum of remaining 22 countries',
        'operating_years':75,'calendar_years':76,'event_day_cells':27375,'event_days_per_year':92,'exclusions':[],
        'f_definition':'All daily seasonal-q75 flags, arranged as 75 complete July–June operating years with 29 February removed.'})


TECHS=['li_lfp','vrfb','fecr_flow','gravity','molten_salt_tes','caes','laes','h2_cavern']
TECH_NAMES=['LFP','VRFB','Fe-Cr flow','Gravity','Molten-salt TES','A-CAES','LAES','Cavern H$_2$']
TECH_COLORS=[BLUE,'#90B8C9','#A9C8D3',GOLD,CORAL,SAGE,TAUPE,'#716A63']
EVIDENCE=['Direct evidence duration','Released configuration','Within-domain component interpolation','H2 structural extrapolation']
EVIDENCE_NAMES=['Direct duration anchor','Released configuration','Component interpolation','H$_2$ extrapolation']


def cost_paths():
    configure()
    data=pd.read_csv(SRC/'cost_paths_53.csv')
    assert len(data)==159 and data.groupby('learning_year').size().eq(53).all()
    assert data.total_capex_eur_kwh.gt(0).all()
    assert data.duration_h.gt(0).all()
    assert not data.duplicated(['tech_id','duration_h','learning_year']).any()
    fig=plt.figure(figsize=(WIDTH_MM*MM,174*MM))
    a=fig.add_axes([.085,.425,.445,.530],label='a')
    b=fig.add_axes([.665,.740,.310,.215],label='b')
    c=fig.add_axes([.665,.425,.310,.230],label='c')
    d=fig.add_axes([.085,.125,.235,.190],label='d')
    e=fig.add_axes([.430,.125,.225,.190],label='e')
    f=fig.add_axes([.775,.125,.200,.190],label='f')
    for tech,name,color in zip(TECHS,TECH_NAMES,TECH_COLORS):
        q=data.loc[data.tech_id.eq(tech)]
        for _,line in q.groupby('duration_h'):
            line=line.sort_values('learning_year')
            a.plot(line.learning_year,line.total_capex_eur_kwh,color=color,lw=.75,alpha=.4,
                   ls=':' if line.structural_extrapolation.any() else '-')
        median=q.groupby('learning_year').total_capex_eur_kwh.median()
        a.plot(median.index,median.values,color=color,lw=1.65,label=name)
    a.set(yscale='log',xticks=[2025,2030,2035],xlabel='Cost year',ylabel='Installed CAPEX (EUR kWh$^{-1}$)')
    handles,names=a.get_legend_handles_labels()
    fig.legend(handles,names,ncol=8,loc='lower center',bbox_to_anchor=(.535,1.005),
               fontsize=5.8,frameon=False,handlelength=1.15,columnspacing=.9)
    label(a,'a',x=-.13,y=1.02)
    d25=data.loc[data.learning_year.eq(2025)].set_index(['tech_id','duration_h'])
    d35=data.loc[data.learning_year.eq(2035)].set_index(['tech_id','duration_h'])
    assert set(d25.index)==set(d35.index)
    d35=d35.reindex(d25.index)
    for tech,name,color in zip(TECHS,TECH_NAMES,TECH_COLORS):
        use=d25.index.get_level_values('tech_id')==tech
        b.scatter(d25.loc[use,'total_capex_eur_kwh'],d35.loc[use,'total_capex_eur_kwh'],
                  color=color,s=15,edgecolor='white',linewidth=.35)
    low=min(d25.total_capex_eur_kwh.min(),d35.total_capex_eur_kwh.min())
    high=max(d25.total_capex_eur_kwh.max(),d35.total_capex_eur_kwh.max())
    b.plot([low,high],[low,high],ls='--',color=INK,lw=.7)
    b.set(xscale='log',yscale='log',xlabel='2025 CAPEX (EUR kWh$^{-1}$)',ylabel='2035 CAPEX (EUR kWh$^{-1}$)')
    label(b,'b',x=-.18)
    reduction=100*(1-d35.total_capex_eur_kwh/d25.total_capex_eur_kwh)
    for i,(tech,color) in enumerate(zip(TECHS,TECH_COLORS)):
        values=reduction[reduction.index.get_level_values('tech_id')==tech].to_numpy()
        lo,med,hi=np.quantile(values,[.25,.5,.75])
        c.plot([lo,hi],[i,i],color=color,lw=2.7)
        c.scatter(med,i,color=color,s=24,edgecolor='white',linewidth=.4,zorder=3)
    c.set(yticks=range(8),yticklabels=TECH_NAMES,xlabel='2025–2035 CAPEX reduction (%)')
    c.invert_yaxis();label(c,'c',x=-.18)
    for tech,color in zip(TECHS,TECH_COLORS):
        q=d35.xs(tech).sort_index()
        d.plot(q.index,q.total_capex_eur_kwh,color=color,lw=1.0,marker='o',ms=2.3)
    d.set(xscale='log',yscale='log',xlabel='Discharge duration (h)',ylabel='2035 CAPEX (EUR kWh$^{-1}$)')
    label(d,'d',x=-.23)
    counts=d25.reset_index().groupby(['tech_id','evidence_class']).size().unstack(fill_value=0).reindex(TECHS).fillna(0)
    assert int(counts.to_numpy().sum())==53
    left=np.zeros(8)
    for key,name,color in zip(EVIDENCE,EVIDENCE_NAMES,[BLUE,SAGE,GOLD,CORAL]):
        values=counts[key].to_numpy()
        e.barh(range(8),values,left=left,color=color,height=.58,label=name)
        left+=values
    e.set(yticks=range(8),yticklabels=TECH_NAMES,xlabel='Configurations',xlim=(0,17),xticks=[0,4,8,12,16])
    e.invert_yaxis();label(e,'e',x=-.24)
    counts.to_csv(SRC/'extended_data_figure3e_source_counts.csv')
    for i,(tech,color) in enumerate(zip(TECHS,TECH_COLORS)):
        vals=d35.xs(tech).total_capex_eur_kwh.to_numpy()
        jitter=.13*np.sin(np.arange(len(vals))*2.3999632297)
        f.scatter(vals,i+jitter,s=10,color=color,alpha=.55,edgecolor='none')
        lo,med,hi=np.quantile(vals,[.25,.5,.75])
        f.plot([lo,hi],[i,i],color=color,lw=2.4)
        f.scatter(med,i,s=22,color=INK,edgecolor='white',linewidth=.35,zorder=3)
    f.set(xscale='log',yticks=range(8),yticklabels=TECH_NAMES,xlabel='2035 CAPEX (EUR kWh$^{-1}$)')
    f.invert_yaxis();label(f,'f',x=-.27)
    handles,labels=e.get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.54,.011),
               ncol=4,frameon=False,fontsize=5.8,handlelength=1.25,columnspacing=1.15,
               title='Source class (e)',title_fontsize=6.0)
    export(fig,'extended_data_figure3',{
        'cost_records':159,'configurations':53,'years':[2025,2030,2035],
        'source_class_totals':{k:int(counts[k].sum()) for k in EVIDENCE},'exclusions':[],
        'legend_placement':'Dedicated margin below all data panels',
        'aggregation':'Technology medians and interquartile intervals; all individual configuration records retained.'})


def main_figure4():
    old=module('v69_main4_template',REV/'revision_issue1_v67/scripts/plot_v67_figures.py')
    load=old.module
    def patched_load(name,path):
        obj=load(name,path)
        if name=='v67_fig4':
            code=inspect.getsource(obj.plot_integrated_response)
            original='ax.inset_axes([0.16, -0.21, 0.68, 0.065])'
            assert code.count(original)==1
            code=code.replace(original,'ax.inset_axes([0.0, -0.21, 1.0, 0.065])')
            exec(compile(code,str(__file__),'exec'),obj.__dict__)
        return obj
    old.module=patched_load
    old.OUT=OUT
    def save(fig,name):
        fig.canvas.draw()
        candidates=[ax for ax in fig.axes if ax.images and np.shape(ax.images[0].get_array())==(3,2)]
        assert len(candidates)==1
        ax=candidates[0];assert len(ax.child_axes)==1
        cax=ax.child_axes[0]
        box=ax.get_position();cb=cax.get_position()
        error=max(abs(box.x0-cb.x0),abs(box.x1-cb.x1))
        assert error<1e-12
        export(fig,name,{'panel_f_colourbar_left':cb.x0,'panel_f_colourbar_right':cb.x1,
                         'panel_f_left':box.x0,'panel_f_right':box.x1,'alignment_error':error,
                         'panel_f_array':np.asarray(ax.images[0].get_array()).tolist(),
                         'change':'Only panel-f colour-bar origin and width changed in the original rendering function.'})
    old.export_files=save
    old.fig4()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure',choices=['all','ed1','ed2','ed3','main4'],default='all')
    args=parser.parse_args()
    for name,fn in [('ed1',demand_validation),('ed2',input_coverage),('ed3',cost_paths),('main4',main_figure4)]:
        if args.figure in ['all',name]:fn()


if __name__=='__main__':
    main()
