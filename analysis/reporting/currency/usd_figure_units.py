"""Physical currency units for the frozen plotting coordinates.

The exact relation x_USD = 1.13 x_EUR applies to observations, interval endpoints,
axes and annotations. Plotting an unchanged normalized position against converted
axis ticks is the same affine unit transformation as rescaling each artist and
axis limits. The USD CSVs provide the corresponding unrounded plotted values.
"""
from config import *
import re,json
import numpy as np
import pandas as pd
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter,FixedLocator,NullFormatter,MaxNLocator

AXES={
 'main_figure3':{'x':[12,13,14],'y':[15]},
 'main_figure4':{'y':[4,5,6]},
 'main_figure5':{'y':[0,1,2],'x':[18]},
 'main_figure6':{'y':[0,3,9,10,12],'x':[11]},
 'supp_figure1':{'y':[0],'x':[3,4]},
 'supp_figure2':{'y':[0,1,2]},
 'supp_figure3':{'x':[0,1,2,3]},
 'supp_figure7':{'x':[1,2],'y':[2]},
 'extended_data_figure3':{'y':[0,1,3],'x':[1,5]},
 'extended_data_figure5':{'y':[3]},
 'extended_data_figure6':{'y':[0],'x':[2,3]},
}

def all_axes(fig):
    out=[]
    def add(a):
        if a in out:return
        out.append(a)
        for b in a.child_axes:add(b)
    for a in fig.axes:add(a)
    return out

def money(v,signed=False):
    v=float(v)*USD_PER_EUR
    if abs(v)<.005:v=0.
    return f'{v:+.2f}' if signed and v else f'{v:.2f}'

def axis_format(axis,scale):
    lo,hi=sorted(axis.get_view_interval()*USD_PER_EUR)
    def fmt(x,pos):
        v=x*USD_PER_EUR
        if abs(v)<max(1e-30,1e-14*max(abs(lo),abs(hi))):return '0'
        if abs(v)<1e-3 or abs(v)>=1e5:
            exp=int(np.floor(np.log10(abs(v))));mant=v/10**exp
            if np.isclose(mant,1):return r'$10^{'+str(exp)+'}$'
            return r'$'+f'{mant:.3g}'+r'\times10^{'+str(exp)+'}$'
        return f'{v:.3g}'
    # For log axes use simple USD decade ticks at the corresponding exact EUR
    # locations. Linear and symlog axes retain their physically identical ticks.
    if scale=='symlog':
        # Sparse dollar-decade ticks avoid a long factor on every residual tick.
        threshold=axis.get_transform().linthresh*USD_PER_EUR
        exponents=np.arange(np.ceil(np.log10(threshold)),np.floor(np.log10(max(abs(lo),abs(hi))))+1,2)
        ticks=np.r_[0.,10.**exponents]
        axis.set_major_locator(FixedLocator(ticks/USD_PER_EUR));axis.set_minor_formatter(NullFormatter())
    elif scale=='log':
        if lo>0:
            power=np.arange(np.floor(np.log10(lo))-1,np.ceil(np.log10(hi))+2)
            axis.set_major_locator(FixedLocator(10.**power/USD_PER_EUR))
            axis.set_minor_locator(FixedLocator(np.concatenate([np.arange(2,10)*10.**p/USD_PER_EUR for p in power])))
            axis.set_minor_formatter(NullFormatter())
    elif scale=='linear':
        n=len([v for v in axis.get_majorticklocs() if lo<=v*USD_PER_EUR<=hi])
        ticks=MaxNLocator(nbins=max(3,min(5,n-1)),steps=[1,2,2.5,5,10]).tick_values(lo,hi)
        axis.set_major_locator(FixedLocator(ticks/USD_PER_EUR))
    axis.set_major_formatter(FuncFormatter(fmt))
    axis.offsetText.set_visible(False)

def convert_figure(fig,name):
    axes=all_axes(fig);audit=[]
    original_positions=[list(a.get_position().bounds) for a in axes]
    # Freeze the established layout before changing tick widths.
    fig.canvas.draw()
    if fig.get_layout_engine() is not None:fig.set_layout_engine(None)
    for direction,indices in AXES[name].items():
        for i in indices:
            ax=axes[i];axis=getattr(ax,direction+'axis');lim=getattr(ax,'get_'+direction+'lim')()
            axis_format(axis,getattr(ax,'get_'+direction+'scale')())
            audit.append({'axis':i,'direction':direction,'eur_limits':list(lim),'usd_limits':[x*USD_PER_EUR for x in lim]})
    natural=json.loads((BASE/'source_data/natural_years/pecd_v42_75y_parity_summary.json').read_text())
    threshold=natural['cost_improvement_2025_2035_eur_kw_year']
    span=natural['natural_q95_q05_span_eur_kw_year'];full=natural['natural_full_span_eur_kw_year']
    records=pd.read_csv(BASE/'source_data/main_efc/figure3_24h_value_records.csv')
    meds={g:float(q.signed_sequencing_effect_eur_kw_year.median()) for g,q in records.groupby('regime')}
    changes=[]
    def set_text(t,new):
        old=t.get_text()
        if old!=new:changes.append({'before':old,'after':new});t.set_text(new)
    for t in fig.findobj(Text):
        s=t.get_text().replace('EUR','USD')
        s=s.replace('zero-hour gate:\n','zero-hour gate\n').replace('central: ','central ').replace('solid / circle: dispersed    dashed / square: clustered','solid / circle  dispersed    dashed / square  clustered')
        if name=='main_figure3':
            for reg,axid in zip(['Europe','China','Japan'],[12,13,14]):
                if t in axes[axid].texts and 'median ' in s:s=re.sub(r'median [+-]?\d+\.\d+',f'median {money(meds[reg],signed=True)}',s)
            if t in axes[19].texts and re.fullmatch(r'[+-]?\d+\.\d+',s):
                for reg in meds:
                    if s==f'{meds[reg]:+.2f}' or s==f'{meds[reg]:.2f}':s=money(meds[reg],signed=True);break
            if 'cost scale\n' in s:s=s.replace('−5.30','−'+money(threshold))
        elif name=='main_figure5' and s=='cost scale\n5.30':s='cost scale\n'+money(threshold)
        elif name=='main_figure6':
            if s.startswith('most adverse matched decade'):
                s='most adverse matched decade  '+money(natural['worst_matched_decade']['weather_change_eur_kw_year'])
            elif s=='learning +5.30':s='learning +'+money(threshold)
            elif ' |  value ' in s:
                y=s.split('  |')[0]
                vals=pd.read_csv(BASE/'source_data/natural_years/pecd_v42_75_window_relative_values_2025.csv')
                window_col=next(c for c in vals if c in ['operating_year','window_label','window','weather_window'])
                val=float(vals.loc[vals[window_col].eq(y),'value_2025_eur_kw_year'].iloc[0]);s=y+'  |  value '+money(val)
            elif '2035 net -5.97' in s:s=s.replace('-5.97',money(natural['worst_matched_decade']['net_change_after_learning_eur_kw_year']))
            elif s=='cost scale 5.30':s='cost scale '+money(threshold)
            elif t in axes[11].texts and s=='15.06':s=money(full)
            elif t in axes[11].texts and s=='7.64':s=money(span)
        elif name=='supp_figure1':
            s=s.replace('central 5.30','central '+money(threshold)).replace('natural span 7.64','natural span '+money(span))
        set_text(t,s)
    if name=='main_figure4':
        rec=pd.read_csv(REV/'revision_issue1_v67/source_data/recharge_efc_records.csv')
        for reg,idx in zip(['Europe','China','Japan'],[4,5,6]):
            q=rec[rec.regime.eq(reg)].pivot(index=['system_id','observation_year'],columns='recharge_multiplier',values='chronology_effect')
            delta=q[4]-q[1];med,lo,hi=np.quantile(delta,[.5,.05,.95])
            line=next(t for t in axes[idx].texts if '4× median' in t.get_text())
            set_text(line,f'4× median {money(med,signed=True)}  [5-95% {money(lo,signed=True)}, {money(hi,signed=True)}]')
        numeric=[t for t in axes[7].texts if re.match(r'^\[?[+\-]?\d',t.get_text())]
        assert len(numeric)==12
        for j,reg in enumerate(['Europe','China','Japan']):
            q=rec[rec.regime.eq(reg)].pivot(index=['system_id','observation_year'],columns='recharge_multiplier',values='chronology_effect')
            for k,arr in enumerate([q[1],q[4]-q[1]]):
                med,lo,hi=np.quantile(arr,[.5,.05,.95]);i=j*4+k*2
                set_text(numeric[i],money(med,signed=True));set_text(numeric[i+1],f'[{money(lo,signed=True)}, {money(hi,signed=True)}]')
        # The panel-f colour map divides both the money data and benchmark by
        # the same factor. Its dimensionless colour range remains unchanged.
        box=axes[7].get_position();cb=axes[8].get_position()
        assert max(abs(box.x0-cb.x0),abs(box.x1-cb.x1))<1e-12
    if name=='extended_data_figure6':
        axes[2].set_xlabel('Clustered - dispersed value (USD kW$^{-1}$ yr$^{-1}$)',fontsize=5.0)
        axes[3].set_xlabel('Median paired effect\n(USD kW$^{-1}$ yr$^{-1}$)',fontsize=5.5)
    if name=='main_figure6':axes[11].set_xlabel('natural value span\n(USD kW$^{-1}$ yr$^{-1}$)')
    if name=='extended_data_figure5':
        # The legacy fixed-canvas export clips this label outside the axes.
        # Retain the same visible panel content in the tightly cropped export.
        for a in axes:
            for t in a.texts:
                if t.get_text()=='n.a.' and not min(a.get_ylim())<=t.get_position()[1]<=max(a.get_ylim()):t.set_clip_on(True)
    fig.canvas.draw()
    assert not any('EUR' in t.get_text() or ':' in t.get_text() for t in fig.findobj(Text))
    (WORK/'qa'/f'{name}_usd_conversion.json').write_text(json.dumps({'figure':name,'factor':USD_PER_EUR,'monetary_axes':audit,'annotation_changes':changes,'source_positions':original_positions,'final_positions':[list(a.get_position().bounds) for a in axes]},indent=2))
