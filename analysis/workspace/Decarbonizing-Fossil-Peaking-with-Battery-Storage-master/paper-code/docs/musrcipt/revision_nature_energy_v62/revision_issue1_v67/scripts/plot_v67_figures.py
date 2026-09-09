"""EFC primary figures using existing publication layouts and frozen source data.

All quantitative records are retained. Intervals use the original block units.
Alternative system-cost figures remain unchanged in the supplement.
"""
from __future__ import annotations
from pathlib import Path
import importlib.util,sys,json,inspect
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pandas as pd

WORK=Path(__file__).resolve().parents[1];ROOT=WORK.parent
SRC=WORK/'source_data';NAT=WORK/'natural_results';OUT=WORK/'figures'
MM=1/25.4

def export_files(fig,name):
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight',pad_inches=2/72)
    fig.savefig(OUT/f'{name}.svg',bbox_inches='tight',pad_inches=2/72)
    fig.savefig(OUT/f'{name}.png',bbox_inches='tight',pad_inches=2/72,dpi=600)

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def save(name):
    def export(fig,stem=None):
        # Labels are changed with the response data, not through raster editing.
        for artist in fig.findobj(Text):
            text=artist.get_text()
            for old,new in [('system-cost','EFC service-value'),('System-cost','EFC service-value'),('system cost','EFC service value'),('Objective-aligned','EFC-valued'),('objective-aligned','EFC-valued'),('peak service','EFC service'),('3/65 reinforce','6/65 reinforce'),('3/65 erase','6/65 erase')]:
                text=text.replace(old,new)
            artist.set_text(text)
        matplotlib.rcParams.update({'svg.fonttype':'none','pdf.fonttype':42})
        if name=='supp_figure1':
            counts=pd.read_csv(SRC/'cost_threshold_parity_counts_efc_natural.csv')
            maximum=float((100*counts.natural_pair_hits/counts.natural_pairs).max())
            limit=float(fig.axes[1].get_ylim()[1])
            assert maximum<limit,(maximum,limit)
            (WORK/'qa/figure1_sensitivity_coverage.json').write_text(json.dumps({'maximum_natural_pair_share_percent':maximum,'axis_upper_percent':limit,'all_points_in_range':True},indent=2))
        export_files(fig,name)
        labels=[x.get_text() for x in fig.findobj(Text) if x.get_text()]
        (OUT/f'{name}_labels.json').write_text(json.dumps(labels,ensure_ascii=False,indent=2))
        print('Rendered',name,flush=True)
    return export

def fig3():
    m=module('v67_fig3',ROOT/'figure_work/figure3_relayout_v62/plot_preview_figure3_recovery_relayout.py')
    source=inspect.getsource(m.plot_value_decomposition).replace('            122,','            155,').replace('ax.set_xlim(-150, 128)','ax.set_xlim(-150, 162)')
    exec(compile(source,str(__file__),'exec'),m.__dict__)
    m.SOURCE=SRC;m.PROFILE_SOURCE=SRC;m.OUT=OUT;m.save_bundle=save('main_figure3');m.main()

def fig4():
    m=module('v67_fig4',ROOT/'figure_work/figure4_relayout_v62/plot_preview_figure4_recharge.py')
    # Keep the refill colour-bar labels clear of the adjacent window panel.
    source=inspect.getsource(m.draw_refill_phase_compact)
    source=source.replace('ax.inset_axes([1.008, 0.0, 0.019, 1.0])','ax.inset_axes([0.992, 0.0, 0.019, 1.0])')
    source=source.replace('colorbar.ax.yaxis.set_label_position("left")','colorbar.ax.yaxis.set_label_position("left")\n    colorbar.ax.yaxis.set_ticks_position("left")')
    exec(compile(source,str(__file__),'exec'),m.__dict__)
    source=inspect.getsource(m.draw_window_transition).replace('bbox_to_anchor=(0.5, -0.46)','bbox_to_anchor=(0.5, -0.28)')
    exec(compile(source,str(__file__),'exec'),m.__dict__)
    def load_data():
        summary=pd.read_csv(SRC/'figure4_recharge_intervention_summary.csv')
        recovery=pd.read_csv(SRC/'figure3_recovery_window_records.csv')
        records=pd.read_csv(SRC/'recharge_efc_records.csv')
        records['observation_id']=records.system_id.astype(str)+'|'+records.observation_year.astype(str)
        for regime,n in {'Europe':132,'China':26,'Japan':27}.items():
            group=records[records.regime.eq(regime)]
            assert group.groupby('recharge_multiplier').observation_id.nunique().eq(n).all()
            assert not group.duplicated(['observation_id','recharge_multiplier']).any()
        assert len(recovery)==370 and len(records)==555
        return summary,recovery,records
    m.load_data=load_data;m.OUT=OUT;m.save_bundle=save('main_figure4');m.main()

def fig5():
    m=module('v67_fig5',ROOT/'figure_work/layout_preview_v61/plot_v62_figure5.py')
    m.base.SOURCE=SRC;m.layout.phase.SOURCE=WORK/'surface'
    m.OUT=OUT;m.save_bundle=save('main_figure5');m.main()

def fig6():
    m=module('v67_fig6',ROOT/'figure_work/layout_preview_v61/plot_v62_figure6.py')
    m.SOURCE=SRC/'fig6';m.base.RESULTS=NAT;m.transition.RESULTS=NAT
    for key in ['VALUES_PATH','PAIRS_PATH','DECADES_PATH','PREPARED_PATH','SUMMARY_PATH']:
        setattr(m.transition,key,NAT/getattr(m.transition,key).name)
    m.OUT=OUT;m.save_bundle=save('main_figure6');m.main()

def supp1():
    m=module('v67_supp1',ROOT/'figure_work/supplementary_figures_v62/plot_v62_supplementary_robustness.py')
    cost=pd.read_csv(m.SOURCE/'discount_duration_cost_scale.csv')
    counts=pd.read_csv(ROOT/'robustness_evidence/cost_threshold_parity_counts.csv')
    ensemble=pd.read_csv(ROOT/'cost_uncertainty_v62/results/cost_comparator_scenario_ensemble.csv')
    pairs=pd.read_csv(NAT/'pecd_v42_75_window_all_pairs.csv.gz')
    decades=pd.read_csv(NAT/'pecd_v42_65_matched_decade_transitions.csv')
    for i,row in counts.iterrows():
        t=row.cost_threshold_eur_kw_year
        counts.loc[i,'natural_pair_hits']=pairs.absolute_weather_value_change_eur_kw_year.ge(t).sum()
        counts.loc[i,'adverse_decade_hits']=decades.weather_value_change_eur_kw_year.le(-t).sum()
        counts.loc[i,'beneficial_decade_hits']=decades.weather_value_change_eur_kw_year.ge(t).sum()
    counts.to_csv(SRC/'cost_threshold_parity_counts_efc_natural.csv',index=False)
    assert len(cost)==12 and len(counts)==4 and len(ensemble)==540 and len(pairs)==2775
    assert counts.loc[np.isclose(counts.discount_rate,.036),'natural_pair_hits'].iloc[0]==496
    # The EFC response exceeds the old objective-ledger plotting limit at 2%.
    upper=10*np.ceil((100*(counts.natural_pair_hits/counts.natural_pairs).max()+5)/10)
    source=inspect.getsource(m.plot_discount_sensitivity)
    assert source.count('ylim=(-1, 40)')==1
    source=source.replace('ylim=(-1, 40)',f'ylim=(-1, {upper})')
    exec(compile(source,str(__file__),'exec'),m.__dict__)
    m.OUT=OUT;m.save_bundle=save('supp_figure1');m.configure();m.plot_discount_sensitivity(cost,counts,ensemble,pairs)

def supp7():
    models=pd.read_csv(SRC/'model_comparison_both_ledgers_and_difference.csv')
    coefs=pd.read_csv(SRC/'coefficient_comparison_both_ledgers_and_difference.csv')
    predictions=pd.read_csv(SRC/'loco_predictions.csv')
    matplotlib.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
       'font.size':6.5,'axes.labelsize':6.5,'axes.titlesize':7,'xtick.labelsize':6,'ytick.labelsize':6,
       'legend.fontsize':6,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    width_mm=183.0
    fig,axes=plt.subplots(3,2,figsize=(width_mm/25.4,width_mm/25.4),layout='constrained')
    names=['longest_event','recovery_window','joint_geometry','joint_controlled']
    labels=['Longest event','Recovery window','Joint geometry','+ task / recharge']
    colours=['#356F6B','#8A4F49']
    for col,(response,title,colour) in enumerate(zip(['efc_value','system_cost_value'],['Primary: EFC service value','Supplementary: system-cost value'],colours)):
        d=models[models.response.eq(response)].set_index('model').loc[names]
        ax=axes[0,col];y=np.arange(4)
        ax.scatter(d.within_r2,y-.1,marker='s',s=17,color='#777777',label='Within-country fit',zorder=4)
        ax.errorbar(d.loco_r2,y+.1,xerr=np.vstack([d.loco_r2-d.loco_r2_ci95_low,d.loco_r2_ci95_high-d.loco_r2]),fmt='o',ms=3,color=colour,lw=.8,capsize=2,label='Leave-one-country-out')
        ax.set(yticks=y,yticklabels=labels,xlabel='$R^2$',title=title,xlim=(-.65,.25));ax.invert_yaxis();ax.axvline(0,color='#aaaaaa',lw=.6)
        if col==0:ax.legend(loc='lower left')
        ax=axes[1,col]
        c=coefs[coefs.response.eq(response)&coefs.predictor.eq('natural_zero_recovery_window_share')].set_index('model').loc[names[1:]]
        ax.errorbar(c.reported_coefficient,np.arange(3),xerr=np.vstack([c.reported_coefficient-c.reported_ci95_low,c.reported_ci95_high-c.reported_coefficient]),fmt='o',ms=4,color=colour,lw=.9,capsize=2)
        ax.set(yticks=np.arange(3),yticklabels=labels[1:],xlabel='Value change per +10 percentage points\nzero-recovery share (EUR kW$^{-1}$ yr$^{-1}$)',xlim=(-2.5,5.5));ax.invert_yaxis();ax.axvline(0,color='#aaaaaa',lw=.7)
        ax=axes[2,col];p=predictions[predictions.response.eq(response)&predictions.model_id.eq('joint_geometry')]
        assert len(p)==132
        x=p.observed_country_centred_value_eur_kw_year;y=p.predicted_country_centred_value_eur_kw_year
        ax.scatter(x,y,s=9,color=colour,alpha=.65,edgecolor='white',linewidth=.2)
        lim=max(abs(x).max(),abs(y).max())*1.07
        ax.plot([-lim,lim],[-lim,lim],color='#777777',lw=.7,ls='--');ax.set(xlim=(-lim,lim),ylim=(-lim,lim),xlabel='Observed within-country deviation\n(EUR kW$^{-1}$ yr$^{-1}$)',ylabel='Held-out prediction\n(EUR kW$^{-1}$ yr$^{-1}$)')
        v=d.loc['joint_geometry'];ax.text(.03,.97,f'Joint geometry | n = 132\nLOCO $R^2$ = {v.loco_r2:.3f}',transform=ax.transAxes,va='top',fontsize=6)
    for i,ax in enumerate(axes.flat):
        ax.text(-.15,1.07,chr(97+i),transform=ax.transAxes,fontweight='bold',fontsize=8)
    # Do not apply primary-only label substitutions to the two-ledger comparison.
    export_files(fig,'supp_figure7')
    plt.close(fig);print('Rendered supp_figure7',flush=True)

if __name__=='__main__':
    selected=sys.argv[1:] or ['3','4','5','6','s1','s7']
    for name in selected:{'3':fig3,'4':fig4,'5':fig5,'6':fig6,'s1':supp1,'s7':supp7}[name]()
