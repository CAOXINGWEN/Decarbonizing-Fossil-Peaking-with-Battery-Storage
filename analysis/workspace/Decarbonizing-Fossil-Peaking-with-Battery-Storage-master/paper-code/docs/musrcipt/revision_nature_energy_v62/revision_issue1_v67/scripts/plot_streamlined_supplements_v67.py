"""EFC-only supplementary displays from frozen results; comparisons are in SI Table 7."""
from pathlib import Path
import importlib.util
import inspect
import json
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pandas as pd

WORK=Path(__file__).resolve().parents[1]
ROOT=WORK.parent
SRC=WORK/"source_data"
OUT=WORK/"figures"
AUDIT=WORK/"streamline_efc"
AUDIT.mkdir(parents=True, exist_ok=True)
OUT.mkdir(exist_ok=True)

def configure():
    matplotlib.rcParams.update({
        "font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
        "font.size":6.5,"axes.labelsize":6.5,"axes.titlesize":7,
        "xtick.labelsize":6,"ytick.labelsize":6,"legend.fontsize":6,
        "axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False,
        "svg.fonttype":"none","pdf.fonttype":42,
        "figure.facecolor":"white","savefig.facecolor":"white"})

def export(fig, name):
    fig.canvas.draw()
    fig.savefig(OUT/f"{name}.pdf",bbox_inches="tight",pad_inches=2/72)
    fig.savefig(OUT/f"{name}.svg",bbox_inches="tight",pad_inches=2/72)
    fig.savefig(OUT/f"{name}.png",bbox_inches="tight",pad_inches=2/72,dpi=600)
    fig.savefig(OUT/f"{name}.tiff",bbox_inches="tight",pad_inches=2/72,dpi=600,
                pil_kwargs={"compression":"tiff_lzw"})
    labels=[x.get_text() for x in fig.findobj(Text) if x.get_text()]
    (OUT/f"{name}_labels.json").write_text(json.dumps(labels,ensure_ascii=False,indent=2))
    assert not any("system-cost" in t.lower() or "system cost" in t.lower() for t in labels)
    print("Rendered",name,flush=True)

def supplement1():
    path=ROOT/"figure_work/supplementary_figures_v62/plot_v62_supplementary_robustness.py"
    spec=importlib.util.spec_from_file_location("sensitivity_efc",path)
    m=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=m
    spec.loader.exec_module(m)
    # Structural adaptation: retain the original statistics and five EFC/cost panels.
    source=inspect.getsource(m.plot_discount_sensitivity)
    old='fig, axes = plt.subplots(3, 2, figsize=(WIDTH_MM * MM, 166 * MM), constrained_layout=True)\n    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes.flat'
    new='fig = plt.figure(figsize=(WIDTH_MM * MM, 158 * MM), constrained_layout=True)\n    grid = fig.add_gridspec(3, 2, height_ratios=[1, .88, 1.15])\n    ax_a = fig.add_subplot(grid[0, 0])\n    ax_b = fig.add_subplot(grid[0, 1])\n    ax_c = fig.add_subplot(grid[1, :])\n    ax_e = fig.add_subplot(grid[2, 0])\n    ax_f = fig.add_subplot(grid[2, 1])'
    assert source.count(old)==1
    source=source.replace(old,new)
    old='        "System cost": {"Europe": "objective_europe_hits", "China": "objective_china_hits", "Japan": "objective_japan_hits"},\n'
    assert source.count(old)==1
    source=source.replace(old,"")
    source=source.replace('zip((ax_c, ax_d), ledger_columns.items(), ("c", "d"))','zip((ax_c,), ledger_columns.items(), ("c",))')
    source=source.replace('panel_label(ax_e, "e")','panel_label(ax_e, "d")').replace('panel_label(ax_f, "f")','panel_label(ax_f, "e")')
    source=source.replace('panel_label(ax, label)', 'panel_label(ax, label, x=-0.059)')
    source=source.replace('f"{ledger} ledger"', '"EFC service value"')
    counts=pd.read_csv(SRC/"cost_threshold_parity_counts_efc_natural.csv")
    cost=pd.read_csv(m.SOURCE/"discount_duration_cost_scale.csv")
    ensemble=pd.read_csv(ROOT/"cost_uncertainty_v62/results/cost_comparator_scenario_ensemble.csv")
    pairs=pd.read_csv(WORK/"natural_results/pecd_v42_75_window_all_pairs.csv.gz")
    decades=pd.read_csv(WORK/"natural_results/pecd_v42_65_matched_decade_transitions.csv")
    assert len(counts)==4 and len(cost)==12 and len(ensemble)==540 and len(pairs)==2775 and len(decades)==65
    for row in counts.itertuples():
        assert row.natural_pair_hits==pairs.absolute_weather_value_change_eur_kw_year.ge(row.cost_threshold_eur_kw_year).sum()
        assert row.adverse_decade_hits==decades.weather_value_change_eur_kw_year.le(-row.cost_threshold_eur_kw_year).sum()
    maximum=float((100*counts.natural_pair_hits/counts.natural_pairs).max())
    upper=10*np.ceil((maximum+5)/10)
    assert source.count('ylim=(-1, 40)')==1
    source=source.replace('ylim=(-1, 40)',f'ylim=(-1, {upper})')
    exec(compile(source,str(__file__),"exec"),m.__dict__)
    m.configure()
    def save_bundle(fig,stem):
        assert len(fig.axes)==5 and fig.axes[1].get_ylim()[1]>maximum
        export(fig,"supp_figure1")
    m.save_bundle=save_bundle
    m.plot_discount_sensitivity(cost,counts,ensemble,pairs)
    return {"panels":5,"natural_pairs":2775,"matched_decades":65,
            "discount_rates":4,"cost_cells":[108,432],"controlled_observations":[132,26,27],
            "largest_natural_pair_share_percent":maximum,"axis_upper_percent":upper,
            "removed_display":"system-cost discount-rate panel; source records retained"}

def supplement7():
    models=pd.read_csv(SRC/"model_comparison_both_ledgers_and_difference.csv")
    coefs=pd.read_csv(SRC/"coefficient_comparison_both_ledgers_and_difference.csv")
    predictions=pd.read_csv(SRC/"loco_predictions.csv")
    names=["longest_event","recovery_window","joint_geometry","joint_controlled"]
    labels=["Longest event","Recovery window","Joint geometry","+ task / recharge"]
    # User-authorized display selection: EFC is plotted; the system-cost response is tabulated.
    d=models[models.response.eq("efc_value")].set_index("model").loc[names]
    c=coefs[coefs.response.eq("efc_value") & coefs.predictor.eq("natural_zero_recovery_window_share")].set_index("model").loc[names[1:]]
    p=predictions[predictions.response.eq("efc_value") & predictions.model_id.eq("joint_geometry")]
    assert len(d)==4 and len(c)==3 and len(p)==132
    configure()
    fig=plt.figure(figsize=(183/25.4,94/25.4),layout="constrained")
    grid=fig.add_gridspec(2,2,width_ratios=[1,1.12],hspace=.15,wspace=.05)
    a=fig.add_subplot(grid[0,0]);b=fig.add_subplot(grid[1,0]);ax=fig.add_subplot(grid[:,1])
    colour="#356F6B"
    y=np.arange(4)
    a.scatter(d.within_r2,y-.11,marker="s",s=18,color="#777777",label="Within-country fit",zorder=4)
    a.errorbar(d.loco_r2,y+.11,xerr=np.vstack([d.loco_r2-d.loco_r2_ci95_low,d.loco_r2_ci95_high-d.loco_r2]),
               fmt="o",ms=3.3,color=colour,lw=.8,capsize=2,label="Leave-one-country-out")
    a.set(yticks=y,yticklabels=labels,xlabel="$R^2$",xlim=(-.055,.23),ylim=(3.6,-.6))
    a.axvline(0,color="#aaaaaa",lw=.6)
    a.legend(loc="upper center",bbox_to_anchor=(.48,-.26),ncol=1,fontsize=5.7)
    b.errorbar(c.reported_coefficient,np.arange(3),xerr=np.vstack([c.reported_coefficient-c.reported_ci95_low,c.reported_ci95_high-c.reported_coefficient]),
               fmt="o",ms=4,color=colour,lw=.9,capsize=2)
    b.set(yticks=np.arange(3),yticklabels=labels[1:],xlabel="EFC value change per +10 percentage points\nzero-recovery share (EUR kW$^{-1}$ yr$^{-1}$)",xlim=(-2.3,.15),ylim=(2.5,-.5))
    b.axvline(0,color="#aaaaaa",lw=.7)
    x=p.observed_country_centred_value_eur_kw_year;y=p.predicted_country_centred_value_eur_kw_year
    ax.scatter(x,y,s=12,color=colour,alpha=.7,edgecolor="white",linewidth=.25)
    lim=float(max(abs(x).max(),abs(y).max())*1.07)
    ax.plot([-lim,lim],[-lim,lim],color="#777777",lw=.7,ls="--")
    ax.set(xlim=(-lim,lim),ylim=(-lim,lim),xlabel="Observed within-country EFC deviation\n(EUR kW$^{-1}$ yr$^{-1}$)",
           ylabel="Held-out EFC prediction\n(EUR kW$^{-1}$ yr$^{-1}$)",aspect="equal")
    ax.text(.035,.975,"Joint geometry | n = 132\nLOCO $R^2$ = "+f"{d.loc['joint_geometry','loco_r2']:.3f}",
            transform=ax.transAxes,va="top",fontsize=6)
    for label,panel in zip("abc",[a,b,ax]):
        panel.text(-.16,1.06,label,transform=panel.transAxes,fontweight="bold",fontsize=8)
    export(fig,"supp_figure7")
    plt.close(fig)
    return {"panels":3,"models":4,"coefficient_models":3,"predictions":132,"country_blocks":15,
            "loco_r2":float(d.loc["joint_geometry","loco_r2"]),
            "interval_input":"frozen fitted coefficients and held-out predictions",
            "removed_display":"system-cost response, retained with intervals in Supplementary Table 7"}

if __name__=="__main__":
    result={"supp_figure1":supplement1(),"supp_figure7":supplement7()}
    (AUDIT/"figure_display_contract.json").write_text(json.dumps(result,indent=2))
