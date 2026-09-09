"""Rebuild publication plots with an explicit USD coordinate conversion.

Figure contract: preserve the v70 scientific panel map, every observation,
confidence interval, data transformation and dimensionless comparison. Monetary
coordinates, tick scales and annotations use the same positive exchange factor.
Quantitative grids and mixed-modality figures retain their established 183 mm
Python/matplotlib layouts, editable PDF/SVG text and 600 dpi TIFF exports.
"""
from pathlib import Path
import importlib.util, sys, json, re, argparse, shutil
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
import pandas as pd
import numpy as np
from config import *
sys.path.insert(0,str(REV/'figure_work'))

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m

def all_axes(fig):
    out=[]
    def add(ax):
        if ax in out:return
        out.append(ax)
        for a in ax.child_axes:add(a)
    for ax in fig.axes:add(ax)
    return out

def describe(fig,name):
    fig.canvas.draw();out=[]
    for i,ax in enumerate(all_axes(fig)):
        out.append(dict(axis=i,label=ax.get_label(),position=list(ax.get_position().bounds),
           xlabel=ax.get_xlabel(),ylabel=ax.get_ylabel(),xscale=ax.get_xscale(),yscale=ax.get_yscale(),
           xlim=list(ax.get_xlim()),ylim=list(ax.get_ylim()),
           xticks=[t.get_text() for t in ax.get_xticklabels()],yticks=[t.get_text() for t in ax.get_yticklabels()],
           texts=[t.get_text() for t in ax.texts]))
    (WORK/'qa'/f'{name}_axes.json').write_text(json.dumps(out,indent=2))
    return out

def export(fig,name,*args):
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],'pdf.fonttype':42,'svg.fonttype':'none'})
    describe(fig,name)
    if MODE=='baseline':
        fig.savefig(WORK/'baseline_figures'/f'{name}.pdf',bbox_inches='tight',pad_inches=2/72)
        print('Captured',name,flush=True);return
    from usd_figure_units import convert_figure
    convert_figure(fig,name)
    fig.savefig(WORK/'figures'/f'{name}.pdf',bbox_inches='tight',pad_inches=2/72)
    fig.savefig(WORK/'figures'/f'{name}.svg',bbox_inches='tight',pad_inches=2/72)
    fig.savefig(WORK/'figures'/f'{name}.png',dpi=300,bbox_inches='tight',pad_inches=2/72)
    fig.savefig(WORK/'figures'/f'{name}.tiff',dpi=600,bbox_inches='tight',pad_inches=2/72,pil_kwargs={'compression':'tiff_lzw'})
    print('Rendered USD',name,flush=True)

def main():
    # Original figure functions remain read-only. Copies receive derived tables.
    v67=module('currency_v67',REV/'revision_issue1_v67/scripts/plot_v67_figures.py')
    v67.OUT=WORK/'plot_aux';v67.OUT.mkdir(exist_ok=True)
    v67.export_files=export
    stream=module('currency_stream',REV/'revision_issue1_v67/scripts/plot_streamlined_supplements_v67.py')
    stream.export=export
    revised=module('currency_revised',REV/'figure_revision_v69/scripts/plot_revised_figures.py')
    revised.OUT=WORK/'plot_aux';revised.QA=WORK/'qa/plot_aux';revised.QA.mkdir(exist_ok=True)
    revised.export=export
    copied=WORK/'plot_input_v69'
    shutil.copytree(revised.SRC,copied,dirs_exist_ok=True);revised.SRC=copied
    rob=module('currency_robustness',REV/'figure_work/supplementary_figures_v62/plot_v62_supplementary_robustness.py')
    def s2():
        rob.configure();rob.save_bundle=lambda fig,stem:export(fig,'supp_figure2')
        rob.plot_duration_boundaries(pd.read_csv(REV/'robustness_evidence/duration_boundary_summary.csv'))
    def s3():
        rob.configure();rob.save_bundle=lambda fig,stem:export(fig,'supp_figure3')
        rob.plot_factorial(pd.read_csv(REV/'recharge_factorial/factorial_cell_summary.csv'),pd.read_csv(REV/'recharge_factorial/factorial_contrast_summary.csv'))
    def e5():
        ed=module('currency_extended5',REV/'figure_work/plot_v54_extended_data.py')
        ed.configure()
        ed.save_ed=lambda fig,stem:export(fig,'extended_data_figure5');ed.ed3()
    def e6():
        ed=module('currency_extended6',REV/'figure_work/plot_v54_extended_data.py')
        ed.configure()
        ed.save_ed=lambda fig,stem:export(fig,'extended_data_figure6');ed.ed4()
    jobs={'main_figure3':v67.fig3,'main_figure4':revised.main_figure4,
          'main_figure5':v67.fig5,'main_figure6':v67.fig6,
          'supp_figure1':stream.supplement1,'supp_figure2':s2,'supp_figure3':s3,
          'supp_figure7':stream.supplement7,'extended_data_figure3':revised.cost_paths,
          'extended_data_figure5':e5,'extended_data_figure6':e6}
    for name in SELECTED or jobs:
        jobs[name]();plt.close('all')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['baseline','usd'],default='usd');p.add_argument('figures',nargs='*')
    a=p.parse_args();MODE=a.mode;SELECTED=a.figures;main()
