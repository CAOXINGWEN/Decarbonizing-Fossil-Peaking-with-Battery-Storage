#!/usr/bin/env python3
"""Revise the locator alignment and South China Sea inset in Extended Data 7.

Figure contract
- Conclusion/evidence: retain the same geographical samples and the same
  cross-duration observations in panels b--f.
- Archetype: map-led quantitative composite, exact reuse of numerical panels.
- Backend/export: Python/matplotlib; 183 x 210 mm source, editable PDF/SVG,
  600 dpi TIFF and PNG preview. The existing geographical records are reused.
- Geometry: coordinate extents are reframed without stretching the maps;
  islands and the nine displayed boundary segments move into a dedicated inset.
"""
from pathlib import Path
import argparse,importlib.util,sys,json,hashlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import fitz

HERE=Path(__file__).resolve().parent
DEFAULT_WS=Path('/data2/cxw/Decarbonizing-Fossil-Peaking-with-Battery-Storage')
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--workspace',type=Path,default=DEFAULT_WS)
p.add_argument('--output',type=Path,default=HERE/'outputs')
p.add_argument('--baseline',action='store_true')
args=p.parse_args()
REV=args.workspace/'Decarbonizing-Fossil-Peaking-with-Battery-Storage-master/paper-code/docs/musrcipt/revision_nature_energy_v62'
FW=REV/'figure_work';SRC=FW/'extended_data_figure7_v62/plot_ed7_three_regime_cross_duration.py'
sys.path.insert(0,str(FW));spec=importlib.util.spec_from_file_location('ed7_original',SRC);ed=importlib.util.module_from_spec(spec);spec.loader.exec_module(ed)
import figure2_relayout_v62.plot_preview_figure2_world_callout_relayout as maps
args.output.mkdir(parents=True,exist_ok=True)
GEOMETRY={}
original_locator=maps.draw_world_locator
original_scale=maps.draw_event_scale

def draw_world_locator(fig,ax,inset_axes,europe):
    original_locator(fig,ax,inset_axes,europe)
    ax.set_ylim(17,78)
    # Fix longitude span to the physical viewport ratio. Equal x/y geographic
    # scale now fills the exact width already shared by the three lower frames.
    box=ax.get_position(original=True)
    width=box.width*fig.get_figwidth();height=box.height*fig.get_figheight()
    centre=sum(ax.get_xlim())/2;lat_span=np.diff(ax.get_ylim())[0]
    lon_span=lat_span*width/height
    ax.set_xlim(centre-lon_span/2,centre+lon_span/2)
    ax.set_aspect('equal',adjustable='box')
    ax.add_patch(Rectangle((0,0),1,1,transform=ax.transAxes,fill=False,
      edgecolor='#AEB8B3',lw=0.42,clip_on=False,zorder=12))
    ax.set_gid('world_locator')
    for name,parent in inset_axes.items():parent.set_gid('map_frame_'+name)
    GEOMETRY['locator_extent']=[*ax.get_xlim(),*ax.get_ylim()]

def draw_china(ax,china):
    values=china.set_index('province')['longest_extreme_peaking_run_days'].to_dict()
    lookup={maps.CHINA_NAMES[name]:value for name,value in values.items() if name in maps.CHINA_NAMES}
    geo=json.loads(maps.CHINA_GEO.read_text(encoding='utf-8'))
    hainan=None
    for feature in geo['features']:
        name=feature['properties'].get('name');value=lookup.get(name)
        face=maps.EVENT_CMAP(maps.EVENT_NORM(value)) if value is not None else maps.LAND
        maps.add_geojson(ax,feature['geometry'],face,edge='white',lw=0.25,z=3 if value is not None else 1)
        if name=='海南省':hainan=feature['geometry']
    if hainan is not None:maps.add_geojson(ax,hainan,maps.LAND,edge='#9FA9A4',lw=0.30,z=5)
    ax.set_xlim(73,136);ax.set_ylim(17.5,54);ax.set_aspect('equal');ax.axis('off');ax.set_gid('china_main')
    # The inset lies east of Taiwan in the otherwise empty lower-right margin.
    inset=ax.inset_axes([0.80,0.01,0.19,0.48],zorder=20)
    inset.set_facecolor('white');inset.set_gid('south_china_sea')
    for feature in geo['features']:
        maps.add_geojson(inset,feature['geometry'],maps.LAND,edge='#9FA9A4',lw=0.24,z=2,bounds=(107,124,2.5,25.5))
    south_boundary=next(f for f in geo['features'] if f['properties'].get('name')=='境界线'
      and min(pt[1] for line in f['geometry']['coordinates'] for pt in line)<5)
    segments=south_boundary['geometry']['coordinates'][:9]
    for line in segments:
        points=np.asarray(line,dtype=float)
        inset.plot(points[:,0],points[:,1],color='#86928B',lw=0.52,solid_capstyle='round',zorder=4)
    inset.set_xlim(107,124);inset.set_ylim(2.5,25.5)
    inset.set_aspect('equal');inset.set_anchor('SE')
    inset.set_xticks([]);inset.set_yticks([])
    for spine in inset.spines.values():spine.set_visible(True);spine.set_edgecolor('#9FA9A4');spine.set_linewidth(0.40)
    GEOMETRY['china_extent']=[73,136,17.5,54]
    GEOMETRY['south_china_sea_extent']=[107,124,2.5,25.5]
    GEOMETRY['boundary_segments']=len(segments)
    GEOMETRY['china_observations']=len(china)

def draw_event_scale(ax):
    original_scale(ax)
    pos=ax.get_position()
    ax.set_position([pos.x0,pos.y0-0.010,pos.width,pos.height])

if not args.baseline:
    maps.draw_world_locator=draw_world_locator
    maps.draw_china=draw_china
    maps.draw_event_scale=draw_event_scale

def all_axes(fig):
    def children(ax):
        yield ax
        for sub in ax.child_axes:yield from children(sub)
    for ax in fig.axes:yield from children(ax)

def export(fig):
    fig.canvas.draw()
    if not args.baseline:
        aa={ax.get_gid():ax for ax in all_axes(fig) if ax.get_gid()}
        positions={k:list(v.get_position().bounds) for k,v in aa.items()}
        world=aa['world_locator'].get_position();left=aa['map_frame_Europe'].get_position().x0;right=aa['map_frame_Japan'].get_position().x1
        GEOMETRY['positions']=positions
        GEOMETRY['left_alignment_error_pt']=abs(world.x0-left)*fig.get_figwidth()*72
        GEOMETRY['right_alignment_error_pt']=abs(world.x1-right)*fig.get_figwidth()*72
        assert GEOMETRY['left_alignment_error_pt']<1e-6
        assert GEOMETRY['right_alignment_error_pt']<1e-6
        main=aa['china_main'];GEOMETRY['china_pixels_per_degree']=float(main.transData.transform((100,30))[0]-main.transData.transform((99,30))[0])
    stem=args.output/('extended_data_figure7_baseline' if args.baseline else 'extended_data_figure7')
    fig.savefig(stem.with_suffix('.pdf'),metadata={'Creator':'Python/matplotlib'})
    fig.savefig(stem.with_suffix('.png'),dpi=300)
    if not args.baseline:
        fig.savefig(stem.with_suffix('.svg'))
        fig.savefig(stem.with_suffix('.tiff'),dpi=600,pil_kwargs={'compression':'tiff_lzw'})
        sources=[ed.SOURCE/'ed7_cross_duration_records.csv',ed.SOURCE/'ed7_crossing_status.csv',*[ed.MAP_SOURCE/f'figure2{x}_{s}_map.csv' for x,s in [('a','europe'),('b','china'),('c','japan')]],maps.CHINA_GEO]
        GEOMETRY['source_hashes']={str(f.relative_to(args.workspace)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}
        (args.output/'map_layout_qa.json').write_text(json.dumps(GEOMETRY,ensure_ascii=False,indent=2)+'\n')
ed.export_figure=export
# Retain the original publication parameters throughout the figure.
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
    'font.size':6.2,'pdf.fonttype':42,'svg.fonttype':'none'})
ed.main()
