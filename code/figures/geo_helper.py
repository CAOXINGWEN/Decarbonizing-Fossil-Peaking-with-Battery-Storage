"""Reusable publication-grade cartographic helpers for Nature Energy v86 figures.

Supports:
  - Mainland China 31 provincial polygons + verified national outline + South China Sea inset
  - European country polygons (15, 17, or 33 countries from Natural Earth 110m v5.1.1, mapping both GB and UK -> GBR)
  - Japan Kanto-Shizuoka 5 prefecture polygons + 5,306 500-m community grid cells (rasterized high-DPI squares)
"""
from functools import lru_cache
import hashlib
import json
import numpy as np
import shapefile
from matplotlib.collections import LineCollection
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MPath

import style as S
import data as D

ROOT = D.SUB.parent
WORLD = ROOT / 'dataset/external_geospatial/world_natural_earth_110m_v5_1_1/ne_110m_admin_0_countries.shp'
CHINA = ROOT / 'dataset/中国_省.geojson'
CHINA_OUTLINE = S.V86 / 'figures/assets/china_outer_outline.geojson'
JP_PREFS = S.V86 / 'figures/assets/japan_5prefectures.geojson'

OCEAN = '#EAF2F6'
CN_EDGE, CN_PROVINCE_EDGE, CN_UNSAMPLED = '#4A555B', '#FAF8F4', '#EFECE4'
JP_FILL, JP_EDGE = '#EEF4F8', '#6B889E'

CN_NAME = {
    'Anhui': '安徽省', 'Beijing': '北京市', 'Chongqing': '重庆市', 'Fujian': '福建省', 'Gansu': '甘肃省',
    'Guangdong': '广东省', 'Guangxi': '广西壮族自治区', 'Guizhou': '贵州省', 'Hainan': '海南省', 'Hebei': '河北省',
    'Heilongjiang': '黑龙江省', 'Henan': '河南省', 'Hubei': '湖北省', 'Hunan': '湖南省',
    'Inner Mongolia': '内蒙古自治区', 'Jiangsu': '江苏省', 'Jiangxi': '江西省', 'Jilin': '吉林省',
    'Liaoning': '辽宁省', 'Ningxia': '宁夏回族自治区', 'Qinghai': '青海省', 'Shaanxi': '陕西省',
    'Shandong': '山东省', 'Shanghai': '上海市', 'Shanxi': '山西省', 'Sichuan': '四川省',
    'Tianjin': '天津市', 'Tibet': '西藏自治区', 'Xinjiang': '新疆维吾尔自治区', 'Yunnan': '云南省',
    'Zhejiang': '浙江省',
}
CN_ZH_TO_EN = {v: k for k, v in CN_NAME.items()}


def rings(geom):
    if geom['type'] == 'Polygon':
        return [geom['coordinates'][0]]
    if geom['type'] == 'MultiPolygon':
        return [p[0] for p in geom['coordinates']]
    return []


def patch(ax, ring, face, edge='white', lw=0.22, z=2, alpha=1.0):
    v = np.asarray(ring, float)
    box = (*ax.get_xlim(), *ax.get_ylim())
    if len(v) < 3 or v[:, 0].max() < box[0] or v[:, 0].min() > box[1] or v[:, 1].max() < box[2] or v[:, 1].min() > box[3]:
        return
    v = S.clip_ring(v, box)
    if len(v) < 3:
        return
    v = np.vstack([v, v[0]])
    codes = [MPath.MOVETO] + [MPath.LINETO] * (len(v) - 2) + [MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(v, codes), facecolor=face, edgecolor=edge, lw=lw, zorder=z, alpha=alpha))


@lru_cache(None)
def _world_shapes():
    r = shapefile.Reader(str(WORLD), encoding='utf-8')
    out = []
    iso2_map = {}
    for rec, shp in zip(r.records(), r.shapes()):
        a3 = rec['ADM0_A3']
        iso2 = rec['ISO_A2_EH']
        pts = np.asarray(shp.points, float)
        parts = list(shp.parts) + [len(pts)]
        rgs = [pts[a:b] for a, b in zip(parts[:-1], parts[1:])]
        out.append((a3, iso2, rgs))
        if iso2 and iso2 != '-99':
            iso2_map[iso2] = a3
    iso2_map['UK'] = 'GBR'
    iso2_map['GB'] = 'GBR'
    iso2_map['EL'] = 'GRC'
    iso2_map['GR'] = 'GRC'
    return out, iso2_map


@lru_cache(None)
def _china_geo():
    g = json.loads(CHINA.read_text(encoding='utf-8'))
    outline = json.loads(CHINA_OUTLINE.read_text(encoding='utf-8'))['features'][0]
    assert outline['properties']['source_sha256'] == hashlib.sha256(CHINA.read_bytes()).hexdigest()
    return g, outline


@lru_cache(None)
def _jp_geo():
    return json.loads(JP_PREFS.read_text(encoding='utf-8'))


def mesh_lonlat(code):
    c = str(code).split('-')[-1]
    lat = int(c[:2]) / 1.5 + int(c[4]) * 5 / 60 + int(c[6]) * 0.5 / 60
    lon = int(c[2:4]) + 100 + int(c[5]) * 7.5 / 60 + int(c[7]) * 0.75 / 60
    q = int(c[8])
    lat += (0.25 if q in (3, 4) else 0) / 60 + 0.125 / 60
    lon += (0.375 if q in (2, 4) else 0) / 60 + 0.1875 / 60
    return lon, lat


def map_axes(cv, x, top, hmm, lon, lat, gid):
    latmid = np.radians(np.mean(lat))
    w = hmm * (lon[1] - lon[0]) * np.cos(latmid) / (lat[1] - lat[0])
    ax = cv.ax(x, top, w, hmm, gid)
    ax.set_facecolor(OCEAN)
    ax.set_xlim(*lon)
    ax.set_ylim(*lat)
    ax.set_aspect('auto')
    ax.set_xticks([])
    ax.set_yticks([])
    for gx in np.linspace(lon[0], lon[1], 5)[1:-1]:
        ax.axvline(gx, color='#D9E5EC', lw=0.30, zorder=0.5)
    for gy in np.linspace(lat[0], lat[1], 4)[1:-1]:
        ax.axhline(gy, color='#D9E5EC', lw=0.30, zorder=0.5)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color('#9FB1BA')
        s.set_linewidth(0.50)
    return ax, w


def draw_world_bg(ax, special=None):
    special = special or {}
    shapes, _ = _world_shapes()
    for a3, _, rgs in shapes:
        if a3 in ('CHN', 'TWN'):
            continue
        if a3 in special:
            face, edge, lw, z = special[a3]
        else:
            face, edge, lw, z = S.LAND, 'white', 0.22, 1
        for rg in rgs:
            patch(ax, rg, face, edge, lw, z)


def _fill_china_axis(ax, zh_colors, lw_line=0.32, outline_lw=0.45, prov_lw=0.20):
    g, outline = _china_geo()
    for f in g['features']:
        nm = f['properties']['name']
        if f['geometry']['type'] == 'MultiLineString':
            lines = [np.asarray(s, float) for s in f['geometry']['coordinates']]
            bounds = np.vstack(lines).min(axis=0)
            national = bounds[0] < 80 or bounds[1] < 10
            segs = S.clip_lines(lines, (*ax.get_xlim(), *ax.get_ylim()))
            ax.add_collection(LineCollection(
                segs, colors=CN_EDGE if national else CN_PROVINCE_EDGE,
                linewidths=lw_line if national else 0.16, zorder=7
            ))
            continue
        face = zh_colors.get(nm, CN_UNSAMPLED)
        for rg in rings(f['geometry']):
            patch(ax, rg, face, CN_PROVINCE_EDGE, prov_lw, 5)
    segs = S.clip_lines([np.asarray(s, float) for s in outline['geometry']['coordinates']],
                        (*ax.get_xlim(), *ax.get_ylim()))
    ax.add_collection(LineCollection(segs, colors=CN_EDGE, linewidths=outline_lw, zorder=7.5,
                                     capstyle='round', joinstyle='round'))


def draw_china_choropleth(cv, x, top, hmm, val_by_en, cmap, norm, gid='cn'):
    """Render 31-province choropleth + South China Sea inset. Keys in val_by_en can be 'Inner Mongolia' or 'CN-Inner Mongolia'."""
    zh_colors = {}
    for k, val in val_by_en.items():
        en = k.split('-', 1)[1] if k.startswith('CN-') else k
        zh = CN_NAME.get(en, en)
        zh_colors[zh] = cmap(norm(float(val)))

    ax_cn, w_cn = map_axes(cv, x, top, hmm, (73.0, 136.0), (17.5, 54.0), gid)
    draw_world_bg(ax_cn)
    _fill_china_axis(ax_cn, zh_colors)

    sw = max(6.2, hmm * 0.275)
    sh = sw * (10.0 / 7.6)
    ins = cv.ax(x + w_cn - sw - 0.50, top + hmm - sh - 0.50, sw, sh, f'{gid}_scs')
    ins.set_facecolor(OCEAN)
    ins.set_xlim(105, 124)
    ins.set_ylim(0, 24.5)
    ins.set_xticks([]); ins.set_yticks([])
    for s in ins.spines.values():
        s.set_visible(True); s.set_color('#8E9CA1'); s.set_linewidth(0.42)
    draw_world_bg(ins)
    _fill_china_axis(ins, zh_colors, lw_line=0.28, outline_lw=0.24, prov_lw=0.15)
    return ax_cn, ins, w_cn


def draw_europe_choropleth(cv, x, top, hmm, val_by_iso2, cmap, norm, gid='eu',
                           lon=(-10.5, 30.5), lat=(34.5, 60.5), edge_col='white', edge_lw=0.30):
    """Render European country choropleth for 15, 17, or 33 countries."""
    _, iso2_map = _world_shapes()
    special = {}
    for iso2, val in val_by_iso2.items():
        a3 = iso2_map.get(iso2, iso2)
        special[a3] = (cmap(norm(float(val))), edge_col, edge_lw, 4)
    ax_eu, w_eu = map_axes(cv, x, top, hmm, lon, lat, gid)
    draw_world_bg(ax_eu, special)
    return ax_eu, w_eu


def draw_japan_choropleth(cv, x, top, hmm, val_by_mesh, cmap, norm, gid='jp', pt_size=0.68):
    """Render Kanto-Shizuoka 5 prefectures + 5,306 500-m community grid cells colored continuously."""
    ax_jp, w_jp = map_axes(cv, x, top, hmm, (137.25, 141.00), (34.05, 36.75), gid)
    draw_world_bg(ax_jp)
    gj = _jp_geo()
    for f in gj['features']:
        for rg in rings(f['geometry']):
            patch(ax_jp, rg, JP_FILL, 'white', 0.32, 3)
            patch(ax_jp, rg, 'none', JP_EDGE, 0.24, 5)

    items = sorted(val_by_mesh.items(), key=lambda kv: float(kv[1]))
    pts = np.array([mesh_lonlat(k) for k, _ in items], dtype=float)
    vals = np.array([float(v) for _, v in items], dtype=float)
    cols = cmap(norm(vals))
    ax_jp.scatter(pts[:, 0], pts[:, 1], s=pt_size, marker='s', c=cols,
                  alpha=0.88, lw=0, zorder=7, rasterized=True)
    return ax_jp, w_jp
