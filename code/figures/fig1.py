"""Figure 1. Peaking tasks, charging windows and annual storage cycling across 11,054 annual records.

7 bespoke panels (a-g):
a  Tri-regional cartographic atlas (China 31 provinces + South China Sea inset, Europe 15 countries, Japan 5,306 communities) + study key
b  2D (12 months x 24 hours) diurnal-seasonal chargeable-surplus probability heatmaps with peaking-task frequency contours
c  Three median example years (Shaanxi 2022, France 2016, Tokyo community 2024): mirrored daily peaking task and chargeable hours
d  Normalized charging-window energy exceedance curves (E_window / E_usable) contrasting 4 h LFP and 24 h A-CAES saturation
e  Paired 4 h LFP versus 24 h A-CAES equivalent full cycling comparison (vertical half-densities + median utilization slopes)
f  Bivariate HDR contour and stratified quantile response of A-CAES equivalent full cycles to mean event duration
g  Bivariate response of A-CAES equivalent full cycles to annual chargeable hours (separating zero-curtailment mass) + |rho| bars
"""
import json
import hashlib
import numpy as np
import pandas as pd
import shapefile
from scipy.stats import gaussian_kde
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.collections import LineCollection

import style as S
import data as D

ROOT = D.SUB.parent
WORLD = ROOT / 'dataset/external_geospatial/world_natural_earth_110m_v5_1_1/ne_110m_admin_0_countries.shp'
CHINA = ROOT / 'dataset/中国_省.geojson'
CHINA_OUTLINE = S.V86 / 'figures/assets/china_outer_outline.geojson'
JP_PREFS = S.V86 / 'figures/assets/japan_5prefectures.geojson'
OCEAN = '#EAF2F6'
CN_EDGE, CN_PROVINCE_EDGE, CN_UNSAMPLED = '#4A555B', '#FAF8F4', '#EFECE4'
JP_FILL, JP_EDGE = '#DCEAF4', '#557894'
NATIONAL = 'AUT BEL BGR CZE DEU DNK ESP FRA GBR GRC HRV ITA PRT ROU SVN'.split()
ISO3 = dict(zip('AT BE BG CZ DE DK ES FR GB GR HR IT PT RO SI'.split(), NATIONAL))
COUNTRY = {
    'AT': 'Austria', 'BE': 'Belgium', 'BG': 'Bulgaria', 'CZ': 'Czechia', 'DE': 'Germany',
    'DK': 'Denmark', 'ES': 'Spain', 'FR': 'France', 'GB': 'Great Britain', 'GR': 'Greece',
    'HR': 'Croatia', 'IT': 'Italy', 'PT': 'Portugal', 'RO': 'Romania', 'SI': 'Slovenia',
}
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

H = 182.0
LEFT, RIGHT = 18.0, 180.0


def rings(geom):
    if geom['type'] == 'Polygon':
        return [geom['coordinates'][0]]
    if geom['type'] == 'MultiPolygon':
        return [p[0] for p in geom['coordinates']]
    return []


def patch(ax, ring, face, edge='white', lw=0.2, z=1, hatch=None, alpha=1.0):
    v = np.asarray(ring, float)
    box = (*ax.get_xlim(), *ax.get_ylim())
    if len(v) < 3 or v[:, 0].max() < box[0] or v[:, 0].min() > box[1] or v[:, 1].max() < box[2] or v[:, 1].min() > box[3]:
        return
    v = S.clip_ring(v, box)
    if len(v) < 3:
        return
    v = np.vstack([v, v[0]])
    codes = [MPath.MOVETO] + [MPath.LINETO] * (len(v) - 2) + [MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(v, codes), facecolor=face, edgecolor=edge, lw=lw, zorder=z, hatch=hatch, alpha=alpha))


def shape_rings(shape):
    pts = np.asarray(shape.points, float)
    parts = list(shape.parts) + [len(pts)]
    return [pts[a:b] for a, b in zip(parts[:-1], parts[1:])]


def mesh_lonlat(code):
    c = str(code)
    lat = int(c[:2]) / 1.5 + int(c[4]) * 5 / 60 + int(c[6]) * 0.5 / 60
    lon = int(c[2:4]) + 100 + int(c[5]) * 7.5 / 60 + int(c[7]) * 0.75 / 60
    q = int(c[8])
    lat += (0.25 if q in (3, 4) else 0) / 60 + 0.125 / 60
    lon += (0.375 if q in (2, 4) else 0) / 60 + 0.1875 / 60
    return lon, lat


def map_axes(cv, x, top, hmm, lon, lat, gid, w_scale=1.0):
    latmid = np.radians(np.mean(lat))
    w = w_scale * hmm * (lon[1] - lon[0]) * np.cos(latmid) / (lat[1] - lat[0])
    ax = cv.ax(x, top, w, hmm, gid)
    ax.set_facecolor(OCEAN)
    ax.set_xlim(*lon)
    ax.set_ylim(*lat)
    ax.set_aspect('auto')
    ax.set_xticks([])
    ax.set_yticks([])
    # Subtle cartographic graticules
    for gx in np.linspace(lon[0], lon[1], 5)[1:-1]:
        ax.axvline(gx, color='#D9E5EC', lw=0.32, zorder=0.5)
    for gy in np.linspace(lat[0], lat[1], 4)[1:-1]:
        ax.axhline(gy, color='#D9E5EC', lw=0.32, zorder=0.5)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color('#AFC0C8')
        s.set_linewidth(0.55)
    return ax, w


def world(ax, special):
    r = shapefile.Reader(str(WORLD), encoding='utf-8')
    for rec, shp in zip(r.records(), r.shapes()):
        a3 = rec['ADM0_A3']
        if a3 in ('CHN', 'TWN'):
            continue
        face, edge, lw, z, hatch = S.LAND, 'white', 0.25, 1, None
        if a3 in special:
            face, edge, lw, z, hatch = special[a3]
        for ring in shape_rings(shp):
            patch(ax, ring, face, edge, lw, z, hatch)


def china(ax, covered, highlight, lw_line=0.35, outline_lw=0.48):
    g = json.loads(CHINA.read_text(encoding='utf-8'))
    hi = None
    for f in g['features']:
        nm = f['properties']['name']
        if f['geometry']['type'] == 'MultiLineString':
            lines = [np.asarray(s, float) for s in f['geometry']['coordinates']]
            bounds = np.vstack(lines).min(axis=0)
            national = bounds[0] < 80 or bounds[1] < 10
            segs = S.clip_lines(lines, (*ax.get_xlim(), *ax.get_ylim()))
            ax.add_collection(LineCollection(segs, colors=CN_EDGE if national else CN_PROVINCE_EDGE,
                                             linewidths=lw_line if national else 0.18, zorder=7))
            continue
        face = S.COLOR['provincial'] if nm in covered else CN_UNSAMPLED
        for ring in rings(f['geometry']):
            patch(ax, ring, face, CN_PROVINCE_EDGE, 0.22, 5)
        if nm == highlight:
            hi = f['geometry']
    outline = json.loads(CHINA_OUTLINE.read_text(encoding='utf-8'))['features'][0]
    assert outline['properties']['source_sha256'] == hashlib.sha256(CHINA.read_bytes()).hexdigest()
    segs = S.clip_lines([np.asarray(s, float) for s in outline['geometry']['coordinates']],
                        (*ax.get_xlim(), *ax.get_ylim()))
    ax.add_collection(LineCollection(segs, colors=CN_EDGE, linewidths=outline_lw, zorder=7,
                                     capstyle='round', joinstyle='round'))
    for ring in rings(hi):
        patch(ax, ring, 'none', S.INK, 0.70, 8)


def japan_prefectures(ax):
    gj = json.loads(JP_PREFS.read_text(encoding='utf-8'))
    for f in gj['features']:
        for ring in rings(f['geometry']):
            patch(ax, ring, JP_FILL, 'white', 0.35, 4)
            patch(ax, ring, 'none', JP_EDGE, 0.24, 5)


def draw_maps(cv, top, hmm, left=LEFT, target_right=125.05):
    rec = D.records()
    covered = {CN_NAME[s.split('-', 1)[1]] for s in rec[rec.group == 'provincial'].system_id}
    assert len(covered) == 31
    meta = rec.set_index('uid')
    prov_ex = CN_NAME[meta.loc[D.EXAMPLES['provincial'], 'system_id'].split('-', 1)[1]]
    nat_ex = ISO3[meta.loc[D.EXAMPLES['national'], 'system_id']]
    x = left
    gap_m = 2.45
    specs = [
        ((73.0, 136.0), (17.5, 54.0)),
        ((-10.5, 30.5), (34.5, 60.5)),
        ((137.25, 141.00), (34.05, 36.75)),
    ]
    raw_w_sum = sum(hmm * (lo[1] - lo[0]) * np.cos(np.radians(np.mean(la))) / (la[1] - la[0]) for lo, la in specs)
    w_scale = (target_right - left - 2 * gap_m) / raw_w_sum

    # 1. China first (matching R1 order)
    ax_cn, w_cn = map_axes(cv, x, top, hmm, (73.0, 136.0), (17.5, 54.0), 'a-asia', w_scale=w_scale)
    world(ax_cn, {})
    china(ax_cn, covered, prov_ex)
    sw, sh = 7.9, 10.0
    ins = cv.ax(x + w_cn - sw - 0.55, top + hmm - sh - 0.55, sw, sh, 'a-scs')
    ins.set_facecolor(OCEAN)
    ins.set_xlim(105, 124)
    ins.set_ylim(0, 24.5)
    ins.set_xticks([]); ins.set_yticks([])
    for s in ins.spines.values():
        s.set_visible(True); s.set_color('#8E9CA1'); s.set_linewidth(0.45)
    world(ins, {})
    china(ins, covered, prov_ex, lw_line=0.35, outline_lw=0.25)
    x += w_cn + gap_m

    # 2. Europe second
    ax_eu, w_eu = map_axes(cv, x, top, hmm, (-10.5, 30.5), (34.5, 60.5), 'a-eu', w_scale=w_scale)
    world(ax_eu, {a3: (S.COLOR['national'], 'white', 0.32, 3, None) for a3 in NATIONAL})
    r = shapefile.Reader(str(WORLD), encoding='utf-8')
    for rec_, shp in zip(r.records(), r.shapes()):
        if rec_['ADM0_A3'] == nat_ex:
            for ring in shape_rings(shp):
                patch(ax_eu, ring, 'none', S.INK, 0.70, 8)
    x += w_eu + gap_m

    # 3. Japan communities last
    ax_jp, w_jp = map_axes(cv, x, top, hmm, (137.25, 141.00), (34.05, 36.75), 'a-jp', w_scale=w_scale)
    world(ax_jp, {})
    japan_prefectures(ax_jp)
    com = rec[rec.group == 'community']
    uniq_sys = sorted(set(com.system_id))
    assert len(uniq_sys) == 5306
    pts = np.array([mesh_lonlat(s.split('-')[-1]) for s in uniq_sys])
    ax_jp.scatter(pts[:, 0], pts[:, 1], s=0.55, marker='s', color='#1C6391',
                  alpha=0.55, lw=0, zorder=8, rasterized=True)
    ex = mesh_lonlat(meta.loc[D.EXAMPLES['community'], 'system_id'].split('-')[-1])
    ax_jp.scatter([ex[0]], [ex[1]], s=28, marker=S.MARKER['community'], facecolor='white',
                  edgecolor=S.INK, lw=0.80, zorder=10)
    x += w_jp
    return x, uniq_sys, pts


def draw_key(cv, x0, top, width, height):
    from matplotlib.patches import FancyBboxPatch
    rec = D.records()
    rows = [
        ('provincial', 'Task: electricity load; surplus: curtailed wind & solar', '31 provinces, 2015–2024'),
        ('national', 'Task: gas generation; surplus: wind & solar', '15 countries, 2015–2024'),
        ('community', 'Task: grid purchases; surplus: rooftop PV export', '5,306 communities in 5 prefectures, FY2023–2024'),
    ]
    n = rec.groupby('group', observed=True).size()
    ax = cv.ax(x0, top, width, height, 'a-key')
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis('off')
    ax.add_patch(FancyBboxPatch(
        (0.15, 0.15), width - 0.30, height - 0.30,
        boxstyle='round,pad=0.0,rounding_size=0.6',
        facecolor='#F6F9FB', edgecolor='#C8D6DD', lw=0.52, zorder=0.5, clip_on=False,
    ))
    for k, (g, src, desc) in enumerate(rows):
        y = 1.0 + k * 6.5
        ax.scatter([2.0], [y + 0.9], s=18, marker=S.MARKER[g], color=S.COLOR[g], edgecolor='white', lw=0.28, zorder=3)
        ax.text(4.3, y, S.LABEL[g], fontsize=5.35, fontweight='bold', va='top', zorder=3)
        ax.text(width - 1.5, y, f'n = {n[g]:,}', fontsize=5.15, fontweight='bold', color=S.COLOR[g], ha='right', va='top', zorder=3)
        ax.text(4.3, y + 2.15, src, fontsize=5.0, va='top', zorder=3)
        ax.text(4.3, y + 4.05, desc, fontsize=5.0, color=S.MUTED, va='top', zorder=3)
    ax.plot([1.5, width - 1.5], [20.4, 20.4], color='#DCE6EA', lw=0.45, zorder=2)
    ax.text(4.3, 21.0, f"Dark outlines: example records in {'C' if S.JOULE else 'c'}", fontsize=5.0, va='top', zorder=3)
    ax.add_patch(Rectangle((1.2, 23.4), 2.3, 1.5, facecolor=CN_UNSAMPLED, edgecolor=CN_EDGE, lw=0.35, zorder=3))
    ax.text(4.3, 23.1, 'Unsampled in China (Taiwan, Hong Kong', fontsize=5.0, va='top', zorder=3)
    ax.text(4.3, 24.9, 'and Macao data unavailable)', fontsize=5.0, color=S.MUTED, va='top', zorder=3)
    return ax


def panel_b(cv, x0, top, total_w, h):
    """Panel b: 2D (12 months x 24 hours) diurnal-seasonal chargeable-surplus heatmaps + peaking-task contours."""
    mh = pd.read_csv(S.SD / 'figure1b_month_hour_regimes.csv')
    regimes = [
        ('provincial_pos', 'provincial', 'China'),
        ('national', 'national', 'Europe'),
        ('community', 'community', 'Japan'),
    ]
    sub_w = 16.4
    gap = (total_w - 3 * sub_w) / 2.0
    hm_h = h - 6.8
    b_ids = []
    im_last = None

    for i, (r_key, g_col, s_lab) in enumerate(regimes):
        xx = x0 + i * (sub_w + gap)
        ax = cv.ax(xx, top + 2.2, sub_w, hm_h, f'b{i}')
        b_ids.append(f'b{i}')
        sub = mh[mh.regime == r_key]
        mat_chg = 100.0 * sub.pivot(index='month', columns='hour', values='chargeable_prob').to_numpy()
        mat_ev = 100.0 * sub.pivot(index='month', columns='hour', values='event_prob').to_numpy()

        im_last = ax.imshow(mat_chg, cmap=S.SURPLUS_CMAP, vmin=0.0, vmax=85.0, aspect='auto',
                            extent=(-0.5, 23.5, 12.5, 0.5), zorder=1)
        # Overlay contour where high-demand event task occurs in >= 16% of days
        X_g, Y_g = np.meshgrid(np.arange(24), np.arange(1, 13))
        if mat_ev.max() > 16.0:
            ax.contour(X_g, Y_g, mat_ev, levels=[16.0], colors=[S.COLOR[g_col]],
                       linewidths=0.65, linestyles='--', zorder=3)

        ax.plot([0, 1], [1.04, 1.04], transform=ax.transAxes, color=S.COLOR[g_col], lw=1.3, clip_on=False)
        cv.text(xx + sub_w / 2, top - 1.2, s_lab, ha='center', va='top',
                fontsize=5.5, fontweight='bold', color=S.COLOR[g_col])

        ax.set_xticks([0, 12, 23])
        ax.set_xticklabels(['0', '12', '23'], fontsize=5.0)
        if i == 0:
            ax.set_yticks([1, 4, 7, 10])
            ax.set_yticklabels(['Jan', 'Apr', 'Jul', 'Oct'], fontsize=5.0)
            ax.set_ylabel('Month')
        else:
            ax.set_yticks([])
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(True)
            ax.spines[sp].set_color('#B8C6CC')
            ax.spines[sp].set_linewidth(0.45)
        for sp in ('left', 'bottom'):
            ax.spines[sp].set_color('#B8C6CC')
            ax.spines[sp].set_linewidth(0.45)

    cv.text(x0 + total_w / 2, top + 2.2 + hm_h + 4.3, 'Hour of day', ha='center', va='top', fontsize=5.3)

    # Compact horizontal colorbar above/below? Let's place a sleek colorbar at bottom of panel b
    cb_ax = cv.ax(x0 + 4.0, top + hm_h + 9.2, total_w - 8.0, 1.8, 'cb_b')
    cb = cv.fig.colorbar(im_last, cax=cb_ax, orientation='horizontal')
    cb.set_ticks([0, 40, 80])
    cb.ax.set_xticklabels(['0%', '40%', '80%'], fontsize=5.0)
    cb.outline.set_linewidth(0.45)
    cb.outline.set_edgecolor('#8FA1A9')
    cb.set_label('Chargeable surplus probability  (dashed: event task)', fontsize=5.0, labelpad=1.2)
    return b_ids


def daily(uid):
    arr, ev = D.hourly(uid)
    task = arr[:, 0].reshape(365, 24).sum(1)
    charge = (arr[:, 2] > 1e-9).reshape(365, 24).sum(1)
    return pd.DataFrame(dict(uid=uid, day=np.arange(1, 366), task_mwh_per_mw=task, task_rel=task / task.max(),
                             high_demand_day=ev, chargeable_hours=charge))


def example_title(g, r):
    if g == 'provincial':
        return f'{r.system_id.split("-", 1)[1]}, {r.observation_year}'
    if g == 'national':
        return f'{COUNTRY[r.system_id]}, {r.observation_year}'
    return f'{r.system_id.split("-")[1]} community, {r.observation_year}'


def panel_c(cv, top, left, width):
    """Panel c: Three median example years with mirrored daily peaking task and chargeable hours."""
    rec = D.records().set_index('uid')
    frames, axes = [], []
    y = top
    for k, (g, uid) in enumerate(D.EXAMPLES.items()):
        d = daily(uid)
        frames.append(d.assign(group=g))
        r = rec.loc[uid]
        c = S.COLOR[g]
        cv.text(left, y, example_title(g, r), fontsize=5.5, fontweight='bold', color=c, va='top')
        cv.text(left + width, y,
                f'{r.charge_hours:,.0f} h,  A-CAES {r.efc_acaes:.1f} cycles,  event {r.mean_dur:.2f} d',
                fontsize=5.05, color=S.INK, va='top', ha='right')
        y += 2.4
        at = cv.ax(left, y, width, 4.3, f'c{k}-task')
        ac = cv.ax(left, y + 4.4, width, 3.5, f'c{k}-charge')
        at.set_facecolor('#FAFBFB')
        ac.set_facecolor('#F3F8F8')
        days = d.day.to_numpy()
        hd = d.high_demand_day.to_numpy()
        at.bar(days[~hd], d.task_rel[~hd], width=1.0, color=c, alpha=0.30, lw=0, zorder=2)
        at.bar(days[hd], d.task_rel[hd], width=1.0, color=c, lw=0, zorder=3)
        for s_day in (90.5, 181.5, 273.5):
            at.axvline(s_day, color='#E2E9EC', lw=0.40, zorder=1)
            ac.axvline(s_day, color='#DCE6E8', lw=0.40, zorder=1)
        at.set_xlim(0.5, 365.5); at.set_ylim(0, 1.20); at.set_yticks([0, 1]); at.set_xticks([])
        at.spines['bottom'].set_visible(True); at.spines['bottom'].set_color('#8D9BA0'); at.spines['bottom'].set_linewidth(0.45)

        ac.bar(days, d.chargeable_hours, width=1.0, color='#4C9A96', lw=0, zorder=2)
        ac.set_xlim(0.5, 365.5); ac.set_ylim(24, 0); ac.set_yticks([24])
        ac.spines['bottom'].set_visible(k == 2)
        ac.set_xticks([1, 90, 180, 270, 365] if k == 2 else [])
        if k == 2:
            ac.set_xticklabels(['1', '90', '180', '270', '365'])
            ac.get_xticklabels()[-1].set_ha('right')
            ac.set_xlabel('Day of year')
        axes += [at, ac]
        if k == 0:
            at.set_ylabel('Task', fontsize=5.0, rotation=0, ha='right', va='center', labelpad=2.5)
            ac.set_ylabel('Chg h', fontsize=5.0, rotation=0, ha='right', va='center', labelpad=2.5)
        y += 7.9 + 2.1
    return pd.concat(frames, ignore_index=True), axes


def panel_d(cv, x, top, w, h):
    """Panel d: Normalized charging-window fill exceedance curves (4 h LFP vs 24 h A-CAES)."""
    wdf = pd.read_csv(S.SD / 'figure1d_window_exceedance.csv')
    lgd = cv.ax(x, top - 4.0, w, 3.0, 'leg_d')
    lgd.set_xlim(0, w); lgd.set_ylim(0, 1); lgd.axis('off')
    lgd.plot([0.0, 4.2], [0.5, 0.5], color=S.INK, lw=1.0, ls=(0, (2.2, 1.2)))
    lgd.text(5.2, 0.5, '4 h LFP (3.6 MWh MW⁻¹)', va='center', fontsize=5.0, color=S.INK)
    lgd.plot([36.5, 40.7], [0.5, 0.5], color=S.INK, lw=1.4)
    lgd.text(41.7, 0.5, '24 h A-CAES (21.6, IQR)', va='center', fontsize=5.0, fontweight='bold', color=S.INK)

    ax = cv.ax(x, top, w, h, 'd')
    S.grid(ax, 'y')
    ax.set_xscale('log')
    ax.axvspan(1.0, 4.2, facecolor=S.WIN_PALE, zorder=0)
    ax.axvline(1.0, color=S.WIN, lw=0.80, ls=(0, (3.0, 1.8)), zorder=2.5)

    r_specs = [
        ('provincial_pos', 'provincial'),
        ('national', 'national'),
        ('community', 'community'),
    ]
    for r_key, g_col in r_specs:
        c = S.COLOR[g_col]
        z_l = wdf[(wdf.regime == r_key) & (wdf.tech == '4h_LFP')].sort_values('fill_multiple')
        z_c = wdf[(wdf.regime == r_key) & (wdf.tech == '24h_ACAES')].sort_values('fill_multiple')

        ax.plot(z_l.fill_multiple, z_l.median_windows, color=c, lw=1.05, ls=(0, (2.4, 1.4)), alpha=0.88, zorder=3.5)
        ax.fill_between(z_c.fill_multiple, z_c.q25_windows, z_c.q75_windows, color=c, alpha=0.16, lw=0, zorder=2)
        ax.plot(z_c.fill_multiple, z_c.median_windows, color=c, lw=1.45, zorder=4.2)

        ml1 = float(z_l.loc[np.isclose(z_l.fill_multiple, 1.0), 'median_windows'].iloc[0])
        mc1 = float(z_c.loc[np.isclose(z_c.fill_multiple, 1.0), 'median_windows'].iloc[0])
        ax.scatter([1.0], [ml1], s=13, marker=S.MARKER[g_col], facecolor='white', edgecolor=c, lw=0.70, zorder=5)
        ax.scatter([1.0], [mc1], s=15, marker=S.MARKER[g_col], color=c, edgecolor='white', lw=0.30, zorder=5.5)

    ax.text(1.12, 114.0, 'Full charge (1×)', ha='left', va='top', fontsize=5.05, fontweight='bold', color=S.WIN)
    ax.annotate('JP LFP: 62', (1.0, 62.0), xytext=(1.42, 76.0), textcoords='data',
                ha='left', va='center', fontsize=5.0, color=S.COLOR['community'],
                arrowprops=dict(arrowstyle='-', lw=0.40, color=S.COLOR['community'], shrinkA=1, shrinkB=2.5))
    ax.annotate('EU/JP CAES: 25', (1.0, 25.0), xytext=(0.13, 18.0), textcoords='data',
                ha='left', va='center', fontsize=5.0, fontweight='bold', color=S.INK,
                arrowprops=dict(arrowstyle='-', lw=0.40, color=S.INK, shrinkA=1, shrinkB=2.5))
    ax.annotate('CN pos: 17', (1.0, 17.0), xytext=(0.32, 7.5), textcoords='data',
                ha='left', va='center', fontsize=5.0, color=S.DEEP['provincial'],
                arrowprops=dict(arrowstyle='-', lw=0.40, color=S.DEEP['provincial'], shrinkA=1, shrinkB=2.5))

    ax.set_xlim(0.05, 4.0)
    ax.minorticks_off()
    ax.set_xticks([0.1, 0.25, 0.5, 1.0, 2.0, 4.0])
    ax.set_xticklabels(['0.1×', '0.25×', '0.5×', '1×', '2×', '4×'], fontsize=5.0)
    ax.set_ylim(0, 128)
    ax.set_yticks([0, 30, 60, 90, 120])
    ax.set_xlabel('Window energy / usable storage capacity')
    ax.set_ylabel('Windows exceeding threshold (yr⁻¹)')


def panel_e(cv, x, top, w, h):
    """Panel e: Paired 4 h LFP vs 24 h A-CAES cycling comparison (vertical half-densities + median slopes)."""
    rec = D.records()
    rec = rec[rec.group.isin(S.MAIN)].copy()
    pos_uids = D.cn_positive_uids()

    lge = cv.ax(x, top - 4.0, w, 3.0, 'leg_e')
    lge.set_xlim(0, w); lge.set_ylim(0, 1); lge.axis('off')
    lge.scatter([1.4], [0.5], s=14, marker='D', facecolor='white', edgecolor=S.INK, lw=0.70)
    lge.text(3.4, 0.5, '4 h LFP (left)', va='center', fontsize=5.0, color=S.INK)
    lge.scatter([28.5], [0.5], s=15, marker='o', color=S.INK, edgecolor='white', lw=0.30)
    lge.text(30.5, 0.5, '24 h A-CAES (right)', va='center', fontsize=5.0, fontweight='bold', color=S.INK)

    ax = cv.ax(x, top, w, h, 'e')
    S.grid(ax, 'y')

    dx = 0.17
    for gi, g in enumerate(S.MAIN):
        sub_all = rec[rec.group == g]
        sub = rec[rec.uid.isin(pos_uids)] if g == 'provincial' else sub_all
        c = S.COLOR[g]
        vl = sub.efc_lfp.to_numpy(float)
        vc = sub.efc_acaes.to_numpy(float)

        # Clip each half-violin strictly to empirical data range (no vertical zero-density tail artifacts!)
        ygl = np.linspace(float(vl.min()), min(118.0, float(vl.max())), 100)
        kl = gaussian_kde(vl, bw_method=0.25)(ygl)
        kl = kl / kl.max() * 0.23
        ax.fill_betweenx(ygl, gi - dx - kl, gi - dx, facecolor=S.TINT[g], edgecolor=c, lw=0.55, alpha=0.90, zorder=2)

        ygc = np.linspace(float(vc.min()), min(118.0, float(vc.max())), 100)
        kc = gaussian_kde(vc, bw_method=0.25)(ygc)
        kc = kc / kc.max() * 0.23
        ax.fill_betweenx(ygc, gi + dx, gi + dx + kc, facecolor=c, edgecolor=c, lw=0.55, alpha=0.35, zorder=2)

        ql25, ml, ql75 = np.quantile(vl, [0.25, 0.50, 0.75])
        qc25, mc, qc75 = np.quantile(vc, [0.25, 0.50, 0.75])
        ax.plot([gi - dx, gi - dx], [ql25, ql75], color=S.INK, lw=1.4, zorder=3.5)
        ax.plot([gi + dx, gi + dx], [qc25, qc75], color=S.INK, lw=1.4, zorder=3.5)
        ax.plot([gi - dx, gi + dx], [ml, mc], color=S.INK, lw=1.0, zorder=4)
        ax.scatter([gi - dx], [ml], s=18, marker='D', facecolor='white', edgecolor=c, lw=0.80, zorder=5)
        ax.scatter([gi + dx], [mc], s=20, marker=S.MARKER[g], color=c, edgecolor='white', lw=0.35, zorder=5)

        ratio = round(ml, 1) / round(mc, 1)
        ax.text(gi + 0.04, max(ml, mc) + 5.5, f'{ratio:.2f}×', ha='center', va='bottom',
                fontsize=5.15, fontweight='bold', color=c)

        if g == 'provincial':
            ml_all = float(sub_all.efc_lfp.median())
            mc_all = float(sub_all.efc_acaes.median())
            ax.plot([gi - dx, gi + dx], [ml_all, mc_all], color=S.MUTED, lw=0.65, ls=':', zorder=3)
            ax.text(0.42, 4.8, 'all CN: 4.0 vs 0.2', ha='left', va='center', fontsize=5.0, color=S.MUTED)

    ax.text(-0.44, 108.0, 'Task hours (median):\nCN 3,884, EU 4,293, JP 4,247',
            ha='left', va='top', fontsize=5.0, color=S.INK, linespacing=1.12)

    ax.set_xlim(-0.52, 2.52)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['China\n(pos. supply)', 'Europe', 'Japan'], fontsize=5.2)
    ax.set_ylim(0, 120)
    ax.set_yticks([0, 30, 60, 90, 120])
    ax.set_ylabel('Equivalent full cycles (yr⁻¹)')


def kde_contours_2d(ax, xv, yv, color, xlim, ylim):
    sx, sy = xlim[1] - xlim[0], ylim[1] - ylim[0]
    rng = np.random.default_rng(2026)
    idx = rng.choice(len(xv), size=min(len(xv), 3500), replace=False)
    pts = np.vstack([(xv[idx] - xlim[0]) / sx, (yv[idx] - ylim[0]) / sy])
    kde = gaussian_kde(pts, bw_method=0.22)
    gx, gy = np.linspace(0.01, 0.99, 70), np.linspace(0.01, 0.99, 70)
    GX, GY = np.meshgrid(gx, gy)
    Z = kde(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
    vals = np.sort(Z.ravel())[::-1]
    csum = np.cumsum(vals)
    t85 = vals[min(len(vals) - 1, np.searchsorted(csum, 0.85 * csum[-1]))]
    t50 = vals[min(len(vals) - 1, np.searchsorted(csum, 0.50 * csum[-1]))]
    Xp, Yp = xlim[0] + GX * sx, ylim[0] + GY * sy
    ax.contourf(Xp, Yp, Z, levels=[t85, Z.max() * 1.05], colors=[color], alpha=0.14, zorder=1)
    ax.contourf(Xp, Yp, Z, levels=[t50, Z.max() * 1.05], colors=[color], alpha=0.20, zorder=1.2)
    ax.contour(Xp, Yp, Z, levels=[t85, t50], colors=[color], linewidths=[0.45, 0.65], alpha=0.55, zorder=1.5)


def quantile_ribbon(ax, xv, yv, n_bins, color, marker, zorder_base=3):
    edges = np.unique(np.quantile(xv, np.linspace(0.02, 0.98, n_bins + 1)))
    if len(edges) < 3:
        return
    edges[0] -= 1e-6; edges[-1] += 1e-6
    xm, q25, q50, q75 = [], [], [], []
    for bi, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = ((xv >= lo) if bi == 0 else (xv > lo)) & (xv <= hi)
        if m.sum() >= 4:
            xm.append(float(np.median(xv[m])))
            l, md, u = np.quantile(yv[m], [0.25, 0.50, 0.75])
            q25.append(float(l)); q50.append(float(md)); q75.append(float(u))
    if len(xm) >= 2:
        ax.fill_between(xm, q25, q75, color=color, alpha=0.22, lw=0, zorder=zorder_base)
        ax.plot(xm, q50, color=color, lw=1.25, zorder=zorder_base + 1)
        ax.scatter(xm, q50, s=11, marker=marker, color=color, edgecolor='white', lw=0.28, zorder=zorder_base + 2)


def panels_f_and_g(cv, rec, top, height):
    """Panel f (mean_dur vs efc_acaes) and Panel g (charge_hours vs efc_acaes + |rho| bars)."""
    pos_uids = D.cn_positive_uids()
    summaries = []
    rho_map = {}
    for predictor in ['mean_dur', 'charge_hours']:
        req, rraw, by_g = D.equal_regime_spearman(rec, predictor, 'efc_acaes')
        rho_map[(predictor, 'equal_regime')] = req
        rho_map[(predictor, 'pooled')] = rraw
        summaries.append(dict(predictor=predictor, response='efc_acaes', group='equal_regime', n=len(rec),
                              spearman_rho=req, observation_unit='equal-regime weighted'))
        summaries.append(dict(predictor=predictor, response='efc_acaes', group='all_pooled', n=len(rec),
                              spearman_rho=rraw, observation_unit='annual record'))
        for g in S.MAIN:
            rho_map[(predictor, g)] = by_g[g]
            summaries.append(dict(predictor=predictor, response='efc_acaes', group=g, n=int((rec.group == g).sum()),
                                  spearman_rho=by_g[g], observation_unit='annual record'))
    summary = pd.DataFrame(summaries)

    # Panel f: Event duration vs A-CAES EFC (tight xlim 1.2..5.0, ylim -1.5..52)
    ax_f = cv.ax(LEFT, top, 52.0, height, 'f')
    S.grid(ax_f, 'y')
    ylim_fg = (-1.5, 52.0)
    for g in ('community', 'national', 'provincial'):
        data = rec[rec.group == g]
        c = S.COLOR[g]
        if g == 'community':
            kde_contours_2d(ax_f, data.mean_dur.to_numpy(float), data.efc_acaes.to_numpy(float), c, (1.2, 5.0), ylim_fg)
            quantile_ribbon(ax_f, data.mean_dur.to_numpy(float), data.efc_acaes.to_numpy(float), 8, c, S.MARKER[g], 3)
        elif g == 'national':
            ax_f.scatter(data.mean_dur, data.efc_acaes, s=6.5, marker=S.MARKER[g], color=c, alpha=0.30, lw=0, zorder=2)
            quantile_ribbon(ax_f, data.mean_dur.to_numpy(float), data.efc_acaes.to_numpy(float), 6, c, S.MARKER[g], 4)
        else:
            d_pos = data[data.uid.isin(pos_uids)]
            d_zero = data[~data.uid.isin(pos_uids)]
            ax_f.scatter(d_zero.mean_dur, d_zero.efc_acaes, s=5.0, marker='s', facecolor='white',
                         edgecolor=c, lw=0.40, alpha=0.45, zorder=1.8)
            ax_f.scatter(d_pos.mean_dur, d_pos.efc_acaes, s=6.5, marker='s', color=c, alpha=0.30, lw=0, zorder=2.2)
            quantile_ribbon(ax_f, d_pos.mean_dur.to_numpy(float), d_pos.efc_acaes.to_numpy(float), 5, c, 's', 4)

    ax_f.text(4.88, 48.0, f"Equal-regime ρ = {rho_map[('mean_dur', 'equal_regime')]:.2f}".replace('-', '−'),
              ha='right', va='top', fontsize=5.2, fontweight='bold', color=S.INK)
    ax_f.set_xlim(1.2, 5.0)
    ax_f.set_xticks([1.5, 2.5, 3.5, 4.5])
    ax_f.set_ylim(*ylim_fg)
    ax_f.set_yticks([0, 15, 30, 45])
    ax_f.set_xlabel('Mean event duration (d)')
    ax_f.set_ylabel('A-CAES cycling (yr⁻¹)')

    # Panel g: Chargeable hours vs A-CAES EFC (no bimodal yellow triangle!) + |rho| comparison bars
    ax_g = cv.ax(81.5, top, 52.0, height, 'g_chg')
    S.grid(ax_g, 'y')
    for g in ('community', 'national', 'provincial'):
        data = rec[rec.group == g]
        c = S.COLOR[g]
        if g == 'community':
            kde_contours_2d(ax_g, data.charge_hours.to_numpy(float), data.efc_acaes.to_numpy(float), c, (-200, 6850), ylim_fg)
            quantile_ribbon(ax_g, data.charge_hours.to_numpy(float), data.efc_acaes.to_numpy(float), 8, c, S.MARKER[g], 3)
        elif g == 'national':
            ax_g.scatter(data.charge_hours, data.efc_acaes, s=6.5, marker=S.MARKER[g], color=c, alpha=0.30, lw=0, zorder=2)
            quantile_ribbon(ax_g, data.charge_hours.to_numpy(float), data.efc_acaes.to_numpy(float), 6, c, S.MARKER[g], 4)
        else:
            d_pos = data[data.uid.isin(pos_uids)]
            ax_g.scatter([0.0], [0.0], s=26, marker='s', facecolor='white', edgecolor=S.DEEP['provincial'], lw=0.85, zorder=5)
            ax_g.text(180, 2.0, '0 surplus (n = 132)', ha='left', va='bottom', fontsize=5.0, color=S.DEEP['provincial'])
            ax_g.scatter(d_pos.charge_hours, d_pos.efc_acaes, s=6.5, marker='s', color=c, alpha=0.30, lw=0, zorder=2.2)
            quantile_ribbon(ax_g, d_pos.charge_hours.to_numpy(float), d_pos.efc_acaes.to_numpy(float), 5, c, 's', 4)

    ax_g.text(180, 48.0, f"Equal-regime ρ = {rho_map[('charge_hours', 'equal_regime')]:.2f}",
              ha='left', va='top', fontsize=5.2, fontweight='bold', color=S.INK)
    ax_g.set_xlim(-200, 6850)
    ax_g.set_xticks([0, 2000, 4000, 6000])
    ax_g.set_xticklabels(['0', '2,000', '4,000', '6,000'])
    ax_g.set_ylim(*ylim_fg)
    ax_g.set_yticks([0, 15, 30, 45])
    ax_g.tick_params(axis='y', labelleft=False)
    ax_g.set_xlabel('Chargeable hours (h yr⁻¹)')

    # Right subpanel of g: Paired horizontal bars of |Spearman rho| (uniform 11.5 mm gutter)
    ax_bar = cv.ax(145.0, top, RIGHT - 145.0, height, 'g_strength')
    ax_bar.grid(False)
    row_keys = ['provincial', 'national', 'community', 'equal_regime']
    row_labs = ['China', 'Europe', 'Japan', 'Equal-\nregime']
    bh = 0.30
    for yi, rk in enumerate(row_keys):
        c_main = S.INK if rk == 'equal_regime' else S.COLOR[rk]
        r_dur = abs(rho_map[('mean_dur', rk)])
        r_chg = abs(rho_map[('charge_hours', rk)])
        ax_bar.barh(yi - 0.17, r_dur, height=bh, facecolor=c_main, edgecolor='none', zorder=3)
        ax_bar.barh(yi + 0.17, r_chg, height=bh, facecolor='#DCE5E8', edgecolor=c_main, lw=0.55, zorder=3)
        ax_bar.text(r_dur + 0.03, yi - 0.17, f'{r_dur:.2f}', va='center', ha='left', fontsize=5.0,
                    fontweight='bold', color=c_main)
        ax_bar.text(r_chg + 0.03, yi + 0.17, f'{r_chg:.2f}', va='center', ha='left', fontsize=5.0, color=S.MUTED)

    ax_bar.axhline(2.52, color='#CBD4D8', lw=0.55, ls='--', zorder=2)
    ax_bar.set_xlim(0, 1.24)
    ax_bar.set_xticks([0, 0.5, 1.0])
    ax_bar.set_ylim(3.55, -0.55)
    ax_bar.set_yticks(range(4))
    ax_bar.set_yticklabels(row_labs, fontsize=5.1, fontweight='bold')
    ax_bar.tick_params(axis='y', length=0)
    ax_bar.set_xlabel('Constraint |ρ|')

    lg_g = cv.ax(141.0, top - 4.2, 39.0, 3.0, 'g_leg')
    lg_g.set_xlim(0, 39.0); lg_g.set_ylim(0, 1); lg_g.axis('off')
    lg_g.add_patch(Rectangle((0.0, 0.18), 2.8, 0.64, facecolor=S.INK, lw=0))
    lg_g.text(3.5, 0.5, 'Duration', va='center', fontsize=5.0, fontweight='bold', color=S.INK)
    lg_g.add_patch(Rectangle((17.5, 0.18), 2.8, 0.64, facecolor='#DCE5E8', edgecolor=S.INK, lw=0.5))
    lg_g.text(21.0, 0.5, 'Chargeable h', va='center', fontsize=5.0, color=S.INK)

    return summary


def main():
    S.setup()
    cv = S.Canvas(H)

    # Row 1: a (Maps + Key)
    cv.letter(1.5, 1.5, 'a')
    xend, uniq_sys, pts = draw_maps(cv, 3.0, 27.0)
    assert xend < 126.5, xend
    draw_key(cv, 127.5, 3.0, RIGHT - 127.5, 27.0)

    # Row 2: b (12x24 Diurnal-Seasonal Fingerprints) + c (365-Day Mirrored Horizons)
    top2 = 36.5
    cv.letter(1.5, top2 - 2.0, 'b')
    b_ids = panel_b(cv, LEFT, top2, 54.0, 31.0)

    cv.letter(75.5, top2 - 2.0, 'c')
    ex_daily, c_axes = panel_c(cv, top2 - 1.2, 84.5, RIGHT - 84.5)

    # Row 3: d (Window Fill Exceedance Curves) + e (Paired LFP vs A-CAES Cycling)
    top3, h3 = 83.5, 36.0
    cv.letter(1.5, top3 - 3.8, 'd')
    panel_d(cv, LEFT, top3, 72.5, h3)

    cv.letter(95.5, top3 - 3.8, 'e')
    panel_e(cv, 106.0, top3, RIGHT - 106.0, h3)

    # Row 4: f (Duration vs EFC) + g (Chargeable h vs EFC + |rho| bars)
    top4, h4 = 135.5, 35.5
    cv.letter(1.5, top4 - 3.8, 'f')
    cv.letter(74.0, top4 - 3.8, 'g')
    rec = D.records()
    rec = rec[rec.group.isin(S.MAIN)].copy()
    rho = panels_f_and_g(cv, rec, top4, h4)

    # Source data exports (backward-compatible + new panels)
    S.SD.mkdir(parents=True, exist_ok=True)
    cols = ['uid', 'group', 'system_id', 'observation_year', 'mean_dur', 'longest', 'events', 'event_days',
            'charge_hours', 'efc_acaes', 'efc_lfp', 'task_hours']
    rec[cols].to_csv(S.SD / 'figure1cf_records.csv', index=False)
    rec[cols].to_csv(S.SD / 'figure1c_records.csv', index=False)
    rec[cols].to_csv(S.SD / 'figure1e_records.csv', index=False)
    rec[['uid', 'group', 'system_id', 'observation_year', 'mean_dur', 'charge_hours', 'efc_acaes']].to_csv(
        S.SD / 'figure1d_correlation_points.csv', index=False)
    rho.to_csv(S.SD / 'figure1d_spearman.csv', index=False)
    rho.to_csv(S.SD / 'figure1fg_spearman.csv', index=False)
    ex_daily.to_csv(S.SD / 'figure1b_example_days.csv', index=False)
    ex_daily.to_csv(S.SD / 'figure1c_example_days.csv', index=False)
    pd.DataFrame([dict(group='community', system_id=s, lon=lo, lat=la) for s, (lo, la) in
                  zip(uniq_sys, pts)]).to_csv(S.SD / 'figure1a_point_locations.csv', index=False)

    S.save(
        cv,
        'main_figure1',
        rows=[b_ids, ['d', 'e'], ['f', 'g_chg', 'g_strength']],
        cols=[['c0-task', 'c1-task', 'c2-task'], ['c0-charge', 'c1-charge', 'c2-charge']],
        exempt=[
            dict(panels=['a-eu', 'a-asia', 'a-scs', 'a-jp'], checks=['all'], reason='Locator maps sized by geographic aspect'),
            dict(panels=['a-key', 'cb_b', 'leg_d', 'leg_e', 'g_leg'], checks=['all'], reason='Study key, colorbar and legend strips'),
            dict(panels=b_ids + ['d', 'e', 'f', 'g_chg', 'g_strength', 'c0-task', 'c1-task', 'c2-task', 'c0-charge', 'c1-charge', 'c2-charge'],
                 checks=['panel-width'], reason='Asymmetric multi-modality design across 7 panels'),
        ],
        audit=dict(
            panels=['a', 'b', 'c', 'd', 'e', 'f', 'g'],
            panel_count=7,
            examples=D.EXAMPLES,
            records=11054,
            excluded_main_records=0,
            correlations=rho.to_dict('records'),
        ),
    )


if __name__ == '__main__':
    main()
