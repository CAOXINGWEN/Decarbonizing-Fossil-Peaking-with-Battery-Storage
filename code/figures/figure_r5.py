"""Figure 5. Eight storage technologies and 58 configurations across 11,054 annual records (641,132 optimizations).

Architecturally grid-locked, counter-staggered 7-panel layout (LEFT = 21.0 mm, RIGHT = 180.0 mm across all 3 rows):
Row 1 (3 panels):  a (4-16 h short-medium, 49 mm) | b (8-24 h A-CAES vs LAES, 54 mm) | c (4-720 h 200 bar H2, 38.5 mm)
Row 2 (wide+comp): d (58 x 150 B landscape carpet, 85 mm + cb) | e (45-deg iso-net-benefit phase plane, 56 mm)
Row 3 (comp+wide): f (58-node log-duration bubble spectrum, 63 mm) | g (3-stage Bezier alluvial Sankey, 85 mm)
"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle, Polygon
from matplotlib.colors import TwoSlopeNorm

W = Path(__file__).resolve().parents[1]
V = W.parents[1]
sys.path.insert(0, str(V / 'figures/code'))
import style as S
import data as D

H = 188.0
LEFT, RIGHT = 21.0, 180.0
TECH = ['li_lfp', 'vrfb', 'fecr_flow', 'gravity', 'molten_salt_tes', 'caes', 'laes', 'h2_tank']
SHORT = {
    'li_lfp': 'LFP',
    'vrfb': 'VRFB',
    'fecr_flow': 'Fe–Cr flow',
    'gravity': 'Gravity',
    'molten_salt_tes': 'Molten-salt TES',
    'caes': 'A-CAES',
    'laes': 'LAES',
    'h2_tank': 'Tank H₂',
}
TECH_PAL = {
    'li_lfp': '#356E96',
    'vrfb': '#4E8FA8',
    'fecr_flow': '#7A6899',
    'molten_salt_tes': '#A67633',
    'gravity': '#5A6B75',
    'caes': '#19646B',
    'laes': '#8A4F5A',
    'h2_tank': '#A84A38',
}
WIN_C = '#19646B'


def panel_a(cv, x, top, w, h, c):
    """Panel a: Short-to-medium duration technologies (4-16 h: LFP, VRFB, Fe-Cr, Molten-salt TES, Gravity)."""
    ax = cv.ax(x, top, w, h, 'a')
    S.grid(ax, 'y')
    ax.axhspan(0, 18, facecolor=S.WIN_PALE, zorder=0)
    ax.axhline(0.0, color=S.INK, lw=0.65, zorder=2)

    fam = ['li_lfp', 'vrfb', 'fecr_flow', 'molten_salt_tes', 'gravity']
    for t in fam:
        zt = c[c.tech == t]
        col = TECH_PAL[t]
        piv = zt.pivot(index='duration_h', columns='group', values='mean')
        xs = piv.index.to_numpy(float)
        ymin = piv.min(axis=1).to_numpy(float)
        ymax = piv.max(axis=1).to_numpy(float)
        ymean = piv.mean(axis=1).to_numpy(float)
        ax.fill_between(xs, ymin, ymax, color=col, alpha=0.22, lw=0, zorder=2.5)
        ax.plot(xs, ymean, color=col, lw=1.35, marker='o', ms=2.8, mec='white', mew=0.25, zorder=4)
        ib = int(np.argmax(ymean))
        ax.scatter([xs[ib]], [ymean[ib]], s=22, facecolor='none', edgecolor=col, lw=0.70, zorder=5)

    ax.text(10.3, -14.0, 'LFP', ha='left', va='center', fontsize=5.0, fontweight='bold', color=TECH_PAL['li_lfp'])
    ax.text(8.3, -44.0, 'VRFB', ha='left', va='center', fontsize=5.0, fontweight='bold', color=TECH_PAL['vrfb'])
    ax.text(12.3, -62.0, 'Fe–Cr', ha='left', va='center', fontsize=5.0, fontweight='bold', color=TECH_PAL['fecr_flow'])
    ax.text(13.1, -82.0, 'Molten-salt TES', ha='center', va='top', fontsize=5.0, fontweight='bold', color=TECH_PAL['molten_salt_tes'])
    ax.text(12.3, -136.0, 'Gravity', ha='left', va='center', fontsize=5.0, fontweight='bold', color=TECH_PAL['gravity'])
    ax.text(4.0, -125.0, 'All 5: endpoint ΔB < 0', ha='left', va='center', fontsize=5.0, color=S.MUTED)

    ax.set_xlim(3.5, 16.8)
    ax.set_xticks([4, 8, 12, 16])
    ax.set_ylim(-156, 16)
    ax.set_yticks([-150, -100, -50, 0])
    ax.set_xlabel('Rated duration (h)')
    ax.set_ylabel(f'Mean B ({S.UNIT})')


def panel_b(cv, x, top, w, h, c):
    """Panel b: Long-duration thermo-mechanical divergence (8-24 h: A-CAES climbing vs LAES falling)."""
    ax = cv.ax(x, top, w, h, 'b')
    S.grid(ax, 'y')
    ax.axhspan(0, 18, facecolor=S.WIN_PALE, zorder=0)
    ax.axhline(0.0, color=S.INK, lw=0.65, zorder=2)

    for t, ls_main in (('caes', '-'), ('laes', (0, (2.6, 1.3)))):
        zt = c[c.tech == t]
        for g in S.MAIN:
            q = zt[zt.group == g].sort_values('duration_h')
            xs = q.duration_h.to_numpy(float)
            ym = q['mean'].to_numpy(float)
            col = S.COLOR[g]
            ax.fill_between(xs, q.q25, q.q75, color=col, alpha=0.14 if t == 'caes' else 0.10, lw=0, zorder=1.5)
            ax.plot(xs, ym, color=col, lw=1.30, ls=ls_main, marker=S.MARKER[g], ms=2.8,
                    mec='white', mew=0.25, zorder=4)
            ib = int(np.argmax(ym))
            ax.scatter([xs[ib]], [ym[ib]], s=22, facecolor='none', edgecolor=col, lw=0.65, zorder=5)

    ax.text(16.0, 5.5, 'A-CAES, solid: ΔB 9.8 to 21.1', ha='center', va='bottom',
            fontsize=5.05, fontweight='bold', color=WIN_C)
    ax.text(16.0, -96.5, 'LAES, dashed: ΔB −23.0 to −34.2', ha='center', va='bottom',
            fontsize=5.05, fontweight='bold', color=TECH_PAL['laes'])
    ax.text(8.0, -72.5, '8–24 h ΔOp. saving: 12.1–23.4\nΔFixed: 2.3 A-CAES vs 46.3 LAES',
            ha='left', va='center', fontsize=5.0, color=S.INK, linespacing=1.12)

    ax.set_xlim(7.2, 24.8)
    ax.set_xticks([8, 12, 16, 20, 24])
    ax.set_ylim(-156, 16)
    ax.set_yticks([-150, -100, -50, 0])
    ax.tick_params(axis='y', labelleft=False)
    ax.set_xlabel('Rated duration (h)')


def panel_c(cv, x, top, w, h, c):
    """Panel c: 200 bar surface-tank hydrogen (4-720 h, log-duration broken-axis envelope)."""
    h_hi = 22.5
    gap_y = 2.5
    h_lo = h - h_hi - gap_y
    ax_hi = cv.ax(x, top, w, h_hi, 'c_hi')
    ax_lo = cv.ax(x, top + h_hi + gap_y, w, h_lo, 'c_lo')
    z = c[c.tech == 'h2_tank']

    for ax_s, ylim, mark_b in ((ax_hi, (-168, 18), True), (ax_lo, (-2150, -180), False)):
        if ylim[0] < 0 < ylim[1]:
            ax_s.axhspan(0, ylim[1], facecolor=S.WIN_PALE, zorder=0)
            ax_s.axhline(0, color=S.INK, lw=0.65, zorder=2)
        for g in S.MAIN:
            q = z[z.group == g].sort_values('duration_h')
            xv = q.duration_h.to_numpy(float)
            ym = q['mean'].to_numpy(float)
            col = S.COLOR[g]
            lx = np.log10(xv)
            box = (np.log10(3.5), np.log10(850), *ylim)
            ring = S.clip_ring(list(zip(np.r_[lx, lx[::-1]], np.r_[q.q25, q.q75[::-1]])), box)
            if len(ring) >= 3:
                ax_s.fill(10 ** ring[:, 0], ring[:, 1], color=col, alpha=0.22, lw=0, zorder=1)
            for seg in S.clip_lines([np.c_[lx, ym]], box):
                ax_s.plot(10 ** seg[:, 0], seg[:, 1], color=col, lw=1.2, zorder=3)
            inside = q['mean'].between(*ylim)
            ax_s.plot(xv[inside], ym[inside], ls='none', marker=S.MARKER[g], ms=2.7,
                      color=col, mec='white', mew=0.22, zorder=4)
            if mark_b:
                ib = int(np.argmax(ym))
                if ylim[0] <= ym[ib] <= ylim[1]:
                    ax_s.scatter([xv[ib]], [ym[ib]], s=22, facecolor='none', edgecolor=col, lw=0.65, zorder=5)
        ax_s.set_xscale('log')
        ax_s.set_xlim(3.5, 850)
        ax_s.set_ylim(*ylim)
        ax_s.minorticks_off()
        S.grid(ax_s, 'y')

    ax_hi.set_xticks([]); ax_hi.spines['bottom'].set_visible(False); ax_hi.set_yticks([-120, 0])
    ax_hi.text(26.0, -34.0, '200 bar Tank H₂', ha='left', va='top', fontsize=5.05, fontweight='bold', color=TECH_PAL['h2_tank'])
    ax_lo.set_xticks([4, 24, 168, 720]); ax_lo.set_xticklabels(['4', '24', '168', '720'])
    ax_lo.get_xticklabels()[-1].set_ha('right')
    ax_lo.set_yticks([-2000, -1000])
    ax_lo.set_xlabel('Rated duration (h)')
    ax_lo.text(4.3, -1520, '4 h best: 99.2%', ha='left', va='center', fontsize=5.0, fontweight='bold', color=S.INK)
    for a_sub, yy in [(ax_hi, 0), (ax_lo, 1)]:
        a_sub.plot([-0.025, 0.025], [yy - 0.04, yy + 0.04], transform=a_sub.transAxes, lw=0.65, color=S.INK, clip_on=False)


def panel_d(cv, x, top, w, h):
    """Panel d: Panoramic 58-configuration x 150-column net-benefit B landscape carpet."""
    agg = pd.read_csv(S.SD / 'figure5d_carpet_58x150.csv')
    row_specs = []
    for t in TECH:
        durs = sorted(agg.loc[agg.tech == t, 'duration_h'].unique())
        for d_h in durs:
            row_specs.append((t, d_h))
    assert len(row_specs) == 58

    mat = np.zeros((58, 150))
    piv = agg.pivot_table(index=['tech', 'duration_h'], columns=['group', 'pbin'], values='B')
    for ri, key in enumerate(row_specs):
        col_idx = 0
        for g in S.MAIN:
            for pb in range(50):
                mat[ri, col_idx] = float(piv.loc[key, (g, pb)])
                col_idx += 1

    ax = cv.ax(x, top, w, h, 'd')
    norm = TwoSlopeNorm(vmin=-180.0, vcenter=0.0, vmax=45.0)
    im = ax.imshow(np.clip(mat, -180.0, 45.0), cmap=S.DIV_B_CMAP, norm=norm, aspect='auto', zorder=1)

    for xv in (49.5, 99.5):
        ax.axvline(xv, color=S.INK, lw=0.65, zorder=3)

    y_centers, y_labels = [], []
    cur = 0
    for t in TECH:
        cnt = sum(1 for tt, _ in row_specs if tt == t)
        if cur > 0:
            ax.axhline(cur - 0.5, color='white', lw=0.55, zorder=2.5)
        y_centers.append(cur + (cnt - 1) / 2.0)
        y_labels.append('TES' if t == 'molten_salt_tes' else SHORT[t])
        cur += cnt

    ax.set_yticks(y_centers)
    ax.set_yticklabels(y_labels, fontsize=5.0)
    ax.set_xticks([24.5, 74.5, 124.5])
    ax.set_xticklabels(['China\n0–100%', 'Europe\n0–100%', 'Japan\n0–100%'], fontsize=5.0)
    ax.tick_params(axis='both', length=0)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color('#8FA0A8'); sp.set_linewidth(0.45)

    cbd = cv.ax(x + w + 1.4, top + 2.5, 2.2, h - 5.0, 'cb_d')
    cb = cv.fig.colorbar(im, cax=cbd)
    cb.set_ticks([-160, -80, 0, 40])
    cb.ax.set_yticklabels(['≤−160', '−80', '0', '40'], fontsize=5.0)
    cb.ax.axhline(0.0, color=S.INK, lw=0.65)
    cb.outline.set_linewidth(0.45); cb.outline.set_edgecolor(S.INK)
    cb.ax.set_title('B', fontsize=5.2, fontweight='bold', pad=2.0)


def add_iso_bands(ax, xlim, ylim):
    box = (*xlim, *ylim)
    bands = [
        (-10.0, -6.0, '#E3DDD4'),
        (-6.0, -4.0, '#ECE6DE'),
        (-4.0, -2.0, '#F4F0EA'),
        (-2.0, 0.0, '#FAF8F5'),
        (0.0, 2.0, '#E2F1EF'),
        (2.0, 6.0, '#CBE6E3'),
    ]
    for c_lo, c_hi, fc in bands:
        poly_raw = [(-5.0, -5.0 + c_lo), (15.0, 15.0 + c_lo), (15.0, 15.0 + c_hi), (-5.0, -5.0 + c_hi)]
        clipped = S.clip_ring(poly_raw, box)
        if len(clipped) >= 3:
            ax.add_patch(Polygon(clipped, closed=True, facecolor=fc, edgecolor='none', zorder=0))
    for c_val in (-8.0, -6.0, -4.0, -2.0, 0.0, 2.0):
        for seg in S.clip_lines([np.array([[-5.0, -5.0 + c_val], [15.0, 15.0 + c_val]])], box):
            if c_val == 0.0:
                ax.plot(seg[:, 0], seg[:, 1], ls=(0, (3.2, 1.8)), lw=0.95, color=WIN_C, zorder=2)
            else:
                ax.plot(seg[:, 0], seg[:, 1], ls=(0, (1.5, 1.8)), lw=0.50, color='#9AA8AD', zorder=1)


def panel_e(cv, x, top, w, h, e):
    """Panel e: 45-degree iso-net-benefit (delta O / delta h vs delta FC / delta h) phase diagram."""
    a = cv.ax(x, top, w, h, 'e')
    add_iso_bands(a, (0.0, 9.2), (0.0, 4.3))
    a.grid(False)

    for x_top, lab, col, fw in [
        (2.30, '2', WIN_C, 'normal'),
        (4.30, 'ΔB = 0', WIN_C, 'bold'),
        (6.30, '−2', S.MUTED, 'normal'),
        (8.30, '−4', S.MUTED, 'normal'),
    ]:
        a.text(x_top, 4.38, lab, ha='center', va='bottom', fontsize=5.0, color=col, fontweight=fw, clip_on=False)
    a.text(8.98, 3.32, '−6', ha='right', va='center', fontsize=5.0, color=S.MUTED)
    a.text(8.98, 1.32, '−8', ha='right', va='center', fontsize=5.0, color=S.MUTED)

    for tech in TECH:
        zt = e[e.tech.eq(tech)]
        fx_val = float(zt.fixed_per_added_kwh.iloc[0])
        ylo, yhi = float(zt.saving_per_added_kwh.min()), float(zt.saving_per_added_kwh.max())
        a.add_patch(Rectangle((fx_val - 0.09, ylo), 0.18, max(yhi - ylo, 0.05),
                              facecolor='white', alpha=0.82, edgecolor='#7C8E96', lw=0.55, zorder=3))

    for g in S.MAIN:
        z = e[e.group.eq(g)]
        a.scatter(z.fixed_per_added_kwh, z.saving_per_added_kwh, color=S.COLOR[g], marker=S.MARKER[g],
                  s=21, lw=0.32, edgecolor='white', zorder=5)

    label_specs = {
        'caes': ('A-CAES', 'max', (1.10, 2.22)),
        'molten_salt_tes': ('TES', 'max', (2.10, 3.12)),
        'laes': ('LAES', 'max', (2.89, 2.22)),
        'h2_tank': ('Tank H₂', 'max', (3.72, 0.78)),
        'vrfb': ('VRFB', 'max', (4.95, 3.82)),
        'fecr_flow': ('Fe–Cr', 'max', (5.38, 2.30)),
        'li_lfp': ('LFP', 'max', (6.40, 3.42)),
        'gravity': ('Gravity', 'max', (7.85, 3.08)),
    }
    for tech, (lab, side, xy_txt) in label_specs.items():
        zt = e[e.tech.eq(tech)]
        fx_v = float(zt.fixed_per_added_kwh.iloc[0])
        sy_v = float(zt.saving_per_added_kwh.max() if side == 'max' else zt.saving_per_added_kwh.min())
        a.annotate(lab, (fx_v, sy_v), xytext=xy_txt, textcoords='data', ha='center', va='center',
                   fontsize=5.05, fontweight='bold' if tech in ('caes', 'h2_tank') else 'normal',
                   color=WIN_C if tech == 'caes' else S.INK,
                   arrowprops=dict(arrowstyle='-', lw=0.45, color=S.MUTED, shrinkA=1, shrinkB=3.2), zorder=6)

    a.set_xlim(0, 9.2); a.set_ylim(0, 4.3)
    a.set_xticks([0, 2, 4, 6, 8]); a.set_yticks([0, 1, 2, 3, 4])
    a.set_xlabel('Fixed-cost increase (USD kWh⁻¹ yr⁻¹)')
    a.set_ylabel('Operating saving\n(USD kWh⁻¹ yr⁻¹)')


def panel_f(cv, x, top, w, h):
    """Panel f: 58-node log-duration optimal-selection bubble spectrum across all 8 technologies."""
    cat = D.catalogue_records()
    cat_s = cat.sort_values(['uid', 'tech', 'B', 'duration_h'], ascending=[True, True, False, True])
    best_t = cat_s.drop_duplicates(['uid', 'tech'], keep='first')
    n_tot = 11054.0

    order_f = ['caes', 'molten_salt_tes', 'vrfb', 'li_lfp', 'fecr_flow', 'laes', 'gravity', 'h2_tank']

    lgf = cv.ax(x, top - 4.3, w, 3.2, 'leg_f')
    lgf.set_xlim(0, w); lgf.set_ylim(0, 1); lgf.axis('off')
    lgf.scatter([1.1], [0.5], s=20, color=WIN_C, edgecolor='white', lw=0.35)
    lgf.text(3.0, 0.5, 'Upper', va='center', fontsize=5.0, fontweight='bold', color=WIN_C)
    lgf.scatter([15.8], [0.5], s=20, color='#C98A2C', edgecolor='white', lw=0.35)
    lgf.text(17.7, 0.5, 'Interior', va='center', fontsize=5.0, color=S.INK)
    lgf.scatter([32.8], [0.5], s=20, color='#4C758F', edgecolor='white', lw=0.35)
    lgf.text(34.7, 0.5, 'Lower', va='center', fontsize=5.0, color=S.INK)
    lgf.scatter([48.5], [0.5], s=8, facecolor='white', edgecolor='#9BAAB2', lw=0.55)
    lgf.text(50.3, 0.5, '0% node', va='center', fontsize=5.0, color=S.MUTED)

    ax = cv.ax(x, top, w, h, 'f')
    ax.set_xscale('log')
    ax.grid(False)

    node_rows = []
    for yi, t in enumerate(order_f):
        if yi % 2 == 1:
            ax.axhspan(yi - 0.46, yi + 0.46, facecolor='#F6F9FA', zorder=0.2)
        durs = sorted(cat.loc[cat.tech == t, 'duration_h'].unique())
        h_min, h_max = float(durs[0]), float(durs[-1])
        ax.plot([h_min, h_max], [yi, yi], color='#CBD6DC', lw=1.15, solid_capstyle='round', zorder=1.5)

        vc = best_t.loc[best_t.tech == t, 'duration_h'].value_counts().to_dict()
        for d_h in durs:
            cnt = int(vc.get(d_h, 0))
            pct = 100.0 * cnt / n_tot
            node_rows.append(dict(tech=t, duration_h=float(d_h), best_records=cnt, best_share_pct=pct))
            if pct < 0.05:
                ax.scatter([d_h], [yi], s=7.0, facecolor='white', edgecolor='#95A5AE', lw=0.50, zorder=2.5)
            else:
                col = WIN_C if d_h == h_max else ('#4C758F' if d_h == h_min else '#C98A2C')
                sz = 11.0 + 74.0 * np.sqrt(pct / 100.0)
                ax.scatter([d_h], [yi], s=sz, color=col, edgecolor='white', lw=0.45, alpha=0.93, zorder=4)

        if t == 'caes':
            ax.text(31.5, yi, '98.3%, 24 h', ha='left', va='center', fontsize=5.0, fontweight='bold', color=WIN_C)
        elif t == 'molten_salt_tes':
            ax.text(20.5, yi, '69.1%, 10–14 h', ha='left', va='center', fontsize=5.0, color='#9A6518')
        elif t == 'vrfb':
            ax.text(10.2, yi, '92.7%, 4 h', ha='left', va='center', fontsize=5.0, color='#4C758F')
        elif t == 'h2_tank':
            ax.text(9.0, yi - 0.30, '99.2% at 4 h', ha='left', va='bottom',
                    fontsize=5.0, fontweight='bold', color='#4C758F')
        else:
            p0 = 100.0 * int(vc.get(h_min, 0)) / n_tot
            ax.text(h_max * 1.22, yi, f'{p0:.1f}%, {int(h_min)} h', ha='left', va='center', fontsize=5.0, color=S.MUTED)

    ax.set_xlim(3.1, 1020.0)
    ax.minorticks_off()
    ax.set_xticks([4, 8, 16, 24, 72, 168, 720])
    ax.set_xticklabels(['4', '8', '16', '24', '72', '168', '720'])
    ax.set_ylim(len(order_f) - 0.35, -0.55)
    ax.set_yticks(range(len(order_f)))
    ax.set_yticklabels(['TES' if t == 'molten_salt_tes' else SHORT[t] for t in order_f],
                       fontsize=5.0, fontweight='bold')
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel('Candidate rated duration (h)')
    return pd.DataFrame(node_rows)


def _sankey_ribbon(ax, x0, x1, y0_top, y0_bot, y1_top, y1_bot, color, alpha=0.38, zorder=2):
    t = np.linspace(0.0, 1.0, 64)
    s = 0.5 - 0.5 * np.cos(np.pi * t)
    xs = x0 + (x1 - x0) * t
    yt = y0_top + (y1_top - y0_top) * s
    yb = y0_bot + (y1_bot - y0_bot) * s
    verts = np.vstack([np.c_[xs, yt], np.c_[xs[::-1], yb[::-1]]])
    ax.add_patch(Polygon(verts, closed=True, facecolor=color, edgecolor='none', alpha=alpha, zorder=zorder))


def panel_g(cv, x, top, w, h):
    """Panel g: Enlarged frameless (92.5 mm x 49.0 mm) 3-stage Bezier alluvial Sankey locked flush to RIGHT = 180.0 mm."""
    cat = D.catalogue_records()
    best_all = cat.sort_values(['uid', 'B', 'duration_h'], ascending=[True, False, True]).drop_duplicates('uid', keep='first')

    # Stage headers aligned directly with Panel f's top legend strip
    lgg = cv.ax(x, top - 4.3, w, 3.2, 'leg_g')
    lgg.set_xlim(0.0, 1.0); lgg.set_ylim(0.0, 1.0); lgg.axis('off')
    lgg.text(0.160, 0.5, 'System regime', ha='center', va='center', fontsize=5.05, fontweight='bold', color=S.MUTED)
    lgg.text(0.505, 0.5, 'Lowest-cost family', ha='center', va='center', fontsize=5.05, fontweight='bold', color=S.MUTED)
    lgg.text(0.865, 0.5, 'Optimal duration', ha='center', va='center', fontsize=5.05, fontweight='bold', color=S.MUTED)

    ax = cv.ax(x, top, w, h, 'g')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(1.0, 0.0)
    ax.axis('off')

    # Frameless enlarged Sankey geometry spanning the full width up to x = 1.000 (180.0 mm)
    x_l = 0.145
    bw_node = 0.044
    y_start = 0.040
    H_flow = 0.855
    gap_l = 0.042
    reg_h = (H_flow - 2 * gap_l) / 3.0

    # Stage 1: 3 equal-height regime blocks
    reg_spans, reg_shares = {}, {}
    cur_y = y_start
    for g in S.MAIN:
        sub = best_all[best_all.group == g]
        n_g = len(sub)
        n_ca24 = int(((sub.tech == 'caes') & (sub.duration_h == 24)).sum())
        n_ca_mid = int(((sub.tech == 'caes') & (sub.duration_h < 24)).sum())
        n_li_mid = int(((sub.tech == 'li_lfp') & (sub.duration_h > 4)).sum())
        n_li4 = int(((sub.tech == 'li_lfp') & (sub.duration_h == 4)).sum())
        assert n_ca24 + n_ca_mid + n_li_mid + n_li4 == n_g
        reg_shares[g] = (n_ca24 / n_g, n_ca_mid / n_g, n_li_mid / n_g, n_li4 / n_g)
        reg_spans[g] = (cur_y, cur_y + reg_h)
        ax.add_patch(Rectangle((x_l, cur_y), bw_node, reg_h, facecolor=S.COLOR[g], edgecolor='white', lw=0.45, zorder=4))
        yc = cur_y + 0.5 * reg_h
        sh_ca = 100.0 * (n_ca24 + n_ca_mid) / n_g
        ax.text(x_l - 0.016, yc - 0.030, S.SHORT_LABEL[g], ha='right', va='center', fontsize=5.2, fontweight='bold', color=S.COLOR[g])
        ax.text(x_l - 0.016, yc + 0.032, f'{sh_ca:.0f}% CAES', ha='right', va='center', fontsize=5.0, color=S.MUTED)
        cur_y += reg_h + gap_l

    # Stage 2: Winning Technology Families (A-CAES vs LFP)
    x_m = 0.485
    gap_m = 0.048
    ca_eq = sum(reg_shares[g][0] + reg_shares[g][1] for g in S.MAIN) / 3.0
    li_eq = 1.0 - ca_eq
    tot_flow_h = 3.0 * reg_h
    h_ca = tot_flow_h * ca_eq
    h_li = tot_flow_h * li_eq
    y_ca0, y_ca1 = 0.055, 0.055 + h_ca
    y_li0, y_li1 = y_ca1 + gap_m, y_ca1 + gap_m + h_li

    ax.add_patch(Rectangle((x_m, y_ca0), bw_node, h_ca, facecolor=WIN_C, edgecolor='white', lw=0.45, zorder=4))
    ax.add_patch(Rectangle((x_m, y_li0), bw_node, h_li, facecolor=TECH_PAL['li_lfp'], edgecolor='white', lw=0.45, zorder=4))
    ax.text(x_m + 0.5 * bw_node, y_ca0 - 0.012, 'A-CAES 17% (1,554)', ha='center', va='bottom',
            fontsize=5.1, fontweight='bold', color=WIN_C)
    ax.text(x_m + bw_node + 0.145, y_li0 + 0.52 * h_li, 'LFP 83%\n9,500 records', ha='center', va='center',
            fontsize=5.15, fontweight='bold', color='#1D3E56', linespacing=1.15, zorder=5)

    # Frameless baseline note aligned cleanly with Panel f's bottom spine
    ax.plot([0.145, 0.890], [0.935, 0.935], color='#CBD7DD', lw=0.50, zorder=2)
    ax.text(0.518, 0.972, 'Other 6 technologies: 0 / 11,054 lowest-cost records',
            ha='center', va='center', fontsize=5.0, color=S.MUTED, zorder=4)

    # Stage 3: Selected Rated Duration Tier locked flush to right edge (x = 1.000 = 180.0 mm)
    x_r = 0.846
    dur24_eq = sum(reg_shares[g][0] for g in S.MAIN) / 3.0
    dur_mid_eq = sum(reg_shares[g][1] + reg_shares[g][2] for g in S.MAIN) / 3.0
    dur4_eq = sum(reg_shares[g][3] for g in S.MAIN) / 3.0
    h_d24 = tot_flow_h * dur24_eq
    h_dmid = max(tot_flow_h * dur_mid_eq, 0.024)
    h_d4 = tot_flow_h * dur4_eq
    y_d24_0 = 0.050
    y_dmid_0 = y_d24_0 + h_d24 + 0.028
    y_d4_0 = y_dmid_0 + h_dmid + 0.028

    ax.add_patch(Rectangle((x_r, y_d24_0), bw_node, h_d24, facecolor=WIN_C, edgecolor='white', lw=0.45, zorder=4))
    ax.add_patch(Rectangle((x_r, y_dmid_0), bw_node, h_dmid, facecolor='#7AB6B2', edgecolor='white', lw=0.45, zorder=4))
    ax.add_patch(Rectangle((x_r, y_d4_0), bw_node, h_d4, facecolor='#4C758F', edgecolor='white', lw=0.45, zorder=4))

    ax.text(1.000, y_d24_0 + 0.5 * h_d24, '24 h\n1,550', ha='right', va='center',
            fontsize=5.1, fontweight='bold', color=WIN_C, linespacing=1.15)
    ax.text(1.000, y_dmid_0 + 0.5 * h_dmid, '6–20 h: 13', ha='right', va='center',
            fontsize=5.0, color=S.INK)
    ax.text(1.000, y_d4_0 + 0.5 * h_d4, '4 h\n9,491', ha='right', va='center',
            fontsize=5.1, fontweight='bold', color='#355B75', linespacing=1.15)

    # Connect Stage 1 -> Stage 2 Bezier ribbons
    ca_cur, li_cur = y_ca0, y_li0
    for g in S.MAIN:
        y0, y1 = reg_spans[g]
        s_ca = reg_shares[g][0] + reg_shares[g][1]
        dh_ca = reg_h * s_ca
        dh_li = reg_h - dh_ca
        _sankey_ribbon(ax, x_l + bw_node, x_m, y0, y0 + dh_ca, ca_cur, ca_cur + dh_ca, S.COLOR[g], alpha=0.42, zorder=2.2)
        _sankey_ribbon(ax, x_l + bw_node, x_m, y0 + dh_ca, y1, li_cur, li_cur + dh_li, S.COLOR[g], alpha=0.26, zorder=2.0)
        ca_cur += dh_ca
        li_cur += dh_li

    # Connect Stage 2 -> Stage 3 Bezier ribbons
    h_ca_to_24 = tot_flow_h * (sum(reg_shares[g][0] for g in S.MAIN) / 3.0)
    h_ca_to_mid = h_ca - h_ca_to_24
    h_li_to_mid = tot_flow_h * (sum(reg_shares[g][2] for g in S.MAIN) / 3.0)
    _sankey_ribbon(ax, x_m + bw_node, x_r, y_ca0, y_ca0 + h_ca_to_24, y_d24_0, y_d24_0 + h_d24, WIN_C, alpha=0.40, zorder=2.2)
    if h_ca_to_mid > 0:
        _sankey_ribbon(ax, x_m + bw_node, x_r, y_ca0 + h_ca_to_24, y_ca1, y_dmid_0, y_dmid_0 + 0.45 * h_dmid, WIN_C, alpha=0.35, zorder=2.2)
    if h_li_to_mid > 0:
        _sankey_ribbon(ax, x_m + bw_node, x_r, y_li0, y_li0 + h_li_to_mid, y_dmid_0 + 0.45 * h_dmid, y_dmid_0 + h_dmid, TECH_PAL['li_lfp'], alpha=0.35, zorder=2.1)
    _sankey_ribbon(ax, x_m + bw_node, x_r, y_li0 + h_li_to_mid, y_li1, y_d4_0, y_d4_0 + h_d4, TECH_PAL['li_lfp'], alpha=0.30, zorder=2.0)


def main():
    S.setup()
    cv = S.Canvas(H)
    c = D.catalogue_summary()
    e = pd.read_csv(W / 'tables/r5_endpoint_decomposition.csv')
    span = e.h_max - e.h_min
    e['fixed_per_added_kwh'] = e.fixed_increment / span
    e['saving_per_added_kwh'] = e.operating_saving / span
    b = D.catalogue_best()

    def _extra_leg(ax, x):
        ax.scatter([x + 1.2], [0.5], s=20, facecolor='none', edgecolor=S.INK, lw=0.65, zorder=4)
        ax.text(x + 3.4, 0.5, f"Best mean duration ({'A–C' if S.JOULE else 'a–c'})", va='center', fontsize=5.15, color=S.INK, zorder=4)
        return x + 36.0

    S.regime_legend_strip(
        cv, 2.0, 1.3, RIGHT - 2.0, 'legend',
        extra_items=(_extra_leg,),
        scope_label='Regimes (B, C, G)' if S.JOULE else 'Regimes (b, c, g)',
        framed=True,
    )

    # Row 1: Strictly LEFT = 21.0 mm to RIGHT = 180.0 mm across 3 duration-scale panels
    top1, h1 = 13.0, 39.5
    cv.letter(2.0, top1 - 4.2, 'a')
    panel_a(cv, LEFT, top1, 49.0, h1, c)

    cv.letter(72.5, top1 - 4.2, 'b')
    panel_b(cv, 77.0, top1, 54.0, h1, c)

    cv.letter(132.5, top1 - 4.2, 'c')
    panel_c(cv, 141.5, top1, RIGHT - 141.5, h1, c)

    # Row 2: Counter-staggered Wide Left (85 mm carpet) + Square-proportioned Right (56 mm phase plane), LEFT = 21.0 to RIGHT = 180.0
    top2, h2 = 68.0, 43.5
    cv.letter(2.0, top2 - 4.2, 'd')
    panel_d(cv, LEFT, top2, 85.0, h2)

    cv.letter(113.5, top2 - 4.2, 'e')
    panel_e(cv, 124.0, top2, RIGHT - 124.0, h2, e)

    # Row 3: Counter-staggered Compact Left (57.5 mm bubble spectrum) + Enlarged Frameless Right (92.5 mm Sankey), strictly LEFT = 21.0 to RIGHT = 180.0
    top3, h3 = 127.0, 49.0
    cv.letter(2.0, top3 - 4.2, 'f')
    bdf = panel_f(cv, LEFT, top3, 57.5, h3)
    bdf.to_csv(S.SD / 'figure5f_node_bubble_spectrum.csv', index=False)

    cv.letter(82.5, top3 - 4.2, 'g')
    panel_g(cv, 87.5, top3, RIGHT - 87.5, h3)

    # Source data exports (preserving exact backward-compatible filenames verified by audit_manuscript.py)
    S.SD.mkdir(parents=True, exist_ok=True)
    c.to_csv(S.SD / 'figure5a_h_all_technologies.csv', index=False)
    c.to_csv(S.SD / 'figure5abc_all_technologies.csv', index=False)
    e.to_csv(S.SD / 'figure5i_duration_cost_decomposition.csv', index=False)
    e.to_csv(S.SD / 'figure5e_duration_cost_decomposition.csv', index=False)
    b.to_csv(S.SD / 'figure5j_cheapest_configuration_counts.csv', index=False)
    b.to_csv(S.SD / 'figure5g_cheapest_configuration_counts.csv', index=False)

    (W / 'qa/figure_contract.json').write_text(json.dumps(dict(
        conclusion='Duration expansion has technology-specific cost-benefit responses under observed inter-event charging conditions.',
        panels={
            'a': 'Short-to-medium duration family across 4-16 h (LFP, VRFB, Fe-Cr, Gravity, Molten-salt TES; 18 configs).',
            'b': 'Long-duration thermo-mechanical divergence across 8-24 h (A-CAES vs LAES; 18 configs).',
            'c': '200 bar surface-tank hydrogen across 4-720 h (22 configs, log-duration broken y axis).',
            'd': 'Panoramic 58-configuration x 150 regime-percentile carpet summarizing all 641,132 optimizations.',
            'e': '45-degree iso-net-benefit phase diagram of endpoint operating savings vs fixed-cost increments.',
            'f': '58-node log-duration optimal-selection bubble spectrum across all 8 technologies.',
            'g': 'Full-span three-stage Bezier alluvial Sankey of cross-technology lowest-cost attribution.',
        },
        backend='python',
        dimensions_mm=[183, H],
        exports=['PDF', 'SVG', 'PNG 300 dpi', 'TIFF 600 dpi'],
    ), indent=2) + '\n')

    S.save(
        cv,
        'main_figure5',
        rows=[['a', 'b'], ['d', 'e'], ['f', 'g']],
        cols=[['c_hi', 'c_lo']],
        exempt=[
            dict(panels=['legend', 'cb_d', 'leg_f', 'leg_g'], checks=['all'], reason='Legends and colorbar'),
            dict(panels=['a', 'b', 'c_hi', 'c_lo', 'd', 'e', 'f', 'g'], checks=['panel-width'],
                 reason='Intentional counter-staggered asymmetric widths across 3 strictly left-right aligned rows'),
            dict(panels=['c_hi', 'c_lo'], checks=['row'],
                 reason='Broken y-axis pair spanning row 1 height'),
        ],
        audit=dict(
            panels=['a', 'b', 'c', 'd', 'e', 'f', 'g'],
            panel_count=7,
            technologies=8,
            configurations=58,
            records=11054,
            source_rows=641132,
        ),
    )


if __name__ == '__main__':
    main()
