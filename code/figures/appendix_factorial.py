"""Supplementary Figure S2 (appendix_figure_factorial).
Surplus-energy x charge-power x charging-access factorial across 11,054 records (2035 costs, 3.6 %).

Upgraded diverse visual styles:
a, b  Grouped vertical half-violin + IQR box + median crossbars (real dates vs 12-day extension)
c     8x3 proportional outcome gauge matrix
d     Shaded 1:1 asymmetry plane + 2D HDR density of added LFP-fill vs long windows under 4x charge power
e     3-regime cascading horizontal ridgeline landscape of annual one-hour charging windows
"""
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle, Polygon
from scipy.stats import gaussian_kde

import style as S
import data as D
import econ as E

H = 116.0
NAME = 'appendix_figure_factorial'
MS = {'provincial': 7, 'national': 7, 'community': 8}
CON = [
    ('surplus_energy_4x', 'Surplus\nenergy × 4'),
    ('charge_power_4x', 'Charge\npower × 4'),
    ('expanded_access', 'Surplus over all\ntask-free hours'),
    ('power_x_access', 'Power × access\ninteraction'),
]
CELLS = [(e, p, a) for a in ('observed', 'expanded') for p in (1, 4) for e in (1, 4)]


def rain_strips(ax, fx, lim, ticks):
    off = {'provincial': -0.26, 'national': 0.0, 'community': 0.26}
    S.grid(ax, 'y')
    ax.axhspan(0, lim[1], facecolor=S.WIN_PALE, alpha=0.45, zorder=0)
    ys_grid = np.linspace(lim[0], lim[1], 140)
    for i, (k, _) in enumerate(CON):
        for g in S.MAIN:
            v = np.clip(fx[fx.group == g][k].to_numpy(float), lim[0] + 0.2, lim[1] - 0.2)
            c = S.COLOR[g]
            xc = i + off[g]
            if np.std(v) > 0.25:
                kd = gaussian_kde(v, bw_method=0.28)(ys_grid)
                kd = kd / kd.max() * 0.11
                ax.fill_betweenx(ys_grid, xc, xc + kd, facecolor=c, alpha=0.30, edgecolor=c, lw=0.45, zorder=2.5)
            q10, q25, q75, q90 = np.quantile(v, [0.10, 0.25, 0.75, 0.90])
            ax.plot([xc, xc], [q10, q90], color=c, lw=0.75, zorder=3)
            ax.add_patch(Rectangle((xc - 0.045, q25), 0.09, max(q75 - q25, 0.25),
                                   facecolor='white', edgecolor=c, lw=0.65, zorder=3.5))
            m = float(np.median(v))
            ax.plot([xc - 0.065, xc + 0.065], [m, m], color=S.INK, lw=1.15, zorder=5, solid_capstyle='butt')
    ax.axhline(0, color=S.INK, lw=0.65, zorder=2)
    ax.set_ylim(*lim)
    ax.set_yticks(ticks)
    ax.set_xlim(-0.55, len(CON) - 0.40)
    ax.set_xticks(range(len(CON)))
    ax.set_xticklabels([l for _, l in CON], fontsize=5.2, linespacing=1.20)
    ax.tick_params(axis='x', length=0)


def main():
    S.setup()
    cv = S.Canvas(H)
    S.regime_legend_strip(cv, 1.5, 1.6, 178.5, 'leg', scope_label='Shared regime key', framed=True)

    top1, h1, top2, h2 = 13.8, 36.2, 66.5, 35.0
    real, d12 = D.factorial('real'), D.factorial('dB12')

    cv.letter(1.5, top1 - 3.8, 'a')
    ax = cv.ax(19.0, top1, 72.0, h1, 'a')
    rain_strips(ax, real, (-18, 25), [-15, 0, 15])
    ax.set_ylabel(f'Change in B, real dates\n({S.UNIT})')

    cv.letter(96.0, top1 - 3.8, 'b')
    ax = cv.ax(108.0, top1, 72.0, h1, 'b')
    rain_strips(ax, d12, (-15, 32), [-10, 0, 10, 20, 30])
    ax.set_ylabel(f'Change in ΔB, 12-day ext.\n({S.UNIT})')

    # c: 8x3 proportional outcome progress tiles
    cells = D.factorial_cells()
    cv.letter(1.5, top2 - 3.8, 'c')
    ax = cv.ax(19.0, top2, 44.0, h2, 'c')
    bw, bh = 0.82, 0.72
    for j, g in enumerate(S.MAIN):
        for i, (e, p, a) in enumerate(CELLS):
            r = cells[(cells.group == g) & (cells.energy_multiplier == e) & (cells.charge_power_multiplier == p) & (cells.access == a)].iloc[0]
            s = r.hybrid_cheaper / r.n
            ax.add_patch(Rectangle((j - bw / 2, i - bh / 2), bw, bh, facecolor='#EEF3F5', edgecolor='#CEDAE0', lw=0.35))
            ax.add_patch(Rectangle((j - bw / 2, i - bh / 2), bw * s, bh, facecolor=S.WIN, lw=0))
            lab = '100%' if r.hybrid_cheaper == r.n else ('>99%' if s >= 0.995 else ('0%' if r.hybrid_cheaper == 0 else f'{100 * s:.0f}%'))
            if s >= 0.50:
                tx, tc = j - bw / 2 + (bw * s) / 2, 'white'
            else:
                tx, tc = j - bw / 2 + bw * s + (bw * (1.0 - s)) / 2, S.INK
            ax.text(tx, i, lab, ha='center', va='center', fontsize=5.0, fontweight='bold' if s >= 0.5 else 'normal', color=tc)
        ax.text(j, -0.72, S.SHORT_LABEL[g], ha='center', va='bottom', fontsize=5.3, fontweight='bold', color=S.COLOR[g])
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(len(CELLS) - 0.5, -0.5)
    ax.set_yticks(range(len(CELLS)))
    ax.set_yticklabels([f'E×{e} P×{p} {"obs." if a == "observed" else "all"}' for e, p, a in CELLS], fontsize=5.05)
    ax.set_xticks([]); ax.tick_params(axis='y', length=0)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_visible(False)

    # d: Shaded 1:1 asymmetry plane + HDR contour of added windows under 4x charge power
    cw = D.charge_power_windows()
    cw['d_lfp'] = cw.lfp_fill_x4 - cw.lfp_fill_x1
    cw['d_long'] = cw.long_x4 - cw.long_x1
    cv.letter(70.0, top2 - 3.8, 'd')
    ax = cv.ax(81.0, top2, 40.0, h2, 'd')
    xlim_d = (min(-4.0, float(cw.d_lfp.min()) - 2.0), max(170.0, float(cw.d_lfp.max()) + 6.0))
    ylim_d = (min(-4.0, float(cw.d_long.min()) - 2.0), max(150.0, float(cw.d_long.max()) + 6.0))
    poly_asym = S.clip_ring([(0, 0), (180, 0), (180, 180)], (*xlim_d, *ylim_d))
    if len(poly_asym) >= 3:
        ax.add_patch(Polygon(poly_asym, closed=True, facecolor='#F6F9FA', edgecolor='none', zorder=0))
    ax.plot([0, 150], [0, 150], color=S.INK, lw=0.65, ls=(0, (3, 2)), zorder=2)

    jp_cw = cw[cw.group == 'community']
    jx, jy = jp_cw.d_lfp.to_numpy(float), jp_cw.d_long.to_numpy(float)
    rng = np.random.default_rng(2026)
    idx_s = rng.choice(len(jx), size=min(len(jx), 3500), replace=False)
    sx, sy = xlim_d[1] - xlim_d[0], ylim_d[1] - ylim_d[0]
    kde2 = gaussian_kde(np.vstack([(jx[idx_s] - xlim_d[0]) / sx, (jy[idx_s] - ylim_d[0]) / sy]), bw_method=0.22)
    gx, gy = np.linspace(0.01, 0.99, 60), np.linspace(0.01, 0.99, 60)
    GX, GY = np.meshgrid(gx, gy)
    Z = kde2(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
    vals = np.sort(Z.ravel())[::-1]
    csum = np.cumsum(vals)
    t88 = vals[min(len(vals) - 1, np.searchsorted(csum, 0.88 * csum[-1]))]
    t50 = vals[min(len(vals) - 1, np.searchsorted(csum, 0.50 * csum[-1]))]
    ax.contourf(xlim_d[0] + GX * sx, ylim_d[0] + GY * sy, Z, levels=[t88, Z.max() * 1.05], colors=[S.COLOR['community']], alpha=0.22, zorder=2.2)
    ax.contour(xlim_d[0] + GX * sx, ylim_d[0] + GY * sy, Z, levels=[t88, t50], colors=[S.COLOR['community']], linewidths=[0.50, 0.75], alpha=0.75, zorder=2.5)

    for g in ('provincial', 'national'):
        z = cw[cw.group == g]
        ax.scatter(z.d_lfp, z.d_long, s=MS[g], marker=S.MARKER[g], color=S.COLOR[g], edgecolor='white', lw=0.2, alpha=0.80, zorder=3)
    ax.set_xlim(*xlim_d); ax.set_ylim(*ylim_d)
    ax.set_xticks([0, 50, 100, 150]); ax.set_yticks([0, 50, 100, 150])
    ax.set_xlabel('Added 4 h LFP windows (yr⁻¹)')
    ax.set_ylabel('Added long windows (yr⁻¹)')

    # e: 3-regime cascading horizontal ridgeline landscape of one-hour windows
    cv.letter(128.0, top2 - 3.8, 'e')
    ax = cv.ax(139.0, top2, 41.0, h2, 'e')
    S.grid(ax, 'x')
    xmax_e = 100.0
    xs_k = np.linspace(0.0, xmax_e, 180)
    for i, g in enumerate(S.MAIN):
        v = np.clip(cw[cw.group == g].one_hour_windows.to_numpy(float), 0.0, xmax_e)
        c = S.COLOR[g]
        kd = gaussian_kde(v, bw_method=0.26)(xs_k)
        kd = kd / kd.max() * 0.68
        ax.fill_between(xs_k, i, i - kd, facecolor=c, alpha=0.34, edgecolor=c, lw=0.75, zorder=2 + i * 0.3)
        q25, m, q75 = np.quantile(v, [0.25, 0.50, 0.75])
        ax.plot([q25, q75], [i + 0.08, i + 0.08], color=c, lw=1.8, solid_capstyle='butt', zorder=4)
        ax.scatter([m], [i + 0.08], s=11, marker=S.MARKER[g], color=c, edgecolor='white', lw=0.35, zorder=5)
        ax.text(xmax_e - 2.0, i - 0.25, f'med {m:.0f}', ha='right', va='center', fontsize=5.0, fontweight='bold', color=c)
    ax.set_xlim(0, xmax_e); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(['0', '25', '50', '75', '100'])
    ax.get_xticklabels()[-1].set_ha('right')
    ax.set_ylim(2.45, -0.82)
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(['CN', 'EU', 'JP'], fontsize=5.1)
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('One-hour windows (yr⁻¹)')

    lw_ = E.win_legend(cv, 19.0, top2 + h2 + 8.5, 65.0)
    real.merge(d12, on=['uid', 'group'], suffixes=('_real', '_dB12')).to_csv(S.SD / 'appendix_factorial_contrasts.csv', index=False)
    cells.to_csv(S.SD / 'appendix_factorial_cells.csv', index=False)
    cw.to_csv(S.SD / 'appendix_factorial_charge_power_windows.csv', index=False)
    S.save(cv, NAME, rows=[['a', 'b'], ['c', 'd', 'e']],
           exempt=[dict(panels=['leg', lw_], checks=['all'], reason='Legend strips'),
                   dict(panels=['c', 'd', 'e'], checks=['panel-width'], reason='Distinct widths for cell gauges, scatter, and ridgelines')],
           audit=dict(records=11054, one_hour_windows_median=cw.groupby('group').one_hour_windows.median().to_dict()))


if __name__ == '__main__':
    main()
