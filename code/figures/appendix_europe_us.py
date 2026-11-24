"""Supplementary Figure S4 (appendix_figure_europe_us).
Weather-year variability in 75 European operating years (1950/51-2024/25) and 32 US operating-region records (2035 costs).

Upgraded diverse visual styles (zero middle dots, continuous silk envelopes, drought-year stratification):
a  75-year continuous 5th-95th chronology-envelope ribbon + threshold-split real-date return trajectory + >=8 d drought ticks
b  Stratified bivariate + group IQR crosshair boxes of mean event duration vs B (rho = -0.50)
c  Stratified bivariate + group IQR crosshair boxes of annual chargeable hours vs B (rho = -0.17)
d  3-region faceted paired chronology ribbon + Observed-vs-Fixed VRE mix comparison across CISO, ERCO, and MISO
"""
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy import stats

import style as S
import data as D

H = 120.0
DARK, LIGHT = '#7B3E6D', '#C9A3BE'
NAME = 'appendix_figure_europe_us'


def main():
    S.setup()
    cv = S.Canvas(H)
    rec = D.records().set_index('uid')
    p = D.placements().set_index('uid')
    h = rec[rec.group == 'hist'].join(p[['mc_q05', 'mc_q50', 'mc_q95', 'roi_real', 'mc_win36']])
    h['year'] = h.observation_year.astype(int)
    h = h.sort_values('year')
    assert len(h) == 75 and h.year.tolist() == list(range(1950, 2025))
    u = rec[rec.group == 'us'].join(p[['mc_q05', 'mc_q95', 'roi_real', 'mc_win36']])
    assert len(u) == 32

    # a: 75-year continuous 5th-95th chronology ribbon + threshold-split return curve
    cv.letter(1.5, 2.2, 'a')
    ax = cv.ax(19.0, 8.0, 161.0, 36.5, 'a')
    yrs = h.year.to_numpy(float)
    yv = h.roi_real.to_numpy(float)
    w = h.win36.to_numpy(bool)
    long8 = (h.longest >= 8).to_numpy(bool)

    ax.axhspan(3.6, 5.5, facecolor=S.WIN_PALE, alpha=0.55, zorder=0)
    ax.fill_between(yrs, h.mc_q05, h.mc_q95, facecolor='#D7CFE0', edgecolor='#9F8CB0', lw=0.40, alpha=0.65, zorder=1.5)
    ax.plot(yrs, h.mc_q50, color='#8C6D9E', lw=0.75, ls=':', zorder=2)
    ax.plot(yrs, yv, color=DARK, lw=0.85, alpha=0.65, zorder=3)
    ax.scatter(yrs[w], yv[w], s=11, color=DARK, edgecolor='white', lw=0.25, zorder=4)
    ax.scatter(yrs[~w], yv[~w], s=10, facecolor='white', edgecolor=DARK, lw=0.65, zorder=4)
    # Top rug ticks for >=8 d drought years
    ax.scatter(yrs[long8], np.full(long8.sum(), 5.32), s=8, marker='v', color='#C9543B', lw=0, zorder=4.5)

    ax.axhline(3.6, color=S.WIN, lw=0.75, ls=(0, (3, 2)), zorder=3.5)
    ax.text(1946.0, 3.68, '3.6%', ha='left', va='bottom', fontsize=5.2, fontweight='bold', color=S.WIN)
    ax.set_xlim(1945.5, 2025.5); ax.set_ylim(1.5, 5.5)
    ax.set_xticks(range(1950, 2025, 10)); ax.set_yticks([2, 3, 4, 5])
    ax.set_xlabel('Weather year (start of July–June operating year)')
    ax.set_ylabel('Incremental return (%)')
    S.grid(ax, 'y')
    ax.text(0.995, 1.02,
            f'Hybrid cheaper at 3.6%: {int(h.win36.sum())}/75 on real dates, {100 * h.mc_win36.mean():.0f}% in random placements (▼ longest ≥ 8 d)',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=5.2)

    # b, c: Stratified bivariate + IQR boxes for Longest >= 8 d (29 yr) vs < 8 d (46 yr)
    top2, hh = 63.0, 40.0
    specs = [
        ('b', 'mean_dur', 'Mean event duration (d)', (1.8, 2.85), [1.8, 2.1, 2.4, 2.7], 19.0),
        ('c', 'charge_hours', 'Chargeable hours (h yr⁻¹)', (3500, 3920), [3500, 3700, 3900], 73.5),
    ]
    for letter, col, lab, lim, ticks, x0 in specs:
        cv.letter(1.5 if letter == 'b' else x0 - 6.5, top2 - 4.0, letter)
        ax = cv.ax(x0, top2, 44.0, hh, letter)
        S.grid(ax, 'both')
        ax.axhspan(0, 7, facecolor=S.WIN_PALE, alpha=0.45, zorder=0)
        ax.axhline(0, color=S.INK, lw=0.65, zorder=2)
        for mk_mask, c_grp, z_ord in ((~long8, LIGHT, 3), (long8, DARK, 4)):
            xv, bv = h[col][mk_mask].to_numpy(float), h.B[mk_mask].to_numpy(float)
            xq1, xmed, xq3 = np.quantile(xv, [0.25, 0.50, 0.75])
            bq1, bmed, bq3 = np.quantile(bv, [0.25, 0.50, 0.75])
            ax.add_patch(Rectangle((xq1, bq1), xq3 - xq1, bq3 - bq1,
                                   facecolor=c_grp, alpha=0.16, edgecolor=c_grp, lw=0.60, zorder=z_ord - 0.5))
            ax.scatter(xv, bv, s=10, marker='D', color=c_grp, edgecolor='white', lw=0.2, zorder=z_ord)
            ax.scatter([xmed], [bmed], s=22, marker='P', color=c_grp, edgecolor='white', lw=0.45, zorder=z_ord + 0.5)
        r = stats.spearmanr(h[col], h.B)[0]
        ax.text(0.5, 1.02, f'Spearman ρ = {r:.2f}'.replace('-', '−'),
                transform=ax.transAxes, ha='center', va='bottom', fontsize=5.15, fontweight='bold')
        ax.set_xlim(*lim); ax.set_xticks(ticks)
        if col == 'charge_hours':
            ax.set_xticklabels([f'{t:,}' for t in ticks])
        ax.set_ylim(-9, 7); ax.set_yticks([-8, -4, 0, 4])
        ax.set_xlabel(lab)
        if letter == 'b':
            ax.set_ylabel(f'B ({S.UNIT})')
        else:
            ax.tick_params(axis='y', labelleft=False)

    lg = cv.ax(19.0, top2 - 8.2, 98.5, 2.6, 'legbc')
    lg.set_xlim(0, 98.5); lg.set_ylim(0, 1); lg.axis('off')
    lg.scatter([1], [0.5], s=11, marker='D', color=DARK, lw=0)
    lg.text(2.6, 0.5, 'Longest ≥ 8 d (29 yr, med −0.51)', va='center', fontsize=5.15)
    lg.scatter([52.5], [0.5], s=11, marker='D', color=LIGHT, lw=0)
    lg.text(54.1, 0.5, '< 8 d (46 yr, med 0.35)', va='center', fontsize=5.15)

    # d: US operating regions (CISO, ERCO, MISO) with shaded region columns and zero middle dots
    x0 = 128.0
    cv.letter(x0 - 9.5, top2 - 3.8, 'd')
    ax = cv.ax(x0, top2, 52.0, hh, 'd')
    S.grid(ax, 'y')
    ax.axhspan(3.6, 6.3, facecolor=S.WIN_PALE, alpha=0.45, zorder=0)

    ticks, labels, cx, reg_mid, order = [], [], 0.0, {}, []
    for ri, reg in enumerate(['CISO', 'ERCO', 'MISO']):
        yrs_r = sorted(u[u.system_id == reg].observation_year.astype(int).unique())
        start = cx
        for y in yrs_r:
            for mix, dx in (('observed_mix', -0.18), ('fixed_mix', 0.18)):
                order.append((f'us_{reg}_{y}_{mix}', cx + dx, mix))
            ticks.append(cx); labels.append(str(y)[2:])
            cx += 1
        end = cx - 1
        reg_mid[reg] = (start + end) / 2
        if ri == 1:
            ax.axvspan(start - 0.45, end + 0.45, facecolor='#F1F5F7', zorder=0.2)
        cx += 0.8

    z = u.loc[[o[0] for o in order]]
    xx = np.array([o[1] for o in order])
    obs = np.array([o[2] == 'observed_mix' for o in order])
    for m_mask, mk in ((obs, 'o'), (~obs, 's')):
        zz = z[m_mask]; w = zz.win36.to_numpy()
        ax.vlines(xx[m_mask], zz.mc_q05, zz.mc_q95, color='#8CA89E', lw=0.85, zorder=2)
        ax.scatter(xx[m_mask][w], zz.roi_real[w], s=10, marker=mk, color=S.COLOR['us'], edgecolor='white', lw=0.2, zorder=4)
        ax.scatter(xx[m_mask][~w], zz.roi_real[~w], s=10, marker=mk, facecolor='white', edgecolor=S.COLOR['us'], lw=0.65, zorder=4)

    ax.axhline(3.6, color=S.WIN, lw=0.70, ls=(0, (3, 2)), zorder=3)
    ax.set_xlim(-0.7, cx - 0.8 + 0.7 - 1)
    ax.set_ylim(2.5, 6.3); ax.set_yticks([3, 4, 5, 6])
    ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=5.0)
    ax.tick_params(axis='x', length=0)
    for reg, mid in reg_mid.items():
        ax.text(mid, 6.18, reg, ha='center', va='top', fontsize=5.4, fontweight='bold', color=S.INK)
    ax.set_ylabel('Incremental return (%)')
    ax.set_xlabel('Region and year (20xx)')
    ax.scatter([], [], s=10, marker='o', color=S.COLOR['us'], lw=0, label='Observed mix')
    ax.scatter([], [], s=10, marker='s', color=S.COLOR['us'], lw=0, label='Fixed mix')
    ax.legend(loc='lower right', handletextpad=0.2, borderaxespad=0.25, fontsize=5.0, ncol=2, columnspacing=0.6)
    ax.text(0.995, 1.02, f'{int(u.win36.sum())}/32 cheaper on real dates, random {100 * u.mc_win36.mean():.0f}%',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=5.15)

    h.reset_index()[['uid', 'year', 'roi_real', 'mc_q05', 'mc_q95', 'mc_win36', 'win36', 'B', 'mean_dur', 'longest', 'events',
                     'charge_hours']].to_csv(S.SD / 'appendix_europe_weather_years.csv', index=False)
    z.reset_index()[['uid', 'system_id', 'observation_year', 'roi_real', 'mc_q05', 'mc_q95', 'mc_win36', 'win36', 'B']].to_csv(
        S.SD / 'appendix_us_regions.csv', index=False)
    S.save(cv, NAME, rows=[['b', 'c', 'd']],
           exempt=[dict(panels=['d'], checks=['panel-width'], reason='d is wider to hold 32 records across 16 region-years'),
                   dict(panels=['legbc'], checks=['all'], reason='Legend strip')])


if __name__ == '__main__':
    main()
