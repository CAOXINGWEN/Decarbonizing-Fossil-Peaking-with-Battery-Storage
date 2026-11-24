"""Extended Data Figure 8 (appendix_figure_capacity). Multi-year capacity planning across 15 European countries (q50 task, 2035 costs, 3.6 %).

7 bespoke panels (a-f, with cartographic spatial view):
a  15-country European choropleth map of 24 h A-CAES share in total procured power (left) + stacked procured power under Mean-Cost objective (right)
b  Stacked procured power under Risk-Averse (0.5 Mean + 0.5 CVaR90) objective with A-CAES share curve
c  Paired range/slope comparison of multi-technology cost saving vs best single technology (%) across Mean and Risk objectives
d  Horizontal/vertical diverging range bars of A-CAES power shift (Extended minus Dispersed chronology, pp)
e  Quantile fan ribbon (IQR + 10th-90th) and median trajectory of total procured power across 2, 4, and 6 h planning steps
f  Nested bullet tolerance intervals (0%, 0.1%, 1.0% cost budget) for 1 h planning in CZ, RO, and ES
"""
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

import style as S
import data as D
import geo_helper as G

H = 166.0
NAME = 'appendix_figure_capacity'
CT = D.H2_REVISION / 'tables'
TECHS = ['li_lfp_4h', 'caes_24h', 'h2_tank_168h']
TL = {'li_lfp_4h': '4 h LFP', 'caes_24h': '24 h A-CAES', 'h2_tank_168h': '168 h Tank H₂ (0%)'}
TC = {'li_lfp_4h': '#9AA8AE', 'caes_24h': '#1C6B73', 'h2_tank_168h': '#C58922'}
RISK = {0.0: ('Mean cost', '#276F9E', 'o'), 0.5: ('Mean + CVaR90', '#C58922', 's')}
ENS = [
    ('observed', 'Real dates', '#607178'),
    ('dispersed', 'Dispersed', '#6BAEA9'),
    ('clustered', 'Extended', '#C58922'),
    ('combined', 'Combined', '#19646B'),
]
OBJ = 'risk_adjusted_cost_eur_per_normalized_mw'


def main():
    S.setup()
    cv = S.Canvas(H)
    s = pd.read_csv(CT / 'capacity_design_summary.csv')
    c = pd.read_csv(CT / 'capacity_technology_power.csv')
    countries = sorted(s.country.unique())
    assert len(countries) == 15
    x = np.arange(15)

    # Top shared technology + objective legend strip
    lg = cv.ax(18.0, 2.0, 162.0, 3.5, 'legt')
    lg.set_xlim(0, 162); lg.set_ylim(0, 1); lg.axis('off')
    offs = [0.0, 28.0, 61.0]
    for off, t in zip(offs, TECHS):
        lg.add_patch(Rectangle((off, 0.16), 3.0, 0.68, facecolor=TC[t], lw=0))
        lg.text(off + 3.8, 0.5, TL[t], va='center', fontsize=5.2, fontweight='bold' if t != 'h2_tank_168h' else 'normal', color=S.INK)
    lg.scatter([105.0], [0.5], s=12, marker='o', color='#276F9E', lw=0)
    lg.text(106.8, 0.5, 'Mean cost', va='center', fontsize=5.2, color=S.INK)
    lg.scatter([129.0], [0.5], s=12, marker='s', color='#C58922', lw=0)
    lg.text(130.8, 0.5, 'Mean + CVaR90', va='center', fontsize=5.2, color=S.INK)

    # Row 1: a (European A-CAES share choropleth map + Mean-cost stacked procurement) & b (Mean+CVaR90 stacked procurement)
    top1, h1 = 13.5, 34.0
    cv.letter(1.5, top1 - 4.2, 'a')

    q0 = c[c.step.eq(4) & c.risk.eq(0.0) & c.ensemble.eq('combined') & c['mode'].eq('diversified')]
    pv0 = q0.pivot(index='country', columns='config_id', values='capacity_per_normalized_mw').reindex(countries)[TECHS] * 100
    tot0 = pv0.sum(axis=1)
    caes_share0 = (100.0 * pv0['caes_24h'] / tot0).to_dict()

    ax_map, w_map = G.draw_europe_choropleth(
        cv, 15.0, top1 + 1.0, 31.5, caes_share0, S.PHASE_CMAP, Normalize(vmin=20, vmax=100), 'a_map'
    )
    cv.text(15.0 + w_map / 2, top1 - 2.3, 'A-CAES share (20–100%)', ha='center', va='top',
            fontsize=5.0, fontweight='bold', color='#19646B')

    ax_a = cv.ax(61.5, top1, 53.5, h1, 'a')
    base = np.zeros(15)
    for t in TECHS:
        ax_a.bar(x, pv0[t], bottom=base, width=0.72, color=TC[t], lw=0, zorder=3)
        base += pv0[t].to_numpy()
    ax_a.set_ylim(0, 900); ax_a.set_yticks([0, 300, 600, 900])
    ax_a.set_xlim(-0.7, 14.7); ax_a.set_xticks(x); ax_a.set_xticklabels(countries, fontsize=5.0, rotation=90)
    ax_a.tick_params(axis='x', length=0)
    S.grid(ax_a, 'y')
    ax_a.set_ylabel('Power (% gas)')
    ax_a.text(0.03, 0.94, 'Mean cost (med 89% CAES)', transform=ax_a.transAxes,
              ha='left', va='top', fontsize=5.0, fontweight='bold', color='#276F9E')

    cv.letter(118.5, top1 - 4.2, 'b')
    ax_b = cv.ax(126.5, top1, 53.5, h1, 'b')
    q5 = c[c.step.eq(4) & c.risk.eq(0.5) & c.ensemble.eq('combined') & c['mode'].eq('diversified')]
    pv5 = q5.pivot(index='country', columns='config_id', values='capacity_per_normalized_mw').reindex(countries)[TECHS] * 100
    base = np.zeros(15)
    for t in TECHS:
        ax_b.bar(x, pv5[t], bottom=base, width=0.72, color=TC[t], lw=0, zorder=3)
        base += pv5[t].to_numpy()
    ax_b.set_ylim(0, 900); ax_b.set_yticks([0, 300, 600, 900]); ax_b.set_yticklabels([])
    ax_b.set_xlim(-0.7, 14.7); ax_b.set_xticks(x); ax_b.set_xticklabels(countries, fontsize=5.0, rotation=90)
    ax_b.tick_params(axis='x', length=0)
    S.grid(ax_b, 'y')
    ax_b.text(0.03, 0.94, 'Mean + CVaR90 (med 100%)', transform=ax_b.transAxes,
              ha='left', va='top', fontsize=5.0, fontweight='bold', color='#8A5D0E')

    # Row 2: c (Paired range bars for multi-tech cost saving), d (Diverging A-CAES shift bars), e (Step quantile fan)
    top2, h2 = 63.0, 34.0
    X, W = [19.0, 77.5, 136.0], 44.0

    cv.letter(1.5, top2 - 4.2, 'c')
    ax = cv.ax(X[0], top2, W, h2, 'c')
    ax.axhspan(-6.2, 0.0, facecolor=S.WIN_PALE, zorder=0)
    cc, dd = {}, {}
    for risk, (lab, col, mk) in RISK.items():
        q = s[s.step.eq(4) & s.risk.eq(risk) & s.ensemble.eq('combined')]
        multi = q[q['mode'].eq('diversified')].set_index('country')[OBJ].reindex(countries)
        single = q[q['mode'].ne('diversified')].groupby('country')[OBJ].min().reindex(countries)
        cc[risk] = 100 * (multi - single) / single
    for xi in range(15):
        pass
    ax.bar(x - 0.18, cc[0.0], width=0.34, color='#276F9E', alpha=0.88, lw=0, zorder=3)
    ax.bar(x + 0.18, cc[0.5], width=0.34, color='#C58922', alpha=0.88, lw=0, zorder=3)
    ax.axhline(0, color=S.INK, lw=0.6, zorder=3.5)
    ax.set_ylim(-6.0, 0.5); ax.set_yticks([-6, -4, -2, 0])
    ax.set_xlim(-0.7, 14.7); ax.set_xticks(x); ax.set_xticklabels(countries, fontsize=5.0, rotation=90)
    ax.tick_params(axis='x', length=0); S.grid(ax, 'y')
    ax.set_ylabel('Multi-tech cost Δ (%)')
    ax.text(0.5, 1.03, 'DK −5.4%, SI −4.8%', transform=ax.transAxes,
            ha='center', va='bottom', fontsize=5.0, fontweight='bold', color=S.WIN)

    cv.letter(X[1] - 13.0, top2 - 4.2, 'd')
    ax = cv.ax(X[1], top2, W, h2, 'd')
    ax.axhspan(0, 100, facecolor=S.WIN_PALE, alpha=0.75, zorder=0)
    ax.axhspan(-100, 0, facecolor='#FAF0EC', alpha=0.75, zorder=0)
    for risk, (lab, col, mk) in RISK.items():
        q = c[c.step.eq(4) & c.risk.eq(risk) & c.config_id.eq('caes_24h') & c['mode'].eq('diversified')]
        pv = q.pivot(index='country', columns='ensemble', values='capacity_per_normalized_mw').reindex(countries)
        dd[risk] = 100 * (pv.clustered - pv.dispersed)
    ax.bar(x - 0.18, dd[0.0], width=0.34, color='#276F9E', alpha=0.82, lw=0, zorder=3)
    ax.bar(x + 0.18, dd[0.5], width=0.34, color='#C58922', alpha=0.82, lw=0, zorder=3)
    ax.axhline(0, color=S.INK, lw=0.6, zorder=3.5)
    ax.set_ylim(-100, 100); ax.set_yticks([-100, -50, 0, 50, 100])
    ax.set_xlim(-0.7, 14.7); ax.set_xticks(x); ax.set_xticklabels(countries, fontsize=5.0, rotation=90)
    ax.tick_params(axis='x', length=0); S.grid(ax, 'y')
    ax.set_ylabel('A-CAES, ext. − disp. (pp)')

    cv.letter(X[2] - 13.0, top2 - 4.2, 'e')
    ax = cv.ax(X[2], top2, W, h2, 'e')
    steps = [2, 4, 6]
    for risk, (lab, col, mk) in RISK.items():
        q = s[s.step.isin(steps) & s.risk.eq(risk) & s.ensemble.eq('combined') & s['mode'].eq('diversified')]
        pv = q.pivot(index='country', columns='step', values='total_power_capacity_per_normalized_mw').reindex(columns=steps) * 100
        q25, q50, q75 = pv.quantile(0.25), pv.median(), pv.quantile(0.75)
        ax.fill_between(steps, q25, q75, color=col, alpha=0.18, lw=0, zorder=2)
        for v in pv.to_numpy():
            ax.plot(steps, v, color=col, lw=0.38, alpha=0.25, zorder=1.5)
        ax.plot(steps, q50, color=col, lw=1.45, marker=mk, ms=3.4, mec='white', mew=0.25, zorder=4)
    ax.set_xlim(1.6, 6.4); ax.set_xticks(steps); ax.set_ylim(0, 1000); ax.set_yticks([0, 250, 500, 750, 1000])
    S.grid(ax, 'y')
    ax.set_xlabel('Planning step (h)'); ax.set_ylabel('Total power (% gas)')

    # Row 3: f (Near-optimal 1 h A-CAES nested bullet intervals in CZ, RO, ES)
    n = pd.read_csv(CT / 'nearoptimal_intervals.csv')
    n = n[(n.technology == 'caes_24h') & (n.risk == 0.0)]
    top3, h3 = 113.5, 29.0
    cv.letter(1.5, top3 - 4.2, 'f')
    for k, cty in enumerate(['CZ', 'RO', 'ES']):
        ax = cv.ax(X[k], top3, W, h3, f'f{k}')
        S.grid(ax, 'x')
        g = n[n.country == cty]
        for j, (ens, lab, col) in enumerate(ENS):
            v1 = g[(g.epsilon == 0.01) & (g.ensemble == ens)].iloc[0]
            v01 = g[(g.epsilon == 0.001) & (g.ensemble == ens)].iloc[0]
            v0 = g[(g.epsilon == 0.0) & (g.ensemble == ens)].iloc[0]
            yy = 3 - j
            lo1, hi1 = 100 * v1['min'], 100 * v1['max']
            lo01, hi01 = 100 * v01['min'], 100 * v01['max']
            lo0, hi0 = 100 * v0['min'], 100 * v0['max']
            ax.barh(yy, max(hi1 - lo1, 2.0), left=lo1, height=0.58, color=col, alpha=0.25, lw=0, zorder=2)
            ax.barh(yy, max(hi01 - lo01, 1.4), left=lo01, height=0.32, color=col, alpha=0.62, lw=0, zorder=3)
            ax.scatter([0.5 * (lo0 + hi0)], [yy], s=13, color=col, edgecolor='white', lw=0.35, zorder=4)
        hi = 100 * g['max'].max()
        ax.set_xlim(0, np.ceil(hi / 50) * 50 + 12)
        ax.set_ylim(-0.6, 3.6)
        ax.set_yticks([3, 2, 1, 0])
        ax.set_yticklabels([x_[1] for x_ in ENS] if k == 0 else [], fontsize=5.0)
        ax.tick_params(axis='y', length=0)
        ax.text(0.5, 1.03, cty, transform=ax.transAxes, ha='center', va='bottom', fontsize=5.6, fontweight='bold', color=S.INK)
        ax.set_xlabel('A-CAES power (% gas)')

    lf = cv.ax(19.0, top3 + h3 + 10.5, 161.0, 3.2, 'legf')
    lf.set_xlim(0, 161); lf.set_ylim(0, 1); lf.axis('off')
    lf.add_patch(Rectangle((0.0, 0.2), 4.5, 0.6, facecolor='#19646B', alpha=0.25, lw=0))
    lf.text(5.5, 0.5, '±1.0% cost tolerance', va='center', fontsize=5.1)
    lf.add_patch(Rectangle((46.0, 0.25), 4.5, 0.5, facecolor='#19646B', alpha=0.65, lw=0))
    lf.text(51.5, 0.5, '±0.1% cost tolerance', va='center', fontsize=5.1)
    lf.scatter([96.5], [0.5], s=13, color='#19646B', edgecolor='white', lw=0.35)
    lf.text(98.5, 0.5, '0% optimum (1 h step)', va='center', fontsize=5.1)

    pd.DataFrame({'country': countries, 'cost_change_pct_mean': cc[0.0].to_numpy(), 'cost_change_pct_risk': cc[0.5].to_numpy(),
                  'caes_extended_minus_dispersed_pp_mean': dd[0.0].to_numpy(), 'caes_extended_minus_dispersed_pp_risk': dd[0.5].to_numpy()}).to_csv(
        S.SD / 'appendix_capacity_countries.csv', index=False)
    c[c.step.eq(4) & c.ensemble.eq('combined') & c['mode'].eq('diversified')][['country', 'risk', 'config_id', 'capacity_per_normalized_mw']].to_csv(
        S.SD / 'appendix_capacity_procurement.csv', index=False)
    n.to_csv(S.SD / 'appendix_capacity_nearoptimal_caes.csv', index=False)
    S.save(cv, NAME, rows=[['a', 'b'], ['c', 'd', 'e'], ['f0', 'f1', 'f2']], cols=[['c', 'f0'], ['d', 'f1'], ['e', 'f2']],
           exempt=[dict(panels=['legt', 'legf', 'a_map'], checks=['all'], reason='Legend strips and European choropleth map inset')],
           audit=dict(cost_change_median=float(cc[0.0].median()), caes_delta_median=float(dd[0.0].median())))


if __name__ == '__main__':
    main()
