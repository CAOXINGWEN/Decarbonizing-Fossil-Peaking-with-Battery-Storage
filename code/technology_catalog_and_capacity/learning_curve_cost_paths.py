"""Extended Data Figure 7 (appendix_figure_cost_paths). Capital learning trajectories, cost reduction spans, annualized fixed-cost gap, and provenance across all 58 configurations."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import parameters as P

sys.path.insert(0, str(P.V86 / 'figures/code'))
import style as S

TECH = ['li_lfp', 'vrfb', 'fecr_flow', 'gravity', 'molten_salt_tes', 'caes', 'laes', 'h2_tank']
TLAB = {
    'li_lfp': 'LFP',
    'vrfb': 'VRFB',
    'fecr_flow': 'Fe–Cr flow',
    'gravity': 'Gravity',
    'molten_salt_tes': 'Molten-salt TES',
    'caes': 'A-CAES',
    'laes': 'LAES',
    'h2_tank': 'Tank H₂ (200 bar)',
}
TCOL = {
    'li_lfp': '#276F9E',
    'vrfb': '#3A8F99',
    'fecr_flow': '#7A5C8F',
    'gravity': '#607178',
    'molten_salt_tes': '#C58922',
    'caes': '#19646B',
    'laes': '#9E5B3E',
    'h2_tank': '#C9543B',
}
TMARK = {
    'li_lfp': 'o',
    'vrfb': 's',
    'fecr_flow': 'D',
    'gravity': '^',
    'molten_salt_tes': 'v',
    'caes': 'o',
    'laes': 's',
    'h2_tank': 'D',
}
PROVENANCE = {
    'li_lfp': (4, 0, 0, 0),
    'vrfb': (1, 2, 0, 0),
    'fecr_flow': (0, 3, 0, 0),
    'gravity': (1, 2, 0, 0),
    'molten_salt_tes': (1, 3, 1, 0),
    'caes': (2, 3, 4, 0),
    'laes': (1, 4, 4, 0),
    'h2_tank': (0, 0, 0, 22),
}


def main():
    P.export()
    rows, compare = [], []
    for cid, c in P.catalogue().items():
        for y in (2025, 2030, 2035):
            x = P.cost_components(cid, y)
            rows.append(dict(
                tech=c['tech_id'],
                config_id=cid,
                cost_year=y,
                duration_h=c['duration_h'],
                power_usd_kw=x['power_capex_usd_kw'],
                energy_usd_kwh=x['energy_capex_usd_kwh'],
                total_capex_usd_kwh=x['power_capex_usd_kw'] / c['duration_h'] + x['energy_capex_usd_kwh'],
                fixed_usd_kw_year=x['annual_fixed_cost_usd_kw_year'],
            ))
    d = pd.DataFrame(rows)
    assert (d.duration_h > 0).all() and (d.total_capex_usd_kwh > 0).all()
    d.to_csv(P.WORK / 'tables/cost_paths.csv', index=False)
    d.to_csv(S.SD / 'appendix_cost_paths.csv', index=False)
    for cid in ('caes_24h', 'h2_tank_24h', 'h2_tank_168h'):
        for y in (2025, 2030, 2035):
            a = P.OLD.cost_components(cid.replace('h2_tank_', 'h2_cavern_'), y, P.RATE)
            b = P.cost_components(cid, y)
            for k in ('power_capex_usd_kw', 'energy_capex_usd_kwh', 'fixed_om_usd_kw_year', 'calendar_life_years', 'annual_fixed_cost_usd_kw_year'):
                compare.append(dict(config=cid, cost_year=y, parameter=k, original=a[k], updated=b[k], difference=b[k] - a[k]))
    pd.DataFrame(compare).to_csv(P.WORK / 'parameters/parameter_comparison.csv', index=False)

    S.setup()
    cv = S.Canvas(156.0)
    d35 = d[d.cost_year == 2035]
    d25 = d[d.cost_year == 2025]

    # Panel a: 2x4 technology grid showing 2025 -> 2035 installed capital cost with 2035 envelope silhouette
    cv.letter(2.0, 5.0, 'a')
    xs_a = [16.5, 43.0, 69.5, 96.0]
    ys_a = [12.0, 53.0]
    w_a, h_a = 23.0, 28.0
    a_ids = []
    for idx, t in enumerate(TECH):
        r, c = divmod(idx, 4)
        gid = f'a{idx + 1}'
        a_ids.append(gid)
        ax = cv.ax(xs_a[c], ys_a[r], w_a, h_a, gid)
        ax.set_facecolor('#FAFDFD')
        for ot in TECH:
            if ot == t:
                continue
            oz = d35[d35.tech == ot].sort_values('duration_h')
            ax.plot(oz.duration_h, oz.total_capex_usd_kwh, color='#DEE6E9', lw=0.65, zorder=1)
        z25 = d25[d25.tech == t].sort_values('duration_h')
        z35 = d35[d35.tech == t].sort_values('duration_h')
        col, mk = TCOL[t], TMARK[t]
        ax.fill_between(z35.duration_h, z35.total_capex_usd_kwh, z25.total_capex_usd_kwh, color=col, alpha=0.22, lw=0, zorder=2)
        ax.plot(z25.duration_h, z25.total_capex_usd_kwh, color=col, lw=0.85, ls='--', marker=mk, ms=2.1, mfc='white', mew=0.65, zorder=3)
        ax.plot(z35.duration_h, z35.total_capex_usd_kwh, color=col, lw=1.15, marker=mk, ms=2.2, zorder=4)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(3.1, 950); ax.set_ylim(24, 720)
        ax.set_xticks([4, 24, 168, 720])
        ax.set_xticklabels(['4', '24', '168', '720'] if r == 1 else [], fontsize=5.1)
        ax.set_yticks([30, 100, 300])
        ax.set_yticklabels(['30', '100', '300'] if c == 0 else [], fontsize=5.2)
        ax.minorticks_off()
        S.grid(ax, 'both')
        ax.text(0.5, 1.07, f'{TLAB[t]} (n={len(z35)})', transform=ax.transAxes, ha='center', va='bottom',
                fontsize=5.2, fontweight='bold', color=col)
    cv.text(5.8, 47.0, 'Installed capital cost (USD kWh⁻¹)', rotation=90, va='center', ha='center', fontsize=5.9)
    cv.text(67.5, 89.5, 'Discharge duration (h)', ha='center', va='center', fontsize=5.9)
    lga = cv.ax(16.5, 92.5, 86.0, 4.0, 'leg_a')
    lga.set_xlim(0, 86); lga.set_ylim(0, 1); lga.axis('off')
    lga.plot([1, 10], [0.5, 0.5], color=S.INK, lw=0.85, ls='--', marker='o', ms=2.3, mfc='white')
    lga.text(12, 0.5, '2025 capital cost', va='center', fontsize=5.3)
    lga.plot([42, 51], [0.5, 0.5], color=S.INK, lw=1.15, marker='o', ms=2.3)
    lga.text(53, 0.5, '2035 capital cost', va='center', fontsize=5.3)

    # Panel b: Horizontal mean bar + min-max range span across duration configurations (no lollipop!)
    cv.letter(123.0, 5.0, 'b')
    axb = cv.ax(145.5, 12.0, 34.5, 28.0, 'b')
    mg = d25.merge(d35, on=['tech', 'config_id', 'duration_h'], suffixes=('_25', '_35'))
    mg['pct_red'] = 100 * (mg.total_capex_usd_kwh_25 - mg.total_capex_usd_kwh_35) / mg.total_capex_usd_kwh_25
    y_pos = np.arange(len(TECH))[::-1]
    for yi, t in zip(y_pos, TECH):
        sub = mg[mg.tech == t]
        mn, md, mx = float(sub.pct_red.min()), float(sub.pct_red.mean()), float(sub.pct_red.max())
        axb.barh(yi, md, height=0.56, facecolor=TCOL[t], alpha=0.28, edgecolor=TCOL[t], lw=0.60, zorder=2)
        if mx - mn > 0.8:
            axb.plot([mn, mx], [yi, yi], color=TCOL[t], lw=1.6, solid_capstyle='round', zorder=3.5)
        axb.scatter([md], [yi], s=11, marker=TMARK[t], color=TCOL[t], edgecolor='white', lw=0.3, zorder=4)
    axb.set_yticks(y_pos)
    axb.set_yticklabels([TLAB[t] for t in TECH], fontsize=5.05)
    axb.tick_params(axis='y', length=0)
    axb.set_xlim(0, 78)
    axb.set_xticks([0, 25, 50, 75])
    axb.set_xlabel('2025–2035 drop (%)')
    S.grid(axb, 'x')

    # Panel c: 2035 annualized fixed cost (USD kW-1 yr-1) with shaded benchmark premium band
    cv.letter(123.0, 46.0, 'c')
    axc = cv.ax(145.5, 53.0, 34.5, 28.0, 'c')
    lfp4 = d35[d35.config_id == 'li_lfp_4h'].iloc[0]
    caes24 = d35[d35.config_id == 'caes_24h'].iloc[0]
    axc.axhspan(lfp4.fixed_usd_kw_year, caes24.fixed_usd_kw_year, facecolor=S.WIN_PALE, zorder=0)
    for t in TECH:
        z = d35[d35.tech == t].sort_values('duration_h')
        axc.plot(z.duration_h, z.fixed_usd_kw_year, color=TCOL[t], marker=TMARK[t], ms=2.0, lw=1.0, zorder=3)
    axc.scatter([lfp4.duration_h, caes24.duration_h], [lfp4.fixed_usd_kw_year, caes24.fixed_usd_kw_year],
                s=28, fc='none', ec=S.INK, lw=0.85, zorder=5)
    axc.annotate(f'LFP 4 h: {lfp4.fixed_usd_kw_year:.1f}', xy=(lfp4.duration_h, lfp4.fixed_usd_kw_year),
                 xytext=(12.0, 14.0), fontsize=5.0, va='center', ha='left',
                 arrowprops=dict(arrowstyle='-', color=S.MUTED, lw=0.45, shrinkA=1, shrinkB=3))
    axc.annotate(f'CAES 24 h\n{caes24.fixed_usd_kw_year:.1f}', xy=(caes24.duration_h, caes24.fixed_usd_kw_year),
                 xytext=(48.0, 26.5), fontsize=5.0, fontweight='bold', color='#19646B', va='center', ha='left',
                 arrowprops=dict(arrowstyle='-', color='#19646B', lw=0.45, shrinkA=1, shrinkB=3))
    axc.set_xscale('log'); axc.set_yscale('log')
    axc.set_xlim(3.1, 950); axc.set_ylim(10, 3000)
    axc.set_xticks([4, 24, 168, 720]); axc.set_xticklabels(['4', '24', '168', '720'], fontsize=5.1)
    axc.set_yticks([10, 100, 1000]); axc.set_yticklabels(['10', '100', '1,000'], fontsize=5.1)
    axc.minorticks_off()
    S.grid(axc, 'y')
    axc.set_xlabel('Duration (h)')
    axc.set_ylabel('2035 fixed cost\n(USD kW⁻¹ yr⁻¹)')

    # Panel d: Configuration provenance Marimekko/stacked bars across all 58 nodes
    cv.letter(2.0, 101.0, 'd')
    axd = cv.ax(31.0, 111.0, 149.0, 33.0, 'd')
    prov_cols = ['#19646B', '#7EB5B0', '#C58922', '#C9543B']
    prov_labs = ['Direct duration anchor', 'Released configuration', 'Component scaling/interpolation', '200 bar surface tank + PNNL power']
    yp = np.arange(len(TECH))[::-1]
    for yi, t in zip(yp, TECH):
        counts = PROVENANCE[t]
        left = 0
        for cnt, pcol in zip(counts, prov_cols):
            if cnt == 0:
                continue
            axd.barh(yi, cnt, left=left, height=0.66, color=pcol, edgecolor='white', lw=0.65, zorder=3)
            txt_col = 'white' if pcol in ('#19646B', '#C9543B') else S.INK
            axd.text(left + cnt / 2, yi, str(cnt), ha='center', va='center', fontsize=5.2, color=txt_col, fontweight='bold')
            left += cnt
    axd.set_yticks(yp)
    axd.set_yticklabels([TLAB[t] for t in TECH], fontsize=5.3)
    axd.tick_params(axis='y', length=0)
    axd.set_xlim(0, 23.5)
    axd.set_xticks([0, 4, 8, 12, 16, 20, 22])
    axd.set_xlabel('Evaluated duration configurations (total = 58)')
    S.grid(axd, 'x')
    lgd = cv.ax(31.0, 103.5, 149.0, 4.0, 'leg_d')
    lgd.set_xlim(0, 149); lgd.set_ylim(0, 1); lgd.axis('off')
    offs = [0, 32, 65, 106]
    for off, pcol, plab in zip(offs, prov_cols, prov_labs):
        lgd.add_patch(Rectangle((off, 0.22), 3.2, 0.56, facecolor=pcol, lw=0))
        lgd.text(off + 4.2, 0.5, plab, va='center', fontsize=5.15)

    S.save(
        cv,
        'appendix_figure_cost_paths',
        rows=[a_ids[:4], a_ids[4:]],
        cols=[[a_ids[i], a_ids[i + 4]] for i in range(4)] + [['b', 'c']],
        exempt=[
            dict(panels=['leg_a', 'leg_d', 'b', 'c', 'd'], checks=['all'],
                 reason='Panel a grid and right/bottom panels have distinct widths for technology labels and provenance bars'),
        ],
        audit=dict(configurations=58, years=[2025, 2030, 2035], unchanged_nonhydrogen_configurations=36,
                   hydrogen_medium='200 bar surface Type-I steel tanks'),
    )


if __name__ == '__main__':
    main()
