"""Supplementary Figure S1 (appendix_figure_provincial_allocation).
Chinese provincial charging-reconstruction robustness across 310 province-years (a-c)
and single-year 26-province gas-peaking intraday task-allocation sensitivity (d-f).

Upgraded diverse visual styles (zero lollipop charts, zero descriptive titles inside panels, zero middle dots / '+' / arrows):
a  Paired 7-baseline (q5-q50) bar-and-trajectory comparison with shaded 96.1-99.0% agreement header
b  Split half-violin + interquartile forest of 2x2x2 factorial contrasts across 178 positive-supply province-years
c  Diverging horizontal endpoint delta-B bars across 7 non-H2 technologies with direct value callouts
d  5-shape horizontal ridgeline + quantile box landscape of incremental return across 26 gas-peaking provinces
e  5-component diverging stacked cost-bridge bars with net-B diamond markers
f  Paired A-CAES half-violin density + 4 h LFP median slope comparison across the 5 intraday task shapes
"""
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

import style as S
import data as D
import econ as E

H = 136.0
NAME = 'appendix_figure_provincial_allocation'
VAR = ['prop', 'flat24', 'cap1400', 'top12', 'top8']
LAB = {
    'prop': 'Typical-curve proportional',
    'flat24': '24 h average',
    'cap1400': 'Capacity-limited',
    'top12': 'Highest 12 h',
    'top8': 'Highest 8 h',
}
VCOL = {'prop': S.COLOR['provincial'], 'flat24': '#607178', 'cap1400': '#276F9E', 'top12': '#7A5C8F', 'top8': '#C9543B'}
PROP_C, CLIP_C = '#C58922', '#19646B'


def load_gas26():
    r = pd.read_csv(D.V86 / 'analysis/provincial_intraday/results/intraday_records.csv', dtype={'uid': str})
    r = r[r.variant.isin(VAR)].copy()
    assert r.groupby('variant').size().to_dict() == {v: 26 for v in VAR}
    main26 = D.rd(D.PROV / 'provincial_real.csv').set_index('uid')
    x = r[r.variant == 'prop'].set_index('uid')
    assert np.allclose(x.B, main26.loc[x.index, 'B'], atol=1e-9)
    r['win36'] = r.B >= 0
    return r


def factorial_contrasts(path, keep_uids):
    d = D.rd(path)
    d = d[(d.order == 'real') & (d.uid.isin(keep_uids))]
    wide = d.pivot_table(index='uid', columns=['energy_multiplier', 'charge_power_multiplier', 'access'], values='B')
    col = lambda e, p, a: wide[(e, p, a)]
    return pd.DataFrame({
        'surplus_energy_4x': sum(col(4, p, a) - col(1, p, a) for p in (1, 4) for a in ('observed', 'expanded')) / 4,
        'charge_power_4x': sum(col(e, 4, a) - col(e, 1, a) for e in (1, 4) for a in ('observed', 'expanded')) / 4,
        'expanded_access': sum(col(e, p, 'expanded') - col(e, p, 'observed') for e in (1, 4) for p in (1, 4)) / 4,
        'power_x_access': sum(col(e, 4, 'expanded') - col(e, 1, 'expanded')
                              - col(e, 4, 'observed') + col(e, 1, 'observed') for e in (1, 4)) / 2,
    })


def endpoint_dB(cat_path):
    c = D.rd(cat_path)
    out = {}
    for t in ['caes', 'vrfb', 'molten_salt_tes', 'fecr_flow', 'li_lfp', 'gravity', 'laes']:
        z = c[c.tech == t]
        hs = sorted(z.duration_h.unique())
        lo = z[z.duration_h == hs[0]].set_index('uid').B
        hi = z[z.duration_h == hs[-1]].set_index('uid').B.reindex(lo.index)
        out[t] = float((hi - lo).mean())
    return out


def main():
    S.setup()
    cv = S.Canvas(H)

    lg1 = cv.ax(19.0, 2.2, 161.0, 3.4, 'leg_top')
    lg1.set_xlim(0, 161); lg1.set_ylim(0, 1); lg1.axis('off')
    lg1.add_patch(Rectangle((0.0, 0.18), 3.2, 0.64, facecolor=PROP_C, lw=0))
    lg1.text(4.0, 0.5, 'Proportional curtailment allocation (main text, n = 310)', va='center', fontsize=5.3, fontweight='bold', color=S.INK)
    lg1.add_patch(Rectangle((85.0, 0.18), 3.2, 0.64, facecolor=CLIP_C, lw=0))
    lg1.text(89.0, 0.5, 'Monthly peak-clipping allocation (n = 310)', va='center', fontsize=5.3, fontweight='bold', color=S.INK)

    top1, h1 = 12.5, 42.0
    X1, W1 = [19.0, 77.5, 136.0], 44.0

    # a: 7 load baselines (q5..q50) real bars & MC ribbon/trajectories
    q7 = pd.read_csv(D.V86 / 'revision/中日基线与蒙特卡洛实验报告_20261118/中国_两种充电方法_七个负荷基线_混合组合胜出比例.csv')
    cv.letter(1.5, top1 - 4.2, 'a')
    axa = cv.ax(X1[0], top1, W1, h1, 'a')
    S.grid(axa, 'y')
    xs = np.arange(len(q7))
    qlabs = q7['负荷基线'].tolist()
    p_real = q7['原始日期按出力比例分配胜出率(%)'].to_numpy(float)
    c_real = q7['原始日期月度削峰胜出率(%)'].to_numpy(float)
    p_mc = q7['蒙特卡洛按出力比例分配胜出率(%)'].to_numpy(float)
    c_mc = q7['蒙特卡洛月度削峰胜出率(%)'].to_numpy(float)
    agree = 100.0 * (310.0 - q7['原始日期仅按出力比例分配胜出数'] - q7['原始日期仅月度削峰胜出数']) / 310.0

    axa.fill_between(xs, np.minimum(p_real, c_real), np.maximum(p_real, c_real), color='#DCE8EA', alpha=0.6, lw=0, zorder=1)
    axa.bar(xs - 0.18, p_real, width=0.33, facecolor=PROP_C, alpha=0.88, lw=0, zorder=2.5)
    axa.bar(xs + 0.18, c_real, width=0.33, facecolor=CLIP_C, alpha=0.88, lw=0, zorder=2.5)
    axa.plot(xs - 0.18, p_mc, color='#7D5310', lw=1.0, ls='--', marker='o', ms=2.3, zorder=4)
    axa.plot(xs + 0.18, c_mc, color='#0D3B43', lw=1.0, ls='--', marker='s', ms=2.3, zorder=4)
    axa.set_xticks(xs, qlabs, fontsize=5.0)
    axa.set_xlim(-0.6, len(xs) - 0.4)
    axa.set_ylim(0, 27.5); axa.set_yticks([0, 10, 20])
    axa.set_xlabel('Load baseline quantile')
    axa.set_ylabel('Hybrid-advantaged share (%)')
    axa.text(0.03, 0.95, f'Bars: real; dashed: 1,000 MC\nAgreement: {agree.min():.1f}–{agree.max():.1f}%',
             transform=axa.transAxes, ha='left', va='top', fontsize=5.0, color=S.INK, linespacing=1.15)

    # b: Split half-violin + IQR forest across 178 positive-curtailment province-years
    pos_uids = D.cn_positive_uids()
    fx_prop = factorial_contrasts(D.CN_SUITE / 'china_factorial.csv.gz', pos_uids)
    fx_clip = factorial_contrasts(D.CN_SUITE_CLIP / 'china_factorial.csv.gz', pos_uids)
    cv.letter(X1[1] - 12.5, top1 - 4.2, 'b')
    axb = cv.ax(X1[1], top1, W1, h1, 'b')
    S.grid(axb, 'x')
    axb.axvspan(0, 14.5, facecolor=S.WIN_PALE, alpha=0.55, zorder=0)
    fac_cols = ['surplus_energy_4x', 'charge_power_4x', 'expanded_access', 'power_x_access']
    fac_labs = ['Surplus\n× 4', 'Power\n× 4', 'All task-\nfree h', 'Power ×\naccess']
    xg_b = np.linspace(-9.2, 14.0, 140)
    for yi, fc in enumerate(fac_cols):
        vp = np.clip(fx_prop[fc].to_numpy(float), -9.2, 14.0)
        vc = np.clip(fx_clip[fc].to_numpy(float), -9.2, 14.0)
        for sgn, arr_v, col in [(-1.0, vp, PROP_C), (1.0, vc, CLIP_C)]:
            if np.std(arr_v) > 0.18:
                kd = gaussian_kde(arr_v, bw_method=0.30)(xg_b)
                kd = kd / kd.max() * 0.30
                axb.fill_between(xg_b, yi, yi + sgn * kd, facecolor=col, alpha=0.32, edgecolor=col, lw=0.50, zorder=2.5)
            q10, q25, q50, q75, q90 = np.quantile(arr_v, [0.10, 0.25, 0.50, 0.75, 0.90])
            yy = yi + sgn * 0.12
            axb.plot([q10, q90], [yy, yy], color=col, lw=0.75, zorder=3)
            axb.plot([q25, q75], [yy, yy], color=col, lw=2.1, solid_capstyle='butt', zorder=3.5)
            axb.scatter([q50], [yy], s=10, color='white', edgecolor=col, lw=0.75, zorder=4.5)
    axb.axvline(0, color=S.INK, lw=0.65, zorder=2)
    axb.set_ylim(3.65, -0.65)
    axb.set_yticks(range(4), fac_labs, fontsize=5.0)
    axb.tick_params(axis='y', length=0)
    axb.set_xlim(-9.5, 14.2); axb.set_xticks([-8, 0, 8])
    axb.set_xlabel(f'ΔB, positive supply ({S.UNIT})')
    axb.text(1.0, 1.02, 'All h med: 0.13 vs 2.79', transform=axb.transAxes, ha='right', va='bottom',
             fontsize=5.0, fontweight='bold', color=CLIP_C)

    # c: Diverging horizontal endpoint ΔB bars across 7 non-H2 technologies
    dB_prop = endpoint_dB(D.CN_SUITE / 'china_catalogue58.csv.gz')
    dB_clip = endpoint_dB(D.CN_SUITE_CLIP / 'china_catalogue58.csv.gz')
    cv.letter(X1[2] - 12.5, top1 - 4.2, 'c')
    axc = cv.ax(X1[2], top1, W1, h1, 'c')
    S.grid(axc, 'x')
    axc.axvspan(0, 16.5, facecolor=S.WIN_PALE, zorder=0)
    t_order = ['caes', 'molten_salt_tes', 'vrfb', 'fecr_flow', 'li_lfp', 'gravity', 'laes']
    t_names = ['A-CAES', 'TES', 'VRFB', 'Fe–Cr', 'LFP', 'Gravity', 'LAES']
    yk = np.arange(len(t_order))
    axc.barh(yk - 0.17, [dB_prop[t] for t in t_order], height=0.32, facecolor=PROP_C, lw=0, zorder=3)
    axc.barh(yk + 0.17, [dB_clip[t] for t in t_order], height=0.32, facecolor=CLIP_C, lw=0, zorder=3)
    axc.text(dB_prop['caes'] + 0.8, -0.17, f"{dB_prop['caes']:.1f}", va='center', ha='left', fontsize=5.0, fontweight='bold', color='#8A5D0E')
    axc.axvline(0, color=S.INK, lw=0.65, zorder=4)
    axc.set_ylim(len(t_order) - 0.4, -0.6)
    axc.set_yticks(yk, t_names, fontsize=5.0)
    axc.tick_params(axis='y', length=0)
    axc.set_xlim(-38, 16.5); axc.set_xticks([-30, -15, 0, 15])
    axc.set_xlabel(f'Endpoint ΔB ({S.UNIT})')

    # Row 2 (d, e, f): Legacy single-year 26-province gas-peaking case across 5 intraday task shapes
    r = load_gas26()
    top2, h2 = 73.5, 39.0
    X2, W2 = [35.0, 86.0, 137.0], 43.0
    rows_idx = list(range(len(VAR)))

    def rows_axis(a, labels):
        a.set_ylim(len(VAR) - 0.38, -0.78)
        a.set_yticks(rows_idx)
        a.set_yticklabels([LAB[v] for v in VAR] if labels else [], fontsize=5.15)
        a.tick_params(axis='y', length=0)
        if labels:
            for t_lbl, v in zip(a.get_yticklabels(), VAR):
                t_lbl.set_color(VCOL[v])
                t_lbl.set_fontweight('bold')
        a.spines['left'].set_visible(False)

    # d: Horizontal ridgeline + IQR box + markers across the 5 shapes
    cv.letter(1.5, top2 - 4.2, 'd')
    ad = cv.ax(X2[0], top2, W2, h2, 'd')
    S.grid(ad, 'x')
    ad.axvspan(3.6, 21.5, facecolor=S.WIN_PALE, alpha=0.55, zorder=0)
    xg_d = np.linspace(-3.8, 21.0, 140)
    for i, v in enumerate(VAR):
        z = r[r.variant == v]
        x = z.roi_2035.to_numpy(float)
        col = VCOL[v]
        kd = gaussian_kde(x, bw_method=0.36)(xg_d)
        kd = kd / kd.max() * 0.56
        ad.fill_between(xg_d, i, i - kd, facecolor=col, alpha=0.28, edgecolor=col, lw=0.60, zorder=2.5)
        q25, m_v, q75 = np.quantile(x, [0.25, 0.50, 0.75])
        ad.plot([q25, q75], [i + 0.10, i + 0.10], color=col, lw=1.8, solid_capstyle='butt', zorder=3.5)
        ad.scatter([m_v], [i + 0.10], s=11, color='white', edgecolor=S.INK, lw=0.7, zorder=4.5)
    ad.axvline(3.6, color=S.WIN, lw=0.70, ls=(0, (3, 2)), zorder=3)
    ad.set_xlim(-4, 21.5); ad.set_xticks([0, 5, 10, 15, 20])
    rows_axis(ad, True)
    ad.set_xlabel('Incremental return (%)')
    ad.text(1.0, 1.02, '3.6% hurdle (23–26/26 win)', transform=ad.transAxes, ha='right', va='bottom', fontsize=5.0, color=S.WIN)

    # e: Mean B and 5 cost components
    cv.letter(81.5, top2 - 4.2, 'e')
    ae = cv.ax(X2[1], top2, W2, h2, 'e')
    m = r.groupby('variant')[S.COMP + ['B']].mean().reindex(VAR)
    for i, v in enumerate(VAR):
        pos = neg = 0.0
        for c_k in S.COMP:
            val = m.loc[v, c_k]
            left = pos if val >= 0 else neg + val
            ae.barh(i, abs(val), left=left, height=0.58, color=S.COMP_COLOR[c_k], lw=0, zorder=3)
            if val >= 0:
                pos += val
            else:
                neg += val
        ae.scatter(m.loc[v, 'B'], i, marker='D', s=11, color=S.INK, edgecolor='white', lw=0.4, zorder=5)
        ae.text(max(pos, 0) + 5, i, f'{m.loc[v, "B"]:.1f}'.replace('-', '−'), va='center', ha='left', fontsize=5.05, fontweight='bold')
    ae.axvline(0, color=S.INK, lw=0.6, zorder=4)
    ae.set_xlim(-60, 200); ae.set_xticks([-50, 0, 50, 100, 150])
    rows_axis(ae, False)
    ae.set_xlabel(f'Mean B ({S.UNIT})')
    for k, c_k in enumerate(S.COMP):
        yy = top2 + h2 + 10.5 + (k % 3) * 2.6
        xx = 86.0 + (k // 3) * 24.5
        cv.fig.patches.append(Rectangle((xx / S.WIDTH_MM, 1 - (yy + 1.9) / H), 2.6 / S.WIDTH_MM, 1.8 / H,
                                        transform=cv.fig.transFigure, facecolor=S.COMP_COLOR[c_k], lw=0))
        cv.text(xx + 3.4, yy + 0.1, S.COMP_LABEL[c_k], fontsize=5.05, va='top')
    cv.fig.lines.append(Line2D([(86.0 + 24.5 + 1.3) / S.WIDTH_MM], [1 - (top2 + h2 + 10.5 + 2 * 2.6 + 1.0) / H],
                               marker='D', markersize=3.0, color=S.INK, transform=cv.fig.transFigure))
    cv.text(86.0 + 24.5 + 3.4, top2 + h2 + 10.5 + 2 * 2.6 + 0.1, 'Total B', fontsize=5.05, va='top')

    # f: Paired A-CAES half-violin density + 4 h LFP median connected slope
    cv.letter(132.5, top2 - 4.2, 'f')
    af = cv.ax(X2[2], top2, W2, h2, 'f')
    S.grid(af, 'x')
    xg_f = np.linspace(0.0, 125.0, 140)
    caes_meds, lfp_meds = [], []
    for i, v in enumerate(VAR):
        z = r[r.variant == v]
        xv = z.efc_acaes.to_numpy(float)
        col = VCOL[v]
        if np.std(xv) > 0.6:
            kd = gaussian_kde(xv, bw_method=0.35)(xg_f)
            kd = kd / kd.max() * 0.50
            af.fill_between(xg_f, i, i - kd, facecolor=col, alpha=0.28, edgecolor=col, lw=0.55, zorder=2.5)
        cm, lm = float(z.efc_acaes.median()), float(z.efc_lfp.median())
        caes_meds.append(cm); lfp_meds.append(lm)
        af.plot([cm, lm], [i, i], color='#9BAAB2', lw=0.80, ls=':', zorder=3)
        af.scatter([cm], [i], marker='s', s=12, color=col, edgecolor='white', lw=0.35, zorder=4.5)
        af.scatter([lm], [i], marker='D', s=13, facecolor='white', edgecolor=S.INK, lw=0.65, zorder=5)
    af.set_xlim(0, 125); af.set_xticks([0, 30, 60, 90, 120])
    rows_axis(af, False)
    af.set_xlabel('Equivalent full cycles (yr⁻¹)')
    af.text(1.0, 1.02, '■ A-CAES med; ◇ 4 h LFP med', transform=af.transAxes, ha='right', va='bottom', fontsize=5.0, color=S.MUTED)

    q7.to_csv(S.SD / 'appendix_provincial_charging_7baselines.csv', index=False)
    pd.concat([fx_prop.assign(charging='proportional'), fx_clip.assign(charging='monthly_clipped')]).reset_index().to_csv(
        S.SD / 'appendix_provincial_charging_factorial.csv', index=False)
    keep = ['uid', 'province', 'variant', 'B', 'roi_2035', 'win36', 'efc_acaes', 'efc_lfp', 'chargeable_hours', 'task_hours'] + S.COMP
    r[keep].to_csv(S.SD / 'appendix_provincial_allocation_records.csv', index=False)

    S.save(cv, NAME, rows=[['a', 'b', 'c'], ['d', 'e', 'f']],
           exempt=[dict(panels=['leg_top'], checks=['all'], reason='Top legend strip')],
           audit=dict(records_310=310, positive_supply_records=178, gas26_records=26,
                      agreement_pct_range=[float(agree.min()), float(agree.max())]))


if __name__ == '__main__':
    main()
