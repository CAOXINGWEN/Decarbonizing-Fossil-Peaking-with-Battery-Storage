"""Figure 3. Event start and end placement chronology and portfolio reversals across 11,054,000 placements.

7 bespoke panels (a-g):
a  Chronology rearrangement calendar barcode (France 2016: real dates vs 10th, 50th and 90th percentile placements)
b  Ranked real-date incremental return vs 5th-95th placement envelope and majority-reversal ribbon
c  Continuous discount-rate (0-10%) frontier trajectories (real vs random + shaded chronology uplift + cost-year inset)
d  Cross-regime 7-baseline (q5-q50) real vs random portfolio trajectories (including China proportional vs monthly peak-clipped)
e  Empirical cumulative calibration curves of real-date return percentile against the 45-degree uniform benchmark
f  2D joint density and quantile trajectory of delta long charging windows versus real-date return percentile
g  Tri-regional spatial chronology-uplift atlas (MC minus real hybrid win-share gain across China, Europe, and 5,306 Japanese grids) + reversal anatomy
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

import style as S
import data as D
import econ as E
import geo_helper as G

sys.path.insert(0, str(D.REV / 'placement_mc_20261027/code'))
from run_obs_placement import obs_random_order, ce  # noqa: E402

H = 188.5
LEFT, RIGHT = 17.5, 180.0
EX = D.EXAMPLES['national']
RESTORE_TAB = S.V86 / 'revision/v85_restore_20261115/tables'

UPLIFT_CMAP = LinearSegmentedColormap.from_list(
    'ne_uplift',
    [(0.00, '#F5F2EA'), (0.14, '#DCEEEB'), (0.40, '#62B0AE'), (0.72, '#1E6F88'), (1.00, '#0D3554')],
)


def long_window_spans(arr):
    call = arr[:, 1] > 1e-10
    r = np.where(call, 0.0, np.minimum(arr[:, 2], 1.5))
    run = np.cumsum(np.r_[0, np.diff(call.astype(np.int8)) != 0])
    e = np.bincount(run, weights=r)
    out = []
    for k in np.flatnonzero(e * 0.55 >= 21.6 - 1e-9):
        h = np.flatnonzero(run == k)
        if not call[h[0]]:
            out.append((h[0] / 24 + 1, (h[-1] + 1) / 24 + 1))
    return out


def panel_a(cv, top, left, width):
    """Panel a: 4-track annual chronology rearrangement barcode for France, 2016."""
    a = pd.read_csv(D.RC / 'primary_all.csv.gz', dtype={'uid': str})
    a = a[a.uid == EX]
    rnd = a[a.seed >= 0]
    picks = [(-1, 'Real dates')]
    for q, lab in ((0.1, '10th percentile'), (0.5, '50th percentile'), (0.9, '90th percentile')):
        v = rnd.roi_2035.quantile(q)
        picks.append((int(rnd.loc[(rnd.roi_2035 - v).abs().idxmin(), 'seed']), f'Placement, {lab}'))
    arr, ev = D.hourly(EX)
    z = np.load(D.INP / f'{EX}.npz')
    ens = ce.ChronologyEnsemble(pd.DataFrame({'segment': z['segment'], 'event': z['event']}), str(z['unit_key']))
    pw = D.placement_windows().set_index(['uid', 'seed']).long_windows
    roi = a.set_index('seed').roi_2035
    rows, y = [], top
    for k, (seed, lab) in enumerate(picks):
        o = np.arange(365) if seed < 0 else np.asarray(obs_random_order(ens, seed))
        aa = arr if seed < 0 else ce.reorder_days(arr, o)
        hd = ev[o]
        ch = (aa[:, 2] > 1e-9).reshape(365, 24).sum(1)
        spans = long_window_spans(aa)
        assert len(spans) == pw.loc[(EX, seed)]
        ax = cv.ax(left, y, width, 4.2, f'a{k}')
        days = np.arange(1, 366)
        ax.add_patch(Rectangle((0.5, 0.0), 365.0, 1.0, facecolor='#F6F9FA', edgecolor='#D5DFE3', lw=0.36, zorder=1))
        ax.add_patch(Rectangle((0.5, -0.36), 365.0, 0.22, facecolor='#ECF1F3', edgecolor='none', zorder=1))
        ax.bar(days, ch / 24, width=1.0, color='#55A09B', lw=0, zorder=2)
        ax.bar(days[hd], np.ones(hd.sum()), width=1.0, color=S.COLOR['national'], lw=0, zorder=3)
        for s0, s1 in spans:
            ax.add_patch(Rectangle((s0, -0.35), s1 - s0, 0.20, facecolor=S.INK, lw=0, zorder=4, clip_on=False))
        ax.set_xlim(0.5, 365.5)
        ax.set_ylim(-0.40, 1.02)
        ax.axis('off')
        cv.text(left - 1.6, y + 1.6, lab, ha='right', va='center', fontsize=5.3,
                fontweight='bold' if seed < 0 else 'normal', color=S.INK)
        cv.text(left + width, y - 0.28,
                f'return {roi.loc[seed]:.2f}%, {len(spans)} long windows',
                ha='right', va='bottom', fontsize=5.05, fontweight='bold' if seed < 0 else 'normal')
        rows.append(dict(seed=seed, label=lab, roi_2035=roi.loc[seed], long_windows=len(spans)))
        y += 6.9

    for dy, ha in ((1, 'left'), (90, 'center'), (180, 'center'), (270, 'center'), (365, 'right')):
        xpos = left + (dy - 0.5) / 365.0 * width
        cv.text(xpos, y - 1.9, f'Day {dy}' if dy == 1 else str(dy), ha=ha, va='top', fontsize=5.0, color=S.MUTED)

    cv.text(left, top - 3.4, 'France, 2016', ha='left', va='bottom',
            fontsize=5.7, fontweight='bold', color=S.COLOR['national'])
    lg = cv.ax(left + 54.0, top - 6.0, 86.0, 2.5, 'lega_ex')
    lg.set_xlim(0, 86); lg.set_ylim(0, 1); lg.axis('off')
    lg.add_patch(Rectangle((0, 0.15), 2.5, 0.7, facecolor=S.COLOR['national'], lw=0))
    lg.text(3.3, 0.5, 'High-demand day', va='center', fontsize=5.1)
    lg.add_patch(Rectangle((27.0, 0.15), 2.5, 0.7, facecolor='#55A09B', lw=0))
    lg.text(30.3, 0.5, 'Chargeable hours', va='center', fontsize=5.1)
    lg.add_patch(Rectangle((54.5, 0.32), 3.6, 0.36, facecolor=S.INK, lw=0))
    lg.text(59.0, 0.5, 'Long charging window', va='center', fontsize=5.1)
    return pd.DataFrame(rows), y


def panel_b(cv, top, flips):
    """Panel b: Ranked real-date curve vs 5th-95th placement envelope & clean majority-reversal highlight."""
    rec = D.records().set_index('uid')
    p = D.placements().set_index('uid')

    widths = {'provincial': 29.5, 'national': 31.0, 'community': 29.5}
    gap_x = 3.0
    h_hi, h_lo, gap_y = 8.2, 26.3, 1.5
    HI, LO = (5.5, 16.2), (-5.2, 5.5)

    x = LEFT
    hi_ids, lo_ids = [], []
    for i, g in enumerate(S.MAIN):
        z = p[p.group == g].join(rec[['win36']]).sort_values('roi_real')
        n = len(z)
        w_mm = widths[g]
        c = S.COLOR[g]
        w_win = z.win36.to_numpy(bool)
        yv = z.roi_real.to_numpy(float)
        fl = z.index.isin(flips)

        hi = cv.ax(x, top, w_mm, h_hi, f'b{i}hi')
        lo = cv.ax(x, top + h_hi + gap_y, w_mm, h_lo, f'b{i}lo')
        hi_ids.append(f'b{i}hi')
        lo_ids.append(f'b{i}lo')

        if n <= 200:
            xs = np.arange(n, dtype=float)
            for ax, (y0, y1) in ((hi, HI), (lo, LO)):
                q0 = np.clip(z.mc_q05.to_numpy(float), y0, y1)
                q1 = np.clip(z.mc_q95.to_numpy(float), y0, y1)
                k = q1 > q0
                if k.any():
                    ax.fill_between(xs, q0, q1, where=k, color='#D4E0E4', alpha=0.55, lw=0, zorder=1)
                    ax.vlines(xs[k], q0[k], q1[k], color='#8EA0A7', lw=0.45, zorder=2)
                inside = (yv >= y0) & (yv <= y1)
                ax.scatter(xs[w_win & inside], yv[w_win & inside], s=5.8, marker=S.MARKER[g],
                           color=c, edgecolor='white', lw=0.22, zorder=4)
                ax.scatter(xs[~w_win & inside], yv[~w_win & inside], s=5.3, marker=S.MARKER[g],
                           facecolor='white', edgecolor=c, lw=0.52, zorder=4)
                if fl.any():
                    ax.scatter(xs[fl & inside], yv[fl & inside], s=16.0, facecolor='none',
                               edgecolor=S.INK, lw=0.58, zorder=5)
                ax.set_xlim(-2.0, n + 1.0); ax.set_ylim(y0, y1); S.grid(ax, 'y')
                ax.set_xticks([]); ax.spines['bottom'].set_visible(y0 == LO[0])
                if y0 == LO[0]:
                    ax.set_xticks([0, 66, 131]); ax.set_xticklabels(['1', '66', '132'], fontsize=5.0)
            lo.set_xlabel('Ranked records', fontsize=5.1, color=S.MUTED, labelpad=1.2)
        else:
            xs_pct = np.linspace(0.0, 100.0, n)
            fin = np.isfinite(yv)
            win_k = max(7, int(round(n * 0.015)) | 1)
            min_p = max(3, win_k // 5)
            q0_sm = pd.Series(np.where(np.isfinite(z.mc_q05), z.mc_q05, LO[0])).rolling(win_k, center=True, min_periods=min_p).median().to_numpy()
            q1_sm = pd.Series(np.where(np.isfinite(z.mc_q95), z.mc_q95, LO[0])).rolling(win_k, center=True, min_periods=min_p).median().to_numpy()

            if g == 'community' and fl.sum() > 100:
                f_lo, f_hi = float(np.quantile(xs_pct[fl], 0.02)), float(np.quantile(xs_pct[fl], 0.98))
                lo.axvspan(f_lo, f_hi, facecolor='#FCEEE8', alpha=0.80, zorder=0.4)
                lo.text(0.5 * (f_lo + f_hi), -3.0, f'5,045 flips ({100 * fl.mean():.0f}%)',
                        ha='center', va='center', fontsize=5.0, fontweight='bold', color=S.DEEP['national'])

            for ax, (y0, y1) in ((hi, HI), (lo, LO)):
                q0c, q1c = np.clip(q0_sm, y0, y1), np.clip(q1_sm, y0, y1)
                k = q1c > q0c
                if k.any():
                    ax.fill_between(xs_pct, q0c, q1c, where=k, color='#D4E0E4', alpha=0.65, lw=0, zorder=1)
                    ax.plot(xs_pct[k], q0c[k], color='#889AA1', lw=0.45, zorder=2)
                    ax.plot(xs_pct[k], q1c[k], color='#889AA1', lw=0.45, zorder=2)
                if g == 'provincial' and fl.any():
                    in_b = fl & fin & (yv >= y0) & (yv <= y1)
                    if in_b.any():
                        ax.scatter(xs_pct[in_b], yv[in_b], s=16.0, facecolor='none', edgecolor=S.INK, lw=0.62, zorder=5)
                if y0 == LO[0] and (~fin).sum() >= 2:
                    ax.plot(xs_pct[~fin], np.full((~fin).sum(), LO[0] + 0.30),
                            color='#95A4AB', lw=0.80, ls=(0, (1.5, 1.1)), zorder=3)
                m_lose, m_win = fin & (~w_win), fin & w_win
                if m_lose.any():
                    for seg in S.clip_lines([np.c_[xs_pct[m_lose], yv[m_lose]]], (-2, 102, y0, y1)):
                        ax.plot(seg[:, 0], seg[:, 1], color=c, lw=1.05, ls=(0, (2.2, 1.1)), zorder=4)
                if m_win.any():
                    for seg in S.clip_lines([np.c_[xs_pct[m_win], yv[m_win]]], (-2, 102, y0, y1)):
                        ax.plot(seg[:, 0], seg[:, 1], color=c, lw=1.55, zorder=4.5)
                ax.set_xlim(-2.0, 102.0); ax.set_ylim(y0, y1); S.grid(ax, 'y')
                if y0 == LO[0]:
                    ax.set_xticks([0, 50, 100]); ax.set_xticklabels(['0%', '50%', '100%'], fontsize=5.0)
                else:
                    ax.set_xticks([]); ax.spines['bottom'].set_visible(False)
            lo.set_xlabel('Percentile rank', fontsize=5.1, color=S.MUTED, labelpad=1.2)

        lo.axhline(3.6, color=S.WIN, lw=0.75, ls=(0, (3.0, 1.8)), zorder=3.5)
        hi.set_yticks([8, 12, 16])
        lo.set_yticks([-4, -2, 0, 2, 4])
        if i == 0:
            for ax_s, yy in ((hi, 0), (lo, 1)):
                ax_s.plot([-0.03, 0.03], [yy - 0.04, yy + 0.04], transform=ax_s.transAxes,
                          color=S.INK, lw=0.65, clip_on=False)
            lo.text(4.0, 4.15, '3.6%', ha='left', va='bottom', fontsize=5.1, fontweight='bold', color=S.WIN)
        else:
            for ax_s in (hi, lo):
                ax_s.spines['left'].set_visible(False)
                ax_s.tick_params(axis='y', length=0, labelleft=False)

        hi.plot([0, 1], [1.04, 1.04], transform=hi.transAxes, color=c, lw=1.35, clip_on=False)
        cv.text(x + w_mm / 2, top - 5.3, S.SHORT_LABEL[g], ha='center', va='top', fontsize=5.6, fontweight='bold', color=c)
        cv.text(x + w_mm / 2, top - 2.7, f'real {100 * w_win.mean():.0f}% | MC {100 * p[p.group == g].mc_win36.mean():.0f}%',
                ha='center', va='top', fontsize=5.0, color=S.INK)
        x += w_mm + gap_x

    cv.text(7.2, top + (h_hi + gap_y + h_lo) / 2, 'Incremental return (%)',
            rotation=90, ha='center', va='center', fontsize=5.7)
    return hi_ids, lo_ids


def panel_c(cv, x, top, w, h):
    """Panel c: Continuous discount-rate (0-10%) frontier curves (Real solid vs Random dashed + shaded uplift + cost-year inset)."""
    s = D.summary()
    s = s[s.group.isin(S.MAIN)]
    ax = cv.ax(x, top, w, h, 'c')
    S.grid(ax, 'y')
    ax.axvline(3.6, color=S.WIN, lw=0.65, ls=':', zorder=1.5)

    rates = E.RATES
    rows_out = []
    s35 = s[s.year == 2035].set_index('group')
    for g in S.MAIN:
        c = S.COLOR[g]
        r_real = [100.0 * float(s35.loc[g, f'real_win_{rt:g}']) / float(s35.loc[g, 'n']) for rt in rates]
        r_rnd = [100.0 * float(s35.loc[g, f'mc_win_{rt:g}']) for rt in rates]
        ax.fill_between(rates, r_real, r_rnd, color=c, alpha=0.18, lw=0, zorder=2)
        ax.plot(rates, r_real, color=c, lw=1.35, zorder=3.5)
        ax.plot(rates, r_rnd, color=c, lw=1.15, ls=(0, (2.4, 1.3)), zorder=4)
        for rt, vr, vm in zip(rates, r_real, r_rnd):
            rows_out.append(dict(panel='c_rate', cost_year=2035, rate_pct=rt, group=g, dates='real', hybrid_cheaper_share=vr / 100.0))
            rows_out.append(dict(panel='c_rate', cost_year=2035, rate_pct=rt, group=g, dates='random', hybrid_cheaper_share=vm / 100.0))

    ax.annotate('JP: 14 to 56%', (3.6, 56.1), xytext=(4.85, 43.0), textcoords='data',
                ha='left', va='center', fontsize=5.0, fontweight='bold', color=S.COLOR['community'],
                arrowprops=dict(arrowstyle='-', lw=0.42, color=S.COLOR['community'], shrinkA=1, shrinkB=2))
    ax.annotate('EU: 18 to 34%', (3.6, 34.2), xytext=(4.85, 31.0), textcoords='data',
                ha='left', va='center', fontsize=5.0, fontweight='bold', color=S.COLOR['national'],
                arrowprops=dict(arrowstyle='-', lw=0.42, color=S.COLOR['national'], shrinkA=1, shrinkB=2))
    ax.text(0.25, 6.0, 'Solid: real\nDashed: MC', ha='left', va='bottom', fontsize=5.0, color=S.MUTED, linespacing=1.08)

    ins = cv.ax(x + w - 19.5, top + 1.9, 18.2, 11.4, 'c_ins')
    ins.set_facecolor('#FAFDFD')
    yrs = [2025, 2030, 2035]
    for g in S.MAIN:
        sg = s[s.group == g].set_index('year')
        yr_r = [100.0 * float(sg.loc[yy, 'real_win_3.6']) / float(sg.loc[yy, 'n']) for yy in yrs]
        yr_m = [100.0 * float(sg.loc[yy, 'mc_win_3.6']) for yy in yrs]
        ins.plot([0, 1, 2], yr_r, color=S.COLOR[g], lw=0.95)
        ins.plot([0, 1, 2], yr_m, color=S.COLOR[g], lw=0.95, ls=(0, (1.8, 1.0)))
        for yy, vr, vm in zip(yrs, yr_r, yr_m):
            rows_out.append(dict(panel='c_year', cost_year=yy, rate_pct=3.6, group=g, dates='real', hybrid_cheaper_share=vr / 100.0))
            rows_out.append(dict(panel='c_year', cost_year=yy, rate_pct=3.6, group=g, dates='random', hybrid_cheaper_share=vm / 100.0))
    ins.set_xlim(-0.2, 2.2); ins.set_ylim(-2, 62)
    ins.set_xticks([0, 1, 2]); ins.set_xticklabels(["'25", "'30", "'35"], fontsize=5.0)
    ins.set_yticks([0, 50]); ins.set_yticklabels(['0', '50'], fontsize=5.0)
    ins.text(0.05, 56.0, 'At 3.6%', ha='left', va='top', fontsize=5.0, color=S.INK)

    ax.set_xlim(-0.2, 10.2)
    ax.set_xticks([0, 2, 3.6, 6, 8, 10])
    ax.set_xticklabels(['0', '2', '3.6', '6', '8', '10'], fontsize=5.0)
    ax.set_ylim(-2, 103)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel('Real discount rate (%, 2035 costs)')
    ax.set_ylabel('Hybrid cheaper (%)')
    return pd.DataFrame(rows_out)


def panel_d(cv, x, top, w, h):
    """Panel d: Cross-regime 7-baseline (q5-q50) real vs random portfolio trajectories."""
    mq = pd.read_csv(RESTORE_TAB / 'multi_q_7baselines_current.csv')
    ax = cv.ax(x, top, w, h, 'd')
    S.grid(ax, 'y')

    specs = [
        ('China_proportional', S.COLOR['provincial'], 's', 1.25),
        ('China_clipped', '#6E5015', 'D', 0.85),
        ('Europe', S.COLOR['national'], 'o', 1.25),
        ('Japan', S.COLOR['community'], '^', 1.25),
    ]
    for ser, col, mk, lw in specs:
        sub = mq[mq.series == ser].sort_values('q')
        ax.plot(sub.q, sub.real_share_pct, color=col, lw=lw, zorder=3.5)
        ax.plot(sub.q, sub.mc_share_pct, color=col, lw=lw, ls=(0, (2.2, 1.2)), marker=mk, ms=2.6,
                mec='white', mew=0.25, zorder=4)
        if ser in ('Europe', 'Japan'):
            ax.fill_between(sub.q, sub.real_share_pct, sub.mc_share_pct, color=col, alpha=0.14, lw=0, zorder=2)

    ax.text(6.0, 56.5, 'Dashed: MC; solid: real\nCN 2-method: 96–99% agree.',
            ha='left', va='top', fontsize=5.0, color=S.INK, linespacing=1.12)

    ax.set_xlim(3, 52.5)
    ax.set_xticks([5, 20, 35, 50])
    ax.set_xticklabels(['q5', 'q20', 'q35', 'q50'], fontsize=5.0)
    ax.set_ylim(-2, 61)
    ax.set_yticks([0, 20, 40, 60])
    ax.set_xlabel('Task baseline quantile')
    ax.set_ylabel('Hybrid cheaper (%)')
    return mq


def panel_e(cv, x, top, w, h):
    """Panel e: Empirical cumulative calibration curves of real-date return percentile vs 45-degree benchmark."""
    p = D.placements()
    pos_uids = D.cn_positive_uids()

    ax = cv.ax(x, top, w, h, 'e')
    S.grid(ax, 'both')
    ax.plot([0, 100], [0, 100], color='#8D9CA3', lw=0.75, ls=(0, (2.6, 1.6)), zorder=2)

    xg = np.linspace(0, 100, 250)
    for g in S.MAIN:
        v = np.sort(100.0 * p.loc[p.group == g, 'real_pct'].to_numpy(float))
        cdf = 100.0 * np.searchsorted(v, xg, side='right') / len(v)
        ax.plot(xg, cdf, color=S.COLOR[g], lw=1.35, zorder=4)
        med = float(np.median(v))
        ax.scatter([med], [50.0], s=14, marker=S.MARKER[g], color=S.COLOR[g], edgecolor='white', lw=0.35, zorder=5)

    vp = np.sort(100.0 * p.loc[p.uid.isin(pos_uids), 'real_pct'].to_numpy(float))
    cdf_p = 100.0 * np.searchsorted(vp, xg, side='right') / len(vp)
    ax.plot(xg, cdf_p, color=S.COLOR['provincial'], lw=1.0, ls=(0, (2.2, 1.2)), zorder=3.8)
    ax.scatter([float(np.median(vp))], [50.0], s=14, marker='s', facecolor='white',
               edgecolor=S.COLOR['provincial'], lw=0.65, zorder=5)

    ax.axhline(50.0, color='#CBD4D8', lw=0.45, zorder=1)
    ax.text(97.5, 4.5, 'Medians:\nJP 5th, EU 9th\nCN pos 24th',
            ha='right', va='bottom', fontsize=5.0, color=S.INK, linespacing=1.08)

    ax.set_xlim(0, 100); ax.set_ylim(0, 102)
    ax.set_xticks([0, 50, 100]); ax.set_yticks([0, 50, 100])
    ax.set_xlabel('Real-date return percentile')
    ax.set_ylabel('Cumulative records (%)')


def panel_f(cv, x, top, w, h):
    """Panel f: 2D joint density & quantile curve of delta long windows vs real-date return percentile."""
    dlw_df = D.placement_delta_long_windows().set_index('uid')
    p = D.placements().set_index('uid')
    p = p[p.group.isin(S.MAIN)]
    d = dlw_df[['dlw']].join(p[['group', 'real_pct']], how='inner').reset_index()
    rho_eq, _, _ = D.equal_regime_spearman(d, 'dlw', 'real_pct')

    ax = cv.ax(x, top, w, h, 'f')
    S.grid(ax, 'y')
    xlim = (-11.5, 8.5)
    ax.add_patch(Rectangle((0, 50), xlim[1], 50, facecolor=S.WIN_PALE, edgecolor='none', zorder=0))
    ax.add_patch(Rectangle((xlim[0], 0), -xlim[0], 50, facecolor='#F4F7F8', edgecolor='none', zorder=0))

    jp = d[d.group == 'community']
    jx = jp.dlw.to_numpy(float)
    jy = 100.0 * jp.real_pct.to_numpy(float)
    c_jp = S.COLOR['community']
    rng = np.random.default_rng(2026)
    idx_s = rng.choice(len(jx), size=min(len(jx), 3500), replace=False)
    jx_j = np.clip(jx[idx_s] + rng.normal(0, 0.32, len(idx_s)), *xlim)
    sx, sy = xlim[1] - xlim[0], 100.0
    kde2 = gaussian_kde(np.vstack([(jx_j - xlim[0]) / sx, jy[idx_s] / sy]), bw_method=0.22)
    gx, gy = np.linspace(0.01, 0.99, 70), np.linspace(0.01, 0.99, 70)
    GX, GY = np.meshgrid(gx, gy)
    Z = kde2(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
    vals = np.sort(Z.ravel())[::-1]
    csum = np.cumsum(vals)
    t88 = vals[min(len(vals) - 1, np.searchsorted(csum, 0.88 * csum[-1]))]
    t50 = vals[min(len(vals) - 1, np.searchsorted(csum, 0.50 * csum[-1]))]
    Xp, Yp = xlim[0] + GX * sx, GY * sy
    ax.contourf(Xp, Yp, Z, levels=[t88, Z.max() * 1.05], colors=[c_jp], alpha=0.16, zorder=1.2)
    ax.contourf(Xp, Yp, Z, levels=[t50, Z.max() * 1.05], colors=[c_jp], alpha=0.24, zorder=1.4)
    ax.contour(Xp, Yp, Z, levels=[t88, t50], colors=[c_jp], linewidths=[0.45, 0.65], alpha=0.60, zorder=1.8)

    edges_j = np.unique(np.quantile(jx, np.linspace(0.02, 0.98, 9)))
    xm_j, ym_j = [], []
    for lo_e, hi_e in zip(edges_j[:-1], edges_j[1:]):
        mk = (jx >= lo_e) & (jx <= hi_e)
        if mk.sum() >= 10:
            xm_j.append(float(np.median(jx[mk]))); ym_j.append(float(np.median(jy[mk])))
    if len(xm_j) >= 2:
        ax.plot(xm_j, ym_j, color=c_jp, lw=1.25, zorder=2.5)

    for g in ('national', 'provincial'):
        z = d[d.group == g]
        ax.scatter(np.clip(z.dlw + E.jitter(len(z), 31) * 0.40, xlim[0] + 0.3, xlim[1] - 0.3), 100 * z.real_pct,
                   s=7.0 if g == 'provincial' else 8.5, marker=S.MARKER[g], color=S.COLOR[g],
                   edgecolor='white', lw=0.22, alpha=0.82, zorder=3.5)

    ax.axvline(0, color=S.INK, lw=0.65, zorder=2)
    ax.axhline(50, color=S.MUTED, lw=0.60, ls=(0, (2.6, 1.6)), zorder=2)
    ax.text(0.5, 1.03, f'Equal-regime ρ = {rho_eq:.2f}', transform=ax.transAxes,
            ha='center', va='bottom', fontsize=5.05, fontweight='bold', color=S.INK)
    ax.set_xlim(*xlim); ax.set_xticks([-10, -5, 0, 5])
    ax.set_ylim(0, 102); ax.set_yticks([0, 50, 100])
    ax.set_xlabel('Δ long windows (real − MC)')
    ax.set_ylabel('Return percentile')
    return d, rho_eq


def panel_g_and_h(cv, x0, top, hmm, flips):
    """Panel g: Enlarged tri-regional spatial chronology-uplift atlas (MC minus real hybrid win-share gain).
    Panel h: Dedicated, clearly labeled portfolio-sensitivity & majority-reversal breakdown locked to RIGHT = 180.0 mm.
    """
    p = D.placements().set_index('uid')
    rec = D.records().set_index('uid')
    dlw = D.placement_delta_long_windows().set_index('uid')
    m = p[p.group.isin(S.MAIN)].join(rec[['system_id', 'win36']]).join(dlw[['dlw']])
    m['uplift_pp'] = 100.0 * (m.mc_win36 - m.win36.astype(float))

    cn_up = m[m.group == 'provincial'].groupby('system_id')['uplift_pp'].mean().to_dict()
    eu_up = m[m.group == 'national'].groupby('system_id')['uplift_pp'].mean().to_dict()
    jp_up = m[m.group == 'community'].groupby('system_id')['uplift_pp'].mean().to_dict()

    norm = Normalize(vmin=-5.0, vmax=75.0)
    x = x0
    _, _, w_cn = G.draw_china_choropleth(cv, x, top, hmm, cn_up, UPLIFT_CMAP, norm, 'g_cn')
    cv.text(x + w_cn / 2, top - 2.8, 'China (mean 0.4 pp)', ha='center', va='top',
            fontsize=5.0, fontweight='bold', color=S.COLOR['provincial'])
    x += w_cn + 2.6

    _, w_eu = G.draw_europe_choropleth(cv, x, top, hmm, eu_up, UPLIFT_CMAP, norm, 'g_eu')
    cv.text(x + w_eu / 2, top - 2.8, 'Europe (mean 16 pp)', ha='center', va='top',
            fontsize=5.0, fontweight='bold', color=S.COLOR['national'])
    x += w_eu + 2.6

    _, w_jp = G.draw_japan_choropleth(cv, x, top, hmm, jp_up, UPLIFT_CMAP, norm, 'g_jp', pt_size=0.72)
    cv.text(x + w_jp / 2, top - 2.8, 'Japan (mean 42 pp)', ha='center', va='top',
            fontsize=5.0, fontweight='bold', color=S.COLOR['community'])
    x += w_jp + 2.0

    cax = cv.ax(x, top + 1.2, 2.2, hmm - 2.4, 'cb_g')
    sm = S.mpl.cm.ScalarMappable(norm=norm, cmap=UPLIFT_CMAP)
    cb = cv.fig.colorbar(sm, cax=cax, ticks=[0, 35, 75])
    cb.ax.set_yticklabels(['0', '35', '75'], fontsize=5.0)
    cb.ax.tick_params(length=1.5, pad=1.1)
    cb.outline.set_linewidth(0.45)
    cb.outline.set_edgecolor('#8FA0A8')
    cv.text(x + 1.1, top - 2.8, 'MC−real', ha='center', va='top', fontsize=5.0, fontweight='bold', color=S.INK)
    cv.text(x + 1.1, top + hmm + 0.8, '(pp)', ha='center', va='top', fontsize=5.0, color=S.MUTED)

    # Panel h: Dedicated labeled panel for timing-sensitive & majority-reversal shares
    xh_let = x + 6.8
    cv.letter(xh_let, top - 3.8, 'h')
    xb = xh_let + 7.2
    wb = RIGHT - xb
    ax = cv.ax(xb, top, wb, hmm, 'h')
    ax.grid(False)

    ys = np.arange(3)[::-1]
    strad = [100.0 * float(((m[m.group == g].mc_win36 > 0.05) & (m[m.group == g].mc_win36 < 0.95)).mean()) for g in S.MAIN]
    flp = [100.0 * float(m[m.group == g].index.isin(flips).mean()) for g in S.MAIN]

    ax.barh(ys + 0.16, strad, height=0.28, facecolor='#DCE6EA', edgecolor='#7B8E97', lw=0.45, zorder=3)
    for gi, g in enumerate(S.MAIN):
        yy = ys[gi]
        ax.barh([yy - 0.16], [flp[gi]], height=0.28, facecolor=S.COLOR[g], edgecolor='none', zorder=3.5)
        ax.text(strad[gi] + 2.0, yy + 0.16, f'{strad[gi]:.0f}%', va='center', ha='left', fontsize=5.0, color=S.MUTED)
        ax.text(flp[gi] + 2.0, yy - 0.16, f'{flp[gi]:.0f}%', va='center', ha='left', fontsize=5.0,
                fontweight='bold', color=S.COLOR[g])

    fl_sub = m.loc[flips]
    pct_lfp_to_hyb = 100.0 * float((~fl_sub.win36).mean())
    pct_dlw_neg = 100.0 * float((fl_sub.dlw < 0).mean())

    ax.text(0.0, 1.04, f'Flips: {pct_lfp_to_hyb:.0f}% LFP to Hyb, {pct_dlw_neg:.0f}% Δlw < 0',
            transform=ax.transAxes, ha='left', va='bottom', fontsize=5.0, fontweight='bold', color=S.INK)
    ax.set_yticks(ys)
    ax.set_yticklabels(['CN', 'EU', 'JP'], fontsize=5.1, fontweight='bold')
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(0, 108)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel('Records (%, grey: >5% both)')
    return ['g_cn', 'g_eu', 'g_jp']


def main():
    S.setup()
    cv = S.Canvas(H)

    p = D.placements().set_index('uid')
    rec = D.records().set_index('uid')
    flips = p.index[(p.group.isin(S.MAIN)) & ((p.mc_win36 > 0.5) != rec.loc[p.index, 'win36'])]

    # Row 1: a (France 2016 chronology tracks)
    cv.letter(1.5, 1.8, 'a')
    ex_rows, y_end = panel_a(cv, 6.8, 40.0, RIGHT - 40.0)

    # Row 2: b (Ranked MC fan) + c (Discount-rate frontier curves + cost-year inset)
    top2 = y_end + 10.2
    cv.letter(1.5, top2 - 5.6, 'b')
    hi_ids, lo_ids = panel_b(cv, top2, flips)

    cv.letter(116.5, top2 - 5.6, 'c')
    c_df = panel_c(cv, 125.5, top2, RIGHT - 125.5, 36.0)

    # Row 3: d (7 baselines q5-q50), e (45-deg ECDF calibration), f (delta long windows) — exact 13.0 mm gutters
    top3, h3 = top2 + 49.5, 33.5
    cv.letter(1.5, top3 - 4.0, 'd')
    mq_df = panel_d(cv, LEFT, top3, 45.5, h3)

    cv.letter(67.0, top3 - 4.0, 'e')
    panel_e(cv, 76.0, top3, 45.5, h3)

    cv.letter(125.5, top3 - 4.0, 'f')
    fd, rho_f = panel_f(cv, 134.5, top3, 45.5, h3)

    # Row 4: g (Spatial chronology-uplift atlas) + h (Reversal & timing-sensitivity breakdown)
    top4 = top3 + h3 + 13.5
    cv.letter(1.5, top4 - 3.8, 'g')
    g_map_ids = panel_g_and_h(cv, LEFT, top4, 29.2, flips)

    # Source data exports
    S.SD.mkdir(parents=True, exist_ok=True)
    ex_rows.assign(uid=EX).to_csv(S.SD / 'figure3a_example_placements.csv', index=False)
    pp = p[p.group.isin(S.MAIN)].join(rec[['win36']])
    pp.assign(majority_differs=pp.index.isin(flips)).reset_index().to_csv(S.SD / 'figure3be_records.csv', index=False)
    c_df.to_csv(S.SD / 'figure3cd_who_is_cheaper.csv', index=False)
    mq_df.to_csv(S.SD / 'figure3d_7baselines_who_is_cheaper.csv', index=False)
    fd.to_csv(S.SD / 'figure3f_long_windows.csv', index=False)

    S.save(
        cv,
        'main_figure3',
        rows=[hi_ids, lo_ids, ['d', 'e', 'f'], g_map_ids + ['h']],
        cols=[['a0', 'a1', 'a2', 'a3']],
        exempt=[
            dict(panels=['lega_ex', 'c_ins', 'c', 'g_cn_scs', 'cb_g'], checks=['all'],
                 reason='Legend strip, frontier panel c + inset, South China Sea inset, and colorbar'),
            dict(panels=hi_ids + lo_ids + g_map_ids + ['h'], checks=['panel-width'],
                 reason='Intentional cartographic and ranked-curve panel widths'),
            dict(panels=['h'], checks=['horizontal-gutter'],
                 reason='Dedicated panel h positioned beside Panel g colorbar'),
        ],
        audit=dict(
            panels=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
            panel_count=8,
            example=EX,
            records=11054,
            majority_flips=int(len(flips)),
            spearman_long_window_gap_vs_percentile_equal_regime=rho_f,
        ),
    )


if __name__ == '__main__':
    main()
