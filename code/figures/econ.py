"""Panels shared by Fig. 2 (real dates) and Fig. 3 (random start/end placement) across 11,054 records."""
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

import style as S
import data as D

RATES = [0, 1, 2, 3, 3.6, 5, 6, 8, 10]
WIDTHS = {'provincial': 46.0, 'national': 62.0, 'community': 48.0}
LO, HI = (-5.4, 9.0), (9.0, 16.2)
RANGE = '#98A8AE'
BAND_FILL = '#D3DCE0'
WIN_C = '#1B5E66'
RND_WIN_C = '#3D8C87'
LOSE_C = '#EDF1F3'
LOSE_EC = '#C9D3D8'
POS_BG = '#EBF5F4'

COMP_PAL = {
    'B_fixed': '#8FA0A8',
    'B_variable_om': '#CBB49D',
    'B_charging': '#DC9E46',
    'B_residual_energy': '#3B968B',
    'B_residual_peak': '#2A5C8A',
}


def roi_panel(cv, top, h_hi, h_lo, prefix, with_random, flips=None, gap=1.6):
    """Incremental return across 10,770 records, broken axis.

    Discrete ranked markers for provincial (26) and national (132); continuous 0–100% percentile-rank
    return curve (+ smooth rolling 5th–95th placement envelope in Fig. 3b) for community (10,612).
    """
    rec = D.records().set_index('uid')
    p = D.placements().set_index('uid')
    x = 19.0
    axes = []
    for i, g in enumerate(S.MAIN):
        z = p[p.group == g].join(rec[['win36']]).sort_values('roi_real')
        n = len(z)
        c = S.COLOR[g]
        w = z.win36.to_numpy(bool)
        hi = cv.ax(x, top, WIDTHS[g], h_hi, f'{prefix}{i}hi')
        lo = cv.ax(x, top + h_hi + gap, WIDTHS[g], h_lo, f'{prefix}{i}lo')
        yv = z.roi_real.to_numpy(float)
        f = z.index.isin(flips) if flips is not None else np.zeros(n, bool)

        hi.axhspan(HI[0], HI[1], facecolor=POS_BG, zorder=0)
        lo.axhspan(3.6, LO[1], facecolor=POS_BG, zorder=0)

        if n <= 200:
            xs = np.arange(n, dtype=float)
            ms = 6.2 if n > 60 else 11.5
            fin_vals = z[['mc_q05', 'mc_q95', 'roi_real'] if with_random else ['roi_real']].to_numpy(float)
            fin_vals = fin_vals[np.isfinite(fin_vals)]
            assert fin_vals.min() > LO[0] and fin_vals.max() < HI[1]
            for ax, (y0, y1) in ((hi, HI), (lo, LO)):
                if with_random:
                    q0 = np.clip(z.mc_q05.to_numpy(float), y0, y1)
                    q1 = np.clip(z.mc_q95.to_numpy(float), y0, y1)
                    k = q1 > q0
                    if k.any():
                        ax.fill_between(xs, q0, q1, where=k, color=BAND_FILL, alpha=0.52, lw=0, zorder=1)
                        ax.vlines(xs[k], q0[k], q1[k], color=RANGE, lw=0.45 if n > 60 else 0.72, zorder=2)
                else:
                    ref = np.clip(3.6, y0, y1)
                    yc = np.clip(yv, y0, y1)
                    above = yc > ref
                    below = yc < ref
                    if above.any():
                        ax.fill_between(xs, ref, yc, where=above, interpolate=True, color=c, alpha=0.15, lw=0, zorder=1)
                    if below.any():
                        ax.fill_between(xs, ref, yc, where=below, interpolate=True, color='#B8C2C6', alpha=0.15, lw=0, zorder=1)

                inside = (yv >= y0) & (yv <= y1)
                ax.scatter(xs[w & inside], yv[w & inside], s=ms, color=c, edgecolor='white', lw=0.25, zorder=4)
                ax.scatter(xs[~w & inside], yv[~w & inside], s=ms, facecolor='white', edgecolor=c, lw=0.62, zorder=4)
                if f.any():
                    ax.scatter(xs[f & inside], yv[f & inside], s=ms * 2.9, facecolor='none', edgecolor=S.INK, lw=0.62, zorder=5)
                ax.set_xlim(-1.5, n + 0.5)
                ax.set_ylim(y0, y1)
                S.grid(ax, 'y')
                ax.set_xticks([])
                ax.spines['bottom'].set_visible(False)
            lo.set_xlabel(f'{n} records, ranked', fontsize=5.4, color=S.MUTED, labelpad=1.8)
            head = (f'real {int(w.sum())}/{n}, random {100 * p[p.group == g].mc_win36.mean():.0f}%' if with_random
                    else f'{int(w.sum())}/{n} hybrid cheaper at 3.6%')
        else:
            # Continuous 0–100% percentile-rank curve for n > 200 (China 310 & Japan 10,612)
            xs_pct = np.linspace(0.0, 100.0, n)
            fin_mask = np.isfinite(yv)
            assert yv[fin_mask].min() > LO[0] and yv[fin_mask].max() < HI[1]
            # Smooth rolling placement quantiles (5th..95th) across ranked curve for Fig. 3b
            if with_random:
                win_k = max(7, int(round(n * 0.015)) | 1)
                min_p = max(3, win_k // 5)
                q0_raw = np.where(np.isfinite(z.mc_q05.to_numpy(float)), z.mc_q05.to_numpy(float), LO[0])
                q1_raw = np.where(np.isfinite(z.mc_q95.to_numpy(float)), z.mc_q95.to_numpy(float), LO[0])
                q0_sm = pd.Series(q0_raw).rolling(win_k, center=True, min_periods=min_p).median().to_numpy()
                q1_sm = pd.Series(q1_raw).rolling(win_k, center=True, min_periods=min_p).median().to_numpy()

            for ax, (y0, y1) in ((hi, HI), (lo, LO)):
                if with_random:
                    q0c = np.clip(q0_sm, y0, y1)
                    q1c = np.clip(q1_sm, y0, y1)
                    k = q1c > q0c
                    if k.any():
                        ax.fill_between(xs_pct, q0c, q1c, where=k, color=BAND_FILL, alpha=0.62, lw=0, zorder=1)
                        ax.plot(xs_pct[k], q0c[k], color=RANGE, lw=0.45, zorder=2)
                        ax.plot(xs_pct[k], q1c[k], color=RANGE, lw=0.45, zorder=2)
                    if f.any():
                        in_band = f & fin_mask & (yv >= y0) & (yv <= y1)
                        fx = xs_pct[in_band]
                        fy = yv[in_band]
                        if len(fx) > 0:
                            step_f = max(1, len(fx) // 35)
                            ax.scatter(fx[::step_f], fy[::step_f], s=11.0 if n < 1000 else 8.5,
                                       facecolor='none', edgecolor=S.INK, lw=0.55 if n < 1000 else 0.48, alpha=0.80, zorder=5)
                else:
                    ref = np.clip(3.6, y0, y1)
                    yc = np.clip(yv[fin_mask], y0, y1)
                    xf = xs_pct[fin_mask]
                    above = yc > ref
                    below = yc < ref
                    if above.any():
                        ax.fill_between(xf, ref, yc, where=above, interpolate=True, color=c, alpha=0.22, lw=0, zorder=1)
                    if below.any():
                        ax.fill_between(xf, ref, yc, where=below, interpolate=True, color='#B8C2C6', alpha=0.18, lw=0, zorder=1)

                # Continuous real-date curve across both lo and hi axes (dashed below 3.6%, solid bold above 3.6%)
                if y0 == LO[0] and (~fin_mask).sum() >= 2:
                    ax.plot(xs_pct[~fin_mask], np.full((~fin_mask).sum(), LO[0] + 0.28),
                            color='#9AA8AE', lw=0.75, ls=(0, (1.6, 1.2)), zorder=3.2)
                m_lose = fin_mask & (~w)
                m_win = fin_mask & w
                if m_lose.any() and np.any((yv[m_lose] >= y0 - 0.5) & (yv[m_lose] <= y1 + 0.5)):
                    ax.plot(xs_pct[m_lose], yv[m_lose], color=c, lw=1.10, ls=(0, (2.2, 1.2)), zorder=4)
                if m_win.any() and np.any((yv[m_win] >= y0 - 0.5) & (yv[m_win] <= y1 + 0.5)):
                    ax.plot(xs_pct[m_win], yv[m_win], color=c, lw=1.55, zorder=4.5)
                if y0 == LO[0] and m_win.any():
                    x_cross = float(xs_pct[m_win][0])
                    ax.scatter([x_cross], [3.6], s=14, color=c, edgecolor='white', lw=0.35, zorder=5)

                ax.set_xlim(-2.0, 102.0)
                ax.set_ylim(y0, y1)
                S.grid(ax, 'y')
                if y0 == LO[0]:
                    ax.set_xticks([0, 50, 100])
                    ax.set_xticklabels(['0%', '50%', '100%'], fontsize=5.0)
                else:
                    ax.set_xticks([])
                    ax.spines['bottom'].set_visible(False)

            lo.set_xlabel(f'{n:,} records (percentile)', fontsize=5.3, color=S.MUTED, labelpad=1.4)
            head = (f'real {int(w.sum()):,}/{n:,}, random {100 * p[p.group == g].mc_win36.mean():.0f}%' if with_random
                    else f'{int(w.sum()):,}/{n:,} ({100 * w.mean():.0f}%) at 3.6%')

        lo.axhline(3.6, color=S.INK, lw=0.7, ls=(0, (3, 2)), zorder=3)
        lo.axhline(0.0, color='#C9D2D5', lw=0.45, zorder=1)
        hi.set_yticks([10, 13, 16])
        lo.set_yticks([-4, -2, 0, 2, 4, 6, 8])
        if i == 0:
            d = 0.85
            for ax, yy in ((hi, 0), (lo, 1)):
                ax.plot([-d / WIDTHS[g], d / WIDTHS[g]], [yy - 0.025, yy + 0.025],
                        transform=ax.transAxes, color=S.INK, lw=0.65, clip_on=False)
            x_lbl = (n - 0.6) if n <= 200 else 18.0
            ha_lbl = 'right' if n <= 200 else 'left'
            lo.text(x_lbl, 4.45, '3.6%', ha=ha_lbl, va='center', fontsize=5.4, fontweight='bold')
        else:
            for ax in (hi, lo):
                ax.spines['left'].set_visible(False)
                ax.tick_params(axis='y', length=0, labelleft=False)

        hi.plot([0, 1], [1.025, 1.025], transform=hi.transAxes, color=c, lw=1.3, solid_capstyle='butt', clip_on=False)
        cv.text(x + WIDTHS[g] / 2, top - 6.0, S.LABEL[g], ha='center', va='top', fontsize=5.9, fontweight='bold', color=c)
        cv.text(x + WIDTHS[g] / 2, top - 3.2, head, ha='center', va='top', fontsize=5.25, color=S.INK)
        axes.append((hi, lo))
        x += WIDTHS[g] + 3.0
    cv.text(8.2, top + (h_hi + gap + h_lo) / 2, 'Incremental return (%)', rotation=90, ha='center', va='center', fontsize=6.0)
    return [f'{prefix}{i}hi' for i in range(3)], [f'{prefix}{i}lo' for i in range(3)]


def shares_from(s, year=None, rate=None):
    out = {}
    for _, r in s.iterrows():
        for rt in (RATES if rate is None else [rate]):
            if year is not None and r.year != year:
                continue
            key = rt if rate is None else int(r.year)
            out[(key, r.group, 'real')] = r[f'real_win_{rt:g}'] / r.n
            out[(key, r.group, 'random')] = r[f'mc_win_{rt:g}']
    return out


def bar_grid(ax, rows, shares, ylabels, kinds=('real', 'random'), show_scale=False):
    """Left-to-right 0–100 % horizontal progress/funnel bars with frontier spine."""
    paired = len(kinds) == 2
    pair_step = 1.12 if paired else 0.0
    group_step = 2.52 if paired else 1.30
    pos = []
    for gi, g in enumerate(S.MAIN):
        base = gi * group_step
        for ji, k in enumerate(kinds):
            pos.append((g, k, base + ji * pair_step))

    bh = 0.66 if paired else 0.64
    xmin = -0.08
    xmax = pos[-1][2] + 1.08

    if 3.6 in rows:
        r36 = rows.index(3.6)
        ax.add_patch(Rectangle((xmin, r36 - 0.46), xmax - xmin, 0.92,
                               facecolor='#E9F3F4', edgecolor='none', zorder=0))

    for gi in (0, 1):
        x_right = gi * group_step + (pair_step if paired else 0.0) + 1.00
        x_next = (gi + 1) * group_step
        xm = 0.5 * (x_right + x_next)
        ax.plot([xm, xm], [-0.45, len(rows) - 0.55], color='#DAE2E5', lw=0.5, zorder=1)

    for g, kind, x0 in pos:
        fc_win = WIN_C if kind == 'real' else RND_WIN_C
        xs_front, ys_front = [], []
        for r, row in enumerate(rows):
            v = float(shares[(row, g, kind)])
            y_top = r - bh / 2.0
            ax.add_patch(Rectangle((x0, y_top), 1.00, bh, facecolor=LOSE_C, edgecolor=LOSE_EC, lw=0.34, zorder=2))
            if v > 0.002:
                ax.add_patch(Rectangle((x0, y_top), v, bh, facecolor=fc_win, lw=0, zorder=3))
            xs_front.append(x0 + v)
            ys_front.append(r)

            pct = 100.0 * v
            if 0.05 <= pct <= 99.95:
                txt = '<1' if pct < 1 else ('>99' if pct > 99 else f'{pct:.0f}')
                dark = v >= 0.46
                xx = x0 + 0.5 * v if dark else x0 + 0.5 * (v + 1.0)
                ax.text(xx, r, txt, ha='center', va='center', fontsize=5.1,
                        fontweight='bold' if row == 3.6 else 'normal',
                        color='white' if dark else S.INK, zorder=5)

        if len(rows) >= 5:
            ax.plot(xs_front, ys_front, color='#0F3E45' if kind == 'real' else '#276662',
                    lw=0.58, alpha=0.75, zorder=4)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(ylabels)
    if paired:
        ax.set_xticks([x0 + 0.50 for _, _, x0 in pos])
        ax.set_xticklabels(['Real' if k == 'real' else 'Random' for _, k, _ in pos], fontsize=5.1)
    elif show_scale:
        xt, xl = [], []
        for _, _, x0 in pos:
            xt.extend([x0, x0 + 0.5, x0 + 1.0])
            xl.extend(['0', '50', '100%'])
        ax.set_xticks(xt)
        ax.set_xticklabels(xl, fontsize=5.0)
    else:
        ax.set_xticks([])
    ax.tick_params(axis='both', length=0)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_visible(False)
    return {g: np.mean([x0 + 0.50 for gg, _, x0 in pos if gg == g]) for g in S.MAIN}


def group_heads(ax, centres, y=-0.72):
    for g, xc in centres.items():
        ax.text(xc, y, S.LABEL2[g], ha='center', va='bottom', fontsize=5.6, fontweight='bold', color=S.COLOR[g],
                transform=ax.transData, linespacing=1.05)


def win_legend(cv, x, top, w, with_random=False):
    lg = cv.ax(x, top, w, 3.0, 'legwin')
    lg.set_xlim(0, w)
    lg.set_ylim(0, 3.0)
    lg.axis('off')
    if with_random:
        lg.add_patch(Rectangle((0, 0.5), 3.0, 2.0, facecolor=WIN_C, lw=0))
        lg.text(3.8, 1.5, 'Hybrid cheaper (real)', va='center', fontsize=5.2)
        x2 = w * 0.36
        lg.add_patch(Rectangle((x2, 0.5), 3.0, 2.0, facecolor=RND_WIN_C, lw=0))
        lg.text(x2 + 3.8, 1.5, 'Hybrid cheaper (random)', va='center', fontsize=5.2)
        x3 = w * 0.75
        lg.add_patch(Rectangle((x3, 0.5), 3.0, 2.0, facecolor=LOSE_C, edgecolor=LOSE_EC, lw=0.4))
        lg.text(x3 + 3.8, 1.5, 'LFP-only', va='center', fontsize=5.2)
    else:
        lg.add_patch(Rectangle((0, 0.5), 3.8, 2.0, facecolor=WIN_C, lw=0))
        lg.text(4.8, 1.5, 'Hybrid portfolio cheaper', va='center', fontsize=5.4)
        x2 = w * 0.50
        lg.add_patch(Rectangle((x2, 0.5), 3.8, 2.0, facecolor=LOSE_C, edgecolor=LOSE_EC, lw=0.4))
        lg.text(x2 + 4.8, 1.5, 'LFP-only portfolio cheaper', va='center', fontsize=5.4)
    return 'legwin'


def strip_axis(ax, labels=True):
    ax.set_ylim(2.6, -0.6)
    ax.set_yticks(range(3))
    ax.set_yticklabels([S.LABEL2[g] for g in S.MAIN] if labels else [])
    ax.tick_params(axis='y', length=0)


def components(cv, x, top, w, h, gid):
    rec = D.records()
    rec = rec[rec.group.isin(S.MAIN)]
    m = rec.groupby('group', observed=True)[S.COMP + ['B']].mean()
    assert np.allclose(m[S.COMP].sum(1), m.B, atol=1e-6)
    ax = cv.ax(x, top, w, h, gid)
    S.grid(ax, 'x')
    ax.axhspan(0.5, 1.5, facecolor='#F7F9FA', zorder=0)
    for i, g in enumerate(S.MAIN):
        pos = neg = 0.0
        for c in S.COMP:
            v = m.loc[g, c]
            left = pos if v >= 0 else neg + v
            ax.barh(i, abs(v), left=left, height=0.58, color=COMP_PAL[c], edgecolor='white', lw=0.25, zorder=3)
            if v >= 0:
                pos += v
            else:
                neg += v
        ax.scatter(m.loc[g, 'B'], i, marker='D', s=17, color=S.INK, edgecolor='white', lw=0.55, zorder=5)
        ax.text(92.0, i, f'{m.loc[g, "B"]:.1f}'.replace('-', '−'),
                va='center', ha='left', fontsize=5.5, fontweight='bold', color=S.INK, clip_on=False, zorder=5)
    ax.axvline(0, color=S.INK, lw=0.65, zorder=4)
    strip_axis(ax)
    ax.set_xlim(-55, 90)
    ax.set_xticks([-45, 0, 45, 90])
    ax.set_xlabel(f'Mean B ({S.UNIT})')
    return ax, m


def components_legend(cv, x, top, H):
    col_w = 23.2
    for k, c in enumerate(S.COMP):
        row, col = divmod(k, 3)
        yy = top + row * 3.1
        xx = x + col * col_w
        cv.fig.patches.append(Rectangle((xx / S.WIDTH_MM, 1 - (yy + 2.0) / H), 2.5 / S.WIDTH_MM, 1.8 / H,
                                        transform=cv.fig.transFigure, facecolor=COMP_PAL[c], lw=0))
        cv.text(xx + 3.2, yy + 0.1, S.COMP_LABEL[c], fontsize=5.2, va='top')
    xx = x + 2 * col_w
    yy = top + 3.1
    cv.fig.lines.append(Line2D([(xx + 1.25) / S.WIDTH_MM], [1 - (yy + 1.0) / H], marker='D', markersize=3.3,
                               markeredgecolor='white', markeredgewidth=0.4, color=S.INK, transform=cv.fig.transFigure))
    cv.text(xx + 3.2, yy + 0.1, 'Total B', fontsize=5.2, fontweight='bold', va='top')


def jitter(n, seed):
    return np.random.default_rng(seed).uniform(-0.26, 0.26, n)
