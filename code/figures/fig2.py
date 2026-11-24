"""Figure 2. Real-date economics of long-duration storage relative to lithium-ion across 11,054 records (2035 costs unless stated).

7 bespoke panels (a-g):
a  Ranked incremental return curves across Chinese provinces (310), European countries (132), and Japanese communities (10,612)
b  2D (9 discount rates x 3 cost years) hybrid-advantage phase heatmaps across the three study regimes
c  Five-component cost-bridge waterfall decomposition of mean B (fixed cost, VOM & charging, task energy, peak, net B)
d  Tri-regional spatial economic atlas of real-date mean system-cost saving B across 31 Chinese provinces, 15 European countries, and 5,306 Japanese communities
e  2D value-source plane: residual task-energy saving versus residual peak saving across 11,054 records
f  Joint scaling of A-CAES annual discharge and incremental return (2D HDR density contours + energy backbone + peak-saving markers)
g  Cascading threshold-split ridgeline density landscape across residual energy price (0.5x-1.25x) and service-boundary scenarios
"""
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle, Polygon

import style as S
import data as D
import econ as E
import geo_helper as G

H = 178.0
LEFT, RIGHT = 17.5, 180.0
RESTORE_TAB = S.V86 / 'revision/v85_restore_20261115/tables'


def panel_a_legend(cv, x, top, w):
    """Frameless compact legend aligned from Panel A's first subplot left edge (17.5) to Panel A's right edge (115.0)."""
    ax = cv.ax(x, top, w, 4.0, 'leg_a')
    ax.set_xlim(0, w)
    ax.set_ylim(0, 1)
    ax.axis('off')
    items = [
        ('provincial', 'China (310)', 0.0),
        ('national', 'Europe (132)', 21.8),
        ('community', 'Japan (10,612)', 45.0),
    ]
    for g, lab, xx in items:
        ax.scatter([xx + 0.85], [0.5], s=14, marker=S.MARKER[g], color=S.COLOR[g], edgecolor='white', lw=0.28, zorder=4, clip_on=False)
        ax.text(xx + 2.35, 0.5, lab, va='center', ha='left', fontsize=5.15, fontweight='bold', color=S.INK, zorder=4)
    ax.plot([70.4, 70.4], [0.22, 0.78], color='#BCCBD2', lw=0.52, zorder=3)
    ax.scatter([73.6], [0.5], s=12, marker='o', color=S.INK, edgecolor='white', lw=0.25, zorder=4)
    ax.text(75.1, 0.5, 'Hybrid', va='center', ha='left', fontsize=5.05, color=S.INK, zorder=4)
    t_lfp = ax.text(w, 0.5, 'LFP-only', va='center', ha='right', fontsize=5.05, color=S.MUTED, zorder=4)
    cv.fig.canvas.draw()
    bb = ax.transData.inverted().transform(t_lfp.get_window_extent(renderer=cv.fig.canvas.get_renderer()))
    ax.scatter([float(bb[0, 0]) - 1.45], [0.5], s=12, marker='o', facecolor='white', edgecolor=S.INK, lw=0.55, zorder=4)
    return ax


def panel_a(cv, top):
    """Panel a: Clean broken-axis ranked incremental return curves."""
    rec = D.records().set_index('uid')
    p = D.placements().set_index('uid')

    widths = {'provincial': 30.0, 'national': 31.5, 'community': 30.0}
    gap_x = 3.0
    h_hi, h_lo, gap_y = 9.2, 28.8, 1.5
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

        hi = cv.ax(x, top, w_mm, h_hi, f'a{i}hi')
        lo = cv.ax(x, top + h_hi + gap_y, w_mm, h_lo, f'a{i}lo')
        hi_ids.append(f'a{i}hi')
        lo_ids.append(f'a{i}lo')

        if n <= 200:
            xs = np.arange(n, dtype=float)
            for ax, (y0, y1) in ((hi, HI), (lo, LO)):
                ref = np.clip(3.6, y0, y1)
                yc = np.clip(yv, y0, y1)
                above = yc > ref
                below = yc < ref
                if above.any():
                    ax.fill_between(xs, ref, yc, where=above, interpolate=True, color=S.WIN, alpha=0.18, lw=0, zorder=1)
                if below.any():
                    ax.fill_between(xs, ref, yc, where=below, interpolate=True, color='#B8C4C8', alpha=0.18, lw=0, zorder=1)
                inside = (yv >= y0) & (yv <= y1)
                ax.scatter(xs[w_win & inside], yv[w_win & inside], s=6.5, marker=S.MARKER[g],
                           color=c, edgecolor='white', lw=0.22, zorder=4)
                ax.scatter(xs[~w_win & inside], yv[~w_win & inside], s=6.0, marker=S.MARKER[g],
                           facecolor='white', edgecolor=c, lw=0.55, zorder=4)
                ax.set_xlim(-2.0, n + 1.0)
                ax.set_ylim(y0, y1)
                S.grid(ax, 'y')
                ax.set_xticks([])
                ax.spines['bottom'].set_visible(y0 == LO[0])
                if y0 == LO[0]:
                    ax.set_xticks([0, 66, 131])
                    ax.set_xticklabels(['1', '66', '132'], fontsize=5.0)
            lo.set_xlabel('Ranked records', fontsize=5.2, color=S.MUTED, labelpad=1.4)
        else:
            xs_pct = np.linspace(0.0, 100.0, n)
            fin = np.isfinite(yv)
            for ax, (y0, y1) in ((hi, HI), (lo, LO)):
                ref = np.clip(3.6, y0, y1)
                yc = np.clip(yv[fin], y0, y1)
                xf = xs_pct[fin]
                above = yc > ref
                below = yc < ref
                if above.any():
                    ax.fill_between(xf, ref, yc, where=above, interpolate=True, color=S.WIN, alpha=0.20, lw=0, zorder=1)
                if below.any():
                    ax.fill_between(xf, ref, yc, where=below, interpolate=True, color='#B8C4C8', alpha=0.18, lw=0, zorder=1)
                if y0 == LO[0] and (~fin).sum() >= 2:
                    ax.plot(xs_pct[~fin], np.full((~fin).sum(), LO[0] + 0.30),
                            color='#95A4AB', lw=0.80, ls=(0, (1.5, 1.1)), zorder=3)
                m_lose = fin & (~w_win)
                m_win = fin & w_win
                if m_lose.any():
                    for seg in S.clip_lines([np.c_[xs_pct[m_lose], yv[m_lose]]], (-2, 102, y0, y1)):
                        ax.plot(seg[:, 0], seg[:, 1], color=c, lw=1.05, ls=(0, (2.2, 1.1)), zorder=4)
                if m_win.any():
                    for seg in S.clip_lines([np.c_[xs_pct[m_win], yv[m_win]]], (-2, 102, y0, y1)):
                        ax.plot(seg[:, 0], seg[:, 1], color=c, lw=1.55, zorder=4.5)
                if y0 == LO[0] and m_win.any():
                    ax.scatter([float(xs_pct[m_win][0])], [3.6], s=14, color=S.WIN, edgecolor='white', lw=0.35, zorder=5)
                ax.set_xlim(-2.0, 102.0)
                ax.set_ylim(y0, y1)
                S.grid(ax, 'y')
                if y0 == LO[0]:
                    ax.set_xticks([0, 50, 100])
                    ax.set_xticklabels(['0%', '50%', '100%'], fontsize=5.0)
                else:
                    ax.set_xticks([])
                    ax.spines['bottom'].set_visible(False)
            lo.set_xlabel('Percentile rank', fontsize=5.2, color=S.MUTED, labelpad=1.4)

        lo.axhline(3.6, color=S.WIN, lw=0.75, ls=(0, (3.0, 1.8)), zorder=3.5)
        lo.axhline(0.0, color='#CBD4D8', lw=0.45, zorder=1)
        hi.set_yticks([8, 12, 16])
        lo.set_yticks([-4, -2, 0, 2, 4])

        if i == 0:
            for ax_s, yy in ((hi, 0), (lo, 1)):
                ax_s.plot([-0.03, 0.03], [yy - 0.04, yy + 0.04], transform=ax_s.transAxes,
                          color=S.INK, lw=0.65, clip_on=False)
            lo.text(4.0, 4.15, '3.6%', ha='left', va='bottom', fontsize=5.2, fontweight='bold', color=S.WIN)
        else:
            for ax_s in (hi, lo):
                ax_s.spines['left'].set_visible(False)
                ax_s.tick_params(axis='y', length=0, labelleft=False)

        hi.plot([0, 1], [1.04, 1.04], transform=hi.transAxes, color=c, lw=1.35, solid_capstyle='butt', clip_on=False)
        cv.text(x + w_mm / 2, top - 5.5, S.SHORT_LABEL[g], ha='center', va='top', fontsize=5.7, fontweight='bold', color=c)
        cv.text(x + w_mm / 2, top - 2.9, f'{int(w_win.sum()):,}/{n:,} ({100 * w_win.mean():.0f}%)',
                ha='center', va='top', fontsize=5.1, color=S.INK)
        x += w_mm + gap_x

    cv.text(7.2, top + (h_hi + gap_y + h_lo) / 2, 'Incremental return (%)',
            rotation=90, ha='center', va='center', fontsize=5.9)
    return hi_ids, lo_ids


def panel_b(cv, x0, top, total_w, h):
    """Panel b: 2D (9 discount rates x 3 cost years) hybrid-advantage phase heatmaps across the 3 regimes."""
    s = D.summary()
    s = s[s.group.isin(S.MAIN)]
    years = [2025, 2030, 2035]
    rates = E.RATES

    sub_w = 15.6
    gap = (total_w - 3 * sub_w) / 2.0
    b_ids = []
    out_rows = []

    for gi, g in enumerate(S.MAIN):
        xx = x0 + gi * (sub_w + gap)
        ax = cv.ax(xx, top, sub_w, h, f'b{gi}')
        b_ids.append(f'b{gi}')
        mat = np.zeros((len(rates), len(years)))
        sub_s = s[s.group == g].set_index('year')
        for ri, rt in enumerate(rates):
            for yi, yr in enumerate(years):
                val = 100.0 * float(sub_s.loc[yr, f'real_win_{rt:g}']) / float(sub_s.loc[yr, 'n'])
                mat[ri, yi] = val
                out_rows.append(dict(group=g, cost_year=yr, rate_pct=rt, hybrid_cheaper_pct=val))

        ax.imshow(mat, cmap=S.PHASE_CMAP, vmin=0.0, vmax=100.0, aspect='auto', zorder=1)
        for r_line in np.arange(-0.5, len(rates), 1.0):
            ax.axhline(r_line, color='white', lw=0.45, zorder=2)
        for c_line in np.arange(-0.5, len(years), 1.0):
            ax.axvline(c_line, color='white', lw=0.45, zorder=2)

        r36 = rates.index(3.6)
        ax.add_patch(Rectangle((1.52, r36 - 0.48), 0.96, 0.96, facecolor='none',
                               edgecolor=S.COLOR[g], lw=0.95, zorder=4))

        for ri, rt in enumerate(rates):
            for yi, yr in enumerate(years):
                v = mat[ri, yi]
                show = (yi == 2) or (rt in (0, 2, 3.6, 6, 10))
                if not show or v < 0.05:
                    continue
                txt = '<1' if v < 1.0 else ('>99' if v > 99.0 and v < 99.99 else f'{v:.0f}')
                rgba = S.PHASE_CMAP(v / 100.0)
                tc = 'white' if S.luminance(rgba) < 0.52 else S.INK
                ax.text(yi, ri, txt, ha='center', va='center', fontsize=5.0,
                        fontweight='bold' if (ri == r36 and yi == 2) else 'normal', color=tc, zorder=5)

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["'25", "'30", "'35"], fontsize=5.0)
        ax.set_yticks(range(len(rates)))
        if gi == 0:
            ax.set_yticklabels([f'{r:g}%' for r in rates], fontsize=5.1)
            ax.get_yticklabels()[r36].set_fontweight('bold')
            ax.set_ylabel('Real discount rate')
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis='both', length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)

        ax.plot([0, 1], [1.03, 1.03], transform=ax.transAxes, color=S.COLOR[g], lw=1.35, clip_on=False)
        cv.text(xx + sub_w / 2, top - 3.2, S.SHORT_LABEL[g], ha='center', va='top',
                fontsize=5.6, fontweight='bold', color=S.COLOR[g])

    cv.text(x0 + total_w / 2, top - 6.0, 'Hybrid cheaper share (%)', ha='center', va='top',
            fontsize=5.1, color=S.MUTED)
    cv.text(x0 + total_w / 2, top + h + 4.3, 'Technology cost year', ha='center', va='top',
            fontsize=5.4, color=S.INK)
    return b_ids, pd.DataFrame(out_rows)


def panel_c(cv, x, top, w, h):
    """Panel c: Stepwise 5-stage cost-bridge waterfall comparing China, Europe and Japan."""
    rec = D.records()
    rec_m = rec[rec.group.isin(S.MAIN)]
    m = rec_m.groupby('group', observed=True)[S.COMP + ['B']].mean()
    pos_uids = D.cn_positive_uids()
    m_pos = rec[rec.uid.isin(pos_uids)][S.COMP + ['B']].mean()

    ax = cv.ax(x, top, w, h, 'c')
    S.grid(ax, 'y')
    ax.axhline(0.0, color=S.INK, lw=0.65, zorder=2)

    stages = ['Fixed', 'Chg &\nVOM', 'Task\nenergy', 'Peak', 'Net B']
    offsets = [-0.24, 0.0, 0.24]
    bw = 0.21

    for gi, g in enumerate(S.MAIN):
        row = m.loc[g]
        c = S.COLOR[g]
        d_fix = float(row['B_fixed'])
        d_cv = float(row['B_charging'] + row['B_variable_om'])
        d_en = float(row['B_residual_energy'])
        tot = float(row['B'])

        starts = [0.0, d_fix, d_fix + d_cv, d_fix + d_cv + d_en, 0.0]
        ends = [d_fix, d_fix + d_cv, d_fix + d_cv + d_en, tot, tot]

        xs_c = [si + offsets[gi] for si in range(5)]
        for si in range(5):
            y0, y1 = starts[si], ends[si]
            bot, ht = min(y0, y1), abs(y1 - y0)
            is_net = (si == 4)
            ax.add_patch(Rectangle(
                (xs_c[si] - bw / 2, bot), bw, max(ht, 0.7),
                facecolor=c if is_net else S.TINT[g],
                edgecolor=S.INK if is_net else c,
                lw=0.65 if is_net else 0.60,
                zorder=4 if is_net else 3,
            ))
            if si < 4:
                ax.plot([xs_c[si] + bw / 2, xs_c[si + 1] - bw / 2], [y1, y1],
                        color=c, lw=0.45, ls=':', alpha=0.75, zorder=2.5)

    ax.scatter([4 + offsets[0]], [float(m_pos['B'])], s=17, marker='D',
               facecolor='white', edgecolor=S.DEEP['provincial'], lw=0.80, zorder=6)
    ax.annotate(f"CN pos: {m_pos['B']:.1f}", (4 + offsets[0], float(m_pos['B'])),
                xytext=(2.68, 7.8), textcoords='data', ha='center', va='bottom',
                fontsize=5.0, fontweight='bold', color=S.DEEP['provincial'],
                arrowprops=dict(arrowstyle='-', lw=0.45, color=S.DEEP['provincial'], shrinkA=1, shrinkB=2))

    ax.text(4 + offsets[0] - 0.04, -15.8, '−13.7', ha='center', va='top', fontsize=5.0, fontweight='bold', color=S.DEEP['provincial'])
    ax.annotate('−3.2', (4 + offsets[1], -3.2), xytext=(3.28, -11.5), textcoords='data',
                ha='center', va='center', fontsize=5.0, fontweight='bold', color=S.DEEP['national'],
                arrowprops=dict(arrowstyle='-', lw=0.40, color=S.DEEP['national'], shrinkA=1, shrinkB=2))
    ax.text(4 + offsets[2] + 0.06, -5.2, '−3.0', ha='center', va='top', fontsize=5.0, fontweight='bold', color=S.DEEP['community'])

    ax.set_xlim(-0.55, 4.55)
    ax.set_ylim(-48.0, 17.5)
    ax.set_xticks(range(5))
    ax.set_xticklabels(stages, fontsize=5.0)
    ax.set_yticks([-40, -20, 0, 15])
    ax.set_ylabel(f'Mean B bridge ({S.UNIT})')
    return m


def panel_d(cv, x0, top, hmm):
    """Panel d: Tri-regional spatial economic atlas of real-date mean system-cost saving B."""
    rec = D.records()
    rec_m = rec[rec.group.isin(S.MAIN)].copy()
    prem = float(D.RM.PREM36[2035])
    rec_m['op'] = rec_m.B + prem

    cn_agg = rec_m[rec_m.group == 'provincial'].groupby('system_id').agg(
        mean_B=('B', 'mean'), win_share=('win36', 'mean')
    )
    eu_agg = rec_m[rec_m.group == 'national'].groupby('system_id').agg(
        mean_B=('B', 'mean'), win_share=('win36', 'mean')
    )
    jp_agg = rec_m[rec_m.group == 'community'].groupby('system_id').agg(
        mean_B=('B', 'mean'), win_share=('win36', 'mean')
    )

    norm = TwoSlopeNorm(vmin=-35.0, vcenter=0.0, vmax=45.0)
    cmap = S.DIV_B_CMAP

    x = x0
    ax_cn, ax_scs, w_cn = G.draw_china_choropleth(cv, x, top, hmm, cn_agg['mean_B'].to_dict(), cmap, norm, 'd_cn')
    n_cn_pos = int((cn_agg['mean_B'] >= 0).sum())
    cv.text(x + w_cn / 2, top - 2.9, f'China ({n_cn_pos}/31 B ≥ 0)', ha='center', va='top',
            fontsize=5.05, fontweight='bold', color=S.COLOR['provincial'])
    x += w_cn + 2.1

    ax_eu, w_eu = G.draw_europe_choropleth(cv, x, top, hmm, eu_agg['mean_B'].to_dict(), cmap, norm, 'd_eu')
    n_eu_pos = int((eu_agg['mean_B'] >= 0).sum())
    cv.text(x + w_eu / 2, top - 2.9, f'Europe ({n_eu_pos}/15 B ≥ 0)', ha='center', va='top',
            fontsize=5.05, fontweight='bold', color=S.COLOR['national'])
    x += w_eu + 2.1

    ax_jp, w_jp = G.draw_japan_choropleth(cv, x, top, hmm, jp_agg['mean_B'].to_dict(), cmap, norm, 'd_jp', pt_size=0.68)
    n_jp_pos = int((jp_agg['mean_B'] >= 0).sum())
    cv.text(x + w_jp / 2, top - 2.9, f'Japan ({n_jp_pos:,}/5,306 B ≥ 0)', ha='center', va='top',
            fontsize=5.05, fontweight='bold', color=S.COLOR['community'])
    x += w_jp + 2.2

    # Vertical diverging colorbar locked so right edge + labels sit flush at RIGHT = 180.0 mm
    cax = cv.ax(173.4, top + 1.5, 2.1, hmm - 3.0, 'cb_d')
    sm = S.mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = cv.fig.colorbar(sm, cax=cax, ticks=[-35, 0, 45])
    cb.ax.set_yticklabels(['−35', '0', '45'], fontsize=5.0)
    cb.ax.tick_params(length=1.6, pad=1.2)
    cb.outline.set_linewidth(0.45)
    cb.outline.set_edgecolor('#8FA0A8')
    cv.text(RIGHT, top - 2.9, 'Mean B', ha='right', va='top', fontsize=5.0, fontweight='bold', color=S.INK)
    cv.text(RIGHT, top + hmm + 0.8, '(USD kW⁻¹ yr⁻¹)', ha='right', va='top', fontsize=5.0, color=S.MUTED)

    map_rows = pd.concat([
        cn_agg.assign(group='provincial').reset_index(),
        eu_agg.assign(group='national').reset_index(),
        jp_agg.assign(group='community').reset_index(),
    ], ignore_index=True)
    return ['d_cn', 'd_eu', 'd_jp'], rec_m[['uid', 'group', 'op']], map_rows


def panel_e(cv, x, top, w, h):
    """Panel e: 2D Value-Source Plane (Residual task-energy saving vs Residual peak saving)."""
    rec = D.records()
    rec = rec[rec.group.isin(S.MAIN)].copy()

    ax = cv.ax(x, top, w, h, 'e')
    S.grid(ax, 'y')

    poly = S.clip_ring([(0, 38), (60, -22), (60, 108), (-3, 108)], (-3, 58, -16, 106))
    if len(poly) >= 3:
        ax.add_patch(Polygon(poly, closed=True, facecolor=S.WIN_PALE, edgecolor='none', zorder=0))
    seg_be = S.clip_lines([np.array([[-3.0, 41.0], [57.0, -19.0]])], (-3, 58, -16, 106))
    for sg in seg_be:
        ax.plot(sg[:, 0], sg[:, 1], color=S.WIN, lw=0.65, ls=(0, (2.6, 1.6)), zorder=2)

    ax.axhline(0.0, color='#C6D1D5', lw=0.50, zorder=1)

    jp = rec[rec.group == 'community']
    jq05, jq25, jq75, jq95 = np.quantile(jp.B_residual_energy, [0.05, 0.25, 0.75, 0.95])
    ax.plot([jq05, jq95], [0.0, 0.0], color=S.COLOR['community'], lw=1.6, zorder=4)
    ax.add_patch(Rectangle((jq25, -2.4), jq75 - jq25, 4.8, facecolor=S.COLOR['community'], alpha=0.65, lw=0, zorder=4.2))

    for g in ('national', 'provincial'):
        z = rec[rec.group == g]
        w_m = z.win36.to_numpy(bool)
        c = S.COLOR[g]
        ax.scatter(z.B_residual_energy[~w_m], z.B_residual_peak[~w_m], s=7.5, marker=S.MARKER[g],
                   facecolor='white', edgecolor=c, lw=0.50, alpha=0.78, zorder=3.5)
        ax.scatter(z.B_residual_energy[w_m], z.B_residual_peak[w_m], s=9.5, marker=S.MARKER[g],
                   color=c, edgecolor='white', lw=0.25, alpha=0.88, zorder=4.5)

    ax.text(0.5, 1.03, 'Peak + energy (74 CN, 9 EU)', transform=ax.transAxes,
            ha='center', va='bottom', fontsize=5.0, fontweight='bold', color=S.INK)
    ax.text(19.0, -11.5, 'Energy-only regime', ha='center', va='center', fontsize=5.0, color=S.MUTED)

    ax.set_xlim(-3, 58)
    ax.set_ylim(-16, 106)
    ax.set_xticks([0, 20, 40, 55])
    ax.set_yticks([0, 30, 60, 90])
    ax.set_xlabel(f'Task-energy saving ({S.UNIT})')
    ax.set_ylabel(f'Peak saving ({S.UNIT})')


def panel_f(cv, x, top, w, h):
    """Panel f: Joint A-CAES annual discharge vs incremental return."""
    rec = D.records()
    rec = rec[rec.group.isin(S.MAIN)].copy()
    rho_eq, _, by_g = D.equal_regime_spearman(rec, 'hybrid_tail_discharge_mwh_per_mw', 'roi_2035')
    assert round(rho_eq, 2) == 0.66 and round(by_g['community'], 2) == 0.87

    ax = cv.ax(x, top, w, h, 'f')
    S.grid(ax, 'y')
    xlim, ylim = (-18.0, 1170.0), (-5.3, 16.2)
    ax.axhspan(3.6, ylim[1], facecolor=S.WIN_PALE, zorder=0)

    jp = rec[(rec.group == 'community') & np.isfinite(rec.roi_2035)]
    jx = jp.hybrid_tail_discharge_mwh_per_mw.to_numpy(float)
    jy = jp.roi_2035.to_numpy(float)
    c_jp = S.COLOR['community']
    rng = np.random.default_rng(2026)
    idx_s = rng.choice(len(jx), size=min(len(jx), 3500), replace=False)
    sx, sy = xlim[1] - xlim[0], ylim[1] - ylim[0]
    kde2 = gaussian_kde(np.vstack([(jx[idx_s] - xlim[0]) / sx, (jy[idx_s] - ylim[0]) / sy]), bw_method=0.20)
    gx, gy = np.linspace(0.01, 0.99, 75), np.linspace(0.01, 0.99, 75)
    GX, GY = np.meshgrid(gx, gy)
    Z = kde2(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
    vals = np.sort(Z.ravel())[::-1]
    csum = np.cumsum(vals)
    t88 = vals[min(len(vals) - 1, np.searchsorted(csum, 0.88 * csum[-1]))]
    t50 = vals[min(len(vals) - 1, np.searchsorted(csum, 0.50 * csum[-1]))]
    Xp, Yp = xlim[0] + GX * sx, ylim[0] + GY * sy
    ax.contourf(Xp, Yp, Z, levels=[t88, Z.max() * 1.05], colors=[c_jp], alpha=0.16, zorder=1.2)
    ax.contourf(Xp, Yp, Z, levels=[t50, Z.max() * 1.05], colors=[c_jp], alpha=0.24, zorder=1.4)
    ax.contour(Xp, Yp, Z, levels=[t88, t50], colors=[c_jp], linewidths=[0.48, 0.70], alpha=0.65, zorder=1.8)

    edges_jp = np.quantile(jx, np.linspace(0.01, 0.99, 9))
    xm_j, ym_j = [], []
    for lo_e, hi_e in zip(edges_jp[:-1], edges_jp[1:]):
        mk = (jx >= lo_e) & (jx <= hi_e)
        if mk.sum() >= 10:
            xm_j.append(float(np.median(jx[mk])))
            ym_j.append(float(np.median(jy[mk])))
    ax.plot(xm_j, ym_j, color=c_jp, lw=1.3, zorder=2.6)

    cn_eu_np = rec[(rec.group != 'community') & np.isfinite(rec.roi_2035) & (rec.B_residual_peak <= 0.5)]
    xv_all = np.r_[cn_eu_np.hybrid_tail_discharge_mwh_per_mw.to_numpy(float), jx[::40]]
    yv_all = np.r_[cn_eu_np.roi_2035.to_numpy(float), jy[::40]]
    edges = np.quantile(xv_all, np.linspace(0, 1, 7))
    edges[0] -= 1e-6; edges[-1] += 1e-6
    xm, ql, qm, qu = [], [], [], []
    for lo_e, hi_e in zip(edges[:-1], edges[1:]):
        m = (xv_all >= lo_e) & (xv_all <= hi_e)
        if m.sum() >= 5:
            xm.append(np.median(xv_all[m]))
            l, md, u = np.quantile(yv_all[m], [0.20, 0.50, 0.80])
            ql.append(l); qm.append(md); qu.append(u)
    ax.fill_between(xm, ql, qu, color='#D7E3E6', alpha=0.45, lw=0, zorder=1.0)
    ax.plot(xm, qm, color='#455D66', lw=0.75, ls=(0, (3.0, 1.8)), zorder=2.0)

    for g in ('provincial', 'national'):
        z = rec[(rec.group == g) & np.isfinite(rec.roi_2035)]
        pk = z.B_residual_peak > 0.5
        ax.scatter(z.hybrid_tail_discharge_mwh_per_mw[~pk], z.roi_2035[~pk], s=7.5, marker=S.MARKER[g],
                   color=S.COLOR[g], edgecolor='white', lw=0.22, alpha=0.85, zorder=3.5)
        if pk.any():
            ax.scatter(z.hybrid_tail_discharge_mwh_per_mw[pk], z.roi_2035[pk], s=11.5, marker=S.MARKER[g],
                       color=S.COLOR[g], edgecolor=S.INK, lw=0.58, alpha=0.92, zorder=4.5)

    ax.axhline(3.6, color=S.WIN, lw=0.72, ls=(0, (3.0, 1.8)), zorder=2.4)
    ax.text(1145, 4.35, '3.6%', ha='right', va='bottom', fontsize=5.05, fontweight='bold', color=S.WIN)
    ax.text(0.5, 1.03, f'ρ = {rho_eq:.2f} (CN {by_g["provincial"]:.2f}, EU {by_g["national"]:.2f}, JP {by_g["community"]:.2f})',
            transform=ax.transAxes, ha='center', va='bottom', fontsize=5.0, fontweight='bold', color=S.INK)
    ax.text(1145, -3.5, 'Outlined: peak > 0.5', ha='right', va='center', fontsize=5.0, color=S.MUTED)

    ax.set_xlim(*xlim)
    ax.set_xticks([0, 500, 1000])
    ax.set_xticklabels(['0', '500', '1,000'])
    ax.set_ylim(*ylim)
    ax.set_yticks([-4, 0, 4, 8, 12, 16])
    ax.set_xlabel(f'A-CAES discharge ({S.DUNIT})')
    ax.set_ylabel('Incremental return (%)')
    return rho_eq


def panel_g(cv, x0, top, w_ridge, h):
    """Panel g: Cascading threshold-split ridgeline density landscape across residual-price & service-scope scenarios."""
    pr = pd.read_csv(RESTORE_TAB / 'price_current.csv', dtype={'uid': str})
    ru = pd.read_csv(RESTORE_TAB / 'rule_current.csv', dtype={'uid': str})
    pr_s = pd.read_csv(RESTORE_TAB / 'price_summary_current.csv')
    ru_s = pd.read_csv(RESTORE_TAB / 'rule_summary_current.csv')

    scenarios = [
        ('price', 'x0.5', '0.5× price'),
        ('price', 'x0.75', '0.75× price'),
        ('price', 'x1', '1× central'),
        ('price', 'x1.25', '1.25× price'),
        ('rule', 'all_task', 'All task h'),
        ('rule', 'netted_all_task', 'Netted all h'),
    ]

    ax = cv.ax(x0, top, w_ridge, h, 'g')
    ax.grid(False)

    xlim = (-40.0, 44.0)
    xg = np.linspace(xlim[0], xlim[1], 240)
    n_sc = len(scenarios)

    ax.axvspan(0.0, xlim[1], facecolor=S.WIN_PALE, zorder=0)
    ax.plot([0.0, 0.0], [n_sc - 0.72, -0.76], color=S.WIN, lw=0.75, ls=(0, (2.8, 1.6)), zorder=5)

    sens_rows = []
    for si, (src_kind, var_key, _) in enumerate(scenarios):
        df_src = pr[pr.variant == var_key] if src_kind == 'price' else ru[ru.variant == var_key]
        sm_src = (pr_s[pr_s.variant == var_key] if src_kind == 'price' else ru_s[ru_s.variant == var_key]).set_index('group')

        dens_mix = np.zeros_like(xg)
        meds_g = {}
        shares_txt = []
        for g in S.MAIN:
            vg = np.clip(df_src.loc[df_src.group == g, 'B'].to_numpy(float), xlim[0] + 0.8, xlim[1] - 0.8)
            meds_g[g] = float(np.median(vg))
            if np.std(vg) > 0.4:
                kg = gaussian_kde(vg, bw_method=0.26)(xg)
            else:
                kg = np.exp(-0.5 * ((xg - float(np.mean(vg))) / 1.4) ** 2)
            if kg.max() > 0:
                kg = kg / np.trapezoid(kg, xg)
            dens_mix += kg / 3.0

            sh = 100.0 * float(sm_src.loc[g, 'win']) / float(sm_src.loc[g, 'n'])
            sens_rows.append(dict(dimension=src_kind, level=var_key, group=g, hybrid_cheaper_pct=sh))
            shares_txt.append('100' if sh > 99.5 else ('<1' if 0 < sh < 0.5 else f'{sh:.0f}'))

        y_base = float(si)
        h_wave = dens_mix / dens_mix.max() * 0.76
        y_top_curve = y_base - h_wave
        z_base = 2.0 + si * 0.4

        ax.plot([xlim[0], xlim[1]], [y_base, y_base], color='#D5E0E4', lw=0.45, zorder=z_base)

        neg_m = xg <= 0.2
        pos_m = xg >= -0.2
        ax.fill_between(xg[neg_m], y_top_curve[neg_m], y_base,
                        facecolor='#DCE5E9', edgecolor='none', alpha=0.88, zorder=z_base + 0.1)
        ax.fill_between(xg[pos_m], y_top_curve[pos_m], y_base,
                        facecolor=S.WIN_MID if var_key != 'x1' else S.WIN,
                        edgecolor='none', alpha=0.82 if var_key != 'x1' else 0.92, zorder=z_base + 0.15)
        ax.plot(xg, y_top_curve, color=S.INK if var_key == 'x1' else '#4B5E67',
                lw=0.85 if var_key == 'x1' else 0.58, zorder=z_base + 0.2)

        for g in S.MAIN:
            ax.scatter([meds_g[g]], [y_base - 0.08], s=8.5, marker=S.MARKER[g],
                       color=S.COLOR[g], edgecolor='white', lw=0.22, zorder=z_base + 0.3)

        # Right-align percentage summary strictly at RIGHT = 180.0 mm in canvas coordinates
        y_frac = ((y_base - 0.22) - (-0.88)) / ((n_sc - 0.72) - (-0.88))
        cv.text(RIGHT, top + y_frac * h, '/'.join(shares_txt) + '%',
                ha='right', va='center', fontsize=5.0,
                fontweight='bold' if var_key in ('x1', 'x1.25', 'all_task') else 'normal',
                color=S.INK if var_key == 'x1' else S.MUTED)

    ax.axhline(3.22, color='#B8C6CC', lw=0.55, ls=':', zorder=1.5)
    cv.text(RIGHT, top - 0.3, 'CN/EU/JP', ha='right', va='bottom',
            fontsize=5.0, fontweight='bold', color=S.INK)
    ax.text(0.0, -0.88, 'B = 0', ha='center', va='bottom',
            fontsize=5.0, fontweight='bold', color=S.WIN, clip_on=False)

    ax.set_xlim(*xlim)
    ax.set_xticks([-35, 0, 35])
    ax.set_xticklabels(['−35', '0', '35'])
    ax.set_ylim(n_sc - 0.72, -0.88)
    ax.set_yticks([i - 0.20 for i in range(n_sc)])
    ax.set_yticklabels([s[2] for s in scenarios], fontsize=5.0)
    ax.get_yticklabels()[2].set_fontweight('bold')
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel(f'System-cost saving B ({S.UNIT})')
    return pd.DataFrame(sens_rows)


def main():
    S.setup()
    cv = S.Canvas(H)

    # Strict frameless legend above Panel a (spanning LEFT = 17.5 .. 115.0, aligned with Panel a left & right edges)
    panel_a_legend(cv, LEFT, 1.6, 115.0 - LEFT)

    # Row 1: a (Ranked returns) + b (2D Rate x Year phase heatmaps)
    top1 = 14.5
    cv.letter(1.5, top1 - 6.0, 'a')
    hi_ids, lo_ids = panel_a(cv, top1)

    cv.letter(119.5, top1 - 6.0, 'b')
    b_ids, b_df = panel_b(cv, 128.0, top1, RIGHT - 128.0, 39.5)

    # Row 2: c (5-stage Waterfall bridge) + d (Tri-regional spatial economic atlas of mean B, locked to RIGHT = 180.0)
    top2 = 68.5
    cv.letter(1.5, top2 - 4.2, 'c')
    m_comp = panel_c(cv, LEFT, top2, 43.5, 31.5)

    cv.letter(63.5, top2 - 4.2, 'd')
    d_map_ids, op_df, map_df = panel_d(cv, 69.2, top2 + 2.0, 27.1)

    # Row 3: e (2D Energy vs Peak plane), f (Discharge vs ROI), g (Cascading ridgeline + right-aligned shares at RIGHT = 180.0)
    top3, h3 = 117.5, 44.5
    cv.letter(1.5, top3 - 4.2, 'e')
    panel_e(cv, LEFT, top3, 39.5, h3)

    cv.letter(60.5, top3 - 4.2, 'f')
    rho_f = panel_f(cv, 70.0, top3, 42.5, h3)

    cv.letter(115.0, top3 - 4.2, 'g')
    g_df = panel_g(cv, 125.0, top3, 43.0, h3)

    # Export complete source data
    S.SD.mkdir(parents=True, exist_ok=True)
    rec = D.records()
    rec = rec[rec.group.isin(S.MAIN)]
    rec[['uid', 'group', 'system_id', 'observation_year', 'roi_2035', 'win36', 'B', *S.COMP, 'efc_acaes',
         'hybrid_tail_discharge_mwh_per_mw']].merge(op_df[['uid', 'op']], on='uid').rename(
        columns={'op': 'operating_saving_usd_kw_year'}).to_csv(S.SD / 'figure2_records.csv', index=False)
    b_df.to_csv(S.SD / 'figure2bc_who_is_cheaper_real.csv', index=False)
    b_df.to_csv(S.SD / 'figure2b_rate_year_phase.csv', index=False)
    m_comp.reset_index().to_csv(S.SD / 'figure2d_B_components_mean.csv', index=False)
    m_comp.reset_index().to_csv(S.SD / 'figure2c_B_components_mean.csv', index=False)
    map_df.to_csv(S.SD / 'figure2d_spatial_mean_B.csv', index=False)
    g_df.to_csv(S.SD / 'figure2g_price_and_rule_sensitivity.csv', index=False)

    S.save(
        cv,
        'main_figure2',
        rows=[hi_ids, lo_ids, b_ids, d_map_ids, ['e', 'f', 'g']],
        cols=[],
        exempt=[
            dict(panels=['leg_a', 'd_cn_scs', 'cb_d', 'c'], checks=['all'],
                 reason='Panel a legend, South China Sea inset, colorbar, and waterfall paired beside map atlas'),
            dict(panels=hi_ids + lo_ids + b_ids + d_map_ids + ['e', 'f', 'g'],
                 checks=['panel-width'], reason='Intentional multi-panel widths tailored to maps, heatmaps, and ridgelines'),
        ],
        audit=dict(
            panels=['a', 'b', 'c', 'd', 'e', 'f', 'g'],
            panel_count=7,
            spearman_annual_discharge_vs_return_equal_regime=rho_f,
            records=11054,
        ),
    )


if __name__ == '__main__':
    main()
