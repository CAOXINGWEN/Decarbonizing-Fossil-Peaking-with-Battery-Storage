"""Figure 4. Charging-window saturation mechanism, multi-day event extension and 2x2x2 factorial constraints across 11,054 records.

7 bespoke panels (a-g):
a  Dual-technology scaling against long charging windows: steep 24 h A-CAES climb (rho = 0.77) vs flat 4 h LFP plateau (JP rho = 0.00)
b  4 h LFP annual discharge vs LFP-fill windows (rho = 0.97) with embedded short-window selectivity comparison
c  Hexagonal mean-B surface in (long charging windows x additional hybrid delivery) space across 11,054 records
d  Coupled two-window analytical saturation diagram: stored/unstored fate bars (top) and two-call delivery tent (bottom)
e  Continuous 1-to-12-day event-extension delta-B decay trajectories and IQR ribbons across the three study regimes
f  Multi-regime 2x2x2 factorial quantile interval forest (including China proportional vs monthly peak-clipping contrast)
g  8-cell x 3-regime factorial portfolio win-share heatmap matrix across (E x P x Access) combinations
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

import style as S
import data as D

H = 184.0
LEFT, RIGHT = 17.5, 180.0
A_CAES, LFP_C = '#19646B', '#5D6F78'
WIN2_C = '#5EA9A3'
SPILL_FC, SPILL_EC = '#F7E7D8', '#BF8454'
MS = {'provincial': 8.5, 'national': 10.0, 'community': 11.0}

B_CMAP = LinearSegmentedColormap.from_list(
    'b_div_f4',
    [
        (0.00, '#C4533A'),
        (0.24, '#E8B39D'),
        (0.3500, '#F7F4EC'),
        (0.58, '#469B97'),
        (1.00, '#0F4D56'),
    ],
)


def striped_rect(ax, x0, y0, w, h, fc, ec, lw=0.42, n_lines=5, zorder=3):
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor=fc, edgecolor='none', lw=0, zorder=zorder))
    box = (x0, x0 + w, y0, y0 + h)
    raw = []
    for t in np.linspace(-0.85, 0.85, n_lines):
        xc = x0 + (0.5 + t) * w
        raw.append(np.array([[xc - 0.55 * w, y0 - 0.2 * h], [xc + 0.55 * w, y0 + 1.2 * h]]))
    for seg in S.clip_lines(raw, box):
        ax.plot(seg[:, 0], seg[:, 1], color=ec, lw=0.40, zorder=zorder + 0.1)
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor='none', edgecolor=ec, lw=lw, zorder=zorder + 0.2))


def kde_2d_and_spine(ax, xv, yv, color, xlim, ylim, ls='-', alpha_scale=1.0):
    rng = np.random.default_rng(2026)
    idx = rng.choice(len(xv), size=min(len(xv), 3500), replace=False)
    sx, sy = xlim[1] - xlim[0], ylim[1] - ylim[0]
    pts = np.vstack([(xv[idx] - xlim[0]) / sx, (yv[idx] - ylim[0]) / sy])
    kde = gaussian_kde(pts, bw_method=0.22)
    gx, gy = np.linspace(0.01, 0.99, 70), np.linspace(0.01, 0.99, 70)
    GX, GY = np.meshgrid(gx, gy)
    Z = kde(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
    vals = np.sort(Z.ravel())[::-1]
    csum = np.cumsum(vals)
    t88 = vals[min(len(vals) - 1, np.searchsorted(csum, 0.88 * csum[-1]))]
    t50 = vals[min(len(vals) - 1, np.searchsorted(csum, 0.50 * csum[-1]))]
    Xp, Yp = xlim[0] + GX * sx, ylim[0] + GY * sy
    ax.contourf(Xp, Yp, Z, levels=[t88, Z.max() * 1.05], colors=[color], alpha=0.14 * alpha_scale, zorder=1.2)
    ax.contourf(Xp, Yp, Z, levels=[t50, Z.max() * 1.05], colors=[color], alpha=0.21 * alpha_scale, zorder=1.4)
    ax.contour(Xp, Yp, Z, levels=[t88, t50], colors=[color], linewidths=[0.45, 0.65],
               linestyles=ls, alpha=0.58 * alpha_scale, zorder=1.8)

    edges = np.unique(np.quantile(xv, np.linspace(0.02, 0.98, 9)))
    xm, ym = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (xv >= lo) & (xv <= hi)
        if m.sum() >= 10:
            xm.append(float(np.median(xv[m])))
            ym.append(float(np.median(yv[m])))
    if len(xm) >= 2:
        ax.plot(xm, ym, color=color, lw=1.25, ls=ls, zorder=2.6)


def panel_a(cv, x, top, w, h, d, r_a, bg_a, r_b, bg_b):
    """Panel a: Dual-technology scaling against long charging windows (24 h A-CAES vs 4 h LFP on tight axes)."""
    ax = cv.ax(x, top, w, h, 'a')
    S.grid(ax, 'y')
    xlim, ylim = (-1.5, 56.0), (-15.0, 1180.0)

    # 1. 24 h A-CAES (vivid regime hues)
    for g in ('community', 'national', 'provincial'):
        z = d[d.group == g]
        xv = z.long_windows.to_numpy(float)
        yv = z.hybrid_tail_discharge_mwh_per_mw.to_numpy(float)
        if g == 'community':
            kde_2d_and_spine(ax, xv, yv, S.COLOR[g], xlim, ylim)
        else:
            ax.scatter(xv, yv, s=MS[g], marker=S.MARKER[g], color=S.COLOR[g],
                       edgecolor='white', lw=0.25, alpha=0.86, zorder=3.5)

    # 2. 4 h LFP plateau band (muted cool slate dashed spine + IQR envelope across positive-window bins)
    pos_d = d[d.long_windows > 0]
    edges = np.unique(np.quantile(pos_d.long_windows, np.linspace(0.02, 0.98, 9)))
    xl_m, ql_lo, ql_md, ql_hi = [], [], [], []
    for lo_e, hi_e in zip(edges[:-1], edges[1:]):
        mk = (pos_d.long_windows >= lo_e) & (pos_d.long_windows <= hi_e)
        if mk.sum() >= 10:
            xl_m.append(float(np.median(pos_d.long_windows[mk])))
            l, md, u = np.quantile(pos_d.lfp_tail_discharge_mwh_per_mw[mk], [0.20, 0.50, 0.80])
            ql_lo.append(float(l)); ql_md.append(float(md)); ql_hi.append(float(u))
    ax.fill_between(xl_m, ql_lo, ql_hi, color='#CFDADF', alpha=0.55, lw=0, zorder=2.0)
    ax.plot(xl_m, ql_md, color='#465862', lw=1.25, ls=(0, (2.4, 1.3)), zorder=3.0)

    ax.text(0.0, 1.03, f'A-CAES ρ = {r_a:.2f} (CN {bg_a["provincial"]:.2f}, EU {bg_a["national"]:.2f}, JP {bg_a["community"]:.2f})',
            transform=ax.transAxes, ha='left', va='bottom', fontsize=5.0, fontweight='bold', color=S.INK)
    ax.text(54.5, 115, f'4 h LFP (dashed): JP ρ = {bg_b["community"]:.2f}, EU {bg_b["national"]:.2f}',
            ha='right', va='top', fontsize=5.0, color='#465862')

    ax.set_xlim(*xlim); ax.set_xticks([0, 15, 30, 45, 55])
    ax.set_ylim(*ylim); ax.set_yticks([0, 350, 700, 1050]); ax.set_yticklabels(['0', '350', '700', '1,050'])
    ax.set_xlabel('Long charging windows (yr⁻¹)')
    ax.set_ylabel(f'Annual discharge ({S.DUNIT})')


def panel_b(cv, x, top, w, h, d, r_c, bg_c):
    """Panel b: 4 h LFP discharge vs LFP-fill windows + embedded short-window selectivity inset."""
    ax = cv.ax(x, top, w, h, 'b')
    S.grid(ax, 'y')
    xlim, ylim = (-4.0, 184.0), (-10.0, 545.0)

    for g in ('community', 'national', 'provincial'):
        z = d[d.group == g]
        xv = z.lfp_fill_windows.to_numpy(float)
        yv = z.lfp_tail_discharge_mwh_per_mw.to_numpy(float)
        if g == 'community':
            kde_2d_and_spine(ax, xv, yv, S.COLOR[g], xlim, ylim)
        else:
            ax.scatter(xv, yv, s=MS[g], marker=S.MARKER[g], color=S.COLOR[g],
                       edgecolor='white', lw=0.25, alpha=0.86, zorder=3.5)

    ax.text(3.5, 520, f'4 h LFP ρ = {r_c:.2f}\n(CN {bg_c["provincial"]:.2f}, EU {bg_c["national"]:.2f}, JP {bg_c["community"]:.2f})',
            ha='left', va='top', fontsize=5.0, fontweight='bold', color=S.INK, linespacing=1.08)

    # Embedded short-window selectivity inset placed cleanly inside lower-right open space
    d_sw = d.assign(short_w=d.lfp_fill_windows - d.long_windows)
    ins = cv.ax(x + w - 18.8, top + h - 14.5, 17.4, 10.5, 'b_ins')
    ins.set_facecolor('white')
    vals_lfp, vals_ca = [], []
    for g in ('national', 'community'):
        sub = d_sw[d_sw.group == g]
        vals_lfp.append(float(stats.spearmanr(sub.short_w, sub.lfp_tail_discharge_mwh_per_mw).statistic))
        vals_ca.append(float(stats.spearmanr(sub.short_w, sub.hybrid_tail_discharge_mwh_per_mw).statistic))
    yy = np.array([0.0, 1.15])
    ins.barh(yy - 0.20, vals_lfp, height=0.30, color='#465862', zorder=3)
    ins.barh(yy + 0.20, vals_ca, height=0.30, color=A_CAES, alpha=0.80, zorder=3)
    for yi_s, y_pos in enumerate(yy):
        ins.text(vals_lfp[yi_s] + 0.05, y_pos - 0.20, f'{vals_lfp[yi_s]:.2f}', va='center', ha='left', fontsize=5.0, color='#465862')
        ins.text(vals_ca[yi_s] + 0.05, y_pos + 0.20, f'{vals_ca[yi_s]:.2f}', va='center', ha='left', fontsize=5.0, color=A_CAES)
    ins.set_xlim(0, 1.35); ins.set_xticks([])
    ins.set_ylim(1.75, -0.65); ins.set_yticks(yy); ins.set_yticklabels(['EU', 'JP'], fontsize=5.0)
    ins.tick_params(axis='y', length=0)
    for sp in ('bottom', 'top', 'right'):
        ins.spines[sp].set_visible(False)
    ins.text(0.0, -0.78, 'Short win. ρ', ha='left', va='bottom', fontsize=5.0, fontweight='bold', color=S.INK, clip_on=False)

    ax.set_xlim(*xlim); ax.set_xticks([0, 60, 120, 180])
    ax.set_ylim(*ylim); ax.set_yticks([0, 200, 400, 540])
    ax.set_xlabel('Windows filling 4 h LFP (yr⁻¹)')
    ax.set_ylabel(f'4 h LFP discharge ({S.DUNIT})')


def panel_c(cv, x, top, w, h, d, r_d, r_dB, bg_dB):
    """Panel c: Hexagonal mean-B surface in (long windows x additional delivery) space."""
    ax = cv.ax(x, top, w, h, 'c')
    S.grid(ax, 'y')
    xlim = (-1.5, 56.0)
    ymax = max(900.0, float(d.additional_delivery_mwh_per_mw.max()) * 1.06)
    ylim = (-20.0, ymax)
    norm_b = TwoSlopeNorm(vmin=-35.0, vcenter=0.0, vmax=65.0)

    jp = d[d.group == 'community']
    hb = ax.hexbin(
        jp.long_windows.to_numpy(float),
        jp.additional_delivery_mwh_per_mw.to_numpy(float),
        C=np.clip(jp.B.to_numpy(float), -35.0, 65.0),
        reduce_C_function=np.mean,
        gridsize=15,
        mincnt=4,
        extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
        cmap=B_CMAP,
        norm=norm_b,
        edgecolors='#536770',
        linewidths=0.18,
        alpha=0.90,
        zorder=2.5,
    )
    sc = hb
    for g in ('national', 'provincial'):
        z = d[d.group == g]
        sc = ax.scatter(
            z.long_windows, z.additional_delivery_mwh_per_mw,
            c=np.clip(z.B.to_numpy(float), -35.0, 65.0), cmap=B_CMAP, norm=norm_b,
            s=MS[g] + 3.5, marker=S.MARKER[g], edgecolor='#22353E', lw=0.32, zorder=4,
        )

    ax.text(0.0, 1.03, f'ρ delivery {r_d:.2f}, B {r_dB:.2f} (JP {bg_dB["community"]:.2f})',
            transform=ax.transAxes, ha='left', va='bottom', fontsize=5.0, fontweight='bold', color=S.INK)
    ax.text(0.03, 0.96, 'Hexbin: JP  |  ■ CN  ● EU\n(color: mean B)',
            transform=ax.transAxes, ha='left', va='top', fontsize=5.0, color='#3A4D56', linespacing=1.12, zorder=5)

    ax.set_xlim(*xlim); ax.set_xticks([0, 15, 30, 45, 55])
    ax.set_ylim(*ylim); ax.set_yticks([0, 300, 600, 900])
    ax.set_xlabel('Long charging windows (yr⁻¹)')
    ax.set_ylabel(f'Additional delivery ({S.DUNIT})')

    cbd = cv.ax(x + w + 1.3, top + 1.5, 2.2, h - 3.0, 'cb_c')
    cb = cv.fig.colorbar(sc, cax=cbd)
    cb.set_ticks([-30, 0, 30, 60])
    cb.ax.set_yticklabels(['−30', '0', '30', '60'], fontsize=5.0)
    cb.ax.tick_params(length=1.6, pad=1.2)
    cb.ax.axhline(0.0, color=S.INK, lw=0.65)
    cb.outline.set_linewidth(0.50); cb.outline.set_edgecolor(S.INK)
    cb.ax.set_title('B', fontsize=5.2, fontweight='bold', pad=2.0)


def panel_d(cv, x, top, w, h):
    """Panel d: Coupled 2D (E1, E2) saturation phase-space contour surface (left) + continuous fate stream (right)."""
    a = pd.read_csv(D.TAB / 'analytic_two_window_example.csv')

    w_ph = 34.5
    gap_d = 7.5
    w_st = w - w_ph - gap_d

    # 1. Left sub-panel: 2D (E1, E2) continuous two-window saturation phase surface
    ax_p = cv.ax(x, top, w_ph, h, 'd_phase')
    e_grid = np.linspace(0.0, 43.2, 180)
    E1, E2 = np.meshgrid(e_grid, e_grid)
    D_caes = np.minimum(E1, 21.6) + np.minimum(E2, 21.6)

    phase_cmap = LinearSegmentedColormap.from_list(
        'caes_phase_2w',
        [(0.00, '#F7EFE8'), (0.35, '#D7EBE9'), (0.72, '#4E9F9B'), (1.00, '#105059')],
    )
    ax_p.imshow(D_caes, origin='lower', extent=(0.0, 43.2, 0.0, 43.2),
                cmap=phase_cmap, vmin=0.0, vmax=43.2, aspect='auto', zorder=1)
    ax_p.contour(E1, E2, D_caes, levels=[21.6, 30.0, 38.0],
                 colors=['white'], linewidths=0.50, alpha=0.82, zorder=2)

    # Saturation thresholds at E1 = 21.6 and E2 = 21.6, plus 4 h LFP corner (3.6, 3.6)
    ax_p.axvline(21.6, color='white', lw=0.65, ls='--', zorder=2.5)
    ax_p.axhline(21.6, color='white', lw=0.65, ls='--', zorder=2.5)
    ax_p.plot([0, 3.6, 3.6], [3.6, 3.6, 0], color=S.INK, lw=0.65, ls=':', zorder=3)

    # Constant-budget diagonal E1 + E2 = 43.2 (f in [0, 1])
    ax_p.plot([0.0, 43.2], [43.2, 0.0], color='#C9543B', lw=1.35, zorder=4)
    ax_p.scatter([21.6], [21.6], s=22, marker='o', facecolor='white', edgecolor='#C9543B', lw=0.90, zorder=5)
    ax_p.scatter([0.0, 43.2], [43.2, 0.0], s=12, marker='o', color='#C9543B', edgecolor='white', lw=0.30, zorder=5)

    ax_p.text(0.5, 1.03, 'Diagonal: E₁ + E₂ = 43.2', transform=ax_p.transAxes,
              ha='center', va='bottom', fontsize=5.0, fontweight='bold', color='#C9543B')
    ax_p.text(32.8, 32.8, '43.2\nmax', ha='center', va='center', fontsize=5.0, fontweight='bold', color='white', linespacing=1.20, zorder=6)
    ax_p.text(4.8, 2.8, 'LFP 3.6', ha='left', va='center', fontsize=5.0, color=S.INK, zorder=6)

    ax_p.set_xlim(0, 43.2); ax_p.set_ylim(0, 43.2)
    ax_p.set_xticks([0, 21.6, 43.2]); ax_p.set_xticklabels(['0', '21.6', '43.2'], fontsize=5.0)
    ax_p.set_yticks([0, 21.6, 43.2]); ax_p.set_yticklabels(['0', '21.6', '43.2'], fontsize=5.0)
    for sp in ('top', 'right'):
        ax_p.spines[sp].set_visible(True); ax_p.spines[sp].set_color('#8FA0A8'); ax_p.spines[sp].set_linewidth(0.45)
    ax_p.set_xlabel('Window 1 energy (MWh MW⁻¹)')
    ax_p.set_ylabel('Window 2 energy\n(MWh MW⁻¹)')

    # 2. Right sub-panel: Continuous stacked energy-fate stream along diagonal f in [0, 1]
    lgd = cv.ax(x + w_ph + gap_d - 2.0, top - 4.3, w_st + 2.0, 3.2, 'leg_d')
    lgd.set_xlim(0, w_st + 2.0); lgd.set_ylim(0, 1); lgd.axis('off')
    lgd.add_patch(Rectangle((0.0, 0.18), 2.4, 0.64, facecolor=A_CAES, lw=0))
    lgd.text(3.0, 0.5, 'Win. 1', va='center', fontsize=5.0)
    lgd.add_patch(Rectangle((12.5, 0.18), 2.4, 0.64, facecolor=WIN2_C, lw=0))
    lgd.text(15.5, 0.5, 'Win. 2', va='center', fontsize=5.0)
    lgd.add_patch(Rectangle((25.0, 0.18), 2.4, 0.64, facecolor='#F6DEC9', edgecolor='#C9543B', lw=0.45))
    lgd.text(28.0, 0.5, 'Spill', va='center', fontsize=5.0, color='#9E432E')

    ax_s = cv.ax(x + w_ph + gap_d, top, w_st, h, 'd_stream')
    fs = np.linspace(0.0, 1.0, 201)
    w1_s = np.minimum(21.6, 43.2 * fs)
    w2_s = np.minimum(21.6, 43.2 * (1.0 - fs))
    tot_c = w1_s + w2_s
    lfp_c = np.minimum(3.6, 43.2 * (0.90 / 0.55) * (1.0 / 1.5) * fs)  # exact zl curve from CSV below
    zl = a[a.technology == 'LFP'].sort_values('first_window_share')

    ax_s.fill_between(fs, 0.0, w1_s, facecolor=A_CAES, alpha=0.90, lw=0, zorder=2)
    ax_s.fill_between(fs, w1_s, tot_c, facecolor=WIN2_C, alpha=0.85, lw=0, zorder=2.2)
    ax_s.fill_between(fs, tot_c, 43.2, facecolor='#F7E3D2', alpha=0.88, lw=0, zorder=1.8)
    ax_s.plot(fs, w1_s, color='white', lw=0.55, zorder=2.5)
    ax_s.plot(fs, tot_c, color='#0E424A', lw=1.35, zorder=4)
    ax_s.plot([0, 1], [43.2, 43.2], color='#C9543B', lw=0.70, ls='--', zorder=3)
    ax_s.plot(zl.first_window_share, zl.delivery, color='white', lw=1.9, zorder=4.2)
    ax_s.plot(zl.first_window_share, zl.delivery, color='#2B3A42', lw=1.15, ls=(0, (2.4, 1.2)), zorder=4.5)
    ax_s.scatter([0.5], [43.2], s=18, facecolor='white', edgecolor='#C9543B', lw=0.85, zorder=5)

    ax_s.text(0.50, 27.5, 'A-CAES\n43.2 at f = 0.5', ha='center', va='center',
              fontsize=5.0, fontweight='bold', color='white', linespacing=1.05, zorder=6)
    ax_s.text(0.50, 9.8, '4 h LFP: 7.2', ha='center', va='bottom',
              fontsize=5.0, fontweight='bold', color='white', zorder=6)

    ax_s.set_xlim(0.0, 1.0); ax_s.set_ylim(0.0, 46.5)
    ax_s.set_xticks([0, 0.5, 1.0]); ax_s.set_xticklabels(['0', '0.5', '1'], fontsize=5.0)
    ax_s.set_yticks([0, 21.6, 43.2]); ax_s.set_yticklabels(['0', '21.6', '43.2'], fontsize=5.0)
    ax_s.set_xlabel('First-window share (f)')
    return a


def panel_e(cv, x, top, w, h):
    """Panel e: Continuous 1-to-12-day event-extension delta-B decay trajectories."""
    ext = D.extension()
    pos_uids = D.cn_positive_uids()
    b0 = ext[ext.target == 0].set_index('uid')['B']
    ext = ext.assign(dB=ext.B - ext.uid.map(b0))

    ax = cv.ax(x, top, w, h, 'e')
    S.grid(ax, 'y')
    ax.axhline(0.0, color=S.INK, lw=0.65, zorder=2)

    targets = list(range(0, 13))
    rows_out = []
    box_e = (-0.3, 12.4, -31.5, 45.5)
    for g, sub_fn, ls_style in [
        ('provincial', lambda df: df[(df.group == 'provincial') & (df.uid.isin(pos_uids))], '-'),
        ('national', lambda df: df[df.group == 'national'], '-'),
        ('community', lambda df: df[df.group == 'community'], '-'),
    ]:
        sub = sub_fn(ext)
        c = S.COLOR[g]
        q25, med, q75 = [], [], []
        for t in targets:
            vt = sub.loc[sub.target == t, 'dB'].to_numpy(float)
            l, m, u = np.quantile(vt, [0.25, 0.50, 0.75])
            q25.append(float(l)); med.append(float(m)); q75.append(float(u))
            rows_out.append(dict(group=f'{g}_pos' if g == 'provincial' else g, target_days=t, q25_dB=l, median_dB=m, q75_dB=u))
        ring = S.clip_ring(list(zip(np.r_[targets, targets[::-1]], np.r_[q25, q75[::-1]])), box_e)
        if len(ring) >= 3:
            ax.fill(ring[:, 0], ring[:, 1], color=c, alpha=0.18, lw=0, zorder=2)
        for seg in S.clip_lines([np.c_[targets, med]], box_e):
            ax.plot(seg[:, 0], seg[:, 1], color=c, lw=1.35, ls=ls_style, zorder=4)
        t_arr, m_arr = np.asarray(targets, float), np.asarray(med, float)
        in_y = (m_arr >= box_e[2]) & (m_arr <= box_e[3])
        ax.plot(t_arr[in_y], m_arr[in_y], ls='none', marker=S.MARKER[g], ms=2.8,
                color=c, mec='white', mew=0.25, zorder=4.2)

    ax.text(0.35, -21.5, '12 d median ΔB:\nCN pos −12.2, EU −22.5, JP −24.9\n(99.6% of pos-supply records fall)',
            ha='left', va='center', fontsize=5.0, color=S.INK, linespacing=1.12)

    ax.set_xlim(-0.3, 12.4)
    ax.set_xticks([0, 2, 4, 6, 8, 10, 12])
    ax.set_xticklabels(['Real', '2', '4', '6', '8', '10', '12'])
    ax.set_ylim(-31.5, 45.5)
    ax.set_yticks([-30, -15, 0, 15, 30, 45])
    ax.set_xlabel('Extended event duration (d)')
    ax.set_ylabel(f'ΔB vs real dates ({S.UNIT})')
    return pd.DataFrame(rows_out)


def _clipped_china_factorial():
    cf = pd.read_csv(D.CN_SUITE_CLIP / 'china_factorial.csv.gz', dtype={'uid': str})
    w = cf.pivot_table(index=['uid', 'energy_multiplier', 'charge_power_multiplier', 'access'],
                       columns='order', values='B').reset_index()
    wide = w.pivot_table(index='uid', columns=['energy_multiplier', 'charge_power_multiplier', 'access'], values='real')
    col = lambda e, p, a: wide[(e, p, a)]
    return pd.DataFrame({
        'uid': wide.index,
        'surplus_energy_4x': sum(col(4, p, a) - col(1, p, a) for p in (1, 4) for a in ('observed', 'expanded')) / 4,
        'charge_power_4x': sum(col(e, 4, a) - col(e, 1, a) for e in (1, 4) for a in ('observed', 'expanded')) / 4,
        'expanded_access': sum(col(e, p, 'expanded') - col(e, p, 'observed') for e in (1, 4) for p in (1, 4)) / 4,
    }).reset_index(drop=True)


def panel_f(cv, x, top, w, h):
    """Panel f: Multi-regime 2x2x2 factorial quantile interval forest (zero lollipops, zero broken empty rectangles)."""
    fx = D.factorial()
    fx_clip = _clipped_china_factorial()
    pos_uids = D.cn_positive_uids()

    cohorts = [
        ('China (prop., pos)', fx[(fx.group == 'provincial') & (fx.uid.isin(pos_uids))], S.COLOR['provincial'], 's'),
        ('China (clipped, pos)', fx_clip[fx_clip.uid.isin(pos_uids)], '#805712', 'D'),
        ('Europe', fx[fx.group == 'national'], S.COLOR['national'], 'o'),
        ('Japan', fx[fx.group == 'community'], S.COLOR['community'], '^'),
    ]
    effects = [
        ('surplus_energy_4x', 'Surplus × 4', -0.22, 'white', 0.85),
        ('charge_power_4x', 'Power × 4', 0.00, '#DCE6E9', 1.25),
        ('expanded_access', 'All task-free h', 0.22, S.WIN, 2.2),
    ]

    lgf = cv.ax(x, top - 4.4, w, 3.2, 'leg_f')
    lgf.set_xlim(0, w); lgf.set_ylim(0, 1); lgf.axis('off')
    lgf.scatter([1.4], [0.5], s=14, marker='o', facecolor='white', edgecolor=S.INK, lw=0.70)
    lgf.text(3.4, 0.5, 'Surplus × 4', va='center', fontsize=5.0)
    lgf.scatter([24.5], [0.5], s=14, marker='o', facecolor='#DCE6E9', edgecolor=S.INK, lw=0.70)
    lgf.text(26.5, 0.5, 'Charge power × 4', va='center', fontsize=5.0)
    lgf.scatter([52.0], [0.5], s=16, marker='o', facecolor=S.WIN, edgecolor='white', lw=0.35)
    lgf.text(54.2, 0.5, 'All task-free hours (IQR & 10th–90th)', va='center', fontsize=5.0, fontweight='bold', color=S.WIN)

    ax = cv.ax(x, top, w, h, 'f')
    S.grid(ax, 'x')
    ax.axvspan(0, 14.5, facecolor=S.WIN_PALE, zorder=0)
    ax.axvline(0.0, color=S.INK, lw=0.65, zorder=2)

    for ci, (c_lab, sub, col, mk) in enumerate(cohorts):
        if ci % 2 == 1:
            ax.axhspan(ci - 0.45, ci + 0.45, facecolor='#F6F9FA', zorder=0.2)
        for ef_col, ef_name, dy, m_fc, bar_lw in effects:
            vals = sub[ef_col].to_numpy(float)
            q10, q25, q50, q75, q90 = np.quantile(vals, [0.10, 0.25, 0.50, 0.75, 0.90])
            yy = ci + dy
            ax.plot([max(-2.0, q10), min(14.2, q90)], [yy, yy], color=col, lw=0.75, alpha=0.75, zorder=3)
            ax.plot([max(-2.0, q25), min(14.2, q75)], [yy, yy], color=S.WIN if ef_col == 'expanded_access' else col,
                    lw=2.4 if ef_col == 'expanded_access' else 1.5, solid_capstyle='butt', zorder=3.8)
            fc_use = col if ef_col == 'expanded_access' else m_fc
            ec_use = 'white' if ef_col == 'expanded_access' else col
            ax.scatter([q50], [yy], s=15, marker=mk, facecolor=fc_use, edgecolor=ec_use, lw=0.55, zorder=5)

        # Right-side key median callouts (Surplus x4 | All hours)
        ms = float(np.median(sub['surplus_energy_4x']))
        ma = float(np.median(sub['expanded_access']))
        ax.text(14.65, ci - 0.14, f'E×4: {ms:.2f}', ha='left', va='center', fontsize=5.0, color=S.MUTED, clip_on=False)
        ax.text(14.65, ci + 0.16, f'All h: {ma:.2f}', ha='left', va='center', fontsize=5.0, fontweight='bold', color=col, clip_on=False)

    ax.set_xlim(-2.2, 14.4)
    ax.set_xticks([0, 3, 6, 9, 12])
    ax.set_ylim(3.55, -0.55)
    ax.set_yticks(range(4))
    ax.set_yticklabels([c[0] for c in cohorts], fontsize=5.05, fontweight='bold')
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel(f'Factorial main-effect ΔB ({S.UNIT})')
    return fx


def panel_g(cv, x, top, w, h):
    """Panel g: Symmetric 8-sector 3-petal Nightingale polar rose diagram (zero numeric heatmap!)."""
    from matplotlib.patches import Polygon as MPolygon

    fc = D.factorial_cells()
    fc_idx = fc.set_index(['group', 'energy_multiplier', 'charge_power_multiplier', 'access'])

    # Right hemisphere (sectors 0..3, clockwise from top-right): All task-free hours
    # Left hemisphere (sectors 4..7, clockwise from bottom-left): Observed hours
    sectors = [
        (1, 1, 'expanded', '1×E, 1×P'),
        (4, 1, 'expanded', '4×E, 1×P'),
        (1, 4, 'expanded', '1×E, 4×P'),
        (4, 4, 'expanded', '4×E, 4×P'),
        (4, 4, 'observed', '4×E, 4×P'),
        (1, 4, 'observed', '1×E, 4×P'),
        (4, 1, 'observed', '4×E, 1×P'),
        (1, 1, 'observed', '1×E, 1×P'),
    ]

    # Cartesian square axis rendering an exact vector polar Nightingale Coxcomb rose (full control over clipping & labels!)
    ax = cv.ax(x, top, w, h, 'g')
    ax.set_xlim(-1.38, 1.38)
    ax.set_ylim(-1.22 * (h / w), 1.30 * (h / w))
    ax.set_aspect('equal')
    ax.axis('off')

    r_in = 0.24
    r_span = 0.76  # r = 0.24 (0%) to 1.00 (100%)

    # Hemisphere background arcs (Right: All task-free hours in pale emerald; Left: Observed hours in soft slate)
    th_right = np.linspace(np.pi / 2, -np.pi / 2, 120)
    th_left = np.linspace(np.pi / 2, 3 * np.pi / 2, 120)
    for th_arc, fc_bg in ((th_right, S.WIN_PALE), (th_left, '#F3F6F8')):
        pts_o = np.c_[1.02 * np.cos(th_arc), 1.02 * np.sin(th_arc)]
        pts_i = np.c_[r_in * np.cos(th_arc[::-1]), r_in * np.sin(th_arc[::-1])]
        ax.add_patch(MPolygon(np.vstack([pts_o, pts_i]), closed=True, facecolor=fc_bg, edgecolor='none', zorder=0.5))

    # Concentric percentage rings (25%, 50%, 75%, 100%)
    th_full = np.linspace(0, 2 * np.pi, 180)
    for pct_r, lw_r, ls_r in ((25, 0.38, ':'), (50, 0.45, '--'), (75, 0.38, ':'), (100, 0.55, '-')):
        rr = r_in + r_span * (pct_r / 100.0)
        ax.plot(rr * np.cos(th_full), rr * np.sin(th_full),
                color='#B9C7CD' if pct_r < 100 else '#8DA0A8', lw=lw_r, ls=ls_r, zorder=1)
    ax.plot(r_in * np.cos(th_full), r_in * np.sin(th_full), color=S.INK, lw=0.55, zorder=3)

    # Vertical hemisphere divider split cleanly around central hub
    ax.plot([0, 0], [r_in, 1.06], color=S.INK, lw=0.75, zorder=3.5)
    ax.plot([0, 0], [-1.06, -r_in], color=S.INK, lw=0.75, zorder=3.5)
    for si in range(8):
        ang_edge = np.pi / 2 - si * (np.pi / 4)
        ax.plot([r_in * np.cos(ang_edge), 1.02 * np.cos(ang_edge)],
                [r_in * np.sin(ang_edge), 1.02 * np.sin(ang_edge)],
                color='#C8D4D9', lw=0.40, zorder=1.5)

    # Draw 3 radial petals (China, Europe, Japan) inside each 45-degree sector
    dtheta = np.radians(10.8)
    offsets_th = [np.radians(11.5), 0.0, -np.radians(11.5)]

    for si, (em, pm, acc, s_lab) in enumerate(sectors):
        ang_mid = np.pi / 2 - (si + 0.5) * (np.pi / 4)
        for gi, g in enumerate(S.MAIN):
            row = fc_idx.loc[(g, em, pm, acc)]
            sh = float(row['hybrid_cheaper']) / float(row['n'])
            r_out = r_in + r_span * max(sh, 0.025)
            th_a = ang_mid + offsets_th[gi] + dtheta / 2
            th_b = ang_mid + offsets_th[gi] - dtheta / 2
            arc = np.linspace(th_a, th_b, 18)
            poly = np.vstack([
                np.c_[r_out * np.cos(arc), r_out * np.sin(arc)],
                np.c_[r_in * np.cos(arc[::-1]), r_in * np.sin(arc[::-1])],
            ])
            ax.add_patch(MPolygon(poly, closed=True, facecolor=S.COLOR[g], edgecolor='white',
                                  lw=0.28, alpha=0.90, zorder=2.8))

        # Spoke label outside the 100% ring
        rl = 1.18
        lx, ly = rl * np.cos(ang_mid), rl * np.sin(ang_mid)
        ha = 'left' if np.cos(ang_mid) > 0.15 else ('right' if np.cos(ang_mid) < -0.15 else 'center')
        ax.text(lx, ly, s_lab, ha=ha, va='center', fontsize=5.0,
                fontweight='bold' if (em == 1 and pm == 1) else 'normal', color=S.INK)

    # Center hub percentage key + clean top header (zero dangling bottom text, zero stroke overlap!)
    ax.text(0.0, 0.0, '0%', ha='center', va='center', fontsize=5.0, fontweight='bold', color=S.MUTED)
    ax.text(-0.74, 1.24, 'Observed h', ha='center', va='bottom', fontsize=5.1, fontweight='bold', color=S.MUTED)
    ax.text(0.0, 1.24, 'Rings: 25–100%', ha='center', va='bottom', fontsize=5.0, color=S.MUTED)
    ax.text(0.74, 1.24, 'All task-free h', ha='center', va='bottom', fontsize=5.1, fontweight='bold', color=S.WIN)
    return fc


def main():
    S.setup()
    cv = S.Canvas(H)

    rec = D.records()
    rec = rec[rec.group.isin(S.MAIN)].copy()
    rec['group'] = rec.group.astype(str)
    d = rec.merge(D.windows()[['uid', 'windows', 'lfp_fill_windows', 'long_windows']], on='uid', validate='1:1')

    r_a, _, bg_a = D.equal_regime_spearman(d, 'long_windows', 'hybrid_tail_discharge_mwh_per_mw')
    r_b, _, bg_b = D.equal_regime_spearman(d, 'long_windows', 'lfp_tail_discharge_mwh_per_mw')
    r_c, _, bg_c = D.equal_regime_spearman(d, 'lfp_fill_windows', 'lfp_tail_discharge_mwh_per_mw')
    r_d, _, bg_d = D.equal_regime_spearman(d, 'long_windows', 'additional_delivery_mwh_per_mw')
    r_dB, _, bg_dB = D.equal_regime_spearman(d, 'long_windows', 'B')
    assert [round(v, 2) for v in (r_a, r_b, r_c, r_d, r_dB)] == [0.77, 0.61, 0.97, 0.86, 0.74]

    S.regime_legend_strip(
        cv, 1.5, 1.3, RIGHT - 1.5, 'leg_top',
        scope_label='Shared regime key (Panels A–C, E–G)' if S.JOULE else 'Shared regime key (panels a–c, e–g)',
        framed=True,
    )

    # Row 1: a, b, c (uniform gutters and exact right alignment at RIGHT = 180.0 mm)
    top1, h1 = 13.8, 36.2
    cv.letter(1.5, top1 - 4.5, 'a')
    panel_a(cv, LEFT, top1, 41.5, h1, d, r_a, bg_a, r_b, bg_b)

    cv.letter(63.5, top1 - 4.5, 'b')
    panel_b(cv, 74.5, top1, 41.5, h1, d, r_c, bg_c)

    cv.letter(120.5, top1 - 4.5, 'c')
    panel_c(cv, 131.5, top1, 40.9, h1, d, r_d, r_dB, bg_dB)

    # Row 2: d (2D Saturation Phase + Stream) + e (1-12 d event extension trajectories)
    top2, h2 = 66.0, 39.5
    cv.letter(1.5, top2 - 4.5, 'd')
    a_two = panel_d(cv, LEFT, top2, 78.0, h2)

    cv.letter(98.5, top2 - 4.5, 'e')
    ext_df = panel_e(cv, 110.0, top2, RIGHT - 110.0, h2)

    # Row 3: f (Multi-regime factorial interval forest) + g (8-sector polar rose diagram locked to RIGHT = 180.0 mm)
    top3, h3 = 123.5, 46.0
    cv.letter(1.5, top3 - 4.5, 'f')
    fx_df = panel_f(cv, 28.5, top3, 76.0, h3)

    cv.letter(122.0, top3 - 4.5, 'g')
    fc_df = panel_g(cv, 126.4, top3, 51.5, h3)

    # Source data exports
    S.SD.mkdir(parents=True, exist_ok=True)
    d[['uid', 'group', 'windows', 'lfp_fill_windows', 'long_windows', 'hybrid_tail_discharge_mwh_per_mw',
       'lfp_tail_discharge_mwh_per_mw', 'additional_delivery_mwh_per_mw', 'B']].to_csv(
        S.SD / 'figure4abcd_windows_and_discharge.csv', index=False)
    a_two.to_csv(S.SD / 'figure4d_two_window_example.csv', index=False)
    a_two.to_csv(S.SD / 'figure4e_two_window_example.csv', index=False)
    a_two.to_csv(S.SD / 'figure4d_two_window_example.csv', index=False)
    ext_df.to_csv(S.SD / 'figure4e_event_extension_curves.csv', index=False)
    fx_df.to_csv(S.SD / 'figure4f_factorial.csv', index=False)
    fc_df.to_csv(S.SD / 'figure4g_factorial_cells.csv', index=False)

    S.save(
        cv,
        'main_figure4',
        rows=[['a', 'b', 'c'], ['d_phase', 'd_stream', 'e']],
        cols=[],
        exempt=[
            dict(panels=['leg_top', 'leg_d', 'leg_f', 'cb_c', 'b_ins', 'g'], checks=['all'],
                 reason='Legend strips, colorbar, selectivity inset and bespoke radial rose panel g'),
            dict(panels=['a', 'b', 'c', 'd_phase', 'd_stream', 'e', 'f'], checks=['panel-width'],
                 reason='Intentional multi-panel widths across 7 panels'),
            dict(panels=['d_stream'], checks=['horizontal-gutter'],
                 reason='Coupled sub-panel inside two-window mechanism panel d'),
        ],
        audit=dict(
            panels=['a', 'b', 'c', 'd', 'e', 'f', 'g'],
            panel_count=7,
            records=11054,
            spearman_equal_regime=dict(acaes_long=r_a, lfp_long=r_b, lfp_fill=r_c, additional_long=r_d, B_long=r_dB),
        ),
    )


if __name__ == '__main__':
    main()
