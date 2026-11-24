"""Supplementary Figure S3 (appendix_figure_event_extension).
Central 1-to-12-day event extension across 11,054 records (Chinese provinces, European countries, Japanese communities; 2035 costs, 3.6 %).

Upgraded layout & visual diversity (eliminating the empty bottom-right hole with wide balanced panels d and e):
a  2D HDR contour + stratified quantile response of A-CAES equivalent full cycles to mean event duration
b  Multi-quantile fan ribbons (25th-75th) of A-CAES cycling across 1-12 d extensions (including China positive-supply trajectory)
c  Multi-quantile fan ribbons of system-cost saving B across 1-12 d extensions with B >= 0 emerald zone
d  Shaded 1:1 dominance half-plane (99.4% below diagonal) + HDR contours of % cycling change (4 h LFP vs 24 h A-CAES)
e  Quadrant-shaded hexagonal/HDR joint response of delta-B to delta additional delivery (equal-regime rho = 0.98)
"""
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon, Rectangle
from scipy.stats import gaussian_kde

import style as S
import data as D

H = 122.0
MS = {'provincial': 7.5, 'national': 8.0, 'community': 9.0}


def multi_scatter(ax, d, x, y, xlim, ylim):
    jp = d[d.group == 'community']
    jx, jy = jp[x].to_numpy(float), jp[y].to_numpy(float)
    rng = np.random.default_rng(2026)
    idx_s = rng.choice(len(jx), size=min(len(jx), 3500), replace=False)
    sx, sy = xlim[1] - xlim[0], ylim[1] - ylim[0]
    kde2 = gaussian_kde(np.vstack([(np.clip(jx[idx_s], *xlim) - xlim[0]) / sx,
                                   (np.clip(jy[idx_s], *ylim) - ylim[0]) / sy]), bw_method=0.22)
    gx, gy = np.linspace(0.01, 0.99, 65), np.linspace(0.01, 0.99, 65)
    GX, GY = np.meshgrid(gx, gy)
    Z = kde2(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
    vals = np.sort(Z.ravel())[::-1]
    csum = np.cumsum(vals)
    t88 = vals[min(len(vals) - 1, np.searchsorted(csum, 0.88 * csum[-1]))]
    t50 = vals[min(len(vals) - 1, np.searchsorted(csum, 0.50 * csum[-1]))]
    c_jp = S.COLOR['community']
    Xp, Yp = xlim[0] + GX * sx, ylim[0] + GY * sy
    ax.contourf(Xp, Yp, Z, levels=[t88, Z.max() * 1.05], colors=[c_jp], alpha=0.18, zorder=2.0)
    ax.contourf(Xp, Yp, Z, levels=[t50, Z.max() * 1.05], colors=[c_jp], alpha=0.25, zorder=2.1)
    ax.contour(Xp, Yp, Z, levels=[t88, t50], colors=[c_jp], linewidths=[0.48, 0.70], alpha=0.68, zorder=2.4)

    for g in ('provincial', 'national'):
        z = d[d.group == g]
        ax.scatter(z[x], z[y], s=MS[g], marker=S.MARKER[g], color=S.COLOR[g],
                   edgecolor='white', lw=0.2, alpha=0.80, zorder=3)

    edges_j = np.unique(np.quantile(jx, np.linspace(0.03, 0.97, 8)))
    xm_j, ym_j = [], []
    for lo_e, hi_e in zip(edges_j[:-1], edges_j[1:]):
        mk = (jx >= lo_e) & (jx <= hi_e)
        if mk.sum() >= 10:
            xm_j.append(float(np.median(jx[mk])))
            ym_j.append(float(np.median(jy[mk])))
    if len(xm_j) >= 2:
        ax.plot(xm_j, ym_j, color=c_jp, lw=1.25, alpha=0.95, zorder=3.6)
    ax.scatter([float(np.median(jx))], [float(np.median(jy))], s=17, marker='^',
               color=c_jp, edgecolor='white', lw=0.45, zorder=4.2)


def band(ax, x, col, pos_uids):
    for g in S.MAIN:
        z = x[(x.group == g) & (x.target >= 1)].groupby('target')[col]
        m, q1, q3 = z.median(), z.quantile(0.25), z.quantile(0.75)
        ax.fill_between(m.index, q1, q3, color=S.COLOR[g], alpha=0.18, lw=0, zorder=2)
        ax.plot(m.index, m, color=S.COLOR[g], lw=1.2, zorder=3.5)
        ax.scatter(m.index, m, s=MS[g] * 0.85, marker=S.MARKER[g], color=S.COLOR[g], edgecolor='white', lw=0.2, zorder=4)
    # Also draw China positive-supply dashed trajectory so China cycling/B decay is clearly resolved
    zp = x[(x.uid.isin(pos_uids)) & (x.target >= 1)].groupby('target')[col].median()
    ax.plot(zp.index, zp, color=S.DEEP['provincial'], lw=1.0, ls='--', zorder=3.8)
    ax.set_xlim(0.5, 12.5)
    ax.set_xticks([1, 3, 6, 9, 12])
    ax.set_xlabel('Extended event duration (d)')


def main():
    S.setup()
    cv = S.Canvas(H)
    rec = D.records()
    rec = rec[rec.group.isin(S.MAIN)].copy()
    rec['group'] = rec.group.astype(str)
    pos_uids = D.cn_positive_uids()
    rho_a, _, _ = D.equal_regime_spearman(rec, 'mean_dur', 'efc_acaes')
    assert round(rho_a, 2) == -0.85

    x = D.extension()
    c = D.central12().copy()
    rho_e, _, _ = D.equal_regime_spearman(c, 'delta_additional_delivery_mwh_per_mw', 'delta_B')

    x12 = x[x.target == 12].set_index('uid')
    x0 = x[x.target == 0].set_index('uid').reindex(x12.index)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = pd.DataFrame({
            'group': x0.group,
            'acaes': np.where(x0.efc_acaes > 1e-6, 100 * (x12.efc_acaes / x0.efc_acaes - 1), np.nan),
            'lfp': np.where(x0.efc_lfp > 1e-6, 100 * (x12.efc_lfp / x0.efc_lfp - 1), np.nan),
        })
    rel_fin = rel.dropna().copy()
    n_below = int((rel_fin.acaes < rel_fin.lfp).sum())
    pct_below = 100.0 * n_below / len(rel_fin)

    def _cn_pos_leg(ax_l, xx):
        ax_l.plot([xx, xx + 4.5], [0.5, 0.5], color=S.DEEP['provincial'], lw=1.0, ls='--')
        ax_l.text(xx + 5.5, 0.5, f"CN pos. supply ({'B, C' if S.JOULE else 'b, c'})", va='center', fontsize=5.1, color=S.DEEP['provincial'])
        return xx + 34.0

    S.regime_legend_strip(cv, 1.5, 1.6, 178.5, 'leg', extra_items=(_cn_pos_leg,), framed=True)

    X = [19.0, 77.5, 136.0]
    W, Hh = 44.0, 36.2
    top1, top2 = 13.8, 68.5

    # Row 1: a, b, c
    cv.letter(1.5, top1 - 3.8, 'a')
    ax = cv.ax(X[0], top1, W, Hh, 'a')
    xlim_a, ylim_a = (1.1, 5.0), (0.0, 62.0)
    S.grid(ax, 'y')
    multi_scatter(ax, rec, 'mean_dur', 'efc_acaes', xlim_a, ylim_a)
    ax.set_xlim(*xlim_a); ax.set_ylim(*ylim_a)
    ax.set_xlabel('Mean event duration (d)')
    ax.set_ylabel('A-CAES cycling (yr⁻¹)')
    ax.text(0.97, 0.95, f'Equal-regime ρ = {rho_a:.2f}', transform=ax.transAxes,
            va='top', ha='right', fontsize=5.1, fontweight='bold')

    cv.letter(X[1] - 12.5, top1 - 3.8, 'b')
    ax = cv.ax(X[1], top1, W, Hh, 'b')
    band(ax, x, 'efc_acaes', pos_uids)
    ax.set_ylim(0, 90); ax.set_yticks([0, 30, 60, 90])
    ax.set_ylabel('A-CAES cycling (yr⁻¹)')
    S.grid(ax, 'y')

    cv.letter(X[2] - 12.5, top1 - 3.8, 'c')
    ax = cv.ax(X[2], top1, W, Hh, 'c')
    ax.axhspan(0, 60, facecolor=S.WIN_PALE, zorder=0)
    band(ax, x, 'B', pos_uids)
    ax.axhline(0, color=S.INK, lw=0.65, zorder=2.5)
    ax.set_ylim(-40, 60); ax.set_yticks([-40, -20, 0, 20, 40, 60])
    ax.set_ylabel(f'B ({S.UNIT})')
    S.grid(ax, 'y')

    # Row 2: Balanced wide panels d and e filling the full width (19.0 .. 180.0 mm)
    X2 = [19.0, 108.0]
    W2 = 72.0
    cv.letter(1.5, top2 - 3.8, 'd')
    ax = cv.ax(X2[0], top2, W2, Hh, 'd')
    xlim_d = (-102.0, 18.0)
    poly_dom = S.clip_ring([(-102, -102), (18, -102), (18, 18)], (*xlim_d, *xlim_d))
    if len(poly_dom) >= 3:
        ax.add_patch(Polygon(poly_dom, closed=True, facecolor='#FCF3EE', edgecolor='none', zorder=0))
    ax.plot([-102, 18], [-102, 18], color=S.INK, lw=0.65, ls=(0, (3, 2)), zorder=2)
    rel_plot = rel_fin.assign(lfp=rel_fin.lfp.clip(*xlim_d), acaes=rel_fin.acaes.clip(*xlim_d))
    multi_scatter(ax, rel_plot, 'lfp', 'acaes', xlim_d, xlim_d)
    ax.set_xlim(*xlim_d); ax.set_ylim(*xlim_d)
    ax.set_xticks([-80, -40, 0]); ax.set_yticks([-80, -40, 0])
    ax.set_xlabel('Change in 4 h LFP cycling (%)')
    ax.set_ylabel('Change in A-CAES cycling (%)')
    ax.text(0.97, 0.06, f'12-day extension: {n_below:,}/{len(rel_fin):,} ({pct_below:.1f}%) below 1:1',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=5.05, fontweight='bold', color=S.DEEP['national'])

    cv.letter(X2[1] - 12.5, top2 - 3.8, 'e')
    ax = cv.ax(X2[1], top2, W2, Hh, 'e')
    xlim_e, ylim_e = (-1000.0, 45.0), (-60.0, 25.0)
    ax.add_patch(Rectangle((xlim_e[0], ylim_e[0]), -xlim_e[0], -ylim_e[0], facecolor='#F5F8FA', edgecolor='none', zorder=0))
    multi_scatter(ax, c, 'delta_additional_delivery_mwh_per_mw', 'delta_B', xlim_e, ylim_e)
    ax.set_xlim(*xlim_e); ax.set_ylim(*ylim_e)
    ax.axhline(0, color=S.INK, lw=0.6); ax.axvline(0, color=S.INK, lw=0.6)
    ax.set_xlabel(f'Change in additional delivery ({S.DUNIT})')
    ax.set_ylabel(f'Change in B ({S.UNIT})')
    ax.text(0.04, 0.92, f'12-day extension\nEqual-regime ρ = {rho_e:.2f}',
            transform=ax.transAxes, va='top', fontsize=5.15, fontweight='bold')

    rec[['uid', 'group', 'mean_dur', 'efc_acaes', 'hybrid_tail_discharge_mwh_per_mw', 'additional_delivery_mwh_per_mw', 'B']].to_csv(
        S.SD / 'appendix_event_extension_records.csv', index=False)
    x[['uid', 'group', 'target', 'mean_dur', 'efc_acaes', 'efc_lfp', 'B', 'additional_delivery_mwh_per_mw']].to_csv(
        S.SD / 'appendix_event_extension_curves.csv', index=False)
    c.join(rel[['acaes', 'lfp']].rename(columns={'acaes': 'pct_change_efc_acaes', 'lfp': 'pct_change_efc_lfp'}), on='uid')[
        ['uid', 'group', 'pct_change_efc_acaes', 'pct_change_efc_lfp', 'delta_additional_delivery_mwh_per_mw', 'delta_B']].to_csv(
        S.SD / 'appendix_event_extension_12day.csv', index=False)
    S.save(cv, 'appendix_figure_event_extension', rows=[['a', 'b', 'c'], ['d', 'e']],
           exempt=[dict(panels=['leg'], checks=['all'], reason='Legend strip'),
                   dict(panels=['d', 'e'], checks=['panel-width'], reason='Two-column bottom row spanning full figure width')],
           audit=dict(records=11054, finite_efc_records=int(len(rel_fin)),
                      spearman_mean_duration_vs_efc=rho_a, spearman_delta_delivery_vs_delta_B=rho_e, acaes_loses_more=n_below))


if __name__ == '__main__':
    main()
