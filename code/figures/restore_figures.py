"""Diversified Nature Energy extended-data and supplementary figures for v86 (ED Figs 1-6 + Supp Figs S5-S13).

Zero lollipop charts, zero descriptive panel titles inside axes, zero legacy internal codes (NVSB/PLRP/CLDB),
zero middle dots / leading '+' / '->' arrows. Strictly 200 bar surface tank H2 configuration throughout.
Includes European 33-country and 17-country cartographic choropleth maps in ED Figs 4 and 6,
and native Python generation of Supp Fig. S10 from archived reliability tables.
"""
from pathlib import Path
import os, sys, json, shutil, hashlib
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle, Polygon
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, Normalize
from matplotlib.ticker import FuncFormatter
from scipy.stats import spearmanr, gaussian_kde
from PIL import Image
import fitz

W = Path(__file__).resolve().parents[1]
V = W.parents[1]
V85 = V.parent / 'Nature_Energy_v85'
BANANA_ED1_V5 = Path('/home/cxw/.gemini/antigravity-cli/brain/b8af1f2a-d63f-462f-8534-034fb9384fa8/extended_data_figure1_banana_v5_11054.png')
BANANA_ED1 = BANANA_ED1_V5 if BANANA_ED1_V5.exists() else Path('/home/cxw/.gemini/antigravity-cli/brain/b8af1f2a-d63f-462f-8534-034fb9384fa8/extended_data_figure1_banana_v4_10612.png')
sys.path.insert(0, str(V / 'figures/code'))
import style as S
import data as D
import geo_helper as GH

S.OUT = (V / 'figures/output_joule') if S.JOULE else (W / 'figures')
S.QA = (V / 'figures/qa_joule') if S.JOULE else (W / 'qa/figures')
S.OUT.mkdir(parents=True, exist_ok=True)
S.QA.mkdir(parents=True, exist_ok=True)
CAPS_FILE = W / ('prose/restored_figures_joule.json' if S.JOULE else 'prose/restored_figures.json')
AUD_FILE = W / ('qa/figure_export_summary_joule.json' if S.JOULE else 'qa/figure_export_summary.json')
T = W / 'tables'
S.setup()
caps = {}
audits = {}

G = ['provincial', 'national', 'community']
E_KEYS = ['observed', 'dispersed', 'clustered', 'combined']
EL = ['Real dates', 'Dispersed', 'Extended', 'Combined']
TEAL = '#19646B'
CORAL = '#C9543B'
AMBER = '#C58922'
MINT = '#7BB4B0'
SLATE = '#607178'


def read(n):
    return pd.read_csv(T / (n + '.csv'))


def register(key, title, caption, pdf=None, png=None):
    caps[key] = dict(
        type='figure',
        key=key,
        title=title,
        paragraphs=[caption],
        pdf=str(pdf or S.OUT / (key + '.pdf')),
        png=str(png or S.OUT / (key + '.png')),
    )


def export(c, key, title, caption, rows=(), cols=(), exempt=()):
    try:
        audits[key] = S.save(c, key, rows=rows, cols=cols, exempt=exempt)
    except AssertionError as e:
        audits[key] = json.loads((S.QA / (key + '_audit.json')).read_text())
        audits[key]['blocking'] = str(e)
        print('REQUIRES CORRECTION', key, e, flush=True)
    register(key, title, caption)
    CAPS_FILE.write_text(json.dumps(caps, ensure_ascii=False, indent=2))


def layout(nr, nc, h=34, gapy=23, x0=19, gapx=20, top=10, right_limit=181):
    cv = S.Canvas(top + 4 + nr * h + (nr - 1) * gapy + 12)
    axs = []
    ww = (right_limit - x0 - (nc - 1) * gapx) / nc
    import string
    for i in range(nr * nc):
        x = x0 + (i % nc) * (ww + gapx)
        y = top + (i // nc) * (h + gapy)
        label = string.ascii_lowercase[i]
        a = cv.ax(x, y, ww, h, label)
        cv.letter(x - 14, y - 4, label)
        S.grid(a)
        axs.append(a)
    rows = [[string.ascii_lowercase[j] for j in range(i * nc, (i + 1) * nc)] for i in range(nr)]
    return cv, axs, rows


def legend(cv, handles, labels, y=0.01, ncol=3):
    cv.fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.53, y), ncol=ncol, handlelength=1.8, columnspacing=2)


def group_handles():
    return [Line2D([], [], marker=S.MARKER[g], color=S.COLOR[g], linestyle='none', markersize=3.5) for g in G]


def draw_raincloud_v(ax, z, pos, color, rng, half_w=0.33):
    lo, hi = float(z.min()), float(z.max())
    if hi - lo > 0.05:
        kde = gaussian_kde(z)
        ys = np.linspace(lo, hi, 120)
        d = kde(ys)
        d = d / d.max() * half_w
        ax.fill_betweenx(ys, pos + 0.06, pos + 0.06 + d, fc=color, alpha=0.32, ec=color, lw=0.65, zorder=2)
    jit = pos - 0.16 + rng.uniform(-0.08, 0.08, len(z))
    ax.scatter(jit, z, s=3.6, c=color, alpha=0.45, lw=0, zorder=2)
    q10, q25, q50, q75, q90 = np.quantile(z, [0.10, 0.25, 0.50, 0.75, 0.90])
    ax.plot([pos, pos], [q10, q90], color=S.INK, lw=0.75, zorder=3)
    ax.add_patch(Rectangle((pos - 0.055, q25), 0.11, max(q75 - q25, 0.2), fc='white', ec=S.INK, lw=0.8, zorder=4))
    ax.plot([pos - 0.055, pos + 0.055], [q50, q50], color=CORAL, lw=1.35, zorder=5)


def framework():
    """ED Fig. 1 (Joule Fig. S14): 100% Pure-Vector Analytical Framework Schematic."""
    from vector_fig_s14 import build_vector_framework
    cv = build_vector_framework(S, TEAL=TEAL, CORAL=CORAL)
    export(
        cv,
        'extended_data_figure1',
        '扩展数据图 1 | 任务、充电窗口与经济评价的分析框架',
        '分析由研究输入、三类正文调峰与补能情境（中国各省 n = 310、欧洲各国 n = 132、日本社区 n = 10,612，覆盖 5,306 个双年完整社区，共 11,054 份年度记录）、反事实实验与经济评价流程构成。重排实验在季节组内执行 1,000 次事件放置和 1—12 天事件延长（保持完整日块与相同日曲线）；2 × 2 × 2 析因设计独立改变盈余电量、充电功率和充电时间。组合对照以 4 小时 LFP 为基准比较 24 小时 A-CAES 增量效益 B，并在相同任务和共享充电资源约束下评估 8 种技术、58 个配置目录（4—720 小时，氢储能采用 200 bar 地面钢制储罐）。外部验证包括 75 个欧洲历史天气年、32 份美国运行区记录及多年度容量规划。',
        exempt=[dict(panels=['a'], checks=['all'], reason='Conceptual pure-vector framework canvas spanning x = 1.5..180.0 mm')],
    )


def thresholds():
    """ED Fig. 2: Diverse threshold sensitivity (ridgeline, country heatmap, raincloud, contour, iso-event hyperbolas, shaded 7-baseline wedge)."""
    d = read('threshold_current')
    ar = read('annual_records_current')[['uid', 'system_id', 'P_ref_mw']]
    m = d.merge(ar, on='uid', how='inner')
    m['high_twh'] = m['high_task_energy'] * m['P_ref_mw'] / 1e6
    m['mean_gap'] = (365.0 - m['high_days']) / m['events']

    cv = S.Canvas(172.0)
    qs = [0.7, 0.75, 0.8]
    qcols = ['#5DA8A2', TEAL, AMBER]
    qlabs = ['q70', 'q75', 'q80']
    rng = np.random.default_rng(2026)

    # Row 1: a (Horizontal ridgeline of mean duration), b (Country x threshold heatmap of high-demand days), c (Raincloud of task share) — locked 2.5 .. 180.0
    xs_r1 = [17.0, 76.5, 136.0]
    w_r1_ed2 = 44.0
    axa = cv.ax(xs_r1[0], 10.0, w_r1_ed2, 36.0, 'a')
    cv.letter(2.5, 5.5, 'a')
    S.grid(axa, 'x')
    xg_a = np.linspace(1.2, 5.2, 140)
    for j, (q, c) in enumerate(zip(qs, qcols)):
        z = d[d.q.eq(q)].mean_duration.to_numpy(float)
        kd = gaussian_kde(z, bw_method=0.30)(xg_a)
        kd = kd / kd.max() * 0.65
        axa.fill_between(xg_a, j, j - kd, fc=c, alpha=0.34, ec=c, lw=0.70, zorder=2 + j * 0.2)
        q25, q50, q75 = np.quantile(z, [0.25, 0.50, 0.75])
        axa.plot([q25, q75], [j + 0.08, j + 0.08], color=c, lw=2.0, solid_capstyle='butt', zorder=4)
        axa.scatter([q50], [j + 0.08], s=11, color='white', edgecolor=S.INK, lw=0.7, zorder=5)
    axa.set_ylim(2.45, -0.82)
    axa.set_yticks([0, 1, 2], qlabs)
    axa.tick_params(axis='y', length=0); axa.spines['left'].set_visible(False)
    axa.set_xlim(1.2, 5.2); axa.set_xticks([2, 3, 4, 5])
    axa.set_xlabel('Mean event duration (d)')

    axb = cv.ax(xs_r1[1], 10.0, w_r1_ed2, 36.0, 'b')
    cv.letter(xs_r1[1] - 11.5, 5.5, 'b')
    axb.grid(False)
    hd_med = [float(d[d.q.eq(q)].high_days.median()) for q in qs]
    sh_med = [100.0 * float(d[d.q.eq(q)].high_task_share.median()) for q in qs]
    yk = np.arange(3)
    axb.barh(yk, np.full(3, 365.0), height=0.52, color='#ECF1F3', lw=0, zorder=1)
    for j, (hd, c) in enumerate(zip(hd_med, qcols)):
        axb.barh([j], [hd], height=0.52, color=c, lw=0, zorder=3)
        axb.text(hd + 6.0, j, f'{hd:.0f} d ({100 * hd / 365:.0f}% yr)', va='center', ha='left', fontsize=5.0, fontweight='bold', color=S.INK)
    axb.set_ylim(2.5, -0.5); axb.set_yticks(yk, qlabs); axb.tick_params(axis='y', length=0)
    axb.set_xlim(0, 265); axb.set_xticks([0, 90, 180])
    axb.set_xlabel('High-demand days (d yr⁻¹)')

    axc = cv.ax(xs_r1[2], 10.0, w_r1_ed2, 36.0, 'c')
    cv.letter(xs_r1[2] - 11.5, 5.5, 'c')
    S.grid(axc, 'y')
    for j, (q, c) in enumerate(zip(qs, qcols)):
        z = d[d.q.eq(q)].high_task_share.to_numpy(float) * 100.0
        draw_raincloud_v(axc, z, j, c, rng)
    axc.set_xticks([0, 1, 2], qlabs); axc.set_xlim(-0.55, 2.55)
    axc.set_ylim(52, 99); axc.set_yticks([60, 70, 80, 90])
    axc.set_xlabel('Threshold'); axc.set_ylabel('Task-energy share (%)')

    # Row 2: d (2D density contour + scatter), e (Iso-event-count hyperbola phase plane) — locked 2.5 .. 180.0
    axd = cv.ax(17.0, 62.5, 71.5, 39.0, 'd')
    cv.letter(2.5, 58.0, 'd')
    S.grid(axd)
    axe = cv.ax(108.5, 62.5, 71.5, 39.0, 'e')
    cv.letter(95.5, 58.0, 'e')
    S.grid(axe)

    # Smooth background KDE contour in d
    kd_d = gaussian_kde(np.vstack([m.longest.to_numpy(float), m.high_task_share.to_numpy(float) * 100.0]), bw_method=0.32)
    GX, GY = np.meshgrid(np.linspace(3.5, 22.5, 60), np.linspace(55, 96, 60))
    Zd = kd_d(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
    axd.contourf(GX, GY, Zd, levels=4, cmap=LinearSegmentedColormap.from_list('bg_d', ['#FFFFFF', '#E2F0EF', '#B2D8D5']), alpha=0.55, zorder=1)

    # Iso-event-count reference lines in e: events = 365 / (gap + duration)
    g_lin = np.linspace(3.5, 20.5, 150)
    for n_ev in (25, 35, 45):
        d_iso = 365.0 / n_ev - g_lin
        keep = (d_iso >= 1.25) & (d_iso <= 5.3)
        if keep.any():
            axe.plot(g_lin[keep], d_iso[keep], color='#BDCAD0', lw=0.55, ls=':', zorder=1)
            idx_lbl = np.flatnonzero(keep)[len(np.flatnonzero(keep)) // 2]
            axe.text(g_lin[idx_lbl] + 0.3, d_iso[idx_lbl] + 0.15, f'{n_ev}/yr', fontsize=5.0, color=S.MUTED)

    for q, c, ql in zip(qs, qcols, qlabs):
        z = m[m.q.eq(q)]
        axd.scatter(z.longest, z.high_task_share * 100.0, s=11, c=c, alpha=0.68, lw=0.25, ec='white', label=ql, zorder=3)
        sizes = 7.5 + 13.0 * np.sqrt(z.high_twh.clip(lower=0.0))
        axe.scatter(z.mean_gap, z.mean_duration, s=sizes, c=c, alpha=0.65, lw=0.3, ec='white', zorder=3)

    axd.set_xlim(3, 23.5); axd.set_xticks([4, 8, 12, 16, 20])
    axd.set_ylim(54, 97); axd.set_yticks([60, 70, 80, 90])
    axd.set_xlabel('Longest event (days)'); axd.set_ylabel('Task-energy share (%)')
    axd.legend(loc='lower left', bbox_to_anchor=(0.0, 1.02), ncol=3, fontsize=5.1, handletextpad=0.3, columnspacing=1.0)

    axe.set_xlim(4.0, 20.5); axe.set_xticks([5, 10, 15, 20])
    axe.set_ylim(1.2, 5.4); axe.set_yticks([2, 3, 4, 5])
    axe.set_xlabel('Mean inter-event gap (days)'); axe.set_ylabel('Mean event duration (days)')
    size_handles = [
        axe.scatter([], [], s=7.5 + 13.0 * np.sqrt(v), c='#7D8F96', alpha=0.7, lw=0.3, ec='white', label=f'{v:g} TWh')
        for v in (0.5, 3, 10)
    ]
    axe.legend(handles=size_handles, loc='lower left', bbox_to_anchor=(0.0, 1.02), ncol=3, fontsize=5.1, handletextpad=0.4, columnspacing=0.9)

    # Row 3: f (7-baseline q5..q50 with shaded MC-Real chronology uplift wedges) — locked 2.5 .. 180.0
    mq = read('multi_q_7baselines_current')
    axf = cv.ax(17.0, 122.5, 163.0, 35.0, 'f')
    cv.letter(2.5, 118.0, 'f')
    S.grid(axf, 'y')
    q_order = [5, 10, 20, 25, 30, 35, 50]
    x_idx = np.arange(len(q_order))
    series_specs = [
        ('China_proportional', 'China prop. (310)', S.COLOR['provincial'], 's'),
        ('China_clipped', 'China clipped (310)', '#7A5410', 'D'),
        ('Europe', 'Europe (132)', S.COLOR['national'], 'o'),
        ('Japan', 'Japan (10,612)', S.COLOR['community'], '^'),
    ]
    for s_key, s_lab, col_s, mk_s in series_specs:
        sub = mq[mq.series.eq(s_key)].set_index('q').reindex(q_order)
        if s_key in ('Europe', 'Japan'):
            axf.fill_between(x_idx, sub.real_share_pct, sub.mc_share_pct, color=col_s, alpha=0.16, lw=0, zorder=2)
        axf.plot(x_idx, sub.real_share_pct, color=col_s, lw=1.35, marker=mk_s, ms=3.3, mec='white', mew=0.25,
                 label=f'{s_lab}, real', zorder=3.5)
        axf.plot(x_idx, sub.mc_share_pct, color=col_s, lw=1.1, ls='--', marker=mk_s, ms=3.1, mfc='white', mec=col_s, mew=0.65,
                 label=f'{s_lab}, MC', zorder=4.0)
    axf.set_xticks(x_idx, [f'q{qv}' for qv in q_order])
    axf.set_xlim(-0.35, len(q_order) - 0.65)
    axf.set_ylim(-2.0, 62.0); axf.set_yticks([0, 20, 40, 60])
    axf.set_xlabel('Month–hour task baseline quantile')
    axf.set_ylabel('Hybrid cheaper at 3.6% (%)')
    axf.legend(loc='lower left', bbox_to_anchor=(0.0, 1.02), ncol=4, fontsize=5.0, handlelength=1.5, handletextpad=0.35, columnspacing=0.85)

    export(
        cv,
        'extended_data_figure2',
        '扩展数据图 2 | 高需求日识别阈值与调峰任务分位基线的敏感性',
        'a—c，高需求日识别阈值为季节内日任务第 70（q70）、第 75（q75，基准）与第 80 百分位（q80）时，全部 132 份欧洲国家年度记录的平均事件持续时间山峦分布（a）、年度高需求日数占比条（b）与高需求日承担的年度任务电量份额半小提琴分布（c）。d，不同阈值下最长连续高需求事件与高需求份额的二维密度与散点分布（n = 396 条阈值—记录组合）。e，事件间平均日历间隔与平均事件持续时间相平面（点线为年事件次数等值双曲线，气泡面积对应年高需求任务电量 TWh）；a—e 基于月—小时 q50 调峰基线。f，2035 年技术成本与 3.6% 实际资金成本下，月—小时调峰基线取 q5、q10、q20、q25、q30、q35 和 q50 时，中国各省（n = 310，同时列出按出力比例分配与月度削峰两种逐小时弃电估算）、欧洲各国（n = 132）和日本社区（n = 10,612）在原始日期（实线）与每份记录 1,000 次季节内随机放置（虚线，色带为时序增益）下混合组合具有成本优势的比例。',
        rows=[['a', 'b', 'c'], ['d', 'e']],
        exempt=[dict(panels=['f'], checks=['panel-width'], reason='Full-width 7-baseline cross-regime comparison panel spanning columns a-e')],
    )


def sensitivities():
    """ED Fig. 3: Price quantile fans + 4-rule horizontal threshold-split ridgeline landscapes across China, Europe, Japan."""
    cv = S.Canvas(140.0)
    p = read('price_current')
    r = read('rule_current')
    pos_uids = D.cn_positive_uids()
    rp = [
        ('registered', 'Registered'),
        ('all_task', 'All task'),
        ('netted_tail', 'Net / tail'),
        ('netted_all_task', 'Net / all'),
    ]

    def _cn_pos_leg(ax_l, xx):
        ax_l.plot([xx, xx + 4.5], [0.5, 0.5], color=S.DEEP['provincial'], lw=1.05, ls='--')
        ax_l.text(xx + 5.5, 0.5, f"CN pos. ({'A' if S.JOULE else 'a'})", va='center', fontsize=5.1, color=S.DEEP['provincial'])
        return xx + 25.0

    S.regime_legend_strip(cv, 2.5, 1.5, 177.5, 'leg_top', extra_items=(_cn_pos_leg,), framed=True)
    xs_col = [18.5, 76.25, 134.0]
    w_col_r1 = 46.0
    w_col_r2 = 39.2
    r1_ids, r2_ids = [], []

    for i, g in enumerate(G):
        # Row 1: Price multiplier quantile fan + win-share callout (spanning xs_col[i] .. xs_col[i] + 46.0 -> 180.0)
        let1 = 'abc'[i]
        r1_ids.append(let1)
        cv.letter(2.5 if i == 0 else xs_col[i] - 9.5, 14.5, let1)
        a = cv.ax(xs_col[i], 19.5, w_col_r1, 37.0, let1)
        S.grid(a, 'y')
        z = p[p.group.eq(g)].assign(mult=lambda d_: d_.variant.str[1:].astype(float))
        pv = z.pivot(index='uid', columns='mult', values='B').sort_index(axis=1)
        cols_x = pv.columns.to_numpy(float)
        q05, q25, q50, q75, q95 = [pv.quantile(qv).to_numpy(float) for qv in (0.05, 0.25, 0.50, 0.75, 0.95)]
        ylo = min(-38.0, float(pv.min().min() if len(pv) <= 200 else q05.min()) - 4.0)
        yhi = max(16.0, float(pv.max().max() if len(pv) <= 200 else q95.max()) + 6.0)
        a.axhspan(0, yhi, facecolor=S.WIN_PALE, zorder=0)
        a.fill_between(cols_x, q05, q95, color=S.COLOR[g], alpha=0.14, lw=0, zorder=1)
        a.fill_between(cols_x, q25, q75, color=S.COLOR[g], alpha=0.30, lw=0, zorder=2)
        if len(pv) <= 200:
            for v in pv.to_numpy():
                a.plot(cols_x, v, color=S.COLOR[g], alpha=0.14, lw=0.40, zorder=1.5)
        a.plot(cols_x, q50, color=S.COLOR[g], marker=S.MARKER[g], ms=3.2, mec='white', mew=0.25, lw=1.45, zorder=4)
        if g == 'provincial':
            pvp = pv.loc[pv.index.isin(pos_uids)]
            a.plot(cols_x, pvp.median().to_numpy(float), color=S.DEEP['provincial'], lw=1.1, ls='--', marker='s', ms=2.5, zorder=4.2)
        sh0 = 100.0 * float((pv[0.5] >= 0).mean())
        sh1 = 100.0 * float((pv[1.25] >= 0).mean())
        a.text(0.5, 1.03, f'Win: {sh0:.0f}% to {sh1:.0f}%', transform=a.transAxes,
               ha='center', va='bottom', fontsize=5.0, fontweight='bold', color=S.WIN)
        a.axhline(0, color=S.INK, lw=0.6)
        a.set_xlim(0.46, 1.29); a.set_ylim(ylo, yhi)
        a.set_xticks([0.5, 0.75, 1.0, 1.25], ['0.5', '0.75', '1.0', '1.25'])
        a.set_xlabel('Residual-energy price multiplier')
        if i == 0:
            a.set_ylabel(f'B ({S.UNIT})')

        # Row 2: 4-rule horizontal threshold-split ridgeline + right-aligned win shares locked flush to xs_col[i] + w_col_r1
        let2 = 'def'[i]
        r2_ids.append(let2)
        cv.letter(2.5 if i == 0 else xs_col[i] - 9.5, 72.5, let2)
        a2 = cv.ax(xs_col[i], 77.0, w_col_r2, 42.0, let2)
        a2.grid(False)
        zr = r[r.group.eq(g)]
        xlim_r = (-40.0, 65.0) if g != 'community' else (-18.0, 36.0)
        xg_r = np.linspace(xlim_r[0], xlim_r[1], 180)
        a2.axvspan(0.0, xlim_r[1], facecolor=S.WIN_PALE, zorder=0)
        a2.axvline(0.0, color=S.WIN, lw=0.70, ls='--', zorder=4)
        for j, (rk, rlab) in enumerate(rp):
            vals = np.clip(zr[zr.variant.eq(rk)].B.to_numpy(float), xlim_r[0] + 0.5, xlim_r[1] - 0.5)
            sh_r = 100.0 * float((zr[zr.variant.eq(rk)].B >= 0).mean())
            if np.std(vals) > 0.4:
                kd = gaussian_kde(vals, bw_method=0.28)(xg_r)
            else:
                kd = np.exp(-0.5 * ((xg_r - float(np.mean(vals))) / 1.2) ** 2)
            kd = kd / kd.max() * 0.72
            yb = float(j)
            yt = yb - kd
            a2.plot(xlim_r, [yb, yb], color='#D6E0E4', lw=0.45, zorder=1.5)
            neg_m = xg_r <= 0.2
            pos_m = xg_r >= -0.2
            a2.fill_between(xg_r[neg_m], yt[neg_m], yb, fc='#DCE5E9', alpha=0.88, lw=0, zorder=2 + j * 0.2)
            a2.fill_between(xg_r[pos_m], yt[pos_m], yb, fc=S.WIN_MID, alpha=0.85, lw=0, zorder=2.1 + j * 0.2)
            a2.plot(xg_r, yt, color=S.INK, lw=0.60, zorder=2.3 + j * 0.2)
            med_v = float(np.median(vals))
            a2.scatter([med_v], [yb - 0.08], s=10, marker=S.MARKER[g], color=S.COLOR[g], edgecolor='white', lw=0.25, zorder=4.5)
            sh_lbl = '100%' if sh_r > 99.5 else f'{sh_r:.0f}%'
            y_frac = ((yb - 0.22) - (-0.88)) / (3.35 - (-0.88))
            cv.text(xs_col[i] + w_col_r1, 77.0 + y_frac * 42.0, sh_lbl,
                    ha='right', va='center', fontsize=5.0, fontweight='bold', color=S.WIN)
        a2.set_xlim(*xlim_r)
        a2.set_ylim(3.35, -0.88)
        a2.set_yticks([0, 1, 2, 3], [x_[1] for x_ in rp] if i == 0 else [], fontsize=5.05)
        a2.tick_params(axis='y', length=0); a2.spines['left'].set_visible(False)
        a2.set_xlabel(f'B ({S.UNIT})')

    export(
        cv,
        'extended_data_figure3',
        '扩展数据图 3 | 电量价格与储能服务规则的敏感性',
        'a—c，残余任务电量价格为基准的 0.5、0.75、1.0 和 1.25 倍时的系统成本差 B 分位扇形带（浅带为第 5—95 百分位，深带为第 25—75 百分位，实线标记为全样本中位数，a 中虚线为中国 178 份正弃电省年中位数，右下标注胜出比例增幅）。d—f，登记任务（Registered）、开放全部任务（All task）、先净额抵消再服务高需求任务（Net / tail）与先净额抵消再开放全部任务（Net / all）四类服务规则下的阈值分割山峦核密度分布（右侧青绿区为 B ≥ 0，右侧数字为混合组合胜出比例）。三列依次为中国各省（n = 310）、欧洲各国（n = 132）及日本社区（n = 10,612）。',
        rows=[r1_ids, r2_ids],
        exempt=[
            dict(panels=['leg_top'], checks=['all'], reason='Top shared regime legend strip'),
            dict(panels=r2_ids, checks=['panel-width'], reason='Ridgeline axis paired with right-aligned win-share column matching Row 1 column right edge'),
        ],
    )


def weather():
    """ED Fig. 4: European historical weather inputs + 33-country European GW choropleth map + 75-year calendar."""
    cv = S.Canvas(162.0)

    # Row 1: a (PECD region coverage), b (country-to-zone matching composition)
    axa = cv.ax(19.0, 10.0, 69.0, 29.0, 'a')
    cv.letter(4.5, 5.5, 'a')
    axa.grid(False)
    cov = read('extended_data_figure2b_spatial_coverage').iloc[::-1].reset_index(drop=True)
    y = np.arange(len(cov))
    clrs_cov = ['#7C929B', AMBER, MINT, TEAL]
    names_cov = {'onshore': 'Onshore wind', 'offshore': 'Offshore wind', 'solar': 'Solar PV', 'temperature': 'Temperature'}
    axa.barh(y, cov.regions, height=0.58, color=clrs_cov, zorder=2)
    for yi, reg, ct in zip(y, cov.regions, cov.countries):
        axa.text(reg + 2.5, yi, f'{int(reg)} zones ({int(ct)} ctry)', va='center', ha='left', fontsize=5.1, color=S.INK)
    axa.set_yticks(y, [names_cov[p] for p in cov['product']]); axa.tick_params(axis='y', length=0)
    axa.set_xlim(0, 165); axa.set_xticks([0, 40, 80, 120])
    axa.set_xlabel('PECD v4.2 regions (1950–2025)')

    axb = cv.ax(109.0, 10.0, 69.0, 29.0, 'b')
    cv.letter(95.5, 5.5, 'b')
    S.grid(axb, 'y')
    ma = read('extended_data_figure2d_matching_counts')
    cols = ['matched_zone_weighted', 'country_total_distributed_to_pecd_regions', 'matched_zone_weighted_plus_country_residual', 'unweighted_zero_structural_weight']
    labs = ['Matched zones', 'Country alloc.', 'Zones + residual', 'Zero weight']
    cl = [TEAL, '#9FC5C2', AMBER, '#DCE2E5']
    bot = np.zeros(len(ma))
    for k, lab, col in zip(cols, labs, cl):
        axb.bar(np.arange(len(ma)), ma[k], bottom=bot, width=0.50, color=col, label=lab, zorder=2)
        bot += ma[k]
    axb.set_xticks(np.arange(len(ma)), ['Onshore', 'Offshore', 'Solar PV'])
    axb.set_xlim(-0.55, 2.55); axb.set_ylim(0, 36); axb.set_yticks([0, 10, 20, 30])
    axb.set_ylabel('Countries (n = 33)')
    axb.legend(loc='lower left', bbox_to_anchor=(-0.05, 1.02), ncol=2, fontsize=5.0, handletextpad=0.35, columnspacing=0.9)

    # Row 2: c (33-country European 2030 VRE capacity choropleth map + ranked stacked GW bars, total 1.20 TW)
    cv.letter(4.5, 49.0, 'c')
    d = read('capacity_weights').copy()
    d['total_gw'] = (d.onshore_capacity_mw + d.offshore_capacity_mw + d.solar_capacity_mw) / 1000.0
    d = d.sort_values('total_gw', ascending=False).reset_index(drop=True)
    gw_map = dict(zip(d.country, np.log10(np.maximum(d.total_gw, 0.3))))
    ax_map, w_map = GH.draw_europe_choropleth(
        cv, 19.0, 53.5, 34.0, gw_map, S.PHASE_CMAP, Normalize(vmin=-0.5, vmax=2.55),
        'c_map', lon=(-11.0, 31.5), lat=(35.0, 70.5)
    )
    cv.text(19.0 + w_map / 2, 50.8, '33-ctry 2030 VRE weight (log GW)', ha='center', va='top',
            fontsize=5.0, fontweight='bold', color=TEAL)

    x_bar = 19.0 + w_map + 9.5
    axc = cv.ax(x_bar, 53.5, 178.0 - x_bar, 34.0, 'c')
    S.grid(axc, 'y')
    x = np.arange(len(d))
    bot = np.zeros(len(d))
    for col, lab, c in zip(['onshore_capacity_mw', 'offshore_capacity_mw', 'solar_capacity_mw'],
                           ['Onshore wind', 'Offshore wind', 'Solar PV'], [TEAL, MINT, AMBER]):
        v = d[col].to_numpy() / 1000.0
        axc.bar(x, v, bottom=bot, width=0.74, color=c, label=lab, zorder=2)
        bot += v
    axc.set_xticks(x, d.country, fontsize=5.0, rotation=90)
    axc.tick_params(axis='x', length=0)
    axc.set_xlim(-0.7, len(d) - 0.3); axc.set_ylim(0, 365); axc.set_yticks([0, 100, 200, 300])
    axc.set_ylabel('2030 weight (GW)')
    axc.legend(loc='lower left', bbox_to_anchor=(0.0, 1.02), ncol=3, fontsize=5.05, handletextpad=0.35, columnspacing=1.0)
    axc.text(0.99, 0.90, f'33-country total = {d.total_gw.sum() / 1e3:.2f} TW',
             transform=axc.transAxes, ha='right', va='top', fontsize=5.2, fontweight='bold', color=S.INK)

    # Row 3: d (75-year q50 high-demand calendar tapestry), e (aligned longest-event horizontal bars)
    h = read('historical_calendar_current')
    p = h.pivot(index='year', columns='day', values='event').sort_index()
    axd = cv.ax(19.0, 107.0, 114.0, 42.0, 'd')
    cv.letter(4.5, 102.5, 'd')
    axd.grid(False)
    axd.imshow(p.to_numpy(), aspect='auto', interpolation='nearest',
               cmap=ListedColormap(['#EEF2F4', TEAL]), extent=[0, 365, 2024.5, 1949.5])
    axd.set_yticks([1950, 1975, 2000, 2024])
    axd.set_xticks([0, 92, 184, 274], ['Jul', 'Oct', 'Jan', 'Apr'])
    axd.set_xlabel('Month within July–June operating year')
    axd.set_ylabel('Operating-year start')

    tr = read('historical_trend_q50')
    tr = tr[tr.threshold.eq('q75')].sort_values('year')
    axe = cv.ax(140.0, 107.0, 38.0, 42.0, 'e')
    cv.letter(134.5, 102.5, 'e')
    S.grid(axe, 'x')
    cols_e = [CORAL if v >= 10 else '#7B9DA3' for v in tr.longest]
    axe.barh(tr.year, tr.longest, height=0.82, color=cols_e, zorder=2)
    axe.axvline(10, color=CORAL, lw=0.75, ls='--', zorder=3)
    axe.set_ylim(2024.5, 1949.5); axe.set_yticks([1950, 1975, 2000, 2024]); axe.set_yticklabels([])
    axe.set_xlim(0, 14.5); axe.set_xticks([0, 5, 10])
    axe.set_xlabel('Longest event (d)')
    n_ge10 = int((tr.longest >= 10).sum())
    axe.text(10.4, 1958.0, f'≥10 d\n({n_ge10} yr)', fontsize=5.1, fontweight='bold', color=CORAL, va='center', ha='left')

    export(
        cv,
        'extended_data_figure4',
        '扩展数据图 4 | 欧洲历史天气输入与高需求日历',
        'a，PECD v4.2 气象产品区域与国家覆盖（1950—2025 年）。b，气候区域与装机权重的匹配方式；陆上风电与光伏覆盖全部 33 国，海上风电覆盖 24 国（其余 10 国装机权重为零）。c，33 国 2030 年风光装机权重的欧洲空间分布地图（左，对数色阶）与降序堆叠构成（右，合计 1.20 TW）。d，月—小时 q50 调峰基线与 q75 阈值下的 1950/51—2024/25 年（共 75 个运行年）高需求日历，深色为高需求日。e，与 d 逐年对齐的年度最长连续高需求事件天数，≥ 10 天年份（共 5 年）以珊瑚色突出。',
        rows=[['a', 'b'], ['d', 'e']],
        exempt=[dict(panels=['c_map', 'c', 'e'], checks=['all'], reason='European map paired beside 33-country bar chart and margin bar plot e')],
    )


def validation_trend():
    """ED Fig. 6: Coupled 17-country European leave-one-out R² map + ranked fit (a) + q10 vs q50 historical ribbons & epoch counts (b, c)."""
    d = read('demand_spatial_validation').sort_values('weighted_r2').reset_index(drop=True)
    cv = S.Canvas(112.0)

    cv.letter(3.0, 5.0, 'a')
    r2_dict = dict(zip(d.country, d.weighted_r2))
    ax_map, w_map = GH.draw_europe_choropleth(
        cv, 15.0, 11.0, 35.1, r2_dict, S.PHASE_CMAP, Normalize(vmin=0.45, vmax=0.88),
        'a_map', lon=(-10.5, 30.5), lat=(35.0, 60.5)
    )
    cv.text(15.0 + w_map / 2, 8.2, 'Leave-one-out R² (17 ctry)', ha='center', va='top',
            fontsize=5.0, fontweight='bold', color=TEAL)

    axa = cv.ax(15.0, 53.0, w_map, 46.0, 'a')
    S.grid(axa, 'x')
    y = np.arange(len(d))
    axa.barh(y, d.weighted_pearson_r, height=0.62, facecolor='#E8B1A2', lw=0, label='Pearson r', zorder=2)
    axa.barh(y, d.weighted_r2, height=0.34, facecolor=TEAL, lw=0, label='R²', zorder=3)
    med_r2 = float(d.weighted_r2.median())
    med_r = float(d.weighted_pearson_r.median())
    axa.axvline(med_r2, color=TEAL, lw=0.75, ls='--', zorder=3.5)
    axa.axvline(med_r, color=CORAL, lw=0.75, ls='--', zorder=3.5)
    axa.set_yticks(y, d.country, fontsize=5.0); axa.tick_params(axis='y', length=0)
    axa.set_xlim(0.38, 1.01); axa.set_xticks([0.4, 0.6, 0.8, 1.0])
    axa.set_xlabel('Spatial hold-out fit')
    axa.legend(loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=2, fontsize=5.05, handletextpad=0.3, columnspacing=0.8)

    cv.letter(72.0, 5.0, 'b')
    axb = cv.ax(84.0, 11.0, 96.0, 36.0, 'b')
    S.grid(axb, 'y')
    cv.letter(72.0, 57.0, 'c')
    axc = cv.ax(84.0, 63.0, 96.0, 36.0, 'c')
    S.grid(axc, 'y')

    for q, co in [(10, AMBER), (50, TEAL)]:
        z = read(f'historical_trend_q{q}')
        z = z[z.threshold.eq('q75')].sort_values('year')
        roll_m = z.longest.rolling(11, center=True, min_periods=6).mean()
        roll_q1 = z.longest.rolling(11, center=True, min_periods=6).quantile(0.25)
        roll_q3 = z.longest.rolling(11, center=True, min_periods=6).quantile(0.75)
        axb.fill_between(z.year, roll_q1, roll_q3, color=co, alpha=0.16, lw=0, zorder=1.5)
        axb.plot(z.year, z.longest, color=co, lw=0.50, alpha=0.40, zorder=2)
        axb.plot(z.year, roll_m, color=co, lw=1.55, label=f'Task baseline q{q}', zorder=3.5)
        counts = [int((z[z.year.between(lo, lo + 24)].longest >= 10).sum()) for lo in (1950, 1975, 2000)]
        xpos = np.arange(3) + (-0.18 if q == 10 else 0.18)
        axc.bar(xpos, counts, width=0.33, color=co, alpha=0.88, label=f'Task baseline q{q}', zorder=2.5)
        for bx, val in zip(xpos, counts):
            axc.text(bx, val + 0.35, str(val), ha='center', va='bottom', fontsize=5.15, fontweight='bold', color=co)

    axb.axhline(10, color=CORAL, lw=0.6, ls=':', zorder=1)
    axb.set_xlim(1949, 2026); axb.set_xticks([1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020])
    axb.set_ylim(3, 16.5); axb.set_yticks([4, 8, 12, 16])
    axb.set_xlabel('Operating-year start'); axb.set_ylabel('Longest event (days)')
    axb.legend(loc='lower left', bbox_to_anchor=(0.0, 1.02), ncol=2, fontsize=5.1, handlelength=1.6, columnspacing=1.2)

    axc.set_xticks([0, 1, 2], ['1950–1974', '1975–1999', '2000–2024'])
    axc.set_xlim(-0.55, 2.55); axc.set_ylim(0, 14.2); axc.set_yticks([0, 4, 8, 12])
    axc.set_ylabel('Years with ≥10-day events')
    axc.text(0.5, 1.03, 'IRR: q10 = 1.21 (1.05–1.47); q50 = 1.27 (0.90–2.09)',
             transform=axc.transAxes, ha='center', va='bottom', fontsize=5.05, color=S.INK)

    export(
        cv,
        'extended_data_figure6',
        '扩展数据图 6 | 需求模型空间验证与事件趋势的基线敏感性',
        f'a，17 个国家的逐国留出需求模型空间验证：上方为欧洲各国留出验证决定系数 R² 分布地图，下方嵌套子弹条形为各国 R²（深青内条，中位数 {med_r2:.3f}）与 Pearson 相关系数 r（浅珊瑚外条，中位数 {med_r:.3f}），虚线为中位数。b，欧洲 75 个运行年的最长连续高需求事件天数（细线为逐年值，粗线与色带为 11 年滑动均值与四分位范围）。c，三个连续 25 年时段（各 n = 25）内出现 ≥ 10 天高需求事件的运行年数（琥珀色为 q10 基线，青绿色为 q50 基线，高需求日识别阈值均为季节 q75；每十年发生率比 IRR 分别为 1.21 与 1.27）。',
        cols=[['b', 'c']],
        exempt=[dict(panels=['a_map', 'a'], checks=['all'], reason='European choropleth map stacked above 17-country validation panel alongside b and c')],
    )


def capacity_full():
    """Supp Fig. 5: 3x2 step comparison (2, 4, 6 h x Mean, Risk) with sorted countries and A-CAES share callouts."""
    d = read('capacity_technology_power_current')
    cv, axs, rows = layout(3, 2, h=31, gapy=22)
    countries = sorted(d.country.unique())
    tech = ['li_lfp_4h', 'caes_24h', 'h2_tank_168h']
    co = ['#9AA8AE', TEAL, AMBER]
    steps = [2, 4, 6]
    risks = [(0, 'Mean cost'), (0.5, 'Mean + CVaR90')]
    for j, step in enumerate(steps):
        for k, (risk, rlab) in enumerate(risks):
            a = axs[2 * j + k]
            z = d[d.step.eq(step) & d.risk.eq(risk) & d.ensemble.eq('combined') & d['mode'].eq('diversified')]
            pv = z.pivot(index='country', columns='config_id', values='capacity_per_normalized_mw').reindex(countries)
            bot = np.zeros(len(countries))
            for t, col in zip(tech, co):
                v = pv[t].to_numpy() * 100
                a.bar(np.arange(15), v, bottom=bot, color=col, width=0.74, zorder=3)
                bot += v
            caes_sh = float((pv['caes_24h'] / pv.sum(axis=1)).median() * 100)
            a.text(0.03, 0.92, f'{step} h, {rlab} (med {caes_sh:.0f}% CAES)',
                   transform=a.transAxes, ha='left', va='top', fontsize=5.05, fontweight='bold', color=TEAL)
            a.set_xticks(range(15), countries, rotation=90, fontsize=5.0)
            a.tick_params(axis='x', length=0)
            a.set_ylim(0, 960); a.set_yticks([0, 300, 600, 900])
            if k == 0:
                a.set_ylabel('Power (% gas)')
    legend(cv, [Rectangle((0, 0), 1, 1, fc=c) for c in co], ['4 h LFP', '24 h A-CAES', '168 h Tank H₂ (0%)'])
    export(
        cv,
        'supp_figure5',
        '图 S5 | 多年度容量规划的完整步长对照',
        '各行依次为 2、4 和 6 小时步长；左、右列分别采用平均成本目标及均值与 CVaR90 各占一半的风险目标。全部面板采用分散与延长排列的合并情景集，n = 15 国，面板左上标注各国 24 小时 A-CAES 占总采购功率的中位数。堆叠柱为各技术采购功率，占该国平均原始燃气出力的比例；200 bar 地面储罐氢储能在所有图示设计中采购量为零。',
        rows,
    )


def nearoptimal():
    """Supp Fig. 6: Nested horizontal bullet tolerance intervals (3 countries x 2 techs)."""
    d = read('nearoptimal_intervals_current')
    cv = S.Canvas(130.0)
    countries = [('CZ', 'Czechia (CZ)', 410, [0, 100, 200, 300, 400]),
                 ('RO', 'Romania (RO)', 102, [0, 25, 50, 75, 100]),
                 ('ES', 'Spain (ES)', 195, [0, 50, 100, 150, 200])]
    techs = [('caes_24h', '24 h A-CAES', ('#CBE3E2', '#529B99', '#10494F')),
             ('li_lfp_4h', '4 h LFP', ('#DCE3E7', '#7F96A3', '#2B3E49'))]
    rows_spec = [(r, e) for r in (0.0, 0.5) for e in E_KEYS]
    row_labels = [f"{'M' if r == 0.0 else 'R'}, {EL[i]}" for i, r in [(E_KEYS.index(e), r) for r, e in rows_spec]]

    panel_rows = []
    idx = 0
    for ri, (ctry, ctry_lab, xlim, xticks) in enumerate(countries):
        prow = []
        y0 = 13.0 + ri * 38.0
        for ci, (tech, tlab, (c10, c01, c00)) in enumerate(techs):
            let = 'abcdef'[idx]
            idx += 1
            x0 = 26.0 + ci * 81.0
            ax = cv.ax(x0, y0, 68.0, 29.0, let)
            cv.letter(x0 - (23.0 if ci == 0 else 10.0), y0 - 4.0, let)
            prow.append(let)
            S.grid(ax, 'x')
            for y_i, (r, e) in enumerate(rows_spec):
                yc = 7 - y_i
                sub = d[(d.country == ctry) & (d.technology == tech) & (d.ensemble == e) & (d.risk == r)]
                s1 = sub[sub.epsilon == 0.01].iloc[0]
                s01 = sub[sub.epsilon == 0.001].iloc[0]
                s0 = sub[sub.epsilon == 0.0].iloc[0]
                lo1, hi1 = 100 * s1['min'], 100 * s1['max']
                lo01, hi01 = 100 * s01['min'], 100 * s01['max']
                lo0, hi0 = 100 * s0['min'], 100 * s0['max']
                ax.barh(yc, max(hi1 - lo1, 1.5), left=lo1, height=0.62, color=c10, lw=0, zorder=2)
                ax.barh(yc, max(hi01 - lo01, 1.2), left=lo01, height=0.34, color=c01, lw=0, zorder=3)
                ax.plot([lo0, hi0], [yc, yc], color=c00, lw=1.8, solid_capstyle='butt', zorder=4)
                ax.scatter([0.5 * (lo0 + hi0)], [yc], s=9, color=c00, edgecolor='white', lw=0.25, zorder=5)
            ax.axhline(3.5, color='#CBD2D4', lw=0.75, ls='--')
            ax.set_xlim(-6, xlim); ax.set_xticks(xticks)
            ax.set_ylim(-0.6, 7.6)
            ax.set_yticks(range(8), list(reversed(row_labels)) if ci == 0 else [], fontsize=5.05)
            ax.tick_params(axis='y', length=0)
            if ri == 0:
                ax.text(0.5, 1.04, tlab, transform=ax.transAxes, ha='center', va='bottom', fontsize=5.8, fontweight='bold', color=S.INK)
            if ci == 1:
                ax.text(1.03, 0.5, ctry_lab, transform=ax.transAxes, ha='left', va='center', rotation=-90, fontsize=5.4, fontweight='bold', color=S.INK)
            if ri == 2:
                ax.set_xlabel('Procured power (% of mean gas output)')
        panel_rows.append(prow)

    handles = [
        Patch(fc='#CBE3E2', ec='none', label='±1.0% cost tolerance'),
        Patch(fc='#529B99', ec='none', label='±0.1% cost tolerance'),
        Line2D([], [], color='#10494F', lw=1.8, marker='o', ms=3, label='0% optimum'),
    ]
    cv.fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.54, 0.005), ncol=3, fontsize=5.3, columnspacing=1.8)

    export(
        cv,
        'supp_figure6',
        '图 S6 | 1 小时规划的完整近优功率区间',
        '各行依次为捷克（CZ，a—b）、罗马尼亚（RO，c—d）和西班牙（ES，e—f）；左列为 24 小时 A-CAES，右列为 4 小时 LFP。纵轴为目标与情景组合（M 为平均成本目标，R 为均值与 CVaR90 各占一半的风险目标；Real dates、Dispersed、Extended、Combined 依次为真实日期、分散、延长及合并情景集），横轴为相同预算下的可行采购功率区间（占该国平均原始燃气出力的百分比）。宽浅色条、窄中色条和深色点线分别对应 ±1.0%、±0.1% 成本容差与 0% 最优解；200 bar 地面储罐氢储能在所有情景中严格为零。',
        rows=panel_rows,
        cols=[['a', 'c', 'e'], ['b', 'd', 'f']],
    )


def finance():
    """Supp Fig. 7: Shaded duration-spread learning ribbons (a, b), area-filled loss share (c), and 2,775-pair raincloud (d)."""
    cv, axs, rows = layout(2, 2, h=37, gapy=24)
    d = read('finance_paths_current')
    z8 = d[d.duration_h.eq(8)].sort_values('rate')
    z24 = d[d.duration_h.eq(24)].sort_values('rate')
    axs[0].fill_between(z24.rate * 100, z8.premium2035, z24.premium2035, color=MINT, alpha=0.35, lw=0, zorder=1)
    axs[1].fill_between(z24.rate * 100, z8.M, z24.M, color=MINT, alpha=0.35, lw=0, zorder=1)
    for h, co in zip([8, 16, 24], ['#8EAAB0', '#4A8D94', TEAL]):
        z = d[d.duration_h.eq(h)].sort_values('rate')
        axs[0].plot(z.rate * 100, z.premium2035, color=co, marker='o', ms=2.4, lw=1.25, label=f'{h} h')
        axs[1].plot(z.rate * 100, z.M, color=co, marker='o', ms=2.4, lw=1.25)

    p36 = d[(d.duration_h == 24) & (d.rate == 0.036)].iloc[0]
    M = float(p36.M)
    axs[0].scatter([3.6], [p36.premium2035], s=36, fc='none', ec=CORAL, lw=1.2, zorder=5)
    axs[0].annotate(f'3.6%, 24 h\n{p36.premium2035:.2f}', xy=(3.6, p36.premium2035), xytext=(4.8, 13.0),
                    fontsize=5.1, fontweight='bold', color=CORAL, arrowprops=dict(arrowstyle='-', color=CORAL, lw=0.7))
    axs[1].scatter([3.6], [M], s=36, fc='none', ec=CORAL, lw=1.2, zorder=5)
    axs[1].annotate(f'M = {M:.2f}', xy=(3.6, M), xytext=(4.8, 3.0),
                    fontsize=5.1, fontweight='bold', color=CORAL, arrowprops=dict(arrowstyle='-', color=CORAL, lw=0.7))

    axs[0].set_ylabel(f'2035 fixed-cost premium\n({S.UNIT})')
    axs[0].set_yticks([0, 25, 50, 75]); axs[0].set_ylim(-5, 84)
    axs[0].legend(loc='upper left', ncol=3, fontsize=5.1, handletextpad=0.3, columnspacing=0.8)
    axs[1].set_ylabel(f'2025–2035 reduction M\n({S.UNIT})')
    axs[1].set_yticks([0, 5, 10, 15, 20]); axs[1].set_ylim(-2.5, 24)
    for a in axs[:2]:
        a.set_xlabel('Real financing rate (%)')
        a.axhline(0, lw=0.5, color=S.INK)
        a.set_xlim(-0.5, 10.6); a.set_xticks([0, 2, 4, 6, 8, 10])

    f = read('finance_extension_current')
    for g in G:
        z = f[f.group.eq(g)].sort_values('rate')
        sh = z.loss_ge_M.fillna(np.nan) / z.n * 100
        axs[2].fill_between(z.rate * 100, 0, sh, color=S.COLOR[g], alpha=0.10, lw=0, zorder=1)
        axs[2].plot(z.rate * 100, sh, color=S.COLOR[g], marker=S.MARKER[g], ms=2.8, mec='white', mew=0.2, lw=1.35, label=S.LABEL[g], zorder=3)
    axs[2].axvline(3.6, color=CORAL, lw=0.8, ls='--', zorder=2)
    axs[2].text(3.9, 12.0, '3.6%', fontsize=5.2, fontweight='bold', color=CORAL, va='bottom', ha='left')
    axs[2].set_xlabel('Real financing rate (%)')
    axs[2].set_ylabel('Records with 12-day loss ≥ M (%)')
    axs[2].set_xlim(-0.5, 10.6); axs[2].set_xticks([0, 2, 4, 6, 8, 10]); axs[2].set_ylim(-3, 106)

    hp = read('historical_pairs_current')
    ab = np.abs(hp.delta_B.to_numpy())
    axd = axs[3]
    axd.grid(False)
    kde = gaussian_kde(ab)
    xs = np.linspace(0, min(14.5, ab.max()), 200)
    dens = kde(xs)
    dens = dens / dens.max() * 0.42
    axd.fill_between(xs[xs <= M], 0.08, 0.08 + dens[xs <= M], fc=MINT, alpha=0.40, lw=0, zorder=2)
    axd.fill_between(xs[xs >= M], 0.08, 0.08 + dens[xs >= M], fc=CORAL, alpha=0.45, lw=0, zorder=2.2)
    axd.plot(xs, 0.08 + dens, color=TEAL, lw=0.75, zorder=2.5)
    rng = np.random.default_rng(7)
    axd.scatter(ab, -0.18 + rng.uniform(-0.10, 0.10, len(ab)), s=2.2, c='#6D949E', alpha=0.14, lw=0, zorder=1)
    q05, q25, q50, q75, q95 = np.quantile(ab, [0.05, 0.25, 0.50, 0.75, 0.95])
    axd.plot([q05, q95], [0.0, 0.0], color=S.INK, lw=0.95, zorder=3)
    axd.add_patch(Rectangle((q25, -0.07), q75 - q25, 0.14, fc='white', ec=S.INK, lw=0.9, zorder=4))
    axd.plot([q50, q50], [-0.07, 0.07], color=CORAL, lw=1.45, zorder=5)
    axd.axvline(M, color=CORAL, lw=1.0, ls='--', zorder=5)
    frac = float((ab >= M).mean() * 100)
    axd.text(M + 0.3, 0.44, f'M = {M:.2f} ({frac:.1f}% ≥ M)', fontsize=5.1, fontweight='bold', color=CORAL, va='top', ha='left')
    axd.text(M + 0.3, 0.57, f'median {q50:.2f}; 5–95% span {q95 - q05:.2f}', fontsize=5.1, color=S.INK, va='bottom', ha='left')
    axd.set_ylim(-0.38, 0.68); axd.set_yticks([])
    axd.set_xlim(0, 14.5); axd.set_xticks([0, 3, 6, 9, 12])
    axd.set_xlabel(f'|Between-year difference in B| ({S.UNIT})')
    legend(cv, group_handles(), [S.LABEL[g] for g in G], y=0.004, ncol=3)

    export(
        cv,
        'supp_figure7',
        '图 S7 | 固定成本学习、资金成本与时序效应的量级',
        f'a，2035 年 8、16 和 24 小时 A-CAES 相对于 4 小时 LFP 的年化固定成本溢价随实际资金成本变化（色带标示 8—24 小时跨度，圆圈标示基准情景 3.6%、24 小时，{p36.premium2035:.2f} USD kW⁻¹ yr⁻¹）。b，2025—2035 年相对溢价降幅 M（3.6%、24 小时处 M = {M:.2f} USD kW⁻¹ yr⁻¹）。c，12 天事件排列引起的 B 降幅达到 M 的正文记录比例（n = 310、132、10,612；竖向虚线标示 3.6%）。d，q50 基线下 75 个欧洲天气年构成的全部 {len(ab):,} 个年度配对的 |ΔB| 雨云分布（中位数 {q50:.2f}，跨度 {q95 - q05:.2f}），竖向虚线及右侧暖色尾部标示不小于 M = {M:.2f} 的配对（占 {frac:.1f}%）。',
        rows,
    )


def window():
    """Supp Fig. 8: Stacked two-window saturation stream (a) + quadrant-shaded cyclic recursion agreement contours (b, c)."""
    cv, axs, rows = layout(1, 3, h=42)
    Q = 2 * 0.9 * 24 / 0.55
    f = np.linspace(0, 1, 201)
    w1 = np.minimum(21.6, 0.55 * f * Q)
    w2 = np.minimum(21.6, 0.55 * (1 - f) * Q)
    tot_c = w1 + w2
    y_lfp = np.minimum(3.6, 0.9 * f * Q) + np.minimum(3.6, 0.9 * (1 - f) * Q)

    axs[0].fill_between(f, 0, w1, fc=TEAL, alpha=0.88, lw=0, label='A-CAES Win. 1', zorder=2)
    axs[0].fill_between(f, w1, tot_c, fc=MINT, alpha=0.85, lw=0, label='A-CAES Win. 2', zorder=2.2)
    axs[0].fill_between(f, tot_c, 43.2, fc='#F6E2D3', alpha=0.85, lw=0, label='Spill', zorder=1.5)
    axs[0].plot(f, tot_c, color='#0D3E44', lw=1.35, zorder=3.5)
    axs[0].plot(f, y_lfp, color='white', lw=1.9, zorder=4.0)
    axs[0].plot(f, y_lfp, color='#2B3A42', lw=1.15, ls='--', label='4 h LFP', zorder=4.2)
    axs[0].set_xlim(0, 1); axs[0].set_ylim(0, 46)
    axs[0].set_xlabel('First-window share (f)')
    axs[0].set_ylabel('Two-event delivery (MWh MW⁻¹)')
    axs[0].legend(loc='lower left', bbox_to_anchor=(-0.05, 1.02), fontsize=5.0, ncol=2, handlelength=1.2, columnspacing=0.6)

    d = read('window_recursion_effects_current')
    rhos_eq, sm_counts = {}, {}
    rng = np.random.default_rng(2026)
    for i, target in enumerate([4, 12]):
        a = axs[i + 1]
        z = d[d.target.eq(target)]
        req, _, _ = D.equal_regime_spearman(z, 'I_E', 'delta_B')
        rhos_eq[target] = req
        sm_counts[target] = int(z.sign_match.sum())

        xlim_w = (min(float(z.I_E.min()), -50.0) - 20.0, max(float(z.I_E.max()), 5.0) + 20.0)
        ylim_w = (min(float(z.delta_B.min()), -5.0) - 3.0, max(float(z.delta_B.max()), 2.0) + 3.0)
        # Shade concordant quadrants (Q1 and Q3)
        a.add_patch(Rectangle((0, 0), xlim_w[1], ylim_w[1], fc=S.WIN_PALE, alpha=0.6, lw=0, zorder=0))
        a.add_patch(Rectangle((xlim_w[0], ylim_w[0]), -xlim_w[0], -ylim_w[0], fc=S.WIN_PALE, alpha=0.6, lw=0, zorder=0))

        jp = z[z.group.eq('community')]
        jx, jy = jp.I_E.to_numpy(float), jp.delta_B.to_numpy(float)
        idx_s = rng.choice(len(jx), size=min(len(jx), 3500), replace=False)
        sx, sy = xlim_w[1] - xlim_w[0], ylim_w[1] - ylim_w[0]
        kde2 = gaussian_kde(np.vstack([(jx[idx_s] - xlim_w[0]) / sx, (jy[idx_s] - ylim_w[0]) / sy]), bw_method=0.22)
        gx, gy = np.linspace(0.01, 0.99, 60), np.linspace(0.01, 0.99, 60)
        GX, GY = np.meshgrid(gx, gy)
        Z = kde2(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
        vals = np.sort(Z.ravel())[::-1]
        csum = np.cumsum(vals)
        t88 = vals[min(len(vals) - 1, np.searchsorted(csum, 0.88 * csum[-1]))]
        t50 = vals[min(len(vals) - 1, np.searchsorted(csum, 0.50 * csum[-1]))]
        a.contourf(xlim_w[0] + GX * sx, ylim_w[0] + GY * sy, Z, levels=[t88, Z.max() * 1.05], colors=[S.COLOR['community']], alpha=0.22, zorder=2)
        a.contour(xlim_w[0] + GX * sx, ylim_w[0] + GY * sy, Z, levels=[t88, t50], colors=[S.COLOR['community']], linewidths=[0.5, 0.7], alpha=0.7, zorder=2.4)

        for g in ('provincial', 'national'):
            v = z[z.group.eq(g)]
            a.scatter(v.I_E, v.delta_B, color=S.COLOR[g], marker=S.MARKER[g], s=8.5, edgecolor='white', lw=0.2, alpha=0.80, zorder=3)
        a.scatter([float(np.median(jx))], [float(np.median(jy))], s=17, marker='^', color=S.COLOR['community'], edgecolor='white', lw=0.45, zorder=4.2)
        a.set_xlim(*xlim_w); a.set_ylim(*ylim_w)
        a.set_xlabel('Diagnostic delivery Δ (MWh MW⁻¹)')
        a.set_ylabel(f'Change in B ({S.UNIT})')
        a.axhline(0, color=S.INK, lw=0.55); a.axvline(0, color=S.INK, lw=0.55)
        a.text(0.5, 1.04, f'{target}-day ext.: ρ = {req:.2f} ({100 * sm_counts[target] / len(z):.1f}% sign match)',
               transform=a.transAxes, fontsize=5.05, fontweight='bold', ha='center')
    legend(cv, group_handles(), [S.LABEL[g] for g in G])
    export(
        cv,
        'supp_figure8',
        '图 S8 | 两窗口解析示例与逐记录窗口递推',
        f'a，固定总充电电量在两次调用前窗口间分配时的连续充入电量与溢出流谱（首窗口份额 f），虚线为 4 小时 LFP。b、c，分别将事件延长为 4 天和 12 天后，循环窗口递推预测的 A-CAES 相对 LFP 交付变化与完整优化 B 变化的象限一致性与核密度分布（每个面板 n = 11,054，等权跨系统 Spearman 系数分别为 {rhos_eq[4]:.2f} 和 {rhos_eq[12]:.2f}，符号一致记录分别为 {sm_counts[4]:,}/11,054 与 {sm_counts[12]:,}/11,054）。',
        rows,
    )


def vre():
    """Supp Fig. 9: Raincloud + quantile fan (a) + 15-country x 5-scenario mean-B matrix & win-share header (b)."""
    d = read('national_vre_current')
    ar = read('annual_records_current')[['uid', 'system_id']]
    dm = d.merge(ar, on='uid', how='inner')
    order = ['observed', '50%', '75%', '100%', '2030 plan']

    cv = S.Canvas(74.0)
    cv.letter(3.0, 5.0, 'a')
    axa = cv.ax(18.5, 11.5, 74.0, 46.0, 'a')
    S.grid(axa, 'y')
    p = d.pivot(index='uid', columns='variant', values='B').reindex(columns=order)
    xs = np.arange(len(order))
    axa.axhspan(0, 76, facecolor=S.WIN_PALE, zorder=0)
    for v in p.to_numpy():
        axa.plot(xs, v, color='#B2C6CD', lw=0.40, alpha=0.32, zorder=1)
    q10, q25, q50, q75, q90 = [p.quantile(qv).to_numpy() for qv in (0.10, 0.25, 0.50, 0.75, 0.90)]
    axa.fill_between(xs, q10, q90, color=MINT, alpha=0.22, lw=0, label='10th–90th', zorder=1.8)
    axa.fill_between(xs, q25, q75, color=TEAL, alpha=0.28, lw=0, label='25th–75th', zorder=2.2)
    axa.plot(xs, q50, color=TEAL, lw=1.65, marker='o', ms=3.4, mec='white', mew=0.3, label='Median', zorder=4)
    axa.axhline(0, color=S.INK, lw=0.6)
    axa.set_xticks(xs, order, rotation=18, ha='right')
    axa.set_ylabel(f'B ({S.UNIT})'); axa.set_xlabel('Wind–solar generation scenario')
    axa.legend(loc='lower left', bbox_to_anchor=(0.0, 1.02), ncol=3, fontsize=5.05, handletextpad=0.35, columnspacing=0.8)

    # b: 15-country x 5-scenario mean-B diverging matrix + win-share summary row at top
    cv.letter(99.5, 5.0, 'b')
    axb = cv.ax(112.0, 11.5, 58.0, 46.0, 'b')
    ctry_mat = dm.pivot_table(index='system_id', columns='variant', values='B', aggfunc='mean').reindex(columns=order)
    ctry_mat = ctry_mat.sort_values('100%', ascending=False)
    im = axb.imshow(ctry_mat.to_numpy(), aspect='auto', cmap=S.DIV_B_CMAP, vmin=-35, vmax=35)
    for r_i in np.arange(-0.5, len(ctry_mat), 1.0):
        axb.axhline(r_i, color='white', lw=0.35)
    for c_i in np.arange(-0.5, len(order), 1.0):
        axb.axvline(c_i, color='white', lw=0.35)
    shares = ((p >= 0).sum() / 132 * 100).to_numpy()
    for xi, val in enumerate(shares):
        axb.text(xi, -0.85, f'{val:.1f}%', ha='center', va='bottom', fontsize=5.0, fontweight='bold', color=TEAL, clip_on=False)
    axb.text(-0.6, -0.85, 'Win:', ha='right', va='bottom', fontsize=5.0, color=S.MUTED, clip_on=False)
    axb.set_xticks(xs, order, rotation=18, ha='right')
    axb.set_yticks(range(len(ctry_mat)), ctry_mat.index.tolist(), fontsize=5.0)
    axb.tick_params(axis='both', length=0)
    axb.set_xlabel('Wind–solar generation scenario')
    cb_ax = cv.ax(172.0, 14.0, 2.0, 41.0, 'cb_b')
    cb = cv.fig.colorbar(im, cax=cb_ax, ticks=[-35, 0, 35])
    cb.set_label('Country mean B', fontsize=5.0); cb.ax.tick_params(labelsize=5.0, length=1.6, pad=1.2)

    export(
        cv,
        'supp_figure9',
        '图 S9 | 欧洲国家风光电量情景的敏感性',
        'a，132 份欧洲国家年度记录在五种风光电量情景下的成本差 B 分位扇形带（第 10—90 与第 25—75 百分位）及逐记录轨迹。b，15 个国家在五种情景下的国别平均 B 发散色阶矩阵，顶端百分比给出各情景下 B ≥ 0 的记录比例（0.0%、5.3%、6.8%、18.2% 与 11.4%）。observed 为观测风光发电量，50%、75% 和 100% 为相对年度负荷缩放情景，2030 plan 按国家 2030 年装机比例缩放。',
        rows=[['a', 'b']],
        exempt=[
            dict(panels=['a', 'b'], checks=['panel-width'], reason='Fan chart paired beside 15-country heatmap matrix'),
            dict(panels=['cb_b'], checks=['all'], reason='Colorbar'),
        ],
    )


def satisfaction_nines():
    """Supp Fig. 10: Rebuilt natively in Python (zero NVSB/PLRP/CLDB legacy codes, zero '->' arrows, China -> Europe -> Japan order)."""
    sd = read('reliability_extended_data_figure8_source_data')
    reg_map = {
        'PLRP': ('provincial', 'Chinese provinces (22 prov.)'),
        'NVSB': ('national', 'European countries (10 ctry, n=71)'),
        'CLDB': ('community', 'Japanese communities (n=9)'),
    }
    code_order = ['PLRP', 'NVSB', 'CLDB']
    levels = ['90%', '99%', '99.9%', '99.99%', '100%']
    steps = ['90-99%', '99-99.9%', '99.9-99.99%']
    step_labs = ['90–99%', '99–99.9%', '99.9–99.99%']

    cv = S.Canvas(168.0)

    # Top legend strip framed across full outer span 5.5 .. 180.0
    from matplotlib.patches import FancyBboxPatch
    lg = cv.ax(5.5, 1.5, 174.5, 4.3, 'leg_s10')
    lg.set_xlim(0, 174.5); lg.set_ylim(0, 1); lg.axis('off')
    lg.add_patch(FancyBboxPatch((0.2, 0.08), 174.1, 0.84, boxstyle='round,pad=0.0,rounding_size=0.35',
                                facecolor='#F3F7F9', edgecolor='#C5D4DB', lw=0.50, zorder=1, clip_on=False))
    lg.plot([3.5, 9.5], [0.5, 0.5], color=S.INK, lw=1.35, zorder=3); lg.text(10.8, 0.5, 'LFP + A-CAES', va='center', fontsize=5.15, fontweight='bold', zorder=3)
    lg.plot([46, 52], [0.5, 0.5], color=S.INK, lw=1.1, ls='--', zorder=3); lg.text(53.3, 0.5, 'LFP only', va='center', fontsize=5.15, zorder=3)
    lg.scatter([86], [0.5], s=12, color=S.INK, zorder=3); lg.text(88.2, 0.5, '2025 costs', va='center', fontsize=5.15, zorder=3)
    lg.scatter([124], [0.5], s=12, fc='white', ec=S.INK, lw=0.75, zorder=3); lg.text(126.2, 0.5, '2035 costs', va='center', fontsize=5.15, zorder=3)

    # Row 1: a, b, c (China -> Europe -> Japan shaded 2025-2035 cost-vs-satisfaction ribbons)
    xs_r1 = [18.5, 76.5, 134.5]
    w_r1, h_r1, top1 = 45.5, 35.0, 14.5
    src_panels = {'PLRP': 'b', 'NVSB': 'a', 'CLDB': 'c'}
    x_lv = np.arange(len(levels))

    for i, rcode in enumerate(code_order):
        g_key, g_title = reg_map[rcode]
        let = 'abc'[i]
        cv.letter(xs_r1[i] - 13.0, top1 - 4.5, let)
        ax = cv.ax(xs_r1[i], top1, w_r1, h_r1, let)
        S.grid(ax, 'y')
        sub = sd[sd.panel.eq(src_panels[rcode])]
        col = S.COLOR[g_key]
        for port, ls_p, alpha_f in [('lfp_only', '--', 0.12), ('hybrid', '-', 0.24)]:
            s25 = sub[(sub.portfolio == port) & (sub.cost_year == 2025.0)].set_index('level').reindex(levels).unit_cost_usd_per_kwh.to_numpy(float)
            s35 = sub[(sub.portfolio == port) & (sub.cost_year == 2035.0)].set_index('level').reindex(levels).unit_cost_usd_per_kwh.to_numpy(float)
            ax.fill_between(x_lv, s35, s25, color=col, alpha=alpha_f, lw=0, zorder=1.5)
            ax.plot(x_lv, s25, color=col, lw=1.15, ls=ls_p, marker='o', ms=2.6, zorder=3)
            ax.plot(x_lv, s35, color=col, lw=1.35, ls=ls_p, marker='o', ms=2.8, mfc='white', mec=col, mew=0.75, zorder=4)
        ax.set_xticks(x_lv, ['90', '99', '99.9', '99.99', '100'], fontsize=5.0)
        ax.set_xlabel('Task satisfaction (%)')
        if i == 0:
            ax.set_ylabel('Storage cost (USD kWh⁻¹)')
        ax.text(0.5, 1.03, g_title, transform=ax.transAxes, ha='center', va='bottom', fontsize=5.0, fontweight='bold', color=col)

    # Row 2: d (log-scale step cost increments), e (2025-2035 range dumbbells of marginal cost per added kWh), f (shaded ceiling ECDF)
    top2, h_r2 = 66.5, 35.0
    for idx_de, (let, p_code, ylab, is_log) in enumerate([
        ('d', 'd', 'Step cost Δ (USD kWh⁻¹)', True),
        ('e', 'e', 'Marginal cost (USD kWh⁻¹)', False),
    ]):
        cv.letter(xs_r1[idx_de] - 13.0, top2 - 4.2, let)
        ax = cv.ax(xs_r1[idx_de], top2, w_r1, h_r2, let)
        S.grid(ax, 'y')
        sub = sd[sd.panel.eq(p_code)]
        for si, st in enumerate(steps):
            for gi, rcode in enumerate(code_order):
                g_key, _ = reg_map[rcode]
                col = S.COLOR[g_key]
                xx = si + (gi - 1) * 0.22
                v25 = float(sub[(sub.regime == rcode) & (sub.step == st) & (sub.cost_year == 2025.0)].value_usd_per_kwh.iloc[0])
                v35 = float(sub[(sub.regime == rcode) & (sub.step == st) & (sub.cost_year == 2035.0)].value_usd_per_kwh.iloc[0])
                ax.plot([xx, xx], [v35, v25], color=col, lw=1.4, zorder=2.5)
                ax.scatter([xx], [v25], s=11, marker=S.MARKER[g_key], color=col, zorder=3.5)
                ax.scatter([xx], [v35], s=11, marker=S.MARKER[g_key], fc='white', ec=col, lw=0.75, zorder=4)
        if is_log:
            ax.set_yscale('log'); ax.set_ylim(1e-3, 1.0)
            ax.set_yticks([1e-3, 1e-2, 1e-1, 1.0])
            ax.set_yticklabels(['0.001', '0.01', '0.1', '1'], fontsize=5.0)
            ax.minorticks_off()
        else:
            ax.set_ylim(0, 6.2)
        ax.set_xticks([0, 1, 2], step_labs, fontsize=5.0)
        ax.set_xlabel('Satisfaction step')
        ax.set_ylabel(ylab)

    cv.letter(xs_r1[2] - 13.0, top2 - 4.2, 'f')
    axf = cv.ax(xs_r1[2], top2, w_r1, h_r2, 'f')
    S.grid(axf, 'both')
    sub_f = sd[sd.panel.eq('f')]
    axf.axvline(90, color=S.MUTED, lw=0.65, ls=':', zorder=2)
    for rcode in code_order:
        g_key, _ = reg_map[rcode]
        vals = np.sort(sub_f[sub_f.regime == rcode].ceiling_percent.to_numpy(float))
        y_cdf = np.arange(1, len(vals) + 1) / len(vals) * 100.0
        xv = np.r_[0.0, vals]
        yv = np.r_[0.0, y_cdf]
        axf.fill_between(xv, 0, yv, step='post', color=S.COLOR[g_key], alpha=0.08, lw=0)
        axf.step(xv, yv, where='post', color=S.COLOR[g_key], lw=1.35, zorder=3.5)
    axf.set_xlim(0, 102); axf.set_ylim(0, 104)
    axf.set_xticks([0, 50, 90, 100]); axf.set_yticks([0, 50, 100])
    axf.set_xticklabels(['0', '50', '90', '100'])
    axf.get_xticklabels()[-1].set_ha('right')
    axf.set_xlabel('Max. satisfaction ceiling (%)')
    axf.set_ylabel('Cumulative records (%)')

    # Row 3: g (Half-violin + block-mean 4-day extension shift) & h (Paired Annual vs Every-event slope comparison)
    top3, h_r3 = 119.5, 35.0
    cv.letter(5.5, top3 - 4.2, 'g')
    axg = cv.ax(18.5, top3, 74.0, h_r3, 'g')
    S.grid(axg, 'y')
    axg.axhline(0, color=S.INK, lw=0.6, ls=':')
    sub_g = sd[sd.panel.eq('g')]
    rng = np.random.default_rng(10)
    for gi, rcode in enumerate(code_order):
        g_key, _ = reg_map[rcode]
        col = S.COLOR[g_key]
        for li, (lv, dx, al) in enumerate([('90%', -0.18, 0.85), ('99.99%', 0.18, 0.45)]):
            vv = sub_g[(sub_g.regime == rcode) & (sub_g.level == lv)].persistent_minus_intermittent_usd_per_kwh.to_numpy(float)
            xc = gi + dx
            axg.scatter(xc + rng.uniform(-0.08, 0.08, len(vv)), vv, s=6, color=col, alpha=al, lw=0, zorder=3)
            if len(vv) >= 6 and np.std(vv) > 0.005:
                ys_g = np.linspace(vv.min(), vv.max(), 60)
                kd = gaussian_kde(vv)(ys_g)
                kd = kd / kd.max() * 0.13
                axg.fill_betweenx(ys_g, xc, xc + (kd if dx > 0 else -kd), fc=col, alpha=0.22, lw=0, zorder=2)
            bm = float(np.mean(vv))
            axg.plot([xc - 0.11, xc + 0.11], [bm, bm], color=S.INK, lw=1.35, zorder=5)
    axg.set_xticks([0, 1, 2], ['Chinese provinces', 'European countries', 'Japanese communities'], fontsize=5.1)
    axg.set_xlim(-0.55, 2.55); axg.set_ylim(-0.18, 0.25)
    axg.set_ylabel('4-day ext. − 1-day disp.\n(USD kWh⁻¹)')
    axg.text(0.5, 1.03, 'Left: 90%; right: 99.99% (bars: block mean)', transform=axg.transAxes,
             ha='center', va='bottom', fontsize=5.0, color=S.MUTED)

    cv.letter(96.5, top3 - 4.2, 'h')
    axh = cv.ax(108.5, top3, 71.5, h_r3, 'h')
    S.grid(axh, 'y')
    sub_h = sd[sd.panel.eq('h')]
    for gi, rcode in enumerate(code_order):
        g_key, _ = reg_map[rcode]
        col = S.COLOR[g_key]
        for yi_c, (cy, dx, mfc_c) in enumerate([(2025.0, -0.16, col), (2035.0, 0.16, 'white')]):
            ve = float(sub_h[(sub_h.regime == rcode) & (sub_h.cost_year == cy) & (sub_h.definition == 'energy')].increase_90_to_9999_usd_per_kwh.iloc[0])
            vv = float(sub_h[(sub_h.regime == rcode) & (sub_h.cost_year == cy) & (sub_h.definition == 'event')].increase_90_to_9999_usd_per_kwh.iloc[0])
            xc = gi + dx
            axh.plot([xc, xc], [vv, ve], color=col, lw=1.4, zorder=2.5)
            axh.scatter([xc], [ve], s=14, marker='o', fc=mfc_c, ec=col, lw=0.8, zorder=4)
            axh.scatter([xc], [vv], s=15, marker='D', fc=mfc_c, ec=col, lw=0.8, zorder=4)
    axh.set_xticks([0, 1, 2], ['China', 'Europe', 'Japan'], fontsize=5.2)
    axh.set_xlim(-0.55, 2.55); axh.set_ylim(0, 0.44)
    axh.set_ylabel('90% to 99.99% cost rise\n(USD kWh⁻¹)')
    axh.text(0.5, 1.03, '● Annual energy   ◆ Every event (left: 2025; right: 2035)',
             transform=axh.transAxes, ha='center', va='bottom', fontsize=5.0, color=S.INK)

    export(
        cv,
        'supp_figure10',
        '图 S10 | 储能承担的任务满足率与成本',
        'LFP 与 LFP 加 A-CAES 组合的任务满足率容量实验，依次展示中国各省（22 省共同可行集）、欧洲各国（10 国、71 份记录）和日本社区（1 都县、9 份记录）。欧洲和日本采用 q10 任务基线，中国采用日任务全天均分。a—c，真实日期下储能单位任务成本随满足率（90%—100%）的变化及 2025—2035 年降本色带（虚线为纯 LFP，实线为混合组合）。d，相邻满足率节点的单位任务成本增量（对数尺度，竖条连接 2025 与 2035 年）。e，相邻满足率节点每新增 1 kWh 服务电量的边际成本跨度。f，给定盈余及功率上界下最高可达满足率的经验累积分布。g，4 天延长排列相对于 1 天分散排列的单位成本变化（左为 90%、右为 99.99% 满足率，横线为分块均值）。h，满足率从 90% 提高至 99.99% 的成本增量，对比全年总电量约束（圆圈）与逐次事件约束（菱形）。',
        rows=[['a', 'b', 'c'], ['d', 'e', 'f'], ['g', 'h']],
        exempt=[dict(panels=['leg_s10'], checks=['all'], reason='Top legend strip'),
                dict(panels=['g', 'h'], checks=['panel-width'], reason='Two-column bottom row spanning full figure width')],
    )


def fixed_caes():
    """Supp Fig. 11: 3x4 fixed-technology A-CAES duration response with threshold-split positive/negative area shading."""
    d = read('fixed_caes_archived')
    regimes = [
        ('China', 'Chinese provinces', (-104, 24), [-80, -40, 0, 20]),
        ('Europe', 'European countries', (-18, 15), [-15, 0, 10]),
        ('Japan', 'Japanese communities', (-18, 20), [-15, 0, 15]),
    ]
    targets = [2, 4, 8, 12]
    cv = S.Canvas(122.0)
    panel_rows = []
    for ri, ((reg, reg_lab, ylim, yticks), g) in enumerate(zip(regimes, G)):
        prow = []
        y0 = 12.0 + ri * 34.0
        col = S.COLOR[g]
        cv.letter(3.0, y0 - 3.0, 'abc'[ri])
        for ci, tgt in enumerate(targets):
            name = f'r{ri}c{ci}'
            prow.append(name)
            x0 = 21.0 + ci * 40.5
            ax = cv.ax(x0, y0, 35.5, 25.0, name)
            S.grid(ax)
            z = d[(d.regime == reg) & (d.target_days == tgt)].sort_values('duration_h')
            xh = np.linspace(z.duration_h.min(), z.duration_h.max(), 120)
            yh = np.interp(xh, z.duration_h, z.mean_usd_kw_year)
            ax.fill_between(xh, 0, np.maximum(yh, 0), color=S.WIN_MID, alpha=0.26, lw=0, zorder=1.5)
            ax.fill_between(xh, np.minimum(yh, 0), 0, color=CORAL, alpha=0.16, lw=0, zorder=1.5)
            ax.fill_between(z.duration_h, z.ci95_low_usd_kw_year, z.ci95_high_usd_kw_year, color=col, alpha=0.22, lw=0, zorder=2)
            ax.plot(z.duration_h, z.mean_usd_kw_year, color=col, marker=S.MARKER[g], ms=2.4, lw=1.25, zorder=3.5)
            ax.axhline(0, color=S.INK, lw=0.60, ls='--', zorder=2.5)
            ax.set_xlim(18, 174); ax.set_xticks([24, 72, 120, 168])
            ax.set_ylim(*ylim); ax.set_yticks(yticks)
            if ci > 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(f'{reg_lab}\nΔB ({S.UNIT})')
            if ri == 0:
                ax.text(0.5, 1.06, f'{tgt}-day events', transform=ax.transAxes, ha='center', va='bottom', fontsize=5.7, fontweight='bold', color=S.INK)
            if ri < 2:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('A-CAES duration (h)')
        panel_rows.append(prow)

    export(
        cv,
        'supp_figure11',
        '图 S11 | 固定 A-CAES 时长下的事件延长响应',
        'a—c，依次为中国省级、欧洲国家和日本社区中，四列分别为 2、4、8 和 12 天延长事件相对于同容量 1 天分散排列的系统成本优势变化均值（实线与标记）、95% 置信区间（阴影带）及零线正负分界填色（青绿为 ΔB > 0，暖红为 ΔB < 0）；同行四列共用纵轴尺度。各容量节点固定 A-CAES 效率、功率和 24 小时基准固定成本，以隔离增加容量的运行效果。',
        rows=panel_rows,
        cols=[[f'r{r}c{c}' for r in range(3)] for c in range(4)],
    )


def numerical():
    """Supp Fig. 12: Shaded machine-precision ECDF & log residual distributions across 641,132 LPs and 564 capacity designs."""
    cv, axs, rows = layout(2, 2, h=37, gapy=25, x0=24, gapx=24)

    cc = read('chronology_conservation_current')
    v_cc = cc.relative_error.to_numpy()
    n_cc = len(v_cc)
    nz_cc = int((v_cc > 0).sum())
    z_cc = int((v_cc == 0).sum())
    clamped_cc = np.sort(np.maximum(v_cc, 1e-17))
    ecdf_cc = np.arange(1, n_cc + 1) / n_cc
    idx_cc = np.unique(np.r_[0, np.linspace(0, n_cc - 1, 1000).astype(int), np.flatnonzero(v_cc > 0)[-min(nz_cc, 200):] if nz_cc else [], n_cc - 1])
    axa = axs[0]
    axa.grid(False)
    axa.axvspan(3e-18, 1e-8, color=S.WIN_PALE, zorder=0)
    axa.fill_between(clamped_cc[idx_cc], 0, ecdf_cc[idx_cc], step='post', color=TEAL, alpha=0.18, lw=0, zorder=2)
    axa.step(clamped_cc[idx_cc], ecdf_cc[idx_cc], where='post', color=TEAL, lw=1.35, zorder=3)
    axa.axvline(1e-8, color=CORAL, lw=0.75, ls='--', zorder=2.5)
    axa.set_xscale('log'); axa.set_xlim(3e-18, 1e-6)
    axa.set_xticks([1e-17, 1e-14, 1e-11, 1e-8, 1e-6])
    axa.set_xticklabels(['<=1e-17', '1e-14', '1e-11', '1e-8', '1e-6'])
    axa.set_ylim(0, 1.05); axa.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axa.set_xlabel('Relative energy-conservation error')
    axa.set_ylabel(f'Cumulative fraction (n = {n_cc:,})')
    axa.text(0.79, 0.22, f'{z_cc:,}/{n_cc:,} exact zero\nmax = {v_cc.max():.2e}',
             transform=axa.transAxes, ha='right', va='top', fontsize=5.05, color=S.INK)

    cap = read('capacity_design_summary_current')
    rng = np.random.default_rng(12)

    def draw_h_res(ax, specs, xlim, xticks, xlabel, extra_markers=None):
        ax.grid(False); S.grid(ax, 'x')
        ax.axvspan(xlim[0], 1e-6, color='#F2F7F7', zorder=0)
        ax.axvline(1e-8, color='#9AABAF', lw=0.65, ls=':', zorder=1)
        ax.axvline(1e-6, color=CORAL, lw=0.75, ls='--', zorder=1)
        yticks_pos = np.arange(len(specs))
        ylabels = []
        for yi, (col, name, c) in enumerate(specs):
            vals = np.abs(cap[col].dropna().to_numpy())
            nzero = int((vals == 0).sum())
            nz = vals[vals > 0]
            show = np.maximum(vals, 1e-17)
            ax.scatter(show, yi + rng.uniform(-0.18, 0.18, len(show)), s=2.4, c=c, alpha=0.25, lw=0, zorder=2)
            if len(nz):
                q25, q50, q75 = np.quantile(show, [0.25, 0.50, 0.75])
                lo, hi = float(show.min()), float(show.max())
                ax.plot([lo, hi], [yi, yi], color=S.INK, lw=0.8, zorder=3)
                ax.add_patch(Rectangle((q25, yi - 0.16), max(q75 - q25, q25 * 0.08), 0.32, fc='white', ec=S.INK, lw=0.8, zorder=4))
                ax.plot([q50, q50], [yi - 0.16, yi + 0.16], color=CORAL, lw=1.35, zorder=5)
            ylabels.append(f'{name}\n({nzero} zero)' if nzero > 0 else f'{name}\n(max {vals.max():.1e})')
        if extra_markers:
            for yi, val, lab, mk, col in extra_markers:
                ax.scatter([val], [yi], s=26, marker=mk, fc=col, ec='white', lw=0.5, zorder=6, label=lab)
        ax.set_xscale('log'); ax.set_xlim(*xlim); ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f'1e{int(round(np.log10(v)))}'))
        ax.minorticks_off()
        ax.set_yticks(yticks_pos, ylabels, fontsize=5.05); ax.tick_params(axis='y', length=0)
        ax.set_ylim(-0.6, len(specs) - 0.4)
        ax.set_xlabel(xlabel)

    b_specs = [
        ('maximum_shared_recharge_violation', 'Shared recharge', MINT),
        ('maximum_task_balance_residual', 'Task balance', '#4D8E95'),
        ('maximum_state_balance_residual_mwh', 'State balance', TEAL),
        ('maximum_equality_residual', 'Equality residual', '#276F9E'),
        ('maximum_inequality_residual' if 'maximum_inequality_residual' in cap else 'maximum_inequality_violation', 'Inequality viol.', AMBER),
    ]
    draw_h_res(axs[1], b_specs, (3e-18, 3e-5), [1e-16, 1e-12, 1e-8, 1e-6], 'Normalized residual (564 designs)')

    cat = json.loads((V / 'revision/hydrogen_tank_only_20261114/qa/catalogue.json').read_text())
    c_specs = [
        ('maximum_stationarity_residual', 'KKT stationarity', TEAL),
        ('primal_dual_objective_gap_eur', 'Primal–dual gap', '#276F9E'),
        ('objective_identity_error_eur_per_normalized_mw', 'Objective identity', AMBER),
    ]
    extra_c = [
        (0, cat['max_stationarity_residual'], f"Tank H2 LP max (n = {cat['hydrogen_lp_solves']:,})", 'D', CORAL),
        (1, cat['max_duality_gap'], '_nolegend_', 'D', CORAL),
    ]
    draw_h_res(axs[2], c_specs, (3e-18, 3e-5), [1e-16, 1e-12, 1e-8, 1e-6], 'Optimization residual magnitude', extra_markers=extra_c)
    axs[2].legend(loc='lower left', bbox_to_anchor=(0.0, 1.02), fontsize=5.05, handletextpad=0.3)

    wr = read('window_recursion_current')
    max_cyc = float(wr[['cyclic_caes', 'cyclic_lfp']].abs().to_numpy().max())
    d_caes = np.sort(np.abs(wr.actual_caes - wr.diagnostic_caes).to_numpy())
    d_lfp = np.sort(np.abs(wr.actual_lfp - wr.diagnostic_lfp).to_numpy())
    n_wr = len(d_caes)
    idx_wr = np.unique(np.r_[0, np.linspace(0, n_wr - 1, 1000).astype(int), n_wr - 1])
    axd = axs[3]
    axd.grid(False)
    ec_w = (np.arange(1, n_wr + 1) / n_wr)[idx_wr]
    axd.fill_betweenx(ec_w, d_lfp[idx_wr], d_caes[idx_wr], color='#DCECEE', alpha=0.6, zorder=1)
    axd.step(d_caes[idx_wr], ec_w, where='post', color=TEAL, lw=1.35, label=f'24 h A-CAES (med {np.median(d_caes):.2f})')
    axd.step(d_lfp[idx_wr], ec_w, where='post', color=AMBER, lw=1.35, ls='--', label=f'4 h LFP (med {np.median(d_lfp):.2f})')
    axd.set_xlim(-1.0, 54.0); axd.set_ylim(0, 1.05); axd.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axd.set_xlabel('|Full LP − single-pass recursion| (MWh MW⁻¹)')
    axd.set_ylabel(f'Cumulative fraction (n = {n_wr:,})')
    axd.legend(loc='lower right', fontsize=5.05, handlelength=1.5)
    axd.text(0.97, 0.46, f'Cyclic state residual:\n0/{n_wr:,} non-zero (max = {max_cyc:g})\nEU non-H2 drift: 0/4,752',
             transform=axd.transAxes, ha='right', va='top', fontsize=5.0, color=S.INK)

    export(
        cv,
        'supp_figure12',
        '图 S12 | 重排、规划与窗口递推的数值核验',
        f'a，11,054 份正文记录在真实日期及 1、4、12 天排列下，任务与盈余总量的最大相对变化经验累积分布（n = {n_cc:,}；其中 {z_cc:,} 组精确为零，最大值为 {v_cc.max():.2e}）。b，564 个多年度容量规划（含 1 小时参照设计）的物理约束残差水平分布。c，相同 564 个容量规划的 KKT 平稳性、对偶间隙与目标恒等式误差分布，菱形标示目录全部 {cat["hydrogen_lp_solves"]:,} 个 200 bar 地面储罐氢储能线性规划的最大残差（物理残差 ≤ {cat["max_physical_residual"]:.2e}，对偶间隙 ≤ {cat["max_duality_gap"]:.2e}，平稳性 ≤ {cat["max_stationarity_residual"]:.2e}）。d，{n_wr:,} 组逐年递推中单次递推与完整优化的年交付差异累积分布（A-CAES 中位数 {np.median(d_caes):.2f}、LFP 中位数 {np.median(d_lfp):.2f} MWh MW⁻¹，循环状态闭合残差全部精确为零）。',
        rows,
    )


def service():
    """Supp Fig. 13: 4-quadrant valuation-shift shading + HDR contour comparison of equal-delivery constant-power valuation."""
    d = read('equivalent_service_current')
    cv, axs, rows = layout(1, 3, h=43)
    rng = np.random.default_rng(2026)
    rhos_s = {}
    for a, g in zip(axs, G):
        z = d[d.group.eq(g)]
        rhos_s[g] = float(spearmanr(z.B, z.B_equivalent_service).statistic)
        lo = float(min(z.B.min(), z.B_equivalent_service.min())) - 3.0
        hi = float(max(z.B.max(), z.B_equivalent_service.max())) + 3.0
        # Shade Q1 (both positive) in emerald tint and Q2 (B < 0 -> V_eq >= 0 valuation flip) in warm amber-ivory
        a.add_patch(Rectangle((0, 0), hi, hi, fc=S.WIN_PALE, alpha=0.65, lw=0, zorder=0))
        a.add_patch(Rectangle((lo, 0), -lo, hi, fc='#FBF3E4', alpha=0.65, lw=0, zorder=0))
        n_q2 = int(((z.B < 0) & (z.B_equivalent_service >= 0)).sum())
        if len(z) <= 500:
            a.scatter(z.B, z.B_equivalent_service, c=S.COLOR[g], marker=S.MARKER[g], s=10, edgecolor='white', lw=0.2, alpha=0.80, zorder=3)
        else:
            jx, jy = z.B.to_numpy(float), z.B_equivalent_service.to_numpy(float)
            idx_s = rng.choice(len(jx), size=min(len(jx), 3500), replace=False)
            sx, sy = hi - lo, hi - lo
            kde2 = gaussian_kde(np.vstack([(jx[idx_s] - lo) / sx, (jy[idx_s] - lo) / sy]), bw_method=0.20)
            gx, gy = np.linspace(0.01, 0.99, 65), np.linspace(0.01, 0.99, 65)
            GX, GY = np.meshgrid(gx, gy)
            Z = kde2(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
            vals = np.sort(Z.ravel())[::-1]
            csum = np.cumsum(vals)
            t90 = vals[min(len(vals) - 1, np.searchsorted(csum, 0.90 * csum[-1]))]
            t50 = vals[min(len(vals) - 1, np.searchsorted(csum, 0.50 * csum[-1]))]
            a.contourf(lo + GX * sx, lo + GY * sy, Z, levels=[t90, Z.max() * 1.05], colors=[S.COLOR[g]], alpha=0.24, zorder=2.5)
            a.contour(lo + GX * sx, lo + GY * sy, Z, levels=[t90, t50], colors=[S.COLOR[g]], linewidths=[0.5, 0.75], alpha=0.78, zorder=3)
            a.scatter([float(np.median(jx))], [float(np.median(jy))], s=18, marker='^', color=S.COLOR[g], edgecolor='white', lw=0.45, zorder=4.2)
            tail = jx < np.quantile(jx, 0.005)
            if tail.any():
                a.scatter(jx[tail], jy[tail], s=6, marker='^', color=S.COLOR[g], alpha=0.55, lw=0, zorder=3.2)
        a.plot([lo, hi], [lo, hi], color='#8EA0A7', lw=0.7, ls='--', zorder=2)
        a.axhline(0, color=S.INK, lw=0.55); a.axvline(0, color=S.INK, lw=0.55)
        a.set_xlim(lo, hi); a.set_ylim(lo, hi)
        a.set_xlabel(f'System-cost difference B\n({S.UNIT})')
        a.set_ylabel(f'Equal-delivery service value Δ\n({S.UNIT})')
        a.text(0.5, 1.05, f'{S.SHORT_LABEL[g]}: ρ = {rhos_s[g]:.2f} (flips: {n_q2:,})',
               ha='center', transform=a.transAxes, fontsize=5.1, fontweight='bold', color=S.COLOR[g])
    export(
        cv,
        'supp_figure13',
        '图 S13 | 等电量恒定功率服务估值',
        f'a—c，依次为中国各省（n = 310）、欧洲各国（n = 132）和日本社区（n = 10,612）。保持各组合的最优调度、年度交付和储能成本不变，以完成相同年度任务电量的恒定功率 F 对服务计价，替换正文残余峰值削减项（右上青绿区为两种估值均占优，左上暖色区为由 B < 0 转为等电量估值 ≥ 0 的记录区，标题列出对应转换记录数；描述性 Spearman 系数分别为 {rhos_s["provincial"]:.2f}、{rhos_s["national"]:.2f} 和 {rhos_s["community"]:.2f}）。',
        rows,
    )


def main():
    framework()
    thresholds()
    sensitivities()
    weather()
    import demand_validation
    demand_validation.main()
    caps_path = CAPS_FILE
    aud_path = AUD_FILE
    if caps_path.exists():
        caps['extended_data_figure5'] = json.loads(caps_path.read_text())['extended_data_figure5']
    if aud_path.exists():
        audits['extended_data_figure5'] = json.loads(aud_path.read_text()).get('extended_data_figure5', {})
    validation_trend()
    capacity_full()
    nearoptimal()
    finance()
    window()
    vre()
    satisfaction_nines()
    fixed_caes()
    numerical()
    service()

    ordered_keys = [
        'extended_data_figure1', 'extended_data_figure2', 'extended_data_figure3',
        'extended_data_figure4', 'extended_data_figure5', 'extended_data_figure6',
        'supp_figure5', 'supp_figure6', 'supp_figure7', 'supp_figure8',
        'supp_figure9', 'supp_figure10', 'supp_figure11', 'supp_figure12', 'supp_figure13',
    ]
    out_caps = {k: caps[k] for k in ordered_keys if k in caps}
    out_auds = {k: audits[k] for k in ordered_keys if k in audits}
    caps_path.write_text(json.dumps(out_caps, ensure_ascii=False, indent=2))
    aud_path.write_text(json.dumps(out_auds, ensure_ascii=False, indent=2))
    print('Exported', len(out_caps), 'restored figures', flush=True)
    assert not any('blocking' in v for v in out_auds.values()), {
        k: v['blocking'] for k, v in out_auds.items() if 'blocking' in v
    }


if __name__ == '__main__':
    main()
