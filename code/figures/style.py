"""Shared style, editorial palette and export/audit helper for the v86 main, extended-data and supplementary figures.

Conventions follow Nature Energy visual standards (183 mm double-column width, DejaVu Sans, >= 5.0 pt rendered glyph floor,
bold 8 pt lower-case panel letters, minimalist left/bottom spines, descriptive panel titles in captions only).
Every export runs the nature-figure gates: render-time panel alignment, PDF glyph floor (5 pt) and rendered collision audit.
"""
from pathlib import Path
import json
import os
import subprocess
import sys

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
V86 = HERE.parents[1]
JOULE = os.environ.get('JOULE_MODE') == '1'
OUT = V86 / ('figures/output_joule' if JOULE else 'figures/output')
QA = V86 / ('figures/qa_joule' if JOULE else 'figures/qa')
SD = V86 / 'source_data'
JOULE_FONT_DIR = V86 / 'figures/assets/joule_fonts'
NF = Path.home() / '.claude/skills/nature-figure/scripts'
sys.path.insert(0, str(NF))
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402

WIDTH_MM = 183.0
PHYSICAL_WIDTH_MM = 174.0 if JOULE else 183.0
MM = 1 / 25.4
UNIT = 'USD kW⁻¹ yr⁻¹'
DUNIT = 'MWh MW⁻¹ yr⁻¹'

# Editorial ink, muted slate, whisper-fine grid and neutral land tones
INK, MUTED, GRID, LAND = '#1B2A32', '#5C6D75', '#E7EDF0', '#E8EBE8'
GROUPS = ['provincial', 'national', 'community', 'hist', 'us']
MAIN = ['provincial', 'national', 'community']

LABEL = {
    'provincial': 'Chinese provinces',
    'national': 'European countries',
    'community': 'Japanese communities',
    'hist': 'Europe weather years',
    'us': 'US regions',
}
LABEL2 = {
    'provincial': 'Chinese\nprovinces',
    'national': 'European\ncountries',
    'community': 'Japanese\ncommunities',
    'hist': 'Europe\nweather years',
    'us': 'US\nregions',
}
SHORT_LABEL = {
    'provincial': 'China',
    'national': 'Europe',
    'community': 'Japan',
}

# Harmonized, print- and CVD-safe Nature Energy regime palette
COLOR = {
    'provincial': '#C58922',   # burnished amber-ochre
    'national': '#C9543B',     # editorial terracotta-vermilion
    'community': '#276F9E',    # Prussian cerulean blue
    'hist': '#995D8E',
    'us': '#2A8F70',
}
TINT = {
    'provincial': '#F8EEDB',
    'national': '#FAE7E2',
    'community': '#E3EFF8',
}
DEEP = {
    'provincial': '#8A5D0E',
    'national': '#8E3320',
    'community': '#164769',
}
MARKER = {'provincial': 's', 'national': 'o', 'community': '^', 'hist': 'D', 'us': 'v'}
MSIZE = {'provincial': 9, 'national': 9, 'community': 10, 'hist': 7, 'us': 10}

# Semantic outcome palette
WIN, LOSE = '#19646B', '#E5ECEF'
WIN_MID, WIN_PALE = '#368F95', '#ECF6F6'
LOSS_WARM, LOSS_PALE = '#C45A41', '#F8EAE5'

COMP = ['B_fixed', 'B_variable_om', 'B_charging', 'B_residual_energy', 'B_residual_peak']
COMP_LABEL = {
    'B_fixed': 'Fixed cost',
    'B_variable_om': 'Variable O&M',
    'B_charging': 'Charging',
    'B_residual_energy': 'Residual task energy',
    'B_residual_peak': 'Residual peak',
}
COMP_COLOR = {
    'B_fixed': '#8A9BA3',
    'B_variable_om': '#C5AE97',
    'B_charging': '#D99B3E',
    'B_residual_energy': '#369388',
    'B_residual_peak': '#275B88',
}

# Curated colormaps for heatmaps and phase surfaces
SURPLUS_CMAP = LinearSegmentedColormap.from_list(
    'ne_surplus',
    [(0.00, '#F8FAFB'), (0.22, '#D5ECEB'), (0.55, '#4A9E9B'), (0.82, '#1C6B73'), (1.00, '#0D3B43')],
)
PHASE_CMAP = LinearSegmentedColormap.from_list(
    'ne_phase',
    [(0.00, '#F3F6F8'), (0.18, '#D6EAE9'), (0.45, '#59A6A2'), (0.75, '#1F7179'), (1.00, '#0E424A')],
)
DIV_B_CMAP = LinearSegmentedColormap.from_list(
    'ne_div_b',
    [(0.00, '#BF4F38'), (0.25, '#E8B7A3'), (0.50, '#F7F5EE'), (0.75, '#4C9E99'), (1.00, '#0F4D56')],
)


def _register_joule_fonts():
    if JOULE and JOULE_FONT_DIR.exists():
        loaded = {f.fname for f in font_manager.fontManager.ttflist}
        for ttf in sorted(JOULE_FONT_DIR.glob('*.ttf')):
            if str(ttf) not in loaded:
                font_manager.fontManager.addfont(str(ttf))


def setup():
    _register_joule_fonts()
    sans_list = ['Arial', 'Liberation Sans', 'DejaVu Sans'] if JOULE else ['DejaVu Sans']
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': sans_list,
        'font.size': 5.8,
        'axes.labelsize': 5.9,
        'axes.titlesize': 5.9,
        'xtick.labelsize': 5.4,
        'ytick.labelsize': 5.4,
        'legend.fontsize': 5.4,
        'legend.frameon': False,
        'axes.linewidth': 0.55,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.size': 2.1,
        'ytick.major.size': 2.1,
        'xtick.minor.size': 1.1,
        'ytick.minor.size': 1.1,
        'xtick.major.width': 0.52,
        'ytick.major.width': 0.52,
        'xtick.major.pad': 1.5,
        'ytick.major.pad': 1.5,
        'axes.labelpad': 2.0,
        'axes.edgecolor': INK,
        'axes.labelcolor': INK,
        'xtick.color': INK,
        'ytick.color': INK,
        'text.color': INK,
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'hatch.linewidth': 0.42,
        'pdf.fonttype': 42,
        'svg.fonttype': 'none',
        'axes.unicode_minus': True,
    })


class Canvas:
    """Figure laid out in millimetres from the top-left corner."""

    def __init__(self, height_mm):
        self.h = height_mm
        self.phys_w = PHYSICAL_WIDTH_MM
        self.phys_h = round(height_mm * (PHYSICAL_WIDTH_MM / WIDTH_MM), 2)
        self.fig = plt.figure(figsize=(self.phys_w * MM, self.phys_h * MM))

    def ax(self, x, top, w, h, gid, **kw):
        a = self.fig.add_axes([x / WIDTH_MM, 1 - (top + h) / self.h, w / WIDTH_MM, h / self.h], **kw)
        a.set_gid(gid)
        return a

    def letter(self, x, top, s):
        lbl = s.upper() if JOULE else s
        fsize = 8.5 if JOULE else 8.0
        self.fig.text(x / WIDTH_MM, 1 - top / self.h, lbl, fontsize=fsize, fontweight='bold',
                      color=INK, va='top', ha='left')

    def text(self, x, top, s, **kw):
        return self.fig.text(x / WIDTH_MM, 1 - top / self.h, s, **kw)


def luminance(rgba):
    r, g, b = mpl.colors.to_rgb(rgba)
    return 0.299 * r + 0.587 * g + 0.114 * b


def clip_ring(v, box):
    """Sutherland-Hodgman clip of a polygon ring to the view box (hidden geometry stays out of the PDF)."""
    x0, x1, y0, y1 = box
    for inside, cut in (
        (lambda p: p[0] >= x0, lambda a, b: (x0, a[1] + (b[1] - a[1]) * (x0 - a[0]) / (b[0] - a[0]))),
        (lambda p: p[0] <= x1, lambda a, b: (x1, a[1] + (b[1] - a[1]) * (x1 - a[0]) / (b[0] - a[0]))),
        (lambda p: p[1] >= y0, lambda a, b: (a[0] + (b[0] - a[0]) * (y0 - a[1]) / (b[1] - a[1]), y0)),
        (lambda p: p[1] <= y1, lambda a, b: (a[0] + (b[0] - a[0]) * (y1 - a[1]) / (b[1] - a[1]), y1)),
    ):
        out = []
        for i in range(len(v)):
            a, b = v[i - 1], v[i]
            if inside(b):
                if not inside(a):
                    out.append(cut(a, b))
                out.append(b)
            elif inside(a):
                out.append(cut(a, b))
        v = out
        if not v:
            break
    return np.asarray(v, float)


def clip_lines(segs, box):
    """Keep the parts of polylines inside the view box (Liang-Barsky per segment)."""
    x0, x1, y0, y1 = box
    out = []
    for s in segs:
        for a, b in zip(s[:-1], s[1:]):
            dx, dy = b - a
            t0, t1, ok = 0., 1., True
            for p, q in ((-dx, a[0] - x0), (dx, x1 - a[0]), (-dy, a[1] - y0), (dy, y1 - a[1])):
                if p == 0:
                    if q < 0:
                        ok = False
                else:
                    t = q / p
                    if p < 0:
                        t0 = max(t0, t)
                    else:
                        t1 = min(t1, t)
            if ok and t0 < t1:
                out.append(np.array([a + t0 * (b - a), a + t1 * (b - a)]))
    return out


def grid(ax, axis='y'):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color=GRID, lw=0.42)


def regime_legend_strip(cv, x, top, w, gid='regime_leg', extra_items=(), scope_label=None, framed=False):
    """Top legend strip across Chinese provinces -> European countries -> Japanese communities.
    When framed=True, renders a full-width subtle slate banner with crisp border and optional scope_label.
    """
    from matplotlib.patches import FancyBboxPatch
    h_leg = 4.5 if framed else 3.6
    ax = cv.ax(x, top, w, h_leg, gid)
    ax.set_xlim(0, w)
    ax.set_ylim(0, 1)
    ax.axis('off')
    counts = {'provincial': 310, 'national': 132, 'community': 10612}
    if framed:
        ax.add_patch(FancyBboxPatch(
            (0.25, 0.08), w - 0.50, 0.84,
            boxstyle='round,pad=0.0,rounding_size=0.35',
            facecolor='#F3F7F9', edgecolor='#C5D4DB', lw=0.52, zorder=1, clip_on=False,
        ))
    x0 = 2.4 if framed else 0.0
    if scope_label:
        t_obj = ax.text(x0, 0.5, scope_label, va='center', ha='left', fontsize=5.2, fontweight='bold', color='#2E434C', zorder=4)
        cv.fig.canvas.draw()
        bb_ax = ax.transData.inverted().transform(t_obj.get_window_extent(renderer=cv.fig.canvas.get_renderer()))
        sep_x = float(bb_ax[1, 0]) + 2.6
        ax.plot([sep_x, sep_x], [0.22, 0.78], color='#BCCBD2', lw=0.55, zorder=3)
        x0 = sep_x + 2.8

    rem_w = w - x0 - (2.2 if framed else 0.0)
    n_tot = 3 + len(extra_items)
    if n_tot == 3:
        slots = [x0 + 0.0, x0 + 0.315 * rem_w, x0 + 0.635 * rem_w]
    else:
        slots = [x0 + 0.0, x0 + 0.245 * rem_w, x0 + 0.495 * rem_w]

    for idx, g in enumerate(MAIN):
        xx = slots[idx]
        ax.scatter([xx + 1.3], [0.5], s=15, marker=MARKER[g], color=COLOR[g], edgecolor='white', lw=0.30, zorder=4)
        ax.text(xx + 3.3, 0.5, f'{LABEL[g]} (n = {counts[g]:,})',
                va='center', ha='left', fontsize=5.2, fontweight='bold', color=INK, zorder=4)
    x_cur = (x0 + 0.775 * rem_w) if extra_items else (x0 + rem_w)
    for fn in extra_items:
        x_cur = fn(ax, x_cur)
    return ax


def save(cv, name, rows=(), cols=(), exempt=(), audit=None):
    """Alignment gate, exports, glyph-floor and collision audits; raises on any blocking finding."""
    fig = cv.fig
    QA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rep = require_matplotlib_panel_alignment(
        fig, json_out=QA / f'{name}_alignment.json', overlay_svg=QA / f'{name}_alignment.svg',
        row_groups=[list(r) for r in rows] or None, column_groups=[list(c) for c in cols] or None,
        exemptions=list(exempt), strict=True)
    stem = OUT / name
    fig.savefig(stem.with_suffix('.pdf'), dpi=600)
    fig.savefig(stem.with_suffix('.svg'), dpi=600)
    fig.savefig(stem.with_suffix('.png'), dpi=600 if JOULE else 300)
    fig.savefig(stem.with_suffix('.tiff'), dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    pdf = stem.with_suffix('.pdf')
    col = QA / f'{name}_collisions.json'
    subprocess.run([sys.executable, str(NF / 'audit_figure_collisions.py'), str(pdf), '--json-out', str(col),
                    '--overlay-pdf', str(QA / f'{name}_collisions_overlay.pdf')], capture_output=True)
    cj = json.loads(col.read_text())
    tx = json.loads(subprocess.run([sys.executable, str(NF / 'audit_pdf_text.py'), '--min-pt', '5', '--json', str(pdf)],
                                   capture_output=True, text=True).stdout)
    out = dict(audit or {}, size_mm=[cv.phys_w, cv.phys_h], alignment=rep.get('verdict'),
               collision=dict(verdict=cj['verdict'], **cj['summary']),
               glyph=dict(minimum_pt=tx['minimum_found_pt'], below_5pt=tx['below_minimum_count']))
    (QA / f'{name}_audit.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    fails = [f"{f['kind']}: {str(f.get('text', ''))[:50]}" for f in cj['findings'] if f.get('severity') == 'FAIL']
    warns = [f"{f['kind']}: {str(f.get('text', ''))[:50]}" for f in cj['findings'] if f.get('severity') == 'WARN']
    print(name, 'alignment', rep.get('verdict'), '| collisions', cj['verdict'], cj['summary'], '| min glyph', tx['minimum_found_pt'])
    for w in warns:
        print('  WARN', w)
    assert not fails, fails
    assert tx['below_minimum_count'] == 0, tx['minimum_found_pt']
    return out
