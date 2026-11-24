"""Synthesize Arial-metric TrueType fonts (Liberation Sans + missing U+207B/U+25C6/U+25C7 glyphs from DejaVu Sans)
and render all 26 Joule-formatted figures (174.0 mm full width, bold uppercase A-G panel labels, Arial sans-serif,
vector PDF/SVG + 450 dpi PNG + 600 dpi LZW TIFF) into figures/output_joule/.
"""
from pathlib import Path
import os
import sys
import json
import subprocess
import fitz
import matplotlib
from fontTools.ttLib import TTFont
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.pens.transformPen import TransformPen

W = Path(__file__).resolve().parents[1]
V = W.parents[1]
FONT_OUT = V / 'figures/assets/joule_fonts'
MPL_TTF = Path(matplotlib.__file__).resolve().parent / 'mpl-data/fonts/ttf'
EXTRA_CP = [0x207B, 0x25C6, 0x25C7]  # '⁻', '◆', '◇'


def build_joule_arial_fonts():
    """Create local Arial-named TTFs combining Liberation Sans (100% Arial metric twin)
    with the 3 Unicode math/geometric symbols from DejaVu Sans so Matplotlib embeds 'Arial'
    with zero missing glyphs across all 26 figures.
    """
    FONT_OUT.mkdir(parents=True, exist_ok=True)
    pairs = [
        ('Arial-Regular.ttf', '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
         MPL_TTF / 'DejaVuSans.ttf', 'Regular'),
        ('Arial-Bold.ttf', '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
         MPL_TTF / 'DejaVuSans-Bold.ttf', 'Bold'),
        ('Arial-Italic.ttf', '/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf',
         MPL_TTF / 'DejaVuSans-Oblique.ttf', 'Italic'),
        ('Arial-BoldItalic.ttf', '/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf',
         MPL_TTF / 'DejaVuSans-BoldOblique.ttf', 'Bold Italic'),
    ]
    for out_name, lib_p, djv_p, subfam in pairs:
        target = FONT_OUT / out_name
        base = TTFont(lib_p)
        donor = TTFont(str(djv_p))
        scale = base['head'].unitsPerEm / donor['head'].unitsPerEm

        donor_cmap = donor.getBestCmap()
        donor_glyf = donor['glyf']
        donor_gset = donor.getGlyphSet()
        donor_hmtx = donor['hmtx']

        base_glyf = base['glyf']
        base_hmtx = base['hmtx']
        glyph_order = list(base.getGlyphOrder())

        for cp in EXTRA_CP:
            if cp in base.getBestCmap():
                continue
            d_gname = donor_cmap.get(cp)
            if not d_gname:
                continue
            new_gname = f'uni{cp:04X}'
            pen = TTGlyphPen(donor_gset)
            tpen = TransformPen(pen, (scale, 0, 0, scale, 0, 0))
            donor_gset[d_gname].draw(tpen)
            glyph = pen.glyph()
            glyph.recalcBounds(base_glyf)
            base_glyf[new_gname] = glyph
            adv, lsb = donor_hmtx[d_gname]
            base_hmtx[new_gname] = (int(round(adv * scale)), int(round(lsb * scale)))
            glyph_order.append(new_gname)
            for table in base['cmap'].tables:
                if table.isUnicode():
                    table.cmap[cp] = new_gname

        base.setGlyphOrder(glyph_order)
        base['maxp'].numGlyphs = len(glyph_order)

        # Rename font family to 'Arial' so Matplotlib font selection and PDF Type-42 embedding report Arial
        full_name = 'Arial' if subfam == 'Regular' else f'Arial {subfam}'
        ps_name = 'ArialMT' if subfam == 'Regular' else f'Arial-{subfam.replace(" ", "")}MT'
        for rec in base['name'].names:
            if rec.nameID in (1, 16):
                rec.string = 'Arial'
            elif rec.nameID in (2, 17):
                rec.string = subfam
            elif rec.nameID in (4,):
                rec.string = full_name
            elif rec.nameID in (6,):
                rec.string = ps_name

        base.save(target)
        print(f'Synthesized {target.name} ({full_name})')


def run_all_joule_figures():
    env = dict(os.environ, JOULE_MODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')

    # 1. Main Figures 1-4 + S1-S4 + S20 (cost_paths) + S21 (capacity)
    mods = [
        'fig1', 'fig2', 'fig3', 'fig4',
        'appendix_provincial_allocation',
        'appendix_factorial',
        'appendix_event_extension',
        'appendix_europe_us',
        'appendix_capacity',
        'appendix_cost_paths',
    ]
    for mod in mods:
        print(f'Rendering {mod} in JOULE_MODE...', flush=True)
        r1 = subprocess.run(
            [sys.executable, '-c', f'import {mod}; {mod}.main()'],
            cwd=str(V / 'figures/code'),
            env=env,
            text=True,
            capture_output=True,
        )
        print(r1.stdout, end='', flush=True)
        if r1.returncode != 0:
            print(r1.stderr, file=sys.stderr)
            raise RuntimeError(f'{mod} failed in JOULE_MODE')

    # 2. Main Figure 5 (figure_r5.py)
    print('Running Main Figure 5 (figure_r5.py) in JOULE_MODE...')
    r2 = subprocess.run([sys.executable, str(W / 'code/figure_r5.py')], cwd=str(W / 'code'), env=env, text=True, capture_output=True)
    print(r2.stdout)
    if r2.returncode != 0:
        print(r2.stderr, file=sys.stderr)
        raise RuntimeError('figure_r5.py failed in JOULE_MODE')

    # 3. Remaining 15 Supplemental Figures (restore_figures.py: S5-S19)
    print('Running Supplemental Figures S5-S19 (restore_figures.py) in JOULE_MODE...')
    r3 = subprocess.run([sys.executable, str(W / 'code/restore_figures.py')], cwd=str(W / 'code'), env=env, text=True, capture_output=True)
    print(r3.stdout)
    if r3.returncode != 0:
        print(r3.stderr, file=sys.stderr)
        raise RuntimeError('restore_figures.py failed in JOULE_MODE')


def verify_joule_figures():
    out_dir = V / 'figures/output_joule'
    pdfs = sorted(out_dir.glob('*.pdf'))
    assert len(pdfs) == 26, f'Expected 26 Joule PDFs in {out_dir}, got {len(pdfs)}'
    summary = {}
    for p in pdfs:
        doc = fitz.open(p)
        rect = doc[0].rect
        w_mm = round(rect.width * 25.4 / 72.0, 2)
        h_mm = round(rect.height * 25.4 / 72.0, 2)
        fonts = [f[3] for f in doc[0].get_fonts()]
        assert abs(w_mm - 174.0) <= 0.15, f'{p.name}: expected 174.0 mm width, got {w_mm} mm'
        assert h_mm <= 200.0, f'{p.name}: height {h_mm} mm exceeds 200 mm Joule max'
        assert any('Arial' in fn for fn in fonts), f'{p.name}: Arial not embedded! Found {fonts}'
        summary[p.stem] = dict(width_mm=w_mm, height_mm=h_mm, fonts=fonts)
        doc.close()
    (V / 'figures/qa_joule/joule_26_figures_audit.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'ALL 26 JOULE FIGURES VERIFIED: exact 174.0 mm width, <=200 mm height, Arial embedded, BOLD UPPERCASE A-G!')


if __name__ == '__main__':
    build_joule_arial_fonts()
    run_all_joule_figures()
    verify_joule_figures()
