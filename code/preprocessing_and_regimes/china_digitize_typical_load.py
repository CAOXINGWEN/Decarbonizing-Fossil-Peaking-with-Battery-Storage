"""Digitise the 2020 typical workday and holiday load curves of the 34 provincial grids published by NDRC and NEA
(attachment to 发改运行〔2020〕1784号, external_downloads/ndrc_typical_load_2020/P020201202546044875868.pdf).

The attachment holds raster charts only. For each grid the first two images on its page are the workday and holiday
curves (ordered by their position on the page; the third is the annual daily max/min chart and is not used). The curve
is the coloured line of each chart (blue in most, red or green in a few): per pixel column the mean row of
coloured pixels, restricted to the widest contiguous run
of columns (drops legend samples). The 24 hourly values are means over 24 equal bins of that run, i.e. the curve is
taken to span 00:00-24:00. The load axis is linear, so the shape (value - daily minimum) needs no axis calibration.
Writes ndrc2020_hourly_shapes.csv (grid, day_type, hour 0-23, rel = (v - min) / (max - min), rank 1 = highest) and an
overlay PNG per chart in qa/ for visual checking.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import fitz
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PDF = ROOT / 'external_downloads/ndrc_typical_load_2020/P020201202546044875868.pdf'
QA = HERE / 'qa'; QA.mkdir(exist_ok=True)


def curve_mask(a):
    """Pixels of the curve: coloured pixels whose hue lies within 18 degrees of the dominant hue of the chart.
    Titles, axes and grid lines are grey or black; JPEG fringes around black text have scattered hues."""
    a = a.astype(float)
    mx, mn = a.max(-1), a.min(-1)
    chroma = mx - mn
    col = (chroma >= 45) & (mx >= 80)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    hue = np.degrees(np.arctan2(np.sqrt(3) * (g - b), 2 * r - g - b)) % 360
    hist, edges = np.histogram(hue[col], bins=72, range=(0, 360))
    k = hist.argmax(); h0 = (edges[k] + edges[k + 1]) / 2
    dh = np.abs((hue - h0 + 180) % 360 - 180)
    return col & (dh <= 18)


def track(m, win=15):
    """Row of the curve per column: the colored run closest to a running-median reference."""
    cols = np.flatnonzero(m.sum(0) > 0)
    first = np.full(m.shape[1], np.nan)
    for x in cols:
        first[x] = np.median(np.flatnonzero(m[:, x]))
    ref = pd.Series(first).rolling(win, center=True, min_periods=3).median().to_numpy()
    out = np.full(m.shape[1], np.nan)
    for x in cols:
        rows = np.flatnonzero(m[:, x])
        runs = np.split(rows, np.flatnonzero(np.diff(rows) > 2) + 1)
        centres = np.array([r.mean() for r in runs])
        target = ref[x] if not np.isnan(ref[x]) else np.median(rows)
        out[x] = centres[np.argmin(np.abs(centres - target))]
    return out


def digitise(img):
    a = np.asarray(img.convert('RGB'))
    m = curve_mask(a)
    cols = np.flatnonzero(m.sum(0) > 0)
    if len(cols) < 20:
        raise ValueError('no coloured curve')
    runs, start = [], cols[0]
    for p, q in zip(cols[:-1], cols[1:]):
        if q - p > 6:
            runs.append((start, p)); start = q
    runs.append((start, cols[-1]))
    x0, x1 = max(runs, key=lambda t: t[1] - t[0])
    xs = np.arange(x0, x1 + 1)
    ys_all = track(m)
    ys = ys_all[x0:x1 + 1]
    ok = ~np.isnan(ys)
    ys = np.interp(xs, xs[ok], ys[ok])
    # drop isolated jumps (stray pixels): points more than 6 px from a wide running median, then re-interpolate
    ref = pd.Series(ys).rolling(31, center=True, min_periods=5).median().to_numpy()
    good = np.abs(ys - ref) <= 6
    ys = np.interp(xs, xs[good], ys[good])
    ys = pd.Series(ys).rolling(5, center=True, min_periods=1).median().to_numpy()
    v = -ys
    edges = np.linspace(0, len(xs), 25)
    hourly = np.array([v[int(edges[h]):max(int(edges[h + 1]), int(edges[h]) + 1)].mean() for h in range(24)])
    return hourly, (x0, x1), xs, ys


def main():
    doc = fitz.open(PDF)
    rows, problems = [], []
    for i, page in enumerate(doc):
        lines = [l.strip() for l in page.get_text().split('\n') if l.strip()]
        name = next(l for l in lines if l[0].isdigit() and '.' in l).split('.', 1)[1].strip()
        imgs = sorted(page.get_images(full=True), key=lambda im: page.get_image_rects(im[0])[0].y0)
        for k, day_type in enumerate(('workday', 'holiday')):
            pix = fitz.Pixmap(doc, imgs[k][0])
            if pix.n - pix.alpha > 3:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            img = Image.frombytes('RGB' if pix.n == 3 else 'RGBA', (pix.width, pix.height), pix.samples).convert('RGB')
            try:
                hourly, (x0, x1), xs, ys = digitise(img)
            except ValueError as e:
                problems.append((name, day_type, str(e))); continue
            rel = (hourly - hourly.min()) / (hourly.max() - hourly.min())
            rank = (-hourly).argsort().argsort() + 1
            for h in range(24):
                rows.append(dict(grid=name, day_type=day_type, hour=h, rel=rel[h], rank=int(rank[h]), raw=hourly[h]))
            ov = img.copy(); dr = ImageDraw.Draw(ov)
            dr.line(list(zip(xs.tolist(), ys.tolist())), fill=(255, 0, 0), width=1)
            for h in range(24):
                xc = x0 + (h + .5) * (x1 - x0) / 24
                yc = -hourly[h]
                dr.ellipse([xc - 3, yc - 3, xc + 3, yc + 3], outline=(0, 160, 0), width=2)
            ov.save(QA / f'{i + 1:02d}_{name}_{day_type}.png')
    d = pd.DataFrame(rows)
    d.to_csv(HERE / 'ndrc2020_hourly_shapes.csv', index=False)
    print(d.grid.nunique(), 'grids', len(d) // 24, 'curves', 'problems:', problems)


if __name__ == '__main__':
    main()
