"""100% Pure-Vector Cell Press *Joule* / *Nature Energy* Schematic for Figure S14 (ED Fig. 1):
'Analytical framework for peaking tasks, charging windows, and economic evaluation'.

Zero raster bitmaps — every icon, waveform, daily profile block, storage schematic, card, and arrow
is rendered as crisp native vector geometry (infinite resolution in PDF and SVG, 600 DPI in PNG/TIFF),
with complete directed workflow arrows connecting all 6 modules across Columns 1–3.
"""
from __future__ import annotations
import numpy as np
from matplotlib.patches import (
    FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon, Wedge, Arc
)
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch


def _rbox(ax, x, y, w, h, fc='#EFF3F6', ec='#CBD6DD', r=1.8, lw=0.60, z=1.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f'round,pad=0.0,rounding_size={r}',
        facecolor=fc, edgecolor=ec, lw=lw, zorder=z,
    ))


def _arrow(ax, x0, y0, x1, y1, color='#5B7876', lw=1.15, hl=2.6, hw=1.7, z=6.0):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle=f'-|>,head_length={hl},head_width={hw}',
        lw=lw, color=color, zorder=z, shrinkA=0, shrinkB=0,
        mutation_scale=1.0,
    ))


def _draw_wind_turbine(ax, bx, by, scale=1.0, z=3.5, stroke='#263640'):
    """Draw a sleek architectural 3-blade wind turbine with base at (bx, by) (y downward)."""
    th = 13.5 * scale
    hub_x, hub_y = bx, by - th
    # Foundation pad
    ax.plot([bx - 2.2 * scale, bx + 2.2 * scale], [by, by], color=stroke, lw=0.85 * scale, solid_capstyle='round', zorder=z)
    # Tapered tower
    tower = Polygon([
        (bx - 0.65 * scale, by),
        (bx + 0.65 * scale, by),
        (bx + 0.28 * scale, hub_y),
        (bx - 0.28 * scale, hub_y),
    ], closed=True, facecolor='#FFFFFF', edgecolor=stroke, lw=0.65 * scale, zorder=z)
    ax.add_patch(tower)
    # Blades (3 aerodynamic tapered polygons at -90, 30, 150 deg)
    bl = 6.6 * scale
    for deg in (-90, 30, 150):
        rad = np.radians(deg)
        perp = rad + np.pi / 2
        tip = (hub_x + bl * np.cos(rad), hub_y + bl * np.sin(rad))
        mid1 = (hub_x + 1.6 * scale * np.cos(rad) + 0.52 * scale * np.cos(perp),
                hub_y + 1.6 * scale * np.sin(rad) + 0.52 * scale * np.sin(perp))
        mid2 = (hub_x + 1.2 * scale * np.cos(rad) - 0.22 * scale * np.cos(perp),
                hub_y + 1.2 * scale * np.sin(rad) - 0.22 * scale * np.sin(perp))
        root = (hub_x + 0.25 * scale * np.cos(rad), hub_y + 0.25 * scale * np.sin(rad))
        ax.add_patch(Polygon([root, mid1, tip, mid2], closed=True,
                             facecolor='#DCE7EC', edgecolor=stroke, lw=0.55 * scale, zorder=z + 0.2))
    ax.add_patch(Circle((hub_x, hub_y), 0.58 * scale, facecolor='#FFFFFF', edgecolor=stroke, lw=0.65 * scale, zorder=z + 0.3))


def _draw_pv_array(ax, x0, y_ground, w=8.8, h=5.6, tilt_dx=1.8, z=3.6, stroke='#263640', panel_fc='#27536B'):
    """Draw a tilted crystalline PV array on steel legs resting on y_ground."""
    y_bot = y_ground - 1.3
    y_top = y_bot - h
    # Steel support legs
    ax.plot([x0 + 1.8, x0 + 1.8], [y_bot, y_ground], color=stroke, lw=0.70, zorder=z)
    ax.plot([x0 + w - 1.4, x0 + w - 1.4], [y_top + 1.4, y_ground], color=stroke, lw=0.70, zorder=z)
    ax.plot([x0 + 0.6, x0 + w + 0.6], [y_ground, y_ground], color=stroke, lw=0.75, solid_capstyle='round', zorder=z)
    # Tilted parallelogram module
    p0 = np.array([x0, y_bot])
    p1 = np.array([x0 + w, y_bot])
    p2 = np.array([x0 + w + tilt_dx, y_top])
    p3 = np.array([x0 + tilt_dx, y_top])
    ax.add_patch(Polygon([p0, p1, p2, p3], closed=True, facecolor=panel_fc, edgecolor=stroke, lw=0.65, zorder=z + 0.1))
    # Crisp internal cell grid (4 cols x 3 rows)
    for frac in (0.25, 0.50, 0.75):
        a = p0 * (1 - frac) + p1 * frac
        b = p3 * (1 - frac) + p2 * frac
        ax.plot([a[0], b[0]], [a[1], b[1]], color='#BFD6E2', lw=0.38, zorder=z + 0.2)
    for frac in (0.33, 0.67):
        a = p0 * (1 - frac) + p3 * frac
        b = p1 * (1 - frac) + p2 * frac
        ax.plot([a[0], b[0]], [a[1], b[1]], color='#BFD6E2', lw=0.38, zorder=z + 0.2)


def _draw_sun_cloud_wind(ax, cx, cy, z=3.2):
    """Draw vector sun + cloud + wind streamlines around (cx, cy)."""
    # Sun
    sx, sy = cx + 1.8, cy - 2.6
    for k in range(10):
        ang = np.radians(k * 36)
        ax.plot([sx + 2.55 * np.cos(ang), sx + 3.45 * np.cos(ang)],
                [sy + 2.55 * np.sin(ang), sy + 3.45 * np.sin(ang)],
                color='#D48B22', lw=0.65, solid_capstyle='round', zorder=z)
    ax.add_patch(Circle((sx, sy), 2.05, facecolor='#F2B441', edgecolor='#C57B14', lw=0.55, zorder=z + 0.1))
    # Soft geometric cloud overlapping lower-left of sun
    blobs = [(cx - 2.0, cy - 0.3, 1.8), (cx - 0.1, cy - 1.1, 2.2), (cx + 2.0, cy - 0.4, 1.7), (cx + 3.3, cy + 0.2, 1.3)]
    for bx, by, br in blobs:
        ax.add_patch(Circle((bx, by), br, facecolor='#FFFFFF', edgecolor='#7C909B', lw=0.60, zorder=z + 0.2))
    ax.add_patch(FancyBboxPatch((cx - 3.4, cy - 0.4), 7.6, 1.9, boxstyle='round,pad=0.0,rounding_size=0.9',
                                facecolor='#FFFFFF', edgecolor='none', zorder=z + 0.25))
    for bx, by, br in blobs:
        ax.add_patch(Circle((bx, by), br - 0.08, facecolor='#FFFFFF', edgecolor='none', zorder=z + 0.26))
    ax.plot([cx - 3.0, cx + 4.0], [cy + 1.48, cy + 1.48], color='#7C909B', lw=0.60, solid_capstyle='round', zorder=z + 0.3)
    # Wind streamlines below cloud
    xs = np.linspace(cx - 3.6, cx + 3.8, 60)
    for dy, amp, xend in ((4.2, 0.35, 3.8), (6.0, -0.32, 2.6), (7.7, 0.28, 3.4)):
        mask = xs <= (cx + xend)
        yw = cy + dy + amp * np.sin((xs[mask] - cx) * 0.95)
        ax.plot(xs[mask], yw, color='#3A8580', lw=0.72, solid_capstyle='round', zorder=z + 0.2)


def _draw_card1a_inputs(ax, S):
    _rbox(ax, 1.5, 1.5, 57.5, 45.0)
    ax.text(30.25, 4.0, 'Supply and demand inputs', ha='center', va='top',
            fontsize=6.35, fontweight='bold', color=S.INK, zorder=5)
    ax.text(30.25, 7.9, 'Hourly renewable availability and electricity demand', ha='center', va='top',
            fontsize=5.15, color=S.MUTED, zorder=5)

    # Left graphic zone: Sun/cloud/wind + Wind turbine + PV array
    _draw_sun_cloud_wind(ax, 9.2, 20.8, z=3.2)
    _draw_wind_turbine(ax, 19.8, 41.8, scale=1.22, z=3.5)
    _draw_pv_array(ax, 22.8, 41.8, w=9.2, h=5.8, tilt_dx=1.8, z=3.6)

    # Subtle flow arrow from meteorology/VRE icons to hourly curves
    _arrow(ax, 33.8, 27.5, 36.4, 27.5, color='#7A9399', lw=0.85, hl=2.0, hw=1.3, z=4.0)

    # Right mini-waveforms card: Renewable availability (top) & Electricity demand (bottom)
    _rbox(ax, 37.0, 12.2, 20.4, 31.2, fc='#FFFFFF', ec='#CFDCE2', r=1.2, lw=0.50, z=2.0)
    t = np.linspace(0, 4 * np.pi, 140)
    x_w = 38.3 + (t / (4 * np.pi)) * 17.8
    vre = 0.52 + 0.30 * np.sin(t - 0.6) + 0.16 * np.cos(2.3 * t)
    vre = np.clip(vre, 0.08, 0.95)
    y_vre_base = 24.0
    y_vre = y_vre_base - vre * 7.6
    ax.fill_between(x_w, y_vre_base, y_vre, color='#439892', alpha=0.25, lw=0, zorder=2.6)
    ax.plot(x_w, y_vre, color='#1D6F73', lw=0.85, zorder=3.0)
    ax.plot([38.3, 56.1], [y_vre_base, y_vre_base], color='#CBD6DC', lw=0.42, zorder=2.5)
    ax.text(47.2, 13.4, 'Renewable availability', ha='center', va='top', fontsize=5.0, fontweight='bold', color='#1D6F73', zorder=5)

    ax.plot([38.2, 56.2], [27.2, 27.2], color='#E5EDF0', lw=0.45, zorder=2.5)

    dem = 0.45 + 0.24 * np.sin(t * 1.5 + 0.4) + 0.20 * np.exp(-((t - 7.8) / 1.1) ** 2)
    y_dem_base = 41.6
    y_dem = y_dem_base - dem * 8.2
    ax.fill_between(x_w, y_dem_base, y_dem, color='#8FA2AC', alpha=0.22, lw=0, zorder=2.6)
    ax.plot(x_w, y_dem, color=S.INK, lw=0.85, zorder=3.0)
    ax.plot([38.3, 56.1], [y_dem_base, y_dem_base], color='#CBD6DC', lw=0.42, zorder=2.5)
    ax.text(47.2, 28.5, 'Electricity demand', ha='center', va='top', fontsize=5.0, fontweight='bold', color=S.INK, zorder=5)


def _draw_pylon(ax, cx, y_bot, h=8.6, stroke='#2B3B44', z=3.5):
    y_top = y_bot - h
    # Main A-frame legs
    ax.plot([cx - 2.1, cx - 0.55], [y_bot, y_top], color=stroke, lw=0.65, zorder=z)
    ax.plot([cx + 2.1, cx + 0.55], [y_bot, y_top], color=stroke, lw=0.65, zorder=z)
    ax.plot([cx - 0.55, cx + 0.55], [y_top, y_top], color=stroke, lw=0.65, zorder=z)
    # Cross arms
    for yf, hw in ((0.24, 3.0), (0.52, 2.5)):
        yy = y_top + yf * h
        ax.plot([cx - hw, cx + hw], [yy, yy], color=stroke, lw=0.62, zorder=z)
    # Diagonal lattice X-braces
    for y0f, y1f, w0, w1 in ((0.24, 0.56, 0.9, 1.35), (0.56, 0.92, 1.35, 1.9)):
        ya, yb = y_top + y0f * h, y_top + y1f * h
        ax.plot([cx - w0, cx + w1], [ya, yb], color=stroke, lw=0.42, zorder=z)
        ax.plot([cx + w0, cx - w1], [ya, yb], color=stroke, lw=0.42, zorder=z)


def _draw_gas_plant(ax, cx, y_bot, stroke='#2B3B44', z=3.5):
    # Turbine hall
    ax.add_patch(Rectangle((cx - 4.8, y_bot - 4.4), 5.6, 4.4, facecolor='#DCE4E8', edgecolor=stroke, lw=0.60, zorder=z))
    # Flue stack
    ax.add_patch(Rectangle((cx - 3.9, y_bot - 8.2), 1.1, 3.8, facecolor='#B8C7CF', edgecolor=stroke, lw=0.58, zorder=z))
    # Cooling tower (hyperbolic profile)
    ct = Polygon([
        (cx + 1.3, y_bot),
        (cx + 5.1, y_bot),
        (cx + 4.3, y_bot - 3.8),
        (cx + 4.5, y_bot - 6.6),
        (cx + 1.9, y_bot - 6.6),
        (cx + 2.1, y_bot - 3.8),
    ], closed=True, facecolor='#E5ECEF', edgecolor=stroke, lw=0.60, zorder=z)
    ax.add_patch(ct)
    # Warm peaking gas flame emblem inside hall
    fx, fy = cx - 2.0, y_bot - 1.0
    flame = Polygon([
        (fx, fy - 2.6),
        (fx + 0.95, fy - 1.1),
        (fx + 0.65, fy),
        (fx - 0.65, fy),
        (fx - 0.95, fy - 1.1),
    ], closed=True, facecolor='#D96043', edgecolor='#9E3520', lw=0.45, zorder=z + 0.3)
    ax.add_patch(flame)


def _draw_pv_house(ax, cx, y_bot, scale=1.0, stroke='#2B3B44', z=3.5):
    w_h, h_wall, h_roof = 5.4 * scale, 3.4 * scale, 2.5 * scale
    ax.add_patch(Rectangle((cx - w_h / 2, y_bot - h_wall), w_h, h_wall,
                           facecolor='#F4F7F9', edgecolor=stroke, lw=0.58, zorder=z))
    roof = Polygon([
        (cx - w_h / 2 - 0.55 * scale, y_bot - h_wall),
        (cx, y_bot - h_wall - h_roof),
        (cx + w_h / 2 + 0.55 * scale, y_bot - h_wall),
    ], closed=True, facecolor='#DCE5EA', edgecolor=stroke, lw=0.60, zorder=z + 0.1)
    ax.add_patch(roof)
    # Right roof slope PV panel
    pv = Polygon([
        (cx + 0.15 * scale, y_bot - h_wall - h_roof + 0.35 * scale),
        (cx + w_h / 2 + 0.25 * scale, y_bot - h_wall - 0.22 * scale),
        (cx + w_h / 2 - 0.65 * scale, y_bot - h_wall - 0.22 * scale),
        (cx - 0.55 * scale, y_bot - h_wall - h_roof + 0.35 * scale),
    ], closed=True, facecolor='#276F9E', edgecolor=stroke, lw=0.45, zorder=z + 0.2)
    ax.add_patch(pv)
    # Window
    ax.add_patch(Rectangle((cx - 1.0 * scale, y_bot - h_wall + 0.85 * scale), 2.0 * scale, 1.4 * scale,
                           facecolor='#FFFFFF', edgecolor='#7C8F99', lw=0.42, zorder=z + 0.2))


def _draw_card1b_regimes(ax, S):
    _rbox(ax, 1.5, 50.0, 57.5, 52.5)
    ax.text(30.25, 52.5, 'Three peaking and recharge settings', ha='center', va='top',
            fontsize=6.25, fontweight='bold', color=S.INK, zorder=5)

    cols = [
        (2.8, 17.6, 11.3, 'Chinese\nprovinces', 'Load & curtailed VRE', 'n = 310', S.DEEP['provincial'], S.COLOR['provincial']),
        (21.4, 17.7, 30.25, 'European\ncountries', 'Gas peaking & VRE', 'n = 132', S.DEEP['national'], S.COLOR['national']),
        (40.1, 17.6, 49.1, 'Japanese\ncommunities', 'Grid imports & PV', 'n = 10,612', S.DEEP['community'], S.COLOR['community']),
    ]
    for x0, cw, xc, r_title, r_sub, r_n, r_col, acc in cols:
        _rbox(ax, x0, 57.2, cw, 25.4, fc='#FFFFFF', ec='#D2DEE5', r=1.2, lw=0.50, z=2.0)
        ax.plot([x0 + 1.2, x0 + cw - 1.2], [57.2, 57.2], color=acc, lw=1.25, solid_capstyle='butt', zorder=3.0)
        ax.text(xc, 83.9, r_title, ha='center', va='top', fontsize=5.15, fontweight='bold', color=r_col, linespacing=1.08, zorder=5)
        ax.text(xc, 90.4, r_sub, ha='center', va='top', fontsize=5.0, color=S.INK, zorder=5)
        ax.text(xc, 94.1, r_n, ha='center', va='top', fontsize=5.0, color=S.MUTED, zorder=5)

    # Col 1 icons (China: Wind + PV + HV pylon + curtailment badge — strictly inside white card)
    _draw_wind_turbine(ax, 7.5, 70.4, scale=0.58, z=3.5)
    _draw_pv_array(ax, 10.1, 70.4, w=5.4, h=3.4, tilt_dx=1.0, z=3.6)
    ax.plot([4.6, 18.6], [71.2, 71.2], color='#CBD6DC', lw=0.45, zorder=2.5)
    _draw_pylon(ax, 9.4, 81.0, h=8.0, z=3.5)
    # Curtailment gate symbol
    gx, gy = 15.5, 77.8
    ax.add_patch(Circle((gx, gy), 1.95, facecolor='#FBF1E3', edgecolor=S.COLOR['provincial'], lw=0.65, zorder=3.6))
    ax.plot([gx - 0.90, gx + 0.90], [gy - 0.90, gy + 0.90], color='#B84A32', lw=0.85, solid_capstyle='round', zorder=3.8)
    ax.plot([gx - 0.90, gx + 0.90], [gy + 0.90, gy - 0.90], color='#B84A32', lw=0.85, solid_capstyle='round', zorder=3.8)

    # Col 2 icons (Europe: Wind + PV + Gas peaking plant — strictly inside white card)
    _draw_wind_turbine(ax, 26.1, 70.4, scale=0.58, z=3.5)
    _draw_pv_array(ax, 28.9, 70.4, w=5.4, h=3.4, tilt_dx=1.0, z=3.6)
    ax.plot([23.2, 37.3], [71.2, 71.2], color='#CBD6DC', lw=0.45, zorder=2.5)
    _draw_gas_plant(ax, 30.25, 81.0, z=3.5)

    # Col 3 icons (Japan: Rooftop PV homes + bidirectional grid exchange)
    _draw_pv_house(ax, 45.2, 70.1, scale=0.88, z=3.5)
    _draw_pv_house(ax, 52.3, 70.1, scale=0.88, z=3.5)
    ax.plot([41.8, 55.8], [71.2, 71.2], color='#CBD6DC', lw=0.45, zorder=2.5)
    _draw_pv_house(ax, 45.8, 81.0, scale=0.95, z=3.5)
    # Utility pole + bidirectional arrows
    px = 53.4
    ax.plot([px, px], [72.4, 80.6], color='#2B3B44', lw=0.75, zorder=3.5)
    ax.plot([px - 1.6, px + 1.6], [73.6, 73.6], color='#2B3B44', lw=0.65, zorder=3.5)
    ax.add_patch(Rectangle((px - 0.85, 74.6), 1.7, 2.3, facecolor='#BCCBD3', edgecolor='#2B3B44', lw=0.55, zorder=3.6))
    _arrow(ax, 49.0, 75.5, 52.1, 75.5, color='#276F9E', lw=0.75, hl=1.6, hw=1.1, z=4.0)
    _arrow(ax, 52.1, 78.2, 49.0, 78.2, color='#C9543B', lw=0.75, hl=1.6, hw=1.1, z=4.0)

    ax.text(30.25, 98.5, 'Total sample: N = 11,054 annual records',
            ha='center', va='top', fontsize=5.0, color=S.MUTED, zorder=5)


def _draw_card2a_chronology(ax, S, TEAL, CORAL):
    _rbox(ax, 63.0, 1.5, 56.5, 45.0)
    ax.text(91.25, 4.0, 'Chronology experiments', ha='center', va='top',
            fontsize=6.35, fontweight='bold', color=S.INK, zorder=5)
    ax.text(91.25, 7.9, '1,000 seasonal placements and 1–12 d extensions', ha='center', va='top',
            fontsize=5.15, color=S.MUTED, zorder=5)

    # Crisp vector mirrored annual/seasonal waveform box
    bx, by, bw, bh = 65.0, 12.6, 52.5, 31.2
    _rbox(ax, bx, by, bw, bh, fc='#FFFFFF', ec='#C8D5DC', r=1.1, lw=0.52, z=2.0)

    # 4 seasonal dividers (kept clear of top/bottom corner labels)
    for k in (1, 2, 3):
        sx = bx + k * (bw / 4.0)
        ax.plot([sx, sx], [by + 4.4, by + bh - 4.4], color='#E3ECEF', lw=0.50, ls='--', zorder=2.2)

    y_mid = by + 15.6
    ax.plot([bx + 1.0, bx + bw - 1.0], [y_mid, y_mid], color='#8EA1AA', lw=0.55, zorder=3.0)

    rng = np.random.default_rng(42)
    n_pts = 110
    xs = np.linspace(bx + 1.4, bx + bw - 1.4, n_pts)
    # Upper: Chargeable renewable surplus pulses (teal)
    surp = 0.20 + 0.52 * np.abs(np.sin(np.linspace(0, 11 * np.pi, n_pts))) * rng.uniform(0.55, 1.12, n_pts)
    ev_centers = [15, 35, 50, 70, 86, 99]
    task = np.zeros(n_pts)
    for ec in ev_centers:
        idx = np.arange(n_pts)
        w_ev = 2.0 if ec in (35, 86) else 1.35
        task += np.exp(-((idx - ec) / w_ev) ** 2)
    task = np.clip(task, 0.0, 1.0)
    surp = np.where(task > 0.38, surp * 0.15, surp)
    surp[:26] = np.minimum(surp[:26], 0.72)

    y_surp = y_mid - np.clip(surp, 0, 0.96) * 9.8
    ax.fill_between(xs, y_mid, y_surp, color='#459A95', alpha=0.34, lw=0, zorder=2.6)
    ax.plot(xs, y_surp, color=TEAL, lw=0.72, zorder=2.8)

    # Lower: Peaking task clusters (warm coral/terracotta downward + highlighted event blocks)
    y_task = y_mid + task * 9.6
    ax.fill_between(xs, y_mid, y_task, color='#D9775B', alpha=0.36, lw=0, zorder=2.6)
    ax.plot(xs, y_task, color=CORAL, lw=0.78, zorder=2.9)

    # Curved rearrangement arrows showing intra-seasonal event date shift (placed cleanly in seasons 2 & 3)
    ax.add_patch(FancyArrowPatch(
        (xs[50], y_mid + 10.1), (xs[58], y_mid + 10.1),
        connectionstyle='arc3,rad=0.32',
        arrowstyle='-|>,head_length=1.8,head_width=1.15',
        lw=0.75, color=CORAL, zorder=4.0,
    ))
    ax.add_patch(FancyArrowPatch(
        (xs[86], y_mid + 10.1), (xs[78], y_mid + 10.1),
        connectionstyle='arc3,rad=-0.32',
        arrowstyle='-|>,head_length=1.8,head_width=1.15',
        lw=0.75, color=CORAL, zorder=4.0,
    ))
    ax.text(bx + 1.8, by + 1.2, 'Surplus windows', ha='left', va='top', fontsize=5.0, fontweight='bold', color=TEAL, zorder=5)
    ax.text(bx + 1.8, by + bh - 1.2, 'Peaking events', ha='left', va='bottom', fontsize=5.0, fontweight='bold', color=CORAL, zorder=5)


def _draw_card2b_profiles_and_factorial(ax, S, TEAL, CORAL):
    _rbox(ax, 63.0, 50.0, 56.5, 52.5)
    ax.text(79.7, 52.5, 'Intact daily profiles', ha='center', va='top',
            fontsize=5.65, fontweight='bold', color=S.INK, zorder=5)

    # 5 crisp vector 24-h daily block tiles
    n_tiles = 5
    tx0, ty0, tw, th, tgap = 64.6, 56.8, 5.45, 11.2, 0.72
    t_day = np.linspace(0, 1, 32)
    shapes = [
        (0.75, 0.25),
        (0.22, 0.72),
        (0.82, 0.20),
        (0.28, 0.68),
        (0.70, 0.30),
    ]
    for i, (pk_a, sp_a) in enumerate(shapes):
        xx = tx0 + i * (tw + tgap)
        is_ev = pk_a > 0.5
        fc_t = '#FDF5F2' if is_ev else '#F3FAF9'
        ec_t = '#DDA291' if is_ev else '#9EC8C5'
        _rbox(ax, xx, ty0, tw, th, fc=fc_t, ec=ec_t, r=0.8, lw=0.55, z=2.2)
        ym = ty0 + 6.0
        ax.plot([xx + 0.4, xx + tw - 0.4], [ym, ym], color='#C8D4DA', lw=0.38, zorder=2.5)
        xd = xx + 0.45 + t_day * (tw - 0.90)
        if is_ev:
            yd = ym - pk_a * 4.2 * np.exp(-((t_day - 0.68) / 0.22) ** 2) - 0.35 * pk_a * np.exp(-((t_day - 0.28) / 0.18) ** 2)
            ax.fill_between(xd, ym, yd, color=CORAL, alpha=0.45, lw=0, zorder=2.8)
            ax.plot(xd, yd, color=CORAL, lw=0.65, zorder=3.0)
        else:
            ys = ym + sp_a * 4.0 * np.exp(-((t_day - 0.50) / 0.24) ** 2)
            ax.fill_between(xd, ym, ys, color='#3A918C', alpha=0.42, lw=0, zorder=2.8)
            ax.plot(xd, ys, color=TEAL, lw=0.65, zorder=3.0)

    # Small downward arrow from 5 daily blocks to concatenated multi-day sequence
    _arrow(ax, 79.7, 68.6, 79.7, 71.6, color='#6E8783', lw=0.90, hl=2.0, hw=1.35, z=4.0)

    # Concatenated multi-day dispatch & window sequence box
    sx, sy, sw, sh = 64.6, 72.0, 30.2, 18.4
    _rbox(ax, sx, sy, sw, sh, fc='#FFFFFF', ec='#C8D5DC', r=1.0, lw=0.52, z=2.2)
    yb = sy + 13.0
    ax.plot([sx + 1.0, sx + sw - 1.0], [yb, yb], color='#95A6AF', lw=0.48, zorder=2.5)
    xm = np.linspace(sx + 1.2, sx + sw - 1.2, 120)
    u = (xm - (sx + 1.2)) / (sw - 2.4)
    # 3 consecutive peaking days followed by wide recharge window
    pk_multi = (
        0.85 * np.exp(-((u - 0.14) / 0.065) ** 2)
        + 0.92 * np.exp(-((u - 0.31) / 0.065) ** 2)
        + 0.80 * np.exp(-((u - 0.48) / 0.065) ** 2)
    )
    sp_multi = 0.78 * np.exp(-((u - 0.78) / 0.14) ** 2)
    ax.fill_between(xm, yb, yb - pk_multi * 9.2, color=CORAL, alpha=0.42, lw=0, zorder=2.8)
    ax.plot(xm, yb - pk_multi * 9.2, color=CORAL, lw=0.78, zorder=3.0)
    ax.fill_between(xm, yb, yb + sp_multi * 4.2, color='#459A95', alpha=0.42, lw=0, zorder=2.8)
    ax.plot(xm, yb + sp_multi * 4.2, color=TEAL, lw=0.75, zorder=3.0)
    # Dashed SOC trajectory
    soc = np.interp(u, [0.0, 0.55, 0.96, 1.0], [0.92, 0.12, 0.88, 0.88])
    ax.plot(xm, sy + 15.8 - soc * 12.2, color=S.INK, lw=0.68, ls='--', zorder=3.4)

    ax.text(79.7, 92.0, 'Extended multi-day\ndispatch sequence', ha='center', va='top',
            fontsize=5.1, color=S.INK, linespacing=1.12, zorder=5)

    # Right sub-card: 2 x 2 x 2 Recharge factorial
    _rbox(ax, 96.4, 52.4, 21.6, 47.6, fc='#FFFFFF', ec='#B7C6CF', r=1.4, lw=0.55, z=2.0)
    ax.text(107.2, 54.8, 'Recharge factorial', ha='center', va='top',
            fontsize=5.5, fontweight='bold', color=S.INK, zorder=5)
    ax.text(107.2, 58.6, '2 × 2 × 2 design', ha='center', va='top',
            fontsize=5.05, color=S.MUTED, zorder=5)
    ax.plot([98.0, 116.4], [62.8, 62.8], color='#DCE5EA', lw=0.50, zorder=3)

    fac_specs = [
        (65.0, 'Surplus energy', '1× vs 4× observed'),
        (77.2, 'Charging power', '1× vs 4× rated'),
        (89.4, 'Task-free hours', 'Observed vs all h'),
    ]
    for fy, fh, fs in fac_specs:
        ax.text(107.2, fy, fh, ha='center', va='top', fontsize=5.15, fontweight='bold', color=TEAL, zorder=5)
        ax.text(107.2, fy + 3.8, fs, ha='center', va='top', fontsize=5.0, color=S.INK, zorder=5)
    ax.plot([98.5, 115.9], [75.0, 75.0], color='#E6EEF1', lw=0.45, zorder=3)
    ax.plot([98.5, 115.9], [87.2, 87.2], color='#E6EEF1', lw=0.45, zorder=3)


def _draw_lfp_icon(ax, cx, cy, z=3.5, stroke='#263640'):
    """Draw a crisp modular containerized 4 h LFP battery enclosure centered around (cx, cy)."""
    w, h = 15.6, 13.2
    x0, y0 = cx - w / 2, cy - h / 2
    # Isometric top & side depth planes for clean 3D engineering depth
    dx, dy = 2.0, -1.6
    ax.add_patch(Polygon([(x0, y0), (x0 + dx, y0 + dy), (x0 + w + dx, y0 + dy), (x0 + w, y0)],
                         closed=True, facecolor='#D4E0E6', edgecolor=stroke, lw=0.58, zorder=z))
    ax.add_patch(Polygon([(x0 + w, y0), (x0 + w + dx, y0 + dy), (x0 + w + dx, y0 + h + dy), (x0 + w, y0 + h)],
                         closed=True, facecolor='#B6C7D0', edgecolor=stroke, lw=0.58, zorder=z))
    # Front cabinet face
    _rbox(ax, x0, y0, w, h, fc='#F4F8FA', ec=stroke, r=0.7, lw=0.65, z=z + 0.1)
    # 4 vertical battery cell racks with SOC bars
    cell_cols = ['#2A8582', '#2A8582', '#4AA69D', '#D4922A']
    cw = 2.55
    for i, cc in enumerate(cell_cols):
        rx = x0 + 1.35 + i * 3.35
        ry = y0 + 2.2
        rh = h - 3.6
        ax.add_patch(Rectangle((rx, ry), cw, rh, facecolor='#E5EEF2', edgecolor='#5C707B', lw=0.48, zorder=z + 0.2))
        fill_h = rh * (0.82 if i < 3 else 0.58)
        ax.add_patch(Rectangle((rx + 0.28, ry + rh - fill_h), cw - 0.56, fill_h, facecolor=cc, edgecolor='none', zorder=z + 0.3))
        # Terminal cap
        ax.add_patch(Rectangle((rx + 0.65, ry - 0.65), cw - 1.30, 0.65, facecolor='#5C707B', edgecolor='none', zorder=z + 0.3))


def _draw_acaes_icon(ax, cx, cy, z=3.5, stroke='#263640'):
    """Draw a crisp engineering schematic of 24 h A-CAES (compressor -> TES -> turbine + air storage vessel)."""
    # Horizontal shaft line
    y_sh = cy - 2.6
    ax.plot([cx - 11.8, cx + 11.8], [y_sh, y_sh], color='#5C707B', lw=1.05, zorder=z)
    # Compressor trapezoid (left)
    comp = Polygon([
        (cx - 10.2, y_sh - 2.2),
        (cx - 5.6, y_sh - 3.4),
        (cx - 5.6, y_sh + 3.4),
        (cx - 10.2, y_sh + 2.2),
    ], closed=True, facecolor='#CFDCE3', edgecolor=stroke, lw=0.62, zorder=z + 0.2)
    ax.add_patch(comp)
    # Thermal Energy Storage (TES) insulated vessel (center)
    _rbox(ax, cx - 2.7, y_sh - 3.8, 5.4, 7.6, fc='#F9E8DF', ec=stroke, r=1.2, lw=0.62, z=z + 0.2)
    # Heat-exchanger zig-zag coil inside TES
    hy = np.linspace(y_sh - 2.2, y_sh + 2.2, 7)
    hx = cx + np.array([-1.1, 1.1, -1.1, 1.1, -1.1, 1.1, -1.1])
    ax.plot(hx, hy, color='#C9543B', lw=0.72, zorder=z + 0.3)
    # Expander turbine trapezoid (right)
    turb = Polygon([
        (cx + 5.6, y_sh - 3.4),
        (cx + 10.2, y_sh - 2.2),
        (cx + 10.2, y_sh + 2.2),
        (cx + 5.6, y_sh + 3.4),
    ], closed=True, facecolor='#CFDCE3', edgecolor=stroke, lw=0.62, zorder=z + 0.2)
    ax.add_patch(turb)
    # High-pressure air storage vessel below + manifold pipes
    y_tk = cy + 4.3
    ax.plot([cx - 7.8, cx - 7.8], [y_sh + 2.8, y_tk], color=stroke, lw=0.65, zorder=z + 0.1)
    ax.plot([cx + 7.8, cx + 7.8], [y_sh + 2.8, y_tk], color=stroke, lw=0.65, zorder=z + 0.1)
    _rbox(ax, cx - 10.6, y_tk - 1.9, 21.2, 3.8, fc='#DCE8EE', ec=stroke, r=1.85, lw=0.62, z=z + 0.2)
    ax.plot([cx - 7.5, cx + 7.5], [y_tk, y_tk], color='#3A8580', lw=0.65, ls='--', zorder=z + 0.3)


def _draw_h2_tank_bundle(ax, cx, cy, z=3.5, stroke='#263640'):
    """Draw a crisp 200 bar aboveground horizontal steel-tank H2 storage bundle on structural cradles."""
    w_t = 27.0
    x0 = cx - w_t / 2
    # Two stacked horizontal cylindrical pressure vessels
    for dy, fc_c in ((-2.3, '#E3EDF2'), (1.6, '#CFDDE5')):
        _rbox(ax, x0, cy + dy - 1.65, w_t, 3.3, fc=fc_c, ec=stroke, r=1.6, lw=0.62, z=z + 0.2)
        # Highlight band
        ax.plot([x0 + 2.2, x0 + w_t - 2.2], [cy + dy - 0.65, cy + dy - 0.65],
                color='#FFFFFF', lw=0.65, alpha=0.85, solid_capstyle='round', zorder=z + 0.25)
    # Steel support saddles / cradles
    for sx in (x0 + 5.2, x0 + w_t - 5.2):
        ax.add_patch(Rectangle((sx - 0.8, cy - 4.1), 1.6, 8.2, facecolor='#7D909B', edgecolor=stroke, lw=0.50, zorder=z + 0.3))
        ax.plot([sx - 2.0, sx + 2.0], [cy + 4.3, cy + 4.3], color=stroke, lw=0.75, solid_capstyle='round', zorder=z + 0.3)
    # Right-hand high-pressure manifold & pressure gauge
    mx = x0 + w_t + 1.5
    ax.plot([x0 + w_t, mx, mx, x0 + w_t], [cy - 2.3, cy - 2.3, cy + 1.6, cy + 1.6], color=stroke, lw=0.68, zorder=z + 0.3)
    ax.plot([mx, mx + 1.8], [cy - 0.35, cy - 0.35], color='#2A8582', lw=0.85, zorder=z + 0.3)
    ax.add_patch(Circle((mx + 1.1, cy - 2.5), 1.05, facecolor='#FFFFFF', edgecolor=stroke, lw=0.55, zorder=z + 0.4))
    ax.plot([mx + 1.1, mx + 1.65], [cy - 2.5, cy - 3.0], color='#C9543B', lw=0.55, zorder=z + 0.5)


def _draw_card3_evaluation(ax, S, TEAL):
    # Card 3A: Portfolios and 8-technology catalog
    _rbox(ax, 124.0, 1.5, 56.0, 55.2)
    ax.text(152.0, 4.0, 'Portfolios and 8-technology catalog', ha='center', va='top',
            fontsize=6.2, fontweight='bold', color=S.INK, zorder=5)
    ax.text(152.0, 7.9, '58 configurations across 4–720 h rated duration', ha='center', va='top',
            fontsize=5.15, color=S.MUTED, zorder=5)

    _draw_lfp_icon(ax, 136.0, 20.6, z=3.5)
    ax.text(136.0, 29.6, '4 h LFP', ha='center', va='top', fontsize=5.35, fontweight='bold', color=S.INK, zorder=5)
    ax.text(136.0, 33.2, 'Reference & hybrid base', ha='center', va='top', fontsize=5.0, color=S.MUTED, zorder=5)

    _draw_acaes_icon(ax, 163.2, 20.4, z=3.5)
    ax.text(163.2, 29.6, '24 h A-CAES', ha='center', va='top', fontsize=5.35, fontweight='bold', color=S.INK, zorder=5)
    ax.text(163.2, 33.2, 'Long-duration benchmark', ha='center', va='top', fontsize=5.0, color=S.MUTED, zorder=5)

    ax.plot([127.0, 177.0], [37.0, 37.0], color='#DCE5EA', lw=0.48, zorder=2.5)
    _draw_h2_tank_bundle(ax, 151.2, 43.5, z=3.5)
    ax.text(152.0, 50.8, '200 bar surface H₂ tank and 5 additional technologies',
            ha='center', va='top', fontsize=5.1, fontweight='bold', color=S.INK, zorder=5)

    # Card 3B: Hourly cyclic dispatch optimization
    _rbox(ax, 124.0, 60.0, 56.0, 18.8, fc='#FFFFFF', ec='#B7C6CF', r=1.5, lw=0.58, z=1.5)
    ax.text(152.0, 62.4, 'Hourly cyclic dispatch optimization', ha='center', va='top',
            fontsize=5.75, fontweight='bold', color=S.INK, zorder=5)
    ax.plot([126.5, 177.5], [66.8, 66.8], color='#E2ECEF', lw=0.45, zorder=2)
    ax.text(152.0, 68.3, 'Equivalent full cycles and charging-window utilization', ha='center', va='top',
            fontsize=5.05, color=S.INK, zorder=5)
    ax.text(152.0, 72.8, 'Annual system-cost saving B and incremental return', ha='center', va='top',
            fontsize=5.05, color=S.INK, zorder=5)

    # Card 3C: External validation and multi-year planning
    _rbox(ax, 124.0, 82.1, 56.0, 20.4, fc='#FFFFFF', ec='#B7C6CF', r=1.5, lw=0.58, z=1.5)
    ax.text(152.0, 84.3, 'External validation and multi-year planning', ha='center', va='top',
            fontsize=5.65, fontweight='bold', color=S.INK, zorder=5)
    ax.plot([126.5, 177.5], [88.5, 88.5], color='#E2ECEF', lw=0.45, zorder=2)

    val_rows = [
        (90.0, 'European historical weather', '75 yr (1950–2025)'),
        (94.1, 'US market balancing records', '32 records (3 ISOs)'),
        (98.2, 'Multi-year portfolio planning', 'Mean & CVaR90'),
    ]
    for vy, v_left, v_right in val_rows:
        ax.text(126.5, vy, v_left, ha='left', va='top', fontsize=5.0, color=S.INK, zorder=5)
        ax.text(177.5, vy, v_right, ha='right', va='top', fontsize=5.0, fontweight='bold', color=TEAL, zorder=5)


def build_vector_framework(S, TEAL='#19646B', CORAL='#C9543B'):
    h_mm = 104.0
    cv = S.Canvas(h_mm)
    ax = cv.ax(0.0, 0.0, 183.0, h_mm, 'a')
    ax.set_xlim(0.0, 183.0)
    ax.set_ylim(h_mm, 0.0)
    ax.axis('off')

    _draw_card1a_inputs(ax, S)
    _draw_card1b_regimes(ax, S)
    _draw_card2a_chronology(ax, S, TEAL, CORAL)
    _draw_card2b_profiles_and_factorial(ax, S, TEAL, CORAL)
    _draw_card3_evaluation(ax, S, TEAL)

    # Complete, non-crossing directed workflow arrows across all 6 cards:
    # 1. Card 1A (Supply & demand inputs) -> Card 1B (Three empirical regimes)
    _arrow(ax, 30.25, 46.7, 30.25, 49.8)
    # 2. Card 1A (Supply & demand inputs) -> Card 2A (Chronology experiments)
    _arrow(ax, 59.2, 24.0, 62.8, 24.0)
    # 3. Card 1B (Three empirical regimes) -> Card 2B (Intact daily profiles & Recharge factorial)
    _arrow(ax, 59.2, 76.2, 62.8, 76.2)
    # 4. Card 2A (Chronology experiments) -> Card 2B (Intact daily profiles & Extended sequence)
    _arrow(ax, 91.25, 46.7, 91.25, 49.8)
    # 5. Card 2A (Chronology experiments) -> Card 3A (Portfolios & 8-technology catalog)
    _arrow(ax, 119.7, 24.0, 123.8, 24.0)
    # 6. Card 2B (Daily profiles & Recharge factorial) -> Card 3B (Hourly cyclic dispatch optimization)
    _arrow(ax, 119.7, 69.4, 123.8, 69.4)
    # 7. Card 3A (Portfolios & 8-technology catalog) -> Card 3B (Hourly cyclic dispatch optimization)
    _arrow(ax, 152.0, 56.9, 152.0, 59.8)
    # 8. Card 3B (Hourly cyclic dispatch optimization) -> Card 3C (External validation & multi-year planning)
    _arrow(ax, 152.0, 79.0, 152.0, 81.9)

    return cv
