"""Extended Data Figure 5. Temporal validation of reconstructed European historical hourly electricity demand (6 diverse panels a-f)."""
from pathlib import Path
import json, shutil
import restore_figures as F
from restore_figures import np, pd, plt, S, W, V85, T, read, export
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter


def main():
    F.caps = json.loads(F.CAPS_FILE.read_text()) if F.CAPS_FILE.exists() else {}
    F.audits = json.loads(F.AUD_FILE.read_text()) if F.AUD_FILE.exists() else {}
    src = V85 / 'source_data/archived_v82/extended_data'
    for name in ['extended_data_figure1b_holdout_sample.csv.gz', 'demand_model_qa.json']:
        if (src / name).exists() and not (T / name).exists():
            shutil.copy2(src / name, T / name)
    fit = read('extended_data_figure1a_country_holdout_fit').sort_values('weighted_r2')
    sample = pd.read_csv(T / 'extended_data_figure1b_holdout_sample.csv.gz')
    assert len(sample) == 120000
    qa = json.loads((T / 'demand_model_qa.json').read_text())['holdout']

    S.setup()
    cv = S.Canvas(156.0)

    # Row 1: a, b, c with exact equal widths (39.5 mm) and uniform gutters (23.5 mm), spanning 14.5 .. 180.0 mm
    cv.letter(2.0, 5.0, 'a')
    axa = cv.ax(14.5, 12.0, 39.5, 38.0, 'a')
    y = np.arange(len(fit))
    axa.barh(y, fit.weighted_pearson_r, height=0.62, facecolor='#E8B1A2', edgecolor='none', label='Pearson r', zorder=2)
    axa.barh(y, fit.weighted_r2, height=0.34, facecolor='#19646B', edgecolor='none', label='R²', zorder=3)
    axa.axvline(qa['weighted_r2'], color='#19646B', lw=0.75, ls='--', zorder=3.5)
    axa.axvline(qa['weighted_pearson_r'], color='#C9543B', lw=0.75, ls='--', zorder=3.5)
    axa.set_yticks(y, fit.country, fontsize=5.0)
    axa.tick_params(axis='y', length=0)
    axa.set_xlim(0.35, 1.01); axa.set_xticks([0.4, 0.6, 0.8, 1.0])
    axa.set_xlabel('Hold-out fit (2023–2024)')
    axa.grid(False); S.grid(axa, 'x')
    axa.legend(loc='lower left', bbox_to_anchor=(-0.03, 1.01), ncol=2, fontsize=5.1, handletextpad=0.3, columnspacing=0.8)

    cv.letter(65.5, 5.0, 'b')
    axb = cv.ax(77.5, 12.0, 39.5, 38.0, 'b')
    bins = np.linspace(0, 2.2, 67)
    cnt, _, _ = np.histogram2d(sample.load_pu, sample.predicted_load_pu, bins=[bins, bins])
    teal_cmap = LinearSegmentedColormap.from_list('teal_dens', ['#EEF6F6', '#74B3B0', '#19646B', '#0B353A'])
    im_b = axb.imshow(np.ma.masked_equal(cnt.T, 0), origin='lower', extent=[0, 2.2, 0, 2.2], aspect='auto',
                      cmap=teal_cmap, norm=LogNorm(1, max(2, cnt.max())), zorder=2)
    axb.plot([0, 2.2], [0, 2.2], color='#C9543B', lw=0.9, ls='--', zorder=3)
    axb.text(0.06, 0.92, f"R² = {qa['weighted_r2']:.2f}\nr = {qa['weighted_pearson_r']:.2f}\nRMSE = {qa['weighted_rmse_pu']:.3f}",
             transform=axb.transAxes, va='top', fontsize=5.1)
    axb.set_xlim(0, 2.2); axb.set_ylim(0, 2.2)
    axb.set_xticks([0, 0.5, 1.0, 1.5, 2.0]); axb.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
    axb.set_xlabel('Observed load (p.u.)'); axb.set_ylabel('Predicted load (p.u.)')
    cab = cv.ax(118.4, 14.0, 1.8, 33.0, 'cb_b')
    cbb = cv.fig.colorbar(im_b, cax=cab, orientation='vertical')
    cbb.set_label('Hours per bin', fontsize=5.0, labelpad=1.5)
    cbb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:g}'))
    cbb.ax.minorticks_off(); cbb.ax.tick_params(labelsize=5.0, length=1.6, pad=1.2)

    cv.letter(129.0, 5.0, 'c')
    axc = cv.ax(140.5, 12.0, 39.5, 38.0, 'c')
    d = read('extended_data_figure1d_hourly_error')
    axc.fill_between(d.hour, d.q10, d.q90, color='#D4E7E8', alpha=0.80, lw=0, label='10th–90th')
    if 'q25' in d.columns and 'q75' in d.columns:
        axc.fill_between(d.hour, d.q25, d.q75, color='#82B8BA', alpha=0.55, lw=0, label='25th–75th')
    axc.plot(d.hour, d['median'], color='#19646B', lw=1.25, marker='o', ms=2.1, label='Median')
    axc.axhline(0, color=S.INK, lw=0.60)
    axc.set_ylim(-0.2, 0.2); axc.set_yticks([-0.2, -0.1, 0.0, 0.1, 0.2])
    axc.set_xlim(-0.5, 23.5); axc.set_xticks([0, 6, 12, 18, 23])
    axc.set_xlabel('Hour (UTC)'); axc.set_ylabel('Prediction error (p.u.)')
    S.grid(axc, 'y')
    axc.legend(loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=2, fontsize=5.0, handlelength=1.2, columnspacing=0.7)

    # Row 2: d (Observed, Predicted, Difference 12x24 month-hour surfaces — colorbar label right edge locked to 180.0)
    cv.letter(2.0, 57.5, 'd')
    z = read('demand_month_hour_surface')
    obs = z.pivot(index='month', columns='hour', values='observed_load_pu').to_numpy()
    pred = z.pivot(index='month', columns='hour', values='predicted_load_pu').to_numpy()
    div_cmap = LinearSegmentedColormap.from_list('teal_coral', ['#19646B', '#7BB4AE', '#F7F7F7', '#E3977D', '#C9543B'])
    load_cmap = LinearSegmentedColormap.from_list('teal_load', ['#F7FAF9', '#8FC1BA', '#19646B', '#0D383D'])
    xs_d = [14.5, 71.0, 127.5]
    titles_d = ['Observed', 'Predicted', 'Predicted − observed']
    for i, (mat, xd, td) in enumerate(zip([obs, pred, pred - obs], xs_d, titles_d)):
        axd = cv.ax(xd, 66.5, 42.0, 34.0, f'd{i + 1}')
        axd.grid(False)
        im = axd.imshow(mat, aspect='auto', cmap=div_cmap if i == 2 else load_cmap,
                        vmin=-0.11 if i == 2 else 0.72, vmax=0.11 if i == 2 else 1.26, extent=[-0.5, 23.5, 11.5, -0.5])
        axd.set_xticks([0, 6, 12, 18, 23]); axd.set_yticks([0, 3, 6, 9, 11])
        axd.set_yticklabels(['Jan', 'Apr', 'Jul', 'Oct', 'Dec'] if i == 0 else [])
        axd.set_xlabel('Hour (UTC)')
        axd.text(0.0, 1.04, td, transform=axd.transAxes, fontsize=5.4, fontweight='bold', va='bottom', ha='left')
        if i == 1:
            cal = cv.ax(114.5, 66.5, 2.0, 34.0, 'cb_load')
            cbl = cv.fig.colorbar(im, cax=cal, ticks=[0.8, 1.0, 1.2])
            cbl.set_label('Load (p.u.)', fontsize=5.1); cbl.ax.tick_params(labelsize=5.0, length=1.6, pad=1.4)
        if i == 2:
            cad = cv.ax(170.8, 66.5, 2.0, 34.0, 'cb_diff')
            cbd = cv.fig.colorbar(im, cax=cad, ticks=[-0.10, 0.0, 0.10])
            cbd.set_label('Diff. (p.u.)', fontsize=5.1); cbd.ax.tick_params(labelsize=5.0, length=1.6, pad=1.4)

    # Row 3: e (76-year shaded deviation horizon + 11-year smooth), f (45-degree 2023 vs 2024 country RMSE calibration scatter)
    cv.letter(2.0, 109.0, 'e')
    sh = read('demand_annual_shape').sort_values('calendar_year')
    axe1 = cv.ax(14.5, 115.0, 108.5, 13.5, 'e1')
    axe2 = cv.ax(14.5, 130.5, 108.5, 13.5, 'e2')
    m1 = float(sh.maximum_daily_mean_pu.mean())
    m2 = float(sh.p95_p05_width_pu.mean())

    axe1.fill_between(sh.calendar_year, m1, sh.maximum_daily_mean_pu, color='#F4D1C7', alpha=0.65, lw=0)
    axe1.plot(sh.calendar_year, sh.maximum_daily_mean_pu, color='#C9543B', lw=0.65, alpha=0.65)
    axe1.plot(sh.calendar_year, sh.maximum_daily_mean_pu.rolling(11, center=True, min_periods=5).mean(), color='#8E3320', lw=1.35)
    axe1.axhline(m1, color='#6E7C80', lw=0.60, ls='--')
    axe1.text(0.99, 0.92, f'mean {m1:.3f}', transform=axe1.transAxes, ha='right', va='top', fontsize=5.05, fontweight='bold', color='#8E3320')
    axe1.set_xticklabels([]); axe1.set_ylim(1.198, 1.298); axe1.set_yticks([1.22, 1.25])
    axe1.set_ylabel('Max. daily\nmean (p.u.)')
    axe1.set_xlim(1949, 2026); axe1.set_xticks([1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020])
    S.grid(axe1, 'y')

    axe2.fill_between(sh.calendar_year, m2, sh.p95_p05_width_pu, color='#CBE5E6', alpha=0.70, lw=0)
    axe2.plot(sh.calendar_year, sh.p95_p05_width_pu, color='#19646B', lw=0.65, alpha=0.65)
    axe2.plot(sh.calendar_year, sh.p95_p05_width_pu.rolling(11, center=True, min_periods=5).mean(), color='#0E3E44', lw=1.35)
    axe2.axhline(m2, color='#6E7C80', lw=0.60, ls='--')
    axe2.text(0.99, 0.92, f'mean {m2:.3f}', transform=axe2.transAxes, ha='right', va='top', fontsize=5.05, fontweight='bold', color='#0E3E44')
    axe2.set_ylim(0.515, 0.602); axe2.set_yticks([0.53, 0.56])
    axe2.set_ylabel('P95 − P5\nwidth (p.u.)')
    axe2.set_xlim(1949, 2026); axe2.set_xticks([1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020])
    axe2.set_xlabel('Calendar year (bold: 11-yr rolling mean)')
    S.grid(axe2, 'y')

    cv.letter(133.0, 109.0, 'f')
    axf = cv.ax(144.5, 115.0, 35.5, 29.0, 'f')
    zrmse = read('extended_data_figure1f_yearly_rmse').pivot(index='country', columns='year', values='weighted_rmse_pu').dropna()
    axf.plot([0.05, 0.145], [0.05, 0.145], color='#8EA0A7', lw=0.75, ls='--', zorder=1)
    axf.scatter(zrmse[2023], zrmse[2024], s=14, color='#19646B', edgecolor='white', lw=0.3, zorder=4)
    axf.set_xlim(0.052, 0.145); axf.set_ylim(0.052, 0.145)
    axf.set_xticks([0.06, 0.10, 0.14]); axf.set_yticks([0.06, 0.10, 0.14])
    axf.set_xlabel('2023 RMSE (p.u.)'); axf.set_ylabel('2024 RMSE (p.u.)')
    S.grid(axf, 'both')

    caption = (
        'a，16 个具有 2023—2024 年验证资料国家的嵌套子弹条形对比：深青内条为决定系数 R²，浅珊瑚外条为 Pearson 相关系数 r，虚线为全部 271,484 个验证小时的整体指标（R² = 0.734，r = 0.857）。'
        'b，120,000 条抽样验证小时的观测与预测归一化负荷二维对数密度，虚线为 1:1 等值线，整体 RMSE = 0.096、MAE = 0.073。'
        'c，按 UTC 小时的预测误差中位数及第 10—90 百分位包络。'
        'd，按月份与 UTC 小时的观测负荷、预测负荷（共用负荷色标）以及预测减观测偏差。'
        'e，1950—2025 年重建区域负荷的年度最大日均值（上）与第 95—5 百分位跨度（下，粗线为 11 年滑动均值，虚线为 76 年均值）。'
        'f，各国 2023 与 2024 年验证均方根误差的 45° 一致性散点对比。模型以 2015—2022 年拟合、2023—2024 年检验。'
    )
    F.audits.pop('extended_data_figure5', None)
    export(
        cv,
        'extended_data_figure5',
        '扩展数据图 5 | 欧洲历史需求重建的时间验证',
        caption,
        rows=[['a', 'b', 'c'], ['d1', 'd2', 'd3']],
        exempt=[
            dict(panels=['cb_b', 'cb_load', 'cb_diff'], checks=['all'], reason='Colorbars'),
            dict(panels=['e1', 'e2', 'f'], checks=['all'], reason='Stacked horizon panel e aligned beside calibration scatter f'),
        ],
    )
    F.audits['extended_data_figure5']['density_display_samples'] = len(sample)
    F.audits['extended_data_figure5']['samples_inside_display_extent'] = int(cnt.sum())
    F.AUD_FILE.write_text(json.dumps(F.audits, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
