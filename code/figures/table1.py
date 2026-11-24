"""Table 1. Charging windows and economics of the three main-text contexts (2035 costs, 11,054 records).

Writes figures/output/table1.{csv,md,tex}. Reports full-sample Chinese province-years (n = 310) alongside
parenthetical positive-curtailment province-years (n = 178).
"""
import numpy as np
import pandas as pd

import style as S
import data as D

NAME = {'provincial': '中国各省', 'national': '欧洲各国', 'community': '日本社区'}


def fmt_int(v):
    return f'{v:,.0f}'


def fmt_roi(v):
    return f'{v:.2f}%' if np.isfinite(v) else '无实根'


def main():
    rec = D.records()
    s = D.summary()
    s35 = s[s.year == 2035].set_index('group')
    s_cn_pos = D.summary_cn_positive()
    r_pos = s_cn_pos[s_cn_pos.year == 2035].iloc[0]
    pos_uids = D.cn_positive_uids()

    pdl = D.placement_delta_long_windows().set_index('uid')
    lw_real = pdl.lw_real
    lw_rand = pdl.lw_mc_med
    c = D.central12()
    d12 = c.groupby('group').delta_B.median()
    d12_cn_pos = float(c[c.uid.isin(pos_uids)].delta_B.median())

    rows = {}
    for k in S.MAIN:
        x = rec[rec.group == k]
        r = s35.loc[k]
        n = len(x)
        assert int(r.n) == n
        if k == 'provincial':
            xp = x[x.uid.isin(pos_uids)]
            np_ = len(xp)
            assert np_ == 178 and int(r_pos.n) == 178
            rows[k] = {
                '年度记录（份）': f'{n:,}（{np_:,}）',
                '平均事件持续时间（天）': f'{x.mean_dur.median():.2f}（{xp.mean_dur.median():.2f}）',
                '一年可充电小时': f'{fmt_int(x.charge_hours.median())}（{fmt_int(xp.charge_hours.median())}）',
                '能充满 A-CAES 的充电窗口（个），真实｜随机': (
                    f'{lw_real.reindex(x.uid).median():g}｜{lw_rand.reindex(x.uid).median():g}'
                    f'（{lw_real.reindex(xp.uid).median():g}｜{lw_rand.reindex(xp.uid).median():g}）'
                ),
                'A-CAES 等效满循环次数': f'{x.efc_acaes.median():.1f}（{xp.efc_acaes.median():.1f}）',
                '新增 4 小时 LFP 等效满循环次数': f'{x.efc_lfp.median():.1f}（{xp.efc_lfp.median():.1f}）',
                '增量投资回报率，真实｜随机': (
                    f'{fmt_roi(r.real_roi_med)}｜{fmt_roi(r.mc_roi_med)}\n'
                    f'（{fmt_roi(r_pos.real_roi_med)}｜{fmt_roi(r_pos.mc_roi_med)}）'
                ),
                '混合组合更省钱，2%': (
                    f'{int(r["real_win_2"]):,}/{n:,}｜{100 * r["mc_win_2"]:.0f}%\n'
                    f'（{int(r_pos["real_win_2"]):,}/{np_:,}｜{100 * r_pos["mc_win_2"]:.0f}%）'
                ),
                '混合组合更省钱，3.6%': (
                    f'{int(r["real_win_3.6"]):,}/{n:,}｜{100 * r["mc_win_3.6"]:.0f}%\n'
                    f'（{int(r_pos["real_win_3.6"]):,}/{np_:,}｜{100 * r_pos["mc_win_3.6"]:.0f}%）'
                ),
                '混合组合更省钱，6%': (
                    f'{int(r["real_win_6"]):,}/{n:,}｜{100 * r["mc_win_6"]:.0f}%\n'
                    f'（{int(r_pos["real_win_6"]):,}/{np_:,}｜{100 * r_pos["mc_win_6"]:.0f}%）'
                ),
                '真实日期回报率在随机放置中的百分位': f'{100 * r.real_pct_med:.0f}（{100 * r_pos.real_pct_med:.0f}）',
                '12 天调峰事件延长的 B 变化': (
                    f'{d12[k]:.1f}（{d12_cn_pos:.1f}）'.replace('-', '−')
                ),
            }
        else:
            rows[k] = {
                '年度记录（份）': f'{n:,}',
                '平均事件持续时间（天）': f'{x.mean_dur.median():.2f}',
                '一年可充电小时': fmt_int(x.charge_hours.median()),
                '能充满 A-CAES 的充电窗口（个），真实｜随机': f'{lw_real.reindex(x.uid).median():g}｜{lw_rand.reindex(x.uid).median():g}',
                'A-CAES 等效满循环次数': f'{x.efc_acaes.median():.1f}',
                '新增 4 小时 LFP 等效满循环次数': f'{x.efc_lfp.median():.1f}',
                '增量投资回报率，真实｜随机': f'{r.real_roi_med:.2f}%｜{r.mc_roi_med:.2f}%',
                '混合组合更省钱，2%': f'{int(r["real_win_2"]):,}/{n:,}｜{100 * r["mc_win_2"]:.0f}%',
                '混合组合更省钱，3.6%': f'{int(r["real_win_3.6"]):,}/{n:,}｜{100 * r["mc_win_3.6"]:.0f}%',
                '混合组合更省钱，6%': f'{int(r["real_win_6"]):,}/{n:,}｜{100 * r["mc_win_6"]:.0f}%',
                '真实日期回报率在随机放置中的百分位': f'{100 * r.real_pct_med:.0f}',
                '12 天调峰事件延长的 B 变化': f'{d12[k]:.1f}'.replace('-', '−'),
            }
            assert np.isclose(r.real_roi_med, x.roi_2035.median())

    t = pd.DataFrame(rows)
    assert t['national'].tolist() == [
        '132', '2.30', '2,447', '24.5｜27', '27.8', '37.0',
        '2.97%｜3.35%', '98/132｜81%', '24/132｜34%', '7/132｜5%', '9', '−22.5'
    ], t['national'].tolist()

    t.columns = [NAME[k] for k in t.columns]
    t.index.name = '指标'
    t.to_csv(S.OUT / 'table1.csv', encoding='utf-8-sig')
    md = ['| 指标 | ' + ' | '.join(t.columns) + ' |', '|---|' + '---|' * len(t.columns)]
    md += [f'| {i} | ' + ' | '.join(str(x).replace('\n', ' ') for x in r) + ' |' for i, r in t.iterrows()]
    (S.OUT / 'table1.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    body = '\n'.join(
        f'{i} & ' + ' & '.join(str(v).replace('\n', ' ').replace('%', r'\%') for v in r) + r'\\'
        + ('\n\\midrule' if i in ('一年可充电小时', '真实日期回报率在随机放置中的百分位') else '')
        for i, r in t.iterrows()
    )
    tex = r'''\begin{table}[htbp]
\centering\small
\caption{\textbf{表 1 | 中国各省、欧洲各国与日本社区的充电窗口与经济性。} 2035 年技术成本。“｜”左为真实日期，右为同一组记录的 1,000 种起止位置随机放置。中国各省列给出全部 310 份省年记录统计，括号内为具有正年度弃电预算的 178 份省年结果；无实根表示运行成本节约不大于零、在任意正资金成本下均为纯 LFP 组合成本较低。持续时间、可充电小时、等效满循环次数与回报率为组内记录的中位数，随机放置先取每份记录 1,000 种排列的中位数。具有成本优势一行左为记录数，右为每份记录内具有成本优势的排列比例再对记录等权平均。B 变化为 12 天调峰事件延长中心排列相对真实日期的中位数，单位 USD kW$^{-1}$ yr$^{-1}$。}
\label{tab:five-cases}
\begin{tabularx}{\textwidth}{@{}P{5.6cm}YYY@{}}
\toprule
指标 & 中国各省 & 欧洲各国 & 日本社区\\
\midrule
''' + body + r'''
\bottomrule
\end{tabularx}
\TableNote{能充满 A-CAES 的充电窗口指相邻两次调用之间可用盈余电量足以把 A-CAES 从空状态充至可用容量的窗口，随机一栏为每份记录 1,000 种排列的中位数。等效满循环次数为优化调度中新增储能的年放电量除以额定储能容量，取真实日期。增量投资回报率由年化成本差方程求根，多根取最大实根。}
\end{table}
'''
    (S.OUT / 'table1.tex').write_text(tex, encoding='utf-8')
    print(t.to_string())


if __name__ == '__main__':
    main()
