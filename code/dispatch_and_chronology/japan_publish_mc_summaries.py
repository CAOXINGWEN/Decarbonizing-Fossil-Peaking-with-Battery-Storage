"""Package exact seed results and report real vs randomized dates by baseline."""
import argparse
import shutil

from run_mc import (HERE, REAL_TABLE, RESULTS, STAGING, UIDS, N_RECORDS, QUANTILES, SEEDS,
                    Q50, checkpoint, file_digest, write_json, model)
import h5py
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment

MANUSCRIPT = HERE.parents[1] / "manuscript/results_integrated"
REAL_DATA = pd.read_csv(REAL_TABLE, dtype={"uid": str}).set_index(["q", "row_idx"]).sort_index()
CHUNK = 128
TOL = 1e-8


def locate_archive(q):
    return RESULTS / f"japan_MC_q{q:02d}_seeds.h5"


def read_checkpoint(q, idx):
    with np.load(checkpoint(q, idx), allow_pickle=False) as z:
        b, w = z["B"], z["lw"]
        if (int(z["q"]) != q or int(z["row"]) != idx or str(z["uid"]) != UIDS[idx]
            or b.shape != (SEEDS,) or w.shape != b.shape or not np.isfinite(b).all()
            or not np.issubdtype(w.dtype, np.integer) or (w<0).any()
            or abs(float(z["real_B"])-REAL_DATA.at[(q,idx),"B_usd_kw_yr"])>1e-6):
            raise ValueError(f"Invalid saved Monte Carlo case: q{q}/{idx}")
        return b,w


def pack(q):
    """Write all draws losslessly once; keep the working checkpoints on scratch."""
    target = locate_archive(q)
    other = STAGING / target.name
    for existing in [target, other]:
        if existing.is_file():
            with h5py.File(existing) as h:
                if h.attrs["q"] == q and h.attrs.get("complete", False) and h["B"].shape == (N_RECORDS,SEEDS):
                    return existing
    extra = STAGING / (target.name + ".part")
    with h5py.File(extra, "w") as h:
        h.attrs.update(q=q, complete=False, price_year=2035, real_rate=.036,
                       cost_definition="B = pure LFP - hybrid, USD per added kW per year")
        shape = (N_RECORDS, SEEDS)
        data = dict(chunks=(1,SEEDS), compression="gzip", compression_opts=4, shuffle=True, fletcher32=True)
        b_table = h.create_dataset("B", shape=shape, dtype="<f8", **data)
        windows_table = h.create_dataset("long_windows", shape=shape, dtype="<i2", **data)
        h.create_dataset("seed", data=np.arange(SEEDS, dtype=np.int16))
        h.create_dataset("uid", data=np.asarray(UIDS, dtype="S16"))
        for idx in range(N_RECORDS):
            b, windows = read_checkpoint(q,idx)
            b_table[idx] = b
            windows_table[idx] = windows
        h.attrs["complete"] = True
    extra.replace(other)
    if shutil.disk_usage(RESULTS).free > other.stat().st_size + 100*1024**2:
        tmp = target.with_suffix(".part")
        shutil.copyfile(other, tmp)
        if file_digest(other) != file_digest(tmp):
            raise IOError("Matrix copy failed its checksum")
        tmp.replace(target)
        return target
    return other


def block_stats(q, start, b, windows):
    nrow = len(b)
    assert b.shape == (nrow, SEEDS) and windows.shape == b.shape
    assert np.isfinite(b).all() and (windows>=0).all()
    records = REAL_DATA.loc[q].iloc[start:start+nrow]
    real_b = records.B_usd_kw_yr.to_numpy()
    low, mid, high = np.quantile(b,[.05,.5,.95],axis=1)
    winning = (b>0).sum(axis=1)
    diff = b - real_b[:,None]
    rank = ((diff < -TOL).sum(axis=1) + .5*(np.abs(diff)<=TOL).sum(axis=1))/SEEDS
    output = pd.DataFrame(dict(q=q, row_idx=np.arange(start,start+nrow), uid=records.uid.to_numpy(),
                             system_id=records.system_id.to_numpy(), fiscal_year=records.fiscal_year.to_numpy(),
                             prefecture=records.prefecture.to_numpy(), n_draws=SEEDS, real_B=real_b,
                             real_hybrid_cheaper=real_b>0, real_long_windows=records.long_windows.to_numpy(),
                             mc_winning_draws=winning, mc_win_fraction=winning/SEEDS,
                             mc_ties=(b==0).sum(axis=1), mc_B_mean=b.mean(axis=1), mc_B_p05=low,
                             mc_B_median=mid, mc_B_p95=high, real_B_percentile=rank,
                             mc_long_windows_median=np.median(windows,axis=1)))
    # Fixed costs change by technology year. The registered 3.6% service prices
    # and both dispatches are the same. Use the manuscript's largest-root rule.
    op_mc = b + model.RM.PREM36[2035]
    op_real = real_b + model.RM.PREM36[2035]
    # The ROI is monotone on the reporting branch. Exact medians need only
    # the two central order statistics after transforming each one separately.
    centre = np.partition(op_mc, [SEEDS//2-1, SEEDS//2], axis=1)[:,SEEDS//2-1:SEEDS//2+1]
    for year in model.RM.YEARS:
        premium = model.RM.PREM36[year]
        minimum = model.RM.curve(year)[1]
        output[f"mc_winning_draws_{year}"] = (op_mc>premium).sum(axis=1)
        real_roi = model.RM.roi(op_real,year)
        median_roi = model.RM.roi(centre,year).mean(axis=1)
        output[f"real_roi_{year}_pct"] = np.where(np.isfinite(real_roi),real_roi,np.nan)
        output[f"mc_median_roi_{year}_pct"] = np.where(np.isfinite(median_roi),median_roi,np.nan)
        output[f"mc_no_roi_root_draws_{year}"] = (op_mc<minimum).sum(axis=1)
    return output


def summarize_group(frame):
    probability = frame.mc_win_fraction.to_numpy()
    real = frame.real_hybrid_cheaper.to_numpy()
    return dict(records=len(frame), grids=frame.system_id.nunique(), observed_wins=int(real.sum()),
                observed_win_fraction=float(real.mean()), n_draws=int(frame.n_draws.sum()),
                mc_winning_draws=int(frame.mc_winning_draws.sum()), mc_win_fraction=float(probability.mean()),
                records_hybrid_wins_majority_draws=int((probability>.5).sum()),
                records_50_50=int((probability==.5).sum()), mc_ties=int(frame.mc_ties.sum()),
                real_B_median=float(frame.real_B.median()), mc_B_median_of_medians=float(frame.mc_B_median.median()),
                real_B_percentile_median=float(frame.real_B_percentile.median()),
                real_long_windows_median=float(frame.real_long_windows.median()),
                mc_long_windows_median_of_medians=float(frame.mc_long_windows_median.median()))


def finance_table(units):
    def ordered_median_with_missing(series):
        value=model.RM.ordered_quantile(series.fillna(-np.inf), [.5])[0]
        return float(value) if np.isfinite(value) else None
    rows=[]
    for q,g in units.groupby("q"):
        for year in model.RM.YEARS:
            difference=model.RM.PREM36[2035]-model.RM.PREM36[year]
            real=g.real_B+difference
            draws=int(g.n_draws.sum())
            wins=int(g[f"mc_winning_draws_{year}"].sum())
            rows.append(dict(q=q,price_year=year,records=len(g),observed_wins=int((real>0).sum()),
                             observed_win_fraction=float((real>0).mean()),n_draws=draws,
                             mc_winning_draws=wins,mc_win_fraction=wins/draws,
                             real_B_median=float(real.median()),
                             mc_B_median_of_medians=float(g.mc_B_median.median()+difference),
                             real_roi_pct=ordered_median_with_missing(g[f"real_roi_{year}_pct"]),
                             mc_roi_pct=ordered_median_with_missing(g[f"mc_median_roi_{year}_pct"]),
                             no_root_real=int(g[f"real_roi_{year}_pct"].isna().sum()),
                             no_root_draws=int(g[f"mc_no_roi_root_draws_{year}"].sum())))
    return pd.DataFrame(rows)


def summarise_q(q):
    pieces = []
    controls=pd.read_csv(RESULTS/"full_hourly_control_seeds.csv")
    controls=controls[controls.q==q]
    def check_controls(b, w):
        maximum=0.0
        for check in controls.itertuples(index=False):
            maximum=max(maximum,abs(float(b[check.row_idx,check.seed])-check.legacy_B))
            if int(w[check.row_idx,check.seed]) != check.long_windows:
                raise ValueError(f"Full-hourly window disagreement at {q}, {check.row_idx}, {check.seed}")
        if maximum>1e-6:
            raise ValueError(f"Full-hourly cost disagreement at q{q}: {maximum}")
        return maximum
    if q == 50:
        b = np.load(Q50/"japan_mc_B.npy", mmap_mode="r")
        w = np.load(Q50/"japan_mc_lw.npy", mmap_mode="r")
        assert b.shape == (N_RECORDS,SEEDS) and b.shape == w.shape
        check_controls(b, w)
        for i in range(0,N_RECORDS,CHUNK):
            pieces.append(block_stats(q,i,b[i:i+CHUNK],w[i:i+CHUNK]))
        archive = [Q50/"japan_mc_B.npy", Q50/"japan_mc_lw.npy"]
    else:
        path = pack(q)
        with h5py.File(path,"r") as h:
            assert np.array_equal(np.asarray(h["uid"]).astype(str), UIDS)
            assert np.array_equal(h["seed"][:],np.arange(SEEDS))
            check_controls(h["B"],h["long_windows"])
            for i in range(0,N_RECORDS,CHUNK):
                pieces.append(block_stats(q,i,h["B"][i:i+CHUNK],h["long_windows"][i:i+CHUNK]))
        archive = [path]
    units = pd.concat(pieces,ignore_index=True)
    assert len(units)==N_RECORDS and units.uid.is_unique and units.system_id.nunique()==5306
    data = [dict(path=str(p), sha256=file_digest(p)) for p in archive]
    return units, data


LABELS = {
    "q":"任务基线", "fiscal_year":"财政年度", "records":"年度记录数", "grids":"社区网格数",
    "observed_wins":"真实日期混合胜出记录数", "observed_win_fraction":"真实日期胜出比例",
    "n_draws":"随机放置总次数", "mc_winning_draws":"随机放置混合胜出次数", "mc_win_fraction":"随机放置胜出比例",
    "records_hybrid_wins_majority_draws":"过半放置混合胜出的记录数", "records_50_50":"恰半数放置胜出的记录数",
    "mc_ties":"成本相等放置次数", "real_B_median":"真实日期成本差中位数", "mc_B_median_of_medians":"随机成本差中位数的中位数",
    "real_B_percentile_median":"真实成本差在随机分布中的百分位中位数", "real_long_windows_median":"真实长窗口数中位数",
    "mc_long_windows_median_of_medians":"随机长窗口数中位数的中位数",
    "price_year":"技术成本年", "real_roi_pct":"真实增量回报率中位数（%）",
    "mc_roi_pct":"随机增量回报率中位数的中位数（%）",
    "no_root_real":"真实日期无回报率实根记录数","no_root_draws":"随机放置无回报率实根次数",
}


def write_book(summary, years, finance, target):
    notes = pd.DataFrame([
        ("研究对象","日本5,306个社区在2023与2024财政年度的10,612份记录。每个基线和每份记录的种子均为0至999。"),
        ("随机放置","按各基线独立扣除购电分位数、识别季节q75高需求日。在季节组内保留原有高需求事件块和普通日的各自顺序，将完整日的任务与光伏外送一同移动。两种组合使用同一次放置。"),
        ("成本口径","2035年技术成本和3.6%实际资金成本。纯LFP和混合组合共用4小时LFP基础层，新增层为4小时LFP或24小时A-CAES。成本差B等于纯LFP成本减去混合成本，单位为美元/(kW新增放电功率·年)。B大于零表示混合组合成本更低。"),
        ("随机胜出比例","每份记录有1,000次放置。以B大于零的放置次数除以全部放置次数，等于各记录胜出概率的算术平均。另列在超过500次放置中混合组合胜出的年度记录数。"),
        ("分布","逐记录成本分布保留第5、50和95百分位。真实成本差百分位按随机B小于真实B的比例计算，绝对差在10⁻⁸美元/(kW·年)以内计作并列并取半权重。"),
        ("成本年比较","固定基准服务价格及逐次优化调度，代入2025、2030和2035年对应的年化技术固定成本。增量回报率沿用稿件的最大实根。无实根排在分布下端，中位位置无实根时留空。"),
        ("中国负荷任务","省级总负荷高于同月同钟点基线的正偏差可定义负荷削峰任务。稿件现用的省级燃气调峰日任务与此对应不同服务对象。"),
        ("欧洲与日本","欧洲采用公共电网燃气净发电高于基线的部分，日本采用光伏社区购网电量高于基线的部分。日本低压双向计量将正向购网电与光伏反向送电分别记录，后者用作充电盈余。"),
        ("逐记录与逐次明细","japan_MC_per_record.csv.gz提供各年度记录的胜出概率、成本差分位数和长窗口数。complete.json列出每个基线的完整种子矩阵路径和校验值。"),
    ],columns=["项目","说明"])
    tmp = target.with_name(target.stem+".part.xlsx")
    with pd.ExcelWriter(tmp,engine="openpyxl") as writer:
        for name, sheet_data in [("全样本",summary),("分财政年度",years),("三个技术成本年",finance),("口径",notes)]:
            display = sheet_data.rename(columns=LABELS).copy()
            if "任务基线" in display:
                display["任务基线"] = display["任务基线"].map(lambda x:f"q{x}")
            display.to_excel(writer,sheet_name=name,index=False)
            sheet = writer.sheets[name]
            sheet.freeze_panes="B2"
            sheet.auto_filter.ref=sheet.dimensions
            sheet.row_dimensions[1].height=42
            for cell in sheet[1]:
                cell.font=Font(bold=True,color="FFFFFF")
                cell.fill=PatternFill("solid",fgColor="274B63")
                cell.alignment=Alignment(wrap_text=True,vertical="center",horizontal="center")
                sheet.column_dimensions[cell.column_letter].width=max(14,min(26,len(cell.value)*1.65))
                if cell.value in [LABELS[x] for x in ("observed_win_fraction","mc_win_fraction","real_B_percentile_median")]:
                    for col in sheet.iter_cols(min_col=cell.column,max_col=cell.column,min_row=2):
                        for x in col: x.number_format="0.000%"
            if name=="口径":
                sheet.column_dimensions["B"].width=100
                for i in range(2,sheet.max_row+1):
                    sheet.cell(i,2).alignment=Alignment(wrap_text=True,vertical="center")
                    sheet.row_dimensions[i].height=45
    tmp.replace(target)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--q50-only",action="store_true")
    args=parser.parse_args()
    levels = (50,) if args.q50_only else QUANTILES
    all_units, files = [], {}
    for q in levels:
        units, files[q] = summarise_q(q)
        all_units.append(units)
    units=pd.concat(all_units,ignore_index=True)
    summary=pd.DataFrame([dict(q=q,**summarize_group(g)) for q,g in units.groupby("q")])
    years=pd.DataFrame([dict(q=q,fiscal_year=y,**summarize_group(g)) for (q,y),g in units.groupby(["q","fiscal_year"])])
    finance=finance_table(units)
    orig=pd.read_csv(Q50/"japan_primary_summary.csv").set_index("year")
    check50=finance[finance.q==50].set_index("price_year")
    assert np.array_equal(check50.observed_wins,orig["real_win_3.6"])
    assert np.allclose(check50.mc_win_fraction,orig["mc_win_3.6"],rtol=0,atol=1e-12)
    assert np.allclose(check50.real_roi_pct,orig.real_roi_med,rtol=0,atol=1e-8)
    assert np.allclose(check50.mc_roi_pct,orig.mc_roi_med,rtol=0,atol=1e-8)
    if args.q50_only:
        summary.to_csv(RESULTS/"q50_verified_summary.csv",index=False)
        print(summary.to_string(index=False),flush=True)
        return
    assert len(units)==N_RECORDS*len(levels) and units.n_draws.sum()==74284000
    assert not units.duplicated(["q","uid"]).any()
    summary.to_csv(RESULTS/"japan_MC_baseline_summary.csv",index=False)
    years.to_csv(RESULTS/"japan_MC_by_fiscal_year.csv",index=False)
    finance.to_csv(RESULTS/"japan_MC_by_technology_year.csv",index=False)
    units.to_csv(RESULTS/"japan_MC_per_record.csv.gz",index=False,compression="gzip")
    output=MANUSCRIPT/"日本10612份_各基线1000次蒙特卡洛.xlsx"
    write_book(summary, years, finance, output)
    book=load_workbook(output,read_only=True,data_only=True)
    contents=iter(book["全样本"].values)
    header=next(contents)
    check=pd.DataFrame(list(contents),columns=header)
    assert list(check[LABELS["q"]])==[f"q{q}" for q in levels]
    assert np.array_equal(check[LABELS["mc_winning_draws"]],summary.mc_winning_draws)
    assert (check[LABELS["n_draws"]]==N_RECORDS*SEEDS).all()
    book.close()
    write_json(RESULTS/"complete.json",dict(status="complete",records=N_RECORDS,seeds=SEEDS,baselines=list(levels),
                                           total_placements=int(units.n_draws.sum()),excel=str(output),matrices=files))
    print(summary.to_string(index=False),flush=True)


if __name__ == "__main__":
    main()
