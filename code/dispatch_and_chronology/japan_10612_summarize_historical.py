"""Validate the real-date rerun and write the analysis tables and reading workbook."""
from pathlib import Path
import hashlib
import json

from run_real import WORK, ORIGINAL, QUANTILES, N_RECORDS, SHARD_SIZE, shard_path, model
import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill

OUT = WORK / "results"
MANUSCRIPT_DIR = WORK.parents[1] / "manuscript/results_integrated"
PRICE_YEAR = 2035
CORRECT = model.cp.COPT.OPTIMAL


def load_and_check():
    shards = [shard_path(i) for i in range(0, N_RECORDS, SHARD_SIZE)]
    frame = pd.concat([pd.read_csv(p, dtype={"uid": str}) for p in shards], ignore_index=True)
    frame.sort_values(["row_idx", "q"], inplace=True, ignore_index=True)
    expected = pd.MultiIndex.from_product([range(N_RECORDS), QUANTILES], names=["row_idx", "q"])
    assert pd.MultiIndex.from_frame(frame[["row_idx", "q"]]).equals(expected)
    assert not frame.duplicated(["uid", "q"]).any() and not frame.isna().any().any()
    assert frame.system_id.nunique() == 5306 and set(frame.fiscal_year) == {2023, 2024}
    assert (frame.groupby(["q", "system_id"]).fiscal_year.nunique() == 2).all()
    assert (frame[["lfp_status", "hybrid_status"]] == CORRECT).all().all()
    assert np.isfinite(frame.select_dtypes("number").to_numpy()).all()
    assert np.allclose(frame.lfp_cost_usd_kw_yr - frame.hybrid_cost_usd_kw_yr,
                       frame.B_usd_kw_yr, rtol=0, atol=1e-8)
    assert (frame.groupby("uid").ref_power_mw.nunique() == 1).all()
    assert (frame.groupby("row_idx").task_mwh.diff().dropna() <= 1e-8).all()
    assert (frame.groupby("row_idx").charge_hours.diff().dropna() >= 0).all()

    archived = pd.read_csv(ORIGINAL / "results/full_10612/japan_records.csv.gz", dtype={"uid": str})
    control = frame.query("q == 50").merge(archived, on=["row_idx", "uid", "system_id"],
                                           validate="one_to_one", suffixes=("", "_original"))
    assert len(control) == N_RECORDS and (control.fiscal_year == control.observation_year).all()
    assert ((control.B_usd_kw_yr > 0) == (control.B > 0)).all()
    q50_max_error = float((control.B_usd_kw_yr-control.B).abs().max())
    assert q50_max_error < 1e-5, q50_max_error
    for col in ("task_hours", "charge_hours", "event_days", "events", "long_windows"):
        assert (control[col] == control[col + "_original"]).all(), col

    independent = pd.read_csv(OUT / "reference_8760h_spot_checks.csv").merge(
        frame, on=["row_idx", "q"], validate="one_to_one", suffixes=("_ref", ""))
    assert len(independent) == len(QUANTILES)
    assert (independent.event_days == independent.event_days_ref).all()
    assert (independent.charge_hours == independent.charge_hours_ref).all()
    errors = [(independent.reference_lfp_cost - independent.lfp_cost_usd_kw_yr).abs(),
              (independent.reference_hybrid_cost - independent.hybrid_cost_usd_kw_yr).abs(),
              (independent.reference_B - independent.B_usd_kw_yr).abs()]
    check_max = max(float(e.max()) for e in errors)
    assert check_max < 1e-5, check_max

    frame["hybrid_cheaper"] = frame.B_usd_kw_yr > 0
    savings = frame.B_usd_kw_yr.to_numpy() + model.RM.PREM36[PRICE_YEAR]
    for year in model.RM.YEARS:
        roi = model.RM.roi(savings, year)
        frame[f"roi_{year}_pct"] = np.where(np.isfinite(roi), roi, np.nan)
        frame[f"no_roi_root_{year}"] = ~np.isfinite(roi)
    receipt = dict(community_years=N_RECORDS, distinct_communities=5306,
                   fiscal_years=[2023, 2024], q_new=list(QUANTILES[:-1]),
                   new_record_q_cases=N_RECORDS * (len(QUANTILES)-1),
                   optimized_portfolios_per_case=2, solver_status_all_optimal=True,
                   replicated_q50_cases=N_RECORDS, q50_B_max_abs_difference=q50_max_error,
                   q50_sign_disagreements=int(((control.B_usd_kw_yr>0)!=(control.B>0)).sum()),
                   independent_full_hourly_checks=len(independent), independent_cost_max_abs_difference=check_max,
                   exact_cost_ties=int((frame.B_usd_kw_yr==0).sum()), solver="COPT",
                   chronological_order="observed", fiscal_year_length_hours=8760, financing_rate=0.036,
                   usd_per_eur=model.FX, power_share_of_annual_mean_import=0.01)
    return frame, receipt


def group_summary(frame, keys):
    summary = frame.groupby(keys, as_index=False).agg(
        records=("uid", "size"), grids=("system_id", "nunique"),
        hybrid_wins=("hybrid_cheaper", "sum"), hybrid_share=("hybrid_cheaper", "mean"),
        B_median=("B_usd_kw_yr", "median"), B_p25=("B_usd_kw_yr", lambda x: x.quantile(.25)),
        B_p75=("B_usd_kw_yr", lambda x: x.quantile(.75)),
        task_share_median=("task_share_of_import", "median"),
        task_hours_median=("task_hours", "median"), charge_hours_median=("charge_hours", "median"),
        mean_event_days_median=("mean_event_days", "median"), long_windows_median=("long_windows", "median"),
        acaes_efc_median=("acaes_efc", "median"), lfp_efc_median=("lfp_efc", "median"))
    return summary


def finance_table(frame):
    rows = []
    for q, group in frame.groupby("q"):
        savings = group.B_usd_kw_yr.to_numpy() + model.RM.PREM36[PRICE_YEAR]
        for year in model.RM.YEARS:
            net = savings - model.RM.PREM36[year]
            roi = group[f"roi_{year}_pct"].fillna(-np.inf)
            median_roi = model.RM.ordered_quantile(roi, [.5])[0]
            rows.append(dict(q=q, cost_year=year, records=len(group), hybrid_wins=int((net>0).sum()),
                             hybrid_share=float((net>0).mean()), B_median=float(np.median(net)),
                             roi_median_pct=float(median_roi) if np.isfinite(median_roi) else None,
                             records_without_roi_root=int(group[f"no_roi_root_{year}"].sum())))
    return pd.DataFrame(rows)


LABELS = {
    "q": "任务基线", "fiscal_year": "财政年度", "records": "年度记录数", "grids": "网格数",
    "hybrid_wins": "混合成本较低（份）", "hybrid_share": "混合成本较低比例",
    "B_median": "成本差中位数", "B_p25": "成本差第25百分位", "B_p75": "成本差第75百分位",
    "task_share_median": "任务电量占购电量中位数", "task_hours_median": "任务小时中位数",
    "charge_hours_median": "可充电小时中位数", "mean_event_days_median": "事件天数均值的中位数",
    "long_windows_median": "长充电窗口数中位数", "acaes_efc_median": "A-CAES等效满循环中位数",
    "lfp_efc_median": "新增LFP等效满循环中位数", "cost_year": "技术成本年",
    "roi_median_pct": "增量回报率中位数（%）", "records_without_roi_root": "无盈亏平衡实根（份）",
}


def add_sheet(writer, name, frame, freeze="B2"):
    shown = frame.rename(columns=LABELS).copy()
    if "任务基线" in shown:
        shown["任务基线"] = shown["任务基线"].map(lambda q: f"q{q}")
    shown.to_excel(writer, sheet_name=name, index=False)
    sheet = writer.sheets[name]
    sheet.freeze_panes = freeze
    sheet.auto_filter.ref = sheet.dimensions
    sheet.sheet_view.showGridLines = False
    sheet.row_dimensions[1].height = 36
    for cell in sheet[1]:
        title = str(cell.value)
        cell.fill = PatternFill("solid", fgColor="264760")
        cell.font = Font(name="Calibri", color="FFFFFF", bold=True, size=11)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        width = min(34, max(14, len(title) * 1.9 + 3))
        if title == "社区网格":
            width = 34
        sheet.column_dimensions[cell.column_letter].width = width
    return sheet


def format_numbers(sheet, frame, wide=False):
    for index, name in enumerate(frame.columns, 1):
        if name in ("hybrid_share", "task_share_median"):
            fmt = "0.00%"
        elif name in ("B_median", "B_p25", "B_p75", "roi_median_pct", "acaes_efc_median",
                      "lfp_efc_median", "mean_event_days_median") or (wide and isinstance(name, str) and name.startswith("q")):
            fmt = '0.00;[Red](0.00);0.00'
        elif name in ("charge_hours_median", "task_hours_median", "long_windows_median"):
            fmt = '#,##0.##'
        else:
            continue
        for col in sheet.iter_cols(min_col=index, max_col=index, min_row=2):
            for cell in col:
                cell.number_format = fmt


def write_workbook(frame, overall, annual, finance, receipt):
    output = MANUSCRIPT_DIR / "日本10612份真实日期_任务基线敏感性.xlsx"
    temporary = output.with_name(output.stem + ".part.xlsx")
    wide = frame.pivot(index=["row_idx", "system_id", "prefecture", "fiscal_year", "uid"],
                       columns="q", values="B_usd_kw_yr").reset_index()
    wide.rename(columns={"system_id": "社区网格", "prefecture": "都县", "fiscal_year": "财政年度",
                         **{q: f"q{q} 成本差" for q in QUANTILES}}, inplace=True)
    wide.drop(columns="row_idx", inplace=True)
    notes = pd.DataFrame([
        ("样本", "埼玉、千叶、东京、神奈川、静冈的5,306个500米网格，各有2023和2024两个完整财政年度，按原始日期顺序计算。"),
        ("任务", "每个网格和财政年度内，按月份和钟点求购电量分位数。购电超过分位数的部分构成任务。各q分别按季节日任务的第75百分位识别高需求日（含等号）。"),
        ("充电与功率", "每个情景使用原时点的光伏外送，允许在任务为零且有外送时充电。两支路各取同一记录年均购电功率的1%。"),
        ("组合与成本", "两组共有4小时LFP基础层，新增层分别为4小时LFP和24小时A-CAES。主表使用2035年技术成本和3.6%实际资金成本。"),
        ("成本差", "纯LFP年化系统成本减去混合组合年化系统成本，单位为美元/(kW新增放电功率·年)。正值表示混合组合成本较低，负值表示纯LFP成本较低。"),
        ("统计口径", "按社区年度逐条计数。主表的比例、分位数和中位数均在相同的10,612份记录中计算。分财政年度表的分母均为5,306。"),
        ("成本年与回报率", "各技术成本年的比较沿用已优化的运行节约和相应的年化固定成本。回报率取盈亏平衡方程较大的实根，无实根的记录计入无根数并按原稿定义排在分布下端。"),
        ("q50复核", f"逐记录核对现行q50归档，最大成本差偏差为{receipt['q50_B_max_abs_difference']:.3g}美元/(kW·年)。独立8,760小时模型抽查7组，最大绝对成本或成本差偏差为{receipt['independent_cost_max_abs_difference']:.3g}美元/(kW·年)。"),
        ("中国新资料", "Yi等由典型负荷、年度用电量与逐时气象资料重建2015—2024年各省历史总用电负荷。它可用于构造总负荷削峰任务。"),
        ("中国稿件现用资料", "分省燃气调峰机组的日调峰电量按典型工作日负荷曲线分至各小时，充电盈余来自站点弃风弃光。"),
        ("欧洲原始变量", "Energy-Charts公共电网净发电数据集中的Fossil gas按燃料类别汇总各国燃气发电。其高于分位线的部分构成调峰任务。"),
        ("日本原始变量", "一桥大学低压自发自用数据按500米网格汇总。原始半小时字段「電力量_供給」为电网向用户供电，「電力量_受電」为用户向电网反向送电。稿件用前者构造任务，用后者表示光伏外送盈余。购电字段只给电量和方向。"),
        ("中国数据论文", "https://www.nature.com/articles/s41597-026-07327-8"),
        ("欧洲数据接口", "https://api.energy-charts.info/openapi.json"),
        ("日本计量口径", "https://www.enecho.meti.go.jp/category/electricity_and_gas/electric/measure/faq/010.html"),
        ("全量明细", str((OUT / "japan_real_baseline_records.csv.gz").relative_to(WORK.parents[1]))),
    ], columns=["项目", "说明"])
    with pd.ExcelWriter(temporary, engine="openpyxl") as writer:
        for name, table in [("全样本2035", overall), ("分财政年度2035", annual), ("三个技术成本年", finance)]:
            sheet = add_sheet(writer, name, table)
            format_numbers(sheet, table)
        sheet = add_sheet(writer, "各记录成本差2035", wide, freeze="E2")
        format_numbers(sheet, wide, wide=True)
        sheet = add_sheet(writer, "变量与口径", notes)
        sheet.column_dimensions["A"].width = 23
        sheet.column_dimensions["B"].width = 105
        for row in range(2, sheet.max_row+1):
            sheet.row_dimensions[row].height = 37
            sheet.cell(row, 2).alignment = Alignment(wrap_text=True, vertical="center")
    temporary.replace(output)
    return output


def main():
    frame, receipt = load_and_check()
    overall = group_summary(frame, ["q"])
    annual = group_summary(frame, ["q", "fiscal_year"])
    finance = finance_table(frame)
    frame.to_csv(OUT / "japan_real_baseline_records.csv.gz", index=False, compression="gzip")
    overall.to_csv(OUT / "japan_real_baseline_summary.csv", index=False)
    finance.to_csv(OUT / "japan_real_baseline_finance.csv", index=False)
    receipt["annual_import_array_sha256"] = file_hash(ORIGINAL / "data/japan_dual_hourly.npy")
    receipt["original_q50_records_sha256"] = file_hash(ORIGINAL / "results/full_10612/japan_records.csv.gz")
    receipt["hourly_metadata_sha256"] = file_hash(ORIGINAL / "data/japan_dual_metadata.csv")
    receipt["model_source_sha256"] = file_hash(Path(model.__file__))
    (OUT / "verification.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    output = write_workbook(frame, overall, annual, finance, receipt)
    print(overall.to_string(index=False), flush=True)
    print(json.dumps(receipt, ensure_ascii=False), output, sep="\n", flush=True)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
