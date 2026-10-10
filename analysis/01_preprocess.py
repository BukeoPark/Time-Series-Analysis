"""전처리: 원본 거래 자료 → 일별 매출 테이블.

기준 문서: 전처리_기준.md (1~5단계)
실행: 프로젝트 루트에서 `python analysis/01_preprocess.py`
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "Online Retail.xlsx"
DAILY_PATH = ROOT / "data" / "processed" / "daily_sales.csv"
LOG_PATH = ROOT / "outputs" / "tables" / "preprocessing_exclusion_log.csv"

START = pd.Timestamp("2010-12-01")
END = pd.Timestamp("2011-12-08")  # 양 끝 포함
CUTOFF = pd.Timestamp("2011-12-09")  # 불완전한 마지막 날 시작 시각

UK_HOLIDAYS = pd.to_datetime([
    "2010-12-27", "2010-12-28", "2011-01-03", "2011-04-22", "2011-04-25",
    "2011-04-29", "2011-05-02", "2011-05-30", "2011-08-29",
])

RAW_COLUMNS = ["InvoiceNo", "StockCode", "Description", "Quantity",
               "InvoiceDate", "UnitPrice", "CustomerID", "Country"]


# ---------------------------------------------------------------- 1. 불러오기
def load_raw() -> pd.DataFrame:
    df = pd.read_excel(
        RAW_PATH, sheet_name="Online Retail",
        dtype={"InvoiceNo": str, "StockCode": str, "Description": str},
    )
    df["InvoiceNo"] = df["InvoiceNo"].str.strip()
    df["StockCode"] = df["StockCode"].str.strip()
    df["Description"] = df["Description"].str.strip()
    df["CustomerID"] = df["CustomerID"].astype("Int64").astype("string")

    df["amount"] = df["Quantity"] * df["UnitPrice"]
    df["is_cancel"] = df["InvoiceNo"].str.startswith("C")
    df["date"] = df["InvoiceDate"].dt.normalize()
    return df


# ---------------------------------------------------------------- 2. 행 제외
def apply_exclusions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    code = df["StockCode"].str.upper()  # 판별용. 원래 행과 인덱스를 공유
    rules = [
        ("2-1", "완전 중복 행", lambda d: d.duplicated(subset=RAW_COLUMNS, keep="first")),
        ("2-2", "회계 조정 코드 (AMAZONFEE, BANK CHARGES, CRUK, B)",
         lambda d: code[d.index].isin(["AMAZONFEE", "BANK CHARGES", "CRUK", "B"])),
        ("2-3", "수기 정정·할인·샘플 (M, D, S)",
         lambda d: code[d.index].isin(["M", "D", "S"])),
        ("2-4", "배송비 (POST, DOT, C2)",
         lambda d: code[d.index].isin(["POST", "DOT", "C2"])),
        ("2-5", "상품권 판매 (gift_*)",
         lambda d: code[d.index].str.startswith("GIFT_")),
        ("2-6", "단가 0 이하", lambda d: d["UnitPrice"] <= 0),
        ("2-7", "취소 아닌 음수 수량", lambda d: (d["Quantity"] < 0) & ~d["is_cancel"]),
        ("2-8", "불완전한 마지막 날 (2011-12-09 이후)",
         lambda d: d["InvoiceDate"] >= CUTOFF),
    ]

    log = [{"step": "0", "rule": "원본", "excluded_rows": 0, "excluded_amount": 0.0,
            "remaining_rows": len(df), "remaining_amount": df["amount"].sum()}]
    for step, rule, mask_fn in rules:
        mask = mask_fn(df).to_numpy(dtype=bool)
        excluded = df[mask]
        df = df[~mask]
        log.append({"step": step, "rule": rule,
                    "excluded_rows": len(excluded),
                    "excluded_amount": excluded["amount"].sum(),
                    "remaining_rows": len(df),
                    "remaining_amount": df["amount"].sum()})
    return df, pd.DataFrame(log)


# ---------------------------------------------------------------- 3. 일별 집계
def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    sale = df[~df["is_cancel"]]
    cancel = df[df["is_cancel"]]
    daily = pd.DataFrame({
        "net_sales": df.groupby("date")["amount"].sum(),
        "gross_sales": sale.groupby("date")["amount"].sum(),
        "cancel_amount": cancel.groupby("date")["amount"].sum(),
        "n_invoices": sale.groupby("date")["InvoiceNo"].nunique(),
        "n_cancel_invoices": cancel.groupby("date")["InvoiceNo"].nunique(),
        "n_lines": sale.groupby("date").size(),
        "quantity": sale.groupby("date")["Quantity"].sum(),
        "n_customers": sale.groupby("date")["CustomerID"].nunique(),  # 결측 제외
    })
    return daily


# ---------------------------------------------------------------- 4. 달력
def build_calendar(daily: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    dates = pd.date_range(START, END, freq="D", name="date")
    out = daily.reindex(dates).fillna(0)
    count_cols = ["n_invoices", "n_cancel_invoices", "n_lines", "quantity", "n_customers"]
    out[count_cols] = out[count_cols].astype("int64")

    raw_dates = set(raw["date"].unique())
    out["is_closed"] = (~dates.isin(raw_dates)).astype("int64")

    out["dow"] = dates.dayofweek
    out["is_sunday"] = (dates.dayofweek == 6).astype("int64")
    out["is_uk_holiday"] = dates.isin(UK_HOLIDAYS).astype("int64")
    out["year_month"] = dates.strftime("%Y-%m")
    month_start = dates.to_period("M").start_time
    month_end = dates.to_period("M").end_time.normalize()
    out["is_partial_month"] = ((month_start < START) | (month_end > END)).astype("int64")
    return out


# ---------------------------------------------------------------- 5. 이상치 표시
def flag_outliers(out: pd.DataFrame, threshold: float = 3.5) -> pd.DataFrame:
    open_days = out[out["is_closed"] == 0]
    if (open_days["net_sales"] <= 0).any():
        raise ValueError("영업일 net_sales에 0 이하 값이 있어 로그를 계산할 수 없음")
    y = np.log(open_days["net_sales"])
    grp = y.groupby(open_days["dow"])
    med = grp.transform("median")
    mad = grp.transform(lambda s: (s - s.median()).abs().median())
    z = 0.6745 * (y - med) / mad

    out["outlier_z"] = np.nan
    out.loc[z.index, "outlier_z"] = z
    out["is_outlier"] = 0
    out.loc[z.index[z.abs() > threshold], "is_outlier"] = 1
    return out


def main() -> None:
    raw = load_raw()
    raw_in_period = raw[(raw["date"] >= START) & (raw["date"] <= END)]

    clean, log = apply_exclusions(raw)
    daily = aggregate_daily(clean)

    # 검산: 일별 합계 = 남은 행 합계
    assert abs(daily["net_sales"].sum() - clean["amount"].sum()) < 1e-6
    assert np.allclose(daily["net_sales"],
                       daily["gross_sales"].fillna(0) + daily["cancel_amount"].fillna(0))

    out = build_calendar(daily, raw_in_period)
    out = flag_outliers(out)

    DAILY_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.reset_index().to_csv(DAILY_PATH, index=False, date_format="%Y-%m-%d")
    log.to_csv(LOG_PATH, index=False)

    with pd.option_context("display.float_format", "{:,.2f}".format,
                           "display.width", 200, "display.max_columns", 20):
        print(log.to_string(index=False))
        print(f"\n남은 행 중 CustomerID 결측: {clean['CustomerID'].isna().sum():,}")
        c = clean[clean["is_cancel"]]
        print(f"남은 행 중 취소 행: {len(c):,}건, {c['amount'].sum():,.2f}")
        print(f"\n일별 테이블: {len(out)}행, 영업일 {int((out['is_closed'] == 0).sum())}, "
              f"휴무일 {int(out['is_closed'].sum())}")
        print(f"영업일 net_sales 최소: {out.loc[out['is_closed'] == 0, 'net_sales'].min():,.2f}")
        print(f"휴무일(토요일 제외): "
              f"{out.index[(out['is_closed'] == 1) & (out['dow'] != 5)].strftime('%Y-%m-%d').tolist()}")
        print(f"\n이상치 {int(out['is_outlier'].sum())}일:")
        print(out.loc[out["is_outlier"] == 1, ["dow", "net_sales", "outlier_z"]].to_string())


if __name__ == "__main__":
    main()
