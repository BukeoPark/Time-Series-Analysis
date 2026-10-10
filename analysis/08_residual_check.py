"""분해 잔차에서 점검이 필요한 날짜를 뽑고, 거래 단위로 원인을 확인한다.

입력:
  outputs/tables/daily_decomposition.csv (07_decomposition.py 산출물)
  data/raw/Online Retail.xlsx (01_preprocess.py의 전처리 함수를 그대로 적용)
출력:
  outputs/tables/residual_check_dates.csv      점검 대상 날짜와 원인 지표
  outputs/tables/residual_check_invoices.csv   점검 대상 날짜의 상위 송장
실행: 프로젝트 루트에서 `python analysis/08_residual_check.py`

점검 대상 기준 (영업일만. 추세가 계산된 날만)
- 상방: 잔차의 로버스트 z = 0.6745 × (잔차 − 중앙값) / MAD 가 3 초과
- 하방: 잔차 / 추세 < −0.7  (매출이 추세+요일 효과 기대치보다 추세의 70% 이상 낮음)
  잔차 분포가 오른쪽으로 치우쳐 z 기준으로는 하방이 잡히지 않으므로 상대 기준을 쓴다.
- 휴무일(토요일 제외)은 모형이 0을 예상하지 못해 생기는 잔차이므로 따로 '휴무' 유형으로 표시한다.
"""
import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEC_PATH = ROOT / "outputs" / "tables" / "daily_decomposition.csv"
DATES_PATH = ROOT / "outputs" / "tables" / "residual_check_dates.csv"
INVOICES_PATH = ROOT / "outputs" / "tables" / "residual_check_invoices.csv"

Z_THRESHOLD = 3.0
REL_THRESHOLD = -0.7
DOW_NAMES = ["월", "화", "수", "목", "금", "토", "일"]


def load_clean() -> pd.DataFrame:
    spec = importlib.util.spec_from_file_location("preprocess", ROOT / "analysis" / "01_preprocess.py")
    pp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pp)
    clean, _ = pp.apply_exclusions(pp.load_raw())
    return clean


def flag_dates(dec: pd.DataFrame) -> pd.DataFrame:
    o = dec[(dec["is_closed"] == 0) & dec["residual"].notna()].copy()
    med = o["residual"].median()
    mad = (o["residual"] - med).abs().median()
    o["robust_z"] = 0.6745 * (o["residual"] - med) / mad
    o["resid_to_trend"] = o["residual"] / o["trend"]
    o["type"] = None
    o.loc[o["robust_z"] > Z_THRESHOLD, "type"] = "상방 급등"
    o.loc[o["resid_to_trend"] < REL_THRESHOLD, "type"] = "하방 급락"

    closed = dec[(dec["is_closed"] == 1) & (dec["dow"] != 5)].copy()
    closed["type"] = "휴무(토요일 외)"
    closed["resid_to_trend"] = closed["residual"] / closed["trend"]
    return pd.concat([o[o["type"].notna()], closed]).sort_values("date")


def profile(flags: pd.DataFrame, clean: pd.DataFrame, dec: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sale = clean[~clean["is_cancel"]]
    open_dates = dec.loc[dec["is_closed"] == 0, "date"]

    # 같은 요일 영업일의 중앙값 (비교 기준)
    per_day = sale.groupby("date").agg(n_invoices=("InvoiceNo", "nunique"),
                                       n_customers=("CustomerID", "nunique"))
    per_day["dow"] = per_day.index.dayofweek
    per_day = per_day[per_day.index.isin(open_dates)]
    dow_med = per_day.groupby("dow")[["n_invoices", "n_customers"]].median()

    inv = (clean.groupby(["date", "InvoiceNo"])
           .agg(amount=("amount", "sum"), customer=("CustomerID", "first"),
                country=("Country", "first"), n_lines=("amount", "size"),
                quantity=("Quantity", "sum"))
           .reset_index())

    rows, top_inv = [], []
    for _, f in flags.iterrows():
        day = clean[clean["date"] == f["date"]]
        s = day[~day["is_cancel"]]
        gross = s["amount"].sum()
        row = {"date": f["date"], "weekday": DOW_NAMES[f["dow"]], "type": f["type"],
               "net_sales": f["net_sales"], "trend": f["trend"],
               "weekday_effect": f["weekday_effect"], "residual": f["residual"],
               "robust_z": f.get("robust_z"), "resid_to_trend": f["resid_to_trend"]}
        if len(s):
            d_inv = inv[inv["date"] == f["date"]].sort_values("amount", ascending=False)
            top = d_inv.iloc[0]
            by_cust = s.groupby("CustomerID")["amount"].sum().sort_values(ascending=False)
            by_country = s.groupby("Country")["amount"].sum().sort_values(ascending=False)
            non_uk = by_country.drop("United Kingdom", errors="ignore")
            by_prod = s.groupby("Description")["amount"].sum().sort_values(ascending=False)
            row.update({
                "n_invoices": s["InvoiceNo"].nunique(),
                "dow_median_invoices": dow_med.loc[f["dow"], "n_invoices"],
                "n_customers": s["CustomerID"].nunique(),
                "cancel_amount": day.loc[day["is_cancel"], "amount"].sum(),
                "top_invoice": top["InvoiceNo"], "top_invoice_amount": top["amount"],
                "top_invoice_share": top["amount"] / gross,
                "top3_invoice_share": d_inv["amount"].head(3).sum() / gross,
                "top_customer": by_cust.index[0] if len(by_cust) else None,
                "top_customer_share": by_cust.iloc[0] / gross if len(by_cust) else None,
                "no_customer_id_share": s.loc[s["CustomerID"].isna(), "amount"].sum() / gross,
                "non_uk_share": non_uk.sum() / gross,
                "top_non_uk_country": non_uk.index[0] if len(non_uk) else None,
                "top_product": by_prod.index[0], "top_product_share": by_prod.iloc[0] / gross,
            })
            top_inv.append(d_inv.head(3).assign(share_of_day_gross=lambda x: x["amount"] / gross))
        rows.append(row)
    return pd.DataFrame(rows), pd.concat(top_inv, ignore_index=True)


def main() -> None:
    dec = pd.read_csv(DEC_PATH, parse_dates=["date"])
    flags = flag_dates(dec)
    clean = load_clean()
    dates, invoices = profile(flags, clean, dec)

    DATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    dates.to_csv(DATES_PATH, index=False, date_format="%Y-%m-%d")
    invoices.to_csv(INVOICES_PATH, index=False, date_format="%Y-%m-%d")

    print(f"기준: 상방 robust z > {Z_THRESHOLD}, 하방 잔차/추세 < {REL_THRESHOLD}, 토요일 외 휴무일")
    print(dates["type"].value_counts().to_string())
    with pd.option_context("display.width", 300, "display.max_columns", 40,
                           "display.float_format", "{:,.2f}".format):
        print(dates.drop(columns=["weekday_effect"]).to_string(index=False))
        print(invoices.to_string(index=False))


if __name__ == "__main__":
    main()
