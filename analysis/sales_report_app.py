"""온라인 소매 매출 분석 보고서 (Streamlit).

analysis/01~07, 10~15 스크립트가 만든 산출물을 읽어 보여준다.
8·9장(예측)은 모형 선택, 평가 기간 선택에 따라 오차 지표를 앱에서 다시 계산한다.
사이드바에서 분석 기간을 고르면 요약 지표, 일별·월별·요일별·카테고리별 집계를
그 기간의 일별 자료로 다시 계산한다 (daily_sales.csv, category_daily_sales.csv 기준).
- 전처리 기록과 시계열 분해(추세·요일 효과)는 전체 기간으로 계산한 값을 그대로 쓰고,
  분해 그래프는 선택 기간에 해당하는 날짜만 보여준다.
- 7일 후행 이동평균은 전체 기간으로 계산한 뒤 잘라서 보여준다 (기간 시작일에도 직전 7일을 반영).

실행 (프로젝트 루트에서):
    python analysis/01_preprocess.py          # 산출물이 없을 때 01~07, 10~15 순서대로 먼저 실행 (13은 15 다음)
    ...
    streamlit run analysis/sales_report_app.py
"""
from datetime import date
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"

BLUE = "#2a78d6"
LIGHT_BLUE = "#b9d3f2"
ORANGE = "#eb6834"
GRAY = "#b5b3ac"
DOW_NAMES = ["월", "화", "수", "목", "금", "토", "일"]
OPEN_DOW_ORDER = [n for n in DOW_NAMES if n != "토"]

REQUIRED = {
    "daily": PROCESSED / "daily_sales.csv",
    "log": TABLES / "preprocessing_exclusion_log.csv",
    "category": TABLES / "category_sales_summary.csv",
    "category_daily": TABLES / "category_daily_sales.csv",
    "decomposition": TABLES / "daily_decomposition.csv",
    "ses_params": TABLES / "ses_params.csv",
    "holt_params": TABLES / "holt_params.csv",
    "hw_params": TABLES / "hw_params.csv",
    "sarima_params": TABLES / "sarima_params.csv",
    "sarima_search": TABLES / "sarima_order_search.csv",
    "fc_nov": TABLES / "all_models_forecast_nov.csv",
    "params_dec": TABLES / "es_models_params_dec.csv",
    "fc_dec": TABLES / "es_models_forecast_dec.csv",
}
MODELS = ["SES", "Holt", "Holt-Winters", "SARIMA"]
MODEL_COLORS = {"SES": ORANGE, "Holt": "#1baf7a", "Holt-Winters": "#4a3aa7", "SARIMA": "#e87ba4"}
MODEL_NAMES_KO = {"SES": "단순 지수평활", "Holt": "Holt (선형 추세)", "Holt-Winters": "Holt-Winters (추세+요일)",
                  "SARIMA": "SARIMA (요일 계절)"}


@st.cache_data
def load() -> dict[str, pd.DataFrame]:
    d = {k: pd.read_csv(p) for k, p in REQUIRED.items()}
    for k in ["daily", "category_daily", "decomposition", "fc_nov", "fc_dec"]:
        d[k]["date"] = pd.to_datetime(d[k]["date"])
    daily = d["daily"]
    daily["weekday"] = daily["dow"].map(lambda i: DOW_NAMES[i])
    daily["ma7"] = daily["net_sales"].rolling(window=7, min_periods=7).mean()
    return d


def gbp(v: float, digits: int = 0) -> str:
    return f"£{v:,.{digits}f}"


def pct(v: float, digits: int = 1) -> str:
    return f"{v:.{digits}%}"


def monthly_summary(sel: pd.DataFrame) -> pd.DataFrame:
    g = sel.groupby("year_month")
    out = pd.DataFrame({
        "net_sales_total": g["net_sales"].sum(),
        "n_days": g.size(),
        "n_open_days": g["is_closed"].apply(lambda s: int((s == 0).sum())),
    })
    out["days_in_month"] = [pd.Period(ym).days_in_month for ym in out.index]
    out["is_partial"] = (out["n_days"] < out["days_in_month"]).astype("int64")
    out["avg_per_day"] = out["net_sales_total"] / out["n_days"]
    out["avg_per_open_day"] = out["net_sales_total"] / out["n_open_days"].replace(0, np.nan)
    return out.reset_index()


def weekday_summary(sel: pd.DataFrame) -> pd.DataFrame:
    """05_weekday_sales.py와 같은 정의를 선택 기간에 적용 (영업일만)."""
    o = sel[sel["is_closed"] == 0].copy()
    week = o["date"].dt.to_period("W-SAT")  # 일요일 시작 주
    o["week_index"] = o["net_sales"] / o.groupby(week)["net_sales"].transform("mean")
    g = o.groupby("dow")
    s = pd.DataFrame({
        "n_open_days": g.size(),
        "mean_net_sales": g["net_sales"].mean(),
        "median_net_sales": g["net_sales"].median(),
        "total_net_sales": g["net_sales"].sum(),
        "mean_n_invoices": g["n_invoices"].mean(),
        "mean_week_index": g["week_index"].mean(),
    })
    s["share_of_total"] = s["total_net_sales"] / sel["net_sales"].sum()
    s["mean_of_weekday_means"] = s["mean_net_sales"].mean()
    s["diff_from_weekday_mean"] = s["mean_net_sales"] - s["mean_of_weekday_means"]
    s["weekday"] = [DOW_NAMES[i] for i in s.index]
    return s.reset_index()


def category_summary(cd: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.Timestamp]:
    n_days = (end - start).days + 1
    mid = start + pd.Timedelta(days=n_days // 2)  # 후반부 시작일
    g = cd.groupby("category")
    s = pd.DataFrame({
        "net_sales": g["net_sales"].sum(),
        "gross_sales": g["gross_sales"].sum(),
        "cancel_amount": g["cancel_amount"].sum(),
    })
    s["share"] = s["net_sales"] / s["net_sales"].sum()
    s["cancel_rate"] = -s["cancel_amount"] / s["gross_sales"]
    s["daily_avg"] = s["net_sales"] / n_days
    early = cd[cd["date"] < mid].groupby("category")["net_sales"].sum() / (mid - start).days
    late = cd[cd["date"] >= mid].groupby("category")["net_sales"].sum() / ((end - mid).days + 1)
    s["growth_half"] = late.reindex(s.index).fillna(0) / early.reindex(s.index) - 1
    if n_days < 28:  # 전반부·후반부가 각각 2주 미만이면 비교하지 않음
        s["growth_half"] = np.nan
    return s.reset_index(), mid


def forecast_metrics(actual: pd.Series, pred: pd.Series) -> dict:
    """10~15번 스크립트의 metrics()와 같은 정의."""
    err = actual - pred
    nz = actual != 0
    return {
        "actual_total": actual.sum(), "forecast_total": pred.sum(),
        "total_error_pct": pred.sum() / actual.sum() - 1 if actual.sum() else np.nan,
        "MAE": err.abs().mean(), "RMSE": np.sqrt((err ** 2).mean()),
        "bias_mean_error": (pred - actual).mean(),
        "MAPE": (err[nz].abs() / actual[nz]).mean() if nz.all() else np.nan,
    }


def model_color_scale(models: list[str]) -> alt.Color:
    return alt.Color("model:N", title=None, sort=MODELS,
                     scale=alt.Scale(domain=models, range=[MODEL_COLORS[m] for m in models]),
                     legend=alt.Legend(orient="top"))


def forecast_chart(df: pd.DataFrame, models: list[str], band: bool = False, height: int = 360):
    """실제(막대, 휴무일 회색) + 모형별 예측(선) + 선택 시 SARIMA 95% 예측구간."""
    bars = df.assign(구분=np.where(df["is_closed"] == 1, "휴무일 (실제 0)", "실제 순매출"))
    layers = []
    if band:
        layers.append(alt.Chart(df).mark_area(color="#f6dbe6", opacity=0.8).encode(
            x="date:T", y="SARIMA_lower95:Q", y2="SARIMA_upper95:Q",
            tooltip=[alt.Tooltip("date:T", title="날짜", format="%Y-%m-%d"),
                     alt.Tooltip("SARIMA_lower95:Q", title="SARIMA 하한 (£)", format=",.0f"),
                     alt.Tooltip("SARIMA_upper95:Q", title="SARIMA 상한 (£)", format=",.0f")]))
    layers.append(alt.Chart(bars).mark_bar(opacity=0.9).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(format="%m-%d")),
        y=alt.Y("actual:Q", title="순매출 (£)", axis=alt.Axis(format=",.0f")),
        color=alt.Color("구분:N", title=None, scale=alt.Scale(domain=["실제 순매출", "휴무일 (실제 0)"],
                                                            range=[LIGHT_BLUE, GRAY]),
                        legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("date:T", title="날짜", format="%Y-%m-%d"), alt.Tooltip("weekday:N", title="요일"),
                 alt.Tooltip("actual:Q", title="실제 (£)", format=",.2f")]))
    long = df.melt(id_vars=["date", "weekday"], value_vars=[f"{m}_forecast" for m in models],
                   var_name="model", value_name="forecast")
    long["model"] = long["model"].str.replace("_forecast", "", regex=False)
    layers.append(alt.Chart(long).mark_line(strokeWidth=2.2, point=alt.OverlayMarkDef(size=22)).encode(
        x="date:T", y="forecast:Q",
        stroke=alt.Stroke("model:N", title=None, sort=MODELS,
                          scale=alt.Scale(domain=models, range=[MODEL_COLORS[m] for m in models]),
                          legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("date:T", title="날짜", format="%Y-%m-%d"), alt.Tooltip("weekday:N", title="요일"),
                 alt.Tooltip("model:N", title="모형"), alt.Tooltip("forecast:Q", title="예측 (£)", format=",.2f")]))
    return alt.layer(*layers).resolve_scale(color="independent").properties(height=height)


# ---------------------------------------------------------------- 페이지
st.set_page_config(page_title="온라인 소매 매출 분석 보고서", layout="wide")

missing = [str(p.relative_to(ROOT)) for p in REQUIRED.values() if not p.exists()]
if missing:
    st.error("분석 산출물이 없습니다. analysis/01\\~07, 10\\~15 스크립트를 순서대로 먼저 실행하세요.\n\n"
             + "\n".join(f"- {m}" for m in missing))
    st.stop()

data = load()
daily, log, category_full = data["daily"], data["log"], data["category"]
cat_daily, dec = data["category_daily"], data["decomposition"]

MIN_D, MAX_D = daily["date"].min().date(), daily["date"].max().date()
PRESETS = {
    "전체 기간": (MIN_D, MAX_D),
    "2011년 1~8월 (비수기)": (date(2011, 1, 1), date(2011, 8, 31)),
    "2011년 9월 이후 (성수기)": (date(2011, 9, 1), MAX_D),
    "최근 3개월": (date(2011, 9, 9), MAX_D),
    "2011년 상반기": (date(2011, 1, 1), date(2011, 6, 30)),
    "2011년 하반기": (date(2011, 7, 1), MAX_D),
}
if "period" not in st.session_state:
    st.session_state["period"] = PRESETS["전체 기간"]


def apply_preset() -> None:
    st.session_state["period"] = PRESETS[st.session_state["preset"]]


with st.sidebar:
    st.markdown("### 분석 기간")
    st.selectbox("빠른 선택", list(PRESETS), key="preset", on_change=apply_preset)
    period = st.date_input("시작일 ~ 종료일", key="period", min_value=MIN_D, max_value=MAX_D,
                           format="YYYY-MM-DD")
    st.caption(f"선택 가능 범위: {MIN_D} \\~ {MAX_D}")
    st.markdown("### 목차")
    st.markdown(
        "1. [요약](#summary)\n"
        "2. [데이터와 전처리](#data)\n"
        "3. [일별 매출 추이](#daily)\n"
        "4. [월별 매출](#monthly)\n"
        "5. [요일별 매출](#weekday)\n"
        "6. [카테고리별 매출](#category)\n"
        "7. [시계열 분해](#decomposition)\n"
        "8. [예측 모형 비교 (11월)](#forecast)\n"
        "9. [12월 예측](#december)\n"
        "10. [시사점과 다음 단계](#next)"
    )
    st.caption("자료: UCI Online Retail (Daqing Chen, CC BY 4.0)\n\n"
               "금액 단위: 파운드(£). 모든 금액은 순매출(판매 − 취소).")

if not isinstance(period, (tuple, list)) or len(period) != 2:
    st.info("사이드바에서 종료일까지 선택하세요.")
    st.stop()
start, end = pd.Timestamp(period[0]), pd.Timestamp(period[1])
sel = daily[(daily["date"] >= start) & (daily["date"] <= end)]
open_sel = sel[sel["is_closed"] == 0]
if open_sel.empty:
    st.warning("선택한 기간에 영업일이 없습니다. 기간을 넓혀 주세요.")
    st.stop()
is_full = (start.date(), end.date()) == (MIN_D, MAX_D)

st.title("온라인 소매 매출 분석 보고서")
st.caption(f"선택 기간 {start:%Y-%m-%d} \\~ {end:%Y-%m-%d} ({len(sel)}일, 영업일 {len(open_sel)}일) "
           f"· 영국 기반 온라인 선물·생활용품 소매업체 거래 자료")

# ---------------------------------------------------------------- 1. 요약
st.header("1. 요약", anchor="summary")
total = sel["net_sales"].sum()
monthly = monthly_summary(sel)
peak = monthly.loc[monthly["avg_per_day"].idxmax()]
wd = weekday_summary(sel)
best = wd.loc[wd["mean_net_sales"].idxmax()]

c1, c2, c3, c4 = st.columns(4)
c1.metric("순매출 합계", f"£{total / 1e6:,.2f}백만" if total >= 1e6 else f"£{total / 1e3:,.1f}천",
          help=f"{gbp(total, 2)} · 전체 기간의 {pct(total / daily['net_sales'].sum())}")
c2.metric("영업일 평균 순매출", gbp(open_sel["net_sales"].mean()),
          help=f"영업일 {len(open_sel)}일 기준 (휴무일 {int(sel['is_closed'].sum())}일 제외)")
c3.metric(f"일평균 최고 월 ({peak['year_month']})", gbp(peak["avg_per_day"]) + " / 일",
          help=f"월 합계 {gbp(peak['net_sales_total'], 2)}. 일부 기간만 포함된 월도 일평균으로 비교")
c4.metric(f"매출이 가장 높은 요일 ({best['weekday']}요일)", gbp(best["mean_net_sales"]),
          help="선택 기간 영업일 평균 순매출")

top2 = category_full.sort_values("net_sales", ascending=False).head(2)
xmas = category_full.set_index("category").loc["크리스마스·시즌"]
full_monthly = monthly_summary(daily)
full_peak = full_monthly.loc[full_monthly.loc[full_monthly["is_partial"] == 0, "net_sales_total"].idxmax()]
st.markdown("**전체 기간 분석 결과** (기간 선택과 관계없이 고정)")
st.markdown(f"""
- **연말로 갈수록 매출이 크게 늘어납니다.** 1\\~8월의 일평균은 £16\\~24천 수준이었지만, 9월부터 오르기 시작해
  {full_peak['year_month']}에는 월 순매출 {gbp(full_peak['net_sales_total'])}으로 가장 높았습니다.
- **요일 패턴이 뚜렷합니다.** 토요일은 매주 휴무이고, 일요일 매출은 주 평균의 절반 수준입니다.
  목요일과 화요일이 가장 높습니다.
- **매출 구성:** {top2.iloc[0]['category']}({pct(top2.iloc[0]['share'])})와
  {top2.iloc[1]['category']}({pct(top2.iloc[1]['share'])})가 매출의 절반 가까이를 차지합니다.
  연말 매출 증가는 크리스마스·시즌 상품(9\\~12월 일평균이 1\\~8월의
  {xmas['growth_late_vs_early'] + 1:,.0f}배)과 겨울 생활잡화가 이끕니다.
- **예측에 주는 의미:** 주간 계절성과 휴무일을 반드시 반영해야 합니다. 매출 수준이 높을수록 요일 간 차이도
  커지므로 곱셈형 계절성을 검토할 필요가 있습니다. 자료가 1년뿐이라 연간 계절성은 추정할 수 없습니다.
""")

# ---------------------------------------------------------------- 2. 데이터와 전처리
st.header("2. 데이터와 전처리", anchor="data")
raw_rows = int(log.loc[log["step"].astype(str) == "0", "remaining_rows"].iloc[0])
final = log.iloc[-1]
st.markdown(f"""
전처리는 **전체 기간 기준**입니다. 원본 {raw_rows:,}행에서 아래 규칙을 순서대로 적용해 최종
**{int(final['remaining_rows']):,}행**, 순매출 **{gbp(final['remaining_amount'], 2)}** 를 남겼습니다.
기준은 협업자 문서 `전처리_기준.md`를 따랐습니다.
""")
log_view = log.rename(columns={
    "step": "단계", "rule": "규칙", "excluded_rows": "제외 행", "excluded_amount": "제외 금액 (£)",
    "remaining_rows": "남은 행", "remaining_amount": "남은 금액 (£)"})
st.dataframe(log_view, hide_index=True, width="stretch",
             column_config={
                 "제외 행": st.column_config.NumberColumn(format="%,d"),
                 "남은 행": st.column_config.NumberColumn(format="%,d"),
                 "제외 금액 (£)": st.column_config.NumberColumn(format="%,.2f"),
                 "남은 금액 (£)": st.column_config.NumberColumn(format="%,.2f"),
             })
st.markdown("""
**제외하지 않은 것**
- **`CustomerID` 결측 행:** 실제 판매이므로 매출에 포함했습니다. 고객 수 집계에서만 빠집니다.
- **취소 행:** 원 주문과 짝짓지 않고, 취소가 일어난 날짜의 매출에서 뺐습니다.
  대량 주문 뒤 같은 날 취소된 건(80,995개, 74,215개)은 순매출에서 서로 상쇄됩니다.

**일별 테이블**
- 2010-12-01 \\~ 2011-12-08의 373일로 만들고, 거래가 없는 날은 0으로 채웠습니다.
- 원본에 거래가 한 건도 없는 날은 휴무일(`is_closed`)로 표시했습니다. 토요일 전부, 연말연시, 영국 공휴일입니다.
- 2011-12-09는 12:50까지만 기록되어 제외했습니다.
""")

# ---------------------------------------------------------------- 3. 일별 추이
st.header("3. 일별 매출 추이", anchor="daily")
st.markdown("막대는 일별 순매출, 선은 **7일 후행 이동평균**(당일 포함 직전 7일 평균, 휴무일 0 포함)입니다. "
            "주황 점은 요일별 기준으로 표시한 이상치입니다. 그래프에 마우스를 올리면 값을 볼 수 있습니다.")
axis_fmt = "%Y-%m" if (end - start).days > 92 else "%m-%d"
base = alt.Chart(sel).encode(x=alt.X("date:T", title=None, axis=alt.Axis(format=axis_fmt)))
tooltip = [alt.Tooltip("date:T", title="날짜", format="%Y-%m-%d"),
           alt.Tooltip("weekday:N", title="요일"),
           alt.Tooltip("net_sales:Q", title="순매출 (£)", format=",.2f"),
           alt.Tooltip("ma7:Q", title="7일 이동평균 (£)", format=",.2f"),
           alt.Tooltip("is_closed:Q", title="휴무일(1)")]
bars = base.mark_bar(color=LIGHT_BLUE).encode(
    y=alt.Y("net_sales:Q", title="순매출 (£)", axis=alt.Axis(format=",.0f")), tooltip=tooltip)
line = base.mark_line(color=BLUE, strokeWidth=2).encode(y="ma7:Q", tooltip=tooltip)
points = alt.Chart(sel[sel["is_outlier"] == 1]).mark_circle(
    color=ORANGE, size=80, stroke="white", strokeWidth=1.5).encode(
    x="date:T", y="net_sales:Q", tooltip=tooltip)
st.altair_chart((bars + line + points).properties(height=360).interactive(bind_y=False),
                width="stretch")
st.caption("드래그하거나 스크롤하면 가로축을 확대·이동할 수 있습니다. "
           "이동평균과 이상치 표시는 전체 기간으로 계산한 값입니다.")

# ---------------------------------------------------------------- 4. 월별
st.header("4. 월별 매출", anchor="monthly")
metric = st.radio("지표", ["일평균 (월 합계 ÷ 달력 일수)", "월 합계", "영업일 평균"], horizontal=True)
col = {"일평균 (월 합계 ÷ 달력 일수)": "avg_per_day", "월 합계": "net_sales_total",
       "영업일 평균": "avg_per_open_day"}[metric]
FULL_LABEL, PARTIAL_LABEL = "한 달 전체", "일부 날짜만 포함"
m = monthly.assign(구분=monthly["is_partial"].map({0: FULL_LABEL, 1: PARTIAL_LABEL}))
chart = alt.Chart(m).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
    x=alt.X("year_month:N", title=None, axis=alt.Axis(labelAngle=0)),
    y=alt.Y(f"{col}:Q", title=f"{metric} (£)", axis=alt.Axis(format=",.0f")),
    color=alt.Color("구분:N", scale=alt.Scale(domain=[FULL_LABEL, PARTIAL_LABEL], range=[BLUE, GRAY]),
                    legend=alt.Legend(orient="top", title=None)),
    tooltip=[alt.Tooltip("year_month:N", title="월"),
             alt.Tooltip("net_sales_total:Q", title="월 합계 (£)", format=",.2f"),
             alt.Tooltip("avg_per_day:Q", title="일평균 (£)", format=",.2f"),
             alt.Tooltip("avg_per_open_day:Q", title="영업일 평균 (£)", format=",.2f"),
             alt.Tooltip("n_days:Q", title="포함 일수"),
             alt.Tooltip("n_open_days:Q", title="영업일 수")],
).properties(height=320)
st.altair_chart(chart, width="stretch")
if col == "net_sales_total" and monthly["is_partial"].any():
    st.info("회색 막대는 선택 기간이나 자료 범위 때문에 일부 날짜만 포함된 월이라 합계가 작게 보입니다. "
            "월끼리 비교할 때는 일평균을 쓰세요.")
st.dataframe(monthly.drop(columns=["days_in_month"]).rename(columns={
    "year_month": "월", "net_sales_total": "합계 (£)", "n_days": "포함 일수", "n_open_days": "영업일",
    "is_partial": "일부만 포함", "avg_per_day": "일평균 (£)", "avg_per_open_day": "영업일 평균 (£)"}),
    hide_index=True, width="stretch", height=(len(monthly) + 1) * 35 + 3,
    column_config={c: st.column_config.NumberColumn(format="%,.2f")
                   for c in ["합계 (£)", "일평균 (£)", "영업일 평균 (£)"]})

# ---------------------------------------------------------------- 5. 요일별
st.header("5. 요일별 매출", anchor="weekday")
mean_k = wd["mean_of_weekday_means"].iloc[0]
st.markdown(f"선택 기간의 영업일만 사용했습니다(토요일은 매주 휴무). "
            f"{len(wd)}개 요일 평균 매출의 평균은 **{gbp(mean_k, 2)}**입니다.")
c1, c2 = st.columns([1.1, 1])
wd_chart = wd.assign(방향=wd["diff_from_weekday_mean"].map(lambda v: "평균 이상" if v >= 0 else "평균 미만"))
bar = alt.Chart(wd_chart).mark_bar(cornerRadius=4).encode(
    x=alt.X("weekday:N", sort=OPEN_DOW_ORDER, title=None, axis=alt.Axis(labelAngle=0)),
    y=alt.Y("diff_from_weekday_mean:Q", title="요일 평균 − 요일 평균의 평균 (£)",
            axis=alt.Axis(format=",.0f")),
    color=alt.Color("방향:N", scale=alt.Scale(domain=["평균 이상", "평균 미만"], range=[BLUE, ORANGE]),
                    legend=alt.Legend(orient="top", title=None)),
    tooltip=[alt.Tooltip("weekday:N", title="요일"),
             alt.Tooltip("n_open_days:Q", title="영업일 수"),
             alt.Tooltip("mean_net_sales:Q", title="평균 (£)", format=",.2f"),
             alt.Tooltip("diff_from_weekday_mean:Q", title="차이 (£)", format="+,.2f"),
             alt.Tooltip("mean_week_index:Q", title="주 평균 대비 지수", format=".2f")],
)
rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#52514e").encode(y="y:Q")
c1.altair_chart((bar + rule).properties(height=320), width="stretch")
c2.dataframe(wd[["weekday", "n_open_days", "mean_net_sales", "median_net_sales",
                 "diff_from_weekday_mean", "mean_week_index", "share_of_total"]].rename(columns={
    "weekday": "요일", "n_open_days": "영업일", "mean_net_sales": "평균 (£)",
    "median_net_sales": "중앙값 (£)", "diff_from_weekday_mean": "평균과의 차이 (£)",
    "mean_week_index": "주 평균 대비", "share_of_total": "매출 비중"}),
    hide_index=True, width="stretch",
    column_config={
        "평균 (£)": st.column_config.NumberColumn(format="%,.2f"),
        "중앙값 (£)": st.column_config.NumberColumn(format="%,.2f"),
        "평균과의 차이 (£)": st.column_config.NumberColumn(format="%+,.2f"),
        "주 평균 대비": st.column_config.NumberColumn(format="%.2f"),
        "매출 비중": st.column_config.NumberColumn(format="percent"),
    })
if wd["n_open_days"].min() < 4:
    st.warning("요일별 영업일 수가 4일 미만인 요일이 있어 평균이 불안정합니다. 기간을 넓혀 보세요.")
st.markdown("""
**전체 기간 기준 해석**
- **목·화요일이 가장 높고 일요일이 가장 낮습니다.** 일요일 매출은 주 평균의 약 절반(0.48)입니다.
  송장당 매출도 가장 작아서, 작은 주문이 적게 들어오는 날입니다.
- **모든 요일에서 평균이 중앙값보다 높습니다.** 가끔 있는 대량 주문일이 평균을 끌어올립니다.
- **주 평균 대비 지수**는 각 영업일 매출을 같은 주의 영업일 평균으로 나눈 값입니다.
  기간 전체의 추세를 걷어내고 요일 효과만 봅니다.
""")

# ---------------------------------------------------------------- 6. 카테고리별
st.header("6. 카테고리별 매출", anchor="category")
st.markdown("원본에 카테고리 열이 없어, 상품명(`Description`)의 키워드로 10개 상품군을 만든 **근사 분류**입니다. "
            "규칙과 상품별 분류 결과는 `analysis/06_category_sales.py`와 "
            "`data/processed/product_category.csv`에 있습니다.")
cd = cat_daily[(cat_daily["date"] >= start) & (cat_daily["date"] <= end)]
cat, mid = category_summary(cd, start, end)
cat = cat.sort_values("net_sales", ascending=False)
order = [c for c in cat["category"] if c != "기타"] + (["기타"] if "기타" in set(cat["category"]) else [])
show_growth = len(sel) >= 28
c1, c2 = st.columns(2)
cat_tip = [alt.Tooltip("category:N", title="카테고리"),
           alt.Tooltip("net_sales:Q", title="순매출 (£)", format=",.2f"),
           alt.Tooltip("share:Q", title="비중", format=".1%"),
           alt.Tooltip("daily_avg:Q", title="일평균 (£)", format=",.2f"),
           alt.Tooltip("growth_half:Q", title="후반부 증감", format="+.0%")]
color_other = alt.condition(alt.datum.category == "기타", alt.value(GRAY), alt.value(BLUE))
c1.markdown("**카테고리별 순매출**")
c1.altair_chart(alt.Chart(cat).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
    y=alt.Y("category:N", sort=order, title=None),
    x=alt.X("net_sales:Q", title="순매출 (£)", axis=alt.Axis(format=",.0f")),
    color=color_other, tooltip=cat_tip).properties(height=340), width="stretch")
c2.markdown(f"**후반부 일평균의 전반부 대비 증감률** "
            f"(전반부 {start:%m-%d}\\~{mid - pd.Timedelta(days=1):%m-%d}, 후반부 {mid:%m-%d}\\~{end:%m-%d})")
if show_growth:
    c2.altair_chart(alt.Chart(cat).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
        y=alt.Y("category:N", sort=order, title=None),
        x=alt.X("growth_half:Q", title="증감률", axis=alt.Axis(format="%")),
        color=color_other, tooltip=cat_tip).properties(height=340), width="stretch")
else:
    c2.info("선택 기간이 28일보다 짧아 전반부·후반부 비교를 생략했습니다.")

st.markdown("**카테고리별 월 일평균 추이** (월 합계 ÷ 선택 기간에 포함된 그 달의 일수)")
picked = st.multiselect("카테고리 선택", order, default=order[:4])
if picked:
    days_per_month = sel.groupby("year_month").size()
    cm = (cd.assign(year_month=cd["date"].dt.strftime("%Y-%m"))
          .groupby(["year_month", "category"])["net_sales"].sum().reset_index())
    cm["daily_avg"] = cm["net_sales"] / cm["year_month"].map(days_per_month)
    cm = cm[cm["category"].isin(picked)]
    facet = alt.Chart(cm).mark_line(color=BLUE, strokeWidth=2, point=alt.OverlayMarkDef(size=25)).encode(
        x=alt.X("year_month:N", title=None, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("daily_avg:Q", title="월 일평균 (£)", axis=alt.Axis(format=",.0f")),
        tooltip=[alt.Tooltip("category:N", title="카테고리"), alt.Tooltip("year_month:N", title="월"),
                 alt.Tooltip("net_sales:Q", title="월 합계 (£)", format=",.2f"),
                 alt.Tooltip("daily_avg:Q", title="월 일평균 (£)", format=",.2f")],
    ).properties(width=260, height=180).facet(
        facet=alt.Facet("category:N", sort=picked, title=None), columns=4
    ).resolve_scale(y="independent")
    st.altair_chart(facet)
    st.caption("카테고리마다 세로축이 다릅니다.")
st.dataframe(cat[["category", "net_sales", "share", "daily_avg", "cancel_rate", "growth_half"]].rename(columns={
    "category": "카테고리", "net_sales": "순매출 (£)", "share": "비중", "daily_avg": "일평균 (£)",
    "cancel_rate": "취소율", "growth_half": "후반부 증감"}),
    hide_index=True, width="stretch",
    column_config={
        "순매출 (£)": st.column_config.NumberColumn(format="%,.2f"),
        "비중": st.column_config.NumberColumn(format="percent"),
        "일평균 (£)": st.column_config.NumberColumn(format="%,.2f"),
        "취소율": st.column_config.NumberColumn(format="percent"),
        "후반부 증감": st.column_config.NumberColumn(format="percent"),
    })
st.markdown("""
**전체 기간 기준 해석**
- **주방·식기와 인테리어·장식이 매출의 약 절반입니다.** 주방·식기는 하반기 증가율이 +26%로 가장 안정적입니다.
- **연말 급증은 시즌 상품이 이끕니다.** 크리스마스 상품은 11월이 정점이고 12월 초에 벌써 줄어듭니다.
  패션·생활잡화의 급증은 핫워터보틀, 워머 같은 겨울 상품 때문입니다.
- **보관·수납은 취소율이 10%로 높은데**, 저장용 항아리 74,215개를 주문하고 같은 날 취소한 1건 때문입니다.
""")

# ---------------------------------------------------------------- 7. 분해
st.header("7. 시계열 분해", anchor="decomposition")
st.markdown("""
일별 순매출을 **매출 = 추세 + 요일 효과 + 잔차**로 나눴습니다. 방식은 전통적인 가법 분해(주기 7일)입니다.
분해는 **전체 기간으로 한 번 계산**했고, 여기서는 선택 기간에 해당하는 날짜만 보여줍니다.
- **추세:** 당일을 가운데 둔 7일 이동평균입니다. 미래 값을 쓰므로 흐름을 살펴보는 용도이고, 예측 입력으로는 쓰지 않습니다.
- **요일 효과:** 매출에서 추세를 뺀 값을 요일별로 평균한 것입니다. 7개 요일의 합은 0입니다.
- **잔차:** 매출에서 추세와 요일 효과를 뺀 나머지입니다.
""")
dec_sel = dec[(dec["date"] >= start) & (dec["date"] <= end)]
months = sorted(dec_sel["year_month"].unique())
if len(months) > 1:
    ym = st.select_slider("월 선택", options=months, value=months[-1])
else:
    ym = months[0]
    st.caption(f"표시 월: {ym}")
md = dec_sel[dec_sel["year_month"] == ym].assign(weekday=lambda x: x["dow"].map(lambda i: DOW_NAMES[i]))
dmax = dec["net_sales"].max() * 1.05
emax = dec["weekday_effect"].abs().max() * 1.15
rmax = dec["residual"].abs().max() * 1.1
dtip = [alt.Tooltip("date:T", title="날짜", format="%Y-%m-%d"), alt.Tooltip("weekday:N", title="요일"),
        alt.Tooltip("net_sales:Q", title="매출 (£)", format=",.2f"),
        alt.Tooltip("trend:Q", title="추세 (£)", format=",.2f"),
        alt.Tooltip("weekday_effect:Q", title="요일 효과 (£)", format="+,.2f"),
        alt.Tooltip("residual:Q", title="잔차 (£)", format="+,.2f")]
m_start = max(pd.Timestamp(ym + "-01"), start)
m_end = min(pd.Timestamp(ym + "-01") + pd.offsets.MonthEnd(0), end)
xenc = alt.X("date:T", title=None, axis=alt.Axis(format="%m-%d"),
             scale=alt.Scale(domain=[(m_start - pd.Timedelta(hours=12)).isoformat(),
                                     (m_end + pd.Timedelta(hours=12)).isoformat()]))


def panel(title, field, kind, domain):
    ch = alt.Chart(md, title=alt.TitleParams(title, anchor="start", fontSize=13))
    y = alt.Y(f"{field}:Q", title="£", scale=alt.Scale(domain=domain), axis=alt.Axis(format=",.0f"))
    if kind == "line":
        return ch.mark_line(color=BLUE, strokeWidth=2).encode(x=xenc, y=y, tooltip=dtip)
    color = alt.value(LIGHT_BLUE) if kind == "bar" else alt.condition(
        f"datum.{field} >= 0", alt.value(BLUE), alt.value(ORANGE))
    return ch.mark_bar().encode(x=xenc, y=y, color=color, tooltip=dtip)


decomp = alt.vconcat(
    panel("① 매출", "net_sales", "bar", [0, dmax]).properties(height=170),
    panel("② 요일 효과", "weekday_effect", "signed", [-emax, emax]).properties(height=140),
    panel("③ 추세 (중심 7일 이동평균)", "trend", "line", [0, dmax]).properties(height=170),
    panel("④ 잔차", "residual", "signed", [-rmax, rmax]).properties(height=140),
).resolve_scale(x="shared")
st.altair_chart(decomp, width="stretch")
st.caption("요소별 세로축은 모든 월에서 같습니다. 전체 기간의 처음·마지막 3일은 추세를 계산할 수 없어 비어 있습니다.")

c1, c2 = st.columns(2)
eff = dec.groupby("dow")["weekday_effect"].first().rename(index=lambda i: DOW_NAMES[i])
c1.markdown("**요일 효과 (£, 전체 기간 기준)**")
c1.dataframe(eff.to_frame("요일 효과 (£)").T, width="stretch",
             column_config={n: st.column_config.NumberColumn(format="%+,.2f") for n in DOW_NAMES})
summ = dec_sel.groupby("year_month").agg(추세_평균=("trend", "mean"), 잔차_표준편차=("residual", "std"))
c2.markdown("**월별 추세 평균과 잔차 표준편차 (£, 선택 기간)**")
c2.dataframe(summ.T, width="stretch",
             column_config={c: st.column_config.NumberColumn(format="%,.0f") for c in summ.index})
st.markdown("""
**전체 기간 기준 해석**
- **추세:** 2\\~4월에 약 £17천으로 가장 낮고, 5\\~8월에는 £23천 안팎으로 평평합니다. 9월부터 올라 11\\~12월에는 £47천에 이릅니다.
- **잔차:** 추세가 높은 9월 이후에 잔차 폭도 커집니다.
- **가법 분해의 한계:** 추세가 높은 달에는 **토요일 잔차가 매번 크게 음수**입니다.
  가법 분해는 요일 효과를 기간 내내 같은 금액으로 보지만, 토요일 매출은 항상 0이기 때문입니다.
  요일 효과를 매출 수준에 비례하는 곱셈형으로 보고, 휴무일은 따로 처리해야 합니다.
""")

# ---------------------------------------------------------------- 8. 예측 모형 비교
st.header("8. 예측 모형 비교 (11월)", anchor="forecast")
st.info("8·9장은 사이드바의 분석 기간과 관계없이, 아래의 모형 선택과 기간 선택으로 조절합니다.")
st.markdown("""
- **학습 구간:** 2011-01-01 \\~ 2011-10-31 (304일)
- **비교 구간:** 2011-11-01 \\~ 2011-11-30 (30일, 영업일 26일)
- **예측 방식:** 10월 31일 시점에서 11월 30일치를 한 번에 예측하고, 11월 실제 값은 비교에만 썼습니다.
- **예측 대상:** 일별 순매출(휴무일 0 포함), 변환 없음
- **지수평활 3개 모형:** 평활계수와 초기값을 학습 구간 오차 제곱합이 가장 작아지도록 자동 추정했습니다.
- **SARIMA:** 7일 계절 차분(D=1) 뒤 p, q ∈ {0,1,2}, P, Q ∈ {0,1}, 상수항 유무 72개 후보 중 학습 AIC가 가장 작은 모형을 골랐습니다.
""")

# 모형 설명 표
sp, hp, wp = data["ses_params"].iloc[0], data["holt_params"].iloc[0], data["hw_params"].iloc[0]
sar = data["sarima_params"].set_index("param")["estimate"]
sar_best = data["sarima_search"].query("converged").iloc[0]
sar_spec = (f"({int(sar_best['p'])},{int(sar_best['d'])},{int(sar_best['q'])})"
            f"({int(sar_best['P'])},{int(sar_best['D'])},{int(sar_best['Q'])})"
            f"[{int(sar_best['m'])}]" + (" + 상수항" if sar_best["trend"] == "c" else ""))
model_tbl = pd.DataFrame({
    "모형": [MODEL_NAMES_KO[m] for m in MODELS],
    "구성": ["수준", "수준 + 선형 추세", "수준 + 선형 추세 + 요일 계절(가법)", f"SARIMA{sar_spec}"],
    "추정된 계수": [
        f"α {sp['alpha']:.4f}",
        f"α {hp['alpha']:.4f}, β {hp['beta']:.4f}",
        f"α {wp['alpha']:.4f}, β {wp['beta']:.4f}, γ {wp['gamma']:.4f}",
        ", ".join(f"{k} {v:+.4f}" for k, v in sar.items() if k != "sigma2"),
    ],
    "초기값": [
        f"수준 £{sp['initial_level']:,.0f}",
        f"수준 £{hp['initial_level']:,.0f}, 추세 £{hp['initial_trend']:,.2f}/일",
        f"수준 £{wp['initial_level']:,.0f}, 추세 £{wp['initial_trend']:,.2f}/일",
        f"AIC {sar_best['aic']:,.2f}",
    ],
    "학습 RMSE (£)": [sp["train_rmse"], hp["train_rmse"], wp["train_rmse"], np.nan],
})
model_tbl["학습 RMSE (£)"] = model_tbl["학습 RMSE (£)"].map(lambda v: "–" if pd.isna(v) else f"{v:,.2f}")
st.markdown("**모형 구성과 추정 결과**")
st.dataframe(model_tbl, hide_index=True, width="stretch")
season_cols = [c for c in data["hw_params"].columns if c.startswith("initial_season_")]
season = pd.Series({c.replace("initial_season_", ""): wp[c] for c in season_cols}).reindex(DOW_NAMES)
st.caption("Holt-Winters 요일 계절값 (£, γ = 0이라 기간 내내 고정): "
           + " · ".join(f"{d} {v:+,.0f}" for d, v in season.items())
           + " · SARIMA 계수: ar.L1 = AR(1), ma.L1 = MA(1), ma.S.L7 = 계절 MA(7)")

# ---- 비교 조건 선택
fc_nov = data["fc_nov"].copy()
NOV_MIN, NOV_MAX = fc_nov["date"].min().date(), fc_nov["date"].max().date()
st.subheader("모형 비교")
c1, c2, c3 = st.columns([1.6, 1.2, 1])
cmp_models = c1.multiselect("비교할 모형", MODELS, default=MODELS, key="cmp_models",
                            format_func=lambda m: MODEL_NAMES_KO[m])
eval_period = c2.date_input("평가 기간 (11월 안에서 선택)", value=(NOV_MIN, NOV_MAX),
                            min_value=NOV_MIN, max_value=NOV_MAX, key="eval_period", format="YYYY-MM-DD")
scope = c3.radio("평가 대상 날짜", ["전체", "영업일만"], key="eval_scope",
                 help="전체에는 실제 0인 토요일이 포함되어 평균 절대 백분율 오차(MAPE)를 계산하지 않습니다.")

if not cmp_models:
    st.warning("비교할 모형을 하나 이상 선택하세요.")
elif not isinstance(eval_period, (tuple, list)) or len(eval_period) != 2:
    st.info("평가 기간의 종료일까지 선택하세요.")
else:
    e_start, e_end = pd.Timestamp(eval_period[0]), pd.Timestamp(eval_period[1])
    ev = fc_nov[(fc_nov["date"] >= e_start) & (fc_nov["date"] <= e_end)]
    if scope == "영업일만":
        ev = ev[ev["is_closed"] == 0]
    if ev.empty:
        st.warning("선택한 조건에 해당하는 날짜가 없습니다.")
    else:
        mt = pd.DataFrame([{"model": m, **forecast_metrics(ev["actual"], ev[f"{m}_forecast"])}
                           for m in cmp_models]).set_index("model")
        best_model = mt["MAE"].idxmin()
        st.caption(f"평가 날짜: {e_start:%m-%d} \\~ {e_end:%m-%d} 중 {len(ev)}일"
                   + (" (영업일만)" if scope == "영업일만" else " (휴무일 포함)")
                   + f" · 실제 합계 {gbp(ev['actual'].sum(), 2)}")
        cols = st.columns(len(cmp_models))
        for c, mname in zip(cols, cmp_models):
            r = mt.loc[mname]
            c.metric(MODEL_NAMES_KO[mname] + (" ★" if mname == best_model and len(cmp_models) > 1 else ""),
                     f"MAE {gbp(r['MAE'])}",
                     help=f"평균 절대 오차. 예측 합계 {gbp(r['forecast_total'])} / 실제 {gbp(r['actual_total'])}")
            c.caption(f"합계 오차 {r['total_error_pct']:+.1%} · RMSE {gbp(r['RMSE'])}")
        if len(cmp_models) > 1:
            st.caption("★ = 선택한 조건에서 평균 절대 오차가 가장 작은 모형")

        view = mt.reset_index()
        view["total_error_pct"] *= 100
        cols_show = ["model", "forecast_total", "actual_total", "total_error_pct", "MAE", "RMSE", "bias_mean_error"]
        if view["MAPE"].notna().all():
            view["MAPE"] *= 100
            cols_show.append("MAPE")
        view["model"] = view["model"].map(MODEL_NAMES_KO)
        st.dataframe(view[cols_show].rename(columns={
            "model": "모형", "forecast_total": "예측 합계 (£)", "actual_total": "실제 합계 (£)",
            "total_error_pct": "합계 오차", "MAE": "평균 절대 오차 (£)", "RMSE": "RMSE (£)",
            "bias_mean_error": "평균 오차 (£)", "MAPE": "평균 절대 백분율 오차"}),
            hide_index=True, width="stretch",
            column_config={
                "예측 합계 (£)": st.column_config.NumberColumn(format="%,.2f"),
                "실제 합계 (£)": st.column_config.NumberColumn(format="%,.2f"),
                "합계 오차": st.column_config.NumberColumn(format="%+.1f%%"),
                "평균 절대 오차 (£)": st.column_config.NumberColumn(format="%,.2f"),
                "RMSE (£)": st.column_config.NumberColumn(format="%,.2f"),
                "평균 오차 (£)": st.column_config.NumberColumn(format="%+,.2f"),
                "평균 절대 백분율 오차": st.column_config.NumberColumn(format="%.1f%%"),
            })
        st.caption("평균 오차가 음수이면 실제보다 낮게 예측한 것입니다.")

        # 실제 vs 예측 그래프 (선택 기간 전체를 표시, 영업일만 선택 시 휴무일은 회색 막대)
        show = fc_nov[(fc_nov["date"] >= e_start) & (fc_nov["date"] <= e_end)]
        band = ("SARIMA" in cmp_models) and st.checkbox("SARIMA 95% 예측구간 표시", value=False, key="nov_band")
        st.altair_chart(forecast_chart(show, cmp_models, band=band), width="stretch")
        st.caption("막대 = 11월 실제 순매출(회색 = 휴무일 토요일), 선 = 모형별 예측. 마우스를 올리면 값을 볼 수 있습니다.")

        # 요일별 평균 오차
        err = ev.melt(id_vars=["date", "weekday", "actual"], value_vars=[f"{m}_forecast" for m in cmp_models],
                      var_name="model", value_name="forecast")
        err["model"] = err["model"].str.replace("_forecast", "", regex=False)
        err["error"] = err["forecast"] - err["actual"]
        by_dow = err.groupby(["model", "weekday"])["error"].mean().reset_index()
        st.markdown("**요일별 평균 오차 (예측 − 실제)**")
        st.altair_chart(alt.Chart(by_dow).mark_bar().encode(
            x=alt.X("weekday:N", sort=DOW_NAMES, title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("model:N", sort=MODELS),
            y=alt.Y("error:Q", title="평균 오차 (£)", axis=alt.Axis(format=",.0f")),
            color=model_color_scale(cmp_models),
            tooltip=[alt.Tooltip("model:N", title="모형"), alt.Tooltip("weekday:N", title="요일"),
                     alt.Tooltip("error:Q", title="평균 오차 (£)", format="+,.2f")],
        ).properties(height=280), width="stretch")

        # 두 모형 직접 비교
        if len(cmp_models) >= 2:
            with st.expander("두 모형 직접 비교 (날짜별 절대 오차)", expanded=False):
                a1, a2 = st.columns(2)
                ma = a1.selectbox("모형 A", cmp_models, index=cmp_models.index(best_model),
                                  format_func=lambda m: MODEL_NAMES_KO[m], key="pair_a")
                rest = [m for m in cmp_models if m != ma]
                mb = a2.selectbox("모형 B", rest, format_func=lambda m: MODEL_NAMES_KO[m], key="pair_b")
                pair = ev[["date", "weekday", "actual"]].copy()
                pair["A"] = (ev[f"{ma}_forecast"] - ev["actual"]).abs()
                pair["B"] = (ev[f"{mb}_forecast"] - ev["actual"]).abs()
                pair["diff"] = pair["B"] - pair["A"]  # 양수면 A가 더 정확
                pair["더 정확한 모형"] = np.where(pair["diff"] > 0, MODEL_NAMES_KO[ma],
                                           np.where(pair["diff"] < 0, MODEL_NAMES_KO[mb], "같음"))
                wins_a, wins_b = int((pair["diff"] > 0).sum()), int((pair["diff"] < 0).sum())
                st.markdown(f"{len(pair)}일 중 더 정확했던 날: **A {MODEL_NAMES_KO[ma]}** {wins_a}일 · "
                            f"**B {MODEL_NAMES_KO[mb]}** {wins_b}일 · 같음 {len(pair) - wins_a - wins_b}일")
                st.altair_chart(alt.Chart(pair).mark_bar().encode(
                    x=alt.X("date:T", title=None, axis=alt.Axis(format="%m-%d")),
                    y=alt.Y("diff:Q", title="절대오차 차이 (£)",
                            axis=alt.Axis(format=",.0f")),
                    color=alt.Color("더 정확한 모형:N", title=None,
                                    scale=alt.Scale(domain=[MODEL_NAMES_KO[ma], MODEL_NAMES_KO[mb], "같음"],
                                                    range=[MODEL_COLORS[ma], MODEL_COLORS[mb], GRAY]),
                                    legend=alt.Legend(orient="top")),
                    tooltip=[alt.Tooltip("date:T", title="날짜", format="%Y-%m-%d"),
                             alt.Tooltip("weekday:N", title="요일"),
                             alt.Tooltip("actual:Q", title="실제 (£)", format=",.2f"),
                             alt.Tooltip("A:Q", title="A 절대오차 (£)", format=",.2f"),
                             alt.Tooltip("B:Q", title="B 절대오차 (£)", format=",.2f")],
                ).properties(height=260), width="stretch")
                st.caption("막대 = B의 절대오차 − A의 절대오차. 위로 향하면 그날은 A가 더 정확, 아래로 향하면 B가 더 정확합니다.")

st.markdown("""
**11월 전체 기준 해석**
- **요일 주기를 다루는 두 모형(Holt-Winters, SARIMA)이 확실히 낫습니다.** 평균 절대 오차가 £16천 안팎으로,
  주기가 없는 단순 지수평활·Holt(£24.6천)보다 약 35% 작습니다.
- **SARIMA는 토요일을 가장 잘 맞힙니다.** 7일 계절 차분이 "지난주 같은 요일 값"을 기준으로 삼아 토요일 예측이 거의 0입니다.
  반면 영업일 예측은 더 낮아서, 영업일만 보면 Holt-Winters가 더 정확합니다.
- **Holt는 단순 지수평활보다 나아지지 않았습니다.** 오차를 최소화한 해가 α = β = 0, 즉 1\\~10월 전체에 맞춘 고정 직선입니다.
- **네 모형 모두 과소 예측입니다.** 11월 영업일 평균은 학습 구간의 약 1.9배인데, 1년치 자료로는 연말 성수기를 배울 수 없습니다.
""")

# ---------------------------------------------------------------- 9. 12월 예측
st.header("9. 12월 예측", anchor="december")
st.markdown("""
- **학습 구간 확장:** 11월 비교로 모형을 평가한 뒤, 11월까지 포함한 **2011-01-01 \\~ 11-30**(334일)의 자료로 계수를 다시 추정했습니다.
  모형 설정은 8장과 같고, SARIMA는 8장에서 고른 차수를 그대로 썼습니다.
- **예측 방식:** 11월 30일 시점에서 **12-01 \\~ 12-31** 기간(31일)을 한 번에 예측합니다.
- **12월 실제 값:** 12-01 \\~ 08(8일)만 있습니다. 기준서에 따라 평가에는 쓰지 않고 참고로만 표시합니다.
""")
fc_dec = data["fc_dec"].copy()
pdec = data["params_dec"].set_index("model")
c1, c2 = st.columns([1.6, 1.2])
dec_models = c1.multiselect("표시할 모형", MODELS, default=["Holt-Winters", "SARIMA"], key="dec_models",
                            format_func=lambda m: MODEL_NAMES_KO[m])
VIEW_MIN, VIEW_MAX = date(2011, 9, 1), fc_dec["date"].max().date()
view_period = c2.date_input("그래프 표시 기간", value=(date(2011, 10, 1), VIEW_MAX),
                            min_value=VIEW_MIN, max_value=VIEW_MAX, key="dec_view", format="YYYY-MM-DD")

if dec_models:
    cols = st.columns(len(dec_models))
    for c, mname in zip(cols, dec_models):
        c.metric(f"{MODEL_NAMES_KO[mname]} · 12월 예측 합계",
                 f"£{pdec.loc[mname, 'dec_forecast_total'] / 1e6:,.2f}백만",
                 help=f"학습 RMSE {gbp(pdec.loc[mname, 'train_rmse'])}")

if dec_models and isinstance(view_period, (tuple, list)) and len(view_period) == 2:
    v_start, v_end = pd.Timestamp(view_period[0]), pd.Timestamp(view_period[1])
    hist = daily[(daily["date"] >= v_start) & (daily["date"] <= min(v_end, pd.Timestamp("2011-11-30")))]
    actual_dec = pd.concat([
        hist[["date", "net_sales"]].rename(columns={"net_sales": "actual"}).assign(구분="학습 실제"),
        fc_dec[["date", "actual_reference"]].rename(columns={"actual_reference": "actual"}).dropna()
        .assign(구분="12월 실제 (참고, 평가 제외)"),
    ])
    actual_dec = actual_dec[(actual_dec["date"] >= v_start) & (actual_dec["date"] <= v_end)]
    fshow = fc_dec[(fc_dec["date"] >= v_start) & (fc_dec["date"] <= v_end)]
    layers = []
    closure = pd.DataFrame({"start": [pd.Timestamp("2011-12-24")], "end": [pd.Timestamp("2011-12-31 23:00")]})
    closure = closure[(closure["end"] >= v_start) & (closure["start"] <= v_end)]
    if not closure.empty:
        layers.append(alt.Chart(closure).mark_rect(color="#efeeea", opacity=0.9).encode(x="start:T", x2="end:T"))
    band_dec = "SARIMA" in dec_models and st.checkbox("SARIMA 95% 예측구간 표시", value=False, key="dec_band")
    if band_dec and not fshow.empty:
        layers.append(alt.Chart(fshow).mark_area(color="#f6dbe6", opacity=0.8).encode(
            x="date:T", y="SARIMA_lower95:Q", y2="SARIMA_upper95:Q"))
    layers.append(alt.Chart(actual_dec).mark_bar().encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(format="%m-%d")),
        y=alt.Y("actual:Q", title="순매출 (£)", axis=alt.Axis(format=",.0f")),
        color=alt.Color("구분:N", title=None,
                        scale=alt.Scale(domain=["학습 실제", "12월 실제 (참고, 평가 제외)"],
                                        range=["#d3e3f7", "#8fb6e8"]),
                        legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("date:T", title="날짜", format="%Y-%m-%d"),
                 alt.Tooltip("구분:N"), alt.Tooltip("actual:Q", title="실제 (£)", format=",.2f")]))
    if not fshow.empty:
        long = fshow.melt(id_vars=["date"], value_vars=[f"{m}_forecast" for m in dec_models],
                          var_name="model", value_name="forecast")
        long["model"] = long["model"].str.replace("_forecast", "", regex=False)
        layers.append(alt.Chart(long).mark_line(strokeWidth=2.2, point=alt.OverlayMarkDef(size=20)).encode(
            x="date:T", y="forecast:Q",
            stroke=alt.Stroke("model:N", title=None, sort=MODELS,
                              scale=alt.Scale(domain=dec_models, range=[MODEL_COLORS[m] for m in dec_models]),
                              legend=alt.Legend(orient="top")),
            tooltip=[alt.Tooltip("date:T", title="날짜", format="%Y-%m-%d"),
                     alt.Tooltip("model:N", title="모형"),
                     alt.Tooltip("forecast:Q", title="예측 (£)", format=",.2f")]))
    if v_start <= pd.Timestamp("2011-11-30") <= v_end:
        layers.append(alt.Chart(pd.DataFrame({"d": [pd.Timestamp("2011-11-30 12:00")]})).mark_rule(
            color="#52514e", strokeDash=[4, 3]).encode(x="d:T"))
    st.altair_chart(alt.layer(*layers).resolve_scale(color="independent").properties(height=380), width="stretch")
    st.caption("점선 = 예측 시작(11-30 시점). 회색 구간 = 작년 연말연시 휴무에 해당하는 시기(2010-12-24 \\~ 2011-01-03).")

    ref = fc_dec.dropna(subset=["actual_reference"])
    ref_tbl = pd.DataFrame({
        "모형": [MODEL_NAMES_KO[m] for m in dec_models],
        "12-01\\~08 예측 합계 (£)": [ref[f"{m}_forecast"].sum() for m in dec_models],
        "12-01\\~08 실제 합계 (£)": ref["actual_reference"].sum(),
    })
    ref_tbl["차이"] = (ref_tbl.iloc[:, 1] / ref_tbl.iloc[:, 2] - 1) * 100
    st.markdown("**12-01 \\~ 08 참고 비교 (평가 아님)**")
    st.dataframe(ref_tbl, hide_index=True, width="stretch",
                 column_config={ref_tbl.columns[1]: st.column_config.NumberColumn(format="%,.2f"),
                                ref_tbl.columns[2]: st.column_config.NumberColumn(format="%,.2f"),
                                "차이": st.column_config.NumberColumn(format="%+.1f%%")})
elif dec_models:
    st.info("그래프 표시 기간의 종료일까지 선택하세요.")
else:
    st.warning("표시할 모형을 하나 이상 선택하세요.")

st.markdown("""
**해석과 주의점**
- **Holt-Winters:** 11월을 학습에 넣자 수준이 약 £45.9천으로 올라가고, 요일 계절 평활계수(γ ≈ 0.12)도 0보다 커졌습니다.
  12월 예측은 월\\~금 £51\\~62천, 일요일 £33\\~35천, 토요일 £9\\~11천 수준입니다.
- **SARIMA:** 토요일은 0에 가깝게 예측하지만, 영업일 수준이 Holt-Winters보다 낮아 12월 합계가 가장 작습니다(£1.28백만).
- **12월 초 8일 참고 비교:** 네 모형 모두 실제보다 낮게 예측했습니다. Holt-Winters가 −7%로 가장 가깝고, 나머지는 −13\\~14%입니다.
- **12월 하순은 휴무를 먼저 확인해야 합니다.** 작년에는 12-24부터 연초까지 휴무였습니다.
  모형은 이를 모르기 때문에, 휴무가 반복되면 12월 하순 예측은 크게 과대 예측됩니다.
  **휴무 일정을 받아 그날 예측을 0으로 바꾸는 후처리가 필요**합니다.
""")

# ---------------------------------------------------------------- 10. 다음 단계
st.header("10. 시사점과 다음 단계", anchor="next")
st.markdown("""
1. **휴무일 처리:** 토요일과 공휴일의 0을 모형 밖에서 처리하면 가법 모형의 토요일 오차와 요일 효과 왜곡이 줄어듭니다.
   예를 들어 영업일만 모델링하고, 휴무일 예측은 0으로 둡니다.
2. **두 모형의 장점 결합:** 영업일은 Holt-Winters, 토요일은 SARIMA처럼 0에 가깝게 예측하는 방식을 검토합니다.
3. **곱셈형 요일 효과:** 매출 수준이 높을수록 요일 간 차이가 커집니다. `log1p` 변환 뒤 Holt-Winters나 SARIMA를 적용하면
   곱셈형에 가까운 효과를 낼 수 있습니다.
4. **기준 모형 추가:** 7일 전 값을 그대로 쓰는 계절 나이브 모형과 비교해 개선 폭을 확인합니다.
5. **연말 성수기:** 1년치 자료로는 연간 계절성을 추정할 수 없습니다. 대형 거래처의 발주 계획이나 전년도 연말 자료를 확보하면 보정할 수 있습니다.
6. **전처리 보완:** 2011-06-10의 £38,970 건처럼 `M` 코드로 처리된 취소가 빠진 사례를 협업자와 점검합니다.
""")
st.caption("이 보고서는 analysis/01\\~07, 10\\~15 스크립트의 산출물(data/processed, outputs/tables)을 읽어 만듭니다. "
           "원본 data/raw는 수정하지 않았습니다.")
