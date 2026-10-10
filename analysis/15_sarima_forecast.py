"""SARIMA(p,d,q)(P,D,Q)_7로 2011년 11월 일별 순매출 예측, 지수평활 3개 모형과 비교.

기준: 전처리_기준.md 6-1 (학습 2011-01-01 ~ 10-31, 비교 2011-11-01 ~ 11-30)
- 예측 대상: net_sales (일 단위, 휴무일 0 포함), 변환 없음
- 차분 차수 (학습 자료로만 결정):
    원자료        ADF p≈0.48, KPSS p≈0.01  → 비정상
    7일 계절 차분 ADF p<0.001, KPSS p≥0.10 → 정상
  ACF도 7·14·21일에서 천천히 줄어들어 계절 차분이 필요함 (14_acf_train.py) → d=0, D=1
- 나머지 차수: p,q ∈ {0,1,2}, P,Q ∈ {0,1}, 상수항(drift) 유무를 모두 적합해 학습 AIC가 가장 작은 모형을 채택.
  (차분 차수가 같으므로 AIC 비교가 가능하다.)
- 추정: 최대우도 (statsmodels SARIMAX). 2011-10-31 시점에서 11월 30일치를 한 번에 예측.
- 지수평활 3개 모형의 11월 예측은 12_holt_winters_forecast.py 산출물(es_models_forecast_nov.csv)을 읽어 비교한다.

입력: data/processed/daily_sales.csv, outputs/tables/es_models_forecast_nov.csv
출력:
  outputs/tables/sarima_order_search.csv     후보 모형별 AIC·BIC
  outputs/tables/sarima_params.csv           채택 모형 계수·표준오차
  outputs/tables/all_models_forecast_nov.csv 11월 일별 실제·4개 모형 예측 (+ SARIMA 95% 예측구간)
  outputs/tables/all_models_metrics_nov.csv  4개 모형 오차 지표 (전체 30일 / 영업일 26일)
  outputs/figures/sarima_forecast_nov.png
실행: 프로젝트 루트에서 `python analysis/15_sarima_forecast.py`
"""
import itertools
import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss

ROOT = Path(__file__).resolve().parent.parent
DAILY_PATH = ROOT / "data" / "processed" / "daily_sales.csv"
TABLES = ROOT / "outputs" / "tables"
ES_FORECAST_PATH = TABLES / "es_models_forecast_nov.csv"
SEARCH_PATH = TABLES / "sarima_order_search.csv"
PARAMS_PATH = TABLES / "sarima_params.csv"
FORECAST_PATH = TABLES / "all_models_forecast_nov.csv"
METRICS_PATH = TABLES / "all_models_metrics_nov.csv"
FIG_PATH = ROOT / "outputs" / "figures" / "sarima_forecast_nov.png"

TRAIN = ("2011-01-01", "2011-10-31")
TEST = ("2011-11-01", "2011-11-30")
M = 7
D_ORDER, SD_ORDER = 0, 1
MODELS = ["SES", "Holt", "Holt-Winters", "SARIMA"]

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

BAR_TEST = "#8fb6e8"
COLORS = {"SES": "#eb6834", "Holt": "#1baf7a", "Holt-Winters": "#4a3aa7", "SARIMA": "#e87ba4"}
SARIMA_BAND = "#f6dbe6"
TEXT, TEXT_2, GRID, CLOSED = "#0b0b0b", "#52514e", "#e4e3df", "#efeeea"
DOW_NAMES = ["월", "화", "수", "목", "금", "토", "일"]


def metrics(actual: pd.Series, pred: pd.Series) -> dict:
    err = actual - pred
    nz = actual != 0
    return {
        "n_days": len(actual),
        "actual_total": actual.sum(),
        "forecast_total": pred.sum(),
        "total_error_pct": pred.sum() / actual.sum() - 1,
        "MAE": err.abs().mean(),
        "RMSE": np.sqrt((err ** 2).mean()),
        "bias_mean_error": (pred - actual).mean(),  # 음수면 과소 예측
        "MAPE": (err[nz].abs() / actual[nz]).mean() if nz.all() else np.nan,
    }


def stationarity(train: pd.Series) -> pd.DataFrame:
    rows = []
    for name, s in [("원자료", train), ("7일 계절 차분", train.diff(M).dropna())]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rows.append({"series": name, "adf_p": adfuller(s, autolag="AIC")[1],
                         "kpss_p": kpss(s, regression="c", nlags="auto")[1]})
    return pd.DataFrame(rows)


def search_orders(train: pd.Series) -> pd.DataFrame:
    rows = []
    for p, q, P, Q, trend in itertools.product(range(3), range(3), range(2), range(2), ["n", "c"]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res = SARIMAX(train, order=(p, D_ORDER, q), seasonal_order=(P, SD_ORDER, Q, M), trend=trend,
                              enforce_stationarity=True, enforce_invertibility=True).fit(disp=False, maxiter=500)
                rows.append({"p": p, "d": D_ORDER, "q": q, "P": P, "D": SD_ORDER, "Q": Q, "m": M, "trend": trend,
                             "aic": res.aic, "bic": res.bic, "converged": res.mle_retvals.get("converged", True)})
            except Exception as e:  # noqa: BLE001 - 적합 실패 후보는 기록만 하고 건너뜀
                rows.append({"p": p, "d": D_ORDER, "q": q, "P": P, "D": SD_ORDER, "Q": Q, "m": M, "trend": trend,
                             "aic": np.nan, "bic": np.nan, "converged": False, "error": str(e)[:80]})
    return pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)


def main() -> None:
    daily = pd.read_csv(DAILY_PATH, parse_dates=["date"]).set_index("date").asfreq("D")
    y = daily["net_sales"]
    train = y.loc[TRAIN[0]:TRAIN[1]]
    test = y.loc[TEST[0]:TEST[1]]
    assert len(train) == 304 and len(test) == 30

    stat = stationarity(train)
    search = search_orders(train)
    best = search[search["converged"]].iloc[0]
    order = (int(best["p"]), D_ORDER, int(best["q"]))
    sorder = (int(best["P"]), SD_ORDER, int(best["Q"]), M)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SARIMAX(train, order=order, seasonal_order=sorder, trend=best["trend"],
                      enforce_stationarity=True, enforce_invertibility=True).fit(disp=False, maxiter=500)
    fc_obj = res.get_forecast(len(test))
    fc = pd.Series(fc_obj.predicted_mean.to_numpy(), index=test.index)
    ci = fc_obj.conf_int(alpha=0.05)
    lb = acorr_ljungbox(res.resid.iloc[M + 1:], lags=[7, 14, 21], return_df=True)

    # 저장: 탐색 결과, 계수
    TABLES.mkdir(parents=True, exist_ok=True)
    search.to_csv(SEARCH_PATH, index=False)
    params = pd.DataFrame({"param": res.params.index, "estimate": res.params.to_numpy(),
                           "std_err": res.bse.to_numpy(), "p_value": res.pvalues.to_numpy()})
    params["model"] = f"SARIMA{order}{sorder} trend={best['trend']}"
    params.to_csv(PARAMS_PATH, index=False)

    # 저장: 4개 모형 11월 예측과 지표
    es = pd.read_csv(ES_FORECAST_PATH, parse_dates=["date"]).set_index("date")
    out = es[["weekday", "is_closed", "actual", "SES_forecast", "Holt_forecast", "Holt-Winters_forecast"]].copy()
    assert np.allclose(out["actual"], test.to_numpy())
    out["SARIMA_forecast"] = fc.to_numpy()
    out["SARIMA_lower95"] = ci.iloc[:, 0].to_numpy()
    out["SARIMA_upper95"] = ci.iloc[:, 1].to_numpy()
    out.reset_index().to_csv(FORECAST_PATH, index=False, date_format="%Y-%m-%d")

    open_mask = (out["is_closed"] == 0).to_numpy()
    rows = []
    for mname in MODELS:
        s = out[f"{mname}_forecast"]
        rows.append({"model": mname, "scope": "전체 30일", **metrics(out["actual"], s)})
        rows.append({"model": mname, "scope": "영업일 26일", **metrics(out["actual"][open_mask], s[open_mask])})
    mt = pd.DataFrame(rows)
    mt.to_csv(METRICS_PATH, index=False)

    coverage = ((out["actual"] >= out["SARIMA_lower95"]) & (out["actual"] <= out["SARIMA_upper95"])).mean()
    plot(train, out, order, sorder, best["trend"], coverage)

    with pd.option_context("display.float_format", "{:,.4f}".format, "display.width", 250):
        print(stat.to_string(index=False))
        print("\nAIC 상위 5개 후보:")
        print(search.head(5).to_string(index=False))
        print(f"\n채택: SARIMA{order}{sorder}, trend={best['trend']}, 후보 {len(search)}개 중 "
              f"수렴 {int(search['converged'].sum())}개")
        print(params.drop(columns="model").to_string(index=False))
        print("\n잔차 Ljung-Box:")
        print(lb.to_string())
        print(f"\n11월 실제가 SARIMA 95% 예측구간 안에 든 비율: {coverage:.1%}")
        print(mt.to_string(index=False))


def plot(train, out, order, sorder, trend, coverage) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10), dpi=150,
                                   gridspec_kw={"height_ratios": [1, 1], "hspace": 0.4})
    fig.patch.set_facecolor("white")
    k = FuncFormatter(lambda v, _: f"{v / 1000:,.0f}")
    idx = out.index

    # ① SARIMA 예측과 95% 예측구간
    for d in idx[out["is_closed"] == 1]:
        ax1.axvspan(d - pd.Timedelta(hours=12), d + pd.Timedelta(hours=12), color=CLOSED, zorder=0, linewidth=0)
    tr = train.loc["2011-10-01":]
    ax1.bar(tr.index, tr, width=0.8, color="#d3e3f7", zorder=2, label="학습 실제 (10월 표시)")
    ax1.bar(idx, out["actual"], width=0.8, color=BAR_TEST, zorder=2, label="11월 실제")
    ax1.fill_between(idx, out["SARIMA_lower95"], out["SARIMA_upper95"], color=SARIMA_BAND, zorder=1,
                     label=f"SARIMA 95% 예측구간 (실제 포함 {coverage:.0%})")
    ax1.plot(idx, out["SARIMA_forecast"], color=COLORS["SARIMA"], linewidth=2.5, marker="o", markersize=3.5,
             zorder=4, label="SARIMA 예측")
    ax1.axvline(idx[0] - pd.Timedelta(hours=12), color=TEXT_2, linestyle="--", linewidth=1)
    ax1.set_title(f"① SARIMA{order}{sorder}{' + 상수항' if trend == 'c' else ''} 예측과 95% 예측구간",
                  loc="left", fontsize=12, color=TEXT)
    ax1.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 8, 15, 22]))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax1.set_xlim(tr.index[0] - pd.Timedelta(days=1), idx[-1] + pd.Timedelta(days=1))
    ax1.legend(loc="upper left", frameon=False, fontsize=8.5, ncol=4)
    lo = min(0, out["SARIMA_lower95"].min() * 1.05)
    ax1.set_ylim(lo, max(out["SARIMA_upper95"].max(), out["actual"].max()) * 1.12)

    # ② 11월 4개 모형 비교
    for d in idx[out["is_closed"] == 1]:
        ax2.axvspan(d - pd.Timedelta(hours=12), d + pd.Timedelta(hours=12), color=CLOSED, zorder=0, linewidth=0)
    ax2.bar(idx, out["actual"], width=0.75, color=BAR_TEST, zorder=2, label="11월 실제")
    for mname in MODELS:
        strong = mname in ("Holt-Winters", "SARIMA")
        ax2.plot(idx, out[f"{mname}_forecast"], color=COLORS[mname], linewidth=2.4 if strong else 1.6,
                 marker="o" if strong else None, markersize=3.5, zorder=4, label=mname)
    ax2.set_title("② 11월 실제 vs 4개 모형 예측 (회색 = 휴무일 토요일)", loc="left", fontsize=12, color=TEXT)
    ax2.set_xticks(pd.to_datetime([f"2011-11-{d:02d}" for d in [1, 5, 10, 15, 20, 25, 30]]))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax2.set_xlim(idx[0] - pd.Timedelta(hours=18), idx[-1] + pd.Timedelta(hours=18))
    ax2.set_ylim(min(0, out[[f"{m}_forecast" for m in MODELS]].min().min() * 1.1), out["actual"].max() * 1.15)
    ax2.legend(loc="upper left", frameon=False, fontsize=8.5, ncol=5)

    for ax in (ax1, ax2):
        ax.axhline(0, color=TEXT_2, linewidth=0.8, zorder=1)
        ax.set_ylabel("순매출 (천 £)", color=TEXT_2)
        ax.yaxis.set_major_formatter(k)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(TEXT_2)
        ax.tick_params(colors=TEXT_2, length=0)
        ax.tick_params(axis="x", length=4)

    fig.suptitle("SARIMA vs 지수평활 모형 · 11월 일별 순매출 예측 (학습 2011-01-01 ~ 10-31)",
                 x=0.01, ha="left", fontsize=14, color=TEXT)
    fig.text(0.01, 0.01,
             "SARIMA 차수: 7일 계절 차분(D=1) 후 p,q∈{0,1,2}, P,Q∈{0,1}, 상수항 유무 중 학습 AIC 최소. "
             "모든 모형은 2011-10-31 시점에서 11월 30일을 한 번에 예측.",
             fontsize=8.5, color=TEXT_2)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.07)
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, facecolor="white")


if __name__ == "__main__":
    main()
