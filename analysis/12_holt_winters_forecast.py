"""Holt-Winters(가법 추세 + 가법 요일 계절성, 주기 7)로 2011년 11월 일별 순매출 예측,
단순 지수평활(SES)·Holt와 비교.

기준: 전처리_기준.md 6장 (6-1 구간 정의)
- 학습: 2011-01-01 ~ 2011-10-31 (304일), 비교: 2011-11-01 ~ 2011-11-30 (30일)
- 예측 대상: net_sales (일 단위, 휴무일 0 포함), 변환 없음
- 모형: trend="add", damped_trend=False (Holt와 같게 감쇠 없음), seasonal="add", seasonal_periods=7
  곱셈 계절성은 매주 토요일 값이 0이라 적용할 수 없다 (statsmodels는 양수 자료만 허용).
- α(수준), β(추세), γ(계절), 초기 수준·추세·7개 초기 계절값은 학습 자료로만 추정 (SSE 최소화,
  initialization_method="estimated"). 국소해를 피하려고 결정적인 최적화 방법 여러 개로 적합한 뒤
  학습 SSE가 가장 작은 결과를 채택한다. 모든 방법의 결과는 기록한다.
- 2011-10-31 시점에서 11월 30일치를 한 번에 예측한다.
- SES, Holt는 10·11번 스크립트와 같은 설정으로 다시 적합해 나란히 비교한다.

입력: data/processed/daily_sales.csv
출력:
  outputs/tables/hw_params.csv                   채택 모형의 계수·초기값
  outputs/tables/hw_optimizer_runs.csv           최적화 방법별 SSE와 계수
  outputs/tables/es_models_forecast_nov.csv      11월 일별 실제·SES·Holt·Holt-Winters 예측
  outputs/tables/es_models_metrics_nov.csv       모형별 오차 지표 (전체 30일 / 영업일 26일)
  outputs/figures/holt_winters_forecast_nov.png
실행: 프로젝트 루트에서 `python analysis/12_holt_winters_forecast.py`
"""
import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt, SimpleExpSmoothing

ROOT = Path(__file__).resolve().parent.parent
DAILY_PATH = ROOT / "data" / "processed" / "daily_sales.csv"
TABLES = ROOT / "outputs" / "tables"
PARAMS_PATH = TABLES / "hw_params.csv"
RUNS_PATH = TABLES / "hw_optimizer_runs.csv"
FORECAST_PATH = TABLES / "es_models_forecast_nov.csv"
METRICS_PATH = TABLES / "es_models_metrics_nov.csv"
FIG_PATH = ROOT / "outputs" / "figures" / "holt_winters_forecast_nov.png"

TRAIN = ("2011-01-01", "2011-10-31")
TEST = ("2011-11-01", "2011-11-30")
PERIOD = 7
METHODS = ["L-BFGS-B", "trust-constr", "TNC", "SLSQP", "Powell", "least_squares"]  # 결정적 방법만

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

BLUE = "#2a78d6"
BAR_TRAIN = "#d3e3f7"
BAR_TEST = "#8fb6e8"
ORANGE = "#eb6834"   # SES
AQUA = "#1baf7a"     # Holt
VIOLET = "#4a3aa7"   # Holt-Winters
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
CLOSED = "#efeeea"
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


def fit_holt_winters(train: pd.Series):
    model = ExponentialSmoothing(train, trend="add", damped_trend=False, seasonal="add",
                                 seasonal_periods=PERIOD, initialization_method="estimated")
    runs, best = [], None
    for method in METHODS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = model.fit(optimized=True, method=method)
        p = res.params
        runs.append({"method": method, "train_sse": res.sse, "alpha": p["smoothing_level"],
                     "beta": p["smoothing_trend"], "gamma": p["smoothing_seasonal"],
                     "initial_level": p["initial_level"], "initial_trend": p["initial_trend"]})
        if best is None or res.sse < best[1].sse:
            best = (method, res)
    return best, pd.DataFrame(runs)


def main() -> None:
    daily = pd.read_csv(DAILY_PATH, parse_dates=["date"]).set_index("date").asfreq("D")
    y = daily["net_sales"]
    train = y.loc[TRAIN[0]:TRAIN[1]]
    test = y.loc[TEST[0]:TEST[1]]
    assert len(train) == 304 and len(test) == 30

    ses = SimpleExpSmoothing(train, initialization_method="estimated").fit(optimized=True)
    holt = Holt(train, damped_trend=False, initialization_method="estimated").fit(
        optimized=True, method="trust-constr")
    (hw_method, hw), runs = fit_holt_winters(train)
    assert hw.sse <= holt.sse <= ses.sse, "중첩 모형인데 SSE 순서가 맞지 않음 (최적화 확인 필요)"

    fcs = {
        "SES": pd.Series(ses.forecast(len(test)).to_numpy(), index=test.index),
        "Holt": pd.Series(holt.forecast(len(test)).to_numpy(), index=test.index),
        "Holt-Winters": pd.Series(hw.forecast(len(test)).to_numpy(), index=test.index),
    }

    # 계수와 초기값 (초기 계절값은 학습 첫날부터의 요일 순서로 저장)
    p = hw.params
    init_season = np.asarray(p["initial_seasons"])
    first_dows = [(train.index[0] + pd.Timedelta(days=i)).dayofweek for i in range(PERIOD)]
    params = {"model": "Holt-Winters (add trend, add seasonal, m=7, no damping)", "optimizer": hw_method,
              "alpha": p["smoothing_level"], "beta": p["smoothing_trend"], "gamma": p["smoothing_seasonal"],
              "initial_level": p["initial_level"], "initial_trend": p["initial_trend"],
              "final_level": hw.level.iloc[-1], "final_trend": hw.trend.iloc[-1],
              "train_sse": hw.sse, "train_rmse": np.sqrt(hw.sse / len(train))}
    for dow, s0 in zip(first_dows, init_season):
        params[f"initial_season_{DOW_NAMES[dow]}"] = s0
    last_season = hw.season.iloc[-PERIOD:]
    for d, v in last_season.items():
        params[f"final_season_{DOW_NAMES[d.dayofweek]}"] = v

    TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([params]).to_csv(PARAMS_PATH, index=False)
    runs.to_csv(RUNS_PATH, index=False)

    fc = pd.DataFrame({"date": test.index, "weekday": [DOW_NAMES[d] for d in test.index.dayofweek],
                       "is_closed": daily.loc[test.index, "is_closed"].to_numpy(), "actual": test.to_numpy()})
    for name, s in fcs.items():
        fc[f"{name}_forecast"] = s.to_numpy()
    fc.to_csv(FORECAST_PATH, index=False, date_format="%Y-%m-%d")

    open_mask = (fc["is_closed"] == 0).to_numpy()
    rows = []
    for name, s in fcs.items():
        rows.append({"model": name, "scope": "전체 30일", **metrics(test, s)})
        rows.append({"model": name, "scope": "영업일 26일", **metrics(test[open_mask], s[open_mask])})
    m = pd.DataFrame(rows)
    m.to_csv(METRICS_PATH, index=False)

    plot(train, test, hw, fcs, daily, params)

    with pd.option_context("display.float_format", "{:,.4f}".format, "display.width", 250,
                           "display.max_columns", 40):
        print(f"채택 최적화 방법: {hw_method}")
        print(runs.to_string(index=False))
        print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in params.items()})
        print(f"학습 SSE: HW {hw.sse:,.2f} / Holt {holt.sse:,.2f} / SES {ses.sse:,.2f}")
        print(m.to_string(index=False))


def plot(train, test, hw, fcs, daily, params) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9.5), dpi=150,
                                   gridspec_kw={"height_ratios": [1, 1.1], "hspace": 0.38})
    fig.patch.set_facecolor("white")
    k = FuncFormatter(lambda v, _: f"{v / 1000:,.0f}")
    colors = {"SES": ORANGE, "Holt": AQUA, "Holt-Winters": VIOLET}
    hw_lbl = (f"Holt-Winters (수준 {params['alpha']:.4f}, 추세 {params['beta']:.4f}, "
              f"계절 {params['gamma']:.4f})")
    labels = {"SES": "SES", "Holt": "Holt", "Holt-Winters": hw_lbl}

    # ① 최근 학습 구간(9~10월) 적합값과 11월 예측
    show_from = pd.Timestamp("2011-09-01")
    tr = train.loc[show_from:]
    ax1.bar(tr.index, tr, width=0.8, color=BAR_TRAIN, zorder=2, label="학습 실제 (9~10월 표시)")
    ax1.bar(test.index, test, width=0.8, color=BAR_TEST, zorder=2, label="11월 실제 (비교)")
    fitted = hw.fittedvalues.loc[show_from:]
    ax1.plot(fitted.index, fitted, color=VIOLET, linewidth=1.2, alpha=0.7, zorder=3,
             label="Holt-Winters 학습 적합값")
    for name, s in fcs.items():
        ax1.plot(s.index, s, color=colors[name], linewidth=2.2 if name == "Holt-Winters" else 1.8,
                 zorder=4, label=f"{name} 예측")
    ax1.axvline(pd.Timestamp(TEST[0]) - pd.Timedelta(hours=12), color=TEXT_2, linestyle="--", linewidth=1)
    ax1.text(pd.Timestamp(TEST[0]), 0.97, " 예측 시작", transform=ax1.get_xaxis_transform(),
             va="top", fontsize=9, color=TEXT_2)
    ax1.set_title("① 2011-09 ~ 11 : 학습 적합값과 11월 예측", loc="left", fontsize=12, color=TEXT)
    ax1.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 15]))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax1.set_xlim(show_from - pd.Timedelta(days=1), pd.Timestamp(TEST[1]) + pd.Timedelta(days=1))
    ax1.legend(loc="upper left", frameon=False, fontsize=8.5, ncol=3)

    # ② 11월 확대
    for d in test.index[daily.loc[test.index, "is_closed"] == 1]:
        ax2.axvspan(d - pd.Timedelta(hours=12), d + pd.Timedelta(hours=12), color=CLOSED, zorder=0, linewidth=0)
    ax2.bar(test.index, test, width=0.75, color=BAR_TEST, zorder=2, label="11월 실제 순매출")
    for name, s in fcs.items():
        ax2.plot(s.index, s, color=colors[name], linewidth=2.5 if name == "Holt-Winters" else 1.8,
                 marker="o" if name == "Holt-Winters" else None, markersize=4, zorder=4, label=labels[name])
    ax2.set_title("② 11월 실제 vs 예측 (회색 = 휴무일 토요일)", loc="left", fontsize=12, color=TEXT)
    ax2.set_xticks(pd.to_datetime([f"2011-11-{d:02d}" for d in [1, 5, 10, 15, 20, 25, 30]]))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax2.set_xlim(pd.Timestamp(TEST[0]) - pd.Timedelta(hours=18), pd.Timestamp(TEST[1]) + pd.Timedelta(hours=18))
    ax2.legend(loc="upper left", frameon=False, fontsize=8.5, ncol=2)

    lo = min(0, min(s.min() for s in fcs.values()) * 1.1)
    top = max(train.loc[show_from:].max(), test.max()) * 1.18
    for ax in (ax1, ax2):
        ax.set_ylim(lo, top)
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

    fig.suptitle("Holt-Winters (가법 추세·가법 요일 계절성, 주기 7) vs Holt vs SES · 11월 일별 순매출 예측",
                 x=0.01, ha="left", fontsize=14, color=TEXT)
    fig.text(0.01, 0.01,
             "세 모형 모두 학습 구간(2011-01-01 ~ 10-31, 휴무일 0 포함)의 제곱오차합을 최소화해 계수와 초기값을 추정. "
             "2011-10-31 시점에서 11월 30일을 한 번에 예측.",
             fontsize=8.5, color=TEXT_2)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.07)
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, facecolor="white")


if __name__ == "__main__":
    main()
