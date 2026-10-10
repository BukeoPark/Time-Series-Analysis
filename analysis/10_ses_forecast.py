"""단순 지수평활(SES)로 2011년 11월 일별 순매출 예측.

기준: 전처리_기준.md 6장 (6-1 구간 정의)
- 학습: 2011-01-01 ~ 2011-10-31 (304일), 비교: 2011-11-01 ~ 2011-11-30 (30일)
- 예측 대상: net_sales (일 단위, 휴무일 0 포함), 변환 없음
- 평활계수 α와 초기 수준 ℓ0는 학습 자료로만 추정한다 (SSE 최소화, statsmodels
  SimpleExpSmoothing, initialization_method="estimated").
- 2011-10-31 시점에서 11월 30일치를 한 번에 예측한다. 11월 실제 값은 비교에만 쓴다.
  SES의 h-단계 예측은 마지막 수준 ℓ_T 로 모두 같다.

입력: data/processed/daily_sales.csv
출력:
  outputs/tables/ses_params.csv          α, ℓ0, 마지막 수준, 학습 SSE
  outputs/tables/ses_forecast_nov.csv    11월 일별 실제·예측·오차
  outputs/tables/ses_metrics_nov.csv     오차 지표 (전체 30일 / 영업일 26일)
  outputs/figures/ses_forecast_nov.png
실행: 프로젝트 루트에서 `python analysis/10_ses_forecast.py`
"""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

ROOT = Path(__file__).resolve().parent.parent
DAILY_PATH = ROOT / "data" / "processed" / "daily_sales.csv"
TABLES = ROOT / "outputs" / "tables"
PARAMS_PATH = TABLES / "ses_params.csv"
FORECAST_PATH = TABLES / "ses_forecast_nov.csv"
METRICS_PATH = TABLES / "ses_metrics_nov.csv"
FIG_PATH = ROOT / "outputs" / "figures" / "ses_forecast_nov.png"

TRAIN = ("2011-01-01", "2011-10-31")
TEST = ("2011-11-01", "2011-11-30")

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

BLUE = "#2a78d6"
BAR = "#b9d3f2"
ORANGE = "#eb6834"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
CLOSED = "#efeeea"
DOW_NAMES = ["월", "화", "수", "목", "금", "토", "일"]


def metrics(actual: pd.Series, pred: pd.Series) -> dict:
    err = actual - pred
    out = {
        "n_days": len(actual),
        "actual_total": actual.sum(),
        "forecast_total": pred.sum(),
        "total_error": pred.sum() - actual.sum(),
        "total_error_pct": pred.sum() / actual.sum() - 1,
        "MAE": err.abs().mean(),
        "RMSE": np.sqrt((err ** 2).mean()),
        "bias_mean_error": (pred - actual).mean(),  # 음수면 과소 예측
    }
    nz = actual != 0
    out["MAPE"] = (err[nz].abs() / actual[nz]).mean() if nz.all() else np.nan  # 0이 있으면 계산 안 함
    return out


def main() -> None:
    daily = pd.read_csv(DAILY_PATH, parse_dates=["date"]).set_index("date").asfreq("D")
    y = daily["net_sales"]
    train = y.loc[TRAIN[0]:TRAIN[1]]
    test = y.loc[TEST[0]:TEST[1]]
    assert len(train) == 304 and len(test) == 30

    fit = SimpleExpSmoothing(train, initialization_method="estimated").fit(optimized=True)
    alpha = fit.params["smoothing_level"]
    l0 = fit.params["initial_level"]
    level_T = fit.level.iloc[-1]
    forecast = fit.forecast(len(test))
    forecast.index = test.index

    # 저장: 파라미터
    TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "model": "SimpleExpSmoothing", "train_start": TRAIN[0], "train_end": TRAIN[1],
        "alpha": alpha, "initial_level": l0, "final_level": level_T,
        "train_sse": fit.sse, "train_rmse": np.sqrt(fit.sse / len(train)),
    }]).to_csv(PARAMS_PATH, index=False)

    # 저장: 11월 일별 비교
    fc = pd.DataFrame({
        "date": test.index, "weekday": [DOW_NAMES[d] for d in test.index.dayofweek],
        "is_closed": daily.loc[test.index, "is_closed"].to_numpy(),
        "actual": test.to_numpy(), "forecast": forecast.to_numpy(),
    })
    fc["error"] = fc["forecast"] - fc["actual"]
    fc.to_csv(FORECAST_PATH, index=False, date_format="%Y-%m-%d")

    # 저장: 오차 지표
    open_mask = fc["is_closed"] == 0
    m = pd.DataFrame([
        {"scope": "전체 30일", **metrics(fc["actual"], fc["forecast"])},
        {"scope": "영업일 26일", **metrics(fc.loc[open_mask, "actual"], fc.loc[open_mask, "forecast"])},
    ])
    m.to_csv(METRICS_PATH, index=False)

    plot(y, train, test, fit, forecast, daily, alpha, l0, level_T)

    print(f"α = {alpha:.6f}, 초기 수준 ℓ0 = {l0:,.2f}, 마지막 수준 ℓ_T(=11월 예측값) = {level_T:,.2f}")
    print(f"학습 SSE = {fit.sse:,.2f}, 학습 RMSE = {np.sqrt(fit.sse / len(train)):,.2f}")
    with pd.option_context("display.float_format", "{:,.4f}".format, "display.width", 250):
        print(m.to_string(index=False))
    print(f"\n저장: {PARAMS_PATH.relative_to(ROOT)}, {FORECAST_PATH.relative_to(ROOT)}, "
          f"{METRICS_PATH.relative_to(ROOT)}, {FIG_PATH.relative_to(ROOT)}")


def plot(y, train, test, fit, forecast, daily, alpha, l0, level_T) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9.5), dpi=150,
                                   gridspec_kw={"height_ratios": [1, 1.1], "hspace": 0.38})
    fig.patch.set_facecolor("white")
    k = FuncFormatter(lambda v, _: f"{v / 1000:,.0f}")

    # ① 전체: 학습 실제·적합 수준, 11월 실제·예측
    ax1.bar(train.index, train, width=0.8, color=BAR, zorder=2, label="학습 실제 (2011-01 ~ 10)")
    ax1.bar(test.index, test, width=0.8, color="#8fb6e8", zorder=2, label="11월 실제 (비교)")
    ax1.plot(fit.level.index, fit.level, color=TEXT_2, linewidth=1.3, zorder=3, label="SES 수준 (학습 구간 적합)")
    ax1.plot(forecast.index, forecast, color=ORANGE, linewidth=2.5, zorder=4, label="SES 예측 (11월)")
    ax1.axvline(pd.Timestamp(TEST[0]) - pd.Timedelta(hours=12), color=TEXT_2, linestyle="--", linewidth=1)
    ax1.text(pd.Timestamp(TEST[0]), 0.97, " 예측 시작", transform=ax1.get_xaxis_transform(),
             va="top", fontsize=9, color=TEXT_2)
    ax1.set_title("① 학습 구간과 11월 예측", loc="left", fontsize=12, color=TEXT)
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.set_xlim(pd.Timestamp(TRAIN[0]) - pd.Timedelta(days=3), pd.Timestamp(TEST[1]) + pd.Timedelta(days=3))
    ax1.legend(loc="upper left", frameon=False, fontsize=9, ncol=2)

    # ② 11월 확대: 실제 vs 예측
    for d in test.index[daily.loc[test.index, "is_closed"] == 1]:
        ax2.axvspan(d - pd.Timedelta(hours=12), d + pd.Timedelta(hours=12), color=CLOSED, zorder=0, linewidth=0)
    ax2.bar(test.index, test, width=0.75, color="#8fb6e8", zorder=2, label="11월 실제 순매출")
    ax2.plot(forecast.index, forecast, color=ORANGE, linewidth=2.5, zorder=4,
             label=f"SES 예측 (매일 £{level_T:,.0f})")
    ax2.axhline(test[daily.loc[test.index, "is_closed"] == 0].mean(), color=BLUE, linewidth=1.3,
                linestyle=":", zorder=3, label="11월 실제 영업일 평균")
    ax2.set_title("② 11월 실제 vs 예측 (회색 = 휴무일 토요일)", loc="left", fontsize=12, color=TEXT)
    ax2.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 5, 10, 15, 20, 25, 30]))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax2.set_xlim(pd.Timestamp(TEST[0]) - pd.Timedelta(hours=18), pd.Timestamp(TEST[1]) + pd.Timedelta(hours=18))
    ax2.legend(loc="upper left", frameon=False, fontsize=9, ncol=3)

    top = max(y.loc[TRAIN[0]:TEST[1]].max(), forecast.max()) * 1.12
    for ax in (ax1, ax2):
        ax.set_ylim(0, top)
        ax.set_ylabel("순매출 (천 £)", color=TEXT_2)
        ax.yaxis.set_major_formatter(k)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(TEXT_2)
        ax.tick_params(colors=TEXT_2, length=0)
        ax.tick_params(axis="x", length=4)

    fig.suptitle(f"단순 지수평활(SES) 11월 일별 순매출 예측  ·  평활계수 {alpha:.4f}  ·  "
                 f"초기 수준 £{l0:,.0f}", x=0.01, ha="left", fontsize=14, color=TEXT)
    fig.text(0.01, 0.01,
             "평활계수와 초기 수준은 학습 구간(2011-01-01 ~ 10-31, 휴무일 0 포함)의 제곱오차합을 최소화해 추정. "
             "2011-10-31 시점에서 11월 30일을 한 번에 예측하므로 예측값은 모두 학습 마지막 날의 수준과 같음.",
             fontsize=8.5, color=TEXT_2)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.07)
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, facecolor="white")


if __name__ == "__main__":
    main()
