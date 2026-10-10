"""Holt 선형 추세 모형(감쇠 없음)으로 2011년 11월 일별 순매출 예측, 단순 지수평활(SES)과 비교.

기준: 전처리_기준.md 6장 (6-1 구간 정의)
- 학습: 2011-01-01 ~ 2011-10-31 (304일), 비교: 2011-11-01 ~ 2011-11-30 (30일)
- 예측 대상: net_sales (일 단위, 휴무일 0 포함), 변환 없음
- 수준 평활계수 α, 추세 평활계수 β, 초기 수준 ℓ0, 초기 추세 b0는 학습 자료로만 추정
  (SSE 최소화, statsmodels Holt, damped_trend=False, initialization_method="estimated").
- 최적화: statsmodels 기본(L-BFGS-B)은 이 자료에서 SSE가 SES보다도 큰 국소해에 멈춘다
  (Holt는 β=0, b0=0이면 SES와 같으므로 전역 최적 SSE는 SES 이하여야 함).
  그래서 결정적인 전역 탐색에 가까운 method="trust-constr"로 추정하고, 기본 방법 결과는 기록용으로 함께 저장한다.
- 2011-10-31 시점에서 11월 30일치를 한 번에 예측한다. h일 뒤 예측 = ℓ_T + h × b_T.
- SES는 10_ses_forecast.py와 같은 설정으로 다시 적합해 나란히 비교한다.

입력: data/processed/daily_sales.csv
출력:
  outputs/tables/holt_params.csv              α, β, ℓ0, b0, 마지막 수준·추세, 학습 SSE
  outputs/tables/ses_holt_forecast_nov.csv    11월 일별 실제·SES·Holt 예측
  outputs/tables/ses_holt_metrics_nov.csv     모형별 오차 지표 (전체 30일 / 영업일 26일)
  outputs/figures/ses_holt_forecast_nov.png
실행: 프로젝트 루트에서 `python analysis/11_holt_forecast.py`
"""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from statsmodels.tsa.holtwinters import Holt, SimpleExpSmoothing

ROOT = Path(__file__).resolve().parent.parent
DAILY_PATH = ROOT / "data" / "processed" / "daily_sales.csv"
TABLES = ROOT / "outputs" / "tables"
PARAMS_PATH = TABLES / "holt_params.csv"
FORECAST_PATH = TABLES / "ses_holt_forecast_nov.csv"
METRICS_PATH = TABLES / "ses_holt_metrics_nov.csv"
FIG_PATH = ROOT / "outputs" / "figures" / "ses_holt_forecast_nov.png"

TRAIN = ("2011-01-01", "2011-10-31")
TEST = ("2011-11-01", "2011-11-30")

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

BLUE = "#2a78d6"
BAR_TRAIN = "#d3e3f7"
BAR_TEST = "#8fb6e8"
ORANGE = "#eb6834"   # SES
AQUA = "#1baf7a"     # Holt
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


def main() -> None:
    daily = pd.read_csv(DAILY_PATH, parse_dates=["date"]).set_index("date").asfreq("D")
    y = daily["net_sales"]
    train = y.loc[TRAIN[0]:TRAIN[1]]
    test = y.loc[TEST[0]:TEST[1]]
    assert len(train) == 304 and len(test) == 30

    ses = SimpleExpSmoothing(train, initialization_method="estimated").fit(optimized=True)
    holt_model = Holt(train, damped_trend=False, initialization_method="estimated")
    holt = holt_model.fit(optimized=True, method="trust-constr")
    holt_default = holt_model.fit(optimized=True)  # 기록용: 기본 최적화(L-BFGS-B) 결과
    assert holt.sse <= ses.sse, "Holt SSE가 SES보다 크면 최적화 실패"
    fc_ses = pd.Series(ses.forecast(len(test)).to_numpy(), index=test.index)
    fc_holt = pd.Series(holt.forecast(len(test)).to_numpy(), index=test.index)

    p = holt.params
    params = {
        "model": "Holt (damped_trend=False)", "train_start": TRAIN[0], "train_end": TRAIN[1],
        "alpha": p["smoothing_level"], "beta": p["smoothing_trend"],
        "initial_level": p["initial_level"], "initial_trend": p["initial_trend"],
        "final_level": holt.level.iloc[-1], "final_trend": holt.trend.iloc[-1],
        "train_sse": holt.sse, "train_rmse": np.sqrt(holt.sse / len(train)),
        "ses_alpha": ses.params["smoothing_level"], "ses_train_rmse": np.sqrt(ses.sse / len(train)),
    }
    q = holt_default.params
    default_row = {
        "model": "Holt (damped_trend=False) - 기본 최적화 L-BFGS-B, 참고용", "train_start": TRAIN[0],
        "train_end": TRAIN[1], "alpha": q["smoothing_level"], "beta": q["smoothing_trend"],
        "initial_level": q["initial_level"], "initial_trend": q["initial_trend"],
        "final_level": holt_default.level.iloc[-1], "final_trend": holt_default.trend.iloc[-1],
        "train_sse": holt_default.sse, "train_rmse": np.sqrt(holt_default.sse / len(train)),
    }
    params["model"] = "Holt (damped_trend=False) - trust-constr, 채택"
    TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([params, default_row]).to_csv(PARAMS_PATH, index=False)

    fc = pd.DataFrame({
        "date": test.index, "weekday": [DOW_NAMES[d] for d in test.index.dayofweek],
        "is_closed": daily.loc[test.index, "is_closed"].to_numpy(),
        "actual": test.to_numpy(), "ses_forecast": fc_ses.to_numpy(), "holt_forecast": fc_holt.to_numpy(),
    })
    fc["ses_error"] = fc["ses_forecast"] - fc["actual"]
    fc["holt_error"] = fc["holt_forecast"] - fc["actual"]
    fc.to_csv(FORECAST_PATH, index=False, date_format="%Y-%m-%d")

    open_mask = fc["is_closed"] == 0
    rows = []
    for name, col in [("SES", "ses_forecast"), ("Holt", "holt_forecast")]:
        rows.append({"model": name, "scope": "전체 30일", **metrics(fc["actual"], fc[col])})
        rows.append({"model": name, "scope": "영업일 26일",
                     **metrics(fc.loc[open_mask, "actual"], fc.loc[open_mask, col])})
    m = pd.DataFrame(rows)
    m.to_csv(METRICS_PATH, index=False)

    plot(train, test, ses, holt, fc_ses, fc_holt, daily, params)

    print(f"Holt: α = {params['alpha']:.6f}, β = {params['beta']:.6f}, "
          f"ℓ0 = {params['initial_level']:,.2f}, b0 = {params['initial_trend']:,.4f}")
    print(f"      마지막 수준 = {params['final_level']:,.2f}, 마지막 추세 = {params['final_trend']:,.4f} /일, "
          f"학습 RMSE = {params['train_rmse']:,.2f} (SES {params['ses_train_rmse']:,.2f})")
    print(f"      11월 예측: {fc_holt.iloc[0]:,.2f} (11-01) → {fc_holt.iloc[-1]:,.2f} (11-30)")
    print(f"      학습 SSE: Holt {holt.sse:,.2f} / SES {ses.sse:,.2f} / Holt 기본 최적화 {holt_default.sse:,.2f} "
          f"(α={default_row['alpha']:.5f}, β={default_row['beta']:.5f}, 11월 합계 "
          f"{holt_default.forecast(len(test)).sum():,.2f})")
    with pd.option_context("display.float_format", "{:,.4f}".format, "display.width", 250):
        print(m.to_string(index=False))


def plot(train, test, ses, holt, fc_ses, fc_holt, daily, params) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9.5), dpi=150,
                                   gridspec_kw={"height_ratios": [1, 1.1], "hspace": 0.38})
    fig.patch.set_facecolor("white")
    k = FuncFormatter(lambda v, _: f"{v / 1000:,.0f}")
    ses_lbl = f"SES (평활계수 {ses.params['smoothing_level']:.4f})"
    holt_lbl = f"Holt (수준 {params['alpha']:.4f}, 추세 {params['beta']:.4f})"

    # ① 학습 구간 적합 수준과 11월 예측
    ax1.bar(train.index, train, width=0.8, color=BAR_TRAIN, zorder=2, label="학습 실제 (2011-01 ~ 10)")
    ax1.bar(test.index, test, width=0.8, color=BAR_TEST, zorder=2, label="11월 실제 (비교)")
    ax1.plot(ses.level.index, ses.level, color=ORANGE, linewidth=1.2, alpha=0.8, zorder=3)
    ax1.plot(holt.level.index, holt.level, color=AQUA, linewidth=1.2, alpha=0.8, zorder=3)
    ax1.plot(fc_ses.index, fc_ses, color=ORANGE, linewidth=2.5, zorder=4, label=ses_lbl)
    ax1.plot(fc_holt.index, fc_holt, color=AQUA, linewidth=2.5, zorder=4, label=holt_lbl)
    ax1.axvline(pd.Timestamp(TEST[0]) - pd.Timedelta(hours=12), color=TEXT_2, linestyle="--", linewidth=1)
    ax1.text(pd.Timestamp(TEST[0]), 0.97, " 예측 시작", transform=ax1.get_xaxis_transform(),
             va="top", fontsize=9, color=TEXT_2)
    ax1.set_title("① 학습 구간 적합 수준(가는 선)과 11월 예측(굵은 선)", loc="left", fontsize=12, color=TEXT)
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.set_xlim(pd.Timestamp(TRAIN[0]) - pd.Timedelta(days=3), pd.Timestamp(TEST[1]) + pd.Timedelta(days=3))
    ax1.legend(loc="upper left", frameon=False, fontsize=9, ncol=2)

    # ② 11월 확대
    for d in test.index[daily.loc[test.index, "is_closed"] == 1]:
        ax2.axvspan(d - pd.Timedelta(hours=12), d + pd.Timedelta(hours=12), color=CLOSED, zorder=0, linewidth=0)
    ax2.bar(test.index, test, width=0.75, color=BAR_TEST, zorder=2, label="11월 실제 순매출")
    ax2.plot(fc_ses.index, fc_ses, color=ORANGE, linewidth=2.5, zorder=4, label=ses_lbl)
    ax2.plot(fc_holt.index, fc_holt, color=AQUA, linewidth=2.5, zorder=4, label=holt_lbl)
    upper, lower = (fc_ses, fc_holt) if fc_ses.iloc[-1] >= fc_holt.iloc[-1] else (fc_holt, fc_ses)
    for s, dy, name in [(upper, 9, "SES" if upper is fc_ses else "Holt"),
                        (lower, -9, "SES" if lower is fc_ses else "Holt")]:
        ax2.annotate(f"{name} £{s.iloc[-1] / 1000:,.1f}천", (s.index[-1], s.iloc[-1]), xytext=(8, dy),
                     textcoords="offset points", va="center", fontsize=9, color=TEXT)
    ax2.set_title("② 11월 실제 vs 예측 (회색 = 휴무일 토요일)", loc="left", fontsize=12, color=TEXT)
    ax2.set_xticks(pd.to_datetime([f"2011-11-{d:02d}" for d in [1, 5, 10, 15, 20, 25, 30]]))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax2.set_xlim(pd.Timestamp(TEST[0]) - pd.Timedelta(hours=18), pd.Timestamp(TEST[1]) + pd.Timedelta(days=3))
    ax2.legend(loc="upper left", frameon=False, fontsize=9, ncol=3)

    top = max(train.max(), test.max(), fc_holt.max()) * 1.12
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

    fig.suptitle("단순 지수평활(SES) vs Holt 선형 추세(감쇠 없음) · 11월 일별 순매출 예측",
                 x=0.01, ha="left", fontsize=14, color=TEXT)
    fig.text(0.01, 0.01,
             "두 모형 모두 학습 구간(2011-01-01 ~ 10-31, 휴무일 0 포함)의 제곱오차합을 최소화해 계수와 초기값을 추정. "
             "2011-10-31 시점에서 11월 30일을 한 번에 예측 (Holt: 마지막 수준 + 경과일 × 마지막 추세).",
             fontsize=8.5, color=TEXT_2)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.92, bottom=0.07)
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, facecolor="white")


if __name__ == "__main__":
    main()
