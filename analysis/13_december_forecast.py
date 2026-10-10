"""2011년 12월 일별 순매출 예측 (SES, Holt, Holt-Winters, SARIMA).

- 11월 비교(10~12번 스크립트)로 모형을 평가한 뒤, 12월 예측에는 11월까지 포함한
  2011-01-01 ~ 2011-11-30 (334일)로 계수·초기값을 다시 추정한다. 모형 설정과 추정 방법은 10~12번과 같다.
- SARIMA는 15번에서 11월 비교용으로 고른 차수(sarima_order_search.csv의 AIC 1위)를 그대로 쓰고
  계수만 다시 추정한다.
- 2011-11-30 시점에서 12-01 ~ 12-31 (31일)을 한 번에 예측한다.
- 12월 실제 값은 12-01 ~ 12-08(8일)만 있다. 전처리_기준.md 6-1에 따라 이 기간은 모형 평가에 쓰지 않고,
  그래프와 표에 '참고용 실제 값'으로만 함께 보여준다.
- 작년에는 2010-12-24 ~ 2011-01-03이 연말연시 휴무였다. 모형은 이 휴무를 모르므로
  12월 하순 예측값은 휴무 여부를 확인한 뒤 해석해야 한다 (그래프에 작년 휴무 기간을 표시).

입력: data/processed/daily_sales.csv, outputs/tables/sarima_order_search.csv
출력:
  outputs/tables/es_models_params_dec.csv      12월 예측용 재추정 계수·초기값
  outputs/tables/es_models_forecast_dec.csv    12월 일별 예측 (+ SARIMA 95% 예측구간, 12-01~08 참고용 실제)
  outputs/figures/es_models_forecast_dec.png
실행: 프로젝트 루트에서 `python analysis/13_december_forecast.py`
"""
import importlib.util
import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from statsmodels.tsa.holtwinters import Holt, SimpleExpSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

ROOT = Path(__file__).resolve().parent.parent
DAILY_PATH = ROOT / "data" / "processed" / "daily_sales.csv"
TABLES = ROOT / "outputs" / "tables"
PARAMS_PATH = TABLES / "es_models_params_dec.csv"
FORECAST_PATH = TABLES / "es_models_forecast_dec.csv"
FIG_PATH = ROOT / "outputs" / "figures" / "es_models_forecast_dec.png"
SARIMA_SEARCH_PATH = TABLES / "sarima_order_search.csv"

TRAIN = ("2011-01-01", "2011-11-30")
HORIZON = pd.date_range("2011-12-01", "2011-12-31", freq="D")
LAST_YEAR_CLOSURE = (pd.Timestamp("2011-12-24"), pd.Timestamp("2011-12-31"))  # 작년 12-24~ 휴무에 대응

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

BAR_TRAIN = "#d3e3f7"
BAR_TEST = "#8fb6e8"
ORANGE, AQUA, VIOLET, MAGENTA = "#eb6834", "#1baf7a", "#4a3aa7", "#e87ba4"
TEXT, TEXT_2, GRID, CLOSED = "#0b0b0b", "#52514e", "#e4e3df", "#efeeea"
DOW_NAMES = ["월", "화", "수", "목", "금", "토", "일"]


def load_hw_module():
    spec = importlib.util.spec_from_file_location("hw", ROOT / "analysis" / "12_holt_winters_forecast.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    hw_mod = load_hw_module()
    daily = pd.read_csv(DAILY_PATH, parse_dates=["date"]).set_index("date").asfreq("D")
    y = daily["net_sales"]
    train = y.loc[TRAIN[0]:TRAIN[1]]
    assert len(train) == 334

    ses = SimpleExpSmoothing(train, initialization_method="estimated").fit(optimized=True)
    holt = Holt(train, damped_trend=False, initialization_method="estimated").fit(
        optimized=True, method="trust-constr")
    (hw_method, hw), _ = hw_mod.fit_holt_winters(train)
    assert hw.sse <= holt.sse <= ses.sse, "중첩 모형인데 SSE 순서가 맞지 않음 (최적화 확인 필요)"

    best = pd.read_csv(SARIMA_SEARCH_PATH).query("converged").iloc[0]
    order = (int(best["p"]), int(best["d"]), int(best["q"]))
    sorder = (int(best["P"]), int(best["D"]), int(best["Q"]), int(best["m"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sarima = SARIMAX(train, order=order, seasonal_order=sorder, trend=best["trend"],
                         enforce_stationarity=True, enforce_invertibility=True).fit(disp=False, maxiter=500)

    h = len(HORIZON)
    sar_fc = sarima.get_forecast(h)
    sar_ci = sar_fc.conf_int(alpha=0.05)
    fcs = {"SES": ses.forecast(h).to_numpy(), "Holt": holt.forecast(h).to_numpy(),
           "Holt-Winters": hw.forecast(h).to_numpy(), "SARIMA": sar_fc.predicted_mean.to_numpy()}

    rows = []
    for name, res in [("SES", ses), ("Holt", holt), ("Holt-Winters", hw)]:
        p = res.params
        rows.append({
            "model": name, "train_start": TRAIN[0], "train_end": TRAIN[1],
            "optimizer": {"SES": "L-BFGS-B", "Holt": "trust-constr", "Holt-Winters": hw_method}[name],
            "alpha": p["smoothing_level"],
            "beta": p.get("smoothing_trend", np.nan) if name != "SES" else np.nan,
            "gamma": p.get("smoothing_seasonal", np.nan) if name == "Holt-Winters" else np.nan,
            "initial_level": p["initial_level"],
            "initial_trend": p["initial_trend"] if name != "SES" else np.nan,
            "final_level": res.level.iloc[-1],
            "final_trend": res.trend.iloc[-1] if name != "SES" else np.nan,
            "train_rmse": np.sqrt(res.sse / len(train)),
            "dec_forecast_total": fcs[name].sum(),
        })
    sp = sarima.params
    rows.append({
        "model": "SARIMA", "train_start": TRAIN[0], "train_end": TRAIN[1],
        "optimizer": f"MLE, SARIMA{order}{sorder} trend={best['trend']}",
        "train_rmse": np.sqrt((sarima.resid.iloc[sorder[3] + 1:] ** 2).mean()),
        "dec_forecast_total": fcs["SARIMA"].sum(),
        **{f"sarima_{k}": v for k, v in sp.items()},
    })
    params = pd.DataFrame(rows)
    # 가법 계절 모형은 수준과 계절값 사이에 상수를 옮겨도 적합·예측이 같다 (식별 불가).
    # 해석을 위해 계절값 7개 평균이 0이 되도록 수준을 조정해 기록한다. 예측값은 변하지 않는다.
    init_s = np.asarray(hw.params["initial_seasons"])
    last_s = hw.season.iloc[-7:]
    i = params.index[params["model"] == "Holt-Winters"][0]
    params.loc[i, "initial_level"] += init_s.mean()
    params.loc[i, "final_level"] += last_s.mean()
    for d, v in (last_s - last_s.mean()).items():
        params.loc[i, f"final_season_{DOW_NAMES[d.dayofweek]}"] = v

    out = pd.DataFrame({"date": HORIZON, "weekday": [DOW_NAMES[d] for d in HORIZON.dayofweek]})
    out["actual_reference"] = y.reindex(HORIZON).to_numpy()  # 12-01~08만 값 있음 (평가에 쓰지 않음)
    for name, v in fcs.items():
        out[f"{name}_forecast"] = v
    out["SARIMA_lower95"] = sar_ci.iloc[:, 0].to_numpy()
    out["SARIMA_upper95"] = sar_ci.iloc[:, 1].to_numpy()

    TABLES.mkdir(parents=True, exist_ok=True)
    params.to_csv(PARAMS_PATH, index=False)
    out.to_csv(FORECAST_PATH, index=False, date_format="%Y-%m-%d")
    plot(y, train, out)

    ref = out.dropna(subset=["actual_reference"])
    with pd.option_context("display.float_format", "{:,.4f}".format, "display.width", 250):
        print(params.to_string(index=False))
        print("\n12-01~08 참고 비교 (평가 아님): 실제 합계 "
              f"{ref['actual_reference'].sum():,.2f}, " + ", ".join(
                  f"{n} {ref[f'{n}_forecast'].sum():,.2f}" for n in fcs))


def plot(y, train, out) -> None:
    fig, ax = plt.subplots(figsize=(13, 6), dpi=150)
    fig.patch.set_facecolor("white")
    show_from = pd.Timestamp("2011-10-01")
    tr = train.loc[show_from:]
    ref = out.set_index("date")["actual_reference"].dropna()

    ax.axvspan(LAST_YEAR_CLOSURE[0] - pd.Timedelta(hours=12), LAST_YEAR_CLOSURE[1] + pd.Timedelta(hours=12),
               color=CLOSED, zorder=0, linewidth=0)
    ax.text(LAST_YEAR_CLOSURE[1] + pd.Timedelta(hours=12), 0.97, "작년 이 시기 휴무\n(2010-12-24 ~ 01-03) ",
            transform=ax.get_xaxis_transform(), va="top", ha="right", fontsize=9, color=TEXT_2)
    ax.bar(tr.index, tr, width=0.8, color=BAR_TRAIN, zorder=2, label="학습 실제 (10~11월 표시)")
    ax.bar(ref.index, ref, width=0.8, color=BAR_TEST, zorder=2, label="12-01~08 실제 (참고, 평가 제외)")
    for name, c, lw in [("SES", ORANGE, 1.8), ("Holt", AQUA, 1.8), ("Holt-Winters", VIOLET, 2.5),
                        ("SARIMA", MAGENTA, 2.5)]:
        ax.plot(out["date"], out[f"{name}_forecast"], color=c, linewidth=lw,
                marker="o" if lw > 2 else None, markersize=3.5, zorder=4, label=f"{name} 예측")
    ax.axvline(HORIZON[0] - pd.Timedelta(hours=12), color=TEXT_2, linestyle="--", linewidth=1)
    ax.text(HORIZON[0], 0.80, " 예측 시작\n (11-30 시점)", transform=ax.get_xaxis_transform(),
            va="top", fontsize=9, color=TEXT_2)

    ax.set_xlim(show_from - pd.Timedelta(days=1), HORIZON[-1] + pd.Timedelta(days=1))
    ax.set_ylim(0, max(tr.max(), ref.max()) * 1.15)
    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 15]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.set_ylabel("순매출 (천 £)", color=TEXT_2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:,.0f}"))
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(TEXT_2)
    ax.tick_params(colors=TEXT_2, length=0)
    ax.tick_params(axis="x", length=4)
    ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=4)

    fig.suptitle("2011년 12월 일별 순매출 예측 (학습 2011-01-01 ~ 11-30로 재추정)",
                 x=0.01, ha="left", fontsize=14, color=TEXT)
    fig.text(0.01, 0.01, "모형 설정은 11월 비교와 같음. 모형은 연말 휴무를 알지 못하므로 12월 하순 예측은 휴무 일정 확인 후 해석.",
             fontsize=8.5, color=TEXT_2)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.1)
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, facecolor="white")


if __name__ == "__main__":
    main()
