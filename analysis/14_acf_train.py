"""ARIMA 비교 준비: 학습 구간 일별 순매출과 시차 1~21일 자기상관(ACF).

기준: 전처리_기준.md 6-1 (학습 2011-01-01 ~ 2011-10-31, 304일)
- 대상: daily_sales.csv의 net_sales (휴무일 0 포함, 변환·차분 없음)
- ACF: statsmodels acf (표본 자기상관, 분모는 전체 표본 분산), 95% 신뢰구간은 Bartlett 공식.
- 참고: 시차 k의 두 열(y_t, y_{t-k})을 짝지어 계산한 피어슨 상관계수도 함께 저장한다
  (ACF는 전체 평균·분산을 쓰므로 값이 약간 다르다).

입력: data/processed/daily_sales.csv
출력:
  outputs/tables/acf_train.csv
  outputs/figures/acf_train.png
실행: 프로젝트 루트에서 `python analysis/14_acf_train.py`
"""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from statsmodels.tsa.stattools import acf

ROOT = Path(__file__).resolve().parent.parent
DAILY_PATH = ROOT / "data" / "processed" / "daily_sales.csv"
TABLE_PATH = ROOT / "outputs" / "tables" / "acf_train.csv"
FIG_PATH = ROOT / "outputs" / "figures" / "acf_train.png"

TRAIN = ("2011-01-01", "2011-10-31")
MAX_LAG = 21
KEY_LAGS = [7, 14, 21]

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

BLUE = "#2a78d6"
BAR = "#b9d3f2"
ORANGE = "#eb6834"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
BAND = "#e9e8e3"


def main() -> None:
    daily = pd.read_csv(DAILY_PATH, parse_dates=["date"]).set_index("date").asfreq("D")
    y = daily.loc[TRAIN[0]:TRAIN[1], "net_sales"]
    assert len(y) == 304

    r, ci = acf(y, nlags=MAX_LAG, alpha=0.05, fft=False)
    lags = np.arange(MAX_LAG + 1)
    table = pd.DataFrame({
        "lag": lags[1:],
        "acf": r[1:],
        "ci_lower": ci[1:, 0] - r[1:],  # 0을 중심으로 한 95% 구간 (Bartlett)
        "ci_upper": ci[1:, 1] - r[1:],
        "pearson_pair": [y.corr(y.shift(k)) for k in lags[1:]],
        "n_pairs": [len(y) - k for k in lags[1:]],
    })
    table["significant"] = (table["acf"] > table["ci_upper"]) | (table["acf"] < table["ci_lower"])
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_PATH, index=False)

    plot(y, daily, table)

    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
        print(table.to_string(index=False))
    print(f"\n저장: {TABLE_PATH.relative_to(ROOT)}, {FIG_PATH.relative_to(ROOT)}")


def plot(y: pd.Series, daily: pd.DataFrame, t: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9.5), dpi=150,
                                   gridspec_kw={"height_ratios": [1, 1.05], "hspace": 0.4})
    fig.patch.set_facecolor("white")

    # ① 학습 구간 일별 순매출
    closed = daily.loc[y.index, "is_closed"] == 1
    ax1.bar(y.index[~closed], y[~closed], width=0.8, color=BAR, zorder=2, label="영업일")
    ax1.scatter(y.index[closed], y[closed], s=6, color=ORANGE, zorder=3, label="휴무일 (0)")
    ma = y.rolling(7).mean()
    ax1.plot(ma.index, ma, color=BLUE, linewidth=2, zorder=4, label="7일 후행 이동평균")
    ax1.set_title(f"① 학습 구간 일별 순매출 ({TRAIN[0]} ~ {TRAIN[1]}, {len(y)}일)",
                  loc="left", fontsize=12, color=TEXT)
    ax1.set_ylabel("순매출 (천 £)", color=TEXT_2)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:,.0f}"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.set_xlim(y.index[0] - pd.Timedelta(days=2), y.index[-1] + pd.Timedelta(days=2))
    ax1.set_ylim(0, y.max() * 1.1)
    ax1.legend(loc="upper left", frameon=False, fontsize=9, ncol=3)

    # ② ACF (시차 1~21)
    ax2.fill_between(t["lag"], t["ci_lower"], t["ci_upper"], color=BAND, step="mid", zorder=0,
                     label="95% 신뢰구간 (Bartlett)")
    is_key = t["lag"].isin(KEY_LAGS)
    ax2.vlines(t["lag"], 0, t["acf"], color=np.where(is_key, ORANGE, BLUE), linewidth=2.5, zorder=2)
    ax2.scatter(t["lag"], t["acf"], color=np.where(is_key, ORANGE, BLUE), s=36, zorder=3,
                edgecolor="white", linewidth=1.2)
    ax2.axhline(0, color=TEXT_2, linewidth=1)
    for _, row in t[is_key].iterrows():
        ax2.annotate(f"{int(row['lag'])}일 전\nr = {row['acf']:.3f}", (row["lag"], row["acf"]),
                     xytext=(0, 12), textcoords="offset points", ha="center", va="bottom",
                     fontsize=9.5, color=TEXT)
    ax2.set_title("② 자기상관함수 (ACF), 시차 1~21일  ·  주황 = 7·14·21일",
                  loc="left", fontsize=12, color=TEXT)
    ax2.set_xlabel("시차 (일)", color=TEXT_2)
    ax2.set_ylabel("자기상관계수", color=TEXT_2)
    ax2.set_xticks(range(1, MAX_LAG + 1))
    ax2.set_xlim(0.3, MAX_LAG + 0.7)
    ax2.set_ylim(min(-0.4, t["acf"].min() - 0.1), max(0.8, t["acf"].max() + 0.2))
    ax2.legend(loc="upper right", frameon=False, fontsize=9)

    for ax in (ax1, ax2):
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(TEXT_2)
        ax.tick_params(colors=TEXT_2, length=0)
        ax.tick_params(axis="x", length=4)

    fig.suptitle("ARIMA 비교 준비 · 학습 구간 일별 순매출과 자기상관",
                 x=0.01, ha="left", fontsize=14, color=TEXT)
    fig.text(0.01, 0.01, "대상: 일별 순매출(휴무일 0 포함), 차분·변환 없음. "
             "신뢰구간 밖의 막대는 5% 수준에서 0과 유의하게 다름.",
             fontsize=8.5, color=TEXT_2)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.08)
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, facecolor="white")


if __name__ == "__main__":
    main()
