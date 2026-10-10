"""일별 순매출과 7일 후행 이동평균 그래프.

입력: data/processed/daily_sales.csv (01_preprocess.py 산출물)
출력: outputs/figures/daily_net_sales_ma7.png
실행: 프로젝트 루트에서 `python analysis/03_plot_moving_average.py`

7일 후행 이동평균: 당일 포함 직전 7개 달력일의 net_sales 평균.
휴무일은 net_sales = 0 그대로 포함한다. 7일 창에는 항상 토요일이 1번 들어가므로
요일 효과가 상쇄된 '1주 평균 일매출'이 된다. 7일이 채워지지 않은 첫 6일은 계산하지 않는다.
"""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent.parent
DAILY_PATH = ROOT / "data" / "processed" / "daily_sales.csv"
FIG_PATH = ROOT / "outputs" / "figures" / "daily_net_sales_ma7.png"

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

BLUE = "#2a78d6"
BAR = "#b9d3f2"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"


def main() -> None:
    daily = pd.read_csv(DAILY_PATH, parse_dates=["date"]).set_index("date").asfreq("D")
    ma7 = daily["net_sales"].rolling(window=7, min_periods=7).mean()

    fig, ax = plt.subplots(figsize=(13, 5.5), dpi=150)
    fig.patch.set_facecolor("white")

    ax.bar(daily.index, daily["net_sales"], width=0.8, color=BAR,
           label="일별 순매출", zorder=2)
    ax.plot(ma7.index, ma7, color=BLUE, linewidth=2,
            label="7일 후행 이동평균", zorder=3)

    last = ma7.dropna().index[-1]
    ax.annotate(f"£{ma7[last] / 1000:,.1f}천", (last, ma7[last]), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=9, color=TEXT)

    ax.set_title("일별 순매출과 7일 후행 이동평균 (2010-12-01 ~ 2011-12-08)",
                 loc="left", fontsize=14, color=TEXT, pad=12)
    ax.set_ylabel("순매출 (천 파운드, £)", color=TEXT_2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:,.0f}"))
    ax.set_ylim(bottom=0)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_xlim(daily.index.min() - pd.Timedelta(days=2),
                daily.index.max() + pd.Timedelta(days=12))

    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(TEXT_2)
    ax.tick_params(colors=TEXT_2, length=0)
    ax.tick_params(axis="x", length=4)

    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2, frameon=False,
              fontsize=10, labelcolor=TEXT)
    fig.text(0.01, 0.01,
             "7일 후행 이동평균 = 당일 포함 직전 7일 순매출의 평균 (휴무일은 0으로 포함, "
             "첫 6일은 7일이 채워지지 않아 계산하지 않음).",
             fontsize=8.5, color=TEXT_2)

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, facecolor="white")
    print(f"저장: {FIG_PATH.relative_to(ROOT)}")
    print(f"이동평균 계산일: {ma7.notna().sum()}일, 최소 {ma7.min():,.2f} ({ma7.idxmin():%Y-%m-%d}), "
          f"최대 {ma7.max():,.2f} ({ma7.idxmax():%Y-%m-%d}), 마지막 {ma7[last]:,.2f}")


if __name__ == "__main__":
    main()
