"""일별 순매출 추이 그래프.

입력: data/processed/daily_sales.csv (01_preprocess.py 산출물)
출력: outputs/figures/daily_net_sales.png
실행: 프로젝트 루트에서 `python analysis/02_plot_daily_sales.py`
"""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent.parent
DAILY_PATH = ROOT / "data" / "processed" / "daily_sales.csv"
FIG_PATH = ROOT / "outputs" / "figures" / "daily_net_sales.png"

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

BLUE = "#2a78d6"
ORANGE = "#eb6834"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
CLOSED = "#efeeea"


def main() -> None:
    daily = pd.read_csv(DAILY_PATH, parse_dates=["date"])
    open_days = daily[daily["is_closed"] == 0].set_index("date")

    # 직전 7일(달력 기준, 당일 포함) 영업일 평균. 휴무일 0은 평균에 넣지 않음
    ma7 = open_days["net_sales"].rolling("7D").mean()

    fig, ax = plt.subplots(figsize=(13, 5.5), dpi=150)
    fig.patch.set_facecolor("white")

    # 연속 휴무 구간(2일 이상, 토요일만인 날 제외) 음영
    closed = daily[(daily["is_closed"] == 1) & (daily["dow"] != 5)]["date"]
    block = (closed.diff().dt.days > 2).cumsum()
    block_labels = iter(["연말연시\n휴무", "부활절\n연휴"])
    for _, g in closed.groupby(block):
        if len(g) >= 3:
            ax.axvspan(g.min() - pd.Timedelta(hours=12), g.max() + pd.Timedelta(hours=12),
                       color=CLOSED, zorder=0)
            ax.text(g.min() + (g.max() - g.min()) / 2, 0.97, next(block_labels),
                    transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=9, color=TEXT_2)

    # 3일 이상 거래가 없는 구간은 선을 끊어 보이게 함 (토요일 하루 공백은 이어서 그림)
    gap = open_days.index.to_series().diff().dt.days > 3
    breaks = open_days.index[gap] - pd.Timedelta(days=1)
    line = pd.concat([open_days["net_sales"], pd.Series(float("nan"), index=breaks)]).sort_index()
    ma_line = pd.concat([ma7, pd.Series(float("nan"), index=breaks)]).sort_index()

    ax.plot(line.index, line, color=BLUE, alpha=0.35,
            linewidth=1, label="일별 순매출 (영업일)")
    ax.plot(ma_line.index, ma_line, color=BLUE, linewidth=2, label="7일 이동평균")

    out = open_days[open_days["is_outlier"] == 1]
    ax.scatter(out.index, out["net_sales"], s=40, color=ORANGE, edgecolor="white",
               linewidth=1.5, zorder=5, label="이상치 표시 (요일별 |z| > 3.5)")
    for d, v in out["net_sales"].items():
        ax.annotate(f"{d:%m-%d}\n£{v / 1000:,.1f}천", (d, v), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9, color=TEXT)

    ax.set_title("일별 순매출 추이 (2010-12-01 ~ 2011-12-08)", loc="left",
                 fontsize=14, color=TEXT, pad=12)
    ax.set_ylabel("순매출 (천 파운드, £)", color=TEXT_2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:,.0f}"))
    ax.set_ylim(bottom=0)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_xlim(daily["date"].min() - pd.Timedelta(days=3),
                daily["date"].max() + pd.Timedelta(days=3))

    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(TEXT_2)
    ax.tick_params(colors=TEXT_2, length=0)
    ax.tick_params(axis="x", length=4)

    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=3, frameon=False,
              fontsize=10, labelcolor=TEXT)
    fig.text(0.01, 0.01,
             "휴무일(토요일·공휴일, 순매출 0)은 선에서 제외. "
             "7일 이동평균 = 직전 7일간 영업일 순매출의 평균.",
             fontsize=8.5, color=TEXT_2)

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, facecolor="white")
    print(f"저장: {FIG_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
