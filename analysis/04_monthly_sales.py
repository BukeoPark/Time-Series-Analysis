"""월별 순매출 요약표와 월별 분리 그래프(일별 순매출 + 7일 후행 이동평균).

입력: data/processed/daily_sales.csv (01_preprocess.py 산출물)
출력:
  outputs/tables/monthly_sales_summary.csv
  outputs/figures/monthly_daily_sales_ma7.png
실행: 프로젝트 루트에서 `python analysis/04_monthly_sales.py`

- 일평균 매출 = 월 합계 / 해당 월의 사용 기간 내 달력 일수 (휴무일 포함)
- 영업일 평균 매출 = 월 합계 / 해당 월의 영업일 수 (참고용)
- 이동평균은 전체 기간 일별 순매출로 먼저 계산한 뒤 월별로 나눠 그린다.
  후행 7일이므로 각 월 초 6일은 전월 말 값을 포함한다.
"""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent.parent
DAILY_PATH = ROOT / "data" / "processed" / "daily_sales.csv"
TABLE_PATH = ROOT / "outputs" / "tables" / "monthly_sales_summary.csv"
FIG_PATH = ROOT / "outputs" / "figures" / "monthly_daily_sales_ma7.png"

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

BLUE = "#2a78d6"
BAR = "#b9d3f2"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"


def monthly_summary(daily: pd.DataFrame) -> pd.DataFrame:
    g = daily.groupby("year_month")
    out = pd.DataFrame({
        "net_sales_total": g["net_sales"].sum(),
        "n_days": g.size(),
        "n_open_days": g["is_closed"].apply(lambda s: int((s == 0).sum())),
        "is_partial_month": g["is_partial_month"].max(),
    })
    out["avg_per_day"] = out["net_sales_total"] / out["n_days"]
    out["avg_per_open_day"] = out["net_sales_total"] / out["n_open_days"]
    return out.reset_index()


def plot_monthly(daily: pd.DataFrame, ma7: pd.Series) -> None:
    months = daily["year_month"].unique()
    ncols = 4
    nrows = -(-len(months) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.2 * nrows), dpi=150,
                             sharey=True)
    fig.patch.set_facecolor("white")
    ymax = daily["net_sales"].max() * 1.08

    for ax, ym in zip(axes.flat, months):
        m = daily[daily["year_month"] == ym]
        ax.bar(m["date"], m["net_sales"], width=0.8, color=BAR, zorder=2,
               label="일별 순매출")
        ax.plot(m["date"], ma7[m.index], color=BLUE, linewidth=2, zorder=3,
                label="7일 후행 이동평균")

        total = m["net_sales"].sum()
        title = f"{ym}  합계 £{total / 1000:,.1f}천 · 일평균 £{total / len(m) / 1000:,.1f}천"
        if m["is_partial_month"].iat[0] == 1:
            title += f"  ({m['date'].dt.day.max()}일까지)"
        ax.set_title(title, loc="left", fontsize=10, color=TEXT)

        ax.set_ylim(0, ymax)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:,.0f}"))
        ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 8, 15, 22, 29]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
        start = m["date"].min().to_period("M").start_time
        ax.set_xlim(start - pd.Timedelta(hours=18),
                    start + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=18))

        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(TEXT_2)
        ax.tick_params(colors=TEXT_2, length=0, labelsize=8.5)
        ax.tick_params(axis="x", length=3)

    for ax in axes.flat[len(months):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("순매출 (천 £)", color=TEXT_2, fontsize=9)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False,
               fontsize=10, bbox_to_anchor=(0.99, 0.995))
    fig.suptitle("월별 일 순매출과 7일 후행 이동평균 (가로축: 일, 세로축 공통)",
                 x=0.01, y=0.995, ha="left", fontsize=14, color=TEXT)
    fig.text(0.01, 0.005,
             "일평균 = 월 합계 / 해당 월 달력 일수(휴무일 포함). "
             "이동평균은 전체 기간으로 계산해 월 초에도 전월 말 값을 포함하며, "
             "2010-12-01~06은 7일이 채워지지 않아 없음.",
             fontsize=8.5, color=TEXT_2)
    fig.tight_layout(rect=(0, 0.015, 1, 0.975))
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, facecolor="white")


def main() -> None:
    daily = pd.read_csv(DAILY_PATH, parse_dates=["date"])
    ma7 = daily["net_sales"].rolling(window=7, min_periods=7).mean()

    summary = monthly_summary(daily)
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLE_PATH, index=False)
    plot_monthly(daily, ma7)

    with pd.option_context("display.float_format", "{:,.2f}".format, "display.width", 200):
        print(summary.to_string(index=False))
    print(f"\n저장: {TABLE_PATH.relative_to(ROOT)}, {FIG_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
