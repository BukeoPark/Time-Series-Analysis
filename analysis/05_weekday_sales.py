"""요일별 순매출 통계표와 그래프.

입력: data/processed/daily_sales.csv (01_preprocess.py 산출물)
출력:
  outputs/tables/weekday_sales_summary.csv
  outputs/figures/weekday_sales.png
실행: 프로젝트 루트에서 `python analysis/05_weekday_sales.py`

- 기본 통계는 영업일(is_closed = 0)만으로 계산한다. 휴무일의 0은 '요일 효과'가 아니라 휴무 효과이기 때문이다.
- 참고로 휴무일을 0으로 포함한 달력 기준 평균도 함께 둔다.
- 주 평균 대비 지수: 각 영업일 순매출 / 같은 주(일~토) 영업일 순매출 평균.
  기간 전체의 추세(연말 증가)를 걷어낸 요일 패턴을 보기 위한 값이다.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent.parent
DAILY_PATH = ROOT / "data" / "processed" / "daily_sales.csv"
TABLE_PATH = ROOT / "outputs" / "tables" / "weekday_sales_summary.csv"
FIG_PATH = ROOT / "outputs" / "figures" / "weekday_sales.png"

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

BLUE = "#2a78d6"
BAR = "#b9d3f2"
ORANGE = "#eb6834"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
DOW_NAMES = ["월", "화", "수", "목", "금", "토", "일"]


def weekday_summary(daily: pd.DataFrame) -> pd.DataFrame:
    open_days = daily[daily["is_closed"] == 0].copy()
    week = open_days["date"].dt.to_period("W-SAT")  # 일요일 시작 주
    open_days["week_index"] = open_days["net_sales"] / open_days.groupby(week)["net_sales"].transform("mean")

    g = open_days.groupby("dow")
    s = pd.DataFrame({
        "n_calendar_days": daily.groupby("dow").size(),
        "n_open_days": g.size(),
        "mean_net_sales": g["net_sales"].mean(),
        "median_net_sales": g["net_sales"].median(),
        "std_net_sales": g["net_sales"].std(),
        "min_net_sales": g["net_sales"].min(),
        "max_net_sales": g["net_sales"].max(),
        "mean_net_sales_incl_closed": daily.groupby("dow")["net_sales"].mean(),
        "total_net_sales": daily.groupby("dow")["net_sales"].sum(),
        "mean_n_invoices": g["n_invoices"].mean(),
        "mean_n_customers": g["n_customers"].mean(),
        "mean_cancel_amount": g["cancel_amount"].mean(),
        "mean_week_index": g["week_index"].mean(),
    }).reindex(range(7))
    s["n_open_days"] = s["n_open_days"].fillna(0).astype("int64")
    s["cv"] = s["std_net_sales"] / s["mean_net_sales"]
    s["share_of_total"] = s["total_net_sales"] / s["total_net_sales"].sum()
    s["sales_per_invoice"] = s["mean_net_sales"] / s["mean_n_invoices"]
    # 영업 요일(토요일 제외 6개)의 평균 순매출을 단순 평균한 값과 각 요일의 차이
    s["mean_of_weekday_means"] = s["mean_net_sales"].mean()  # NaN(토요일)은 제외됨
    s["diff_from_weekday_mean"] = s["mean_net_sales"] - s["mean_of_weekday_means"]
    s.insert(0, "weekday", DOW_NAMES)
    return s.reset_index(), open_days


def plot(summary: pd.DataFrame, open_days: pd.DataFrame) -> None:
    days = [d for d in range(7) if d in open_days["dow"].unique()]
    labels = [DOW_NAMES[d] for d in days]
    x = np.arange(len(days))
    s = summary.set_index("dow").loc[days]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=150,
                             gridspec_kw={"width_ratios": [1, 1.25, 1]})
    fig.patch.set_facecolor("white")
    k = FuncFormatter(lambda v, _: f"{v / 1000:,.0f}")

    # (1) 평균 순매출 + 중앙값
    ax = axes[0]
    ax.bar(x, s["mean_net_sales"], width=0.6, color=BLUE, label="평균", zorder=2)
    ax.scatter(x, s["median_net_sales"], marker="_", s=500, linewidths=2.5,
               color=ORANGE, zorder=3, label="중앙값")
    for xi, v in zip(x, s["mean_net_sales"]):
        ax.text(xi, v + 600, f"{v / 1000:,.1f}", ha="center", va="bottom", fontsize=9, color=TEXT)
    ax.set_title("① 요일별 평균 순매출 (영업일)", loc="left", fontsize=11.5, color=TEXT)
    ax.set_ylabel("순매출 (천 £)", color=TEXT_2)
    ax.yaxis.set_major_formatter(k)
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    # (2) 분포: 상자그림 + 개별 점
    ax = axes[1]
    data = [open_days.loc[open_days["dow"] == d, "net_sales"] for d in days]
    ax.boxplot(data, positions=x, widths=0.5, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor=BAR, edgecolor=TEXT_2, linewidth=1),
               medianprops=dict(color=ORANGE, linewidth=2),
               whiskerprops=dict(color=TEXT_2), capprops=dict(color=TEXT_2), zorder=2)
    rng = np.random.default_rng(0)  # 점 흩뿌림 위치 고정
    for xi, vals in zip(x, data):
        ax.scatter(xi + rng.uniform(-0.18, 0.18, len(vals)), vals, s=9,
                   color=BLUE, alpha=0.45, linewidths=0, zorder=3)
    ax.set_title("② 요일별 일 순매출 분포 (점 = 영업일 하루)", loc="left", fontsize=11.5, color=TEXT)
    ax.set_ylabel("순매출 (천 £)", color=TEXT_2)
    ax.yaxis.set_major_formatter(k)

    # (3) 추세 제거 요일 지수
    ax = axes[2]
    idx = s["mean_week_index"]
    ax.bar(x, idx - 1, bottom=1, width=0.6,
           color=[BLUE if v >= 1 else BAR for v in idx], zorder=2)
    ax.axhline(1, color=TEXT_2, linewidth=1, zorder=3)
    for xi, v in zip(x, idx):
        ax.text(xi, v + (0.01 if v >= 1 else -0.01), f"{v:.2f}", ha="center",
                va="bottom" if v >= 1 else "top", fontsize=9, color=TEXT)
    ax.set_title("③ 주 평균 대비 요일 지수 (1 = 그 주 평균)", loc="left", fontsize=11.5, color=TEXT)
    ax.set_ylim(min(idx.min(), 1) - 0.1, max(idx.max(), 1) + 0.1)

    for ax in axes:
        ax.set_xticks(x, labels)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(TEXT_2)
        ax.tick_params(colors=TEXT_2, length=0)
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(bottom=0)

    fig.suptitle("요일별 순매출 (2010-12-01 ~ 2011-12-08, 토요일은 전부 휴무)",
                 x=0.01, ha="left", fontsize=14, color=TEXT)
    fig.text(0.01, 0.01,
             "영업일만 사용(휴무·공휴일 제외). ② 상자 = 사분위 범위, 수염 = 1.5×IQR. "
             "③ 각 영업일 순매출 ÷ 같은 주(일~토) 영업일 평균의 요일별 평균 — 기간 추세를 걷어낸 요일 효과.",
             fontsize=8.5, color=TEXT_2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, facecolor="white")


def main() -> None:
    daily = pd.read_csv(DAILY_PATH, parse_dates=["date"])
    summary, open_days = weekday_summary(daily)
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLE_PATH, index=False)
    plot(summary, open_days)

    with pd.option_context("display.float_format", "{:,.3f}".format,
                           "display.width", 250, "display.max_columns", 30):
        print(summary.to_string(index=False))
    print(f"\n저장: {TABLE_PATH.relative_to(ROOT)}, {FIG_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
