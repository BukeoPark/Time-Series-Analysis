"""일별 순매출의 가법 분해 (매출 = 추세 + 요일 효과 + 잔차) 와 월별 그래프.

방법: 고전적 가법 분해 (statsmodels seasonal_decompose, model='additive', period=7)
- 입력은 373일 전체 달력의 net_sales (휴무일 0 포함). 달력 7일마다 요일이 한 번씩 돌아오므로 주기 7.
- 추세: 중심 7일 이동평균 (당일 기준 앞뒤 3일). 양 끝 3일은 계산되지 않는다.
  미래 값을 쓰므로 탐색용 분해이며 예측 입력으로 쓰지 않는다.
- 요일 효과: (매출 - 추세)의 요일별 평균을 7개 요일 합이 0이 되도록 맞춘 값. 전체 기간에서 요일마다 하나의 값.
- 잔차: 매출 - 추세 - 요일 효과.
- 모든 요소는 전체 기간으로 한 번 계산한 뒤 월별로 나눠 그린다.
  월끼리 비교할 수 있도록 요소마다 세로축 범위를 모든 월에서 같게 둔다.

입력: data/processed/daily_sales.csv
출력:
  outputs/tables/daily_decomposition.csv
  outputs/figures/decomposition/decomposition_YYYY-MM.png (월별 13개)
실행: 프로젝트 루트에서 `python analysis/07_decomposition.py`
"""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from statsmodels.tsa.seasonal import seasonal_decompose

ROOT = Path(__file__).resolve().parent.parent
DAILY_PATH = ROOT / "data" / "processed" / "daily_sales.csv"
TABLE_PATH = ROOT / "outputs" / "tables" / "daily_decomposition.csv"
FIG_DIR = ROOT / "outputs" / "figures" / "decomposition"

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


def decompose(daily: pd.DataFrame) -> pd.DataFrame:
    y = daily.set_index("date")["net_sales"].asfreq("D")
    res = seasonal_decompose(y, model="additive", period=7)
    out = pd.DataFrame({
        "net_sales": y,
        "trend": res.trend,
        "weekday_effect": res.seasonal,
        "residual": res.resid,
    })
    out["dow"] = out.index.dayofweek
    out["is_closed"] = daily.set_index("date")["is_closed"]
    out["year_month"] = out.index.strftime("%Y-%m")
    ok = out["trend"].notna()
    assert np.allclose(out.loc[ok, "net_sales"],
                       out.loc[ok, ["trend", "weekday_effect", "residual"]].sum(axis=1))
    return out


def sym_limit(s: pd.Series) -> float:
    return float(np.nanmax(np.abs(s))) * 1.1


def plot_month(m: pd.DataFrame, ym: str, lims: dict) -> Path:
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), dpi=150, sharex=True,
                             gridspec_kw={"hspace": 0.45})
    fig.patch.set_facecolor("white")
    k = FuncFormatter(lambda v, _: f"{v / 1000:,.0f}")
    x = m.index

    panels = [
        ("① 매출 (일별 순매출)", "net_sales"),
        ("② 요일 효과", "weekday_effect"),
        ("③ 추세 (중심 7일 이동평균)", "trend"),
        ("④ 잔차 (매출 - 추세 - 요일 효과)", "residual"),
    ]
    for ax, (title, col) in zip(axes, panels):
        for d in x[m["is_closed"] == 1]:
            ax.axvspan(d - pd.Timedelta(hours=12), d + pd.Timedelta(hours=12),
                       color=CLOSED, zorder=0, linewidth=0)
        v = m[col]
        if col == "net_sales":
            ax.bar(x, v, width=0.75, color=BAR, zorder=2)
            ax.set_ylim(0, lims[col])
        elif col == "trend":
            ax.plot(x, v, color=BLUE, linewidth=2, zorder=3)
            ax.set_ylim(0, lims[col])
        else:
            ax.bar(x, v, width=0.75, color=[BLUE if a >= 0 else ORANGE for a in v.fillna(0)],
                   zorder=2)
            ax.axhline(0, color=TEXT_2, linewidth=1, zorder=3)
            ax.set_ylim(-lims[col], lims[col])
        if col == "weekday_effect":
            for d, a in v.items():
                ax.text(d, a + (lims[col] * 0.04 if a >= 0 else -lims[col] * 0.04),
                        DOW_NAMES[d.dayofweek], ha="center",
                        va="bottom" if a >= 0 else "top", fontsize=7.5, color=TEXT_2)
        if v.isna().any():
            ax.text(0.99, 0.95, "전체 기간의 처음·마지막 3일은 중심 이동평균을 계산할 수 없어 비어 있음",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8, color=TEXT_2)

        ax.set_title(title, loc="left", fontsize=11.5, color=TEXT)
        ax.set_ylabel("천 £", color=TEXT_2, fontsize=9)
        ax.yaxis.set_major_formatter(k)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(TEXT_2)
        ax.tick_params(colors=TEXT_2, length=0, labelsize=8.5)

    start = x.min().to_period("M").start_time
    axes[-1].set_xlim(start - pd.Timedelta(hours=18),
                      start + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=18))
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 5, 10, 15, 20, 25, 30]))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axes[-1].tick_params(axis="x", length=3, labelbottom=True)
    axes[-1].set_xlabel("날짜", color=TEXT_2, fontsize=9)

    partial = "  (8일까지)" if ym == "2011-12" else ""
    fig.suptitle(f"{ym} 일별 순매출의 가법 분해{partial}  ·  "
                 f"월 합계 £{m['net_sales'].sum() / 1000:,.1f}천",
                 x=0.01, ha="left", y=0.985, fontsize=14, color=TEXT)
    fig.text(0.01, 0.01,
             "매출 = 추세 + 요일 효과 + 잔차 (고전적 가법 분해, 주기 7일, 휴무일은 0으로 포함). "
             "회색 음영 = 휴무일. 요소별 세로축은 모든 월에서 같음.",
             fontsize=8.5, color=TEXT_2)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.07)
    path = FIG_DIR / f"decomposition_{ym}.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    daily = pd.read_csv(DAILY_PATH, parse_dates=["date"])
    dec = decompose(daily)

    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    dec.reset_index().to_csv(TABLE_PATH, index=False, date_format="%Y-%m-%d")

    lims = {
        "net_sales": dec["net_sales"].max() * 1.05,
        "trend": dec["net_sales"].max() * 1.05,  # 매출과 같은 척도로 비교
        "weekday_effect": sym_limit(dec["weekday_effect"]) * 1.15,
        "residual": sym_limit(dec["residual"]),
    }
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ym, m in dec.groupby("year_month"):
        print(f"저장: {plot_month(m, ym, lims).relative_to(ROOT)}")

    effect = dec.groupby("dow")["weekday_effect"].first()
    effect.index = [DOW_NAMES[i] for i in effect.index]
    print("\n요일 효과 (£):")
    print(effect.map("{:,.2f}".format).to_string())
    summary = dec.groupby("year_month").agg(
        net_sales=("net_sales", "sum"), trend_mean=("trend", "mean"),
        resid_std=("residual", "std"), resid_abs_max=("residual", lambda s: s.abs().max()))
    with pd.option_context("display.float_format", "{:,.2f}".format):
        print("\n월별 요약:")
        print(summary.to_string())


if __name__ == "__main__":
    main()
