"""주문당 평균 금액(AOV)의 가법 분해 (AOV = 추세 + 요일 효과 + 잔차) 와 월별 그래프.

정의: AOV = gross_sales / n_invoices  (정상 판매 송장 기준. 취소 송장은 '주문'이 아니므로 제외)
- 휴무일은 주문이 없어 AOV가 정의되지 않는다(0이 아님).
- 토요일은 전 기간 휴무이므로 달력에서 빼고 일~금 6일을 1주로 본다 → 주기 6.
- 토요일 외 휴무일(16일)은 분해 계산을 위해 앞뒤 영업일 값의 선형 보간으로 채우고
  is_imputed = 1로 표시한다. 보간한 날의 AOV와 잔차는 그래프·요약에 쓰지 않는다.
- 추세: 중심 이동평균 (짝수 주기이므로 2×6 이동평균). 앞뒤 값을 쓰므로 탐색용이다.
- 요일 효과: (AOV − 추세)의 요일별 평균을 6개 요일 합이 0이 되도록 맞춘 값.

입력: data/processed/daily_sales.csv
출력:
  outputs/tables/daily_aov_decomposition.csv
  outputs/figures/aov_decomposition/aov_decomposition_YYYY-MM.png (월별 13개)
실행: 프로젝트 루트에서 `python analysis/09_aov_decomposition.py`
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
TABLE_PATH = ROOT / "outputs" / "tables" / "daily_aov_decomposition.csv"
FIG_DIR = ROOT / "outputs" / "figures" / "aov_decomposition"

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
PERIOD = 6


def decompose(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily[daily["dow"] != 5].set_index("date").copy()  # 토요일 제외 달력
    d["aov"] = np.where(d["is_closed"] == 0, d["gross_sales"] / d["n_invoices"].replace(0, np.nan), np.nan)
    d["is_imputed"] = d["aov"].isna().astype("int64")
    filled = d["aov"].interpolate(method="time")

    res = seasonal_decompose(filled.to_numpy(), model="additive", period=PERIOD)
    out = pd.DataFrame({
        "aov": d["aov"],
        "aov_filled": filled,
        "trend": res.trend,
        "weekday_effect": res.seasonal,
        "residual": res.resid,
        "n_invoices": d["n_invoices"],
        "gross_sales": d["gross_sales"],
        "is_imputed": d["is_imputed"],
    }, index=d.index)
    out.loc[out["is_imputed"] == 1, "residual"] = np.nan  # 보간값의 잔차는 의미 없음
    out["dow"] = out.index.dayofweek
    out["year_month"] = out.index.strftime("%Y-%m")

    ok = out["trend"].notna() & (out["is_imputed"] == 0)
    assert np.allclose(out.loc[ok, "aov"], out.loc[ok, ["trend", "weekday_effect", "residual"]].sum(axis=1))
    return out


def sym_limit(s: pd.Series) -> float:
    return float(np.nanmax(np.abs(s))) * 1.1


def second_max(s: pd.Series) -> float:
    """세로축 상한: 극단값 하나가 모든 월의 축을 결정하지 않도록 두 번째로 큰 값 기준."""
    return float(np.sort(s.dropna().to_numpy())[-2])


def label_clipped(ax, x: pd.Index, v: pd.Series, lim: float, symmetric: bool) -> None:
    for d, a in v.items():
        if a > lim or (symmetric and a < -lim):
            y = lim if a > 0 else -lim
            ax.text(d, y * 0.97, f"▲ £{a:,.0f}" if a > 0 else f"▼ £{a:,.0f}", ha="left",
                    va="top" if a > 0 else "bottom", fontsize=8, color=TEXT)


def plot_month(m: pd.DataFrame, ym: str, lims: dict) -> Path:
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), dpi=150, sharex=True,
                             gridspec_kw={"hspace": 0.45})
    fig.patch.set_facecolor("white")
    fmt = FuncFormatter(lambda v, _: f"{v:,.0f}")
    x = m.index

    start = x.min().to_period("M").start_time
    month_days = pd.date_range(start, start + pd.offsets.MonthEnd(0), freq="D")
    closed_days = month_days[month_days.dayofweek == 5].append(x[m["is_imputed"] == 1])

    panels = [
        ("① 주문당 평균 금액 (정상 판매 금액 ÷ 판매 송장 수)", "aov"),
        ("② 요일 효과", "weekday_effect"),
        ("③ 추세 (중심 6영업일 이동평균)", "trend"),
        ("④ 잔차 (주문당 평균 금액 - 추세 - 요일 효과)", "residual"),
    ]
    for ax, (title, col) in zip(axes, panels):
        for d in closed_days:
            ax.axvspan(d - pd.Timedelta(hours=12), d + pd.Timedelta(hours=12),
                       color=CLOSED, zorder=0, linewidth=0)
        v = m[col]
        if col == "aov":
            ax.bar(x, v.clip(upper=lims[col]), width=0.75, color=BAR, zorder=2)
            ax.set_ylim(0, lims[col])
            label_clipped(ax, x, v, lims[col], symmetric=False)
        elif col == "trend":
            ax.plot(x, v, color=BLUE, linewidth=2, zorder=3)
            ax.set_ylim(0, lims[col])  # 추세는 상한(1,600 안팎)을 넘지 않음
        else:
            ax.bar(x, v.clip(-lims[col], lims[col]), width=0.75,
                   color=[BLUE if a >= 0 else ORANGE for a in v.fillna(0)], zorder=2)
            ax.axhline(0, color=TEXT_2, linewidth=1, zorder=3)
            ax.set_ylim(-lims[col], lims[col])
            label_clipped(ax, x, v, lims[col], symmetric=True)
        if col == "weekday_effect":
            for d, a in v.items():
                ax.text(d, a + (lims[col] * 0.04 if a >= 0 else -lims[col] * 0.04),
                        DOW_NAMES[d.dayofweek], ha="center",
                        va="bottom" if a >= 0 else "top", fontsize=7.5, color=TEXT_2)
        if col == "trend" and v.isna().any():
            ax.text(0.99, 0.95, "전체 기간의 처음·마지막 3영업일은 중심 이동평균을 계산할 수 없어 비어 있음",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8, color=TEXT_2)

        ax.set_title(title, loc="left", fontsize=11.5, color=TEXT)
        ax.set_ylabel("£", color=TEXT_2, fontsize=9)
        ax.yaxis.set_major_formatter(fmt)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(TEXT_2)
        ax.tick_params(colors=TEXT_2, length=0, labelsize=8.5)

    axes[-1].set_xlim(start - pd.Timedelta(hours=18),
                      start + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=18))
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 5, 10, 15, 20, 25, 30]))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axes[-1].tick_params(axis="x", length=3, labelbottom=True)
    axes[-1].set_xlabel("날짜", color=TEXT_2, fontsize=9)

    real = m[m["is_imputed"] == 0]
    month_aov = real["gross_sales"].sum() / real["n_invoices"].sum()
    partial = "  (8일까지)" if ym == "2011-12" else ""
    fig.suptitle(f"{ym} 주문당 평균 금액의 가법 분해{partial}  ·  "
                 f"월 주문당 평균 £{month_aov:,.0f} (판매 송장 {int(real['n_invoices'].sum()):,}건)",
                 x=0.01, ha="left", y=0.985, fontsize=14, color=TEXT)
    fig.text(0.01, 0.01,
             "주문당 평균 금액 = 추세 + 요일 효과 + 잔차 (고전적 가법 분해, 토요일 제외 6영업일 주기). "
             "회색 음영 = 휴무일(값 없음, 분해 계산에는 보간값 사용).\n"
             "요소별 세로축은 모든 월에서 같음. 축 범위를 넘는 값(2011-01-18)은 막대를 잘라 그리고 실제 값을 ▲/▼로 표시.",
             fontsize=8.5, color=TEXT_2)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.08)
    path = FIG_DIR / f"aov_decomposition_{ym}.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    daily = pd.read_csv(DAILY_PATH, parse_dates=["date"])
    dec = decompose(daily)

    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    dec.reset_index().to_csv(TABLE_PATH, index=False, date_format="%Y-%m-%d")

    lims = {
        "aov": second_max(dec["aov"]) * 1.05,
        "trend": second_max(dec["aov"]) * 1.05,  # AOV와 같은 척도로 비교
        "weekday_effect": sym_limit(dec["weekday_effect"]) * 1.15,
        "residual": second_max(dec["residual"].abs()) * 1.1,
    }
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ym, m in dec.groupby("year_month"):
        print(f"저장: {plot_month(m, ym, lims).relative_to(ROOT)}")

    print(f"\n보간한 휴무일(토요일 외): {int(dec['is_imputed'].sum())}일")
    effect = dec.groupby("dow")["weekday_effect"].first()
    effect.index = [DOW_NAMES[i] for i in effect.index]
    print("요일 효과 (£):")
    print(effect.map("{:+,.2f}".format).to_string())

    real = dec[dec["is_imputed"] == 0]
    g = real.groupby("year_month")
    summary = pd.DataFrame({
        "order_value": g["gross_sales"].sum() / g["n_invoices"].sum(),
        "daily_aov_median": g["aov"].median(),
        "trend_mean": g["trend"].mean(),
        "resid_std": g["residual"].std(),
    })
    with pd.option_context("display.float_format", "{:,.2f}".format):
        print("\n월별 요약:")
        print(summary.to_string())
    med = real["residual"].median()
    mad = (real["residual"] - med).abs().median()
    z = 0.6745 * (real["residual"] - med) / mad
    top = real.assign(z=z).loc[z.abs() > 3.0, ["aov", "trend", "residual", "n_invoices", "z"]]
    with pd.option_context("display.float_format", "{:,.2f}".format):
        print(f"\n잔차 robust |z| > 3: {len(top)}일")
        print(top.to_string())


if __name__ == "__main__":
    main()
