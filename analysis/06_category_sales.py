"""상품명(Description) 키워드 기반 카테고리별 매출 분석.

원본에 카테고리 열이 없으므로 규칙 기반으로 상품군을 만든다 (근사 분류).
- 전처리 기준(전처리_기준.md 1~2단계)은 01_preprocess.py의 함수를 그대로 불러 써서 동일하게 적용한다.
- 분류: Description을 대문자·영숫자 토큰으로 나눈 뒤, 아래 RULES를 위에서부터 차례로 확인해
  처음 맞는 키워드의 카테고리를 준다. 어느 규칙에도 맞지 않으면 '기타'.
- 월 비교는 2011-12가 8일까지뿐이므로 '월 일평균(월 합계 / 달력 일수)'으로 한다.

입력: data/raw/Online Retail.xlsx
출력:
  data/processed/product_category.csv          상품명별 카테고리와 일치 키워드
  outputs/tables/category_sales_summary.csv    카테고리별 요약
  outputs/tables/category_monthly_sales.csv    카테고리 × 월 순매출·일평균
  outputs/tables/category_daily_sales.csv      카테고리 × 일 순매출·판매·취소 (보고서 기간 필터용)
  outputs/figures/category_sales.png
실행: 프로젝트 루트에서 `python analysis/06_category_sales.py`
"""
import importlib.util
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter

ROOT = Path(__file__).resolve().parent.parent
MAP_PATH = ROOT / "data" / "processed" / "product_category.csv"
SUMMARY_PATH = ROOT / "outputs" / "tables" / "category_sales_summary.csv"
MONTHLY_PATH = ROOT / "outputs" / "tables" / "category_monthly_sales.csv"
DAILY_CAT_PATH = ROOT / "outputs" / "tables" / "category_daily_sales.csv"
FIG_PATH = ROOT / "outputs" / "figures" / "category_sales.png"

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

BLUE = "#2a78d6"
BAR = "#b9d3f2"
GRAY = "#b5b3ac"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"

# 우선순위 순서. 예) "CHRISTMAS LIGHTS"는 크리스마스·시즌, "LUNCH BAG"은 가방·파우치
RULES = [
    ("크리스마스·시즌", ["CHRISTMAS", "XMAS", "ADVENT", "EASTER", "HALLOWEEN", "SANTA",
                     "REINDEER", "SNOWFLAKE", "BAUBLE", "NOEL", "NATIVITY", "SNOWMAN",
                     "SNOWMEN", "GINGERBREAD"]),
    ("조명·캔들", ["LIGHT", "LIGHTS", "T-LIGHT", "T-LIGHTS", "CANDLE", "CANDLES", "LANTERN",
               "LAMP", "CHANDELIER", "NIGHTLIGHT", "TEALIGHT", "INCENSE", "OIL BURNER",
               "DIFFUSER", "CANDLEHOLDER", "TORCH", "CHOCOLATECANDLE", "DISCO BALL"]),
    ("가방·파우치", ["BAG", "BAGS", "PURSE", "WALLET", "SHOPPER", "TOTE", "HANDBAG",
                "BACKPACK", "RUCKSACK"]),
    ("주방·식기", ["CAKE", "CAKESTAND", "MUG", "MUGS", "CUP", "CUPS", "TEACUP", "SAUCER",
               "PLATE", "PLATES", "BOWL", "BOWLS", "CUTLERY", "TEA", "COFFEE", "JAM",
               "BAKING", "KITCHEN", "PANTRY", "APRON", "NAPKIN", "NAPKINS", "DOILY",
               "DOILIES", "SPOON", "SPOONS", "FORK", "KNIFE", "TRAY", "TRAYS", "TEAPOT",
               "JUG", "BEAKER", "GLASS", "GLASSES", "TUMBLER", "CHOPSTICKS", "EGG", "BREAD",
               "BISCUIT", "SUGAR", "MILK", "DISH", "LUNCH", "PICNIC", "STRAWS",
               "BOTTLE OPENER", "OVEN", "SCALES", "COASTER", "COASTERS", "PLACEMATS",
               "PLACEMAT", "RECIPE", "DINER", "MOULD", "ICE", "POPCORN", "BUNTING",
               "COOKIE CUTTER", "COLANDER", "TABLECLOTH", "FOOD COVER", "SALT", "PEPPER",
               "CONTAINER"]),
    ("보관·수납", ["BOX", "BOXES", "TIN", "TINS", "JAR", "JARS", "STORAGE", "BASKET",
               "BASKETS", "CASE", "CASES", "DRAWER", "DRAWERS", "HAMPER", "CRATE", "CRATES",
               "TRINKET", "CABINET", "CHEST", "POT", "POTS", "HOLDER", "RACK", "CADDY",
               "TIDY", "PEGS"]),
    ("문구·카드·포장", ["CARD", "CARDS", "PAPER", "WRAP", "NOTEBOOK", "NOTEBOOKS", "PEN",
                   "PENS", "PENCIL", "PENCILS", "STICKER", "STICKERS", "RIBBON", "RIBBONS",
                   "TAPE", "ENVELOPE", "ENVELOPES", "CRAYONS", "CHALK", "ERASER", "RUBBER",
                   "STATIONERY", "JOURNAL", "DIARY", "BOOK", "BOOKS", "NOTE", "PAD",
                   "NOTEPAD", "TAG", "TAGS", "CONFETTI", "INVITES", "PARTY", "BALLOON",
                   "BALLOONS", "STAMP", "STAMPS", "CALENDAR", "POSTCARD", "GIFT", "RULER",
                   "MEMOBOARD", "MEMO", "MAGNETS", "MAGNET", "SCISSOR", "STENCIL"]),
    ("완구·어린이", ["TOY", "TOYS", "DOLLY", "DOLL", "CHILDRENS", "CHILDS", "KIDS", "BABY",
                "GAME", "GAMES", "PUZZLE", "JIGSAW", "SKITTLES", "DOMINOES", "PLAYHOUSE",
                "TEDDY", "SPACEBOY", "FELTCRAFT", "KIT", "CRAYON", "COLOURING", "SOLDIER",
                "MAGIC", "BINGO", "SNAKES", "LADDERS", "SKIPPING", "BUBBLES", "PIGGY",
                "HELICOPTER", "ROCKET", "MARBLES", "SPINNING", "SOFT", "BUILDING BLOCK",
                "PAINT SET", "MODELLING CLAY", "KNITTING NANCY", "GLIDERS", "LUDO",
                "NAUGHTS", "SPORTING"]),
    ("인테리어·장식", ["SIGN", "FRAME", "FRAMES", "CLOCK", "DOORMAT", "CUSHION", "MIRROR",
                  "DECORATION", "DECORATIONS", "ORNAMENT", "WALL", "HOOK", "HOOKS", "KNOB",
                  "KNOBS", "DRAWERKNOB", "HANGER", "HANGERS", "PHOTO", "VASE", "WREATH",
                  "GARLAND", "CHAIN", "PLAQUE", "DOORSTOP", "BLACKBOARD", "BLACK BOARD",
                  "CHALKBOARD", "ART", "PRINT", "POSTER", "FLAG", "BIRDHOUSE", "BIRD",
                  "HANGING", "DOOR", "SHELF", "CURTAIN", "THROW", "QUILT", "STOOL", "TABLE",
                  "CHAIR", "LETTERS", "ALPHABET", "NUMBER", "MOBILE", "FEATHER", "FLOWER",
                  "FLOWERS", "STAR", "CROWN", "BELL", "BELLS", "HEART", "HEARTS",
                  "BUTTERFLY", "WINDMILL", "PARASOL", "UMBRELLA", "CANVAS", "TILE", "GNOME",
                  "GARDEN", "PLANT", "PLANTER", "WATERING", "CHIME", "CHIMES", "SPINNER",
                  "MAT", "HAMMOCK", "LOVE SEAT", "ROSETTE", "FLYING DUCKS", "METAL PEARS"]),
    ("패션·생활잡화", ["NECKLACE", "BRACELET", "EARRINGS", "RING", "HAIR", "SCARF", "GLOVES",
                  "SLIPPERS", "HAT", "HOT WATER BOTTLE", "WARMER", "SOAP", "TISSUE",
                  "TISSUES", "TOWEL", "COMPACT", "KEYRING", "KEY", "CHARM", "BROOCH",
                  "HAIRCLIP", "SLIDE", "BANGLE", "DIAMANTE", "JEWEL", "JEWELLERY",
                  "LIPSTICK", "LIP GLOSS", "SPONGE", "PASSPORT", "LUGGAGE", "POUCH",
                  "SEWING", "NEEDLE", "BADGE", "PHONE", "HAND", "SHOWER", "BATH", "WASH",
                  "FAN", "PONCHO", "SOMBRERO", "FLANNEL", "BICYCLE"]),
]
OTHER = "기타"
CATEGORIES = [c for c, _ in RULES] + [OTHER]


def load_preprocess_module():
    spec = importlib.util.spec_from_file_location("preprocess", ROOT / "analysis" / "01_preprocess.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def classify(desc: str) -> tuple[str, str]:
    text = " " + re.sub(r"[^A-Z0-9\- ]", " ", str(desc).upper()) + " "
    text = re.sub(r"\s+", " ", text)
    tokens = set(text.split())
    for category, keywords in RULES:
        for kw in keywords:
            if (" " in kw and f" {kw} " in text) or kw in tokens:
                return category, kw
    return OTHER, ""


def summarize(clean: pd.DataFrame, daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sale = clean[~clean["is_cancel"]]
    cancel = clean[clean["is_cancel"]]
    g = clean.groupby("category")
    summary = pd.DataFrame({
        "net_sales": g["amount"].sum(),
        "gross_sales": sale.groupby("category")["amount"].sum(),
        "cancel_amount": cancel.groupby("category")["amount"].sum(),
        "n_products": sale.groupby("category")["StockCode"].nunique(),
        "n_lines": sale.groupby("category").size(),
        "quantity": sale.groupby("category")["Quantity"].sum(),
    }).reindex(CATEGORIES).fillna(0)
    summary["share"] = summary["net_sales"] / summary["net_sales"].sum()
    summary["cancel_rate"] = -summary["cancel_amount"] / summary["gross_sales"]
    summary["avg_unit_price"] = summary["gross_sales"] / summary["quantity"]  # 수량 가중 평균 단가

    # 월별 순매출과 월 일평균 (월 합계 / 해당 월 사용 기간 내 달력 일수)
    clean = clean.assign(year_month=clean["date"].dt.strftime("%Y-%m"))
    n_days = daily.groupby("year_month").size()
    monthly = (clean.pivot_table(index="year_month", columns="category", values="amount",
                                 aggfunc="sum", fill_value=0)
               .reindex(columns=CATEGORIES, fill_value=0))
    monthly_avg = monthly.div(n_days, axis=0)

    # 성장: 2011-09~12 월 일평균 / 2011-01~08 월 일평균
    early = monthly.loc["2011-01":"2011-08"].sum() / n_days.loc["2011-01":"2011-08"].sum()
    late = monthly.loc["2011-09":"2011-12"].sum() / n_days.loc["2011-09":"2011-12"].sum()
    summary["daily_avg_2011_01_08"] = early
    summary["daily_avg_2011_09_12"] = late
    summary["growth_late_vs_early"] = late / early - 1

    # 요일 편차: 카테고리별 요일 평균 순매출 / 해당 카테고리 영업 요일 평균의 평균
    open_dates = daily.loc[daily["is_closed"] == 0, "date"]
    cat_daily = (clean.pivot_table(index="date", columns="category", values="amount",
                                   aggfunc="sum", fill_value=0)
                 .reindex(index=open_dates, columns=CATEGORIES, fill_value=0))
    by_dow = cat_daily.groupby(cat_daily.index.dayofweek).mean()
    summary["thu_index"] = by_dow.loc[3] / by_dow.mean()
    summary["sun_index"] = by_dow.loc[6] / by_dow.mean()

    long = (monthly.stack().rename("net_sales").to_frame()
            .join(monthly_avg.stack().rename("daily_avg")).reset_index())
    return summary.reset_index(), long


def plot(summary: pd.DataFrame, monthly: pd.DataFrame) -> None:
    s = summary.set_index("category")
    order = [c for c in s.sort_values("net_sales", ascending=False).index if c != OTHER] + [OTHER]
    k = FuncFormatter(lambda v, _: f"{v / 1000:,.0f}")

    fig = plt.figure(figsize=(17, 11), dpi=150)
    fig.patch.set_facecolor("white")

    def style(ax):
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(TEXT_2)
        ax.tick_params(colors=TEXT_2, length=0, labelsize=8.5)

    # ① 카테고리별 순매출과 비중
    ax = fig.add_axes([0.085, 0.665, 0.37, 0.25])
    y = np.arange(len(order))[::-1]
    vals = s.loc[order, "net_sales"]
    ax.barh(y, vals, color=[GRAY if c == OTHER else BLUE for c in order], height=0.62, zorder=2)
    for yi, c in zip(y, order):
        ax.text(vals[c] + vals.max() * 0.01, yi,
                f"£{vals[c] / 1000:,.0f}천 ({s.loc[c, 'share']:.1%})",
                va="center", fontsize=9, color=TEXT)
    ax.set_yticks(y, order)
    ax.set_xlim(0, vals.max() * 1.3)
    ax.xaxis.set_major_formatter(k)
    ax.set_xlabel("순매출 (천 £)", color=TEXT_2, fontsize=9)
    ax.set_title("① 카테고리별 순매출과 비중 (전체 기간)", loc="left", fontsize=11.5, color=TEXT)
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.spines["bottom"].set_visible(False)

    # ② 하반기 성장률
    ax = fig.add_axes([0.6, 0.665, 0.37, 0.25])
    g = s.loc[order, "growth_late_vs_early"]
    ax.barh(y, g, color=[GRAY if c == OTHER else (BLUE if v >= 0 else BAR)
                         for c, v in zip(order, g)], height=0.62, zorder=2)
    for yi, c in zip(y, order):
        ax.text(g[c] + 0.03, yi, f"{g[c]:+.0%}", va="center", fontsize=9, color=TEXT)
    ax.axvline(0, color=TEXT_2, linewidth=1)
    ax.set_yticks(y, order)
    ax.set_xlim(min(0, g.min()) - 0.1, g.max() * 1.2)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title("② 9~12월 일평균 순매출의 1~8월 대비 증감률", loc="left", fontsize=11.5, color=TEXT)
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.spines["bottom"].set_visible(False)

    # ③ 카테고리별 월 일평균 추이 (작은 그림 10개, 각자 세로축)
    months = sorted(monthly["year_month"].unique())
    xm = np.arange(len(months))
    for i, c in enumerate(order):
        ax = fig.add_axes([0.055 + (i % 5) * 0.19, 0.335 - (i // 5) * 0.27, 0.16, 0.18])
        m = monthly[monthly["category"] == c].set_index("year_month").reindex(months)
        color = GRAY if c == OTHER else BLUE
        ax.plot(xm[:-1], m["daily_avg"].iloc[:-1], color=color, linewidth=2, zorder=3)
        ax.plot(xm[-2:], m["daily_avg"].iloc[-2:], color=color, linewidth=2, linestyle=":", zorder=3)
        ax.scatter(xm[-1], m["daily_avg"].iloc[-1], s=18, color=color, zorder=4)
        ax.set_title(c, loc="left", fontsize=10.5, color=TEXT)
        ax.set_ylim(bottom=0, top=m["daily_avg"].max() * 1.15)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:,.1f}"))
        ax.set_xticks(xm[::3], [months[j][2:].replace("-", ".") for j in range(0, len(months), 3)])
        style(ax)
        if i % 5 == 0:
            ax.set_ylabel("일평균 (천 £)", color=TEXT_2, fontsize=9)

    fig.text(0.01, 0.975, "상품 카테고리별 순매출 (2010-12-01 ~ 2011-12-08, 상품명 키워드 기반 분류)",
             fontsize=14, color=TEXT, va="top")
    fig.text(0.01, 0.585, "③ 카테고리별 월 일평균 순매출 추이 (월 합계 ÷ 달력 일수, 카테고리마다 세로축 다름)",
             fontsize=11.5, color=TEXT)
    fig.text(0.01, 0.012,
             "카테고리는 Description 키워드 규칙에 의한 근사 분류. 점선·끝점 = 2011-12 (8일까지). "
             "② 증감률 = (2011-09~12 일평균) ÷ (2011-01~08 일평균) - 1.",
             fontsize=8.5, color=TEXT_2)
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, facecolor="white")


def main() -> None:
    pp = load_preprocess_module()
    raw = pp.load_raw()
    clean, _ = pp.apply_exclusions(raw)
    daily = pd.read_csv(pp.DAILY_PATH, parse_dates=["date"])

    # 상품명 단위 분류표
    desc = clean[["StockCode", "Description"]].drop_duplicates()
    desc[["category", "matched_keyword"]] = desc["Description"].map(classify).tolist()
    clean = clean.merge(desc, on=["StockCode", "Description"], how="left")
    assert clean["category"].notna().all()

    summary, monthly = summarize(clean, daily)
    assert abs(summary["net_sales"].sum() - daily["net_sales"].sum()) < 1e-6

    MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    desc.sort_values(["category", "StockCode"]).to_csv(MAP_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    monthly.to_csv(MONTHLY_PATH, index=False)
    daily_cat = (clean.assign(gross=clean["amount"].where(~clean["is_cancel"], 0.0),
                              cancel=clean["amount"].where(clean["is_cancel"], 0.0))
                 .groupby(["date", "category"])
                 .agg(net_sales=("amount", "sum"), gross_sales=("gross", "sum"),
                      cancel_amount=("cancel", "sum"))
                 .reset_index())
    daily_cat.to_csv(DAILY_CAT_PATH, index=False, date_format="%Y-%m-%d")
    plot(summary, monthly)

    n_other = (desc["category"] == OTHER).sum()
    with pd.option_context("display.float_format", "{:,.3f}".format,
                           "display.width", 250, "display.max_columns", 30):
        print(summary.to_string(index=False))
        print(f"\n상품(StockCode×Description) {len(desc):,}개 중 기타 {n_other:,}개 "
              f"({n_other / len(desc):.1%}), 기타 행 {(clean['category'] == OTHER).sum():,}건")


if __name__ == "__main__":
    main()
