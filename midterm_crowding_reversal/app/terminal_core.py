"""Shared data, calculations, theme and chart helpers for the Crowding Terminal (app/terminal.py).

Reads only files that already exist (raw market parquet, data/processed/panel.parquet and
outputs/); never trains a model. Backtests are recomputed from the saved walk-forward
predictions with the same engine as 05_backtest.py (src/backtest.py), so the default
settings reproduce the report's numbers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import backtest as B  # noqa: E402
from src.config import COST_BPS, OUT, PROCESSED, RAW, SETUP  # noqa: E402

SYMS = SETUP["universe"]["symbols"]
MODELS = {"lgbm": "LightGBM", "ridge": "Ridge", "rule": "規則（-funding z）"}
LADDER = SETUP["portfolio_extras"]["drawdown_ladder"]
LADDER_OFF = {"warn": 9, "halve": 9, "stop": 9}

# ---------------------------------------------------------------- theme
BG, PANEL, PANEL2, GRID, TXT, MUTED, ACCENT = "#0b0e11", "#151a21", "#1c222b", "#232a34", "#d6dae0", "#848e9c", "#f0b90b"
BLUE, PURPLE, CYAN = "#3d8bfd", "#a970ff", "#22c3d6"
FONT = "Inter, 'Noto Sans TC', 'Microsoft JhengHei', sans-serif"


def colors() -> tuple[str, str]:
    """(up, down) colours following the user's convention setting."""
    if st.session_state.get("color_mode", "綠漲紅跌") == "紅漲綠跌":
        return "#f6465d", "#0ecb81"
    return "#0ecb81", "#f6465d"


def inject_css():
    up, down = colors()
    st.markdown(f"""
<style>
:root {{ --up:{up}; --down:{down}; --accent:{ACCENT}; --panel:{PANEL}; --panel2:{PANEL2}; --grid:{GRID}; --muted:{MUTED}; }}
html, body, [class*="css"] {{ font-family:{FONT}; }}
.block-container {{ padding-top:3.2rem; padding-bottom:2rem; max-width:1600px; }}
#MainMenu, footer {{ visibility:hidden; }}
[data-testid="stHeader"] {{ background:transparent; }}
[data-testid="stSidebar"] {{ background:#0f1318; border-right:1px solid var(--grid); }}
h1, h2, h3 {{ letter-spacing:.2px; }}
.brand {{ display:flex; align-items:center; gap:.6rem; font-weight:800; font-size:1.15rem; }}
.brand .dot {{ width:10px; height:10px; border-radius:50%; background:var(--accent); box-shadow:0 0 10px var(--accent); }}
.brand-sub {{ color:var(--muted); font-size:.78rem; font-weight:400; }}
.topbar {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:.5rem; }}
.page-title {{ font-size:1.45rem; font-weight:700; margin:0 0 .15rem 0; }}
.page-sub {{ color:var(--muted); font-size:.86rem; margin-bottom:.8rem; }}
.tape {{ overflow:hidden; white-space:nowrap; border:1px solid var(--grid); background:var(--panel); border-radius:8px; margin-bottom:.9rem; }}
.tape-inner {{ display:inline-block; padding:.45rem 0; animation:tape 70s linear infinite; }}
.tape:hover .tape-inner {{ animation-play-state:paused; }}
.tape span.item {{ margin:0 1.3rem; font-size:.84rem; font-variant-numeric:tabular-nums; }}
.tape b {{ color:#eaecef; margin-right:.35rem; }}
@keyframes tape {{ from {{ transform:translateX(0); }} to {{ transform:translateX(-50%); }} }}
.up {{ color:var(--up) !important; }} .down {{ color:var(--down) !important; }} .muted {{ color:var(--muted); }}
.kpi-grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:.6rem; margin:.2rem 0 .9rem 0; }}
.kpi {{ background:var(--panel); border:1px solid var(--grid); border-radius:8px; padding:.65rem .8rem; }}
.kpi-l {{ color:var(--muted); font-size:.76rem; }}
.kpi-v {{ font-size:1.32rem; font-weight:700; font-variant-numeric:tabular-nums; margin-top:.1rem; }}
.kpi-s {{ color:var(--muted); font-size:.72rem; margin-top:.05rem; }}
.quote {{ display:flex; align-items:baseline; gap:1rem; flex-wrap:wrap; margin:.1rem 0 .4rem 0; }}
.quote .sym {{ font-size:1.5rem; font-weight:800; }}
.quote .px {{ font-size:1.9rem; font-weight:800; font-variant-numeric:tabular-nums; }}
.quote .chg {{ font-size:1.05rem; font-weight:600; font-variant-numeric:tabular-nums; }}
.stats {{ display:flex; gap:1.6rem; flex-wrap:wrap; color:var(--muted); font-size:.8rem; margin-bottom:.6rem; }}
.stats div b {{ display:block; color:#eaecef; font-size:.92rem; font-variant-numeric:tabular-nums; }}
.badge {{ display:inline-block; padding:.12rem .5rem; border-radius:4px; font-size:.75rem; font-weight:600; }}
.badge.long {{ background:color-mix(in srgb, var(--up) 18%, transparent); color:var(--up); }}
.badge.short {{ background:color-mix(in srgb, var(--down) 18%, transparent); color:var(--down); }}
.badge.neutral {{ background:#2b3139; color:var(--muted); }}
.panel {{ background:var(--panel); border:1px solid var(--grid); border-radius:8px; padding:.75rem .9rem; margin-bottom:.7rem; }}
.panel h4 {{ margin:0 0 .45rem 0; font-size:.92rem; }}
.pos-row {{ display:flex; justify-content:space-between; padding:.28rem 0; border-bottom:1px dashed var(--grid); font-size:.86rem; font-variant-numeric:tabular-nums; }}
.pos-row:last-child {{ border-bottom:none; }}
div[data-testid="stMetric"] {{ background:var(--panel); border:1px solid var(--grid); border-radius:8px; padding:.55rem .8rem; }}
div[data-testid="stForm"] {{ background:var(--panel); border:1px solid var(--grid); border-radius:8px; }}
.stTabs [data-baseweb="tab-list"] {{ gap:.2rem; border-bottom:1px solid var(--grid); }}
.stTabs [data-baseweb="tab"] {{ padding:.4rem .9rem; }}
.sq {{ display:inline-block; width:8px; height:8px; border-radius:2px; margin-right:.45rem; vertical-align:middle; }}
.up-bg {{ background:var(--up); }} .down-bg {{ background:var(--down); }}
[data-testid="stSidebarNav"] span[data-testid="stIconMaterial"] {{ color:var(--muted); font-size:1.15rem; }}
[data-testid="stSidebarNav"] a[aria-current="page"] span[data-testid="stIconMaterial"] {{ color:var(--accent); }}
[data-testid="stSidebarNav"] a[aria-current="page"] {{ box-shadow:inset 2px 0 0 var(--accent); border-radius:4px; }}
[data-testid="stNavSectionHeader"] {{ text-transform:uppercase; letter-spacing:.08em; font-size:.72rem; color:var(--muted); }}
</style>""", unsafe_allow_html=True)


def page_header(title: str, sub: str = ""):
    st.markdown(f'<div class="page-title">{title}</div>' + (f'<div class="page-sub">{sub}</div>' if sub else ""),
                unsafe_allow_html=True)


def tone(x: float) -> str:
    return "" if x is None or not np.isfinite(x) or x == 0 else ("up" if x > 0 else "down")


def kpi_row(items):
    """items: (label, value_text, sub_text, tone_class)."""
    cells = "".join(f'<div class="kpi"><div class="kpi-l">{l}</div><div class="kpi-v {t}">{v}</div>'
                    f'<div class="kpi-s">{s}</div></div>' for l, v, s, t in items)
    st.markdown(f'<div class="kpi-grid">{cells}</div>', unsafe_allow_html=True)


def fmt_px(x: float) -> str:
    if not np.isfinite(x):
        return "—"
    return f"{x:,.2f}" if x >= 100 else (f"{x:,.4f}" if x >= 1 else f"{x:,.5f}")


def pct(x: float, d: int = 2, sign: bool = True) -> str:
    return "—" if x is None or not np.isfinite(x) else (f"{x:+.{d}%}" if sign else f"{x:.{d}%}")


def style(fig: go.Figure, h: int = 420, legend: bool = True, right_axis: bool = False) -> go.Figure:
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=PANEL,
                      font=dict(family=FONT, color=TXT, size=12), height=h, hovermode="x unified",
                      margin=dict(l=8, r=8, t=56, b=8), showlegend=legend,
                      legend=dict(orientation="h", yanchor="bottom", y=1.005, x=1, xanchor="right", bgcolor="rgba(0,0,0,0)"),
                      hoverlabel=dict(bgcolor=PANEL2, bordercolor=GRID, font=dict(family=FONT)))
    if fig.layout.title.text:
        fig.update_layout(title=dict(x=0.005, xanchor="left", y=0.995, yanchor="top", font=dict(size=14)))
    fig.update_xaxes(gridcolor=GRID, zeroline=False, showspikes=True, spikemode="across", spikesnap="cursor",
                     spikecolor="#5e6673", spikethickness=1, spikedash="dot")
    fig.update_yaxes(gridcolor=GRID, zeroline=False, side="right" if right_axis else "left")
    return fig


def show(fig: go.Figure):
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": True,
                                                           "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


# ---------------------------------------------------------------- data
@st.cache_data(show_spinner="載入 1 小時 K 線…")
def hourly() -> pd.DataFrame:
    h = pd.read_parquet(RAW["klines_1h"])
    return h.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def sym_hourly(sym: str) -> pd.DataFrame:
    h = hourly()
    return h[h["symbol"] == sym].set_index("timestamp").drop(columns="symbol")


@st.cache_data(show_spinner=False)
def funding_raw() -> pd.DataFrame:
    return pd.read_parquet(RAW["funding"]).sort_values(["symbol", "timestamp"])


@st.cache_data(show_spinner=False)
def premium_raw() -> pd.DataFrame:
    return pd.read_parquet(RAW["premium_1h"], columns=["timestamp", "symbol", "premium_index_close"])


@st.cache_data(show_spinner="整理每日行情…")
def daily_market() -> pd.DataFrame:
    """Calendar-day (UTC) bars per coin with funding, premium and the 30-day funding z-score.

    Row D describes day D itself (known at D+1 00:00), which is what a quote screen shows;
    the research panel shifts the same numbers to decision date D+1.
    """
    h = hourly().assign(day=lambda x: x["timestamp"].dt.floor("D"))
    d = h.groupby(["symbol", "day"]).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                                         close=("close", "last"), volume=("volume", "sum"),
                                         quote_volume=("quote_volume", "sum")).reset_index()
    f = funding_raw()
    f = f.assign(day=(f["timestamp"] - pd.Timedelta(microseconds=1)).dt.floor("D"))
    fd = f.groupby(["symbol", "day"]).agg(funding_day=("funding_rate", "sum"),
                                          funding_last=("funding_rate", "last")).reset_index()
    p = premium_raw()
    pdl = (p.assign(day=p["timestamp"].dt.floor("D")).groupby(["symbol", "day"])["premium_index_close"]
           .mean().rename("premium").reset_index())
    d = d.merge(fd, on=["symbol", "day"], how="left").merge(pdl, on=["symbol", "day"], how="left")
    d = d.sort_values(["symbol", "day"]).reset_index(drop=True)
    g = d.groupby("symbol")
    for n in (1, 7, 30, 90):
        d[f"ret_{n}d"] = g["close"].pct_change(n)
    fm = g["funding_day"].transform(lambda s: s.rolling(30, min_periods=20).mean())
    fs = g["funding_day"].transform(lambda s: s.rolling(30, min_periods=20).std())
    d["funding_z"] = (d["funding_day"] - fm) / fs.replace(0, np.nan)
    d["adv_30d"] = g["quote_volume"].transform(lambda s: s.rolling(30, min_periods=1).mean())
    d["rvol_30d"] = g["close"].transform(lambda s: np.log(s).diff().rolling(30, min_periods=10).std() * np.sqrt(365))
    d["funding_ann"] = d["funding_day"] * 365
    return d


@st.cache_data(show_spinner=False)
def J(name: str):
    p = OUT / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8").replace("Infinity", "1e999")) if p.exists() else None


@st.cache_data(show_spinner=False)
def out_parquet(name: str) -> pd.DataFrame:
    return pd.read_parquet(OUT / f"{name}.parquet")


@st.cache_data(show_spinner="載入模型預測…")
def bt_base() -> pd.DataFrame:
    """Exactly the frame 05_backtest.py backtests on."""
    D = out_parquet("model_panel")
    P = out_parquet("wf_predictions")
    base = D[["date", "symbol", "ret_long_simple", "rvol_20d", "ret_long"]].merge(P, on=["date", "symbol"])
    pp = pd.read_parquet(PROCESSED / "panel.parquet", columns=["date", "symbol", "ret_price", "funding_hold", "tradable"])
    return base.merge(pp, on=["date", "symbol"], how="left")


@st.cache_data(show_spinner=False)
def signal_frame() -> pd.DataFrame:
    """Predictions plus the raw signals behind them, for the daily ranking screen."""
    base = bt_base()
    extra = out_parquet("model_panel")[["date", "symbol", "funding_z_30d_raw"]]
    pnl = pd.read_parquet(PROCESSED / "panel.parquet", columns=["date", "symbol", "funding_day", "premium_mean_1d", "close"])
    return base.merge(extra, on=["date", "symbol"], how="left").merge(pnl, on=["date", "symbol"], how="left")


# ---------------------------------------------------------------- backtest
def _day_weights(day: pd.DataFrame, score: str, k: int, allocator: str) -> pd.Series:
    """src.backtest.day_weights with the number of names per side as a parameter."""
    d = day[day["tradable"] & day[score].notna()].sort_values(score)
    if len(d) < B.MIN_NAMES:
        return pd.Series(dtype=float)
    longs, shorts = d.iloc[-k:], d.iloc[:k]
    if allocator == "inv_vol":
        wl = 1 / longs.set_index("symbol")["rvol_20d"].clip(lower=1e-4)
        ws = 1 / shorts.set_index("symbol")["rvol_20d"].clip(lower=1e-4)
    else:
        wl = pd.Series(1.0, index=longs["symbol"])
        ws = pd.Series(1.0, index=shorts["symbol"])
    return pd.concat([B._cap_side(wl, 0.5), -B._cap_side(ws, 0.5)])


@st.cache_data(show_spinner="計算每日持倉…")
def weights(score: str, k: int = 3, allocator: str = "equal") -> pd.DataFrame:
    base = bt_base()
    if k == B.K:
        return B.weights(base, score, allocator)
    W = {dt: _day_weights(g, score, k, allocator) for dt, g in base.groupby("date")}
    return pd.DataFrame(W).T.sort_index().fillna(0.0)


@st.cache_data(show_spinner="回測中…")
def backtest(score: str, k: int, allocator: str, cost: float, start, end,
             vol_target: float | None, ladder: bool) -> pd.DataFrame:
    base = bt_base()
    W = weights(score, k, allocator)
    W = W[(W.index >= start) & (W.index <= end)]
    sub = base[(base["date"] >= start) & (base["date"] <= end)]
    bt = B.run(sub, score, cost, W=W)
    # 05_backtest.py saved these exact series; reuse them so the default view matches the report to the
    # last digit (a score tie can sort differently across machines and move a single day)
    saved = out_parquet("backtest_daily").set_index("date")
    col = score if allocator == "equal" else f"{score}_invvol"
    if k == B.K and f"{col}_gross" in saved:
        bt["gross"] = saved[f"{col}_gross"].reindex(bt.index).fillna(bt["gross"])
        bt["turnover"] = saved[f"{col}_turnover"].reindex(bt.index).fillna(bt["turnover"])
        bt.iloc[0, bt.columns.get_loc("turnover")] = W.iloc[0].abs().sum()
        bt["net"] = bt["gross"] - bt["turnover"] * cost / 1e4
    bt["scale"] = 1.0
    if vol_target or ladder:
        o = B.apply_overlays(bt, cost, vol_target=vol_target, ladder=LADDER if ladder else LADDER_OFF)
        bt["net"], bt["scale"] = o["net"], o["scale"]
    return bt


@st.cache_data(show_spinner=False)
def benchmarks(start, end) -> pd.DataFrame:
    base = bt_base()
    b = base[(base["date"] >= start) & (base["date"] <= end)]
    ew = b[b["tradable"]].groupby("date")["ret_long_simple"].mean()
    btc = b[b["symbol"] == "BTCUSDT"].set_index("date")["ret_long_simple"]
    return pd.DataFrame({"等權做多全部": ew, "BTC 做多": btc}).fillna(0)


def perf(net: pd.Series, turnover: pd.Series | None = None) -> dict:
    r = net.dropna()
    n = len(r)
    eq = (1 + r).cumprod()
    sd = r.std()
    sr = r.mean() / sd * np.sqrt(365) if sd else np.nan
    se = np.sqrt((1 + sr ** 2 / 2) / (n / 365)) if n > 2 else np.nan
    dd = eq / eq.cummax() - 1
    active = r[r != 0]
    out = dict(total=eq.iloc[-1] - 1 if n else np.nan,
               cagr=eq.iloc[-1] ** (365 / n) - 1 if n else np.nan,
               vol=sd * np.sqrt(365), sharpe=sr, sharpe_lo=sr - 1.96 * se, sharpe_hi=sr + 1.96 * se,
               max_dd=dd.min(), hit=(active > 0).mean() if len(active) else np.nan,
               best=r.max(), worst=r.min(), days=n,
               calmar=(eq.iloc[-1] ** (365 / n) - 1) / abs(dd.min()) if n and dd.min() < 0 else np.nan)
    if turnover is not None:
        out["ann_turnover"] = turnover.mean() * 365
    return out


def monthly_table(net: pd.Series) -> pd.DataFrame:
    r = net.dropna()
    m = (1 + r).groupby([r.index.year, r.index.month]).prod() - 1
    t = m.unstack()
    t.columns = [f"{c}月" for c in t.columns]
    t["全年"] = (1 + r).groupby(r.index.year).prod() - 1
    t.index.name = "年度"
    return t


def to_utc(d) -> pd.Timestamp:
    return pd.Timestamp(d).tz_localize("UTC") if pd.Timestamp(d).tzinfo is None else pd.Timestamp(d)


def quick_range(choice: str, d0: pd.Timestamp, d1: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    days = {"7D": 7, "1M": 30, "3M": 91, "6M": 182, "1Y": 365, "3Y": 1095}
    if choice in days:
        return max(d0, d1 - pd.Timedelta(days=days[choice])), d1
    if choice == "YTD":
        return max(d0, pd.Timestamp(year=d1.year, month=1, day=1, tz="UTC")), d1
    return d0, d1


def date_range_picker(key: str, d0: pd.Timestamp, d1: pd.Timestamp, default: str = "1Y",
                      options=("1M", "3M", "6M", "YTD", "1Y", "3Y", "全部", "自訂")):
    """Pills for quick ranges plus a custom calendar, like a quote terminal."""
    choice = st.pills("期間", list(options), default=default, key=f"{key}_rng", label_visibility="collapsed") or default
    if choice == "自訂":
        lo, hi = quick_range(default, d0, d1)
        v = st.date_input("自訂日期區間", value=(lo.date(), hi.date()), min_value=d0.date(), max_value=d1.date(),
                          key=f"{key}_custom")
        if isinstance(v, (tuple, list)) and len(v) == 2:
            return to_utc(v[0]), to_utc(v[1])
        return lo, hi
    return quick_range(choice, d0, d1)


# ---------------------------------------------------------------- live quotes
def _live_binance() -> pd.DataFrame:
    from app.live_signal import fetch_live
    return fetch_live(timeout=6)


def _live_bybit() -> pd.DataFrame:
    import requests
    js = requests.get("https://api.bybit.com/v5/market/tickers", params={"category": "linear"}, timeout=6).json()
    df = pd.DataFrame(js["result"]["list"])
    df = df[df["symbol"].isin(SYMS)].copy()
    for c in ["markPrice", "indexPrice", "fundingRate"]:
        df[c] = df[c].astype(float)
    out = pd.DataFrame({"symbol": df["symbol"], "lastFundingRate": df["fundingRate"],
                        "premium": df["markPrice"] / df["indexPrice"] - 1, "markPrice": df["markPrice"],
                        "next_funding": pd.to_datetime(df["nextFundingTime"].astype("int64"), unit="ms", utc=True)})
    return out


def _live_hyperliquid() -> pd.DataFrame:
    """Hyperliquid perps settle hourly; scale to an 8h-equivalent rate so the ranking is comparable."""
    import requests
    meta, ctx = requests.post("https://api.hyperliquid.xyz/info", json={"type": "metaAndAssetCtxs"}, timeout=6).json()
    rows = []
    for a, c in zip(meta["universe"], ctx):
        sym = f"{a['name']}USDT"
        if sym in SYMS and c.get("markPx") and c.get("oraclePx"):
            mark, orc = float(c["markPx"]), float(c["oraclePx"])
            rows.append({"symbol": sym, "lastFundingRate": float(c["funding"]) * 8, "premium": mark / orc - 1, "markPrice": mark})
    now = pd.Timestamp.now(tz="UTC")
    out = pd.DataFrame(rows)
    out["next_funding"] = now.ceil("h")
    return out


LIVE_SOURCES = [("Binance", _live_binance), ("Bybit", _live_bybit), ("Hyperliquid", _live_hyperliquid)]


@st.cache_data(ttl=60, show_spinner="抓取即時資金費率…")
def live_quotes() -> tuple[pd.DataFrame, str, list[str]]:
    """Try each exchange in turn (cloud hosts in the US are blocked by Binance and Bybit).

    Returns (frame, source, errors). Falls back to the last settled day in the dataset.
    """
    errors = []
    for name, fn in LIVE_SOURCES:
        try:
            df = fn()
            if len(df) >= 8:
                df["fetched_utc"] = pd.Timestamp.now(tz="UTC")
                return df.reset_index(drop=True), name, errors
            errors.append(f"{name}: 只有 {len(df)} 檔")
        except Exception as e:  # blocked region, timeout, schema change
            errors.append(f"{name}: {type(e).__name__}")
    d = daily_market()
    last = d[d["day"] == d["day"].max()]
    df = pd.DataFrame({"symbol": last["symbol"], "lastFundingRate": last["funding_last"], "premium": last["premium"],
                       "markPrice": last["close"], "next_funding": pd.NaT,
                       "fetched_utc": last["day"] + pd.Timedelta(days=1)}).dropna(subset=["lastFundingRate"])
    return df.reset_index(drop=True), "最近一次已結算資料", errors
