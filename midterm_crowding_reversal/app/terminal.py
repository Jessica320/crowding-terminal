"""Crowding Terminal: an interactive quote-and-strategy system for the crowding-reversal project.

    streamlit run app/terminal.py

Market pages read the raw 1h bars, funding and premium files; strategy and research pages
read outputs/ (walk-forward predictions, risk, holdout). Nothing here trains a model.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))
import terminal_core as C  # noqa: E402
from terminal_core import L  # noqa: E402

st.set_page_config(page_title="Crowding Terminal｜擁擠度反轉交易系統", page_icon=":material/candlestick_chart:", layout="wide",
                   initial_sidebar_state="expanded")
ss = st.session_state
ss.setdefault("sym", "BTCUSDT")
ss.setdefault("color_mode", "green_up")
ss.setdefault("lang", "EN")
ss.setdefault("model", (C.J("results") or {}).get("chosen", "lgbm"))
ss.setdefault("scr_n", 0)


def ticker_tape():
    d = C.daily_market()
    last = d[(d["day"] == d["day"].max()) & (d["quote_volume"] > 0)]   # skip coins with no trading that day
    items = "".join(
        f'<span class="item"><b>{r.symbol.replace("USDT", "")}</b>{C.fmt_px(r.close)} '
        f'<span class="{C.tone(r.ret_1d)}">{C.pct(r.ret_1d)}</span> '
        f'<span class="muted">· {L("資金費", "funding")} {r.funding_ann:+.1%}{L("/年", "/yr")}</span></span>'
        for r in last.itertuples())
    st.markdown(f'<div class="tape"><div class="tape-inner">{items}{items}</div></div>', unsafe_allow_html=True)


# ==================================================================== 行情：市場總覽
def page_market():
    C.page_header(L("市場總覽", "Market Overview"), L("19 檔 Binance USDT 永續合約的報價、資金費率與擁擠度。點表格任一列可直接打開該幣的 K 線。", "Quotes, funding rates and crowding for 19 Binance USDT perpetuals. Click any row to open that coin's chart."))
    d = C.daily_market()
    days = d["day"].drop_duplicates().sort_values()
    c1, c2, c3, c4 = st.columns([1.2, 1.4, 1.6, 1.2])
    asof = C.to_utc(c1.date_input(L("資料日期（UTC）", "Data date (UTC)"), value=days.iloc[-1].date(),
                                   min_value=days.iloc[30].date(), max_value=days.iloc[-1].date()))
    asof = days[days <= asof].iloc[-1]
    view = c2.segmented_control(L("篩選", "Filter"), [L("全部", "All"), L("多方擁擠", "Longs crowded"), L("空方擁擠", "Shorts crowded")], default=L("全部", "All")) or L("全部", "All")
    q = c3.text_input(L("搜尋幣種", "Search symbol"), placeholder=L("例如 BTC、SOL…", "e.g. BTC, SOL…")).strip().upper()
    sort_by = c4.selectbox(L("排序", "Sort by"), [L("擁擠度 z", "Crowding z"), L("24h 漲跌", "24h change"), L("7 日漲跌", "7D change"), L("30 日漲跌", "30D change"), L("年化資金費", "Funding (ann.)"), L("30 日均成交額", "30D avg volume")])

    today = d[(d["day"] == asof) & (d["quote_volume"] > 0)].copy()   # drops coins delisted or halted that day
    today["rank"] = today["funding_z"].rank(ascending=False, method="first")
    n = today["funding_z"].notna().sum()
    today[L("狀態", "Status")] = np.where(today["rank"] <= 3, L("做空候選（多方擁擠）", "Short candidate (longs crowded)"),
                           np.where(today["rank"] > n - 3, L("做多候選（空方擁擠）", "Long candidate (shorts crowded)"), L("中性", "Neutral")))
    hist = d[(d["day"] <= asof) & (d["day"] > asof - pd.Timedelta(days=30))]
    spark = hist.groupby("symbol")["close"].apply(list)
    today[L("30 日走勢", "30D trend")] = today["symbol"].map(spark)

    up, down = C.colors()
    breadth = (today["ret_1d"] > 0).mean()
    top = today.loc[today["funding_z"].idxmax()] if n else None
    bot = today.loc[today["funding_z"].idxmin()] if n else None
    C.kpi_row([
        ("BTC", C.fmt_px(today.loc[today.symbol == "BTCUSDT", "close"].squeeze()),
         C.pct(today.loc[today.symbol == "BTCUSDT", "ret_1d"].squeeze()), C.tone(today.loc[today.symbol == "BTCUSDT", "ret_1d"].squeeze())),
        (L("上漲家數", "Advancers"), f"{(today['ret_1d'] > 0).sum()} / {len(today)}", L(f"市場寬度 {breadth:.0%}", f"Breadth {breadth:.0%}"), "up" if breadth >= .5 else "down"),
        (L("平均年化資金費", "Avg funding (ann.)"), f"{today['funding_ann'].mean():+.1%}", L("正值＝多方付費給空方", "Positive = longs pay shorts"), C.tone(today['funding_ann'].mean())),
        (L("最擁擠（多方）", "Most crowded (longs)"), top.symbol.replace("USDT", "") if top is not None else "—", f"z = {top.funding_z:+.2f}" if top is not None else "", "down"),
        (L("最擁擠（空方）", "Most crowded (shorts)"), bot.symbol.replace("USDT", "") if bot is not None else "—", f"z = {bot.funding_z:+.2f}" if bot is not None else "", "up"),
        (L("資金費分散度", "Funding dispersion"), f"{today['funding_day'].std() * 1e4:.1f} bp", L("各幣日資金費的標準差", "Cross-sectional std of daily funding"), ""),
    ])

    t = today
    if view == L("多方擁擠", "Longs crowded"):
        t = t[t["funding_z"] > 0]
    elif view == L("空方擁擠", "Shorts crowded"):
        t = t[t["funding_z"] < 0]
    if q:
        t = t[t["symbol"].str.contains(q)]
    key = {L("擁擠度 z", "Crowding z"): "funding_z", L("24h 漲跌", "24h change"): "ret_1d", L("7 日漲跌", "7D change"): "ret_7d", L("30 日漲跌", "30D change"): "ret_30d",
           L("年化資金費", "Funding (ann.)"): "funding_ann", L("30 日均成交額", "30D avg volume"): "adv_30d"}[sort_by]
    t = t.sort_values(key, ascending=False)
    table = pd.DataFrame({
        L("擁擠排名", "Crowding rank"): t["rank"].astype("Int64"), L("幣種", "Symbol"): t["symbol"].str.replace("USDT", ""), L("收盤價", "Close"): t["close"],
        "24h": t["ret_1d"] * 100, L("7 日", "7D"): t["ret_7d"] * 100, L("30 日", "30D"): t["ret_30d"] * 100, L("30 日走勢", "30D trend"): t[L("30 日走勢", "30D trend")],
        L("日資金費 (bp)", "Daily funding (bp)"): t["funding_day"] * 1e4, L("年化資金費", "Funding (ann.)"): t["funding_ann"] * 100, L("擁擠度 z", "Crowding z"): t["funding_z"],
        L("溢價 (bp)", "Premium (bp)"): t["premium"] * 1e4, L("30 日均成交額 (百萬)", "30D avg volume ($M)"): t["adv_30d"] / 1e6, L("狀態", "Status"): t[L("狀態", "Status")],
    }).reset_index(drop=True)
    styled = (table.style
              .map(lambda v: f"color:{up}" if isinstance(v, (int, float)) and v > 0 else (f"color:{down}" if isinstance(v, (int, float)) and v < 0 else ""),
                   subset=["24h", L("7 日", "7D"), L("30 日", "30D"), L("年化資金費", "Funding (ann.)")])
              .map(lambda v: f"color:{down}" if str(v).startswith(L("做空", "Short")) else (f"color:{up}" if str(v).startswith(L("做多", "Long")) else f"color:{C.MUTED}"),
                   subset=[L("狀態", "Status")])
              .background_gradient(cmap="RdBu_r", vmin=-3, vmax=3, subset=[L("擁擠度 z", "Crowding z")])
              .format({L("收盤價", "Close"): C.fmt_px, "24h": "{:+.2f}%", L("7 日", "7D"): "{:+.2f}%", L("30 日", "30D"): "{:+.2f}%",
                       L("日資金費 (bp)", "Daily funding (bp)"): "{:+.2f}", L("年化資金費", "Funding (ann.)"): "{:+.1f}%", L("擁擠度 z", "Crowding z"): "{:+.2f}", L("溢價 (bp)", "Premium (bp)"): "{:+.1f}",
                       L("30 日均成交額 (百萬)", "30D avg volume ($M)"): "{:,.0f}"}, na_rep="—"))
    ev = st.dataframe(
        styled, use_container_width=True, hide_index=True, height=min(35 * len(table) + 38, 760),
        on_select="rerun", selection_mode="single-row", key=f"screener_{ss.scr_n}",
        column_config={
            L("30 日走勢", "30D trend"): st.column_config.LineChartColumn(width="small"),
            L("擁擠度 z", "Crowding z"): st.column_config.Column(help=L("當日資金費相對自身 30 日的 z 分數；越高＝多方越擠", "Today's funding vs its own 30-day history (z-score); higher = longs more crowded")),
        })
    rows = ev.selection.rows if ev and hasattr(ev, "selection") else []
    if rows:
        ss.sym = table.iloc[rows[0]][L("幣種", "Symbol")] + "USDT"
        ss.scr_n += 1
        st.switch_page(PAGES["chart"])
    st.caption(L("排名依『規則模型』：擁擠度 z 最高 3 檔為做空候選、最低 3 檔為做多候選。實際策略在隔日 01:00 UTC 進場。", "Ranked by the rule model: the 3 highest crowding z are short candidates, the 3 lowest are long candidates. The strategy enters at the next 01:00 UTC open."))

    left, right = st.columns([1, 1.35])
    with left:
        b = today.dropna(subset=["funding_z"]).sort_values("funding_z")
        fig = go.Figure(go.Bar(x=b["funding_z"], y=b["symbol"].str.replace("USDT", ""), orientation="h",
                               marker_color=np.where(b["funding_z"] > 0, down, up),
                               text=b["funding_z"].map("{:+.2f}".format), textposition="outside", cliponaxis=False))
        fig.add_vline(x=0, line_color=C.MUTED, line_width=1)
        lim = float(b["funding_z"].abs().max() or 1) * 1.25
        fig.update_xaxes(range=[-lim, lim])
        fig.update_layout(title=L(f"擁擠度 z 分布（{asof:%Y-%m-%d}）", f"Crowding z by coin ({asof:%Y-%m-%d})"))
        C.show(C.style(fig, 520, legend=False))
    with right:
        rng = st.segmented_control(L("熱力圖期間", "Heatmap window"), [L("30 天", "30 days"), L("90 天", "90 days"), L("180 天", "180 days"), L("1 年", "1 year")], default=L("90 天", "90 days"), key="hm_rng") or L("90 天", "90 days")
        nd = {L("30 天", "30 days"): 30, L("90 天", "90 days"): 90, L("180 天", "180 days"): 180, L("1 年", "1 year"): 365}[rng]
        hm = d[(d["day"] <= asof) & (d["day"] > asof - pd.Timedelta(days=nd))]
        m = hm.pivot(index="symbol", columns="day", values="funding_z")
        m.index = m.index.str.replace("USDT", "")
        fig = go.Figure(go.Heatmap(z=m.values, x=m.columns, y=m.index, colorscale="RdBu_r", zmin=-3, zmax=3,
                                   colorbar=dict(title="z", thickness=10),
                                   hovertemplate="%{y}<br>%{x|%Y-%m-%d}<br>z = %{z:+.2f}<extra></extra>"))
        fig.update_layout(title=L("擁擠度熱力圖（紅＝多方擁擠，藍＝空方擁擠）", "Crowding heatmap (red = longs crowded, blue = shorts crowded)"), hovermode="closest")
        C.show(C.style(fig, 470, legend=False))


# ==================================================================== 行情：K 線
def indicator_panel_rows(subs):
    return 1 + len(subs)


def page_chart():
    d = C.daily_market()
    tab_k, tab_cmp = st.tabs([L(":material/candlestick_chart: K 線分析", ":material/candlestick_chart: Chart"), L(":material/compare_arrows: 多幣比較", ":material/compare_arrows: Compare")])
    with tab_k:
        c1, c2, _ = st.columns([1.3, 1.6, 2.6])
        sym = c1.selectbox(L("幣種", "Symbol"), C.SYMS, index=C.SYMS.index(ss.sym),
                           format_func=lambda s: L(f"{s.replace('USDT', '')} / USDT 永續", f"{s.replace('USDT', '')} / USDT perpetual"))
        ss.sym = sym
        itv = c2.segmented_control(L("週期", "Interval"), ["1H", "4H", "1D", "1W"], default="1D", key="itv") or "1D"
        h = C.sym_hourly(sym)
        d0, d1 = h.index.min().normalize(), h.index.max()
        r1, r2, r3 = st.columns([2.2, 2.2, 2.2])
        with r1:
            start, end = C.date_range_picker("k", d0, d1, default="3M" if itv in ("1H", "4H") else "1Y")
        ind = r2.multiselect(L("主圖指標", "Overlays"), ["MA7", "MA25", "MA99", L("布林通道", "Bollinger Bands"), L("成交量加權均價", "VWAP")], default=["MA25", "MA99"])
        subs = r3.multiselect(L("副圖", "Sub-panels"), [L("成交量", "Volume"), L("資金費率", "Funding rate"), L("擁擠度 z", "Crowding z"), L("溢價指數", "Premium index")], default=[L("成交量", "Volume"), L("資金費率", "Funding rate"), L("擁擠度 z", "Crowding z")])
        o1, o2, o3 = st.columns([1.4, 1.4, 3])
        show_sig = o1.toggle(L("標示模型進出場", "Show model signals"), value=True, help=L("2021–2024 驗證期，模型把這檔排進多/空 3 檔的日子", "Days in the 2021–2024 validation period when the model put this coin in its top/bottom 3"))
        model = o2.selectbox(L("訊號模型", "Signal model"), list(C.MODELS), index=list(C.MODELS).index(ss.model),
                             format_func=C.mname, disabled=not show_sig)
        log_y = o3.toggle(L("對數座標", "Log scale"), value=False)

        if itv == "1H" and (end - start).days > 180:
            start = end - pd.Timedelta(days=180)
            st.info(L("1 小時週期最多顯示 180 天，已自動縮短期間。", "The 1H interval shows at most 180 days; the range was shortened."))
        s = h[(h.index >= start) & (h.index <= end + pd.Timedelta(days=1))]
        rule = {"1H": None, "4H": "4h", "1D": "1D", "1W": "W-MON"}[itv]
        if rule:
            s = s.resample(rule, label="left", closed="left").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "quote_volume": "sum"}).dropna()
        if s.empty:
            st.warning(L("這段期間沒有資料。", "No data in this range."))
            return

        dd = d[d["symbol"] == sym].set_index("day")
        last = dd.iloc[-1] if end >= dd.index.max() else dd[dd.index <= end].iloc[-1]
        chg = s["close"].iloc[-1] / s["open"].iloc[0] - 1
        pct_rank = (dd["funding_z"] <= last["funding_z"]).mean()
        st.markdown(
            f'<div class="quote"><span class="sym">{sym.replace("USDT", "")}<span class="muted" style="font-size:1rem">/USDT {L("永續", "Perp")}</span></span>'
            f'<span class="px {C.tone(last.ret_1d)}">{C.fmt_px(s["close"].iloc[-1])}</span>'
            f'<span class="chg {C.tone(last.ret_1d)}">{C.pct(last.ret_1d)} 24h</span>'
            f'<span class="chg {C.tone(chg)}">{C.pct(chg)} {L("區間", "range")}</span></div>'
            f'<div class="stats"><div>{L("區間最高", "High")}<b>{C.fmt_px(s["high"].max())}</b></div><div>{L("區間最低", "Low")}<b>{C.fmt_px(s["low"].min())}</b></div>'
            f'<div>{L("區間成交額", "Volume")}<b>{s["quote_volume"].sum() / 1e9:,.2f} B</b></div>'
            f'<div>{L("日資金費", "Daily funding")}<b class="{C.tone(last.funding_day)}">{last.funding_day * 1e4:+.2f} bp</b></div>'
            f'<div>{L("年化資金費", "Funding (ann.)")}<b class="{C.tone(last.funding_ann)}">{last.funding_ann:+.1%}</b></div>'
            f'<div>{L("擁擠度 z", "Crowding z")}<b>{last.funding_z:+.2f}</b></div><div>{L("擁擠度歷史分位", "Crowding percentile")}<b>{pct_rank:.0%}</b></div>'
            f'<div>{L("30 日年化波動", "30D vol (ann.)")}<b>{last.rvol_30d:.0%}</b></div><div>{L("資料截至", "Data as of")}<b>{last.name:%Y-%m-%d} UTC</b></div></div>',
            unsafe_allow_html=True)

        up, down = C.colors()
        rows = 1 + len(subs)
        heights = [0.58] + [0.42 / len(subs)] * len(subs) if subs else [1.0]
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.025, row_heights=heights)
        fig.add_trace(go.Candlestick(x=s.index, open=s["open"], high=s["high"], low=s["low"], close=s["close"], name=sym,
                                     increasing=dict(line=dict(color=up), fillcolor=up),
                                     decreasing=dict(line=dict(color=down), fillcolor=down)), 1, 1)
        palette = {"MA7": C.ACCENT, "MA25": C.PURPLE, "MA99": C.CYAN}
        for k in ind:
            if k.startswith("MA"):
                n = int(k[2:])
                fig.add_trace(go.Scatter(x=s.index, y=s["close"].rolling(n).mean(), name=k,
                                         line=dict(width=1.3, color=palette[k])), 1, 1)
        if L("布林通道", "Bollinger Bands") in ind:
            m, sd = s["close"].rolling(20).mean(), s["close"].rolling(20).std()
            fig.add_trace(go.Scatter(x=s.index, y=m + 2 * sd, name=L("BB 上軌", "BB upper"), line=dict(width=1, color="#5e6673")), 1, 1)
            fig.add_trace(go.Scatter(x=s.index, y=m - 2 * sd, name=L("BB 下軌", "BB lower"), line=dict(width=1, color="#5e6673"),
                                     fill="tonexty", fillcolor="rgba(94,102,115,.08)"), 1, 1)
        if L("成交量加權均價", "VWAP") in ind:
            tp = (s["high"] + s["low"] + s["close"]) / 3
            fig.add_trace(go.Scatter(x=s.index, y=(tp * s["volume"]).cumsum() / s["volume"].cumsum(), name=L("VWAP（區間）", "VWAP (range)"),
                                     line=dict(width=1.3, dash="dot", color=C.BLUE)), 1, 1)
        if show_sig:
            W = C.weights(model, 3, "equal")
            if sym in W.columns:
                w = W[sym]
                w = w[(w.index >= start) & (w.index <= end)]
                for side, mask, color, sym_ in [(L("模型做多", "Model long"), w > 0, up, "triangle-up"), (L("模型做空", "Model short"), w < 0, down, "triangle-down")]:
                    dts = w.index[mask]
                    if len(dts):
                        # decision date t trades at the 01:00 open; mark the daily low/high of that day
                        px = h["low"].resample("1D").min() if side == L("模型做多", "Model long") else h["high"].resample("1D").max()
                        y = px.reindex(dts) * (0.985 if side == L("模型做多", "Model long") else 1.015)
                        fig.add_trace(go.Scatter(x=dts, y=y, mode="markers", name=side,
                                                 marker=dict(symbol=sym_, size=8, color=color, line=dict(width=0))), 1, 1)
        row = 2
        dz = dd[(dd.index >= start) & (dd.index <= end)]
        for sp in subs:
            if sp == L("成交量", "Volume"):
                colr = np.where(s["close"] >= s["open"], up, down)
                fig.add_trace(go.Bar(x=s.index, y=s["quote_volume"] / 1e6, name=L("成交額（百萬）", "Volume ($M)"), marker_color=colr, opacity=.7), row, 1)
            elif sp == L("資金費率", "Funding rate"):
                fig.add_trace(go.Bar(x=dz.index, y=dz["funding_day"] * 1e4, name=L("日資金費 (bp)", "Daily funding (bp)"),
                                     marker_color=np.where(dz["funding_day"] > 0, down, up)), row, 1)
            elif sp == L("擁擠度 z", "Crowding z"):
                fig.add_trace(go.Scatter(x=dz.index, y=dz["funding_z"], name=L("擁擠度 z", "Crowding z"), line=dict(color=C.ACCENT, width=1.4)), row, 1)
                for lv in (2, -2):
                    fig.add_hline(y=lv, line_dash="dot", line_color="#5e6673", row=row, col=1)
            elif sp == L("溢價指數", "Premium index"):
                fig.add_trace(go.Scatter(x=dz.index, y=dz["premium"] * 1e4, name=L("溢價 (bp)", "Premium (bp)"), line=dict(color=C.CYAN, width=1.2),
                                         fill="tozeroy", fillcolor="rgba(34,195,214,.08)"), row, 1)
            fig.update_yaxes(title_text=sp, title_font=dict(size=11), row=row, col=1)
            row += 1
        fig.update_layout(xaxis_rangeslider_visible=False)
        if log_y:
            fig.update_yaxes(type="log", row=1, col=1)
        C.style(fig, 520 + 140 * len(subs), right_axis=True)
        C.show(fig)
        st.caption(L("時間皆為 UTC。▲／▼ 為模型在該決策日（00:00 UTC 出訊號、01:00 開盤進場）把此幣排入多／空前 3 名。", "All times UTC. ▲/▼ mark decision dates (signal at 00:00 UTC, entry at the 01:00 open) when the model ranked this coin in its long/short top 3."))

        with st.expander(L("這檔幣的模型訊號紀錄", "Model signal history for this coin"), icon=":material/history:"):
            sf = C.signal_frame().dropna(subset=[model])
            sf = sf[sf["tradable"]]
            allr = sf.groupby("date")[model].rank(ascending=False, method="first")
            mine = sf[sf["symbol"] == sym].copy()
            mine[L("排名", "Rank")] = allr.loc[mine.index].astype(int)
            cnt = sf.groupby("date")[model].count()
            mine[L("檔數", "Names")] = mine["date"].map(cnt)
            mine[L("動作", "Action")] = np.where(mine[L("排名", "Rank")] <= 3, L("做多", "Long"), np.where(mine[L("排名", "Rank")] > mine[L("檔數", "Names")] - 3, L("做空", "Short"), "—"))
            mine = mine[(mine["date"] >= start) & (mine["date"] <= end)].sort_values("date", ascending=False)
            st.dataframe(pd.DataFrame({
                L("決策日", "Decision date"): mine["date"].dt.strftime("%Y-%m-%d"), L("模型分數", "Model score"): mine[model], L("排名", "Rank"): mine[L("排名", "Rank")], L("檔數", "Names"): mine[L("檔數", "Names")],
                L("動作", "Action"): mine[L("動作", "Action")], L("擁擠度 z", "Crowding z"): mine["funding_z_30d_raw"], L("隔日報酬（含資金費）", "Next-day return (incl. funding)"): mine["ret_long_simple"] * 100,
            }), use_container_width=True, hide_index=True, height=360,
                column_config={L("模型分數", "Model score"): st.column_config.NumberColumn(format="%.4f"),
                               L("擁擠度 z", "Crowding z"): st.column_config.NumberColumn(format="%+.2f"),
                               L("隔日報酬（含資金費）", "Next-day return (incl. funding)"): st.column_config.NumberColumn(format="%+.2f%%")})

    with tab_cmp:
        c1, c2 = st.columns([3, 2])
        pick = c1.multiselect(L("比較幣種", "Symbols"), C.SYMS, default=["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"],
                              format_func=lambda s: s.replace("USDT", ""), max_selections=8)
        metric = c2.segmented_control(L("比較項目", "Metric"), [L("累積報酬", "Cumulative return"), L("擁擠度 z", "Crowding z"), L("年化資金費", "Funding (ann.)")], default=L("累積報酬", "Cumulative return")) or L("累積報酬", "Cumulative return")
        d0, d1 = d["day"].min(), d["day"].max()
        start, end = C.date_range_picker("cmp", d0, d1, default="1Y")
        sub = d[(d["symbol"].isin(pick)) & (d["day"] >= start) & (d["day"] <= end)]
        fig = go.Figure()
        stats = []
        for i, sname in enumerate(pick):
            x = sub[sub["symbol"] == sname].set_index("day")
            if x.empty:
                continue
            if metric == L("累積報酬", "Cumulative return"):
                y = x["close"] / x["close"].iloc[0] - 1
            elif metric == L("擁擠度 z", "Crowding z"):
                y = x["funding_z"].rolling(7, min_periods=1).mean()
            else:
                y = x["funding_ann"].rolling(7, min_periods=1).mean()
            fig.add_trace(go.Scatter(x=y.index, y=y, name=sname.replace("USDT", ""), line=dict(width=1.6)))
            r = np.log(x["close"]).diff()
            stats.append({L("幣種", "Symbol"): sname.replace("USDT", ""), L("區間報酬", "Period return"): (x["close"].iloc[-1] / x["close"].iloc[0] - 1) * 100,
                          L("年化波動", "Ann. volatility"): r.std() * np.sqrt(365) * 100,
                          L("最大回撤", "Max drawdown"): ((x["close"] / x["close"].cummax()) - 1).min() * 100,
                          L("平均年化資金費", "Avg funding (ann.)"): x["funding_ann"].mean() * 100, L("平均擁擠度 z", "Avg crowding z"): x["funding_z"].mean()})
        if metric == L("累積報酬", "Cumulative return"):
            fig.update_yaxes(tickformat=".0%")
        elif metric == L("年化資金費", "Funding (ann.)"):
            fig.update_yaxes(tickformat=".0%")
        fig.update_layout(title=L(f"{metric}（{'7 日平滑' if metric != '累積報酬' else '以期初為 0'}）", f"{metric} ({'7-day smoothed' if metric != 'Cumulative return' else 'from 0 at start'})"))
        C.show(C.style(fig, 480))
        if stats:
            st.dataframe(pd.DataFrame(stats), use_container_width=True, hide_index=True, column_config={
                k: st.column_config.NumberColumn(format="%+.1f%%" if k != L("年化波動", "Ann. volatility") else "%.1f%%")
                for k in [L("區間報酬", "Period return"), L("年化波動", "Ann. volatility"), L("最大回撤", "Max drawdown"), L("平均年化資金費", "Avg funding (ann.)")]} | {
                L("平均擁擠度 z", "Avg crowding z"): st.column_config.NumberColumn(format="%+.2f")})


# ==================================================================== 策略：每日選股
def page_ranking():
    C.page_header(L("每日選股排行", "Daily Signal Ranking"), L("選一個決策日與模型，看當天 19 檔的模型分數排行、被選進的多空各 N 檔，以及隔天實際的報酬。", "Pick a decision date and a model to see that day's score ranking, the N longs and N shorts it picked, and what they actually returned the next day."))
    sf = C.signal_frame()
    dates = sf["date"].drop_duplicates().sort_values()
    ss.setdefault("rk_date", dates.iloc[-1].date())

    def shift(step):
        cur = dates[dates <= C.to_utc(ss.rk_date)]
        i = len(cur) - 1
        if step == 0:
            ss.rk_date = dates.sample(1).iloc[0].date()
        else:
            ss.rk_date = dates.iloc[int(np.clip(i + step, 0, len(dates) - 1))].date()

    c1, c2, c3, c4 = st.columns([1.2, 1.3, 1, 2.8])
    dt = C.to_utc(c1.date_input(L("決策日（00:00 UTC）", "Decision date (00:00 UTC)"), key="rk_date", min_value=dates.iloc[0].date(),
                                 max_value=dates.iloc[-1].date()))
    dt = dates[dates <= dt].iloc[-1]
    model = c2.selectbox(L("模型", "Model"), list(C.MODELS), index=list(C.MODELS).index(ss.model), format_func=C.mname, key="rk_model")
    k = c3.number_input(L("每邊檔數", "Names per side"), 2, 6, 3)
    with c4:
        st.write("")
        b1, b2, b3 = st.columns(3)
        b1.button(L("前一天", "Prev day"), icon=":material/chevron_left:", use_container_width=True, on_click=shift, args=(-1,))
        b2.button(L("後一天", "Next day"), icon=":material/chevron_right:", use_container_width=True, on_click=shift, args=(1,))
        b3.button(L("隨機", "Random"), icon=":material/shuffle:", use_container_width=True, on_click=shift, args=(0,))

    day = sf[(sf["date"] == dt) & sf["tradable"]].dropna(subset=[model]).sort_values(model, ascending=False).reset_index(drop=True)
    n = len(day)
    day[L("排名", "Rank")] = np.arange(1, n + 1)
    day["rk"] = day[L("排名", "Rank")]
    day[L("動作", "Action")] = np.where(day[L("排名", "Rank")] <= k, L("做多", "Long"), np.where(day[L("排名", "Rank")] > n - k, L("做空", "Short"), "—"))
    W = C.weights(model, int(k), "equal")
    w = W.loc[dt] if dt in W.index else pd.Series(dtype=float)
    pnl = float((w * day.set_index("symbol")["ret_long_simple"].reindex(w.index).fillna(0)).sum()) if len(w) else np.nan
    ic = day[[model, "ret_long_simple"]].corr(method="spearman").iloc[0, 1] if n > 4 else np.nan
    longs, shorts = day[day[L("動作", "Action")] == L("做多", "Long")], day[day[L("動作", "Action")] == L("做空", "Short")]
    C.kpi_row([
        (L("決策日", "Decision date"), f"{dt:%Y-%m-%d}", f"fold {int(day['fold'].iloc[0]) if n else '—'}", ""),
        (L("可交易檔數", "Tradable names"), f"{n}", L("少於 8 檔不交易", "No trade below 8 names"), ""),
        (L("當日組合報酬（未扣成本）", "Portfolio return (before costs)"), C.pct(pnl), L(f"多 {k} 空 {k} 等權", f"Long {k} / short {k}, equal weight"), C.tone(pnl)),
        (L("多頭平均", "Long book avg"), C.pct(longs["ret_long_simple"].mean()), L("24h，含資金費", "24h, incl. funding"), C.tone(longs["ret_long_simple"].mean())),
        (L("空頭平均（做空所得）", "Short book avg (short P&L)"), C.pct(-shorts["ret_long_simple"].mean()), L("24h，含資金費", "24h, incl. funding"), C.tone(-shorts["ret_long_simple"].mean())),
        (L("當日 IC", "Daily IC"), f"{ic:+.3f}" if np.isfinite(ic) else "—", L("分數與實際報酬的等級相關", "Rank correlation of score vs realised return"), C.tone(ic)),
    ])
    up, down = C.colors()
    left, right = st.columns([1.5, 1])
    with left:
        b = day.iloc[::-1]
        colr = np.where(b[L("動作", "Action")] == L("做多", "Long"), up, np.where(b[L("動作", "Action")] == L("做空", "Short"), down, "#3a414c"))
        fig = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[.55, .45], horizontal_spacing=.03,
                            subplot_titles=(L(f"{C.mname(model)} 分數", f"{C.mname(model)} score"), L("隔天實際報酬（含資金費）", "Next-day return (incl. funding)")))
        fig.add_trace(go.Bar(y=b["symbol"].str.replace("USDT", ""), x=b[model], orientation="h", marker_color=colr, name=L("分數", "Score"),
                             hovertemplate="%{y}: %{x:.4f}<extra></extra>"), 1, 1)
        fig.add_trace(go.Bar(y=b["symbol"].str.replace("USDT", ""), x=b["ret_long_simple"], orientation="h", name=L("實際報酬", "Realised return"),
                             marker_color=np.where(b["ret_long_simple"] > 0, up, down), opacity=.85,
                             hovertemplate="%{y}: %{x:+.2%}<extra></extra>"), 1, 2)
        fig.update_xaxes(tickformat="+.1%", row=1, col=2)
        C.show(C.style(fig, 40 * n + 90, legend=False))
    with right:
        for title, part, sign in [(L('<span class="sq up-bg"></span>做多清單', '<span class="sq up-bg"></span>Long book'), longs, 1), (L('<span class="sq down-bg"></span>做空清單', '<span class="sq down-bg"></span>Short book'), shorts.iloc[::-1], -1)]:
            rows = "".join(
                f'<div class="pos-row"><span><b>{r.symbol.replace("USDT", "")}</b> <span class="muted">#{r.rk}</span></span>'
                f'<span class="muted">z {r.funding_z_30d_raw:+.2f}</span>'
                f'<span class="muted">{L("資金費", "funding")} {r.funding_day * 1e4:+.1f}bp</span>'
                f'<span class="{C.tone(sign * r.ret_long_simple)}">{C.pct(sign * r.ret_long_simple)}</span></div>'
                for r in part.itertuples())
            st.markdown(f'<div class="panel"><h4>{title}</h4>{rows or L("<span class=muted>無</span>", "<span class=muted>none</span>")}</div>', unsafe_allow_html=True)
        st.caption(L("報酬欄是『這個部位』賺的：做空那欄已經把正負號反過來。z＝擁擠度（資金費 30 日 z 分數）。", "Returns are what each position earned (the short side is already sign-flipped). z = crowding (30-day funding z-score)."))

    st.markdown(L("#### 近 60 個決策日的模型排名", "#### Model rank over the last 60 decision dates"))
    lo = dt - pd.Timedelta(days=60)
    hist = sf[(sf["date"] > lo) & (sf["date"] <= dt) & sf["tradable"]].copy()
    hist["rk"] = hist.groupby("date")[model].rank(ascending=False, method="first")
    m = hist.pivot(index="symbol", columns="date", values="rk")
    m.index = m.index.str.replace("USDT", "")
    fig = go.Figure(go.Heatmap(z=m.values, x=m.columns, y=m.index, colorscale=[[0, up], [.5, "#1c222b"], [1, down]],
                               colorbar=dict(title=L("排名", "Rank"), thickness=10),
                               hovertemplate=L("%{y}<br>%{x|%Y-%m-%d}<br>第 %{z:.0f} 名<extra></extra>", "%{y}<br>%{x|%Y-%m-%d}<br>rank %{z:.0f}<extra></extra>")))
    fig.update_layout(hovermode="closest")
    C.show(C.style(fig, 430, legend=False))


# ==================================================================== 策略：回測實驗室
def page_backtest():
    C.page_header(L("回測實驗室", "Backtest Lab"), L("用 2021–2024 walk-forward 驗證期的模型預測重新回測。預設值＝報告設定（多 3 空 3、等權、每邊 7 bp）。", "Re-run the backtest on the 2021–2024 walk-forward predictions. Defaults = the report's setup (long 3 / short 3, equal weight, 7 bp per side)."))
    base = C.bt_base()
    d0, d1 = base["date"].min(), base["date"].max()
    with st.form("bt_form", border=True):
        c1, c2, c3, c4 = st.columns(4)
        model = c1.selectbox(L("模型", "Model"), list(C.MODELS), index=list(C.MODELS).index(ss.model), format_func=C.mname)
        k = c2.slider(L("每邊檔數", "Names per side"), 2, 6, 3)
        alloc = c3.radio(L("配置方式", "Allocation"), ["equal", "inv_vol"], format_func={"equal": L("等權", "Equal weight"), "inv_vol": L("反波動加權", "Inverse vol")}.get, horizontal=True)
        cost = c4.slider(L("每邊交易成本（bp）", "Cost per side (bp)"), 0.0, 30.0, float(C.COST_BPS), 0.5)
        c5, c6, c7, c8 = st.columns(4)
        dr = c5.date_input(L("回測期間", "Backtest period"), value=(d0.date(), d1.date()), min_value=d0.date(), max_value=d1.date())
        vt_on = c6.toggle(L("波動目標", "Vol target"), value=False)
        vt = c6.slider(L("目標年化波動", "Target annual vol"), 0.05, 0.60, 0.20, 0.05, disabled=False, label_visibility="collapsed")
        ladder = c7.toggle(L("回撤停損階梯", "Drawdown ladder"), value=False, help=L("回撤 10% 減半、15% 停止（依基礎策略的回撤）", "Halve at 10% drawdown, stop at 15% (on the base strategy's drawdown)"))
        bench = c8.multiselect(L("比較基準", "Benchmarks"), [L("規則基準", "Rule baseline"), L("等權做多全部", "Equal-weight long all"), L("BTC 做多", "BTC long")], default=[L("規則基準", "Rule baseline"), L("等權做多全部", "Equal-weight long all")])
        run = st.form_submit_button(L("執行回測", "Run backtest"), icon=":material/play_arrow:", type="primary", use_container_width=True)
    ss.model = model
    start, end = (C.to_utc(dr[0]), C.to_utc(dr[1])) if isinstance(dr, (tuple, list)) and len(dr) == 2 else (d0, d1)
    bt = C.backtest(model, int(k), alloc, float(cost), start, end, float(vt) if vt_on else None, bool(ladder))
    net = bt["net"]
    p = C.perf(net, bt["turnover"])
    g_sc, t_sc = bt["gross"] * bt["scale"], bt["turnover"] * bt["scale"]
    be = float(g_sc.sum() / t_sc.sum() * 1e4) if t_sc.sum() else np.nan
    gross_sr = C.perf(g_sc)["sharpe"]
    C.kpi_row([
        (L("累積報酬", "Cumulative return"), C.pct(p["total"], 1), L(f"{p['days']} 天", f"{p['days']} days"), C.tone(p["total"])),
        (L("年化報酬", "Ann. return"), C.pct(p["cagr"], 1), L(f"年化波動 {p['vol']:.1%}", f"Ann. vol {p['vol']:.1%}"), C.tone(p["cagr"])),
        (L("淨 Sharpe", "Net Sharpe"), f"{p['sharpe']:.2f}", f"95% CI [{p['sharpe_lo']:.2f}, {p['sharpe_hi']:.2f}]", C.tone(p["sharpe"])),
        (L("成本前 Sharpe", "Gross Sharpe"), f"{gross_sr:.2f}", L("未扣交易成本", "Before trading costs"), C.tone(gross_sr)),
        (L("最大回撤", "Max drawdown"), C.pct(p["max_dd"], 1, sign=False), f"Calmar {p['calmar']:.2f}" if np.isfinite(p["calmar"]) else "", "down"),
        (L("損益兩平成本", "Breakeven cost"), f"{be:.1f} bp", L("舒適區 > 2 倍成本", "Comfort zone > 2x cost") if be > 2 * cost else (L("灰色地帶", "Grey zone") if be > cost else L("低於目前成本", "Below current cost")),
         "up" if be > 2 * cost else ("" if be > cost else "down")),
        (L("日勝率", "Daily hit rate"), C.pct(p["hit"], 1, sign=False), L(f"最佳 {p['best']:+.1%}／最差 {p['worst']:+.1%}", f"Best {p['best']:+.1%} / worst {p['worst']:+.1%}"), ""),
        (L("年化換手", "Ann. turnover"), f"{p['ann_turnover']:.0f}×", L(f"平均每天 {bt['n_names'].mean():.1f} 檔", f"{bt['n_names'].mean():.1f} names per day"), ""),
    ])
    curves = {L("策略", "Strategy"): net}
    if L("規則基準", "Rule baseline") in bench and model != "rule":
        rb = C.backtest("rule", 3, "equal", float(cost), start, end, None, False)
        curves[L("規則基準", "Rule baseline")] = rb["net"]
    bm = C.benchmarks(start, end)
    for b_, col in ((L("等權做多全部", "Equal-weight long all"), "ew"), (L("BTC 做多", "BTC long"), "btc")):
        if b_ in bench:
            curves[b_] = bm[col].reindex(net.index).fillna(0)
    up, down = C.colors()
    tabs = st.tabs([L("權益曲線", "Equity curve"), L("月報酬", "Monthly returns"), L("成本敏感度", "Cost sensitivity"), L("損益拆解", "P&L attribution"), L("滾動指標", "Rolling metrics"), L("每日持倉", "Daily positions")])
    with tabs[0]:
        log_y = st.toggle(L("對數座標", "Log scale"), value=True, key="bt_log")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.72, .28], vertical_spacing=.03)
        pal = [C.ACCENT, C.PURPLE, C.CYAN, C.BLUE]
        for i, (name, r) in enumerate(curves.items()):
            eq = (1 + r).cumprod()
            fig.add_trace(go.Scatter(x=eq.index, y=eq, name=name, line=dict(width=2.2 if i == 0 else 1.3, color=pal[i % 4])), 1, 1)
        eq = (1 + net).cumprod()
        ddn = eq / eq.cummax() - 1
        fig.add_trace(go.Scatter(x=ddn.index, y=ddn, name=L("策略回撤", "Strategy drawdown"), fill="tozeroy", line=dict(color=down, width=1),
                                 fillcolor="rgba(246,70,93,.18)"), 2, 1)
        if log_y:
            fig.update_yaxes(type="log", row=1, col=1)
        fig.update_yaxes(tickformat=".0%", row=2, col=1)
        fig.update_yaxes(title_text=L("淨值（期初 = 1）", "NAV (start = 1)"), row=1, col=1)
        C.show(C.style(fig, 600))
        rows = []
        for name, r in curves.items():
            q = C.perf(r)
            rows.append({L("策略", "Strategy"): name, L("累積報酬", "Cumulative return"): q["total"] * 100, L("年化報酬", "Ann. return"): q["cagr"] * 100, L("年化波動", "Ann. volatility"): q["vol"] * 100,
                         "Sharpe": q["sharpe"], L("最大回撤", "Max drawdown"): q["max_dd"] * 100, "Calmar": q["calmar"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, column_config={
            **{c: st.column_config.NumberColumn(format="%+.1f%%") for c in [L("累積報酬", "Cumulative return"), L("年化報酬", "Ann. return"), L("最大回撤", "Max drawdown")]},
            L("年化波動", "Ann. volatility"): st.column_config.NumberColumn(format="%.1f%%"),
            "Sharpe": st.column_config.NumberColumn(format="%.2f"), "Calmar": st.column_config.NumberColumn(format="%.2f")})
    with tabs[1]:
        t = C.monthly_table(net)
        z = t.values * 100
        fig = go.Figure(go.Heatmap(z=z, x=t.columns, y=[str(y) for y in t.index], colorscale=[[0, down], [.5, "#1c222b"], [1, up]],
                                   zmid=0, text=np.where(np.isnan(z), "", np.vectorize(lambda v: f"{v:+.1f}%")(np.nan_to_num(z))),
                                   texttemplate="%{text}", hovertemplate="%{y} %{x}: %{z:+.2f}%<extra></extra>",
                                   colorbar=dict(title="%", thickness=10)))
        fig.update_layout(title=L("月報酬（扣成本後）", "Monthly returns (net of costs)"), hovermode="closest")
        fig.update_yaxes(autorange="reversed")
        C.show(C.style(fig, 90 + 60 * len(t), legend=False))
        yr = t[L("全年", "Year")]
        fig = go.Figure(go.Bar(x=[str(y) for y in yr.index], y=yr, marker_color=np.where(yr > 0, up, down),
                               text=yr.map("{:+.1%}".format), textposition="outside"))
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(title=L("年度報酬", "Annual returns"))
        C.show(C.style(fig, 330, legend=False))
    with tabs[2]:
        grid = [0, 2, 4, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40]
        sr = [C.perf(bt["gross"] * bt["scale"] - bt["turnover"] * bt["scale"] * c / 1e4)["sharpe"] for c in grid]
        fig = go.Figure(go.Scatter(x=grid, y=sr, mode="lines+markers", name=L("淨 Sharpe", "Net Sharpe"), line=dict(color=C.ACCENT, width=2)))
        fig.add_hline(y=0, line_color=C.MUTED, line_width=1)
        fig.add_vline(x=cost, line_dash="dash", line_color=C.BLUE, annotation_text=L(f"目前 {cost:g} bp", f"current {cost:g} bp"))
        if np.isfinite(be) and be > 0:
            fig.add_vline(x=be, line_dash="dot", line_color=down, annotation_text=L(f"損益兩平 {be:.1f} bp", f"breakeven {be:.1f} bp"))
        fig.update_layout(title=L("每邊成本 vs 淨 Sharpe", "Cost per side vs net Sharpe"), xaxis_title=L("每邊成本（bp）", "Cost per side (bp)"), yaxis_title=L("淨 Sharpe", "Net Sharpe"))
        C.show(C.style(fig, 430, legend=False))
        st.caption(L("損益兩平成本＝成本前總報酬 ÷ 總換手。報告用的分區：< 1 倍成本＝硬底線、1–2 倍＝灰色地帶、> 2 倍＝舒適區。", "Breakeven cost = gross return ÷ total turnover. Report zones: < 1x cost = hard floor, 1–2x = grey zone, > 2x = comfort zone."))
    with tabs[3]:
        sc = bt["scale"]
        fund = bt["funding_pnl"] * sc
        price = bt["gross"] * sc - fund
        costs = -(net - bt["gross"] * sc)
        fig = go.Figure()
        for name, s_, colr in [(L("價格損益", "Price P&L"), price, C.BLUE), (L("資金費收付", "Funding P&L"), fund, C.ACCENT), (L("交易成本", "Trading costs"), -costs, down), (L("淨損益", "Net P&L"), net, up)]:
            fig.add_trace(go.Scatter(x=s_.index, y=s_.cumsum(), name=name, line=dict(width=2 if name == L("淨損益", "Net P&L") else 1.4, color=colr)))
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(title=L("累積損益拆解（簡單加總）", "Cumulative P&L attribution (simple sum)"))
        C.show(C.style(fig, 470))
        C.kpi_row([(L("價格損益（年化）", "Price P&L (ann.)"), C.pct(price.mean() * 365, 1), "", C.tone(price.mean())),
                   (L("資金費（年化）", "Funding (ann.)"), C.pct(fund.mean() * 365, 1), L("空擁擠方收取的補償", "Paid by the crowded side"), C.tone(fund.mean())),
                   (L("交易成本（年化）", "Trading costs (ann.)"), C.pct(-costs.mean() * 365, 1), "", "down")])
    with tabs[4]:
        win = st.segmented_control(L("滾動視窗", "Rolling window"), [30, 90, 180, 365], default=90, format_func=lambda x: L(f"{x} 天", f"{x} days")) or 90
        rs = net.rolling(win).mean() / net.rolling(win).std() * np.sqrt(365)
        rv = net.rolling(win).std() * np.sqrt(365)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.05, subplot_titles=(L(f"滾動 {win} 天 Sharpe", f"Rolling {win}-day Sharpe"), L(f"滾動 {win} 天年化波動", f"Rolling {win}-day ann. vol")))
        fig.add_trace(go.Scatter(x=rs.index, y=rs, name="Sharpe", line=dict(color=C.ACCENT)), 1, 1)
        fig.add_hline(y=0, line_color=C.MUTED, row=1, col=1)
        fig.add_trace(go.Scatter(x=rv.index, y=rv, name=L("波動", "Volatility"), line=dict(color=C.PURPLE)), 2, 1)
        fig.update_yaxes(tickformat=".0%", row=2, col=1)
        C.show(C.style(fig, 520))
    with tabs[5]:
        W = C.weights(model, int(k), alloc)
        W = W[(W.index >= start) & (W.index <= end)]
        def names(row, side):
            x = row[row > 0] if side > 0 else row[row < 0]
            return "、".join(f"{s.replace('USDT', '')}({abs(v):.0%})" for s, v in x.sort_values(ascending=side < 0).items())
        log = pd.DataFrame({L("決策日", "Decision date"): W.index.strftime("%Y-%m-%d"),
                            L("做多", "Long"): [names(r, 1) for _, r in W.iterrows()],
                            L("做空", "Short"): [names(r, -1) for _, r in W.iterrows()],
                            L("曝險倍數", "Exposure"): bt["scale"].reindex(W.index).values,
                            L("成本前報酬", "Gross return"): bt["gross"].reindex(W.index).values * 100,
                            L("換手", "Turnover"): bt["turnover"].reindex(W.index).values,
                            L("淨報酬", "Net return"): net.reindex(W.index).values * 100}).iloc[::-1]
        st.dataframe(log, hide_index=True, use_container_width=True, height=480, column_config={
            L("成本前報酬", "Gross return"): st.column_config.NumberColumn(format="%+.2f%%"), L("淨報酬", "Net return"): st.column_config.NumberColumn(format="%+.2f%%"),
            L("換手", "Turnover"): st.column_config.NumberColumn(format="%.2f"), L("曝險倍數", "Exposure"): st.column_config.NumberColumn(format="%.2f")})
        st.download_button(L("下載每日持倉與損益 CSV", "Download positions & P&L (CSV)"), log.to_csv(index=False).encode("utf-8-sig"), icon=":material/download:",
                           file_name=f"backtest_{model}_k{k}_{cost:g}bp.csv", mime="text/csv")
    st.caption(L("所有參數組合都只是『事後觀察』，不是新的研究試驗；報告的正式結論以凍結設定（setup.yaml v1）為準。", "Any setting here is an after-the-fact look, not a new research trial; the report's conclusions rest on the frozen setup (setup.yaml v1)."))


# ==================================================================== 策略：即時訊號
def page_live():
    C.page_header(L("即時訊號（Shadow mode）", "Live Signal (shadow mode)"), L("從交易所公開 API 抓目前的資金費率與溢價（Binance → Bybit → Hyperliquid 依序嘗試），依規則模型排擁擠度。只觀察、不下單。", "Pulls current funding and premium from public exchange APIs (Binance → Bybit → Hyperliquid, in that order) and ranks crowding with the rule model. Observe only, no orders."))
    c1, c2 = st.columns([1, 4])
    if c1.button(L("重新整理", "Refresh"), icon=":material/refresh:", type="primary", use_container_width=True):
        C.live_quotes.clear()
    from app.live_signal import rank
    raw, src, errs = C.live_quotes()
    live = rank(raw, C.funding_raw()) if len(raw) else pd.DataFrame()
    t = live["fetched_utc"].iloc[0] if len(live) else None
    srcname = L("最近一次已結算資料", "last settled data") if src == "settled" else src
    if len(live):
        live["side"] = np.where(live["shadow_action"].str.startswith(L("做空", "Short")), "short",
                                np.where(live["shadow_action"].str.startswith(L("做多", "Long")), "long", ""))
        live["shadow_action"] = live["shadow_action"].map(C.tr)
    if src == "settled":
        c2.warning(L(f"目前連不到交易所即時 API（{'、'.join(errs)}），改顯示資料集最後一天（{t:%Y-%m-%d} 00:00 UTC）的已結算資金費。", f"Exchange APIs unreachable ({', '.join(errs)}); showing settled funding from the last day in the dataset ({t:%Y-%m-%d} 00:00 UTC)."))
    else:
        note = L(f"（{'、'.join(e.split(':')[0] for e in errs)} 連不到，已自動改用 {src}）", f"({', '.join(e.split(':')[0] for e in errs)} unreachable; switched to {src})") if errs else ""
        c2.success(L(f"{src} 即時資料 · {t:%Y-%m-%d %H:%M} UTC {note}", f"{src} live data · {t:%Y-%m-%d %H:%M} UTC {note}"))
    if live.empty:
        return
    up, down = C.colors()
    key = "funding_z_est" if "funding_z_est" in live else "lastFundingRate"
    sh, lg = live[live["side"] == "short"], live[live["side"] == "long"]
    C.kpi_row([
        (L("下次資金費結算", "Next funding"), f"{pd.to_datetime(live['next_funding']).min():%H:%M} UTC" if pd.to_datetime(live['next_funding']).notna().any() else "—", L(f"資料來源：{srcname}", f"Source: {srcname}"), ""),
        (L("平均預估資金費（每期）", "Avg est. funding (per period)"), f"{live['lastFundingRate'].mean() * 1e4:+.2f} bp", L(f"年化約 {live['lastFundingRate'].mean() * 3 * 365:+.1%}", f"≈ {live['lastFundingRate'].mean() * 3 * 365:+.1%} annualised"),
         C.tone(live['lastFundingRate'].mean())),
        (L("做空候選", "Short candidates"), "、".join(sh["symbol"].str.replace("USDT", "")), L("多方最擁擠", "Longs most crowded"), "down"),
        (L("做多候選", "Long candidates"), "、".join(lg["symbol"].str.replace("USDT", "")), L("空方最擁擠", "Shorts most crowded"), "up"),
    ])
    left, right = st.columns([1.2, 1])
    with left:
        b = live.sort_values(key)
        fig = go.Figure(go.Bar(x=b[key], y=b["symbol"].str.replace("USDT", ""), orientation="h",
                               marker_color=np.where(b["side"] == "short", down,
                                                     np.where(b["side"] == "long", up, "#3a414c"))))
        fig.update_layout(title=L("即時擁擠度（預估 z）", "Live crowding (est. z)") if key == "funding_z_est" else L("即時資金費率", "Live funding rate"))
        C.show(C.style(fig, 560, legend=False))
    with right:
        show = pd.DataFrame({L("排名", "Rank"): live["crowding_rank"], L("幣種", "Symbol"): live["symbol"].str.replace("USDT", ""),
                             L("標記價格", "Mark price"): live.get("markPrice", np.nan), L("預估資金費 (bp)", "Est. funding (bp)"): live["lastFundingRate"] * 1e4,
                             L("預估 z", "Est. z"): live.get("funding_z_est", np.nan), L("溢價 (bp)", "Premium (bp)"): live["premium"] * 1e4,
                             L("動作", "Action"): live["shadow_action"]})
        st.dataframe(show, hide_index=True, use_container_width=True, height=560, column_config={
            L("標記價格", "Mark price"): st.column_config.NumberColumn(format="%.4f"),
            L("預估資金費 (bp)", "Est. funding (bp)"): st.column_config.NumberColumn(format="%+.2f"),
            L("預估 z", "Est. z"): st.column_config.NumberColumn(format="%+.2f"), L("溢價 (bp)", "Premium (bp)"): st.column_config.NumberColumn(format="%+.1f"),
            L("動作", "Action"): st.column_config.TextColumn(width="medium")})
    st.caption(L("即時資金費是『預估值』，要到下次結算才確定；研究與回測只用已結算值。", "Live funding is an estimate until the next settlement; research and backtests use settled values only."))


# ==================================================================== 研究：模型診斷
def page_models():
    C.page_header(L("模型診斷", "Model Diagnostics"), L("規則 → Ridge → LightGBM 的模型階梯：16 個季度 walk-forward fold 的 IC 與信賴區間。", "Model ladder rule → Ridge → LightGBM: IC and confidence intervals over 16 quarterly walk-forward folds."))
    mc = C.J("model_comparison")
    s = pd.DataFrame(mc["summary"]).T
    C.kpi_row([(C.mname(m), f"IC {s.loc[m, 'ic']:+.3f}", f"95% CI [{s.loc[m, 'lo']:+.3f}, {s.loc[m, 'hi']:+.3f}] · ICIR {s.loc[m, 'icir']:.2f}",
                C.tone(s.loc[m, "ic"])) for m in ["rule", "ridge", "lgbm"]] +
              [(L("採用模型", "Chosen model"), C.mname(mc["chosen"]), L("複雜模型需成對比較 CI 下界 > 0", "A more complex model must beat the simpler one (paired CI lower bound > 0)"), "")])
    bf = pd.read_csv(C.OUT / "wf_ic_by_fold.csv")
    pick = st.multiselect(L("顯示模型", "Models"), list(C.MODELS), default=list(C.MODELS), format_func=C.mname)
    fig = go.Figure()
    pal = {"rule": C.CYAN, "ridge": C.PURPLE, "lgbm": C.ACCENT}
    for m in pick:
        fig.add_trace(go.Bar(x=bf["val_start"], y=bf[f"{m}_ic"], name=C.mname(m), marker_color=pal[m],
                             error_y=dict(type="data", symmetric=False, array=bf[f"{m}_hi"] - bf[f"{m}_ic"],
                                          arrayminus=bf[f"{m}_ic"] - bf[f"{m}_lo"], thickness=1, color="#8a93a0")))
    fig.add_hline(y=0, line_color=C.MUTED)
    fig.update_layout(barmode="group", title=L("各 fold 每日 IC 平均（Newey-West 95% CI）", "Mean daily IC per fold (Newey-West 95% CI)"), xaxis_title=L("驗證 fold 起始日", "Validation fold start"))
    C.show(C.style(fig, 440))
    c1, c2 = st.columns(2)
    with c1:
        r = C.out_parquet("rolling_ic").set_index("date")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=r.index, y=r["ic"], name=L("每日 IC", "Daily IC"), marker_color="#3a414c"))
        fig.add_trace(go.Scatter(x=r.index, y=r["roll_ic"], name=L("滾動 IC", "Rolling IC"), line=dict(color=C.ACCENT, width=2)))
        fig.add_hline(y=0, line_color=C.MUTED)
        fig.update_layout(title=L(f"{C.mname(mc['chosen'])} 滾動 IC", f"{C.mname(mc['chosen'])} rolling IC"))
        fig.update_yaxes(range=[-0.3, 0.3])
        C.show(C.style(fig, 400))
    with c2:
        st.markdown(L("##### 模型摘要", "##### Model summary"))
        st.dataframe(s[["ic", "lo", "hi", "t", "icir", "hit", "pos_fold_share"]].rename(
            index={k: C.mname(k) for k in C.MODELS}, columns={"ic": "IC", "lo": L("CI 下界", "CI low"), "hi": L("CI 上界", "CI high"), "t": L("t 值", "t-stat"), "icir": "ICIR", "hit": L("IC>0 天數比", "Share of days IC>0"), "pos_fold_share": L("正 IC fold 比", "Share of folds IC>0")}),
            use_container_width=True, column_config={c: st.column_config.NumberColumn(format="%+.3f") for c in ["IC", L("CI 下界", "CI low"), L("CI 上界", "CI high")]})
        st.markdown(L("##### 成對比較（IC 差）", "##### Paired comparison (IC difference)"))
        pr = pd.DataFrame(mc["paired"]).T
        st.dataframe(pr.rename(columns={"d_ic": L("IC 差", "IC diff"), "lo": L("CI 下界", "CI low"), "hi": L("CI 上界", "CI high")}), use_container_width=True,
                     column_config={c: st.column_config.NumberColumn(format="%+.4f") for c in [L("IC 差", "IC diff"), L("CI 下界", "CI low"), L("CI 上界", "CI high")]})
    t1, t2 = st.tabs([L("特徵初篩（triage）", "Feature triage"), L("SHAP 特徵重要度", "SHAP importance")])
    with t1:
        tr = pd.read_csv(C.OUT / "triage.csv")
        cols = [c for c in ["feature", "role", "ic", "lo", "hi", "icir", "pos_fold_share", "spread_bp_per_day", "verdict"] if c in tr]
        if "role" in tr:
            tr["role"] = tr["role"].map(C.tr)
        st.dataframe(tr[cols], use_container_width=True, hide_index=True)
    with t2:
        if (C.OUT / "fig_shap.png").exists():
            st.image(str(C.OUT / "fig_shap.png"), use_container_width=True)


# ==================================================================== 研究：風險
def page_risk():
    C.page_header(L("風險監控", "Risk Monitor"), L("採用模型的回撤、尾部風險、壓力事件與市場狀態切片。", "Drawdown, tail risk, stress events and market-regime slices for the chosen model."))
    rk = C.J("risk")
    r = C.out_parquet("risk_daily").set_index("date")
    dd = rk["drawdown"]
    v = next(iter(rk["var_cvar_95"].values()))   # first entry = whole sample
    C.kpi_row([(L("最大回撤", "Max drawdown"), C.pct(dd["depth"], 1, False), f"{dd['peak']} → {dd['trough']}", "down"),
               (L("回撤期間", "Drawdown length"), L(f"{dd['duration_days']} 天", f"{dd['duration_days']} days"), L(f"恢復：{C.tr(dd['recovered'])}", f"Recovery: {C.tr(dd['recovered'])}"), ""),
               (L("95% VaR（日）", "95% VaR (daily)"), C.pct(v[0], 2, False), "", "down"), (L("95% CVaR（日）", "95% CVaR (daily)"), C.pct(v[1], 2, False), "", "down"),
               (L("BTC beta（HAC）", "BTC beta (HAC)"), f"{rk['beta_hac']['beta']:+.3f}", f"[{rk['beta_hac']['lo']:+.3f}, {rk['beta_hac']['hi']:+.3f}]", "")])
    up, down = C.colors()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.65, .35], vertical_spacing=.05,
                        subplot_titles=(L("回撤與停損階梯", "Drawdown and stop ladder"), L("滾動 BTC beta", "Rolling BTC beta")))
    fig.add_trace(go.Scatter(x=r.index, y=r["dd"], fill="tozeroy", name=L("回撤", "Drawdown"), line=dict(color=down, width=1),
                             fillcolor="rgba(246,70,93,.18)"), 1, 1)
    for lv, lab in [(.05, L("警示", "Warn")), (.10, L("減半", "Halve")), (.15, L("停止", "Stop")), (.20, L("平倉", "Flat"))]:
        fig.add_hline(y=-lv, line_dash="dash", line_color="#5e6673", annotation_text=lab, row=1, col=1)
    fig.add_trace(go.Scatter(x=r.index, y=r["roll_beta"], name="beta", line=dict(color=C.CYAN)), 2, 1)
    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    C.show(C.style(fig, 560))
    c1, c2 = st.columns(2)
    with c1:
        vc = pd.DataFrame(rk["var_cvar_95"], index=["VaR", "CVaR"]).T
        vc.index = vc.index.map(C.tr)
        fig = go.Figure([go.Bar(x=vc.index, y=vc["VaR"], name="VaR 95%", marker_color=C.ACCENT),
                         go.Bar(x=vc.index, y=vc["CVaR"], name="CVaR 95%", marker_color=down)])
        fig.update_yaxes(tickformat=".1%")
        fig.update_layout(title=L("各波動狀態的每日 VaR / CVaR", "Daily VaR / CVaR by volatility regime"), barmode="group")
        C.show(C.style(fig, 380))
    with c2:
        rs = pd.DataFrame(rk["regime_slices"])
        fig = go.Figure(go.Bar(x=rs["regime"].map(C.tr), y=rs["sharpe"], marker_color=np.where(rs["sharpe"] > 0, up, down),
                               text=rs["days"].map(lambda x: L(f"{x} 天", f"{x} days")), textposition="outside"))
        fig.update_layout(title=L("市場狀態切片的 Sharpe", "Sharpe by market regime"))
        C.show(C.style(fig, 380, legend=False))
    st.markdown(L("##### 壓力測試", "##### Stress tests"))
    ev = st.segmented_control(L("事件", "Event"), list(rk["stress"]), default=list(rk["stress"])[0], format_func=C.tr) or list(rk["stress"])[0]
    x = rk["stress"][ev]
    names = {"total": L("期間累積", "Period total"), "worst_day": L("最差單日", "Worst day"), "btc": L("同期 BTC", "BTC same period"), "ann_ret": L("年化報酬", "Ann. return"), "vol": L("年化波動", "Ann. volatility"),
             "sharpe": "Sharpe", "max_dd": L("最大回撤", "Max drawdown"), "sharpe_lo": L("Sharpe CI 下界", "Sharpe CI low"), "sharpe_hi": L("Sharpe CI 上界", "Sharpe CI high"), "days": L("天數", "Days")}
    C.kpi_row([(names.get(k_, k_), f"{v_:.2f}" if "sharpe" in k_ else (f"{v_:,.0f}" if k_ == "days" else C.pct(v_, 1)), "",
                C.tone(v_) if k_ != "days" else "") for k_, v_ in x.items()])


# ==================================================================== 研究：2025 封存驗證
def page_holdout():
    C.page_header(L("2025 封存期驗證", "2025 Holdout Test"), L("2025 年資料在研究期間完全封存，規格凍結後只打開一次。這是策略真正的樣本外成績。", "2025 data was sealed during research and opened once after the spec was frozen. This is the strategy's true out-of-sample result."))
    ho = C.J("holdout")
    if not ho:
        st.info(L("封存期尚未打開。", "The holdout has not been opened."))
        return
    hd = C.out_parquet("holdout_daily").set_index("date")
    ch = ho["chosen"]
    t, ic = ho["table"][ch], ho["ic"][ch]
    C.kpi_row([(L("打開時間", "Opened at"), ho["opened_at"].replace("T", " "), L("只允許一次", "Opened once only"), ""),
               (L("淨 Sharpe", "Net Sharpe"), f"{t['sharpe']:.2f}", f"[{t['sharpe_lo']:.2f}, {t['sharpe_hi']:.2f}]", C.tone(t["sharpe"])),
               ("IC", f"{ic['ic']:+.3f}", f"[{ic['lo']:+.3f}, {ic['hi']:+.3f}]", C.tone(ic["ic"])),
               (L("累積報酬", "Cumulative return"), C.pct(t["total"], 1), L(f"{t['days']} 天", f"{t['days']} days"), C.tone(t["total"])),
               (L("最大回撤", "Max drawdown"), C.pct(t["max_dd"], 1, False), "", "down"),
               (L("確認檢定", "Confirmatory test"), L("通過", "Pass") if (t["sharpe"] > 0 and ic["lo"] > 0) else L("未通過", "Fail"), L("淨 Sharpe > 0 且 IC 下界 > 0", "Net Sharpe > 0 and IC lower bound > 0"),
                "up" if (t["sharpe"] > 0 and ic["lo"] > 0) else "down")])
    fig = go.Figure()
    pal = {"rule_net": C.CYAN, "ridge_net": C.PURPLE, "lgbm_net": C.ACCENT, "ew_long": "#5e6673"}
    lab = {"rule_net": L("規則", "Rule"), "ridge_net": "Ridge", "lgbm_net": "LightGBM", "ew_long": L("等權做多", "EW long")}
    for c in hd.columns:
        fig.add_trace(go.Scatter(x=hd.index, y=(1 + hd[c]).cumprod(), name=lab.get(c, c),
                                 line=dict(color=pal.get(c), width=2.2 if c == f"{ch}_net" else 1.3)))
    fig.update_layout(title=L("2025 年淨值（扣每邊 7 bp）", "2025 NAV (net of 7 bp per side)"))
    C.show(C.style(fig, 460))
    rows = []
    for m in ["rule", "ridge", "lgbm"]:
        a, b = ho["table"][m], ho["ic"][m]
        rows.append({L("模型", "Model"): C.mname(m), "IC": b["ic"], L("IC 下界", "IC low"): b["lo"], L("IC 上界", "IC high"): b["hi"], L("淨 Sharpe", "Net Sharpe"): a["sharpe"],
                     L("年化報酬", "Ann. return"): a["ann_ret"] * 100, L("最大回撤", "Max drawdown"): a["max_dd"] * 100, L("損益兩平 (bp)", "Breakeven (bp)"): a.get("breakeven_bps", np.nan)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, column_config={
        **{c: st.column_config.NumberColumn(format="%+.3f") for c in ["IC", L("IC 下界", "IC low"), L("IC 上界", "IC high")]},
        L("淨 Sharpe", "Net Sharpe"): st.column_config.NumberColumn(format="%.2f"),
        L("年化報酬", "Ann. return"): st.column_config.NumberColumn(format="%+.1f%%"), L("最大回撤", "Max drawdown"): st.column_config.NumberColumn(format="%.1f%%"),
        L("損益兩平 (bp)", "Breakeven (bp)"): st.column_config.NumberColumn(format="%.1f")})


# ==================================================================== 研究：規格
def page_spec():
    C.page_header(L("策略規格與資料健康", "Strategy Spec & Data Health"), L("回測前凍結的交易設定、驗證協定，以及資料時間點（PIT）檢查。", "The trading setup and validation protocol frozen before any backtest, plus point-in-time (PIT) data checks."))
    res = C.J("results")
    if res:
        f = res["falsified"]
        C.kpi_row([(L("採用模型", "Chosen model"), C.mname(res["chosen"]), "", ""), (L("驗證期結論", "Validation verdict"), C.tr(res["pattern_validation"]), "", ""),
                   (L("IC CI 含 0？", "IC CI contains 0?"), L("是", "Yes") if f["ic_ci_contains_0"] else L("否", "No"), L("否＝訊號顯著", "No = signal is significant"), "down" if f["ic_ci_contains_0"] else "up"),
                   (L("淨 Sharpe ≤ 0？", "Net Sharpe ≤ 0?"), L("是", "Yes") if f["net_sharpe_le_0"] else L("否", "No"), L("是＝扣成本後不賺", "Yes = unprofitable after costs"), "down" if f["net_sharpe_le_0"] else "up"),
                   ("Deflated Sharpe", f"{res['inference']['dsr']:.3f}", L(f"試驗數 {res['inference']['n_trials']}", f"{res['inference']['n_trials']} trials"), "")])
    t1, t2, t3 = st.tabs([L("setup.yaml（凍結設定）", "setup.yaml (frozen)"), L("驗證協定", "Validation protocol"), L("資料健康 / PIT", "Data health / PIT")])
    with t1:
        st.code((C.ROOT / "setup.yaml").read_text(encoding="utf-8"), language="yaml")
    with t2:
        md = L((C.ROOT / "protocol.md").read_text(encoding="utf-8"), PROTOCOL_EN)
        st.markdown(md.replace("# ", "#### ", 1).replace("\n## ", "\n##### "))
    with t3:
        if (C.OUT / "pit_table.csv").exists():
            st.dataframe(L(pd.read_csv(C.OUT / "pit_table.csv"), PIT_EN), use_container_width=True, hide_index=True)
        if (C.OUT / "pit_leakage_by_feature.csv").exists():
            st.markdown(L("Ch.02 PITValidator：每個特徵與次日報酬的最大相關（門檻 0.30）", "Ch.02 PITValidator: max correlation of each feature with next-day return (threshold 0.30)"))
            st.dataframe(pd.read_csv(C.OUT / "pit_leakage_by_feature.csv"), use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        for c, f in [(c1, "fig_knowledge_delay.png"), (c2, "fig_tradable_count.png")]:
            if (C.OUT / f).exists():
                c.image(str(C.OUT / f), use_container_width=True)


PROTOCOL_EN = """# Validation protocol (pre-committed, frozen 2026-10-26, before any backtest)

| Design item | Our choice |
|---|---|
| Training window | Expanding, from 2020-01-01, at least 365 days |
| Step | 1 calendar quarter |
| Retraining | Once per fold (quarterly) |
| Label horizon | 1 day (d 01:00 → d+1 01:00, incl. funding) |
| Longest feature lookback | 30 days |
| Validation period | 2021-01-01 to 2024-12-31, 16 folds (2021Q1 … 2024Q4) |
| Holdout | 2025-01-01 to 2025-12-31, sealed; opened once after the spec is frozen, with the owner's approval |
| Purging | Drop 1 day (= label horizon) at the end of training; crypto trades 24/7 so trading days = calendar days |
| Embargo | Not needed (training always precedes validation) |
| Tuning | Nested: last 20% of each training window is inner validation (also purged 1 day) |
| Primary metric | Daily Spearman IC with Newey–West 95% CI; a signal counts only if the CI excludes 0 |
| Model ladder rule | Adopt the more complex model only if its daily IC gain has a HAC CI lower bound > 0; otherwise keep the simpler one |
| Falsification | (1) the chosen model's validation IC CI contains 0; (2) net Sharpe ≤ 0 at 7 bp per side. Either one → stop and report honestly |

Fold dates come from `src/cv.py`, saved to `outputs/folds.csv` and logged in `run_log.csv`.

## Sealing

- `src/config.research_only()`: the first call after reading any data; drops every row on or after 2025-01-01 and asserts it.
- `data/build_panel.py` writes the 2025 rows to `data/processed/sealed/`, which research scripts never read.
- `08_holdout.py` writes `outputs/HOLDOUT_OPENED.lock` before reading the sealed data and refuses a second run.

## Ledger

Every look at results that could change a later choice is logged in `run_log.csv` (`counts_as_trial=True`, `n_configs` = settings tried). Bug fixes are not trials. The Deflated Sharpe trial count N = sum of `n_configs` over counted rows.
"""

PIT_EN = pd.DataFrame({
    "level": ["Values", "Universe", "Symbols", "Dates"],
    "check": ["Every feature available_at <= 00:00 UTC; settled funding only; Ch.02 PITValidator leakage check",
              "Eligibility uses data before t only; delisted coins included",
              "Contract renames and relistings (e.g. 1000SHIB)",
              "PITValidator: monotonic dates, gaps > 5 days"],
    "result": ["PASS", "FAIL (disclosed)", "N/A", "PASS"],
    "note": ["Funding knowledge delay 0–0 h, 0 negative values; the control (tomorrow's return) is caught = True",
             "Listing age and 30-day volume use pre-t data only, but the data holds surviving contracts only (no LUNA, FTT…), so returns are biased upward",
             "All 19 use original tickers, no 1000x or renames; checked symbol by symbol",
             "480 missing 1h bars in total; no trading on a day with a missing bar"],
})


# ==================================================================== shell
PAGES = {
    "market": st.Page(page_market, title=L("市場總覽", "Market Overview"), icon=":material/dashboard:", url_path="market", default=True),
    "chart": st.Page(page_chart, title=L("K 線與比較", "Charts & Compare"), icon=":material/candlestick_chart:", url_path="chart"),
    "ranking": st.Page(page_ranking, title=L("每日選股排行", "Daily Signal Ranking"), icon=":material/leaderboard:", url_path="ranking"),
    "backtest": st.Page(page_backtest, title=L("回測實驗室", "Backtest Lab"), icon=":material/science:", url_path="backtest"),
    "live": st.Page(page_live, title=L("即時訊號", "Live Signal"), icon=":material/sensors:", url_path="live"),
    "models": st.Page(page_models, title=L("模型診斷", "Model Diagnostics"), icon=":material/insights:", url_path="models"),
    "risk": st.Page(page_risk, title=L("風險監控", "Risk Monitor"), icon=":material/shield:", url_path="risk"),
    "holdout": st.Page(page_holdout, title=L("2025 封存驗證", "2025 Holdout"), icon=":material/lock:", url_path="holdout"),
    "spec": st.Page(page_spec, title=L("策略規格", "Strategy Spec"), icon=":material/description:", url_path="spec"),
}

nav = st.navigation({L("行情", "Markets"): [PAGES["market"], PAGES["chart"]],
                     L("策略", "Strategy"): [PAGES["ranking"], PAGES["backtest"], PAGES["live"]],
                     L("研究", "Research"): [PAGES["models"], PAGES["risk"], PAGES["holdout"], PAGES["spec"]]})
with st.sidebar:
    st.divider()
    st.segmented_control("Language / 語言", ["EN", "中文"], key="lang")
    ss.color_mode = st.radio(L("漲跌顏色", "Up/down colours"), ["green_up", "red_up"], index=["green_up", "red_up"].index(ss.color_mode),
                             format_func=lambda m: {"green_up": "綠漲紅跌", "red_up": "紅漲綠跌"}[m] if ss.lang == "中文"
                             else {"green_up": "Green up", "red_up": "Red up"}[m],
                             horizontal=True, help=L("國際/幣圈慣用綠漲紅跌；台股慣用紅漲綠跌", "Crypto/global convention is green up; Taiwan stocks use red up"))
    res = C.J("results")
    bt_ = C.J("backtest")
    OPENED, SEALED = L("已開啟（一次）", "opened (once)"), L("封存中", "sealed")
    st.markdown(
        f'<div class="panel" style="font-size:.8rem;line-height:1.7">'
        f'<b>{L("系統狀態", "System status")}</b><br>'
        f'<span class="muted">{L("幣種", "Universe")}</span> {len(C.SYMS)} {L("檔 USDT 永續", "USDT perps")}<br>'
        f'<span class="muted">{L("行情資料", "Market data")}</span> 2020-01 ～ 2025-12<br>'
        f'<span class="muted">{L("驗證期", "Validation")}</span> {bt_["window"][0]} ～ {bt_["window"][1]}<br>'
        f'<span class="muted">{L("採用模型", "Chosen model")}</span> {C.mname(res["chosen"]) if res else "—"}<br>'
        f'<span class="muted">{L("封存期 2025", "2025 holdout")}</span> {OPENED if (C.OUT / "HOLDOUT_OPENED.lock").exists() else SEALED}<br>'
        f'<span class="muted">{L("設定版本", "Setup")}</span> {C.SETUP["setup_version"]}{L("（", " (frozen ")}{C.SETUP["frozen_at"]}{L(" 凍結）", ")")}</div>',
        unsafe_allow_html=True)
    st.caption(L("前端只讀已存好的資料與結果，不訓練模型。", "Reads saved data and results only; no model training."))

C.inject_css()
st.markdown('<div class="topbar"><div class="brand"><span class="dot"></span>Crowding Terminal'
            f'<span class="brand-sub">{L("加密永續合約 · 擁擠度反轉交易系統", "Crypto perpetuals · crowding-reversal trading system")}</span></div>'
            f'<div class="brand-sub">Binance USDT-M · {L("時間皆為 UTC", "all times UTC")}</div></div>', unsafe_allow_html=True)
ticker_tape()
nav.run()
