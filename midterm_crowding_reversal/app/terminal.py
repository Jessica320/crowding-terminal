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

st.set_page_config(page_title="Crowding Terminal｜擁擠度反轉交易系統", page_icon=":material/candlestick_chart:", layout="wide",
                   initial_sidebar_state="expanded")
ss = st.session_state
ss.setdefault("sym", "BTCUSDT")
ss.setdefault("color_mode", "綠漲紅跌")
ss.setdefault("model", (C.J("results") or {}).get("chosen", "lgbm"))
ss.setdefault("scr_n", 0)


def ticker_tape():
    d = C.daily_market()
    last = d.sort_values("day").groupby("symbol").tail(1)
    items = "".join(
        f'<span class="item"><b>{r.symbol.replace("USDT", "")}</b>{C.fmt_px(r.close)} '
        f'<span class="{C.tone(r.ret_1d)}">{C.pct(r.ret_1d)}</span> '
        f'<span class="muted">· 資金費 {r.funding_ann:+.1%}/年</span></span>'
        for r in last.itertuples())
    st.markdown(f'<div class="tape"><div class="tape-inner">{items}{items}</div></div>', unsafe_allow_html=True)


def badge(action: str) -> str:
    cls = {"做多": "long", "做空": "short"}.get(action[:2], "neutral")
    return f'<span class="badge {cls}">{action}</span>'


# ==================================================================== 行情：市場總覽
def page_market():
    C.page_header("市場總覽", "19 檔 Binance USDT 永續合約的報價、資金費率與擁擠度。點表格任一列可直接打開該幣的 K 線。")
    d = C.daily_market()
    days = d["day"].drop_duplicates().sort_values()
    c1, c2, c3, c4 = st.columns([1.2, 1.4, 1.6, 1.2])
    asof = C.to_utc(c1.date_input("資料日期（UTC）", value=days.iloc[-1].date(),
                                   min_value=days.iloc[30].date(), max_value=days.iloc[-1].date()))
    asof = days[days <= asof].iloc[-1]
    view = c2.segmented_control("篩選", ["全部", "多方擁擠", "空方擁擠"], default="全部") or "全部"
    q = c3.text_input("搜尋幣種", placeholder="例如 BTC、SOL…").strip().upper()
    sort_by = c4.selectbox("排序", ["擁擠度 z", "24h 漲跌", "7 日漲跌", "30 日漲跌", "年化資金費", "30 日均成交額"])

    today = d[(d["day"] == asof) & (d["quote_volume"] > 0)].copy()   # drops coins delisted or halted that day
    today["rank"] = today["funding_z"].rank(ascending=False, method="first")
    n = today["funding_z"].notna().sum()
    today["狀態"] = np.where(today["rank"] <= 3, "做空候選（多方擁擠）",
                           np.where(today["rank"] > n - 3, "做多候選（空方擁擠）", "中性"))
    hist = d[(d["day"] <= asof) & (d["day"] > asof - pd.Timedelta(days=30))]
    spark = hist.groupby("symbol")["close"].apply(list)
    today["30 日走勢"] = today["symbol"].map(spark)

    up, down = C.colors()
    breadth = (today["ret_1d"] > 0).mean()
    top = today.loc[today["funding_z"].idxmax()] if n else None
    bot = today.loc[today["funding_z"].idxmin()] if n else None
    C.kpi_row([
        ("BTC", C.fmt_px(today.loc[today.symbol == "BTCUSDT", "close"].squeeze()),
         C.pct(today.loc[today.symbol == "BTCUSDT", "ret_1d"].squeeze()), C.tone(today.loc[today.symbol == "BTCUSDT", "ret_1d"].squeeze())),
        ("上漲家數", f"{(today['ret_1d'] > 0).sum()} / {len(today)}", f"市場寬度 {breadth:.0%}", "up" if breadth >= .5 else "down"),
        ("平均年化資金費", f"{today['funding_ann'].mean():+.1%}", "正值＝多方付費給空方", C.tone(today['funding_ann'].mean())),
        ("最擁擠（多方）", top.symbol.replace("USDT", "") if top is not None else "—", f"z = {top.funding_z:+.2f}" if top is not None else "", "down"),
        ("最擁擠（空方）", bot.symbol.replace("USDT", "") if bot is not None else "—", f"z = {bot.funding_z:+.2f}" if bot is not None else "", "up"),
        ("資金費分散度", f"{today['funding_day'].std() * 1e4:.1f} bp", "各幣日資金費的標準差", ""),
    ])

    t = today
    if view == "多方擁擠":
        t = t[t["funding_z"] > 0]
    elif view == "空方擁擠":
        t = t[t["funding_z"] < 0]
    if q:
        t = t[t["symbol"].str.contains(q)]
    key = {"擁擠度 z": "funding_z", "24h 漲跌": "ret_1d", "7 日漲跌": "ret_7d", "30 日漲跌": "ret_30d",
           "年化資金費": "funding_ann", "30 日均成交額": "adv_30d"}[sort_by]
    t = t.sort_values(key, ascending=False)
    table = pd.DataFrame({
        "擁擠排名": t["rank"].astype("Int64"), "幣種": t["symbol"].str.replace("USDT", ""), "收盤價": t["close"],
        "24h": t["ret_1d"] * 100, "7 日": t["ret_7d"] * 100, "30 日": t["ret_30d"] * 100, "30 日走勢": t["30 日走勢"],
        "日資金費 (bp)": t["funding_day"] * 1e4, "年化資金費": t["funding_ann"] * 100, "擁擠度 z": t["funding_z"],
        "溢價 (bp)": t["premium"] * 1e4, "30 日均成交額 (百萬)": t["adv_30d"] / 1e6, "狀態": t["狀態"],
    }).reset_index(drop=True)
    styled = (table.style
              .map(lambda v: f"color:{up}" if isinstance(v, (int, float)) and v > 0 else (f"color:{down}" if isinstance(v, (int, float)) and v < 0 else ""),
                   subset=["24h", "7 日", "30 日", "年化資金費"])
              .map(lambda v: f"color:{down}" if str(v).startswith("做空") else (f"color:{up}" if str(v).startswith("做多") else f"color:{C.MUTED}"),
                   subset=["狀態"])
              .background_gradient(cmap="RdBu_r", vmin=-3, vmax=3, subset=["擁擠度 z"])
              .format({"收盤價": C.fmt_px, "24h": "{:+.2f}%", "7 日": "{:+.2f}%", "30 日": "{:+.2f}%",
                       "日資金費 (bp)": "{:+.2f}", "年化資金費": "{:+.1f}%", "擁擠度 z": "{:+.2f}", "溢價 (bp)": "{:+.1f}",
                       "30 日均成交額 (百萬)": "{:,.0f}"}, na_rep="—"))
    ev = st.dataframe(
        styled, use_container_width=True, hide_index=True, height=min(35 * len(table) + 38, 760),
        on_select="rerun", selection_mode="single-row", key=f"screener_{ss.scr_n}",
        column_config={
            "30 日走勢": st.column_config.LineChartColumn(width="small"),
            "擁擠度 z": st.column_config.Column(help="當日資金費相對自身 30 日的 z 分數；越高＝多方越擠"),
        })
    rows = ev.selection.rows if ev and hasattr(ev, "selection") else []
    if rows:
        ss.sym = table.iloc[rows[0]]["幣種"] + "USDT"
        ss.scr_n += 1
        st.switch_page(PAGES["chart"])
    st.caption("排名依『規則模型』：擁擠度 z 最高 3 檔為做空候選、最低 3 檔為做多候選。實際策略在隔日 01:00 UTC 進場。")

    left, right = st.columns([1, 1.35])
    with left:
        b = today.dropna(subset=["funding_z"]).sort_values("funding_z")
        fig = go.Figure(go.Bar(x=b["funding_z"], y=b["symbol"].str.replace("USDT", ""), orientation="h",
                               marker_color=np.where(b["funding_z"] > 0, down, up),
                               text=b["funding_z"].map("{:+.2f}".format), textposition="outside", cliponaxis=False))
        fig.add_vline(x=0, line_color=C.MUTED, line_width=1)
        lim = float(b["funding_z"].abs().max() or 1) * 1.25
        fig.update_xaxes(range=[-lim, lim])
        fig.update_layout(title=f"擁擠度 z 分布（{asof:%Y-%m-%d}）")
        C.show(C.style(fig, 520, legend=False))
    with right:
        rng = st.segmented_control("熱力圖期間", ["30 天", "90 天", "180 天", "1 年"], default="90 天", key="hm_rng") or "90 天"
        nd = {"30 天": 30, "90 天": 90, "180 天": 180, "1 年": 365}[rng]
        hm = d[(d["day"] <= asof) & (d["day"] > asof - pd.Timedelta(days=nd))]
        m = hm.pivot(index="symbol", columns="day", values="funding_z")
        m.index = m.index.str.replace("USDT", "")
        fig = go.Figure(go.Heatmap(z=m.values, x=m.columns, y=m.index, colorscale="RdBu_r", zmin=-3, zmax=3,
                                   colorbar=dict(title="z", thickness=10),
                                   hovertemplate="%{y}<br>%{x|%Y-%m-%d}<br>z = %{z:+.2f}<extra></extra>"))
        fig.update_layout(title="擁擠度熱力圖（紅＝多方擁擠，藍＝空方擁擠）", hovermode="closest")
        C.show(C.style(fig, 470, legend=False))


# ==================================================================== 行情：K 線
def indicator_panel_rows(subs):
    return 1 + len(subs)


def page_chart():
    d = C.daily_market()
    tab_k, tab_cmp = st.tabs([":material/candlestick_chart: K 線分析", ":material/compare_arrows: 多幣比較"])
    with tab_k:
        c1, c2, _ = st.columns([1.3, 1.6, 2.6])
        sym = c1.selectbox("幣種", C.SYMS, index=C.SYMS.index(ss.sym),
                           format_func=lambda s: f"{s.replace('USDT', '')} / USDT 永續")
        ss.sym = sym
        itv = c2.segmented_control("週期", ["1H", "4H", "1D", "1W"], default="1D", key="itv") or "1D"
        h = C.sym_hourly(sym)
        d0, d1 = h.index.min().normalize(), h.index.max()
        r1, r2, r3 = st.columns([2.2, 2.2, 2.2])
        with r1:
            start, end = C.date_range_picker("k", d0, d1, default="3M" if itv in ("1H", "4H") else "1Y")
        ind = r2.multiselect("主圖指標", ["MA7", "MA25", "MA99", "布林通道", "成交量加權均價"], default=["MA25", "MA99"])
        subs = r3.multiselect("副圖", ["成交量", "資金費率", "擁擠度 z", "溢價指數"], default=["成交量", "資金費率", "擁擠度 z"])
        o1, o2, o3 = st.columns([1.4, 1.4, 3])
        show_sig = o1.toggle("標示模型進出場", value=True, help="2021–2024 驗證期，模型把這檔排進多/空 3 檔的日子")
        model = o2.selectbox("訊號模型", list(C.MODELS), index=list(C.MODELS).index(ss.model),
                             format_func=C.MODELS.get, disabled=not show_sig)
        log_y = o3.toggle("對數座標", value=False)

        if itv == "1H" and (end - start).days > 180:
            start = end - pd.Timedelta(days=180)
            st.info("1 小時週期最多顯示 180 天，已自動縮短期間。")
        s = h[(h.index >= start) & (h.index <= end + pd.Timedelta(days=1))]
        rule = {"1H": None, "4H": "4h", "1D": "1D", "1W": "W-MON"}[itv]
        if rule:
            s = s.resample(rule, label="left", closed="left").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "quote_volume": "sum"}).dropna()
        if s.empty:
            st.warning("這段期間沒有資料。")
            return

        dd = d[d["symbol"] == sym].set_index("day")
        last = dd.iloc[-1] if end >= dd.index.max() else dd[dd.index <= end].iloc[-1]
        chg = s["close"].iloc[-1] / s["open"].iloc[0] - 1
        pct_rank = (dd["funding_z"] <= last["funding_z"]).mean()
        st.markdown(
            f'<div class="quote"><span class="sym">{sym.replace("USDT", "")}<span class="muted" style="font-size:1rem">/USDT 永續</span></span>'
            f'<span class="px {C.tone(last.ret_1d)}">{C.fmt_px(s["close"].iloc[-1])}</span>'
            f'<span class="chg {C.tone(last.ret_1d)}">{C.pct(last.ret_1d)} 24h</span>'
            f'<span class="chg {C.tone(chg)}">{C.pct(chg)} 區間</span></div>'
            f'<div class="stats"><div>區間最高<b>{C.fmt_px(s["high"].max())}</b></div><div>區間最低<b>{C.fmt_px(s["low"].min())}</b></div>'
            f'<div>區間成交額<b>{s["quote_volume"].sum() / 1e9:,.2f} B</b></div>'
            f'<div>日資金費<b class="{C.tone(last.funding_day)}">{last.funding_day * 1e4:+.2f} bp</b></div>'
            f'<div>年化資金費<b class="{C.tone(last.funding_ann)}">{last.funding_ann:+.1%}</b></div>'
            f'<div>擁擠度 z<b>{last.funding_z:+.2f}</b></div><div>擁擠度歷史分位<b>{pct_rank:.0%}</b></div>'
            f'<div>30 日年化波動<b>{last.rvol_30d:.0%}</b></div><div>資料截至<b>{last.name:%Y-%m-%d} UTC</b></div></div>',
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
        if "布林通道" in ind:
            m, sd = s["close"].rolling(20).mean(), s["close"].rolling(20).std()
            fig.add_trace(go.Scatter(x=s.index, y=m + 2 * sd, name="BB 上軌", line=dict(width=1, color="#5e6673")), 1, 1)
            fig.add_trace(go.Scatter(x=s.index, y=m - 2 * sd, name="BB 下軌", line=dict(width=1, color="#5e6673"),
                                     fill="tonexty", fillcolor="rgba(94,102,115,.08)"), 1, 1)
        if "成交量加權均價" in ind:
            tp = (s["high"] + s["low"] + s["close"]) / 3
            fig.add_trace(go.Scatter(x=s.index, y=(tp * s["volume"]).cumsum() / s["volume"].cumsum(), name="VWAP（區間）",
                                     line=dict(width=1.3, dash="dot", color=C.BLUE)), 1, 1)
        if show_sig:
            W = C.weights(model, 3, "equal")
            if sym in W.columns:
                w = W[sym]
                w = w[(w.index >= start) & (w.index <= end)]
                for side, mask, color, sym_ in [("模型做多", w > 0, up, "triangle-up"), ("模型做空", w < 0, down, "triangle-down")]:
                    dts = w.index[mask]
                    if len(dts):
                        # decision date t trades at the 01:00 open; mark the daily low/high of that day
                        px = h["low"].resample("1D").min() if side == "模型做多" else h["high"].resample("1D").max()
                        y = px.reindex(dts) * (0.985 if side == "模型做多" else 1.015)
                        fig.add_trace(go.Scatter(x=dts, y=y, mode="markers", name=side,
                                                 marker=dict(symbol=sym_, size=8, color=color, line=dict(width=0))), 1, 1)
        row = 2
        dz = dd[(dd.index >= start) & (dd.index <= end)]
        for sp in subs:
            if sp == "成交量":
                colr = np.where(s["close"] >= s["open"], up, down)
                fig.add_trace(go.Bar(x=s.index, y=s["quote_volume"] / 1e6, name="成交額（百萬）", marker_color=colr, opacity=.7), row, 1)
            elif sp == "資金費率":
                fig.add_trace(go.Bar(x=dz.index, y=dz["funding_day"] * 1e4, name="日資金費 (bp)",
                                     marker_color=np.where(dz["funding_day"] > 0, down, up)), row, 1)
            elif sp == "擁擠度 z":
                fig.add_trace(go.Scatter(x=dz.index, y=dz["funding_z"], name="擁擠度 z", line=dict(color=C.ACCENT, width=1.4)), row, 1)
                for lv in (2, -2):
                    fig.add_hline(y=lv, line_dash="dot", line_color="#5e6673", row=row, col=1)
            elif sp == "溢價指數":
                fig.add_trace(go.Scatter(x=dz.index, y=dz["premium"] * 1e4, name="溢價 (bp)", line=dict(color=C.CYAN, width=1.2),
                                         fill="tozeroy", fillcolor="rgba(34,195,214,.08)"), row, 1)
            fig.update_yaxes(title_text=sp, title_font=dict(size=11), row=row, col=1)
            row += 1
        fig.update_layout(xaxis_rangeslider_visible=False)
        if log_y:
            fig.update_yaxes(type="log", row=1, col=1)
        C.style(fig, 520 + 140 * len(subs), right_axis=True)
        C.show(fig)
        st.caption("時間皆為 UTC。▲／▼ 為模型在該決策日（00:00 UTC 出訊號、01:00 開盤進場）把此幣排入多／空前 3 名。")

        with st.expander("這檔幣的模型訊號紀錄", icon=":material/history:"):
            sf = C.signal_frame().dropna(subset=[model])
            sf = sf[sf["tradable"]]
            allr = sf.groupby("date")[model].rank(ascending=False, method="first")
            mine = sf[sf["symbol"] == sym].copy()
            mine["排名"] = allr.loc[mine.index].astype(int)
            cnt = sf.groupby("date")[model].count()
            mine["檔數"] = mine["date"].map(cnt)
            mine["動作"] = np.where(mine["排名"] <= 3, "做多", np.where(mine["排名"] > mine["檔數"] - 3, "做空", "—"))
            mine = mine[(mine["date"] >= start) & (mine["date"] <= end)].sort_values("date", ascending=False)
            st.dataframe(pd.DataFrame({
                "決策日": mine["date"].dt.strftime("%Y-%m-%d"), "模型分數": mine[model], "排名": mine["排名"], "檔數": mine["檔數"],
                "動作": mine["動作"], "擁擠度 z": mine["funding_z_30d_raw"], "隔日報酬（含資金費）": mine["ret_long_simple"] * 100,
            }), use_container_width=True, hide_index=True, height=360,
                column_config={"模型分數": st.column_config.NumberColumn(format="%.4f"),
                               "擁擠度 z": st.column_config.NumberColumn(format="%+.2f"),
                               "隔日報酬（含資金費）": st.column_config.NumberColumn(format="%+.2f%%")})

    with tab_cmp:
        c1, c2 = st.columns([3, 2])
        pick = c1.multiselect("比較幣種", C.SYMS, default=["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"],
                              format_func=lambda s: s.replace("USDT", ""), max_selections=8)
        metric = c2.segmented_control("比較項目", ["累積報酬", "擁擠度 z", "年化資金費"], default="累積報酬") or "累積報酬"
        d0, d1 = d["day"].min(), d["day"].max()
        start, end = C.date_range_picker("cmp", d0, d1, default="1Y")
        sub = d[(d["symbol"].isin(pick)) & (d["day"] >= start) & (d["day"] <= end)]
        fig = go.Figure()
        stats = []
        for i, sname in enumerate(pick):
            x = sub[sub["symbol"] == sname].set_index("day")
            if x.empty:
                continue
            if metric == "累積報酬":
                y = x["close"] / x["close"].iloc[0] - 1
            elif metric == "擁擠度 z":
                y = x["funding_z"].rolling(7, min_periods=1).mean()
            else:
                y = x["funding_ann"].rolling(7, min_periods=1).mean()
            fig.add_trace(go.Scatter(x=y.index, y=y, name=sname.replace("USDT", ""), line=dict(width=1.6)))
            r = np.log(x["close"]).diff()
            stats.append({"幣種": sname.replace("USDT", ""), "區間報酬": (x["close"].iloc[-1] / x["close"].iloc[0] - 1) * 100,
                          "年化波動": r.std() * np.sqrt(365) * 100,
                          "最大回撤": ((x["close"] / x["close"].cummax()) - 1).min() * 100,
                          "平均年化資金費": x["funding_ann"].mean() * 100, "平均擁擠度 z": x["funding_z"].mean()})
        if metric == "累積報酬":
            fig.update_yaxes(tickformat=".0%")
        elif metric == "年化資金費":
            fig.update_yaxes(tickformat=".0%")
        fig.update_layout(title=f"{metric}（{'7 日平滑' if metric != '累積報酬' else '以期初為 0'}）")
        C.show(C.style(fig, 480))
        if stats:
            st.dataframe(pd.DataFrame(stats), use_container_width=True, hide_index=True, column_config={
                k: st.column_config.NumberColumn(format="%+.1f%%" if k != "年化波動" else "%.1f%%")
                for k in ["區間報酬", "年化波動", "最大回撤", "平均年化資金費"]} | {
                "平均擁擠度 z": st.column_config.NumberColumn(format="%+.2f")})


# ==================================================================== 策略：每日選股
def page_ranking():
    C.page_header("每日選股排行", "選一個決策日與模型，看當天 19 檔的模型分數排行、被選進的多空各 N 檔，以及隔天實際的報酬。")
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
    dt = C.to_utc(c1.date_input("決策日（00:00 UTC）", key="rk_date", min_value=dates.iloc[0].date(),
                                 max_value=dates.iloc[-1].date()))
    dt = dates[dates <= dt].iloc[-1]
    model = c2.selectbox("模型", list(C.MODELS), index=list(C.MODELS).index(ss.model), format_func=C.MODELS.get, key="rk_model")
    k = c3.number_input("每邊檔數", 2, 6, 3)
    with c4:
        st.write("")
        b1, b2, b3 = st.columns(3)
        b1.button("前一天", icon=":material/chevron_left:", use_container_width=True, on_click=shift, args=(-1,))
        b2.button("後一天", icon=":material/chevron_right:", use_container_width=True, on_click=shift, args=(1,))
        b3.button("隨機", icon=":material/shuffle:", use_container_width=True, on_click=shift, args=(0,))

    day = sf[(sf["date"] == dt) & sf["tradable"]].dropna(subset=[model]).sort_values(model, ascending=False).reset_index(drop=True)
    n = len(day)
    day["排名"] = np.arange(1, n + 1)
    day["動作"] = np.where(day["排名"] <= k, "做多", np.where(day["排名"] > n - k, "做空", "—"))
    W = C.weights(model, int(k), "equal")
    w = W.loc[dt] if dt in W.index else pd.Series(dtype=float)
    pnl = float((w * day.set_index("symbol")["ret_long_simple"].reindex(w.index).fillna(0)).sum()) if len(w) else np.nan
    ic = day[[model, "ret_long_simple"]].corr(method="spearman").iloc[0, 1] if n > 4 else np.nan
    longs, shorts = day[day["動作"] == "做多"], day[day["動作"] == "做空"]
    C.kpi_row([
        ("決策日", f"{dt:%Y-%m-%d}", f"fold {int(day['fold'].iloc[0]) if n else '—'}", ""),
        ("可交易檔數", f"{n}", "少於 8 檔不交易", ""),
        ("當日組合報酬（未扣成本）", C.pct(pnl), f"多 {k} 空 {k} 等權", C.tone(pnl)),
        ("多頭平均", C.pct(longs["ret_long_simple"].mean()), "24h，含資金費", C.tone(longs["ret_long_simple"].mean())),
        ("空頭平均（做空所得）", C.pct(-shorts["ret_long_simple"].mean()), "24h，含資金費", C.tone(-shorts["ret_long_simple"].mean())),
        ("當日 IC", f"{ic:+.3f}" if np.isfinite(ic) else "—", "分數與實際報酬的等級相關", C.tone(ic)),
    ])
    up, down = C.colors()
    left, right = st.columns([1.5, 1])
    with left:
        b = day.iloc[::-1]
        colr = np.where(b["動作"] == "做多", up, np.where(b["動作"] == "做空", down, "#3a414c"))
        fig = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[.55, .45], horizontal_spacing=.03,
                            subplot_titles=(f"{C.MODELS[model]} 分數", "隔天實際報酬（含資金費）"))
        fig.add_trace(go.Bar(y=b["symbol"].str.replace("USDT", ""), x=b[model], orientation="h", marker_color=colr, name="分數",
                             hovertemplate="%{y}: %{x:.4f}<extra></extra>"), 1, 1)
        fig.add_trace(go.Bar(y=b["symbol"].str.replace("USDT", ""), x=b["ret_long_simple"], orientation="h", name="實際報酬",
                             marker_color=np.where(b["ret_long_simple"] > 0, up, down), opacity=.85,
                             hovertemplate="%{y}: %{x:+.2%}<extra></extra>"), 1, 2)
        fig.update_xaxes(tickformat="+.1%", row=1, col=2)
        C.show(C.style(fig, 40 * n + 90, legend=False))
    with right:
        for title, part, sign in [('<span class="sq up-bg"></span>做多清單', longs, 1), ('<span class="sq down-bg"></span>做空清單', shorts.iloc[::-1], -1)]:
            rows = "".join(
                f'<div class="pos-row"><span><b>{r.symbol.replace("USDT", "")}</b> <span class="muted">#{r.排名}</span></span>'
                f'<span class="muted">z {r.funding_z_30d_raw:+.2f}</span>'
                f'<span class="muted">資金費 {r.funding_day * 1e4:+.1f}bp</span>'
                f'<span class="{C.tone(sign * r.ret_long_simple)}">{C.pct(sign * r.ret_long_simple)}</span></div>'
                for r in part.itertuples())
            st.markdown(f'<div class="panel"><h4>{title}</h4>{rows or "<span class=muted>無</span>"}</div>', unsafe_allow_html=True)
        st.caption("報酬欄是『這個部位』賺的：做空那欄已經把正負號反過來。z＝擁擠度（資金費 30 日 z 分數）。")

    st.markdown("#### 近 60 個決策日的模型排名")
    lo = dt - pd.Timedelta(days=60)
    hist = sf[(sf["date"] > lo) & (sf["date"] <= dt) & sf["tradable"]].copy()
    hist["rk"] = hist.groupby("date")[model].rank(ascending=False, method="first")
    m = hist.pivot(index="symbol", columns="date", values="rk")
    m.index = m.index.str.replace("USDT", "")
    fig = go.Figure(go.Heatmap(z=m.values, x=m.columns, y=m.index, colorscale=[[0, up], [.5, "#1c222b"], [1, down]],
                               colorbar=dict(title="排名", thickness=10),
                               hovertemplate="%{y}<br>%{x|%Y-%m-%d}<br>第 %{z:.0f} 名<extra></extra>"))
    fig.update_layout(hovermode="closest")
    C.show(C.style(fig, 430, legend=False))


# ==================================================================== 策略：回測實驗室
def page_backtest():
    C.page_header("回測實驗室", "用 2021–2024 walk-forward 驗證期的模型預測重新回測。預設值＝報告設定（多 3 空 3、等權、每邊 7 bp）。")
    base = C.bt_base()
    d0, d1 = base["date"].min(), base["date"].max()
    with st.form("bt_form", border=True):
        c1, c2, c3, c4 = st.columns(4)
        model = c1.selectbox("模型", list(C.MODELS), index=list(C.MODELS).index(ss.model), format_func=C.MODELS.get)
        k = c2.slider("每邊檔數", 2, 6, 3)
        alloc = c3.radio("配置方式", ["equal", "inv_vol"], format_func={"equal": "等權", "inv_vol": "反波動加權"}.get, horizontal=True)
        cost = c4.slider("每邊交易成本（bp）", 0.0, 30.0, float(C.COST_BPS), 0.5)
        c5, c6, c7, c8 = st.columns(4)
        dr = c5.date_input("回測期間", value=(d0.date(), d1.date()), min_value=d0.date(), max_value=d1.date())
        vt_on = c6.toggle("波動目標", value=False)
        vt = c6.slider("目標年化波動", 0.05, 0.60, 0.20, 0.05, disabled=False, label_visibility="collapsed")
        ladder = c7.toggle("回撤停損階梯", value=False, help="回撤 10% 減半、15% 停止（依基礎策略的回撤）")
        bench = c8.multiselect("比較基準", ["規則基準", "等權做多全部", "BTC 做多"], default=["規則基準", "等權做多全部"])
        run = st.form_submit_button("執行回測", icon=":material/play_arrow:", type="primary", use_container_width=True)
    ss.model = model
    start, end = (C.to_utc(dr[0]), C.to_utc(dr[1])) if isinstance(dr, (tuple, list)) and len(dr) == 2 else (d0, d1)
    bt = C.backtest(model, int(k), alloc, float(cost), start, end, float(vt) if vt_on else None, bool(ladder))
    net = bt["net"]
    p = C.perf(net, bt["turnover"])
    g_sc, t_sc = bt["gross"] * bt["scale"], bt["turnover"] * bt["scale"]
    be = float(g_sc.sum() / t_sc.sum() * 1e4) if t_sc.sum() else np.nan
    gross_sr = C.perf(g_sc)["sharpe"]
    C.kpi_row([
        ("累積報酬", C.pct(p["total"], 1), f"{p['days']} 天", C.tone(p["total"])),
        ("年化報酬", C.pct(p["cagr"], 1), f"年化波動 {p['vol']:.1%}", C.tone(p["cagr"])),
        ("淨 Sharpe", f"{p['sharpe']:.2f}", f"95% CI [{p['sharpe_lo']:.2f}, {p['sharpe_hi']:.2f}]", C.tone(p["sharpe"])),
        ("成本前 Sharpe", f"{gross_sr:.2f}", "未扣交易成本", C.tone(gross_sr)),
        ("最大回撤", C.pct(p["max_dd"], 1, sign=False), f"Calmar {p['calmar']:.2f}" if np.isfinite(p["calmar"]) else "", "down"),
        ("損益兩平成本", f"{be:.1f} bp", "舒適區 > 2 倍成本" if be > 2 * cost else ("灰色地帶" if be > cost else "低於目前成本"),
         "up" if be > 2 * cost else ("" if be > cost else "down")),
        ("日勝率", C.pct(p["hit"], 1, sign=False), f"最佳 {p['best']:+.1%}／最差 {p['worst']:+.1%}", ""),
        ("年化換手", f"{p['ann_turnover']:.0f}×", f"平均每天 {bt['n_names'].mean():.1f} 檔", ""),
    ])
    curves = {"策略": net}
    if "規則基準" in bench and model != "rule":
        rb = C.backtest("rule", 3, "equal", float(cost), start, end, None, False)
        curves["規則基準"] = rb["net"]
    bm = C.benchmarks(start, end)
    for b_ in ("等權做多全部", "BTC 做多"):
        if b_ in bench:
            curves[b_] = bm[b_].reindex(net.index).fillna(0)
    up, down = C.colors()
    tabs = st.tabs(["權益曲線", "月報酬", "成本敏感度", "損益拆解", "滾動指標", "每日持倉"])
    with tabs[0]:
        log_y = st.toggle("對數座標", value=True, key="bt_log")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.72, .28], vertical_spacing=.03)
        pal = [C.ACCENT, C.PURPLE, C.CYAN, C.BLUE]
        for i, (name, r) in enumerate(curves.items()):
            eq = (1 + r).cumprod()
            fig.add_trace(go.Scatter(x=eq.index, y=eq, name=name, line=dict(width=2.2 if i == 0 else 1.3, color=pal[i % 4])), 1, 1)
        eq = (1 + net).cumprod()
        ddn = eq / eq.cummax() - 1
        fig.add_trace(go.Scatter(x=ddn.index, y=ddn, name="策略回撤", fill="tozeroy", line=dict(color=down, width=1),
                                 fillcolor="rgba(246,70,93,.18)"), 2, 1)
        if log_y:
            fig.update_yaxes(type="log", row=1, col=1)
        fig.update_yaxes(tickformat=".0%", row=2, col=1)
        fig.update_yaxes(title_text="淨值（期初 = 1）", row=1, col=1)
        C.show(C.style(fig, 600))
        rows = []
        for name, r in curves.items():
            q = C.perf(r)
            rows.append({"策略": name, "累積報酬": q["total"] * 100, "年化報酬": q["cagr"] * 100, "年化波動": q["vol"] * 100,
                         "Sharpe": q["sharpe"], "最大回撤": q["max_dd"] * 100, "Calmar": q["calmar"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, column_config={
            **{c: st.column_config.NumberColumn(format="%+.1f%%") for c in ["累積報酬", "年化報酬", "最大回撤"]},
            "年化波動": st.column_config.NumberColumn(format="%.1f%%"),
            "Sharpe": st.column_config.NumberColumn(format="%.2f"), "Calmar": st.column_config.NumberColumn(format="%.2f")})
    with tabs[1]:
        t = C.monthly_table(net)
        z = t.values * 100
        fig = go.Figure(go.Heatmap(z=z, x=t.columns, y=[str(y) for y in t.index], colorscale=[[0, down], [.5, "#1c222b"], [1, up]],
                                   zmid=0, text=np.where(np.isnan(z), "", np.vectorize(lambda v: f"{v:+.1f}%")(np.nan_to_num(z))),
                                   texttemplate="%{text}", hovertemplate="%{y} %{x}: %{z:+.2f}%<extra></extra>",
                                   colorbar=dict(title="%", thickness=10)))
        fig.update_layout(title="月報酬（扣成本後）", hovermode="closest")
        fig.update_yaxes(autorange="reversed")
        C.show(C.style(fig, 90 + 60 * len(t), legend=False))
        yr = t["全年"]
        fig = go.Figure(go.Bar(x=[str(y) for y in yr.index], y=yr, marker_color=np.where(yr > 0, up, down),
                               text=yr.map("{:+.1%}".format), textposition="outside"))
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(title="年度報酬")
        C.show(C.style(fig, 330, legend=False))
    with tabs[2]:
        grid = [0, 2, 4, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40]
        sr = [C.perf(bt["gross"] * bt["scale"] - bt["turnover"] * bt["scale"] * c / 1e4)["sharpe"] for c in grid]
        fig = go.Figure(go.Scatter(x=grid, y=sr, mode="lines+markers", name="淨 Sharpe", line=dict(color=C.ACCENT, width=2)))
        fig.add_hline(y=0, line_color=C.MUTED, line_width=1)
        fig.add_vline(x=cost, line_dash="dash", line_color=C.BLUE, annotation_text=f"目前 {cost:g} bp")
        if np.isfinite(be) and be > 0:
            fig.add_vline(x=be, line_dash="dot", line_color=down, annotation_text=f"損益兩平 {be:.1f} bp")
        fig.update_layout(title="每邊成本 vs 淨 Sharpe", xaxis_title="每邊成本（bp）", yaxis_title="淨 Sharpe")
        C.show(C.style(fig, 430, legend=False))
        st.caption("損益兩平成本＝成本前總報酬 ÷ 總換手。報告用的分區：< 1 倍成本＝硬底線、1–2 倍＝灰色地帶、> 2 倍＝舒適區。")
    with tabs[3]:
        sc = bt["scale"]
        fund = bt["funding_pnl"] * sc
        price = bt["gross"] * sc - fund
        costs = -(net - bt["gross"] * sc)
        fig = go.Figure()
        for name, s_, colr in [("價格損益", price, C.BLUE), ("資金費收付", fund, C.ACCENT), ("交易成本", -costs, down), ("淨損益", net, up)]:
            fig.add_trace(go.Scatter(x=s_.index, y=s_.cumsum(), name=name, line=dict(width=2 if name == "淨損益" else 1.4, color=colr)))
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(title="累積損益拆解（簡單加總）")
        C.show(C.style(fig, 470))
        C.kpi_row([("價格損益（年化）", C.pct(price.mean() * 365, 1), "", C.tone(price.mean())),
                   ("資金費（年化）", C.pct(fund.mean() * 365, 1), "空擁擠方收取的補償", C.tone(fund.mean())),
                   ("交易成本（年化）", C.pct(-costs.mean() * 365, 1), "", "down")])
    with tabs[4]:
        win = st.segmented_control("滾動視窗", [30, 90, 180, 365], default=90, format_func=lambda x: f"{x} 天") or 90
        rs = net.rolling(win).mean() / net.rolling(win).std() * np.sqrt(365)
        rv = net.rolling(win).std() * np.sqrt(365)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.05, subplot_titles=(f"滾動 {win} 天 Sharpe", f"滾動 {win} 天年化波動"))
        fig.add_trace(go.Scatter(x=rs.index, y=rs, name="Sharpe", line=dict(color=C.ACCENT)), 1, 1)
        fig.add_hline(y=0, line_color=C.MUTED, row=1, col=1)
        fig.add_trace(go.Scatter(x=rv.index, y=rv, name="波動", line=dict(color=C.PURPLE)), 2, 1)
        fig.update_yaxes(tickformat=".0%", row=2, col=1)
        C.show(C.style(fig, 520))
    with tabs[5]:
        W = C.weights(model, int(k), alloc)
        W = W[(W.index >= start) & (W.index <= end)]
        def names(row, side):
            x = row[row > 0] if side > 0 else row[row < 0]
            return "、".join(f"{s.replace('USDT', '')}({abs(v):.0%})" for s, v in x.sort_values(ascending=side < 0).items())
        log = pd.DataFrame({"決策日": W.index.strftime("%Y-%m-%d"),
                            "做多": [names(r, 1) for _, r in W.iterrows()],
                            "做空": [names(r, -1) for _, r in W.iterrows()],
                            "曝險倍數": bt["scale"].reindex(W.index).values,
                            "成本前報酬": bt["gross"].reindex(W.index).values * 100,
                            "換手": bt["turnover"].reindex(W.index).values,
                            "淨報酬": net.reindex(W.index).values * 100}).iloc[::-1]
        st.dataframe(log, hide_index=True, use_container_width=True, height=480, column_config={
            "成本前報酬": st.column_config.NumberColumn(format="%+.2f%%"), "淨報酬": st.column_config.NumberColumn(format="%+.2f%%"),
            "換手": st.column_config.NumberColumn(format="%.2f"), "曝險倍數": st.column_config.NumberColumn(format="%.2f")})
        st.download_button("下載每日持倉與損益 CSV", log.to_csv(index=False).encode("utf-8-sig"), icon=":material/download:",
                           file_name=f"backtest_{model}_k{k}_{cost:g}bp.csv", mime="text/csv")
    st.caption("所有參數組合都只是『事後觀察』，不是新的研究試驗；報告的正式結論以凍結設定（setup.yaml v1）為準。")


# ==================================================================== 策略：即時訊號
def page_live():
    C.page_header("即時訊號（Shadow mode）", "從交易所公開 API 抓目前的資金費率與溢價（Binance → Bybit → Hyperliquid 依序嘗試），依規則模型排擁擠度。只觀察、不下單。")
    c1, c2 = st.columns([1, 4])
    if c1.button("重新整理", icon=":material/refresh:", type="primary", use_container_width=True):
        C.live_quotes.clear()
    from app.live_signal import rank
    raw, src, errs = C.live_quotes()
    live = rank(raw, C.funding_raw()) if len(raw) else pd.DataFrame()
    t = live["fetched_utc"].iloc[0] if len(live) else None
    if src == "最近一次已結算資料":
        c2.warning(f"目前連不到交易所即時 API（{'、'.join(errs)}），改顯示資料集最後一天（{t:%Y-%m-%d} 00:00 UTC）的已結算資金費。")
    else:
        note = f"（{'、'.join(e.split(':')[0] for e in errs)} 連不到，已自動改用 {src}）" if errs else ""
        c2.success(f"{src} 即時資料 · {t:%Y-%m-%d %H:%M} UTC {note}")
    if live.empty:
        return
    up, down = C.colors()
    key = "funding_z_est" if "funding_z_est" in live else "lastFundingRate"
    sh, lg = live[live["shadow_action"].str.startswith("做空")], live[live["shadow_action"].str.startswith("做多")]
    C.kpi_row([
        ("下次資金費結算", f"{pd.to_datetime(live['next_funding']).min():%H:%M} UTC" if pd.to_datetime(live['next_funding']).notna().any() else "—", f"資料來源：{src}", ""),
        ("平均預估資金費（每期）", f"{live['lastFundingRate'].mean() * 1e4:+.2f} bp", f"年化約 {live['lastFundingRate'].mean() * 3 * 365:+.1%}",
         C.tone(live['lastFundingRate'].mean())),
        ("做空候選", "、".join(sh["symbol"].str.replace("USDT", "")), "多方最擁擠", "down"),
        ("做多候選", "、".join(lg["symbol"].str.replace("USDT", "")), "空方最擁擠", "up"),
    ])
    left, right = st.columns([1.2, 1])
    with left:
        b = live.sort_values(key)
        fig = go.Figure(go.Bar(x=b[key], y=b["symbol"].str.replace("USDT", ""), orientation="h",
                               marker_color=np.where(b["shadow_action"].str.startswith("做空"), down,
                                                     np.where(b["shadow_action"].str.startswith("做多"), up, "#3a414c"))))
        fig.update_layout(title="即時擁擠度（預估 z）" if key == "funding_z_est" else "即時資金費率")
        C.show(C.style(fig, 560, legend=False))
    with right:
        show = pd.DataFrame({"排名": live["crowding_rank"], "幣種": live["symbol"].str.replace("USDT", ""),
                             "標記價格": live.get("markPrice", np.nan), "預估資金費 (bp)": live["lastFundingRate"] * 1e4,
                             "預估 z": live.get("funding_z_est", np.nan), "溢價 (bp)": live["premium"] * 1e4,
                             "動作": live["shadow_action"]})
        st.dataframe(show, hide_index=True, use_container_width=True, height=560, column_config={
            "標記價格": st.column_config.NumberColumn(format="%.4f"),
            "預估資金費 (bp)": st.column_config.NumberColumn(format="%+.2f"),
            "預估 z": st.column_config.NumberColumn(format="%+.2f"), "溢價 (bp)": st.column_config.NumberColumn(format="%+.1f"),
            "動作": st.column_config.TextColumn(width="medium")})
    st.caption("即時資金費是『預估值』，要到下次結算才確定；研究與回測只用已結算值。")


# ==================================================================== 研究：模型診斷
def page_models():
    C.page_header("模型診斷", "規則 → Ridge → LightGBM 的模型階梯：16 個季度 walk-forward fold 的 IC 與信賴區間。")
    mc = C.J("model_comparison")
    s = pd.DataFrame(mc["summary"]).T
    C.kpi_row([(C.MODELS[m], f"IC {s.loc[m, 'ic']:+.3f}", f"95% CI [{s.loc[m, 'lo']:+.3f}, {s.loc[m, 'hi']:+.3f}] · ICIR {s.loc[m, 'icir']:.2f}",
                C.tone(s.loc[m, "ic"])) for m in ["rule", "ridge", "lgbm"]] +
              [("採用模型", C.MODELS[mc["chosen"]], "複雜模型需成對比較 CI 下界 > 0", "")])
    bf = pd.read_csv(C.OUT / "wf_ic_by_fold.csv")
    pick = st.multiselect("顯示模型", list(C.MODELS), default=list(C.MODELS), format_func=C.MODELS.get)
    fig = go.Figure()
    pal = {"rule": C.CYAN, "ridge": C.PURPLE, "lgbm": C.ACCENT}
    for m in pick:
        fig.add_trace(go.Bar(x=bf["val_start"], y=bf[f"{m}_ic"], name=C.MODELS[m], marker_color=pal[m],
                             error_y=dict(type="data", symmetric=False, array=bf[f"{m}_hi"] - bf[f"{m}_ic"],
                                          arrayminus=bf[f"{m}_ic"] - bf[f"{m}_lo"], thickness=1, color="#8a93a0")))
    fig.add_hline(y=0, line_color=C.MUTED)
    fig.update_layout(barmode="group", title="各 fold 每日 IC 平均（Newey-West 95% CI）", xaxis_title="驗證 fold 起始日")
    C.show(C.style(fig, 440))
    c1, c2 = st.columns(2)
    with c1:
        r = C.out_parquet("rolling_ic").set_index("date")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=r.index, y=r["ic"], name="每日 IC", marker_color="#3a414c"))
        fig.add_trace(go.Scatter(x=r.index, y=r["roll_ic"], name="滾動 IC", line=dict(color=C.ACCENT, width=2)))
        fig.add_hline(y=0, line_color=C.MUTED)
        fig.update_layout(title=f"{C.MODELS[mc['chosen']]} 滾動 IC")
        fig.update_yaxes(range=[-0.3, 0.3])
        C.show(C.style(fig, 400))
    with c2:
        st.markdown("##### 模型摘要")
        st.dataframe(s[["ic", "lo", "hi", "t", "icir", "hit", "pos_fold_share"]].rename(
            index=C.MODELS, columns={"ic": "IC", "lo": "CI 下界", "hi": "CI 上界", "t": "t 值", "icir": "ICIR", "hit": "IC>0 天數比", "pos_fold_share": "正 IC fold 比"}),
            use_container_width=True, column_config={c: st.column_config.NumberColumn(format="%+.3f") for c in ["IC", "CI 下界", "CI 上界"]})
        st.markdown("##### 成對比較（IC 差）")
        pr = pd.DataFrame(mc["paired"]).T
        st.dataframe(pr.rename(columns={"d_ic": "IC 差", "lo": "CI 下界", "hi": "CI 上界"}), use_container_width=True,
                     column_config={c: st.column_config.NumberColumn(format="%+.4f") for c in ["IC 差", "CI 下界", "CI 上界"]})
    t1, t2 = st.tabs(["特徵初篩（triage）", "SHAP 特徵重要度"])
    with t1:
        tr = pd.read_csv(C.OUT / "triage.csv")
        cols = [c for c in ["feature", "role", "ic", "lo", "hi", "icir", "pos_fold_share", "spread_bp_per_day", "verdict"] if c in tr]
        st.dataframe(tr[cols], use_container_width=True, hide_index=True)
    with t2:
        if (C.OUT / "fig_shap.png").exists():
            st.image(str(C.OUT / "fig_shap.png"), use_container_width=True)


# ==================================================================== 研究：風險
def page_risk():
    C.page_header("風險監控", "採用模型的回撤、尾部風險、壓力事件與市場狀態切片。")
    rk = C.J("risk")
    r = C.out_parquet("risk_daily").set_index("date")
    dd = rk["drawdown"]
    v = rk["var_cvar_95"]["全部"]
    C.kpi_row([("最大回撤", C.pct(dd["depth"], 1, False), f"{dd['peak']} → {dd['trough']}", "down"),
               ("回撤期間", f"{dd['duration_days']} 天", f"恢復：{dd['recovered']}", ""),
               ("95% VaR（日）", C.pct(v[0], 2, False), "", "down"), ("95% CVaR（日）", C.pct(v[1], 2, False), "", "down"),
               ("BTC beta（HAC）", f"{rk['beta_hac']['beta']:+.3f}", f"[{rk['beta_hac']['lo']:+.3f}, {rk['beta_hac']['hi']:+.3f}]", "")])
    up, down = C.colors()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.65, .35], vertical_spacing=.05,
                        subplot_titles=("回撤與停損階梯", "滾動 BTC beta"))
    fig.add_trace(go.Scatter(x=r.index, y=r["dd"], fill="tozeroy", name="回撤", line=dict(color=down, width=1),
                             fillcolor="rgba(246,70,93,.18)"), 1, 1)
    for lv, lab in [(.05, "警示"), (.10, "減半"), (.15, "停止"), (.20, "平倉")]:
        fig.add_hline(y=-lv, line_dash="dash", line_color="#5e6673", annotation_text=lab, row=1, col=1)
    fig.add_trace(go.Scatter(x=r.index, y=r["roll_beta"], name="beta", line=dict(color=C.CYAN)), 2, 1)
    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    C.show(C.style(fig, 560))
    c1, c2 = st.columns(2)
    with c1:
        vc = pd.DataFrame(rk["var_cvar_95"], index=["VaR", "CVaR"]).T
        fig = go.Figure([go.Bar(x=vc.index, y=vc["VaR"], name="VaR 95%", marker_color=C.ACCENT),
                         go.Bar(x=vc.index, y=vc["CVaR"], name="CVaR 95%", marker_color=down)])
        fig.update_yaxes(tickformat=".1%")
        fig.update_layout(title="各波動狀態的每日 VaR / CVaR", barmode="group")
        C.show(C.style(fig, 380))
    with c2:
        rs = pd.DataFrame(rk["regime_slices"])
        fig = go.Figure(go.Bar(x=rs["regime"], y=rs["sharpe"], marker_color=np.where(rs["sharpe"] > 0, up, down),
                               text=rs["days"].map(lambda x: f"{x} 天"), textposition="outside"))
        fig.update_layout(title="市場狀態切片的 Sharpe")
        C.show(C.style(fig, 380, legend=False))
    st.markdown("##### 壓力測試")
    ev = st.segmented_control("事件", list(rk["stress"]), default=list(rk["stress"])[0]) or list(rk["stress"])[0]
    x = rk["stress"][ev]
    names = {"total": "期間累積", "worst_day": "最差單日", "btc": "同期 BTC", "ann_ret": "年化報酬", "vol": "年化波動",
             "sharpe": "Sharpe", "max_dd": "最大回撤", "sharpe_lo": "Sharpe CI 下界", "sharpe_hi": "Sharpe CI 上界", "days": "天數"}
    C.kpi_row([(names.get(k_, k_), f"{v_:.2f}" if "sharpe" in k_ else (f"{v_:,.0f}" if k_ == "days" else C.pct(v_, 1)), "",
                C.tone(v_) if k_ != "days" else "") for k_, v_ in x.items()])


# ==================================================================== 研究：2025 封存驗證
def page_holdout():
    C.page_header("2025 封存期驗證", "2025 年資料在研究期間完全封存，規格凍結後只打開一次。這是策略真正的樣本外成績。")
    ho = C.J("holdout")
    if not ho:
        st.info("封存期尚未打開。")
        return
    hd = C.out_parquet("holdout_daily").set_index("date")
    ch = ho["chosen"]
    t, ic = ho["table"][ch], ho["ic"][ch]
    C.kpi_row([("打開時間", ho["opened_at"].replace("T", " "), "只允許一次", ""),
               ("淨 Sharpe", f"{t['sharpe']:.2f}", f"[{t['sharpe_lo']:.2f}, {t['sharpe_hi']:.2f}]", C.tone(t["sharpe"])),
               ("IC", f"{ic['ic']:+.3f}", f"[{ic['lo']:+.3f}, {ic['hi']:+.3f}]", C.tone(ic["ic"])),
               ("累積報酬", C.pct(t["total"], 1), f"{t['days']} 天", C.tone(t["total"])),
               ("最大回撤", C.pct(t["max_dd"], 1, False), "", "down"),
               ("確認檢定", "通過" if (t["sharpe"] > 0 and ic["lo"] > 0) else "未通過", "淨 Sharpe > 0 且 IC 下界 > 0",
                "up" if (t["sharpe"] > 0 and ic["lo"] > 0) else "down")])
    fig = go.Figure()
    pal = {"rule_net": C.CYAN, "ridge_net": C.PURPLE, "lgbm_net": C.ACCENT, "ew_long": "#5e6673"}
    lab = {"rule_net": "規則", "ridge_net": "Ridge", "lgbm_net": "LightGBM", "ew_long": "等權做多"}
    for c in hd.columns:
        fig.add_trace(go.Scatter(x=hd.index, y=(1 + hd[c]).cumprod(), name=lab.get(c, c),
                                 line=dict(color=pal.get(c), width=2.2 if c == f"{ch}_net" else 1.3)))
    fig.update_layout(title="2025 年淨值（扣每邊 7 bp）")
    C.show(C.style(fig, 460))
    rows = []
    for m in ["rule", "ridge", "lgbm"]:
        a, b = ho["table"][m], ho["ic"][m]
        rows.append({"模型": C.MODELS[m], "IC": b["ic"], "IC 下界": b["lo"], "IC 上界": b["hi"], "淨 Sharpe": a["sharpe"],
                     "年化報酬": a["ann_ret"] * 100, "最大回撤": a["max_dd"] * 100, "損益兩平 (bp)": a.get("breakeven_bps", np.nan)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, column_config={
        **{c: st.column_config.NumberColumn(format="%+.3f") for c in ["IC", "IC 下界", "IC 上界"]},
        "淨 Sharpe": st.column_config.NumberColumn(format="%.2f"),
        "年化報酬": st.column_config.NumberColumn(format="%+.1f%%"), "最大回撤": st.column_config.NumberColumn(format="%.1f%%"),
        "損益兩平 (bp)": st.column_config.NumberColumn(format="%.1f")})


# ==================================================================== 研究：規格
def page_spec():
    C.page_header("策略規格與資料健康", "回測前凍結的交易設定、驗證協定，以及資料時間點（PIT）檢查。")
    res = C.J("results")
    if res:
        f = res["falsified"]
        C.kpi_row([("採用模型", C.MODELS[res["chosen"]], "", ""), ("驗證期結論", res["pattern_validation"], "", ""),
                   ("IC CI 含 0？", "是" if f["ic_ci_contains_0"] else "否", "否＝訊號顯著", "down" if f["ic_ci_contains_0"] else "up"),
                   ("淨 Sharpe ≤ 0？", "是" if f["net_sharpe_le_0"] else "否", "是＝扣成本後不賺", "down" if f["net_sharpe_le_0"] else "up"),
                   ("Deflated Sharpe", f"{res['inference']['dsr']:.3f}", f"試驗數 {res['inference']['n_trials']}", "")])
    t1, t2, t3 = st.tabs(["setup.yaml（凍結設定）", "驗證協定", "資料健康 / PIT"])
    with t1:
        st.code((C.ROOT / "setup.yaml").read_text(encoding="utf-8"), language="yaml")
    with t2:
        st.markdown((C.ROOT / "protocol.md").read_text(encoding="utf-8"))
    with t3:
        if (C.OUT / "pit_table.csv").exists():
            st.dataframe(pd.read_csv(C.OUT / "pit_table.csv"), use_container_width=True, hide_index=True)
        if (C.OUT / "pit_leakage_by_feature.csv").exists():
            st.markdown("Ch.02 PITValidator：每個特徵與次日報酬的最大相關（門檻 0.30）")
            st.dataframe(pd.read_csv(C.OUT / "pit_leakage_by_feature.csv"), use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        for c, f in [(c1, "fig_knowledge_delay.png"), (c2, "fig_tradable_count.png")]:
            if (C.OUT / f).exists():
                c.image(str(C.OUT / f), use_container_width=True)


# ==================================================================== shell
PAGES = {
    "market": st.Page(page_market, title="市場總覽", icon=":material/dashboard:", url_path="market", default=True),
    "chart": st.Page(page_chart, title="K 線與比較", icon=":material/candlestick_chart:", url_path="chart"),
    "ranking": st.Page(page_ranking, title="每日選股排行", icon=":material/leaderboard:", url_path="ranking"),
    "backtest": st.Page(page_backtest, title="回測實驗室", icon=":material/science:", url_path="backtest"),
    "live": st.Page(page_live, title="即時訊號", icon=":material/sensors:", url_path="live"),
    "models": st.Page(page_models, title="模型診斷", icon=":material/insights:", url_path="models"),
    "risk": st.Page(page_risk, title="風險監控", icon=":material/shield:", url_path="risk"),
    "holdout": st.Page(page_holdout, title="2025 封存驗證", icon=":material/lock:", url_path="holdout"),
    "spec": st.Page(page_spec, title="策略規格", icon=":material/description:", url_path="spec"),
}

nav = st.navigation({"行情": [PAGES["market"], PAGES["chart"]],
                     "策略": [PAGES["ranking"], PAGES["backtest"], PAGES["live"]],
                     "研究": [PAGES["models"], PAGES["risk"], PAGES["holdout"], PAGES["spec"]]})
with st.sidebar:
    st.divider()
    ss.color_mode = st.radio("漲跌顏色", ["綠漲紅跌", "紅漲綠跌"], index=["綠漲紅跌", "紅漲綠跌"].index(ss.color_mode),
                             horizontal=True, help="國際/幣圈慣用綠漲紅跌；台股慣用紅漲綠跌")
    res = C.J("results")
    bt_ = C.J("backtest")
    st.markdown(
        f'<div class="panel" style="font-size:.8rem;line-height:1.7">'
        f'<b>系統狀態</b><br>'
        f'<span class="muted">幣種</span> {len(C.SYMS)} 檔 USDT 永續<br>'
        f'<span class="muted">行情資料</span> 2020-01 ～ 2025-12<br>'
        f'<span class="muted">驗證期</span> {bt_["window"][0]} ～ {bt_["window"][1]}<br>'
        f'<span class="muted">採用模型</span> {C.MODELS.get(res["chosen"], res["chosen"]) if res else "—"}<br>'
        f'<span class="muted">封存期 2025</span> {"已開啟（一次）" if (C.OUT / "HOLDOUT_OPENED.lock").exists() else "封存中"}<br>'
        f'<span class="muted">設定版本</span> {C.SETUP["setup_version"]}（{C.SETUP["frozen_at"]} 凍結）</div>',
        unsafe_allow_html=True)
    st.caption("前端只讀已存好的資料與結果，不訓練模型。")

C.inject_css()
st.markdown('<div class="topbar"><div class="brand"><span class="dot"></span>Crowding Terminal'
            '<span class="brand-sub">加密永續合約 · 擁擠度反轉交易系統</span></div>'
            '<div class="brand-sub">Binance USDT-M · 時間皆為 UTC</div></div>', unsafe_allow_html=True)
ticker_tape()
nav.run()
