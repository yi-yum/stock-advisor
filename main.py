import os
from datetime import datetime
import streamlit as st
import time
from stock_data import get_stock_data, safe_round
from claude_analyzer import analyze_stock, analyze_btc_short, analyze_stocktwits
from history import load_history, save_result, delete_record, save_short_result
from sector_scan import scan_sectors
from btc_short import get_btc_short_data, SUPPORTED_COINS
from stocktwits import (fetch_posts, fetch_most_active, fetch_symbol_with_sentiment,
                        summarize_sentiment, build_st_gemini_message, build_trending_gemini_message)

st.set_page_config(
    page_title="股票 & 加密貨幣買入分析",
    page_icon="📈",
    layout="wide",
)

def gemini_download_btn(label: str, analysis: str, filename: str):
    """在 Gemini 分析結果下方加下載按鈕"""
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    st.download_button(
        label=f"💾 下載分析結果（.txt）",
        data=analysis.encode("utf-8"),
        file_name=f"{filename}_{ts}.txt",
        mime="text/plain",
        key=f"dl_{filename}_{ts}",
    )


ASSET_LABELS = {
    "us":     ("🇺🇸", "美股"),
    "taiwan": ("🇹🇼", "台股"),
    "crypto": ("🪙", "加密貨幣"),
}


def render_analysis(stock_data: dict, analysis: str, claude_prompt: str):
    """顯示分析結果（新分析和歷史紀錄共用）"""
    asset_type = stock_data["asset_type"]
    icon, label = ASSET_LABELS[asset_type]
    cur = stock_data["currency"]

    st.subheader(f"{icon} {stock_data['symbol']}  `{label}`")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("當前價格", f"{cur}{stock_data['current_price']}", f"{stock_data['change_pct']}%")
    col2.metric("RSI(14)", stock_data["rsi"],
                "超買" if stock_data["rsi"] > 70 else ("超賣" if stock_data["rsi"] < 30 else "中性"))
    col3.metric("量比", f"{stock_data['volume_ratio']}x",
                "放量" if stock_data["volume_ratio"] > 1.5 else ("縮量" if stock_data["volume_ratio"] < 0.7 else "正常"))
    col4.metric("5日報酬", f"{stock_data['return_5d']}%")
    col5.metric("距年高點", f"{stock_data['pct_from_52w_high']}%")

    st.markdown(analysis)

    with st.expander("📋 複製給 Claude 分析（點開 → 全選複製）"):
        st.code(claude_prompt, language=None)


# ── 標題 ─────────────────────────────────────────────
st.title("📈 買入時機分析系統")
st.caption("Gemini AI 自動診斷狀態 → 選擇策略 → 給出買入建議｜支援美股、台股、加密貨幣")

# ── 分頁 ─────────────────────────────────────────────
tab_analyze, tab_sector, tab_btc, tab_st, tab_history, tab_logic = st.tabs(["🔍 新分析", "🗺️ 板塊掃描", "⚡ 短線", "💬 社群情緒", "📁 歷史紀錄", "📖 判斷邏輯"])

# ════════════════════════════════════════════════════
# 新分析頁
# ════════════════════════════════════════════════════
with tab_analyze:
    col_input, col_main = st.columns([1, 3])

    with col_input:
        st.markdown("#### 輸入代號")
        symbols_input = st.text_area(
            "每行一個",
            value="AAPL\n2330.TW\nBTC",
            height=160,
            help="美股：AAPL｜台股上市：2330.TW｜台股上櫃：6547.TWO｜加密貨幣：BTC",
            label_visibility="collapsed",
        )
        analyze_btn = st.button("🔍 開始分析", type="primary", use_container_width=True)

        st.markdown("""
**代號格式**
| 市場 | 範例 |
|------|------|
| 美股 | `AAPL` |
| 台股 | `2330.TW` |
| 加密貨幣 | `BTC` |

**做多策略**
- 🚀 動能追蹤
- 🔄 均值回歸
- 💥 突破策略
- 🛡️ 逆勢反彈
- ⏸️ 觀望

**做空策略（加密貨幣）**
- 📉 趨勢做空
- 💣 跌破支撐做空

**持倉管理**
- 📈 獲利中：持有/了結/加碼
- 📉 虧損中：停損/持有/攤平
""")

    with col_main:
        # ── 抓取市場數據 ──────────────────────────────
        if analyze_btn:
            symbols = [s.strip().upper() for s in symbols_input.strip().split("\n") if s.strip()]
            if not symbols:
                st.error("請至少輸入一個代號")
            else:
                fetched = {}
                for symbol in symbols:
                    with st.spinner(f"取得 {symbol} 市場數據..."):
                        stock_data, df, err = get_stock_data(symbol)
                    fetched[symbol] = {"stock_data": stock_data, "err": err}
                st.session_state["analysis_fetched"] = fetched
                # 清除舊的 Gemini 結果
                st.session_state.pop("analysis_gemini", None)

        # ── 顯示技術數據 + Gemini 按鈕 ────────────────
        if "analysis_fetched" in st.session_state:
            fetched = st.session_state["analysis_fetched"]
            gemini_results = st.session_state.get("analysis_gemini", {})

            for symbol, item in fetched.items():
                st.markdown("---")
                if item["err"]:
                    st.error(item["err"])
                    continue

                stock_data = item["stock_data"]
                asset_type = stock_data["asset_type"]
                cur        = stock_data["currency"]

                # 多時框架偏向
                tf = stock_data.get("timeframes", {})
                tf_cols = st.columns(4)
                for idx, (key, label) in enumerate({"weekly":"週線","daily":"日線","h4":"4H","h1":"1H"}.items()):
                    d = tf.get(key)
                    if d:
                        color = "🟢" if d["bias"] == "偏多" else ("🔴" if d["bias"] == "偏空" else "🟡")
                        tf_cols[idx].metric(label, f"{color} {d['bias']}", f"RSI {d['rsi']}")
                    else:
                        tf_cols[idx].metric(label, "—")

                # ── 社群情緒（可選）─────────────────────
                sentiment_key = f"analysis_sentiment_{symbol}"
                sentiment_data = st.session_state.get(sentiment_key)

                st_col, gemini_col = st.columns([1, 2])
                if st_col.button(f"📊 抓取 StockTwits 情緒", key=f"st_fetch_{symbol}"):
                    with st.spinner(f"抓取 ${symbol} 社群討論..."):
                        fetched_sentiment = fetch_symbol_with_sentiment(symbol)
                    if fetched_sentiment:
                        st.session_state[sentiment_key] = fetched_sentiment
                        sentiment_data = fetched_sentiment
                    else:
                        st.warning(f"StockTwits 無 ${symbol} 的近期貼文")

                if sentiment_data:
                    mood_icon = {"偏多": "🟢", "偏空": "🔴", "分歧": "🟡", "無標記": "⚪"}
                    mood = sentiment_data.get("mood", "無標記")
                    bull = sentiment_data.get("bullish_pct")
                    bear = sentiment_data.get("bearish_pct")
                    total = sentiment_data.get("total", 0)
                    gemini_col.info(
                        f"{mood_icon.get(mood,'⚪')} 社群情緒：{mood}　"
                        f"{'看多 ' + str(bull) + '% / 看空 ' + str(bear) + '%' if bull else '無標記'}　"
                        f"（{total} 筆討論）"
                    )

                # Gemini 分析按鈕
                if st.button(f"🤖 Gemini 分析 {symbol}{'（含情緒面）' if sentiment_data else ''}", key=f"gemini_{symbol}"):
                    with st.spinner(f"Gemini 正在分析 {symbol}..."):
                        result = analyze_stock(stock_data, sentiment=sentiment_data)
                    if not st.session_state.get("analysis_gemini"):
                        st.session_state["analysis_gemini"] = {}
                    st.session_state["analysis_gemini"][symbol] = result

                # 顯示 Gemini 結果（若已分析）
                if symbol in gemini_results:
                    result = gemini_results[symbol]
                    if result.get("error"):
                        st.error(result["error"])
                    else:
                        macd_dir = "金叉（多頭）" if stock_data['macd_histogram'] and stock_data['macd_histogram'] > 0 else "死叉（空頭）"
                        if asset_type == "taiwan":
                            ma_block = f"- MA20：{stock_data['ma20']}（{'上方' if stock_data.get('above_ma20') else '下方'}，斜率 {stock_data['ma20_slope']}%/5日）\n- MA60（季線）：{stock_data['ma60'] or '資料不足'}"
                            strategy_block = "1. 動能追蹤\n2. 均值回歸\n3. 突破策略\n4. 逆勢反彈\n5. 觀望"
                        elif asset_type == "us":
                            ma_block = f"- MA20：{stock_data['ma20']}（{'上方' if stock_data.get('above_ma20') else '下方'}，斜率 {stock_data['ma20_slope']}%/5日）\n- MA50：{stock_data['ma50'] or '資料不足'}\n- MA200：{stock_data['ma200'] or '資料不足'}"
                            strategy_block = "1. 動能追蹤\n2. 均值回歸\n3. 突破策略\n4. 逆勢反彈\n5. 觀望"
                        else:
                            ma_block = f"- MA20：{stock_data['ma20']}（{'上方' if stock_data.get('above_ma20') else '下方'}，斜率 {stock_data['ma20_slope']}%/5日）\n- MA50：{stock_data['ma50'] or '資料不足'}"
                            strategy_block = "1. 動能追蹤\n2. 區間交易\n3. 突破策略\n4. 分批建倉\n5. 觀望"

                        claude_prompt = f"""你是一位專業的技術分析師。請【先根據下方原始數據獨立分析】，再與 Gemini 的結論比較，最後給出你自己的判斷。

══════════════════════════════
【策略框架】
══════════════════════════════
{strategy_block}

══════════════════════════════
【原始技術數據】{symbol}
══════════════════════════════
價格
- 當前價格：{cur}{stock_data['current_price']}（今日 {stock_data['change_pct']}%）
- 5日報酬：{stock_data['return_5d']}%

均線
{ma_block}

動能指標
- RSI(14)：{stock_data['rsi']}　背離：{stock_data.get('rsi_divergence', 'N/A')}
- MACD Histogram：{stock_data['macd_histogram']}（{macd_dir}）

K線型態（多時框架）
{chr(10).join([f"- {label}：{'、'.join(stock_data.get('candlestick_patterns', {}).get(key, ['資料不足']))}" for key, label in [('weekly','週線'),('daily','日線'),('h4','4H')]])}

布林通道：{stock_data.get('bb_position', 'N/A')}
量比：{stock_data['volume_ratio']}x　OBV：{stock_data.get('obv_signal', 'N/A')}
ATR 建議停損：{cur}{stock_data.get('atr_stop')}
52週：高 {cur}{stock_data['high_52w']}（{stock_data['pct_from_52w_high']}%）/ 低 {cur}{stock_data['low_52w']}
{f"下次財報：{stock_data['earnings_info']}" if stock_data.get('earnings_info') else ""}

消息面
{chr(10).join([f"- [{n['date']}] {n['title']}" for n in stock_data.get('news', [])]) or "- 無近期新聞"}

多時框架
{chr(10).join([f"- {label}：RSI {d['rsi']}　MACD {d['macd_direction']}　→ {d['bias']}" if (d := stock_data.get('timeframes', {}).get(key)) else f"- {label}：資料不足" for key, label in [('weekly','週線'),('daily','日線'),('h4','4H'),('h1','1H')]])}

══════════════════════════════
【Gemini 分析結論】
══════════════════════════════
{result["analysis"]}

══════════════════════════════
【請依序回答】
══════════════════════════════
1. 你看完原始數據後的判斷與策略？{'（含做多/做空方向）' if asset_type == 'crypto' else ''}
2. 你與 Gemini 的相同與不同之處？
3. 最終操作建議（進場、停損、目標）
4. 持倉管理（獲利中/虧損中各如何處理）

請用繁體中文回答。"""

                        render_analysis(stock_data, result["analysis"], claude_prompt)
                        gemini_download_btn("下載", result["analysis"], symbol)
                        save_result(stock_data, result["analysis"], claude_prompt)
                        st.toast(f"✅ {symbol} 已儲存", icon="💾")
        else:
            st.info("在左側輸入代號，點擊「開始分析」取得技術數據，再視需要點擊 Gemini 分析")
            st.markdown("""
| 類型 | 範例代號 |
|------|---------|
| 🇺🇸 美股 | `AAPL` `NVDA` `TSLA` `META` `MSFT` |
| 🇹🇼 台股上市 | `2330.TW` `2317.TW` `2454.TW` |
| 🇹🇼 台股上櫃 | `6547.TWO` `3231.TWO` |
| 🪙 加密貨幣 | `BTC` `ETH` `SOL` `BNB` `DOGE` |
""")

# ════════════════════════════════════════════════════
# 板塊掃描頁
# ════════════════════════════════════════════════════
with tab_sector:
    st.markdown("#### 🗺️ 美股板塊觸底雷達")
    st.caption("掃描各板塊龍頭股，依觸底訊號強度排名 — 分數越高代表越可能處於底部區間")

    scan_btn = st.button("🔄 開始掃描", type="primary")

    if scan_btn:
        with st.spinner("掃描各板塊龍頭股中，約需 30 秒..."):
            sector_results = scan_sectors()

        # 整體觸底排行（跨板塊前5）
        all_stocks = [s for stocks in sector_results.values() for s in stocks]
        all_stocks.sort(key=lambda x: x["score"], reverse=True)
        top5 = [s for s in all_stocks if s["score"] >= 5][:5]

        if top5:
            st.markdown("### 🏆 全市場觸底強訊號 Top")
            cols = st.columns(len(top5))
            for i, s in enumerate(top5):
                cols[i].metric(
                    f"{s['symbol']}",
                    f"分數 {s['score']}　RSI {s['rsi']}",
                    f"今日 {s['change']}%",
                )
            st.divider()

        # 各板塊展開顯示
        for sector, stocks in sector_results.items():
            if not stocks:
                continue

            best_score = stocks[0]["score"]
            badge = "🟢" if best_score >= 6 else "🟡" if best_score >= 3 else "⚪"

            with st.expander(f"{badge} **{sector}**　｜　最高觸底分 {best_score}　｜　{len(stocks)} 隻"):
                for s in stocks:
                    score_bar = "█" * s["score"] + "░" * max(0, 10 - s["score"])
                    c1, c2, c3, c4, c5 = st.columns([1.5, 1, 1, 1, 3])
                    c1.markdown(f"**{s['symbol']}**")
                    c2.markdown(f"${s['close']}　`{s['change']:+}%`")
                    c3.markdown(f"RSI {s['rsi']}")
                    c4.markdown(f"分數 **{s['score']}**　`{score_bar}`")
                    if s["signals"]:
                        c5.markdown("　".join(s["signals"]))
                    else:
                        c5.markdown("—")
                    st.divider()

        st.info("💡 點擊個股代號可至「新分析」頁做深入技術分析")
    else:
        st.info("點擊「開始掃描」分析各板塊龍頭股的觸底狀況")

# ════════════════════════════════════════════════════
# BTC 短線頁
# ════════════════════════════════════════════════════
with tab_btc:
    st.markdown("#### ⚡ BTC 短線訊號")
    st.caption("整合 Binance 籌碼數據（資金費率 / OI / 多空比）+ 多時框架技術分析（1H / 15M / 5M）")

    coin_col, anchor_col, len_col, btn_col = st.columns([1, 1, 1, 2])
    selected_coin   = coin_col.selectbox("幣種", SUPPORTED_COINS, key="short_coin")
    selected_anchor = anchor_col.selectbox("VWAP 錨點", ["週錨（本週一）", "日錨（今日00:00）"], key="short_anchor")
    vwap_length     = len_col.number_input("VWAP 長度", min_value=5, max_value=100, value=14, step=1, key="short_vwap_len",
                                           help="帶狀滾動標準差的計算視窗，對應 TradingView 的「長度」參數（預設 14）")
    anchor_val      = "week" if "週" in selected_anchor else "day"
    btc_btn = btn_col.button("🔄 更新訊號", type="primary", key="btc_scan", use_container_width=True)

    if btc_btn:
        with st.spinner(f"抓取 {selected_coin} Binance 籌碼數據 + 計算技術指標..."):
            st.session_state["btc_data"] = get_btc_short_data(selected_coin, anchor=anchor_val, vwap_length=int(vwap_length))
        st.session_state.pop("btc_gemini_result", None)  # 更新數據時清除舊分析

    if "btc_data" not in st.session_state:
        st.info("點擊「更新訊號」載入最新數據")
        st.markdown("""
**數據來源**
| 項目 | 來源 |
|------|------|
| K線 / 技術指標 | yfinance（BTC-USD） |
| 資金費率 | Binance Futures 公開 API |
| 未平倉量 OI | Binance Futures 公開 API |
| 多空比 | Binance Futures 公開 API |
""")
    else:
        btc = st.session_state["btc_data"]

        # ── 方向判斷 ──────────────────────────────
        dir_color = {"做多": "🟢", "偏多觀望": "🟡", "觀望": "⚪", "偏空觀望": "🟠", "做空": "🔴"}
        direction = btc["direction"]
        score = btc["score"]
        st.markdown(f"## {dir_color.get(direction, '⚪')} 當前方向：**{direction}**")

        # 分數說明列
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("訊號分數", f"{score:+d}", help="多方訊號 +1、空方訊號 −1，累加後判斷方向")
        sc2.metric("最大可能分數", f"+{6 + 3 + 1}",  help="籌碼面最多 ±6，VWAP ±1，技術面 3 個時框各 ±1")
        sc3.metric("方向門檻", "≥+4 做多 ／ ≤−4 做空", help="±2~3 為觀望偏向，±1 以內為觀望")
        st.divider()

        # ── 籌碼面 ────────────────────────────────
        st.markdown("### 📊 籌碼面")
        cm1, cm2, cm3 = st.columns(3)

        if btc["funding"]:
            fr = btc["funding"]["funding_rate"]
            fr_label = "偏高⚠️" if fr > 0.02 else ("為負✅" if fr < 0 else "中性")
            cm1.metric("資金費率", f"{fr}%", fr_label)
        else:
            cm1.metric("資金費率", "無法取得")

        if btc["oi"]:
            chg = btc["oi"]["oi_change_6h"]
            cm2.metric("未平倉量 OI", f"{btc['oi']['oi']:,.0f} BTC", f"6h {chg:+}%" if chg is not None else "")
        else:
            cm2.metric("未平倉量 OI", "無法取得")

        if btc["ls_ratio"]:
            ls = btc["ls_ratio"]
            cm3.metric("多空比", f"{ls['ratio']}",
                       f"多 {ls['long_pct']}% / 空 {ls['short_pct']}%")
        else:
            cm3.metric("多空比", "無法取得")

        st.divider()

        # ── 技術面（多時框架）─────────────────────
        st.markdown("### 📈 技術面")
        tf_labels = [("1h", "1H"), ("15m", "15M"), ("5m", "5M")]
        tf_cols = st.columns(3)

        for i, (key, label) in enumerate(tf_labels):
            d = btc["tf_data"].get(key)
            if d:
                bias_icon = "🟢" if d["bias"] == "偏多" else "🔴"
                tf_cols[i].markdown(f"**{label}** {bias_icon} {d['bias']}")
                tf_cols[i].markdown(f"RSI `{d['rsi']}`")
                tf_cols[i].markdown(f"MACD {d['macd_status']}")
                tf_cols[i].markdown(f"布林 {d['bb_position']}")
                patterns = [p for p in d["patterns"] if p != "無明顯K線型態"]
                if patterns:
                    tf_cols[i].markdown(f"K線 `{'、'.join(patterns)}`")
            else:
                tf_cols[i].markdown(f"**{label}** — 資料不足")

        st.divider()

        # ── 錨定 VWAP（15m）────────────────────────
        vwap = btc.get("vwap_data")
        if vwap:
            st.markdown(f"### 📐 錨定 VWAP（15m {vwap.get('anchor_label','')}，錨點 {vwap.get('anchor_time','')}，帶狀長度 {vwap.get('length',14)}，共 {vwap.get('bars',0)} 根）")
            bias_icon = "🟢" if vwap["bias"] == "偏多" else ("🔴" if vwap["bias"] == "偏空" else "🟡")
            st.markdown(f"{bias_icon} **{vwap['position']}**　偏離 VWAP `{vwap['diff_pct']:+}%`　1σ = `${vwap['std']:,}`")

            v1, v2, v3, v4, v5 = st.columns(5)
            v1.metric("+2σ",  f"${vwap['upper_2']:,}", f"+{safe_round((vwap['upper_2']-vwap['vwap'])/vwap['vwap']*100,2)}%")
            v2.metric("+1σ",  f"${vwap['upper_1']:,}", f"+{safe_round((vwap['upper_1']-vwap['vwap'])/vwap['vwap']*100,2)}%")
            v3.metric("VWAP", f"${vwap['vwap']:,}")
            v4.metric("-1σ",  f"${vwap['lower_1']:,}", f"{safe_round((vwap['lower_1']-vwap['vwap'])/vwap['vwap']*100,2)}%")
            v5.metric("-2σ",  f"${vwap['lower_2']:,}", f"{safe_round((vwap['lower_2']-vwap['vwap'])/vwap['vwap']*100,2)}%")

            st.markdown("**🟢 做多方案**（回踩 VWAP 進場）")
            vl1, vl2, vl3, vl4 = st.columns(4)
            vl1.metric("進場",   f"${vwap['long_entry']:,}",  "VWAP")
            vl2.metric("停損",   f"${vwap['long_stop']:,}",   "VWAP − 1σ")
            vl3.metric("目標",   f"${vwap['long_target']:,}", "VWAP + 2σ")
            vl4.metric("風報比", f"1 : {vwap['rr_long']}")

            st.markdown("**🔴 做空方案**（反彈 VWAP 進場）")
            vs1, vs2, vs3, vs4 = st.columns(4)
            vs1.metric("進場",   f"${vwap['short_entry']:,}",  "VWAP")
            vs2.metric("停損",   f"${vwap['short_stop']:,}",   "VWAP + 1σ")
            vs3.metric("目標",   f"${vwap['short_target']:,}", "VWAP − 2σ")
            vs4.metric("風報比", f"1 : {vwap['rr_short']}")
        else:
            st.warning("VWAP 資料無法取得")

        st.divider()

        # ── 進場參考 ──────────────────────────────
        st.markdown("### 🎯 ATR 停損參考（1H）")
        st.markdown("**🟢 做多**")
        lg1, lg2, lg3, lg4 = st.columns(4)
        lg1.metric("現價參考",  f"${btc['entry_ref']:,}"   if btc["entry_ref"]   else "—")
        lg2.metric("停損",      f"${btc['stop_long']:,}"   if btc["stop_long"]   else "—")
        lg3.metric("目標價",    f"${btc['target_long']:,}" if btc["target_long"] else "—")
        lg4.metric("風報比",    f"1 : {btc['rr_long']}"   if btc["rr_long"]     else "—")

        st.markdown("**🔴 做空**")
        sh1, sh2, sh3, sh4 = st.columns(4)
        sh1.metric("現價參考",  f"${btc['entry_ref']:,}"    if btc["entry_ref"]    else "—")
        sh2.metric("停損",      f"${btc['stop_short']:,}"   if btc["stop_short"]   else "—")
        sh3.metric("目標價",    f"${btc['target_short']:,}" if btc["target_short"] else "—")
        sh4.metric("風報比",    f"1 : {btc['rr_short']}"   if btc["rr_short"]     else "—")

        st.divider()

        # ── 訊號清單 ──────────────────────────────
        st.markdown("### 📋 訊號明細")
        signal_icon  = {"多": "🟢", "空": "🔴", "中": "⚪"}
        signal_score = {"多": "+1", "空": "−1", "中": "0"}
        running = 0
        for side, msg in btc["signals"]:
            if side == "多":
                running += 1
            elif side == "空":
                running -= 1
            col_a, col_b, col_c = st.columns([0.5, 5, 1])
            col_a.markdown(signal_icon.get(side, "⚪"))
            col_b.markdown(msg)
            col_c.markdown(f"`{signal_score.get(side, '0')}` → **{running:+d}**")

        st.divider()

        # ── Gemini 分析（累積歷史版）────────────────
        st.markdown("### 🤖 Gemini 短線分析")

        history_key = f"btc_gemini_history_{btc.get('coin','BTC')}"
        gemini_history = st.session_state.get(history_key, [])

        btn_col1, btn_col2 = st.columns([2, 1])
        round_label = f"✨ 第 {len(gemini_history)+1} 次 Gemini 分析"
        if btn_col1.button(round_label, key="btc_gemini"):
            with st.spinner("Gemini 分析中..."):
                result = analyze_btc_short(btc, history=gemini_history if gemini_history else None)
            if not result["error"]:
                entry = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "analysis":  result["analysis"],
                    "snapshot": {
                        "entry_ref": btc.get("entry_ref"),
                        "direction": btc.get("direction"),
                        "score":     btc.get("score", 0),
                    },
                }
                gemini_history.append(entry)
                st.session_state[history_key] = gemini_history
            else:
                st.error(result["error"])

        if btn_col2.button("🗑️ 清除歷史", key="btc_gemini_clear"):
            st.session_state.pop(history_key, None)
            gemini_history = []
            st.rerun()

        if gemini_history:
            st.caption(f"共 {len(gemini_history)} 次分析，最新在上")

            for i, entry in enumerate(reversed(gemini_history)):
                round_n = len(gemini_history) - i
                snap = entry.get("snapshot", {})
                snap_str = (
                    f"現價 ${snap.get('entry_ref','?')}　{snap.get('direction','?')}　分數 {snap.get('score',0):+d}"
                    if snap else ""
                )
                is_latest = (i == 0)
                with st.expander(
                    f"第 {round_n} 次分析　{entry['timestamp']}　`{snap_str}`",
                    expanded=is_latest,
                ):
                    st.markdown(entry["analysis"])

                    # Claude 複製區塊（只在最新一次顯示）
                    if is_latest:
                        tf_data = btc.get("tf_data", {})
                        tf_lines = "\n".join([
                            f"- {label}：RSI {d['rsi']}　MACD {d['macd_status']}　布林 {d['bb_position']}　→ {d['bias']}"
                            if (d := tf_data.get(key)) else f"- {label}：資料不足"
                            for key, label in [("1h","1H"),("15m","15M"),("5m","5M")]
                        ])
                        funding = btc.get("funding") or {}
                        oi      = btc.get("oi") or {}
                        ls      = btc.get("ls_ratio") or {}

                        history_summary = "\n".join([
                            f"第 {j+1} 次（{h['timestamp']}）：{h['analysis'][:200]}..."
                            for j, h in enumerate(gemini_history[:-1])
                        ]) if len(gemini_history) > 1 else "（本次為第一次分析）"

                        claude_btc_prompt = f"""你是一位專業的 BTC 短線交易員。請【先根據下方原始數據獨立分析】，再與 Gemini 的結論比較，最後給出你自己的判斷。

══════════════════════════════
【籌碼面數據】
══════════════════════════════
- 資金費率：{funding.get('funding_rate', 'N/A')}%
- OI 6h 變化：{oi.get('oi_change_6h', 'N/A')}%
- 多空比：{ls.get('ratio', 'N/A')}（多 {ls.get('long_pct', 'N/A')}% / 空 {ls.get('short_pct', 'N/A')}%）

══════════════════════════════
【技術面數據】
══════════════════════════════
{tf_lines}

══════════════════════════════
【ATR 進場參考】
══════════════════════════════
現價：${btc.get('entry_ref', 'N/A')}
🟢 做多：停損 ${btc.get('stop_long', 'N/A')}　目標 ${btc.get('target_long', 'N/A')}　風報比 1:{btc.get('rr_long', 'N/A')}
🔴 做空：停損 ${btc.get('stop_short', 'N/A')}　目標 ${btc.get('target_short', 'N/A')}　風報比 1:{btc.get('rr_short', 'N/A')}

══════════════════════════════
【前次 Gemini 分析摘要】
══════════════════════════════
{history_summary}

══════════════════════════════
【本次（第 {len(gemini_history)} 次）Gemini 分析結論】
══════════════════════════════
{entry['analysis']}

══════════════════════════════
【請依序回答】
══════════════════════════════
1. 你看完數據後，籌碼面與技術面各自的判斷為何？
2. 你與 Gemini 的結論有哪些相同？哪些不同？為什麼？
3. 你的做多方案：進場價、停損、目標一、目標二、風報比
4. 你的做空方案：進場價、停損、目標一、目標二、風報比
5. 目前優先方向與理由（一句話）

請用繁體中文回答。"""

                        with st.expander("📋 複製給 Claude 做第二意見"):
                            st.code(claude_btc_prompt, language=None)

                        gemini_download_btn("下載", entry["analysis"], f"{btc.get('coin')}_short")
                        save_short_result(btc, entry["analysis"], claude_btc_prompt)
                        st.toast(f"✅ {btc.get('coin')} 第 {len(gemini_history)} 次分析已儲存", icon="💾")
        st.markdown("""
**數據來源**
| 項目 | 來源 |
|------|------|
| K線 / 技術指標 | yfinance（BTC-USD） |
| 資金費率 | Binance Futures 公開 API |
| 未平倉量 OI | Binance Futures 公開 API |
| 多空比 | Binance Futures 公開 API |
""")

# ════════════════════════════════════════════════════
# 社群情緒頁（StockTwits）
# ════════════════════════════════════════════════════
with tab_st:
    st.markdown("#### 💬 社群熱度雷達（StockTwits）")
    st.caption("掃描今日熱門討論股票，依討論量排名，並可展開查看各股情緒與貼文")

    # 自訂清單 or 預設清單
    with st.expander("⚙️ 自訂掃描清單（選填，留空使用預設）"):
        custom_input = st.text_area(
            "每行或用逗號分隔，輸入股票代號",
            placeholder="AAPL, NVDA, TSLA ...",
            height=80, key="st_custom"
        )

    trend_btn = st.button("🔄 開始掃描", type="primary", key="st_trending")

    if trend_btn:
        if custom_input.strip():
            symbols = [s.strip().upper() for s in custom_input.replace(",", "\n").split("\n") if s.strip()]
            source_label = f"自訂清單（{len(symbols)} 支）"
        else:
            with st.spinner("從 Yahoo Finance 抓取今日最活躍股票..."):
                symbols = fetch_most_active(20)
            if not symbols:
                st.error("無法取得 Yahoo Finance 最活躍股票，請稍後重試或使用自訂清單")
                st.stop()
            source_label = f"Yahoo Finance 今日最活躍（{len(symbols)} 支）"

        st.session_state["st_source_label"] = source_label

        progress = st.progress(0, text="掃描 StockTwits 社群討論中...")
        results  = []
        for i, sym in enumerate(symbols):
            data = fetch_symbol_with_sentiment(sym)
            if data:
                results.append(data)
            progress.progress((i + 1) / len(symbols), text=f"抓取 ${sym}... ({i+1}/{len(symbols)})")
        progress.empty()

        results.sort(key=lambda x: x["total"], reverse=True)
        st.session_state["st_trending_data"] = results
        st.session_state.pop("st_trending_gemini", None)
        st.session_state.pop("st_detail", None)

    if "st_trending_data" in st.session_state:
        results = st.session_state["st_trending_data"]

        if not results:
            st.warning("未取得任何股票數據")
        else:
            # ── 熱門排行表 ────────────────────────
            source = st.session_state.get("st_source_label", "")
            st.markdown(f"### 📊 社群討論排行　`{source}`")
            st.caption("依 StockTwits 近期貼文數排序，數量越多代表討論越熱")
            mood_icon = {"偏多": "🟢", "偏空": "🔴", "分歧": "🟡", "無標記": "⚪"}

            for r in results:
                sym      = r["symbol"]
                mood     = r["mood"]
                icon     = mood_icon.get(mood, "⚪")
                bull_str = f"🟢 {r['bullish_pct']}%" if r.get("bullish_pct") else "—"
                bear_str = f"🔴 {r['bearish_pct']}%" if r.get("bearish_pct") else "—"
                header   = f"**${sym}**　{icon} {mood}　{bull_str} / {bear_str}　（{r['total']} 筆討論）"

                with st.expander(header):
                    d1, d2, d3 = st.columns(3)
                    d1.metric("整體氛圍", f"{icon} {mood}")
                    d2.metric("看多", f"{r['bullish_pct']}%" if r.get("bullish_pct") else "—", f"{r['bullish']} 筆")
                    d3.metric("看空", f"{r['bearish_pct']}%" if r.get("bearish_pct") else "—", f"{r['bearish']} 筆")

                    if st.button(f"🤖 Gemini 分析 ${sym} 社群", key=f"st_sym_gemini_{sym}"):
                        with st.spinner(f"分析 ${sym} 社群貼文..."):
                            msg    = build_st_gemini_message(sym, r["posts"], r)
                            result = analyze_stocktwits(sym, msg)
                        st.session_state[f"st_sym_result_{sym}"] = result

                    key = f"st_sym_result_{sym}"
                    if key in st.session_state:
                        res = st.session_state[key]
                        if res["error"]:
                            st.error(res["error"])
                        else:
                            st.markdown(res["analysis"])
                            gemini_download_btn("下載", res["analysis"], f"{sym}_stocktwits")

                    with st.expander(f"📄 原始貼文（{len(r['posts'])} 筆）"):
                        for p in r["posts"]:
                            badge = "🟢" if p["sentiment"] == "Bullish" else ("🔴" if p["sentiment"] == "Bearish" else "⚪")
                            st.markdown(f"{badge} **@{p['user']}** `{p['created']}`　👍 {p['likes']}")
                            st.markdown(f"> {p['body']}")
                            st.markdown("---")

            st.divider()

            # ── Gemini 總覽分析 ───────────────────
            if st.button("🤖 Gemini 分析所有熱門股票趨勢", key="st_trend_gemini_btn"):
                with st.spinner("Gemini 分析中..."):
                    msg    = build_trending_gemini_message(results)
                    result = analyze_stocktwits("熱門股票總覽", msg)
                st.session_state["st_trending_gemini"] = result

            if "st_trending_gemini" in st.session_state:
                r = st.session_state["st_trending_gemini"]
                if r["error"]:
                    st.error(r["error"])
                else:
                    st.markdown("#### 🤖 Gemini 市場熱點分析")
                    st.markdown(r["analysis"])
                    gemini_download_btn("下載", r["analysis"], "trending_stocktwits")
    else:
        st.info("點擊「掃描今日熱門股票」查看近期討論最熱的股票")
        st.markdown("""
**說明**
- 自動抓取 StockTwits 今日討論量最高的美股
- 每支股票顯示看多/看空比例
- 可請 Gemini 一次分析所有熱門股票的市場主題
- 也可點「展開」查看單支股票的詳細貼文
""")

# ════════════════════════════════════════════════════
# 歷史紀錄頁
# ════════════════════════════════════════════════════
with tab_history:
    history = load_history()

    if not history:
        st.info("尚無歷史紀錄，請先進行分析")
    else:
        st.caption(f"共 {len(history)} 筆紀錄，同一代號只保留最新一筆")
        if os.environ.get("STREAMLIT_SHARING") or not os.path.exists(".env"):
            st.warning("⚠️ 雲端版的歷史紀錄僅在本次執行期間保存，重新部署後會清空")

        for record in history:
            is_short = record.get("asset_type") == "short"

            if is_short:
                icon, label = "⚡", f"{record.get('coin', '')} 短線"
            else:
                icon, label = ASSET_LABELS.get(record["asset_type"], ("📊", ""))

            cur = record["currency"]
            price_str = f"{cur}{record['current_price']}" if record.get("current_price") else ""

            with st.expander(
                f"{icon} **{record['symbol']}** `{label}`　{price_str}　{record['timestamp']}",
                expanded=False,
            ):
                if is_short:
                    sc1, sc2, sc3, sc4 = st.columns(4)
                    sc1.metric("方向",     record.get("direction", "—"))
                    sc2.metric("資金費率", f"{record.get('funding_rate', '—')}%")
                    sc3.metric("多空比",   record.get("ls_ratio", "—"))
                    sc4.metric("風報比",   f"多1:{record.get('rr_long','—')} / 空1:{record.get('rr_short','—')}")
                else:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("RSI(14)",  record["rsi"])
                    col2.metric("5日報酬",  f"{record['return_5d']}%")
                    col3.metric("距年高點", f"{record['pct_from_52w_high']}%")

                st.markdown(record["analysis"])

                with st.expander("📋 複製給 Claude 分析"):
                    st.code(record["claude_prompt"], language=None)

                ts_key = record["timestamp"].replace(" ", "_").replace(":", "")
                if st.button(f"🗑️ 刪除此筆紀錄", key=f"del_{record['symbol']}_{ts_key}"):
                    delete_record(record["symbol"], record["timestamp"])
                    st.rerun()

# ════════════════════════════════════════════════════
# 判斷邏輯頁
# ════════════════════════════════════════════════════
with tab_logic:
    st.markdown("## 📖 所有判斷邏輯說明")
    st.caption("本頁說明系統各模組所使用的評分與判斷規則")

    # ── 1. 多時框架偏向（新分析頁）────────────────────
    with st.expander("🔍 新分析｜多時框架偏向判斷", expanded=True):
        st.markdown("""
每個時框（週線 / 日線 / 4H / 1H）獨立計算 4 項條件，加總後判斷偏向：

| 條件 | 符合得分 |
|------|---------|
| 收盤價 > MA20 | +1 |
| 收盤價 > MA50 | +1 |
| MACD Histogram > 0（金叉） | +1 |
| RSI > 50 | +1 |

**結論：** 得分 ≥ 3 → 偏多 ／ 得分 ≤ 1 → 偏空 ／ 其餘 → 中性
""")

    # ── 2. 技術指標說明（新分析頁）───────────────────
    with st.expander("🔍 新分析｜技術指標判讀規則"):
        st.markdown("""
**RSI(14)**
| 數值 | 意義 |
|------|------|
| > 70 | 超買，留意回調風險 |
| 30 ~ 70 | 正常區間 |
| < 30 | 超賣，留意反彈機會 |

**RSI 背離（近30根K棒前後各15根比較）**
| 型態 | 條件 | 訊號 |
|------|------|------|
| 底背離 | 後半價格創新低，但 RSI 未創新低（差距 > 3） | 看漲 |
| 頂背離 | 後半價格創新高，但 RSI 未創新高（差距 > 3） | 看跌 |

**MACD（12/26/9）**
| 狀態 | 條件 |
|------|------|
| 金叉 | Histogram > 0 |
| 死叉 | Histogram < 0 |

**布林通道（20日，2倍標準差）**
| 位置 | 條件 |
|------|------|
| 觸及上軌（超買） | 收盤 ≥ 上軌 × 99% |
| 觸及下軌（超賣） | 收盤 ≤ 下軌 × 101% |
| 中軌上方（偏強） | 收盤 > 中軌 |
| 中軌下方（偏弱） | 收盤 ≤ 中軌 |

**OBV 量能訊號**
| 條件 | 訊號 |
|------|------|
| OBV 上升 + 價格上漲 | 量價齊升（健康） |
| OBV 上升 + 價格下跌 | 底部蓄勢（看漲） |
| OBV 下降 + 價格上漲 | 量能背離，可能假突破（警示） |
| OBV 下降 + 價格下跌 | 量價齊跌（確認下跌） |

**ATR 停損（14日）**
- 建議停損 = 收盤價 − 1.5 × ATR（僅供參考，非絕對）
""")

    # ── 3. K 線型態（所有頁面共用）───────────────────
    with st.expander("🕯️ K線型態判斷條件（日線 / 週線 / 4H / 1H 共用）"):
        st.markdown("""
以下條件使用最近 3 根 K 棒，變數定義：
- **Body**（實體）= |收盤 − 開盤|
- **Range**（全幅）= 最高 − 最低
- **上影線** = 最高 − max(收盤, 開盤)
- **下影線** = min(收盤, 開盤) − 最低

| 型態 | 條件 | 訊號方向 |
|------|------|---------|
| 十字星 | Body / Range < 10% | 中性，等待確認 |
| 錘頭 | Body/Range < 35%，下影線 ≥ 2×Body，上影線 ≤ 0.5×Body | 看漲（底部） |
| 射擊之星 | Body/Range < 35%，上影線 ≥ 2×Body，下影線 ≤ 0.5×Body | 看跌（頂部） |
| 上吊線 | 同錘頭條件 + 陰線 | 看跌（高位警示） |
| 多頭吞噬 | 前根陰線，後根陽線完全包覆前根 | 看漲反轉 |
| 空頭吞噬 | 前根陽線，後根陰線完全包覆前根 | 看跌反轉 |
| 早晨之星 | c2 陰線 → c1 小實體 → c0 陽線且收盤超過 c2 中點 | 強力看漲反轉 |
| 黃昏之星 | c2 陽線 → c1 小實體 → c0 陰線且收盤低於 c2 中點 | 強力看跌反轉 |
""")

    # ── 4. 板塊觸底評分（板塊掃描頁）────────────────
    with st.expander("🗺️ 板塊掃描｜觸底評分規則"):
        st.markdown("""
對每隻龍頭股計算以下分數，總分越高代表越接近底部：

**RSI 超賣**
| 條件 | 分數 |
|------|------|
| RSI < 30 | +3 |
| RSI < 40 | +2 |
| RSI < 50 | +1 |

**接近 52 週低點**
| 距低點距離 | 分數 |
|-----------|------|
| < 5% | +3 |
| < 15% | +2 |
| < 25% | +1 |

**其他訊號**
| 條件 | 分數 |
|------|------|
| RSI 底背離 | +2 |
| MACD 死叉收窄（空頭動能減弱） | +2 |
| MACD 剛形成金叉 | +3 |
| 出現看漲 K 線型態（錘頭、多頭吞噬、早晨之星） | +1 |
| 價跌縮量（底部蓄勢） | +1 |

**顯示分級：** 🟢 ≥ 6 高度觸底 ／ 🟡 3–5 觀察名單 ／ ⚪ < 3 無訊號
""")

    # ── 5. 短線綜合訊號評分（短線頁）──────────────
    with st.expander("⚡ 短線｜綜合訊號評分規則"):
        st.markdown("""
**籌碼面（Binance）**

| 條件 | 分數 |
|------|------|
| 資金費率 > 0.05%（多方極度擁擠） | −2 |
| 資金費率 > 0.02% | −1 |
| 資金費率 < −0.02%（空方付費） | +2 |
| 資金費率 < 0% | +1 |
| OI 6h 增加 + 技術面偏多 | +1 |
| OI 6h 增加 + 技術面偏空 | −1 |
| 多空比 > 1.8（多方極度擁擠，逆向） | −2 |
| 多空比 > 1.4 | −1 |
| 多空比 < 0.7（空方極度擁擠，逆向） | +2 |
| 多空比 < 0.9 | +1 |

**15m 錨定 VWAP（仿 TradingView VWAP 自動錨定）**

錨點選項：週錨（本週一 00:00 UTC）或日錨（今日 00:00 UTC）
從錨點起累積計算所有 15m K 棒的成交量加權平均價。

| 條件 | 分數 |
|------|------|
| 現價偏離 VWAP > +0.3% | +1 |
| 現價偏離 VWAP < −0.3% | −1 |
| 現價在 VWAP ±0.3% 以內 | 0 |

**VWAP 進場邏輯**
| 方向 | 進場 | 停損 | 目標 | 風報比 |
|------|------|------|------|--------|
| 做多 | 回踩至 VWAP | VWAP − 1σ | VWAP + 2σ | 1:2 |
| 做空 | 反彈至 VWAP | VWAP + 1σ | VWAP − 2σ | 1:2 |

σ = 加權標準差：√(Σ(volume × (典型價 − VWAP)²) / Σ(volume))

**技術面（每個時框各 ±1）**

每個時框（1H / 15M / 5M）計算 3 項條件：RSI > 50、MACD Histogram > 0、收盤 > 布林中軌
符合 ≥ 2 項 → 偏多（+1）／ 符合 < 2 項 → 偏空（−1）

**最終方向判斷**
| 總分 | 方向 |
|------|------|
| ≥ +4 | 🟢 做多 |
| +2 ~ +3 | 🟡 偏多觀望 |
| −1 ~ +1 | ⚪ 觀望 |
| −3 ~ −2 | 🟠 偏空觀望 |
| ≤ −4 | 🔴 做空 |

**進場參考計算**
| 項目 | 計算方式 |
|------|---------|
| 停損（做多） | 現價 − 1.5 × ATR(1H) |
| 停損（做空） | 現價 + 1.5 × ATR(1H) |
| 目標價 | 近50根1H K棒的最近擺盪高/低點；找不到時退回 現價 ± 2.0 × ATR |
| 風報比 | (目標 − 現價) ÷ (現價 − 停損) |
""")
