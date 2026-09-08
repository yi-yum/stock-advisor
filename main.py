import os
import streamlit as st
import time
from stock_data import get_stock_data
from claude_analyzer import analyze_stock
from history import load_history, save_result, delete_record

st.set_page_config(
    page_title="股票 & 加密貨幣買入分析",
    page_icon="📈",
    layout="wide",
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
tab_analyze, tab_history = st.tabs(["🔍 新分析", "📁 歷史紀錄"])

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
            help="美股：AAPL｜台股：2330.TW｜加密貨幣：BTC",
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
        if analyze_btn:
            symbols = [s.strip().upper() for s in symbols_input.strip().split("\n") if s.strip()]
            if not symbols:
                st.error("請至少輸入一個代號")
            else:
                for i, symbol in enumerate(symbols):
                    st.markdown("---")
                    with st.spinner(f"取得 {symbol} 市場數據..."):
                        stock_data, df, err = get_stock_data(symbol)

                    if err:
                        st.error(err)
                        continue

                    asset_type = stock_data["asset_type"]
                    cur = stock_data["currency"]

                    # 多時框架偏向顯示
                    tf = stock_data.get("timeframes", {})
                    tf_labels = {"weekly": "週線", "daily": "日線", "h4": "4H", "h1": "1H"}
                    tf_cols = st.columns(4)
                    for idx, (key, label) in enumerate(tf_labels.items()):
                        d = tf.get(key)
                        if d:
                            color = "🟢" if d["bias"] == "偏多" else ("🔴" if d["bias"] == "偏空" else "🟡")
                            tf_cols[idx].metric(label, f"{color} {d['bias']}", f"RSI {d['rsi']}")
                        else:
                            tf_cols[idx].metric(label, "—")

                    with st.spinner(f"Gemini 正在分析 {symbol}..."):
                        result = analyze_stock(stock_data)

                    if result.get("error"):
                        st.error(result["error"])
                        continue

                    # 產生 Claude prompt
                    macd_dir = "金叉（多頭）" if stock_data['macd_histogram'] and stock_data['macd_histogram'] > 0 else "死叉（空頭）"

                    if asset_type == "taiwan":
                        ma_block = f"- MA20：{stock_data['ma20']}（{'上方' if stock_data.get('above_ma20') else '下方'}，斜率 {stock_data['ma20_slope']}%/5日）\n- MA60（季線）：{stock_data['ma60'] or '資料不足'}"
                        strategy_block = """可選策略（請從以下五種選一種最適合的）：
1. 動能追蹤 — 均線多頭排列，回踩月線買入
2. 均值回歸 — 季線支撐買入
3. 突破策略 — 等待放量突破整理區間高點
4. 逆勢反彈 — 超賣且接近重要支撐，分批承接
5. 觀望      — 趨勢不明或超買，等待訊號"""
                    elif asset_type == "us":
                        ma_block = f"- MA20：{stock_data['ma20']}（{'上方' if stock_data.get('above_ma20') else '下方'}，斜率 {stock_data['ma20_slope']}%/5日）\n- MA50：{stock_data['ma50'] or '資料不足'}（{'上方' if stock_data.get('above_ma50') else '下方' if stock_data['ma50'] else 'N/A'}）\n- MA200：{stock_data['ma200'] or '資料不足'}（{'上方' if stock_data.get('above_ma200') else '下方' if stock_data['ma200'] else 'N/A'}）"
                        strategy_block = """可選策略（請從以下五種選一種最適合的）：
1. 動能追蹤 — 趨勢明確，回踩 MA20 或 MA50 買入
2. 均值回歸 — 橫盤整理，支撐位買入
3. 突破策略 — 等待放量突破關鍵阻力後進場
4. 逆勢反彈 — 超賣且接近重要支撐，分批建倉
5. 觀望      — 多空不明或風險過高，等待時機"""
                    else:
                        ma_block = f"- MA20：{stock_data['ma20']}（{'上方' if stock_data.get('above_ma20') else '下方'}，斜率 {stock_data['ma20_slope']}%/5日）\n- MA50：{stock_data['ma50'] or '資料不足'}（{'上方' if stock_data.get('above_ma50') else '下方' if stock_data['ma50'] else 'N/A'}）"
                        strategy_block = """可選策略（請從以下五種選一種最適合的）：
1. 動能追蹤 — 趨勢明確，回踩 MA20 或關鍵支撐買入
2. 區間交易 — 盤整期在下緣買入
3. 突破策略 — 等待放量突破確認後進場
4. 分批建倉 — 超賣恐慌時分批佈局，控制整體倉位
5. 觀望      — 方向不明或過熱，等待確認"""

                    claude_prompt = f"""你是一位專業的技術分析師。請【先根據下方原始數據獨立分析】，再與 Gemini 的結論比較，最後給出你自己的判斷。

══════════════════════════════
【策略框架】（與 Gemini 使用相同框架，方便比較）
══════════════════════════════
{strategy_block}

══════════════════════════════
【原始技術數據】{stock_data['symbol']}
══════════════════════════════
價格
- 當前價格：{cur}{stock_data['current_price']}（今日 {stock_data['change_pct']}%）
- 5日報酬：{stock_data['return_5d']}%（5日前 {cur}{stock_data['price_5d_ago']}）

均線
{ma_block}

動能指標
- RSI(14)：{stock_data['rsi']}　背離：{stock_data.get('rsi_divergence', 'N/A')}
- MACD：{stock_data['macd']}，Signal：{stock_data['macd_signal']}，Histogram：{stock_data['macd_histogram']}（{macd_dir}）

布林通道
- 上軌：{cur}{stock_data.get('bb_upper')}　下軌：{cur}{stock_data.get('bb_lower')}
- 當前位置：{stock_data.get('bb_position', 'N/A')}

量能
- 量比：{stock_data['volume_ratio']}x
- OBV：{stock_data.get('obv_signal', 'N/A')}

波動率
- ATR(14)：{cur}{stock_data.get('atr')}（波動率 {stock_data.get('atr_pct')}%）
- ATR 建議停損：{cur}{stock_data.get('atr_stop')}

價格位置
- 52週高點：{cur}{stock_data['high_52w']}（距高點 {stock_data['pct_from_52w_high']}%）
- 52週低點：{cur}{stock_data['low_52w']}（距低點 +{stock_data['pct_from_52w_low']}%）
{f"- 下次財報：{stock_data['earnings_info']}" if stock_data.get('earnings_info') else ""}

══════════════════════════════
【Gemini 的分析結論】
══════════════════════════════
{result["analysis"]}

══════════════════════════════
【請依序回答】
══════════════════════════════
1. 你自己看完原始數據後，判斷的股票狀態和策略為何？
   {'（加密貨幣請同時判斷做多或做空方向）' if asset_type == 'crypto' else ''}
2. 你與 Gemini 的結論有哪些相同？哪些不同？為什麼？
3. 你的最終操作建議（進場價位、停損、目標價）
4. 持倉管理建議：
   📈 若目前獲利中 → 繼續持有 / 部分獲利了結 / 加碼？
   📉 若目前虧損中 → 停損 / 繼續持有 / 攤平？
   🎯 停利參考位與持倉停損位各為何？

請用繁體中文回答。"""

                    # 顯示結果
                    render_analysis(stock_data, result["analysis"], claude_prompt)

                    # 自動儲存
                    save_result(stock_data, result["analysis"], claude_prompt)
                    st.toast(f"✅ {symbol} 已自動儲存", icon="💾")

                    if i < len(symbols) - 1:
                        time.sleep(5)  # 免費版每分鐘15次，間隔5秒較安全

                st.success("✅ 分析完成，結果已儲存至歷史紀錄")
        else:
            st.info("在左側輸入代號，點擊「開始分析」")
            st.markdown("""
### 支援資產範例

| 類型 | 範例代號 |
|------|---------|
| 🇺🇸 美股 | `AAPL` `NVDA` `TSLA` `META` `MSFT` |
| 🇹🇼 台股 | `2330.TW` `2317.TW` `2454.TW` `2382.TW` |
| 🪙 加密貨幣 | `BTC` `ETH` `SOL` `BNB` `DOGE` |
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
            icon, label = ASSET_LABELS.get(record["asset_type"], ("📊", ""))
            cur = record["currency"]

            with st.expander(
                f"{icon} **{record['symbol']}** `{label}`　{cur}{record['current_price']}　{record['timestamp']}",
                expanded=False,
            ):
                col1, col2, col3 = st.columns(3)
                col1.metric("RSI(14)", record["rsi"])
                col2.metric("5日報酬", f"{record['return_5d']}%")
                col3.metric("距年高點", f"{record['pct_from_52w_high']}%")

                st.markdown(record["analysis"])

                with st.expander("📋 複製給 Claude 分析"):
                    st.code(record["claude_prompt"], language=None)

                if st.button(f"🗑️ 刪除 {record['symbol']} 紀錄", key=f"del_{record['symbol']}"):
                    delete_record(record["symbol"])
                    st.rerun()
