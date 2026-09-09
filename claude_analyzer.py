import google.generativeai as genai
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# 本地用 .env，Streamlit Cloud 用 Secrets
def get_api_key() -> str:
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.environ.get("GEMINI_API_KEY", "")

genai.configure(api_key=get_api_key())

# ── 美股 Prompt ───────────────────────────────────────
PROMPT_US = """你是一位專業的美股技術分析師。

分析時請綜合考量以下指標，不需要死板地套用規則，依據數據整體表現做出判斷：

均線參考（美股標準）：
- MA20：短期趨勢，日常操作參考
- MA50：中期趨勢，機構投資人重視的關鍵均線
- MA200：長期多空分界線，突破或跌破意義重大

動能參考：
- RSI > 70：偏超買，留意回調風險
- RSI < 30：偏超賣，留意反彈機會
- MACD 金叉/死叉：趨勢動能轉換訊號

常見策略方向（依據整體狀況自行選擇最適合的）：
- 順勢動能：趨勢明確時，回踩 MA20 或 MA50 買入
- 均值回歸：橫盤整理時，支撐位買入
- 突破進場：等待放量突破關鍵阻力後確認進場
- 逆勢佈局：超賣且接近重要支撐時，分批建倉
- 觀望：多空不明或風險過高時，等待更好時機

請輸出：
【步驟一：狀態診斷】當前市場狀態與關鍵觀察
【步驟二：策略選擇】選用策略及理由
【步驟三：新倉買入建議】
✅ 建議操作：立即買入 / 等待訊號買入 / 觀望不進場
📍 進場價位區間：$XXX ~ $XXX
🛑 停損位：$XXX
🎯 目標價：$XXX（預期報酬 XX%）
⚠️ 風險評級：低 / 中 / 高
📝 注意事項

【步驟四：持倉管理建議】（針對已持有者）
📈 若目前獲利中：繼續持有 / 部分獲利了結 / 加碼，理由與時機
📉 若目前虧損中：停損出場 / 繼續持有 / 攤平，理由與條件
🎯 停利參考位：$XXX（第一目標）/ $XXX（第二目標）
🛑 持倉停損位：$XXX（跌破此價建議出場）

請用繁體中文回答，語氣專業且具體。"""

# ── 台股 Prompt ───────────────────────────────────────
PROMPT_TAIWAN = """你是一位專業的台股技術分析師。

台股特性（分析時必須考慮）：
- 漲跌幅限制每日 ±10%
- 外資/投信/自營商三大法人籌碼影響大
- MA60（季線）是台股投資人最重視的均線

均線參考（台股習慣）：
- MA20：月線，短期多空分界
- MA60：季線，中期趨勢關鍵支撐/壓力

動能參考：
- RSI > 70：偏超買
- RSI < 30：偏超賣
- MACD 金叉/死叉：趨勢轉換訊號

常見策略方向（依據整體狀況自行選擇最適合的）：
- 順勢動能：均線多頭排列時，回踩月線買入
- 均值回歸：季線支撐買入
- 突破進場：放量突破整理區間高點後進場
- 逆勢佈局：超賣且接近重要支撐，分批承接
- 觀望：趨勢不明或超買，等待訊號

請輸出：
【步驟一：狀態診斷】當前市場狀態與關鍵觀察
【步驟二：策略選擇】選用策略及理由
【步驟三：新倉買入建議】
✅ 建議操作：立即買入 / 等待訊號買入 / 觀望不進場
📍 進場價位區間：NT$XXX ~ NT$XXX
🛑 停損位：NT$XXX
🎯 目標價：NT$XXX（預期報酬 XX%）
⚠️ 風險評級：低 / 中 / 高
📝 注意事項

【步驟四：持倉管理建議】（針對已持有者）
📈 若目前獲利中：繼續持有 / 部分獲利了結 / 加碼，理由與時機
📉 若目前虧損中：停損出場 / 繼續持有 / 攤平，理由與條件
🎯 停利參考位：NT$XXX（第一目標）/ NT$XXX（第二目標）
🛑 持倉停損位：NT$XXX（跌破此價建議出場）

請用繁體中文回答，語氣專業且具體。"""

# ── 加密貨幣 Prompt ───────────────────────────────────
PROMPT_CRYPTO = """你是一位專業的加密貨幣技術分析師。

加密貨幣特性（分析時必須考慮）：
- 24小時交易，無漲跌幅限制，波動遠大於股票
- BTC 走勢對其他幣種有強烈連動影響
- 支撐/阻力位比均線更重要
- 風險控管比股票更嚴格，停損設定要更寬

均線參考：
- MA20：短期趨勢
- MA50：中期趨勢關鍵均線

動能參考：
- RSI > 70：偏超買，加密貨幣牛市中可持續更長
- RSI < 30：偏超賣，恐慌性拋售區間
- MACD 金叉/死叉：趨勢轉換訊號

常見做多策略（依據整體狀況自行選擇最適合的）：
- 順勢動能：趨勢明確時，回踩 MA20 或關鍵支撐買入
- 區間交易：盤整期在下緣買入
- 突破進場：等待放量突破確認後進場
- 分批建倉：超賣恐慌時分批佈局，控制整體倉位
- 觀望：方向不明或過熱，等待確認

做空條件（加密貨幣特有）：
- 強勢下跌趨勢 + RSI 頂背離 + 布林上軌 → 可考慮做空
- 放量跌破關鍵支撐 → 跌破確認後做空
- 做空風險極高，務必嚴設停損

請輸出：
【步驟一：狀態診斷】當前市場狀態與關鍵觀察
【步驟二：策略選擇】選用策略及理由（含判斷做多或做空）
【步驟三：操作建議】
方向：做多 / 做空 / 觀望
📍 進場價位區間：$XXX ~ $XXX
🛑 停損位：$XXX
🎯 目標價：$XXX（預期報酬 XX%）
⚠️ 風險評級：中 / 高 / 極高
📝 注意事項

【步驟四：持倉管理建議】（針對已持有多單者）
📈 若目前獲利中：繼續持有 / 部分獲利了結 / 加碼，理由與時機
📉 若目前虧損中：停損出場 / 繼續持有 / 攤平，理由與條件
🎯 停利參考位：$XXX（第一目標）/ $XXX（第二目標）
🛑 持倉停損位：$XXX（跌破此價建議出場）

請用繁體中文回答，語氣專業且具體。"""

PROMPTS = {
    "us": PROMPT_US,
    "taiwan": PROMPT_TAIWAN,
    "crypto": PROMPT_CRYPTO,
}

# ── BTC 短線 Prompt ───────────────────────────────────
PROMPT_BTC_SHORT = """你是一位專業的 BTC 短線交易員，擅長結合籌碼面與技術面做出精準的短線判斷。

分析框架：
1. 籌碼面優先：資金費率、OI 變化、多空比是短線最重要的領先指標
2. 技術面確認：1H 看方向，15M 找結構，5M 找進場時機
3. 風報比至少 1:2，停損嚴格執行

籌碼面判讀：
- 資金費率 > 0.05%：多方過擁擠，逢高做空或不做多
- 資金費率 < -0.02%：空方付費，逢低做多機會
- OI 增加 + 價漲：多方加倉，趨勢延續
- OI 增加 + 價跌：空方加倉，下跌動能強
- OI 減少：倉位出清，方向不明
- 多空比 > 1.8：散戶過度樂觀，逆向偏空
- 多空比 < 0.7：散戶過度悲觀，逆向偏多

技術面判讀：
- 多時框架共鳴（1H + 15M 同向）才進場
- RSI 超買/超賣配合布林通道上下軌使用
- MACD 金叉/死叉作為進場觸發條件

請輸出：
【步驟一：籌碼面診斷】
- 資金費率解讀
- OI 趨勢解讀
- 多空比解讀
- 整體籌碼面偏向：偏多 / 偏空 / 中性

【步驟二：技術面確認】
- 1H / 15M / 5M 各時框狀態
- 關鍵支撐與壓力位
- 整體技術面偏向

【步驟三：做多方案】
📍 進場價位：$XXX（觸發條件）
🛑 停損位：$XXX
🎯 目標一：$XXX
🎯 目標二：$XXX（延伸目標）
⚖️ 風報比：1 : X
📝 進場條件 / 注意事項

【步驟四：做空方案】
📍 進場價位：$XXX（觸發條件）
🛑 停損位：$XXX
🎯 目標一：$XXX
🎯 目標二：$XXX（延伸目標）
⚖️ 風報比：1 : X
📝 進場條件 / 注意事項

【步驟五：優先方向與結論】
目前優先方向：做多 / 做空 / 觀望等訊號
理由（一句話）

請用繁體中文回答，語氣簡潔直接，像交易員對交易員說話。"""


def build_tf_block(tf: dict) -> str:
    """把多時框架摘要轉成文字"""
    labels = {"weekly": "週線", "daily": "日線", "h4": "4H", "h1": "1H"}
    lines = []
    for key, label in labels.items():
        d = tf.get(key)
        if not d:
            lines.append(f"- {label}：資料不足")
            continue
        ma20_str = f"MA20 {'上方' if d.get('above_ma20') else '下方'}" if d.get('above_ma20') is not None else ""
        ma50_str = f"MA50 {'上方' if d.get('above_ma50') else '下方'}" if d.get('above_ma50') is not None else ""
        lines.append(
            f"- {label}：RSI {d['rsi']}　MACD {d['macd_direction']}　{ma20_str}　{ma50_str}　→ **{d['bias']}**"
        )
    return "\n".join(lines)



def build_sentiment_block(sentiment: dict) -> str:
    """把 StockTwits 情緒數據轉成 prompt 文字區塊"""
    posts = sentiment.get("posts", [])
    mood  = sentiment.get("mood", "無標記")
    bull  = sentiment.get("bullish_pct")
    bear  = sentiment.get("bearish_pct")
    total = sentiment.get("total", 0)

    sentiment_line = (
        f"看多 {bull}% / 看空 {bear}%（共 {total} 筆，整體氛圍：{mood}）"
        if bull is not None
        else f"無標記數據（共 {total} 筆）"
    )

    top_posts = "\n".join([
        f"- [{p['created']}] @{p['user']} "
        f"{'[看多]' if p['sentiment']=='Bullish' else '[看空]' if p['sentiment']=='Bearish' else '[未標記]'} "
        f"{p['body'][:100]}"
        for p in posts[:10]
    ])

    return f"""
**社群情緒面（StockTwits）**
- 情緒分佈：{sentiment_line}
- 近期貼文摘要：
{top_posts}"""


def build_user_message(data: dict, sentiment: dict | None = None) -> str:
    """組裝給 Gemini 的技術分析 prompt，可選加入社群情緒"""
    cur = data["currency"]
    asset_type = data["asset_type"]
    macd_dir = "金叉（多頭）" if data["macd_histogram"] and data["macd_histogram"] > 0 else "死叉（空頭）"
    volume_desc = (
        "放量" if data["volume_ratio"] and data["volume_ratio"] > 1.5
        else "縮量" if data["volume_ratio"] and data["volume_ratio"] < 0.7
        else "正常量"
    )

    if asset_type == "taiwan":
        ma_lines = f"""- MA20：{data['ma20']}（{'上方' if data.get('above_ma20') else '下方'}，斜率 {data['ma20_slope']}%/5日）
- MA60：{data['ma60'] or '資料不足'}（{'上方' if data.get('above_ma60') else '下方' if data['ma60'] else 'N/A'}）"""
    elif asset_type == "us":
        ma_lines = f"""- MA20：{data['ma20']}（{'上方' if data.get('above_ma20') else '下方'}，斜率 {data['ma20_slope']}%/5日）
- MA50：{data['ma50'] or '資料不足'}（{'上方' if data.get('above_ma50') else '下方' if data['ma50'] else 'N/A'}）
- MA200：{data['ma200'] or '資料不足'}（{'上方' if data.get('above_ma200') else '下方' if data['ma200'] else 'N/A'}）"""
    else:
        ma_lines = f"""- MA20：{data['ma20']}（{'上方' if data.get('above_ma20') else '下方'}，斜率 {data['ma20_slope']}%/5日）
- MA50：{data['ma50'] or '資料不足'}（{'上方' if data.get('above_ma50') else '下方' if data['ma50'] else 'N/A'}）"""

    tf_block = build_tf_block(data.get("timeframes", {}))

    cp = data.get("candlestick_patterns", {})
    tf_pattern_labels = [("weekly", "週線"), ("daily", "日線"), ("h4", "4H")]
    patterns_str = "\n".join([
        f"- {label}：{'、'.join(cp.get(key, ['資料不足']))}"
        for key, label in tf_pattern_labels
    ])

    news_list = data.get("news", [])
    if news_list:
        news_str = "\n".join([
            f"- [{n['date']}] {n['title']}" + (f"\n  摘要：{n['summary']}" if n.get('summary') else "")
            for n in news_list
        ])
    else:
        news_str = "- 無近期新聞"

    sentiment_section = build_sentiment_block(sentiment) if sentiment else ""
    has_sentiment = bool(sentiment)

    return f"""請分析以下數據並給出買入建議：

## {data['symbol']} 技術數據

**價格**
- 當前價格：{cur}{data['current_price']}（今日 {'+' if data['change_pct'] > 0 else ''}{data['change_pct']}%）
- 5日報酬：{'+' if data['return_5d'] > 0 else ''}{data['return_5d']}%（5日前 {cur}{data['price_5d_ago']}）

**多時框架偏向**
{tf_block}

**日線均線**
{ma_lines}

**日線動能指標**
- RSI(14)：{data['rsi']}　背離：{data.get('rsi_divergence', 'N/A')}
- MACD：{data['macd']}，Signal：{data['macd_signal']}，Histogram：{data['macd_histogram']}（{macd_dir}）

**K線型態（多時框架）**
{patterns_str}

**布林通道**
- 上軌：{cur}{data.get('bb_upper')}　下軌：{cur}{data.get('bb_lower')}
- 當前位置：{data.get('bb_position', 'N/A')}　通道寬度：{data.get('bb_width')}%

**量能**
- 量比：{data['volume_ratio']}x（{volume_desc}）
- OBV：{data.get('obv_signal', 'N/A')}

**波動率與停損參考**
- ATR(14)：{cur}{data.get('atr')}（波動率 {data.get('atr_pct')}%）
- ATR 建議停損位：{cur}{data.get('atr_stop')}（1.5x ATR）

**價格位置**
- 52週高點：{cur}{data['high_52w']}（距高點 {data['pct_from_52w_high']}%）
- 52週低點：{cur}{data['low_52w']}（距低點 +{data['pct_from_52w_low']}%）
{f"- 下次財報：{data['earnings_info']}" if data.get('earnings_info') else ""}

**消息面（近期新聞）**
{news_str}
{sentiment_section}

分析時請優先從週線判斷大趨勢方向，再看日線確認中期趨勢，最後用4H/1H找進場時機。停損位請參考ATR建議，不要設定固定百分比。消息面若有重大利多/利空請在步驟一診斷中特別標注。{"步驟一診斷中請一併說明社群情緒面（看多/看空/分歧）與技術面是否共鳴或背離，並判斷社群情緒對近期股價的潛在影響。" if has_sentiment else ""}請依照步驟格式進行分析。"""


def build_btc_short_message(btc: dict, history: list | None = None) -> str:
    tf_data = btc.get("tf_data", {})
    tf_lines = []
    for key, label in [("1h", "1H"), ("15m", "15M"), ("5m", "5M")]:
        d = tf_data.get(key)
        if d:
            patterns = [p for p in d.get("patterns", []) if p != "無明顯K線型態"]
            pattern_str = f"　K線：{'、'.join(patterns)}" if patterns else ""
            tf_lines.append(
                f"- {label}：RSI {d['rsi']}　MACD {d['macd_status']}　布林 {d['bb_position']}　→ **{d['bias']}**{pattern_str}"
            )
        else:
            tf_lines.append(f"- {label}：資料不足")

    funding = btc.get("funding") or {}
    oi      = btc.get("oi") or {}
    ls      = btc.get("ls_ratio") or {}
    n       = len(history) + 1 if history else 1

    # 歷史分析區塊
    history_block = ""
    if history:
        entries = []
        for i, h in enumerate(history, 1):
            snap = h.get("snapshot", {})
            snap_line = (
                f"  當時現價 ${snap.get('entry_ref','?')}　方向 {snap.get('direction','?')}　"
                f"訊號分數 {snap.get('score', 0):+d}"
            ) if snap else ""
            entries.append(
                f"=== 第 {i} 次分析（{h['timestamp']}）===\n"
                f"{snap_line}\n"
                f"{h['analysis']}"
            )
        history_block = f"""
══════════════════════════════
【前次分析記錄（共 {len(history)} 次）】
══════════════════════════════
{chr(10).join(entries)}

══════════════════════════════
【本次為第 {n} 次分析，請對照前次記錄回答：】
1. 哪些籌碼面 / 技術面數據發生了明顯改變？
2. 前次建議的進場或停損條件是否已觸發或失效？
3. 根據最新數據，是否需要調整操作方向或倉位計畫？
══════════════════════════════
"""

    vd = btc.get("vwap_data") or {}
    if vd:
        vwap_block = f"""
**15m 錨定 VWAP（{vd.get('anchor_label','')}，錨點 {vd.get('anchor_time','')}）**
- VWAP：${vd.get('vwap','N/A')}　現價偏離 {vd.get('diff_pct','N/A'):+}%（{vd.get('position','N/A')}）
- 標準差帶：+2σ ${vd.get('upper_2','N/A')} / +1σ ${vd.get('upper_1','N/A')} / -1σ ${vd.get('lower_1','N/A')} / -2σ ${vd.get('lower_2','N/A')}
- 做多方案：進場 ${vd.get('long_entry','N/A')}（回踩VWAP）　停損 ${vd.get('long_stop','N/A')}（-1σ）　目標 ${vd.get('long_target','N/A')}（+2σ）　風報比 1:{vd.get('rr_long','N/A')}
- 做空方案：進場 ${vd.get('short_entry','N/A')}（反彈VWAP）　停損 ${vd.get('short_stop','N/A')}（+1σ）　目標 ${vd.get('short_target','N/A')}（-2σ）　風報比 1:{vd.get('rr_short','N/A')}"""
    else:
        vwap_block = "\n**15m 錨定 VWAP**：資料無法取得"

    return f"""請分析以下 {btc.get('coin', 'BTC')} 短線數據（第 {n} 次分析）：
{history_block}
**最新籌碼面**
- 資金費率：{funding.get('funding_rate', 'N/A')}%
- 未平倉量 OI 6h 變化：{oi.get('oi_change_6h', 'N/A')}%
- 多空比：{ls.get('ratio', 'N/A')}（多 {ls.get('long_pct', 'N/A')}% / 空 {ls.get('short_pct', 'N/A')}%）

**最新技術面（多時框架）**
{chr(10).join(tf_lines)}
{vwap_block}

**ATR 停損參考（1H）**
- 做多停損：${btc.get('stop_long', 'N/A')}　目標：${btc.get('target_long', 'N/A')}
- 做空停損：${btc.get('stop_short', 'N/A')}　目標：${btc.get('target_short', 'N/A')}

**程式初步判斷**：{btc.get('direction', 'N/A')}（訊號分數 {btc.get('score', 0):+d}）

請依照步驟格式給出第 {n} 次短線操作建議。分析時請結合錨定 VWAP 的位置與標準差帶，說明現價在 VWAP 上方/下方的含義，以及 VWAP ±1σ / ±2σ 作為支撐/壓力的判斷。"""


def analyze_stocktwits(symbol: str, message: str) -> dict:
    import time
    import streamlit as st

    prompt = "你是一位專業的金融社群情緒分析師。分析社群貼文的市場情緒，指出關鍵主題與對股價的潛在影響。請用繁體中文回答，簡潔扼要。"
    model = genai.GenerativeModel(model_name="gemini-2.5-flash", system_instruction=prompt)

    for attempt in range(3):
        try:
            response = model.generate_content(message)
            return {"symbol": symbol, "analysis": response.text, "error": None}
        except Exception as e:
            err = str(e)
            if "ResourceExhausted" in err or "429" in err:
                wait = (attempt + 1) * 30
                st.warning(f"⏳ Gemini 速率限制，{wait} 秒後重試...")
                time.sleep(wait)
            else:
                return {"symbol": symbol, "analysis": "", "error": f"Gemini 分析失敗：{err[:200]}"}

    return {"symbol": symbol, "analysis": "", "error": "已達 Gemini 速率上限，請稍後重試"}


def analyze_btc_short(btc: dict, history: list | None = None) -> dict:
    import time
    import streamlit as st

    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=PROMPT_BTC_SHORT,
    )

    last_error = ""
    for attempt in range(3):
        try:
            response = model.generate_content(build_btc_short_message(btc, history))
            return {"analysis": response.text, "error": None}
        except Exception as e:
            last_error = str(e)
            if "ResourceExhausted" in last_error or "429" in last_error:
                wait = (attempt + 1) * 30
                st.warning(f"⏳ Gemini 速率限制，{wait} 秒後重試（第 {attempt + 1}/3 次）...")
                time.sleep(wait)
            else:
                return {"analysis": "", "error": f"Gemini 分析失敗：{last_error[:200]}"}

    return {"analysis": "", "error": "已達 Gemini 速率上限，請稍候幾分鐘後重試"}


def analyze_stock(stock_data: dict, sentiment: dict | None = None) -> dict:
    import time
    import streamlit as st
    asset_type = stock_data.get("asset_type", "us")

    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=PROMPTS[asset_type],
    )

    last_error = ""
    for attempt in range(3):
        try:
            response = model.generate_content(build_user_message(stock_data, sentiment))
            return {
                "symbol": stock_data["symbol"],
                "asset_type": asset_type,
                "price": stock_data["current_price"],
                "analysis": response.text,
                "error": None,
            }
        except Exception as e:
            last_error = str(e)
            if "ResourceExhausted" in last_error or "429" in last_error:
                wait = (attempt + 1) * 30  # 30秒、60秒、90秒
                st.warning(f"⏳ Gemini 速率限制，{wait} 秒後自動重試（第 {attempt + 1}/3 次）...")
                time.sleep(wait)
            else:
                # 非速率限制的錯誤，直接回傳錯誤訊息不崩潰
                return {
                    "symbol": stock_data["symbol"],
                    "asset_type": asset_type,
                    "price": stock_data["current_price"],
                    "analysis": "",
                    "error": f"Gemini 分析失敗：{last_error[:200]}",
                }

    return {
        "symbol": stock_data["symbol"],
        "asset_type": asset_type,
        "price": stock_data["current_price"],
        "analysis": "",
        "error": "已達 Gemini 速率上限，請稍候幾分鐘後重試（免費版每分鐘限 15 次）",
    }
