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


def build_user_message(data: dict) -> str:
    cur = data["currency"]
    asset_type = data["asset_type"]
    macd_dir = "金叉（多頭）" if data["macd_histogram"] and data["macd_histogram"] > 0 else "死叉（空頭）"
    volume_desc = (
        "放量" if data["volume_ratio"] and data["volume_ratio"] > 1.5
        else "縮量" if data["volume_ratio"] and data["volume_ratio"] < 0.7
        else "正常量"
    )

    # 依資產類型選擇顯示的均線
    if asset_type == "taiwan":
        ma_lines = f"""- MA20：{data['ma20']}（{'上方' if data.get('above_ma20') else '下方'}，斜率 {data['ma20_slope']}%/5日）
- MA60：{data['ma60'] or '資料不足'}（{'上方' if data.get('above_ma60') else '下方' if data['ma60'] else 'N/A'}）"""
    elif asset_type == "us":
        ma_lines = f"""- MA20：{data['ma20']}（{'上方' if data.get('above_ma20') else '下方'}，斜率 {data['ma20_slope']}%/5日）
- MA50：{data['ma50'] or '資料不足'}（{'上方' if data.get('above_ma50') else '下方' if data['ma50'] else 'N/A'}）
- MA200：{data['ma200'] or '資料不足'}（{'上方' if data.get('above_ma200') else '下方' if data['ma200'] else 'N/A'}）"""
    else:  # crypto
        ma_lines = f"""- MA20：{data['ma20']}（{'上方' if data.get('above_ma20') else '下方'}，斜率 {data['ma20_slope']}%/5日）
- MA50：{data['ma50'] or '資料不足'}（{'上方' if data.get('above_ma50') else '下方' if data['ma50'] else 'N/A'}）"""

    tf_block = build_tf_block(data.get("timeframes", {}))

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

分析時請優先從週線判斷大趨勢方向，再看日線確認中期趨勢，最後用4H/1H找進場時機。停損位請參考ATR建議，不要設定固定百分比。請依照三步驟格式進行分析。"""


def analyze_stock(stock_data: dict) -> dict:
    import time
    asset_type = stock_data.get("asset_type", "us")

    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=PROMPTS[asset_type],
    )

    # 自動重試，遇到速率限制最多等待 3 次
    for attempt in range(3):
        try:
            response = model.generate_content(build_user_message(stock_data))
            return {
                "symbol": stock_data["symbol"],
                "asset_type": asset_type,
                "price": stock_data["current_price"],
                "analysis": response.text,
            }
        except Exception as e:
            err = str(e)
            if "ResourceExhausted" in err or "429" in err:
                wait = (attempt + 1) * 15  # 15秒、30秒、45秒
                import streamlit as st
                st.warning(f"⏳ Gemini 速率限制，{wait} 秒後自動重試（第 {attempt + 1}/3 次）...")
                time.sleep(wait)
            else:
                raise

    raise Exception("Gemini API 已達速率上限，請稍後再試（免費版每分鐘限 15 次請求）")
