import json
import os
from datetime import datetime

# Streamlit Cloud 的檔案系統是暫時的，重新部署後會清空
# 本地端執行時歷史紀錄會永久保存
HISTORY_FILE = "history.json"


def load_history() -> list:
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_result(stock_data: dict, analysis: str, claude_prompt: str):
    history = load_history()

    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "symbol": stock_data["symbol"],
        "asset_type": stock_data["asset_type"],
        "currency": stock_data["currency"],
        "current_price": stock_data["current_price"],
        "change_pct": stock_data["change_pct"],
        "rsi": stock_data["rsi"],
        "volume_ratio": stock_data["volume_ratio"],
        "return_5d": stock_data["return_5d"],
        "pct_from_52w_high": stock_data["pct_from_52w_high"],
        "analysis": analysis,
        "claude_prompt": claude_prompt,
    }

    history.insert(0, record)

    # 最多保留 100 筆
    history = history[:100]

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def save_short_result(btc_data: dict, analysis: str, claude_prompt: str):
    history = load_history()
    coin = btc_data.get("coin", "BTC")
    symbol = f"{coin}-SHORT"

    funding  = btc_data.get("funding") or {}
    oi       = btc_data.get("oi") or {}
    ls       = btc_data.get("ls_ratio") or {}
    h1       = (btc_data.get("tf_data") or {}).get("1h") or {}

    record = {
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M"),
        "symbol":       symbol,
        "asset_type":   "short",
        "currency":     "$",
        "current_price": btc_data.get("entry_ref"),
        "change_pct":   None,
        "rsi":          h1.get("rsi"),
        "volume_ratio": None,
        "return_5d":    None,
        "pct_from_52w_high": None,
        # 短線專屬欄位
        "coin":          coin,
        "direction":     btc_data.get("direction"),
        "score":         btc_data.get("score"),
        "funding_rate":  funding.get("funding_rate"),
        "oi_change_6h":  oi.get("oi_change_6h"),
        "ls_ratio":      ls.get("ratio"),
        "rr_long":       btc_data.get("rr_long"),
        "rr_short":      btc_data.get("rr_short"),
        "analysis":      analysis,
        "claude_prompt": claude_prompt,
    }

    history.insert(0, record)
    history = history[:100]

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def delete_record(symbol: str, timestamp: str | None = None):
    history = load_history()
    if timestamp:
        # 刪除特定一筆
        history = [r for r in history if not (r["symbol"] == symbol and r["timestamp"] == timestamp)]
    else:
        # 刪除該 symbol 的全部
        history = [r for r in history if r["symbol"] != symbol]
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
