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

    # 同一支股票只保留最新一筆，避免重複
    history = [r for r in history if r["symbol"] != stock_data["symbol"]]
    history.insert(0, record)

    # 最多保留 50 筆
    history = history[:50]

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def delete_record(symbol: str):
    history = load_history()
    history = [r for r in history if r["symbol"] != symbol]
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
