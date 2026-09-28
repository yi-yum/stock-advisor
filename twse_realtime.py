"""
台灣證券交易所（TWSE）即時報價模組

提供台股交易時間內的即時報價功能。
台股交易時間：週一到週五 09:00-13:30 (台灣時間 UTC+8)
"""

import requests
from datetime import datetime
from typing import Dict, Optional, List
import pytz

# 預設 User-Agent，避免被伺服器擋掉
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Referer": "https://mis.twse.com.tw"
}

# TWSE API 基礎 URL
TWSE_REALTIME_URL = "https://mis.twse.com.tw/stock/api/getStockInfo.jsp"

# 台灣時區
TW_TZ = pytz.timezone('Asia/Taipei')


def is_tw_market_open() -> bool:
    """
    判斷目前是否為台股交易時間（週一到週五 09:00-13:30 台灣時間）

    Returns:
        bool: True 表示交易時間內，False 表示非交易時間
    """
    now_tw = datetime.now(TW_TZ)
    weekday = now_tw.weekday()  # 0=Monday, 6=Sunday
    hour = now_tw.hour
    minute = now_tw.minute

    # 週一到週五 (weekday 0-4)
    if weekday >= 5:  # Saturday or Sunday
        return False

    # 09:00 - 13:30
    time_seconds = hour * 3600 + minute * 60
    market_open = 9 * 3600  # 09:00
    market_close = 13 * 3600 + 30 * 60  # 13:30

    return market_open <= time_seconds <= market_close


def _normalize_symbol(symbol: str) -> tuple[str, str]:
    """
    將股票代號標準化為 TWSE API 格式

    Args:
        symbol: 股票代號 (支援格式: "2330", "2330.TW", "6547.TWO")

    Returns:
        tuple: (純數字代號, 市場代碼)
        - 上市股返回 ("2330", "tse")
        - 上櫃股返回 ("6547", "otc")

    Raises:
        ValueError: 若無法識別代號格式
    """
    symbol = symbol.upper().strip()

    # 分離代號和市場代碼
    if "." in symbol:
        code, market = symbol.split(".")
        code = code.strip()
    else:
        code = symbol
        market = None

    # 確保代號是純數字
    if not code.isdigit():
        raise ValueError(f"無效的股票代號格式: {symbol}")

    # 如果已指定市場代碼，使用該代碼
    if market and market in ["TW", "TWO"]:
        market_type = "tse" if market == "TW" else "otc"
        return code, market_type

    # 根據代號範圍猜測市場（上市：0001-9999, 上櫃：6000-9999 多數為上櫃）
    # 簡易判斷：代號在 1000-5999 通常是上市，6000+ 通常是上櫃
    code_int = int(code)
    if code_int < 6000:
        return code, "tse"
    else:
        return code, "otc"


def get_twse_realtime_price(symbol: str) -> Dict:
    """
    取得台股即時報價（交易時間內）

    API：https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch=tse_2330.tw&json=1&delay=0
    上櫃用：https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch=otc_6547.tw&json=1&delay=0

    Args:
        symbol: 股票代號（支援格式: "2330", "2330.TW", "6547.TWO"）

    Returns:
        dict: 回傳格式如下：
        {
            "symbol": "2330",                # 股票代號
            "name": "台積電",                # 股票名稱
            "price": 850.0,                  # 最新成交價
            "change": 5.0,                   # 漲跌絕對值
            "change_pct": 0.59,              # 漲跌百分比
            "high": 855.0,                   # 最高價
            "low": 845.0,                    # 最低價
            "open": 848.0,                   # 開盤價
            "volume": 12345,                 # 成交張數
            "total_volume": 50000,           # 累計成交量
            "buy": 850.0,                    # 買進價
            "sell": 851.0,                   # 賣出價
            "is_market_open": True,          # 是否交易時間
            "timestamp": "14:30:00",         # 時間戳記
            "error": None                    # 錯誤訊息 (若無則為 None)
        }
    """
    result = {
        "symbol": symbol.split(".")[0].upper(),
        "name": None,
        "price": None,
        "change": None,
        "change_pct": None,
        "high": None,
        "low": None,
        "open": None,
        "volume": None,
        "total_volume": None,
        "buy": None,
        "sell": None,
        "is_market_open": is_tw_market_open(),
        "timestamp": datetime.now(TW_TZ).strftime("%H:%M:%S"),
        "error": None
    }

    try:
        # 標準化符號
        code, market_type = _normalize_symbol(symbol)
        result["symbol"] = code

        # 構建 API 參數
        ex_ch = f"{market_type}_{code}.tw"
        params = {
            "ex_ch": ex_ch,
            "json": "1",
            "delay": "0"
        }

        # 發送請求
        response = requests.get(
            TWSE_REALTIME_URL,
            params=params,
            headers=DEFAULT_HEADERS,
            timeout=5
        )
        response.raise_for_status()

        data = response.json()

        # 檢查 API 回傳
        if not data.get("msgArray") or len(data["msgArray"]) == 0:
            result["error"] = f"查無代號 {code} 的報價"
            return result

        # 取第一筆股票資料
        stock_info = data["msgArray"][0]

        # 解析 API 回傳的欄位
        # 根據 TWSE API 文檔，msgArray 結構為：
        # [0] 股票代號, [1] 名稱, [2] 時間, [3] 成交, [4] 漲跌, [5] 最高,
        # [6] 最低, [7] 開盤, [8] 成交張數, [9] 成交金額, [10] 買進價, [11] 賣出價

        try:
            # 提取並轉換數值
            price = float(stock_info.get("z", 0)) if stock_info.get("z") else None
            change = float(stock_info.get("d", 0)) if stock_info.get("d") else None
            change_pct = float(stock_info.get("f", 0)) if stock_info.get("f") else None

            # 若無成交價，標記為非交易時間或尚未開盤
            if price is None or price == 0:
                if not result["is_market_open"]:
                    result["error"] = "非交易時間"
                else:
                    result["error"] = "尚未開盤"
                return result

            result["name"] = stock_info.get("n", "")
            result["price"] = price
            result["change"] = change
            result["change_pct"] = change_pct
            result["high"] = float(stock_info.get("y", 0)) if stock_info.get("y") else None
            result["low"] = float(stock_info.get("l", 0)) if stock_info.get("l") else None
            result["open"] = float(stock_info.get("o", 0)) if stock_info.get("o") else None
            result["volume"] = int(stock_info.get("v", 0).replace(",", "")) if stock_info.get("v") else None
            result["buy"] = float(stock_info.get("bp", 0)) if stock_info.get("bp") else None
            result["sell"] = float(stock_info.get("ap", 0)) if stock_info.get("ap") else None
            result["timestamp"] = stock_info.get("tlong", "")[:2] + ":" + stock_info.get("tlong", "")[2:4] + ":" + stock_info.get("tlong", "")[4:6] if stock_info.get("tlong") else result["timestamp"]

        except (ValueError, KeyError, TypeError) as e:
            result["error"] = f"數據解析異常: {str(e)}"
            return result

        return result

    except ValueError as e:
        result["error"] = str(e)
        return result
    except requests.RequestException as e:
        result["error"] = f"API 請求失敗: {str(e)}"
        return result
    except Exception as e:
        result["error"] = f"未預期的錯誤: {str(e)}"
        return result


def get_multiple_twse_realtime(symbols: List[str]) -> Dict[str, Dict]:
    """
    批次取得多支台股即時報價

    支援單一 API call 取得多支：ex_ch=tse_2330.tw|tse_2317.tw|otc_6547.tw

    Args:
        symbols: 股票代號列表，支援格式 ["2330", "2330.TW", "6547.TWO", ...]

    Returns:
        dict: 回傳格式 {"2330": {...}, "2317": {...}, ...}
        每個股票的資料結構同 get_twse_realtime_price() 的回傳值
    """
    if not symbols:
        return {}

    result_dict = {}

    try:
        # 分組 tse 和 otc 符號
        tse_codes = []
        otc_codes = []
        symbol_map = {}  # 用於將標準化的 code 對應回原始 symbol

        for symbol in symbols:
            try:
                code, market_type = _normalize_symbol(symbol)
                symbol_map[code] = symbol

                if market_type == "tse":
                    tse_codes.append(code)
                else:
                    otc_codes.append(code)
            except ValueError:
                # 若符號無效，直接返回錯誤
                result_dict[symbol] = {
                    "symbol": symbol.split(".")[0].upper(),
                    "error": f"無效的股票代號格式: {symbol}",
                    "is_market_open": is_tw_market_open(),
                    "timestamp": datetime.now(TW_TZ).strftime("%H:%M:%S"),
                }

        # 構建 ex_ch 參數
        ex_ch_parts = []
        ex_ch_parts.extend([f"tse_{code}.tw" for code in tse_codes])
        ex_ch_parts.extend([f"otc_{code}.tw" for code in otc_codes])

        if not ex_ch_parts:
            return {}

        ex_ch = "|".join(ex_ch_parts)

        # 發送單一 API 請求
        params = {
            "ex_ch": ex_ch,
            "json": "1",
            "delay": "0"
        }

        response = requests.get(
            TWSE_REALTIME_URL,
            params=params,
            headers=DEFAULT_HEADERS,
            timeout=5
        )
        response.raise_for_status()

        data = response.json()

        # 處理每支股票的回傳資料
        is_market_open = is_tw_market_open()
        timestamp = datetime.now(TW_TZ).strftime("%H:%M:%S")

        if data.get("msgArray"):
            for stock_info in data["msgArray"]:
                try:
                    code = stock_info.get("c", "").strip()
                    if not code:
                        continue

                    price = float(stock_info.get("z", 0)) if stock_info.get("z") else None

                    # 若無成交價
                    if price is None or price == 0:
                        result_dict[code] = {
                            "symbol": code,
                            "name": stock_info.get("n", ""),
                            "price": None,
                            "change": None,
                            "change_pct": None,
                            "high": None,
                            "low": None,
                            "open": None,
                            "volume": None,
                            "total_volume": None,
                            "buy": None,
                            "sell": None,
                            "is_market_open": is_market_open,
                            "timestamp": timestamp,
                            "error": "非交易時間" if not is_market_open else "尚未開盤"
                        }
                        continue

                    result_dict[code] = {
                        "symbol": code,
                        "name": stock_info.get("n", ""),
                        "price": price,
                        "change": float(stock_info.get("d", 0)) if stock_info.get("d") else None,
                        "change_pct": float(stock_info.get("f", 0)) if stock_info.get("f") else None,
                        "high": float(stock_info.get("y", 0)) if stock_info.get("y") else None,
                        "low": float(stock_info.get("l", 0)) if stock_info.get("l") else None,
                        "open": float(stock_info.get("o", 0)) if stock_info.get("o") else None,
                        "volume": int(stock_info.get("v", 0).replace(",", "")) if stock_info.get("v") else None,
                        "buy": float(stock_info.get("bp", 0)) if stock_info.get("bp") else None,
                        "sell": float(stock_info.get("ap", 0)) if stock_info.get("ap") else None,
                        "is_market_open": is_market_open,
                        "timestamp": timestamp,
                        "error": None
                    }
                except (ValueError, KeyError, TypeError):
                    code = stock_info.get("c", "Unknown")
                    result_dict[code] = {
                        "symbol": code,
                        "error": "數據解析異常",
                        "is_market_open": is_market_open,
                        "timestamp": timestamp,
                    }

        # 確保所有輸入的代號都有結果（即使出錯）
        for symbol in symbols:
            code = symbol.split(".")[0].upper()
            if code not in result_dict:
                result_dict[code] = {
                    "symbol": code,
                    "name": None,
                    "price": None,
                    "change": None,
                    "change_pct": None,
                    "high": None,
                    "low": None,
                    "open": None,
                    "volume": None,
                    "buy": None,
                    "sell": None,
                    "is_market_open": is_market_open,
                    "timestamp": timestamp,
                    "error": "查無此代號"
                }

        return result_dict

    except requests.RequestException as e:
        # API 請求失敗時，為所有輸入代號返回錯誤
        is_market_open = is_tw_market_open()
        timestamp = datetime.now(TW_TZ).strftime("%H:%M:%S")

        for symbol in symbols:
            code = symbol.split(".")[0].upper()
            result_dict[code] = {
                "symbol": code,
                "error": f"API 請求失敗: {str(e)}",
                "is_market_open": is_market_open,
                "timestamp": timestamp,
            }
        return result_dict
    except Exception as e:
        is_market_open = is_tw_market_open()
        timestamp = datetime.now(TW_TZ).strftime("%H:%M:%S")

        for symbol in symbols:
            code = symbol.split(".")[0].upper()
            result_dict[code] = {
                "symbol": code,
                "error": f"未預期的錯誤: {str(e)}",
                "is_market_open": is_market_open,
                "timestamp": timestamp,
            }
        return result_dict


if __name__ == "__main__":
    # 測試範例
    print("=== 測試單支股票 ===")
    result = get_twse_realtime_price("2330.TW")
    print(f"2330 (台積電):")
    print(f"  價格: {result['price']}")
    print(f"  漲跌%: {result['change_pct']}%")
    print(f"  交易時間: {result['is_market_open']}")
    if result.get("error"):
        print(f"  錯誤: {result['error']}")

    print("\n=== 測試多支股票 ===")
    results = get_multiple_twse_realtime(["2330.TW", "2317.TW", "6547.TWO"])
    for code, data in results.items():
        print(f"{code}:")
        print(f"  價格: {data['price']}")
        if data.get("error"):
            print(f"  錯誤: {data['error']}")

    print("\n=== 市場開盤狀態 ===")
    print(f"台股交易時間內: {is_tw_market_open()}")
    print(f"目前時間 (台灣): {datetime.now(TW_TZ).strftime('%Y-%m-%d %H:%M:%S')}")
