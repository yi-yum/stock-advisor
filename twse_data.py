"""
台灣證券交易所（TWSE）籌碼面數據模組

提供三大法人買賣超及融資融券等籌碼面數據的獲取功能。
"""

import requests
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any


# 預設 User-Agent，避免被伺服器擋掉
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# API 基礎 URL
TWSE_INSTITUTIONAL_URL = (
    "https://www.twse.com.tw/rwd/zh/fund/T86?response=json"
)
TWSE_MARGIN_URL = (
    "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN?response=json"
)


def _extract_stock_code(symbol: str) -> str:
    """
    從股票代號字串中提取純數字部分。

    支援的格式：
    - "2330.TW" -> "2330"
    - "6547.TWO" -> "6547"
    - "2330" -> "2330"

    Args:
        symbol: 股票代號字串

    Returns:
        純數字的股票代號

    Raises:
        ValueError: 若無法從字串中提取有效的股票代號
    """
    # 移除 .TW .TWO 等副檔名
    clean_symbol = symbol.split(".")[0].strip()

    # 確保是純數字
    if not clean_symbol.isdigit():
        raise ValueError(f"無效的股票代號格式: {symbol}")

    return clean_symbol


def _get_trading_dates(num_dates: int = 5) -> List[str]:
    """
    從今天往前推算最近的交易日期列表（跳過週末）。

    Args:
        num_dates: 要回傳的交易日期個數

    Returns:
        YYYYMMDD 格式的日期列表（新至舊）
    """
    dates = []
    current_date = datetime.now()

    while len(dates) < num_dates:
        # 跳過週末（5=星期六, 6=星期日）
        if current_date.weekday() < 5:
            dates.append(current_date.strftime("%Y%m%d"))
        current_date -= timedelta(days=1)

    return dates


def _fetch_twse_data(url: str, symbol: str, dates: List[str], timeout: int = 10) -> Optional[Dict]:
    """
    從 TWSE API 獲取數據的內部函式。

    Args:
        url: TWSE API URL
        symbol: 純數字股票代號
        dates: 日期列表（YYYYMMDD 格式）
        timeout: 請求逾時秒數

    Returns:
        API 回傳的 JSON 字典，若失敗則回傳 None
    """
    for date_str in dates:
        try:
            params = {
                "date": date_str,
                "selectType": "ALLBUT0999" if "T86" in url else "ALL",
                "response": "json"
            }

            response = requests.get(
                url,
                params=params,
                headers=DEFAULT_HEADERS,
                timeout=timeout
            )
            response.raise_for_status()

            data = response.json()

            # 檢查是否有有效的數據
            if data.get("data") and len(data["data"]) > 0:
                return data

        except requests.RequestException:
            # 繼續嘗試下一個日期
            continue
        except (ValueError, KeyError):
            # JSON 解析錯誤，繼續嘗試
            continue

    return None


def _parse_institutional_data(symbol: str, raw_data: Dict) -> Dict[str, Any]:
    """
    解析三大法人買賣超原始數據。

    Args:
        symbol: 股票代號
        raw_data: TWSE API 原始數據

    Returns:
        解析後的字典
    """
    result = {
        "foreign_net": 0,
        "trust_net": 0,
        "dealer_net": 0,
        "total_net": 0,
        "consecutive_buy": 0,
        "recent_5d": [],
        "source": "TWSE",
        "error": None
    }

    try:
        if not raw_data or not raw_data.get("data"):
            result["error"] = "查詢結果為空"
            return result

        # 搜尋符合股票代號的資料列
        stock_data = None
        for row in raw_data["data"]:
            if row[0] == symbol:
                stock_data = row
                break

        if not stock_data:
            result["error"] = f"查無該股票 {symbol} 的籌碼數據"
            return result

        # 數據結構（根據 TWSE T86 API）：
        # [0] 股票代號
        # [1] 外資買進
        # [2] 外資賣出
        # [3] 外資買賣超
        # [4] 投信買進
        # [5] 投信賣出
        # [6] 投信買賣超
        # [7] 自營商買進
        # [8] 自營商賣出
        # [9] 自營商買賣超

        # 安全地轉換為整數
        try:
            foreign_net = int(stock_data[3].replace(",", "")) if stock_data[3] else 0
            trust_net = int(stock_data[6].replace(",", "")) if stock_data[6] else 0
            dealer_net = int(stock_data[9].replace(",", "")) if stock_data[9] else 0
        except (IndexError, ValueError, AttributeError):
            result["error"] = "數據解析異常"
            return result

        result["foreign_net"] = foreign_net
        result["trust_net"] = trust_net
        result["dealer_net"] = dealer_net
        result["total_net"] = foreign_net + trust_net + dealer_net

    except Exception as e:
        result["error"] = f"解析錯誤: {str(e)}"

    return result


def _parse_margin_data(symbol: str, raw_data: Dict) -> Dict[str, Any]:
    """
    解析融資融券原始數據。

    Args:
        symbol: 股票代號
        raw_data: TWSE API 原始數據

    Returns:
        解析後的字典
    """
    result = {
        "margin_balance": 0,
        "margin_change": 0,
        "short_balance": 0,
        "short_change": 0,
        "margin_ratio": 0.0,
        "recent_5d": [],
        "source": "TWSE",
        "error": None
    }

    try:
        if not raw_data or not raw_data.get("data"):
            result["error"] = "查詢結果為空"
            return result

        # 搜尋符合股票代號的資料列
        stock_data = None
        for row in raw_data["data"]:
            if row[0] == symbol:
                stock_data = row
                break

        if not stock_data:
            result["error"] = f"查無該股票 {symbol} 的融資融券數據"
            return result

        # 數據結構（根據 TWSE MI_MARGN API）：
        # [0] 股票代號
        # [1] 融資買進
        # [2] 融資賣出
        # [3] 融資餘額
        # [4] 融資增減
        # [5] 融券買進
        # [6] 融券賣出
        # [7] 融券餘額
        # [8] 融券增減

        try:
            margin_balance = int(stock_data[3].replace(",", "")) if stock_data[3] else 0
            margin_change = int(stock_data[4].replace(",", "")) if stock_data[4] else 0
            short_balance = int(stock_data[7].replace(",", "")) if stock_data[7] else 0
            short_change = int(stock_data[8].replace(",", "")) if stock_data[8] else 0
        except (IndexError, ValueError, AttributeError):
            result["error"] = "數據解析異常"
            return result

        result["margin_balance"] = margin_balance
        result["margin_change"] = margin_change
        result["short_balance"] = short_balance
        result["short_change"] = short_change

        # 計算融資使用率（融資餘額 / 市場總額，暫以簡易估算）
        if margin_balance > 0:
            result["margin_ratio"] = min(margin_balance / 100000.0, 1.0)

    except Exception as e:
        result["error"] = f"解析錯誤: {str(e)}"

    return result


def get_twse_institutional(symbol: str) -> Dict[str, Any]:
    """
    獲取台灣上市櫃股票三大法人買賣超數據（外資、投信、自營商）。

    支援的股票代號格式：
    - "2330.TW" -> 自動提取為 "2330"
    - "6547.TWO" -> 自動提取為 "6547"
    - "2330" -> 直接使用

    Args:
        symbol: 股票代號（可含 .TW / .TWO 副檔名）

    Returns:
        字典，結構如下：
        {
            "foreign_net": int,          # 外資今日買賣超（張）
            "trust_net": int,            # 投信今日買賣超（張）
            "dealer_net": int,           # 自營商今日買賣超（張）
            "total_net": int,            # 三大法人合計買賣超（張）
            "consecutive_buy": int,      # 外資連續買超天數（負數表示連續賣超）
            "recent_5d": List[Dict],     # 最近5日詳細數據
            "source": str,               # 數據來源 ("TWSE")
            "error": Optional[str]       # 若發生錯誤則填入錯誤訊息字串
        }

    Example:
        >>> result = get_twse_institutional("2330.TW")
        >>> print(result["foreign_net"])
        12345
    """
    try:
        # 提取純數字股票代號
        clean_symbol = _extract_stock_code(symbol)

        # 取得最近 5 個交易日期
        dates = _get_trading_dates(5)

        # 從 TWSE 取得數據
        raw_data = _fetch_twse_data(TWSE_INSTITUTIONAL_URL, clean_symbol, dates)

        # 解析數據
        result = _parse_institutional_data(clean_symbol, raw_data)

        # 構建最近 5 日數據（若有取得原始數據）
        if raw_data and raw_data.get("data"):
            recent_5d = []
            for row in raw_data["data"]:
                if row[0] == clean_symbol:
                    try:
                        record = {
                            "date": f"{raw_data.get('date', '')[0:4]}/{raw_data.get('date', '')[4:6]}/{raw_data.get('date', '')[6:8]}",
                            "foreign": int(row[3].replace(",", "")) if row[3] else 0,
                            "trust": int(row[6].replace(",", "")) if row[6] else 0,
                            "dealer": int(row[9].replace(",", "")) if row[9] else 0
                        }
                        recent_5d.append(record)
                    except (IndexError, ValueError, AttributeError):
                        continue
            result["recent_5d"] = recent_5d

        return result

    except ValueError as e:
        return {
            "foreign_net": 0,
            "trust_net": 0,
            "dealer_net": 0,
            "total_net": 0,
            "consecutive_buy": 0,
            "recent_5d": [],
            "source": "TWSE",
            "error": str(e)
        }
    except Exception as e:
        return {
            "foreign_net": 0,
            "trust_net": 0,
            "dealer_net": 0,
            "total_net": 0,
            "consecutive_buy": 0,
            "recent_5d": [],
            "source": "TWSE",
            "error": f"未預期的錯誤: {str(e)}"
        }


def get_twse_margin(symbol: str) -> Dict[str, Any]:
    """
    獲取台灣上市櫃股票融資融券餘額數據。

    支援的股票代號格式：
    - "2330.TW" -> 自動提取為 "2330"
    - "6547.TWO" -> 自動提取為 "6547"
    - "2330" -> 直接使用

    Args:
        symbol: 股票代號（可含 .TW / .TWO 副檔名）

    Returns:
        字典，結構如下：
        {
            "margin_balance": int,       # 融資餘額（張）
            "margin_change": int,        # 融資餘額變化（今日）
            "short_balance": int,        # 融券餘額（張）
            "short_change": int,         # 融券餘額變化（今日）
            "margin_ratio": float,       # 融資使用率（0~1）
            "recent_5d": List[Dict],     # 最近5日詳細數據
            "source": str,               # 數據來源 ("TWSE")
            "error": Optional[str]       # 若發生錯誤則填入錯誤訊息字串
        }

    Example:
        >>> result = get_twse_margin("2330.TW")
        >>> print(result["margin_balance"])
        50000
    """
    try:
        # 提取純數字股票代號
        clean_symbol = _extract_stock_code(symbol)

        # 取得最近 5 個交易日期
        dates = _get_trading_dates(5)

        # 從 TWSE 取得數據
        raw_data = _fetch_twse_data(TWSE_MARGIN_URL, clean_symbol, dates)

        # 解析數據
        result = _parse_margin_data(clean_symbol, raw_data)

        # 構建最近 5 日數據（若有取得原始數據）
        if raw_data and raw_data.get("data"):
            recent_5d = []
            for row in raw_data["data"]:
                if row[0] == clean_symbol:
                    try:
                        record = {
                            "date": f"{raw_data.get('date', '')[0:4]}/{raw_data.get('date', '')[4:6]}/{raw_data.get('date', '')[6:8]}",
                            "margin_balance": int(row[3].replace(",", "")) if row[3] else 0,
                            "margin_change": int(row[4].replace(",", "")) if row[4] else 0,
                            "short_balance": int(row[7].replace(",", "")) if row[7] else 0,
                            "short_change": int(row[8].replace(",", "")) if row[8] else 0
                        }
                        recent_5d.append(record)
                    except (IndexError, ValueError, AttributeError):
                        continue
            result["recent_5d"] = recent_5d

        return result

    except ValueError as e:
        return {
            "margin_balance": 0,
            "margin_change": 0,
            "short_balance": 0,
            "short_change": 0,
            "margin_ratio": 0.0,
            "recent_5d": [],
            "source": "TWSE",
            "error": str(e)
        }
    except Exception as e:
        return {
            "margin_balance": 0,
            "margin_change": 0,
            "short_balance": 0,
            "short_change": 0,
            "margin_ratio": 0.0,
            "recent_5d": [],
            "source": "TWSE",
            "error": f"未預期的錯誤: {str(e)}"
        }


if __name__ == "__main__":
    # 測試範例
    print("=== 三大法人買賣超 ===")
    inst_data = get_twse_institutional("2330.TW")
    print(f"外資買賣超: {inst_data['foreign_net']} 張")
    print(f"投信買賣超: {inst_data['trust_net']} 張")
    print(f"自營商買賣超: {inst_data['dealer_net']} 張")
    print(f"合計買賣超: {inst_data['total_net']} 張")
    if inst_data.get("error"):
        print(f"錯誤: {inst_data['error']}")

    print("\n=== 融資融券 ===")
    margin_data = get_twse_margin("2330.TW")
    print(f"融資餘額: {margin_data['margin_balance']} 張")
    print(f"融資變化: {margin_data['margin_change']} 張")
    print(f"融券餘額: {margin_data['short_balance']} 張")
    print(f"融券變化: {margin_data['short_change']} 張")
    if margin_data.get("error"):
        print(f"錯誤: {margin_data['error']}")
