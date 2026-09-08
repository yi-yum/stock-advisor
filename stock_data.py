import yfinance as yf
import pandas as pd
import numpy as np

# 常見加密貨幣簡寫，輸入 BTC 自動轉為 BTC-USD
CRYPTO_SYMBOLS = {
    "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX",
    "DOT", "MATIC", "LINK", "UNI", "LTC", "ATOM", "FIL", "SUI",
}


def detect_asset_type(symbol: str) -> str:
    if symbol.endswith(".TW"):
        return "taiwan"
    elif symbol.endswith("-USD") or symbol in CRYPTO_SYMBOLS:
        return "crypto"
    else:
        return "us"


def normalize_symbol(symbol: str) -> str:
    if symbol in CRYPTO_SYMBOLS:
        return f"{symbol}-USD"
    return symbol


def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram


def safe_round(val, decimals=2):
    return round(float(val), decimals) if not pd.isna(val) else None


def get_stock_data(symbol: str):
    try:
        symbol = symbol.upper()
        asset_type = detect_asset_type(symbol)
        normalized = normalize_symbol(symbol)

        # 美股/加密貨幣需要 MA200，抓 1 年數據才夠
        period = "1y" if asset_type in ("us", "crypto") else "6mo"
        ticker = yf.Ticker(normalized)
        df = ticker.history(period=period)

        if df.empty:
            return None, None, f"無法取得 {symbol} 數據，請確認代號是否正確"

        # 均線：美股用 MA20/MA50/MA200，台股用 MA20/MA60，加密貨幣用 MA20/MA50
        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA50"] = df["Close"].rolling(50).mean()
        df["MA60"] = df["Close"].rolling(60).mean()
        df["MA200"] = df["Close"].rolling(200).mean()

        df["RSI"] = calculate_rsi(df["Close"])
        df["MACD"], df["Signal"], df["Histogram"] = calculate_macd(df["Close"])
        df["VolumeMA20"] = df["Volume"].rolling(20).mean()
        df["VolumeRatio"] = df["Volume"] / df["VolumeMA20"]

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # 52 週高低點
        df_1y = ticker.history(period="1y")
        high_52w = df_1y["High"].max()
        low_52w = df_1y["Low"].min()

        # MA20 斜率（5日變化率）
        ma20_slope = 0.0
        if len(df) >= 5 and not pd.isna(df.iloc[-5]["MA20"]):
            ma20_slope = (latest["MA20"] - df.iloc[-5]["MA20"]) / df.iloc[-5]["MA20"] * 100

        # 5日報酬
        return_5d = 0.0
        price_5d_ago = latest["Close"]
        if len(df) >= 5:
            price_5d_ago = df.iloc[-5]["Close"]
            return_5d = (latest["Close"] - price_5d_ago) / price_5d_ago * 100

        currency = "NT$" if asset_type == "taiwan" else "$"

        summary = {
            "symbol": symbol,
            "normalized_symbol": normalized,
            "asset_type": asset_type,
            "currency": currency,
            "current_price": round(float(latest["Close"]), 2),
            "change_pct": round(float((latest["Close"] - prev["Close"]) / prev["Close"] * 100), 2),
            "ma20": safe_round(latest["MA20"]),
            "ma50": safe_round(latest["MA50"]),   # 美股/加密貨幣主要均線
            "ma60": safe_round(latest["MA60"]),   # 台股季線
            "ma200": safe_round(latest["MA200"]), # 美股年線
            "rsi": safe_round(latest["RSI"], 1),
            "macd": safe_round(latest["MACD"], 4),
            "macd_signal": safe_round(latest["Signal"], 4),
            "macd_histogram": safe_round(latest["Histogram"], 4),
            "volume_ratio": safe_round(latest["VolumeRatio"]),
            "high_52w": safe_round(high_52w),
            "low_52w": safe_round(low_52w),
            "pct_from_52w_high": safe_round((latest["Close"] - high_52w) / high_52w * 100),
            "pct_from_52w_low": safe_round((latest["Close"] - low_52w) / low_52w * 100),
            "above_ma20": bool(latest["Close"] > latest["MA20"]) if not pd.isna(latest["MA20"]) else None,
            "above_ma50": bool(latest["Close"] > latest["MA50"]) if not pd.isna(latest["MA50"]) else None,
            "above_ma60": bool(latest["Close"] > latest["MA60"]) if not pd.isna(latest["MA60"]) else None,
            "above_ma200": bool(latest["Close"] > latest["MA200"]) if not pd.isna(latest["MA200"]) else None,
            "ma20_slope": safe_round(ma20_slope, 3),
            "return_5d": safe_round(return_5d),
            "price_5d_ago": safe_round(price_5d_ago),
        }

        return summary, df, None

    except Exception as e:
        return None, None, f"取得 {symbol} 數據時發生錯誤：{str(e)}"
