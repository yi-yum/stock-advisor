import yfinance as yf
import pandas as pd
import numpy as np

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
    return round(float(val), decimals) if (val is not None and not pd.isna(val)) else None


def calc_indicators(df: pd.DataFrame) -> dict:
    """計算一個 DataFrame 的技術指標，回傳最新一根的摘要"""
    if len(df) < 5:
        return None

    df = df.copy()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["RSI"]  = calculate_rsi(df["Close"])
    df["MACD"], df["Signal"], df["Hist"] = calculate_macd(df["Close"])

    latest = df.iloc[-1]

    rsi   = safe_round(latest["RSI"], 1)
    macd  = safe_round(latest["MACD"], 4)
    hist  = safe_round(latest["Hist"], 4)
    ma20  = safe_round(latest["MA20"])
    ma50  = safe_round(latest["MA50"])
    close = safe_round(latest["Close"])

    bias = "中性"
    if rsi and ma20 and close:
        bullish = sum([
            close > ma20,
            ma50 and close > ma50,
            hist and hist > 0,
            rsi and rsi > 50,
        ])
        if bullish >= 3:
            bias = "偏多"
        elif bullish <= 1:
            bias = "偏空"

    return {
        "close":  close,
        "ma20":   ma20,
        "ma50":   ma50,
        "rsi":    rsi,
        "macd_histogram": hist,
        "macd_direction": "金叉" if hist and hist > 0 else "死叉",
        "above_ma20": bool(close > ma20) if (close and ma20) else None,
        "above_ma50": bool(close > ma50) if (close and ma50) else None,
        "bias": bias,
    }


def resample_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """把 1H 數據重採樣成 4H"""
    df = df_1h.copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df.resample("4h").agg({
        "Open": "first",
        "High": "max",
        "Low":  "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna()


def get_stock_data(symbol: str):
    try:
        symbol = symbol.upper()
        asset_type = detect_asset_type(symbol)
        normalized = normalize_symbol(symbol)
        ticker = yf.Ticker(normalized)

        # ── 週線（2年）────────────────────────────
        df_w = ticker.history(period="2y", interval="1wk")
        # ── 日線（1年）────────────────────────────
        df_d = ticker.history(period="1y",  interval="1d")
        # ── 1H（60天，用來組 4H）─────────────────
        df_1h = ticker.history(period="60d", interval="1h")

        if df_d.empty:
            return None, None, f"無法取得 {symbol} 數據，請確認代號是否正確"

        # 4H 重採樣
        df_4h = resample_to_4h(df_1h) if not df_1h.empty else pd.DataFrame()

        # ── 日線額外指標 ──────────────────────────
        df_d["MA20"]  = df_d["Close"].rolling(20).mean()
        df_d["MA50"]  = df_d["Close"].rolling(50).mean()
        df_d["MA60"]  = df_d["Close"].rolling(60).mean()
        df_d["MA200"] = df_d["Close"].rolling(200).mean()
        df_d["RSI"]   = calculate_rsi(df_d["Close"])
        df_d["MACD"], df_d["Signal"], df_d["Histogram"] = calculate_macd(df_d["Close"])
        df_d["VolumeMA20"] = df_d["Volume"].rolling(20).mean()
        df_d["VolumeRatio"] = df_d["Volume"] / df_d["VolumeMA20"]

        latest = df_d.iloc[-1]
        prev   = df_d.iloc[-2]

        df_1y    = ticker.history(period="1y")
        high_52w = df_1y["High"].max()
        low_52w  = df_1y["Low"].min()

        ma20_slope = 0.0
        if len(df_d) >= 5 and not pd.isna(df_d.iloc[-5]["MA20"]):
            ma20_slope = (latest["MA20"] - df_d.iloc[-5]["MA20"]) / df_d.iloc[-5]["MA20"] * 100

        return_5d   = 0.0
        price_5d_ago = float(latest["Close"])
        if len(df_d) >= 5:
            price_5d_ago = float(df_d.iloc[-5]["Close"])
            return_5d = (float(latest["Close"]) - price_5d_ago) / price_5d_ago * 100

        currency = "NT$" if asset_type == "taiwan" else "$"

        # ── 多時框架摘要 ──────────────────────────
        tf = {
            "weekly": calc_indicators(df_w),
            "daily":  calc_indicators(df_d),
            "h4":     calc_indicators(df_4h) if not df_4h.empty else None,
            "h1":     calc_indicators(df_1h) if not df_1h.empty else None,
        }

        summary = {
            "symbol":           symbol,
            "normalized_symbol": normalized,
            "asset_type":       asset_type,
            "currency":         currency,
            "current_price":    safe_round(latest["Close"]),
            "change_pct":       safe_round((latest["Close"] - prev["Close"]) / prev["Close"] * 100),
            "ma20":   safe_round(latest["MA20"]),
            "ma50":   safe_round(latest["MA50"]),
            "ma60":   safe_round(latest["MA60"]),
            "ma200":  safe_round(latest["MA200"]),
            "rsi":    safe_round(latest["RSI"], 1),
            "macd":         safe_round(latest["MACD"], 4),
            "macd_signal":  safe_round(latest["Signal"], 4),
            "macd_histogram": safe_round(latest["Histogram"], 4),
            "volume_ratio": safe_round(latest["VolumeRatio"]),
            "high_52w": safe_round(high_52w),
            "low_52w":  safe_round(low_52w),
            "pct_from_52w_high": safe_round((latest["Close"] - high_52w) / high_52w * 100),
            "pct_from_52w_low":  safe_round((latest["Close"] - low_52w)  / low_52w  * 100),
            "above_ma20":  bool(latest["Close"] > latest["MA20"])  if not pd.isna(latest["MA20"])  else None,
            "above_ma50":  bool(latest["Close"] > latest["MA50"])  if not pd.isna(latest["MA50"])  else None,
            "above_ma60":  bool(latest["Close"] > latest["MA60"])  if not pd.isna(latest["MA60"])  else None,
            "above_ma200": bool(latest["Close"] > latest["MA200"]) if not pd.isna(latest["MA200"]) else None,
            "ma20_slope":  safe_round(ma20_slope, 3),
            "return_5d":   safe_round(return_5d),
            "price_5d_ago": safe_round(price_5d_ago),
            "timeframes":  tf,
        }

        return summary, df_d, None

    except Exception as e:
        return None, None, f"取得 {symbol} 數據時發生錯誤：{str(e)}"
