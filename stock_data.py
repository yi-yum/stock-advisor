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


def safe_round(val, decimals=2):
    return round(float(val), decimals) if (val is not None and not pd.isna(val)) else None


# ── 指標計算 ──────────────────────────────────────────

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
    return macd, signal_line, macd - signal_line


def calculate_bollinger(prices, period=20, std=2):
    mid = prices.rolling(period).mean()
    band = prices.rolling(period).std()
    return mid + std * band, mid, mid - std * band


def calculate_atr(df, period=14):
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"]  - df["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calculate_obv(df):
    return (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()


def detect_rsi_divergence(df):
    """
    最近30根K棒內偵測背離：
    - 底背離：價格創新低，RSI 未創新低
    - 頂背離：價格創新高，RSI 未創新高
    """
    if len(df) < 30:
        return "資料不足"

    recent = df.iloc[-30:]
    mid = 15  # 前半 / 後半切割點

    first_half = recent.iloc[:mid]
    second_half = recent.iloc[mid:]

    price_low_1  = first_half["Close"].min()
    price_low_2  = second_half["Close"].min()
    rsi_low_1    = first_half["RSI"].min()
    rsi_low_2    = second_half["RSI"].min()

    price_high_1 = first_half["Close"].max()
    price_high_2 = second_half["Close"].max()
    rsi_high_1   = first_half["RSI"].max()
    rsi_high_2   = second_half["RSI"].max()

    if price_low_2 < price_low_1 and rsi_low_2 > rsi_low_1 + 3:
        return "底背離（看漲訊號：價格創新低但RSI未跟隨）"
    elif price_high_2 > price_high_1 and rsi_high_2 < rsi_high_1 - 3:
        return "頂背離（看跌訊號：價格創新高但RSI未跟隨）"
    else:
        return "無明顯背離"


def get_earnings_info(ticker, asset_type):
    """抓美股下次財報日期"""
    if asset_type != "us":
        return None
    try:
        cal = ticker.calendar
        if cal is not None and not cal.empty:
            earnings_date = cal.iloc[0, 0]
            today = pd.Timestamp.now()
            days_to_earnings = (pd.Timestamp(earnings_date) - today).days
            if 0 <= days_to_earnings <= 90:
                return f"{days_to_earnings} 天後（{str(earnings_date)[:10]}）"
    except Exception:
        pass
    return None


# ── 多時框架 ─────────────────────────────────────────

def calc_indicators(df: pd.DataFrame) -> dict:
    if len(df) < 5:
        return None
    df = df.copy()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["RSI"]  = calculate_rsi(df["Close"])
    df["MACD"], df["Signal"], df["Hist"] = calculate_macd(df["Close"])

    latest = df.iloc[-1]
    rsi  = safe_round(latest["RSI"], 1)
    hist = safe_round(latest["Hist"], 4)
    ma20 = safe_round(latest["MA20"])
    ma50 = safe_round(latest["MA50"])
    close = safe_round(latest["Close"])

    bullish = sum([
        bool(close and ma20 and close > ma20),
        bool(close and ma50 and close > ma50),
        bool(hist and hist > 0),
        bool(rsi and rsi > 50),
    ])
    bias = "偏多" if bullish >= 3 else ("偏空" if bullish <= 1 else "中性")

    return {
        "close": close, "ma20": ma20, "ma50": ma50, "rsi": rsi,
        "macd_histogram": hist,
        "macd_direction": "金叉" if hist and hist > 0 else "死叉",
        "above_ma20": bool(close > ma20) if (close and ma20) else None,
        "above_ma50": bool(close > ma50) if (close and ma50) else None,
        "bias": bias,
    }


def resample_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    df = df_1h.copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df.resample("4h").agg({
        "Open": "first", "High": "max",
        "Low": "min", "Close": "last", "Volume": "sum",
    }).dropna()


# ── 主函式 ───────────────────────────────────────────

def get_stock_data(symbol: str):
    try:
        symbol = symbol.upper()
        asset_type = detect_asset_type(symbol)
        normalized = normalize_symbol(symbol)
        ticker = yf.Ticker(normalized)

        # 抓各時框數據
        df_w  = ticker.history(period="2y",  interval="1wk")
        df_d  = ticker.history(period="1y",  interval="1d")
        df_1h = ticker.history(period="60d", interval="1h")

        if df_d.empty:
            return None, None, f"無法取得 {symbol} 數據，請確認代號是否正確"

        df_4h = resample_to_4h(df_1h) if not df_1h.empty else pd.DataFrame()

        # ── 日線指標 ──────────────────────────────
        df_d["MA20"]  = df_d["Close"].rolling(20).mean()
        df_d["MA50"]  = df_d["Close"].rolling(50).mean()
        df_d["MA60"]  = df_d["Close"].rolling(60).mean()
        df_d["MA200"] = df_d["Close"].rolling(200).mean()
        df_d["RSI"]   = calculate_rsi(df_d["Close"])
        df_d["MACD"], df_d["Signal"], df_d["Histogram"] = calculate_macd(df_d["Close"])
        df_d["VolumeMA20"]  = df_d["Volume"].rolling(20).mean()
        df_d["VolumeRatio"] = df_d["Volume"] / df_d["VolumeMA20"]

        # 布林通道
        df_d["BB_upper"], df_d["BB_mid"], df_d["BB_lower"] = calculate_bollinger(df_d["Close"])

        # ATR
        df_d["ATR"] = calculate_atr(df_d)

        # OBV
        df_d["OBV"] = calculate_obv(df_d)

        latest = df_d.iloc[-1]
        prev   = df_d.iloc[-2]

        # 52 週高低
        df_1y    = ticker.history(period="1y")
        high_52w = df_1y["High"].max()
        low_52w  = df_1y["Low"].min()

        # MA20 斜率
        ma20_slope = 0.0
        if len(df_d) >= 5 and not pd.isna(df_d.iloc[-5]["MA20"]):
            ma20_slope = (latest["MA20"] - df_d.iloc[-5]["MA20"]) / df_d.iloc[-5]["MA20"] * 100

        # 5日報酬
        return_5d = 0.0
        price_5d_ago = float(latest["Close"])
        if len(df_d) >= 5:
            price_5d_ago = float(df_d.iloc[-5]["Close"])
            return_5d = (float(latest["Close"]) - price_5d_ago) / price_5d_ago * 100

        # ── 布林通道位置 ──────────────────────────
        close = float(latest["Close"])
        bb_upper = float(latest["BB_upper"])
        bb_lower = float(latest["BB_lower"])
        bb_mid   = float(latest["BB_mid"])
        bb_width = safe_round((bb_upper - bb_lower) / bb_mid * 100, 1)

        if close >= bb_upper * 0.99:
            bb_position = "觸及上軌（超買區域）"
        elif close <= bb_lower * 1.01:
            bb_position = "觸及下軌（超賣區域）"
        elif close > bb_mid:
            bb_position = "中軌上方（偏強）"
        else:
            bb_position = "中軌下方（偏弱）"

        # ── ATR 停損建議 ──────────────────────────
        atr = safe_round(latest["ATR"])
        atr_pct = safe_round(latest["ATR"] / close * 100, 1) if atr else None
        atr_stop = safe_round(close - 1.5 * float(latest["ATR"])) if atr else None

        # ── OBV 趨勢 ─────────────────────────────
        if len(df_d) >= 10:
            obv_now  = float(df_d.iloc[-1]["OBV"])
            obv_past = float(df_d.iloc[-10]["OBV"])
            price_chg = float(latest["Close"]) - float(df_d.iloc[-10]["Close"])
            if obv_now > obv_past and price_chg > 0:
                obv_signal = "OBV 上升，量能確認價格走勢（健康）"
            elif obv_now > obv_past and price_chg < 0:
                obv_signal = "OBV 上升但價格下跌，底部蓄勢（看漲）"
            elif obv_now < obv_past and price_chg > 0:
                obv_signal = "OBV 下降但價格上漲，量能背離（警示，可能假突破）"
            else:
                obv_signal = "OBV 下降，量能確認下跌趨勢"
        else:
            obv_signal = "資料不足"

        # ── RSI 背離 ─────────────────────────────
        rsi_divergence = detect_rsi_divergence(df_d)

        # ── 財報日期（美股）──────────────────────
        earnings_info = get_earnings_info(ticker, asset_type)

        currency = "NT$" if asset_type == "taiwan" else "$"

        summary = {
            "symbol": symbol,
            "normalized_symbol": normalized,
            "asset_type": asset_type,
            "currency": currency,
            "current_price": safe_round(latest["Close"]),
            "change_pct":    safe_round((latest["Close"] - prev["Close"]) / prev["Close"] * 100),
            "ma20":  safe_round(latest["MA20"]),
            "ma50":  safe_round(latest["MA50"]),
            "ma60":  safe_round(latest["MA60"]),
            "ma200": safe_round(latest["MA200"]),
            "rsi":   safe_round(latest["RSI"], 1),
            "macd":          safe_round(latest["MACD"], 4),
            "macd_signal":   safe_round(latest["Signal"], 4),
            "macd_histogram":safe_round(latest["Histogram"], 4),
            "volume_ratio":  safe_round(latest["VolumeRatio"]),
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
            # 新增指標
            "bb_upper":    safe_round(bb_upper),
            "bb_lower":    safe_round(bb_lower),
            "bb_position": bb_position,
            "bb_width":    bb_width,
            "atr":         atr,
            "atr_pct":     atr_pct,
            "atr_stop":    atr_stop,
            "obv_signal":  obv_signal,
            "rsi_divergence": rsi_divergence,
            "earnings_info":  earnings_info,
            # 多時框架
            "timeframes": {
                "weekly": calc_indicators(df_w),
                "daily":  calc_indicators(df_d),
                "h4":     calc_indicators(df_4h) if not df_4h.empty else None,
                "h1":     calc_indicators(df_1h) if not df_1h.empty else None,
            },
        }

        return summary, df_d, None

    except Exception as e:
        return None, None, f"取得 {symbol} 數據時發生錯誤：{str(e)}"
