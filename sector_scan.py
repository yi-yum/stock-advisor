import yfinance as yf
import pandas as pd
from stock_data import calculate_rsi, calculate_macd, detect_rsi_divergence, detect_candlestick_patterns, safe_round

SECTOR_STOCKS = {
    "半導體":     ["NVDA", "AMD", "AVGO", "QCOM", "AMAT", "MU", "TXN", "LRCX"],
    "軟體/SaaS":  ["MSFT", "ORCL", "CRM", "ADBE", "NOW", "INTU"],
    "通訊/社群":  ["GOOGL", "META", "NFLX", "DIS"],
    "電商/消費":  ["AMZN", "TSLA", "NKE", "MCD"],
    "金融":       ["JPM", "BAC", "GS", "V", "MA", "BRK-B"],
    "醫療/生技":  ["JNJ", "UNH", "LLY", "ABBV", "MRK", "PFE"],
    "能源":       ["XOM", "CVX", "COP", "SLB"],
    "工業":       ["CAT", "DE", "HON", "GE", "RTX"],
    "防禦/公用":  ["WMT", "COST", "PG", "NEE", "DUK"],
}


def _analyze_stock(symbol: str) -> dict | None:
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1y", interval="1d")
        if df.empty or len(df) < 30:
            return None

        df["RSI"] = calculate_rsi(df["Close"])
        df["MACD"], df["Signal"], df["Hist"] = calculate_macd(df["Close"])
        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA50"] = df["Close"].rolling(50).mean()

        close    = safe_round(float(df["Close"].iloc[-1]))
        prev     = safe_round(float(df["Close"].iloc[-2]))
        high_52w = safe_round(float(df["High"].max()))
        low_52w  = safe_round(float(df["Low"].min()))
        rsi      = safe_round(float(df["RSI"].iloc[-1]), 1)
        change   = safe_round((close - prev) / prev * 100) if prev else None
        ma20     = safe_round(float(df["MA20"].iloc[-1]))
        ma50     = safe_round(float(df["MA50"].iloc[-1]))
        hist_now  = float(df["Hist"].iloc[-1])
        hist_prev = float(df["Hist"].iloc[-2])
        pct_from_low  = safe_round((close - low_52w)  / low_52w  * 100, 1) if low_52w  else None
        pct_from_high = safe_round((close - high_52w) / high_52w * 100, 1) if high_52w else None

        score = 0
        signals = []

        # RSI
        if rsi is not None:
            if rsi < 30:
                score += 3
                signals.append(f"RSI {rsi} 深度超賣")
            elif rsi < 40:
                score += 2
                signals.append(f"RSI {rsi} 超賣")
            elif rsi < 50:
                score += 1
                signals.append(f"RSI {rsi} 偏弱")

        # 接近52週低點
        if pct_from_low is not None:
            if pct_from_low < 5:
                score += 3
                signals.append(f"距52週低點僅 {pct_from_low}%")
            elif pct_from_low < 15:
                score += 2
                signals.append(f"距52週低點 {pct_from_low}%")
            elif pct_from_low < 25:
                score += 1

        # RSI 底背離
        divergence = detect_rsi_divergence(df)
        if "底背離" in divergence:
            score += 2
            signals.append("RSI 底背離")

        # MACD 收窄或金叉
        if hist_now < 0 and hist_now > hist_prev:
            score += 2
            signals.append("MACD 死叉收窄")
        elif hist_now > 0 and hist_prev <= 0:
            score += 3
            signals.append("MACD 剛金叉")

        # K線型態
        patterns = detect_candlestick_patterns(df)
        bullish_kw = ["錘頭", "多頭吞噬", "早晨之星"]
        for p in patterns:
            if any(k in p for k in bullish_kw):
                score += 1
                signals.append(f"K線 {p.split('（')[0]}")

        # OBV 蓄勢
        if len(df) >= 10:
            price_chg = float(df["Close"].iloc[-1]) - float(df["Close"].iloc[-10])
            vol_chg   = float(df["Volume"].iloc[-5:].mean()) - float(df["Volume"].iloc[-10:-5].mean())
            if price_chg < 0 and vol_chg < 0:
                score += 1
                signals.append("價跌縮量（底部蓄勢）")

        return {
            "symbol":        symbol,
            "close":         close,
            "change":        change,
            "rsi":           rsi,
            "ma20":          ma20,
            "ma50":          ma50,
            "above_ma20":    bool(close > ma20) if (close and ma20) else None,
            "above_ma50":    bool(close > ma50) if (close and ma50) else None,
            "pct_from_low":  pct_from_low,
            "pct_from_high": pct_from_high,
            "score":         score,
            "signals":       signals,
        }

    except Exception:
        return None


def scan_sectors() -> dict[str, list[dict]]:
    """回傳 {板塊名稱: [股票分析結果, ...]}，每個板塊內依分數排序"""
    results = {}
    for sector, symbols in SECTOR_STOCKS.items():
        stocks = []
        for symbol in symbols:
            data = _analyze_stock(symbol)
            if data:
                stocks.append(data)
        stocks.sort(key=lambda x: x["score"], reverse=True)
        results[sector] = stocks
    return results
