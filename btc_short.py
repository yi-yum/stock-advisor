import requests
import yfinance as yf
import pandas as pd
from stock_data import (
    calculate_rsi, calculate_macd, calculate_bollinger,
    calculate_atr, detect_candlestick_patterns, safe_round
)

BINANCE_FAPI = "https://fapi.binance.com"

# 支援的幣種（需有 Binance 合約）
SUPPORTED_COINS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "AVAX", "LINK", "ADA", "SUI"]


def _binance_symbol(coin: str) -> str:
    return f"{coin.upper()}USDT"

def _yf_symbol(coin: str) -> str:
    return f"{coin.upper()}-USD"


# ── Binance 公開 API ──────────────────────────────────

def get_funding_rate(coin: str) -> dict | None:
    try:
        r = requests.get(f"{BINANCE_FAPI}/fapi/v1/premiumIndex",
                         params={"symbol": _binance_symbol(coin)}, timeout=5)
        d = r.json()
        fr = safe_round(float(d["lastFundingRate"]) * 100, 4)
        return {
            "funding_rate": fr,
            "mark_price":   safe_round(float(d["markPrice"])),
        }
    except Exception:
        return None


def get_open_interest(coin: str) -> dict | None:
    try:
        r = requests.get(f"{BINANCE_FAPI}/futures/data/openInterestHist",
                         params={"symbol": _binance_symbol(coin), "period": "1h", "limit": 7}, timeout=5)
        hist = r.json()
        if not hist or len(hist) < 2:
            return None
        oi_now    = float(hist[-1]["sumOpenInterest"])
        oi_6h_ago = float(hist[0]["sumOpenInterest"])
        oi_change = safe_round((oi_now - oi_6h_ago) / oi_6h_ago * 100, 2)
        return {
            "oi":           safe_round(oi_now),
            "oi_change_6h": oi_change,
        }
    except Exception:
        return None


def get_long_short_ratio(coin: str) -> dict | None:
    try:
        r = requests.get(f"{BINANCE_FAPI}/futures/data/globalLongShortAccountRatio",
                         params={"symbol": _binance_symbol(coin), "period": "5m", "limit": 1}, timeout=5)
        data = r.json()
        if not data:
            return None
        return {
            "ratio":     safe_round(float(data[0]["longShortRatio"]), 3),
            "long_pct":  safe_round(float(data[0]["longAccount"])  * 100, 1),
            "short_pct": safe_round(float(data[0]["shortAccount"]) * 100, 1),
        }
    except Exception:
        return None


# ── 多時框架技術分析 ─────────────────────────────────

def _calc_tf(df: pd.DataFrame) -> dict | None:
    if df.empty or len(df) < 20:
        return None

    df = df.copy()
    df["RSI"]  = calculate_rsi(df["Close"])
    df["MACD"], df["Signal"], df["Hist"] = calculate_macd(df["Close"])
    df["BB_upper"], df["BB_mid"], df["BB_lower"] = calculate_bollinger(df["Close"])
    df["ATR"]  = calculate_atr(df)

    latest    = df.iloc[-1]
    close     = float(latest["Close"])
    rsi       = safe_round(float(df["RSI"].iloc[-1]), 1)
    hist_now  = float(df["Hist"].iloc[-1])
    hist_prev = float(df["Hist"].iloc[-2])
    atr       = float(latest["ATR"]) if not pd.isna(latest["ATR"]) else 0
    bb_mid    = float(latest["BB_mid"])
    bb_upper  = float(latest["BB_upper"])
    bb_lower  = float(latest["BB_lower"])

    if close >= bb_upper * 0.99:
        bb_pos = "觸上軌（超買）"
    elif close <= bb_lower * 1.01:
        bb_pos = "觸下軌（超賣）"
    elif close > bb_mid:
        bb_pos = "中軌上方"
    else:
        bb_pos = "中軌下方"

    if hist_now > 0 and hist_prev <= 0:
        macd_status = "剛金叉 🟢"
    elif hist_now < 0 and hist_prev >= 0:
        macd_status = "剛死叉 🔴"
    elif hist_now > 0:
        macd_status = "金叉持續"
    else:
        macd_status = "死叉持續"

    bull = sum([rsi > 50, hist_now > 0, close > bb_mid])
    bias = "偏多" if bull >= 2 else "偏空"

    patterns = detect_candlestick_patterns(df)

    return {
        "close":         safe_round(close),
        "rsi":           rsi,
        "macd_status":   macd_status,
        "bb_position":   bb_pos,
        "atr":           safe_round(atr),
        "stop_long":     safe_round(close - 1.5 * atr),
        "stop_short":    safe_round(close + 1.5 * atr),
        "bias":          bias,
        "patterns":      patterns,
        "df":            df,  # 供後續計算擺盪點使用
    }


def _find_swing_target(df: pd.DataFrame, close: float, atr: float, direction: str) -> float:
    """
    從近50根1H K棒找最近的擺盪高/低點作為目標價。
    若找不到合理目標則退回 2x ATR。
    """
    recent = df.iloc[-50:] if len(df) >= 50 else df

    if direction == "long":
        # 找價格上方最近的擺盪高點（局部最高：左右各2根更低）
        candidates = []
        for i in range(2, len(recent) - 2):
            h = float(recent["High"].iloc[i])
            if (h > float(recent["High"].iloc[i-1]) and
                h > float(recent["High"].iloc[i-2]) and
                h > float(recent["High"].iloc[i+1]) and
                h > float(recent["High"].iloc[i+2]) and
                h > close):
                candidates.append(h)
        if candidates:
            target = min(candidates)           # 最近（最低）���上方阻力
            if target > close + 0.5 * atr:    # 至少要離現價半個ATR
                return safe_round(target)
        return safe_round(close + 2.0 * atr)  # 退回方案

    else:  # short
        candidates = []
        for i in range(2, len(recent) - 2):
            l = float(recent["Low"].iloc[i])
            if (l < float(recent["Low"].iloc[i-1]) and
                l < float(recent["Low"].iloc[i-2]) and
                l < float(recent["Low"].iloc[i+1]) and
                l < float(recent["Low"].iloc[i+2]) and
                l < close):
                candidates.append(l)
        if candidates:
            target = max(candidates)           # 最近（最高）的下方支撐
            if target < close - 0.5 * atr:
                return safe_round(target)
        return safe_round(close - 2.0 * atr)


def calculate_anchored_vwap(df: pd.DataFrame, anchor: str = "week", length: int = 14) -> dict | None:
    """
    錨定 VWAP + 標準差帶，精確對齊 TradingView Pine Script ta.vwap() 公式：

    每根 K 棒從錨點起累積：
        sumSrcVol    += src × vol          (src = (H+L+C)/3)
        sumVol       += vol
        sumSrcSrcVol += src² × vol

    VWAP     = sumSrcVol / sumVol
    variance = max(sumSrcSrcVol / sumVol − VWAP², 0)
    std      = sqrt(variance)
    帶狀     = VWAP ± n × std

    anchor="week" → 本週一 00:00 UTC
    anchor="day"  → 今日 00:00 UTC
    length 參數保留供 UI 顯示用（TradingView 預設 14）
    """
    if df is None or df.empty or len(df) < 5:
        return None
    try:
        df = df.copy()
        df.index = pd.to_datetime(df.index, utc=True)
        last_ts = df.index[-1]

        # ── 決定錨點 ──────────────────────────────
        if anchor == "week":
            anchor_ts = (last_ts - pd.Timedelta(days=last_ts.weekday())).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        else:  # day
            anchor_ts = last_ts.replace(hour=0, minute=0, second=0, microsecond=0)

        anchored = df[df.index >= anchor_ts].copy()
        if len(anchored) < 8:
            anchored = df.iloc[-96:].copy()
            anchor_ts = anchored.index[0]

        # ── TradingView 精確公式 ───────────────────
        src = (anchored["High"] + anchored["Low"] + anchored["Close"]) / 3  # (H+L+C)/3
        vol = anchored["Volume"]

        cum_src_vol    = (src * vol).cumsum()
        cum_vol        = vol.cumsum()
        cum_src2_vol   = (src ** 2 * vol).cumsum()

        vwap_series    = cum_src_vol / cum_vol
        variance_series = (cum_src2_vol / cum_vol - vwap_series ** 2).clip(lower=0)
        std_series     = variance_series ** 0.5

        vwap_now = float(vwap_series.iloc[-1])
        std_now  = float(std_series.iloc[-1])
        close    = float(df["Close"].iloc[-1])

        if vwap_now == 0:
            return None
        if std_now == 0:
            std_now = vwap_now * 0.005  # fallback 0.5%

        diff_pct = (close - vwap_now) / vwap_now * 100

        if diff_pct > 0.3:
            position, bias = "VWAP 上方", "偏多"
        elif diff_pct < -0.3:
            position, bias = "VWAP 下方", "偏空"
        else:
            position, bias = "貼近 VWAP", "中性"

        return {
            "vwap":         safe_round(vwap_now),
            "upper_1":      safe_round(vwap_now + std_now),
            "upper_2":      safe_round(vwap_now + 2 * std_now),
            "lower_1":      safe_round(vwap_now - std_now),
            "lower_2":      safe_round(vwap_now - 2 * std_now),
            "std":          safe_round(std_now),
            "diff_pct":     safe_round(diff_pct, 2),
            "position":     position,
            "bias":         bias,
            "anchor_time":  anchor_ts.strftime("%m/%d %H:%M"),
            "anchor_label": "週錨" if anchor == "week" else "日錨",
            "bars":         len(anchored),
            "length":       length,
            "long_entry":   safe_round(vwap_now),
            "long_stop":    safe_round(vwap_now - std_now),
            "long_target":  safe_round(vwap_now + 2 * std_now),
            "rr_long":      2.0,
            "short_entry":  safe_round(vwap_now),
            "short_stop":   safe_round(vwap_now + std_now),
            "short_target": safe_round(vwap_now - 2 * std_now),
            "rr_short":     2.0,
        }
    except Exception:
        return None


def get_tf_data(coin: str) -> tuple[dict, pd.DataFrame | None]:
    """回傳 (tf_data dict, 15m 原始 df)，15m df 供 VWAP 計算使用"""
    ticker = yf.Ticker(_yf_symbol(coin))
    configs = [
        ("1h",  "60d",  "1h"),
        ("15m", "20d",  "15m"),
        ("5m",  "7d",   "5m"),
    ]
    results   = {}
    df_15m_raw = None
    for label, period, interval in configs:
        try:
            df = ticker.history(period=period, interval=interval)
            results[label] = _calc_tf(df)
            if label == "15m":
                df_15m_raw = df
        except Exception:
            results[label] = None
    return results, df_15m_raw


# ── 綜合訊號評分 ─────────────────────────────────────

def get_btc_short_data(coin: str = "BTC", anchor: str = "week", vwap_length: int = 14) -> dict:
    coin     = coin.upper()
    funding  = get_funding_rate(coin)
    oi       = get_open_interest(coin)
    ls_ratio = get_long_short_ratio(coin)
    tf_data, df_15m_raw = get_tf_data(coin)
    vwap_data = calculate_anchored_vwap(df_15m_raw, anchor=anchor, length=vwap_length)

    score   = 0
    signals = []

    # 資金費率
    if funding:
        fr = funding["funding_rate"]
        if fr > 0.05:
            score -= 2
            signals.append(("空", f"資金費率 {fr}% 極高，多方過擁擠"))
        elif fr > 0.02:
            score -= 1
            signals.append(("空", f"資金費率 {fr}% 偏高"))
        elif fr < -0.02:
            score += 2
            signals.append(("多", f"資金費率 {fr}% 為負，空方付費"))
        elif fr < 0:
            score += 1
            signals.append(("多", f"資金費率 {fr}% 略負"))
        else:
            signals.append(("中", f"資金費率 {fr}% 中性"))

    # OI
    if oi and oi.get("oi_change_6h") is not None:
        chg = oi["oi_change_6h"]
        h1_bias = (tf_data.get("1h") or {}).get("bias", "")
        if abs(chg) >= 2:
            if (chg > 0 and h1_bias == "偏多"):
                score += 1
                signals.append(("多", f"OI 6h +{chg}%，多方加倉"))
            elif (chg > 0 and h1_bias == "偏空"):
                score -= 1
                signals.append(("空", f"OI 6h +{chg}%，空方加倉"))
            elif chg < 0:
                signals.append(("中", f"OI 6h {chg}%，倉位出清中"))

    # 多空比（逆向指標）
    if ls_ratio:
        ratio = ls_ratio["ratio"]
        if ratio > 1.8:
            score -= 2
            signals.append(("空", f"多空比 {ratio}，多方極度擁擠（逆向空）"))
        elif ratio > 1.4:
            score -= 1
            signals.append(("空", f"多空比 {ratio}，多方偏多"))
        elif ratio < 0.7:
            score += 2
            signals.append(("多", f"多空比 {ratio}，空方極度擁擠（逆向多）"))
        elif ratio < 0.9:
            score += 1
            signals.append(("多", f"多空比 {ratio}，空方偏多"))
        else:
            signals.append(("中", f"多空比 {ratio} 中性"))

    # 錨定 VWAP 位置（15m）
    if vwap_data:
        label = vwap_data.get("anchor_label", "VWAP")
        if vwap_data["bias"] == "偏多":
            score += 1
            signals.append(("多", f"錨定VWAP（{label}）：現價在 {vwap_data['position']}，偏離 {vwap_data['diff_pct']:+}%（VWAP={vwap_data['vwap']}）"))
        elif vwap_data["bias"] == "偏空":
            score -= 1
            signals.append(("空", f"錨定VWAP（{label}）：現價在 {vwap_data['position']}，偏離 {vwap_data['diff_pct']:+}%（VWAP={vwap_data['vwap']}）"))
        else:
            signals.append(("中", f"錨定VWAP（{label}）：現價貼近 VWAP（{vwap_data['vwap']}），方向待確認"))

    # 多時框架技術面
    for tf, label in [("1h", "1H"), ("15m", "15M"), ("5m", "5M")]:
        d = tf_data.get(tf)
        if d:
            if d["bias"] == "偏多":
                score += 1
                signals.append(("多", f"{label} 技術面偏多（RSI {d['rsi']}，{d['macd_status']}）"))
            else:
                score -= 1
                signals.append(("空", f"{label} 技術面偏空（RSI {d['rsi']}，{d['macd_status']}）"))

    # 方向判斷
    if score >= 4:
        direction = "做多"
    elif score <= -4:
        direction = "做空"
    elif score >= 2:
        direction = "偏多觀望"
    elif score <= -2:
        direction = "偏空觀望"
    else:
        direction = "觀望"

    # 進場參考（停損用 ATR，目標用擺盪點）
    entry_ref = stop_long = stop_short = target_long = target_short = None
    rr_long = rr_short = None
    h1 = tf_data.get("1h")
    if h1:
        entry_ref  = h1["close"]
        stop_long  = h1["stop_long"]
        stop_short = h1["stop_short"]
        atr        = h1["atr"]
        df_1h      = h1.get("df")

        if df_1h is not None and atr:
            target_long  = _find_swing_target(df_1h, entry_ref, atr, "long")
            target_short = _find_swing_target(df_1h, entry_ref, atr, "short")
        else:
            target_long  = safe_round(entry_ref + 2.0 * atr) if atr else None
            target_short = safe_round(entry_ref - 2.0 * atr) if atr else None

        # 風報比
        if stop_long and target_long and entry_ref > stop_long:
            rr_long = safe_round((target_long - entry_ref) / (entry_ref - stop_long), 2)
        if stop_short and target_short and stop_short > entry_ref:
            rr_short = safe_round((entry_ref - target_short) / (stop_short - entry_ref), 2)

    return {
        "coin":         coin,
        "funding":      funding,
        "oi":           oi,
        "ls_ratio":     ls_ratio,
        "tf_data":      tf_data,
        "vwap_data":    vwap_data,
        "score":        score,
        "signals":      signals,
        "direction":    direction,
        "entry_ref":    entry_ref,
        "stop_long":    stop_long,
        "stop_short":   stop_short,
        "target_long":  target_long,
        "target_short": target_short,
        "rr_long":      rr_long,
        "rr_short":     rr_short,
    }
