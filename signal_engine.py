"""
signal_engine.py — 統一訊號引擎
所有頁面（自選股、每日掃描、回測）都從這裡取得訊號邏輯。

策略：結構性回調（Structural Pullback）
核心思路：找處於強勢上升趨勢中、回調至關鍵均線附近的股票，
等待出現轉強訊號後進場，利用結構性支撐控制風險。

三層過濾：
  Layer 1 — 趨勢品質（0–6分，需≥4才進入第二層）
    EMA20 > EMA50           +2  短中期多頭排列
    EMA50 > EMA200          +2  長期趨勢向上（需200日資料）
    週線 MACD 柱體 > 0      +1  大週期多頭
    ADX > 20                +1  趨勢有方向性

  Layer 2 — 回調品質（三項需同時滿足，否則降為WATCH上限）
    距 EMA20 在 -8% ~ +2%   回調到位，沒有過度延伸
    RSI 40–62               健康回調（非崩盤、非超買）
    量比 < 1.0              縮量回調，籌碼穩定

  Layer 3 — 轉強訊號（0–10分加分項）
    MACD柱體連續2日遞增     +2  動能恢復
    MACD柱體開始回升        +1  動能初步好轉
    今日收紅K               +2  多方接手
    縮量後放量紅K           +1  反彈訊號強化（額外加分）
    RSI向上穿越50           +2  動能方向確認
    RSI站上50               +1  多頭動能
    收盤在日內高點80%以上   +1  尾盤強勢
    DI+ > DI-               +1  多方主導

決策：
  回調有效 + 轉強≥7 → BUY（風報比<1.5時降為WATCH）
  回調有效 + 轉強≥5，或回調無效 + 轉強≥6 → WATCH
  其餘 → WAIT
"""

import pandas as pd
import numpy as np
from typing import Optional


# ── 加碼點位（分批建倉）───────────────────────────────────────────────────────

def calc_add_on_levels(entry_price: Optional[float], sl: Optional[float]) -> list[dict]:
    """
    以 R 倍數（風險單位）計算金字塔式加碼點位：
      首倉 50% → +1R 加碼 30%（停損上移至成本價，鎖定本金）
             → +2R 加碼 20%（停損上移至加碼點1，鎖定第一階段獲利）
    每次加碼後倉位遞減、停損遞進，避免愈攤愈重、也讓獲利逐步落袋為安。
    entry_price/sl 任一缺失或風險為 0 時回傳空list（無法計算R值）。
    """
    if entry_price is None or sl is None or entry_price <= sl:
        return []
    risk = entry_price - sl
    add1_price = entry_price + 1.0 * risk
    add2_price = entry_price + 2.0 * risk
    return [
        {
            "level": 1,
            "label": "加碼點1（+1R）",
            "price": round(add1_price, 4),
            "weight_pct": 30,
            "new_stop": round(entry_price, 4),
            "new_stop_label": "停損上移至成本價",
        },
        {
            "level": 2,
            "label": "加碼點2（+2R）",
            "price": round(add2_price, 4),
            "weight_pct": 20,
            "new_stop": round(add1_price, 4),
            "new_stop_label": "停損上移至加碼點1",
        },
    ]


# ── 指標計算 ──────────────────────────────────────────────────────────────────

def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    輸入含 Open/High/Low/Close/Volume 的 DataFrame，
    回傳附加所有技術指標欄位的 DataFrame。
    """
    df = df.copy()
    close = df["Close"]

    # RSI(14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).ewm(span=14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(span=14, adjust=False).mean()
    df["RSI"] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = macd - sig

    # MA / EMA
    df["MA20"]   = close.rolling(20).mean()
    df["MA50"]   = close.rolling(50).mean()
    df["EMA20"]  = close.ewm(span=20, adjust=False).mean()
    df["EMA50"]  = close.ewm(span=50, adjust=False).mean()
    df["EMA200"] = close.ewm(span=200, adjust=False).mean()

    # Bollinger Bands
    std20 = close.rolling(20).std()
    df["BB_upper"] = df["MA20"] + 2 * std20
    df["BB_lower"] = df["MA20"] - 2 * std20

    # ATR(14)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - close.shift()).abs(),
        (df["Low"]  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(span=14, adjust=False).mean()

    # ADX + DI+/DI-
    up    = df["High"].diff()
    down  = -df["Low"].diff()
    dm_p  = up.where((up > down) & (up > 0), 0.0)
    dm_m  = down.where((down > up) & (down > 0), 0.0)
    atr_s = tr.ewm(span=14, adjust=False).mean()
    df["DI_P"] = 100 * dm_p.ewm(span=14, adjust=False).mean() / (atr_s + 1e-10)
    df["DI_M"] = 100 * dm_m.ewm(span=14, adjust=False).mean() / (atr_s + 1e-10)
    denom  = (df["DI_P"] + df["DI_M"]).replace(0, float("nan"))
    dx     = 100 * (df["DI_P"] - df["DI_M"]).abs() / denom
    df["ADX"] = dx.ewm(span=14, adjust=False).mean()

    # Volume ratio（截斷異常大量後算均量，避免單日爆量污染）
    vol_median50      = df["Volume"].rolling(50, min_periods=10).median()
    vol_capped        = df["Volume"].clip(upper=vol_median50 * 10)
    df["VolumeMA20"]  = vol_capped.rolling(20).mean()
    df["VolumeRatio"] = df["Volume"] / (df["VolumeMA20"] + 1e-10)

    # 20日高低點（不含當日，避免 look-ahead bias）
    df["High20"] = close.rolling(20).max().shift(1)
    df["Low20"]  = close.rolling(20).min().shift(1)

    return df


def _calc_weekly_macd(df: pd.DataFrame) -> Optional[bool]:
    """
    從日線重採樣至週線，計算週線 MACD 柱體方向。
    回傳 True（正值）/ False（負值）/ None（資料不足）。
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        return None
    try:
        weekly = df["Close"].resample("W").last().dropna()
        if len(weekly) < 35:
            return None
        ema12 = weekly.ewm(span=12, adjust=False).mean()
        ema26 = weekly.ewm(span=26, adjust=False).mean()
        macd  = ema12 - ema26
        sig   = macd.ewm(span=9, adjust=False).mean()
        hist  = float((macd - sig).iloc[-1])
        return hist > 0
    except Exception:
        return None


# ── 主要訊號函式 ───────────────────────────────────────────────────────────────

def get_signal(df: pd.DataFrame) -> dict:
    """
    輸入 OHLCV DataFrame（至少 60 根 K 棒），
    回傳完整的訊號分析結果。

    回傳 dict 包含：
    - regime: 市場狀態（TREND_UP / TREND_DOWN / UNCERTAIN）
    - strategy: 適用策略名稱
    - signal: 'BUY' | 'WATCH' | 'WAIT'
    - signal_label: 中文標籤
    - can_enter: bool
    - score: 綜合分數（趨勢層 + 轉強層）
    - trend_score: 趨勢品質分數（0–6）
    - entry_score: 轉強訊號分數（0–10）
    - pullback_valid: 回調品質是否達標
    - reasons: list[str]
    - entry_price: 建議進場價
    - sl: 止損價
    - tp: 止盈價
    - rr: 風報比
    - trigger_reasons: list[str]（技術觸發條件，供掃描器用）
    - indicators: dict（各指標數值）
    """
    # ── 計算指標 ──────────────────────────────────────────────
    weekly_macd_pos = _calc_weekly_macd(df)   # 重採樣前先算，保留 DatetimeIndex
    df = calc_indicators(df)
    df.dropna(subset=["MA20", "MA50", "EMA20", "EMA50", "RSI", "ATR", "ADX"], inplace=True)

    if len(df) < 10:
        return _empty_signal("資料不足")

    last  = df.iloc[-1]
    prev  = df.iloc[-2] if len(df) >= 2 else last
    prev2 = df.iloc[-3] if len(df) >= 3 else prev

    close      = float(last["Close"])
    open_      = float(last["Open"]) if "Open" in last else close
    high_      = float(last["High"]) if "High" in last else close
    low_       = float(last["Low"])  if "Low"  in last else close
    rsi        = float(last["RSI"])
    macd_hist  = float(last["MACD_hist"])
    prev_macd  = float(prev["MACD_hist"])
    prev2_macd = float(prev2["MACD_hist"])
    ema20      = float(last["EMA20"])
    ema50      = float(last["EMA50"])
    ema200_raw = last.get("EMA200", float("nan"))
    ema200     = float(ema200_raw) if not pd.isna(ema200_raw) else None
    ma20       = float(last["MA20"])
    atr        = float(last["ATR"])
    bb_lower   = float(last["BB_lower"])
    di_p       = float(last["DI_P"])
    di_m       = float(last["DI_M"])
    adx        = float(last["ADX"])
    vol_ratio  = float(last["VolumeRatio"])
    prev_vol   = float(prev["VolumeRatio"])
    high20     = float(last["High20"]) if not pd.isna(last["High20"]) else close
    low20      = float(last["Low20"])  if not pd.isna(last["Low20"])  else close
    prev_rsi   = float(prev["RSI"])
    change_pct = round((close - float(prev["Close"])) / (float(prev["Close"]) + 1e-10) * 100, 2)

    # ── 技術觸發標籤（前端快速顯示用）─────────────────────────
    trigger_reasons = []
    if rsi < 35:              trigger_reasons.append("RSI<35")
    if rsi > 65:              trigger_reasons.append("RSI>65")
    if vol_ratio >= 1.5:      trigger_reasons.append(f"爆量({vol_ratio:.1f}x)")
    if close > high20:        trigger_reasons.append("突破20日高點")
    if close < bb_lower:      trigger_reasons.append("跌破布林下軌")
    if rsi > prev_rsi > float(prev2["RSI"]) and rsi > 45:
        trigger_reasons.append("RSI連升")

    # ── 指標摘要 ──────────────────────────────────────────────
    indicators = {
        "rsi":            round(rsi, 1),
        "macd_hist":      round(macd_hist, 4),
        "adx":            round(adx, 1),
        "di_plus":        round(di_p, 1),
        "di_minus":       round(di_m, 1),
        "ema20":          round(ema20, 4),
        "ema50":          round(ema50, 4),
        "ema200":         round(ema200, 4) if ema200 else None,
        "ma20":           round(ma20, 4),
        "atr":            round(atr, 4),
        "bb_lower":       round(bb_lower, 4),
        "volume_ratio":   round(vol_ratio, 2),
        "change_pct":     change_pct,
        "weekly_macd_pos": weekly_macd_pos,
    }

    reasons = []

    # ════════════════════════════════════════════════════════
    # Layer 1：趨勢品質評分（0–6）
    # ════════════════════════════════════════════════════════
    trend_score = 0

    if ema20 > ema50:
        trend_score += 2
        reasons.append(f"✅ EMA20({ema20:.2f}) > EMA50({ema50:.2f})，短中期多頭排列")
    else:
        reasons.append(f"❌ EMA20({ema20:.2f}) < EMA50({ema50:.2f})，短期趨勢偏弱")

    if ema200 is not None:
        if ema50 > ema200:
            trend_score += 2
            reasons.append(f"✅ EMA50 > EMA200({ema200:.2f})，長期趨勢向上")
        else:
            reasons.append(f"❌ EMA50 < EMA200({ema200:.2f})，長期趨勢偏弱")

    if weekly_macd_pos is True:
        trend_score += 1
        reasons.append("✅ 週線 MACD 柱體正值，大週期多頭")
    elif weekly_macd_pos is False:
        reasons.append("❌ 週線 MACD 柱體負值，大週期偏空")

    if adx > 20:
        trend_score += 1
        reasons.append(f"✅ ADX {adx:.1f} > 20，趨勢有方向性")
    else:
        reasons.append(f"⚠️ ADX {adx:.1f} < 20，趨勢方向不明顯")

    # 趨勢不足：直接返回 WAIT，不進後續評分
    if trend_score < 4:
        regime = "TREND_DOWN" if ema20 < ema50 else "UNCERTAIN"
        reasons.append(f"趨勢品質分數 {trend_score}/6（需≥4），暫不入場")
        return {
            "regime":        regime,
            "strategy":      "趨勢不足",
            "signal":        "WAIT",
            "signal_label":  "趨勢不足",
            "can_enter":     False,
            "score":         trend_score,
            "trend_score":   trend_score,
            "entry_score":   0,
            "pullback_valid": False,
            "reasons":       reasons,
            "entry_price":   round(close, 4),
            "sl": None, "tp": None, "rr": None,
            "trigger_reasons": trigger_reasons,
            "indicators":    indicators,
        }

    # ════════════════════════════════════════════════════════
    # Layer 2：回調品質（三項需同時達標）
    # ════════════════════════════════════════════════════════
    dist_ema20_pct = (close - ema20) / ema20 * 100

    pb_dist  = -8.0 <= dist_ema20_pct <= 2.0
    pb_rsi   = 40 <= rsi <= 62
    pb_vol   = vol_ratio < 1.0

    pullback_valid = pb_dist and pb_rsi and pb_vol

    if pullback_valid:
        reasons.append(
            f"✅ 健康回調：距EMA20 {dist_ema20_pct:+.1f}%，"
            f"RSI {rsi:.1f}，量比 {vol_ratio:.2f}x（縮量）"
        )
    else:
        msgs = []
        if not pb_dist:
            if dist_ema20_pct < -8.0:
                msgs.append(f"回調過深（距EMA20 {dist_ema20_pct:+.1f}%，可能趨勢轉弱）")
            else:
                msgs.append(f"偏離EMA20過遠（{dist_ema20_pct:+.1f}%，追高風險）")
        if not pb_rsi:
            if rsi < 40:
                msgs.append(f"RSI {rsi:.1f} 偏低（賣壓仍重）")
            else:
                msgs.append(f"RSI {rsi:.1f} 偏高（上方空間有限）")
        if not pb_vol:
            msgs.append(f"量比 {vol_ratio:.2f}x（放量回調，賣壓偏重）")
        reasons.append("⚠️ 回調品質未達標：" + "；".join(msgs))

    # ════════════════════════════════════════════════════════
    # Layer 3：轉強訊號評分（0–10）
    # ════════════════════════════════════════════════════════
    entry_score = 0

    # MACD 動能
    macd_rising_2d = macd_hist > prev_macd > prev2_macd
    if macd_rising_2d:
        entry_score += 2
        reasons.append("✅ MACD 柱體連續 2 日遞增，動能恢復")
    elif macd_hist > prev_macd:
        entry_score += 1
        reasons.append("⚠️ MACD 柱體開始回升（1日）")

    # 今日 K 棒顏色
    is_green = close > open_
    if is_green:
        entry_score += 2
        reasons.append(f"✅ 今日收紅 K（+{change_pct:.1f}%），多方接手")

    # 縮量後放量紅K（額外加分）
    if is_green and vol_ratio > 1.0 and prev_vol < 0.9:
        entry_score += 1
        reasons.append(f"✅ 縮量後放量紅 K（量比 {vol_ratio:.1f}x），強烈反彈訊號")

    # RSI 穿越 50
    if rsi > 50 and prev_rsi <= 50:
        entry_score += 2
        reasons.append(f"✅ RSI 由下往上穿越 50（{prev_rsi:.1f}→{rsi:.1f}），動能方向確認")
    elif rsi > 50:
        entry_score += 1
        reasons.append(f"✅ RSI {rsi:.1f} 站上 50，多頭動能")

    # 收盤位置（尾盤強弱）
    day_range = high_ - low_
    if day_range > 1e-6:
        close_pos = (close - low_) / day_range
        if close_pos >= 0.8:
            entry_score += 1
            reasons.append(f"✅ 收盤在日內高點 {close_pos*100:.0f}%，尾盤強勢")

    # DI+ > DI-
    if di_p > di_m:
        entry_score += 1
        reasons.append(f"✅ DI+({di_p:.1f}) > DI-({di_m:.1f})，多方主導")
    else:
        reasons.append(f"⚠️ DI-({di_m:.1f}) > DI+({di_p:.1f})，買方力道偏弱")

    # ════════════════════════════════════════════════════════
    # 止損止盈計算
    # ════════════════════════════════════════════════════════
    # 止損：EMA50 * 0.98 與 20日低點 * 0.99 中取較近者（保護更大）
    sl_candidates = [ema50 * 0.98, low20 * 0.99, close - 2.0 * atr]
    sl_raw = max(sl_candidates)   # 取最高（最靠近價格），保護最大
    risk   = close - sl_raw

    sl = tp = rr = None
    actual_rr = 0.0
    add_on_levels: list[dict] = []
    if risk > 0:
        sl        = round(sl_raw, 4)
        tp        = round(close + 2.5 * risk, 4)
        rr        = 2.5
        actual_rr = (tp - close) / risk
        add_on_levels = calc_add_on_levels(close, sl)

    # ════════════════════════════════════════════════════════
    # 訊號決定
    # ════════════════════════════════════════════════════════
    regime   = "TREND_UP"
    strategy = "結構性回調"

    if pullback_valid and entry_score >= 7:
        if actual_rr >= 1.5:
            signal, signal_label, can_enter = "BUY", "回調再起", True
        else:
            signal, signal_label, can_enter = "WATCH", f"觀察（RR {actual_rr:.1f}不足）", False
            reasons.append(f"⚠️ 風報比 {actual_rr:.1f} < 1.5，評分達標但暫不建議進場")
    elif (pullback_valid and entry_score >= 4) or entry_score >= 6:
        # 有效回調 + 初步轉強，或無回調但轉強訊號非常強
        signal, signal_label, can_enter = "WATCH", "觀察中", False
    else:
        signal, signal_label, can_enter = "WAIT", "等待時機", False

    total_score = trend_score + entry_score

    return {
        "regime":         regime,
        "strategy":       strategy,
        "signal":         signal,
        "signal_label":   signal_label,
        "can_enter":      can_enter,
        "score":          total_score,
        "trend_score":    trend_score,
        "entry_score":    entry_score,
        "pullback_valid": pullback_valid,
        "reasons":        reasons,
        "entry_price":    round(close, 4),
        "sl":             sl,
        "tp":             tp,
        "rr":             rr,
        "add_on_levels":  add_on_levels,
        "trigger_reasons": trigger_reasons,
        "indicators":     indicators,
    }


# ── 三重 SuperTrend 訊號 ──────────────────────────────────────────────────────

_ST_PARAMS = [(11, 2.0), (10, 1.0), (12, 3.0)]


def _wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    n = len(close)
    prev_c = np.empty(n)
    prev_c[0] = close[0]
    prev_c[1:] = close[:-1]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_c), np.abs(low - prev_c)))
    atr = np.zeros(n)
    if n < period:
        return atr
    atr[period - 1] = tr[:period].mean()
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    atr[:period - 1] = atr[period - 1]
    return atr


def _supertrend(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                period: int, mult: float):
    """回傳 (direction 陣列, dynamic_support 陣列)。direction: 1=多, -1=空。"""
    n = len(close)
    atr = _wilder_atr(high, low, close, period)
    hl2 = (high + low) / 2.0
    bu = hl2 + mult * atr
    bd = hl2 - mult * atr
    fu = bu.copy()
    fd = bd.copy()
    direction = np.ones(n, dtype=np.int8)
    for i in range(1, n):
        fu[i] = bu[i] if (bu[i] < fu[i - 1] or close[i - 1] > fu[i - 1]) else fu[i - 1]
        fd[i] = bd[i] if (bd[i] > fd[i - 1] or close[i - 1] < fd[i - 1]) else fd[i - 1]
        if   close[i] > fu[i - 1]: direction[i] =  1
        elif close[i] < fd[i - 1]: direction[i] = -1
        else:                       direction[i] = direction[i - 1]
    support = np.where(direction == 1, fd, fu)
    return direction, support


def get_signal_supertrend(df: pd.DataFrame) -> dict:
    """
    三重 SuperTrend 訊號引擎（供每日掃描器使用）。
    BUY   = 三條 ST 全為多方
    WATCH = 2/3 多方
    WAIT  = 0-1 多方
    """
    if len(df) < 60:
        return _empty_signal("資料不足")

    df = df.copy()
    high  = df["High"].values.astype(float)
    low   = df["Low"].values.astype(float)
    close = df["Close"].values.astype(float)

    # 基本指標（顯示用）
    tr_s = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"]  - df["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr_val   = float(tr_s.ewm(span=14, adjust=False).mean().iloc[-1])
    delta     = df["Close"].diff()
    gain      = delta.where(delta > 0, 0.0).ewm(span=14, adjust=False).mean()
    loss      = (-delta.where(delta < 0, 0.0)).ewm(span=14, adjust=False).mean()
    rsi_val   = float(100 - (100 / (1 + gain / (loss + 1e-10))).iloc[-1])
    # 截斷異常大量後再算均量（避免單日爆量污染，cap 在 50 日中位數 × 10）
    vol_median50 = df["Volume"].rolling(50, min_periods=10).median()
    vol_capped   = df["Volume"].clip(upper=vol_median50 * 10)
    vol_ma20  = float(vol_capped.rolling(20).mean().iloc[-1])
    vol_ratio = float(df["Volume"].iloc[-1]) / (vol_ma20 + 1e-10)
    close_last = close[-1]
    close_prev = close[-2] if len(close) >= 2 else close_last
    change_pct = round((close_last - close_prev) / (close_prev + 1e-10) * 100, 2)

    # EMA / 近期高低（供 Claude 支撐壓力分析用）
    ema20_val  = float(df["Close"].ewm(span=20,  adjust=False).mean().iloc[-1])
    ema50_val  = float(df["Close"].ewm(span=50,  adjust=False).mean().iloc[-1])
    ema200_val = float(df["Close"].ewm(span=200, adjust=False).mean().iloc[-1])
    high20_val = float(df["High"].rolling(20).max().iloc[-1])
    low20_val  = float(df["Low"].rolling(20).min().iloc[-1])
    high60_val = float(df["High"].rolling(60).max().iloc[-1]) if len(df) >= 60 else high20_val
    low60_val  = float(df["Low"].rolling(60).min().iloc[-1])  if len(df) >= 60 else low20_val

    # 三條 SuperTrend
    st_results = [_supertrend(high, low, close, p, m) for p, m in _ST_PARAMS]
    dirs  = [r[0] for r in st_results]
    lines = [r[1] for r in st_results]

    cur_dirs   = [int(d[-1]) for d in dirs]
    prev_dirs  = [int(d[-2]) for d in dirs] if len(close) >= 2 else cur_dirs
    green_count = sum(1 for d in cur_dirs if d == 1)
    prev_green  = sum(1 for d in prev_dirs if d == 1)

    reasons = []
    for i, (p, m) in enumerate(_ST_PARAMS):
        icon = "✅" if cur_dirs[i] == 1 else "❌"
        state = "多方" if cur_dirs[i] == 1 else "空方"
        reasons.append(f"{icon} SuperTrend({p},{m})：{state}")

    # 只標「由紅轉綠」：今日狀態比昨日更好才觸發訊號
    trigger_reasons = []
    if green_count == 3 and prev_green < 3:
        trigger_reasons.append("三重ST今日翻多")
    elif green_count == 2 and prev_green < 2:
        trigger_reasons.append("ST今日達2/3多方")

    # SL / TP（全多時才計算）
    sl = tp = rr = None
    add_on_levels: list[dict] = []
    if green_count == 3:
        sl_raw = max(float(lines[i][-1]) for i in range(3))
        risk   = close_last - sl_raw
        if risk > 0:
            sl = round(sl_raw, 4)
            tp = round(close_last + 2.0 * risk, 4)
            rr = 2.0
            add_on_levels = calc_add_on_levels(close_last, sl)

    # BUY：三條全綠且今日剛翻多（由紅轉綠）
    # WATCH：2/3 且今日新增一條翻多（接近觸發）
    # 其餘（持續多頭、持續整理）→ WAIT，不重複追蹤
    if green_count == 3 and prev_green < 3:
        signal, label, can_enter = "BUY",   "三重ST翻多",  True
    elif green_count == 2 and prev_green < 2:
        signal, label, can_enter = "WATCH", "ST 2/3 翻多", False
    else:
        signal, label, can_enter = "WAIT",  "趨勢未確立",  False

    regime = "TREND_UP" if green_count == 3 else ("UNCERTAIN" if green_count == 2 else "TREND_DOWN")

    return {
        "regime":         regime,
        "strategy":       "三重SuperTrend",
        "signal":         signal,
        "signal_label":   label,
        "can_enter":      can_enter,
        "score":          green_count * 2,
        "trend_score":    green_count * 2,
        "entry_score":    2 if (green_count == 3 and prev_green < 3) else 0,
        "pullback_valid": green_count == 3,
        "reasons":        reasons,
        "entry_price":    round(close_last, 4),
        "sl":             sl,
        "tp":             tp,
        "rr":             rr,
        "add_on_levels":  add_on_levels,
        "trigger_reasons": trigger_reasons,
        "indicators": {
            "rsi":            round(rsi_val, 1),
            "atr":            round(atr_val, 4),
            "volume_ratio":   round(vol_ratio, 2),
            "change_pct":     change_pct,
            "st_green_count": green_count,
            "st_directions":  cur_dirs,
            "ema20":          round(ema20_val, 4),
            "ema50":          round(ema50_val, 4),
            "ema200":         round(ema200_val, 4),
            "high20":         round(high20_val, 4),
            "low20":          round(low20_val, 4),
            "high60":         round(high60_val, 4),
            "low60":          round(low60_val, 4),
        },
    }


def _empty_signal(reason: str = "") -> dict:
    return {
        "regime":         "UNKNOWN",
        "strategy":       "",
        "signal":         "WAIT",
        "signal_label":   "資料不足",
        "can_enter":      False,
        "score":          0,
        "trend_score":    0,
        "entry_score":    0,
        "pullback_valid": False,
        "reasons":        [reason] if reason else [],
        "entry_price":    None,
        "sl":             None,
        "tp":             None,
        "rr":             None,
        "add_on_levels":  [],
        "trigger_reasons": [],
        "indicators":     {},
    }
