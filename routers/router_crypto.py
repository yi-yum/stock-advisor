from fastapi import APIRouter, HTTPException, Query
from pathlib import Path
from datetime import datetime, timezone
import sys
import math
import json
import os
import requests
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from json_utils import atomic_write_json

router = APIRouter(prefix="/api/crypto", tags=["crypto"])

# ── 支援幣種 ────────────────────────────────────────────────────────────────────
SUPPORTED_SYMBOLS = {
    "BTCUSDT", "ETHUSDT", "LINKUSDT", "SOLUSDT", "BNBUSDT",
    "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT",
}

# ── VB 策略參數 ─────────────────────────────────────────────────────────────────
SQUEEZE_BARS  = 20    # 最少連續收縮 K 棒數
ATR_THRESH    = 0.75  # ATR < ATR_MA × 0.75 才算收縮
ATR_PERIOD    = 14    # ATR EMA 週期
ATR_MA_BARS   = 50    # ATR_MA SMA 週期
TP_MULT       = 1.5   # 止盈倍數
STOP_PCT_CAP  = 0.05  # SL 最大 5%

BINANCE_BASE  = "https://fapi.binance.com/fapi/v1/klines"  # 永續合約
MIN_BARS      = 150   # 資料不足閾值

SIGNAL_HISTORY_FILE = Path(__file__).parent.parent / "scan_results" / "crypto_signal_history.json"
LINE_API_PUSH       = "https://api.line.me/v2/bot/message/push"


# ── 工具函式 ────────────────────────────────────────────────────────────────────

def _fetch_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """呼叫 Binance 公開 K 線 API，回傳 DataFrame。"""
    url = f"{BINANCE_BASE}?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=500, detail=f"Binance API 呼叫失敗：{exc}")

    raw = resp.json()
    if not raw:
        raise HTTPException(status_code=500, detail="Binance 回傳空資料")

    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["open_time"] = pd.to_numeric(df["open_time"])
    df = df.sort_values("open_time").reset_index(drop=True)
    return df


def _round_price(v):
    """將價格四捨五入至 4 位有效小數（對高價幣避免過多小數）。"""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return round(float(v), 4)


def _safe(v, digits=4):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    return round(float(v), digits)


# ── 主要端點 ────────────────────────────────────────────────────────────────────

@router.get("/ohlcv")
def get_ohlcv(
    symbol: str = Query(default="BTCUSDT", description="交易對"),
    bars: int = Query(default=80, ge=20, le=300, description="返回最近幾根 K 棒"),
):
    """返回 15m K 線資料 + EMA + ATR 收縮標記，供前端 K 線圖使用"""
    symbol = symbol.upper()
    if symbol not in SUPPORTED_SYMBOLS:
        raise HTTPException(status_code=400, detail=f"不支援的幣種：{symbol}")

    df = _fetch_klines(symbol, "15m", bars + 120)  # 多拉一些用於指標暖機

    # EMA20 / EMA50（15m）
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

    # ATR 收縮標記
    df["prev_close"] = df["close"].shift(1)
    df["tr"] = df.apply(
        lambda r: max(
            r["high"] - r["low"],
            abs(r["high"] - r["prev_close"]) if not pd.isna(r["prev_close"]) else 0,
            abs(r["low"]  - r["prev_close"]) if not pd.isna(r["prev_close"]) else 0,
        ), axis=1,
    )
    df["atr"]    = df["tr"].ewm(span=ATR_PERIOD, adjust=False).mean()
    df["atr_ma"] = df["atr"].rolling(ATR_MA_BARS).mean()
    df["squeeze"] = (df["atr"] < df["atr_ma"] * ATR_THRESH) & df["atr_ma"].notna()

    recent = df.tail(bars).reset_index(drop=True)

    candles = []
    for _, row in recent.iterrows():
        candles.append({
            "time":    int(row["open_time"]) // 1000,
            "open":    _safe(row["open"],  2),
            "high":    _safe(row["high"],  2),
            "low":     _safe(row["low"],   2),
            "close":   _safe(row["close"], 2),
            "ema20":   _safe(row["ema20"], 2),
            "ema50":   _safe(row["ema50"], 2),
            "squeeze": bool(row["squeeze"]),
        })

    return {"symbol": symbol, "interval": "15m", "candles": candles}


@router.get("/signal")
def get_vb_signal(
    symbol: str = Query(default="BTCUSDT", description="交易對，如 BTCUSDT"),
):
    symbol = symbol.upper()
    if symbol not in SUPPORTED_SYMBOLS:
        raise HTTPException(
            status_code=400,
            detail=f"不支援的幣種：{symbol}。支援：{sorted(SUPPORTED_SYMBOLS)}",
        )

    # ── 取得 K 線資料 ──────────────────────────────────────────────────────────
    df    = _fetch_klines(symbol, "15m", 300)
    df_4h = _fetch_klines(symbol, "4h",  100)

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # 資料不足檢查
    if len(df) < MIN_BARS:
        return {
            "symbol": symbol,
            "signal": "WAIT",
            "can_enter": False,
            "reason": "資料不足，暖機中",
            "updated_at": now_utc,
        }

    # ── 步驟 1：計算 ATR（14 週期 EMA）─────────────────────────────────────────
    df["prev_close"] = df["close"].shift(1)
    df["tr"] = df.apply(
        lambda r: max(
            r["high"] - r["low"],
            abs(r["high"] - r["prev_close"]) if not pd.isna(r["prev_close"]) else 0,
            abs(r["low"]  - r["prev_close"]) if not pd.isna(r["prev_close"]) else 0,
        ),
        axis=1,
    )
    df["atr"]    = df["tr"].ewm(span=ATR_PERIOD, adjust=False).mean()
    df["atr_ma"] = df["atr"].rolling(ATR_MA_BARS).mean()

    # ── 步驟 2：掃描 Squeeze（從倒數第2根往前）────────────────────────────────
    sq_count = 0
    sq_high  = float("-inf")
    sq_low   = float("+inf")

    for i in range(2, len(df)):
        row = df.iloc[-i]
        if pd.isna(row["atr_ma"]):
            break
        if row["atr"] < row["atr_ma"] * ATR_THRESH:
            sq_count += 1
            sq_high = max(sq_high, row["high"])
            sq_low  = min(sq_low,  row["low"])
        else:
            break

    # ── 步驟 3：4H 趨勢判斷 ────────────────────────────────────────────────────
    df_4h["ema20"] = df_4h["close"].ewm(span=20, adjust=False).mean()
    df_4h["ema50"] = df_4h["close"].ewm(span=50, adjust=False).mean()

    last_4h  = df_4h.iloc[-1]
    ema20_4h = float(last_4h["ema20"])
    ema50_4h = float(last_4h["ema50"])
    trend    = "up" if ema20_4h > ema50_4h else "down"

    # ── 步驟 4：突破判斷（用倒數第2根，即最新完成 K 棒）──────────────────────
    last_completed = df.iloc[-2]
    close          = float(last_completed["close"])
    atr_val        = float(last_completed["atr"])
    atr_ma_val     = float(last_completed["atr_ma"]) if not pd.isna(last_completed["atr_ma"]) else None

    signal  = "WAIT"
    sl      = None
    tp      = None
    rr      = None
    reason  = ""

    if sq_count >= SQUEEZE_BARS:
        rng = sq_high - sq_low

        if close > sq_high and trend == "up":
            sl     = max(sq_low, close * (1 - STOP_PCT_CAP))
            tp     = close + rng * TP_MULT
            signal = "LONG"
            reason = "突破收縮上軌，4H 上升趨勢"
        elif close < sq_low and trend == "down":
            sl     = min(sq_high, close * (1 + STOP_PCT_CAP))
            tp     = close - rng * TP_MULT
            signal = "SHORT"
            reason = "跌破收縮下軌，4H 下降趨勢"
        else:
            signal = "WAIT"
            reason = f"收縮充分（{sq_count} 根），等待突破方向"
    else:
        signal = "WAIT"
        reason = f"收縮根數不足（{sq_count} < {SQUEEZE_BARS}），繼續觀察"

    # ── 風報比計算 ────────────────────────────────────────────────────────────
    if sl is not None and tp is not None:
        risk   = abs(close - sl)
        reward = abs(tp - close)
        rr     = reward / risk if risk > 0 else 0

    can_enter = signal != "WAIT"

    # ── sq_high / sq_low 在收縮不足時設為 None ────────────────────────────────
    out_sq_high = sq_high  if sq_count > 0 else None
    out_sq_low  = sq_low   if sq_count > 0 else None

    # range_pct：(sq_high-sq_low)/price × 100，只在有收縮時計算
    range_pct = None
    if out_sq_high is not None and out_sq_low is not None and close > 0:
        range_pct = _safe((out_sq_high - out_sq_low) / close * 100, 2)

    # atr_ratio
    atr_ratio = None
    if atr_ma_val and atr_ma_val > 0:
        atr_ratio = _safe(atr_val / atr_ma_val, 4)

    # 未突破但收縮充分時，entry/sl/tp/rr 為 None；已突破則帶入值
    entry_out = _round_price(close) if can_enter else None
    sl_out    = _round_price(sl)    if can_enter else None
    tp_out    = _round_price(tp)    if can_enter else None
    rr_out    = _safe(rr, 2)        if can_enter else None

    return {
        "symbol":    symbol,
        "price":     _round_price(close),
        "signal":    signal,
        "can_enter": can_enter,
        "reason":    reason,
        "sq_count":  sq_count,
        "sq_high":   _round_price(out_sq_high),
        "sq_low":    _round_price(out_sq_low),
        "range_pct": range_pct,
        "trend_4h":  trend,
        "ema20_4h":  _round_price(ema20_4h),
        "ema50_4h":  _round_price(ema50_4h),
        "entry":     entry_out,
        "sl":        sl_out,
        "tp":        tp_out,
        "rr":        rr_out,
        "atr":       _safe(atr_val, 4),
        "atr_ma":    _safe(atr_ma_val, 4),
        "atr_ratio": atr_ratio,
        "updated_at": now_utc,
    }


# ── Per-coin 即時策略狀態（指標計算，非回測）─────────────────────────────────

def _strategy_status_vb(symbol: str, cfg: dict) -> dict:
    """VB：ATR 收縮 + 突破偵測（即時）"""
    atr_thresh   = cfg.get("atr_thresh", 0.75)
    squeeze_bars = cfg.get("squeeze_bars", 20)
    tp_mult      = cfg.get("tp_mult", 1.5)

    df = _fetch_klines(symbol, "1h", squeeze_bars * 4 + 200)
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)

    c = df["close"].values
    h = df["high"].values
    lo = df["low"].values

    tr = np.zeros(len(c))
    tr[0] = h[0] - lo[0]
    for i in range(1, len(c)):
        tr[i] = max(h[i]-lo[i], abs(h[i]-c[i-1]), abs(lo[i]-c[i-1]))
    atr    = pd.Series(tr).ewm(span=ATR_PERIOD, adjust=False).mean().values
    atr_ma = pd.Series(atr).rolling(ATR_MA_BARS).mean().values

    ema20 = pd.Series(c).ewm(span=20, adjust=False).mean().values
    ema50 = pd.Series(c).ewm(span=50, adjust=False).mean().values

    # 掃描最新收縮（從倒數第2根往前）
    sq_count = 0
    sq_high  = float("-inf")
    sq_low   = float("+inf")
    for i in range(2, len(c)):
        row_atr    = atr[-i]
        row_atr_ma = atr_ma[-i]
        if np.isnan(row_atr_ma):
            break
        if row_atr < row_atr_ma * atr_thresh:
            sq_count += 1
            sq_high = max(sq_high, h[-i])
            sq_low  = min(sq_low,  lo[-i])
        else:
            break

    cur_close      = float(c[-2])
    atr_ratio      = float(atr[-2] / atr_ma[-2]) if not np.isnan(atr_ma[-2]) else None
    trend_up       = bool(ema20[-2] > ema50[-2])
    prev_trend_up  = bool(ema20[-3] > ema50[-3])
    just_trend_down = prev_trend_up and not trend_up
    price          = float(c[-1])

    signal = "WAIT"
    reason = ""
    sl = tp = rr = None

    if just_trend_down:
        signal = "EXIT"
        reason = "EMA 趨勢由多轉空（本根剛翻），出場訊號"
    elif sq_count >= squeeze_bars:
        rng = sq_high - sq_low
        if cur_close > sq_high and trend_up:
            signal = "LONG"
            reason = f"突破收縮上軌，趨勢向上（{sq_count} 根收縮）"
            sl = max(sq_low, cur_close * (1 - STOP_PCT_CAP))
            tp = cur_close + rng * tp_mult
        elif cur_close < sq_low and not trend_up:
            signal = "SHORT"
            reason = f"跌破收縮下軌，趨勢向下（{sq_count} 根收縮）"
            sl = min(sq_high, cur_close * (1 + STOP_PCT_CAP))
            tp = cur_close - rng * tp_mult
        else:
            reason = f"收縮充分（{sq_count} 根），等待突破方向"
    else:
        reason = f"收縮 {sq_count}/{squeeze_bars} 根，繼續觀察"

    if sl and tp:
        risk = abs(cur_close - sl)
        rr = round(abs(tp - cur_close) / risk, 2) if risk > 0 else None

    return {
        "signal": signal, "can_enter": signal != "WAIT", "reason": reason,
        "price": _round_price(price),
        "sq_count": sq_count, "sq_bars_needed": squeeze_bars,
        "sq_high": _round_price(sq_high) if sq_count > 0 else None,
        "sq_low":  _round_price(sq_low)  if sq_count > 0 else None,
        "atr_ratio": _safe(atr_ratio, 4),
        "trend_up": trend_up,
        "sl": _round_price(sl), "tp": _round_price(tp), "rr": rr,
    }


def _strategy_status_3st(symbol: str, cfg: dict, bull_mask_val: bool | None) -> dict:
    """3ST：三重 SuperTrend 當前方向"""
    from signal_engine import _supertrend
    ST_PARAMS = [(11, 2.0), (10, 1.0), (12, 3.0)]

    df = _fetch_klines(symbol, "4h", 300)
    for col in ("high", "low", "close", "open"):
        df[col] = df[col].astype(float)
    h = df["high"].values
    lo = df["low"].values
    c  = df["close"].values
    o  = df["open"].values

    dirs = [int(_supertrend(h, lo, c, p, m)[0][-1]) for p, m in ST_PARAMS]
    prev_dirs = [int(_supertrend(h, lo, c, p, m)[0][-2]) for p, m in ST_PARAMS]

    green      = sum(1 for d in dirs if d == 1)
    prev_green = sum(1 for d in prev_dirs if d == 1)
    price      = float(c[-1])

    just_entry = (green == 3 and prev_green < 3)
    just_exit  = (prev_green == 3 and green < 3)
    signal = "WAIT"
    reason = ""

    if just_exit:
        signal = "EXIT"
        reason = f"ST 多頭由 3 降至 {green}，出場訊號"
    elif just_entry and (bull_mask_val is None or bull_mask_val):
        signal = "LONG"
        reason = "三條 ST 全翻多（本根剛觸發）"
    elif green == 3:
        signal = "HOLD"
        reason = f"三條 ST 全多頭（持續 {green}/3）"
    elif green == 2:
        missing = [f"ST({p},{m})" for (p, m), d in zip(ST_PARAMS, dirs) if d != 1]
        reason = f"2/3 多頭，待翻：{missing[0] if missing else '?'}"
    elif green == 0:
        reason = "三條 ST 全空頭"
    else:
        reason = f"{green}/3 條多頭"

    if bull_mask_val is False and signal == "LONG":
        signal = "WAIT"
        reason += "（EMA200 過濾：空頭區間，暫不進場）"

    return {
        "signal": signal, "can_enter": signal == "LONG", "reason": reason,
        "price": _round_price(price),
        "green_count": green, "prev_green": prev_green,
        "st_dirs": dirs,
        "st_labels": [f"ST({p},{m})" for p, m in ST_PARAMS],
    }


def _strategy_status_ema(symbol: str, cfg: dict, bull_mask_val: bool | None) -> dict:
    """EMA Cross + Trend Filter 當前狀態"""
    fast  = cfg.get("fast", 9)
    slow  = cfg.get("slow", 21)
    trend = cfg.get("trend", 200)

    warmup = trend + 10
    df = _fetch_klines(symbol, "4h", warmup + 50)
    c  = df["close"].astype(float).values
    o  = df["open"].astype(float).values
    price = float(c[-1])

    ema_f = pd.Series(c).ewm(span=fast,  adjust=False).mean().values
    ema_s = pd.Series(c).ewm(span=slow,  adjust=False).mean().values
    ema_t = pd.Series(c).ewm(span=trend, adjust=False).mean().values

    fast_above    = bool(ema_f[-1] > ema_s[-1])
    above_trend   = bool(c[-1] > ema_t[-1])
    prev_f_above  = bool(ema_f[-2] > ema_s[-2])
    golden_cross  = fast_above and not prev_f_above   # 剛發生
    death_cross   = not fast_above and prev_f_above

    # 計算上次交叉距今幾根
    bars_since_cross = 0
    for i in range(1, min(50, len(c))):
        if (ema_f[-i] > ema_s[-i]) != fast_above:
            bars_since_cross = i - 1
            break

    signal = "WAIT"
    reason = ""

    if death_cross:
        signal = "EXIT"
        reason = "死亡交叉（本根觸發），出場訊號"
    elif golden_cross and above_trend and (bull_mask_val is None or bull_mask_val):
        signal = "LONG"
        reason = f"黃金交叉（本根觸發），收盤 > EMA{trend}"
    elif fast_above and above_trend and (bull_mask_val is None or bull_mask_val):
        signal = "HOLD"
        reason = f"快線在慢線上方（{bars_since_cross} 根），收盤 > EMA{trend}"
    elif fast_above and not above_trend:
        reason = f"快線在慢線上方，但收盤 < EMA{trend}（趨勢過濾未通過）"
    else:
        cross_dir = "空" if not fast_above else "多"
        reason = f"快線在慢線{'上' if fast_above else '下'}方（{bars_since_cross} 根），{cross_dir}頭"

    if bull_mask_val is False and signal in ("LONG", "HOLD"):
        signal = "WAIT"
        reason += "（EMA200 日線過濾：空頭區間）"

    pct_gap = (ema_f[-1] - ema_s[-1]) / ema_s[-1] * 100

    return {
        "signal": signal, "can_enter": signal == "LONG", "reason": reason,
        "price": _round_price(price),
        "ema_fast": _round_price(ema_f[-1]),
        "ema_slow": _round_price(ema_s[-1]),
        "ema_trend": _round_price(ema_t[-1]),
        "fast_above_slow": fast_above,
        "above_trend": above_trend,
        "golden_cross": golden_cross,
        "bars_since_cross": bars_since_cross,
        "pct_gap": _safe(pct_gap, 3),
        "fast_period": fast, "slow_period": slow, "trend_period": trend,
    }


def _strategy_status_dc(symbol: str, cfg: dict) -> dict:
    """Donchian Channel 突破偵測"""
    entry_period = cfg.get("entry_period", 20)
    exit_period  = cfg.get("exit_period", 10)

    df = _fetch_klines(symbol, "4h", entry_period + 50)
    c  = df["close"].astype(float).values
    h  = df["high"].astype(float).values
    lo = df["low"].astype(float).values
    price = float(c[-1])

    # 用倒數第2根判斷（最新已收盤K棒）
    dc_high      = float(np.max(h[-entry_period-2:-2]))
    dc_low       = float(np.min(lo[-entry_period-2:-2]))
    dc_low_exit  = float(np.min(lo[-exit_period-2:-2]))   # 出場低點通道
    cur          = float(c[-2])
    prev         = float(c[-3])

    signal = "WAIT"
    reason = ""
    pct_to_high = (dc_high - cur) / cur * 100
    pct_to_low  = (cur - dc_low)  / cur * 100

    # 出場優先：跌破 exit_period 根低點
    if prev >= dc_low_exit and cur < dc_low_exit:
        signal = "EXIT"
        reason = f"跌破 {exit_period} 根低點（{dc_low_exit:.2f}），出場訊號"
    elif cur > dc_high:
        signal = "LONG"
        reason = f"突破 {entry_period} 根高點（{dc_high:.2f}）"
    else:
        reason = f"距 {entry_period} 根高點 {pct_to_high:.2f}%（高點 {dc_high:.2f}）"

    return {
        "signal": signal, "can_enter": signal == "LONG", "reason": reason,
        "price": _round_price(price),
        "dc_high":     _round_price(dc_high),
        "dc_low":      _round_price(dc_low),
        "dc_low_exit": _round_price(dc_low_exit),
        "pct_to_high": _safe(pct_to_high, 2),
        "entry_period": entry_period, "exit_period": exit_period,
    }


@router.get("/strategy-status")
def get_strategy_status():
    """各幣種套用最佳策略的即時進場訊號狀態。"""
    from backtest_crypto import COIN_BEST
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    # 預先載入有 trend_filter 幣種的日線 EMA200
    def _macro_bull(symbol: str) -> bool | None:
        try:
            df_d = _fetch_klines(symbol.replace("USDT", "USDT"), "1d", 250)
            c = df_d["close"].astype(float)
            ema200 = c.ewm(span=200, adjust=False).mean()
            return bool(float(c.iloc[-1]) > float(ema200.iloc[-1]))
        except Exception:
            return None

    results = []
    for symbol, cfg in COIN_BEST.items():
        strat = cfg["strategy"]
        bull  = _macro_bull(symbol) if cfg.get("trend_filter") else None
        try:
            if strat == "VB":
                s = _strategy_status_vb(symbol, cfg)
            elif strat == "3ST":
                s = _strategy_status_3st(symbol, cfg, bull)
            elif strat == "EMA":
                s = _strategy_status_ema(symbol, cfg, bull)
            elif strat == "DC":
                s = _strategy_status_dc(symbol, cfg)
            else:
                s = {"signal": "WAIT", "can_enter": False, "reason": f"未知策略 {strat}", "price": None}
        except Exception as e:
            s = {"signal": "WAIT", "can_enter": False, "reason": f"錯誤：{e}", "price": None}

        results.append({
            "symbol":     symbol,
            "strategy":   strat,
            "macro_bull": bull,
            **s,
        })

    return {
        "signals":    results,
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# ── 策略 K 線圖資料（含進場標記）────────────────────────────────────────────

@router.get("/strategy-chart")
def get_strategy_chart(
    symbol: str,
    bars: int = Query(default=120, ge=50, le=300),
):
    """返回指定幣種的策略 K 線圖資料，包含指標線與進場標記時間點。"""
    from backtest_crypto import COIN_BEST
    from signal_engine import _supertrend

    symbol = symbol.upper()
    cfg = COIN_BEST.get(symbol)
    if cfg is None:
        raise HTTPException(status_code=404, detail=f"無此幣種配置：{symbol}")

    strat    = cfg["strategy"]
    interval = "1h" if strat == "VB" else "4h"
    warmup   = 260  # 讓 EMA200 等長週期指標充分暖機

    df_raw = _fetch_klines(symbol, interval, bars + warmup)
    for col in ("open", "high", "low", "close", "volume"):
        df_raw[col] = df_raw[col].astype(float)

    vol_all   = df_raw["volume"].values
    high_all  = df_raw["high"].values
    low_all   = df_raw["low"].values
    close_all = df_raw["close"].values
    open_all  = df_raw["open"].values
    times_all = (df_raw["open_time"].values // 1000).astype(int)  # ms → s

    overlays: dict    = {}
    entry_bars: list  = []
    exit_bars: list   = []
    signal = "WAIT"

    # ── EMA Cross ─────────────────────────────────────────────────────────
    if strat == "EMA":
        fast  = cfg.get("fast",  9)
        slow  = cfg.get("slow",  21)
        trend = cfg.get("trend", 200)

        ema_f = pd.Series(close_all).ewm(span=fast,  adjust=False).mean().values
        ema_s = pd.Series(close_all).ewm(span=slow,  adjust=False).mean().values
        ema_t = pd.Series(close_all).ewm(span=trend, adjust=False).mean().values

        # 取最後 bars 根
        n = len(close_all)
        s = n - bars
        ts     = times_all[s:]
        c      = close_all[s:]
        h      = high_all[s:]
        lo     = low_all[s:]
        op     = open_all[s:]
        vol    = vol_all[s:]
        ema_f  = ema_f[s:]
        ema_s  = ema_s[s:]
        ema_t  = ema_t[s:]

        # 掃描視窗內所有黃金交叉（進場）與死亡交叉（出場）
        for i in range(1, len(ts)):
            if ema_f[i] > ema_s[i] and ema_f[i - 1] <= ema_s[i - 1]:
                entry_bars.append(int(ts[i]))
            elif ema_f[i] < ema_s[i] and ema_f[i - 1] >= ema_s[i - 1]:
                exit_bars.append(int(ts[i]))

        if ema_f[-1] > ema_s[-1]:
            signal = "LONG" if (entry_bars and entry_bars[-1] == int(ts[-1])) else "HOLD"
        elif exit_bars and exit_bars[-1] == int(ts[-1]):
            signal = "EXIT"

        overlays = {
            f"EMA{fast}":  [{"time": int(ts[i]), "value": _safe(ema_f[i])} for i in range(len(ts))],
            f"EMA{slow}":  [{"time": int(ts[i]), "value": _safe(ema_s[i])} for i in range(len(ts))],
            f"EMA{trend}": [{"time": int(ts[i]), "value": _safe(ema_t[i])} for i in range(len(ts))],
        }
        candles = [
            {"time": int(ts[i]), "open": _safe(op[i]), "high": _safe(h[i]),
             "low": _safe(lo[i]), "close": _safe(c[i]), "volume": _safe(vol[i], 2)}
            for i in range(len(ts))
        ]

    # ── 三重 SuperTrend ────────────────────────────────────────────────────
    elif strat == "3ST":
        ST_PARAMS = [(11, 2.0), (10, 1.0), (12, 3.0)]

        dirs_all_list = []
        for p, m in ST_PARAMS:
            d, _ = _supertrend(high_all, low_all, close_all, p, m)
            dirs_all_list.append(d)

        n = len(close_all)
        s = n - bars
        ts  = times_all[s:]
        c   = close_all[s:]
        h   = high_all[s:]
        lo  = low_all[s:]
        op  = open_all[s:]
        vol = vol_all[s:]
        dirs_trim = [d[s:] for d in dirs_all_list]

        gc = np.array([sum(1 for d in dirs_trim if d[i] == 1) for i in range(bars)])

        # 掃描視窗內所有進場（3條全翻多）與出場（從3條掉下）
        for i in range(1, len(gc)):
            if gc[i] == 3 and gc[i - 1] < 3:
                entry_bars.append(int(ts[i]))
            elif gc[i - 1] == 3 and gc[i] < 3:
                exit_bars.append(int(ts[i]))

        if gc[-1] == 3:
            signal = "LONG" if (entry_bars and entry_bars[-1] == int(ts[-1])) else "HOLD"
        elif exit_bars and exit_bars[-1] == int(ts[-1]):
            signal = "EXIT"

        overlays = {
            "green_count": [{"time": int(ts[i]), "value": int(gc[i])} for i in range(len(ts))]
        }
        candles = [
            {"time": int(ts[i]), "open": _safe(op[i]), "high": _safe(h[i]),
             "low": _safe(lo[i]), "close": _safe(c[i]), "gc": int(gc[i]),
             "volume": _safe(vol[i], 2)}
            for i in range(len(ts))
        ]

    # ── VB 波動率收縮突破 ──────────────────────────────────────────────────
    elif strat == "VB":
        atr_thresh          = cfg.get("atr_thresh",   0.75)
        squeeze_bars_needed = cfg.get("squeeze_bars", 20)

        tr_arr = np.concatenate([[0.0], np.maximum(
            np.maximum(high_all[1:] - low_all[1:],
                       np.abs(high_all[1:] - close_all[:-1])),
            np.abs(low_all[1:] - close_all[:-1])
        )])
        atr_arr    = pd.Series(tr_arr).ewm(span=14, adjust=False).mean().values
        atr_ma_arr = pd.Series(atr_arr).rolling(50, min_periods=1).mean().values
        ratio_arr  = np.where(atr_ma_arr > 0, atr_arr / atr_ma_arr, 1.0)
        squeeze_arr = ratio_arr < atr_thresh

        # EMA20/50 用於出場偵測（趨勢由多轉空）
        ema20_all = pd.Series(close_all).ewm(span=20, adjust=False).mean().values
        ema50_all = pd.Series(close_all).ewm(span=50, adjust=False).mean().values

        n = len(close_all)
        s = n - bars
        ts       = times_all[s:]
        c        = close_all[s:]
        h        = high_all[s:]
        lo       = low_all[s:]
        op       = open_all[s:]
        vol      = vol_all[s:]
        sq_t     = squeeze_arr[s:]
        ema20_t  = ema20_all[s:]
        ema50_t  = ema50_all[s:]

        # 進場：收縮結束後的突破根（全部）
        for i in range(squeeze_bars_needed, len(sq_t)):
            if not sq_t[i] and sq_t[i - 1]:
                sq_len = 0
                j = i - 1
                while j >= 0 and sq_t[j]:
                    sq_len += 1
                    j -= 1
                if sq_len >= squeeze_bars_needed:
                    entry_bars.append(int(ts[i]))

        # 出場：EMA20 跌破 EMA50
        for i in range(1, len(ts)):
            if ema20_t[i - 1] > ema50_t[i - 1] and ema20_t[i] <= ema50_t[i]:
                exit_bars.append(int(ts[i]))

        if entry_bars:
            signal = "HOLD"
        if exit_bars and exit_bars[-1] == int(ts[-1]):
            signal = "EXIT"

        overlays = {}  # 收縮資訊直接標在 candles 上
        candles = [
            {
                "time":    int(ts[i]),
                "open":    _safe(op[i]),
                "high":    _safe(h[i]),
                "low":     _safe(lo[i]),
                "close":   _safe(c[i]),
                "squeeze": bool(sq_t[i]),
                "volume":  _safe(vol[i], 2),
            }
            for i in range(len(ts))
        ]

    # ── Donchian Channel ───────────────────────────────────────────────────
    elif strat == "DC":
        entry_period = cfg.get("entry_period", 20)
        exit_period  = cfg.get("exit_period",  10)

        dc_high_arr      = pd.Series(high_all).rolling(entry_period).max().shift(1).values
        dc_low_arr       = pd.Series(low_all).rolling(entry_period).min().shift(1).values
        dc_low_exit_arr  = pd.Series(low_all).rolling(exit_period).min().shift(1).values

        n = len(close_all)
        s = n - bars
        ts        = times_all[s:]
        c         = close_all[s:]
        h         = high_all[s:]
        lo        = low_all[s:]
        op        = open_all[s:]
        vol       = vol_all[s:]
        dc_h      = dc_high_arr[s:]
        dc_l      = dc_low_arr[s:]
        dc_le     = dc_low_exit_arr[s:]

        # 掃描視窗內所有進場（突破高點）與出場（跌破出場低點）
        for i in range(1, len(ts)):
            if not np.isnan(dc_h[i]) and c[i] > dc_h[i] and c[i - 1] <= dc_h[i - 1]:
                entry_bars.append(int(ts[i]))
            if not np.isnan(dc_le[i]) and c[i] < dc_le[i] and c[i - 1] >= dc_le[i - 1]:
                exit_bars.append(int(ts[i]))

        if entry_bars:
            signal = "HOLD"
        if exit_bars and exit_bars[-1] == int(ts[-1]):
            signal = "EXIT"

        overlays = {
            "dc_high":     [{"time": int(ts[i]), "value": _safe(dc_h[i])}
                            for i in range(len(ts)) if not np.isnan(dc_h[i])],
            "dc_low":      [{"time": int(ts[i]), "value": _safe(dc_l[i])}
                            for i in range(len(ts)) if not np.isnan(dc_l[i])],
            "dc_low_exit": [{"time": int(ts[i]), "value": _safe(dc_le[i])}
                            for i in range(len(ts)) if not np.isnan(dc_le[i])],
        }
        candles = [
            {"time": int(ts[i]), "open": _safe(op[i]), "high": _safe(h[i]),
             "low": _safe(lo[i]), "close": _safe(c[i]), "volume": _safe(vol[i], 2)}
            for i in range(len(ts))
        ]

    else:
        raise HTTPException(status_code=400, detail=f"未知策略：{strat}")

    return {
        "symbol":     symbol,
        "strategy":   strat,
        "interval":   interval,
        "signal":     signal,
        "entry_bars": entry_bars,
        "exit_bars":  exit_bars,
        "candles":    candles,
        "overlays":   overlays,
    }


# ── 訊號歷史 & LINE 推播 ─────────────────────────────────────────────────────

@router.get("/signal-history")
def get_signal_history():
    """回傳加密貨幣訊號變化歷史記錄"""
    if not SIGNAL_HISTORY_FILE.exists():
        return {"history": [], "last_signals": {}, "last_checked": None}
    try:
        data = json.loads(SIGNAL_HISTORY_FILE.read_text(encoding="utf-8"))
        last_signals = data.get("last_signals", {})
        timestamps = [v.get("checked_at", "") for v in last_signals.values() if v.get("checked_at")]
        last_checked = max(timestamps) if timestamps else None
        return {"history": data.get("history", []), "last_signals": last_signals, "last_checked": last_checked}
    except Exception:
        return {"history": [], "last_signals": {}, "last_checked": None}


@router.get("/notify")
def check_and_notify():
    """
    檢查各幣種訊號是否有新進場/出場，有則寫入歷史並發送 LINE 推播。
    由 APScheduler 每 4 小時自動呼叫，也可手動觸發。

    LINE Messaging API 設定（.env）：
      LINE_CHANNEL_TOKEN=<長效 Channel Access Token>
      LINE_USER_ID=<你的 LINE User ID，Uxxxxxxxxx 格式>
    """
    channel_token = os.environ.get("LINE_CHANNEL_TOKEN", "")
    user_id       = os.environ.get("LINE_USER_ID", "")

    # ── 取得目前策略狀態 ──────────────────────────────────────────────────────
    current_data = get_strategy_status()
    current_map  = {r["symbol"]: r for r in current_data["signals"]}

    # ── 讀取前次狀態 ──────────────────────────────────────────────────────────
    prev_map = {}
    history  = []
    if SIGNAL_HISTORY_FILE.exists():
        try:
            saved    = json.loads(SIGNAL_HISTORY_FILE.read_text(encoding="utf-8"))
            prev_map = saved.get("last_signals", {})
            history  = saved.get("history", [])
        except Exception:
            pass

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    alerts  = []

    for sym, curr in current_map.items():
        prev_sig = prev_map.get(sym, {}).get("signal", "WAIT")
        curr_sig = curr["signal"]

        # 新進場：非 LONG/SHORT/HOLD → LONG 或 SHORT
        # 新出場：非 EXIT → EXIT
        if (curr_sig in ("LONG", "SHORT") and prev_sig not in ("LONG", "SHORT", "HOLD")) or \
           (curr_sig == "EXIT" and prev_sig != "EXIT"):
            alerts.append(curr)
            history.append({
                "symbol":    sym,
                "signal":    curr_sig,
                "strategy":  curr["strategy"],
                "price":     curr.get("price"),
                "reason":    curr.get("reason", ""),
                "timestamp": now_str,
            })

    # ── 儲存最新狀態 ──────────────────────────────────────────────────────────
    atomic_write_json(SIGNAL_HISTORY_FILE, {
        "last_signals": {
            sym: {"signal": d["signal"], "checked_at": now_str}
            for sym, d in current_map.items()
        },
        "history": history[-500:],
    }, ensure_ascii=False, indent=2)

    # ── LINE 推播 ─────────────────────────────────────────────────────────────
    line_results = []
    if channel_token and user_id and alerts:
        for a in alerts:
            emoji = "🟢" if a["signal"] == "LONG" else "🔵" if a["signal"] == "SHORT" else "🔴"
            label = "進場訊號" if a["signal"] == "LONG" else "做空訊號" if a["signal"] == "SHORT" else "出場訊號"
            p     = a.get("price")
            price_str = f"${p:,.2f}" if p else "—"
            text  = (
                f"{emoji} {a['symbol']} {label}\n"
                f"策略：{a['strategy']}\n"
                f"價格：{price_str}\n"
                f"原因：{a.get('reason', '')}"
            )
            try:
                r = requests.post(
                    LINE_API_PUSH,
                    headers={
                        "Authorization": f"Bearer {channel_token}",
                        "Content-Type": "application/json",
                    },
                    json={"to": user_id, "messages": [{"type": "text", "text": text}]},
                    timeout=10,
                )
                line_results.append({"symbol": a["symbol"], "status": r.status_code})
            except Exception as e:
                line_results.append({"symbol": a["symbol"], "error": str(e)})

    return {
        "checked_at":       now_str,
        "alerts":           len(alerts),
        "symbols_alerted":  [a["symbol"] for a in alerts],
        "line_configured":  bool(channel_token and user_id),
        "line_results":     line_results,
    }


# ── Per-coin 持倉狀態（回測式，已停用於 dashboard）──────────────────────────

@router.get("/signals")
def get_per_coin_signals():
    """回傳 COIN_BEST 各幣種當前信號狀態（持倉中 / 空倉）。"""
    from backtest_crypto import detect_signal, COIN_BEST

    results = []
    for symbol in COIN_BEST:
        try:
            r = detect_signal(symbol)
        except Exception as e:
            r = {"symbol": symbol, "error": str(e), "strategy": "—"}
        results.append(r)

    return {
        "signals":    results,
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
