"""
backtest_crypto.py — 加密貨幣策略比較回測
==========================================
策略 A：VB 波動率收縮突破（4H K線）
  - ATR(14) < ATR_MA(50) × 0.75，連續 ≥ 20 根 → 蓄積能量
  - 4H 收盤突破收縮上軌 → LONG；跌破下軌 → SHORT
  - SL：收縮下/上軌（最大 ±5%）；TP：進場 + 區間 × 1.5

策略 B：三重 SuperTrend（4H K線，做多 + 做空）
  - 三條 ST(11,2)/(10,1)/(12,3) 全翻多 → LONG
  - 三條 ST 全翻空 → SHORT
  - 出場：全翻至反向（flat 期間 = 1~2 條翻）

執行：
  python backtest_crypto.py
  python backtest_crypto.py --symbols BTCUSDT ETHUSDT --start 2022-01-01
  python backtest_crypto.py --interval 1h
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from signal_engine import _supertrend

# ── 常數 ──────────────────────────────────────────────────────────────────────

COINS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
         "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT"]

# ── Per-coin 最佳策略配置（來自優化結果 2023-2026）─────────────────────────────
# strategy: "VB" / "3ST" / "DC" / "EMA"
COIN_BEST = {
    "BTCUSDT":  {"strategy": "VB",  "atr_thresh": 0.9,  "squeeze_bars": 20, "tp_mult": 1.0, "trend_filter": True},
    "ETHUSDT":  {"strategy": "3ST", "mode": "long_only", "stop_loss": 0.08, "trend_filter": True},
    "SOLUSDT":  {"strategy": "EMA", "fast": 12, "slow": 26, "trend": 200, "stop_pct": 0.07},
    "BNBUSDT":  {"strategy": "EMA", "fast": 9,  "slow": 21, "trend": 200, "stop_pct": 0.05},
    "XRPUSDT":  {"strategy": "VB",  "atr_thresh": 0.8,  "squeeze_bars": 10, "tp_mult": 1.0},
    "AVAXUSDT": {"strategy": "DC",  "entry_period": 10, "exit_period": 7,   "stop_pct": 0.06},
    "DOGEUSDT": {"strategy": "EMA", "fast": 7,  "slow": 21, "trend": 100,  "stop_pct": 0.07},
    "LINKUSDT": {"strategy": "VB",  "atr_thresh": 0.75, "squeeze_bars": 15, "tp_mult": 1.0},
    "ADAUSDT":  {"strategy": "EMA", "fast": 9,  "slow": 26, "trend": 200,  "stop_pct": 0.07},
}

BINANCE_BASE = "https://fapi.binance.com/fapi/v1/klines"
CACHE_DIR    = Path("backtest_cache/crypto")
RESULT_DIR   = Path("backtest_results")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

FEE = 0.0005          # 單邊手續費 0.05%（Binance 永續合約 taker）

# VB 策略參數
ATR_PERIOD   = 14
ATR_MA_BARS  = 50
ATR_THRESH   = 0.75
SQUEEZE_BARS = 20
TP_MULT      = 1.5
STOP_PCT_CAP = 0.05

# 三重 SuperTrend 參數
ST_PARAMS = [(11, 2.0), (10, 1.0), (12, 3.0)]


# ── Binance 資料 ───────────────────────────────────────────────────────────────

def _fetch_page(symbol: str, interval: str, start_ms: int, end_ms: int) -> list:
    url = (f"{BINANCE_BASE}?symbol={symbol}&interval={interval}"
           f"&startTime={start_ms}&endTime={end_ms}&limit=1000")
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  Binance 請求失敗：{e}")
        return []


def fetch_klines(symbol: str, interval: str, start: str, end: str) -> pd.DataFrame:
    """分頁抓取 Binance K 線，回傳 DataFrame（open/high/low/close/volume）。"""
    start_ms = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms   = int(pd.Timestamp(end).timestamp() * 1000)

    all_rows = []
    cur = start_ms
    while cur < end_ms:
        rows = _fetch_page(symbol, interval, cur, end_ms)
        if not rows:
            break
        all_rows.extend(rows)
        cur = int(rows[-1][0]) + 1
        if len(rows) < 1000:
            break
        time.sleep(0.08)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qv","trades","tbb","tbq","ignore",
    ])
    for col in ("open","high","low","close","volume"):
        df[col] = pd.to_numeric(df[col])
    df.index = pd.to_datetime(pd.to_numeric(df["open_time"]), unit="ms", utc=True)
    df.index = df.index.tz_localize(None)
    return df[["open","high","low","close","volume"]]


def load_ohlcv(symbol: str, interval: str, start: str, end: str) -> pd.DataFrame | None:
    """讀快取，沒有則下載並快取。"""
    cache = CACHE_DIR / f"{symbol}_{interval}.parquet"

    if cache.exists():
        try:
            df = pd.read_parquet(cache)
            c_start = int(pd.Timestamp(start).timestamp() * 1000)
            c_end   = int(pd.Timestamp(end).timestamp() * 1000)
            d_start = int(df.index[0].timestamp() * 1000)
            d_end   = int(df.index[-1].timestamp() * 1000)
            if d_start <= c_start and d_end >= c_end - 86_400_000 * 2:
                return df[(df.index >= pd.Timestamp(start)) &
                          (df.index <= pd.Timestamp(end))]
        except Exception:
            pass

    print(f"  下載 {symbol} {interval} 資料中...")
    df = fetch_klines(symbol, interval, start, end)
    if df.empty:
        return None
    df.to_parquet(cache)
    return df


# ── 統計計算 ───────────────────────────────────────────────────────────────────

def _calc_stats(trades: list, equity_curve: list, initial: float) -> dict:
    if not trades:
        return {
            "trades": 0, "long_trades": 0, "short_trades": 0,
            "total_return_pct": 0.0, "win_rate": 0.0,
            "avg_ret_pct": 0.0, "avg_win_pct": 0.0, "avg_loss_pct": 0.0,
            "max_dd_pct": 0.0, "sharpe": 0.0,
            "long_win_rate": 0.0, "short_win_rate": 0.0,
        }

    rets = np.array([t["ret_pct"] for t in trades])
    longs  = [t for t in trades if t["dir"] == "long"]
    shorts = [t for t in trades if t["dir"] == "short"]

    wins  = rets[rets > 0]
    loses = rets[rets <= 0]

    # 最大回撤（從 equity curve）
    eq_vals = [v for _, v in equity_curve]
    peak, max_dd = eq_vals[0], 0.0
    for v in eq_vals:
        peak = max(peak, v)
        dd = (peak - v) / peak * 100
        max_dd = max(max_dd, dd)

    final_equity = equity_curve[-1][1] if equity_curve else initial
    total_ret = (final_equity / initial - 1) * 100

    sharpe = 0.0
    if len(rets) > 1 and rets.std() > 0:
        sharpe = float(rets.mean() / rets.std() * np.sqrt(len(rets)))

    lw = len([t for t in longs  if t["ret_pct"] > 0])
    sw = len([t for t in shorts if t["ret_pct"] > 0])

    return {
        "trades":           len(trades),
        "long_trades":      len(longs),
        "short_trades":     len(shorts),
        "total_return_pct": round(total_ret, 2),
        "win_rate":         round(len(wins) / len(rets) * 100, 1),
        "avg_ret_pct":      round(float(rets.mean()), 3),
        "avg_win_pct":      round(float(wins.mean()),  3) if len(wins)  > 0 else 0.0,
        "avg_loss_pct":     round(float(loses.mean()), 3) if len(loses) > 0 else 0.0,
        "max_dd_pct":       round(max_dd, 2),
        "sharpe":           round(sharpe, 3),
        "long_win_rate":    round(lw / len(longs)  * 100, 1) if longs  else 0.0,
        "short_win_rate":   round(sw / len(shorts) * 100, 1) if shorts else 0.0,
    }


# ── VB 策略回測 ────────────────────────────────────────────────────────────────

def backtest_vb(df_raw: pd.DataFrame, initial: float = 10_000.0,
                atr_thresh: float = ATR_THRESH,
                squeeze_bars: int = SQUEEZE_BARS,
                tp_mult: float = TP_MULT,
                stop_pct: float = STOP_PCT_CAP,
                bull_mask: pd.Series | None = None,
                _return_position: bool = False) -> dict:
    """
    VB 波動率收縮突破策略（1H K線）。
    訊號確認：bar i 收盤突破 → 下一根 bar i+1 開盤進場。
    出場：在 bar 內檢查 SL/TP（兩者同時命中 → 取 SL，保守估計）。
    """
    close  = df_raw["close"].values.astype(float)
    high   = df_raw["high"].values.astype(float)
    low    = df_raw["low"].values.astype(float)
    open_  = df_raw["open"].values.astype(float)
    n      = len(close)

    # ATR(14 EMA) 與 ATR_MA(50 SMA)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    atr    = pd.Series(tr).ewm(span=ATR_PERIOD, adjust=False).mean().values
    atr_ma = pd.Series(atr).rolling(ATR_MA_BARS).mean().values
    squeeze = np.where(~np.isnan(atr_ma), atr < atr_ma * atr_thresh, False)

    # 趨勢過濾（EMA20 vs EMA50）
    ema20 = pd.Series(close).ewm(span=20, adjust=False).mean().values
    ema50 = pd.Series(close).ewm(span=50, adjust=False).mean().values

    trades: list[dict] = []
    equity = initial
    position = None
    equity_curve = [(df_raw.index[0], initial)]
    WARMUP = ATR_MA_BARS + squeeze_bars + 5

    for i in range(WARMUP, n - 1):
        # ── 檢查持倉出場 ──────────────────────────────────────────────────────
        if position is not None:
            hit_sl = hit_tp = False

            if position["dir"] == "long":
                hit_sl = low[i]  <= position["sl"]
                hit_tp = high[i] >= position["tp"]
                if hit_sl and hit_tp:   # 同根命中 → 保守取 SL
                    hit_tp = False
                exit_price = position["sl"] if hit_sl else (position["tp"] if hit_tp else None)
            else:  # short
                hit_sl = high[i] >= position["sl"]
                hit_tp = low[i]  <= position["tp"]
                if hit_sl and hit_tp:
                    hit_tp = False
                exit_price = position["sl"] if hit_sl else (position["tp"] if hit_tp else None)

            if exit_price is not None:
                ret = (exit_price - position["entry"]) / position["entry"]
                if position["dir"] == "short":
                    ret = -ret
                ret -= FEE * 2
                equity *= (1 + ret)
                trades.append({
                    "entry_date": position["date"],
                    "exit_date":  str(df_raw.index[i].date()),
                    "dir":        position["dir"],
                    "entry":      position["entry"],
                    "exit":       exit_price,
                    "ret_pct":    round(ret * 100, 3),
                    "result":     "tp" if hit_tp else "sl",
                })
                equity_curve.append((df_raw.index[i], equity))
                position = None
            continue  # 持倉中不找新訊號

        # ── 掃描收縮區間（從 bar i-1 往前）─────────────────────────────────
        # 注意：突破 bar i 的 ATR 已擴張，squeeze[i] 通常為 False；
        # 必須從前一根 i-1 往前掃，確認 squeeze 區間，再看 close[i] 是否突破。
        sq_count = 0
        sq_high  = -np.inf
        sq_low   =  np.inf
        for j in range(i - 1, max(i - 250, WARMUP - 1), -1):
            if squeeze[j]:
                sq_count += 1
                sq_high = max(sq_high, high[j])
                sq_low  = min(sq_low,  low[j])
            else:
                break

        if sq_count < squeeze_bars:
            continue

        cur_close = close[i]
        trend_up  = ema20[i] > ema50[i]
        rng       = sq_high - sq_low

        entry_price = open_[i + 1]
        signal_dir  = None

        macro_bull = (bull_mask is None or bool(bull_mask.iloc[i]))
        if cur_close > sq_high and trend_up and macro_bull:
            sl = max(sq_low, cur_close * (1 - stop_pct))
            tp = cur_close + rng * tp_mult
            signal_dir = "long"
        elif cur_close < sq_low and not trend_up:
            sl = min(sq_high, cur_close * (1 + stop_pct))
            tp = cur_close - rng * tp_mult
            signal_dir = "short"

        if signal_dir is not None:
            position = {
                "dir":   signal_dir,
                "entry": entry_price,
                "sl":    sl,
                "tp":    tp,
                "date":  str(df_raw.index[i + 1].date()),
            }

    # 結束時強制平倉
    final_position = dict(position) if position is not None else None
    if position is not None:
        exit_price = close[-1]
        ret = (exit_price - position["entry"]) / position["entry"]
        if position["dir"] == "short":
            ret = -ret
        ret -= FEE * 2
        equity *= (1 + ret)
        trades.append({
            "entry_date": position["date"],
            "exit_date":  str(df_raw.index[-1].date()),
            "dir":        position["dir"],
            "entry":      position["entry"],
            "exit":       exit_price,
            "ret_pct":    round(ret * 100, 3),
            "result":     "open",
        })
    equity_curve.append((df_raw.index[-1], equity))

    stats = _calc_stats(trades, equity_curve, initial)
    if _return_position:
        return stats, final_position
    return stats


# ── 三重 SuperTrend 回測 ───────────────────────────────────────────────────────

def backtest_triple_st(df_raw: pd.DataFrame, initial: float = 10_000.0,
                       stop_loss_pct: float | None = None,
                       long_only: bool = False,
                       bull_mask: pd.Series | None = None,
                       _return_position: bool = False) -> dict:
    """
    三重 SuperTrend 策略（含做空）。
    stop_loss_pct: 固定止損百分比，None = 無止損（只靠 ST 訊號出場）
    long_only:     True = 只做多，不做空
    進場規則：三條 ST 全翻多 → LONG；全翻空 → SHORT（long_only=False 時）
    出場規則：全翻至反向，或觸碰 stop_loss_pct 止損
    次根 open 進出場。
    """
    close  = df_raw["close"].values.astype(float)
    high   = df_raw["high"].values.astype(float)
    low    = df_raw["low"].values.astype(float)
    open_  = df_raw["open"].values.astype(float)
    n      = len(close)

    # 計算三條 ST 方向
    dirs = np.array([
        _supertrend(high, low, close, p, m)[0]
        for p, m in ST_PARAMS
    ])  # (3, n)
    green = (dirs == 1).sum(axis=0)  # 0–3

    trades: list[dict] = []
    equity    = initial
    position  = None
    equity_curve = [(df_raw.index[0], initial)]

    for i in range(1, n - 1):
        cur  = int(green[i])
        prev = int(green[i - 1])
        next_open = open_[i + 1]

        # ── 出場檢查 ────────────────────────────────────────────────────────
        if position is not None:
            # ST 訊號出場
            should_exit = (
                (position["dir"] == "long"  and cur == 0 and prev > 0) or
                (position["dir"] == "short" and cur == 3 and prev < 3)
            )
            # 止損出場（bar 內用 high/low 判斷是否觸碰）
            hit_sl = False
            sl_price = None
            if stop_loss_pct is not None:
                entry = position["entry"]
                if position["dir"] == "long":
                    sl_price = entry * (1 - stop_loss_pct)
                    hit_sl = low[i] <= sl_price
                else:
                    sl_price = entry * (1 + stop_loss_pct)
                    hit_sl = high[i] >= sl_price

            if hit_sl or should_exit:
                if hit_sl:
                    exit_price = sl_price
                    result = "sl"
                else:
                    exit_price = next_open
                    result = "exit"
                ret = (exit_price - position["entry"]) / position["entry"]
                if position["dir"] == "short":
                    ret = -ret
                ret -= FEE * 2
                equity *= (1 + ret)
                trades.append({
                    "entry_date": position["date"],
                    "exit_date":  str(df_raw.index[i + 1].date()),
                    "dir":        position["dir"],
                    "entry":      position["entry"],
                    "exit":       exit_price,
                    "ret_pct":    round(ret * 100, 3),
                    "result":     result,
                })
                equity_curve.append((df_raw.index[i + 1], equity))
                position = None

        # ── 進場檢查 ────────────────────────────────────────────────────────
        if position is None:
            macro_bull = (bull_mask is None or bool(bull_mask.iloc[i]))
            if cur == 3 and prev < 3 and macro_bull:
                position = {
                    "dir":   "long",
                    "entry": next_open,
                    "date":  str(df_raw.index[i + 1].date()),
                }
            elif not long_only and cur == 0 and prev > 0:
                position = {
                    "dir":   "short",
                    "entry": next_open,
                    "date":  str(df_raw.index[i + 1].date()),
                }

    # 結束強制平倉
    final_position = dict(position) if position is not None else None
    if position is not None:
        exit_price = close[-1]
        ret = (exit_price - position["entry"]) / position["entry"]
        if position["dir"] == "short":
            ret = -ret
        ret -= FEE * 2
        equity *= (1 + ret)
        trades.append({
            "entry_date": position["date"],
            "exit_date":  str(df_raw.index[-1].date()),
            "dir":        position["dir"],
            "entry":      position["entry"],
            "exit":       exit_price,
            "ret_pct":    round(ret * 100, 3),
            "result":     "open",
        })
    equity_curve.append((df_raw.index[-1], equity))

    stats = _calc_stats(trades, equity_curve, initial)
    if _return_position:
        return stats, final_position
    return stats


# ── Donchian Channel Breakout 回測 ────────────────────────────────────────────

def backtest_donchian(df_raw: pd.DataFrame, initial: float = 10_000.0,
                      entry_period: int = 20, exit_period: int = 10,
                      stop_pct: float = 0.08,
                      long_only: bool = False,
                      _return_position: bool = False) -> dict:
    """
    Donchian Channel 龜式突破策略。
    訊號：收盤突破前 entry_period 根高/低點 → 次根 open 進場。
    出場：收盤跌破前 exit_period 根低點（多單）/ 漲破高點（空單），或止損。
    """
    close  = df_raw["close"].values.astype(float)
    high   = df_raw["high"].values.astype(float)
    low    = df_raw["low"].values.astype(float)
    open_  = df_raw["open"].values.astype(float)
    n      = len(close)
    WARMUP = max(entry_period, exit_period) + 5

    trades: list[dict] = []
    equity        = initial
    position      = None
    equity_curve  = [(df_raw.index[0], initial)]

    for i in range(WARMUP, n - 1):
        # ── 出場檢查 ────────────────────────────────────────────────────────
        if position is not None:
            # Donchian 出場通道（前 exit_period 根，不含當根）
            dc_exit_high = float(np.max(high[i - exit_period: i]))
            dc_exit_low  = float(np.min(low[i  - exit_period: i]))
            entry        = position["entry"]

            hit_sl = False
            dc_exit = False
            if position["dir"] == "long":
                sl_price = entry * (1 - stop_pct)
                hit_sl   = low[i] <= sl_price
                dc_exit  = close[i] < dc_exit_low
                exit_p   = sl_price if hit_sl else (open_[i + 1] if dc_exit else None)
            else:
                sl_price = entry * (1 + stop_pct)
                hit_sl   = high[i] >= sl_price
                dc_exit  = close[i] > dc_exit_high
                exit_p   = sl_price if hit_sl else (open_[i + 1] if dc_exit else None)

            if exit_p is not None:
                ret = (exit_p - entry) / entry
                if position["dir"] == "short":
                    ret = -ret
                ret -= FEE * 2
                equity *= (1 + ret)
                trades.append({
                    "entry_date": position["date"],
                    "exit_date":  str(df_raw.index[i + 1].date()),
                    "dir":        position["dir"],
                    "entry":      entry,
                    "exit":       exit_p,
                    "ret_pct":    round(ret * 100, 3),
                    "result":     "sl" if hit_sl else "dc_exit",
                })
                equity_curve.append((df_raw.index[i], equity))
                position = None

        # ── 進場檢查 ────────────────────────────────────────────────────────
        if position is None:
            dc_entry_high = float(np.max(high[i - entry_period: i]))
            dc_entry_low  = float(np.min(low[i  - entry_period: i]))
            if close[i] > dc_entry_high:
                position = {
                    "dir":   "long",
                    "entry": open_[i + 1],
                    "date":  str(df_raw.index[i + 1].date()),
                }
            elif not long_only and close[i] < dc_entry_low:
                position = {
                    "dir":   "short",
                    "entry": open_[i + 1],
                    "date":  str(df_raw.index[i + 1].date()),
                }

    final_position = dict(position) if position is not None else None
    if position is not None:
        exit_p = close[-1]
        ret = (exit_p - position["entry"]) / position["entry"]
        if position["dir"] == "short":
            ret = -ret
        ret -= FEE * 2
        equity *= (1 + ret)
        trades.append({
            "entry_date": position["date"],
            "exit_date":  str(df_raw.index[-1].date()),
            "dir":        position["dir"],
            "entry":      position["entry"],
            "exit":       exit_p,
            "ret_pct":    round(ret * 100, 3),
            "result":     "open",
        })
    equity_curve.append((df_raw.index[-1], equity))
    stats = _calc_stats(trades, equity_curve, initial)
    if _return_position:
        return stats, final_position
    return stats


# ── EMA Cross + Trend Filter 回測 ──────────────────────────────────────────────

def backtest_ema_cross(df_raw: pd.DataFrame, initial: float = 10_000.0,
                       fast: int = 9, slow: int = 21, trend: int = 200,
                       stop_pct: float = 0.06,
                       long_only: bool = False,
                       _return_position: bool = False) -> dict:
    """
    EMA Cross + Trend Filter 策略。
    進場：fast EMA 上穿 slow EMA 且 close > trend EMA → LONG
          fast EMA 下穿 slow EMA 且 close < trend EMA → SHORT
    出場：fast EMA 反向穿越 slow EMA，或止損。
    次根 open 進出場。
    """
    close  = df_raw["close"].values.astype(float)
    high   = df_raw["high"].values.astype(float)
    low    = df_raw["low"].values.astype(float)
    open_  = df_raw["open"].values.astype(float)
    n      = len(close)

    s_close = pd.Series(close)
    ema_fast  = s_close.ewm(span=fast,  adjust=False).mean().values
    ema_slow  = s_close.ewm(span=slow,  adjust=False).mean().values
    ema_trend = s_close.ewm(span=trend, adjust=False).mean().values

    WARMUP = trend + 5
    trades: list[dict] = []
    equity       = initial
    position     = None
    equity_curve = [(df_raw.index[0], initial)]

    for i in range(WARMUP, n - 1):
        cur_cross  = ema_fast[i]  - ema_slow[i]
        prev_cross = ema_fast[i-1] - ema_slow[i-1]

        # ── 出場檢查 ────────────────────────────────────────────────────────
        if position is not None:
            entry = position["entry"]
            crossed_against = (
                (position["dir"] == "long"  and cur_cross < 0 and prev_cross >= 0) or
                (position["dir"] == "short" and cur_cross > 0 and prev_cross <= 0)
            )
            hit_sl = False
            sl_price = None
            if position["dir"] == "long":
                sl_price = entry * (1 - stop_pct)
                hit_sl   = low[i] <= sl_price
            else:
                sl_price = entry * (1 + stop_pct)
                hit_sl   = high[i] >= sl_price

            if hit_sl or crossed_against:
                exit_p = sl_price if hit_sl else open_[i + 1]
                ret = (exit_p - entry) / entry
                if position["dir"] == "short":
                    ret = -ret
                ret -= FEE * 2
                equity *= (1 + ret)
                trades.append({
                    "entry_date": position["date"],
                    "exit_date":  str(df_raw.index[i + 1].date()),
                    "dir":        position["dir"],
                    "entry":      entry,
                    "exit":       exit_p,
                    "ret_pct":    round(ret * 100, 3),
                    "result":     "sl" if hit_sl else "cross_exit",
                })
                equity_curve.append((df_raw.index[i], equity))
                position = None

        # ── 進場檢查 ────────────────────────────────────────────────────────
        if position is None:
            golden_cross = cur_cross > 0 and prev_cross <= 0
            death_cross  = cur_cross < 0 and prev_cross >= 0
            if golden_cross and close[i] > ema_trend[i]:
                position = {
                    "dir":   "long",
                    "entry": open_[i + 1],
                    "date":  str(df_raw.index[i + 1].date()),
                }
            elif not long_only and death_cross and close[i] < ema_trend[i]:
                position = {
                    "dir":   "short",
                    "entry": open_[i + 1],
                    "date":  str(df_raw.index[i + 1].date()),
                }

    final_position = dict(position) if position is not None else None
    if position is not None:
        exit_p = close[-1]
        ret = (exit_p - position["entry"]) / position["entry"]
        if position["dir"] == "short":
            ret = -ret
        ret -= FEE * 2
        equity *= (1 + ret)
        trades.append({
            "entry_date": position["date"],
            "exit_date":  str(df_raw.index[-1].date()),
            "dir":        position["dir"],
            "entry":      position["entry"],
            "exit":       exit_p,
            "ret_pct":    round(ret * 100, 3),
            "result":     "open",
        })
    equity_curve.append((df_raw.index[-1], equity))
    stats = _calc_stats(trades, equity_curve, initial)
    if _return_position:
        return stats, final_position
    return stats


# ── 優化函式 ───────────────────────────────────────────────────────────────────

def _score(r: dict) -> float:
    """
    綜合評分：Calmar ratio 風格，報酬除以最大回撤，再乘以勝率。
    交易次數 < 5 的結果直接排除（樣本太少）。
    """
    if r["trades"] < 5 or r["max_dd_pct"] <= 0:
        return -999.0
    calmar = r["total_return_pct"] / r["max_dd_pct"]
    return calmar * (r["win_rate"] / 100)


def optimize_vb(df: pd.DataFrame, verbose: bool = False) -> dict:
    """
    VB 參數網格搜尋。
    回傳最佳組合與各組合結果表格。
    """
    grid = {
        "atr_thresh":   [0.70, 0.75, 0.80, 0.85, 0.90],
        "squeeze_bars": [10, 15, 20, 25],
        "tp_mult":      [1.0, 1.5, 2.0, 2.5],
    }

    best_score  = -999.0
    best_params = {}
    best_result = {}
    rows = []

    total = len(grid["atr_thresh"]) * len(grid["squeeze_bars"]) * len(grid["tp_mult"])
    done  = 0

    for at in grid["atr_thresh"]:
        for sq in grid["squeeze_bars"]:
            for tp in grid["tp_mult"]:
                r = backtest_vb(df, atr_thresh=at, squeeze_bars=sq, tp_mult=tp)
                sc = _score(r)
                rows.append({
                    "atr_thresh": at, "squeeze_bars": sq, "tp_mult": tp,
                    **r, "score": round(sc, 4),
                })
                if sc > best_score:
                    best_score  = sc
                    best_params = {"atr_thresh": at, "squeeze_bars": sq, "tp_mult": tp}
                    best_result = r
                done += 1
                if verbose:
                    print(f"  [{done}/{total}] at={at} sq={sq} tp={tp} "
                          f"→ ret={r['total_return_pct']:+.1f}% dd={r['max_dd_pct']:.1f}% "
                          f"n={r['trades']} score={sc:.3f}")

    return {
        "best_params":  best_params,
        "best_result":  best_result,
        "best_score":   round(best_score, 4),
        "all_rows":     sorted(rows, key=lambda x: x["score"], reverse=True),
    }


def optimize_3st(df: pd.DataFrame) -> dict:
    """
    三重 ST 優化：測試不同止損水準 + 純多模式。
    """
    sl_levels  = [0.04, 0.06, 0.08, 0.10, 0.12, 0.15, None]
    modes      = [("long+short", False), ("long_only", True)]
    best_score = -999.0
    best_cfg   = {}
    best_result = {}
    rows = []

    for mode_label, lo in modes:
        for sl in sl_levels:
            r  = backtest_triple_st(df, stop_loss_pct=sl, long_only=lo)
            sc = _score(r)
            sl_label = f"{sl*100:.0f}%" if sl is not None else "無止損"
            rows.append({
                "mode": mode_label, "stop_loss": sl_label,
                **r, "score": round(sc, 4),
            })
            if sc > best_score:
                best_score  = sc
                best_cfg    = {"mode": mode_label, "stop_loss": sl_label,
                               "stop_loss_pct": sl, "long_only": lo}
                best_result = r

    return {
        "best_cfg":    best_cfg,
        "best_result": best_result,
        "best_score":  round(best_score, 4),
        "all_rows":    sorted(rows, key=lambda x: x["score"], reverse=True),
    }


def optimize_donchian(df: pd.DataFrame) -> dict:
    """Donchian Channel 參數網格搜尋。"""
    grid = {
        "entry_period": [10, 15, 20, 25, 30],
        "exit_period":  [5, 7, 10, 13],
        "stop_pct":     [0.06, 0.08, 0.10],
    }
    best_score, best_params, best_result = -999.0, {}, {}
    rows = []
    for ep in grid["entry_period"]:
        for xp in grid["exit_period"]:
            if xp >= ep:
                continue   # exit channel 不能大於 entry channel
            for sp in grid["stop_pct"]:
                r = backtest_donchian(df, entry_period=ep, exit_period=xp, stop_pct=sp)
                sc = _score(r)
                rows.append({"entry_period": ep, "exit_period": xp, "stop_pct": sp,
                              **r, "score": round(sc, 4)})
                if sc > best_score:
                    best_score  = sc
                    best_params = {"entry_period": ep, "exit_period": xp, "stop_pct": sp}
                    best_result = r
    return {
        "best_params": best_params, "best_result": best_result,
        "best_score": round(best_score, 4),
        "all_rows": sorted(rows, key=lambda x: x["score"], reverse=True),
    }


def optimize_ema_cross(df: pd.DataFrame) -> dict:
    """EMA Cross 參數網格搜尋。"""
    grid = {
        "fast":     [7, 9, 12],
        "slow":     [21, 26, 34],
        "trend":    [100, 200],
        "stop_pct": [0.05, 0.07, 0.10],
    }
    best_score, best_params, best_result = -999.0, {}, {}
    rows = []
    for f in grid["fast"]:
        for s in grid["slow"]:
            for t in grid["trend"]:
                for sp in grid["stop_pct"]:
                    r = backtest_ema_cross(df, fast=f, slow=s, trend=t, stop_pct=sp)
                    sc = _score(r)
                    rows.append({"fast": f, "slow": s, "trend": t, "stop_pct": sp,
                                 **r, "score": round(sc, 4)})
                    if sc > best_score:
                        best_score  = sc
                        best_params = {"fast": f, "slow": s, "trend": t, "stop_pct": sp}
                        best_result = r
    return {
        "best_params": best_params, "best_result": best_result,
        "best_score": round(best_score, 4),
        "all_rows": sorted(rows, key=lambda x: x["score"], reverse=True),
    }


def print_optimization_summary(symbol: str, vb_opt: dict, st_opt: dict,
                                dc_opt: dict = None, ema_opt: dict = None):
    print(f"\n{'─'*70}")
    print(f"[{symbol}] 優化結果")
    print(f"{'─'*70}")

    # VB 最佳
    bp = vb_opt["best_params"]
    br = vb_opt["best_result"]
    print(f"  VB 最佳：ATR閾值={bp.get('atr_thresh')}  壓縮根數={bp.get('squeeze_bars')}  "
          f"TP倍數={bp.get('tp_mult')}")
    print(f"         報酬={br['total_return_pct']:+.1f}%  勝率={br['win_rate']}%  "
          f"MaxDD={br['max_dd_pct']:.1f}%  次數={br['trades']}  "
          f"評分={vb_opt['best_score']:.3f}")

    # VB Top-5
    print("  VB Top-5：")
    for row in vb_opt["all_rows"][:5]:
        print(f"    at={row['atr_thresh']} sq={row['squeeze_bars']} tp={row['tp_mult']}  "
              f"ret={row['total_return_pct']:+.1f}%  dd={row['max_dd_pct']:.1f}%  "
              f"n={row['trades']}  score={row['score']:.3f}")

    # 3ST 最佳
    bc = st_opt["best_cfg"]
    br2 = st_opt["best_result"]
    print(f"  3ST 最佳：模式={bc.get('mode')}  止損={bc.get('stop_loss')}")
    print(f"          報酬={br2['total_return_pct']:+.1f}%  勝率={br2['win_rate']}%  "
          f"MaxDD={br2['max_dd_pct']:.1f}%  次數={br2['trades']}  "
          f"評分={st_opt['best_score']:.3f}")

    # 3ST 所有組合
    print("  3ST 全部組合：")
    for row in st_opt["all_rows"]:
        print(f"    {row['mode']:<12} SL={row['stop_loss']:<6}  "
              f"ret={row['total_return_pct']:+.1f}%  dd={row['max_dd_pct']:.1f}%  "
              f"n={row['trades']}  score={row['score']:.3f}")

    # DC 最佳
    if dc_opt:
        bp = dc_opt["best_params"]
        br = dc_opt["best_result"]
        print(f"  DC 最佳：entry={bp.get('entry_period')}  exit={bp.get('exit_period')}  "
              f"stop={bp.get('stop_pct')}")
        print(f"         報酬={br['total_return_pct']:+.1f}%  勝率={br['win_rate']}%  "
              f"MaxDD={br['max_dd_pct']:.1f}%  次數={br['trades']}  "
              f"評分={dc_opt['best_score']:.3f}")
        print("  DC Top-5：")
        for row in dc_opt["all_rows"][:5]:
            print(f"    en={row['entry_period']} ex={row['exit_period']} sl={row['stop_pct']}  "
                  f"ret={row['total_return_pct']:+.1f}%  dd={row['max_dd_pct']:.1f}%  "
                  f"n={row['trades']}  score={row['score']:.3f}")

    # EMA Cross 最佳
    if ema_opt:
        bp = ema_opt["best_params"]
        br = ema_opt["best_result"]
        print(f"  EMA 最佳：fast={bp.get('fast')}  slow={bp.get('slow')}  "
              f"trend={bp.get('trend')}  stop={bp.get('stop_pct')}")
        print(f"          報酬={br['total_return_pct']:+.1f}%  勝率={br['win_rate']}%  "
              f"MaxDD={br['max_dd_pct']:.1f}%  次數={br['trades']}  "
              f"評分={ema_opt['best_score']:.3f}")
        print("  EMA Top-5：")
        for row in ema_opt["all_rows"][:5]:
            print(f"    f={row['fast']} s={row['slow']} t={row['trend']} sl={row['stop_pct']}  "
                  f"ret={row['total_return_pct']:+.1f}%  dd={row['max_dd_pct']:.1f}%  "
                  f"n={row['trades']}  score={row['score']:.3f}")


# ── 輸出格式 ───────────────────────────────────────────────────────────────────

def print_summary(all_results: dict):
    print("\n" + "=" * 90)
    print("策略比較摘要（每幣種）")
    print("=" * 90)
    header = (
        f"{'幣種':<12} "
        f"{'策略':<6} "
        f"{'總報酬':>9} "
        f"{'勝率':>7} "
        f"{'次數':>6} "
        f"{'多勝率':>8} "
        f"{'空勝率':>8} "
        f"{'多次':>6} "
        f"{'空次':>6} "
        f"{'均損益':>8} "
        f"{'MaxDD':>8} "
        f"{'Sharpe':>8}"
    )
    print(header)
    print("-" * 90)

    totals = {"VB": [], "3ST": [], "DC": [], "EMA": []}

    for symbol, res in all_results.items():
        strategies = [("VB", "VB"), ("3ST", "3ST")]
        if "DC" in res:
            strategies.append(("DC", "DC"))
        if "EMA" in res:
            strategies.append(("EMA", "EMA"))
        for key, label in strategies:
            r = res[key]
            print(
                f"{symbol:<12} "
                f"{label:<6} "
                f"{r['total_return_pct']:>8.1f}% "
                f"{r['win_rate']:>6.1f}% "
                f"{r['trades']:>6} "
                f"{r['long_win_rate']:>7.1f}% "
                f"{r['short_win_rate']:>7.1f}% "
                f"{r['long_trades']:>6} "
                f"{r['short_trades']:>6} "
                f"{r['avg_ret_pct']:>7.2f}% "
                f"{r['max_dd_pct']:>7.1f}% "
                f"{r['sharpe']:>8.3f}"
            )
            totals[key].append(r["total_return_pct"])
        print()

    print("=" * 90)
    for key, label in [("VB", "VB"), ("3ST", "3ST"), ("DC", "DC"), ("EMA", "EMA")]:
        if totals[key]:
            print(f"  {label} 平均總報酬：{sum(totals[key])/len(totals[key]):+.1f}%")


def print_detail(symbol: str, res: dict):
    """印出單一幣種的詳細逐筆紀錄（可選）。"""
    print(f"\n── {symbol} 詳細交易紀錄 ──")
    for key in ("VB", "3ST", "DC", "EMA"):
        if key not in res:
            continue
        r = res[key]
        print(f"  [{key}] 共 {r['trades']} 筆，勝率 {r['win_rate']}%，"
              f"總報酬 {r['total_return_pct']:+.1f}%，MaxDD {r['max_dd_pct']:.1f}%")


# ── 主程式 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="加密貨幣策略比較回測")
    parser.add_argument("--symbols",     nargs="+", default=COINS)
    parser.add_argument("--start",       default="2023-01-01")
    parser.add_argument("--end",         default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--vb-interval", default="1h",
                        choices=["15m", "30m", "1h", "2h", "4h"],
                        help="VB 策略時框（預設 1h）")
    parser.add_argument("--st-interval", default="4h",
                        choices=["1h", "2h", "4h", "6h", "8h", "12h", "1d"],
                        help="三重 ST 策略時框（預設 4h）")
    parser.add_argument("--detail",      action="store_true",
                        help="顯示每幣種詳細摘要")
    parser.add_argument("--optimize",    action="store_true",
                        help="執行參數優化（VB 網格搜尋 + 3ST 止損掃描）")
    parser.add_argument("--mode",        default="compare",
                        choices=["compare", "per-coin"],
                        help="compare=四策略並排比較（預設）；per-coin=每幣用最佳已知策略")
    parser.add_argument("--opt-symbols", nargs="+",
                        help="指定優化幣種（預設同 --symbols）")
    args = parser.parse_args()

    print(f"回測期間：{args.start} ～ {args.end}")
    print(f"VB 時框：{args.vb_interval}  |  三重 ST 時框：{args.st_interval}")
    print(f"幣種：{' '.join(args.symbols)}")
    print(f"手續費：{FEE*100:.3f}% 單邊（往返 {FEE*2*100:.3f}%）")
    print("=" * 60)

    all_results = {}

    # ── Per-coin 最佳策略模式 ─────────────────────────────────────────────────
    if not args.optimize and args.mode == "per-coin":
        print("\n[Per-coin 模式] 每幣套用最佳已知策略\n")
        pc_rows = []

        for symbol in args.symbols:
            cfg = COIN_BEST.get(symbol)
            if cfg is None:
                print(f"  [{symbol}] 無預設配置，跳過（可先跑 --optimize 取得最佳參數）")
                continue

            strat = cfg["strategy"]
            # 決定需要哪個時框的資料
            need_vb  = strat == "VB"
            interval = args.vb_interval if need_vb else args.st_interval
            df = load_ohlcv(symbol, interval, args.start, args.end)
            if df is None or len(df) < 150:
                print(f"  [{symbol}] 資料不足，跳過。")
                continue

            # ── 大趨勢過濾（日線 EMA200）────────────────────────────────────
            bull_mask = None
            if cfg.get("trend_filter"):
                df_d = load_ohlcv(symbol, "1d", args.start, args.end)
                if df_d is not None and len(df_d) >= 200:
                    ema200 = df_d["close"].ewm(span=200, adjust=False).mean()
                    daily_bull = (df_d["close"] > ema200).rename("bull")
                    # forward-fill 到目標時框
                    bull_mask = (
                        daily_bull
                        .reindex(df.index, method="ffill")
                        .fillna(False)
                    )
                    pct = bull_mask.mean() * 100
                    print(f"  趨勢過濾：日線 EMA200，多頭區間佔 {pct:.0f}%")

            print(f"[{symbol}]  策略={strat}  時框={interval}  資料={len(df)}根")

            if strat == "VB":
                r = backtest_vb(df,
                                atr_thresh=cfg.get("atr_thresh", ATR_THRESH),
                                squeeze_bars=cfg.get("squeeze_bars", SQUEEZE_BARS),
                                tp_mult=cfg.get("tp_mult", TP_MULT),
                                bull_mask=bull_mask)
            elif strat == "3ST":
                long_only = cfg.get("mode", "long_only") == "long_only"
                stop_loss = cfg.get("stop_loss")
                r = backtest_triple_st(df, long_only=long_only, stop_loss_pct=stop_loss,
                                       bull_mask=bull_mask)
            elif strat == "DC":
                r = backtest_donchian(df,
                                      entry_period=cfg.get("entry_period", 20),
                                      exit_period=cfg.get("exit_period", 10),
                                      stop_pct=cfg.get("stop_pct", 0.08))
            elif strat == "EMA":
                r = backtest_ema_cross(df,
                                       fast=cfg.get("fast", 9),
                                       slow=cfg.get("slow", 21),
                                       trend=cfg.get("trend", 200),
                                       stop_pct=cfg.get("stop_pct", 0.07))
            else:
                print(f"  [{symbol}] 未知策略 {strat}，跳過。")
                continue

            print(f"  報酬 {r['total_return_pct']:>+8.1f}%  勝率 {r['win_rate']:>5.1f}%  "
                  f"次數 {r['trades']:>3}  MaxDD {r['max_dd_pct']:>5.1f}%  "
                  f"Sharpe {r['sharpe']:>6.3f}")
            pc_rows.append({"symbol": symbol, "strategy": strat, **r})

        if pc_rows:
            print("\n" + "=" * 75)
            print(f"  Per-coin 彙總  （{args.start} ～ {args.end}）")
            print("=" * 75)
            print(f"  {'幣種':<12} {'策略':<5} {'報酬':>9} {'勝率':>7} {'次數':>5} "
                  f"{'MaxDD':>7} {'Sharpe':>8}")
            print("  " + "-" * 60)
            total_ret = []
            for row in pc_rows:
                print(f"  {row['symbol']:<12} {row['strategy']:<5} "
                      f"{row['total_return_pct']:>8.1f}% {row['win_rate']:>6.1f}% "
                      f"{row['trades']:>5}  {row['max_dd_pct']:>6.1f}% "
                      f"{row['sharpe']:>8.3f}")
                total_ret.append(row["total_return_pct"])
            print("=" * 75)
            print(f"  平均報酬：{sum(total_ret)/len(total_ret):+.1f}%")

            ts  = datetime.now().strftime("%Y%m%d_%H%M")
            out = RESULT_DIR / f"crypto_percoin_{args.start}_{ts}.json"
            with open(out, "w", encoding="utf-8") as f:
                json.dump({
                    "meta": {"start": args.start, "end": args.end,
                             "mode": "per-coin", "generated": ts},
                    "results": pc_rows,
                }, f, ensure_ascii=False, indent=2)
            print(f"\n結果已儲存至 {out}")
        return

    # ── 一般比較模式 ──────────────────────────────────────────────────────────
    if not args.optimize:
        for symbol in args.symbols:
            print(f"\n[{symbol}]")

            df_vb = load_ohlcv(symbol, args.vb_interval, args.start, args.end)
            df_st = load_ohlcv(symbol, args.st_interval, args.start, args.end)

            if df_vb is None or len(df_vb) < 150:
                print(f"  VB 資料不足，跳過。")
                continue
            if df_st is None or len(df_st) < 150:
                print(f"  3ST 資料不足，跳過。")
                continue

            print(f"  VB  資料：{len(df_vb):>5} 根 {args.vb_interval}")
            print(f"  3ST 資料：{len(df_st):>5} 根 {args.st_interval}")

            vb_res  = backtest_vb(df_vb)
            st_res  = backtest_triple_st(df_st)
            dc_res  = backtest_donchian(df_st)
            ema_res = backtest_ema_cross(df_st)

            for label, interval, r in [
                ("VB ", args.vb_interval, vb_res),
                ("3ST", args.st_interval, st_res),
                ("DC ", args.st_interval, dc_res),
                ("EMA", args.st_interval, ema_res),
            ]:
                print(f"  {label}({interval})：交易 {r['trades']:>3} 次 | "
                      f"勝率 {r['win_rate']:>5.1f}% | "
                      f"多 {r['long_trades']}筆/{r['long_win_rate']:.0f}% "
                      f"空 {r['short_trades']}筆/{r['short_win_rate']:.0f}% | "
                      f"總報酬 {r['total_return_pct']:>+7.1f}% | "
                      f"MaxDD {r['max_dd_pct']:.1f}%")

            all_results[symbol] = {"VB": vb_res, "3ST": st_res, "DC": dc_res, "EMA": ema_res}
            if args.detail:
                print_detail(symbol, all_results[symbol])

        if not all_results:
            print("無結果，請檢查資料來源。")
            return

        print_summary(all_results)

        ts  = datetime.now().strftime("%Y%m%d_%H%M")
        out = RESULT_DIR / f"crypto_cmp_VB{args.vb_interval}_3ST{args.st_interval}_{args.start}_{ts}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump({
                "meta": {
                    "start": args.start, "end": args.end,
                    "vb_interval": args.vb_interval,
                    "st_interval": args.st_interval,
                    "generated": ts,
                },
                "results": all_results,
            }, f, ensure_ascii=False, indent=2)
        print(f"\n結果已儲存至 {out}")

    # ── 優化模式 ──────────────────────────────────────────────────────────────
    else:
        opt_symbols = args.opt_symbols or args.symbols
        print(f"\n優化幣種：{' '.join(opt_symbols)}")
        print("VB 網格：ATR閾值×5 × 壓縮根數×4 × TP倍數×4 = 80 組合 / 幣種")
        print("3ST 掃描：止損水準×7 × 模式×2 = 14 組合 / 幣種")
        print("─" * 60)

        opt_results = {}
        cross_vb  = {}   # 跨幣種統計：{params_key: [scores]}
        cross_st  = {}
        cross_dc  = {}
        cross_ema = {}

        for symbol in opt_symbols:
            print(f"\n[{symbol}] 計算中...")
            df_vb = load_ohlcv(symbol, args.vb_interval, args.start, args.end)
            df_st = load_ohlcv(symbol, args.st_interval, args.start, args.end)

            if df_vb is None or df_st is None:
                print(f"  資料不足，跳過。")
                continue

            vb_opt  = optimize_vb(df_vb)
            st_opt  = optimize_3st(df_st)
            dc_opt  = optimize_donchian(df_st)
            ema_opt = optimize_ema_cross(df_st)
            print_optimization_summary(symbol, vb_opt, st_opt, dc_opt, ema_opt)
            opt_results[symbol] = {"vb": vb_opt, "st": st_opt, "dc": dc_opt, "ema": ema_opt}

            # 跨幣種累積分數（找最穩定通用參數）
            for row in vb_opt["all_rows"]:
                k = f"at{row['atr_thresh']}_sq{row['squeeze_bars']}_tp{row['tp_mult']}"
                cross_vb.setdefault(k, []).append(row["score"])
            for row in st_opt["all_rows"]:
                k = f"{row['mode']}_sl{row['stop_loss']}"
                cross_st.setdefault(k, []).append(row["score"])
            for row in dc_opt["all_rows"]:
                k = f"en{row['entry_period']}_ex{row['exit_period']}_sl{row['stop_pct']}"
                cross_dc.setdefault(k, []).append(row["score"])
            for row in ema_opt["all_rows"]:
                k = f"f{row['fast']}_s{row['slow']}_t{row['trend']}_sl{row['stop_pct']}"
                cross_ema.setdefault(k, []).append(row["score"])

        # 跨幣種最穩定參數（所有幣種平均分最高）
        if len(opt_symbols) > 1:
            import statistics
            print(f"\n{'='*70}")
            print("跨幣種最穩定參數（所有幣種平均評分最高）")
            print(f"{'='*70}")

            for label, cross in [
                ("VB",  cross_vb),
                ("3ST", cross_st),
                ("DC",  cross_dc),
                ("EMA", cross_ema),
            ]:
                avg = [(k, statistics.mean(v), min(v))
                       for k, v in cross.items() if len(v) == len(opt_symbols)]
                avg.sort(key=lambda x: x[1], reverse=True)
                print(f"{label} Top-5 通用參數：")
                for k, avg_sc, min_sc in avg[:5]:
                    print(f"  {k:<40} avg={avg_sc:.3f}  min={min_sc:.3f}")

        # 儲存優化結果
        ts  = datetime.now().strftime("%Y%m%d_%H%M")
        out = RESULT_DIR / f"crypto_opt_{args.start}_{ts}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump({
                "meta": {"start": args.start, "end": args.end, "generated": ts},
                "results": {
                    sym: {
                        "vb_best":   v["vb"]["best_params"],
                        "vb_result": v["vb"]["best_result"],
                        "st_best":   v["st"]["best_cfg"],
                        "st_result": v["st"]["best_result"],
                        "dc_best":   v["dc"]["best_params"],
                        "dc_result": v["dc"]["best_result"],
                        "ema_best":  v["ema"]["best_params"],
                        "ema_result": v["ema"]["best_result"],
                    } for sym, v in opt_results.items()
                },
            }, f, ensure_ascii=False, indent=2)
        print(f"\n優化結果已儲存至 {out}")


# ── 信號偵測（供 API 使用）────────────────────────────────────────────────────

def detect_signal(symbol: str, lookback_start: str = "2024-06-01") -> dict:
    """
    判斷指定幣種當前信號狀態。
    使用 COIN_BEST 配置，回傳持倉中/剛進場/空倉等資訊。
    """
    cfg = COIN_BEST.get(symbol)
    if cfg is None:
        return {"symbol": symbol, "error": "無預設配置", "strategy": "—"}

    strat    = cfg["strategy"]
    interval = "1h" if strat == "VB" else "4h"
    end      = datetime.now().strftime("%Y-%m-%d")

    df = load_ohlcv(symbol, interval, lookback_start, end)
    if df is None or len(df) < 150:
        return {"symbol": symbol, "error": "資料不足", "strategy": strat}

    current_price = float(df["close"].iloc[-1])
    current_bar   = str(df.index[-1])

    # ── 大趨勢過濾 ───────────────────────────────────────────────────────────
    bull_mask  = None
    macro_bull = None
    if cfg.get("trend_filter"):
        df_d = load_ohlcv(symbol, "1d", lookback_start, end)
        if df_d is not None and len(df_d) >= 200:
            ema200      = df_d["close"].ewm(span=200, adjust=False).mean()
            daily_bull  = (df_d["close"] > ema200)
            bull_mask   = daily_bull.reindex(df.index, method="ffill").fillna(False)
            macro_bull  = bool(bull_mask.iloc[-1])

    # ── 執行策略，取得最終持倉狀態 ─────────────────────────────────────────
    try:
        if strat == "VB":
            _, pos = backtest_vb(
                df,
                atr_thresh=cfg.get("atr_thresh", ATR_THRESH),
                squeeze_bars=cfg.get("squeeze_bars", SQUEEZE_BARS),
                tp_mult=cfg.get("tp_mult", TP_MULT),
                bull_mask=bull_mask,
                _return_position=True,
            )
        elif strat == "3ST":
            long_only = cfg.get("mode", "long_only") == "long_only"
            _, pos = backtest_triple_st(
                df,
                long_only=long_only,
                stop_loss_pct=cfg.get("stop_loss"),
                bull_mask=bull_mask,
                _return_position=True,
            )
        elif strat == "DC":
            _, pos = backtest_donchian(
                df,
                entry_period=cfg.get("entry_period", 20),
                exit_period=cfg.get("exit_period", 10),
                stop_pct=cfg.get("stop_pct", 0.08),
                _return_position=True,
            )
        elif strat == "EMA":
            _, pos = backtest_ema_cross(
                df,
                fast=cfg.get("fast", 9),
                slow=cfg.get("slow", 21),
                trend=cfg.get("trend", 200),
                stop_pct=cfg.get("stop_pct", 0.07),
                _return_position=True,
            )
        else:
            return {"symbol": symbol, "error": f"未知策略 {strat}", "strategy": strat}
    except Exception as e:
        return {"symbol": symbol, "error": str(e), "strategy": strat}

    in_position = pos is not None
    result: dict = {
        "symbol":        symbol,
        "strategy":      strat,
        "interval":      interval,
        "current_price": round(current_price, 6),
        "current_bar":   current_bar,
        "in_position":   in_position,
        "direction":     pos["dir"] if in_position else None,
        "entry_price":   round(float(pos["entry"]), 6) if in_position else None,
        "entry_date":    pos["date"] if in_position else None,
        "pnl_pct":       None,
        "macro_bull":    macro_bull,
    }

    if in_position:
        ep = float(pos["entry"])
        pnl = (current_price - ep) / ep * 100
        if pos["dir"] == "short":
            pnl = -pnl
        result["pnl_pct"] = round(pnl, 2)

    return result


if __name__ == "__main__":
    main()
