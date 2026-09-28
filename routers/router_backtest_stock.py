import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yfinance as yf
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from datetime import datetime, timedelta

router = APIRouter(prefix="/api/backtest", tags=["backtest"])

# ── 三重 SuperTrend 參數 ───────────────────────────────────────────────────────
ST_PARAMS = [(11, 2.0), (10, 1.0), (12, 3.0)]


# ── SuperTrend 計算 ────────────────────────────────────────────────────────────

def _calc_atr_wilder(high: np.ndarray, low: np.ndarray,
                     close: np.ndarray, period: int) -> np.ndarray:
    """Wilder's ATR"""
    n = len(close)
    prev_c = np.empty(n)
    prev_c[0] = close[0]
    prev_c[1:] = close[:-1]
    tr = np.maximum(high - low,
         np.maximum(np.abs(high - prev_c), np.abs(low - prev_c)))
    atr = np.zeros(n)
    if n < period:
        return atr
    atr[period - 1] = tr[:period].mean()
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    atr[:period - 1] = atr[period - 1]
    return atr


def calc_supertrend(high: np.ndarray, low: np.ndarray,
                    close: np.ndarray,
                    period: int = 10, mult: float = 3.0):
    """
    計算 SuperTrend 方向與支撐線。
    回傳：(direction, support)
      direction: 1 = 多方（綠）, -1 = 空方（紅）
      support:   多方時為下軌支撐（fd），空方時為上軌壓力（fu）
    """
    n = len(close)
    if n < period + 5:
        return np.full(n, -1, dtype=np.int8), np.full(n, 0.0)

    atr = _calc_atr_wilder(high, low, close, period)
    hl2 = (high + low) / 2.0
    basic_up = hl2 + mult * atr
    basic_dn = hl2 - mult * atr

    final_up  = basic_up.copy()
    final_dn  = basic_dn.copy()
    direction = np.ones(n, dtype=np.int8)

    for i in range(1, n):
        if basic_up[i] < final_up[i - 1] or close[i - 1] > final_up[i - 1]:
            final_up[i] = basic_up[i]
        else:
            final_up[i] = final_up[i - 1]

        if basic_dn[i] > final_dn[i - 1] or close[i - 1] < final_dn[i - 1]:
            final_dn[i] = basic_dn[i]
        else:
            final_dn[i] = final_dn[i - 1]

        if close[i] > final_up[i - 1]:
            direction[i] = 1
        elif close[i] < final_dn[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]

    # 多方支撐 = 下軌 fd；空方壓力 = 上軌 fu
    support = np.where(direction == 1, final_dn, final_up)
    return direction, support


# ── 結構性回調回測 ─────────────────────────────────────────────────────────────

def _run_structural_pullback(df: pd.DataFrame, df_bt: pd.DataFrame,
                             initial_capital: float) -> tuple:
    """
    結構性回調策略回測（邏輯與 signal_engine.get_signal 一致，不含週線MACD）。
    進場：三層過濾全通過（趨勢/回調/轉強）且 entry_score ≥ 7、風報比 ≥ 1.5
    出場：EMA20 跌破 EMA50（趨勢出場）或收盤低於 SL（止損出場）
    回傳：(trades, equity_curve)
    """
    from signal_engine import calc_indicators

    df_calc = calc_indicators(df.copy())
    df_calc.dropna(
        subset=["MA20", "MA50", "EMA20", "EMA50", "RSI", "ATR", "ADX"],
        inplace=True,
    )
    df_calc = df_calc[df_calc.index >= df_bt.index[0]].copy()
    n = len(df_calc)
    if n < 10:
        return [], []

    c_arr   = df_calc["Close"].values.astype(float)
    o_arr   = df_calc["Open"].values.astype(float) if "Open" in df_calc.columns else c_arr.copy()
    h_arr   = df_calc["High"].values.astype(float)
    l_arr   = df_calc["Low"].values.astype(float)
    e20     = df_calc["EMA20"].values.astype(float)
    e50     = df_calc["EMA50"].values.astype(float)
    e200    = df_calc["EMA200"].values.astype(float)
    macd    = df_calc["MACD_hist"].values.astype(float)
    rsi     = df_calc["RSI"].values.astype(float)
    vr      = df_calc["VolumeRatio"].values.astype(float)
    atr     = df_calc["ATR"].values.astype(float)
    di_p    = df_calc["DI_P"].values.astype(float)
    di_m    = df_calc["DI_M"].values.astype(float)
    low20   = df_calc["Low20"].values.astype(float)

    position = None
    trades = []
    equity_curve = []

    for i in range(n):
        date = str(df_calc.index[i].date())
        c = c_arr[i]

        realized = sum(t["pnl"] for t in trades)
        if position:
            unreal = (c - position["entry"]) / position["entry"] * initial_capital * 0.95
            equity_curve.append({"date": date, "value": round(initial_capital + realized + unreal, 2)})
        else:
            equity_curve.append({"date": date, "value": round(initial_capital + realized, 2)})

        # 出場
        if position:
            hit_sl      = c <= position["sl"]
            trend_break = e20[i] < e50[i]
            if hit_sl or trend_break:
                pnl_pct = (c - position["entry"]) / position["entry"] * 100
                pnl     = (c - position["entry"]) / position["entry"] * initial_capital * 0.95
                trades.append({
                    "entry_date":  position["entry_date"],
                    "exit_date":   date,
                    "entry_price": round(position["entry"], 4),
                    "exit_price":  round(c, 4),
                    "strategy":    "結構性回調",
                    "exit_reason": "止損出場" if hit_sl else "趨勢出場",
                    "return_pct":  round(pnl_pct, 2),
                    "pnl":         round(pnl, 2),
                    "days_held":   i - position["entry_idx"],
                })
                position = None

        # 進場（需 i ≥ 2 以計算 MACD 連升）
        if position is None and i >= 2:
            # Layer 1：趨勢品質（簡化，不含週線MACD）
            ts = 0
            if e20[i] > e50[i]:                                    ts += 2
            if not np.isnan(e200[i]) and e50[i] > e200[i]:         ts += 2
            if ts < 4:
                continue

            # Layer 2：回調品質
            dist = (c - e20[i]) / e20[i] * 100
            if not (-8.0 <= dist <= 2.0):       continue
            if not (40 <= rsi[i] <= 62):         continue
            if not (vr[i] < 1.0):                continue

            # Layer 3：轉強評分
            es = 0
            if macd[i] > macd[i-1] > macd[i-2]: es += 2
            elif macd[i] > macd[i-1]:            es += 1

            is_green = c > o_arr[i]
            if is_green:
                es += 2
                if vr[i] > 1.0 and vr[i-1] < 0.9: es += 1

            if rsi[i] > 50 and rsi[i-1] <= 50:  es += 2
            elif rsi[i] > 50:                    es += 1

            dr = h_arr[i] - l_arr[i]
            if dr > 1e-6 and (c - l_arr[i]) / dr >= 0.8: es += 1
            if di_p[i] > di_m[i]:               es += 1

            if es < 7:
                continue

            # SL / TP
            l20v    = low20[i] if not np.isnan(low20[i]) else c * 0.92
            sl_raw  = max(e50[i] * 0.98, l20v * 0.99, c - 2.0 * atr[i])
            risk    = c - sl_raw
            if risk <= 0:
                continue
            actual_rr = (c + 2.5 * risk - c) / risk
            if actual_rr < 1.5:
                continue

            position = {
                "entry":      c,
                "entry_date": date,
                "entry_idx":  i,
                "sl":         sl_raw,
                "tp":         c + 2.5 * risk,
            }

    # 強制平倉
    if position:
        c = c_arr[-1]
        d = str(df_calc.index[-1].date())
        pnl_pct = (c - position["entry"]) / position["entry"] * 100
        pnl     = (c - position["entry"]) / position["entry"] * initial_capital * 0.95
        trades.append({
            "entry_date":  position["entry_date"],
            "exit_date":   d,
            "entry_price": round(position["entry"], 4),
            "exit_price":  round(c, 4),
            "strategy":    "結構性回調",
            "exit_reason": "回測結束",
            "return_pct":  round(pnl_pct, 2),
            "pnl":         round(pnl, 2),
            "days_held":   n - 1 - position["entry_idx"],
        })

    return trades, equity_curve


# ── 工具函式 ───────────────────────────────────────────────────────────────────

def _calc_max_drawdown(equity_curve: list) -> float:
    if not equity_curve:
        return 0.0
    values = [p["value"] for p in equity_curve]
    peak = values[0]
    max_dd = 0.0
    for v in values:
        if v > peak:
            peak = v
        dd = (v - peak) / peak * 100
        if dd < max_dd:
            max_dd = dd
    return round(max_dd, 2)


# ── 主要 endpoint ──────────────────────────────────────────────────────────────

@router.get("/stock")
def backtest_stock(
    symbol:     str = Query(...,    description="股票代號，例如 AAPL"),
    start_date: str = Query(None,   description="回測起始日 YYYY-MM-DD（優先於 period_years）"),
    end_date:   str = Query(None,   description="回測結束日 YYYY-MM-DD，預設今日"),
    period_years: int = Query(2, ge=1, le=10, description="回測年數（start_date 未傳時使用，預設 2）"),
    strategy:   str = Query("supertrend", description="supertrend 或 structural_pullback"),
):
    """
    策略回測。strategy 可選：
    - supertrend：三重 SuperTrend（預設）
    - structural_pullback：結構性回調
    日期可用 start_date/end_date（YYYY-MM-DD）或 period_years（1-10）。
    """
    symbol = symbol.upper().strip()

    # ── 解析日期範圍 ───────────────────────────────────────────────────────────
    try:
        bt_end = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()
    except ValueError:
        raise HTTPException(status_code=400, detail=f"end_date 格式錯誤，請使用 YYYY-MM-DD")

    try:
        bt_start = datetime.strptime(start_date, "%Y-%m-%d") if start_date else (
            bt_end - timedelta(days=period_years * 365)
        )
    except ValueError:
        raise HTTPException(status_code=400, detail=f"start_date 格式錯誤，請使用 YYYY-MM-DD")

    if bt_start >= bt_end:
        raise HTTPException(status_code=400, detail="start_date 必須早於 end_date")

    # 多抓 90 天讓指標充分暖機
    fetch_start = bt_start - timedelta(days=90)

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=fetch_start.strftime("%Y-%m-%d"),
            end=(bt_end + timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1d",
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"無法取得 {symbol} 的資料：{e}")

    if df.empty:
        raise HTTPException(status_code=400, detail=f"找不到股票代號 {symbol} 的資料，請確認代號是否正確")

    df = df.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    if len(df) < 100:
        raise HTTPException(
            status_code=400,
            detail=f"{symbol} 資料不足（僅 {len(df)} 根 K 棒，需至少 100 根）"
        )

    # ── 計算三條 SuperTrend ────────────────────────────────────────────────────
    high  = df["High"].values.astype(float)
    low   = df["Low"].values.astype(float)
    close = df["Close"].values.astype(float)

    st_results = [calc_supertrend(high, low, close, p, m) for p, m in ST_PARAMS]
    st_dirs    = [r[0] for r in st_results]
    st_sups    = [r[1] for r in st_results]   # 各條 ST 支撐/壓力線

    # 三條全綠時，止損線 = 三條支撐線中最高者（最保守）
    st_matrix   = np.stack(st_dirs, axis=1)           # shape (n, 3)
    all_green   = np.all(st_matrix == 1, axis=1)      # bool
    max_support = np.maximum(np.maximum(st_sups[0], st_sups[1]), st_sups[2])

    df["all_green"]   = all_green
    df["max_support"] = max_support

    # ── 只回測指定期間 ─────────────────────────────────────────────────────────
    df_bt = df[(df.index >= bt_start) & (df.index <= bt_end)].copy()

    if len(df_bt) < 30:
        raise HTTPException(
            status_code=400,
            detail=f"{symbol} 在指定期間內資料不足（{len(df_bt)} 根），無法進行回測"
        )

    initial_capital = 100_000.0

    # ── 手續費設定 ─────────────────────────────────────────────────────────────
    # 台股：買 0.1425% + 賣 0.1425% + 交易稅 0.3% = round-trip ~0.585%
    # 美股：多數券商免手續費，保守估計 0.1% round-trip
    is_tw    = symbol.endswith(".TW") or symbol.endswith(".TWO")
    fee_buy  = 0.001425 if is_tw else 0.0005
    fee_sell = 0.004425 if is_tw else 0.0005   # 台股含交易稅

    # ── 策略分支 ───────────────────────────────────────────────────────────────
    if strategy == "structural_pullback":
        trades, equity_curve = _run_structural_pullback(df, df_bt, initial_capital)
        no_signal_msg = "回測期間內沒有符合結構性回調全部條件的進場訊號"
    else:
        # ── 三重 SuperTrend（公正版）────────────────────────────────────────
        # 進場：三條 ST 由非全綠翻成全綠 → 次日開盤買入（消除 look-ahead bias）
        # 出場：收盤跌破最高 ST 支撐線 → 次日開盤賣出（移動止損）
        # 手續費：進出各計一次，台股含交易稅
        ag     = df_bt["all_green"].values
        sup    = df_bt["max_support"].values      # 三條 ST 支撐線中最高者
        closes = df_bt["Close"].values.astype(float)
        opens  = df_bt["Open"].values.astype(float)
        n_bt   = len(df_bt)

        equity   = initial_capital
        position = None
        trades   = []
        equity_curve = []

        for i in range(n_bt):
            date = str(df_bt.index[i].date())
            c    = closes[i]
            sl   = sup[i]    # 當日移動止損線

            # ── 標記資金曲線（mark-to-market）──────────────────────────────
            if position is not None:
                equity_curve.append({"date": date, "value": round(equity * (c / position["entry"]), 2)})
            else:
                equity_curve.append({"date": date, "value": round(equity, 2)})

            # ── 出場：收盤跌破 ST 支撐 → 次日開盤出場 ──────────────────────
            if position is not None and c < sl:
                exit_price = opens[i + 1] if i + 1 < n_bt else c
                exit_date  = str(df_bt.index[i + 1].date()) if i + 1 < n_bt else date
                pnl_pct    = (exit_price - position["entry"]) / position["entry"] * 100
                equity     = equity * (exit_price / position["entry"]) * (1 - fee_sell)
                trades.append({
                    "entry_date":  position["entry_date"],
                    "exit_date":   exit_date,
                    "entry_price": round(position["entry"], 4),
                    "exit_price":  round(exit_price, 4),
                    "strategy":    "三重SuperTrend",
                    "exit_reason": "ST止損出場",
                    "return_pct":  round(pnl_pct, 2),
                    "pnl":         round(equity - initial_capital, 2),
                    "days_held":   i - position["entry_idx"],
                })
                position = None

            # ── 進場：今日翻全綠 → 次日開盤買入 ────────────────────────────
            if position is None and ag[i]:
                prev_green = ag[i - 1] if i > 0 else False
                if not prev_green and i + 1 < n_bt:   # 確保有次日資料
                    entry_price = opens[i + 1]
                    entry_date  = str(df_bt.index[i + 1].date())
                    equity     *= (1 - fee_buy)
                    position    = {
                        "entry":      entry_price,
                        "entry_date": entry_date,
                        "entry_idx":  i,
                    }

        # ── 回測結束強制平倉 ────────────────────────────────────────────────
        if position is not None:
            exit_price = closes[-1]
            exit_date  = str(df_bt.index[-1].date())
            pnl_pct    = (exit_price - position["entry"]) / position["entry"] * 100
            equity     = equity * (exit_price / position["entry"]) * (1 - fee_sell)
            trades.append({
                "entry_date":  position["entry_date"],
                "exit_date":   exit_date,
                "entry_price": round(position["entry"], 4),
                "exit_price":  round(exit_price, 4),
                "strategy":    "三重SuperTrend",
                "exit_reason": "回測結束",
                "return_pct":  round(pnl_pct, 2),
                "pnl":         round(equity - initial_capital, 2),
                "days_held":   n_bt - 1 - position["entry_idx"],
            })
        no_signal_msg = "回測期間內沒有符合三重 SuperTrend 全綠條件的進場訊號"

    # ── 統計計算 ───────────────────────────────────────────────────────────────
    closes_bt     = df_bt["Close"].values.astype(float)
    total_trades  = len(trades)
    buy_hold_return = round(
        (float(closes_bt[-1]) - float(closes_bt[0])) / float(closes_bt[0]) * 100, 2
    )

    if total_trades == 0:
        return {
            "symbol":          symbol,
            "start_date":      bt_start.strftime("%Y-%m-%d"),
            "end_date":        bt_end.strftime("%Y-%m-%d"),
            "strategy":        strategy,
            "total_return":    0.0,
            "buy_hold_return": buy_hold_return,
            "total_trades":    0,
            "win_rate":        0.0,
            "avg_return":      0.0,
            "max_drawdown":    0.0,
            "by_strategy":     {},
            "equity_curve":    equity_curve,
            "trades":          [],
            "message":         no_signal_msg,
        }

    wins       = [t for t in trades if t["return_pct"] > 0]
    win_rate   = round(len(wins) / total_trades, 4)
    avg_return = round(sum(t["return_pct"] for t in trades) / total_trades, 2)
    max_drawdown = _calc_max_drawdown(equity_curve)
    total_return = round(
        (equity_curve[-1]["value"] - initial_capital) / initial_capital * 100, 2
    )

    def _stats(t_list):
        w = [t for t in t_list if t["return_pct"] > 0]
        return {
            "trades":     len(t_list),
            "win_rate":   round(len(w) / len(t_list), 4),
            "avg_return": round(sum(t["return_pct"] for t in t_list) / len(t_list), 2),
        }

    trades_out = [{k: v for k, v in t.items() if k != "pnl"} for t in trades]

    bh_start = float(df_bt["Close"].iloc[0])
    buy_hold_curve = [
        {
            "date":  str(df_bt.index[i].date()),
            "value": round(initial_capital * float(df_bt["Close"].iloc[i]) / bh_start, 2),
        }
        for i in range(len(df_bt))
    ]

    return {
        "symbol":          symbol,
        "start_date":      bt_start.strftime("%Y-%m-%d"),
        "end_date":        bt_end.strftime("%Y-%m-%d"),
        "strategy":        strategy,
        "total_return":    total_return,
        "buy_hold_return": buy_hold_return,
        "total_trades":    total_trades,
        "win_rate":        win_rate,
        "avg_return":      avg_return,
        "max_drawdown":    max_drawdown,
        "by_strategy":     {
            ("三重SuperTrend" if strategy != "structural_pullback" else "結構性回調"): _stats(trades)
        },
        "equity_curve":    equity_curve,
        "buy_hold_curve":  buy_hold_curve,
        "trades":          trades_out,
    }
